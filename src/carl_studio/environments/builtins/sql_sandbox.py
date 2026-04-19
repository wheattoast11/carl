"""SQL sandbox environment -- in-memory SQLite, query execution, binary+continuous reward.

Second CARL environment lane (query). Proves the protocol generalizes beyond code.
Isolated per-instance via in-memory SQLite. No filesystem access.

Tools: execute_query (SELECT/WITH only), list_tables, describe_table, insert_data.
Safety: read-only by default. insert_data only writes to tables listed in writable_tables.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)

from carl_studio.environments.connection import EnvironmentConnection
from carl_studio.environments.protocol import EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import register_environment

_MAX_ROWS_DISPLAY = 50
_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|ALTER|CREATE|ATTACH|DETACH|LOAD|PRAGMA\s+(?!table_info|database_list))\b",
    re.IGNORECASE,
)


@register_environment
class SQLSandboxEnv(EnvironmentConnection):
    """SQL query sandbox -- in-memory SQLite, schema inspection, query execution.

    Supports two usage patterns:
      1. Query-only: model explores schema and answers questions via execute_query.
      2. Query+write: model inserts data into designated writable tables.

    Instrumentation attributes:
        _tool_call_count: Total tool invocations this episode.
        _tool_failure_count: Tool calls that returned errors.
        _query_count: Number of successful queries executed.
        _insert_count: Number of successful inserts.
    """

    spec = EnvironmentSpec(
        lane=EnvironmentLane.QUERY,
        name="sqlite-sandbox",
        tools=("execute_query", "list_tables", "describe_table", "insert_data"),
        max_turns=8,
        reward_type="continuous",
        system_prompt=(
            "You are a data analyst. Answer questions by querying the database.\n\n"
            "First inspect the schema with list_tables and describe_table, "
            "then write SQL queries with execute_query to find the answer. "
            "Use insert_data when you need to write rows to writable tables.\n\n"
            "Do NOT explain your reasoning. Just query and answer."
        ),
        dataset_columns=("task_description", "schema_ddl"),
    )

    connection_spec = ConnectionSpec(
        name="carl.env.sql",
        scope=ConnectionScope.ONE_P,
        kind=ConnectionKind.ENVIRONMENT,
        direction=ConnectionDirection.BIDIRECTIONAL,
        transport=ConnectionTransport.IN_PROCESS,
        trust=ConnectionTrust.PUBLIC,
        metadata={"lane": "query", "sandbox": "sqlite-memory"},
    )

    def __init__(self) -> None:
        super().__init__()
        self._conn: sqlite3.Connection | None = None
        self._expected: str = ""
        self._expected_pattern: str = ""
        self._writable_tables: set[str] = set()
        self._tool_call_count: int = 0
        self._tool_failure_count: int = 0
        self._query_count: int = 0
        self._insert_count: int = 0
        self._last_query_result: str = ""

    def reset(self, **kwargs: Any) -> str | None:
        """Create fresh SQLite database with the provided schema.

        Args:
            **kwargs: Dataset columns. Expected: task_description (str),
                schema_ddl (str), seed_data (str, optional),
                expected_result (str, optional), expected_pattern (str, optional),
                writable_tables (list[str], optional).
        """
        super().reset(**kwargs)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._expected = str(kwargs.get("expected_result", ""))
        self._expected_pattern = str(kwargs.get("expected_pattern", ""))
        self._last_query_result = ""
        self._tool_call_count = 0
        self._tool_failure_count = 0
        self._query_count = 0
        self._insert_count = 0

        # Writable tables -- empty set means fully read-only
        writable = kwargs.get("writable_tables", [])
        if isinstance(writable, (list, tuple)):
            self._writable_tables = {t.lower() for t in writable}
        else:
            self._writable_tables = set()

        # Apply schema DDL
        ddl = kwargs.get("schema_ddl", "")
        if ddl:
            try:
                self._conn.executescript(ddl)
            except sqlite3.Error as e:
                return f"Schema error: {e}"

        # Insert seed data
        seed_sql = kwargs.get("seed_data", "")
        if seed_sql:
            try:
                self._conn.executescript(seed_sql)
            except sqlite3.Error as e:
                return f"Seed data error: {e}"

        return kwargs.get("task_description", None)

    def _ensure_conn(self) -> sqlite3.Connection:
        """Return the active connection or raise."""
        if self._conn is None:
            raise RuntimeError("Database not initialized -- call reset() first")
        return self._conn

    def list_tables(self) -> str:
        """List all tables in the database with row counts.

        Args:
            (none)

        Returns:
            Table listing with row counts.
        """
        self._tool_call_count += 1
        try:
            conn = self._ensure_conn()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                result = "No tables found."
                self._record_turn("list_tables", {}, result)  # pyright: ignore[reportPrivateUsage]
                return result

            lines = []
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                writable_tag = " [writable]" if table.lower() in self._writable_tables else ""
                lines.append(f"  {table} ({count} rows){writable_tag}")
            result = "Tables:\n" + "\n".join(lines)
        except (sqlite3.Error, RuntimeError) as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
        self._record_turn("list_tables", {}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def describe_table(self, table: str) -> str:
        """Show columns, types, and constraints for a table.

        Args:
            table: Name of the table to describe.

        Returns:
            Column listing with types and constraints.
        """
        self._tool_call_count += 1
        try:
            conn = self._ensure_conn()
            cursor = conn.execute(f"PRAGMA table_info([{table}])")
            cols = cursor.fetchall()
            if not cols:
                self._tool_failure_count += 1
                result = f"Error: Table '{table}' not found or has no columns."
                self._record_turn("describe_table", {"table": table}, result)  # pyright: ignore[reportPrivateUsage]
                return result

            col_lines = []
            for c in cols:
                # c: (cid, name, type, notnull, dflt_value, pk)
                parts = [f"  {c[1]} {c[2]}"]
                if c[3]:
                    parts.append("NOT NULL")
                if c[5]:
                    parts.append("PK")
                if c[4] is not None:
                    parts.append(f"DEFAULT {c[4]}")
                col_lines.append(" ".join(parts))

            count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            writable_tag = " [writable]" if table.lower() in self._writable_tables else " [read-only]"
            result = f"TABLE {table} ({count} rows){writable_tag}:\n" + "\n".join(col_lines)
        except (sqlite3.Error, RuntimeError) as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
        self._record_turn("describe_table", {"table": table}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def execute_query(self, sql: str) -> str:
        """Execute a read-only SQL query and return results.

        Args:
            sql: The SQL query to execute (SELECT and WITH only).

        Returns:
            Query results as formatted text, or an error message.
        """
        self._tool_call_count += 1
        try:
            conn = self._ensure_conn()
        except RuntimeError as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
            self._record_turn("execute_query", {"sql": sql}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        # Safety: only SELECT/WITH allowed
        stripped = sql.strip()
        upper = stripped.upper()
        if not upper.startswith("SELECT") and not upper.startswith("WITH"):
            self._tool_failure_count += 1
            result = "Error: Only SELECT and WITH queries allowed. Use insert_data for writes."
            self._record_turn("execute_query", {"sql": sql}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        # Block dangerous embedded statements
        if _DANGEROUS_KEYWORDS.search(stripped):
            self._tool_failure_count += 1
            result = "Error: Query contains disallowed keywords (DROP, ALTER, CREATE, ATTACH, etc.)."
            self._record_turn("execute_query", {"sql": sql}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            self._query_count += 1

            if not rows:
                result = f"Columns: {', '.join(columns)}\n(0 rows)"
            else:
                header = " | ".join(columns)
                separator = "-" * len(header)
                row_strs = [" | ".join(str(v) for v in row) for row in rows[:_MAX_ROWS_DISPLAY]]
                result = f"{header}\n{separator}\n" + "\n".join(row_strs)
                if len(rows) > _MAX_ROWS_DISPLAY:
                    result += f"\n... ({len(rows)} total rows, showing first {_MAX_ROWS_DISPLAY})"

            self._last_query_result = result
            self._score_result(result)
        except sqlite3.Error as e:
            self._tool_failure_count += 1
            result = f"SQL Error: {e}"
        self._record_turn("execute_query", {"sql": sql}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def insert_data(self, table: str, columns: str, values: str) -> str:
        """Insert a row into a writable table.

        Args:
            table: Target table name (must be in writable_tables).
            columns: Comma-separated column names, e.g. "name, age, city".
            values: Comma-separated values, e.g. "'Alice', 30, 'NYC'".

        Returns:
            Confirmation message or error.
        """
        self._tool_call_count += 1
        try:
            conn = self._ensure_conn()
        except RuntimeError as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
            self._record_turn("insert_data", {"table": table}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        # Safety: only writable tables
        if table.lower() not in self._writable_tables:
            self._tool_failure_count += 1
            if self._writable_tables:
                allowed = ", ".join(sorted(self._writable_tables))
                result = f"Error: Table '{table}' is read-only. Writable tables: {allowed}"
            else:
                result = "Error: No writable tables configured. All tables are read-only."
            self._record_turn("insert_data", {"table": table}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        # Validate column names are simple identifiers (prevent SQL injection)
        import re as _re
        for col_name in columns.split(','):
            if not _re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', col_name.strip()):
                self._tool_failure_count += 1
                return f"Error: Invalid column name: {col_name.strip()}"

        # Validate no dangerous keywords in the values
        combined = f"{table} {columns} {values}"
        if _DANGEROUS_KEYWORDS.search(combined):
            self._tool_failure_count += 1
            result = "Error: Input contains disallowed keywords."
            self._record_turn("insert_data", {"table": table}, result)  # pyright: ignore[reportPrivateUsage]
            return result

        sql = f"INSERT INTO [{table}] ({columns}) VALUES ({values})"
        try:
            conn.execute(sql)
            conn.commit()
            self._insert_count += 1
            result = f"Inserted 1 row into {table}."
        except sqlite3.Error as e:
            self._tool_failure_count += 1
            result = f"SQL Error: {e}"
        self._record_turn("insert_data", {"table": table, "columns": columns}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def _score_result(self, query_result: str) -> None:
        """Update reward based on query result matching expected output.

        Scoring hierarchy:
          1. expected_pattern (regex match) -> 1.0 if match, no change if not
          2. expected_result (substring match) -> 1.0 exact, 0.5 partial, 0.0 miss
          3. No expected -> 1.0 if any query returned rows
        """
        if self._expected_pattern:
            try:
                if re.search(self._expected_pattern, query_result, re.IGNORECASE):
                    self.reward = 1.0
                    return
            except re.error:
                pass  # Invalid pattern -- fall through to substring match

        if self._expected:
            expected_lower = self._expected.lower().strip()
            result_lower = query_result.lower().strip()
            if expected_lower in result_lower or result_lower in expected_lower:
                self.reward = 1.0
            elif any(tok in result_lower for tok in expected_lower.split() if len(tok) > 1):
                self.reward = max(self.reward, 0.5)
            # Don't decrease reward -- best-of-N across turns
        else:
            # No expected result -- reward for producing any results
            if "(0 rows)" not in query_result:
                self.reward = max(self.reward, 1.0)

    def __del__(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except (TypeError, AttributeError):
            pass
        # Delegate to EnvironmentConnection.__del__ so the underlying
        # connection adapter is closed cleanly on shutdown.
        try:
            super().__del__()
        except (TypeError, AttributeError):
            pass
