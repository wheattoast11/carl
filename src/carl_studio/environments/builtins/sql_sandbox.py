"""SQL sandbox environment — in-memory SQLite, query execution, binary+continuous reward.

Second CARL environment lane (query). Proves the protocol generalizes beyond code.
Isolated per-instance via in-memory SQLite.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import register_environment


@register_environment
class SQLSandboxEnv(BaseEnvironment):
    """SQL query sandbox — in-memory SQLite, schema inspection, query execution."""

    spec = EnvironmentSpec(
        lane=EnvironmentLane.QUERY,
        name="sqlite-sandbox",
        tools=("describe_schema", "run_query", "submit_answer"),
        max_turns=5,
        reward_type="continuous",
        system_prompt=(
            "You are a data analyst. Answer questions by querying the database.\n\n"
            "First inspect the schema, then write SQL queries to find the answer. "
            "Submit your final answer when ready.\n\n"
            "Do NOT explain your reasoning. Just query and answer."
        ),
        dataset_columns=("task_description", "schema_ddl"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._conn: sqlite3.Connection | None = None
        self._expected: str = ""
        self._last_query_result: str = ""

    def reset(self, **kwargs: Any) -> str | None:
        """Create fresh SQLite database with the provided schema.

        Args:
            **kwargs: Dataset columns. Expected: task_description (str),
                schema_ddl (str), expected_result (str, optional).
        """
        super().reset(**kwargs)
        if self._conn is not None:
            self._conn.close()

        self._conn = sqlite3.connect(":memory:")
        self._expected = kwargs.get("expected_result", "")
        self._last_query_result = ""

        ddl = kwargs.get("schema_ddl", "")
        if ddl:
            try:
                self._conn.executescript(ddl)
            except sqlite3.Error as e:
                return f"Schema error: {e}"

        # Insert seed data if provided
        seed_sql = kwargs.get("seed_data", "")
        if seed_sql:
            try:
                self._conn.executescript(seed_sql)
            except sqlite3.Error as e:
                return f"Seed data error: {e}"

        return kwargs.get("task_description", None)

    def describe_schema(self) -> str:
        """List all tables and their columns in the database.

        Args:
            (none)

        Returns:
            Schema description as text.
        """
        if self._conn is None:
            return "Error: Database not initialized"
        try:
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            if not tables:
                result = "No tables found."
                self._record_turn("describe_schema", {}, result)
                return result

            parts = []
            for table in tables:
                cursor = self._conn.execute(f"PRAGMA table_info({table})")
                cols = cursor.fetchall()
                col_strs = [f"  {c[1]} {c[2]}{' NOT NULL' if c[3] else ''}{' PK' if c[5] else ''}" for c in cols]
                # Row count
                count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                parts.append(f"TABLE {table} ({count} rows):\n" + "\n".join(col_strs))
            result = "\n\n".join(parts)
        except sqlite3.Error as e:
            result = f"Error: {e}"
        self._record_turn("describe_schema", {}, result)
        return result

    def run_query(self, sql: str) -> str:
        """Execute a SQL query and return results.

        Args:
            sql: The SQL query to execute (SELECT only).

        Returns:
            Query results as formatted text.
        """
        if self._conn is None:
            return "Error: Database not initialized"

        # Only allow SELECT queries for safety
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
            result = "Error: Only SELECT/WITH queries allowed. Use submit_answer to provide your final answer."
            self._record_turn("run_query", {"sql": sql}, result)
            return result

        try:
            cursor = self._conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            if not rows:
                result = f"Columns: {', '.join(columns)}\n(0 rows)"
            else:
                header = " | ".join(columns)
                separator = "-" * len(header)
                row_strs = [" | ".join(str(v) for v in row) for row in rows[:50]]
                result = f"{header}\n{separator}\n" + "\n".join(row_strs)
                if len(rows) > 50:
                    result += f"\n... ({len(rows)} total rows, showing first 50)"

            self._last_query_result = result
        except sqlite3.Error as e:
            result = f"SQL Error: {e}"
        self._record_turn("run_query", {"sql": sql}, result)
        return result

    def submit_answer(self, answer: str) -> str:
        """Submit your final answer to the question.

        Args:
            answer: Your answer based on the query results.

        Returns:
            Confirmation message.
        """
        self.done = True

        # Score: simple substring match if expected_result provided
        if self._expected:
            expected_lower = self._expected.lower().strip()
            answer_lower = answer.lower().strip()
            if expected_lower in answer_lower or answer_lower in expected_lower:
                self.reward = 1.0
            elif any(tok in answer_lower for tok in expected_lower.split()):
                self.reward = 0.5
            else:
                self.reward = 0.0
        else:
            # No expected result — binary: did they query at all?
            self.reward = 1.0 if self._last_query_result else 0.0

        result = f"Answer submitted. Score: {self.reward}"
        self._record_turn("submit_answer", {"answer_length": len(answer)}, result)
        return result

    def __del__(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except (TypeError, AttributeError):
            pass
