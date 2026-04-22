---
last_updated: 2026-04-21
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.16
---

# v0.16 Utility Inventory — Handle-Runtime Python Deps

Curated short-list of battle-tested Python libraries for the Carl handle
runtime. Selection bias: libraries that return cursors / iterators /
references rather than materialized blobs; first-class streaming;
audit-friendly (deterministic hash of input → output).

One pick per category unless a genuine sync/async or pure-python/native
axis forces two. Each entry: three to six lines.

---

### HTTP client: `httpx` (0.28.1, BSD-3-Clause) [[1]]
- **Why:** One library for sync + async with identical API; HTTP/2 and
  streaming primitives built-in. `requests` has no async; `aiohttp` is
  async-only and no HTTP/2.
- **Handle-fit:** `client.stream()` returns a `Response` context manager
  whose `.iter_bytes()` / `.iter_raw()` yields chunks without buffering
  the body — we pass the iterator as a handle, never the full payload.
- **Install:** `pip install httpx`

### Hashing (native): `blake3` (1.0.8, Apache-2.0 OR MIT) [[2]]
- **Why:** Parallel SIMD Rust implementation; 5–10× faster than
  `hashlib.sha256` on typical input and supports incremental `.update()`
  on mmapped files. `hashlib` stays for FIPS compat but is not the
  primary fingerprinter.
- **Handle-fit:** Fingerprint = content reference. BLAKE3 output doubles
  as a deterministic handle ID for the audit trail; extendable-output
  (XOF) mode lets us derive sub-handles without re-hashing.
- **Install:** `pip install blake3`

### Serialization: `msgspec` (0.20.0, BSD-3-Clause) [[3]]
- **Why:** Fastest JSON + MessagePack for Python; validation is built
  into the decoder (single pass, zero-copy where possible). `orjson` is
  serialize-only; `pydantic` validates after a separate decode step.
- **Handle-fit:** `msgspec.Struct` gives frozen dataclass-like handles
  with `__hash__` + stable field order — decode straight into a typed
  reference object that carries its own provenance.
- **Install:** `pip install msgspec`

### Compression: `zstandard` (0.25.0, BSD-3-Clause) [[4]]
- **Why:** Best ratio-vs-speed of any mainstream codec; streaming
  compressor/decompressor with frame boundaries we can use as seek
  points. `gzip` stays in stdlib for interop but is not competitive on
  ratio or throughput.
- **Handle-fit:** `ZstdDecompressor().stream_reader(fh)` returns a
  file-like handle — readers pull chunks lazily, and the frame index
  becomes a random-access cursor for audit replay.
- **Install:** `pip install zstandard`

### PDF extraction: `pypdfium2` (5.7.0, Apache-2.0 OR BSD-3-Clause) [[5]]
- **Why:** Google's PDFium bindings — renders and extracts the same way
  Chrome does, which is the real compatibility target. `pypdf` is pure
  Python and slower on complex layouts; `pdfplumber` is a wrapper on
  pdfminer.six with narrower fidelity.
- **Handle-fit:** `PdfDocument` exposes `PdfPage` iterators — each page
  is a lazy handle with its own text, bitmap, and object tree; pages
  never leave the native side until explicitly materialized.
- **Install:** `pip install pypdfium2`

### Embedded analytics DB: `duckdb` (1.5.0, MIT) [[6]]
- **Why:** OLAP engine with zero-copy Arrow interop, Parquet/CSV/JSON
  scanners, and vectorized execution. SQLite stays the *transactional*
  store (already `~/.carl/state.db`) — DuckDB is the *analytical*
  sibling for run ledgers, trajectory joins, trace windows.
- **Handle-fit:** `duckdb.sql("…")` returns a `DuckDBPyRelation` — a
  lazy query handle. No rows materialize until `.fetchall()` /
  `.arrow()` / `.pl()`; chains compose without intermediate copies.
- **Install:** `pip install duckdb`

### Process/resource probing: `psutil` (7.2.2, BSD-3-Clause) [[7]]
- **Why:** The canonical cross-platform process/system probe. No real
  competition at this scope. 7.2.2 uses `pidfd_open` on Linux 5.3+ so
  `Process.wait()` no longer busy-loops.
- **Handle-fit:** `Process(pid)` is the handle; methods (`cpu_times`,
  `memory_info`, `open_files`) are cursor reads against a stable
  subject. We never serialize full process trees — we pass the PID as
  the reference.
- **Install:** `pip install psutil`

### Filesystem watch: `watchfiles` (1.1.1, MIT) [[8]]
- **Why:** Rust (Notify) backend, modern async API, smaller surface.
  `watchdog` is pure-Python + older event model and debounces badly
  under burst loads.
- **Handle-fit:** `watch(path)` / `awatch(path)` are generators yielding
  `(Change, path)` tuples — the path IS the handle; consumers pull
  events when ready, no subscription object to leak.
- **Install:** `pip install watchfiles`

### Tabular data: `polars` (1.40.0, MIT) [[9]]
- **Why:** Lazy query engine (`scan_csv` / `scan_parquet`) that
  composes a plan before touching rows; Arrow-native, multithreaded,
  out-of-core. Pandas is eager-by-default and thus hostile to the
  handle pattern.
- **Handle-fit:** `pl.scan_parquet(path)` returns a `LazyFrame` — a
  pure query handle with no materialized data. `.collect()` is the only
  place rows appear; `.sink_parquet()` streams straight to disk without
  ever leaving the native side.
- **Install:** `pip install polars`

### Fuzzy matching: `rapidfuzz` (3.14.5, MIT) [[10]]
- **Why:** C++ core, same API as `thefuzz` / FuzzyWuzzy but 10–30×
  faster and MIT-licensed (the original `fuzzywuzzy` is GPL).
  `thefuzz` is the rebranded GPL-liberated fork but still slower.
- **Handle-fit:** `process.cdist(queries, choices)` returns an ndarray
  of scores indexed by handle positions — we can store score matrices
  without materializing strings, matching against interned handles.
- **Install:** `pip install rapidfuzz`

### Retries/resilience: `stamina` (26.1.0, MIT) [[11]]
- **Why:** Opinionated wrapper over `tenacity` (9.1.4, [[12]]) with
  safer defaults (exp backoff + jitter + total-time cap by default),
  async-native, and built-in structlog / Prometheus instrumentation
  that lines up with `carl_core.resilience`. Raw `tenacity` is fine but
  every team mis-configures it.
- **Handle-fit:** `@stamina.retry` and `stamina.retry_context()` are
  decorators/context managers — they wrap a *call site*, not the
  callable's return; retry attempts emit events we thread into the
  `InteractionChain` as `Step.probe_call` entries.
- **Install:** `pip install stamina`

### Archives / 7z: `py7zr` (1.1.2, LGPL-2.1-or-later) [[13]]
- **Why:** Only well-maintained pure-Python 7z impl with Zstd/LZMA2/
  PPMd + AES. Stdlib `zipfile` handles zip and nothing else;
  `tarfile` handles tar; we need 7z for cross-tool artifact drops.
  LGPL is acceptable because we dynamically link via import.
- **Handle-fit:** `SevenZipFile.readall()` streams member-by-member;
  `.read([name])` returns a `BytesIO` handle per member — we can tee
  the stream into both the hash fingerprint and the consumer without
  double-reading.
- **Install:** `pip install py7zr`

### Image I/O: `pillow` (12.2.0, MIT-CMU) [[14]]
- **Why:** The universal Python image library. `pillow-simd` exists
  for SIMD-accelerated resize but is unmaintained and pins to old
  Pillow; stock Pillow 10+ added most of the SIMD paths anyway.
- **Handle-fit:** `Image.open(path)` is lazy — pixels don't decode
  until `.load()` or pixel access; `.crop()` returns a new `Image`
  that shares the source until modified. We pass `Image` instances
  as handles and fingerprint via `.tobytes()` only at checkpoint.
- **Install:** `pip install pillow`

### Parallel task dispatch: `anyio` (4.13.0, MIT) [[15]]
- **Why:** Already a transitive dep (via `mcp`); single API over
  asyncio + trio with proper structured concurrency (`TaskGroup`,
  cancellation scopes). Raw `trio` is excellent but asyncio-only
  teams can't use it without an `anyio` bridge anyway.
- **Handle-fit:** `anyio.create_task_group()` returns a scope whose
  child tasks are referenced by their cancel scopes, not the
  coroutine objects themselves — the scope IS the handle for the
  whole parallel branch.
- **Install:** `pip install anyio`

### Date parsing: `whenever` (0.10.0, MIT) [[16]]
- **Why:** Rust-backed, typesafe datetime library; distinguishes
  `Instant` / `ZonedDateTime` / `PlainDateTime` at the type level so
  tz bugs become type errors. `pendulum` and `arrow` both conflate
  aware/naive and neither has the Rust perf floor.
- **Handle-fit:** Every type is immutable and `__hash__`able — a
  `ZonedDateTime` is its own stable reference, and round-trips via
  ISO-8601 with zone suffix (`2026-04-21T12:00[America/New_York]`)
  for lossless audit serialization.
- **Install:** `pip install whenever`

---

## Skip list

Libraries people instinctively reach for in this space; we decline:

- **`deep` / `deepdiff`** — attractive for "diff the Step input vs
  output," but the hash chain already IS the diff. Adding deepdiff
  imports ~100k LOC and still can't beat a BLAKE3 fingerprint for
  audit purposes.
- **`requests`** — no async, no HTTP/2, blocking-only, and the
  maintainer put it on life-support mode. `httpx` is the successor
  the author himself points to.
- **`simplejson`** — `json` (stdlib) + `msgspec` cover the space.
  `simplejson` is slower than both and predates `Decimal` support
  in stdlib that was the original reason to use it.
- **`sh`** — shells out with magic attribute access; security auditing
  nightmare and cannot be capability-constrained. `subprocess.run` or
  `anyio.run_process` are the only acceptable answers.
- **`fabric`** — SSH-based remote-exec framework; wrong abstraction
  for an agent runtime where every remote side effect must be gated
  and witnessed. If we need remote exec, it flows through the
  authenticated carl.camp HTTP surface, not a fire-and-forget SSH.

---

## Sources

- [1] [httpx on PyPI — 0.28.1](https://pypi.org/project/httpx/)
- [2] [blake3 on PyPI — 1.0.8](https://pypi.org/project/blake3/)
- [3] [msgspec on PyPI — 0.20.0](https://pypi.org/project/msgspec/)
- [4] [zstandard on PyPI — 0.25.0](https://pypi.org/project/zstandard/)
- [5] [pypdfium2 on PyPI — 5.7.0](https://pypi.org/project/pypdfium2/)
- [6] [DuckDB Python 1.5.0 release](https://pypi.org/project/duckdb/)
- [7] [psutil on PyPI — 7.2.2](https://pypi.org/project/psutil/)
- [8] [watchfiles on PyPI — 1.1.1](https://pypi.org/project/watchfiles/)
- [9] [polars on PyPI — 1.40.0](https://pypi.org/project/polars/)
- [10] [RapidFuzz on PyPI — 3.14.5](https://pypi.org/project/RapidFuzz/)
- [11] [stamina on PyPI — 26.1.0](https://pypi.org/project/stamina/)
- [12] [tenacity on PyPI — 9.1.4](https://pypi.org/project/tenacity/)
- [13] [py7zr on PyPI — 1.1.2](https://pypi.org/project/py7zr/)
- [14] [Pillow 12.2.0 release](https://pypi.org/project/Pillow/)
- [15] [AnyIO on PyPI — 4.13.0](https://pypi.org/project/anyio/)
- [16] [whenever on PyPI — 0.10.0](https://pypi.org/project/whenever/)
