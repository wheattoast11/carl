---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9.0
---

# EML wire format — cross-boundary isomorphism

CARL ships two EML expression-tree implementations. They **share the
semantic model** (`eml(x, y) = exp(x) - ln(y)`, MAX_DEPTH=4) but diverge
on numeric precision and wire layout. This document is the authoritative
reference for how the two formats relate, what they share, and what an
interop layer must translate.

## 1. Canonical semantic model (shared)

| Axiom | Value |
|---|---|
| Binary op | `eml(x, y) = exp(x) - ln(y)` |
| x clamp | ±20.0 (`CLAMP_X`) |
| y floor | Python: `max(y, 1e-12)`. Rust: rejects `y ≤ 0` with `DomainError` |
| Max depth | 4 |
| Leaf kinds | `CONST`, `VAR_X` |
| Binary kind | `EML` |

**Divergence (intentional)**: Python clamps non-positive `y` into `EPS=1e-12`
and returns a finite value; Rust returns `EmlError::DomainError(y)` for
`y ≤ 0`. This is a deliberate safety-vs-strictness tradeoff — Python
powers training loops where dropping a batch on a numeric blip is
expensive; Rust powers on-chain eval where silent saturation is a bug.

## 2. Opcode table (aligned on first three values)

| Name (sem.) | Python `EMLOp` | Rust `EmlOp` | Wire byte |
|---|---|---|---|
| CONST | `CONST = 0` | `Const = 0` | Python stream tag: `0x01` · Rust: `0x00` |
| VAR_X | `VAR_X = 1` | `VarX = 1` | Python stream tag: `0x02` · Rust: `0x01` |
| EML | `EML = 2` | `Eml = 2` | Python stream tag: `0x03` · Rust: `0x02` |
| ADD | — | `Add = 3` | Rust: `0x03` |
| SUB | — | `Sub = 4` | Rust: `0x04` |
| MUL | — | `Mul = 5` | Rust: `0x05` |
| NEG | — | `Neg = 6` | Rust: `0x06` |

**Reading the table**: the *enum numeric values* (`EMLOp`/`EmlOp`) agree
on `{CONST=0, VAR_X=1, EML=2}`. The *wire bytes* that appear in the
serialized stream do NOT agree: Python uses `+1` offsets (`0x01/0x02/0x03`)
in its postfix stream, Rust uses the bare enum value (`0/1/2`) in its
bytecode. Any interop layer MUST know both mappings.

Python is a pure-EML subset; Rust extends the algebra with the arithmetic
ring (`Add/Sub/Mul/Neg`) so that wasm-side stacks can compose EML outputs
with cheap linear ops without re-serializing. Trees that use only
`{Const, VarX, Eml}` are **semantically isomorphic** across both sides;
trees that use `Add/Sub/Mul/Neg` are **Rust-only** and have no Python
equivalent (yet).

## 3. Wire layouts

### Python `EMLTree.to_bytes()`

```
magic:     b"EML\x01"          4 bytes   (bytes 0x45 4d 4c 01)
input_dim: uint16 LE           2 bytes
stream:    postfix tag stream  variable  (see below)
```

Stream tags (inline operands, NOT a separate const pool):
- `0x01` (CONST) + `float64` LE (8 bytes)
- `0x02` (VAR_X) + `uint16` LE (2 bytes)
- `0x03` (EML) — no operand; consumes the top 2 stack entries

Constants are stored **f64, inline, in postorder traversal order**. There
is no constant pool — a tree with 10 identical constants serializes 10
f64 payloads.

Example — `exp_x` tree (`eml(x0, 1)`):
```
45 4d 4c 01              magic
01 00                    input_dim=1
02 00 00                 VAR_X 0
01 00 00 00 00 00 00 f0 3f   CONST 1.0 (f64 LE)
03                       EML
```

### Rust `EmlTree::encode()`

```
magic:     b"EML1"          4 bytes   (bytes 0x45 4d 4c 31)
input_dim: uint32 LE        4 bytes
n_consts:  uint32 LE        4 bytes
consts:    f32 LE × n       4×n bytes  (constant pool)
bc_len:    uint32 LE        4 bytes
bytecode:  u8 × bc_len      variable  (postfix opcodes)
```

Bytecode opcodes (reference a const pool, NOT inline):
- `0x00` (Const) + `u8` const_index (1 byte)
- `0x01` (VarX) + `u8` input_index (1 byte)
- `0x02` (Eml), `0x03` (Add), `0x04` (Sub), `0x05` (Mul), `0x06` (Neg) — no operand

Constants are **f32, in a separate pool**, indexed by the `Const`
opcode's operand. Reuses are free.

Example — `exp_x` tree:
```
45 4d 4c 31              magic
01 00 00 00              input_dim=1
01 00 00 00              n_consts=1
00 00 80 3f              const[0] = 1.0 (f32 LE)
05 00 00 00              bc_len=5
01 00                    VarX 0
00 00                    Const 0
02                       Eml
```

## 4. Error taxonomy bijection

Python → Rust (where a direct mapping exists):

| Python code | Rust variant | Notes |
|---|---|---|
| `carl.eml.depth_exceeded` | `EmlError::DepthExceeded` | Bijective |
| `carl.eml.domain_error` (structural) | `EmlError::InputOutOfRange` / `EmlError::ConstOutOfRange` | Python lumps structural errors under a single code; Rust partitions them |
| `carl.eml.decode_error` | `EmlError::DecodeError` / `EmlError::MalformedBytecode` | Python has one code for both shape errors and unknown tags |
| — (silent clamp) | `EmlError::Overflow(f32)` | Python clamps instead of erroring |
| — (silent floor) | `EmlError::DomainError(f32)` (on `y ≤ 0`) | Python applies `max(y, EPS)` instead of erroring |
| — | `EmlError::StackUnderflow` | Python raises `carl.eml.decode_error` on postfix stack imbalance |

The mapping is **injective from Python to Rust** (every Python code has
a Rust target) but **not surjective** — Rust exposes finer-grained
errors that Python collapses. An interop layer that round-trips
Rust-originated errors through Python SHOULD preserve the
Rust-variant name in the error `context` dict.

## 5. Numerical agreement

Python evaluation uses f64 with libm `math.exp` / `math.log`.
Rust `eval_det` uses f32 with a 16-term Taylor softfloat `exp_det` and
12-term Mercator `ln_det`. Expected max relative error between
identical expression trees evaluated on both sides:

| Input class | Max relative error | Cause |
|---|---|---|
| `exp(x), x ∈ [-5, 5]` | ~1.2e-7 | f32 mantissa (~24 bits) |
| `ln(y), y ∈ [0.1, 100]` | ~1.5e-7 | f32 mantissa |
| `eml(x, y), composite` | ~2e-7 per EML op | accumulates per depth |
| depth-4 tree | ~1e-6 | cumulative |

These bounds were measured on a fixed test battery. They are the
**expected** precision gap, not a bug. Any cross-boundary test that
asserts bit-identity MUST compare with a tolerance of at least
`1e-5 * |truth|` for depth ≤ 4 trees.

## 6. What "isomorphic" means for CARL

The two implementations are **semantically isomorphic on the shared
opcode subset** `{Const, VarX, Eml}`:

- Same expression tree → same computed value, modulo f32/f64 precision.
- Same canonical tree structure → same hashable identity (via the Python
  `content_hash`-based `hash()` and the Rust FNV-1a `hash()` — different
  byte strings, different semantics, both deterministic within their
  ecosystem).

They are **NOT wire-compatible**: bytes emitted by `EMLTree.to_bytes()`
cannot be decoded by `EmlTree::decode()` and vice versa. Interop
requires an explicit translator.

The canonical cross-boundary test lives at
`tests/test_eml_py_rust_roundtrip.py` — it consumes the vectors emitted
by the Rust example `crates/terminals-core/examples/eml_vectors.rs` and
asserts numerical agreement within the bounds above.

## 7. Future alignment (non-goals for v0.9.0)

A v2 wire format that unifies the two layouts is possible (f32 on both
sides, shared magic, shared opcode bytes, shared const-pool structure)
but requires breaking the on-disk format in both ecosystems. Not in
scope for v0.9.0 — tracked as `carl.eml.v2_wire`.
