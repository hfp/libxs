# Memory Comparison Benchmark

Benchmarks `libxs_diff`, `libxs_memcmp`, and the C standard library's `memcmp` for both large contiguous comparisons and many small fixed-size comparisons. A Fortran variant compares `libxs_diff` against Fortran's `ALL(a .EQ. b)` and `.NOT. ANY(a .NE. b)` array intrinsics.

## Building

```bash
cd samples/memory
make
```

Produces `memory.x` (C) and, if a Fortran compiler is found, `memoryf.x` and `matcopyf.x`.

## Usage (C)

```
./memory.x [elsize [stride [nelems [niters]]]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `elsize` | `INSIZE`/`MAXSIZE` (compile-time, 64) | Element comparison size in bytes |
| `stride` | `max(MAXSIZE, elsize)` | Distance between successive elements in bytes |
| `nelems` | `2 GB / stride` | Number of elements |
| `niters` | 5 | Repetitions (arithmetic average is reported) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRIDED` | 0 | **0**: `libxs_memcmp` compares the entire buffer in one call when `stride == elsize`, otherwise loops; **1**: sequential loop over elements; **>=2**: random-order traversal using a coprime-based permutation |
| `CHECK` | 0 | **Non-zero**: run a validation pass that injects random byte differences and cross-checks `libxs_diff` / `libxs_memcmp` against `memcmp` |
| `MISMATCH` | 0 | **0**: compare equal buffers (baseline); **1**: inject difference at first byte of each element; **2**: middle; **3**: last byte; **>=4**: random position. Tests early-out behavior of each implementation |
| `OFFSET` | 0 | Shift buffer `b` by this many bytes off its natural alignment. Tests SIMD alignment sensitivity |

### Compile-time Knobs

| Macro | Default | Description |
|-------|---------|-------------|
| `MAXSIZE` | 64 | Stride floor; elements are spaced at least this far apart |
| `INSIZE` | `MAXSIZE` | Default element size when the first argument is omitted or zero |

### Example

```bash
# 32-byte elements, sequential strided access, 10 repetitions
./memory.x 32 0 0 10

# 8-byte elements, random traversal, 3 repetitions
STRIDED=2 ./memory.x 8 0 0 3

# contiguous (single-call) comparison, ~2 GB
./memory.x 64 64 0 5

# mismatch at last byte, buffer b misaligned by 3 bytes
MISMATCH=3 OFFSET=3 ./memory.x 32 0 0 5
```

## Usage (Fortran)

```
./memoryf.x [nelements [nrepeat]]
```

Element type is hard-coded as `INTEGER(4)`. Compares ~2 GB by default… reports per-iteration times (ms) and average throughput (MB/s).

## What Is Measured

1. **`libxs_diff`** — SIMD-dispatched short-buffer comparison (SSE/AVX2/AVX-512 depending on CPU). Limited to `elsize < 256` bytes. Always called per-element (strided).
2. **`libxs_memcmp`** — SIMD-dispatched `memcmp` replacement. Used in a single call for contiguous buffers, or per-element for strided access.
3. **stdlib `memcmp`** — The C library implementation, tested under the same access pattern.

Each kernel gets freshly initialized data (`libxs_rng_seq` + `memcpy`) before its timed section, so that caches are refilled from scratch and one kernel's residency does not gift the next one. An untimed warm-up pass runs before the first measured iteration to demand-page the buffers and stabilize CPU frequency.

When `MISMATCH` is set, a controlled byte difference is injected into buffer `b` at the configured position of each element after the `memcpy`. This allows measuring early-out behavior independently from the equal-buffer baseline.

The summary reports **min / median / max** across iterations plus peak MB/s (from the minimum time), giving a more robust picture than a simple arithmetic average.

The random traversal mode (`STRIDED>=2`) uses `libxs_coprime2(size)` to produce a full permutation of the element indices, stressing TLB/cache miss behavior.

## Design Notes

### Resolved Issues

The following concerns from an earlier review have been addressed:

1. **Mismatch testing** (`MISMATCH` env var) — Buffers can now differ at a configurable position (first byte, middle, last byte, or random), testing early-out performance in addition to the equal-buffer baseline.

2. **Min / median / max reporting** — The summary section now shows the minimum, median, and maximum time across iterations, plus peak MB/s derived from the fastest run. This is more robust for memory-bandwidth benchmarks than an arithmetic mean.

3. **Warm-up iteration** — An untimed pass through `libxs_rng_seq` + `memcpy` is run before the first measured iteration to demand-page the allocations and give the CPU time to ramp its frequency.

4. **Misalignment testing** (`OFFSET` env var) — Buffer `b` can be shifted off its natural alignment to reveal SIMD alignment sensitivity.

5. **`libxs_diff` skip message** — When `elsize >= 256`, the benchmark now prints a note instead of silently omitting the column.

6. **Deterministic validation** — The `CHECK` mode now seeds `srand(42)` before the validation loop for reproducible results.

7. **POT element-size sweep** — The test script (`tests/memcmp.sh`) now includes power-of-two sizes (1, 2, 4, 8, 16, 32, 64, 128) and non-POT neighbors (3, 5, 7, 9, 15, 31, 33, 63, 65, 127) to expose stdlib regressions on short fixed-size comparisons, which historically affected certain glibc versions.

### Throughput Convention

Reported MB/s counts both buffers (`2 * nbytes / time`) since both `a` and `b` are read during comparison. This differs from some published benchmarks that report single-buffer bandwidth.
