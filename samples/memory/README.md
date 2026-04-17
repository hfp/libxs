# Memory Comparison Benchmark

Benchmarks `libxs_diff`, `libxs_memcmp`, and the C standard library's
`memcmp` for both large contiguous comparisons and many small
fixed-size comparisons. A Fortran variant compares `libxs_diff`
against Fortran array intrinsics.

## Building

```bash
cd samples/memory
make GNU=1
```

Produces `memcmp.x` (C) and, if a Fortran compiler is found,
`memcmpf.x` and `matcpyf.x`.

## Usage (C)

```bash
./memcmp.x [elsize [stride [nelems [niters]]]]
```

| Argument | Default                      | Description                          |
|----------|------------------------------|--------------------------------------|
| elsize   | INSIZE/MAXSIZE (compile, 64) | Element comparison size in bytes     |
| stride   | max(MAXSIZE, elsize)         | Distance between successive elements |
| nelems   | 2 GB / stride                | Number of elements                   |
| niters   | 5                            | Repetitions (arithmetic average)     |

### Environment Variables

| Variable | Default | Description                                                           |
|----------|---------|-----------------------------------------------------------------------|
| STRIDED  | 0       | 0: one-call when stride==elsize, else loop; 1: seq loop; >=2: random  |
| CHECK    | 0       | Non-zero: validation pass (inject diffs, cross-check implementations) |
| MISMATCH | 0       | 0: equal buffers; 1: first byte; 2: middle; 3: last; >=4: random      |
| OFFSET   | 0       | Shift buffer b off its natural alignment by this many bytes           |

### Compile-Time Knobs

| Macro   | Default | Description                                   |
|---------|---------|-----------------------------------------------|
| MAXSIZE | 64      | Stride floor (elements spaced >= this far)    |
| INSIZE  | MAXSIZE | Default element size when argument is omitted |

### Example

```bash
# 32-byte elements, sequential strided access, 10 repetitions
./memcmp.x 32 0 0 10

# 8-byte elements, random traversal, 3 repetitions
STRIDED=2 ./memcmp.x 8 0 0 3

# contiguous (single-call) comparison, ~2 GB
./memcmp.x 64 64 0 5

# mismatch at last byte, buffer b misaligned by 3 bytes
MISMATCH=3 OFFSET=3 ./memcmp.x 32 0 0 5
```

Environment variables and compile-time macros apply only to `memcmp.x`.

## Usage (Fortran -- memcmpf)

```bash
./memcmpf.x [nelements [nrepeat]]
```

Element type is hard-coded as INTEGER(4). Compares ~2 GB by default.
Reports per-iteration times (ms) and average throughput (MB/s).

## Usage (Fortran -- matcpyf)

```bash
./matcpyf.x [m [n [ldi [ldo [nrepeat [nmb]]]]]]
```

| Argument | Default        | Description                 |
|----------|----------------|-----------------------------|
| m        | 4096           | First matrix dimension      |
| n        | m              | Second matrix dimension     |
| ldi      | m (enforced m) | Leading dimension of input  |
| ldo      | ldi            | Leading dimension of output |
| nrepeat  | 2              | Number of repetitions       |
| nmb      | 2048           | Memory budget in MB         |

Benchmarks `libxs_matcopy` and `libxs_matcopy_task` for matrix copy
and zeroing. When the matrix is square and ldi==ldo, also benchmarks
`libxs_otrans` and `libxs_otrans_task` for out-of-place transpose.

## What Is Measured

Three comparison implementations are benchmarked:

- `libxs_diff` -- SIMD-dispatched short-buffer comparison
  (SSE/AVX2/AVX-512). Limited to elsize < 256 bytes, per-element.
- `libxs_memcmp` -- SIMD-dispatched memcmp replacement. Single call
  for contiguous buffers, or per-element for strided access.
- stdlib `memcmp` -- C library implementation under the same pattern.

Each kernel gets freshly initialized data before its timed section.
An untimed warm-up pass runs before the first measured iteration.
The summary reports min / median / max across iterations plus peak
MB/s. Reported MB/s counts both buffers (2 * nbytes / time).
