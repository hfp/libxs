# Shuffle

Benchmarks three shuffling strategies and compares their throughput:

| Label       | Method                                                            |
|-------------|-------------------------------------------------------------------|
| RNG-shuffle | Fisher-Yates via `libxs_rng_u32` (reference baseline)             |
| DS1-shuffle | `libxs_shuffle` — deterministic in-place (coprime stride)         |
| DS2-shuffle | `libxs_shuffle2` — deterministic out-of-place (src to dst)        |

Each iteration works on an array of unsigned integers initialized to
0..n-1. On the final iteration the shuffled result can be written as a
PNG image (see below). Every run ends with round-trip checks of
`libxs_shuffle2`/`libxs_unshuffle2`, once with the given offset and once
with offset 42, each reported as PASS or FAIL.

## Build

```bash
cd samples/shuffle
make
```

OpenMP is enabled by default; `make OMP=0` retains a serial build.

## Usage

```bash
./shuffle.x [nelems [elemsize [niters [repeat]]]]
```

| Positional | Default          | Description                            |
|------------|------------------|----------------------------------------|
| nelems     | 64 MB / elemsize | Number of elements                     |
| elemsize   | 4                | Element size in bytes (1, 2, 4, or 8)  |
| niters     | 1                | Shuffle passes per timed measurement   |
| repeat     | 3                | Timed iterations after one warm-up     |

### Environment Variables

| Variable | Default | Description                                                   |
|----------|---------|---------------------------------------------------------------|
| BIAS     | 0       | Coprime selection in [-1, +1] (`libxs_coprime_bias`)          |
| OFFSET   | 0       | Offset of the deterministic shuffles                          |
| RANDOM   | 0       | Non-zero: count inversions and report randomness (rand=%)     |
| SPLIT    | 1       | Partitioning depth for the STATS metrics                      |
| STATS    | 0       | Non-zero: print partition metrics, displacement, and cyclic lag energies |

BIAS=-1 selects the smallest non-trivial coprime, 0 one near sqrt(n), and
+1 one near n/2. If BIAS, OFFSET, RANDOM or STATS is non-zero, the sample
first prints n, the selected coprime, bias and offset.

### Example

```bash
# Default run (64 MB of 4-byte elements, 3 repeats)
./shuffle.x

# 1M elements of 8 bytes, 4 shuffle passes, 5 repeats
./shuffle.x 1000000 8 4 5

# Enable randomness quality metric
RANDOM=1 ./shuffle.x 10000 4 1 3

# Enable partition statistics with split depth 2
STATS=1 SPLIT=2 ./shuffle.x
```

## What Is Measured

Reported bandwidth is 2 * data-size / time (in MB/s, with 1 MB = 2^20
Bytes). The factor of two accounts for reading and writing each element
during the shuffle. The sample runs `repeat + 1` iterations: the first
is a warm-up, and when `repeat` is larger than 1 the average over the
remaining `repeat` iterations is printed at the end.

Optional quality metrics:

- rand — Enabled by RANDOM=1. Inversions are counted via merge-sort
  and compared against the expected count for a random permutation
  (n*(n-1)/4) to give a percentage, which exceeds 100% for a permutation
  with more inversions than expected.
- sum — Distance of the element sum from the expected uniform value,
  split into hierarchical partitions (SPLIT). At depth 1 the sum of all
  elements is fixed, hence zero.
- disp — Mean distance of each value from its position, relative to n.
  A uniform draw expects 33.3%. Not printed if the element size repeats
  values.
- imb — Partition imbalance, measuring how unevenly element sums
  are distributed across sub-partitions (SPLIT).
- With STATS=1 and distinct identity labels, the sample prints cyclic
  squared-increment energies at lags 1, 2, 3, 7, 16, and 63 below n.
  `Q` divides the energy by the exact random-permutation expectation,
  n*(n+1)/6. The deterministic paths additionally check their measured
  energies against d*(n-d), with d=(C^niters*lag) mod n.
  Energies do not depend on OFFSET; large Q is not necessarily better.
  These metrics require no fingerprint computation.

## Irregular Work Scheduling

Set `LIBXS_SHUFFLE_WORK=spmv` or `gemm` to replace the shuffle-throughput
run with an irregular-work scheduling benchmark. The first positional
argument is the number of tasks (default 4096); the second and third,
if supplied, must be 4 and 1. The fourth remains the number of timed
repetitions after one warm-up.

```bash
# Irregular CSR rows; increase the size to investigate bandwidth limits.
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  LIBXS_SHUFFLE_WORK=spmv ./shuffle.x 1000000 4 1 5

# Cache-resident 16x16 DGEMMs, with a variable number of calls per task.
LIBXS_GEMM_BACKEND=4 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  LIBXS_SHUFFLE_WORK=gemm ./shuffle.x 4096 4 1 5
```

SpMV uses a generated CSR matrix, hashed column indices, and one output
per row. Its cost units are nonzeros. GEMM tasks share read-only resident
operands, accumulate into private 16x16 matrices, and store one checksum
per task. Their cost units are DGEMM calls (8192 floating-point operations
each). GEMM requires `LIBXS_GEMM_BACKEND=4`, selecting LIBXS's built-in,
serial kernel rather than a potentially threaded external BLAS.

| Variable | Default | Description |
|----------|---------|-------------|
| LIBXS_SHUFFLE_WORK | unset | `spmv` or `gemm`; unset preserves the original benchmark |
| LIBXS_SHUFFLE_PATTERN | hotspot | `uniform`, `ramp`, `hotspot`, or `periodic` task costs |
| LIBXS_SHUFFLE_LOW | 8 / 1 | Minimum nonzeros per row / DGEMM calls per task |
| LIBXS_SHUFFLE_HIGH | 128 / 16 | Maximum cost, at least LOW |
| LIBXS_SHUFFLE_WIDTH | max(n/8,1) | Number of expensive tasks in the hotspot |
| LIBXS_SHUFFLE_SHIFT | 0 | Cyclic phase of the cost profile, from 0 to n-1 |
| LIBXS_SHUFFLE_PERIOD | min(n,64) | Spacing of expensive tasks in the periodic profile |
| LIBXS_SHUFFLE_SEED | 1 | Seed for the stored Fisher-Yates permutation |

BIAS and OFFSET also apply. Uniform costs are LOW everywhere; the ramp
increases from LOW to HIGH over the shifted task domain. The hotspot has
WIDTH consecutive HIGH-cost tasks; the periodic profile has one HIGH-cost
task every PERIOD positions. Other tasks cost LOW. Costs must be between
1 and 1000000. Task count must be at least two; task count and seed
must fit signed int.

Every case runs six assignments:

- `blocked`: balanced contiguous task counts in original order.
- `cyclic`: worker t visits t, t+T, ... .
- `affine`: blocked traversal positions mapped through SHUFFLE_INDEX.
- `table`: blocked traversal positions mapped through a seeded Fisher-Yates table.
- `dynamic`: OpenMP dynamic scheduling with chunk size one.
- `guided`: OpenMP guided scheduling with minimum chunk size one.

Output records beginning with `work,` use the printed CSV header.
`seconds` includes the OpenMP parallel region and per-worker accounting,
but excludes input construction, index setup, output initialization, and
verification. `index_setup_s` reports coprime selection or table construction
separately; the table is reused across repetitions. `index_bytes` counts
the permutation table only, not the common workload and verification arrays.
For a one-use comparison, add index setup to execution time.

`units` is total work and `max_units` the largest assigned worker load.
`imbalance_pct` is 100*(T*max_units/units-1), using the actual team size.
Additional records have the form
`worker,kernel,pattern,N,C,B,seed,low,high,width,shift,period,method,threads,repeat,tid,tasks,units,seconds`;
their times end before the final team barrier. Worksharing imbalance is
the observed assignment, not a prediction of runtime or bandwidth.
Method order rotates between repetitions. Every execution, including
warm-up, checks all task outputs against an independent reference and
checks total task/work counts; a failure exits nonzero.

Compare multiple thread counts, domain sizes, phases, offsets, and random
seeds. The periodic profile is a resonance control, not a promise of an
affine win. Small SpMV instances may fit in cache; only sufficiently large
instances can investigate memory-system limits. GEMM is a resident compute
comparison, not a representative end-to-end scientific application.

### Image Output

When RANDOM is not set and the element size is 1, 2, 4, or 8 bytes,
the final shuffled array is written as a PNG image per method
(shuffle_rng.png, shuffle_ds1.png, shuffle_ds2.png). The images are
shaped close to square (isqrt(n) x n/isqrt(n)); each pixel is an 8-bit
hash of the element value and its position.

## Compile-Time Knobs

| Variable    | Default   | Description                                              |
|-------------|-----------|----------------------------------------------------------|
| OMP         | 1         | Enable OpenMP for the irregular-work benchmark          |
| SYM         | 1         | Include debug symbols (-g)                               |
| BUBBLE_SORT | undefined | Define (-DBUBBLE_SORT) for O(n^2) inversion counter      |
