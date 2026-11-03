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
| STATS    | 0       | Non-zero: print partition metrics (sum, imb) and displacement |

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
- With STATS=1, the sample also prints the fingerprint distance
  (Sobolev, order 4) between the original and each shuffled array,
  and between the RNG-shuffled and each deterministic result.

### Image Output

When RANDOM is not set and the element size is 1, 2, 4, or 8 bytes,
the final shuffled array is written as a PNG image per method
(shuffle_rng.png, shuffle_ds1.png, shuffle_ds2.png). The images are
shaped close to square (isqrt(n) x n/isqrt(n)); each pixel is an 8-bit
hash of the element value and its position.

## Compile-Time Knobs

| Variable    | Default   | Description                                              |
|-------------|-----------|----------------------------------------------------------|
| OMP         | 0         | Enable OpenMP (not used by this sample)                  |
| SYM         | 1         | Include debug symbols (-g)                               |
| BUBBLE_SORT | undefined | Define (-DBUBBLE_SORT) for O(n^2) inversion counter      |
