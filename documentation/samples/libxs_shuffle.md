# Shuffle

Benchmarks three shuffling strategies and compares their throughput:

| Label       | Method                                                            |
|-------------|-------------------------------------------------------------------|
| RNG-shuffle | Fisher-Yates via `libxs_rng_u32` (reference baseline)             |
| DS1-shuffle | `libxs_shuffle` -- deterministic in-place (coprime stride)        |
| DS2-shuffle | `libxs_shuffle2` -- deterministic out-of-place (src to dst)       |

Each iteration works on an array of unsigned integers initialized to
0..n-1. On the final iteration the shuffled result can optionally be
written as an MHD image file or analyzed for randomness quality.

## Build

```bash
cd samples/shuffle
make GNU=1
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
| repeat     | 3                | Timed iterations (first is warmup)     |

### Environment Variables

| Variable | Default | Description                                                   |
|----------|---------|---------------------------------------------------------------|
| RANDOM   | 0       | Non-zero: count inversions and report randomness (rand=%)     |
| SPLIT    | 1       | Partitioning depth for the STATS metrics                      |
| STATS    | 0       | Non-zero: print partition imbalance (imb) and distance (dst)  |

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

Reported bandwidth is 2 * data-size / time (in MB/s). The factor of
two accounts for reading and writing each element during the shuffle.
The first iteration is excluded from the average.

Optional quality metrics:

- rand -- Enabled by RANDOM=1. Inversions are counted via merge-sort
  and compared against the expected count for a random permutation
  (n*(n-1)/4) to give a percentage.
- dst -- Manhattan distance of the element sum from the expected
  uniform value, split into hierarchical partitions (SPLIT).
- imb -- Partition imbalance, measuring how unevenly element sums
  are distributed across sub-partitions (SPLIT).

### MHD Image Output

When RANDOM is not set and the element size is 1, 2, 4, or 8 bytes,
the final shuffled array is written as an MHD image per method
(shuffle_rng.mhd, shuffle_ds1.mhd, shuffle_ds2.mhd). The images are
shaped close to square (isqrt(n) x n/isqrt(n)) and converted to
unsigned 8-bit via modulus.

## Compile-Time Knobs

| Variable    | Default   | Description                                              |
|-------------|-----------|----------------------------------------------------------|
| OMP         | 0         | Enable OpenMP (not used by this sample)                  |
| SYM         | 1         | Include debug symbols (-g)                               |
| BUBBLE_SORT | undefined | Define (-DBUBBLE_SORT) for O(n^2) inversion counter      |
