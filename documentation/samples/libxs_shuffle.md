# Shuffle

Benchmarks three shuffling strategies and compares their throughput:

| Label | Method |
|:------|:-------|
| **RNG-shuffle** | Fisher–Yates shuffle using `libxs_rng_u32` for random indices and `LIBXS_MEMSWP` for element swaps (reference baseline). |
| **DS1-shuffle** | `libxs_shuffle` – deterministic in-place shuffle based on a coprime stride. |
| **DS2-shuffle** | `libxs_shuffle2` – deterministic out-of-place shuffle (source → destination). |

Each iteration works on an array of unsigned integers initialized to `0 .. n-1`. On the final iteration the shuffled result can optionally be written as an MHD image file (one per method) or analyzed for randomness quality.

## Build

```bash
cd samples/shuffle
make
```

## Usage

```
./shuffle.x [nelems [elemsize [niters [repeat]]]]
```

| Positional | Default | Description |
|:-----------|:--------|:------------|
| `nelems`   | 64 MB / `elemsize` | Number of elements. |
| `elemsize` | 4       | Size of each element in bytes (1, 2, 4, or 8). |
| `niters`   | 1       | Number of shuffle passes per timed measurement (averaged). |
| `repeat`   | 3       | Number of timed iterations; the first is treated as warm-up. |

### Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `RANDOM` | 0 (off) | If nonzero, count inversions (via merge-sort) of the shuffled data and report a randomness percentage (`rand=%`) instead of writing MHD files. |
| `SPLIT`  | 1       | Partitioning depth for the `STATS` metrics (imbalance and distance). |
| `STATS`  | 0 (off) | If nonzero, compute and print partition imbalance (`imb`) and Manhattan distance (`dst`) metrics per method. |

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

Reported bandwidth is `2 × data-size / time` (in MB/s). The factor of two accounts for reading and writing each element during the shuffle. For the RNG (Fisher–Yates) method this is an approximation: elements where the random index equals the current index (~37%) are not swapped, so actual data movement is slightly lower. The first iteration is excluded from the arithmetic average.

### Optional Quality Metrics

* **`rand`** – Enabled by `RANDOM=1`. After shuffling, inversions are counted via merge-sort (O(n log n)) and compared against the expected count for a random permutation (n(n−1)/4) to give a percentage. For the stochastic RNG-shuffle, quality metrics are averaged across all non-warm-up iterations. For the deterministic DS1/DS2 methods, metrics are taken from the final iteration only.
* **`dst`** – Manhattan distance of the element sum from the expected uniform value, split into hierarchical partitions (`SPLIT`).
* **`imb`** – Partition imbalance, measuring how unevenly element sums are distributed across sub-partitions (`SPLIT`).

### MHD Image Output

When `RANDOM` is not set and the element size matches a supported type (1, 2, 4, or 8 bytes), the final shuffled array is written as an MHD image:

| File | Source |
|:-----|:-------|
| `shuffle_rng.mhd` | RNG-shuffle result |
| `shuffle_ds1.mhd` | DS1-shuffle result |
| `shuffle_ds2.mhd` | DS2-shuffle result |

The images are shaped as close to square as possible (`isqrt(n) × n/isqrt(n)`) and converted to unsigned 8-bit via modulus, allowing visual inspection of shuffle uniformity.

## Compile-Time Knobs

The Makefile inherits settings from `Makefile.inc`. Relevant flags:

| Variable | Default | Effect |
|:---------|:--------|:-------|
| `OMP`    | 0       | Enable OpenMP (not used by this sample). |
| `SYM`    | 1       | Include debug symbols (`-g`). |
| `BUBBLE_SORT` | *undefined* | Define (`-DBUBBLE_SORT`) to use O(n²) bubble-sort for the `RANDOM` metric instead of the default O(n log n) merge-sort inversion counter. |
