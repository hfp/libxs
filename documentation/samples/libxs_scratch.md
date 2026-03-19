# Memory Allocation (Microbenchmark)

Benchmarks pooled scratch memory allocation (`libxs_malloc` / `libxs_free`) against the standard C library `malloc` / `free`. The pool-based allocator recycles memory after an initial warm-up, avoiding repeated system calls in steady state. The benchmark measures allocation throughput (calls/s) under a randomized mix of allocation sizes and counts, optionally with multiple threads. A Fortran variant compares `libxs_malloc` against Fortran `ALLOCATE` / `DEALLOCATE`.

## Building

```bash
cd samples/scratch
make
```

Produces `scratch.x` (C) and, if a Fortran compiler is found, `scratchf.x`.

## Usage (C)

```
./scratch.x [ncycles [max_nactive [nthreads]]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ncycles` | 100 | Number of allocation/deallocation cycles |
| `max_nactive` | 4 | Maximum number of concurrent live allocations per cycle (clamped to `MAX_MALLOC_N`, default 24) |
| `nthreads` | 1 | OpenMP threads (clamped to `omp_get_max_threads()`) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECK` | 0 | Non-zero: `memset` each allocation to detect mapping errors (slower, validates that pages are writable) |

### Compile-time Knobs

| Macro | Default | Description |
|-------|---------|-------------|
| `MAX_MALLOC_MB` | 100 | Upper bound on individual allocation size in MB (each allocation is 1–`MAX_MALLOC_MB` MB) |
| `MAX_MALLOC_N` | 24 | Array size for the random-number table and per-cycle allocation slots |

### Example

```bash
# default: 100 cycles, up to 4 active allocations, 1 thread
./scratch.x

# heavy load: 500 cycles, up to 12 active allocations, 4 threads
./scratch.x 500 12 4

# validate pages are writable
CHECK=1 ./scratch.x 43 8 4
```

## Usage (Fortran)

```
./scratchf.x [mbytes [nrepeat]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `mbytes` | 100 | Allocation size in MB |
| `nrepeat` | 20 | Number of repetitions |

A simpler single-threaded benchmark that times a single `libxs_malloc` / `libxs_free` pair against Fortran `ALLOCATE` / `DEALLOCATE` per iteration. Reports per-iteration times (ms) and an average speedup ratio.

```bash
# default: 100 MB, 20 repetitions
./scratchf.x

# 256 MB, 50 repetitions
./scratchf.x 256 50
```

## What Is Measured

Each cycle draws a random number of allocations (1 to `max_nactive`) with random sizes (1–`MAX_MALLOC_MB` MB each). The cycle allocates all buffers, optionally touches them (`CHECK`), then frees them in order.

1. **Standard malloc** — `malloc(nbytes)` / `free(p)` (or TBB / Intel KMP variants depending on compile-time configuration). Runs first to avoid warm-page bias from the pool path.
2. **libxs scratch pool** — `libxs_malloc(pool, nbytes, 0)` / `libxs_free(p)`. The pool grows on first use and recycles memory thereafter.

An untimed warm-up cycle runs both allocators before the measured loops to demand-page memory and stabilize CPU frequency. Both `malloc` and `free` (or their equivalents) are individually timed and included in the throughput calculation.

Both paths use the same randomized sequence and are timed with `libxs_timer_ncycles`. Reported metrics:

| Metric | Description |
|--------|-------------|
| **calls/s (kHz)** | Allocation+free throughput for each allocator |
| **Scratch size** | High-water mark of the pool (MB) |
| **Malloc size** | Peak aggregate allocation per cycle (MB) |
| **Scratch Speedup** | Ratio of pool throughput to stdlib throughput |
| **Fair** | Size-adjusted speedup: `(malloc_size / scratch_size) * speedup`, accounting for the pool's memory overhead |

