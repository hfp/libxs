# Memory Allocation (Microbenchmark)

Benchmarks pooled scratch memory allocation (`libxs_malloc` /
`libxs_free`) against the standard C library `malloc` / `free`. The
pool-based allocator recycles memory after an initial warm-up,
avoiding repeated system calls in steady state. A Fortran variant
compares `libxs_malloc` against Fortran ALLOCATE / DEALLOCATE.

## Building

```bash
cd samples/scratch
make GNU=1
```

Produces `scratch.x` (C) and, if a Fortran compiler is found,
`scratchf.x`.

## Usage (C)

```bash
./scratch.x [ncycles [max_nactive [nthreads]]]
```

| Argument    | Default | Description                                           |
|-------------|---------|-------------------------------------------------------|
| ncycles     | 100     | Number of allocation/deallocation cycles              |
| max_nactive | 4       | Max concurrent live allocations per cycle (max 24)    |
| nthreads    | 1       | OpenMP threads (clamped to omp_get_max_threads)       |

### Environment Variables

| Variable | Default | Description                                            |
|----------|---------|--------------------------------------------------------|
| CHECK    | 0       | Non-zero: memset each allocation (validates writeable) |

### Compile-Time Knobs

| Macro         | Default | Description                                        |
|---------------|---------|----------------------------------------------------|
| MAX_MALLOC_MB | 100     | Upper bound on individual allocation size in MB    |
| MAX_MALLOC_N  | 24      | Array size for random-number table and alloc slots |

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

```bash
./scratchf.x [mbytes [nrepeat]]
```

| Argument | Default | Description              |
|----------|---------|--------------------------|
| mbytes   | 100     | Allocation size in MB    |
| nrepeat  | 20      | Number of repetitions    |

Single-threaded benchmark comparing `libxs_malloc` / `libxs_free`
against Fortran ALLOCATE / DEALLOCATE. Reports per-iteration times
(ms) and an average speedup ratio.

## What Is Measured

Each cycle draws a random number of allocations (1 to max_nactive)
with random sizes (1 to MAX_MALLOC_MB MB each). The cycle allocates
all buffers, optionally touches them (CHECK), then frees them.

Both paths use the same randomized sequence. An untimed warm-up cycle
runs before the measured loops. Reported metrics:

- calls/s (kHz) -- allocation+free throughput for each allocator
- Scratch size -- high-water mark of the pool (MB)
- Malloc size -- peak aggregate allocation per cycle (MB)
- Scratch Speedup -- ratio of pool throughput to stdlib throughput
- Fair -- size-adjusted speedup: (malloc_size / scratch_size) * speedup
