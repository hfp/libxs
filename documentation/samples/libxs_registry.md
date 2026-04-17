# Registry Microbenchmark

Benchmarks the performance of the LIBXS registry (key-value store)
dispatch path under different access patterns and concurrency levels.

## Programs

### registry.x (C)

```bash
./registry.x [total [nrepeat [nthreads]]]
```

| Argument | Default       | Description                               |
|----------|---------------|-------------------------------------------|
| total    | 10000         | Number of unique keys to register (min 2) |
| nrepeat  | 10            | Repeat iterations for lookup phases       |
| nthreads | max available | Number of OpenMP threads                  |

Measurements:

- Duration to register (insert) all keys into the registry.
- Cold lookup with shuffled access pattern (defeats thread-local cache).
- Cached lookup with sequential/repeating pattern (hits thread-local cache).
- Multi-threaded parallel reads across all threads.
- Contended parallel writes (each thread registers its own key range).
- Mixed read/write: one writer thread while remaining threads read.

The multi-threaded benchmarks require at least 2 threads and are
skipped when running single-threaded.

### registryf.x (Fortran)

```bash
./registryf.x
```

Fortran variant with hardcoded parameters (10000 keys, 10 repeats).
Measures registration, cold lookup, and cached lookup.

## Scaling Behavior

Read-only accesses stay roughly constant in per-op duration due to
the thread-local cache. Write accesses are serialized and duration
scales with the number of threads.
