 Microbenchmark

This code sample benchmarks the performance of the registry (key-value store) dispatch path. Various scenarios are measured to characterize the registry under different access patterns and concurrency levels.

## Programs

### registry.x (C)

```
./registry.x [total [nrepeat [nthreads]]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `total` | 10000 | Number of unique keys to register (minimum 2) |
| `nrepeat` | 10 | Number of repeat iterations for lookup phases |
| `nthreads` | max available | Number of OpenMP threads (clamped to available) |

**Measurements:**

* Duration to register (insert) all keys into the registry (write path).
* Duration to look up keys with a shuffled access pattern that defeats the thread-local cache ("cold lookup").
* Duration to look up keys with a sequential/repeating pattern that hits the thread-local cache ("cached lookup").
* Duration of multi-threaded parallel reads across all threads.
* Duration of contended parallel writes (each thread registers its own key range).
* Duration of mixed read/write: one writer thread registering keys while remaining threads read concurrently.

The multi-threaded benchmarks (parallel reads, contended writes, mixed) require at least 2 threads and are skipped when running single-threaded.

### registryf.x (Fortran)

```
./registryf.x
```

A Fortran variant using hardcoded parameters (10000 keys, 10 repeats). Accepts no command-line arguments. Measures three scenarios: registration, cold lookup (coprime-shuffled access), and cached lookup (cycling over 16 entries). Also verifies all registered entries via `libxs_registry_has()`.

## Scaling Behavior

In case of a multi-threaded benchmark, the timings represent contended requests. For thread-scaling, read-only accesses (lookup) stay roughly constant in duration per-op due to the thread-local cache, whereas write-accesses (registration) are serialized and duration scales with the number of threads.
