 Microbenchmark

This code sample benchmarks the performance of the registry (key-value store) dispatch path. Various scenarios are measured to characterize the registry under different access patterns and concurrency levels.

**Command Line Interface (CLI)**

* Optionally takes the number of unique keys to register (default:&#160;10000).
* Optionally takes the number of repeat iterations for lookup phases (default:&#160;10).
* Optionally takes the number of threads (default:&#160;max available).

**Measurements (Benchmark)**

* Duration to register (insert) all keys into the registry (write path).
* Duration to look up keys with a shuffled access pattern that defeats the thread-local cache ("cold lookup").
* Duration to look up keys with a sequential/repeating pattern that hits the thread-local cache ("cached lookup").
* Duration of multi-threaded parallel reads across all threads.
* Duration of contended parallel writes (each thread registers its own key range).
* Duration of mixed read/write: one writer thread registering keys while remaining threads read concurrently.

In case of a multi-threaded benchmark, the timings represent contended requests. For thread-scaling, read-only accesses (lookup) stay roughly constant in duration per-op due to the thread-local cache, whereas write-accesses (registration) are serialized and duration scales with the number of threads.

