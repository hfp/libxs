# Synchronization Primitives

Micro-benchmark for the lock implementations provided by LIBXS
(`libxs_sync.h`). Measures single-thread latency (uncontended
acquire/release) and multi-thread throughput (mixed read/write
workload) for every compiled lock kind:

| Lock kind            | Description                                      |
|----------------------|--------------------------------------------------|
| LIBXS_LOCK_DEFAULT   | Compile-time default (typically atomic)          |
| LIBXS_LOCK_SPINLOCK  | OS-native or CAS-based spin lock (if available)  |
| LIBXS_LOCK_MUTEX     | OS-native mutex / pthread_mutex_t (if available) |
| LIBXS_LOCK_RWLOCK    | Reader/writer lock / pthread_rwlock_t            |

The default lock is always benchmarked. The remaining kinds are
conditionally compiled depending on platform support.

## Build

```bash
cd samples/sync
make GNU=1
```

OpenMP is enabled by default (OMP=1) for multi-threaded tests.

## Run

```bash
./sync.x [nthreads] [wratio%] [work_r] [work_w] [nlat] [ntpt]
```

| Argument | Default       | Description                                      |
|----------|---------------|--------------------------------------------------|
| nthreads | all available | Number of OpenMP threads                         |
| wratio%  | 5             | Percentage of write operations (0-100)           |
| work_r   | 100           | Simulated work inside read-critical section (cy) |
| work_w   | 10 * work_r   | Simulated work inside write-critical section     |
| nlat     | 2000000       | Iterations for latency measurement               |
| ntpt     | 10000         | Iterations per thread for throughput measurement |

### Example

```bash
./sync.x 4 5 100 1000
```

```text
LIBXS: default lock-kind "atomic" (Other)

Latency and throughput of "atomic" (default) for nthreads=4 wratio=5% ...
        ro-latency: 11 ns (call/s 91 MHz, 33 cycles)
        rw-latency: 11 ns (call/s 90 MHz, 33 cycles)
        throughput: 0 us (call/s 9128 kHz, 328 cycles)
```

## Measurement Details

- RO-latency: uncontended read-lock acquire/release pairs (4x
  unrolled), reported as nanoseconds per operation and TSC cycles.
- RW-latency: uncontended write-lock acquire/release pairs (4x
  unrolled).
- Throughput: all threads run a mixed read/write workload governed by
  wratio%. Simulated work inside the critical section is subtracted
  so only synchronization overhead is reported.
