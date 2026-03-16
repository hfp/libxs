# Synchronization Primitives

Micro-benchmark for the lock implementations provided by LIBXS (`libxs_sync.h`).
The program measures **single-thread latency** (uncontended acquire/release)
and **multi-thread throughput** (mixed read/write workload) for every lock kind
compiled into the library:

| Lock kind | Description |
|---|---|
| `LIBXS_LOCK_DEFAULT` | Compile-time default (typically `atomic`) |
| `LIBXS_LOCK_SPINLOCK` | OS-native or CAS-based spin lock |
| `LIBXS_LOCK_MUTEX` | OS-native mutex (`pthread_mutex_t`) |
| `LIBXS_LOCK_RWLOCK` | Reader/writer lock (atomic or `pthread_rwlock_t`) |

## Build

```bash
cd samples/sync
make GNU=1
```

OpenMP is enabled by default (`OMP=1`) for multi-threaded throughput tests.

## Run

```
./sync.x [nthreads] [wratio%] [work_r] [work_w] [nlat] [ntpt]
```

| Argument | Default | Description |
|---|---|---|
| `nthreads` | all available | Number of OpenMP threads |
| `wratio%` | 5 | Percentage of write operations (0-100) |
| `work_r` | 100 | Simulated work inside read-critical section (cycles) |
| `work_w` | 10 x work_r | Simulated work inside write-critical section (cycles) |
| `nlat` | 2000000 | Number of iterations for latency measurement |
| `ntpt` | 10000 | Number of iterations per thread for throughput measurement |

### Example

```
$ ./sync.x 4 5 100 1000
LIBXS: default lock-kind "atomic" (Other)

Latency and throughput of "atomic" (default) for nthreads=4 wratio=5% ...
        ro-latency: 11 ns (call/s 91 MHz, 33 cycles)
        rw-latency: 11 ns (call/s 90 MHz, 33 cycles)
        throughput: 0 us (call/s 9128 kHz, 328 cycles)
...
```

## Measurement Details

- **RO-latency**: Uncontended read-lock acquire/release pairs (4x unrolled),
  reported as nanoseconds per operation and TSC cycles.
- **RW-latency**: Uncontended write-lock acquire/release pairs (4x unrolled).
- **Throughput**: All threads run a mixed read/write workload governed by
  `wratio%`. Simulated work inside the critical section is subtracted
  so only synchronization overhead is reported.
