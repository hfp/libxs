# Synchronization Primitives

Two binaries over `libxs_sync.h`: `sync_lock.x` for the lock kinds and
`sync_barrier.x` for `libxs_barrier_t`.

## sync_lock.x

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
make
```

OpenMP is enabled by default (OMP=1) for multi-threaded tests.

## Run

```bash
./sync_lock.x [nthreads] [wratio%] [work_r] [work_w] [nlat] [ntpt]
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
./sync_lock.x 4 5 100 1000
```

```text
LIBXS: default lock-kind "atomic" (Other)
LIBXS: pauses before yielding 4096 (lock), 1024 (barrier)

Latency and throughput of "atomic" (default) for nthreads=4 wratio=5% ...
        atomic ro-latency: 11 ns (call/s 91 MHz, 33 cycles)
        atomic rw-latency: 11 ns (call/s 90 MHz, 33 cycles)
        atomic throughput: 0 us (call/s 9128 kHz, 328 cycles)
```

Each measured line names the lock kind it belongs to, and the run reports the
pause budgets it was built with, because a contended figure depends on those
more than on anything else on the command line.

To reach the regime a hot internal lock lives in - a critical section of a few
instructions, taken by every thread - ask for a short one and nothing but
writes:

```bash
./sync_lock.x 384 100 1 1 100000 5000
```

The defaults instead describe a long critical section, where the lock is held
long enough that waiting for it is not the cost.

## Measurement Details

- RO-latency: uncontended read-lock acquire/release pairs (4x
  unrolled), reported as nanoseconds per operation and TSC cycles.
- RW-latency: uncontended write-lock acquire/release pairs (4x
  unrolled).
- Throughput: all threads run a mixed read/write workload governed by
  wratio%. Simulated work inside the critical section is subtracted
  so only synchronization overhead is reported.

## sync_barrier.x

Cost of a rendezvous over `libxs_barrier_t`, and of the broadcast that
hands one task's value to the rest.

```bash
./sync_barrier.x [nthreads] [nrepeat]
```

| Argument | Default       | Description                          |
|----------|---------------|--------------------------------------|
| nthreads | all available | Tasks in the team                    |
| nrepeat  | 100000        | Rendezvous per measurement           |

The team is asked how large it actually is rather than told: a barrier
initialized for more tasks than the runtime grants waits for one that
never arrives.

Both are checked and the binary fails if either is wrong, because a
rendezvous that releases a task early is fast and worthless: every task
stamps a slot of its own and reads all of them after the rendezvous,
and the broadcast publishes a value that changes each round.

The stamp check runs in a pass of its own, outside the timing. Reading
every stamp is work proportional to the team, so at a wide one it costs
more than the rendezvous it checks, and timing the two together would
report the check as if it were the barrier.

Expect a flat barrier to cost more once the team reaches the number of
cores: every waiting task spins, so a team that over-subscribes the
machine spends its time taking cycles away from the task it is waiting
for. That is a property of the primitive, not of the measurement, and
it is the reason to keep a rendezvous out of a tight loop.
