# Synchronization

Header: `libxs_sync.h`

Thread-local storage, atomic operations, lock abstractions (spin/mutex/rwlock/atomic), file locking, and stdio synchronization.

## Thread-Local Storage

```C
LIBXS_TLS
```

Storage-class qualifier for thread-local variables. Maps to `__thread`, `__declspec(thread)`, or `thread_local` depending on the compiler. Defined as empty when TLS is unavailable or disabled (`LIBXS_NO_TLS`).

## Atomic Operations

The header provides a suite of macros for atomic loads, stores, compare-and-swap, and arithmetic. The implementation selects among GCC builtins (`__atomic_*`), legacy GCC sync builtins, Windows Interlocked intrinsics, or no-op fallbacks depending on the compiler and `LIBXS_SYNC` setting.

| Macro | Description |
|:------|:------------|
| `LIBXS_ATOMIC_LOAD(SRC, KIND)` | Atomic load |
| `LIBXS_ATOMIC_STORE(DST, VALUE, KIND)` | Atomic store |
| `LIBXS_ATOMIC_STORE_ZERO(DST, KIND)` | Atomic store of zero |
| `LIBXS_ATOMIC_CMPSWP(DST, OLDVAL, NEWVAL, KIND)` | Compare-and-swap |
| `LIBXS_ATOMIC_FETCH_OR(DST, VALUE, KIND)` | Fetch-and-or |
| `LIBXS_ATOMIC_FETCH_ADD(DST, VALUE, KIND)` | Fetch-and-add |
| `LIBXS_ATOMIC_FETCH_SUB(DST, VALUE, KIND)` | Fetch-and-subtract |
| `LIBXS_ATOMIC_ADD_FETCH(DST, VALUE, KIND)` | Add-and-fetch |
| `LIBXS_ATOMIC_SUB_FETCH(DST, VALUE, KIND)` | Subtract-and-fetch |
| `LIBXS_ATOMIC_TRYLOCK(DST, KIND)` | Try-lock (returns acquired state) |
| `LIBXS_ATOMIC_ACQUIRE(DST, NPAUSE, KIND)` | Spin-acquire with backoff |
| `LIBXS_ATOMIC_RELEASE(DST, KIND)` | Release (store zero) |
| `LIBXS_ATOMIC_SYNC(KIND)` | Full memory fence |

The `KIND` parameter selects the memory order but is ignored on most backends (sequential consistency is always used).

## Lock Abstraction

```C
LIBXS_LOCK_TYPE(KIND)
LIBXS_LOCK_INIT(KIND, LOCK, ATTR)
LIBXS_LOCK_DESTROY(KIND, LOCK)
LIBXS_LOCK_ACQUIRE(KIND, LOCK)
LIBXS_LOCK_TRYLOCK(KIND, LOCK)
LIBXS_LOCK_RELEASE(KIND, LOCK)
```

Generic lock interface parameterized by `KIND`:

| KIND | Backend |
|:-----|:--------|
| `LIBXS_LOCK_SPINLOCK` | Atomic spin-lock |
| `LIBXS_LOCK_MUTEX` | Pthreads / Windows CRITICAL_SECTION |
| `LIBXS_LOCK_RWLOCK` | Pthreads reader-writer lock |
| `LIBXS_LOCK_ATOMIC` | Lightweight atomic lock |

The default lock kind used by the library is `LIBXS_LOCK` (resolves to one of the above based on build configuration).

```C
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_t;
```

General-purpose lock type. Instances of `libxs_lock_t` are used by the registry and other library components; users may also create their own.

Reader-writer variants are available for `LIBXS_LOCK_RWLOCK`:

```C
LIBXS_LOCK_ACQREAD(KIND, LOCK)
LIBXS_LOCK_RELREAD(KIND, LOCK)
LIBXS_LOCK_TRYREAD(KIND, LOCK)
```

## Barrier

```C
libxs_barrier_t barrier;
libxs_barrier_init(&barrier, ntasks);
libxs_barrier_wait(&barrier);
value = libxs_barrier_bcast(&barrier, tid, root, value);
```

Rendezvous over a fixed number of tasks. The caller owns the storage and nothing is allocated inside, so the barrier can live in whatever structure already describes the team; `libxs_barrier_init` is called once, before any task waits, rather than by every task. A task is only a number here, so the team need not be a thread team.

It is flat, one counter for the whole team, which suits a team that meets between stages of work rather than inside a tight loop. Every waiting task spins, so a team that over-subscribes the machine takes cycles away from the task it is waiting for.

`libxs_barrier_bcast` waits and returns the value the `root` task carried into the call, which is how one task hands a decision to the rest. Writing that decision into a word of one's own and reading it after a plain `libxs_barrier_wait` is not equivalent and is not safe: nothing stops the publisher from writing the word again, and a reader still between the release and its own load then reads the newer value. The broadcast alternates between two slots by the parity of the rendezvous, so a late reader still finds what was published to it.

Note that a lock is not a barrier and cannot stand in for one. A task holding a lock while waiting for tasks that need that lock to arrive does not proceed, so the two serve opposite purposes: a lock admits one task and needs to know nothing about the team, a barrier requires the whole team.

## File Locking

```C
LIBXS_FLOCK(FILE)
LIBXS_FUNLOCK(FILE)
```

Per-file locking for thread-safe I/O. Maps to `flockfile`/`funlockfile` on POSIX, `_lock_file`/`_unlock_file` on Windows, or no-ops when synchronization is disabled.

## Functions

```C
unsigned int libxs_nranks(void);
```

Return the number of MPI ranks.

```C
unsigned int libxs_nrank(void);
```

Return the MPI rank of the calling process.

```C
unsigned int libxs_rid(void);
```

Return a rank ID of the calling process.

```C
unsigned int libxs_pid(void);
```

Return the process ID of the calling process.

```C
unsigned int libxs_tid(void);
```

Return a zero-based, consecutive thread ID for the calling thread. TID = 0 does not necessarily correspond to the main thread.

```C
void libxs_stdio_acquire(void);
void libxs_stdio_release(void);
```

Acquire/release a global lock around console output. The macros `LIBXS_STDIO_ACQUIRE()` and `LIBXS_STDIO_RELEASE()` expand to these calls.
