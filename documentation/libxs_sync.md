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
| `LIBXS_ATOMIC_FETCHADD(SRC, VALUE, KIND)` | Fetch-and-add |
| `LIBXS_ATOMIC_ADDFETCH(SRC, VALUE, KIND)` | Add-and-fetch |
| `LIBXS_ATOMIC_SUBFETCH(SRC, VALUE, KIND)` | Subtract-and-fetch |
| `LIBXS_ATOMIC_TRYLOCK(DST, KIND)` | Try-lock (returns acquired state) |
| `LIBXS_ATOMIC_ACQUIRE(DST, KIND)` | Spin-acquire |
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

## File Locking

```C
LIBXS_FLOCK(FILE)
LIBXS_FUNLOCK(FILE)
```

Per-file locking for thread-safe I/O. Maps to `flockfile`/`funlockfile` on POSIX, `_lock_file`/`_unlock_file` on Windows, or no-ops when synchronization is disabled.

## Functions

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
