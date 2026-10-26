# Synchronization

Header: `libxs_sync.h`

Thread-local storage, atomic operations, lock abstractions (spin/mutex/rwlock/atomic), file locking, and stdio synchronization.

## Thread-Local Storage

```C
LIBXS_TLS
```

Storage-class qualifier for thread-local variables. Maps to `__thread`, `__declspec(thread)`, or `thread_local` depending on the compiler. Defined as empty when TLS is unavailable or disabled (`LIBXS_NO_TLS`).

Key-based thread-local values map to POSIX thread keys or Windows TLS slots (defined unless `LIBXS_SYNC` is 0); return values are those of the platform:

| Macro | Description |
|:------|:------------|
| `LIBXS_TLS_TYPE` | Key type (`pthread_key_t`, or `DWORD` on Windows) |
| `LIBXS_TLS_CREATE(KEYPTR)` | Create a key (no destructor) |
| `LIBXS_TLS_DESTROY(KEY)` | Delete a key |
| `LIBXS_TLS_SETVALUE(KEY, PTR)` | Set the calling thread's value |
| `LIBXS_TLS_GETVALUE(KEY)` | The calling thread's value (NULL if not set) |

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

```C
typedef enum libxs_atomic_kind_t {
  LIBXS_ATOMIC_SEQ_CST, /* __ATOMIC_SEQ_CST if available */
  LIBXS_ATOMIC_RELAXED  /* __ATOMIC_RELAXED if available */
} libxs_atomic_kind_t;
```

The `KIND` parameter selects the memory order (`LIBXS_ATOMIC_SEQ_CST` or `LIBXS_ATOMIC_RELAXED`) with the GCC builtins (`__atomic_*`). The legacy sync builtins and the Windows Interlocked intrinsics ignore it, and `LIBXS_ATOMIC_SYNC` is always a full fence.

The unqualified operations are 32-bit in some configurations. Select a width explicitly with `LIBXS_ATOMIC(FN, BITS)`, where `BITS` is 8, 16, 32, or 64, e.g., `LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, 64)(&counter, 1, LIBXS_ATOMIC_RELAXED)`. `LIBXS_ATOMIC_SIZE(FN)` selects the width of `size_t`.

| Macro | Description |
|:------|:------------|
| `LIBXS_SYNC_PAUSE` | CPU pause hint for spin loops (no-op where unavailable) |
| `LIBXS_SYNC_YIELD` | Yield the processor to another thread |
| `LIBXS_NPAUSE_LOCK` | Pauses before a waiting lock yields (default 4096); suits `NPAUSE` of `LIBXS_ATOMIC_ACQUIRE` |
| `LIBXS_NPAUSE_BARRIER` | Pauses before a waiting barrier task yields (default 1024) |
| `LIBXS_SYNC_CYCLE(PTR, STATE, NPAUSE)` | Spin until the lowest bit of `*PTR` equals that of `STATE`; pauses double up to `NPAUSE`, then each round yields |
| `LIBXS_SYNC_CYCLE_EQ(PTR, VALUE, NPAUSE)` | Like `LIBXS_SYNC_CYCLE`, but until `*PTR` equals `VALUE` |
| `LIBXS_SYNC_CYCLE_ELSE(PTR, STATE, NPAUSE, ELSE)`, `LIBXS_SYNC_CYCLE_EQ_ELSE(...)` | As above; `ELSE` is a statement run after each yield |

## Lock Abstraction

```C
LIBXS_LOCK_TYPE(KIND)
LIBXS_LOCK_INIT(KIND, LOCK, ATTR)
LIBXS_LOCK_DESTROY(KIND, LOCK)
LIBXS_LOCK_ACQUIRE(KIND, LOCK)
LIBXS_LOCK_TRYLOCK(KIND, LOCK)
LIBXS_LOCK_RELEASE(KIND, LOCK)
```

Generic lock interface parameterized by `KIND`. `LIBXS_LOCK_TRYLOCK` succeeded if its result equals `LIBXS_LOCK_ACQUIRED(KIND)`, a value that depends on the backend.

| KIND | Backend |
|:-----|:--------|
| `LIBXS_LOCK_SPINLOCK` | Atomic spin-lock |
| `LIBXS_LOCK_MUTEX` | Pthreads mutex / Windows SRW lock (Windows mutex if targeting pre-Vista) |
| `LIBXS_LOCK_RWLOCK` | Pthreads reader-writer lock / Windows SRW lock (Windows mutex if targeting pre-Vista) |
| `LIBXS_LOCK_ATOMIC` | Lightweight atomic lock |

The lock kind used by the library is `LIBXS_LOCK`, which defaults to `LIBXS_LOCK_DEFAULT` (`LIBXS_LOCK_ATOMIC`). Define `LIBXS_LOCK` before including LIBXS to choose another kind.

`ATTR` is a pointer to lock attributes, which are created per kind:

```C
LIBXS_LOCK_ATTR_TYPE(KIND) attr;
LIBXS_LOCK_ATTR_INIT(KIND, &attr);
LIBXS_LOCK_INIT(KIND, &lock, &attr);
LIBXS_LOCK_ATTR_DESTROY(KIND, &attr);
```

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

`LIBXS_LOCK_TYPE_ISRW(KIND)` is 1 if the kind provides genuine shared (reader) locking; otherwise the read variants lock exclusively. `LIBXS_LOCK_TYPE_ISPOD(KIND)` is 1 if the lock type is plain data.

## Barrier

```C
typedef struct libxs_barrier_t libxs_barrier_t; /* POD defined in the header */

void libxs_barrier_init(libxs_barrier_t* barrier, int ntasks);  /* inline */
void libxs_barrier_wait(libxs_barrier_t* barrier);              /* inline */
int libxs_barrier_bcast(libxs_barrier_t* barrier,
  int tid, int root, int value);                                /* inline */
```

Rendezvous over a fixed number of tasks. The caller owns the storage (nothing is allocated inside), and one task calls `libxs_barrier_init` once, before any task waits (`ntasks` below 1 counts as 1). A task is only a number, so the team need not be a thread team.

Every waiting task spins, so the barrier suits a team that meets between stages of work, and a team that over-subscribes the machine slows down.

`libxs_barrier_bcast` waits and returns the `int` value the `root` task passed in (`tid` is the calling task's number). Use it to hand a value from one task to the rest of the team.

## Ring

```C
typedef enum libxs_ring_kind_t {
  LIBXS_RING_PRIVATE = 0,
  LIBXS_RING_SHARED = 1
} libxs_ring_kind_t;
typedef struct libxs_ring_t libxs_ring_t; /* header placed in caller memory */

size_t libxs_ring_size(size_t capacity);
libxs_ring_t* libxs_ring_init(void* memory, size_t size, libxs_ring_kind_t kind);
void* libxs_ring_reserve(libxs_ring_t* ring, size_t nbytes, int timeout_ms);
void libxs_ring_commit(libxs_ring_t* ring);
const void* libxs_ring_peek(libxs_ring_t* ring, size_t* nbytes, int timeout_ms);
void libxs_ring_release(libxs_ring_t* ring);
void libxs_ring_close(libxs_ring_t* ring);
```

Usage:

```C
const size_t size = libxs_ring_size(capacity);
libxs_ring_t* ring = libxs_ring_init(memory, size, LIBXS_RING_SHARED);
void* record = libxs_ring_reserve(ring, nbytes, timeout_ms); /* producer */
libxs_ring_commit(ring);
const void* data = libxs_ring_peek(ring, &nbytes, timeout_ms); /* consumer */
libxs_ring_release(ring);
libxs_ring_close(ring);
```

Records of varying size passed from one producer to one consumer, in order. The caller owns the memory (aligned to a cache line), and the ring keeps no pointer in it, so two processes can map the same memory (`MAP_SHARED`) at different addresses; one side calls `libxs_ring_init` and the other uses the memory as is. `LIBXS_RING_PRIVATE` is for threads of one process, `LIBXS_RING_SHARED` for processes.

A record is written in place: `libxs_ring_reserve` returns room for it, contiguous and aligned to eight bytes, and `libxs_ring_commit` publishes it. The consumer reads it in place as well, until `libxs_ring_release`. A record takes at most half the capacity.

- `libxs_ring_size` returns the bytes of memory needed for `capacity` bytes of records (rounded up to a power of two); a record takes its size rounded up to eight bytes, plus eight.
- `libxs_ring_init` returns NULL if the memory cannot hold a ring.
- `libxs_ring_reserve` returns NULL if there was no room in time, the ring is closed, or the record exceeds half the capacity.
- `libxs_ring_peek` writes the record size to `*nbytes` and returns NULL if no record arrived in time, or if the ring is closed and empty.

A side that has to wait spins briefly and then sleeps (a futex on Linux) until the other side moves; `timeout_ms` of zero does not wait and a negative one waits indefinitely. `libxs_ring_close` ends the waiting on both sides, while the consumer still receives what was committed. A process that dies does not close its side, hence a timeout is how the other side notices.

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

Return the number of MPI ranks on the node of the calling process, as the launcher
states it (`MPI_LOCALNRANKS`, `OMPI_COMM_WORLD_LOCAL_SIZE`, or
`MV2_COMM_WORLD_LOCAL_SIZE`). Without these, it counts the user's processes on the
node that initialized LIBXS (each holds a slot, a locked file in `/tmp`), or one.
Such a count grows while processes start, hence it is better queried when needed
than right at initialization.

```C
unsigned int libxs_nrank(void);
```

Return the MPI rank of the calling process among the ranks on its node
(`MPI_LOCALRANKID`, `OMPI_COMM_WORLD_LOCAL_RANK`, `MV2_COMM_WORLD_LOCAL_RANK`, or
`SLURM_LOCALID`). Without these, it returns the index of the process' slot (see
`libxs_nranks`), which is unique on the node but not necessarily below
`libxs_nranks`. The global `PMI_RANK` is the last resort, which is local only if
ranks are placed in blocks of `libxs_nranks` per node.

```C
unsigned int libxs_rid(void);
```

Return the node-local MPI rank (`libxs_nrank`) if there are several ranks on the node (`libxs_nranks`), and the process ID otherwise.

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

```C
int libxs_singleton_acquire(const char key[]);
void libxs_singleton_release(int handle);
```

Elect one process per node for a key: of the user's processes on a node, the first to acquire the key owns it until it releases the handle or exits, and every other acquisition fails, including a second one by the owner. A non-negative handle means the calling process owns the key. Unlike `libxs_nrank`, the election does not depend on what the launcher states, so it also holds across jobs and without MPI. On POSIX the lock is a file `/tmp/.libxs.<uid>.<key>`, hence `/tmp` must be local to the node; the key carries no path separator, and an empty key is the one LIBXS uses itself.
