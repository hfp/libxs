# Memory Allocation

Header: `libxs_malloc.h`

Pool-based memory allocator designed for steady-state performance: after an initial warm-up phase, allocations are served from a recycled pool without system calls.

## Types

```C
typedef struct libxs_malloc_info_t {
  size_t size;  /* allocated size in bytes */
} libxs_malloc_info_t;
```

```C
typedef struct libxs_malloc_pool_hist_t {
  size_t count;         /* allocations in this bucket */
  size_t nreuses;       /* chunk reused without reallocation */
  size_t ngrows;        /* chunk reallocated (too small) */
  size_t nevicts_age;   /* evictions triggered by age */
  size_t nevicts_limit; /* evictions triggered by memory limit */
} libxs_malloc_pool_hist_t;
```

```C
typedef struct libxs_malloc_pool_info_t {
  size_t used;      /* memory currently in use (sum of requested sizes) */
  size_t size;      /* total allocated memory (sum of actual chunk sizes) */
  size_t peak;      /* peak of the total allocated memory */
  size_t nactive;   /* pending (not yet freed) allocations */
  size_t nmallocs;  /* total allocation count */
  libxs_malloc_pool_hist_t hist[6]; /* per-bucket histogram */
} libxs_malloc_pool_info_t;
```

The histogram uses 6 logarithmic buckets keyed on requested size: <1M, 1-4M, 4-16M, 16-64M, 64-256M, >=256M. Each bucket tracks how allocations in that size class were served (reuse vs. grow) and how many were evicted by the pool's memory-pressure heuristics.

```C
typedef struct libxs_malloc_pool_t libxs_malloc_pool_t; /* opaque */
```

```C
typedef void* (*libxs_malloc_fn)(size_t size);
typedef void  (*libxs_free_fn)(void* pointer);
```

```C
typedef void* (*libxs_malloc_xfn)(size_t size, const void* extra);
typedef void  (*libxs_free_xfn)(void* pointer, const void* extra);
```

## Pool Management

```C
libxs_malloc_pool_t* libxs_malloc_pool(libxs_malloc_fn malloc_fn, libxs_free_fn free_fn);
```

Create a memory pool. If `malloc_fn` and `free_fn` are both NULL, the standard `malloc`/`free` are used. Both must be NULL or both must be non-NULL.

```C
libxs_malloc_pool_t* libxs_malloc_xpool(libxs_malloc_xfn malloc_fn, libxs_free_xfn free_fn,
  int max_nthreads);
```

Create a memory pool with extended allocator functions that receive a per-thread extra argument (see `libxs_malloc_arg`). The `max_nthreads` parameter determines the size of the internal per-thread argument table (indexed by `libxs_tid`). Both function pointers must be non-NULL and `max_nthreads` must be positive; otherwise NULL is returned.

```C
void libxs_malloc_arg(libxs_malloc_pool_t* pool, const void* extra);
```

Set the per-thread extra argument for an extended pool (created by `libxs_malloc_xpool`). The pointer is stored at the calling thread's slot (`libxs_tid() % max_nthreads`) and passed to the registered `malloc_xfn`/`free_xfn` on subsequent allocations and frees from this thread. No-op for standard pools. Access is lock-free since each thread writes only its own slot.

```C
void libxs_free_pool(libxs_malloc_pool_t* pool);
```

Destroy the pool and release all associated memory. Accepts NULL.

```C
int libxs_malloc_pool_info(const libxs_malloc_pool_t* pool, libxs_malloc_pool_info_t* info);
```

Query aggregate pool statistics including the per-bucket histogram. A NULL `pool` selects the default pool.

```C
void libxs_malloc_pool_print(FILE* ostream, const char prefix[],
  const libxs_malloc_pool_t* pool);
```

Print pool statistics and per-bucket histogram to `ostream`. Each non-empty bucket is printed on its own line showing count, reuse, grow, and eviction numbers. The `prefix` string (may be NULL) is prepended to each line. No output is produced if the pool has zero allocations or `ostream` is NULL.

```C
libxs_malloc_pool_t* libxs_default_pool(void);
```

Access the default host memory pool (created at `libxs_init`). Printed automatically at `libxs_finalize` when `LIBXS_VERBOSE` >= 3 (or negative), as is every pool destroyed by `libxs_free_pool`.

## General Allocation

```C
void* libxs_malloc(libxs_malloc_pool_t* pool, size_t size, int alignment);
```

Allocate `size` bytes from the given `pool`. `LIBXS_MALLOC_AUTO` (0) uses automatic alignment with inline metadata. `LIBXS_MALLOC_NATIVE` (1) preserves the allocator's native pointer (out-of-band metadata via registry). Values greater than 1 are interpreted as explicit alignment in Bytes (inline metadata). A NULL `pool` selects the default pool. Returns NULL on failure or if `size` is zero.

```C
void libxs_free(void* pointer);
```

Return memory to its originating pool (derived internally). Accepts NULL.

```C
int libxs_malloc_info(const void* pointer, libxs_malloc_info_t* info);
```

Query the size of an allocation made with `libxs_malloc`. The pool is derived internally.

## Environment

`LIBXS_MALLOC_LIMIT` bounds the pool in MB. Beyond it, an allocation first evicts
a cached chunk, which returns its memory to the system. A negative value removes
the bound. Unset, the bound follows the pool's high-water mark plus headroom. A
bound below the high-water mark does not reduce what is held; it only causes more
eviction. `LIBXS_VERBOSE=3` prints the pool report, including the high-water mark
and the bound.

`LIBXS_MALLOC_SHARE` caps the derived bound at a percentage of the memory the
process may use (default 50). With several MPI ranks on a node (`libxs_nranks`),
each rank receives its part of the share. It does not apply when
`LIBXS_MALLOC_LIMIT` is set.

`LIBXS_MALLOC_EVICT_AGE` evicts a cached chunk that was not used for this many
ticks of 1024 allocations from its pool (default 8). A negative value disables
eviction by age, leaving the bound above as the only eviction. A larger age
reallocates less often, at the cost of holding idle memory longer. The pool report
(`LIBXS_VERBOSE=3`) counts both kinds of eviction per size class.

`LIBXS_MALLOC_NATIVE` selects where `libxs_malloc` keeps its metadata. Unset,
`LIBXS_MALLOC_NATIVE` allocations keep the native pointer (out-of-band metadata)
and `LIBXS_MALLOC_AUTO` uses inline metadata. A non-zero value makes
`LIBXS_MALLOC_AUTO` keep the native pointer as well; zero makes both use inline
metadata.

## Fixed-Size Pool

A lightweight fixed-size pool for scenarios where the element size is known at initialization time.

```C
void libxs_pmalloc_init(size_t size, size_t* num,
  void* pool[], void* storage);
```

Partition `storage` into `*num` chunks of `size` bytes and register them in `pool`.

```C
void* libxs_pmalloc(void* pool[], size_t* num);
void* libxs_pmalloc_lock(void* pool[], size_t* num,
  libxs_lock_t* lock);
```

Pop one chunk from the pool. The `_lock` variant uses a caller-provided lock; the plain variant uses an internal lock.

```C
void libxs_pfree(void* pointer, void* pool[], size_t* num);
void libxs_pfree_lock(void* pointer, void* pool[],
  size_t* num, libxs_lock_t* lock);
```

Push a chunk back into the pool.
