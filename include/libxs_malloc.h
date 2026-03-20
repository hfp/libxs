/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MALLOC_H
#define LIBXS_MALLOC_H

#include "libxs_sync.h"


/** Automatic alignment with inline metadata (default). */
#define LIBXS_MALLOC_AUTO 0
/** Preserve allocator's native pointer (out-of-band metadata). */
#define LIBXS_MALLOC_NATIVE 1
/* Values > 1 are interpreted as explicit alignment (inline metadata). */


/** Information about allocated memory (pointer). */
LIBXS_EXTERN_C typedef struct libxs_malloc_info_t {
  size_t size;
} libxs_malloc_info_t;

/** Information about pooled memory. */
LIBXS_EXTERN_C typedef struct libxs_malloc_pool_info_t {
  size_t size;
  /** Peak memory consumption. */
  size_t peak;
  /** Pending allocations (not released). */
  size_t nactive;
  /** Number of allocations so far. */
  size_t nmallocs;
} libxs_malloc_pool_info_t;

/** Opaque pool type (created by libxs_malloc_pool). */
LIBXS_EXTERN_C typedef struct libxs_malloc_pool_t libxs_malloc_pool_t;

/** Function type for custom memory allocation (malloc-compatible). */
LIBXS_EXTERN_C typedef void* (*libxs_malloc_fn)(size_t size);
/** Function type for custom memory deallocation (free-compatible). */
LIBXS_EXTERN_C typedef void (*libxs_free_fn)(void* pointer);

/** Extended function type for custom allocation (extra context per thread). */
LIBXS_EXTERN_C typedef void* (*libxs_malloc_xfn)(size_t size, const void* extra);
/** Extended function type for custom deallocation (extra context per thread). */
LIBXS_EXTERN_C typedef void (*libxs_free_xfn)(void* pointer, const void* extra);


/**
 * Initialize the pool by drawing from the given storage a number of chunks of the given size.
 * If the capacity of the pool is num, the storage must be at least num x size.
 * The same num-counter must be used for pmalloc/pfree when referring to the same pool.
 */
LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Allocate from the given pool by using the original num-counter (libxs_pmalloc_init). */
LIBXS_API void* libxs_pmalloc_lock(void* pool[], size_t* num, libxs_lock_t* lock);
/** Similar to libxs_pmalloc_lock but using an internal lock. */
LIBXS_API void* libxs_pmalloc(void* pool[], size_t* num);

/** Bring pointer back into the pool by using original num-counter (libxs_pmalloc_init). */
LIBXS_API void libxs_pfree_lock(void* pointer, void* pool[], size_t* num, libxs_lock_t* lock);
/** Similar to libxs_pfree_lock but using an internal lock. */
LIBXS_API void libxs_pfree(void* pointer, void* pool[], size_t* num);

/**
 * Create a memory pool, optionally with custom allocator functions.
 * If malloc_fn or free_fn is NULL, the standard malloc/free is used.
 * Both function pointers must be NULL or both must be non-NULL.
 */
LIBXS_API libxs_malloc_pool_t* libxs_malloc_pool(libxs_malloc_fn malloc_fn, libxs_free_fn free_fn);

/**
 * Create a memory pool with extended allocator functions that receive a per-thread
 * extra argument (see libxs_malloc_arg). The max_nthreads parameter determines the
 * size of the internal per-thread argument table (indexed by libxs_tid).
 * Both function pointers must be non-NULL.
 */
LIBXS_API libxs_malloc_pool_t* libxs_malloc_xpool(libxs_malloc_xfn malloc_fn, libxs_free_xfn free_fn,
  int max_nthreads);

/**
 * Set the per-thread extra argument for an extended pool (created by libxs_malloc_xpool).
 * The extra pointer is stored at the calling thread's slot (libxs_tid) and passed to
 * the registered malloc_xfn/free_xfn on subsequent allocations from this thread.
 * No-op for standard pools (created by libxs_malloc_pool).
 */
LIBXS_API void libxs_malloc_arg(libxs_malloc_pool_t* pool, const void* extra);

/** Destroy the pool and free all associated memory. */
LIBXS_API void libxs_free_pool(libxs_malloc_pool_t* pool);

/**
 * Allocate from the given pool.
 * LIBXS_MALLOC_AUTO: automatic alignment with inline metadata.
 * LIBXS_MALLOC_NATIVE: preserve allocator's native pointer.
 * Values > 1: explicit alignment in Bytes (inline metadata).
 */
LIBXS_API void* libxs_malloc(libxs_malloc_pool_t* pool, size_t size, int alignment);
/** Free memory allocated by libxs_malloc (pool is derived internally). */
LIBXS_API void libxs_free(void* pointer);

/** Retrieve information about allocated memory (pool is derived internally). */
LIBXS_API int libxs_malloc_info(const void* pointer, libxs_malloc_info_t* info);
/** Retrieve information about the pooled memory domain. */
LIBXS_API int libxs_malloc_pool_info(const libxs_malloc_pool_t* pool, libxs_malloc_pool_info_t* info);

#endif /*LIBXS_MALLOC_H*/
