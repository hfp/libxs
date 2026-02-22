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


/** Lock-type used for this domain. */
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_malloc_lock_t;

/**
 * Initialize the pool by drawing from the given storage a number of chunks of the given size.
 * If the capacity of the pool is num, the storage must be at least num x size.
 * The same num-counter must be used for pmalloc/pfree when referring to the same pool.
 */
LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Allocate from the given pool by using the original num-counter (libxs_pmalloc_init). */
LIBXS_API void* libxs_pmalloc_lock(void* pool[], size_t* num, libxs_malloc_lock_t* lock);
/** Similar to libxs_pmalloc_lock but using an internal lock. */
LIBXS_API void* libxs_pmalloc(void* pool[], size_t* num);

/** Bring pointer back into the pool by using original num-counter (libxs_pmalloc_init). */
LIBXS_API void libxs_pfree_lock(const void* pointer, void* pool[], size_t* num, libxs_malloc_lock_t* lock);
/** Similar to libxs_pfree_lock but using an internal lock. */
LIBXS_API void libxs_pfree(const void* pointer, void* pool[], size_t* num);

/** Allocate from a pool which can reach steady-sate (libxs_malloc_pool). */
LIBXS_API void* libxs_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);
/** Free memory allocated by libxs_malloc. */
LIBXS_API void libxs_free(const void* pointer);

/** Information about allocated memory (pointer). */
LIBXS_EXTERN_C typedef struct libxs_malloc_info_t {
  size_t size;
} libxs_malloc_info_t;

/** Retrieve information about allocated memory (pointer). */
LIBXS_API int libxs_malloc_info(const void* pointer, libxs_malloc_info_t* info);

/** Allocate a pool for libxs_malloc (shall be called before libxs_malloc). */
LIBXS_API void libxs_malloc_pool(void);
/** Free unused memory (libxs_malloc_pool). */
LIBXS_API void libxs_free_pool(void);

/** Information about pooled memory. */
LIBXS_EXTERN_C typedef struct libxs_malloc_pool_info_t {
  size_t size;
  /** Pending allocations (not released). */
  size_t nactive;
  /** Number of allocations so far. */
  size_t nmallocs;
} libxs_malloc_pool_info_t;

/** Retrieve information about the pooled memory domain. */
LIBXS_API int libxs_malloc_pool_info(libxs_malloc_pool_info_t* info);

#endif /*LIBXS_MALLOC_H*/
