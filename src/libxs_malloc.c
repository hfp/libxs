/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_malloc.h>
#include <libxs_sync.h>
#include <libxs_mem.h>
#include "libxs_hash.h"

#if !defined(LIBXS_MALLOC_SEED)
# define LIBXS_MALLOC_SEED 1051981
#endif
#if !defined(LIBXS_MALLOC_NLOCKS)
# define LIBXS_MALLOC_NLOCKS 16
#endif
#if !defined(LIBXS_MALLOC_UPSIZE)
# define LIBXS_MALLOC_UPSIZE (2 << 20)
#endif
#if !defined(LIBXS_MALLOC_EVICT_SIZE)
# define LIBXS_MALLOC_EVICT_SIZE (8 * LIBXS_MALLOC_UPSIZE)
#endif
#if !defined(LIBXS_MALLOC_EVICT_LIMIT)
# define LIBXS_MALLOC_EVICT_LIMIT (64 * LIBXS_MALLOC_EVICT_SIZE)
#endif
#if !defined(LIBXS_MALLOC_EVICT) && 0
# define LIBXS_MALLOC_EVICT
#endif
#if !defined(LIBXS_MALLOC_SEARCH) && 1
# define LIBXS_MALLOC_SEARCH
#endif
#if !defined(LIBXS_MALLOC_PRUNE) && 0
# define LIBXS_MALLOC_PRUNE
#endif


typedef struct internal_malloc_chunk_t {
  char *pointer;
  size_t size, nmallocs;
} internal_malloc_chunk_t;

LIBXS_APIVAR_DEFINE(volatile int internal_malloc_plocks[LIBXS_MALLOC_NLOCKS]);
LIBXS_APIVAR_DEFINE(internal_malloc_chunk_t* internal_malloc_storage);
LIBXS_APIVAR_DEFINE(internal_malloc_chunk_t** internal_malloc_pool);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_pool_size);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_pool_num);
LIBXS_APIVAR_DEFINE(int internal_malloc_pool_maxnt);

#if defined(LIBXS_MALLOC_EVICT)
LIBXS_APIVAR_DEFINE(size_t internal_malloc_pool_bytes);
#endif


LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage)
{
  char *p = (char*)storage;
  volatile int *lock;
  unsigned int hash;
  size_t n, i = 0;
  LIBXS_ASSERT(0 < size && NULL != num && NULL != pool && NULL != storage);
  libxs_hash_init(libxs_cpuid(NULL)); /* CRC-facility must be initialized upfront */
  hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
  lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
  LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
  for (n = *num; i < n; ++i, p += size) pool[i] = p;
  LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
}


LIBXS_API void* libxs_pmalloc(void* pool[], size_t* num)
{
  const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
  volatile int *const lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
  void *pointer;
  LIBXS_ASSERT(NULL != pool && NULL != num);
  LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
  assert(0 < *num && ((size_t)-1) != *num); /* !LIBXS_ASSERT */
  pointer = pool[--(*num)];
  LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
  LIBXS_ASSERT(NULL != pointer);
  return pointer;
}


LIBXS_API void libxs_pfree(const void* pointer, void* pool[], size_t* num)
{
  LIBXS_ASSERT(NULL != pool && NULL != num);
  if (NULL != pointer) {
    const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
    volatile int *const lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
    LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    LIBXS_VALUE_ASSIGN(pool[*num], pointer); ++(*num);
    LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
  }
}


LIBXS_API void* libxs_malloc(size_t size, size_t alignment)
{
  void *result = NULL;
  if (0 != size) {
    const size_t alignpot = LIBXS_UP2POT(LIBXS_MAX(sizeof(void*) + 1,
      0 == alignment ? LIBXS_ALIGNMENT : alignment));
    internal_malloc_chunk_t *chunk = NULL;
    void **info = NULL;
#if defined(LIBXS_MALLOC_SEARCH)
    const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, internal_malloc_pool);
    volatile int *const lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
    internal_malloc_chunk_t **hit = NULL;
    size_t i = 0, diff = (size_t)-1;
    LIBXS_ASSERT(NULL != internal_malloc_pool);
    LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    assert(0 < internal_malloc_pool_num && ((size_t)-1) != internal_malloc_pool_num);
    do {
      internal_malloc_chunk_t *const c = internal_malloc_pool[i];
      const size_t delta = LIBXS_DELTA(c->size, size);
      if (delta < diff) {
        diff = delta; hit = internal_malloc_pool + i;
        if (size <= c->size && c->size < (size + LIBXS_MALLOC_UPSIZE)) break;
      }
    } while (++i < internal_malloc_pool_num);
    { /* allocate slot and eventually reuse */
      internal_malloc_chunk_t **const alloc = internal_malloc_pool + --internal_malloc_pool_num;
      if (hit != alloc && NULL != hit && (
# if defined(LIBXS_MALLOC_PRUNE)
        (*hit)->nmallocs < (*alloc)->nmallocs ||
# endif
        size <= (*hit)->size))
      {
        LIBXS_VALUE_SWAP(*alloc, *hit);
      }
      chunk = *alloc;
    }
    LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
#else
    chunk = libxs_pmalloc((void**)internal_malloc_pool, &internal_malloc_pool_num);
#endif
    LIBXS_ASSERT(NULL != chunk);
    if (NULL != chunk->pointer) {
      if (chunk->size < size) {
#if defined(LIBXS_MALLOC_EVICT)
        const size_t old_size = chunk->size;
#endif
        char *const pointer = realloc(chunk->pointer, size + alignpot - 1);
        result = LIBXS_ALIGN(pointer, alignpot);
        info = (void**)((uintptr_t)result - sizeof(void*));
        chunk->pointer = pointer;
        chunk->size = size;
        ++chunk->nmallocs;
#if defined(LIBXS_MALLOC_EVICT)
        LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&internal_malloc_pool_bytes, size - old_size, LIBXS_ATOMIC_SEQ_CST);
#endif
        *info = chunk;
      }
      else { /* reuse */
        result = LIBXS_ALIGN(chunk->pointer, alignpot);
      }
    }
    else {
      char *const pointer = malloc(size + alignpot - 1);
      result = LIBXS_ALIGN(pointer, alignpot);
      info = (void**)((uintptr_t)result - sizeof(void*));
      LIBXS_ASSERT(0 == chunk->size);
      chunk->pointer = pointer;
      chunk->size = size;
      ++chunk->nmallocs;
#if defined(LIBXS_MALLOC_EVICT)
      LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&internal_malloc_pool_bytes, size, LIBXS_ATOMIC_SEQ_CST);
#endif
      *info = chunk;
    }
  }
  return result;
}


LIBXS_API void libxs_free(const void* pointer)
{
  if (NULL != pointer) {
    internal_malloc_chunk_t *const chunk = *(void**)((uintptr_t)pointer - sizeof(void*));
    LIBXS_ASSERT(NULL != chunk && NULL != internal_malloc_pool);
#if defined(LIBXS_MALLOC_EVICT)
    if (NULL != chunk->pointer && LIBXS_MALLOC_EVICT_SIZE <= chunk->size) {
      const size_t total_bytes = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&internal_malloc_pool_bytes, LIBXS_ATOMIC_RELAXED);
      if (LIBXS_MALLOC_EVICT_LIMIT < total_bytes) {
        const size_t reclaimed = chunk->size;
        free(chunk->pointer);
        chunk->pointer = NULL;
        chunk->size = 0;
        LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(&internal_malloc_pool_bytes, reclaimed, LIBXS_ATOMIC_SEQ_CST);
      }
    }
#endif
    libxs_pfree(chunk, (void**)internal_malloc_pool, &internal_malloc_pool_num);
  }
}


LIBXS_API int libxs_malloc_info(const void* pointer, libxs_malloc_info_t* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    LIBXS_MEMZERO(info);
    if (NULL != pointer) {
      const internal_malloc_chunk_t *const chunk = *(void**)((uintptr_t)pointer - sizeof(void*));
      info->size = chunk->size;
    }
  }
  else if (NULL != pointer) {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void libxs_malloc_pool(int max_nthreads, int max_nactive)
{
  if (NULL == internal_malloc_storage) {
    LIBXS_ASSERT(NULL == internal_malloc_pool && 0 < max_nthreads && 0 < max_nactive);
    internal_malloc_pool_size = internal_malloc_pool_num = max_nthreads * max_nactive;
    internal_malloc_storage = calloc(internal_malloc_pool_num, sizeof(internal_malloc_chunk_t));
    internal_malloc_pool = malloc(internal_malloc_pool_num * sizeof(void*));
    internal_malloc_pool_maxnt = max_nthreads;
#if defined(LIBXS_MALLOC_EVICT)
    internal_malloc_pool_bytes = 0;
#endif
    if (NULL != internal_malloc_storage && NULL != internal_malloc_pool) {
      libxs_pmalloc_init(sizeof(internal_malloc_chunk_t), &internal_malloc_pool_num,
        (void**)internal_malloc_pool, internal_malloc_storage);
    }
    else {
      internal_malloc_pool_size = internal_malloc_pool_num = 0;
      internal_malloc_pool_maxnt = 0;
      free(internal_malloc_storage);
      free(internal_malloc_pool);
      internal_malloc_storage = NULL;
      internal_malloc_pool = NULL;
    }
  }
}


LIBXS_API void libxs_free_pool(void)
{
  if (NULL != internal_malloc_storage) {
    size_t i = 0;
    LIBXS_ASSERT(NULL != internal_malloc_pool);
    for (; i < internal_malloc_pool_num/*internal_malloc_pool_size*/; ++i) {
      internal_malloc_chunk_t *const chunk = internal_malloc_pool[i];
      LIBXS_ASSERT(NULL != chunk);
      free(chunk->pointer);
    }
#if defined(LIBXS_MALLOC_EVICT)
    internal_malloc_pool_bytes = 0;
#endif
    internal_malloc_pool_maxnt = 0;
    internal_malloc_pool_size = internal_malloc_pool_num = 0;
    free(internal_malloc_storage);
    free(internal_malloc_pool);
    internal_malloc_storage = NULL;
    internal_malloc_pool = NULL;
  }
#if defined(LIBXS_MALLOC_EVICT)
  LIBXS_ASSERT(0 == internal_malloc_pool_bytes);
#endif
  LIBXS_ASSERT(0 == internal_malloc_pool_maxnt);
  LIBXS_ASSERT(0 == internal_malloc_pool_size);
  LIBXS_ASSERT(0 == internal_malloc_pool_num);
}


LIBXS_API int libxs_malloc_pool_info(libxs_malloc_pool_info_t* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    LIBXS_MEMZERO(info);
    if (NULL != internal_malloc_pool) {
      size_t i = 0;
      for (; i < internal_malloc_pool_size; ++i) {
        internal_malloc_chunk_t *const chunk = internal_malloc_pool[i];
        LIBXS_ASSERT(NULL != chunk);
        info->nmallocs += chunk->nmallocs;
        info->size += chunk->size;
      }
      info->nactive = (internal_malloc_pool_size / internal_malloc_pool_maxnt) -
        LIBXS_UPDIV(internal_malloc_pool_num, internal_malloc_pool_maxnt);
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}
