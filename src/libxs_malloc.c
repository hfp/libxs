/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_reg.h>
#include "libxs_crc32.h"
#include "libxs_main.h"

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
# define LIBXS_MALLOC_EVICT_LIMIT ((size_t)512 * LIBXS_MALLOC_EVICT_SIZE)
#endif
#if !defined(LIBXS_MALLOC_EVICT_AGE)
# define LIBXS_MALLOC_EVICT_AGE 8
#endif
#if !defined(LIBXS_MALLOC_TICK_RATE)
# define LIBXS_MALLOC_TICK_RATE 1024
#endif
#if !defined(LIBXS_MALLOC_EVICT) && 1
# define LIBXS_MALLOC_EVICT
#endif
#if !defined(LIBXS_MALLOC_SEARCH) && 1
# define LIBXS_MALLOC_SEARCH
#endif
#if !defined(LIBXS_MALLOC_POOL_INIT)
# define LIBXS_MALLOC_POOL_INIT 16
#endif
#if !defined(LIBXS_MALLOC_HIST_SHIFT)
# define LIBXS_MALLOC_HIST_SHIFT 20
#endif
#define LIBXS_MALLOC_HIST_NBUCKETS \
  (sizeof(((libxs_malloc_pool_info_t*)0)->hist) / sizeof(libxs_malloc_pool_hist_t))


typedef struct internal_libxs_malloc_chunk_t {
  char *pointer;
  struct internal_libxs_malloc_chunk_t *next;
  struct libxs_malloc_pool_t *pool; /* back-pointer to owning pool */
  size_t used, size, nmallocs;
#if defined(LIBXS_MALLOC_EVICT)
  size_t last_tick; /* generation when last used */
#endif
} internal_libxs_malloc_chunk_t;

struct libxs_malloc_pool_t {
  union { libxs_malloc_fn std; libxs_malloc_xfn ext; } fn_malloc;
  union { libxs_free_fn std; libxs_free_xfn ext; } fn_free;
  const void** extra; /* per-thread extra args, NULL for standard pools */
  int max_nthreads; /* 0 for standard pools */
  libxs_lock_t plock;
  internal_libxs_malloc_chunk_t *all;
  internal_libxs_malloc_chunk_t **slots;
  size_t slots_size;
  size_t slots_num;
#if defined(LIBXS_MALLOC_EVICT)
  size_t generation;
  size_t pool_bytes;
  size_t pool_peak;
#endif
  libxs_malloc_pool_hist_t hist[LIBXS_MALLOC_HIST_NBUCKETS];
};

LIBXS_APIVAR_DEFINE(libxs_lock_t internal_libxs_malloc_plocks[LIBXS_MALLOC_NLOCKS]);
LIBXS_APIVAR_DEFINE(libxs_registry_t* internal_libxs_malloc_registry);
#if defined(LIBXS_MALLOC_EVICT)
LIBXS_APIVAR_DEFINE(size_t internal_libxs_malloc_evict_limit);
#endif


LIBXS_API_INLINE libxs_registry_t* internal_libxs_malloc_get_registry(int alignment)
{
  libxs_registry_t* result = NULL;
  if (LIBXS_MALLOC_NATIVE >= alignment) { /* AUTO or NATIVE */
    static int malloc_native = -1;
    if (0 > malloc_native) {
      const char *const env = getenv("LIBXS_MALLOC_NATIVE");
      malloc_native = (NULL == env ? 1 : (2 * (0 != atoi(env))));
    }
    if  ((LIBXS_MALLOC_AUTO >= alignment && 1 < malloc_native)
      || (LIBXS_MALLOC_NATIVE == alignment && 0 != malloc_native))
    {
      if (NULL == internal_libxs_malloc_registry) {
        static libxs_lock_t lock;
        LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &lock);
        if (NULL == internal_libxs_malloc_registry) {
          internal_libxs_malloc_registry = libxs_registry_create();
        }
        LIBXS_LOCK_RELEASE(LIBXS_LOCK, &lock);
      }
      result = internal_libxs_malloc_registry;
    }
  }
  return result;
}


LIBXS_API_INLINE libxs_malloc_pool_t* internal_libxs_malloc_default_pool(void)
{
  libxs_malloc_pool_t *result = internal_libxs_default_pool;
  if (NULL == result && 0 == libxs_ninit) {
    libxs_init();
    result = internal_libxs_default_pool;
  }
  return result;
}


LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage)
{
  char *p = (char*)storage;
  size_t n, i = 0;
  LIBXS_ASSERT(0 < size && NULL != num && NULL != pool && NULL != storage);
  internal_libxs_hash_init(libxs_cpuid(NULL)); /* CRC-facility must be initialized upfront */
  for (n = *num; i < n; ++i, p += size) pool[i] = p;
}


LIBXS_API void* libxs_pmalloc_lock(void* pool[], size_t* num, libxs_lock_t* lock)
{
  void *pointer;
  LIBXS_ASSERT(NULL != pool && NULL != num);
  if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  assert(0 < *num && ((size_t)-1) != *num); /* !LIBXS_ASSERT */
  pointer = pool[--(*num)];
  if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  LIBXS_ASSERT(NULL != pointer);
  return pointer;
}


LIBXS_API void* libxs_pmalloc(void* pool[], size_t* num)
{
  const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
  libxs_lock_t *const lock = internal_libxs_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
  return libxs_pmalloc_lock(pool, num, lock);
}


LIBXS_API void libxs_pfree_lock(void* pointer, void* pool[], size_t* num, libxs_lock_t* lock)
{
  LIBXS_ASSERT(NULL != pool && NULL != num);
  if (NULL != pointer) {
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    pool[*num] = pointer; ++(*num);
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_pfree(void* pointer, void* pool[], size_t* num)
{
  LIBXS_ASSERT(NULL != pool && NULL != num);
  if (NULL != pointer) {
    const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
    libxs_lock_t *const lock = internal_libxs_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
    libxs_pfree_lock(pointer, pool, num, lock);
  }
}


LIBXS_API_INLINE void internal_libxs_malloc_pool_return(internal_libxs_malloc_chunk_t *chunk)
{
  libxs_malloc_pool_t *const pool = chunk->pool;
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &pool->plock);
  if (pool->slots_num >= pool->slots_size) {
    const size_t new_size = (0 < pool->slots_size
      ? (2 * pool->slots_size) : LIBXS_MALLOC_POOL_INIT);
    internal_libxs_malloc_chunk_t **const np = realloc(
      pool->slots, new_size * sizeof(void*));
    if (NULL != np) {
      pool->slots_size = new_size;
      pool->slots = np;
    }
  }
  if (pool->slots_num < pool->slots_size) {
    pool->slots[pool->slots_num++] = chunk;
  }
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &pool->plock);
}


LIBXS_API_INLINE int internal_libxs_malloc_hist_bucket(size_t size)
{
  int b = 0;
  size >>= LIBXS_MALLOC_HIST_SHIFT;
  while (0 != size && b < (int)(LIBXS_MALLOC_HIST_NBUCKETS) - 1) { size >>= 2; ++b; }
  return b;
}


LIBXS_API_INLINE void* internal_libxs_malloc_allocate(
  libxs_malloc_pool_t *pool, size_t size)
{
  if (0 < pool->max_nthreads) {
    return pool->fn_malloc.ext(size, pool->extra[libxs_tid() % pool->max_nthreads]);
  }
  else if (NULL != pool->fn_malloc.std) return pool->fn_malloc.std(size);
  else return malloc(size);
}


LIBXS_API_INLINE void internal_libxs_malloc_deallocate(
  libxs_malloc_pool_t *pool, void *pointer)
{
  if (0 < pool->max_nthreads) {
    pool->fn_free.ext(pointer, pool->extra[libxs_tid() % pool->max_nthreads]);
  }
  else if (NULL != pool->fn_free.std) pool->fn_free.std(pointer);
  else free(pointer);
}


#if defined(LIBXS_MALLOC_EVICT)
LIBXS_API_INLINE void internal_libxs_malloc_peak_update(libxs_malloc_pool_t *pool, size_t candidate)
{
  size_t peak = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_peak, LIBXS_ATOMIC_RELAXED);
  while (candidate > peak) {
    if (LIBXS_ATOMIC(LIBXS_ATOMIC_CMPSWP, LIBXS_BITS)(&pool->pool_peak, peak, candidate, LIBXS_ATOMIC_LOCKORDER)) break;
    peak = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_peak, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_API_INLINE size_t internal_libxs_malloc_evict_available(
  libxs_malloc_pool_t *pool, internal_libxs_malloc_chunk_t *current)
{
  size_t reclaimed = 0;
  for (;;) {
    void *pointer = NULL;
    size_t used = 0, i;
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &pool->plock);
    for (i = 0; i < pool->slots_num; ++i) {
      internal_libxs_malloc_chunk_t *const chunk = pool->slots[i];
      if (NULL != chunk->pointer) {
        const int hist_b = internal_libxs_malloc_hist_bucket(chunk->used);
        pointer = chunk->pointer;
        used = chunk->used;
        chunk->pointer = NULL;
        chunk->used = 0;
        chunk->size = 0;
        LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
          &pool->hist[hist_b].nevicts_limit, 1, LIBXS_ATOMIC_RELAXED);
        break;
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &pool->plock);
    if (NULL == pointer) break;
    internal_libxs_malloc_deallocate(pool, pointer);
    reclaimed += used;
  }
  if (NULL != current->pointer) {
    const int hist_b = internal_libxs_malloc_hist_bucket(current->used);
    internal_libxs_malloc_deallocate(pool, current->pointer);
    reclaimed += current->used;
    current->pointer = NULL;
    current->used = 0;
    current->size = 0;
    LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
      &pool->hist[hist_b].nevicts_limit, 1, LIBXS_ATOMIC_RELAXED);
  }
  if (0 != reclaimed) {
    LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(
      &pool->pool_bytes, reclaimed, LIBXS_ATOMIC_LOCKORDER);
  }
  return reclaimed;
}
#endif


LIBXS_API void* libxs_malloc(libxs_malloc_pool_t* pool, size_t size, int alignment)
{
  void *result = NULL;
  if (NULL == pool) pool = internal_libxs_malloc_default_pool();
  if (NULL != pool && 0 != size) {
    internal_libxs_malloc_chunk_t *chunk = NULL;
    void **info = NULL;
#if defined(LIBXS_MALLOC_SEARCH)
    internal_libxs_malloc_chunk_t **hit = NULL;
    size_t i = 0, diff = (size_t)-1;
    LIBXS_ASSERT(NULL != pool->slots);
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &pool->plock);
    if (0 < pool->slots_num) {
      do {
        internal_libxs_malloc_chunk_t *const c = pool->slots[i];
        if (size <= c->size) {
          const size_t delta = c->size - size;
          if (delta < diff) {
            diff = delta; hit = pool->slots + i;
            if (c->size < (size + LIBXS_MALLOC_UPSIZE)) break;
          }
        }
      } while (++i < pool->slots_num);
      { /* allocate slot and eventually reuse */
        internal_libxs_malloc_chunk_t **const alloc = pool->slots + --pool->slots_num;
        if (hit != alloc && NULL != hit && size <= (*hit)->size) {
          LIBXS_VALUE_SWAP(*alloc, *hit);
        }
        chunk = *alloc;
      }
    }
    else { /* pool empty: allocate new chunk on demand */
      chunk = (internal_libxs_malloc_chunk_t*)calloc(1, sizeof(internal_libxs_malloc_chunk_t));
      if (NULL != chunk) {
        chunk->pool = pool;
        chunk->next = pool->all;
        pool->all = chunk;
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &pool->plock);
#else
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &pool->plock);
    if (0 < pool->slots_num) {
      chunk = pool->slots[--pool->slots_num];
    }
    else { /* pool empty: allocate new chunk on demand */
      chunk = (internal_libxs_malloc_chunk_t*)calloc(1, sizeof(internal_libxs_malloc_chunk_t));
      if (NULL != chunk) {
        chunk->pool = pool;
        chunk->next = pool->all;
        pool->all = chunk;
      }
    }
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &pool->plock);
#endif
    if (NULL != chunk) {
      const int hist_b = internal_libxs_malloc_hist_bucket(size);
      libxs_registry_t *const reg = internal_libxs_malloc_get_registry(alignment);
      const size_t alignpot = LIBXS_UP2POT(LIBXS_MAX(sizeof(void*) + 1,
        1 < alignment ? (size_t)alignment : LIBXS_ALIGNMENT));
      const size_t alloc_size = (NULL != reg)
        ? size : (size + sizeof(void*) + alignpot - 1);
#if defined(LIBXS_MALLOC_EVICT)
      const size_t nmallocs = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
        &pool->generation, 1, LIBXS_ATOMIC_LOCKORDER);
#endif
      LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&pool->hist[hist_b].count, 1, LIBXS_ATOMIC_RELAXED);
#if defined(LIBXS_MALLOC_EVICT)
      if (NULL != chunk->pointer && chunk->last_tick != 0 &&
          LIBXS_MALLOC_EVICT_AGE < nmallocs / LIBXS_MALLOC_TICK_RATE - chunk->last_tick)
      {
        internal_libxs_malloc_deallocate(pool, chunk->pointer);
        chunk->pointer = NULL;
        LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(&pool->pool_bytes, chunk->used, LIBXS_ATOMIC_LOCKORDER);
        LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&pool->hist[hist_b].nevicts_age, 1, LIBXS_ATOMIC_RELAXED);
        chunk->used = 0;
        chunk->size = 0;
      }
#endif
      if (NULL != chunk->pointer) {
        if (chunk->size < alloc_size) {
          char *pointer;
          if (0 < pool->max_nthreads || NULL != pool->fn_malloc.std) {
            pointer = (char*)internal_libxs_malloc_allocate(pool, alloc_size);
#if defined(LIBXS_MALLOC_EVICT)
            if (NULL == pointer && 0 != internal_libxs_malloc_evict_available(pool, chunk)) {
              pointer = (char*)internal_libxs_malloc_allocate(pool, alloc_size);
            }
#endif
            if (NULL != pointer) internal_libxs_malloc_deallocate(pool, chunk->pointer);
          }
          else { /* default: realloc */
            pointer = realloc(chunk->pointer, alloc_size);
#if defined(LIBXS_MALLOC_EVICT)
            if (NULL == pointer && 0 != internal_libxs_malloc_evict_available(pool, chunk)) {
              pointer = realloc(chunk->pointer, alloc_size);
            }
#endif
          }
          if (NULL != pointer) {
#if defined(LIBXS_MALLOC_EVICT)
            const size_t old_used = chunk->used;
            const int moved = (pointer != chunk->pointer);
#endif
            chunk->pointer = pointer;
            chunk->used = size;
            chunk->size = alloc_size;
            ++chunk->nmallocs;
            LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&pool->hist[hist_b].ngrows, 1, LIBXS_ATOMIC_RELAXED);
#if defined(LIBXS_MALLOC_EVICT)
            { const size_t new_bytes = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                &pool->pool_bytes, size - old_used, LIBXS_ATOMIC_LOCKORDER);
              internal_libxs_malloc_peak_update(pool, 0 != moved ? new_bytes + old_used : new_bytes);
            }
#endif
            if (NULL != reg) {
              result = pointer;
              if (NULL == libxs_registry_set(reg,
                &result, sizeof(void*), &chunk, sizeof(void*), libxs_registry_lock(reg)))
              {
                internal_libxs_malloc_pool_return(chunk);
                result = NULL;
              }
            }
            else {
              result = LIBXS_ALIGN(pointer + sizeof(void*), alignpot);
              info = (void**)((uintptr_t)result - sizeof(void*));
              *info = chunk;
            }
          }
          else { /* alloc failed: original pointer intact, return chunk to pool */
            internal_libxs_malloc_pool_return(chunk);
          }
        }
        else { /* reuse */
          chunk->used = size;
          LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&pool->hist[hist_b].nreuses, 1, LIBXS_ATOMIC_RELAXED);
          if (NULL != reg) {
            result = chunk->pointer;
            if (NULL == libxs_registry_set(reg,
              &result, sizeof(void*), &chunk, sizeof(void*), libxs_registry_lock(reg)))
            {
              internal_libxs_malloc_pool_return(chunk);
              result = NULL;
            }
          }
          else {
            result = LIBXS_ALIGN(chunk->pointer + sizeof(void*), alignpot);
          }
        }
      }
      else { char *pointer;
        pointer = (char*)internal_libxs_malloc_allocate(pool, alloc_size);
#if defined(LIBXS_MALLOC_EVICT)
        if (NULL == pointer && 0 != internal_libxs_malloc_evict_available(pool, chunk)) {
          pointer = (char*)internal_libxs_malloc_allocate(pool, alloc_size);
        }
#endif
        if (NULL != pointer) {
          LIBXS_ASSERT(0 == chunk->used);
          chunk->pointer = pointer;
          chunk->used = size;
          chunk->size = alloc_size;
          ++chunk->nmallocs;
#if defined(LIBXS_MALLOC_EVICT)
          { const size_t new_bytes = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
              &pool->pool_bytes, size, LIBXS_ATOMIC_LOCKORDER);
            internal_libxs_malloc_peak_update(pool, new_bytes);
          }
#endif
          if (NULL != reg) {
            result = pointer;
            if (NULL == libxs_registry_set(reg,
              &result, sizeof(void*), &chunk, sizeof(void*), libxs_registry_lock(reg)))
            {
              internal_libxs_malloc_pool_return(chunk);
              result = NULL;
            }
          }
          else {
            result = LIBXS_ALIGN(pointer + sizeof(void*), alignpot);
            info = (void**)((uintptr_t)result - sizeof(void*));
            *info = chunk;
          }
        }
        else { /* malloc failed: return empty chunk to pool */
          internal_libxs_malloc_pool_return(chunk);
        }
      }
    }
  }
  return result;
}


LIBXS_API void libxs_free(void* pointer)
{
  if (NULL != pointer) {
    internal_libxs_malloc_chunk_t *chunk = NULL;
    libxs_malloc_pool_t *pool;
    const int found = (NULL != internal_libxs_malloc_registry)
      ? libxs_registry_extract(internal_libxs_malloc_registry,
          &pointer, sizeof(void*), &chunk, sizeof(chunk),
          libxs_registry_lock(internal_libxs_malloc_registry))
      : 0;
    if (0 == found) {
      chunk = *(void**)((uintptr_t)pointer - sizeof(void*));
    }
    LIBXS_ASSERT(NULL != chunk && NULL != chunk->pool);
    pool = chunk->pool;
    LIBXS_ASSERT(NULL != pool && NULL != pool->slots);
#if defined(LIBXS_MALLOC_EVICT)
    if (NULL != chunk->pointer && LIBXS_MALLOC_EVICT_SIZE <= chunk->used) {
      const size_t total_bytes = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_bytes, LIBXS_ATOMIC_RELAXED);
      if (0 == internal_libxs_malloc_evict_limit) {
        const char *const env = getenv("LIBXS_MALLOC_EVICT_LIMIT");
        internal_libxs_malloc_evict_limit = (NULL != env && '\0' != *env)
          ? (size_t)atol(env) << 20 : LIBXS_MALLOC_EVICT_LIMIT;
      }
      if (internal_libxs_malloc_evict_limit < total_bytes) {
        const int evict_b = internal_libxs_malloc_hist_bucket(chunk->used);
        const size_t reclaimed = chunk->used;
        internal_libxs_malloc_deallocate(pool, chunk->pointer);
        chunk->pointer = NULL;
        chunk->used = 0;
        chunk->size = 0;
        LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(&pool->pool_bytes, reclaimed, LIBXS_ATOMIC_LOCKORDER);
        LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(&pool->hist[evict_b].nevicts_limit, 1, LIBXS_ATOMIC_RELAXED);
      }
    }
    chunk->last_tick = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->generation, LIBXS_ATOMIC_RELAXED)
      / LIBXS_MALLOC_TICK_RATE;
#endif
    internal_libxs_malloc_pool_return(chunk);
  }
}


LIBXS_API int libxs_malloc_info(const void* pointer, libxs_malloc_info_t* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    LIBXS_MEMZERO(info);
    if (NULL != pointer) {
      internal_libxs_malloc_chunk_t *chunk = NULL;
      const int found = (NULL != internal_libxs_malloc_registry)
        ? libxs_registry_get_copy(internal_libxs_malloc_registry,
            &pointer, sizeof(void*), &chunk, sizeof(chunk),
            libxs_registry_lock(internal_libxs_malloc_registry))
        : 0;
      if (0 == found) {
        chunk = *(void**)((uintptr_t)pointer - sizeof(void*));
      }
      info->size = chunk->used;
    }
  }
  else if (NULL != pointer) {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API libxs_malloc_pool_t* libxs_malloc_pool(libxs_malloc_fn malloc_fn, libxs_free_fn free_fn)
{
  libxs_malloc_pool_t *pool;
  internal_libxs_hash_init(libxs_cpuid(NULL));
  pool = (libxs_malloc_pool_t*)calloc(1, sizeof(libxs_malloc_pool_t));
  if (NULL != pool) {
    pool->slots = (internal_libxs_malloc_chunk_t**)malloc(
      LIBXS_MALLOC_POOL_INIT * sizeof(void*));
    if (NULL != pool->slots) {
      pool->fn_malloc.std = malloc_fn;
      pool->fn_free.std = free_fn;
      pool->extra = NULL;
      pool->max_nthreads = 0;
      pool->slots_size = LIBXS_MALLOC_POOL_INIT;
      pool->slots_num = 0;
      pool->all = NULL;
    }
    else {
      free(pool);
      pool = NULL;
    }
  }
  return pool;
}


LIBXS_API libxs_malloc_pool_t* libxs_malloc_xpool(libxs_malloc_xfn malloc_fn, libxs_free_xfn free_fn,
  int max_nthreads)
{
  libxs_malloc_pool_t *pool;
  if (NULL == malloc_fn || NULL == free_fn || 1 > max_nthreads) return NULL;
  internal_libxs_hash_init(libxs_cpuid(NULL));
  pool = (libxs_malloc_pool_t*)calloc(1, sizeof(libxs_malloc_pool_t));
  if (NULL != pool) {
    pool->extra = (const void**)calloc(max_nthreads, sizeof(void*));
    pool->slots = (internal_libxs_malloc_chunk_t**)malloc(
      LIBXS_MALLOC_POOL_INIT * sizeof(void*));
    if (NULL != pool->slots && NULL != pool->extra) {
      pool->fn_malloc.ext = malloc_fn;
      pool->fn_free.ext = free_fn;
      pool->max_nthreads = max_nthreads;
      pool->slots_size = LIBXS_MALLOC_POOL_INIT;
      pool->slots_num = 0;
      pool->all = NULL;
    }
    else {
      free(pool->extra);
      free(pool->slots);
      free(pool);
      pool = NULL;
    }
  }
  return pool;
}


LIBXS_API void libxs_malloc_arg(libxs_malloc_pool_t* pool, const void* extra)
{
  if (NULL != pool && 0 < pool->max_nthreads) {
    pool->extra[libxs_tid() % pool->max_nthreads] = extra;
  }
}


LIBXS_API void libxs_free_pool(libxs_malloc_pool_t* pool)
{
  if (NULL != pool) {
    if (LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity) {
      libxs_malloc_pool_print(stderr,
        0 < pool->max_nthreads ? "INFO LIBXS: xpool " : "INFO LIBXS: pool ", pool);
    }
    if (NULL != pool->slots) {
      libxs_registry_t *const reg = internal_libxs_malloc_registry;
      internal_libxs_malloc_chunk_t *chunk = pool->all;
      while (NULL != chunk) {
        internal_libxs_malloc_chunk_t *const next = chunk->next;
        if (NULL != chunk->pointer) {
          if (NULL != reg) {
            const void *const ptr = chunk->pointer;
            libxs_registry_remove(reg, &ptr, sizeof(void*), libxs_registry_lock(reg));
          }
          internal_libxs_malloc_deallocate(pool, chunk->pointer);
        }
        free(chunk);
        chunk = next;
      }
      pool->all = NULL;
#if defined(LIBXS_MALLOC_EVICT)
      pool->pool_bytes = 0;
      pool->pool_peak = 0;
#endif
      pool->slots_size = pool->slots_num = 0;
      free(pool->slots);
      pool->slots = NULL;
    }
    free(pool->extra);
    pool->extra = NULL;
    free(pool);
  }
}


LIBXS_API int libxs_malloc_pool_info(const libxs_malloc_pool_t* pool, libxs_malloc_pool_info_t* info)
{
  int result = EXIT_SUCCESS;
  if (NULL == pool) pool = internal_libxs_malloc_default_pool();
  if (NULL != info) {
    memset(info, 0, sizeof(*info));
    if (NULL != pool && NULL != pool->slots) {
      const internal_libxs_malloc_chunk_t *chunk = pool->all;
      size_t nchunks = 0, i;
      while (NULL != chunk) {
        info->nmallocs += chunk->nmallocs;
        info->used += chunk->used;
        info->size += chunk->size;
        chunk = chunk->next;
        ++nchunks;
      }
      info->nactive = nchunks - pool->slots_num;
#if defined(LIBXS_MALLOC_EVICT)
      info->peak = pool->pool_peak;
#endif
      for (i = 0; i < LIBXS_MALLOC_HIST_NBUCKETS; ++i) info->hist[i] = pool->hist[i];
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void libxs_malloc_pool_print(FILE* ostream, const char prefix[],
  const libxs_malloc_pool_t* pool)
{
  static const char *const labels[] = { "<1M", "1-4M", "4-16M", "16-64M", "64-256M", ">=256M" };
  libxs_malloc_pool_info_t info;
  if (NULL != ostream && EXIT_SUCCESS == libxs_malloc_pool_info(pool, &info) && 0 != info.nmallocs) {
    size_t i, total_evict_limit = 0, total_evict_age = 0;
    if (NULL != prefix) fprintf(ostream, "%s", prefix);
    fprintf(ostream, "used=%llu size=%llu peak=%llu allocs=%llu active=%llu\n",
      (unsigned long long)(info.used >> 20), (unsigned long long)(info.size >> 20),
      (unsigned long long)(info.peak >> 20), (unsigned long long)info.nmallocs,
      (unsigned long long)info.nactive);
    for (i = 0; i < LIBXS_MALLOC_HIST_NBUCKETS; ++i) {
      const libxs_malloc_pool_hist_t *const h = info.hist + i;
      if (0 != h->count) {
        if (NULL != prefix) fprintf(ostream, "%s", prefix);
        fprintf(ostream, "  %s: %llu (reuse=%llu grow=%llu evict_age=%llu evict_limit=%llu)\n",
          labels[i], (unsigned long long)h->count, (unsigned long long)h->nreuses,
          (unsigned long long)h->ngrows, (unsigned long long)h->nevicts_age,
          (unsigned long long)h->nevicts_limit);
      }
      total_evict_limit += h->nevicts_limit;
      total_evict_age += h->nevicts_age;
    }
    if (0 != total_evict_limit || 0 != total_evict_age) {
      const size_t suggest_limit = info.peak + (info.peak >> 2);
      if (NULL != prefix) fprintf(ostream, "%s", prefix);
      fprintf(ostream, "  suggest: LIBXS_MALLOC_EVICT_LIMIT=%llu",
        (unsigned long long)(suggest_limit >> 20));
      if (0 != total_evict_age) {
        fprintf(ostream, " LIBXS_MALLOC_EVICT_AGE=+");
      }
      fprintf(ostream, "\n");
    }
  }
}


LIBXS_API libxs_malloc_pool_t* libxs_default_pool(void)
{
  return internal_libxs_malloc_default_pool();
}
