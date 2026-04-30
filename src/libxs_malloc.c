/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_malloc.h>
#include <libxs_mem.h>
#include <libxs_reg.h>
#include "libxs_crc32.h"

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
#if !defined(LIBXS_MALLOC_EVICT) && 1
# define LIBXS_MALLOC_EVICT
#endif
#if !defined(LIBXS_MALLOC_SEARCH) && 1
# define LIBXS_MALLOC_SEARCH
#endif
#if !defined(LIBXS_MALLOC_POOL_INIT)
# define LIBXS_MALLOC_POOL_INIT 16
#endif


typedef struct internal_libxs_malloc_chunk_t {
  char *pointer;
  struct internal_libxs_malloc_chunk_t *next;
  struct libxs_malloc_pool_t *pool; /* back-pointer to owning pool */
  size_t used, size, nmallocs;
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
  size_t pool_bytes;
  size_t pool_peak;
#endif
};

LIBXS_APIVAR_DEFINE(libxs_lock_t internal_libxs_malloc_plocks[LIBXS_MALLOC_NLOCKS]);
LIBXS_APIVAR_DEFINE(libxs_registry_t* internal_libxs_malloc_registry);


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


#if defined(LIBXS_MALLOC_EVICT)
LIBXS_API_INLINE void internal_libxs_malloc_peak_update(libxs_malloc_pool_t *pool, size_t candidate)
{
  size_t peak = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_peak, LIBXS_ATOMIC_RELAXED);
  while (candidate > peak) {
    if (LIBXS_ATOMIC(LIBXS_ATOMIC_CMPSWP, LIBXS_BITS)(&pool->pool_peak, peak, candidate, LIBXS_ATOMIC_LOCKORDER)) break;
    peak = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_peak, LIBXS_ATOMIC_RELAXED);
  }
}
#endif


LIBXS_API void* libxs_malloc(libxs_malloc_pool_t* pool, size_t size, int alignment)
{
  void *result = NULL;
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
        const size_t delta = LIBXS_DELTA(c->size, size);
        if (delta < diff) {
          diff = delta; hit = pool->slots + i;
          if (size <= c->size && c->size < (size + LIBXS_MALLOC_UPSIZE)) break;
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
      libxs_registry_t *const reg = internal_libxs_malloc_get_registry(alignment);
      const size_t alignpot = LIBXS_UP2POT(LIBXS_MAX(sizeof(void*) + 1,
        1 < alignment ? (size_t)alignment : LIBXS_ALIGNMENT));
      const size_t alloc_size = (NULL != reg)
        ? size : (size + sizeof(void*) + alignpot - 1);
      if (NULL != chunk->pointer) {
        if (chunk->size < alloc_size) {
          char *pointer;
          if (0 < pool->max_nthreads) { /* extended pool */
            const void *const xarg = pool->extra[libxs_tid() % pool->max_nthreads];
            pointer = (char*)pool->fn_malloc.ext(alloc_size, xarg);
            if (NULL != pointer) pool->fn_free.ext(chunk->pointer, xarg);
          }
          else if (pool->fn_malloc.std) { /* standard custom pool */
            pointer = (char*)pool->fn_malloc.std(alloc_size);
            if (NULL != pointer) pool->fn_free.std(chunk->pointer);
          }
          else { /* default: realloc */
            pointer = realloc(chunk->pointer, alloc_size);
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
        if (0 < pool->max_nthreads) {
          pointer = (char*)pool->fn_malloc.ext(alloc_size, pool->extra[libxs_tid() % pool->max_nthreads]);
        }
        else if (pool->fn_malloc.std) pointer = (char*)pool->fn_malloc.std(alloc_size);
        else pointer = (char*)malloc(alloc_size);
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
    LIBXS_ASSERT(NULL != chunk);
    pool = chunk->pool;
    LIBXS_ASSERT(NULL != pool && NULL != pool->slots);
#if defined(LIBXS_MALLOC_EVICT)
    if (NULL != chunk->pointer && LIBXS_MALLOC_EVICT_SIZE <= chunk->used) {
      const size_t total_bytes = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&pool->pool_bytes, LIBXS_ATOMIC_RELAXED);
      if (LIBXS_MALLOC_EVICT_LIMIT < total_bytes) {
        const size_t reclaimed = chunk->used;
        if (0 < pool->max_nthreads) {
          pool->fn_free.ext(chunk->pointer, pool->extra[libxs_tid() % pool->max_nthreads]);
        }
        else if (pool->fn_free.std) pool->fn_free.std(chunk->pointer);
        else free(chunk->pointer);
        chunk->pointer = NULL;
        chunk->used = 0;
        chunk->size = 0;
        LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(&pool->pool_bytes, reclaimed, LIBXS_ATOMIC_LOCKORDER);
      }
    }
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
          if (0 < pool->max_nthreads) {
            pool->fn_free.ext(chunk->pointer, pool->extra[libxs_tid() % pool->max_nthreads]);
          }
          else if (pool->fn_free.std) pool->fn_free.std(chunk->pointer);
          else free(chunk->pointer);
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
  if (NULL != info) {
    LIBXS_MEMZERO(info);
    if (NULL != pool && NULL != pool->slots) {
      const internal_libxs_malloc_chunk_t *chunk = pool->all;
      size_t nchunks = 0;
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
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}
