/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_reg.h>
#include <libxs_mem.h>
#include "libxs_hash.h"
#include "libxs_main.h"
#include "libxs_diff.h"

#include <signal.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if !defined(_WIN32)
# if defined(__GNUC__) || defined(__PGI) || defined(_CRAYC)
#   include <sys/time.h>
#   include <time.h>
# endif
# include <sys/types.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <fcntl.h>
#endif
#if defined(__APPLE__)
# include <libkern/OSCacheControl.h>
/*# include <mach/mach_time.h>*/
# include <pthread.h>
#endif
#if defined(__powerpc64__)
# include <sys/platform/ppc.h>
#endif

/* used internally to re-implement certain exit-handler */
#if !defined(LIBXS_EXIT_SUCCESS)
# define LIBXS_EXIT_SUCCESS() exit(EXIT_SUCCESS)
#endif
#if !defined(LIBXS_CODE_MAXSIZE)
# define LIBXS_CODE_MAXSIZE 131072
#endif
#if !defined(LIBXS_DIFF_SIZE)
# define LIBXS_DESCRIPTOR_SIGSIZE LIBXS_DESCRIPTOR_MAXSIZE
#endif
#if !defined(LIBXS_DIFF_SIZE)
# define LIBXS_DIFF_SIZE LIBXS_DESCRIPTOR_SIGSIZE
#endif
#if !defined(LIBXS_HASH_SIZE)
/* can be smaller than MAXSIZE/SIGSIZE at the expense of collisions */
# define LIBXS_HASH_SIZE 32
#endif
#if !defined(LIBXS_HASH_SEED)
# define LIBXS_HASH_SEED 25071975
#endif
#if !defined(LIBXS_ENABLE_DEREG) && 0
# define LIBXS_ENABLE_DEREG
#endif
#if !defined(LIBXS_REGUSER_HASH) && 1
# define LIBXS_REGUSER_HASH
#endif
#if !defined(LIBXS_REGUSER_ALIGN) && 1
# define LIBXS_REGUSER_ALIGN
#endif
#if !defined(LIBXS_REGLOCK_TRY) && 0
# define LIBXS_REGLOCK_TRY
#endif
#if !defined(LIBXS_UNIFY_LOCKS) && 1
# define LIBXS_UNIFY_LOCKS
#endif
#if !defined(LIBXS_REGKEY_PAD) && 0
# define LIBXS_REGKEY_PAD
#endif
#if !defined(LIBXS_CACHE_PAD) && 1
# define LIBXS_CACHE_PAD
#endif
#if !defined(LIBXS_AUTOPIN) && 0
# define LIBXS_AUTOPIN
#endif
#if !defined(LIBXS_MAIN_DELIMS)
# define LIBXS_MAIN_DELIMS ";,:"
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 1 /* beneficial when registry approaches capacity (collisions) */
# define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif
#if !defined(LIBXS_COLLISION_COUNT_STATIC) && 0
# define LIBXS_COLLISION_COUNT_STATIC
#endif

#if (0 != LIBXS_SYNC)
# if !defined(INTERNAL_REGLOCK_MAXN)
#   if defined(_MSC_VER)
#     define INTERNAL_REGLOCK_MAXN 0
#   else
#     define INTERNAL_REGLOCK_MAXN 0
#   endif
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   if !defined(LIBXS_CACHE_MAXSIZE) && (8 > INTERNAL_REGLOCK_MAXN)
#     define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#   endif
#   if !defined(LIBXS_REGLOCK)
#     define LIBXS_REGLOCK LIBXS_LOCK_DEFAULT
#   endif
#   if !defined(LIBXS_CLEANUP_NTRY)
#     define LIBXS_CLEANUP_NTRY 7
#   endif
#   if LIBXS_LOCK_TYPE_ISPOD(LIBXS_REGLOCK)
LIBXS_EXTERN_C typedef union internal_reglocktype {
  char pad[LIBXS_CACHELINE];
  LIBXS_LOCK_TYPE(LIBXS_REGLOCK) state;
} internal_reglocktype;
#   else
LIBXS_EXTERN_C typedef union internal_reglocktype {
  LIBXS_LOCK_TYPE(LIBXS_REGLOCK) state;
} internal_reglocktype;
#   endif
LIBXS_APIVAR_DEFINE(internal_reglocktype internal_reglock[INTERNAL_REGLOCK_MAXN]);
# else /* RW-lock */
#   if !defined(LIBXS_CACHE_MAXSIZE)
#     define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#   endif
#   if !defined(LIBXS_REGLOCK)
#     if defined(LIBXS_UNIFY_LOCKS)
#       define LIBXS_REGLOCK LIBXS_LOCK
#     elif defined(_MSC_VER)
#       define LIBXS_REGLOCK LIBXS_LOCK_MUTEX
#     elif 0
#       define LIBXS_REGLOCK LIBXS_LOCK_RWLOCK
#     else
#       define LIBXS_REGLOCK LIBXS_LOCK_DEFAULT
#     endif
#   endif
LIBXS_APIVAR_DEFINE(LIBXS_LOCK_TYPE(LIBXS_REGLOCK)* internal_reglock_ptr);
# endif
#elif !defined(LIBXS_CACHE_MAXSIZE)
# define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#endif
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
# define LIBXS_CACHE_STRIDE LIBXS_MAX(sizeof(libxs_descriptor_t), LIBXS_DESCRIPTOR_MAXSIZE)
#else
# define LIBXS_CACHE_STRIDE LIBXS_DESCRIPTOR_MAXSIZE
#endif

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# define INTERNAL_FIND_CODE_CACHE_GROW(RESULT_INDEX, CACHE_SIZE) \
    RESULT_INDEX = CACHE_SIZE; CACHE_SIZE = (unsigned char)(0 != (CACHE_SIZE) ? ((CACHE_SIZE) << 1) : 1)
# define INTERNAL_FIND_CODE_CACHE_EVICT(RESULT_INDEX, CACHE_SIZE, CACHE_HIT) \
    RESULT_INDEX = (unsigned char)LIBXS_MOD2((CACHE_HIT) + ((CACHE_SIZE) - 1), CACHE_SIZE)
#endif

#if (0 == LIBXS_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) {
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# if defined(LIBXS_REGLOCK_TRY)
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
    if (1 != internal_reglock_count) { /* (re-)try and get (meanwhile) generated code */ \
      LIBXS_ASSERT(NULL != internal_registry_state && NULL != internal_registry_state->registry); /* engine is not shut down */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; break; \
    }
# else
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
      LIBXS_ASSERT(NULL != internal_registry_state && NULL != internal_registry_state->registry); /* engine is not shut down */ \
      continue
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) /*do*/ { \
      const unsigned int LOCKINDEX = (0 != internal_reglock_count ? LIBXS_MOD2(INDEX, internal_reglock_count) : 0); \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state); } while(0)
# else /* RW-lock */
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) /*do*/ { \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, internal_reglock_ptr)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr); } /*while(0)*/
# endif
#endif

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
LIBXS_EXTERN_C typedef struct internal_cache_entry_type {
  libxs_descriptor_t keys[LIBXS_CACHE_MAXSIZE];
  libxs_code_pointer_t code[LIBXS_CACHE_MAXSIZE];
  unsigned int id; /* to invalidate */
  unsigned char size, hit;
} internal_cache_entry_type;

LIBXS_EXTERN_C typedef union internal_cache_type {
# if defined(LIBXS_CACHE_PAD)
  char pad[LIBXS_UP2(sizeof(internal_cache_entry_type),LIBXS_CACHELINE)];
# endif
  internal_cache_entry_type entry;
} internal_cache_type;

#endif /*defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))*/
LIBXS_EXTERN_C typedef union internal_regkey_type {
#if defined(LIBXS_REGKEY_PAD)
  char pad[LIBXS_UP2(sizeof(libxs_descriptor_t), LIBXS_CACHELINE)];
#endif
  libxs_descriptor_t entry;
} internal_regkey_type;

struct libxs_registry_t {
  libxs_code_pointer_t* registry;
  internal_regkey_type* keys;
  unsigned int capacity;
  unsigned int size;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
  internal_cache_type* cache_buffer;
  int cache_size;
#endif
};

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXS_APIVAR_DEFINE(int internal_reglock_count);
LIBXS_APIVAR_DEFINE(libxs_registry_t* internal_registry_state);


LIBXS_API void libxs_registry_create(libxs_registry_t** registry)
{
  libxs_registry_t *const r = malloc(sizeof(libxs_registry_t));
  LIBXS_ASSERT(NULL != registry);
  if (NULL != r) {
    *registry = r;
  }
}


LIBXS_API void libxs_registry_destroy(libxs_registry_t* registry)
{
  if (NULL != registry) {
    free(registry);
  }
}


LIBXS_API_INLINE void internal_pad_descriptor(libxs_descriptor_t* desc, signed char size)
{
  LIBXS_ASSERT(LIBXS_DESCRIPTOR_MAXSIZE < 128 && NULL != desc);
  LIBXS_ASSERT(LIBXS_DIFF_SIZE <= LIBXS_DESCRIPTOR_MAXSIZE);
  LIBXS_ASSERT(LIBXS_HASH_SIZE <= LIBXS_DIFF_SIZE);
  for (; size < LIBXS_DIFF_SIZE; ++size) desc->data[size] = 0;
}


LIBXS_API_INLINE libxs_code_pointer_t internal_find_code(libxs_registry_t* registry_state, libxs_descriptor_t* desc, size_t desc_size, size_t user_size)
{
  libxs_code_pointer_t flux_entry = { 0 };
  libxs_code_pointer_t *const registry = (NULL != registry_state ? registry_state->registry : NULL);
  internal_regkey_type *const registry_keys = (NULL != registry_state ? registry_state->keys : NULL);
  const unsigned int registry_capacity = (NULL != registry_state ? registry_state->capacity : 0);
  const int is_big_desc = 0; /*LIBXS_DESCRIPTOR_ISBIG(desc->kind);*/
  const signed char size = (signed char)(sizeof(libxs_descriptor_kind_t) + desc_size);
  LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
  const int registry_cache_size = (NULL != registry_state ? registry_state->cache_size : 0);
# if defined(LIBXS_NTHREADS_USE)
  const unsigned int tid = libxs_tid();
  internal_cache_type *const cache = (NULL != registry_state && NULL != registry_state->cache_buffer
    ? registry_state->cache_buffer + tid : NULL);
# else
  static LIBXS_TLS internal_cache_type internal_cache_buffer /*= { 0 }*/;
  internal_cache_type *const cache = &internal_cache_buffer;
# endif
  unsigned char cache_index;
  const unsigned int ninit = LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_SEQ_CST);
  LIBXS_ASSERT(NULL != registry_state && NULL != registry && NULL != registry_keys);
  LIBXS_ASSERT(0 != registry_capacity);
  internal_pad_descriptor(desc, size);
  if (0 == is_big_desc) {
    LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
    LIBXS_DIFF_N(unsigned char, cache_index, LIBXS_DIFF(LIBXS_DIFF_SIZE), xdesc, cache->entry.keys,
      LIBXS_DIFF_SIZE, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  else {
    cache_index = (unsigned char)libxs_diff_n(desc, cache->entry.keys,
      size, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  if (0 != registry_cache_size && NULL != cache
    && ninit == cache->entry.id && cache_index < cache->entry.size) { /* valid hit */
    flux_entry = cache->entry.code[cache_index];
    cache->entry.hit = cache_index;
  }
  else
#else
  internal_pad_descriptor(desc, size);
#endif
  {
    unsigned int i, i0, mode = 0, diff = 1;
    unsigned int hash = LIBXS_CRC32(LIBXS_HASH_SIZE)(LIBXS_HASH_SEED, desc);
    i0 = i = LIBXS_MOD2(hash, registry_capacity);
    LIBXS_ASSERT(NULL != registry);
    do { /* use calculated location and check if the requested code is already JITted */
#if (1 < INTERNAL_REGLOCK_MAXN) || !LIBXS_LOCK_TYPE_ISRW(LIBXS_REGLOCK) /* read registered code */
# if 1 /* omitting an atomic load is safe but avoids race-detectors to highlight this location */
      uintptr_t *const fluxaddr = &registry[i].uval;
      flux_entry.uval = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(fluxaddr, LIBXS_ATOMIC_RELAXED);
# else
      flux_entry = registry[i];
# endif
#else
      LIBXS_LOCK_ACQREAD(LIBXS_REGLOCK, internal_reglock_ptr);
      flux_entry = registry[i]; /* read registered code */
      LIBXS_LOCK_RELREAD(LIBXS_REGLOCK, internal_reglock_ptr);
#endif
      if ((NULL != flux_entry.ptr_const || 1 == mode) && 2 > mode) { /* confirm entry */
        if (NULL != flux_entry.ptr_const) {
          if (0 == is_big_desc) {
#if !defined(LIBXS_CACHE_MAXSIZE) || (0 == (LIBXS_CACHE_MAXSIZE))
            LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
#endif
            diff = LIBXS_DIFF(LIBXS_DIFF_SIZE)(xdesc, registry_keys + i, 0/*dummy*/);
          }
          else {
            diff = libxs_diff(desc, registry_keys + i, size);
          }
        }
#if !defined(NDEBUG)
        else LIBXS_ASSERT(0 != diff);
#endif
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXS_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXS_HASH_COLLISION & flux_entry.uval)) {
              LIBXS_ASSERT(NULL != flux_entry.ptr_const); /* collision */
              mode = 3;
            }
            else
#endif      /* search for an existing code version */
            mode = 1; /* else */
          }
          i = LIBXS_MOD2(i + 1, registry_capacity);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXS_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
          }
          LIBXS_ASSERT(0 != diff); /* continue */
        }
      }
#if 0
      else { /* enter code generation (there is no code version yet) */
        LIBXS_ASSERT(0 == mode || 1 < mode);
        if (LIBXS_X86_GENERIC <= libxs_target_archid || /* check if JIT is supported (CPUID) */
           (LIBXS_KERNEL_KIND_USER == LIBXS_DESCRIPTOR_KIND(desc->kind)))
        {
          LIBXS_ASSERT(0 != mode || NULL == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.ptr); /* lock the registry entry */
              if (NULL == registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxs_build_request request /*= { 0 }*/; /* setup the code build request */
            LIBXS_ASSERT(LIBXS_KERNEL_UNREGISTERED > LIBXS_DESCRIPTOR_KIND(desc->kind));
            request.kind = (libxs_build_kind)LIBXS_DESCRIPTOR_KIND(desc->kind);
            request.descriptor.ptr = &desc->gemm.desc;
            request.user_size = user_size;
# if defined(NDEBUG)
            if (EXIT_SUCCESS == libxs_build(&request, i, &flux_entry) && NULL != flux_entry.ptr_const)
# else
            build = libxs_build(&request, i, &flux_entry);
            if (EXIT_SUCCESS == build && NULL != flux_entry.ptr_const)
# endif
            {
              LIBXS_ASSIGN127(registry_keys + i, desc);
# if (1 < INTERNAL_REGLOCK_MAXN)
              LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&registry[i].ptr, flux_entry.ptr, LIBXS_ATOMIC_SEQ_CST);
# else
              registry[i] = flux_entry;
# endif
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer_t fix_entry;
#   if (1 < INTERNAL_REGLOCK_MAXN)
                fix_entry.ptr = (void*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
                  &registry[i0].ptr, LIBXS_ATOMIC_RELAXED);
#   else
                fix_entry = registry[i0];
#   endif
                LIBXS_ASSERT(NULL != fix_entry.ptr_const);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
#   if (1 < INTERNAL_REGLOCK_MAXN)
                  LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&registry[i0].ptr,
                    fix_entry.ptr, LIBXS_ATOMIC_RELAXED);
#   else
                  registry[i0] = fix_entry;
#   endif
                }
              }
# endif
              if (registry_state->size < registry_capacity && NULL != flux_entry.ptr_const) {
                ++registry_state->size;
              }
            }
            if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
            }
            /* leave here even in case of a build-error; do not use break (inside of locked region) */
            diff = 0;
          }
          INTERNAL_FIND_CODE_UNLOCK(lock);
          if (0 != diff) { /* acquire registry slot */
            if (0 == mode) { /* initial condition */
              mode = 2; /* continue to linearly search for an empty slot */
              i0 = i; /* keep current position on record */
            }
            do { /* continue to linearly search for an available slot */
              i = LIBXS_MOD2(i + 1, registry_capacity);
              if (NULL == registry[i].ptr_const) break;
            } while (i != i0);
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* do not use break if inside of locked region */
            }
            flux_entry.ptr = NULL; /* no result */
          }
        }
        else /* JIT-code generation not available */
        { /* leave the dispatch loop */
          flux_entry.ptr = NULL;
          diff = 0;
        }
      }
#endif
    } while (0 != diff);
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    if (NULL != flux_entry.ptr_const) { /* keep code version on record (cache) */
      LIBXS_ASSERT(0 == diff);
      if (ninit == cache->entry.id) { /* maintain cache */
        if (cache->entry.size < registry_cache_size) { /* grow */
          INTERNAL_FIND_CODE_CACHE_GROW(cache_index, cache->entry.size);
          LIBXS_ASSERT(cache->entry.size <= registry_cache_size);
        }
        else { /* evict */
          LIBXS_ASSERT(cache->entry.hit < cache->entry.size);
          INTERNAL_FIND_CODE_CACHE_EVICT(cache_index, cache->entry.size, cache->entry.hit);
        }
      }
      else if (0 != registry_cache_size) { /* reset cache */
        /* INTERNAL_FIND_CODE_CACHE_GROW doubles size (and would expose invalid entries) */
        memset(cache->entry.keys, 0, LIBXS_CACHE_MAXSIZE * sizeof(*cache->entry.keys));
        cache->entry.id = ninit;
        cache->entry.size = 1;
        cache_index = 0;
      }
      LIBXS_MEMCPY127(cache->entry.keys + cache_index, desc, 0 == is_big_desc ? LIBXS_DIFF_SIZE : size);
      cache->entry.code[cache_index] = flux_entry;
      cache->entry.hit = cache_index;
    }
# if !defined(NDEBUG)
    else {
      memset(cache, 0, sizeof(*cache));
    }
# endif
#endif
  }
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
  return flux_entry;
}


#if 0
LIBXS_API_INTERN const libxs_kernel_xinfo* libxs_get_kernel_xinfo(libxs_registry_t* registry_state,
  libxs_code_pointer_t code, const libxs_descriptor_t** desc, size_t* code_size)
{
  libxs_kernel_xinfo* result = NULL;
  void *const result_address = &result;
  int flags = LIBXS_MALLOC_FLAG_X;
  if (NULL != code.ptr_const && EXIT_SUCCESS == libxs_get_malloc_xinfo(
    code.ptr_const, code_size, &flags, (void**)result_address) && NULL != result)
  {
    if (NULL != desc) {
      if (NULL != registry_state && NULL != registry_state->registry && NULL != registry_state->keys
        && result->registered < registry_state->capacity
#if defined(LIBXS_HASH_COLLISION)
        && code.uval == (~LIBXS_HASH_COLLISION & registry_state->registry[result->registered].uval)
#else
        && code.ptr_const == registry_state->registry[result->registered].ptr_const
#endif
        && LIBXS_KERNEL_UNREGISTERED > LIBXS_DESCRIPTOR_KIND(registry_state->keys[result->registered].entry.kind))
      {
        *desc = &registry_state->keys[result->registered].entry;
      }
      else *desc = NULL;
    }
  }
  else {
    LIBXS_ASSERT(NULL == result);
    if (NULL != code_size) *code_size = 0;
    if (NULL != desc) *desc = NULL;
  }
  return result;
}
#endif


#if 0
LIBXS_API int libxs_get_kernel_info(const void* kernel, libxs_kernel_info* info)
{
  int result;
  const libxs_kernel_xinfo* xinfo;
  libxs_kernel_info result_info /*= { 0 }*/;
  const libxs_descriptor_t* desc;
  libxs_code_pointer_t code = { 0 };
  code.ptr_const = kernel;
  LIBXS_MEMZERO127(&result_info);
  xinfo = libxs_get_kernel_xinfo(code, &desc, &result_info.code_size);
  result_info.is_reference_kernel = xinfo->is_reference_kernel;
  if (NULL != xinfo) {
    if (NULL != desc) {
      const libxs_kernel_kind kind = (libxs_kernel_kind)LIBXS_DESCRIPTOR_KIND(desc->kind);
      result_info.kind = kind;
      if (LIBXS_KERNEL_KIND_USER == kind) {
        result_info.code_size = 0; /* invalid */
      }
    }
    else {
      result_info.kind = LIBXS_KERNEL_UNREGISTERED;
    }
    result_info.nflops = xinfo->nflops;
    LIBXS_ASSIGN127(info, &result_info);
    result = EXIT_SUCCESS;
  }
  else {
    LIBXS_ASSERT(NULL == desc);
    if (NULL != info) {
      LIBXS_ASSIGN127(info, &result_info);
      result = EXIT_FAILURE;
    }
    else {
      result = EXIT_SUCCESS;
    }
  }
  return result;
}
#endif


LIBXS_API int libxs_registry_info(libxs_registry_t* registry, libxs_registry_info_t* info)
{
  int result = EXIT_SUCCESS;
  /*LIBXS_INIT*/ /* verbosity */
  if (0 != info && NULL != registry && NULL != registry->registry) {
    size_t i;
    LIBXS_MEMZERO127(info); /* info->nstatic = 0; info->size = 0; */
    info->nbytes = registry->capacity * (sizeof(libxs_code_pointer_t) + sizeof(libxs_descriptor_t));
    info->capacity = registry->capacity;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    info->ncache = registry->cache_size;
#else
    info->ncache = 0;
#endif
    for (i = 0; i < registry->capacity; ++i) {
      libxs_code_pointer_t code = registry->registry[i];
      if (0 != code.ptr_const && EXIT_SUCCESS == result) {
        if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
          size_t buffer_size = 0;
          void* buffer = 0;
#if defined(LIBXS_HASH_COLLISION)
          code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
#if 0
          result = libxs_get_malloc_xinfo(code.ptr_const, &buffer_size, NULL/*flags*/, &buffer);
#endif
          if (EXIT_SUCCESS == result) {
            info->nbytes += buffer_size + (((const char*)code.ptr_const) - (char*)buffer);
          }
        }
        else {
          ++info->nstatic;
        }
        ++info->size;
      }
    }
    registry->size = (unsigned int)info->size;
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE void* internal_get_registry_entry(libxs_registry_t* registry,
  int i, int kind, const void** key)
{
  void* result = NULL;
  LIBXS_ASSERT(NULL != registry && NULL != registry->registry);
  for (; NULL != registry && NULL != registry->registry && i < (int)registry->capacity; ++i) {
#if 0
    const libxs_code_pointer_t regentry = registry->registry[i];
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(regentry.ptr_const,
      NULL/*code_size*/, NULL/*flags*/, &result) && NULL != result)
    {
      const libxs_kernel_xinfo info = *(const libxs_kernel_xinfo*)result;
      const libxs_descriptor_t *const desc = &registry->keys[info.registered].entry;
      if (LIBXS_DESCRIPTOR_KIND(desc->kind) == (int)kind) {
        if (NULL != key) {
#if defined(LIBXS_REGUSER_ALIGN)
          if (LIBXS_KERNEL_KIND_USER == kind) {
            const size_t offset = LIBXS_UP2(desc->user.desc - desc->data, 4 < desc->user.size ? 8 : 4);
            *key = desc->data + offset;
          }
          else
#endif
          *key = desc->user.desc;
        }
        result = regentry.ptr;
        break;
      }
    }
#endif
  }
  return result;
}


LIBXS_API void* libxs_registry_begin(libxs_registry_t* registry, const void** key)
{
  void* result = NULL;
  if (NULL != registry && NULL != registry->registry) {
    /*result = internal_get_registry_entry(registry, 0, key);*/
  }
  return result;
}


LIBXS_API void* libxs_registry_next(libxs_registry_t* registry, const void* regentry, const void** key)
{
  void* result = NULL;
#if 0
  const libxs_descriptor_t* desc;
  libxs_code_pointer_t entry = { 0 };
  entry.ptr_const = regentry;
  if (NULL != libxs_get_kernel_xinfo(entry, &desc, NULL/*code_size*/)
    /* given regentry is indeed a registered kernel */
    && NULL != desc)
  {
    result = internal_get_registry_entry(
      (int)(desc - &registry->keys->entry + 1),
      (libxs_kernel_kind)LIBXS_DESCRIPTOR_KIND(desc->kind), key);
  }
#endif
  return result;
}


LIBXS_API void* libxs_registry_set(libxs_registry_t* registry, const void* key, size_t key_size,
  size_t value_size, const void* value_init)
{
  void* result = NULL;
#if 0
  libxs_descriptor_t wrap /*= { 0 }*/;
#if defined(LIBXS_REGUSER_ALIGN)
  const size_t offset = LIBXS_UP2(wrap.user.desc - wrap.data, 4 < key_size ? 8 : 4);
#else
  const size_t offset = wrap.user.desc - wrap.data;
#endif
  static int error_once = 0;
  /*LIBXS_INIT*/ /* verbosity */
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= (offset + key_size)) {
    void* dst;
#if defined(LIBXS_UNPACKED) || defined(LIBXS_REGUSER_ALIGN)
    LIBXS_MEMSET127(&wrap, 0, offset);
#endif
    LIBXS_MEMCPY127(wrap.data + offset, key, key_size);
    wrap.user.size = LIBXS_CAST_UCHAR(key_size);
    wrap.kind = (libxs_descriptor_kind_t)(LIBXS_DESCRIPTOR_SIGSIZE >= (offset + key_size)
      ? ((libxs_descriptor_kind_t)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    dst = internal_find_code(&wrap, offset + key_size - sizeof(libxs_descriptor_kind_t), value_size).ptr;
    if (NULL != dst) {
      size_t size;
      if (EXIT_SUCCESS == libxs_get_malloc_xinfo(dst, &size, NULL/*flags*/, NULL/*extra*/)
        && value_size <= size)
      {
        if (NULL != value_init) memcpy(dst, value_init, value_size);
        result = dst;
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
          /*&& 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)*/)
        {
          fprintf(stderr, "LIBXS ERROR: value too large for previously registered key!\n");
        }
        result = NULL;
      }
    }
    else result = NULL;
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      if (LIBXS_DESCRIPTOR_MAXSIZE >= key_size) {
        fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_registry_set specified!\n");
      }
      else {
        fprintf(stderr, "LIBXS ERROR: libxs_registry_set has maximum key-size of %i Byte!\n",
          LIBXS_DESCRIPTOR_MAXSIZE);
      }
    }
    result = NULL;
  }
#endif
  return result;
}


LIBXS_API void* libxs_registry_get(libxs_registry_t* registry, const void* key, size_t key_size)
{
  void* result = NULL;
#if 0
  libxs_descriptor_t wrap /*= { 0 }*/;
#if defined(LIBXS_REGUSER_ALIGN)
  const size_t offset = LIBXS_UP2(wrap.user.desc - wrap.data, 4 < key_size ? 8 : 4);
#else
  const size_t offset = wrap.user.desc - wrap.data;
#endif
  /*LIBXS_INIT*/ /* verbosity */
#if !defined(NDEBUG)
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= (offset + key_size))
#endif
  {
#if defined(LIBXS_UNPACKED) || defined(LIBXS_REGUSER_ALIGN)
    LIBXS_MEMSET127(&wrap, 0, offset);
#endif
    LIBXS_MEMCPY127(wrap.data + offset, key, key_size);
    wrap.user.size = LIBXS_CAST_UCHAR(key_size);
    wrap.kind = (libxs_descriptor_kind_t)(LIBXS_DESCRIPTOR_SIGSIZE >= (offset + key_size)
      ? ((libxs_descriptor_kind_t)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    result = internal_find_code(&wrap, offset + key_size - sizeof(libxs_descriptor_kind_t), 0/*user_size*/).ptr;
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_registry_get specified!\n");
    }
    result = NULL;
  }
#endif
#endif
  return result;
}


LIBXS_API void libxs_registry_free(libxs_registry_t* registry, const void* key, size_t key_size)
{
  /*libxs_release_kernel(libxs_registry_get(key, key_size));*/
}


LIBXS_API void libxs_release_kernel(const void* kernel)
{
#if 0
  if (NULL != kernel) {
    static int error_once = 0;
    libxs_kernel_xinfo* extra = NULL;
    void *const extra_address = &extra;
    /*LIBXS_INIT*/
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(
      kernel, NULL/*size*/, NULL/*flags*/, (void**)extra_address) && NULL != extra)
    {
      const unsigned int regindex = extra->registered;
      libxs_registry_t *const registry_state = internal_registry_state;
      libxs_code_pointer_t *const registry = (NULL != registry_state ? registry_state->registry : NULL);
      internal_regkey_type *const registry_keys = (NULL != registry_state ? registry_state->keys : NULL);
      if (NULL == registry_state || regindex >= registry_state->capacity) {
        libxs_xfree(kernel, 0/*no check*/);
      }
      else { /* attempt to unregister kernel */
        libxs_kernel_info info /*= { 0 }*/;
#if !defined(LIBXS_ENABLE_DEREG)
        if (EXIT_SUCCESS == libxs_get_kernel_info(kernel, &info)
          && LIBXS_KERNEL_KIND_USER == info.kind)
#endif
        {
          LIBXS_ASSERT(LIBXS_KERNEL_UNREGISTERED > info.kind);
          /* coverity[check_return] */
          LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST); /* invalidate code cache (TLS) */
          if (NULL != registry && NULL != registry[regindex].ptr_const && 0 < registry_state->size) {
            --registry_state->size;
          }
          registry[regindex].ptr = NULL;
#if !defined(NDEBUG)
          if (NULL != registry_keys) memset(registry_keys + regindex, 0, sizeof(*registry_keys));
#endif
          libxs_xfree(kernel, 0/*no check*/);
        }
#if !defined(LIBXS_ENABLE_DEREG)
        else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS WARNING: attempt to unregister JIT-kernel!\n");
        }
#endif
      }
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: failed to release kernel!\n");
    }
  }
#endif
}
