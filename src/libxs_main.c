/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_trace.h"
#include "libxs_xcopy.h"
#include "libxs_gemm.h"
#include "libxs_hash.h"
#include "libxs_diff.h"
#include "libxs_main.h"
#if defined(LIBXS_PERF)
# include "libxs_perf.h"
#endif
#include "generator_common.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/types.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(__APPLE__)
# include <libkern/OSCacheControl.h>
# include <pthread.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_CODE_MAXSIZE)
# define LIBXS_CODE_MAXSIZE 131072
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
#if !defined(LIBXS_MALLOC_HOOK_ALIGN) && 1
# define LIBXS_MALLOC_HOOK_ALIGN
#endif
#if !defined(LIBXS_MALLOC_HOOK_INIT) && 0
# define LIBXS_MALLOC_HOOK_INIT
#endif
#if !defined(LIBXS_ENABLE_DEREG) && 0
# define LIBXS_ENABLE_DEREG
#endif
#if !defined(LIBXS_REGUSER_HASH) && 1
# define LIBXS_REGUSER_HASH
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
#if !defined(LIBXS_AUTOPIN) && 1
# define LIBXS_AUTOPIN
#endif
#if !defined(INTERNAL_DELIMS)
# define INTERNAL_DELIMS ";,:"
#endif

#if !defined(_WIN32) && !defined(__CYGWIN__)
LIBXS_EXTERN int posix_memalign(void**, size_t, size_t) LIBXS_THROW;
#endif
#if defined(LIBXS_AUTOPIN) && !defined(_WIN32)
LIBXS_EXTERN int putenv(char*) LIBXS_THROW;
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 1 /* beneficial when registry approaches capacity (collisions) */
# define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXS_PREFETCH) /* auto-prefetch (frontend) */ || (defined(_WIN32) || defined(__CYGWIN__))
# define INTERNAL_PREFETCH LIBXS_GEMM_PREFETCH_NONE
#else
# define INTERNAL_PREFETCH ((libxs_gemm_prefetch_type)LIBXS_PREFETCH)
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
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_reglocktype {
  char pad[LIBXS_CACHELINE];
  LIBXS_LOCK_TYPE(LIBXS_REGLOCK) state;
} internal_reglocktype;
#   else
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_reglocktype {
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
# define LIBXS_CACHE_STRIDE LIBXS_MAX(sizeof(libxs_descriptor), LIBXS_DESCRIPTOR_MAXSIZE)
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
      LIBXS_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; break; \
    }
# else
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
      LIBXS_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
      const unsigned int LOCKINDEX = (0 != internal_reglock_count ? LIBXS_MOD2(INDEX, internal_reglock_count) : 0); \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state); }
# else /* RW-lock */
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, internal_reglock_ptr)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr); }
# endif
#endif


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_cache_entry_type {
  libxs_descriptor keys[LIBXS_CACHE_MAXSIZE];
  libxs_code_pointer code[LIBXS_CACHE_MAXSIZE];
  unsigned int id; /* to invalidate */
  unsigned char size, hit;
} internal_cache_entry_type;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_cache_type {
# if defined(LIBXS_CACHE_PAD)
  char pad[LIBXS_UP2(sizeof(internal_cache_entry_type),LIBXS_CACHELINE)];
# endif
  internal_cache_entry_type entry;
} internal_cache_type;

# if defined(LIBXS_NTHREADS_USE)
LIBXS_APIVAR_DEFINE(internal_cache_type* internal_cache_buffer);
# endif
LIBXS_APIVAR_DEFINE(int internal_cache_size);
#endif /*defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))*/
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_regkey_type {
#if defined(LIBXS_REGKEY_PAD)
  char pad[LIBXS_UP2(sizeof(libxs_descriptor), LIBXS_CACHELINE)];
#endif
  libxs_descriptor entry;
} internal_regkey_type;

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXS_APIVAR_DEFINE(int internal_reglock_count);
LIBXS_APIVAR_DEFINE(size_t internal_registry_nbytes);
LIBXS_APIVAR_DEFINE(unsigned int internal_registry_nleaks);
LIBXS_APIVAR_DEFINE(internal_regkey_type* internal_registry_keys);
LIBXS_APIVAR_DEFINE(libxs_code_pointer* internal_registry);
LIBXS_APIVAR_DEFINE(internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/]);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_sml);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_med);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_mnk);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_gemv);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_mcopy);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_meltw);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_tcopy);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_trsm);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_trmm);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_user);
LIBXS_APIVAR_DEFINE(int internal_gemm_auto_prefetch_locked);
LIBXS_APIVAR_DEFINE(const char* internal_build_state);
/** Time stamp (startup time of library). */
LIBXS_APIVAR_DEFINE(libxs_timer_tickint internal_timer_start);
LIBXS_APIVAR_DEFINE(libxs_cpuid_x86_info internal_cpuid_info);

#if defined(_WIN32)
# define INTERNAL_SINGLETON_HANDLE HANDLE
# define INTERNAL_SINGLETON(HANDLE) (NULL != (HANDLE))
#else
# define INTERNAL_SINGLETON_HANDLE int
# define INTERNAL_SINGLETON(HANDLE) (0 <= (HANDLE) && 0 != *internal_singleton_fname)
LIBXS_APIVAR_DEFINE(char internal_singleton_fname[64]);
#endif
LIBXS_APIVAR_DEFINE(INTERNAL_SINGLETON_HANDLE internal_singleton_handle);

/* definition of corresponding variables */
LIBXS_APIVAR_PRIVATE_DEF(libxs_malloc_function libxs_default_malloc_fn);
LIBXS_APIVAR_PRIVATE_DEF(libxs_malloc_function libxs_scratch_malloc_fn);
LIBXS_APIVAR_PRIVATE_DEF(libxs_free_function libxs_default_free_fn);
LIBXS_APIVAR_PRIVATE_DEF(libxs_free_function libxs_scratch_free_fn);
LIBXS_APIVAR_PRIVATE_DEF(const void* libxs_default_allocator_context);
LIBXS_APIVAR_PRIVATE_DEF(const void* libxs_scratch_allocator_context);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_scratch_pools);
LIBXS_APIVAR_PRIVATE_DEF(double libxs_scratch_scale);
LIBXS_APIVAR_PRIVATE_DEF(double libxs_timer_scale);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_statistic_num_spmdm);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_thread_count);
/* definition of corresponding variables */
LIBXS_APIVAR_PUBLIC_DEF(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
LIBXS_APIVAR_PUBLIC_DEF(int libxs_nosync);

#if (0 != LIBXS_SYNC)
LIBXS_APIVAR_PRIVATE_DEF(LIBXS_TLS_TYPE libxs_tlskey);
#endif

LIBXS_API_INTERN void* libxs_memalign_internal(size_t alignment, size_t size)
{
  void* result;
#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
  result = __libc_memalign(alignment, size);
#elif defined(LIBXS_BUILD) && ( /*C11*/ \
  defined(__STDC_VERSION__) && (201112L <= __STDC_VERSION__))
  result = aligned_alloc(alignment, size);
#elif (defined(_WIN32) || defined(__CYGWIN__))
  LIBXS_UNUSED(alignment);
  result = malloc(size);
#else
  if (0 != posix_memalign(&result, alignment, size)) result = NULL;
#endif
  return result;
}


LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_memalign(size_t alignment, size_t size)
{
  void* result;
#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
  if (
# if defined(LIBXS_MALLOC_HOOK_INIT)
    1 < libxs_ninit &&
# endif
    NULL != libxs_malloc_fn.memalign.ptr)
  {
    result = libxs_malloc_fn.memalign.ptr(alignment, size);
  }
  else
#endif
  result = libxs_memalign_internal(alignment, size);
  return result;
}


LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_malloc(size_t size)
{
  void* result;
#if defined(LIBXS_MALLOC_HOOK_ALIGN)
  const size_t alignment = libxs_alignment(size, 0/*auto*/);
  result = __real_memalign(alignment, size);
#else
# if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
  if (
#   if defined(LIBXS_MALLOC_HOOK_INIT)
    1 < libxs_ninit &&
#   endif
    NULL != libxs_malloc_fn.malloc.ptr)
  {
    LIBXS_ASSERT(malloc != libxs_malloc_fn.malloc.ptr);
    result = libxs_malloc_fn.malloc.ptr(size);
  }
  else
# endif
# if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
  result = __libc_malloc(size);
# else
  result = malloc(size);
# endif
#endif
  return result;
}


#if defined(LIBXS_MALLOC_HOOK_CALLOC)
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_calloc(size_t num, size_t size)
{
  void* result;
#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
  if (
# if defined(LIBXS_MALLOC_HOOK_INIT)
    1 < libxs_ninit &&
# endif
    NULL != libxs_malloc_fn.calloc.ptr)
  {
    LIBXS_ASSERT(calloc != libxs_malloc_fn.calloc.ptr);
    result = libxs_malloc_fn.calloc.ptr(num, size);
  }
  else
#endif
#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
  result = __libc_calloc(num, size);
#else
  result = calloc(num, size);
#endif
  return result;
}
#endif


#if defined(LIBXS_MALLOC_HOOK_REALLOC)
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_realloc(void* ptr, size_t size)
{
  void* result;
#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
  if (
# if defined(LIBXS_MALLOC_HOOK_INIT)
    1 < libxs_ninit &&
# endif
    NULL != libxs_malloc_fn.realloc.ptr)
  {
    LIBXS_ASSERT(realloc != libxs_malloc_fn.realloc.ptr);
    result = libxs_malloc_fn.realloc.ptr(ptr, size);
  }
  else
#endif
#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
  result = __libc_realloc(ptr, size);
#else
  result = realloc(ptr, size);
#endif
  return result;
}
#endif


LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void __real_free(void* ptr)
{
  if (NULL != ptr) {
#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
    if (
# if defined(LIBXS_MALLOC_HOOK_INIT)
      1 < libxs_ninit &&
# endif
      NULL != libxs_malloc_fn.free.ptr)
    {
      LIBXS_ASSERT(free != libxs_malloc_fn.free.ptr);
      libxs_malloc_fn.free.ptr(ptr);
    }
    else
#endif
#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
    __libc_free(ptr);
#else
    free(ptr);
#endif
  }
}


LIBXS_API_INLINE void internal_update_mmstatistic(const libxs_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol, unsigned int njit, unsigned int nsta)
{
  LIBXS_ASSERT(NULL != desc);
  if (1 < desc->m && 1 < desc->n) { /* only record matrix-matrix multiplication */
    const unsigned long long kernel_size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
    const int idx = (LIBXS_GEMM_PRECISION_F64 == LIBXS_GETENUM_OUT(desc->datatype) ? 0 : 1);
    int bucket;
    if (LIBXS_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
      bucket = 0;
    }
    else if (LIBXS_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
      bucket = 1;
    }
    else if (LIBXS_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
      bucket = 2;
    }
    else { /*huge*/
      bucket = 3;
    }
    if (0 != ncol) ncol/*dummy assignment*/ = LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ncol, ncol, LIBXS_ATOMIC_RELAXED);
    if (0 != ntry) ntry/*dummy assignment*/ = LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ntry, ntry, LIBXS_ATOMIC_RELAXED);
    /* the following counters are not manipulated concurrently (no need for atomic increment) */
    if (0 != njit) internal_statistic[idx][bucket].njit += njit;
    if (0 != nsta) internal_statistic[idx][bucket].nsta += nsta;
  }
}


LIBXS_API_INLINE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  LIBXS_ASSERT(NULL != unit);
  *unit = default_unit;
  if ((1000000) <= n) {
    number = (n + 500000) / 1000000;
    *unit = 'm';
  }
  else if (9999 < n) {
    number = (n + 500) / 1000;
    *unit = 'k';
  }
  return number;
}


LIBXS_API_INLINE unsigned int internal_print_statistic(FILE* ostream,
  const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  const internal_statistic_type statistic_sml = internal_statistic[precision][0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic[precision][1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic[precision][2/*BIG*/];
  const internal_statistic_type statistic_xxx = internal_statistic[precision][3/*XXX*/];
  int printed = 0;
  LIBXS_ASSERT(NULL != ostream && (0 <= precision && precision < 2));
  if (/* omit to print anything if it is superfluous */
    0 != statistic_sml.ntry || 0 != statistic_sml.njit || 0 != statistic_sml.nsta || 0 != statistic_sml.ncol ||
    0 != statistic_med.ntry || 0 != statistic_med.njit || 0 != statistic_med.nsta || 0 != statistic_med.ncol ||
    0 != statistic_big.ntry || 0 != statistic_big.njit || 0 != statistic_big.nsta || 0 != statistic_big.ncol ||
    0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol)
  {
    char title[256], range[256], unit[4];
    unsigned int counter[4];
    {
      unsigned int n;
      if (NULL != target_arch && 0 != *target_arch) {
        assert(strlen(target_arch) < sizeof(title)); /* !LIBXS_ASSERT */
        for (n = 0; 0 != target_arch[n] /*avoid code-gen. issue with some clang versions: && n < sizeof(title)*/; ++n) {
          const char c = target_arch[n];
          title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c); /* toupper */
        }
        LIBXS_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      }
      else {
        LIBXS_SNPRINTF(title, sizeof(title), "%s", 0 == precision ? "DP" : "SP");
      }
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-8s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY", "JIT", "STA", "COL");
    LIBXS_SNPRINTF(range, sizeof(range), "%u..%u", 0u, internal_statistic_sml);
    counter[0] = internal_print_number(statistic_sml.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_sml.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_sml.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_sml.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXS_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_sml + 1u, internal_statistic_med);
    counter[0] = internal_print_number(statistic_med.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_med.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_med.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_med.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXS_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_med + 1u, internal_statistic_mnk);
    counter[0] = internal_print_number(statistic_big.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_big.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_big.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_big.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    if (0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol) {
      LIBXS_SNPRINTF(range, sizeof(range), "> %u", internal_statistic_mnk);
      counter[0] = internal_print_number(statistic_xxx.ntry, ' ', unit + 0);
      counter[1] = internal_print_number(statistic_xxx.njit, ' ', unit + 1);
      counter[2] = internal_print_number(statistic_xxx.nsta, ' ', unit + 2);
      counter[3] = internal_print_number(statistic_xxx.ncol, ' ', unit + 3);
      fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
        counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    }
    printed = 1;
  }
  return printed;
}


#if !(defined(_WIN32) || defined(__CYGWIN__))
LIBXS_API_INLINE unsigned int internal_statistic_ntry(int precision)
{
  return internal_statistic[precision][0/*SML*/].ntry + internal_statistic[precision][1/*MED*/].ntry
       + internal_statistic[precision][2/*BIG*/].ntry + internal_statistic[precision][3/*XXX*/].ntry;
}
#endif


#if !defined(_WIN32)
LIBXS_API_INLINE void internal_register_static_code(
  libxs_gemm_precision precision, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  libxs_xmmfunction xgemm, libxs_code_pointer* registry)
{
  const libxs_blasint lda = m, ldb = k, ldc = m;
  /*const*/ int precondition = LIBXS_GEMM_NO_BYPASS_DIMS(m, n, k) && LIBXS_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc);
  if (precondition) {
    const size_t size = (LIBXS_HASH_SIZE) - sizeof(libxs_descriptor_kind);
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_dinit(&blob, precision,
      m, n, k, lda, ldb, ldc, LIBXS_ALPHA, LIBXS_BETA, LIBXS_FLAGS, INTERNAL_PREFETCH);
    unsigned int i = LIBXS_MOD2(
      libxs_crc32(LIBXS_HASH_SEED, desc, LIBXS_MIN(sizeof(libxs_gemm_descriptor), size)),
      LIBXS_CAPACITY_REGISTRY);
    libxs_code_pointer* dst_entry = registry + i;
#if !defined(NDEBUG)
    libxs_code_pointer code; code.xgemm = xgemm;
    LIBXS_ASSERT(NULL != code.ptr_const && NULL != registry);
    LIBXS_ASSERT(0 == (LIBXS_CODE_STATIC & code.uval));
#endif
    if (NULL != dst_entry->ptr_const) { /* collision */
      const unsigned int i0 = i;
      do { /* continue to linearly search for an available slot */
        i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
        if (NULL == registry[i].ptr_const) break;
      } while (i != i0);
#if defined(LIBXS_HASH_COLLISION) /* mark entry as a collision */
      dst_entry->uval |= LIBXS_HASH_COLLISION;
#endif
      dst_entry = registry + i; /* update destination */
      internal_update_mmstatistic(desc, 0, 1/*collision*/, 0, 0);
      /* out of capacity (no registry slot available) */
      LIBXS_ASSERT(NULL == dst_entry->ptr_const || i == i0);
    }
    if (NULL == dst_entry->ptr_const) { /* registry not exhausted */
      internal_registry_keys[i].entry.kind = LIBXS_KERNEL_KIND_MATMUL;
      LIBXS_ASSIGN127(&internal_registry_keys[i].entry.gemm.desc, desc);
      dst_entry->xgemm = xgemm;
      /* mark current entry as static code (non-JIT) */
      dst_entry->uval |= LIBXS_CODE_STATIC;
    }
    internal_update_mmstatistic(desc, 1/*try*/, 0, 0, 0);
  }
}
#endif


LIBXS_API_INTERN void internal_release_scratch(void);
LIBXS_API_INTERN void internal_release_scratch(void)
{
  libxs_xrelease_scratch(NULL/*lock*/);
  /* release global services */
  libxs_memory_finalize();
  libxs_hash_finalize();
  libxs_malloc_finalize();
}


/* Caution: cannot be used multiple times in a single expression! */
LIBXS_API_INTERN size_t libxs_format_value(char buffer[32], int buffer_size, size_t nbytes, const char scale[], const char* unit, int base)
{
  const int len = (NULL != scale ? ((int)strlen(scale)) : 0);
  const int m = LIBXS_INTRINSICS_BITSCANBWD64(nbytes) / base, n = LIBXS_MIN(m, len);
  int i;
  buffer[0] = 0; /* clear */
  LIBXS_ASSERT(NULL != unit && 0 <= base);
  for (i = 0; i < n; ++i) nbytes >>= base;
  LIBXS_SNPRINTF(buffer, buffer_size, "%i %c%s",
    (int)nbytes, 0 < n ? scale[n-1] : *unit, 0 < n ? unit : "");
  return nbytes;
}


LIBXS_API_INTERN LIBXS_ATTRIBUTE_NO_TRACE void internal_dump(FILE* ostream, int urgent);
LIBXS_API_INTERN void internal_dump(FILE* ostream, int urgent)
{
  char *const env_dump_build = getenv("LIBXS_DUMP_BUILD");
  char *const env_dump_files = (NULL != getenv("LIBXS_DUMP_FILES")
    ? getenv("LIBXS_DUMP_FILES")
    : getenv("LIBXS_DUMP_FILE"));
  LIBXS_ASSERT_MSG(INTERNAL_SINGLETON(internal_singleton_handle), "Invalid handle");
  /* determine whether this instance is unique or not */
  if (NULL != env_dump_files && 0 != *env_dump_files && 0 == urgent) { /* dump per-node info */
    const char* filename = strtok(env_dump_files, INTERNAL_DELIMS);
    for (; NULL != filename; filename = strtok(NULL, INTERNAL_DELIMS)) {
      FILE* const file = fopen(filename, "r");
      if (NULL != file) {
        int c = fgetc(file);
        fprintf(ostream, "\n\nLIBXS_DUMP_FILE: %s\n", filename);
        /* coverity[tainted_data] */
        while (EOF != c) {
          fputc(c, stdout);
          c = fgetc(file);
        }
        fputc('\n', stdout);
        fclose(file);
      }
    }
  }
  if  (NULL != internal_build_state /* dump build state */
    && NULL != env_dump_build && 0 != *env_dump_build)
  {
    const int dump_build = atoi(env_dump_build);
    if (0 == urgent ? (0 < dump_build) : (0 > dump_build)) {
      fprintf(ostream, "\n\nBUILD_DATE=%i\n", LIBXS_CONFIG_BUILD_DATE);
      fprintf(ostream, "%s\n", internal_build_state);
    }
  }
}


LIBXS_API_INTERN void internal_finalize(void);
LIBXS_API_INTERN void internal_finalize(void)
{
  libxs_finalize();
  LIBXS_STDIO_ACQUIRE(); /* synchronize I/O */
  if (0 != libxs_verbosity) { /* print statistic on termination */
    const char *const env_target_hidden = getenv("LIBXS_TARGET_HIDDEN");
    const char *const target_arch = (NULL == env_target_hidden || 0 == atoi(env_target_hidden))
      ? libxs_cpuid_name(libxs_target_archid) : NULL/*hidden*/;
    fprintf(stderr, "\nLIBXS_VERSION: %s%s%s (%i)", LIBXS_BRANCH,
      0 != *(LIBXS_BRANCH) ? "-" : "", 0 != *(LIBXS_VERSION) ? (LIBXS_VERSION) : "unconfigured",
      LIBXS_VERSION4(LIBXS_VERSION_MAJOR, LIBXS_VERSION_MINOR, LIBXS_VERSION_UPDATE, LIBXS_VERSION_PATCH));
    if (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) {
      unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      const int high_verbosity = (LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity);
      char number_format_buffer[32];
      libxs_scratch_info scratch_info;
      libxs_cpuid_x86_info info;
      libxs_cpuid_x86(&info);
      if ((LIBXS_VERBOSITY_HIGH < libxs_verbosity || 0 > libxs_verbosity) &&
        0 == internal_cpuid_info.has_context && 0 != info.has_context)
      {
        fprintf(stderr, "\nLIBXS: CPU features have been promoted.");
      }
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak && NULL != target_arch) {
        fprintf(stderr, "\nLIBXS_TARGET: %s\n", target_arch);
      }
      if (0 != libxs_format_value(number_format_buffer, sizeof(number_format_buffer),
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
        sizeof(internal_cache_type) * (LIBXS_NTHREADS_MAX) +
#endif
        (sizeof(internal_regkey_type) + sizeof(libxs_code_pointer)) * (LIBXS_CAPACITY_REGISTRY),
        "KM", "B", 10))
      {
        fprintf(stderr, "Registry and code: %s", number_format_buffer);
        if (0 != libxs_format_value(number_format_buffer, sizeof(number_format_buffer), internal_registry_nbytes, "KM", "B", 10)) {
          fprintf(stderr, " + %s", number_format_buffer);
        }
        if (0 != high_verbosity) {
          unsigned int ngemms = 0;
          int i; for (i = 0; i < 4; ++i) {
            ngemms += internal_statistic[0/*DP*/][i].nsta + internal_statistic[1/*SP*/][i].nsta;
            ngemms += internal_statistic[0/*DP*/][i].njit + internal_statistic[1/*SP*/][i].njit;
          }
          if (0 != ngemms || 0 != internal_statistic_num_gemv
            || 0 != internal_statistic_num_mcopy || 0 != internal_statistic_num_tcopy
            || 0 != libxs_statistic_num_spmdm
            || 0 != internal_statistic_num_user
            || 0 != internal_registry_nleaks)
          {
            const char sep[] = " ", *s = "";
            fprintf(stderr, " (");
            if (0 != ngemms) { fprintf(stderr, "gemm=%u", ngemms); s = sep; }
            if (0 != internal_statistic_num_gemv) { fprintf(stderr, "%sgemv=%u", s, internal_statistic_num_gemv); s = sep; }
            if (0 != internal_statistic_num_mcopy) { fprintf(stderr, "%smcopy=%u", s, internal_statistic_num_mcopy); s = sep; }
            if (0 != internal_statistic_num_meltw) { fprintf(stderr, "%smeltw=%u", s, internal_statistic_num_meltw); s = sep; }
            if (0 != internal_statistic_num_tcopy) { fprintf(stderr, "%stcopy=%u", s, internal_statistic_num_tcopy); s = sep; }
            if (0 != libxs_statistic_num_spmdm) { fprintf(stderr, "%sspmdm=%u", s, libxs_statistic_num_spmdm); s = sep; }
            if (0 != internal_statistic_num_user) { fprintf(stderr, "%suser=%u", s, internal_statistic_num_user); s = sep; }
            if (0 != internal_registry_nleaks) { fprintf(stderr, "%snleaks=%u", s, internal_registry_nleaks); s = sep; }
            fprintf(stderr, ")");
          }
        }
        fprintf(stderr, "\n");
      }
      if (EXIT_SUCCESS == libxs_get_scratch_info(&scratch_info)) {
        if (0 != scratch_info.size &&
          0 != libxs_format_value(number_format_buffer, sizeof(number_format_buffer), scratch_info.size, "KM", "B", 10))
        {
          fprintf(stderr, "Scratch: %s", number_format_buffer);
          if (0 != high_verbosity) {
            fprintf(stderr, " (mallocs=%lu, pools=%u)\n", (unsigned long int)scratch_info.nmallocs, scratch_info.npools);
          }
          else {
            fprintf(stderr, "\n");
          }
        }
        if (0 != scratch_info.internal && 0 != high_verbosity &&
          libxs_format_value(number_format_buffer, sizeof(number_format_buffer), scratch_info.internal, "KM", "B", 10))
        {
          fprintf(stderr, "Private: %s\n", number_format_buffer);
        }
      }
      if (LIBXS_VERBOSITY_HIGH < libxs_verbosity || 0 > libxs_verbosity) {
        fprintf(stderr, "Uptime: %f s", libxs_timer_duration(internal_timer_start, libxs_timer_tick()));
        if (1 < libxs_thread_count && INT_MAX == libxs_verbosity) {
          fprintf(stderr, " (nthreads=%u)", libxs_thread_count);
        }
        fprintf(stderr, "\n");
      }
    }
    else {
      fprintf(stderr, "\nLIBXS_TARGET: %s\n", target_arch);
    }
  }
  /* release scratch memory pool */
  if (EXIT_SUCCESS != atexit(internal_release_scratch) && 0 != libxs_verbosity) {
    fprintf(stderr, "LIBXS ERROR: failed to perform final cleanup!\n");
  }
  /* determine whether this instance is unique or not */
  if (INTERNAL_SINGLETON(internal_singleton_handle)) {
    internal_dump(stdout, 0/*urgent*/);
    /* cleanup singleton */
#if defined(_WIN32)
    ReleaseMutex(internal_singleton_handle);
    CloseHandle(internal_singleton_handle);
#else
    unlink(internal_singleton_fname);
    close(internal_singleton_handle);
#endif
  }
  LIBXS_STDIO_RELEASE(); /* synchronize I/O */
#if (0 != LIBXS_SYNC)
  { /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
    int i; for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_DESTROY(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_DESTROY(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
    LIBXS_LOCK_DESTROY(LIBXS_LOCK, &libxs_lock_global);
  }
#endif
}


#if defined(LIBXS_INTERCEPT_DYNAMIC)
LIBXS_API LIBXS_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* /*message*/, int /*len*/, int /*quiet*/);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* message, int len, int quiet)
{ /* STOP termination handler for GNU Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(const char*, int, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "_gfortran_stop_string");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len, quiet);
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}

LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core(const char* /*message*/, int /*len*/);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core(const char* message, int len)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(const char*, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "for_stop_core");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len);
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}

LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core_quiet(void);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core_quiet(void)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(void); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "for_stop_core_quiet");
    if (NULL != stop.dlsym) {
      stop.ptr();
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}
#endif


LIBXS_API_INTERN size_t internal_strlen(const char* /*cstr*/, size_t /*maxlen*/);
LIBXS_API_INTERN size_t internal_strlen(const char* cstr, size_t maxlen)
{
  size_t result = 0;
  if (NULL != cstr) {
    while (0 != cstr[result] && result < maxlen) ++result;
  }
  return result;
}


LIBXS_API_INTERN size_t internal_parse_nbytes(const char* /*nbytes*/, size_t /*ndefault*/);
LIBXS_API_INTERN size_t internal_parse_nbytes(const char* nbytes, size_t ndefault)
{
  size_t result = ndefault;
  if (NULL != nbytes && 0 != *nbytes) {
    size_t u = internal_strlen(nbytes, 32) - 1;
    const char unit[] = "kmgKMG", *const hit = strchr(unit, nbytes[u]);
    const long long int ibytes = atol(nbytes); /* take with increased type-width */
    result = (size_t)ibytes;
    if ((size_t)LIBXS_UNLIMITED != result) {
      u = (0 != hit ? ((hit - unit) % 3) : 3);
      if (u < 3) {
        result <<= (u + 1) * 10;
      }
    }
  }
  return result;
}


LIBXS_API_INTERN LIBXS_ATTRIBUTE_NO_TRACE void internal_init(void);
LIBXS_API_INTERN void internal_init(void)
{
  int i;
#if (0 != LIBXS_SYNC) /* setup the locks in a thread-safe fashion */
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
  LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
#endif
  if (NULL == internal_registry) { /* double-check after acquiring the lock(s) */
#if defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_AUTOPIN)
    /* clear error status (dummy condition: it does not matter if MPI_Init or MPI_Abort) */
    const char* const dlsymname = (NULL == dlerror() ? "MPI_Init" : "MPI_Abort");
    const void* const dlsymbol = dlsym(LIBXS_RTLD_NEXT, dlsymname);
    const void* const dlmpi = (NULL == dlerror() ? dlsymbol : NULL);
#endif
    const char* const env_verbose = getenv("LIBXS_VERBOSE");
    void* new_registry = NULL, * new_keys = NULL;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# if defined(LIBXS_NTHREADS_USE)
    void* new_cache = NULL;
# endif
    const char* const env_cache = getenv("LIBXS_CACHE");
    if (NULL != env_cache && 0 != *env_cache) {
      const int cache_size = atoi(env_cache), cache_size2 = LIBXS_UP2POT(cache_size);
      internal_cache_size = LIBXS_MIN(cache_size2, LIBXS_CACHE_MAXSIZE);
    }
    else {
      internal_cache_size = LIBXS_CACHE_MAXSIZE;
    }
#endif
    /* setup verbosity as early as possible since below code may rely on verbose output */
    if (NULL != env_verbose && 0 != *env_verbose) {
      libxs_verbosity = atoi(env_verbose);
    }
#if !defined(NDEBUG)
    else {
      libxs_verbosity = INT_MAX; /* quiet -> verbose */
    }
#endif
#if (0 == LIBXS_JIT)
    if (2 > libxs_ninit && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)) {
      fprintf(stderr, "LIBXS: JIT-code generation was disabled at compile-time.\n");
    }
#endif
#if defined(LIBXS_AUTOPIN)
# if defined(LIBXS_INTERCEPT_DYNAMIC)
    /* MPI: unwanted affinity can slow-down unrelated jobs (over-subscription), e.g., CP2K regtests */
    if (NULL == dlmpi)
# endif
    { /* setup some viable affinity if nothing else is present */
      const char *const gomp_cpu_affinity = getenv("GOMP_CPU_AFFINITY");
      const char *const kmp_affinity = getenv("KMP_AFFINITY");
      const char *const omp_proc_bind = getenv("OMP_PROC_BIND");
      if  ((NULL == gomp_cpu_affinity || 0 == *gomp_cpu_affinity)
        && (NULL == kmp_affinity || 0 == *kmp_affinity)
        && (NULL == omp_proc_bind || 0 == *omp_proc_bind))
      {
        static char affinity[] = "OMP_PROC_BIND=TRUE";
        LIBXS_EXPECT(EXIT_SUCCESS, LIBXS_PUTENV(affinity));
        if (LIBXS_VERBOSITY_HIGH < libxs_verbosity || 0 > libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS: prepared to pin threads.\n");
        }
      }
    }
# if defined(LIBXS_INTERCEPT_DYNAMIC) && 1
    else if (NULL == getenv("I_MPI_SHM_HEAP")) {
      static char shmheap[] = "I_MPI_SHM_HEAP=1";
      LIBXS_EXPECT(EXIT_SUCCESS, LIBXS_PUTENV(shmheap));
    }
# endif
#endif
#if !defined(_WIN32) && 0
    umask(S_IRUSR | S_IWUSR); /* setup default/secure file mask */
#endif
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    { const char *const env = getenv("LIBXS_SCRATCH_POOLS");
      if (NULL == env || 0 == *env) {
        libxs_scratch_pools = LIBXS_MALLOC_SCRATCH_MAX_NPOOLS;
      }
      else {
        libxs_scratch_pools = LIBXS_CLMP(atoi(env), 0, LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
        /*libxs_scratch_pools_locked = 1;*/
      }
      LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    }
    { const char *const env = getenv("LIBXS_SCRATCH_SCALE");
      if (NULL == env || 0 == *env) {
        libxs_scratch_scale = LIBXS_MALLOC_SCRATCH_SCALE;
      }
      else {
        libxs_scratch_scale = LIBXS_CLMP(atof(env), 1.0, 10.0);
        /*libxs_scratch_scale_locked = 1;*/
      }
      LIBXS_ASSERT(1 <= libxs_scratch_scale);
    }
    libxs_set_scratch_limit(internal_parse_nbytes(getenv("LIBXS_SCRATCH_LIMIT"), LIBXS_SCRATCH_DEFAULT));
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
    { /* setup malloc-interception after internal allocations */
      const libxs_malloc_function null_malloc_fn = { 0 };
      const libxs_free_function null_free_fn = { 0 };
      const char *const env_k = getenv("LIBXS_MALLOC");
      char *const env_t = getenv("LIBXS_MALLOC_LIMIT");
      const char* env_i = (NULL != env_t ? strtok(env_t, INTERNAL_DELIMS) : NULL);
      const size_t malloc_lo = internal_parse_nbytes(env_i, LIBXS_MALLOC_LIMIT);
      const size_t malloc_hi = (NULL != env_i ? internal_parse_nbytes(
        strtok(NULL, INTERNAL_DELIMS), LIBXS_SCRATCH_UNLIMITED) : LIBXS_SCRATCH_UNLIMITED);
      const int malloc_kind = ((NULL == env_k || 0 == *env_k) ? 0/*disabled*/ : atoi(env_k));
      libxs_xset_default_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
      libxs_xset_scratch_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
      libxs_set_malloc(malloc_kind, &malloc_lo, &malloc_hi); /* implies libxs_malloc_init */
    }
#if defined(LIBXS_MAXTARGET)
    libxs_set_target_arch(LIBXS_STRINGIFY(LIBXS_MAXTARGET));
#else /* attempt to set libxs_target_archid per environment variable */
    libxs_set_target_arch(getenv("LIBXS_TARGET"));
#endif
    { const char *const env = getenv("LIBXS_SYNC");
      libxs_nosync = (NULL == env || 0 == *env) ? 0/*default*/ : atoi(env);
    }
    /* clear internal counters/statistic */
    for (i = 0; i < 4/*sml/med/big/xxx*/; ++i) {
      LIBXS_MEMZERO127(&internal_statistic[0/*DP*/][i]);
      LIBXS_MEMZERO127(&internal_statistic[1/*SP*/][i]);
    }
    internal_statistic_mnk = LIBXS_MAX_DIM;
    internal_statistic_sml = 13;
    internal_statistic_med = 23;
    LIBXS_ASSERT(LIBXS_ISPOT(LIBXS_CAPACITY_REGISTRY));
    libxs_hash_init(libxs_target_archid); /* used by debug memory allocation (checksum) */
    libxs_memory_init(libxs_target_archid);
    if (
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      (EXIT_SUCCESS == libxs_xmalloc(&new_cache, /* if internal_cache_size is zero, allocation must still happen (later control-flow too expensive) */
        sizeof(internal_cache_type) * (LIBXS_NTHREADS_MAX), LIBXS_CACHELINE/*alignment*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_cache) &&
#endif
      (EXIT_SUCCESS == libxs_xmalloc(&new_keys, (LIBXS_CAPACITY_REGISTRY) * sizeof(internal_regkey_type), 0/*auto-align*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_keys) &&
      (EXIT_SUCCESS == libxs_xmalloc(&new_registry, (LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer), 0/*auto-align*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_registry))
    {
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL != new_cache); /* SA: suppress false positive */
      memset(new_cache, 0, (LIBXS_NTHREADS_MAX) * sizeof(internal_cache_type));
#endif
      libxs_xcopy_init(libxs_target_archid);
      libxs_dnn_init(libxs_target_archid);
      { const char *const env = getenv("LIBXS_GEMM_PREFETCH");
#if (defined(_WIN32) || defined(__CYGWIN__))
        libxs_gemm_auto_prefetch_default = INTERNAL_PREFETCH;
#else
        libxs_gemm_auto_prefetch_default = (0 == internal_statistic_ntry(0/*DP*/) && 0 == internal_statistic_ntry(1/*SP*/))
          /* avoid special prefetch if static code is present, since such code uses INTERNAL_PREFETCH */
          ? (((LIBXS_X86_AVX512 >= libxs_target_archid || LIBXS_X86_AVX512_CORE <= libxs_target_archid))
            ? LIBXS_GEMM_PREFETCH_AL2BL2_VIA_C : LIBXS_GEMM_PREFETCH_BL2_VIA_C)
          : INTERNAL_PREFETCH;
#endif
        libxs_gemm_auto_prefetch = INTERNAL_PREFETCH;
        if (NULL != env && 0 != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            libxs_gemm_auto_prefetch_default = libxs_gemm_uid2prefetch(uid);
            libxs_gemm_auto_prefetch = libxs_gemm_auto_prefetch_default;
            internal_gemm_auto_prefetch_locked = 1;
          }
        }
      }
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) ((libxs_code_pointer*)new_registry)[i].ptr = NULL;
      LIBXS_ASSERT(NULL == internal_registry && NULL == internal_registry_keys);
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL == internal_cache_buffer);
      internal_cache_buffer = (internal_cache_type*)new_cache;
#endif
      internal_registry_keys = (internal_regkey_type*)new_keys; /* prior to registering static kernels */
#if defined(LIBXS_BUILD) && !defined(LIBXS_DEFAULT_CONFIG)
#     include <libxs_dispatch.h>
#endif
      libxs_gemm_init(libxs_target_archid);
#if defined(LIBXS_TRACE)
      { int filter_threadid = 0/*only main-thread*/, filter_mindepth = 0, filter_maxnsyms = 0;
        const int init_code = libxs_trace_init(filter_threadid, filter_mindepth, filter_maxnsyms);
        if (EXIT_SUCCESS != init_code && 0 != libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS ERROR: failed to initialize TRACE (error #%i)!\n", init_code);
        }
      }
#endif
      { /* commit the registry buffer and enable global visibility */
        void *const pv_registry = &internal_registry;
        LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)((void**)pv_registry, (void*)new_registry, LIBXS_ATOMIC_SEQ_CST);
      }
    }
    else {
      if (0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to allocate internal buffers!\n");
      }
      libxs_xfree(new_registry, 0/*no check*/);
      libxs_xfree(new_keys, 0/*no check*/);
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      libxs_xfree(new_cache, 0/*no check*/);
#endif
    }
  }
#if (0 != LIBXS_SYNC) /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
  LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
}


LIBXS_API LIBXS_ATTRIBUTE_CTOR void libxs_init(void)
{
  if (0 == LIBXS_ATOMIC_LOAD(&internal_registry, LIBXS_ATOMIC_RELAXED)) {
    static unsigned int ninit = 0, gid = 0;
    const unsigned int tid = LIBXS_ATOMIC_ADD_FETCH(&ninit, 1, LIBXS_ATOMIC_SEQ_CST);
    LIBXS_ASSERT(0 < tid);
    /* libxs_ninit (1: initialization started, 2: library initialized, higher: to invalidate code-TLS) */
    if (1 == tid) {
      libxs_timer_tickint s0 = libxs_timer_tick_rtc(); /* warm-up */
      libxs_timer_tickint t0 = libxs_timer_tick_tsc(); /* warm-up */
      s0 = libxs_timer_tick_rtc(); t0 = libxs_timer_tick_tsc(); /* start timing */
      assert(0 == LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_SEQ_CST)); /* !LIBXS_ASSERT */
      /* coverity[check_return] */
      LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST);
      gid = tid; /* protect initialization */
#if (0 != LIBXS_SYNC)
      /* coverity[check_return] */
      LIBXS_TLS_CREATE(&libxs_tlskey);
      { /* construct and initialize locks */
# if defined(LIBXS_REGLOCK_TRY)
        const char *const env_trylock = getenv("LIBXS_TRYLOCK");
# endif
        LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr_global;
# if (1 < INTERNAL_REGLOCK_MAXN)
        int i;
        LIBXS_LOCK_ATTR_TYPE(LIBXS_REGLOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_REGLOCK, &attr);
# elif defined(LIBXS_UNIFY_LOCKS)
        internal_reglock_ptr = &libxs_lock_global;
# else
        static LIBXS_LOCK_TYPE(LIBXS_REGLOCK) internal_reglock;
        internal_reglock_ptr = &internal_reglock;
        LIBXS_LOCK_ATTR_TYPE(LIBXS_REGLOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_REGLOCK, &attr);
        LIBXS_LOCK_INIT(LIBXS_REGLOCK, internal_reglock_ptr, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_REGLOCK, &attr);
# endif
        LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr_global);
        LIBXS_LOCK_INIT(LIBXS_LOCK, &libxs_lock_global, &attr_global);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr_global);
        /* control number of locks needed; LIBXS_TRYLOCK implies only 1 lock */
# if defined(LIBXS_REGLOCK_TRY)
        if (NULL == env_trylock || 0 == *env_trylock)
# endif
        { /* no LIBXS_TRYLOCK */
# if defined(LIBXS_VTUNE)
          internal_reglock_count = 1; /* avoid duplicated kernels */
# elif (1 < INTERNAL_REGLOCK_MAXN)
          const char *const env_nlocks = getenv("LIBXS_NLOCKS");
          const int reglock_count = (NULL == env_nlocks || 0 == *env_nlocks || 1 > atoi(env_nlocks))
            ? (INTERNAL_REGLOCK_MAXN) : LIBXS_MIN(atoi(env_nlocks), INTERNAL_REGLOCK_MAXN);
          internal_reglock_count = LIBXS_LO2POT(reglock_count);
# else
          internal_reglock_count = 0;
# endif
        }
# if defined(LIBXS_REGLOCK_TRY)
        else { /* LIBXS_TRYLOCK environment variable specified */
          internal_reglock_count = (0 != atoi(env_trylock) ? 1
#   if (1 < INTERNAL_REGLOCK_MAXN)
            : INTERNAL_REGLOCK_MAXN);
#   else
            : 0);
#   endif
        }
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
        LIBXS_ASSERT(1 <= internal_reglock_count);
        for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_INIT(LIBXS_REGLOCK, &internal_reglock[i].state, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_REGLOCK, &attr);
# endif
      }
#endif
      { /* determine whether this instance is unique or not */
#if defined(_WIN32)
        internal_singleton_handle = CreateMutex(NULL, TRUE, "GlobalLIBXS");
#else
        const int result = LIBXS_SNPRINTF(internal_singleton_fname, sizeof(internal_singleton_fname), "/tmp/.libxs.%u",
          /*rely on user id to avoid permission issues in case of left-over files*/(unsigned int)getuid());
        struct flock singleton_flock;
        int singleton_handle;
        singleton_flock.l_start = 0;
        singleton_flock.l_len = 0; /* entire file */
        singleton_flock.l_type = F_WRLCK; /* exclusive across PIDs */
        singleton_flock.l_whence = SEEK_SET;
        singleton_handle = ((0 < result && (int)sizeof(internal_singleton_fname) > result) ? open(
          internal_singleton_fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR) : -1);
        internal_singleton_handle = fcntl(singleton_handle, F_SETLK, &singleton_flock);
        if (0 > internal_singleton_handle && 0 <= singleton_handle) close(singleton_handle);
#endif  /* coverity[leaked_handle] */
      }
      { /* calibrate timer */
        int register_termination_proc;
        libxs_timer_tickint s1, t1;
        internal_init(); /* must be first to initialize verbosity, etc. */
        if (INTERNAL_SINGLETON(internal_singleton_handle)) { /* after internal_init */
          internal_dump(stdout, 1/*urgent*/);
        }
        s1 = libxs_timer_tick_rtc(); t1 = libxs_timer_tick_tsc(); /* mid-timing */
        libxs_cpuid_x86(&internal_cpuid_info);
        if (0 != internal_cpuid_info.constant_tsc && t0 < t1) {
          libxs_timer_scale = libxs_timer_duration_rtc(s0, s1) / (t1 - t0);
        }
        register_termination_proc = atexit(internal_finalize);
        s1 = libxs_timer_tick_rtc(); t1 = libxs_timer_tick_tsc(); /* final timing */
        /* set timer-scale and determine start of the "uptime" (shown at termination) */
        if (t0 < t1 && 0.0 < libxs_timer_scale) {
          const double scale = libxs_timer_duration_rtc(s0, s1) / (t1 - t0);
          const double diff = LIBXS_DELTA(libxs_timer_scale, scale) / scale;
          if (5E-5 > diff) {
            libxs_timer_scale = scale;
            internal_timer_start = t0;
          }
          else {
            libxs_timer_scale = 0;
            internal_timer_start = s0;
#if defined(_DEBUG)
            libxs_se = 1;
#endif
          }
        }
        else {
          internal_timer_start = s0;
          libxs_timer_scale = 0;
        }
        if (0 != libxs_verbosity) { /* library code is expected to be mute */
          if (EXIT_SUCCESS != register_termination_proc) {
            fprintf(stderr, "LIBXS ERROR: failed to register termination procedure!\n");
          }
          if (0 == libxs_timer_scale) {
            fprintf(stderr, "LIBXS WARNING: timer is maybe not cycle-accurate!\n");
          }
        }
      }
      assert(1 == LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_SEQ_CST)); /* !LIBXS_ASSERT */
      /* coverity[check_return] */
      LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST);
    }
    else /*if (gid != tid)*/ { /* avoid recursion */
      LIBXS_ASSERT(gid != tid);
      LIBXS_UNUSED(gid);
      while (2 > LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_RELAXED)) LIBXS_SYNC_YIELD;
      internal_init();
    }
#if defined(LIBXS_PERF)
    libxs_perf_init();
#endif
  }
  LIBXS_ASSERT(1 < libxs_ninit);
}


LIBXS_API LIBXS_ATTRIBUTE_NO_TRACE void libxs_finalize(void);
LIBXS_API LIBXS_ATTRIBUTE_DTOR void libxs_finalize(void)
{
  void *const regaddr = &internal_registry;
  uintptr_t regptr = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)((uintptr_t*)regaddr, LIBXS_ATOMIC_RELAXED);
  libxs_code_pointer* registry = (libxs_code_pointer*)regptr;
  if (NULL != registry) {
    int i;
#if (0 != LIBXS_SYNC)
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
# if (1 < INTERNAL_REGLOCK_MAXN)
    { /* acquire locks and thereby shortcut lazy initialization later on */
      int ntry = 0, n;
      do {
        for (i = 0, n = 0; i < internal_reglock_count; ++i) {
          if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) == LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, &internal_reglock[i].state)) ++n;
        }
        ntry += (0 == n ? 1 : 0);
      } while (n < internal_reglock_count && ntry < LIBXS_CLEANUP_NTRY);
    }
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
#endif
    regptr = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)((uintptr_t*)regaddr, LIBXS_ATOMIC_RELAXED);
    registry = (libxs_code_pointer*)regptr;
    if (NULL != registry) {
      internal_regkey_type*const registry_keys = internal_registry_keys;
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      internal_cache_type *const cache_buffer = internal_cache_buffer;
#endif
      unsigned int rest = 0, errors = 0;
#if defined(LIBXS_TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
#if defined(LIBXS_PERF)
      libxs_perf_finalize();
#endif
      libxs_xcopy_finalize();
      libxs_gemm_finalize();
      libxs_dnn_finalize();
      /* coverity[check_return] */
      LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_RELAXED); /* invalidate code cache (TLS) */
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      internal_cache_buffer = NULL;
#endif
      internal_registry_keys = NULL; /* make registry keys unavailable */
      LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, LIBXS_BITS)((uintptr_t*)regaddr, LIBXS_ATOMIC_SEQ_CST);
      internal_registry_nbytes = 0; internal_registry_nleaks = 0;
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        /*const*/ libxs_code_pointer code = registry[i];
        if (NULL != code.ptr_const) {
          /* check if the registered entity is a GEMM kernel */
          switch (LIBXS_DESCRIPTOR_KIND(registry_keys[i].entry.kind)) {
            case LIBXS_KERNEL_KIND_MATMUL: {
              const libxs_gemm_descriptor *const desc = &registry_keys[i].entry.gemm.desc;
              if (1 < desc->m && 1 < desc->n) {
                const unsigned int njit = (0 == (LIBXS_CODE_STATIC & code.uval) ? 1 : 0);
                const unsigned int nsta = (0 != (LIBXS_CODE_STATIC & code.uval) ? 1 : 0);
                /* count whether kernel is static or JIT-code */
                internal_update_mmstatistic(desc, 0, 0, njit, nsta);
              }
              else {
                ++internal_statistic_num_gemv;
              }
              ++rest;
            } break;
            case LIBXS_KERNEL_KIND_MCOPY: {
              ++internal_statistic_num_mcopy;
            } break;
            case LIBXS_KERNEL_KIND_MELTW: {
              ++internal_statistic_num_meltw;
            } break;
            case LIBXS_KERNEL_KIND_TRANS: {
              ++internal_statistic_num_tcopy;
            } break;
            case LIBXS_KERNEL_KIND_TRSM: {
              ++internal_statistic_num_trsm;
            } break;
            case LIBXS_KERNEL_KIND_TRMM: {
              ++internal_statistic_num_trmm;
            } break;
            case LIBXS_KERNEL_KIND_USER: {
              ++internal_statistic_num_user;
            } break;
            default: if (LIBXS_KERNEL_UNREGISTERED <= LIBXS_DESCRIPTOR_KIND(registry_keys[i].entry.kind)) {
              ++errors;
            }
            else {
              ++rest;
            }
          }
          if (0 != libxs_verbosity) { /* library code is expected to be mute */
            if (0 != errors) {
              fprintf(stderr, "LIBXS ERROR: code registry is corrupted!\n");
            }
            if (LIBXS_CAPACITY_REGISTRY == (rest + errors + internal_statistic_num_gemv +
              internal_statistic_num_mcopy + internal_statistic_num_meltw +
              internal_statistic_num_tcopy + internal_statistic_num_trsm +
              internal_statistic_num_trmm + internal_statistic_num_user))
            {
              fprintf(stderr, "LIBXS WARNING: code registry was exhausted!\n");
            }
          }
          if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
            void* buffer = NULL;
            size_t size = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            if (EXIT_SUCCESS == libxs_get_malloc_xinfo(code.ptr_const, &size, NULL/*flags*/, &buffer)) {
              if (LIBXS_KERNEL_KIND_USER == LIBXS_DESCRIPTOR_KIND(registry_keys[i].entry.kind)
                /* dump user-data just like JIT'ted code */
                && 0 > libxs_verbosity)
              {
                char name[16];
                int nchar;
#if defined(LIBXS_REGUSER_HASH)
                const size_t descsize = LIBXS_DESCRIPTOR_ISBIG(registry_keys[i].entry.kind)
                  ? LIBXS_DESCRIPTOR_MAXSIZE : LIBXS_DESCRIPTOR_SIGSIZE;
                const unsigned int id = libxs_crc32(LIBXS_HASH_SEED, registry_keys[i].entry.user.desc,
                  descsize - sizeof(libxs_descriptor_kind));
                LIBXS_ASSERT(descsize > sizeof(libxs_descriptor_kind));
#else
                const unsigned int id = internal_statistic_num_user;
#endif
                nchar = LIBXS_SNPRINTF(name, sizeof(name), "%010u.user", id);
                if (0 < nchar && (int)sizeof(name) > nchar) {
                  LIBXS_EXPECT(EXIT_SUCCESS, libxs_dump("LIBXS-USER-DUMP", name, code.ptr_const, size, 0/*unique*/));
                }
              }
#if !defined(NDEBUG)
              registry[i].ptr = NULL;
#endif
              libxs_xfree(code.ptr_const, 0/*no check*/);
              /* round-up size (it is fine to assume 4 KB pages since it is likely more accurate than not rounding up) */
              internal_registry_nbytes += LIBXS_UP2(size + (((char*)code.ptr_const) - (char*)buffer), LIBXS_PAGE_MINSIZE);
            }
            else ++internal_registry_nleaks;
          }
        }
      }
      /* release buffers (registry, keys, cache) */
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      libxs_xfree(cache_buffer, 0/*no check*/);
#endif
      libxs_xfree(registry_keys, 0/*no check*/);
      libxs_xfree(registry, 0/*no check*/);
    }
#if (0 != LIBXS_SYNC) /* LIBXS_LOCK_RELEASE, but no LIBXS_LOCK_DESTROY */
# if (1 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
    /* coverity[check_return] */
    LIBXS_TLS_DESTROY(libxs_tlskey);
#endif
  }
}


LIBXS_API void libxs_sink(LIBXS_VARIADIC)
{
  /* does nothing else but sinking given arguments */
}


LIBXS_API int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(__MIC__)
  return libxs_target_archid;
#else /* no JIT support */
  return LIBXS_MIN(libxs_target_archid, LIBXS_X86_GENERIC);
#endif
}


LIBXS_API void libxs_set_target_archid(int id)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXS_X86_AVX512_SPR:
    case LIBXS_X86_AVX512_CPX:
    case LIBXS_X86_AVX512_CLX:
    case LIBXS_X86_AVX512_CORE:
    case LIBXS_X86_AVX512_KNM:
    case LIBXS_X86_AVX512_MIC:
    case LIBXS_X86_AVX512:
    case LIBXS_X86_AVX2:
    case LIBXS_X86_AVX:
    case LIBXS_X86_SSE42:
    case LIBXS_X86_SSE3:
    case LIBXS_AARCH64_V81:
    case LIBXS_TARGET_ARCH_GENERIC: {
      target_archid = id;
    } break;
    default: if (LIBXS_X86_GENERIC <= id) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else {
      target_archid = libxs_cpuid();
    }
  }
  LIBXS_ATOMIC_STORE(&libxs_target_archid, target_archid, LIBXS_ATOMIC_RELAXED);
  if (0 != libxs_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxs_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = libxs_cpuid_name(target_archid);
      fprintf(stderr, "LIBXS WARNING: \"%s\" code may fail to run on \"%s\"!\n",
        target_arch, libxs_cpuid_name(cpuid));
    }
  }
}


LIBXS_API const char* libxs_get_target_arch(void)
{
  LIBXS_INIT
  return libxs_cpuid_name(libxs_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXS_API const char* libxsf_get_target_arch(int* length);
LIBXS_API const char* libxsf_get_target_arch(int* length)
{
  const char *const arch = libxs_get_target_arch();
  /* valid here since function is not in the public interface */
  LIBXS_ASSERT(NULL != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXS_API void libxs_set_target_arch(const char* arch)
{
  const int cpuid = libxs_cpuid();
  int target_archid;
  if (NULL != arch && 0 != *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 < jit) {
      target_archid = LIBXS_X86_GENERIC + jit;
    }
    else if (arch == libxs_stristr(arch, "spr") || arch == libxs_stristr(arch, "amx")) {
      target_archid = LIBXS_X86_AVX512_SPR;
    }
    else if (arch == libxs_stristr(arch, "cpx")) {
      target_archid = LIBXS_X86_AVX512_CPX;
    }
    else if (arch == libxs_stristr(arch, "clx")) {
      target_archid = LIBXS_X86_AVX512_CLX;
    }
    else if (arch == libxs_stristr(arch, "skx") || arch == libxs_stristr(arch, "skl")
          /* "avx3"/"avx512" previously enabled LIBXS_X86_AVX512 */
          || arch == libxs_stristr(arch, "avx3") || arch == libxs_stristr(arch, "avx512"))
    {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (arch == libxs_stristr(arch, "knm")) {
      target_archid = LIBXS_X86_AVX512_KNM;
    }
    else if (arch == libxs_stristr(arch, "knl") || arch == libxs_stristr(arch, "mic")) {
      target_archid = LIBXS_X86_AVX512_MIC;
    }
    else if (arch == libxs_stristr(arch, "hsw") || arch == libxs_stristr(arch, "avx2")) {
      target_archid = LIBXS_X86_AVX2;
    }
    else if (arch == libxs_stristr(arch, "snb") || arch == libxs_stristr(arch, "avx")) {
      target_archid = LIBXS_X86_AVX;
    }
    else if (arch == libxs_stristr(arch, "wsm") || arch == libxs_stristr(arch, "nhm")
       || arch == libxs_stristr(arch, "sse4_2") || arch == libxs_stristr(arch, "sse4.2")
       || arch == libxs_stristr(arch, "sse4"))
    {
      target_archid = LIBXS_X86_SSE42;
    }
    else if (arch == libxs_stristr(arch, "sse3"))
    {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (arch == libxs_stristr(arch, "x86") || arch == libxs_stristr(arch, "x64")
          || arch == libxs_stristr(arch, "x86_64") || arch == libxs_stristr(arch, "sse2"))
    {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (arch == libxs_stristr(arch, "aarch64"))
    {
      target_archid = LIBXS_AARCH64_V81;
    }
    else if (arch == libxs_stristr(arch, "generic")
          || arch == libxs_stristr(arch, "none"))
    {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
    else {
      target_archid = cpuid;
    }
  }
  else {
    target_archid = cpuid;
  }
  if (cpuid < target_archid) { /* warn about code path if beyond CPUID */
    static int error_once = 0;
    if ( 0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      const char *const target_arch = libxs_cpuid_name(target_archid);
      fprintf(stderr, "LIBXS WARNING: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, libxs_cpuid_name(cpuid));
    }
#if 0 /* limit code path to confirmed features */
    target_archid = cpuid;
#endif
  }
  LIBXS_ATOMIC_STORE(&libxs_target_archid, target_archid, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API int libxs_get_verbosity(void)
{
  LIBXS_INIT
  return libxs_verbosity;
}


LIBXS_API void libxs_set_verbosity(int level)
{
  LIBXS_INIT
  LIBXS_ATOMIC_STORE(&libxs_verbosity, level, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_auto_prefetch(void)
{
  return (libxs_gemm_prefetch_type)libxs_gemm_auto_prefetch;
}


LIBXS_API void libxs_set_gemm_auto_prefetch(libxs_gemm_prefetch_type strategy)
{
  if (0 == internal_gemm_auto_prefetch_locked) { /* LIBXS_GEMM_PREFETCH environment takes precedence */
    LIBXS_ATOMIC_STORE(&libxs_gemm_auto_prefetch_default, strategy, LIBXS_ATOMIC_RELAXED);
    LIBXS_ATOMIC_STORE(&libxs_gemm_auto_prefetch, strategy, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_API unsigned char libxs_typesize(libxs_datatype datatype)
{
  const unsigned char result = (unsigned char)LIBXS_TYPESIZE(datatype);
  if (0 != result) {
    return result;
  }
  else {
    static int error_once = 0;
    LIBXS_ASSERT_MSG(0, "unsupported data type");
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: unsupported data type!\n");
    }
    return 1; /* avoid to return 0 to avoid div-by-zero in static analysis of depending code */
  }
}


LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue)
{
  int result = EXIT_SUCCESS;
  if (NULL != value && NULL != dvalue) {
    switch (datatype) {
      case LIBXS_DATATYPE_F64: *dvalue =         (*(const double   *)value); break;
      case LIBXS_DATATYPE_F32: *dvalue = (double)(*(const float    *)value); break;
      case LIBXS_DATATYPE_I64: *dvalue = (double)(*(const long long*)value); break;
      case LIBXS_DATATYPE_I32: *dvalue = (double)(*(const int      *)value); break;
      case LIBXS_DATATYPE_I16: *dvalue = (double)(*(const short    *)value); break;
      case LIBXS_DATATYPE_I8:  *dvalue = (double)(*(const char     *)value); break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INTERN const char* libxs_typename(libxs_datatype datatype)
{
  switch (datatype) {
    case LIBXS_DATATYPE_F64:  return "f64";
    case LIBXS_DATATYPE_F32:  return "f32";
    case LIBXS_DATATYPE_BF16: return "bf16";
    case LIBXS_DATATYPE_F16:  return "f16";
    case LIBXS_DATATYPE_I64:  return "i64";
    case LIBXS_DATATYPE_I32:  return "i32";
    case LIBXS_DATATYPE_I16:  return "i16";
    case LIBXS_DATATYPE_I8:   return "i8";
    default: {
      if (LIBXS_GEMM_PRECISION_I16 == LIBXS_GETENUM_INP(datatype) &&
          LIBXS_GEMM_PRECISION_I32 == LIBXS_GETENUM_OUT(datatype))
      {
        return "i16i32";
      }
      else if (LIBXS_GEMM_PRECISION_I16 == LIBXS_GETENUM_INP(datatype) &&
               LIBXS_GEMM_PRECISION_F32 == LIBXS_GETENUM_OUT(datatype))
      {
        return "i16f32";
      }
      else if (LIBXS_GEMM_PRECISION_I8 == LIBXS_GETENUM_INP(datatype) &&
               LIBXS_GEMM_PRECISION_I32 == LIBXS_GETENUM_OUT(datatype))
      {
        return "i8i32";
      }
      else if (LIBXS_GEMM_PRECISION_BF16 == LIBXS_GETENUM_INP(datatype) &&
               LIBXS_GEMM_PRECISION_F32 == LIBXS_GETENUM_OUT(datatype))
      {
        return "bf16f32";
      }
      else {
        return "void";
      }
    }
  }
}


LIBXS_API_INLINE void internal_get_typesize_string(char buffer[4], int buffer_size, size_t typesize)
{
  LIBXS_ASSERT(256 > typesize && 4 <= buffer_size);
  if (10 > typesize) {
    buffer[0] = (char)('0' + typesize);
    buffer[1] = 0;
  }
  else {
    LIBXS_SNPRINTF(buffer, buffer_size, "%i", (int)typesize);
  }
}


LIBXS_API_INTERN int libxs_dump(const char* title, const char* name, const void* data, size_t size, int unique)
{
  int result;
  if (NULL != name && '\0' != *name && NULL != data && 0 != size) {
    FILE* data_file = fopen(name, "rb");
    int diff = 0;
    if (NULL == data_file) { /* file does not exist */
      data_file = fopen(name, "wb");
      if (NULL != data_file) { /* dump data into a file */
        if (size != fwrite(data, 1, size, data_file)) result = EXIT_FAILURE;;
        result = fclose(data_file);
      }
      else result = EXIT_FAILURE;
    }
    else if (0 != unique) { /* check existing file */
      const char* check_a = (const char*)data;
      char check_b[4096];
      size_t rest = size;
      do {
        const size_t n = fread(check_b, 1, LIBXS_MIN(sizeof(check_b), rest), data_file);
        diff += memcmp(check_a, check_b, LIBXS_MIN(sizeof(check_b), n));
        check_a += n;
        rest -= n;
      } while (0 < rest && 0 == diff);
      result = fclose(data_file);
    }
    else {
      result = EXIT_SUCCESS;
    }
    if (EXIT_SUCCESS == result && NULL != title && '\0' != *title) {
      fprintf(stderr, "%s(ptr:file) %p : %s\n", title, data, name);
    }
    if (0 != diff) { /* override existing dump and warn about erroneous condition */
      fprintf(stderr, "LIBXS ERROR: %s is not a unique filename!\n", name);
      data_file = fopen(name, "wb");
      if (NULL != data_file) { /* dump data into a file */
        LIBXS_EXPECT(size, fwrite(data, 1, size, data_file));
        LIBXS_EXPECT(EXIT_SUCCESS, fclose(data_file));
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INTERN int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__)
  const char * /*const*/ target_arch = libxs_cpuid_name(libxs_target_archid);
  /* large enough temporary buffer for generated code */
  char jit_buffer[LIBXS_CODE_MAXSIZE], jit_name[256] = { 0 };
  libxs_generated_code generated_code;
  libxs_kernel_xinfo extra;

  LIBXS_MEMZERO127(&generated_code);
  generated_code.generated_code = jit_buffer;
  generated_code.buffer_size = sizeof(jit_buffer);
  /* setup code generation */
  generated_code.arch = libxs_target_archid;
  generated_code.code_type = 2;

# if !defined(NDEBUG) /* should not be needed (all members will be initialized below) */
  LIBXS_MEMZERO127(&extra);
# endif
  extra.registered = regindex;
  extra.nflops = 0;

  LIBXS_ASSERT(NULL != generated_code.generated_code || 0 == generated_code.buffer_size);
  LIBXS_ASSERT(NULL != request && 0 != libxs_target_archid);
  LIBXS_ASSERT(NULL != code && NULL == code->ptr_const);
  LIBXS_ASSERT(0 == LIBXS_DESCRIPTOR_ISBIG(request->kind));

  switch (request->kind) { /* generate kernel */
    case LIBXS_BUILD_KIND_GEMM: { /* small MxM kernel */
      LIBXS_ASSERT(NULL != request->descriptor.gemm);
# if 0 /* dummy kernel for an empty shape is desired */
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
# endif
      {
        const unsigned int m = request->descriptor.gemm->m, n = request->descriptor.gemm->n, k = request->descriptor.gemm->k;
        extra.nflops = 2 * m * n * k;
# if !defined(LIBXS_DENY_RETARGET) /* disable: ECFLAGS=-DLIBXS_DENY_RETARGET */
        if ((LIBXS_X86_AVX2 < libxs_target_archid) && (libxs_target_archid <= LIBXS_X86_ALLFEAT) &&
           (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.gemm->datatype) ||
            LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.gemm->datatype)) &&
           (16 >= (m * k) || 16 >= (k * n) || 16 >= (m * n)))
        {
          /* TODO: shall we update variable "target_arch" (name)? */
          generated_code.arch = LIBXS_X86_AVX2;
        }
# endif
        LIBXS_NO_OFFLOAD(void, libxs_generator_gemm_kernel, &generated_code, request->descriptor.gemm);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.gemm->datatype);
          const char *const meltw_tname = libxs_typename((libxs_datatype)request->descriptor.gemm->meltw_datatype_aux);
          int typesigns = 0, br = 0;
          char tc_option[16] = { 0 };
          int decompress_A = 0;
          int sparsity_factor_A = 1;
          if ((request->descriptor.gemm->meltw_operation == LIBXS_MELTW_OPERATION_DECOMPRESS_A) ||
              (request->descriptor.gemm->meltw_operation == LIBXS_MELTW_OPERATION_COLBIAS_ACT_DECOMPRESS_A))
          {
            decompress_A = 1;
            sparsity_factor_A = (int)request->descriptor.gemm->meltw_param;
          }

          /* query batch reduce variant */
          if ( (LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS & request->descriptor.gemm->flags) > 1 ) {
            br = 1;
          } else if ( (LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET & request->descriptor.gemm->flags) > 1 ) {
            br = 2;
          } else if ( (LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE & request->descriptor.gemm->flags) > 1 ) {
            br = 3;
          } else {
            br = 0;
          }
          /* query A/B sign combinations */
          if ( (LIBXS_GEMM_FLAG_A_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 1;
          } else if ( (LIBXS_GEMM_FLAG_B_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 2;
          } else if ( (LIBXS_GEMM_FLAG_AB_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 3;
          } else {
            typesigns = 0;
          }
          /* query tileconfig options */
          if (((LIBXS_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) != 0) &&
              ((LIBXS_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) == 0) ) {
            LIBXS_SNPRINTF(tc_option, sizeof(tc_option), "conf");
          } else if (((LIBXS_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) == 0) &&
                     ((LIBXS_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) != 0) ) {
            LIBXS_SNPRINTF(tc_option, sizeof(tc_option), "rele");
          } else if (((LIBXS_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) != 0) &&
                     ((LIBXS_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) != 0)) {
            LIBXS_SNPRINTF(tc_option, sizeof(tc_option), "none");
          } else {
            LIBXS_SNPRINTF(tc_option, sizeof(tc_option), "abid");
          }

          if ( request->descriptor.gemm->meltw_operation != 0 ) {
            /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
            LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_br%i_uh%u_si%i_tc-%s_avnni%i_bvnni%i_cvnni%i_meop%u-%s_mefl%u_meld%u-%u-%u_decompress_A%i_spfactor%i.mxm", target_arch, tname,
              0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
              0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't', m, n, k,
              request->descriptor.gemm->lda, request->descriptor.gemm->ldb, request->descriptor.gemm->ldc,
              /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.gemm->flags) ? 0 : */1,
              0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.gemm->flags) ? 0 : 1, uid,
              br, (unsigned int)request->descriptor.gemm->c3, typesigns, tc_option,
              0 != (LIBXS_GEMM_FLAG_VNNI_A  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXS_GEMM_FLAG_VNNI_B  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXS_GEMM_FLAG_VNNI_C  & request->descriptor.gemm->flags) ? 1 : 0,
              (unsigned int)request->descriptor.gemm->meltw_operation, meltw_tname, (unsigned int)request->descriptor.gemm->meltw_flags,
              request->descriptor.gemm->meltw_ldx, request->descriptor.gemm->meltw_ldy, request->descriptor.gemm->meltw_ldz, decompress_A, sparsity_factor_A );
          } else {
            /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
            LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_br%i_uh%u_si%i_tc-%s_avnni%i_bvnni%i_cvnni%i_decompress_A%i_spfactor%i.mxm", target_arch, tname,
              0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
              0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't', m, n, k,
              request->descriptor.gemm->lda, request->descriptor.gemm->ldb, request->descriptor.gemm->ldc,
              /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.gemm->flags) ? 0 : */1,
              0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.gemm->flags) ? 0 : 1, uid,
              br, (unsigned int)request->descriptor.gemm->c3, typesigns, tc_option,
              0 != (LIBXS_GEMM_FLAG_VNNI_A  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXS_GEMM_FLAG_VNNI_B  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXS_GEMM_FLAG_VNNI_C  & request->descriptor.gemm->flags) ? 1 : 0, decompress_A, sparsity_factor_A );
          }
        }
      }
    } break;
    case LIBXS_BUILD_KIND_PSPGEMM_CSR: { /* packed sparse gemm kernel, CSR format */
      LIBXS_ASSERT(NULL != request->descriptor.pspgemm_csr && 0 != request->descriptor.pspgemm_csr->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.pspgemm_csr->row_ptr && 0 != request->descriptor.pspgemm_csr->column_idx && 0 != request->descriptor.pspgemm_csr->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pspgemm_csr->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pspgemm_csr->gemm->datatype))
      {
        const unsigned int nnz = (request->descriptor.pspgemm_csr->gemm->lda == 0) ?
            request->descriptor.pspgemm_csr->row_ptr[request->descriptor.pspgemm_csr->gemm->m] : request->descriptor.pspgemm_csr->row_ptr[request->descriptor.pspgemm_csr->gemm->k];
        const unsigned int gemm_factor = (request->descriptor.pspgemm_csr->gemm->lda == 0) ? request->descriptor.pspgemm_csr->gemm->n : request->descriptor.pspgemm_csr->gemm->m;
        extra.nflops = 2 * nnz * gemm_factor * request->descriptor.pspgemm_csr->packed_width;
        LIBXS_NO_OFFLOAD(void, libxs_generator_packed_spgemm_csr_kernel, &generated_code, request->descriptor.pspgemm_csr->gemm,
          request->descriptor.pspgemm_csr->row_ptr, request->descriptor.pspgemm_csr->column_idx, request->descriptor.pspgemm_csr->values, request->descriptor.pspgemm_csr->packed_width);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.pspgemm_csr->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.pspgemm_csr->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i_nnz%u.pspgemm_csr", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.pspgemm_csr->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.pspgemm_csr->gemm->flags) ? 'n' : 't',
            request->descriptor.pspgemm_csr->gemm->m,   request->descriptor.pspgemm_csr->gemm->n,   request->descriptor.pspgemm_csr->gemm->k,
            request->descriptor.pspgemm_csr->gemm->lda, request->descriptor.pspgemm_csr->gemm->ldb, request->descriptor.pspgemm_csr->gemm->ldc,
            request->descriptor.pspgemm_csr->packed_width,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.pspgemm_csr->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.pspgemm_csr->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_PSPGEMM_CSC: { /* packed sparse gemm kernel, CSC format */
      LIBXS_ASSERT(NULL != request->descriptor.pspgemm_csc && 0 != request->descriptor.pspgemm_csc->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.pspgemm_csc->row_idx && 0 != request->descriptor.pspgemm_csc->column_ptr && 0 != request->descriptor.pspgemm_csc->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pspgemm_csc->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pspgemm_csc->gemm->datatype))
      {
        const unsigned int nnz = (request->descriptor.pspgemm_csc->gemm->lda == 0) ?
            request->descriptor.pspgemm_csc->column_ptr[request->descriptor.pspgemm_csc->gemm->k] : request->descriptor.pspgemm_csc->column_ptr[request->descriptor.pspgemm_csc->gemm->n];
        const unsigned int gemm_factor = (request->descriptor.pspgemm_csc->gemm->lda == 0) ? request->descriptor.pspgemm_csc->gemm->n : request->descriptor.pspgemm_csc->gemm->m;
        extra.nflops = 2 * nnz * gemm_factor * request->descriptor.pspgemm_csc->packed_width;
        LIBXS_NO_OFFLOAD(void, libxs_generator_packed_spgemm_csc_kernel, &generated_code, request->descriptor.pspgemm_csc->gemm,
          request->descriptor.pspgemm_csc->row_idx, request->descriptor.pspgemm_csc->column_ptr, request->descriptor.pspgemm_csc->values, request->descriptor.pspgemm_csc->packed_width);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.pspgemm_csc->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.pspgemm_csc->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i_nnz%u.pspgemm_csc", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.pspgemm_csc->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.pspgemm_csc->gemm->flags) ? 'n' : 't',
            request->descriptor.pspgemm_csc->gemm->m,   request->descriptor.pspgemm_csc->gemm->n,   request->descriptor.pspgemm_csc->gemm->k,
            request->descriptor.pspgemm_csc->gemm->lda, request->descriptor.pspgemm_csc->gemm->ldb, request->descriptor.pspgemm_csc->gemm->ldc,
            request->descriptor.pspgemm_csc->packed_width,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.pspgemm_csc->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.pspgemm_csc->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_PGEMMRMAC: { /* packed GEMM, B regular matrix, row-major */
      LIBXS_ASSERT(NULL != request->descriptor.pgemmacrm && 0 != request->descriptor.pgemmacrm->gemm);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pgemmacrm->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pgemmacrm->gemm->datatype))
      {
        extra.nflops = 2 * request->descriptor.pgemmacrm->packed_width * request->descriptor.pgemmacrm->gemm->m * request->descriptor.pgemmacrm->gemm->n * request->descriptor.pgemmacrm->gemm->k;
        LIBXS_NO_OFFLOAD(void, libxs_generator_packed_gemm_ac_rm, &generated_code, request->descriptor.pgemmacrm->gemm, request->descriptor.pgemmacrm->packed_width);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.pgemmacrm->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.pgemmacrm->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i.pgemmacrm", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.pgemmacrm->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.pgemmacrm->gemm->flags) ? 'n' : 't',
            request->descriptor.pgemmacrm->gemm->m,   request->descriptor.pgemmacrm->gemm->n,   request->descriptor.pgemmacrm->gemm->k,
            request->descriptor.pgemmacrm->gemm->lda, request->descriptor.pgemmacrm->gemm->ldb, request->descriptor.pgemmacrm->gemm->ldc,
            request->descriptor.pgemmacrm->packed_width,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.pgemmacrm->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.pgemmacrm->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_PGEMMRMBC: { /* packed GEMM, A regular matrix, row-major */
      LIBXS_ASSERT(NULL != request->descriptor.pgemmbcrm && 0 != request->descriptor.pgemmbcrm->gemm);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pgemmbcrm->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.pgemmbcrm->gemm->datatype))
      {
        extra.nflops = 2 * request->descriptor.pgemmbcrm->packed_width * request->descriptor.pgemmbcrm->gemm->m * request->descriptor.pgemmbcrm->gemm->n * request->descriptor.pgemmbcrm->gemm->k;
        LIBXS_NO_OFFLOAD(void, libxs_generator_packed_gemm_bc_rm, &generated_code, request->descriptor.pgemmbcrm->gemm, request->descriptor.pgemmbcrm->packed_width);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.pgemmbcrm->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.pgemmbcrm->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i.pgemmbcrm", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.pgemmbcrm->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.pgemmbcrm->gemm->flags) ? 'n' : 't',
            request->descriptor.pgemmbcrm->gemm->m,   request->descriptor.pgemmbcrm->gemm->n,   request->descriptor.pgemmbcrm->gemm->k,
            request->descriptor.pgemmbcrm->gemm->lda, request->descriptor.pgemmbcrm->gemm->ldb, request->descriptor.pgemmbcrm->gemm->ldc,
            request->descriptor.pgemmbcrm->packed_width,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.pgemmbcrm->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.pgemmbcrm->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SREG: { /* sparse register kernel */
      LIBXS_ASSERT(NULL != request->descriptor.sreg && 0 != request->descriptor.sreg->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.sreg->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.sreg->gemm->datatype))
      {
        const unsigned int nnz = request->descriptor.sreg->row_ptr[request->descriptor.sreg->gemm->m];
        extra.nflops = 2 * libxs_cpuid_vlen32(libxs_target_archid)/2 * request->descriptor.sreg->gemm->n * nnz;
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_reg_kernel, &generated_code, request->descriptor.sreg->gemm, target_arch,
          request->descriptor.sreg->row_ptr, request->descriptor.sreg->column_idx,
          (const double*)request->descriptor.sreg->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.sreg->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.sreg->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.sreg", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            request->descriptor.sreg->gemm->m,   request->descriptor.sreg->gemm->n,   request->descriptor.sreg->gemm->k,
            request->descriptor.sreg->gemm->lda, request->descriptor.sreg->gemm->ldb, request->descriptor.sreg->gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.sreg->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.sreg->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_MCOPY: { /* matcopy kernel */
      LIBXS_ASSERT(NULL != request->descriptor.mcopy);
# if 0 /* TODO: backend supports typesize <= 4, but kernels for typesize < 4 are incorrect */
      if (4 == request->descriptor.mcopy->typesize)
# endif
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_matcopy_kernel, &generated_code, request->descriptor.mcopy, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.mcopy->typesize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%ux%u_p%u.mcopy", target_arch, tsizename,
            request->descriptor.mcopy->m, request->descriptor.mcopy->n, request->descriptor.mcopy->ldi, request->descriptor.mcopy->ldo,
            (unsigned int)request->descriptor.mcopy->prefetch);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_MELTW: { /* matcopy kernel */
      LIBXS_ASSERT(NULL != request->descriptor.meltw);
      {
        /* dispatch eltwise code with AVX512_BF16 by demoting seemlessly to the current CPU arch */
        if ( ( generated_code.arch >= LIBXS_X86_AVX512_SPR ) &&
             ( generated_code.arch <= LIBXS_X86_ALLFEAT )       ) {
          int emu_amx = 0;
          const char *const env_emu_amx = getenv("EMULATE_AMX");
          if ( 0 == env_emu_amx ) {
          } else {
            emu_amx = atoi(env_emu_amx);
          }
          if (emu_amx > 0) {
            generated_code.arch = libxs_cpuid();
          }
        }

        LIBXS_NO_OFFLOAD(void, libxs_generator_mateltwise_kernel, &generated_code, request->descriptor.meltw);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.meltw->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          if ( request->descriptor.meltw->operation == LIBXS_MELTW_OPERATION_REDUCE_COLS_IDX ) {
            LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_idxtsize%u_%u_%ux%u_opcode%u_flags%u.meltw", target_arch, tsizename,
              request->descriptor.meltw->n, request->descriptor.meltw->m, request->descriptor.meltw->ldi, request->descriptor.meltw->ldo,
              (unsigned int)request->descriptor.meltw->operation, (unsigned int)request->descriptor.meltw->flags);
          } else {
            LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%ux%u_opcode%u_flags%u.meltw", target_arch, tsizename,
              request->descriptor.meltw->m, request->descriptor.meltw->n, request->descriptor.meltw->ldi, request->descriptor.meltw->ldo,
              (unsigned int)request->descriptor.meltw->operation, (unsigned int)request->descriptor.meltw->flags);
          }
        }
      }
    } break;
    case LIBXS_BUILD_KIND_MEQN: { /* matequation kernel */
      LIBXS_ASSERT(NULL != request->descriptor.meltw);
      {
        /* dispatch eltwise code with AVX512_BF16 by demoting seemlessly to the current CPU arch */
        if ( ( generated_code.arch >= LIBXS_X86_AVX512_SPR ) &&
             ( generated_code.arch <= LIBXS_X86_ALLFEAT )       ) {
          int emu_amx = 0;
          const char *const env_emu_amx = getenv("EMULATE_AMX");
          if ( 0 == env_emu_amx ) {
          } else {
            emu_amx = atoi(env_emu_amx);
          }
          if (emu_amx > 0) {
            generated_code.arch = libxs_cpuid();
          }
        }

        LIBXS_NO_OFFLOAD(void, libxs_generator_matequation_kernel, &generated_code, request->descriptor.meqn);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.meqn->datatype);
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%u_eqn-idx%u.meltw", target_arch, tsizename,
            request->descriptor.meqn->m, request->descriptor.meqn->n, request->descriptor.meqn->ldo,
            (unsigned int)request->descriptor.meqn->eqn_idx );
        }
      }
    } break;
    case LIBXS_BUILD_KIND_TRANS: { /* transpose kernel */
      LIBXS_ASSERT(NULL != request->descriptor.trans);
      if (4 == request->descriptor.trans->typesize || 8 == request->descriptor.trans->typesize) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_transpose_kernel, &generated_code, request->descriptor.trans, libxs_target_archid);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.trans->typesize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%u.trans", target_arch, tsizename,
            request->descriptor.trans->m, request->descriptor.trans->n, request->descriptor.trans->ldo);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_PGEMM: { /* compact P/GEMM-kernel (packed) */
      unsigned int tsize;
      LIBXS_ASSERT(NULL != request->descriptor.pgemm);
      tsize = (unsigned int)request->descriptor.pgemm->typesize;
      if (4 == tsize || 8 == tsize) {
        extra.nflops = 0; /* TODO */
        LIBXS_NO_OFFLOAD(void, libxs_generator_pgemm_kernel, &generated_code, request->descriptor.pgemm, libxs_target_archid);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c%c%c_%ux%ux%u_%u_%u_%u_%i.pgemm", target_arch, tsizename,
            request->descriptor.pgemm->transa, request->descriptor.pgemm->transb, request->descriptor.pgemm->layout,
            request->descriptor.pgemm->m, request->descriptor.pgemm->n, request->descriptor.pgemm->k,
            request->descriptor.pgemm->lda, request->descriptor.pgemm->ldb, request->descriptor.pgemm->ldc,
            (int)request->descriptor.pgemm->alpha_val);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_GETRF: { /* compact GETRF kernel (packed) */
      unsigned int tsize;
      LIBXS_ASSERT(NULL != request->descriptor.getrf);
      tsize = (unsigned int)request->descriptor.getrf->typesize;
      if (4 == tsize || 8 == tsize) {
        extra.nflops = 0; /* TODO */
        LIBXS_NO_OFFLOAD(void, libxs_generator_getrf_kernel, &generated_code, request->descriptor.getrf, libxs_target_archid);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c_%ux%u_%u.getrf", target_arch, tsizename,
            request->descriptor.getrf->layout, request->descriptor.getrf->m, request->descriptor.getrf->n, request->descriptor.getrf->lda);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_TRMM: { /* compact TRMM kernel (packed) */
      unsigned int tsize;
      LIBXS_ASSERT(NULL != request->descriptor.trmm);
      tsize = (unsigned int)request->descriptor.trmm->typesize;
      if (4 == tsize || 8 == tsize) {
        extra.nflops = 0; /* TODO */
        LIBXS_NO_OFFLOAD(void, libxs_generator_trmm_kernel, &generated_code, request->descriptor.trmm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c%c%c%c_%ux%u_%u_%u.trmm", target_arch, tsizename,
            request->descriptor.trmm->transa, request->descriptor.trmm->layout, request->descriptor.trmm->side, request->descriptor.trmm->uplo,
            request->descriptor.trmm->m, request->descriptor.trmm->n, request->descriptor.trmm->lda, request->descriptor.trmm->ldb); /* TODO: alpha */
        }
      }
    } break;
    case LIBXS_BUILD_KIND_TRSM: if (NULL != request->descriptor.trsm) { /* compact TRSM kernel (packed) */
      const unsigned int tsize = (unsigned int)request->descriptor.trsm->typesize;
      if (4 == tsize || 8 == tsize) {
        extra.nflops = 0; /* TODO */
        LIBXS_NO_OFFLOAD(void, libxs_generator_trsm_kernel, &generated_code, request->descriptor.trsm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c%c%c%c_%ux%u_%u_%u.trsm", target_arch, tsizename,
            request->descriptor.trsm->transa, request->descriptor.trsm->layout, request->descriptor.trsm->side, request->descriptor.trsm->uplo,
            request->descriptor.trsm->m, request->descriptor.trsm->n, request->descriptor.trsm->lda, request->descriptor.trsm->ldb); /* TODO: alpha */
        }
      }
    } break;
    case LIBXS_BUILD_KIND_USER: break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid build request discovered!\n");
      }
      /*result = EXIT_FAILURE;*/
    }
# endif
  }

  if  (0 == generated_code.last_error /* no error raised */
    && 0 != generated_code.code_size /*check (tcopy issue?)*/)
  {
    char* code_buffer = NULL;
    void* code_buffer_result = &code_buffer;
    LIBXS_ASSERT(generated_code.code_size <= LIBXS_CODE_MAXSIZE);
    LIBXS_ASSERT(NULL != generated_code.generated_code);
    /* attempt to create executable buffer */
    result = libxs_xmalloc((void**)code_buffer_result, generated_code.code_size, 0/*auto*/,
      /* flag must be a superset of what's populated by libxs_malloc_attrib */
      LIBXS_MALLOC_FLAG_RWX, &extra, sizeof(extra));
    if (EXIT_SUCCESS == result) { /* check for success */
      LIBXS_ASSERT(NULL != code_buffer);
#if defined(__APPLE__) && defined(__arm64__)
      pthread_jit_write_protect_np( false );
#endif
      /* copy temporary buffer into the prepared executable buffer */
# if defined(NDEBUG)
      { int i; /* precondition: jit_buffer == generated_code.generated_code */
        for (i = 0; i < (int)generated_code.code_size; ++i) code_buffer[i] = jit_buffer[i];
      }
# else
      memcpy(code_buffer, generated_code.generated_code, generated_code.code_size);
# endif
#if defined(__APPLE__) && defined(__arm64__)
      pthread_jit_write_protect_np( true );
      sys_icache_invalidate( code_buffer, generated_code.code_size );
#else
      /* attribute/protect buffer and revoke unnecessary flags */
      result = libxs_malloc_attrib((void**)code_buffer_result, LIBXS_MALLOC_FLAG_X, jit_name);
      if (EXIT_SUCCESS == result) { /* check for success */
        code->ptr = code_buffer; /* commit buffer */
        LIBXS_ASSERT(NULL != code->ptr && 0 == (LIBXS_CODE_STATIC & code->uval));
#if defined(__aarch64__)
        __builtin___clear_cache( code_buffer, code_buffer+generated_code.code_size );
#endif
      }
      else { /* release buffer */
        libxs_xfree(code_buffer, 0/*no check*/);
      }
#endif
    }
  }
  else if (request->kind == LIBXS_BUILD_KIND_USER && NULL != request->descriptor.ptr) { /* user-data */
    if (0 != request->user_size) {
      void* user_data = &code->ptr;
      result = libxs_xmalloc((void**)user_data, request->user_size, 0/*auto*/,
        LIBXS_MALLOC_FLAG_PRIVATE, &extra, sizeof(extra));
    }
    else {
      result = EXIT_SUCCESS;
      code->ptr = NULL;
    }
  }
  else {
    result = (0 != generated_code.last_error ? generated_code.last_error : EXIT_FAILURE);
  }
#else /* unsupported platform */
  LIBXS_UNUSED(request); LIBXS_UNUSED(regindex); LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXS_X86_GENERIC <= libxs_target_archid) result = EXIT_FAILURE;
#endif
  return result;
}


LIBXS_API_INLINE void internal_pad_descriptor(libxs_descriptor* desc, signed char size)
{
  LIBXS_ASSERT(LIBXS_DESCRIPTOR_MAXSIZE < 128 && NULL != desc);
  LIBXS_ASSERT(LIBXS_DIFF_SIZE <= LIBXS_DESCRIPTOR_MAXSIZE);
  LIBXS_ASSERT(LIBXS_HASH_SIZE <= LIBXS_DIFF_SIZE);
  for (; size < LIBXS_DIFF_SIZE; ++size) desc->data[size] = 0;
}


LIBXS_API_INLINE libxs_code_pointer internal_find_code(libxs_descriptor* desc, size_t desc_size, size_t user_size, unsigned int* hash)
{
  libxs_code_pointer flux_entry = { 0 };
  const int is_big_desc = LIBXS_DESCRIPTOR_ISBIG(desc->kind);
  const signed char size = (signed char)(sizeof(libxs_descriptor_kind) + desc_size);
  LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
  int build = EXIT_SUCCESS;
#endif
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# if defined(LIBXS_NTHREADS_USE)
  const unsigned int tid = libxs_get_tid();
  internal_cache_type *const cache = internal_cache_buffer + tid;
# else
  static LIBXS_TLS internal_cache_type internal_cache_buffer;
  internal_cache_type *const cache = &internal_cache_buffer;
# endif
  unsigned char cache_index;
  internal_pad_descriptor(desc, size);
  LIBXS_ASSERT(NULL != hash);
  if (0 == is_big_desc) {
    LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
    LIBXS_DIFF_N(unsigned char, cache_index, LIBXS_DIFF(LIBXS_DIFF_SIZE), xdesc, cache->entry.keys,
      LIBXS_DIFF_SIZE, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  else {
    cache_index = (unsigned char)libxs_diff_n(desc, cache->entry.keys,
      size, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  if (cache->entry.id == libxs_ninit && cache_index < cache->entry.size) { /* valid hit */
    flux_entry = cache->entry.code[cache_index];
    cache->entry.hit = cache_index;
  }
  else
#else
  internal_pad_descriptor(desc, size);
  LIBXS_ASSERT(NULL != hash);
#endif
  {
    unsigned int i, i0, mode = 0, diff = 1;
    *hash = LIBXS_CRC32(LIBXS_HASH_SIZE)(LIBXS_HASH_SEED, desc);
    i0 = i = LIBXS_MOD2(*hash, LIBXS_CAPACITY_REGISTRY);
    LIBXS_ASSERT(NULL != internal_registry);
    LIBXS_ASSERT(&desc->kind == &desc->gemm.pad && desc->kind == desc->gemm.pad);
    do { /* use calculated location and check if the requested code is already JITted */
#if (1 < INTERNAL_REGLOCK_MAXN) || !LIBXS_LOCK_TYPE_ISRW(LIBXS_REGLOCK) /* read registered code */
# if 1 /* omitting an atomic load is safe but avoids race-detectors to highlight this location */
      uintptr_t *const fluxaddr = &internal_registry[i].uval;
      flux_entry.uval = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(fluxaddr, LIBXS_ATOMIC_RELAXED);
# else
      flux_entry = internal_registry[i];
# endif
#else
      LIBXS_LOCK_ACQREAD(LIBXS_REGLOCK, internal_reglock_ptr);
      flux_entry = internal_registry[i]; /* read registered code */
      LIBXS_LOCK_RELREAD(LIBXS_REGLOCK, internal_reglock_ptr);
#endif
      if ((NULL != flux_entry.ptr_const || 1 == mode) && 2 > mode) { /* check existing entry further */
        if (NULL != flux_entry.ptr_const) {
          if (0 == is_big_desc) {
#if !defined(LIBXS_CACHE_MAXSIZE) || (0 == (LIBXS_CACHE_MAXSIZE))
            LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
#endif
            diff = LIBXS_DIFF(LIBXS_DIFF_SIZE)(xdesc, internal_registry_keys + i, 0/*dummy*/);
          }
          else {
            diff = libxs_diff(desc, internal_registry_keys + i, size);
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
          i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXS_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
            if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 0, 1/*collision*/, 0, 0);
            }
          }
          LIBXS_ASSERT(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        LIBXS_ASSERT(0 == mode || 1 < mode);
#if (0 == LIBXS_JIT)
        LIBXS_UNUSED(user_size);
#else
        if (LIBXS_X86_AVX <= libxs_target_archid || /* check if JIT is supported (CPUID) */
           (LIBXS_X86_GENERIC <= libxs_target_archid && LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) ||
           (LIBXS_KERNEL_KIND_USER == LIBXS_DESCRIPTOR_KIND(desc->kind)))
        {
          LIBXS_ASSERT(0 != mode || NULL == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.ptr); /* lock the registry entry */
          if (NULL == internal_registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxs_build_request request; /* setup the code build request */
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
              LIBXS_ASSIGN127(internal_registry_keys + i, desc);
# if (1 < INTERNAL_REGLOCK_MAXN)
              LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_registry[i].ptr, flux_entry.ptr, LIBXS_ATOMIC_SEQ_CST);
# else
              internal_registry[i] = flux_entry;
# endif
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer fix_entry;
#   if (1 < INTERNAL_REGLOCK_MAXN)
                fix_entry.ptr = LIBXS_ATOMIC_LOAD(&internal_registry[i0].ptr, LIBXS_ATOMIC_RELAXED);
#   else
                fix_entry = internal_registry[i0];
#   endif
                LIBXS_ASSERT(NULL != fix_entry.ptr_const);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
#   if (1 < INTERNAL_REGLOCK_MAXN)
                  LIBXS_ATOMIC_STORE(&internal_registry[i0].ptr, fix_entry.ptr, LIBXS_ATOMIC_RELAXED);
#   else
                  internal_registry[i0] = fix_entry;
#   endif
                }
              }
# endif
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
              i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
              if (NULL == internal_registry[i].ptr_const) break;
            } while (i != i0);
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* do not use break if inside of locked region */
            }
            flux_entry.ptr = NULL; /* no result */
          }
        }
        else /* JIT-code generation not available */
#endif
        { /* leave the dispatch loop */
          if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
            internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
          }
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
          build = EXIT_FAILURE;
#endif
          flux_entry.ptr = NULL;
          diff = 0;
        }
      }
    } while (0 != diff);
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    if (NULL != flux_entry.ptr_const) { /* keep code version on record (cache) */
      LIBXS_ASSERT(0 == diff);
      if (cache->entry.id == libxs_ninit) { /* maintain cache */
        if (cache->entry.size < internal_cache_size) { /* grow */
          INTERNAL_FIND_CODE_CACHE_GROW(cache_index, cache->entry.size);
          LIBXS_ASSERT(cache->entry.size <= internal_cache_size);
        }
        else { /* evict */
          LIBXS_ASSERT(cache->entry.hit < cache->entry.size);
          INTERNAL_FIND_CODE_CACHE_EVICT(cache_index, cache->entry.size, cache->entry.hit);
        }
      }
      else if (0 != internal_cache_size) { /* reset cache */
# if !defined(NDEBUG)
        LIBXS_MEMZERO127(cache->entry.keys);
# endif
        cache->entry.id = libxs_ninit;
        cache->entry.size = 1;
        cache_index = 0;
      }
      LIBXS_MEMCPY127(cache->entry.keys + cache_index, desc, 0 == is_big_desc ? LIBXS_DIFF_SIZE : size);
      cache->entry.code[cache_index] = flux_entry;
      cache->entry.hit = cache_index;
    }
#endif
  }
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
#if (0 != LIBXS_JIT)
  assert( /*!LIBXS_ASSERT*/
    LIBXS_KERNEL_KIND_MATMUL != LIBXS_DESCRIPTOR_KIND(desc->kind)
    || NULL != flux_entry.ptr_const
    || 1 == internal_reglock_count
    || EXIT_SUCCESS != build);
#endif
  return flux_entry;
}


LIBXS_API_INTERN const libxs_kernel_xinfo* libxs_get_kernel_xinfo(libxs_code_pointer code,
  const libxs_descriptor** desc, size_t* code_size)
{
  libxs_kernel_xinfo* result = NULL;
  void *const result_address = &result;
  int flags = LIBXS_MALLOC_FLAG_X;
  if (NULL != code.ptr_const && EXIT_SUCCESS == libxs_get_malloc_xinfo(
    code.ptr_const, code_size, &flags, (void**)result_address) && NULL != result)
  {
    if (NULL != desc) {
      if (NULL != internal_registry && NULL != internal_registry_keys && result->registered < (LIBXS_CAPACITY_REGISTRY)
#if defined(LIBXS_HASH_COLLISION)
        && code.uval == (~LIBXS_HASH_COLLISION & internal_registry[result->registered].uval)
#else
        && code.ptr_const == internal_registry[result->registered].ptr_const
#endif
        && LIBXS_KERNEL_UNREGISTERED > LIBXS_DESCRIPTOR_KIND(internal_registry_keys[result->registered].entry.kind))
      {
        *desc = &internal_registry_keys[result->registered].entry;
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


LIBXS_API int libxs_get_kernel_info(const void* kernel, libxs_kernel_info* info)
{
  int result;
  const libxs_kernel_xinfo* xinfo;
  libxs_kernel_info result_info;
  const libxs_descriptor* desc;
  libxs_code_pointer code;
  code.ptr_const = kernel;
  LIBXS_MEMZERO127(&result_info);
  xinfo = libxs_get_kernel_xinfo(code, &desc, &result_info.code_size);
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


LIBXS_API int libxs_get_mmkernel_info(libxs_xmmfunction kernel, libxs_mmkernel_info* info)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xgemm = kernel;
  if (NULL != info) {
    const libxs_descriptor* desc;
    if (NULL != libxs_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind))
    {
      info->iprecision = (libxs_gemm_precision)LIBXS_GETENUM_INP(desc->gemm.desc.datatype);
      info->oprecision = (libxs_gemm_precision)LIBXS_GETENUM_OUT(desc->gemm.desc.datatype);
      info->prefetch = (libxs_gemm_prefetch_type)desc->gemm.desc.prefetch;
      info->flags = desc->gemm.desc.flags;
      info->lda = desc->gemm.desc.lda;
      info->ldb = desc->gemm.desc.ldb;
      info->ldc = desc->gemm.desc.ldc;
      info->m = desc->gemm.desc.m;
      info->n = desc->gemm.desc.n;
      info->k = desc->gemm.desc.k;
      result = EXIT_SUCCESS;
    }
    else {
      if ( 0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        if (NULL == code.ptr_const) {
          fprintf(stderr, "LIBXS ERROR: NULL-kernel cannot be inspected!\n");
        }
        else {
          fprintf(stderr, "LIBXS ERROR: invalid kernel cannot be inspected!\n");
        }
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_transkernel_info(libxs_xtransfunction kernel, libxs_transkernel_info* info)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xtrans = kernel;
  if (NULL != info) {
    const libxs_descriptor* desc;
    if (NULL != libxs_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXS_KERNEL_KIND_TRANS == LIBXS_DESCRIPTOR_KIND(desc->kind))
    {
      info->typesize = desc->trans.desc.typesize;
      info->ldo = desc->trans.desc.ldo;
      info->m = desc->trans.desc.m;
      info->n = desc->trans.desc.n;
      result = EXIT_SUCCESS;
    }
    else {
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: invalid kernel cannot be inspected!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_mcopykernel_info(libxs_xmcopyfunction kernel, libxs_mcopykernel_info* info)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xmatcopy = kernel;
  if (NULL != info) {
    const libxs_descriptor* desc;
    if (NULL != libxs_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXS_KERNEL_KIND_MCOPY == LIBXS_DESCRIPTOR_KIND(desc->kind))
    {
      info->typesize = desc->mcopy.desc.typesize;
      info->prefetch = desc->mcopy.desc.prefetch;
      info->flags = desc->mcopy.desc.flags;
      info->ldi = desc->mcopy.desc.ldi;
      info->ldo = desc->mcopy.desc.ldo;
      info->m = desc->mcopy.desc.m;
      info->n = desc->mcopy.desc.n;
      result = EXIT_SUCCESS;
    }
    else {
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: invalid kernel cannot be inspected!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_meltwkernel_info(libxs_xmeltwfunction kernel, libxs_meltwkernel_info* info)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xmateltw = kernel;
  if (NULL != info) {
    const libxs_descriptor* desc;
    if (NULL != libxs_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXS_KERNEL_KIND_MELTW == LIBXS_DESCRIPTOR_KIND(desc->kind))
    {
      info->datatype = desc->meltw.desc.datatype;
      info->operation = desc->meltw.desc.operation;
      info->flags = desc->meltw.desc.flags;
      info->ldi = desc->meltw.desc.ldi;
      info->ldo = desc->meltw.desc.ldo;
      info->m = desc->meltw.desc.m;
      info->n = desc->meltw.desc.n;
      result = EXIT_SUCCESS;
    }
    else {
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: invalid kernel cannot be inspected!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_registry_info(libxs_registry_info* info)
{
  int result = EXIT_SUCCESS;
  LIBXS_INIT /* verbosity */
  if (0 != info && 0 != internal_registry) {
    size_t i;
    LIBXS_MEMZERO127(info); /* info->nstatic = 0; info->size = 0; */
    info->nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(libxs_descriptor));
    info->capacity = LIBXS_CAPACITY_REGISTRY;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    info->ncache = internal_cache_size;
#else
    info->ncache = 0;
#endif
    for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
      libxs_code_pointer code = internal_registry[i];
      if (0 != code.ptr_const && EXIT_SUCCESS == result) {
        if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
          size_t buffer_size = 0;
          void* buffer = 0;
#if defined(LIBXS_HASH_COLLISION)
          code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
          result = libxs_get_malloc_xinfo(code.ptr_const, &buffer_size, NULL/*flags*/, &buffer);
          if (EXIT_SUCCESS == result) {
            info->nbytes += LIBXS_UP2(buffer_size + (((char*)code.ptr_const) - (char*)buffer), LIBXS_PAGE_MINSIZE);
          }
        }
        else {
          ++info->nstatic;
        }
        ++info->size;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void* libxs_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init, unsigned int* key_hash)
{
  static int error_once = 0;
  void* result;
  LIBXS_INIT /* verbosity */
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= key_size) {
    libxs_descriptor wrap;
    unsigned int hash = 0;
    void* dst;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, key_size);
#endif
    LIBXS_MEMCPY127(wrap.user.desc, key, key_size);
    wrap.kind = (libxs_descriptor_kind)(LIBXS_DESCRIPTOR_SIGSIZE >= key_size
      ? ((libxs_descriptor_kind)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    dst = internal_find_code(&wrap, key_size, value_size, &hash).ptr;
    if (NULL != key_hash) *key_hash = hash;
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
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
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
        fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xregister specified!\n");
      }
      else {
        fprintf(stderr, "LIBXS ERROR: libxs_xregister has maximum key-size of %i Byte!\n",
          LIBXS_DESCRIPTOR_MAXSIZE);
      }
    }
    result = NULL;
  }
  return result;
}


LIBXS_API void* libxs_xdispatch(const void* key, size_t key_size, unsigned int* key_hash)
{
  void* result;
  LIBXS_INIT /* verbosity */
#if !defined(NDEBUG)
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= key_size)
#endif
  {
    unsigned int hash = 0;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, key_size);
#endif
    LIBXS_MEMCPY127(wrap.user.desc, key, key_size);
    wrap.kind = (libxs_descriptor_kind)(LIBXS_DESCRIPTOR_SIGSIZE >= key_size
      ? ((libxs_descriptor_kind)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    result = internal_find_code(&wrap, key_size, 0/*user_size*/, &hash).ptr;
    if (NULL != key_hash) *key_hash = hash;
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdispatch specified!\n");
    }
    result = NULL;
  }
#endif
  return result;
}


LIBXS_API void libxs_xrelease(const void* key, size_t key_size)
{
  libxs_release_kernel(libxs_xdispatch(key, key_size, NULL/*key_hash*/));
}


LIBXS_API libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  libxs_xmmfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    const int batch_reduce =
      LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS |
      LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET |
      LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.gemm.desc, descriptor);
    wrap.kind = (libxs_descriptor_kind)(0 == (batch_reduce & descriptor->flags)
      ? ((libxs_descriptor_kind)LIBXS_KERNEL_KIND_MATMUL)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_MATMUL));
    if (0 != (0x80 & descriptor->prefetch)) { /* "sign"-bit of byte-value is set */
      wrap.gemm.desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
    }
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xgemm;
#if defined(_DEBUG)
    if (LIBXS_VERBOSITY_HIGH <= libxs_verbosity && INT_MAX != libxs_verbosity && NULL != result.xmm) {
      LIBXS_STDIO_ACQUIRE();
      fprintf(stderr, "\nLIBXS: ");
      libxs_gemm_xprint(stderr, result, NULL/*a*/, NULL/*b*/, NULL/*c*/);
      LIBXS_STDIO_RELEASE();
    }
#endif
  }
  else { /* quietly accept NULL-descriptor */
    result.xmm = NULL;
  }
  return result;
}


LIBXS_API libxs_dmmfunction libxs_dmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.dmm;
}


LIBXS_API libxs_smmfunction libxs_smmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.smm;
}


LIBXS_API libxs_bsmmfunction libxs_bsmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bsmm;
}


LIBXS_API libxs_bmmfunction libxs_bmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bmm;
}


LIBXS_API libxs_wimmfunction libxs_wimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.wimm;
}


LIBXS_API libxs_ssbimmfunction libxs_ssbimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.ssbimm;
}


LIBXS_API libxs_usbimmfunction libxs_usbimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.usbimm;
}


LIBXS_API libxs_subimmfunction libxs_subimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.subimm;
}


LIBXS_API libxs_uubimmfunction libxs_uubimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.uubimm;
}


LIBXS_API libxs_sububmmfunction libxs_sububmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.sububmm;
}


LIBXS_API libxs_dmmfunction_reducebatch_addr libxs_dmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.dmra;
}


LIBXS_API libxs_smmfunction_reducebatch_addr libxs_smmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.smra;
}


LIBXS_API libxs_bsmmfunction_reducebatch_addr libxs_bsmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bsmra;
}


LIBXS_API libxs_bmmfunction_reducebatch_addr libxs_bmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bmra;
}


LIBXS_API libxs_wimmfunction_reducebatch_addr libxs_wimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.wimra;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_addr libxs_ssbimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.ssbimra;
}


LIBXS_API libxs_usbimmfunction_reducebatch_addr libxs_usbimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.usbimra;
}


LIBXS_API libxs_subimmfunction_reducebatch_addr libxs_subimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.subimra;
}


LIBXS_API libxs_uubimmfunction_reducebatch_addr libxs_uubimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.uubimra;
}


LIBXS_API libxs_sububmmfunction_reducebatch_addr libxs_sububmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.sububmra;
}


LIBXS_API libxs_dmmfunction_reducebatch_addr libxs_dmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.dmra;
}


LIBXS_API libxs_smmfunction_reducebatch_addr libxs_smmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.smra;
}


LIBXS_API libxs_bsmmfunction_reducebatch_addr libxs_bsmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.bsmra;
}


LIBXS_API libxs_bmmfunction_reducebatch_addr libxs_bmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.bmra;
}


LIBXS_API libxs_wimmfunction_reducebatch_addr libxs_wimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.wimra;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_addr libxs_ssbimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.ssbimra;
}


LIBXS_API libxs_usbimmfunction_reducebatch_addr libxs_usbimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.usbimra;
}


LIBXS_API libxs_subimmfunction_reducebatch_addr libxs_subimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.subimra;
}


LIBXS_API libxs_uubimmfunction_reducebatch_addr libxs_uubimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.uubimra;
}


LIBXS_API libxs_sububmmfunction_reducebatch_addr libxs_sububmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.sububmra;
}


LIBXS_API libxs_dmmfunction_reducebatch_offs libxs_dmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.dmro;
}


LIBXS_API libxs_smmfunction_reducebatch_offs libxs_smmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.smro;
}


LIBXS_API libxs_bsmmfunction_reducebatch_offs libxs_bsmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bsmro;
}


LIBXS_API libxs_bmmfunction_reducebatch_offs libxs_bmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bmro;
}


LIBXS_API libxs_wimmfunction_reducebatch_offs libxs_wimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.wimro;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_offs libxs_ssbimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.ssbimro;
}


LIBXS_API libxs_usbimmfunction_reducebatch_offs libxs_usbimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.usbimro;
}


LIBXS_API libxs_subimmfunction_reducebatch_offs libxs_subimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.subimro;
}


LIBXS_API libxs_uubimmfunction_reducebatch_offs libxs_uubimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.uubimro;
}


LIBXS_API libxs_sububmmfunction_reducebatch_offs libxs_sububmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.sububmro;
}


LIBXS_API libxs_dmmfunction_reducebatch_offs libxs_dmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.dmro;
}


LIBXS_API libxs_smmfunction_reducebatch_offs libxs_smmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.smro;
}


LIBXS_API libxs_bsmmfunction_reducebatch_offs libxs_bsmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.bsmro;
}


LIBXS_API libxs_bmmfunction_reducebatch_offs libxs_bmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.bmro;
}


LIBXS_API libxs_wimmfunction_reducebatch_offs libxs_wimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.wimro;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_offs libxs_ssbimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.ssbimro;
}


LIBXS_API libxs_usbimmfunction_reducebatch_offs libxs_usbimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.usbimro;
}


LIBXS_API libxs_subimmfunction_reducebatch_offs libxs_subimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.subimro;
}


LIBXS_API libxs_uubimmfunction_reducebatch_offs libxs_uubimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.uubimro;
}


LIBXS_API libxs_sububmmfunction_reducebatch_offs libxs_sububmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  result = libxs_xmmdispatch(desc);
  return result.sububmro;
}


LIBXS_API libxs_dmmfunction_reducebatch_strd libxs_dmmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.dmrs;
}


LIBXS_API libxs_smmfunction_reducebatch_strd libxs_smmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.smrs;
}


LIBXS_API libxs_bsmmfunction_reducebatch_strd libxs_bsmmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.bsmrs;
}


LIBXS_API libxs_bmmfunction_reducebatch_strd libxs_bmmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.bmrs;
}


LIBXS_API libxs_wimmfunction_reducebatch_strd libxs_wimmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.wimrs;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_strd libxs_ssbimmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.ssbimrs;
}


LIBXS_API libxs_usbimmfunction_reducebatch_strd libxs_usbimmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.usbimrs;
}


LIBXS_API libxs_subimmfunction_reducebatch_strd libxs_subimmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.subimrs;
}


LIBXS_API libxs_uubimmfunction_reducebatch_strd libxs_uubimmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.uubimrs;
}


LIBXS_API libxs_sububmmfunction_reducebatch_strd libxs_sububmmdispatch_reducebatch_strd(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE,
    libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.sububmrs;
}


LIBXS_API libxs_dmmfunction_reducebatch_strd libxs_dmmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.dmrs;
}


LIBXS_API libxs_smmfunction_reducebatch_strd libxs_smmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.smrs;
}


LIBXS_API libxs_bsmmfunction_reducebatch_strd libxs_bsmmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.bsmrs;
}


LIBXS_API libxs_bmmfunction_reducebatch_strd libxs_bmmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.bmrs;
}


LIBXS_API libxs_wimmfunction_reducebatch_strd libxs_wimmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.wimrs;
}


LIBXS_API libxs_ssbimmfunction_reducebatch_strd libxs_ssbimmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.ssbimrs;
}


LIBXS_API libxs_usbimmfunction_reducebatch_strd libxs_usbimmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_A_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.usbimrs;
}


LIBXS_API libxs_subimmfunction_reducebatch_strd libxs_subimmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.subimrs;
}


LIBXS_API libxs_uubimmfunction_reducebatch_strd libxs_uubimmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_AB_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.uubimrs;
}


LIBXS_API libxs_sububmmfunction_reducebatch_strd libxs_sububmmdispatch_reducebatch_strd_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_B_UNSIGNED | LIBXS_GEMM_FLAG_C_UNSIGNED | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE,
    libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxs_xmmdispatch(desc);
  return result.sububmrs;
}


/* GEMMs fused with eltwise kernels */
LIBXS_API libxs_bmmfunction_reducebatch_strd_meltwfused libxs_bmmdispatch_reducebatch_strd_meltwfused(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxs_xmmdispatch(desc);
  return result.bmrs_meltwfused;
}


LIBXS_API libxs_bmmfunction_reducebatch_strd_meltwfused libxs_bmmdispatch_reducebatch_strd_meltwfused_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxs_xmmdispatch(desc);
  return result.bmrs_meltwfused;
}


LIBXS_API libxs_bsmmfunction_reducebatch_strd_meltwfused libxs_bsmmdispatch_reducebatch_strd_meltwfused(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxs_xmmdispatch(desc);
  return result.bsmrs_meltwfused;
}


LIBXS_API libxs_bsmmfunction_reducebatch_strd_meltwfused libxs_bsmmdispatch_reducebatch_strd_meltwfused_unroll(
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXS_FLAGS | LIBXS_GEMM_FLAG_VNNI_A) : *flags);
  libxs_descriptor_blob blob;
  /*const*/ libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(unroll_hint < 127 ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxs_xmmdispatch(desc);
  return result.bsmrs_meltwfused;
}


LIBXS_API libxs_xmcopyfunction libxs_dispatch_mcopy(const libxs_mcopy_descriptor* descriptor)
{
  libxs_xmcopyfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.mcopy.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_MCOPY;
#if (defined(_WIN32) || defined(__CYGWIN__))
    wrap.mcopy.desc.prefetch = 0;
#endif
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xmatcopy;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_xmeltwfunction libxs_dispatch_meltw(const libxs_meltw_descriptor* descriptor)
{
  libxs_xmeltwfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.meltw.desc, descriptor);
    wrap.kind = LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_MELTW);
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xmateltw;
  }
  else {
    result.xmeltw = NULL;
  }
  return result;
}


LIBXS_API libxs_meltwfunction_copy libxs_dispatch_meltw_copy(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_copy_flags flags)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, 0, LIBXS_MELTW_OPERATION_COPY);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_copy;
}


LIBXS_API libxs_meltwfunction_zero libxs_dispatch_meltw_zero(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    0, 0, LIBXS_MELTW_OPERATION_ZERO);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_zero;
}


LIBXS_API libxs_meltwfunction_add libxs_dispatch_meltw_add(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    0, 0, LIBXS_MELTW_OPERATION_ADD);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_add;
}


LIBXS_API libxs_meltwfunction_mul libxs_dispatch_meltw_mul(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    0, 0, LIBXS_MELTW_OPERATION_MUL);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_mul;
}


LIBXS_API libxs_meltwfunction_relu libxs_dispatch_meltw_relu(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_relu_flags flags, unsigned char param)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, param, LIBXS_MELTW_OPERATION_RELU);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_relu;
}


LIBXS_API libxs_meltwfunction_cvtfp32bf16 libxs_dispatch_meltw_cvtfp32bf16(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_cvt_flags flags)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, 0, LIBXS_MELTW_OPERATION_CVTFP32BF16);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_cvtfp32bf16;
}


LIBXS_API libxs_meltwfunction_cvtfp32bf16_act libxs_dispatch_meltw_cvtfp32bf16_act(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_cvta_flags flags, unsigned char param)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, param, LIBXS_MELTW_OPERATION_CVTFP32BF16_ACT);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_cvtfp32bf16_act;
}


LIBXS_API libxs_meltwfunction_act_cvtfp32bf16 libxs_dispatch_meltw_act_cvtfp32bf16(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_acvt_flags flags, unsigned char param)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, param, LIBXS_MELTW_OPERATION_ACT_CVTFP32BF16);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_act_cvtfp32bf16;
}


LIBXS_API libxs_meltwfunction_reduce libxs_dispatch_meltw_reduce(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_redu_flags flags, unsigned char param)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, param, LIBXS_MELTW_OPERATION_REDUCE);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_reduce;
}


LIBXS_API libxs_meltwfunction_reduce_cols_idx libxs_dispatch_meltw_reduce_cols_idx(
  libxs_blasint m, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_datatype idx_type)
{
  libxs_descriptor_blob blob;
  libxs_blasint idx_dtype_size = libxs_typesize(idx_type);
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, idx_dtype_size, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    0, 0, LIBXS_MELTW_OPERATION_REDUCE_COLS_IDX);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_reduce_cols_idx;
}


LIBXS_API libxs_meltwfunction_opreduce_vecs_idx libxs_dispatch_meltw_opreduce_vecs_idx(
  libxs_blasint m, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_datatype idx_type, libxs_meltw_opreduce_vecs_flags flags)
{
  libxs_descriptor_blob blob;
  libxs_blasint idx_dtype_size = libxs_typesize(idx_type);
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, idx_dtype_size, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, 0, LIBXS_MELTW_OPERATION_OPREDUCE_VECS_IDX);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_opreduce_vecs_idx;
}


LIBXS_API libxs_meltwfunction_scale libxs_dispatch_meltw_scale(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_scal_flags flags, unsigned char param)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, param, LIBXS_MELTW_OPERATION_SCALE);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_scale;
}


LIBXS_API libxs_meltwfunction_transform libxs_dispatch_meltw_transform(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_transform_flags flags)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, 0, LIBXS_MELTW_OPERATION_TRANSFORM);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_transform;
}


LIBXS_API libxs_meltwfunction_dropout libxs_dispatch_meltw_dropout(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype out_type, libxs_meltw_dropout_flags flags)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init(&blob,
    in_type, out_type, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, 0, LIBXS_MELTW_OPERATION_DROPOUT);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_dropout;
}


LIBXS_API libxs_meltwfunction_unary libxs_dispatch_meltw_unary(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype compute_type, libxs_datatype out_type, libxs_meltw_unary_flags flags, libxs_meltw_unary_type type)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init2(&blob,
    in_type, compute_type, out_type, LIBXS_DATATYPE_UNSUPPORTED, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo, 0, 0,
    (unsigned short)flags, (unsigned char)type, LIBXS_MELTW_OPERATION_UNARY);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_unary;
}


LIBXS_API libxs_meltwfunction_binary libxs_dispatch_meltw_binary(
  libxs_blasint m, libxs_blasint n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  libxs_datatype in_type, libxs_datatype compute_type, libxs_datatype out_type, libxs_meltw_binary_flags flags, libxs_meltw_binary_type type)
{
  libxs_descriptor_blob blob;
  const libxs_meltw_descriptor *const desc = libxs_meltw_descriptor_init2(&blob,
    in_type, compute_type, out_type, LIBXS_DATATYPE_UNSUPPORTED, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo, 0, 0,
    (unsigned short)flags, (unsigned char)type, LIBXS_MELTW_OPERATION_BINARY);

  libxs_xmeltwfunction result = libxs_dispatch_meltw(desc);

  return result.meltw_binary;
}


LIBXS_API libxs_matrix_eqn_function libxs_dispatch_matrix_eqn_desc( const libxs_meqn_descriptor* descriptor ) {
  libxs_matrix_eqn_function result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;

    /* check if equation is ready for JIT */
    if ( libxs_matrix_eqn_is_ready_for_jit( descriptor->eqn_idx) == 0 ) {
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
      LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
      LIBXS_ASSIGN127(&wrap.meqn.desc, descriptor);
      wrap.kind = LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_MEQN);
      result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xmateqn;
    } else {
      result = NULL;
    }
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_matrix_eqn_function libxs_dispatch_matrix_eqn( const libxs_blasint m, const libxs_blasint n, const libxs_blasint* ldo, const libxs_datatype out_type, const unsigned int eqn_idx ) {
  libxs_descriptor_blob blob;
  const libxs_meqn_descriptor *const desc = libxs_meqn_descriptor_init(&blob,
    out_type, m, n, (ldo == NULL) ? m : *ldo, eqn_idx );

  return libxs_dispatch_matrix_eqn_desc( desc );
}


LIBXS_API libxs_xtransfunction libxs_dispatch_trans(const libxs_trans_descriptor* descriptor)
{
  libxs_xtransfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.trans.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_TRANS;
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xtrans;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_pgemm_xfunction libxs_dispatch_pgemm(const libxs_pgemm_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.pgemm.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_PGEMM;
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xpgemm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_getrf_xfunction libxs_dispatch_getrf(const libxs_getrf_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.getrf.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_GETRF;
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xgetrf;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_trmm_xfunction libxs_dispatch_trmm(const libxs_trmm_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.trmm.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_TRMM;
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xtrmm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_trsm_xfunction libxs_dispatch_trsm(const libxs_trsm_descriptor* descriptor)
{
  libxs_trsm_xfunction result;
  LIBXS_INIT /* verbosity */
#if !defined(LIBXS_UNPACKED) /* CCE/Classic */
  LIBXS_ASSERT((sizeof(*descriptor) + sizeof(libxs_descriptor_kind)) <= (LIBXS_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxs_descriptor wrap;
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
    LIBXS_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXS_ASSIGN127(&wrap.trsm.desc, descriptor);
    wrap.kind = LIBXS_KERNEL_KIND_TRSM;
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xtrsm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_xmmfunction libxs_create_packed_spxgemm_csr(const libxs_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxs_pspgemm_csr_descriptor pspgemm_csr;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pspgemm_csr.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXS_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      pspgemm_csr.gemm = &desc;
    }
    pspgemm_csr.row_ptr = row_ptr;
    pspgemm_csr.column_idx = column_idx;
    pspgemm_csr.values = values;
    pspgemm_csr.packed_width = packed_width;
    request.descriptor.pspgemm_csr = &pspgemm_csr;
    request.kind = LIBXS_BUILD_KIND_PSPGEMM_CSR;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_packed_spxgemm_csc(const libxs_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor && NULL != column_ptr && NULL != row_idx && NULL != values) {
    libxs_pspgemm_csc_descriptor pspgemm_csc;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pspgemm_csc.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXS_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      pspgemm_csc.gemm = &desc;
    }
    pspgemm_csc.column_ptr = column_ptr;
    pspgemm_csc.row_idx = row_idx;
    pspgemm_csc.values = values;
    pspgemm_csc.packed_width = packed_width;
    request.descriptor.pspgemm_csc = &pspgemm_csc;
    request.kind = LIBXS_BUILD_KIND_PSPGEMM_CSC;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_packed_xgemm_ac_rm(const libxs_gemm_descriptor* descriptor, unsigned int packed_width)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor) {
    libxs_pgemm_ac_rm_descriptor pgemmacrm;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pgemmacrm.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXS_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      pgemmacrm.gemm = &desc;
    }
    pgemmacrm.packed_width = packed_width;
    request.descriptor.pgemmacrm = &pgemmacrm;
    request.kind = LIBXS_BUILD_KIND_PGEMMRMAC;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_packed_xgemm_bc_rm(const libxs_gemm_descriptor* descriptor, unsigned int packed_width)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor) {
    libxs_pgemm_bc_rm_descriptor pgemmbcrm;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pgemmbcrm.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXS_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      pgemmbcrm.gemm = &desc;
    }
    pgemmbcrm.packed_width = packed_width;
    request.descriptor.pgemmbcrm = &pgemmbcrm;
    request.kind = LIBXS_BUILD_KIND_PGEMMRMBC;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_dmmfunction libxs_create_dcsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxs_csr_reg_descriptor sreg;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      sreg.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXS_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      sreg.gemm = &desc;
    }
    sreg.row_ptr = row_ptr;
    sreg.column_idx = column_idx;
    sreg.values = values;
    request.descriptor.sreg = &sreg;
    request.kind = LIBXS_BUILD_KIND_SREG;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm.dmm;
}


LIBXS_API libxs_smmfunction libxs_create_scsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values)
{
  libxs_code_pointer result = { 0 };
  LIBXS_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxs_csr_reg_descriptor sreg;
    libxs_build_request request;
    const unsigned int n = row_ptr[descriptor->m];
    double *const d_values = (double*)(0 != n ? malloc(n * sizeof(double)) : NULL);
    if (NULL != d_values) {
      libxs_gemm_descriptor desc;
      unsigned int i;
      /* we need to copy the values into a double precision buffer */
      for (i = 0; i < n; ++i) d_values[i] = (double)values[i];
      if (0 == (0x80 & descriptor->prefetch)) {
        sreg.gemm = descriptor;
      }
      else { /* "sign"-bit of byte-value is set */
        LIBXS_ASSIGN127(&desc, descriptor);
        desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
        sreg.gemm = &desc;
      }
      sreg.row_ptr = row_ptr;
      sreg.column_idx = column_idx;
      sreg.values = d_values;
      request.descriptor.sreg = &sreg;
      request.kind = LIBXS_BUILD_KIND_SREG;
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
      free(d_values);
    }
  }
  return result.xgemm.smm;
}


LIBXS_API void libxs_release_kernel(const void* kernel)
{
  if (NULL != kernel) {
    static int error_once = 0;
    libxs_kernel_xinfo* extra = NULL;
    void *const extra_address = &extra;
    LIBXS_INIT
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(
      kernel, NULL/*size*/, NULL/*flags*/, (void**)extra_address) && NULL != extra)
    {
      const unsigned int regindex = extra->registered;
      if ((LIBXS_CAPACITY_REGISTRY) <= regindex) {
        libxs_xfree(kernel, 0/*no check*/);
      }
      else { /* attempt to unregister kernel */
        libxs_kernel_info info;
#if !defined(LIBXS_ENABLE_DEREG)
        if (EXIT_SUCCESS == libxs_get_kernel_info(kernel, &info)
          && LIBXS_KERNEL_KIND_USER == info.kind)
#endif
        {
          LIBXS_ASSERT(LIBXS_KERNEL_UNREGISTERED > info.kind);
          /* coverity[check_return] */
          LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_RELAXED); /* invalidate code cache (TLS) */
          internal_registry[regindex].ptr = NULL;
#if !defined(NDEBUG)
          memset(internal_registry_keys + regindex, 0, sizeof(*internal_registry_keys));
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
}


#if defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_init)(void);
LIBXS_API void LIBXS_FSYMBOL(libxs_init)(void)
{
  libxs_init();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_finalize)(void);
LIBXS_API void LIBXS_FSYMBOL(libxs_finalize)(void)
{
  libxs_finalize();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** /*kernel*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** kernel)
{
#if !defined(NDEBUG)
  if (NULL != kernel)
#endif
  {
    libxs_release_kernel(*kernel);
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument passed into libxs_release_kernel!\n");
    }
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch2)(intptr_t* /*fn*/, const int* /*iprec*/, const int* /*oprec*/,
  const libxs_blasint* /*m*/, const libxs_blasint* /*n*/, const libxs_blasint* /*k*/,
  const libxs_blasint* /*lda*/, const libxs_blasint* /*ldb*/, const libxs_blasint* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/, const int* /*flags*/, const int* /*prefetch*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch2)(intptr_t* fn, const int* iprec, const int* oprec,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
#if !defined(NDEBUG)
  if (NULL != fn && NULL != m
    && (NULL == iprec || (0 <= *iprec && *iprec < LIBXS_DATATYPE_UNSUPPORTED))
    && (NULL == oprec || (0 <= *oprec && *oprec < LIBXS_DATATYPE_UNSUPPORTED)))
#endif
  {
    const int gemm_flags = (NULL != flags ? *flags : LIBXS_FLAGS);
    const libxs_gemm_descriptor* descriptor;
    libxs_gemm_prefetch_type gemm_prefetch;
    libxs_descriptor_blob blob;
    libxs_code_pointer result;
#if !defined(NDEBUG)
    const libxs_gemm_precision itype = (NULL != iprec ? ((libxs_gemm_precision)*iprec) : LIBXS_GEMM_PRECISION_F64);
    const libxs_gemm_precision otype = (NULL != oprec ? ((libxs_gemm_precision)*oprec) : itype);
    const libxs_blasint kk = *(NULL != k ? k : m), nn = (NULL != n ? *n : kk);
#else
    const libxs_gemm_precision itype = (libxs_gemm_precision)*iprec, otype = (libxs_gemm_precision)*oprec;
    const libxs_blasint kk = *k, nn = *n;
#endif
    LIBXS_PRAGMA_FORCEINLINE
    gemm_prefetch = libxs_get_gemm_xprefetch(prefetch);
    LIBXS_PRAGMA_FORCEINLINE
    descriptor = libxs_gemm_descriptor_init2(&blob, itype, otype, *m, nn, kk,
        NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? *m : kk),
        NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? kk : nn),
      *(NULL != ldc ? ldc : m), alpha, beta, gemm_flags, gemm_prefetch);
#if !defined(NDEBUG)
    if (NULL != descriptor)
#endif
    {
      LIBXS_PRAGMA_FORCEINLINE
      result.xgemm = libxs_xmmdispatch(descriptor);
      *fn = result.ival;
    }
#if !defined(NDEBUG)
    else { /* quiet */
      *fn = 0;
    }
#endif
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument passed into libxs_xmmdispatch!\n");
    }
    if (NULL != fn) *fn = 0;
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* /*fn*/, const int* /*precision*/,
  const libxs_blasint* /*m*/, const libxs_blasint* /*n*/, const libxs_blasint* /*k*/,
  const libxs_blasint* /*lda*/, const libxs_blasint* /*ldb*/, const libxs_blasint* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/, const int* /*flags*/, const int* /*prefetch*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* fn, const int* precision,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
  LIBXS_FSYMBOL(libxs_xmmdispatch2)(fn, precision, precision, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_abc)(
  const libxs_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_abc)(
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != fn && NULL != a && NULL != b && NULL != c)
#endif
  {
#if !defined(NDEBUG)
    if (NULL != fn->xmm)
#endif
    {
      fn->xmm(a, b, c);
    }
#if !defined(NDEBUG)
    else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: NULL-function passed into libxs_xmmcall_abc!\n");
    }
#endif
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xmmcall_abc specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_prf)(
  const libxs_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/,
  const void* /*pa*/, const void* /*pb*/, const void* /*pc*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_prf)(
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != fn && NULL != a && NULL != b && NULL != c)
#endif
  {
#if !defined(NDEBUG)
    if (NULL != fn->xmm)
#endif
    {
      fn->xmm(a, b, c, pa, pb, pc);
    }
#if !defined(NDEBUG)
    else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: NULL-function passed into libxs_xmmcall_prf!\n");
    }
#endif
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xmmcall_prf specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall)(
  const libxs_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/,
  const void* /*pa*/, const void* /*pb*/, const void* /*pc*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall)(
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
  LIBXS_FSYMBOL(libxs_xmmcall_prf)(fn, a, b, c, pa, pb, pc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xregister)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/,
  const int* /*valsize*/, const void* /*valinit*/, int* /*keyhash*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xregister)(void** regval, const void* key, const int* keysize,
  const int* valsize, const void* valinit, int* keyhash)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize && NULL != valsize)
#endif
  {
    unsigned int hash = 0;
    *regval = libxs_xregister(key, *keysize, *valsize, valinit, &hash);
    if (NULL != keyhash) {
      *keyhash = (hash & 0x7FFFFFFF/*sign-bit*/);
    }
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xregister specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xdispatch)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/, int* /*keyhash*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xdispatch)(void** regval, const void* key, const int* keysize, int* keyhash)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize)
#endif
  {
    unsigned int hash = 0;
    *regval = libxs_xdispatch(key, *keysize, &hash);
    if (NULL != keyhash) {
      *keyhash = (hash & 0x7FFFFFFF/*sign-bit*/);
    }
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdispatch specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xrelease)(const void* /*key*/, const int* /*keysize*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xrelease)(const void* key, const int* keysize)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != key && NULL != keysize)
#endif
  {
    libxs_xrelease(key, *keysize);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xrelease specified!\n");
  }
#endif
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
