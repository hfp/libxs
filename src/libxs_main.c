/******************************************************************************
** Copyright (c) 2014-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
#include <libxs_intrinsics_x86.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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
# define LIBXS_HASH_SIZE LIBXS_DESCRIPTOR_SIGSIZE
#endif
#if !defined(LIBXS_HASH_SEED)
# define LIBXS_HASH_SEED 25071975
#endif
#if !defined(LIBXS_UNIFY_LOCKS)
# define LIBXS_UNIFY_LOCKS
#endif
#if !defined(LIBXS_ENABLE_DEREG) && 0
# define LIBXS_ENABLE_DEREG
#endif
#if !defined(LIBXS_REGLOCK_TRY) && 0
# define LIBXS_REGLOCK_TRY
#endif
#if !defined(LIBXS_DIFF_INLINE) && 0
# define LIBXS_DIFF_INLINE
#endif
#if !defined(LIBXS_DESC_INLINE) && 0
# define LIBXS_DESC_INLINE
#endif
#if !defined(LIBXS_DESC_PAD) && 1
# define LIBXS_DESC_PAD
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
LIBXS_APIVAR_ARRAY(internal_reglocktype internal_reglock, INTERNAL_REGLOCK_MAXN);
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
LIBXS_APIVAR(LIBXS_LOCK_TYPE(LIBXS_REGLOCK)* internal_reglock_ptr);
# endif
#endif

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# define INTERNAL_FIND_CODE_CACHE_GROW(RESULT_INDEX, CACHE_SIZE) \
    RESULT_INDEX = CACHE_SIZE; CACHE_SIZE = (unsigned char)(0 != (CACHE_SIZE) ? ((CACHE_SIZE) << 1) : 1)
# define INTERNAL_FIND_CODE_CACHE_EVICT(RESULT_INDEX, CACHE_SIZE, CACHE_HIT) \
    RESULT_INDEX = (unsigned char)LIBXS_MOD2((CACHE_HIT) + ((CACHE_SIZE) - 1), CACHE_SIZE)
#endif

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXS_APIVAR(int internal_reglock_count);
LIBXS_APIVAR(size_t internal_registry_nbytes);
LIBXS_APIVAR(libxs_descriptor* internal_registry_keys);
LIBXS_APIVAR(libxs_code_pointer* internal_registry);
LIBXS_APIVAR_ARRAY(internal_statistic_type internal_statistic[2/*DP/SP*/], 4/*sml/med/big/xxx*/);
LIBXS_APIVAR(unsigned int internal_statistic_sml);
LIBXS_APIVAR(unsigned int internal_statistic_med);
LIBXS_APIVAR(unsigned int internal_statistic_mnk);
LIBXS_APIVAR(unsigned int internal_statistic_num_mcopy);
LIBXS_APIVAR(unsigned int internal_statistic_num_tcopy);
LIBXS_APIVAR(unsigned int internal_statistic_num_trsm);
LIBXS_APIVAR(unsigned int internal_statistic_num_trmm);
LIBXS_APIVAR(int internal_gemm_auto_prefetch_locked);
LIBXS_APIVAR(const char* internal_build_state);

#if defined(_WIN32)
# define INTERNAL_DELIMS ";,"
LIBXS_APIVAR(HANDLE internal_singleton_handle);
#else
# define INTERNAL_DELIMS ";,:"
LIBXS_APIVAR_ARRAY(char internal_singleton_fname, 64);
LIBXS_APIVAR(int internal_singleton_handle);
#endif

#if (0 == LIBXS_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) {
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# if defined(LIBXS_REGLOCK_TRY)
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
    if (1 != internal_reglock_count) { /* (re-)try and get (meanwhile) generated code */ \
      LIBXS_ASSERT(0 != internal_registry); /* engine is not shut down */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; break; \
    }
# else
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
      LIBXS_ASSERT(0 != internal_registry); /* engine is not shut down */ \
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


LIBXS_API_INTERN unsigned int libxs_update_mmstatistic(libxs_gemm_precision precision,
  libxs_blasint m, libxs_blasint n, libxs_blasint k, unsigned int ntry, unsigned int ncol)
{
  const unsigned long long kernel_size = LIBXS_MNK_SIZE(m, n, k);
  const int idx = (LIBXS_GEMM_PRECISION_F64 == precision ? 0 : 1);
  int bucket = 3/*huge*/;

  if (LIBXS_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
    bucket = 0;
  }
  else if (LIBXS_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
    bucket = 1;
  }
  else if (LIBXS_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
    bucket = 2;
  }

  LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ncol, ncol, LIBXS_ATOMIC_RELAXED);
  return LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ntry, ntry, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API_INLINE unsigned int internal_update_mmstatistic(const libxs_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol)
{
  LIBXS_ASSERT(NULL != desc);
  return libxs_update_mmstatistic((libxs_gemm_precision)LIBXS_GETENUM_OUT(desc->datatype),
    desc->m, desc->n, desc->k, ntry, ncol);
}


LIBXS_API_INLINE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  LIBXS_ASSERT(0 != unit);
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
  LIBXS_ASSERT(0 != ostream && (0 <= precision && precision < 2));

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


LIBXS_API_INLINE unsigned int internal_statistic_ntry(int precision)
{
  return internal_statistic[precision][0/*SML*/].ntry + internal_statistic[precision][1/*MED*/].ntry
       + internal_statistic[precision][2/*BIG*/].ntry + internal_statistic[precision][3/*XXX*/].ntry;
}


LIBXS_API_INLINE void internal_register_static_code(
  libxs_gemm_precision precision, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  libxs_xmmfunction xgemm, libxs_code_pointer* registry)
{
  const libxs_blasint lda = m, ldb = k, ldc = m;
  /*const*/ int precondition = LIBXS_GEMM_NO_BYPASS_DIMS(m, n, k) && LIBXS_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc);
  if (precondition) {
    const size_t size = (LIBXS_HASH_SIZE)-sizeof(libxs_descriptor_kind);
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
      internal_update_mmstatistic(desc, 0, 1/*collision*/);
      /* out of capacity (no registry slot available) */
      LIBXS_ASSERT(NULL == dst_entry->ptr_const || i == i0);
    }
    if (NULL == dst_entry->ptr_const) { /* registry not exhausted */
      internal_registry_keys[i].kind = LIBXS_KERNEL_KIND_MATMUL;
      internal_registry_keys[i].gemm.desc = *desc;
      dst_entry->xgemm = xgemm;
      /* mark current entry as static code (non-JIT) */
      dst_entry->uval |= LIBXS_CODE_STATIC;
    }
    internal_update_mmstatistic(desc, 1/*try*/, 0);
  }
}


LIBXS_API_INLINE void internal_finalize(void)
{
  char *const env_dump_build = getenv("LIBXS_DUMP_BUILD");
  char *const env_dump_files = (NULL != getenv("LIBXS_DUMP_FILES")
    ? getenv("LIBXS_DUMP_FILES") : getenv("LIBXS_DUMP_FILE"));
  libxs_finalize();
  if (0 != libxs_verbosity) { /* print statistic on termination */
    const char *const env_target_hidden = getenv("LIBXS_TARGET_HIDDEN");
    const char *const target_arch = (NULL == env_target_hidden || 0 == atoi(env_target_hidden))
      ? libxs_cpuid_name(libxs_target_archid)
      : NULL/*hidden*/;
    /* synchronize I/O */
    LIBXS_STDIO_ACQUIRE();
#if !defined(NDEBUG) && defined(__OPTIMIZE__)
    fprintf(stderr, "LIBXS WARNING: library is optimized without -DNDEBUG and contains debug code!\n");
#endif
    fprintf(stderr, "\nLIBXS_VERSION: %s-%s (%i)", LIBXS_BRANCH, LIBXS_VERSION, LIBXS_VERSION4(
      LIBXS_VERSION_MAJOR, LIBXS_VERSION_MINOR, LIBXS_VERSION_UPDATE, LIBXS_VERSION_PATCH));
    if (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) {
      const int high_verbosity = (LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity);
      libxs_scratch_info scratch_info; size_t size_scratch = 0, size_private = 0;
      unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak && NULL != target_arch) {
        if (0 == libxs_se) {
          fprintf(stderr, "\nLIBXS_TARGET: %s\n", target_arch);
        }
        else {
          fprintf(stderr, "\nLIBXS_TARGET: %s*\n", target_arch);
        }
      }
      if (EXIT_SUCCESS == libxs_get_scratch_info(&scratch_info)) {
        size_private = scratch_info.internal;
        size_scratch = scratch_info.size;
      }
      fprintf(stderr, "Memory: %.f MB", 1.0 * (internal_registry_nbytes + size_private) / (1ULL << 20));
      if (0 != high_verbosity) {
        size_t ngemms = 0;
        int i; for (i = 0; i < 4; ++i) {
          ngemms += (size_t)internal_statistic[0/*DP*/][i].nsta + internal_statistic[1/*SP*/][i].nsta;
          ngemms += (size_t)internal_statistic[0/*DP*/][i].njit + internal_statistic[1/*SP*/][i].njit;
        }
        fprintf(stderr, " (gemm=%lu mcopy=%u tcopy=%u)\n", (unsigned long int)ngemms,
          internal_statistic_num_mcopy, internal_statistic_num_tcopy);
      }
      else {
        fprintf(stderr, "\n");
      }
      if (0 != size_scratch) {
        fprintf(stderr, "Scratch: %.f MB", 1.0 * size_scratch / (1ULL << 20));
        if (0 != high_verbosity) {
#if (0 != LIBXS_SYNC)
          if (1 < libxs_threads_count) {
            fprintf(stderr, " (mallocs=%lu, pools=%u, threads=%u)\n",
              (unsigned long int)scratch_info.nmallocs, scratch_info.npools, libxs_threads_count);
          }
          else
#endif
          {
            fprintf(stderr, " (mallocs=%lu, pools=%u)\n",
              (unsigned long int)scratch_info.nmallocs, scratch_info.npools);
          }
        }
        else {
          fprintf(stderr, "\n");
        }
      }
    }
    else {
      fprintf(stderr, "\nLIBXS_TARGET: %s\n", target_arch);
    }
    /* synchronize I/O */
    LIBXS_STDIO_RELEASE();
  }
  /* release scratch memory pool */
  libxs_release_scratch();
  /* turn-off redirected memory allocations */
  libxs_malloc_kind = 0;
  /* release global services */
  libxs_hash_finalize();
#if defined(_WIN32)
  if (NULL != internal_singleton_handle)
#else
  if (0 <= internal_singleton_handle && 0 != *internal_singleton_fname)
#endif
  { /* dump per-node info */
    if (NULL != env_dump_build || NULL != env_dump_files) {
      LIBXS_STDIO_ACQUIRE();
      if (NULL != env_dump_files && 0 != *env_dump_files) {
        const char *filename = strtok(env_dump_files, INTERNAL_DELIMS);
        for (; NULL != filename; filename = strtok(NULL, INTERNAL_DELIMS)) {
          FILE *const file = fopen(filename, "r");
          if (NULL != file) {
            int c = fgetc(file);
            fprintf(stdout, "\n\nLIBXS_DUMP_FILE: %s\n", filename);
            while (EOF != c) {
              fputc(c, stdout);
              c = fgetc(file);
            }
            fputc('\n', stdout);
            fclose(file);
          }
        }
      }
      if (NULL != env_dump_build && 0 != *env_dump_build && '0' != *env_dump_build) {
        fprintf(stdout, "\n\nBUILD_DATE=%i\n", LIBXS_CONFIG_BUILD_DATE);
        if (NULL != internal_build_state) {
          fprintf(stdout, "%s\n", internal_build_state);
        }
      }
      LIBXS_STDIO_RELEASE();
    }
    /* cleanup singleton */
#if defined(_WIN32)
    ReleaseMutex(internal_singleton_handle);
#else
    unlink(internal_singleton_fname);
    close(internal_singleton_handle);
#endif
  }
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


LIBXS_API_INLINE size_t internal_strlen(const char* cstr, size_t maxlen)
{
  size_t result = 0;
  if (NULL != cstr) {
    while (0 != cstr[result] && result < maxlen) ++result;
  }
  return result;
}


LIBXS_API_INTERN
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(no_instrument_function)
#endif
void internal_init(void);

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
    libxs_code_pointer* new_registry;
    /* setup verbosity as early as possible since below code may rely on verbose output */
    const char *const env_verbose = getenv("LIBXS_VERBOSE");
    if (NULL != env_verbose && 0 != *env_verbose) {
      libxs_verbosity = atoi(env_verbose);
    }
#if !defined(NDEBUG)
    else {
      libxs_verbosity = INT_MAX; /* quiet -> verbose */
    }
#endif
    LIBXS_ASSERT(NULL == internal_registry_keys); /* should never happen */
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
    { const char *const env = getenv("LIBXS_SCRATCH_LIMIT");
      if (NULL == env || 0 == *env) {
        /*const*/ unsigned long long limit = LIBXS_MALLOC_SCRATCH_LIMIT;
        libxs_scratch_limit = (size_t)limit;
      }
      else {
        size_t u = internal_strlen(env, 32) - 1;
        const char *const unit = "kmgKMG", *const hit = strchr(unit, env[u]);
        libxs_scratch_limit = (size_t)strtoul(env, 0, 10);
        u = (0 != hit ? ((hit - unit) % 3) : 3);
        if (u < 3) {
          libxs_scratch_limit <<= (u + 1) * 10;
        }
        /*libxs_scratch_limit_locked = 1;*/
      }
    }
    { const char *const env = getenv("LIBXS_SCRATCH_SCALE");
      if (NULL == env || 0 == *env) {
        libxs_scratch_scale = LIBXS_MALLOC_SCRATCH_SCALE;
      }
      else {
        libxs_scratch_scale = LIBXS_CLMP(atof(env), 1.1, 3.0);
        /*libxs_scratch_scale_locked = 1;*/
      }
      LIBXS_ASSERT(1 <= libxs_scratch_scale);
    }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
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
      memset(&internal_statistic[0/*DP*/][i], 0, sizeof(internal_statistic_type));
      memset(&internal_statistic[1/*SP*/][i], 0, sizeof(internal_statistic_type));
    }
    libxs_nt = 2;
#if !defined(__MIC__) && (LIBXS_X86_AVX512_MIC != LIBXS_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512_MIC == libxs_target_archid)
#endif
    {
      libxs_nt = 4;
    }
    internal_statistic_mnk = LIBXS_MAX_DIM;
    internal_statistic_sml = 13;
    internal_statistic_med = 23;
#if !defined(NDEBUG) /* LIBXS_CAPACITY_REGISTRY: power of two */
    { const unsigned int npot = LIBXS_UP2POT(LIBXS_CAPACITY_REGISTRY);
      assert(LIBXS_CAPACITY_REGISTRY == npot); /* !LIBXS_ASSERT */
    }
#endif
    libxs_hash_init(libxs_target_archid); /* used by debug memory allocation (checksum) */
    if  ((EXIT_SUCCESS == libxs_xmalloc((void**)&new_registry, (LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer), 0/*auto-align*/,
          LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/))
      && (EXIT_SUCCESS == libxs_xmalloc((void**)&internal_registry_keys, (LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_descriptor), 0/*auto-align*/,
          LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/)))
    {
      LIBXS_ASSERT(NULL != new_registry && NULL != internal_registry_keys);
      libxs_trans_init(libxs_target_archid);
      libxs_dnn_init(libxs_target_archid);
#if defined(LIBXS_PERF)
      libxs_perf_init();
#endif
      { const char *const env = getenv("LIBXS_GEMM_PREFETCH");
#if defined(_WIN32) || defined(__CYGWIN__)
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
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) new_registry[i].pmm = NULL;
#if defined(LIBXS_BUILD)
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
      { /* setup libxs_malloc_kind after internal allocations */
        const libxs_malloc_function null_malloc_fn = { 0 };
        const libxs_free_function null_free_fn = { 0 };
        const char *const env = getenv("LIBXS_MALLOC");
        if (NULL != env && 0 != *env) libxs_malloc_kind = atoi(env);
        libxs_xset_default_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
        libxs_xset_scratch_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
      }
      { /* commit the registry buffer and enable global visibility */
        void *const pv_registry = &internal_registry;
        LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)((void**)pv_registry, (void*)new_registry, LIBXS_ATOMIC_SEQ_CST);
      }
    }
    else {
      if (0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to allocate code registry!\n");
      }
      libxs_xfree(internal_registry_keys);
      libxs_xfree(new_registry);
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
#if (0 != LIBXS_SYNC)
    static int once = 0;
    /* libxs_ninit: serves as an ID to invalidate the thread-local cache; never decremented */
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST)) {
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
#endif
      }
      { /* calibrate timer */
        libxs_timer_tickint s0, t0, s1, t1;
        libxs_timer_tick_rtc(); libxs_timer_tick(); /* warm-up */
        s0 = libxs_timer_tick_rtc(); t0 = libxs_timer_tick(); /* start timing */
        internal_init();
        atexit(internal_finalize); /* once */
        s1 = libxs_timer_tick_rtc(); t1 = libxs_timer_tick(); /* final timing */
        if (LIBXS_FEQ(0, libxs_timer_scale) && s0 != s1 && t0 != t1) {
          libxs_timer_scale = libxs_timer_duration(s0, s1) / (t0 < t1 ? (t1 - t0) : (t0 - t1));
        }
      }
#if (0 != LIBXS_SYNC)
      LIBXS_ATOMIC_STORE(&once, 1, LIBXS_ATOMIC_RELAXED); /* inc? */
    }
    else while (1) {
      if (0 != LIBXS_ATOMIC_LOAD(&once, LIBXS_ATOMIC_RELAXED)) {
        break;
      }
# if 1
      else LIBXS_SYNC_YIELD();
# else
      else LIBXS_SYNC_PAUSE;
# endif
    }
#endif
    internal_init();
  }
}


LIBXS_API
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(no_instrument_function)
#endif
void libxs_finalize(void);

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
      libxs_descriptor *const registry_keys = internal_registry_keys;
      unsigned int rest = 0, errors = 0;
      internal_registry_nbytes = 0;
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        /*const*/ libxs_code_pointer code = registry[i];
        if (NULL != code.ptr_const) {
          /* check if the registered entity is a GEMM kernel */
          switch (registry_keys[i].kind) {
            case LIBXS_KERNEL_KIND_MATMUL: {
              const libxs_gemm_descriptor *const desc = &registry_keys[i].gemm.desc;
              const unsigned long long kernel_size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
              const int precision = (LIBXS_GEMM_PRECISION_F64 == desc->datatype ? 0 : 1);
              int bucket = 3/*huge*/;
              LIBXS_ASSERT(0 < kernel_size);
              if (LIBXS_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
                bucket = 0;
              }
              else if (LIBXS_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
                bucket = 1;
              }
              else if (LIBXS_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
                bucket = 2;
              }
              if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* count whether kernel is static or JIT-code */
                ++internal_statistic[precision][bucket].njit;
              }
              else {
                ++internal_statistic[precision][bucket].nsta;
              }
              ++rest;
            } break;
            case LIBXS_KERNEL_KIND_MCOPY: {
              ++internal_statistic_num_mcopy;
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
            default: if (LIBXS_KERNEL_KIND_INVALID <= registry_keys[i].kind) {
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
            if (LIBXS_CAPACITY_REGISTRY == (rest + errors +
              internal_statistic_num_mcopy + internal_statistic_num_tcopy +
              internal_statistic_num_trsm + internal_statistic_num_trmm))
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
              libxs_xfree(code.ptr_const);
              /* round-up size (it is fine to assume 4 KB pages since it is likely more accurate than not rounding up) */
              internal_registry_nbytes += (unsigned int)LIBXS_UP2(size + (((char*)code.ptr_const) - (char*)buffer), 4096/*4KB*/);
            }
          }
        }
      }
#if defined(LIBXS_TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
#if defined(LIBXS_PERF)
      libxs_perf_finalize();
#endif
      libxs_gemm_finalize();
      libxs_trans_finalize();
      libxs_dnn_finalize();
      /* make internal registry globally unavailable */
      LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, LIBXS_BITS)((uintptr_t*)regaddr, LIBXS_ATOMIC_SEQ_CST);
      internal_registry_keys = NULL;
      libxs_xfree(registry_keys);
      libxs_xfree(registry);
    }
#if (0 != LIBXS_SYNC) /* LIBXS_LOCK_RELEASE, but no LIBXS_LOCK_DESTROY */
# if (1 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
  }
}



LIBXS_API void libxs_sink(LIBXS_VARIADIC)
{
  /* does nothing else but sink the given arguments */
}


LIBXS_API int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(__MIC__)
  return libxs_target_archid;
#else /* no JIT support */
  return LIBXS_MIN(libxs_target_archid, LIBXS_X86_SSE3);
#endif
}


LIBXS_API void libxs_set_target_archid(int id)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXS_X86_AVX512_CPX:
    case LIBXS_X86_AVX512_CLX:
    case LIBXS_X86_AVX512_CORE:
    case LIBXS_X86_AVX512_KNM:
    case LIBXS_X86_AVX512_MIC:
    case LIBXS_X86_AVX512:
    case LIBXS_X86_AVX2:
    case LIBXS_X86_AVX:
    case LIBXS_X86_SSE4:
    case LIBXS_X86_SSE3:
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
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 < jit) {
      target_archid = LIBXS_X86_GENERIC + jit;
    }
    else if (0 == strcmp("cpx", arch)) {
      target_archid = LIBXS_X86_AVX512_CPX;
    }
    else if (0 == strcmp("clx", arch)) {
      target_archid = LIBXS_X86_AVX512_CLX;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("skl", arch)
          /* "avx3"/"avx512" previously enabled LIBXS_X86_AVX512 */
          || 0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch))
    {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knm", arch)) {
      target_archid = LIBXS_X86_AVX512_KNM;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic", arch)) {
      target_archid = LIBXS_X86_AVX512_MIC;
    }
    else if (0 == strcmp("hsw", arch) || 0 == strcmp("avx2", arch)) {
      target_archid = LIBXS_X86_AVX2;
    }
    else if (0 == strcmp("snb", arch) || 0 == strcmp("avx", arch)) {
      target_archid = LIBXS_X86_AVX;
    }
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch) || 0 == strcmp("sse4", arch)
       || 0 == strcmp("sse4_1", arch) || 0 == strcmp("sse4.1", arch)
       || 0 == strcmp("sse4_2", arch) || 0 == strcmp("sse4.2", arch))
    {
      target_archid = LIBXS_X86_SSE4;
    }
    else if (0 == strcmp("sse", arch) || 0 == strcmp("sse3", arch)
        || 0 == strcmp("ssse3", arch) || 0 == strcmp("ssse", arch))
    {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("x64", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
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
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
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
  switch (datatype) {
    case LIBXS_DATATYPE_F64:  return 8;
    case LIBXS_DATATYPE_F32:  return 4;
    case LIBXS_DATATYPE_BF16: return 2;
    case LIBXS_DATATYPE_I64:  return 8;
    case LIBXS_DATATYPE_I32:  return 4;
    case LIBXS_DATATYPE_I16:  return 2;
    case LIBXS_DATATYPE_I8:   return 1;
    case LIBXS_DATATYPE_UNSUPPORTED: {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: unsupported data type!\n");
      }
    } break;
  }
  LIBXS_ASSERT_MSG(0, "unsupported data type");
  return 1; /* avoid to return 0 to avoid div-by-zero in static analysis of depending code */
}


LIBXS_API_INTERN int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue)
{
  int result = EXIT_SUCCESS;
  if (NULL != value && NULL != dvalue) {
    switch (datatype) {
      case LIBXS_DATATYPE_F64: *dvalue =         (*(const double*)value); break;
      case LIBXS_DATATYPE_F32: *dvalue = (double)(*(const float *)value); break;
      case LIBXS_DATATYPE_I32: *dvalue = (double)(*(const int   *)value); break;
      case LIBXS_DATATYPE_I16: *dvalue = (double)(*(const short *)value); break;
      case LIBXS_DATATYPE_I8:  *dvalue = (double)(*(const char  *)value); break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_cast(libxs_datatype datatype, double dvalue, void* value)
{
  int result = EXIT_SUCCESS;
  if (NULL != value) {
    switch (datatype) {
      case LIBXS_DATATYPE_F64: *(double     *)value =              dvalue; break;
      case LIBXS_DATATYPE_F32: *(float      *)value =       (float)dvalue; break;
      case LIBXS_DATATYPE_I32: *(int        *)value =         (int)dvalue; break;
      case LIBXS_DATATYPE_I16: *(short      *)value =       (short)dvalue; break;
      case LIBXS_DATATYPE_I8:  *(signed char*)value = (signed char)dvalue; break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API const char* libxs_typename(libxs_datatype datatype)
{
  switch (datatype) {
    case LIBXS_DATATYPE_F64:  return "f64";
    case LIBXS_DATATYPE_F32:  return "f32";
    case LIBXS_DATATYPE_BF16: return "bf16";
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


LIBXS_API_INLINE const char* internal_get_typesize_string(size_t typesize)
{
  static LIBXS_TLS char result[4];
  LIBXS_ASSERT(256 > typesize);
  if (10 > typesize) {
    result[0] = (char)('0' + typesize);
    result[1] = 0;
  }
  else {
    LIBXS_SNPRINTF(result, sizeof(result), "%i", (int)typesize);
  }
  return result;
}


LIBXS_API_INTERN int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__)
  const char* target_arch = libxs_cpuid_name(libxs_target_archid);
  libxs_generated_code generated_code;
  char jit_name[256] = { 0 };

  /* large enough temporary buffer for generated code */
#if defined(NDEBUG)
  char jit_buffer[LIBXS_CODE_MAXSIZE];
  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.generated_code = jit_buffer;
  generated_code.buffer_size = sizeof(jit_buffer);
#else
  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.generated_code = malloc(LIBXS_CODE_MAXSIZE);
  generated_code.buffer_size = (NULL != generated_code.generated_code ? LIBXS_CODE_MAXSIZE : 0);
#endif
  /* setup code generation */
  generated_code.code_type = 2;
  generated_code.arch = libxs_target_archid;

  LIBXS_ASSERT(NULL != generated_code.generated_code || 0 == generated_code.buffer_size);
  LIBXS_ASSERT(NULL != request && 0 != libxs_target_archid);
  LIBXS_ASSERT(NULL != code && NULL == code->ptr_const);

  switch (request->kind) { /* generate kernel */
    case LIBXS_BUILD_KIND_GEMM: { /* small MxM kernel */
      LIBXS_ASSERT(NULL != request->descriptor.gemm);
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
      {
        const unsigned int m = request->descriptor.gemm->m, n = request->descriptor.gemm->n, k = request->descriptor.gemm->k;
# if !defined(LIBXS_DENY_RETARGET) /* disable: ECFLAGS=-DLIBXS_DENY_RETARGET */
        if (LIBXS_X86_AVX2 < libxs_target_archid &&
           (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.gemm->datatype) ||
            LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.gemm->datatype)) &&
           (16 >= (m * k) || 16 >= (k * n) || 16 >= (m * n)))
        {
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
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_br%i.mxm", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't', m, n, k,
            request->descriptor.gemm->lda, request->descriptor.gemm->ldb, request->descriptor.gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.gemm->flags) ? 0 : 1, uid,
            0 == (LIBXS_GEMM_FLAG_BATCH_REDUCE & request->descriptor.gemm->flags) ? 0 : 1);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SRSOA: { /* sparse SOA kernel, CSR format */
      LIBXS_ASSERT(NULL != request->descriptor.srsoa && 0 != request->descriptor.srsoa->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.srsoa->row_ptr && 0 != request->descriptor.srsoa->column_idx && 0 != request->descriptor.srsoa->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.srsoa->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.srsoa->gemm->datatype))
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_soa_kernel, &generated_code, request->descriptor.srsoa->gemm, target_arch,
          request->descriptor.srsoa->row_ptr, request->descriptor.srsoa->column_idx, request->descriptor.srsoa->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.srsoa->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.srsoa->gemm->datatype);
          const unsigned int nnz = (request->descriptor.srsoa->gemm->lda == 0) ?
            request->descriptor.srsoa->row_ptr[request->descriptor.srsoa->gemm->m] : request->descriptor.srsoa->row_ptr[request->descriptor.srsoa->gemm->k];
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_nnz%u.srsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.srsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.srsoa->gemm->flags) ? 'n' : 't',
            request->descriptor.srsoa->gemm->m,   request->descriptor.srsoa->gemm->n,   request->descriptor.srsoa->gemm->k,
            request->descriptor.srsoa->gemm->lda, request->descriptor.srsoa->gemm->ldb, request->descriptor.srsoa->gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.srsoa->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.srsoa->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SCSOA: { /* sparse SOA kernel, CSC format */
      LIBXS_ASSERT(NULL != request->descriptor.scsoa && 0 != request->descriptor.scsoa->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.scsoa->row_idx && 0 != request->descriptor.scsoa->column_ptr && 0 != request->descriptor.scsoa->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.scsoa->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.scsoa->gemm->datatype))
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csc_soa_kernel, &generated_code, request->descriptor.scsoa->gemm, target_arch,
          request->descriptor.scsoa->row_idx, request->descriptor.scsoa->column_ptr, request->descriptor.scsoa->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.scsoa->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.scsoa->gemm->datatype);
          const unsigned int nnz = (request->descriptor.scsoa->gemm->lda == 0) ?
            request->descriptor.scsoa->column_ptr[request->descriptor.scsoa->gemm->k] : request->descriptor.scsoa->column_ptr[request->descriptor.scsoa->gemm->n];
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_nnz%u.scsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.scsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.scsoa->gemm->flags) ? 'n' : 't',
            request->descriptor.scsoa->gemm->m,   request->descriptor.scsoa->gemm->n,   request->descriptor.scsoa->gemm->k,
            request->descriptor.scsoa->gemm->lda, request->descriptor.scsoa->gemm->ldb, request->descriptor.scsoa->gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.scsoa->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.scsoa->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_RMACSOA: { /* dense SOA kernel, CSC format */
      LIBXS_ASSERT(NULL != request->descriptor.rmacsoa && 0 != request->descriptor.rmacsoa->gemm);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.rmacsoa->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.rmacsoa->gemm->datatype))
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_gemm_rm_ac_soa, &generated_code, request->descriptor.rmacsoa->gemm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.rmacsoa->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.rmacsoa->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.rmacsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.rmacsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.rmacsoa->gemm->flags) ? 'n' : 't',
            request->descriptor.rmacsoa->gemm->m,   request->descriptor.rmacsoa->gemm->n,   request->descriptor.rmacsoa->gemm->k,
            request->descriptor.rmacsoa->gemm->lda, request->descriptor.rmacsoa->gemm->ldb, request->descriptor.rmacsoa->gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.rmacsoa->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.rmacsoa->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_RMBCSOA: { /* sparse SOA kernel, CSC format */
      LIBXS_ASSERT(NULL != request->descriptor.rmbcsoa && 0 != request->descriptor.rmbcsoa->gemm);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.rmbcsoa->gemm->datatype) ||
          LIBXS_GEMM_PRECISION_F32 == /*LIBXS_GETENUM_OUT*/(request->descriptor.rmbcsoa->gemm->datatype))
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_gemm_rm_bc_soa, &generated_code, request->descriptor.rmbcsoa->gemm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.rmbcsoa->gemm->prefetch);
          const char *const tname = libxs_typename((libxs_datatype)request->descriptor.rmbcsoa->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.rmbcsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.rmbcsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.rmbcsoa->gemm->flags) ? 'n' : 't',
            request->descriptor.rmbcsoa->gemm->m,   request->descriptor.rmbcsoa->gemm->n,   request->descriptor.rmbcsoa->gemm->k,
            request->descriptor.rmbcsoa->gemm->lda, request->descriptor.rmbcsoa->gemm->ldb, request->descriptor.rmbcsoa->gemm->ldc,
          /*0 != (LIBXS_GEMM_FLAG_ALPHA_0 & request->descriptor.rmbcsoa->gemm->flags) ? 0 : */1,
            0 != (LIBXS_GEMM_FLAG_BETA_0  & request->descriptor.rmbcsoa->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SREG: { /* sparse register kernel */
      LIBXS_ASSERT(NULL != request->descriptor.sreg && 0 != request->descriptor.sreg->gemm);
      LIBXS_ASSERT(NULL != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
#if 1
      if (LIBXS_GEMM_PRECISION_F64 == /*LIBXS_GETENUM_OUT*/(request->descriptor.sreg->gemm->datatype)) /* only double-precision */
#endif
      {
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
      if (4 == request->descriptor.mcopy->typesize) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_matcopy_kernel, &generated_code, request->descriptor.mcopy, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(request->descriptor.mcopy->typesize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%ux%u_p%u.mcopy", target_arch, tsizename,
            request->descriptor.mcopy->m, request->descriptor.mcopy->n, request->descriptor.mcopy->ldi, request->descriptor.mcopy->ldo,
            (unsigned int)request->descriptor.mcopy->prefetch);
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
          const char *const tsizename = internal_get_typesize_string(request->descriptor.trans->typesize);
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
        LIBXS_NO_OFFLOAD(void, libxs_generator_pgemm_kernel, &generated_code, request->descriptor.pgemm, libxs_target_archid);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(tsize);
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
        LIBXS_NO_OFFLOAD(void, libxs_generator_getrf_kernel, &generated_code, request->descriptor.getrf, libxs_target_archid);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(tsize);
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
        LIBXS_NO_OFFLOAD(void, libxs_generator_trmm_kernel, &generated_code, request->descriptor.trmm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c%c%c%c_%ux%u_%u_%u.trmm", target_arch, tsizename,
            request->descriptor.trmm->transa, request->descriptor.trmm->layout, request->descriptor.trmm->side, request->descriptor.trmm->uplo,
            request->descriptor.trmm->m, request->descriptor.trmm->n, request->descriptor.trmm->lda, request->descriptor.trmm->ldb); /* TODO: alpha */
        }
      }
    } break;
    case LIBXS_BUILD_KIND_TRSM: { /* compact TRSM kernel (packed) */
      unsigned int tsize;
      LIBXS_ASSERT(NULL != request->descriptor.trsm);
      tsize = (unsigned int)request->descriptor.trsm->typesize;
      if (4 == tsize || 8 == tsize) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_trsm_kernel, &generated_code, request->descriptor.trsm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(tsize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%c%c%c%c_%ux%u_%u_%u.trsm", target_arch, tsizename,
            request->descriptor.trsm->transa, request->descriptor.trsm->layout, request->descriptor.trsm->side, request->descriptor.trsm->uplo,
            request->descriptor.trsm->m, request->descriptor.trsm->n, request->descriptor.trsm->lda, request->descriptor.trsm->ldb); /* TODO: alpha */
        }
      }
    } break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid build request discovered!\n");
      }
      result = EXIT_FAILURE;
    }
# endif
  }

  /* handle an eventual error in the else-branch */
  if (0 < generated_code.code_size) {
    LIBXS_ASSERT(generated_code.code_size <= LIBXS_CODE_MAXSIZE);
    LIBXS_ASSERT(NULL != generated_code.generated_code);
    if (0 == generated_code.last_error) { /* no error raised */
      char* code_buffer = NULL;
      void* code_buffer_result = &code_buffer;
      /* attempt to create executable buffer */
      result = libxs_xmalloc((void**)code_buffer_result, generated_code.code_size, 0/*auto*/,
        /* flag must be a superset of what's populated by libxs_malloc_attrib */
        LIBXS_MALLOC_FLAG_RWX, &regindex, sizeof(regindex));
      if (EXIT_SUCCESS == result) { /* check for success */
        LIBXS_ASSERT(NULL != code_buffer);
        /* copy temporary buffer into the prepared executable buffer */
#if defined(NDEBUG)
        { int i; /* precondition: jit_buffer == generated_code.generated_code */
          for (i = 0; i < (int)generated_code.code_size; ++i) code_buffer[i] = jit_buffer[i];
        }
#else
        memcpy(code_buffer, generated_code.generated_code, generated_code.code_size);
#endif
        /* attribute/protect buffer and revoke unnecessary flags */
        result = libxs_malloc_attrib((void**)code_buffer_result, LIBXS_MALLOC_FLAG_X, jit_name);
        if (EXIT_SUCCESS == result) { /* check for success */
          code->pmm = code_buffer; /* commit buffer */
          LIBXS_ASSERT(NULL != code->pmm && 0 == (LIBXS_CODE_STATIC & code->uval));
        }
        else { /* release buffer */
          libxs_xfree(code_buffer);
        }
      }
    }
    else {
      result = generated_code.last_error;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
# if !defined(NDEBUG)
  free(generated_code.generated_code); /* free temporary/initial code buffer */
# endif
#else /* unsupported platform */
  LIBXS_UNUSED(request); LIBXS_UNUSED(regindex); LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXS_X86_SSE3 <= libxs_target_archid) result = EXIT_FAILURE;
#endif
  LIBXS_ASSERT(NULL != code->pmm || EXIT_FAILURE == result);
  return result;
}


#if defined(LIBXS_DESC_PAD)
LIBXS_API_INLINE void internal_pad_descriptor(libxs_descriptor* desc, size_t size)
{
  size_t i = size;
  size = LIBXS_MAX(LIBXS_DIFF_SIZE, LIBXS_HASH_SIZE);
  LIBXS_ASSERT(NULL != desc && i <= size && size <= LIBXS_DESCRIPTOR_MAXSIZE);
  for (; i < size; ++i) desc->data[i] = 0;
}
#endif


LIBXS_API_INLINE libxs_code_pointer internal_find_code(libxs_descriptor* desc, size_t desc_size)
{
  libxs_code_pointer flux_entry = { 0 };
  const size_t size = sizeof(libxs_descriptor_kind) + desc_size;
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
  int build = EXIT_SUCCESS;
#endif
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
  static LIBXS_TLS struct {
    libxs_descriptor keys[LIBXS_CACHE_MAXSIZE];
    libxs_code_pointer code[LIBXS_CACHE_MAXSIZE];
    unsigned int id; /* to invalidate cache */
    unsigned char size, hit;
  } cache;
  unsigned char cache_index;
# if defined(LIBXS_DESC_PAD)
#   if defined(LIBXS_DESC_INLINE)
  LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
  internal_pad_descriptor(desc, size);
  LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
  LIBXS_DIFF_N(unsigned char, cache_index, LIBXS_DIFF(LIBXS_DIFF_SIZE),
    xdesc, &cache.keys, LIBXS_DIFF_SIZE, LIBXS_DESCRIPTOR_MAXSIZE, cache.hit, cache.size);
#   else
  internal_pad_descriptor(desc, size);
  cache_index = (unsigned char)libxs_diff_n(desc, &cache.keys,
    LIBXS_DIFF_SIZE, LIBXS_DESCRIPTOR_MAXSIZE, cache.hit, cache.size);
#   endif
# else
  LIBXS_ASSERT(NULL != desc);
  cache_index = (unsigned char)libxs_diff_n(desc, &cache.keys,
    LIBXS_MIN(size, LIBXS_DIFF_SIZE), LIBXS_DESCRIPTOR_MAXSIZE, cache.hit, cache.size);
# endif
  if (cache_index < cache.size && cache.id == libxs_ninit) { /* valid hit */
    flux_entry = cache.code[cache_index];
    cache.hit = cache_index;
  }
  else
#else
  LIBXS_ASSERT(NULL != desc);
# if defined(LIBXS_DESC_PAD)
# if defined(LIBXS_DESC_INLINE)
  LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
  internal_pad_descriptor(desc, size);
  LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
# else
  internal_pad_descriptor(desc, size);
# endif
# endif
#endif
  {
#if defined(LIBXS_DESC_PAD)
    unsigned int i = LIBXS_CONCATENATE(libxs_crc32_b, LIBXS_HASH_SIZE)(LIBXS_HASH_SEED, desc);
#else
    unsigned int i = libxs_crc32(LIBXS_HASH_SEED, desc, LIBXS_MIN(size, LIBXS_HASH_SIZE));
#endif
    unsigned int i0 = i = LIBXS_MOD2(i, LIBXS_CAPACITY_REGISTRY), mode = 0, diff = 1;
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
#if defined(LIBXS_DESC_PAD)
# if defined(LIBXS_DIFF_INLINE)
#   if !defined(LIBXS_DESC_INLINE)
          LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
          LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
#   endif
          diff = LIBXS_DIFF(LIBXS_DIFF_SIZE)(xdesc, internal_registry_keys + i, 0/*dummy*/);
# else
          diff = libxs_diff(desc, internal_registry_keys + i, LIBXS_DIFF_SIZE);
# endif
#else
          diff = libxs_diff(desc, internal_registry_keys + i, LIBXS_MIN(size, LIBXS_DIFF_SIZE));
#endif
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
            if (LIBXS_KERNEL_KIND_MATMUL == desc->kind) {
              internal_update_mmstatistic(&desc->gemm.desc, 0, 1/*collision*/);
            }
          }
          LIBXS_ASSERT(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        LIBXS_ASSERT(0 == mode || 1 < mode);
#if (0 != LIBXS_JIT)
        if (LIBXS_X86_AVX <= libxs_target_archid || /* check if JIT is supported (CPUID) */
           (LIBXS_X86_SSE3 <= libxs_target_archid && LIBXS_BUILD_KIND_GEMM == desc->kind))
        {
          LIBXS_ASSERT(0 != mode || NULL == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.pmm); /* lock the registry entry */
          if (NULL == internal_registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxs_build_request request; /* setup the code build request */
            LIBXS_ASSERT(desc->kind < LIBXS_KERNEL_KIND_INVALID);
            request.kind = (libxs_build_kind)desc->kind;
            request.descriptor.ptr = &desc->gemm.desc;
#if defined(NDEBUG)
            if (EXIT_SUCCESS == libxs_build(&request, i, &flux_entry) && NULL != flux_entry.ptr_const)
#else
            build = libxs_build(&request, i, &flux_entry);
            if (EXIT_SUCCESS == build && NULL != flux_entry.ptr_const)
#endif
            {
              internal_registry_keys[i] = *desc;
# if (1 < INTERNAL_REGLOCK_MAXN)
              LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_registry[i].pmm, flux_entry.pmm, LIBXS_ATOMIC_SEQ_CST);
# else
              internal_registry[i] = flux_entry;
# endif
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer fix_entry;
#   if (1 < INTERNAL_REGLOCK_MAXN)
                fix_entry.pmm = LIBXS_ATOMIC_LOAD(&internal_registry[i0].pmm, LIBXS_ATOMIC_RELAXED);
#   else
                fix_entry = internal_registry[i0];
#   endif
                LIBXS_ASSERT(NULL != fix_entry.ptr_const);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
#   if (1 < INTERNAL_REGLOCK_MAXN)
                  LIBXS_ATOMIC_STORE(&internal_registry[i0].pmm, fix_entry.pmm, LIBXS_ATOMIC_RELAXED);
#   else
                  internal_registry[i0] = fix_entry;
#   endif
                }
              }
# endif
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
              diff = 0; /* inside of locked region (do not use break!) */
            }
            flux_entry.pmm = NULL; /* no result */
          }
        }
        else /* JIT-code generation not available */
#endif
        { /* leave the dispatch loop */
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
          build = EXIT_FAILURE;
#endif
          flux_entry.pmm = NULL;
          diff = 0;
        }
        if (((int)LIBXS_KERNEL_KIND_MATMUL) == desc->kind) {
          internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0);
        }
      }
    } while (0 != diff);
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    if (NULL != flux_entry.ptr_const) { /* keep code version on record (cache) */
      if (cache.id != libxs_ninit) { /* invalidate */
        memset(cache.keys, 0, sizeof(cache.keys));
        cache.id = libxs_ninit;
        cache.size = cache.hit = 0;
      }
      if (cache.size < (LIBXS_CACHE_MAXSIZE)) { /* grow */
        INTERNAL_FIND_CODE_CACHE_GROW(cache_index, cache.size);
        LIBXS_ASSERT(cache.size <= LIBXS_CACHE_MAXSIZE);
      }
      else { /* evict */
        INTERNAL_FIND_CODE_CACHE_EVICT(cache_index, cache.size, cache.hit);
      }
      cache.keys[cache_index] = *desc;
      cache.code[cache_index] = flux_entry;
      cache.hit = cache_index;
      LIBXS_ASSERT(0 == diff);
    }
#endif
  }
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
#if (0 != LIBXS_JIT)
  assert(LIBXS_BUILD_KIND_GEMM != desc->kind || NULL != flux_entry.ptr_const || EXIT_SUCCESS != build || 1 == internal_reglock_count); /*!LIBXS_ASSERT*/
#endif
  return flux_entry;
}


LIBXS_API const libxs_descriptor* libxs_get_kernel_info(libxs_code_pointer code, size_t* size)
{
  const libxs_descriptor* result;
  void* extra = NULL;
  if (NULL != size) *size = 0;
  if (NULL != code.ptr_const && NULL != internal_registry && NULL != internal_registry_keys
    && EXIT_SUCCESS == libxs_get_malloc_xinfo(code.ptr_const, size, NULL/*flags*/, &extra)
    && NULL != extra && *((const unsigned int*)extra) < (LIBXS_CAPACITY_REGISTRY)
#if defined(LIBXS_HASH_COLLISION)
    && code.uval == (~LIBXS_HASH_COLLISION & internal_registry[*((const unsigned int*)extra)].uval)
#else
    && code.ptr_const == internal_registry[*((const unsigned int*)extra)].ptr_const
#endif
    && internal_registry_keys[*((const unsigned int*)extra)].kind < LIBXS_KERNEL_KIND_INVALID)
  {
    result = internal_registry_keys + *((const unsigned int*)extra);
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API int libxs_get_kernel_kind(const void* kernel, libxs_kernel_kind* kind)
{
  const libxs_descriptor* info;
  libxs_code_pointer code;
  int result;
  code.ptr_const = kernel;
  info = libxs_get_kernel_info(code, NULL/*code_size*/);
  if (NULL != info && NULL != kind) {
    *kind = (libxs_kernel_kind)info->kind;
    result = EXIT_SUCCESS;
  }
  else {
    if (NULL != kind) *kind = LIBXS_KERNEL_KIND_INVALID;
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_mmkernel_info(libxs_xmmfunction kernel, libxs_mmkernel_info* info, size_t* code_size)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xgemm = kernel;
  if (NULL != info || NULL != code_size) {
    const libxs_descriptor *const kernel_info = libxs_get_kernel_info(code, code_size);
    if (NULL != kernel_info && LIBXS_KERNEL_KIND_MATMUL == kernel_info->kind) {
      if (NULL != info) {
        info->iprecision = (libxs_gemm_precision)LIBXS_GETENUM_INP(kernel_info->gemm.desc.datatype);
        info->oprecision = (libxs_gemm_precision)LIBXS_GETENUM_OUT(kernel_info->gemm.desc.datatype);
        info->prefetch = (libxs_gemm_prefetch_type)kernel_info->gemm.desc.prefetch;
        info->flags = kernel_info->gemm.desc.flags;
        info->lda = kernel_info->gemm.desc.lda;
        info->ldb = kernel_info->gemm.desc.ldb;
        info->ldc = kernel_info->gemm.desc.ldc;
        info->m = kernel_info->gemm.desc.m;
        info->n = kernel_info->gemm.desc.n;
        info->k = kernel_info->gemm.desc.k;
      }
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


LIBXS_API int libxs_get_transkernel_info(libxs_xtransfunction kernel, libxs_transkernel_info* info, size_t* code_size)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xtrans = kernel;
  if (NULL != info || 0 != code_size) {
    const libxs_descriptor *const kernel_info = libxs_get_kernel_info(code, code_size);
    if (NULL != kernel_info && LIBXS_KERNEL_KIND_TRANS == kernel_info->kind) {
      if (NULL != info) {
        info->typesize = kernel_info->trans.desc.typesize;
        info->ldo = kernel_info->trans.desc.ldo;
        info->m = kernel_info->trans.desc.m;
        info->n = kernel_info->trans.desc.n;
      }
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


LIBXS_API int libxs_get_mcopykernel_info(libxs_xmcopyfunction kernel, libxs_mcopykernel_info* info, size_t* code_size)
{
  libxs_code_pointer code;
  static int error_once = 0;
  int result;
  code.xmatcopy = kernel;
  if (NULL != info || 0 != code_size) {
    const libxs_descriptor *const kernel_info = libxs_get_kernel_info(code, code_size);
    if (NULL != kernel_info && LIBXS_KERNEL_KIND_MCOPY == kernel_info->kind) {
      if (NULL != info) {
        info->typesize = kernel_info->mcopy.desc.typesize;
        info->prefetch = kernel_info->mcopy.desc.prefetch;
        info->flags = kernel_info->mcopy.desc.flags;
        info->ldi = kernel_info->mcopy.desc.ldi;
        info->ldo = kernel_info->mcopy.desc.ldo;
        info->m = kernel_info->mcopy.desc.m;
        info->n = kernel_info->mcopy.desc.n;
      }
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
  if (0 != info) {
    LIBXS_INIT
    if (0 != internal_registry) {
      size_t i;
      memset(info, 0, sizeof(libxs_registry_info)); /* info->nstatic = 0; info->size = 0; */
      info->nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(libxs_descriptor));
      info->capacity = LIBXS_CAPACITY_REGISTRY;
#if defined(LIBXS_CACHE_MAXSIZE)
      info->ncache = LIBXS_CACHE_MAXSIZE;
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
              info->nbytes += (unsigned int)(buffer_size + (((char*)code.ptr_const) - (char*)buffer));
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
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  libxs_xmmfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    wrap.gemm.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_MATMUL;
    if (0 != (0x80 & descriptor->prefetch)) { /* "sign"-bit of byte-value is set */
      wrap.gemm.desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
    }
    result = internal_find_code(&wrap, sizeof(*descriptor)).xgemm;
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


LIBXS_API libxs_wimmfunction libxs_wimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.wimm;
}


LIBXS_API libxs_wsmmfunction libxs_wsmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_wsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.wsmm;
}


LIBXS_API libxs_bsmmfunction libxs_bsmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
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
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bmm;
}


LIBXS_API libxs_dmmfunction_reducebatch libxs_dmmdispatch_reducebatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_dgemm_descriptor_init(&blob,
    m, n, k, NULL != lda ? *lda : m, NULL != ldb ? *ldb : k, NULL != ldc ? *ldc : m,
    NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.dmr;
}


LIBXS_API libxs_smmfunction_reducebatch libxs_smmdispatch_reducebatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_sgemm_descriptor_init(&blob,
    m, n, k, NULL != lda ? *lda : m, NULL != ldb ? *ldb : k, NULL != ldc ? *ldc : m,
    NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.smr;
}


LIBXS_API libxs_bsmmfunction_reducebatch libxs_bsmmdispatch_reducebatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bsgemm_descriptor_init(&blob,
    m, n, k, NULL != lda ? *lda : m, NULL != ldb ? *ldb : k, NULL != ldc ? *ldc : m,
    NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bsmr;
}


LIBXS_API libxs_bmmfunction_reducebatch libxs_bmmdispatch_reducebatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXS_FLAGS : *flags);
  libxs_descriptor_blob blob;
  const libxs_gemm_descriptor *const desc = libxs_bgemm_descriptor_init(&blob,
    m, n, k, NULL != lda ? *lda : m, NULL != ldb ? *ldb : k, NULL != ldc ? *ldc : m,
    NULL != alpha ? *alpha : LIBXS_ALPHA, NULL != beta ? *beta : LIBXS_BETA,
    gemm_flags | LIBXS_GEMM_FLAG_BATCH_REDUCE, libxs_get_gemm_xprefetch(prefetch));
  /*const*/ libxs_xmmfunction result = libxs_xmmdispatch(desc);
  return result.bmr;
}


LIBXS_API libxs_xmcopyfunction libxs_dispatch_mcopy(const libxs_mcopy_descriptor* descriptor)
{
  libxs_xmcopyfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.mcopy.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_MCOPY;
#if defined(_WIN32) || defined(__CYGWIN__)
    wrap.mcopy.desc.prefetch = 0;
#endif
    result = internal_find_code(&wrap, sizeof(*descriptor)).xmatcopy;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_xtransfunction libxs_dispatch_trans(const libxs_trans_descriptor* descriptor)
{
  libxs_xtransfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.trans.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_TRANS;
    result = internal_find_code(&wrap, sizeof(*descriptor)).xtrans;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_pgemm_xfunction libxs_dispatch_pgemm(const libxs_pgemm_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.pgemm.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_PGEMM;
    result = internal_find_code(&wrap, sizeof(*descriptor)).xpgemm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_getrf_xfunction libxs_dispatch_getrf(const libxs_getrf_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.getrf.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_GETRF;
    result = internal_find_code(&wrap, sizeof(*descriptor)).xgetrf;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_trmm_xfunction libxs_dispatch_trmm(const libxs_trmm_descriptor* descriptor)
{
  libxs_trmm_xfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.trmm.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_TRMM;
    result = internal_find_code(&wrap, sizeof(*descriptor)).xtrmm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_trsm_xfunction libxs_dispatch_trsm(const libxs_trsm_descriptor* descriptor)
{
  libxs_trsm_xfunction result;
  if (NULL != descriptor) {
    libxs_descriptor wrap;
    LIBXS_INIT
    wrap.trsm.desc = *descriptor;
    wrap.kind = LIBXS_KERNEL_KIND_TRSM;
    result = internal_find_code(&wrap, sizeof(*descriptor)).xtrsm;
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXS_API libxs_xmmfunction libxs_create_xcsr_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxs_csr_soa_descriptor srsoa;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      srsoa.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      desc = *descriptor;
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      srsoa.gemm = &desc;
    }
    srsoa.row_ptr = row_ptr;
    srsoa.column_idx = column_idx;
    srsoa.values = values;
    request.descriptor.srsoa = &srsoa;
    request.kind = LIBXS_BUILD_KIND_SRSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_xcsc_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  if (NULL != descriptor && NULL != column_ptr && NULL != row_idx && NULL != values) {
    libxs_csc_soa_descriptor scsoa;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      scsoa.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      desc = *descriptor;
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      scsoa.gemm = &desc;
    }
    scsoa.column_ptr = column_ptr;
    scsoa.row_idx = row_idx;
    scsoa.values = values;
    request.descriptor.scsoa = &scsoa;
    request.kind = LIBXS_BUILD_KIND_SCSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_rm_ac_soa(const libxs_gemm_descriptor* descriptor)
{
  libxs_code_pointer result = { 0 };
  if (NULL != descriptor) {
    libxs_rm_ac_soa_descriptor rmacsoa;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      rmacsoa.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      desc = *descriptor;
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      rmacsoa.gemm = &desc;
    }
    request.descriptor.rmacsoa = &rmacsoa;
    request.kind = LIBXS_BUILD_KIND_RMACSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_xmmfunction libxs_create_rm_bc_soa(const libxs_gemm_descriptor* descriptor)
{
  libxs_code_pointer result = { 0 };
  if (NULL != descriptor) {
    libxs_rm_bc_soa_descriptor rmbcsoa;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      rmbcsoa.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      desc = *descriptor;
      desc.prefetch = (unsigned char)libxs_get_gemm_prefetch(LIBXS_PREFETCH_AUTO);
      rmbcsoa.gemm = &desc;
    }
    request.descriptor.rmbcsoa = &rmbcsoa;
    request.kind = LIBXS_BUILD_KIND_RMBCSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API libxs_dmmfunction libxs_create_dcsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxs_code_pointer result = { 0 };
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxs_csr_reg_descriptor sreg;
    libxs_build_request request;
    libxs_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      sreg.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      desc = *descriptor;
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
        desc = *descriptor;
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


LIBXS_API void libxs_release_kernel(const void* jit_kernel)
{
  if (NULL != jit_kernel) {
    static int error_once = 0;
    void* extra = NULL;
    LIBXS_INIT
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(jit_kernel, NULL/*size*/, NULL/*flags*/, &extra) && NULL != extra) {
      const unsigned int regindex = *((const unsigned int*)extra);
      if ((LIBXS_CAPACITY_REGISTRY) <= regindex) {
        libxs_xfree(jit_kernel);
      }
      else
#if !defined(LIBXS_ENABLE_DEREG)
      if (0 != libxs_verbosity /* library code is expected to be mute */
       && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: attempt to unregister a JIT-kernel!\n");
      }
#else
      { /* unregister kernel */
        internal_registry[regindex].pmm = NULL;
# if !defined(NDEBUG)
        memset(internal_registry_keys + regindex, 0, sizeof(libxs_descriptor));
# endif
        libxs_xfree(jit_kernel);
      }
#endif
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: failed to release kernel!\n");
    }
  }
}


#if defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)

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
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** jit_kernel);
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** jit_kernel)
{
#if !defined(NDEBUG)
  if (NULL != jit_kernel)
#endif
  {
    libxs_release_kernel(*jit_kernel);
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
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch2)(intptr_t* fn,
  const libxs_gemm_precision* iprec, const libxs_gemm_precision* oprec,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch2)(intptr_t* fn,
  const libxs_gemm_precision* iprec, const libxs_gemm_precision* oprec,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
#if !defined(NDEBUG)
  if (NULL != fn && NULL != m)
#endif
  {
    const libxs_gemm_precision precision = (NULL != iprec ? *iprec : LIBXS_GEMM_PRECISION_F64);
    const libxs_blasint kk = *(NULL != k ? k : m), nn = (NULL != n ? *n : kk);
    const int gemm_flags = (NULL != flags ? *flags : LIBXS_FLAGS);
    libxs_descriptor_blob blob;
    libxs_gemm_descriptor *descriptor = libxs_gemm_descriptor_init2(&blob,
      precision, NULL != oprec ? *oprec : precision, *m, nn, kk,
      NULL != lda ? *lda : (0 == (LIBXS_GEMM_FLAG_TRANS_A & gemm_flags) ? *m : kk),
      NULL != ldb ? *ldb : (0 == (LIBXS_GEMM_FLAG_TRANS_B & gemm_flags) ? kk : nn),
      *(NULL != ldc ? ldc : m), alpha, beta, gemm_flags, libxs_get_gemm_xprefetch(prefetch));
    if (NULL != descriptor) {
      libxs_code_pointer result;
      result.xgemm = libxs_xmmdispatch(descriptor);
      *fn = result.ival;
    }
    else { /* quiet */
      *fn = 0;
    }
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
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* fn,
  const libxs_gemm_precision* precision, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* fn,
  const libxs_gemm_precision* precision, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
  LIBXS_FSYMBOL(libxs_xmmdispatch2)(fn, precision, precision, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_abc)(
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c);
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
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc);
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
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc);
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall)(
  const libxs_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
  LIBXS_FSYMBOL(libxs_xmmcall_prf)(fn, a, b, c, pa, pb, pc);
}

#endif /*defined(LIBXS_BUILD)*/
