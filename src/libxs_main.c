/******************************************************************************
** Copyright (c) 2014-2018, Intel Corporation                                **
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
#include "libxs_gemm_diff.h"
#include "libxs_trace.h"
#include "libxs_trans.h"
#include "libxs_gemm.h"
#include "libxs_hash.h"
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
#include <math.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/mman.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_CAPACITY_CACHE)
# define LIBXS_CAPACITY_CACHE 4
#endif

#if !defined(LIBXS_CODE_MAXSIZE)
# define LIBXS_CODE_MAXSIZE 131072
#endif

#if !defined(LIBXS_HASH_SEED)
# define LIBXS_HASH_SEED 25071975
#endif

#if 0
# define LIBXS_HASH_MOD(N, NGEN) ((N) % (NGEN))
#else /* LIBXS_CAPACITY_REGISTRY is POT */
# define LIBXS_HASH_MOD(N, NPOT) LIBXS_MOD2(N, NPOT)
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 0 /* disabled due to no performance advantage */
# define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXS_PREFETCH) /* auto-prefetch (frontend) */ || \
  (defined(_WIN32) || defined(__CYGWIN__)) /* TODO: full support for Windows calling convention */
# define INTERNAL_PREFETCH LIBXS_PREFETCH_NONE
#else
# define INTERNAL_PREFETCH LIBXS_PREFETCH
#endif

#if defined(LIBXS_GEMM_DIFF_SW) && (2 == (LIBXS_GEMM_DIFF_SW)) /* most general implementation */
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    RESULT_INDEX = ((CACHE_HIT) + ((LIBXS_CAPACITY_CACHE) - 1)) % (LIBXS_CAPACITY_CACHE)
#else
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    assert(/*is pot*/(LIBXS_CAPACITY_CACHE) == LIBXS_UP2POT(LIBXS_CAPACITY_CACHE)); \
    RESULT_INDEX = LIBXS_MOD2((CACHE_HIT) + ((LIBXS_CAPACITY_CACHE) - 1), LIBXS_CAPACITY_CACHE)
#endif

#if defined(_DEBUG)
# define INTERNAL_DISPATCH_DEBUG(RESULT, TYPE, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA) \
  if (0 != libxs_verbosity && INT_MAX != libxs_verbosity && 0 != (RESULT).pmm) { \
    const libxs_blasint internal_dispatch_debug_m_ = M, internal_dispatch_debug_n_ = N, internal_dispatch_debug_k_ = K; \
    const libxs_blasint internal_dispatch_debug_lda_ = (0 == (PLDA) ? M : *(PLDA)); \
    const libxs_blasint internal_dispatch_debug_ldb_ = (0 == (PLDB) ? K : *(PLDB)); \
    const libxs_blasint internal_dispatch_debug_ldc_ = (0 == (PLDC) ? M : *(PLDC)); \
    LIBXS_FLOCK(stdout); \
    fprintf(stdout, "LIBXS: "); \
    LIBXS_GEMM_PRINT(stdout, LIBXS_GEMM_PRECISION(TYPE), FLAGS, \
      &internal_dispatch_debug_m_, &internal_dispatch_debug_n_, &internal_dispatch_debug_k_, \
      PALPHA, 0/*a*/, &internal_dispatch_debug_lda_, 0/*b*/, &internal_dispatch_debug_ldb_, PBETA, 0/*c*/, &internal_dispatch_debug_ldc_); \
    fprintf(stdout, " = %p\n", (RESULT).pmm); \
    LIBXS_FUNLOCK(stdout); \
  }
#else
# define INTERNAL_DISPATCH_DEBUG(RESULT, TYPE, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA)
#endif

#define INTERNAL_DISPATCH(TYPE, DESC, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) { \
  const libxs_blasint ilda = (0 == (PLDA) ? m : *(PLDA)), ildb = (0 == (PLDB) ? k : *(PLDB)), ildc = (0 == (PLDC) ? m : *(PLDC)); \
  const int internal_prefetch = (0 == (PREFETCH) ? libxs_gemm_auto_prefetch : *(PREFETCH)); \
  const int iflags = (0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)); \
  libxs_code_pointer internal_dispatch_result_; \
  libxs_gemm_descriptor DESC; \
  if (EXIT_SUCCESS == LIBXS_CONCATENATE(LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX_NAME(TYPE)), gemm_descriptor_init)( \
    &(DESC), M, N, K, ilda, ildb, ildc, \
    0 != (PALPHA) ? *((const TYPE*)(PALPHA)) : (LIBXS_ALPHA), \
    0 != (PBETA) ? *((const TYPE*)(PBETA)) : (LIBXS_BETA), \
    iflags, internal_prefetch)) \
  { \
    internal_dispatch_result_ = internal_find_code(&(DESC)); \
  } \
  else { /* unsupported */ \
    libxs_update_mmstatistic(LIBXS_GEMM_PRECISION(TYPE), M, N, K, 1/*try*/, 0); \
    internal_dispatch_result_.pmm = 0; \
  } \
  INTERNAL_DISPATCH_DEBUG(internal_dispatch_result_, TYPE, \
    0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS), \
    M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA); \
  return internal_dispatch_result_.xgemm.LIBXS_TPREFIX(TYPE, mm); \
}

#if !defined(LIBXS_NO_SYNC)
# if !defined(INTERNAL_REGLOCK_MAXN)
#   if defined(_MSC_VER)
#     define INTERNAL_REGLOCK_MAXN 0
#   else
#     define INTERNAL_REGLOCK_MAXN 256
#   endif
# endif
# if (0 < INTERNAL_REGLOCK_MAXN)
#   if !defined(LIBXS_REGNLOCK)
#     define LIBXS_REGNLOCK LIBXS_LOCK_DEFAULT
#   endif
#   if LIBXS_LOCK_TYPE_ISPOD(LIBXS_REGNLOCK)
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_reglocktype {
  char pad[LIBXS_CACHELINE];
  LIBXS_LOCK_TYPE(LIBXS_REGNLOCK) state;
} internal_reglocktype;
#   else
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_reglocktype {
  LIBXS_LOCK_TYPE(LIBXS_REGNLOCK) state;
} internal_reglocktype;
#   endif
LIBXS_API_VARIABLE(internal_reglocktype internal_reglock[INTERNAL_REGLOCK_MAXN]);
# else /* RW-lock */
#   if !defined(LIBXS_REG1LOCK)
#     if defined(_MSC_VER)
#       define LIBXS_REG1LOCK LIBXS_LOCK_MUTEX
#     else
#       define LIBXS_REG1LOCK LIBXS_LOCK_RWLOCK
#     endif
#   endif
LIBXS_API_VARIABLE(LIBXS_LOCK_TYPE(LIBXS_REG1LOCK) internal_reglock);
# endif
#endif

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXS_API_VARIABLE(int internal_reglock_count);
LIBXS_API_VARIABLE(size_t internal_registry_nbytes);
LIBXS_API_VARIABLE(libxs_kernel_info* internal_registry_keys);
LIBXS_API_VARIABLE(libxs_code_pointer* internal_registry);
LIBXS_API_VARIABLE(internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/]);
LIBXS_API_VARIABLE(unsigned int internal_statistic_sml);
LIBXS_API_VARIABLE(unsigned int internal_statistic_med);
LIBXS_API_VARIABLE(unsigned int internal_statistic_mnk);
LIBXS_API_VARIABLE(unsigned int internal_statistic_num_mcopy);
LIBXS_API_VARIABLE(unsigned int internal_statistic_num_tcopy);
LIBXS_API_VARIABLE(unsigned int internal_teardown);
LIBXS_API_VARIABLE(int internal_dispatch_trylock_locked);
LIBXS_API_VARIABLE(int internal_gemm_auto_prefetch_locked);


#if defined(LIBXS_NO_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE)
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX)
#elif (0 < INTERNAL_REGLOCK_MAXN)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
  const unsigned int LOCKINDEX = LIBXS_MOD2(INDEX, internal_reglock_count); \
  if (LIBXS_LOCK_ACQUIRED(LIBXS_REGNLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGNLOCK, &internal_reglock[LOCKINDEX].state)) { \
    if (1 != internal_reglock_count && /* (re-)try and get (meanwhile) generated code */ \
        0 != internal_registry) /* ensure engine is not shut down */ \
    { \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; \
      break; \
    } \
  }
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGNLOCK, &internal_reglock[LOCKINDEX].state); }
#else /* RW-lock */
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
  if (LIBXS_LOCK_ACQUIRED(LIBXS_REG1LOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REG1LOCK, &internal_reglock)) { \
    if (1 != internal_reglock_count && /* (re-)try and get (meanwhile) generated code */ \
        0 != internal_registry) /* ensure engine is not shut down */ \
    { \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; \
      break; \
    } \
  }
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REG1LOCK, &internal_reglock); }
#endif


LIBXS_API_DEFINITION unsigned int libxs_update_mmstatistic(libxs_gemm_precision precision,
  libxs_blasint m, libxs_blasint n, libxs_blasint k, unsigned int ntry, unsigned int ncol)
{
  const unsigned long long kernel_size = LIBXS_MNK_SIZE(m, n, k);
  const int index = (LIBXS_GEMM_PRECISION_F64 == precision ? 0 : 1);
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

  LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[index][bucket].ncol, ncol, LIBXS_ATOMIC_RELAXED);
  return LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[index][bucket].ntry, ntry, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API_INLINE unsigned int internal_update_mmstatistic(const libxs_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol)
{
  assert(0 != desc && LIBXS_KERNEL_KIND_MATMUL == desc->iflags);
  return libxs_update_mmstatistic((libxs_gemm_precision)desc->datatype, desc->m, desc->n, desc->k, ntry, ncol);
}


LIBXS_API_INLINE const char* internal_get_target_arch(int id);
LIBXS_API_INLINE const char* internal_get_target_arch(int id)
{
  const char* target_arch = 0;
  switch (id) {
    case LIBXS_X86_AVX512_ICL: {
      target_arch = "icl";
    } break;
    case LIBXS_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXS_X86_AVX512_KNM: {
      target_arch = "knm";
    } break;
    case LIBXS_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXS_X86_AVX512: {
      target_arch = "avx3";
    } break;
    case LIBXS_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXS_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXS_X86_SSE4: {
      target_arch = "wsm";
    } break;
    case LIBXS_X86_SSE3: {
      target_arch = "sse3";
    } break;
    case LIBXS_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXS_X86_GENERIC <= id) {
      target_arch = "x86";
    }
    else {
      target_arch = "unknown";
    }
  }

  assert(0 != target_arch);
  return target_arch;
}


LIBXS_API_INLINE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  assert(0 != unit);
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
  assert(0 != ostream && (0 <= precision && precision < 2));

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
      if (0 != target_arch && 0 != *target_arch) {
        assert(strlen(target_arch) < sizeof(title));
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


LIBXS_API void internal_register_static_code(const libxs_gemm_descriptor*,
  unsigned int, unsigned int, libxs_xmmfunction, libxs_code_pointer*);
LIBXS_API_DEFINITION void internal_register_static_code(const libxs_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxs_xmmfunction src, libxs_code_pointer* registry)
{
  libxs_kernel_info* dst_key = internal_registry_keys + index;
  libxs_code_pointer* dst_entry = registry + index;
#if !defined(NDEBUG)
  libxs_code_pointer code; code.xgemm = src;
  assert(0 != desc && 0 != code.ptr_const && 0 != dst_key && 0 != registry);
  assert(0 == (LIBXS_CODE_STATIC & code.uval));
#endif

  if (0 != dst_entry->ptr_const) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXS_HASH_MOD(libxs_crc32_u32(151981/*seed*/, hash), LIBXS_CAPACITY_REGISTRY);
    unsigned int i0, i, next;
#if defined(LIBXS_HASH_COLLISION)
    /* mark current entry as a collision (this might be already the case) */
    dst_entry->uval |= LIBXS_HASH_COLLISION;
#endif
    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXS_HASH_MOD(start + 1, LIBXS_CAPACITY_REGISTRY), i0 = i, next = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY);
      0 != registry[i].ptr_const && next != i0; i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY));

    /* calculate destinations */
    dst_key = internal_registry_keys + i;
    dst_entry = registry + i;

    internal_update_mmstatistic(desc, 0, 1/*collision*/);
  }

  if (0 == dst_entry->ptr_const) { /* registry not (yet) exhausted */
    dst_key->xgemm = *desc;
    dst_entry->xgemm = src;
    /* mark current entry as static code (non-JIT) */
    dst_entry->uval |= LIBXS_CODE_STATIC;
  }

  internal_update_mmstatistic(desc, 1/*try*/, 0);
}


LIBXS_API_INLINE void internal_finalize(void)
{
  libxs_finalize();
  if (0 != libxs_verbosity) { /* print statistic on termination */
    fflush(stdout); /* synchronize with standard output */
    {
      const char *const env_target_hidden = getenv("LIBXS_TARGET_HIDDEN");
      const char *const target_arch = (0 == env_target_hidden || 0 == atoi(env_target_hidden))
        ? internal_get_target_arch(libxs_target_archid)
        : 0/*hidden*/;
      const double regsize = 1.0 * internal_registry_nbytes / (1 << 20);
      libxs_scratch_info scratch_info;
      unsigned int linebreak;

      if (1 < libxs_verbosity || 0 > libxs_verbosity) {
        fprintf(stderr, "\nLIBXS_VERSION=%s-%s", LIBXS_BRANCH, LIBXS_VERSION);
      }
      linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak && 0 != target_arch) {
        fprintf(stderr, "\nLIBXS_TARGET=%s", target_arch);
      }
      fprintf(stderr, "\nRegistry: %.f MB", regsize);
      if (1 < libxs_verbosity || 0 > libxs_verbosity) {
        size_t ngemms = 0;
        int i; for (i = 0; i < 4; ++i) {
          ngemms += internal_statistic[0/*DP*/][i].nsta + internal_statistic[1/*SP*/][i].nsta;
          ngemms += internal_statistic[0/*DP*/][i].njit + internal_statistic[1/*SP*/][i].njit;
        }
        fprintf(stderr, " (gemm=%lu mcopy=%u tcopy=%u)", (unsigned long int)ngemms,
          internal_statistic_num_mcopy, internal_statistic_num_tcopy);
      }
      if (EXIT_SUCCESS == libxs_get_scratch_info(&scratch_info) && 0 < scratch_info.size) {
        fprintf(stderr, "\nScratch: %.f MB", 1.0 * scratch_info.size / (1 << 20));
        if (1 < libxs_verbosity || 0 > libxs_verbosity) {
#if !defined(LIBXS_NO_SYNC)
          if (1 < libxs_threads_count) {
            fprintf(stderr, " (mallocs=%lu, pools=%u, threads=%u)\n",
              (unsigned long int)scratch_info.nmallocs,
              scratch_info.npools, libxs_threads_count);
          }
          else
#endif
          {
            fprintf(stderr, " (mallocs=%lu, pools=%u)\n",
              (unsigned long int)scratch_info.nmallocs,
              scratch_info.npools);
          }
        }
        else {
          fprintf(stderr, "\n");
        }
      }
      else {
        fprintf(stderr, "\n");
      }
    }
  }

  /* release scratch memory pool */
  libxs_release_scratch();

#if !defined(LIBXS_NO_SYNC)
  { /* release locks */
# if (0 < INTERNAL_REGLOCK_MAXN)
    int i; for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_DESTROY(LIBXS_REGNLOCK, &internal_reglock[i].state);
# else
    LIBXS_LOCK_DESTROY(LIBXS_REG1LOCK, &internal_reglock);
# endif
    LIBXS_LOCK_DESTROY(LIBXS_LOCK, &libxs_lock_global);
  }
#endif
}


LIBXS_API_INLINE void internal_init(void)
{
#if defined(LIBXS_TRACE)
  int filter_threadid = 0, filter_mindepth = -1, filter_maxnsyms = 0, init_code = EXIT_SUCCESS;
#endif
  int i;
  const libxs_malloc_function null_malloc_fn = { 0 };
  const libxs_free_function null_free_fn = { 0 };
#if !defined(LIBXS_NO_SYNC) /* setup the locks in a thread-safe fashion */
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
# if (0 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_ACQUIRE(LIBXS_REGNLOCK, &internal_reglock[i].state);
# else
  LIBXS_LOCK_ACQUIRE(LIBXS_REG1LOCK, &internal_reglock);
# endif
#endif
  if (0 == internal_registry) { /* double-check after acquiring the lock(s) */
    assert(0 == internal_registry_keys); /* should never happen */
    libxs_xset_default_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
    libxs_xset_scratch_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    { const char *const env = getenv("LIBXS_SCRATCH_POOLS");
      if (0 == env || 0 == *env) {
        libxs_scratch_pools = LIBXS_MALLOC_SCRATCH_MAX_NPOOLS;
      }
      else {
        libxs_scratch_pools = LIBXS_CLMP(atoi(env), 0, LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
        /*libxs_scratch_pools_locked = 1;*/
      }
      assert(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    }
    { const char *const env = getenv("LIBXS_SCRATCH_LIMIT");
      if (0 == env || 0 == *env) {
        /*const*/ unsigned long long limit = LIBXS_MALLOC_SCRATCH_LIMIT;
        libxs_scratch_limit = (size_t)limit;
      }
      else {
        size_t u = strlen(env) - 1; /* 0 < strlen(env) */
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
      if (0 == env || 0 == *env) {
        libxs_scratch_scale = LIBXS_MALLOC_SCRATCH_SCALE;
      }
      else {
        libxs_scratch_scale = LIBXS_CLMP(atof(env), 1.1, 3.0);
        /*libxs_scratch_scale_locked = 1;*/
      }
      assert(1 <= libxs_scratch_scale);
    }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
    libxs_set_target_arch(getenv("LIBXS_TARGET")); /* set libxs_target_archid */
    { const char *const env = getenv("LIBXS_SYNC");
      libxs_nosync = (0 == env || 0 == *env) ? 0/*default*/ : atoi(env);
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
    {
      const char *const env = getenv("LIBXS_VERBOSE");
      if (0 != env && 0 != *env) {
        libxs_verbosity = atoi(env);
      }
#if !defined(NDEBUG)
      else {
        libxs_verbosity = INT_MAX; /* quiet -> verbose */
      }
#endif
    }
    internal_statistic_mnk = libxs_cbrt_u64(LIBXS_MAX_MNK);
    internal_statistic_sml = 13;
    internal_statistic_med = 23;
#if defined(LIBXS_TRACE)
    { const char *const env = getenv("LIBXS_TRACE");
      if (0 != env && 0 != *env) {
        char buffer[32] = { 0 };
        if (1 == sscanf(env, "%32[^,],", buffer)) {
          init_code = (0 <= sscanf(buffer, "%i", &filter_threadid) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        if (1 == sscanf(env, "%*[^,],%32[^,],", buffer)) {
          init_code = (0 <= sscanf(buffer, "%i", &filter_mindepth) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        if (1 == sscanf(env, "%*[^,],%*[^,],%32s", buffer)) {
          init_code = (0 <= sscanf(buffer, "%i", &filter_maxnsyms) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        else {
          filter_maxnsyms = -1; /* all */
        }
      }
    }
    if (EXIT_SUCCESS == init_code) {
#endif
      libxs_code_pointer *const new_registry = (libxs_code_pointer*)malloc((LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer));
      internal_registry_keys = (libxs_kernel_info*)malloc((LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_kernel_info));
      if (0 != new_registry && 0 != internal_registry_keys) {
        const char *const env = getenv("LIBXS_GEMM_PREFETCH");
        libxs_gemm_diff_init(libxs_target_archid);
        libxs_trans_init(libxs_target_archid);
        libxs_hash_init(libxs_target_archid);
        libxs_dnn_init(libxs_target_archid);
#if defined(LIBXS_PERF)
        libxs_perf_init();
#endif
        for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) new_registry[i].pmm = 0;
        /* omit registering code if JIT is enabled and if an ISA extension is found
         * which is beyond the static code path used to compile the library
         */
#if defined(LIBXS_BUILD)
# if (0 != LIBXS_JIT) && !defined(__MIC__)
        /* check if target arch. permits execution (arch. may be overridden) */
        if (LIBXS_STATIC_TARGET_ARCH <= libxs_target_archid &&
           (LIBXS_X86_AVX > libxs_target_archid /* JIT code gen. is not available */
            /* condition allows to avoid JIT (if static code is good enough) */
            || LIBXS_STATIC_TARGET_ARCH == libxs_target_archid))
# endif
        { /* opening a scope for eventually declaring variables */
          /* setup the dispatch table for the statically generated code */
#           include <libxs_dispatch.h>
        }
#endif
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
        libxs_gemm_auto_prefetch_default = INTERNAL_PREFETCH;
#else
        libxs_gemm_auto_prefetch_default = (0 == internal_statistic_ntry(0/*DP*/) && 0 == internal_statistic_ntry(1/*SP*/))
          /* avoid special prefetch if static code is present, since such code uses INTERNAL_PREFETCH */
          ? (((LIBXS_X86_AVX512 >= libxs_target_archid || LIBXS_X86_AVX512_CORE <= libxs_target_archid))
            ? LIBXS_PREFETCH_AL2BL2_VIA_C : LIBXS_PREFETCH_BL2_VIA_C)
          : INTERNAL_PREFETCH;
#endif
        libxs_gemm_auto_prefetch = INTERNAL_PREFETCH;
        if (0 != env && 0 != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            libxs_gemm_auto_prefetch_default = libxs_gemm_uid2prefetch(uid);
            libxs_gemm_auto_prefetch = libxs_gemm_auto_prefetch_default;
            internal_gemm_auto_prefetch_locked = 1;
          }
        }
        libxs_gemm_init(libxs_target_archid);
        if (0 == internal_teardown) {
          atexit(internal_finalize);
        }
        {
          void *const pv_registry = &internal_registry;
          LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)((void**)pv_registry, (void*)new_registry, LIBXS_ATOMIC_SEQ_CST);
        }
      }
      else {
        if (0 != libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS ERROR: failed to allocate code registry!\n");
        }
        free(internal_registry_keys);
        free(new_registry);
      }
#if defined(LIBXS_TRACE)
      init_code = libxs_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
      if (EXIT_SUCCESS != init_code && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to initialize TRACE (error #%i)!\n", init_code);
      }
    }
    else if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: failed to parse LIBXS_TRACE!\n");
    }
#endif
  }
#if !defined(LIBXS_NO_SYNC) /* release locks */
# if (0 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGNLOCK, &internal_reglock[i].state);
# else
  LIBXS_LOCK_RELEASE(LIBXS_REG1LOCK, &internal_reglock);
# endif
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
}


LIBXS_API_DEFINITION LIBXS_ATTRIBUTE_CTOR void libxs_init(void)
{
  if (0 == LIBXS_ATOMIC_LOAD(&internal_registry, LIBXS_ATOMIC_RELAXED)) {
    unsigned long long s1 = libxs_timer_tick(), t1; /* warm-up */
    const unsigned long long s0 = libxs_timer_tick(), t0 = libxs_timer_tick_rdtsc();
#if !defined(LIBXS_NO_SYNC) /* setup the locks in a thread-safe fashion */
    static int counter = 0, once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&counter, 1, LIBXS_ATOMIC_SEQ_CST)) {
      const char *const env_trylock = getenv("LIBXS_TRYLOCK");
      LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr_global;
# if (0 < INTERNAL_REGLOCK_MAXN)
      int i;
      LIBXS_LOCK_ATTR_TYPE(LIBXS_REGNLOCK) attr;
      LIBXS_LOCK_ATTR_INIT(LIBXS_REGNLOCK, &attr);
# else
      LIBXS_LOCK_ATTR_TYPE(LIBXS_REG1LOCK) attr;
      LIBXS_LOCK_ATTR_INIT(LIBXS_REG1LOCK, &attr);
      LIBXS_LOCK_INIT(LIBXS_REG1LOCK, &internal_reglock, &attr);
      LIBXS_LOCK_ATTR_DESTROY(LIBXS_REG1LOCK, &attr);
# endif
      LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr_global);
      LIBXS_LOCK_INIT(LIBXS_LOCK, &libxs_lock_global, &attr_global);
      LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr_global);
      /* control number of locks needed; LIBXS_TRYLOCK implies only 1 lock */
      if (0 == env_trylock || 0 == *env_trylock) { /* no LIBXS_TRYLOCK */
        internal_reglock_count = INTERNAL_REGLOCK_MAXN;
      }
      else { /* LIBXS_TRYLOCK environment variable specified */
        internal_reglock_count = (0 != atoi(env_trylock) ? 1 : (INTERNAL_REGLOCK_MAXN));
        internal_dispatch_trylock_locked = 1;
      }
# if (0 < INTERNAL_REGLOCK_MAXN)
      assert(1 <= internal_reglock_count);
      for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_INIT(LIBXS_REGNLOCK, &internal_reglock[i].state, &attr);
      LIBXS_LOCK_ATTR_DESTROY(LIBXS_REGNLOCK, &attr);
# endif
      once = 1;
    }
    else while (1) {
      if (0 != once) break;
      else LIBXS_SYNC_PAUSE;
    }
#endif
    internal_init();
    s1 = libxs_timer_tick(); t1 = libxs_timer_tick_rdtsc(); /* final timings */
    if (0 != LIBXS_FEQ(0, libxs_timer_scale) && s0 != s1 && t0 != t1) {
      libxs_timer_scale = libxs_timer_duration(s0, s1) / (t0 < t1 ? (t1 - t0) : (t0 - t1));
    }
  }
}


LIBXS_API
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(no_instrument_function)
#endif
void libxs_finalize(void);

LIBXS_API_DEFINITION LIBXS_ATTRIBUTE_DTOR void libxs_finalize(void)
{
  uintptr_t regptr = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)((uintptr_t*)&internal_registry, LIBXS_ATOMIC_SEQ_CST);
  libxs_code_pointer* registry = (libxs_code_pointer*)regptr;
  if (0 != registry) {
    int i;
#if !defined(LIBXS_NO_SYNC)
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
    /* acquire locks and thereby shortcut lazy initialization later on */
# if (0 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_ACQUIRE(LIBXS_REGNLOCK, &internal_reglock[i].state);
# else
    LIBXS_LOCK_ACQUIRE(LIBXS_REG1LOCK, &internal_reglock);
# endif
#endif
    regptr = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)((uintptr_t*)&internal_registry, LIBXS_ATOMIC_RELAXED);
    registry = (libxs_code_pointer*)regptr;

    if (0 != registry) {
      libxs_kernel_info *const registry_keys = internal_registry_keys;
      internal_registry_nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(libxs_kernel_info));

      /* serves as an id to invalidate the thread-local cache; never decremented */
      ++internal_teardown;
#if defined(LIBXS_TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
      libxs_gemm_finalize();
      libxs_gemm_diff_finalize();
      libxs_trans_finalize();
      libxs_hash_finalize();
      libxs_dnn_finalize();
#if defined(LIBXS_PERF)
      libxs_perf_finalize();
#endif
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        /*const*/ libxs_code_pointer code = registry[i];
        if (0 != code.ptr_const) {
          /* check if the registered entity is a GEMM kernel */
          if (LIBXS_KERNEL_KIND_MATMUL == registry_keys[i].xgemm.iflags) {
            const libxs_gemm_descriptor *const desc = &registry_keys[i].xgemm;
            const unsigned long long kernel_size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
            const int precision = (LIBXS_GEMM_PRECISION_F64 == desc->datatype ? 0 : 1);
            int bucket = 3/*huge*/;
            assert(0 < kernel_size);
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
          }
          else if (LIBXS_KERNEL_KIND_MCOPY == registry_keys[i].xgemm.iflags) {
            ++internal_statistic_num_mcopy;
          }
          else if (LIBXS_KERNEL_KIND_TCOPY == registry_keys[i].xgemm.iflags) {
            ++internal_statistic_num_tcopy;
          }
          else {
            fprintf(stderr, "LIBXS ERROR: code registry is corrupted!\n");
          }
          if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
            void* buffer = 0;
            size_t size = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            if (EXIT_SUCCESS == libxs_get_malloc_xinfo(code.ptr_const, &size, 0/*flags*/, &buffer)) {
              libxs_xfree(code.ptr_const);
              /* round-up size (it is fine to assume 4 KB pages since it is likely more accurate than not rounding up) */
              internal_registry_nbytes += (unsigned int)LIBXS_UP2(size + (((char*)code.ptr_const) - (char*)buffer), 4096/*4KB*/);
            }
          }
        }
      }
      /* make internal registry globally unavailable */
      LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, LIBXS_BITS)(&internal_registry, LIBXS_ATOMIC_SEQ_CST);
      internal_registry_keys = 0;
      free(registry_keys);
      free(registry);
    }
#if !defined(LIBXS_NO_SYNC) /* LIBXS_LOCK_RELEASE, but no LIBXS_LOCK_DESTROY */
# if (0 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGNLOCK, &internal_reglock[i].state);
# else
    LIBXS_LOCK_RELEASE(LIBXS_REG1LOCK, &internal_reglock);
# endif
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
  }
}


LIBXS_API_DEFINITION int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(__MIC__)
  return libxs_target_archid;
#else /* no JIT support */
  return LIBXS_MIN(libxs_target_archid, LIBXS_X86_SSE4);
#endif
}


LIBXS_API_DEFINITION void libxs_set_target_archid(int id)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  switch (id) {
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
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXS WARNING: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
}


LIBXS_API_DEFINITION const char* libxs_get_target_arch(void)
{
  LIBXS_INIT
  return internal_get_target_arch(libxs_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXS_API const char* get_target_arch(int* length);
LIBXS_API_DEFINITION const char* get_target_arch(int* length)
{
  const char *const arch = libxs_get_target_arch();
  /* valid here since function is not in the public interface */
  assert(0 != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXS_API_DEFINITION void libxs_set_target_arch(const char* arch)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  if (0 != arch && 0 != *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXS_X86_GENERIC + jit;
    }
    else if (0 == strcmp("icl", arch) || 0 == strcmp("icx", arch)) {
      target_archid = LIBXS_X86_AVX512_ICL;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("skl", arch)) {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knm", arch)) {
      target_archid = LIBXS_X86_AVX512_KNM;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic", arch)) {
      target_archid = LIBXS_X86_AVX512_MIC;
    }
    else if (0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch)) {
      target_archid = LIBXS_X86_AVX512;
    }
    else if (0 == strcmp("hsw", arch) || 0 == strcmp("avx2", arch)) {
      target_archid = LIBXS_X86_AVX2;
    }
    else if (0 == strcmp("snb", arch) || 0 == strcmp("avx", arch)) {
      target_archid = LIBXS_X86_AVX;
    }
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch)
          || 0 == strcmp("sse", arch) || 0 == strcmp("sse4", arch)
          || 0 == strcmp("sse4_2", arch)
          || 0 == strcmp("sse4.2", arch))
    {
      target_archid = LIBXS_X86_SSE4;
    }
    else if (0 == strcmp("sse3", arch)) {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXS_TARGET_ARCH_UNKNOWN == target_archid || LIBXS_X86_AVX512_ICL < target_archid) {
    target_archid = libxs_cpuid();
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxs_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXS WARNING: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
  LIBXS_ATOMIC_STORE(&libxs_target_archid, target_archid, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API_DEFINITION int libxs_get_verbosity(void)
{
  LIBXS_INIT
  return libxs_verbosity;
}


LIBXS_API_DEFINITION void libxs_set_verbosity(int level)
{
  LIBXS_INIT
  LIBXS_ATOMIC_STORE(&libxs_verbosity, level, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API_DEFINITION int libxs_get_dispatch_trylock(void)
{
  LIBXS_INIT
  return 1 == internal_reglock_count ? 1 : 0;
}


LIBXS_API_DEFINITION void libxs_set_dispatch_trylock(int trylock)
{
#if defined(LIBXS_NO_SYNC)
  LIBXS_UNUSED(trylock);
#else
  LIBXS_INIT
  if (0 == internal_dispatch_trylock_locked) { /* LIBXS_TRYLOCK environment takes precedence */
    LIBXS_ATOMIC_STORE(&internal_reglock_count, 0 != trylock ? 1 : INTERNAL_REGLOCK_MAXN, LIBXS_ATOMIC_RELAXED);
  }
#endif
}


LIBXS_API_DEFINITION libxs_gemm_prefetch_type libxs_get_gemm_auto_prefetch(void)
{
  return (libxs_gemm_prefetch_type)libxs_gemm_auto_prefetch;
}


LIBXS_API_DEFINITION void libxs_set_gemm_auto_prefetch(libxs_gemm_prefetch_type strategy)
{
  if (0 == internal_gemm_auto_prefetch_locked) { /* LIBXS_GEMM_PREFETCH environment takes precedence */
    LIBXS_ATOMIC_STORE(&libxs_gemm_auto_prefetch_default, strategy, LIBXS_ATOMIC_RELAXED);
    LIBXS_ATOMIC_STORE(&libxs_gemm_auto_prefetch, strategy, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_API_DEFINITION unsigned char libxs_typesize(libxs_datatype datatype)
{
  switch (datatype) {
    case LIBXS_DATATYPE_F64: return 8;
    case LIBXS_DATATYPE_F32: return 4;
    case LIBXS_DATATYPE_I32: return 4;
    case LIBXS_DATATYPE_I16: return 2;
    case LIBXS_DATATYPE_I8:  return 1;
  }
  return 0;
}


LIBXS_API const char* internal_get_typename(int /*datatype*/);
LIBXS_API_DEFINITION const char* internal_get_typename(int datatype)
{
  switch (datatype) {
    case LIBXS_DATATYPE_F64: return "f64";
    case LIBXS_DATATYPE_F32: return "f32";
    case LIBXS_DATATYPE_I32: return "i32";
    case LIBXS_DATATYPE_I16: return "i16";
    case LIBXS_DATATYPE_I8:  return "i8";
  }
  return "void";
}


LIBXS_API const char* internal_get_typesize_string(size_t typesize);
LIBXS_API_DEFINITION const char* internal_get_typesize_string(size_t typesize)
{
  static LIBXS_TLS char result[4];
  assert(256 > typesize);
  if (1 < typesize) {
    LIBXS_SNPRINTF(result, sizeof(result), "%i", (int)typesize);
  }
  else {
    result[0] = 0;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__)
  const char *const target_arch = internal_get_target_arch(libxs_target_archid);
  libxs_generated_code generated_code = { 0 };
  char jit_name[256] = { 0 };

  /* large enough temporary buffer for generated code */
#if defined(NDEBUG)
  char jit_buffer[LIBXS_CODE_MAXSIZE];
  generated_code.generated_code = jit_buffer;
  generated_code.buffer_size = sizeof(jit_buffer);
#else
  generated_code.generated_code = malloc(LIBXS_CODE_MAXSIZE);
  generated_code.buffer_size = (0 != generated_code.generated_code ? LIBXS_CODE_MAXSIZE : 0);
#endif
  /* setup code generation */
  generated_code.code_type = 2;

  assert(0 != request && 0 != libxs_target_archid);
  assert(0 != code && 0 == code->ptr_const);

  switch (request->kind) { /* generate kernel */
    case LIBXS_BUILD_KIND_GEMM: { /* small MxM kernel */
      assert(0 != request->descriptor.gemm);
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_gemm_kernel, &generated_code, request->descriptor.gemm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.gemm->prefetch);
          const char *const tname = internal_get_typename(request->descriptor.gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.mxm", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.gemm->m,   (unsigned int)request->descriptor.gemm->n,   (unsigned int)request->descriptor.gemm->k,
            (unsigned int)request->descriptor.gemm->lda, (unsigned int)request->descriptor.gemm->ldb, (unsigned int)request->descriptor.gemm->ldc,
            request->descriptor.gemm->alpha, request->descriptor.gemm->beta, uid);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SRSOA: { /* sparse SOA kernel, CSR format */
      assert(0 != request->descriptor.srsoa && 0 != request->descriptor.srsoa->gemm);
      assert(0 != request->descriptor.srsoa->row_ptr && 0 != request->descriptor.srsoa->column_idx && 0 != request->descriptor.srsoa->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == request->descriptor.srsoa->gemm->datatype || LIBXS_GEMM_PRECISION_F32 == request->descriptor.srsoa->gemm->datatype) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_soa_kernel, &generated_code, request->descriptor.srsoa->gemm, target_arch,
          request->descriptor.srsoa->row_ptr, request->descriptor.srsoa->column_idx, request->descriptor.srsoa->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.srsoa->gemm->prefetch);
          const char *const tname = internal_get_typename(request->descriptor.srsoa->gemm->datatype);
          const unsigned int nnz = ((unsigned int)request->descriptor.srsoa->gemm->lda == 0) ?
            request->descriptor.srsoa->row_ptr[request->descriptor.srsoa->gemm->m] : request->descriptor.srsoa->row_ptr[request->descriptor.srsoa->gemm->k];
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_nnz%u.srsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.srsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.srsoa->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.srsoa->gemm->m,   (unsigned int)request->descriptor.srsoa->gemm->n,   (unsigned int)request->descriptor.srsoa->gemm->k,
            (unsigned int)request->descriptor.srsoa->gemm->lda, (unsigned int)request->descriptor.srsoa->gemm->ldb, (unsigned int)request->descriptor.srsoa->gemm->ldc,
            request->descriptor.srsoa->gemm->alpha, request->descriptor.srsoa->gemm->beta, uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SCSOA: { /* sparse SOA kernel, CSC format */
      assert(0 != request->descriptor.scsoa && 0 != request->descriptor.scsoa->gemm);
      assert(0 != request->descriptor.scsoa->row_idx && 0 != request->descriptor.scsoa->column_ptr && 0 != request->descriptor.scsoa->values);
      /* only floating point */
      if (LIBXS_GEMM_PRECISION_F64 == request->descriptor.scsoa->gemm->datatype || LIBXS_GEMM_PRECISION_F32 == request->descriptor.scsoa->gemm->datatype) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csc_soa_kernel, &generated_code, request->descriptor.scsoa->gemm, target_arch,
          request->descriptor.scsoa->row_idx, request->descriptor.scsoa->column_ptr, request->descriptor.scsoa->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.scsoa->gemm->prefetch);
          const char *const tname = internal_get_typename(request->descriptor.scsoa->gemm->datatype);
          const unsigned int nnz = ((unsigned int)request->descriptor.srsoa->gemm->lda == 0) ?
            request->descriptor.scsoa->column_ptr[request->descriptor.scsoa->gemm->k] : request->descriptor.scsoa->column_ptr[request->descriptor.scsoa->gemm->n];
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_nnz%u.scsoa", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.scsoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.scsoa->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.scsoa->gemm->m,   (unsigned int)request->descriptor.scsoa->gemm->n,   (unsigned int)request->descriptor.scsoa->gemm->k,
            (unsigned int)request->descriptor.scsoa->gemm->lda, (unsigned int)request->descriptor.scsoa->gemm->ldb, (unsigned int)request->descriptor.scsoa->gemm->ldc,
            request->descriptor.scsoa->gemm->alpha, request->descriptor.scsoa->gemm->beta, uid, nnz);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_SREG: { /* sparse register kernel */
      assert(0 != request->descriptor.sreg && 0 != request->descriptor.sreg->gemm);
      assert(0 != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
#if 1
      if (LIBXS_GEMM_PRECISION_F64 == request->descriptor.sreg->gemm->flags) { /* only double-precision */
#endif
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_reg_kernel, &generated_code, request->descriptor.sreg->gemm, target_arch,
          request->descriptor.sreg->row_ptr, request->descriptor.sreg->column_idx,
          (const double*)request->descriptor.sreg->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.sreg->gemm->prefetch);
          const char *const tname = internal_get_typename(request->descriptor.sreg->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.sreg", target_arch, tname,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.sreg->gemm->m,   (unsigned int)request->descriptor.sreg->gemm->n,   (unsigned int)request->descriptor.sreg->gemm->k,
            (unsigned int)request->descriptor.sreg->gemm->lda, (unsigned int)request->descriptor.sreg->gemm->ldb, (unsigned int)request->descriptor.sreg->gemm->ldc,
            request->descriptor.sreg->gemm->alpha, request->descriptor.sreg->gemm->beta, uid);
        }
#if 1
      }
#endif
    } break;
    case LIBXS_BUILD_KIND_CFWD: { /* forward convolution */
      assert(0 != request->descriptor.cfwd);
      if (0 < request->descriptor.cfwd->kw && 0 < request->descriptor.cfwd->kh &&
          0 != request->descriptor.cfwd->stride_w && 0 != request->descriptor.cfwd->stride_h)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_forward_kernel, &generated_code, request->descriptor.cfwd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(request->descriptor.cfwd->datatype);
          const char *const precision_out = internal_get_typename(request->descriptor.cfwd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_fwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cfwd->kw/*kernel width*/, (unsigned int)request->descriptor.cfwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cfwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cfwd->unroll_kh/*height*/,
            (int)request->descriptor.cfwd->stride_w/*input offset*/, (int)request->descriptor.cfwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cfwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cfwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cfwd->ifw_padded, (unsigned int)request->descriptor.cfwd->ifh_padded,
            (unsigned int)request->descriptor.cfwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cfwd->ofh_rb/*register block ofh*/,
            (int)request->descriptor.cfwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cfwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CBWD: { /* backward convolution */
      assert(0 != request->descriptor.cbwd);
      if (0 < request->descriptor.cbwd->kw && 0 < request->descriptor.cbwd->kh &&
          0 != request->descriptor.cbwd->stride_w && 0 != request->descriptor.cbwd->stride_h)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_backward_kernel, &generated_code, request->descriptor.cbwd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(request->descriptor.cbwd->datatype);
          const char *const precision_out = internal_get_typename(request->descriptor.cbwd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_bwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cbwd->kw/*kernel width*/, (unsigned int)request->descriptor.cbwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cbwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cbwd->unroll_kh/*height*/,
            (int)request->descriptor.cbwd->stride_w/*input offset*/, (int)request->descriptor.cbwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cbwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cbwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cbwd->ifw_padded, (unsigned int)request->descriptor.cbwd->ifh_padded,
            (unsigned int)request->descriptor.cbwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cbwd->ofh_rb/*register block ofh*/,
            (int)request->descriptor.cbwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cbwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CUPD: { /* convolution update weights */
      assert(0 != request->descriptor.cupd);
      if (0 < request->descriptor.cupd->kw &&
          0 != request->descriptor.cupd->stride_w && 0 != request->descriptor.cupd->stride_h)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_weight_update_kernel, &generated_code, request->descriptor.cupd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(request->descriptor.cupd->datatype);
          const char *const precision_out = internal_get_typename(request->descriptor.cupd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_upd_%s_%s_%ux%u_%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%ux%uu%u_if%uu_t%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cupd->kw/*kernel width*/, (unsigned int)request->descriptor.cupd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cupd->unroll_kw/*width*/,
            (int)request->descriptor.cupd->stride_w/*input offset*/, (int)request->descriptor.cupd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cupd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cupd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cupd->ifw_padded, (unsigned int)request->descriptor.cupd->ifh_padded,
            (unsigned int)request->descriptor.cupd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cupd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cupd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cupd->ofh_rb/*register block ofh*/,
            (unsigned int)request->descriptor.cupd->ofw/*ofw*/, (unsigned int)request->descriptor.cupd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cupd->ofh/*ofh*/, (unsigned int)request->descriptor.cupd->ofh_unroll/*ofh_unroll*/,
            (unsigned int)request->descriptor.cupd->ifm_unroll/*ifm unroll*/,
            (unsigned int)request->descriptor.cupd->transpose_ofw_ifm/*transpose_ofw_ifm*/,
            (int)request->descriptor.cupd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cupd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CWFWD: { /* convolution Winograd forward */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles &&
          0 < request->descriptor.cwino->bimg && 0 < request->descriptor.cwino->ur)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wfwd_%s_%s_t%ux%u_mb%u_u%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/,
            (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur/*unrolling*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CWBWD: { /* convolution Winograd backward */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles &&
          0 < request->descriptor.cwino->bimg && 0 < request->descriptor.cwino->ur)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wbwd_%s_%s_t%ux%u_mb%u_u%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/,
            (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur/*unrolling*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CWUPD: { /* convolution Winograd update */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles &&
          0 < request->descriptor.cwino->bimg && 0 < request->descriptor.cwino->ur)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_weight_update_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_typename(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wupd_%s_%s_t%ux%u_mb%u_u%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/,
            (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur/*unrolling*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_MCOPY: { /* matcopy kernel */
      assert(0 != request->descriptor.matcopy);
      if (4 == request->descriptor.matcopy->typesize || 8 == request->descriptor.matcopy->typesize
       || 2 == request->descriptor.matcopy->typesize || 1 == request->descriptor.matcopy->typesize)
      {
        LIBXS_NO_OFFLOAD(void, libxs_generator_matcopy_kernel, &generated_code, request->descriptor.matcopy, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(request->descriptor.matcopy->typesize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u_%ux%u_p%u.mcopy", target_arch, tsizename,
            request->descriptor.matcopy->m, request->descriptor.matcopy->n,
            request->descriptor.matcopy->ldi, request->descriptor.matcopy->ldo,
            (unsigned int)request->descriptor.matcopy->prefetch);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_TRANS: { /* transpose kernel */
      assert(0 != request->descriptor.trans);
      if (4 == request->descriptor.trans->typesize || 8 == request->descriptor.trans->typesize) {
        LIBXS_NO_OFFLOAD(void, libxs_generator_transpose_kernel, &generated_code, request->descriptor.trans, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const tsizename = internal_get_typesize_string(request->descriptor.trans->typesize);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_tsize%s_%ux%u.trans", target_arch, tsizename,
            request->descriptor.trans->m, request->descriptor.trans->n);
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
  if (0 != generated_code.generated_code) {
    if (0 == generated_code.last_error) { /* no error raised */
      if (0 < generated_code.code_size) { /* sanity check */
        /* attempt to create executable buffer */
        result = libxs_xmalloc(&code->pmm, generated_code.code_size, 0/*auto*/,
          /* flag must be a superset of what's populated by libxs_malloc_attrib */
          LIBXS_MALLOC_FLAG_RWX, &regindex, sizeof(regindex));
        if (EXIT_SUCCESS == result) { /* check for success */
          assert(0 != code->pmm && 0 == (LIBXS_CODE_STATIC & code->uval));
          assert(0 != generated_code.generated_code/*sanity check*/);
          /* copy temporary buffer into the prepared executable buffer */
          memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);
          /* attribute/protect buffer and revoke unnecessary flags */
          result = libxs_malloc_attrib(&code->pmm, LIBXS_MALLOC_FLAG_X, jit_name);
        }
      }
    }
    else {
# if !defined(LIBXS_VERBOSE_BACKEND) /* avoid duplicated error messages */
      if (0 != libxs_verbosity) { /* library code is expected to be mute */
        static int error_once = 0;
        if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
          LIBXS_NO_OFFLOAD(int, fprintf, stderr, "LIBXS ERROR: %s\n",
            LIBXS_NO_OFFLOAD(const char*, libxs_strerror, generated_code.last_error));
        }
      }
# endif
      result = EXIT_FAILURE;
    }
# if !defined(NDEBUG)
    free(generated_code.generated_code); /* free temporary/initial code buffer */
# endif
  }
#else /* unsupported platform */
  LIBXS_UNUSED(request); LIBXS_UNUSED(regindex); LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXS_X86_AVX <= libxs_target_archid) result = EXIT_FAILURE;
#endif
  return result;
}


LIBXS_API_INLINE libxs_code_pointer internal_find_code(const libxs_gemm_descriptor* descriptor)
{
  libxs_code_pointer flux_entry = { 0 };
  unsigned int hash, i0, i = 0, mode = 0, diff = 1;
#if !defined(NDEBUG)
  const libxs_gemm_descriptor* refdesc = 0;
#endif
#if defined(LIBXS_CAPACITY_CACHE) && (0 < (LIBXS_CAPACITY_CACHE))
  static LIBXS_TLS struct {
    libxs_gemm_descriptor keys[LIBXS_CAPACITY_CACHE];
    libxs_code_pointer code[LIBXS_CAPACITY_CACHE];
    unsigned int hit, id;
  } cache;
  unsigned int cache_index;
  assert(0 != descriptor);
  /* search small cache starting with the last hit on record */
  cache_index = libxs_gemm_diffn(descriptor, &cache.keys, cache.hit, LIBXS_CAPACITY_CACHE, LIBXS_GEMM_DESCRIPTOR_SIZE);
  if ((LIBXS_CAPACITY_CACHE) > cache_index && cache.id == internal_teardown) { /* cache hit, and valid */
    flux_entry = cache.code[cache_index];
    cache.hit = cache_index;
#if !defined(NDEBUG)
    if (0 == (LIBXS_CODE_STATIC & flux_entry.uval)) { /* JIT only */
      void* extra = 0;
# if defined(LIBXS_HASH_COLLISION)
      flux_entry.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
# endif
      if (EXIT_SUCCESS == libxs_get_malloc_xinfo(flux_entry.ptr_const, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
        refdesc = &internal_registry_keys[*((const unsigned int*)extra)].xgemm;
      }
    }
#endif
  }
  else
#else
  assert(0 != descriptor);
#endif
  {
    assert(0 != internal_registry);
    /* calculate registry location (and check if the requested code is already JITted) */
    hash = libxs_crc32(descriptor, LIBXS_GEMM_DESCRIPTOR_SIZE, LIBXS_HASH_SEED);
    i = i0 = LIBXS_HASH_MOD(hash, LIBXS_CAPACITY_REGISTRY);

    while (0 != diff) {
#if (0 < INTERNAL_REGLOCK_MAXN) || defined(LIBXS_NO_SYNC) /* read registered code */
      flux_entry.uval = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)((uintptr_t*)&internal_registry[i].pmm, LIBXS_ATOMIC_RELAXED);
#else
      LIBXS_LOCK_ACQREAD(LIBXS_REG1LOCK, &internal_reglock);
      flux_entry.pmm = internal_registry[i].pmm; /* read registered code */
      LIBXS_LOCK_RELREAD(LIBXS_REG1LOCK, &internal_reglock);
#endif
      if ((0 != flux_entry.ptr_const || 1 == mode) && 2 > mode) { /* check existing entry further */
        diff = 0 != flux_entry.ptr_const ? libxs_gemm_diff(descriptor, &internal_registry_keys[i].xgemm) : 1;
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXS_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXS_HASH_COLLISION & flux_entry.uval)) {
              assert(0 != flux_entry.ptr_const); /* collision */
              mode = 3;
            }
            else
#endif      /* search for an existing code version */
            {
              mode = 1;
            }
          }
          i = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXS_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
          }
          assert(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        assert(0 == mode || 1 < mode);
#if (0 != LIBXS_JIT)
        if (LIBXS_X86_AVX <= libxs_target_archid) { /* check if JIT is supported (CPUID) */
          assert(0 != mode || 0 == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.pmm); /* lock the registry entry */
          if (0 == internal_registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxs_build_request request; /* setup the code build request */
            request.descriptor.gemm = descriptor;
            if (LIBXS_KERNEL_KIND_MCOPY != descriptor->iflags) {
              if (LIBXS_KERNEL_KIND_TCOPY != descriptor->iflags) { /* GEMM */
                internal_update_mmstatistic(descriptor, 1/*try*/, 0); /* count attempt */
                request.kind = LIBXS_BUILD_KIND_GEMM;
              }
              else { /* transpose */
                request.kind = LIBXS_BUILD_KIND_TRANS;
              }
            }
            else { /* matcopy */
              request.kind = LIBXS_BUILD_KIND_MCOPY;
            }
            if (EXIT_SUCCESS == libxs_build(&request, i, &flux_entry) && 0 != flux_entry.ptr_const) {
              internal_registry_keys[i].xgemm = *descriptor;
# if (0 < INTERNAL_REGLOCK_MAXN)
              LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_registry[i].pmm, flux_entry.pmm, LIBXS_ATOMIC_RELAXED);
# else
              internal_registry[i].pmm = flux_entry.pmm;
# endif
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer fix_entry;
#   if (0 < INTERNAL_REGLOCK_MAXN)
                fix_entry.pmm = LIBXS_ATOMIC_LOAD(&internal_registry[i0].pmm, LIBXS_ATOMIC_RELAXED);
#   else
                fix_entry.pmm = internal_registry[i0].pmm;
#   endif
                assert(0 != fix_entry.ptr_const);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
#   if (0 < INTERNAL_REGLOCK_MAXN)
                  LIBXS_ATOMIC_STORE(&internal_registry[i0].pmm, fix_entry.pmm, LIBXS_ATOMIC_RELAXED);
#   else
                  internal_registry[i0].pmm = fix_entry.pmm;
#   endif
                }
              }
# endif
            }
            diff = 0; /* inside of locked region (do not use break!) */
          }
          INTERNAL_FIND_CODE_UNLOCK(lock);
          if (0 != diff) { /* acquire registry slot */
            if (0 == mode) { /* initial condition */
              mode = 2; /* continue to linearly search for an empty slot */
              i0 = i; /* keep current position on record */
            }
            for (i = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY); i != i0 && 0 != internal_registry[i].ptr_const;
                 i = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY)); /* continue to linearly search code */
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* inside of locked region (do not use break!) */
            }
            flux_entry.pmm = 0; /* no result */
          }
        }
        else /* JIT-code generation not available */
#endif
        { /* leave the dispatch loop */
          flux_entry.pmm = 0;
          diff = 0;
        }
      }
    }
#if defined(LIBXS_CAPACITY_CACHE) && (0 < (LIBXS_CAPACITY_CACHE))
    if (0 != flux_entry.ptr_const) { /* keep code version on record (cache) */
      INTERNAL_FIND_CODE_CACHE_INDEX(cache.hit, cache_index);
      cache.keys[cache_index] = *descriptor;
      cache.code[cache_index] = flux_entry;
      cache.hit = cache_index;
      assert(0 == diff);
    }
    if (cache.id != internal_teardown) {
      memset(cache.keys, 0, sizeof(cache.keys));
      cache.id = internal_teardown;
    }
#endif
#if !defined(NDEBUG)
    refdesc = &internal_registry_keys[i].xgemm;
#endif
  }
  assert(0 == flux_entry.ptr_const || 0 == refdesc || 0 == memcmp(refdesc, descriptor, LIBXS_GEMM_DESCRIPTOR_SIZE));
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
  return flux_entry;
}


LIBXS_API_DEFINITION const libxs_kernel_info* libxs_get_kernel_info(libxs_code_pointer code, libxs_kernel_kind* kind, size_t* size)
{
  const libxs_kernel_info* result;
  void* extra = 0;
  if (0 != code.ptr_const && 0 != internal_registry && 0 != internal_registry_keys
    && EXIT_SUCCESS == libxs_get_malloc_xinfo(code.ptr_const, size, 0/*flags*/, &extra)
    && 0 != extra && *((const unsigned int*)extra) < (LIBXS_CAPACITY_REGISTRY)
    && code.ptr_const == internal_registry[*((const unsigned int*)extra)].ptr_const
    /* the kernel kind is stored in the internal flags of the libxs_gemm_descriptor (iflags). */
    && internal_registry_keys[*((const unsigned int*)extra)].xgemm.iflags < LIBXS_KERNEL_KIND_INVALID)
  {
    if (0 != kind) *kind = (libxs_kernel_kind)internal_registry_keys[*((const unsigned int*)extra)].xgemm.iflags;
    result = internal_registry_keys + *((const unsigned int*)extra);
  }
  else {
    if (0 != kind) *kind = LIBXS_KERNEL_KIND_INVALID;
    result = 0;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_get_kernel_kind(const void* kernel, libxs_kernel_kind* kind)
{
  libxs_code_pointer code; code.ptr_const = kernel;
  return (0 != libxs_get_kernel_info(code, kind, 0/*code_size*/) ? EXIT_SUCCESS : EXIT_FAILURE);
}


LIBXS_API_DEFINITION int libxs_get_mmkernel_info(libxs_xmmfunction kernel, libxs_gemm_descriptor* info, size_t* code_size)
{
  libxs_code_pointer code;
  libxs_kernel_kind kind;
  static int error_once = 0;
  int result;
  code.xgemm = kernel;
  if (0 != info || 0 != code_size) {
    const libxs_kernel_info *const kernel_info = libxs_get_kernel_info(code, &kind, code_size);
    if (0 != kernel_info && LIBXS_KERNEL_KIND_MATMUL == kind) {
      if (0 != info) *info = kernel_info->xgemm;
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


LIBXS_API_DEFINITION int libxs_get_transkernel_info(libxs_xtransfunction kernel, libxs_transpose_descriptor* info, size_t* code_size)
{
  libxs_code_pointer code;
  libxs_kernel_kind kind;
  static int error_once = 0;
  int result;
  code.xtrans = kernel;
  if (0 != info || 0 != code_size) {
    const libxs_kernel_info *const kernel_info = libxs_get_kernel_info(code, &kind, code_size);
    if (0 != kernel_info && LIBXS_KERNEL_KIND_TCOPY == kind) {
      if (0 != info) *info = kernel_info->trans;
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


LIBXS_API_DEFINITION int libxs_get_matcopykernel_info(libxs_xmatcopyfunction kernel, libxs_matcopy_descriptor* info, size_t* code_size)
{
  libxs_code_pointer code;
  libxs_kernel_kind kind;
  static int error_once = 0;
  int result;
  code.xmatcopy = kernel;
  if (0 != info || 0 != code_size) {
    const libxs_kernel_info *const kernel_info = libxs_get_kernel_info(code, &kind, code_size);
    if (0 != kernel_info && LIBXS_KERNEL_KIND_MCOPY == kind) {
      if (0 != info) *info = kernel_info->mcopy;
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


LIBXS_API_DEFINITION int libxs_get_registry_info(libxs_registry_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
    LIBXS_INIT
    if (0 != internal_registry) {
      size_t i;
      memset(info, 0, sizeof(libxs_registry_info)); /* info->nstatic = 0; info->size = 0; */
      info->nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(libxs_kernel_info));
      info->capacity = LIBXS_CAPACITY_REGISTRY;
      info->ncache = LIBXS_CAPACITY_CACHE;
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        libxs_code_pointer code = internal_registry[i];
        if (0 != code.ptr_const && EXIT_SUCCESS == result) {
          if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
            size_t buffer_size = 0;
            void* buffer = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            result = libxs_get_malloc_xinfo(code.ptr_const, &buffer_size, 0/*flags*/, &buffer);
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


LIBXS_API_DEFINITION int libxs_gemm_descriptor_init(libxs_gemm_descriptor* descriptor,
  libxs_gemm_precision precision, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
  const libxs_blasint ilda = (0 == lda ? m : *lda), ildb = (0 == ldb ? k : *ldb), ildc = (0 == ldc ? m : *ldc);
  const int internal_prefetch = (0 == prefetch ? libxs_gemm_auto_prefetch : *prefetch);
  const int iflags = (0 == flags ? LIBXS_FLAGS : *flags);
  int result;

  switch (precision) {
    case LIBXS_GEMM_PRECISION_F64: {
      result = libxs_dgemm_descriptor_init(descriptor, m, n, k, ilda, ildb, ildc,
        0 != alpha ? *((const double*)alpha) : (LIBXS_ALPHA),
        0 != beta ? *((const double*)beta) : (LIBXS_BETA),
        iflags, internal_prefetch);
    } break;
    case LIBXS_GEMM_PRECISION_F32: {
      result = libxs_sgemm_descriptor_init(descriptor, m, n, k, ilda, ildb, ildc,
        0 != alpha ? *((const float*)alpha) : (LIBXS_ALPHA),
        0 != beta ? *((const float*)beta) : (LIBXS_BETA),
        iflags, internal_prefetch);
    } break;
    case LIBXS_GEMM_PRECISION_I16: {
      /**
       * Take alpha and beta as short data although wgemm works on integers.
       * However, alpha and beta are only JIT-supported for certain values,
       * and the call-side may not distinct different input and output types
       * (integer/short), hence it is safer to only read short data.
       */
      result = libxs_wgemm_descriptor_init(descriptor, m, n, k, ilda, ildb, ildc,
        0 != alpha ? *((const short*)alpha) : (LIBXS_ALPHA),
        0 != beta ? *((const short*)beta) : (LIBXS_BETA),
        iflags, internal_prefetch);
    } break;
    default: {
      static int error_once = 0;
      if (0 != libxs_verbosity /* library code is expected to be mute */
       && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: GEMM precision is not supported!\n");
      }
      result = EXIT_FAILURE;
    }
  }

  return result;
}


LIBXS_API_DEFINITION libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  libxs_xmmfunction result = { 0 };
  /* there is no need to check LIBXS_GEMM_NO_BYPASS_DIMS (M, N, K, LDx) since we already got a descriptor */
  if (0 != descriptor && LIBXS_GEMM_NO_BYPASS(descriptor->flags, descriptor->alpha, descriptor->beta)) {
    libxs_gemm_descriptor backend_descriptor;
    LIBXS_INIT
    if (0 > (int)descriptor->prefetch) {
      backend_descriptor = *descriptor;
      LIBXS_GEMM_DESCRIPTOR_PREFETCH(backend_descriptor, libxs_gemm_auto_prefetch);
      descriptor = &backend_descriptor;
    }
    result = internal_find_code(descriptor).xgemm;
  }
  else { /* bypass (not supported) */
    internal_update_mmstatistic(descriptor, 1/*try*/, 0);
  }
  return result;
}


#if !defined(LIBXS_BUILD) && defined(__APPLE__) && defined(__MACH__) && defined(__clang__) && !defined(__INTEL_COMPILER)
LIBXS_PRAGMA_OPTIMIZE_OFF
#endif

LIBXS_API_DEFINITION libxs_dmmfunction libxs_dmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  LIBXS_INIT
  INTERNAL_DISPATCH(double, descriptor, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_API_DEFINITION libxs_smmfunction libxs_smmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  LIBXS_INIT
  INTERNAL_DISPATCH(float, descriptor, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_API_DEFINITION libxs_wmmfunction libxs_wmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta,
  const int* flags, const int* prefetch)
{
  LIBXS_INIT
  INTERNAL_DISPATCH(short, descriptor, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


#if !defined(LIBXS_BUILD) && defined(__APPLE__) && defined(__MACH__) && defined(__clang__) && !defined(__INTEL_COMPILER)
LIBXS_PRAGMA_OPTIMIZE_ON
#endif

LIBXS_API_DEFINITION libxs_xmatcopyfunction libxs_xmatcopydispatch(const libxs_matcopy_descriptor* descriptor)
{
  libxs_xmatcopyfunction result = { 0 };
  if (0 != descriptor) {
    libxs_kernel_info query = { { 0 } };
    assert(LIBXS_SIZEOF(descriptor, &descriptor->flags) < sizeof(query));
    LIBXS_INIT
    query.mcopy = *descriptor;
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
    query.mcopy.prefetch = 0;
#endif
    query.xgemm.iflags = LIBXS_KERNEL_KIND_MCOPY;
    result = internal_find_code(&query.xgemm).xmatcopy;
  }
  return result;
}


LIBXS_API_DEFINITION libxs_xtransfunction libxs_xtransdispatch(const libxs_transpose_descriptor* descriptor)
{
  libxs_xtransfunction result = { 0 };
  if (0 != descriptor && 0 != LIBXS_TRANS_NO_BYPASS_DIMS(descriptor->m, descriptor->n, descriptor->ldo)) {
    libxs_kernel_info query = { { 0 } };
    assert(LIBXS_SIZEOF(descriptor, &descriptor->typesize) < sizeof(query));
    LIBXS_INIT
    query.trans = *descriptor;
    query.xgemm.iflags = LIBXS_KERNEL_KIND_TCOPY;
    result = internal_find_code(&query.xgemm).xtrans;
  }
  return result;
}


LIBXS_API_DEFINITION libxs_xmmfunction libxs_create_xcsr_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  if (0 != descriptor && 0 != row_ptr && 0 != column_idx && 0 != values) {
    libxs_csr_soa_descriptor srsoa;
    libxs_build_request request;
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
    libxs_gemm_descriptor gemm = *descriptor;
    LIBXS_GEMM_DESCRIPTOR_PREFETCH(gemm, LIBXS_PREFETCH_NONE);
    descriptor = &gemm;
#endif
    LIBXS_INIT
    srsoa.gemm = descriptor;
    srsoa.row_ptr = row_ptr;
    srsoa.column_idx = column_idx;
    srsoa.values = values;
    request.descriptor.srsoa = &srsoa;
    request.kind = LIBXS_BUILD_KIND_SRSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API_DEFINITION libxs_xmmfunction libxs_create_xcsc_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values)
{
  libxs_code_pointer result = { 0 };
  if (0 != descriptor && 0 != column_ptr && 0 != row_idx && 0 != values) {
    libxs_csc_soa_descriptor scsoa;
    libxs_build_request request;
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
    libxs_gemm_descriptor gemm = *descriptor;
    LIBXS_GEMM_DESCRIPTOR_PREFETCH(gemm, LIBXS_PREFETCH_NONE);
    descriptor = &gemm;
#endif
    LIBXS_INIT
    scsoa.gemm = descriptor;
    scsoa.column_ptr = column_ptr;
    scsoa.row_idx = row_idx;
    scsoa.values = values;
    request.descriptor.scsoa = &scsoa;
    request.kind = LIBXS_BUILD_KIND_SCSOA;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXS_API_DEFINITION libxs_dmmfunction libxs_create_dcsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxs_code_pointer result = { 0 };
  if (0 != descriptor && 0 != row_ptr && 0 != column_idx && 0 != values) {
    libxs_csr_reg_descriptor sreg;
    libxs_build_request request;
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
    libxs_gemm_descriptor gemm = *descriptor;
    LIBXS_GEMM_DESCRIPTOR_PREFETCH(gemm, LIBXS_PREFETCH_NONE);
    descriptor = &gemm;
#endif
    LIBXS_INIT
    sreg.gemm = descriptor;
    sreg.row_ptr = row_ptr;
    sreg.column_idx = column_idx;
    sreg.values = values;
    request.descriptor.sreg = &sreg;
    request.kind = LIBXS_BUILD_KIND_SREG;
    libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm.dmm;
}


LIBXS_API_DEFINITION libxs_smmfunction libxs_create_scsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values)
{
  libxs_code_pointer result = { 0 };
  if (0 != descriptor && 0 != row_ptr && 0 != column_idx && 0 != values) {
    libxs_csr_reg_descriptor sreg;
    libxs_build_request request;
    const unsigned int n = row_ptr[descriptor->m];
    double *const d_values = (double*)malloc(n * sizeof(double));
#if defined(_WIN32) || defined(__CYGWIN__) /* TODO: full support for Windows calling convention */
    libxs_gemm_descriptor gemm = *descriptor;
    LIBXS_GEMM_DESCRIPTOR_PREFETCH(gemm, LIBXS_PREFETCH_NONE);
    descriptor = &gemm;
#endif
    if (0 != d_values) {
      unsigned int i;
      LIBXS_INIT
      /* we need to copy the values into a double precision buffer */
      for (i = 0; i < n; ++i) d_values[i] = (double)values[i];
      sreg.gemm = descriptor;
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


LIBXS_API_DEFINITION void libxs_release_kernel(const void* jit_code)
{
  void* extra = 0;
  LIBXS_INIT
  if (EXIT_SUCCESS == libxs_get_malloc_xinfo(jit_code, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int regindex = *((const unsigned int*)extra);
    if ((LIBXS_CAPACITY_REGISTRY) <= regindex) {
      libxs_xfree(jit_code);
    }
#if !defined(NDEBUG)
    else { /* TODO: implement to unregister GEMM kernels */
      fprintf(stderr, "LIBXS WARNING: attempt to unregister a JIT-kernel!\n");
    }
#endif
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: failed to release kernel!\n");
    }
  }
}


LIBXS_API_DEFINITION int libxs_matdiff(libxs_datatype datatype, libxs_blasint m, libxs_blasint n,
  const void* ref, const void* tst, const libxs_blasint* ldref, const libxs_blasint* ldtst,
  libxs_matdiff_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != ref && 0 != tst && 0 != info) {
    libxs_blasint mm = m, nn = n, ldr = (0 == ldref ? m : *ldref), ldt = (0 == ldtst ? m : *ldtst);
    if (1 == n) { mm = ldr = ldt = 1; nn = m; } /* ensure row-vector shape to standardize results */
    memset(info, 0, sizeof(*info)); /* nullify */
    switch(datatype) {
      case LIBXS_DATATYPE_F64: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE double
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXS_DATATYPE_F32: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE float
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXS_DATATYPE_I32: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE int
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXS_DATATYPE_I16: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE short
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXS_DATATYPE_I8: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE signed char
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: unsupported data-type requested for libxs_matdiff!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* square-root without libm dependency */
    int i;
    if (0 < info->l2_abs) {
      const double squared = info->l2_abs; info->l2_abs *= 0.5;
      for (i = 0; i < 16; ++i) info->l2_abs = 0.5 * (info->l2_abs + squared / info->l2_abs);
    }
    if (0 < info->l2_rel) {
      const double squared = info->l2_rel; info->l2_rel *= 0.5;
      for (i = 0; i < 16; ++i) info->l2_rel = 0.5 * (info->l2_rel + squared / info->l2_rel);
    }
    if (0 < info->normf_rel) {
      const double squared = info->normf_rel; info->normf_rel *= 0.5;
      for (i = 0; i < 16; ++i) info->normf_rel = 0.5 * (info->normf_rel + squared / info->normf_rel);
    }
    if (1 == n) {
      const libxs_blasint tmp = info->linf_abs_m;
      info->linf_abs_m = info->linf_abs_n;
      info->linf_abs_n = tmp;
    }
  }
  return result;
}


#if defined(LIBXS_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_init)(void);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_init)(void)
{
  libxs_init();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_finalize)(void);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_finalize)(void)
{
  libxs_finalize();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* fn,
  const libxs_gemm_precision* precision, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_xmmdispatch)(intptr_t* fn,
  const libxs_gemm_precision* precision, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
#if !defined(NDEBUG) /* this should not happen */
  static int error_once = 0;
  if (0 != fn && 0 != m)
#endif
  {
    const libxs_gemm_precision gemm_precision = (0 != precision ? *precision : LIBXS_GEMM_PRECISION_F64);
    const libxs_blasint kk = *(0 != k ? k : m), nn = (0 != n ? *n : kk);
    switch (gemm_precision) {
      case LIBXS_GEMM_PRECISION_F64: {
        *fn = (intptr_t)libxs_dmmdispatch(*m, nn, kk, lda, ldb, ldc,
          (const double*)alpha, (const double*)beta,
          flags, prefetch);
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
        *fn = (intptr_t)libxs_smmdispatch(*m, nn, kk, lda, ldb, ldc,
          (const float*)alpha, (const float*)beta,
          flags, prefetch);
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
        *fn = (intptr_t)libxs_wmmdispatch(*m, nn, kk, lda, ldb, ldc,
          (const int*)alpha, (const int*)beta,
          flags, prefetch);
      } break;
#if !defined(NDEBUG) /* this should not happen */
      default: {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: invalid precision requested for libxs_xmmdispatch!\n");
        }
        *fn = 0;
      }
#endif
    }
  }
#if !defined(NDEBUG)
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid M, N, or K passed into libxs_xmmdispatch!\n");
    }
    if (0 != fn) *fn = 0;
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xmmcall_abc)(
  const libxs_code_pointer* fn, const void* a, const void* b, void* c);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_xmmcall_abc)(
  const libxs_code_pointer* fn, const void* a, const void* b, void* c)
{
#if !defined(NDEBUG) /* this should not happen */
  static int error_once = 0;
  if (0 != fn && 0 != a && 0 != b && 0 != c)
#endif
  {
#if !defined(NDEBUG) /* this should not happen */
    if (0 != fn->xgemm.xmm)
#endif
    {
      fn->xgemm.xmm(a, b, c);
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
  const libxs_code_pointer* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_xmmcall_prf)(
  const libxs_code_pointer* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
#if !defined(NDEBUG) /* this should not happen */
  static int error_once = 0;
  if (0 != fn && 0 != a && 0 != b && 0 != c)
#endif
  {
#if !defined(NDEBUG) /* this should not happen */
    if (0 != fn->xgemm.xmm)
#endif
    {
      fn->xgemm.xmm(a, b, c, pa, pb, pc);
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
  const libxs_code_pointer* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_xmmcall)(
  const libxs_code_pointer* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
  LIBXS_FSYMBOL(libxs_xmmcall_prf)(fn, a, b, c, pa, pb, pc);
}

#endif /*defined(LIBXS_BUILD)*/
