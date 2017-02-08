/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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
#include "libxs_trans.h"
#include "libxs_gemm.h"
#include "libxs_hash.h"
#include "libxs_main.h"
#if defined(__TRACE)
# include "libxs_trace.h"
#endif
#if defined(LIBXS_PERF)
# include "libxs_perf.h"
#endif
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

/* alternative hash algorithm (instead of CRC32) */
#if !defined(LIBXS_HASH_BASIC)
# if (LIBXS_X86_SSE4 > LIBXS_MAX_STATIC_TARGET_ARCH)
/*#   define LIBXS_HASH_BASIC*/
# endif
#endif

/* LIBXS_CAPACITY_REGISTRY is POT */
/*#define LIBXS_HASH_MOD(N, NGEN) ((N) % (NGEN))*/
#define LIBXS_HASH_MOD(N, NPOT) LIBXS_MOD2(N, NPOT)

#if !defined(LIBXS_CAPACITY_CACHE)
# define LIBXS_CAPACITY_CACHE 4
#endif

#if defined(LIBXS_HASH_BASIC)
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxs_hash_npot(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, LIBXS_CAPACITY_REGISTRY); \
    assert((LIBXS_CAPACITY_REGISTRY) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxs_crc32(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXS_HASH_MOD(HASH, LIBXS_CAPACITY_REGISTRY)
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 0 /* disabled due to no performance advantage */
#define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif

#if 16 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE LIBXS_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXS_RETARGETABLE internal_regkey_type {
  char simd[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxs_gemm_descriptor descriptor;
} internal_regkey_type;

typedef struct LIBXS_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXS_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXS_PREFETCH_NONE
#else
# define INTERNAL_PREFETCH LIBXS_PREFETCH
#endif

#if defined(LIBXS_NO_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE)
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX)
#else
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
  const unsigned int LOCKINDEX = LIBXS_MOD2(INDEX, INTERNAL_REGLOCK_COUNT); \
  if (LIBXS_LOCK_ACQUIRED != LIBXS_LOCK_TRYLOCK(internal_reglock + (LOCKINDEX))) { \
    if (0 == libxs_dispatch_trylock) { /* (re-)try and get (meanwhile) generated code */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; \
      break; \
    } \
  }
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(internal_reglock + (LOCKINDEX)); }
#endif

#if defined(LIBXS_GEMM_DIFF_SW) && (2 == (LIBXS_GEMM_DIFF_SW)) /* most general implementation */
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    RESULT_INDEX = ((CACHE_HIT) + ((LIBXS_CAPACITY_CACHE) - 1)) % (LIBXS_CAPACITY_CACHE)
#else
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    assert(/*is pot*/(LIBXS_CAPACITY_CACHE) == (1 << LIBXS_LOG2(LIBXS_CAPACITY_CACHE))); \
    RESULT_INDEX = LIBXS_MOD2((CACHE_HIT) + ((LIBXS_CAPACITY_CACHE) - 1), LIBXS_CAPACITY_CACHE)
#endif

#define INTERNAL_DISPATCH_MAIN(TYPE, DESCRIPTOR_DECL, DESC, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) { \
  const int internal_dispatch_main_flags_ = (0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)) | LIBXS_GEMM_TYPEFLAG(TYPE); \
  const int internal_dispatch_main_lda_ = (0 == LIBXS_LD(PLDA, PLDB) ? LIBXS_LD(M, N) : *LIBXS_LD(PLDA, PLDB)); \
  const int internal_dispatch_main_ldb_ = (0 == LIBXS_LD(PLDB, PLDA) ? (K) : *LIBXS_LD(PLDB, PLDA)); \
  const int internal_dispatch_main_ldc_ = (0 == (PLDC) ? LIBXS_LD(M, N) : *(PLDC)); \
  const TYPE internal_dispatch_main_alpha_ = (0 == (PALPHA) ? ((TYPE)LIBXS_ALPHA) : *(PALPHA)); \
  const TYPE internal_dispatch_main_beta_ = (0 == (PBETA) ? ((TYPE)LIBXS_BETA) : *(PBETA)); \
  if (LIBXS_GEMM_NO_BYPASS(internal_dispatch_main_flags_, internal_dispatch_main_alpha_, internal_dispatch_main_beta_) && LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) && \
    LIBXS_GEMM_NO_BYPASS_DIMS(internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_)) \
  { \
    const int internal_dispatch_main_prefetch_ = (0 == (PREFETCH) ? libxs_gemm_auto_prefetch : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXS_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXS_ALIGNMENT, \
      internal_dispatch_main_flags_, LIBXS_LD(M, N), LIBXS_LD(N, M), K, internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_, \
      (signed char)(internal_dispatch_main_alpha_), (signed char)(internal_dispatch_main_beta_), \
      (0 > internal_dispatch_main_prefetch_ ? internal_gemm_auto_prefetch : internal_dispatch_main_prefetch_)); \
    { \
      return internal_find_code(DESC).LIBXS_TPREFIX(TYPE, mm); \
    } \
  } \
  else { /* bypass (not supported) */ \
    /* libxs_gemm_print is not suitable here since A, B, and C are unknown at this point */ \
    libxs_update_mmstatistic(internal_dispatch_main_flags_, LIBXS_LD(M, N), LIBXS_LD(N, M), K, 1/*try*/, 0); \
    return 0; \
  } \
}

#if defined(LIBXS_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXS_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
    INTERNAL_DISPATCH_MAIN(TYPE, libxs_gemm_descriptor descriptor = { 0 }, &descriptor, \
      PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#else /* padding: LIBXS_GEMM_DESCRIPTOR_SIZE -> LIBXS_ALIGNMENT */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
    INTERNAL_DISPATCH_MAIN(TYPE, union { libxs_gemm_descriptor desc; char simd[LIBXS_ALIGNMENT]; } simd_descriptor; int i; \
      for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
      PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#endif

#if !defined(LIBXS_NO_SYNC)
# define INTERNAL_REGLOCK_COUNT 256
LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_LOCK_TYPE internal_reglock[INTERNAL_REGLOCK_COUNT];
#endif

LIBXS_EXTERN_C LIBXS_RETARGETABLE internal_regkey_type* internal_registry_keys;
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_code_pointer* internal_registry;
LIBXS_EXTERN_C LIBXS_RETARGETABLE internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/];
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int internal_statistic_sml;
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int internal_statistic_med;
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int internal_statistic_mnk;
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int internal_teardown;
LIBXS_EXTERN_C LIBXS_RETARGETABLE size_t internal_heapmem;
LIBXS_EXTERN_C LIBXS_RETARGETABLE int internal_dispatch_trylock_locked;
LIBXS_EXTERN_C LIBXS_RETARGETABLE int internal_gemm_auto_prefetch_locked;
LIBXS_EXTERN_C LIBXS_RETARGETABLE int internal_gemm_auto_prefetch;


LIBXS_API_DEFINITION unsigned int libxs_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol)
{
  const unsigned long long kernel_size = LIBXS_MNK_SIZE(m, n, k);
  const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & flags) ? 0 : 1);
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

  LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ncol, ncol, LIBXS_ATOMIC_RELAXED);
  return LIBXS_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ntry, ntry, LIBXS_ATOMIC_RELAXED);
}


LIBXS_INLINE LIBXS_RETARGETABLE unsigned int internal_update_mmstatistic(const libxs_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol)
{
  assert(0 != desc);
  return libxs_update_mmstatistic(desc->flags, desc->m, desc->n, desc->k, ntry, ncol);
}


LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int id);
LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int id)
{
  const char* target_arch = 0;
  switch (id) {
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


LIBXS_INLINE LIBXS_RETARGETABLE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
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


LIBXS_INLINE LIBXS_RETARGETABLE unsigned int internal_print_statistic(FILE* ostream,
  const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  const internal_statistic_type statistic_sml = internal_statistic[precision][0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic[precision][1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic[precision][2/*BIG*/];
  const internal_statistic_type statistic_xxx = internal_statistic[precision][3/*XXX*/];
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

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
      assert(strlen(target_arch) < sizeof(title));
      for (n = 0; 0 != target_arch[n] /*avoid code-gen. issue with some clang versions: && n < sizeof(title)*/; ++n) {
        const char c = target_arch[n];
        title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c); /* toupper */
      }
      LIBXS_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-8s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY" ,"JIT", "STA", "COL");
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


LIBXS_INLINE LIBXS_RETARGETABLE unsigned int internal_statistic_ntry(int precision)
{
  return internal_statistic[precision][0/*SML*/].ntry + internal_statistic[precision][1/*MED*/].ntry
       + internal_statistic[precision][2/*BIG*/].ntry + internal_statistic[precision][3/*XXX*/].ntry;
}


LIBXS_API void internal_register_static_code(const libxs_gemm_descriptor*,
  unsigned int, unsigned int, libxs_xmmfunction, libxs_code_pointer*);
LIBXS_API_DEFINITION void internal_register_static_code(const libxs_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxs_xmmfunction src, libxs_code_pointer* registry)
{
  internal_regkey_type* dst_key = internal_registry_keys + index;
  libxs_code_pointer* dst_entry = registry + index;
#if !defined(NDEBUG)
  libxs_code_pointer code; code.xmm = src;
  assert(0 != desc && 0 != code.const_pmm && 0 != dst_key && 0 != registry);
  assert(0 == (LIBXS_CODE_STATIC & code.uimm));
#endif

  if (0 != dst_entry->const_pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_CAPACITY_REGISTRY);
    unsigned int i0, i, next;
#if defined(LIBXS_HASH_COLLISION)
    /* mark current entry as a collision (this might be already the case) */
    dst_entry->uimm |= LIBXS_HASH_COLLISION;
#endif
    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXS_HASH_MOD(start + 1, LIBXS_CAPACITY_REGISTRY), i0 = i, next = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY);
      0 != registry[i].const_pmm && next != i0; i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY));

    /* calculate destinations */
    dst_key = internal_registry_keys + i;
    dst_entry = registry + i;

    internal_update_mmstatistic(desc, 0, 1/*collision*/);
  }

  if (0 == dst_entry->const_pmm) { /* registry not (yet) exhausted */
    dst_key->descriptor = *desc;
    dst_entry->xmm = src;
    /* mark current entry as static code (non-JIT) */
    dst_entry->uimm |= LIBXS_CODE_STATIC;
  }

  internal_update_mmstatistic(desc, 1/*try*/, 0);
}


LIBXS_API_DEFINITION int libxs_gemm_prefetch2uid(libxs_gemm_prefetch_type prefetch)
{
  switch (prefetch) {
    case LIBXS_PREFETCH_SIGONLY:            return 2;
    case LIBXS_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXS_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXS_PREFETCH_AL2:                return 6;
    case LIBXS_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXS_PREFETCH_AL2_JPST:           return 8;
    case LIBXS_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    case LIBXS_PREFETCH_AL2CL2BL2_VIA_C:    return 10;
    default: {
      assert(LIBXS_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXS_API_DEFINITION libxs_gemm_prefetch_type libxs_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  2: return LIBXS_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXS_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXS_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXS_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXS_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXS_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXS_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    case 10: return LIBXS_PREFETCH_AL2CL2BL2_VIA_C;     /* AL2_BL2viaC_CL2 */
    default: return LIBXS_PREFETCH_NONE;
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_finalize(void)
{
  libxs_finalize();
  if (0 != libxs_verbosity) { /* print statistic on termination */
    fflush(stdout); /* synchronize with standard output */
    {
      const char *const target_arch = internal_get_target_arch(libxs_target_archid);
      const unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
        fprintf(stderr, "LIBXS_TARGET=%s ", target_arch);
      }
      fprintf(stderr, "HEAP: %.f MB\n", 1.0 * internal_heapmem / (1 << 20));
    }
  }
  {
    size_t n = 0;
    /* release scratch memory pool */
    libxs_release_scratch(&n);
    if (0 < n && 0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS: pending scratch-memory allocations discovered!\n");
    }
#if !defined(LIBXS_NO_SYNC) /* release locks */
    for (n = 0; n < INTERNAL_REGLOCK_COUNT; ++n) LIBXS_LOCK_DESTROY(internal_reglock + n);
    LIBXS_LOCK_DESTROY(&libxs_lock_global);
#endif
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_init(void)
{
  libxs_code_pointer* result;
  int init_code = EXIT_FAILURE, i;
#if !defined(LIBXS_NO_SYNC) /* setup the locks in a thread-safe fashion */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock + i);
  LIBXS_LOCK_ACQUIRE(&libxs_lock_global);
#endif
  result = internal_registry;
  if (0 == result) {
    const libxs_malloc_function null_malloc_fn = { 0 };
    const libxs_free_function null_free_fn = { 0 };
    libxs_xset_default_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
    libxs_xset_scratch_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
    libxs_set_target_arch(getenv("LIBXS_TARGET")); /* set libxs_target_archid */
    libxs_mt = 2;
    { /* behavior of parallelized routines which are located in libxsext library
       * 0: sequential below-threshold routine (no OpenMP); may fall-back to BLAS,
       * 1: (OpenMP-)parallelized but without internal parallel region,
       * 2: (OpenMP-)parallelized with internal parallel region"
       */
      const char *const env = getenv("LIBXS_MT");
      if (0 != env && 0 != *env) {
        libxs_mt = atoi(env);
      }
    }
    { const char *const env = getenv("LIBXS_TASKS");
      if (0 != env && 0 != *env) {
        libxs_tasks = atoi(env);
      }
    }
    { const char *const env = getenv("LIBXS_TRYLOCK");
      if (0 != env && 0 != *env) {
        libxs_dispatch_trylock = atoi(env);
        internal_dispatch_trylock_locked = 1;
      }
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
      internal_statistic_mnk = (unsigned int)(pow((double)(LIBXS_MAX_MNK), 0.3333333333333333) + 0.5);
      internal_statistic_sml = 13; internal_statistic_med = 23;
      if (0 != env && 0 != *env) {
        libxs_verbosity = atoi(env);
      }
#if !defined(NDEBUG)
      else {
        libxs_verbosity = INT_MAX - 1; /* quiet -> verbose */
      }
#endif
    }
#if !defined(__TRACE)
    LIBXS_UNUSED(init_code);
#else
    {
      int filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = 0;
      const char *const env = getenv("LIBXS_TRACE");
      if (0 != env && 0 != *env) {
        char buffer[32];
        if (1 == sscanf(env, "%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_threadid);
        }
        if (1 == sscanf(env, "%*[^,],%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_mindepth);
        }
        if (1 == sscanf(env, "%*[^,],%*[^,],%32s", buffer)) {
          sscanf(buffer, "%i", &filter_maxnsyms);
        }
        else {
          filter_maxnsyms = -1; /* all */
        }
      }
      init_code = libxs_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
    }
    if (EXIT_SUCCESS == init_code)
#endif
    {
      libxs_gemm_diff_init(libxs_target_archid);
      libxs_trans_init(libxs_target_archid);
      libxs_hash_init(libxs_target_archid);
#if defined(LIBXS_PERF)
      libxs_perf_init();
#endif
      assert(0 == internal_registry_keys && 0 == internal_registry); /* should never happen */
      result = (libxs_code_pointer*)malloc((LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer));
      internal_registry_keys = (internal_regkey_type*)malloc((LIBXS_CAPACITY_REGISTRY) * sizeof(internal_regkey_type));
      if (0 != result && 0 != internal_registry_keys) {
        const char *const env = getenv("LIBXS_GEMM_PREFETCH");
        for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) result[i].pmm = 0;
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
        internal_gemm_auto_prefetch = (0 == internal_statistic_ntry(0/*DP*/) && 0 == internal_statistic_ntry(1/*SP*/))
          /* avoid special prefetch if static code is present, since such code uses INTERNAL_PREFETCH */
          ? (LIBXS_X86_AVX512_MIC != libxs_target_archid ? LIBXS_PREFETCH_AL2BL2_VIA_C : LIBXS_PREFETCH_BL2_VIA_C)
          : INTERNAL_PREFETCH;
        libxs_gemm_auto_prefetch = INTERNAL_PREFETCH;
        if (0 != env && 0 != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            internal_gemm_auto_prefetch = libxs_gemm_uid2prefetch(uid);
            libxs_gemm_auto_prefetch = internal_gemm_auto_prefetch;
            internal_gemm_auto_prefetch_locked = 1;
          }
        }
        libxs_gemm_init(libxs_target_archid, libxs_gemm_auto_prefetch);
        if (0 == internal_teardown) {
          atexit(internal_finalize);
        }
        {
          void *const pv_registry = &internal_registry;
          LIBXS_ATOMIC_STORE((void**)pv_registry, (void*)result, LIBXS_ATOMIC_SEQ_CST);
        }
      }
      else {
        if (0 != libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS: failed to allocate code registry!\n");
        }
        free(internal_registry_keys);
        free(result);
      }
    }
#if defined(__TRACE)
    else if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS: failed to initialize TRACE (error #%i)!\n", init_code);
    }
#endif
  }
#if !defined(LIBXS_NO_SYNC) /* release locks */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_RELEASE(internal_reglock + i);
  LIBXS_LOCK_RELEASE(&libxs_lock_global);
#endif
}


LIBXS_API_DEFINITION LIBXS_ATTRIBUTE_CTOR void libxs_init(void)
{
  const void *const registry = LIBXS_ATOMIC_LOAD(&internal_registry, LIBXS_ATOMIC_RELAXED);
  if (0 == registry) {
#if !defined(LIBXS_NO_SYNC) /* setup the locks in a thread-safe fashion */
    static int reglock_check = 0;
    int i;
    assert(sizeof(internal_reglock) == (INTERNAL_REGLOCK_COUNT * sizeof(*internal_reglock)));
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&reglock_check, 1, LIBXS_ATOMIC_SEQ_CST)) {
      for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_INIT(internal_reglock + i);
      LIBXS_LOCK_INIT(&libxs_lock_global);
    }
    else { /* wait until locks are initialized, or until shutdown */
      while (0 == internal_registry && 0 == internal_teardown) {
        if (0 != LIBXS_ATOMIC_LOAD(&internal_registry, LIBXS_ATOMIC_RELAXED)) break;
        if (0 != LIBXS_ATOMIC_LOAD(&internal_teardown, LIBXS_ATOMIC_RELAXED)) break;
      }
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

LIBXS_API_DEFINITION LIBXS_ATTRIBUTE_DTOR void libxs_finalize(void)
{
  libxs_code_pointer* registry = LIBXS_ATOMIC_LOAD(&internal_registry, LIBXS_ATOMIC_SEQ_CST);
  if (0 != registry) {
    int i;
#if !defined(LIBXS_NO_SYNC)
    /* acquire locks and thereby shortcut lazy initialization later on */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock + i);
#endif
    registry = internal_registry;

    if (0 != registry) {
      internal_regkey_type *const registry_keys = internal_registry_keys;
      internal_heapmem = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(internal_regkey_type));

      /* serves as an id to invalidate the thread-local cache; never decremented */
      ++internal_teardown;
#if defined(__TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS: failed to finalize trace (error #%i)!\n", i);
      }
#endif
      libxs_gemm_finalize();
      libxs_gemm_diff_finalize();
      libxs_trans_finalize();
      libxs_hash_finalize();
#if defined(LIBXS_PERF)
      libxs_perf_finalize();
#endif
      /* make internal registry globally unavailable */
      LIBXS_ATOMIC_STORE_ZERO(&internal_registry, LIBXS_ATOMIC_SEQ_CST);
      internal_registry_keys = 0;

      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        libxs_code_pointer code = registry[i];
        if (0 != code.const_pmm) {
          const libxs_gemm_descriptor *const desc = &registry_keys[i].descriptor;
          const unsigned long long kernel_size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
          const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
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
          if (0 == (LIBXS_CODE_STATIC & code.uimm)) { /* check for allocated/generated JIT-code */
            void* buffer = 0;
            size_t size = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uimm &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            if (EXIT_SUCCESS == libxs_malloc_info(code.const_pmm, &size, 0/*flags*/, &buffer)) {
              libxs_xfree(code.const_pmm);
              ++internal_statistic[precision][bucket].njit;
              internal_heapmem += (unsigned int)(size + (((char*)code.const_pmm) - (char*)buffer));
            }
          }
          else {
            ++internal_statistic[precision][bucket].nsta;
          }
        }
      }
      free(registry_keys);
      free(registry);
    }
#if !defined(LIBXS_NO_SYNC) /* LIBXS_LOCK_RELEASE, but no LIBXS_LOCK_DESTROY */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_RELEASE(internal_reglock + i);
#endif
  }
  /* release scratch memory pool */
  libxs_release_scratch(0);
}


LIBXS_API_DEFINITION int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return libxs_target_archid;
#else /* no JIT support */
  return LIBXS_MIN(libxs_target_archid, LIBXS_X86_SSE4);
#endif
}


LIBXS_API_DEFINITION void libxs_set_target_archid(int id)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXS_X86_AVX512_KNM:
    case LIBXS_X86_AVX512_CORE:
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
    if (cpuid < libxs_target_archid) {
      const char *const target_arch = internal_get_target_arch(libxs_target_archid);
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
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
    else if (0 == strcmp("skx", arch) || 0 == strcmp("skl", arch)) {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knm", arch) || 0 == strcmp("mic2", arch)) {
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
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch) || 0 == strcmp("sse4", arch) || 0 == strcmp("sse4_2", arch) || 0 == strcmp("sse4.2", arch)) {
      target_archid = LIBXS_X86_SSE4;
    }
    else if (0 == strcmp("sse3", arch) || 0 == strcmp("sse", arch)) {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXS_TARGET_ARCH_UNKNOWN == target_archid || LIBXS_X86_AVX512_KNM < target_archid) {
    target_archid = libxs_cpuid();
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxs_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
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
  return libxs_dispatch_trylock;
}


LIBXS_API_DEFINITION void libxs_set_dispatch_trylock(int trylock)
{
  LIBXS_INIT
  if (0 == internal_dispatch_trylock_locked) { /* LIBXS_TRYLOCK environment takes precedence */
    LIBXS_ATOMIC_STORE(&libxs_dispatch_trylock, trylock, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_API_DEFINITION libxs_gemm_prefetch_type libxs_get_gemm_auto_prefetch(void)
{
  return (libxs_gemm_prefetch_type)libxs_gemm_auto_prefetch;
}


LIBXS_API_DEFINITION void libxs_set_gemm_auto_prefetch(libxs_gemm_prefetch_type strategy)
{
  if (0 == internal_gemm_auto_prefetch_locked) { /* LIBXS_GEMM_PREFETCH environment takes precedence */
    LIBXS_ATOMIC_STORE(&internal_gemm_auto_prefetch, strategy, LIBXS_ATOMIC_RELAXED);
    LIBXS_ATOMIC_STORE(&libxs_gemm_auto_prefetch, strategy, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_API const char* internal_get_precision_string(libxs_dnn_datatype);
LIBXS_API_DEFINITION const char* internal_get_precision_string(libxs_dnn_datatype datatype)
{
  const char* result = "unk"; /* unknown */
  switch (datatype) {
    case LIBXS_DNN_DATATYPE_F32: result = "f32"; break;
    case LIBXS_DNN_DATATYPE_I32: result = "i32"; break;
    case LIBXS_DNN_DATATYPE_I16: result = "i16"; break;
    case LIBXS_DNN_DATATYPE_I8:  result = "i8";  break;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  const char *const target_arch = internal_get_target_arch(libxs_target_archid);
  libxs_generated_code generated_code;
  char jit_name[256] = { 0 };

  assert(0 != request && 0 != libxs_target_archid);
  assert(0 != code && 0 == code->const_pmm);
  /* setup code generation */
  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.code_type = 2;

  switch (request->kind) { /* generate kernel */
    case LIBXS_BUILD_KIND_GEMM: { /* small MxM kernel */
      assert(0 != request->descriptor.gemm);
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_gemm_kernel, &generated_code, request->descriptor.gemm, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.mxm", target_arch/*code path name*/,
            0 == (LIBXS_GEMM_FLAG_F32PREC & request->descriptor.gemm->flags) ? "f64" : "f32",
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.gemm->m,   (unsigned int)request->descriptor.gemm->n,   (unsigned int)request->descriptor.gemm->k,
            (unsigned int)request->descriptor.gemm->lda, (unsigned int)request->descriptor.gemm->ldb, (unsigned int)request->descriptor.gemm->ldc,
            request->descriptor.gemm->alpha, request->descriptor.gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return result;
      }
    } break;
    case LIBXS_BUILD_KIND_SSOA: { /* sparse SOA kernel */
      assert(0 != request->descriptor.ssoa && 0 != request->descriptor.ssoa->gemm);
      assert(0 != request->descriptor.ssoa->row_ptr && 0 != request->descriptor.ssoa->column_idx && 0 != request->descriptor.ssoa->values);
      if (0 == (LIBXS_GEMM_FLAG_F32PREC & (request->descriptor.ssoa->gemm->flags))/*only double-precision*/) {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_soa_kernel, &generated_code, request->descriptor.ssoa->gemm, target_arch,
          request->descriptor.ssoa->row_ptr, request->descriptor.ssoa->column_idx,
          (const double*)request->descriptor.ssoa->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.ssoa->gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.ssoa", target_arch/*code path name*/,
            0 == (LIBXS_GEMM_FLAG_F32PREC & request->descriptor.ssoa->gemm->flags) ? "f64" : "f32",
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.ssoa->gemm->m,   (unsigned int)request->descriptor.ssoa->gemm->n,   (unsigned int)request->descriptor.ssoa->gemm->k,
            (unsigned int)request->descriptor.ssoa->gemm->lda, (unsigned int)request->descriptor.ssoa->gemm->ldb, (unsigned int)request->descriptor.ssoa->gemm->ldc,
            request->descriptor.ssoa->gemm->alpha, request->descriptor.ssoa->gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return result;
      }
    } break;
    case LIBXS_BUILD_KIND_SREG: { /* sparse register kernel */
      assert(0 != request->descriptor.sreg && 0 != request->descriptor.ssoa->gemm);
      assert(0 != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
#if 1
      if (0 == (LIBXS_GEMM_FLAG_F32PREC & (request->descriptor.sreg->gemm->flags))/*only double-precision*/) {
#endif
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_spgemm_csr_reg_kernel, &generated_code, request->descriptor.sreg->gemm, target_arch,
          request->descriptor.sreg->row_ptr, request->descriptor.sreg->column_idx,
          (const double*)request->descriptor.sreg->values);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const int uid = libxs_gemm_prefetch2uid((libxs_gemm_prefetch_type)request->descriptor.ssoa->gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.sreg", target_arch/*code path name*/,
            0 == (LIBXS_GEMM_FLAG_F32PREC & request->descriptor.sreg->gemm->flags) ? "f64" : "f32",
            0 == (LIBXS_GEMM_FLAG_TRANS_A & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.sreg->gemm->m,   (unsigned int)request->descriptor.sreg->gemm->n,   (unsigned int)request->descriptor.sreg->gemm->k,
            (unsigned int)request->descriptor.sreg->gemm->lda, (unsigned int)request->descriptor.sreg->gemm->ldb, (unsigned int)request->descriptor.sreg->gemm->ldc,
            request->descriptor.sreg->gemm->alpha, request->descriptor.sreg->gemm->beta, uid);
        }
#if 1
      }
      else { /* this case is not an actual error */
        return result;
      }
#endif
    } break;
    case LIBXS_BUILD_KIND_CFWD: { /* forward convolution */
      assert(0 != request->descriptor.cfwd);
      if (0 < request->descriptor.cfwd->kw && 0 < request->descriptor.cfwd->kh &&
          0 != request->descriptor.cfwd->stride_w && 0 != request->descriptor.cfwd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_forward_kernel, &generated_code, request->descriptor.cfwd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cfwd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cfwd->datatype_itm);
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
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_backward_kernel, &generated_code, request->descriptor.cbwd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cbwd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cbwd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_bwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%u_v%u_pa%u_p%i_f%i.conv",
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
            (unsigned int)request->descriptor.cbwd->ofw/*ofw*/, (unsigned int)request->descriptor.cbwd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cbwd->peeled/*peeled version*/,
            (unsigned int)request->descriptor.cbwd->prefetch_output_ahead/*prefetch kj outputs for jumping from non-peel to peel version*/,
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
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_weight_update_kernel, &generated_code, request->descriptor.cupd, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cupd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cupd->datatype_itm);
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
    case LIBXS_BUILD_KIND_CWFWD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wfwd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CWBWD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wbwd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXS_BUILD_KIND_CWUPD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXS_NO_OFFLOAD(void, libxs_generator_convolution_winograd_weight_update_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXS_VTUNE)
        if (0 > libxs_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXS_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXS to appear in order (Intel VTune, etc.) */
          LIBXS_SNPRINTF(jit_name, sizeof(jit_name), "libxs_%s_wupd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS: invalid build request discovered!\n");
      }
      result = EXIT_FAILURE;
    }
# endif
  }

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
    assert(0 < generated_code.code_size/*sanity check*/);
    /* attempt to create executable buffer */
    result = libxs_xmalloc(&code->pmm, generated_code.code_size, 0/*auto*/,
      /* flag must be a superset of what's populated by libxs_malloc_attrib */
      LIBXS_MALLOC_FLAG_RWX, &regindex, sizeof(regindex));
    if (EXIT_SUCCESS == result) { /* check for success */
      assert(0 != code->const_pmm && 0 == (LIBXS_CODE_STATIC & code->uimm));
      assert(0 != generated_code.generated_code/*sanity check*/);
      /* copy temporary buffer into the prepared executable buffer */
      memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);
      /* attribute/protect buffer and revoke unnecessary flags */
      result = libxs_malloc_attrib(&code->pmm, LIBXS_MALLOC_FLAG_X, jit_name);
    }
  }
  else {
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        LIBXS_NO_OFFLOAD(int, fprintf, stderr, "%s (error #%u)\n",
          LIBXS_NO_OFFLOAD(const char*, libxs_strerror, generated_code.last_error),
          generated_code.last_error);
      }
    }
    result = EXIT_FAILURE;
  }
  free(generated_code.generated_code); /* free temporary/initial code buffer */
#else /* unsupported platform */
  LIBXS_UNUSED(request); LIBXS_UNUSED(regindex); LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXS_X86_AVX <= libxs_target_archid) result = EXIT_FAILURE;
#endif
  return result;
}


/** This function only works for JIT-generated code! */
LIBXS_API const libxs_gemm_descriptor* internal_get_gemm_descriptor(const void* gemm_jit);
LIBXS_API_DEFINITION const libxs_gemm_descriptor* internal_get_gemm_descriptor(const void* gemm_jit)
{
  const libxs_gemm_descriptor* result = 0;
  void* extra = 0;
  if (EXIT_SUCCESS == libxs_malloc_info(gemm_jit, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int i = *((const unsigned int*)extra);
    result = &internal_registry_keys[i].descriptor;
  }
  return result;
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_xmmfunction internal_find_code(const libxs_gemm_descriptor* descriptor)
{
  libxs_code_pointer flux_entry = { 0 };
  unsigned int hash, i0, i = 0, mode = 0, diff = 1;
#if !defined(NDEBUG)
  const libxs_gemm_descriptor* refdesc = 0;
#endif
#if defined(LIBXS_CAPACITY_CACHE) && (0 < (LIBXS_CAPACITY_CACHE))
  static LIBXS_TLS struct {
    union { char padding[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE]; libxs_gemm_descriptor desc; } keys[LIBXS_CAPACITY_CACHE];
    libxs_code_pointer code[LIBXS_CAPACITY_CACHE];
    unsigned int hit, id;
  } cache;
  unsigned int cache_index;
  assert(0 != descriptor && LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE >= LIBXS_GEMM_DESCRIPTOR_SIZE);
  /* search small cache starting with the last hit on record */
  cache_index = libxs_gemm_diffn(descriptor, &cache.keys->desc, cache.hit, LIBXS_CAPACITY_CACHE, LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE);
  if ((LIBXS_CAPACITY_CACHE) > cache_index && cache.id == internal_teardown) { /* cache hit, and valid */
    flux_entry = cache.code[cache_index];
    cache.hit = cache_index;
#if !defined(NDEBUG)
    if (0 == (LIBXS_CODE_STATIC & flux_entry.uimm)) { /* JIT only */
# if defined(LIBXS_HASH_COLLISION)
      flux_entry.uimm &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
# endif
      refdesc = internal_get_gemm_descriptor(flux_entry.const_pmm);
    }
#endif
  }
  else
#else
  assert(0 != descriptor);
#endif
  {
    assert(0 != internal_registry);
    /* check if the requested xGEMM is already JITted */
    LIBXS_HASH_FUNCTION_CALL(hash, i = i0, *descriptor);
    while (0 != diff) {
      flux_entry.pmm = LIBXS_ATOMIC_LOAD(&internal_registry[i].pmm, LIBXS_ATOMIC_RELAXED); /* read registered code */
      if ((0 != flux_entry.const_pmm || 1 == mode) && 2 > mode) { /* check existing entry further */
        diff = 0 != flux_entry.const_pmm ? libxs_gemm_diff(descriptor, &internal_registry_keys[i].descriptor) : 1;
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXS_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXS_HASH_COLLISION & flux_entry.uimm)) {
              assert(0 != flux_entry.const_pmm); /* collision */
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
          assert(0 != mode || 0 == flux_entry.const_pmm/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.pmm); /* lock the registry entry */
          if (0 == internal_registry[i].const_pmm) { /* double-check registry after acquiring the lock */
            libxs_build_request request; /* setup the code build request */
            request.descriptor.gemm = descriptor; request.kind = LIBXS_BUILD_KIND_GEMM;
            internal_update_mmstatistic(descriptor, 1/*try*/, 0); /* count attempt */
            if (EXIT_SUCCESS == libxs_build(&request, i, &flux_entry) && 0 != flux_entry.const_pmm) {
              internal_registry_keys[i].descriptor = *descriptor;
              LIBXS_ATOMIC_STORE(&internal_registry[i].pmm, flux_entry.pmm, LIBXS_ATOMIC_RELAXED); /* sync */
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer fix_entry;
                fix_entry.pmm = LIBXS_ATOMIC_LOAD(&internal_registry[i0].pmm, LIBXS_ATOMIC_RELAXED);
                assert(0 != fix_entry.const_pmm);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uimm)) {
                  fix_entry.uimm |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
                  LIBXS_ATOMIC_STORE(&internal_registry[i0].pmm, fix_entry.pmm, LIBXS_ATOMIC_RELAXED);
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
            for (i = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY); i != i0 && 0 != internal_registry[i].const_pmm;
                 i = LIBXS_HASH_MOD(i + 1, LIBXS_CAPACITY_REGISTRY)); /* continue to linearly search code */
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* inside of locked region (do not use break!) */
            }
            flux_entry.pmm = 0; /* no result */
          }
        }
        else
#endif
        { /* leave the dispatch loop */
          flux_entry.pmm = 0;
          diff = 0;
        }
      }
    }
#if defined(LIBXS_CAPACITY_CACHE) && (0 < (LIBXS_CAPACITY_CACHE))
    if (0 != flux_entry.const_pmm) { /* keep code version on record (cache) */
      INTERNAL_FIND_CODE_CACHE_INDEX(cache.hit, cache_index);
      cache.keys[cache_index].desc = *descriptor;
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
    refdesc = &internal_registry_keys[i].descriptor;
#endif
  }
  assert(0 == flux_entry.const_pmm || 0 == refdesc || 0 == memcmp(refdesc, descriptor, LIBXS_GEMM_DESCRIPTOR_SIZE));
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uimm &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uimm &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
  return flux_entry.xmm;
}


LIBXS_API_DEFINITION int libxs_get_registry_info(libxs_registry_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
    LIBXS_INIT
    if (0 != internal_registry) {
      size_t i;
      memset(info, 0, sizeof(libxs_registry_info)); /* info->nstatic = 0; info->size = 0; */
      info->nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(internal_regkey_type));
      info->capacity = LIBXS_CAPACITY_REGISTRY;
      info->ncache = LIBXS_CAPACITY_CACHE;
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        libxs_code_pointer code = internal_registry[i];
        if (0 != code.const_pmm && EXIT_SUCCESS == result) {
          if (0 == (LIBXS_CODE_STATIC & code.uimm)) { /* check for allocated/generated JIT-code */
            size_t buffer_size = 0;
            void* buffer = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uimm &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            result = libxs_malloc_info(code.const_pmm, &buffer_size, 0/*flags*/, &buffer);
            if (EXIT_SUCCESS == result) {
              info->nbytes += (unsigned int)(buffer_size + (((char*)code.const_pmm) - (char*)buffer));
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


LIBXS_API_DEFINITION libxs_gemm_descriptor* libxs_create_dgemm_descriptor(char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta,
  libxs_gemm_prefetch_type strategy)
{
  libxs_gemm_descriptor *const result = (libxs_gemm_descriptor*)malloc(sizeof(libxs_gemm_descriptor));
  assert(0 != transa && 0 != transb && 0 != strchr("NnTt", transa) && 0 != strchr("NnTt", transb));
  /* filter alpha and beta values since the descriptor cannot store general real values */
  if (0 != result && 0 != LIBXS_GEMM_NO_BYPASS(0, alpha, beta)) {
    LIBXS_GEMM_DESCRIPTOR(*result, 1, LIBXS_GEMM_FLAG_F64PREC |
      (('T' == transa || 't' == transa) ? LIBXS_GEMM_FLAG_TRANS_A : 0) |
      (('T' == transb || 't' == transb) ? LIBXS_GEMM_FLAG_TRANS_B : 0),
      m, n, k, lda, ldb, ldc, alpha, beta, strategy);
  }
  return result;
}


LIBXS_API_DEFINITION void libxs_release_gemm_descriptor(const libxs_gemm_descriptor* descriptor)
{
  free((void*)descriptor);
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
      backend_descriptor.prefetch = (unsigned char)libxs_gemm_auto_prefetch;
      descriptor = &backend_descriptor;
    }
    result = internal_find_code(descriptor);
  }
  else { /* bypass (not supported) */
    internal_update_mmstatistic(descriptor, 1/*try*/, 0);
  }
  return result;
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API intptr_t libxsf_xmmdispatch(const libxs_gemm_precision* /*precision*/,
  const int* /*m*/, const int* /*n*/, const int* /*k*/, const int* /*lda*/, const int* /*ldb*/, const int* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/, const int* /*flags*/, const int* /*prefetch*/);
LIBXS_API_DEFINITION intptr_t libxsf_xmmdispatch(const libxs_gemm_precision* precision,
  const int* m, const int* n, const int* k, const int* lda, const int* ldb, const int* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
  const libxs_gemm_precision gemm_precision = (0 != precision ? *precision : LIBXS_GEMM_FLAG_F64PREC);
  static int error_once = 0;
  intptr_t result = 0;
#if !defined(NDEBUG) /* this should not happen */
  if ((0 == m || 0 == n || 0 == k)
   && 0 != libxs_verbosity /* library code is expected to be mute */
   && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS: invalid M, N, or K argument specified!\n");
  }
#endif
  switch (gemm_precision) {
    case LIBXS_GEMM_FLAG_F64PREC: {
      result = (intptr_t)libxs_dmmdispatch(*m, *n, *k, lda, ldb, ldc,
        (const double*)alpha, (const double*)beta,
        flags, prefetch);
    } break;
    case LIBXS_GEMM_FLAG_F32PREC: {
      result = (intptr_t)libxs_smmdispatch(*m, *n, *k, lda, ldb, ldc,
        (const float*)alpha, (const float*)beta,
        flags, prefetch);
    } break;
    default: if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS: invalid GEMM precision specified!\n");
    }
  }
  return result;
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void libxsf_xmmcall(
  const intptr_t* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/,
  const void* /*pa*/, const void* /*pb*/, const void* /*pc*/);
LIBXS_API_DEFINITION void libxsf_xmmcall(
  const intptr_t* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
  libxs_code_pointer code_pointer = { 0 };
  static int error_once = 0;
#if !defined(NDEBUG) /* this should not happen */
  if ((0 == fn || 0 == a || 0 == b || 0 == c)
   && 0 != libxs_verbosity /* library code is expected to be mute */
   && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS: invalid arguments for libxs_xmmcall specified!\n");
  }
#endif
  if (0 != *fn) {
    code_pointer.imm = *fn;
    code_pointer.vmm(a, b, c, pa, pb, pc);
  }
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS: NULL-function passed into libxs_xmmcall!\n");
  }
}

#if !defined(LIBXS_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXS_PRAGMA_OPTIMIZE_OFF
#endif

LIBXS_API_DEFINITION libxs_smmfunction libxs_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  LIBXS_INIT
  INTERNAL_DISPATCH(float, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_API_DEFINITION libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  LIBXS_INIT
  INTERNAL_DISPATCH(double, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}

#if !defined(LIBXS_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXS_PRAGMA_OPTIMIZE_ON
#endif

LIBXS_API_DEFINITION libxs_xmmfunction libxs_create_dcsr_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxs_code_pointer code = { 0 };
  libxs_csr_soa_descriptor ssoa;
  libxs_build_request request;
  LIBXS_INIT
  ssoa.gemm = descriptor;
  ssoa.row_ptr = row_ptr;
  ssoa.column_idx = column_idx;
  ssoa.values = values;
  request.descriptor.ssoa = &ssoa;
  request.kind = LIBXS_BUILD_KIND_SSOA;
  libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
  return code.xmm;
}


LIBXS_API_DEFINITION libxs_dmmfunction libxs_create_dcsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxs_code_pointer code = { 0 };
  libxs_csr_reg_descriptor sreg;
  libxs_build_request request;
  LIBXS_INIT
  sreg.gemm = descriptor;
  sreg.row_ptr = row_ptr;
  sreg.column_idx = column_idx;
  sreg.values = values;
  request.descriptor.sreg = &sreg;
  request.kind = LIBXS_BUILD_KIND_SREG;
  libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
  return code.xmm.dmm;
}


LIBXS_API_DEFINITION libxs_smmfunction libxs_create_scsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values)
{
  libxs_code_pointer code = { 0 };
  libxs_csr_reg_descriptor sreg;
  libxs_build_request request;
  double* d_values;
  unsigned int i;
  LIBXS_INIT
  /* we need to copy the values into a double precision buffer */
  d_values = (double*)malloc(row_ptr[descriptor->m]*sizeof(double));
  for ( i = 0; i < row_ptr[descriptor->m]; i++) {
    d_values[i] = (double)values[i];
  }
  sreg.gemm = descriptor;
  sreg.row_ptr = row_ptr;
  sreg.column_idx = column_idx;
  sreg.values = d_values;
  request.descriptor.sreg = &sreg;
  request.kind = LIBXS_BUILD_KIND_SREG;
  libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
  free(d_values);
  return code.xmm.smm;
}


LIBXS_API_DEFINITION void libxs_release_kernel(const void* jit_code)
{
  void* extra = 0;
  LIBXS_INIT
  if (EXIT_SUCCESS == libxs_malloc_info(jit_code, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int regindex = *((const unsigned int*)extra);
    if ((LIBXS_CAPACITY_REGISTRY) <= regindex) {
      libxs_xfree(jit_code);
    }
    /* TODO: implement to unregister GEMM kernels */
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: failed to release kernel!\n");
    }
  }
}

