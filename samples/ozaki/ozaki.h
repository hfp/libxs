/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_hist.h>
#include <libxs_malloc.h>
#include <libxs_timer.h>
#include <libxs_mhd.h>
#include <libxs_sync.h>
#if defined(__DNNL)
# include <oneapi/dnnl/dnnl.h>
#endif

#if GEMM_IS_DOUBLE
# define OZ_MANT_BITS 52
# define OZ_EXP_BIAS 1023
#else /* single-precision */
# define OZ_MANT_BITS 23
# define OZ_EXP_BIAS 127
#endif
#define OZ_BIAS_PLUS_MANT (OZ_EXP_BIAS + OZ_MANT_BITS)
#define OZ_EXP_MASK (2U * OZ_EXP_BIAS + 1U)

#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif
#if !defined(BATCH_K)
# define BATCH_K 4
#endif
#if !defined(K_GRP)
# define K_GRP 32768
#endif

#if !defined(MAX_NSLICES)
# if GEMM_IS_DOUBLE
#   define MAX_NSLICES 16
# else
#   define MAX_NSLICES 8
# endif
#endif
#if !defined(NSLICES_DEFAULT)
# if GEMM_IS_DOUBLE
#   define NSLICES_DEFAULT 8
# else
#   define NSLICES_DEFAULT 4
# endif
#endif

/* Runtime flag-set controlling the Ozaki scheme 1 (OZAKI_FLAGS env var).
 * Bit 0 (1): TRIANGULAR  - iterate upper triangle of slice-pair matrix
 * Bit 1 (2): SYMMETRIZE  - compute mirror D(sb,sa) for off-diagonal pairs
 * Default 3 = TRIANGULAR + SYMMETRIZE (correct, fewer loop iterations).
 *
 * The trim parameter (OZAKI_TRIM env var) drops the T least significant
 * diagonals: pairs with sa + sb > 2*(S-1) - T are skipped. Default 0
 * means exact (all pairs). Each dropped diagonal loses ~7 bits. */
#define OZ1_TRIANGULAR 1
#define OZ1_SYMMETRIZE 2
#define OZ1_DEFAULT (OZ1_TRIANGULAR | OZ1_SYMMETRIZE)

#if GEMM_IS_DOUBLE
# define OZ2_NPRIMES_MAX 20
# if defined(OZAKI_I8) && (OZAKI_I8)
#   define OZ2_NPRIMES_DEFAULT 19
# else
#   define OZ2_NPRIMES_DEFAULT 16
# endif
#else /* single-precision */
# define OZ2_NPRIMES_MAX 12
# if defined(OZAKI_I8) && (OZAKI_I8)
#   define OZ2_NPRIMES_DEFAULT 10
# else
#   define OZ2_NPRIMES_DEFAULT 9
# endif
#endif

/* CPU-side profiling helpers (used inside omp parallel regions).
 * GEMM_PROFILE_DECL: declare tick variables.
 * GEMM_PROFILE_TICK(VAR, TID): take a tick on the master thread.
 * GEMM_PROFILE_END(TID, M, N, K): compute duration, push GFLOPS to histogram. */
#define GEMM_PROFILE_DECL libxs_timer_tick_t t_start = 0, t_preprocess = 0, t_kernel = 0
#define GEMM_PROFILE_TICK(VAR, TID) \
  if (0 == (TID) && 0 != ozaki_profile) (VAR) = libxs_timer_tick()
#define GEMM_PROFILE_END(TID, M, N, K) \
  if (0 == (TID) && 0 != ozaki_profile) { \
    const libxs_timer_tick_t t_end = libxs_timer_tick(); \
    const double flops = 2.0 * (M) * (N) * (K); \
    double duration = 0; \
    LIBXS_ASSERT(NULL != ozaki_hist); \
    if (2 == ozaki_profile) duration = libxs_timer_duration(t_kernel, t_end); \
    else if (3 == ozaki_profile || 4 == ozaki_profile) duration = libxs_timer_duration(t_start, t_preprocess); \
    else duration = libxs_timer_duration(t_start, t_end); \
    if (0 < duration) { \
      const double gflops = flops / (duration * 1E9); \
      libxs_hist_push(NULL, ozaki_hist, &gflops); \
    } \
  }

/**
 * Implement the public gemm_ozN function: call the _diff kernel,
 * then handle verbose output, diff accumulation, and matrix dumps.
 * DIFF_FN is the _diff kernel (gemm_oz1_diff or gemm_oz2_diff).
 */
#define OZAKI_GEMM_WRAPPER(DIFF_FN) \
  if (0 == ozaki_verbose) { \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0, NULL); \
  } \
  else { \
    libxs_matdiff_t diff, call_diff; \
    libxs_matdiff_clear(&diff); \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, LIBXS_ABS(ozaki_stat), &diff); \
    call_diff = diff; \
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER); \
    libxs_matdiff_reduce(&gemm_diff, &diff); \
    diff.r = gemm_diff.r; \
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER); \
    call_diff.r = diff.r; \
    if (1 < ozaki_verbose || 0 > ozaki_verbose) { \
      const int nth = (0 < ozaki_verbose ? ozaki_verbose : 1); \
      if (0 == (diff.r % nth)) { \
        if (0 <= ozaki_stat) print_diff(stderr, &diff); \
        else { \
          fprintf(stderr, "GEMM: "); \
          print_gemm(stderr, LIBXS_ABS(ozaki_stat), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
        } \
      } \
    } \
    if (call_diff.rsq < ozaki_rsq || -1 > ozaki_verbose || ozaki_eps < libxs_matdiff_epsilon(&call_diff)) { \
      print_diff(stderr, &call_diff); \
      if (0 != gemm_dump_inhibit) { \
        gemm_dump_inhibit = 2; \
      } \
      else { \
        const int result = gemm_dump_matrices(GEMM_ARGPASS, 1); \
        if (0 != ozaki_exit) exit(EXIT_SUCCESS == result ? EXIT_FAILURE : result); \
      } \
    } \
  }

/* Wrap/real symbol definitions for real GEMM */
#define GEMM_WRAP LIBXS_CONCATENATE(__wrap_, GEMM)
#define GEMM_REAL LIBXS_CONCATENATE(__real_, GEMM)

/* Wrap/real symbol definitions for complex GEMM */
#define ZGEMM_WRAP LIBXS_CONCATENATE(__wrap_, ZGEMM)
#define ZGEMM_REAL LIBXS_CONCATENATE(__real_, ZGEMM)

/* Precision-specific type and variable names (enable dual-precision builds).
 * These macros redirect "friendly" names used throughout the implementation
 * to unique symbols, e.g. gemm_original -> dgemm_original (double) or
 * sgemm_original (float). Both precisions can coexist in one binary. */
#define gemm_function_t LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define zgemm_function_t LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define gemm_original LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_original)
#define zgemm_original LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_original)
#define gemm_lock LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_lock)
#define ozaki_verbose LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_verbose)
#define ozaki LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki)
#define ozaki_n LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_n)
#define ozaki_profile LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_profile)
#define ozaki_hist LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_hist)
#define ozaki_flags LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_flags)
#define ozaki_trim LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_trim)
#define ozaki_stat LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_stat)
#define ozaki_exit LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_exit)
#define ozaki_eps LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_eps)
#define ozaki_rsq LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_rsq)
#define ozaki_target_arch LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_tarch)
#define gemm_oz1 LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz1)
#define gemm_oz2 LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz2)
#define gemm_dump_inhibit LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_inhibit)
#define gemm_dump_matrices LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_mhd)
#define zgemm3m LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm3m)
#define gemm_signal_handler LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_signal_handler)
#define gemm_atexit LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_atexit)
#define gemm_pool LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_pool)
#if defined(__LIBXSTREAM)
# define ozaki_ocl_handle LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_ocl_handle)
# define gemm_oz_ocl_diff LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz_ocl_diff)
# define oz1_host_preprocess_a LIBXS_TPREFIX(GEMM_REAL_TYPE, oz1_host_prep_a)
# define oz1_host_preprocess_b LIBXS_TPREFIX(GEMM_REAL_TYPE, oz1_host_prep_b)
# define oz2_host_preprocess_a LIBXS_TPREFIX(GEMM_REAL_TYPE, oz2_host_prep_a)
# define oz2_host_preprocess_b LIBXS_TPREFIX(GEMM_REAL_TYPE, oz2_host_prep_b)
#endif

/* Int8 dot product dispatch macro (Ozaki-1 slicing).
 * Priority: VPDPBSSD (1 instr) > VPDPBUSD+bias (2 instr) > scalar. */
#if (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
# if (LIBXS_X86_AVX512_INT8 <= LIBXS_STATIC_TARGET_ARCH)
#   define ozaki_dot_i8_init() ozaki_dot_i8_bssd
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
#   define ozaki_dot_i8_init() ozaki_dot_i8_vnni
# elif (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define ozaki_dot_i8_init() \
      ((LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) \
          ? ozaki_dot_i8_bssd \
          : ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_i8_vnni : ozaki_dot_i8_sw))
# elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define ozaki_dot_i8_init() ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_i8_vnni : ozaki_dot_i8_sw)
# else
#   define ozaki_dot_i8_init() ozaki_dot_i8_sw
# endif
#else
# define ozaki_dot_i8_init() ozaki_dot_i8_sw
#endif

/* u8 dot product dispatch macro (Ozaki-2 CRT).
 * Priority: VPDPBUUD (1 instr) > VPDPBUSD+bias (2 instr) > scalar. */
#if (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
# if (LIBXS_X86_AVX512_INT8 <= LIBXS_STATIC_TARGET_ARCH)
#   define ozaki_dot_u8_init() ozaki_dot_u8_buud
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
#   define ozaki_dot_u8_init() ozaki_dot_u8_vnni
# elif (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define ozaki_dot_u8_init() \
      ((LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) \
          ? ozaki_dot_u8_buud \
          : ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_u8_vnni : ozaki_dot_u8_sw))
# elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define ozaki_dot_u8_init() ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_u8_vnni : ozaki_dot_u8_sw)
# else
#   define ozaki_dot_u8_init() ozaki_dot_u8_sw
# endif
#else
# define ozaki_dot_u8_init() ozaki_dot_u8_sw
#endif


/** Function type for complex GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*zgemm_function_t)(GEMM_ARGDECL);
/** Function pointer type for int8 dot product dispatch. */
typedef int32_t (*ozaki_dot_i8_fn)(const int8_t[BLOCK_K], const int8_t[BLOCK_K]);
/** Function pointer type for uint8 dot product dispatch (Ozaki-2 u8 CRT). */
typedef int32_t (*ozaki_dot_u8_fn)(const uint8_t[BLOCK_K], const uint8_t[BLOCK_K]);

/** Function prototypes for wrapped / real / public GEMM and complex GEMM. */
LIBXS_API_INTERN void GEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void GEMM_REAL(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_REAL(GEMM_ARGDECL);
LIBXS_API void ZGEMM(GEMM_ARGDECL);

/** Function prototype for GEMM using low-precision (Ozaki scheme 1). */
LIBXS_API void gemm_oz1(GEMM_ARGDECL);
/** Function prototype for GEMM using CRT modular arithmetic (Ozaki scheme 2). */
LIBXS_API void gemm_oz2(GEMM_ARGDECL);
/** Complex GEMM 3M (Karatsuba) implementation (internal). */
LIBXS_API_INTERN void zgemm3m(GEMM_ARGDECL);

LIBXS_APIVAR_PUBLIC(gemm_function_t gemm_original);
LIBXS_APIVAR_PUBLIC(int ozaki);
LIBXS_APIVAR_PUBLIC(int ozaki_verbose);
LIBXS_APIVAR_PUBLIC(int ozaki_stat);

LIBXS_APIVAR_PRIVATE(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE(zgemm_function_t zgemm_original);
LIBXS_APIVAR_PRIVATE(libxs_malloc_pool_t* gemm_pool);
LIBXS_APIVAR_PRIVATE(int ozaki_target_arch);
LIBXS_APIVAR_PRIVATE(double ozaki_eps);
LIBXS_APIVAR_PRIVATE(double ozaki_rsq);
LIBXS_APIVAR_PRIVATE(int ozaki_flags);
LIBXS_APIVAR_PRIVATE(int ozaki_trim);
LIBXS_APIVAR_PRIVATE(int ozaki_exit);
LIBXS_APIVAR_PRIVATE(int ozaki_n);
LIBXS_APIVAR_PRIVATE(int ozaki_profile);
LIBXS_APIVAR_PRIVATE(libxs_hist_t* ozaki_hist);

extern LIBXS_TLS int gemm_dump_inhibit;

#if defined(__LIBXSTREAM)
/** Opaque OpenCL handle (bridge to LIBXSTREAM Ozaki). */
LIBXS_APIVAR_PRIVATE(void* ozaki_ocl_handle);
void* ozaki_ocl_create(
  int use_double, int kind, int verbosity, int tm, int tn, int ndecomp, int ozflags, int oztrim, int ozgroups, int profiling);
void ozaki_ocl_release(void* handle);
int ozaki_ocl_gemm(void* handle, char transa, char transb, int M, int N, int K, double alpha, const void* a, int lda, const void* b,
  int ldb, double beta, void* c, int ldc, libxs_hist_t* hist, int profile);
int ozaki_ocl_gemm3m(void* handle, char transa, char transb, int M, int N, int K, const double* alpha, const void* a, int lda,
  const void* b, int ldb, const double* beta, void* c, int ldc);
int ozaki_ocl_supports_zgemm3m(void* handle);
void ozaki_ocl_invalidate_cache(void* handle, const void* a, const void* b);
void ozaki_ocl_finalize(void);
#endif

/* Shared int8 dot-product infrastructure (VNNI + scalar fallback).
 *
 * VPDPBSSD (i8*i8): true signed×signed, single instruction, no bias.
 * VPDPBUUD (u8*u8): true unsigned×unsigned, single instruction, no bias.
 * Both require AVX-VNNI-INT8 (LIBXS_X86_AVX512_INT8 / LIBXS_X86_AVX10_256).
 * The LIBXS_INTRINSICS guard + target attribute enable the compiler to
 * emit these instructions via function multi-versioning even when the
 * baseline -march does not include avxvnniint8. */
#if (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH) && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)

# if 16 == BLOCK_K /* 128-bit: one __m128i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_i8_bssd(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m128i va = _mm_loadu_si128((const __m128i*)a);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  __m128i dp = _mm_dpbssd_epi32(_mm_setzero_si128(), va, vb);
  dp = _mm_hadd_epi32(dp, dp);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0);
}

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_u8_buud(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m128i va = _mm_loadu_si128((const __m128i*)a);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  __m128i dp = _mm_dpbuud_epi32(_mm_setzero_si128(), va, vb);
  dp = _mm_hadd_epi32(dp, dp);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0);
}

# elif 32 == BLOCK_K /* 256-bit: one __m256i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_i8_bssd(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m256i va = _mm256_loadu_si256((const __m256i*)a);
  const __m256i vb = _mm256_loadu_si256((const __m256i*)b);
  __m256i dp = _mm256_dpbssd_epi32(_mm256_setzero_si256(), va, vb);
  {
    const __m128i hi = _mm256_extracti128_si256(dp, 1);
    __m128i lo = _mm256_castsi256_si128(dp);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
  }
}

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_u8_buud(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m256i va = _mm256_loadu_si256((const __m256i*)a);
  const __m256i vb = _mm256_loadu_si256((const __m256i*)b);
  __m256i dp = _mm256_dpbuud_epi32(_mm256_setzero_si256(), va, vb);
  {
    const __m128i hi = _mm256_extracti128_si256(dp, 1);
    __m128i lo = _mm256_castsi256_si128(dp);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
  }
}

# elif 64 == BLOCK_K /* 512-bit: one __m512i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_i8_bssd(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m512i va = _mm512_loadu_si512((const __m512i*)a);
  const __m512i vb = _mm512_loadu_si512((const __m512i*)b);
  __m512i dp = _mm512_dpbssd_epi32(_mm512_setzero_si512(), va, vb);
  return _mm512_reduce_add_epi32(dp);
}

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
int32_t ozaki_dot_u8_buud(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m512i va = _mm512_loadu_si512((const __m512i*)a);
  const __m512i vb = _mm512_loadu_si512((const __m512i*)b);
  __m512i dp = _mm512_dpbuud_epi32(_mm512_setzero_si512(), va, vb);
  return _mm512_reduce_add_epi32(dp);
}

# endif /* BLOCK_K width selection */
#endif /* AVX512_INT8 <= MAX_STATIC_TARGET_ARCH */


/* VPDPBUSD: unsigned×signed int8 dot product with bias correction.
 * XOR with 0x80 converts signed a[] to unsigned, then VPDPBUSD
 * computes u8×s8 dot; subtracting 128*sum(b[]) compensates the
 * bias.  Requires two VPDPBUSD calls per vector (one for dp, one
 * for the b-sum used in compensation). */
#if defined(LIBXS_INTRINSICS_AVX512)

# if 16 == BLOCK_K /* 128-bit: one __m128i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i va = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), bias);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  const __m128i ones = _mm_set1_epi8(1);
  __m128i dp = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
  __m128i sb = _mm_dpbusd_epi32(_mm_setzero_si128(), ones, vb);
  dp = _mm_hadd_epi32(dp, sb);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0) - 128 * _mm_extract_epi32(dp, 1);
}

# elif 32 == BLOCK_K /* 256-bit: one __m256i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i va = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)a), bias);
  const __m256i vb = _mm256_loadu_si256((const __m256i*)b);
  const __m256i ones = _mm256_set1_epi8(1);
  __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, vb);
  __m256i sb = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, vb);
  {
    const __m128i hi_dp = _mm256_extracti128_si256(dp, 1);
    const __m128i hi_sb = _mm256_extracti128_si256(sb, 1);
    __m128i lo_dp = _mm256_castsi256_si128(dp);
    __m128i lo_sb = _mm256_castsi256_si128(sb);
    lo_dp = _mm_add_epi32(lo_dp, hi_dp);
    lo_sb = _mm_add_epi32(lo_sb, hi_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_dp);
    return _mm_extract_epi32(lo_dp, 0) - 128 * _mm_extract_epi32(lo_dp, 1);
  }
}

# elif 64 == BLOCK_K /* 512-bit: one __m512i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m512i bias = _mm512_set1_epi8((char)0x80);
  const __m512i va = _mm512_xor_si512(_mm512_loadu_si512((const __m512i*)a), bias);
  const __m512i vb = _mm512_loadu_si512((const __m512i*)b);
  const __m512i ones = _mm512_set1_epi8(1);
  __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, vb);
  __m512i sb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, vb);
  return _mm512_reduce_add_epi32(dp) - 128 * _mm512_reduce_add_epi32(sb);
}

# endif /* BLOCK_K width selection */
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API_INLINE int32_t ozaki_dot_i8_sw(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* ---- Unsigned u8 dot products for Ozaki-2 CRT ----
 *
 * u8 CRT stores residues in [0, p-1] with sign encoded via modular
 * additive inverse (p - r).  Dot products need u8*u8 -> int32.
 *
 * VPDPBUSD is u8*i8.  To get u8*u8 we XOR B with 0x80 to convert
 * u8 -> i8 (subtracting 128), then correct: result + 128*sum(A).
 * Same 2-instruction cost as the i8 bias-correction path above. */

/* VPDPBUSD: u8*u8 dot product via u8*i8 with bias correction on B.
 * XOR B with 0x80 converts u8 -> i8, then VPDPBUSD gives u8*i8;
 * adding 128*sum(A) (via VPDPBUSD(A, ones)) compensates. */
#if defined(LIBXS_INTRINSICS_AVX512)

# if 16 == BLOCK_K /* 128-bit: one __m128i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_u8_vnni(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i va = _mm_loadu_si128((const __m128i*)a);
  const __m128i vb = _mm_xor_si128(_mm_loadu_si128((const __m128i*)b), bias);
  const __m128i ones = _mm_set1_epi8(1);
  __m128i dp = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
  __m128i sa = _mm_dpbusd_epi32(_mm_setzero_si128(), va, ones);
  dp = _mm_hadd_epi32(dp, sa);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0) + 128 * _mm_extract_epi32(dp, 1);
}

# elif 32 == BLOCK_K /* 256-bit: one __m256i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_u8_vnni(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i va = _mm256_loadu_si256((const __m256i*)a);
  const __m256i vb = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)b), bias);
  const __m256i ones = _mm256_set1_epi8(1);
  __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, vb);
  __m256i sa = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, ones);
  {
    const __m128i hi_dp = _mm256_extracti128_si256(dp, 1);
    const __m128i hi_sa = _mm256_extracti128_si256(sa, 1);
    __m128i lo_dp = _mm256_castsi256_si128(dp);
    __m128i lo_sa = _mm256_castsi256_si128(sa);
    lo_dp = _mm_add_epi32(lo_dp, hi_dp);
    lo_sa = _mm_add_epi32(lo_sa, hi_sa);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_sa);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_dp);
    return _mm_extract_epi32(lo_dp, 0) + 128 * _mm_extract_epi32(lo_dp, 1);
  }
}

# elif 64 == BLOCK_K /* 512-bit: one __m512i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_u8_vnni(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  const __m512i bias = _mm512_set1_epi8((char)0x80);
  const __m512i va = _mm512_loadu_si512((const __m512i*)a);
  const __m512i vb = _mm512_xor_si512(_mm512_loadu_si512((const __m512i*)b), bias);
  const __m512i ones = _mm512_set1_epi8(1);
  __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, vb);
  __m512i sa = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, ones);
  return _mm512_reduce_add_epi32(dp) + 128 * _mm512_reduce_add_epi32(sa);
}

# endif /* BLOCK_K width selection */
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API_INLINE int32_t ozaki_dot_u8_sw(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* Unified uint8 GEMM: C[M,N] = A[M,K] * B'[N,K], u8*u8 -> s32.
 * Row-major storage. beta: 0=overwrite C, nonzero=accumulate into C.
 * Requires: transa='N', transb='T', K % BLOCK_K == 0. */
LIBXS_API_INLINE void ozaki_gemm_u8u8s32(char transa, char transb, GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const uint8_t* a, GEMM_INT_TYPE lda, const uint8_t* b, GEMM_INT_TYPE ldb, int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  const ozaki_dot_u8_fn dot = ozaki_dot_u8_init();
  GEMM_INT_TYPE mi, nj, kb;
  LIBXS_ASSERT('N' == transa || 'n' == transa);
  LIBXS_ASSERT('T' == transb || 't' == transb);
  LIBXS_ASSERT(0 == (K % BLOCK_K));
  LIBXS_UNUSED(transa);
  LIBXS_UNUSED(transb);
  LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
  for (mi = 0; mi < M; ++mi) {
    int32_t* const crow = c + mi * ldc;
    const uint8_t* const arow = a + mi * lda;
    if (0 == beta) {
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N)
      for (nj = 0; nj < N; ++nj) crow[nj] = 0;
    }
    for (kb = 0; kb < K; kb += BLOCK_K) {
      const uint8_t* const ak = arow + kb;
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N)
      for (nj = 0; nj < N; ++nj) {
        crow[nj] += dot(ak, b + nj * ldb + kb);
      }
    }
  }
}


/* Unified int8 GEMM: C[M,N] = op(A)[M,K] * op(B)[K,N], s8*s8 -> s32.
 * Row-major storage. beta: 0=overwrite C, nonzero=accumulate into C.
 * With __DNNL: uses dnnl_gemm_s8s8s32.
 * Without: naive dot_i8 loop (transa='N', transb='T', K % BLOCK_K == 0). */
LIBXS_API_INLINE void ozaki_gemm_s8s8s32(char transa, char transb, GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const int8_t* a, GEMM_INT_TYPE lda, const int8_t* b, GEMM_INT_TYPE ldb, int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
#if defined(__DNNL)
  static const int32_t zero = 0;
  dnnl_gemm_s8s8s32(transa, transb, 'F', (dnnl_dim_t)M, (dnnl_dim_t)N, (dnnl_dim_t)K, 1.0f, a, (dnnl_dim_t)lda, 0, b,
    (dnnl_dim_t)ldb, 0, 0 != beta ? 1.0f : 0.0f, c, (dnnl_dim_t)ldc, &zero);
#else
  const ozaki_dot_i8_fn dot = ozaki_dot_i8_init();
  GEMM_INT_TYPE mi, nj, kb;
  LIBXS_ASSERT('N' == transa || 'n' == transa);
  LIBXS_ASSERT('T' == transb || 't' == transb);
  LIBXS_ASSERT(0 == (K % BLOCK_K));
  LIBXS_UNUSED(transa);
  LIBXS_UNUSED(transb);
  /* M x K x N loop order: reuse each A chunk across all N columns. */
  LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
  for (mi = 0; mi < M; ++mi) {
    int32_t* const crow = c + mi * ldc;
    const int8_t* const arow = a + mi * lda;
    if (0 == beta) {
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N)
      for (nj = 0; nj < N; ++nj) crow[nj] = 0;
    }
    for (kb = 0; kb < K; kb += BLOCK_K) {
      const int8_t* const ak = arow + kb;
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N)
      for (nj = 0; nj < N; ++nj) {
        crow[nj] += dot(ak, b + nj * ldb + kb);
      }
    }
  }
#endif
}


/**
 * Extract IEEE-754 biased exponent and full mantissa (with implicit bit)
 * into uint64_t.  Returns sign (+1 or -1); for zero/subnormal/NaN/Inf
 * sets exp_biased=0 and mantissa=0, returns +1.
 *
 * Special-value detection is done entirely via the raw exponent field
 * after bit-extraction, avoiding the previous float cast which was
 * incorrect for double precision (finite doubles > ~3.4e38 overflow
 * to float-Inf, producing a false positive).
 */
LIBXS_API_INLINE int ozaki_extract_ieee(GEMM_REAL_TYPE value, int16_t* exp_biased, uint64_t* mantissa)
{
  uint64_t bits;
  uint64_t frac;
  uint16_t exp_raw;
  int sign = 1;

  /* Fast zero check avoids bit extraction for the common sparse case */
  if (value == (GEMM_REAL_TYPE)0) {
    *exp_biased = 0;
    *mantissa = 0;
  }
  else {
    sign = (value < (GEMM_REAL_TYPE)0) ? -1 : 1;
    if (value < (GEMM_REAL_TYPE)0) value = -value;

#if GEMM_IS_DOUBLE
    {
      union {
        double d;
        uint64_t u;
      } cvt;
      cvt.d = value;
      bits = cvt.u;
    }
#else
    {
      union {
        float f;
        uint32_t u;
      } cvt;
      cvt.f = value;
      bits = cvt.u;
    }
#endif
    exp_raw = (uint16_t)((bits >> OZ_MANT_BITS) & OZ_EXP_MASK);
    frac = bits & ((1ULL << OZ_MANT_BITS) - 1ULL);

    /* exp_raw == 0: subnormal (treated as zero).
     * exp_raw == OZ_EXP_MASK: Inf (frac==0) or NaN (frac!=0). */
    if (0 == exp_raw || exp_raw == (uint16_t)OZ_EXP_MASK) {
      *exp_biased = 0;
      *mantissa = 0;
      if (0 != exp_raw) sign = 1; /* NaN/Inf: return 1 */
    }
    else {
      *exp_biased = (int16_t)exp_raw;
      *mantissa = (1ULL << OZ_MANT_BITS) | frac;
    }
  }
  return sign;
}


/**
 * Scale a tile of C by beta, optionally capturing the pre-scaled block.
 * Per BLAS spec, beta=0 must zero out C unconditionally (C may hold NaN
 * or Inf when uninitialized); a plain multiply would give NaN.
 */
LIBXS_API_INLINE void ozaki_scale_block_beta(GEMM_REAL_TYPE* mb, GEMM_INT_TYPE ldc, GEMM_INT_TYPE iblk, GEMM_INT_TYPE jblk,
  const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* ref_blk, int capture_ref)
{
  const GEMM_REAL_TYPE b = *beta;
  GEMM_INT_TYPE mi, nj;
  for (mi = 0; mi < iblk; ++mi) {
    for (nj = 0; nj < jblk; ++nj) {
      if (0 != capture_ref) ref_blk[LIBXS_INDEX(0, BLOCK_M, mi, nj)] = mb[LIBXS_INDEX(0, ldc, mi, nj)];
      mb[LIBXS_INDEX(0, ldc, mi, nj)] = ((GEMM_REAL_TYPE)0 != b) ? mb[LIBXS_INDEX(0, ldc, mi, nj)] * b : (GEMM_REAL_TYPE)0;
    }
  }
}


/** Store a (reference, reconstructed) value pair into block buffers. */
LIBXS_API_INLINE void ozaki_store_block_pair(GEMM_REAL_TYPE* ref_blk, GEMM_REAL_TYPE* recon_blk, GEMM_INT_TYPE ld,
  GEMM_INT_TYPE row, GEMM_INT_TYPE col, GEMM_REAL_TYPE ref_val, GEMM_REAL_TYPE recon_val)
{
  recon_blk[LIBXS_INDEX(0, ld, row, col)] = recon_val;
  ref_blk[LIBXS_INDEX(0, ld, row, col)] = ref_val;
}


/** Compute matrix diff for one block and reduce into accumulator. */
LIBXS_API_INLINE void ozaki_accumulate_block_diff(libxs_matdiff_t* acc, const GEMM_REAL_TYPE* ref_blk,
  const GEMM_REAL_TYPE* tst_blk, GEMM_INT_TYPE bm, GEMM_INT_TYPE bn, GEMM_INT_TYPE ld_ref, GEMM_INT_TYPE ld_tst)
{
  libxs_matdiff_t block_diff;
  const int ild_ref = (int)ld_ref, ild_tst = (int)ld_tst;
  if (EXIT_SUCCESS == libxs_matdiff(&block_diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), bm, bn, ref_blk, tst_blk, &ild_ref, &ild_tst)) {
    libxs_matdiff_reduce(acc, &block_diff);
  }
}


/**
 * Dump A and B matrices as MHD files.
 * Works for both real (ncomponents=1) and complex (ncomponents=2) matrices.
 * Uses gemm_diff.r as the dump ID and updates ozaki_eps/ozaki_rsq thresholds
 * to avoid repeated dumps.
 */
LIBXS_API_INLINE int gemm_dump_matrices(GEMM_ARGDECL, size_t ncomponents)
{
  char fname[64];
  const char* const env_slurm = getenv("SLURM_JOBID");
  const int slurm = (NULL == env_slurm ? -1 : atoi(env_slurm));
  const int id = (1 < libxs_nranks() ? libxs_nrank() : libxs_pid());
  int result = EXIT_SUCCESS;
  FILE* file;

  gemm_mhd_settings_t settings;
  settings.ozaki = ozaki;
  settings.ozn = ozaki_n;
  settings.ozflags = ozaki_flags;
  settings.oztrim = ozaki_trim;
  settings.ldc = *ldc;
  settings.eps = ozaki_eps;
  settings.rsq = ozaki_rsq;

  LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);

  if (0 > slurm) LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%u-%i-a.mhd", id, gemm_diff.r);
  else LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%i-%u-%i-a.mhd", slurm, id, gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    const int result_a = gemm_mhd_write(fname, a, *m, *k, *lda, *transa, alpha, ncomponents, &settings);
    if (EXIT_SUCCESS != result_a && 0 != ozaki_verbose) {
      fprintf(stderr, "ERROR: dumping A-matrix to %s failed!\n", fname);
    }
    result |= result_a;
  }
  else fclose(file);

  if (0 > slurm) LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%u-%i-b.mhd", id, gemm_diff.r);
  else LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%i-%u-%i-b.mhd", slurm, id, gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    const int result_b = gemm_mhd_write(fname, b, *k, *n, *ldb, *transb, beta, ncomponents, &settings);
    if (EXIT_SUCCESS != result_b && 0 != ozaki_verbose) {
      fprintf(stderr, "ERROR: dumping B-matrix to %s failed!\n", fname);
    }
    result |= result_b;
  }
  else fclose(file);

  if (0 != ozaki_verbose) {
    fprintf(stderr, "GEMM: ");
    print_gemm(stderr, 1 /*compact*/, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  if (EXIT_SUCCESS == result) {
    /* avoid repeated dumps */
    ozaki_eps = libxs_matdiff_epsilon(&gemm_diff);
    ozaki_rsq = gemm_diff.rsq;
  }

  LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  return result;
}
