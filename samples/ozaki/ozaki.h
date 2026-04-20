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
#include <libxs_sync.h>
#include <libxs_mhd.h>
#include <libxs_mem.h>
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
 * Per-phase timing (modes 2-4) is only meaningful on GPU where
 * preprocessing and kernel are distinct operations; on CPU the
 * K-group loop interleaves them, so we always measure total time.
 * GEMM_PROFILE_START(TID): record start tick on master thread.
 * GEMM_PROFILE_END(TID, M, N, K): compute duration, push GFLOPS to histogram. */
#define GEMM_PROFILE_DECL libxs_timer_tick_t gemm_profile_start_ = 0
#define GEMM_PROFILE_START(TID) \
  if (0 == (TID) && 0 != ozaki_profile) gemm_profile_start_ = libxs_timer_tick()
#define GEMM_PROFILE_END(TID, M, N, K) \
  if (0 == (TID) && 0 != ozaki_profile) { \
    const double duration = libxs_timer_duration(gemm_profile_start_, libxs_timer_tick()); \
    if (0 < duration) { \
      const double gflops = 2.0 * (M) * (N) * (K) / (duration * 1E9); \
      libxs_hist_push(NULL, ozaki_hist, &gflops); \
    } \
  }

#define OZAKI_GEMM_WRAPPER(DIFF_FN, LABEL, NCOMP) \
  if (0 == ozaki_verbose) { \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, NULL); \
  } \
  else { \
    libxs_matdiff_t diff; \
    libxs_matdiff_clear(&diff); \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &diff); \
    ozaki_post_diff(GEMM_ARGPASS, LABEL, NCOMP, &diff); \
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
#define ozaki_complex LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_complex)
#define ozaki_maxk LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_maxk)
#define ozaki_n LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_n)
#define ozaki_profile LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_profile)
#define ozaki_hist LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_hist)
#define ozaki_flags LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_flags)
#define ozaki_trim LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_trim)
#define ozaki_stat LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_stat)
#define ozaki_dump LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_dump)
#define ozaki_exit LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_exit)
#define ozaki_idx LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_idx)
#define ozaki_eps LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_eps)
#define ozaki_rsq LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_rsq)
#define ozaki_target_arch LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_tarch)
#define gemm_oz1 LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz1)
#define gemm_oz2 LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz2)
#define gemm_init LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_init)
#define gemm_threshold LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_threshold)
/* gemm_dump_inhibit: not precision-prefixed (like gemm_nozaki) */
#define gemm_dump_matrices LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_mhd)
#define gemm_complex LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_complex)
#define gemm_signal_handler LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_signal_handler)
#define gemm_atexit LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_atexit)
#define gemm_pool LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_pool)
#if defined(__LIBXSTREAM)
# define ozaki_ocl_handle LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_ocl_handle)
# define gemm_oz_ocl_diff LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz_ocl_diff)
#endif

/* Scalar int8 GEMM fallback: C[M,N] += A[M,K] * B'[N,K] via per-element
 * dot product. Used for edge tiles where N < BLOCK_N (panel path not
 * applicable). Auto-vectorizable inner loop. */
#define OZAKI_GEMM_INT8_BODY(ELEM_T, DOT_FN) \
  { \
    GEMM_INT_TYPE mi, nj, kb; \
    LIBXS_ASSERT('N' == transa || 'n' == transa); \
    LIBXS_ASSERT('T' == transb || 't' == transb); \
    LIBXS_ASSERT(0 == (K % BLOCK_K)); \
    LIBXS_UNUSED(transa); \
    LIBXS_UNUSED(transb); \
    LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M) \
    for (mi = 0; mi < M; ++mi) { \
      int32_t* const crow = c + mi * ldc; \
      const ELEM_T* const arow = a + mi * lda; \
      if (0 == beta) { \
        LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N) \
        for (nj = 0; nj < N; ++nj) crow[nj] = 0; \
      } \
      for (kb = 0; kb < K; kb += BLOCK_K) { \
        const ELEM_T* const ak = arow + kb; \
        LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_N, BLOCK_N) \
        for (nj = 0; nj < N; ++nj) { \
          crow[nj] += DOT_FN(ak, b + nj * ldb + kb); \
        } \
      } \
    } \
  }

#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
#define OZAKI_PANEL_REFORMAT_B(B, LDB, KB, N, BUF) do { \
    int rf_kk, rf_nj; \
    for (rf_kk = 0; rf_kk < BLOCK_K; rf_kk += 4) { \
      for (rf_nj = 0; rf_nj < (N); ++rf_nj) { \
        memcpy((BUF) + (rf_kk >> 2) * (N) + rf_nj, \
               (const char*)(B) + (long)rf_nj * (LDB) + (KB) + rf_kk, 4); \
      } \
    } \
  } while(0)

#define OZAKI_PANEL_REFORMAT_B_XOR(B, LDB, KB, N, BUF) do { \
    int rf_kk, rf_nj; \
    for (rf_kk = 0; rf_kk < BLOCK_K; rf_kk += 4) { \
      for (rf_nj = 0; rf_nj < (N); ++rf_nj) { \
        int32_t rf_tmp; \
        memcpy(&rf_tmp, (const char*)(B) + (long)rf_nj * (LDB) + (KB) + rf_kk, 4); \
        (BUF)[(rf_kk >> 2) * (N) + rf_nj] = rf_tmp ^ (int32_t)0x80808080; \
      } \
    } \
  } while(0)
#endif


/** Function type for complex GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*zgemm_function_t)(GEMM_ARGDECL);

/** Function prototypes for wrapped / real / public GEMM and complex GEMM. */
OZAKI_API_INTERN void GEMM_WRAP(GEMM_ARGDECL);
OZAKI_API_INTERN void GEMM_REAL(GEMM_ARGDECL);
OZAKI_API_INTERN void ZGEMM_WRAP(GEMM_ARGDECL);
OZAKI_API_INTERN void ZGEMM_REAL(GEMM_ARGDECL);
OZAKI_API void ZGEMM(GEMM_ARGDECL);

/** Function prototype for GEMM using low-precision (Ozaki scheme 1). */
OZAKI_API void gemm_oz1(GEMM_ARGDECL);
/** Function prototype for GEMM using CRT modular arithmetic (Ozaki scheme 2). */
OZAKI_API void gemm_oz2(GEMM_ARGDECL);
/** Complex GEMM implementation (internal). */
OZAKI_API_INTERN void gemm_complex(GEMM_ARGDECL);

OZAKI_APIVAR_PUBLIC(gemm_function_t gemm_original);
OZAKI_APIVAR_PUBLIC(int ozaki);
OZAKI_APIVAR_PUBLIC(int ozaki_complex);
OZAKI_APIVAR_PUBLIC(int ozaki_maxk);
OZAKI_APIVAR_PUBLIC(int ozaki_verbose);
OZAKI_APIVAR_PUBLIC(int ozaki_stat);

OZAKI_APIVAR_PRIVATE(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
OZAKI_APIVAR_PRIVATE(zgemm_function_t zgemm_original);
OZAKI_APIVAR_PRIVATE(libxs_malloc_pool_t* gemm_pool);
OZAKI_APIVAR_PRIVATE(int ozaki_target_arch);
OZAKI_APIVAR_PRIVATE(int ozaki_idx);
OZAKI_APIVAR_PRIVATE(double ozaki_eps);
OZAKI_APIVAR_PRIVATE(double ozaki_rsq);
OZAKI_APIVAR_PRIVATE(int ozaki_flags);
OZAKI_APIVAR_PRIVATE(int ozaki_trim);
OZAKI_APIVAR_PRIVATE(int ozaki_dump);
OZAKI_APIVAR_PRIVATE(int ozaki_exit);
OZAKI_APIVAR_PRIVATE(int ozaki_n);
OZAKI_APIVAR_PRIVATE(int ozaki_profile);
OZAKI_APIVAR_PRIVATE(libxs_hist_t* ozaki_hist);
OZAKI_APIVAR_PRIVATE(int gemm_threshold);

OZAKI_API_INTERN void gemm_init(void);
LIBXS_API_INLINE void ozaki_post_diff(GEMM_ARGDECL, const char* label, size_t ncomponents, libxs_matdiff_t* diff);

extern LIBXS_TLS int gemm_nozaki; /* not precision-prefixed: bypass must cover all precisions */
extern LIBXS_TLS int gemm_dump_inhibit;

#if defined(__LIBXSTREAM)
/** Opaque OpenCL handle (bridge to LIBXSTREAM Ozaki). */
OZAKI_APIVAR_PRIVATE(void* ozaki_ocl_handle);
void* ozaki_ocl_create(
  int use_double, int kind, int verbosity, int tm, int tn, int ndecomp, int ozflags, int oztrim, int ozgroups, int maxk, int profiling);
void ozaki_ocl_release(void* handle);
int ozaki_ocl_gemm(void* handle, char transa, char transb, int M, int N, int K, double alpha, const void* a, int lda, const void* b,
  int ldb, double beta, void* c, int ldc, libxs_hist_t* hist, int profile);
int ozaki_ocl_gemm_complex(void* handle, char transa, char transb, int M, int N, int K, const double* alpha, const void* a, int lda,
  const void* b, int ldb, const double* beta, void* c, int ldc);
int ozaki_ocl_supports_gemm_complex(void* handle);
void ozaki_ocl_invalidate_cache(void* handle, const void* a, const void* b);
void ozaki_ocl_finalize(void);
#endif

/* Scalar int8 dot product fallback (auto-vectorizable). */
LIBXS_API_INLINE int32_t ozaki_dot_i8_sw(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}

/* Scalar u8 dot product fallback (auto-vectorizable). */
LIBXS_API_INLINE int32_t ozaki_dot_u8_sw(const uint8_t a[BLOCK_K], const uint8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* Panel GEMM kernels: accumulate N=16 output columns simultaneously
 * using broadcast-A + VNNI dot-product-accumulate (no horizontal
 * reduction). B is reformatted to VNNI dword packing on the fly:
 * b_vnni[K/4][16] where each int32 holds 4 consecutive K-bytes
 * from one B column. This layout matches AMX tile-B format. */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)

/* s8*s8 panel via DPBSSD (AVX-VNNI-INT8, 512-bit EVEX). */
#if (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
void ozaki_panel_i8_bssd(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const int8_t* a, GEMM_INT_TYPE lda, const int8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  __m512i acc[BLOCK_M];
  GEMM_INT_TYPE mi, kb;
  int kk;
  for (mi = 0; mi < M; ++mi) {
    acc[mi] = (0 != beta) ? _mm512_loadu_si512((__m512i*)(c + mi * ldc)) : _mm512_setzero_si512();
  }
  for (kb = 0; kb < K; kb += BLOCK_K) {
    LIBXS_ALIGNED(int32_t b_vnni[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
    OZAKI_PANEL_REFORMAT_B(b, ldb, kb, N, b_vnni);
    for (kk = 0; kk < BLOCK_K; kk += 4) {
      const __m512i vb = _mm512_loadu_si512((const __m512i*)(b_vnni + (kk >> 2) * N));
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
      for (mi = 0; mi < M; ++mi) {
        const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kb + kk));
        acc[mi] = _mm512_dpbssd_epi32(acc[mi], va, vb);
      }
    }
  }
  for (mi = 0; mi < M; ++mi) {
    _mm512_storeu_si512((__m512i*)(c + mi * ldc), acc[mi]);
  }
}
#endif /* AVX512_INT8 panel i8 */


/* s8*s8 panel via DPBUSD with bias correction (base AVX-512 VNNI).
 * DPBUSD computes u8*s8; XOR A with 0x80 converts s8 to u8, then
 * subtract 128 * column_sum(B) to correct. Column sum is computed
 * once across all K using DPBUSD(ones, B). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void ozaki_panel_i8_vnni(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const int8_t* a, GEMM_INT_TYPE lda, const int8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  const __m512i bias = _mm512_set1_epi32((int32_t)0x80808080);
  const __m512i ones = _mm512_set1_epi32(0x01010101);
  __m512i acc[BLOCK_M];
  __m512i bsum = _mm512_setzero_si512();
  GEMM_INT_TYPE mi, kb;
  int kk;
  for (mi = 0; mi < M; ++mi) {
    acc[mi] = (0 != beta) ? _mm512_loadu_si512((__m512i*)(c + mi * ldc)) : _mm512_setzero_si512();
  }
  for (kb = 0; kb < K; kb += BLOCK_K) {
    LIBXS_ALIGNED(int32_t b_vnni[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
    OZAKI_PANEL_REFORMAT_B(b, ldb, kb, N, b_vnni);
    for (kk = 0; kk < BLOCK_K; kk += 4) {
      const __m512i vb = _mm512_loadu_si512((const __m512i*)(b_vnni + (kk >> 2) * N));
      bsum = _mm512_dpbusd_epi32(bsum, ones, vb);
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
      for (mi = 0; mi < M; ++mi) {
        const __m512i va = _mm512_xor_si512(_mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kb + kk)), bias);
        acc[mi] = _mm512_dpbusd_epi32(acc[mi], va, vb);
      }
    }
  }
  { const __m512i correction = _mm512_mullo_epi32(_mm512_set1_epi32(128), bsum);
    for (mi = 0; mi < M; ++mi) {
      acc[mi] = _mm512_sub_epi32(acc[mi], correction);
      _mm512_storeu_si512((__m512i*)(c + mi * ldc), acc[mi]);
    }
  }
}


/* u8*u8 panel via DPBUUD (AVX-VNNI-INT8, 512-bit EVEX). */
#if (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8)
void ozaki_panel_u8_buud(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const uint8_t* a, GEMM_INT_TYPE lda, const uint8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  __m512i acc[BLOCK_M];
  GEMM_INT_TYPE mi, kb;
  int kk;
  for (mi = 0; mi < M; ++mi) {
    acc[mi] = (0 != beta) ? _mm512_loadu_si512((__m512i*)(c + mi * ldc)) : _mm512_setzero_si512();
  }
  for (kb = 0; kb < K; kb += BLOCK_K) {
    LIBXS_ALIGNED(int32_t b_vnni[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
    OZAKI_PANEL_REFORMAT_B(b, ldb, kb, N, b_vnni);
    for (kk = 0; kk < BLOCK_K; kk += 4) {
      const __m512i vb = _mm512_loadu_si512((const __m512i*)(b_vnni + (kk >> 2) * N));
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
      for (mi = 0; mi < M; ++mi) {
        const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kb + kk));
        acc[mi] = _mm512_dpbuud_epi32(acc[mi], va, vb);
      }
    }
  }
  for (mi = 0; mi < M; ++mi) {
    _mm512_storeu_si512((__m512i*)(c + mi * ldc), acc[mi]);
  }
}
#endif /* AVX512_INT8 panel u8 */


/* u8*u8 panel via DPBUSD with bias correction (base AVX-512 VNNI).
 * DPBUSD is u8*s8; XOR B with 0x80 converts u8 to s8 during reformat,
 * then add 128 * row_sum(A) to correct. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void ozaki_panel_u8_vnni(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const uint8_t* a, GEMM_INT_TYPE lda, const uint8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  __m512i acc[BLOCK_M];
  GEMM_INT_TYPE mi, kb;
  int kk;
  for (mi = 0; mi < M; ++mi) {
    acc[mi] = (0 != beta) ? _mm512_loadu_si512((__m512i*)(c + mi * ldc)) : _mm512_setzero_si512();
  }
  for (kb = 0; kb < K; kb += BLOCK_K) {
    LIBXS_ALIGNED(int32_t b_vnni[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
    OZAKI_PANEL_REFORMAT_B_XOR(b, ldb, kb, N, b_vnni);
    for (kk = 0; kk < BLOCK_K; kk += 4) {
      const __m512i vb = _mm512_loadu_si512((const __m512i*)(b_vnni + (kk >> 2) * N));
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
      for (mi = 0; mi < M; ++mi) {
        const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kb + kk));
        acc[mi] = _mm512_dpbusd_epi32(acc[mi], va, vb);
      }
    }
  }
  for (mi = 0; mi < M; ++mi) {
    int32_t asum = 0;
    GEMM_INT_TYPE k;
    for (k = 0; k < K; ++k) asum += (int32_t)a[mi * lda + k];
    acc[mi] = _mm512_add_epi32(acc[mi], _mm512_set1_epi32(128 * asum));
    _mm512_storeu_si512((__m512i*)(c + mi * ldc), acc[mi]);
  }
}

#endif /* LIBXS_INTRINSICS_AVX512 && BLOCK_N==16 && BLOCK_K valid */


/* AMX tile GEMM kernels: compute full 16x16 output tile via TDPBUSD.
 * A is loaded as 16 rows x 64 K-bytes per tile operation.
 * B is reformatted to VNNI layout: b_tile[16][16] int32 where each
 * int32 packs 4 consecutive K-bytes from one column (same as panel B
 * format, but 64 bytes deep = one full AMX tile-B).
 * K is processed in chunks of 64; any K-tail falls through to VNNI. */
#if defined(LIBXS_INTRINSICS_AMX) && 16 == BLOCK_M && 16 == BLOCK_N

#define OZAKI_AMX_TILE_C 0
#define OZAKI_AMX_TILE_A 1
#define OZAKI_AMX_TILE_B 2

typedef struct { uint8_t data[64]; } ozaki_amx_tilecfg_t;

LIBXS_API_INLINE void ozaki_amx_tilecfg_init(ozaki_amx_tilecfg_t* cfg,
  int m_rows, int k_bytes)
{
  memset(cfg, 0, sizeof(*cfg));
  cfg->data[0] = 1; /* palette_id */
  /* tile 0 (C): m_rows x 64 colsb (16 int32 columns) */
  *(uint16_t*)(cfg->data + 16 + OZAKI_AMX_TILE_C * 2) = 64;
  cfg->data[48 + OZAKI_AMX_TILE_C] = (uint8_t)m_rows;
  /* tile 1 (A): m_rows x k_bytes colsb */
  *(uint16_t*)(cfg->data + 16 + OZAKI_AMX_TILE_A * 2) = (uint16_t)k_bytes;
  cfg->data[48 + OZAKI_AMX_TILE_A] = (uint8_t)m_rows;
  /* tile 2 (B): (k_bytes/4) rows x 64 colsb (16 columns, 4 bytes packed) */
  *(uint16_t*)(cfg->data + 16 + OZAKI_AMX_TILE_B * 2) = 64;
  cfg->data[48 + OZAKI_AMX_TILE_B] = (uint8_t)(k_bytes / 4);
}


/* AMX u8*u8 panel via TDPBUSD with on-the-fly B reformat + XOR.
 * B is column-contiguous raw u8 residues (transb='T', ldb=K_grp_pad).
 * Reformats 16 columns x 64 K-bytes into VNNI tile with XOR 0x80,
 * then TDPBUSD (u8*s8). Correct via +128*row_sum(A). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_AMX)
void ozaki_panel_u8_amx(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const uint8_t* a, GEMM_INT_TYPE lda, const uint8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  ozaki_amx_tilecfg_t cfg;
  LIBXS_ALIGNED(int32_t c_buf[BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
  LIBXS_ALIGNED(int32_t b_tile[16 * BLOCK_N], LIBXS_ALIGNMENT);
  const int c_stride = BLOCK_N * (int)sizeof(int32_t);
  GEMM_INT_TYPE mi, kb;
  LIBXS_ASSERT(M <= BLOCK_M && N == BLOCK_N);

  if (0 != beta) {
    for (mi = 0; mi < M; ++mi) memcpy(c_buf + mi * BLOCK_N, c + mi * ldc, (size_t)N * sizeof(int32_t));
  }

  if (0 < (K & ~(GEMM_INT_TYPE)63)) {
    ozaki_amx_tilecfg_init(&cfg, (int)M, 64);
    _tile_loadconfig(&cfg);
    if (0 == beta) _tile_zero(OZAKI_AMX_TILE_C);
    else _tile_loadd(OZAKI_AMX_TILE_C, c_buf, c_stride);
    for (kb = 0; kb < (K & ~(GEMM_INT_TYPE)63); kb += 64) {
      int qi;
      for (qi = 0; qi < 16; ++qi) {
        int nj;
        for (nj = 0; nj < BLOCK_N; ++nj) {
          int32_t tmp;
          memcpy(&tmp, b + (long)nj * ldb + kb + qi * 4, 4);
          b_tile[qi * BLOCK_N + nj] = tmp ^ (int32_t)0x80808080;
        }
      }
      _tile_loadd(OZAKI_AMX_TILE_A, a + kb, (int)lda);
      _tile_loadd(OZAKI_AMX_TILE_B, b_tile, c_stride);
      _tile_dpbusd(OZAKI_AMX_TILE_C, OZAKI_AMX_TILE_A, OZAKI_AMX_TILE_B);
    }
    _tile_stored(OZAKI_AMX_TILE_C, c_buf, c_stride);
    _tile_release();
  }

  if (kb < K) {
    const __m512i bxor = _mm512_set1_epi32((int32_t)0x80808080);
    for (mi = 0; mi < M; ++mi) {
      __m512i acc = (0 == kb && 0 == beta) ? _mm512_setzero_si512()
        : _mm512_loadu_si512((__m512i*)(c_buf + mi * BLOCK_N));
      { GEMM_INT_TYPE kk;
        for (kk = kb; kk < K; kk += 4) {
          int32_t bq[BLOCK_N];
          int nj;
          for (nj = 0; nj < BLOCK_N; ++nj) { int32_t t; memcpy(&t, b + (long)nj * ldb + kk, 4); bq[nj] = t ^ (int32_t)0x80808080; }
          { const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kk));
            acc = _mm512_dpbusd_epi32(acc, va, _mm512_loadu_si512((__m512i*)bq));
          }
        }
      }
      _mm512_storeu_si512((__m512i*)(c_buf + mi * BLOCK_N), acc);
    }
  }

  for (mi = 0; mi < M; ++mi) {
    int32_t asum = 0;
    GEMM_INT_TYPE k;
    __m512i vacc = _mm512_loadu_si512((__m512i*)(c_buf + mi * BLOCK_N));
    for (k = 0; k < K; ++k) asum += (int32_t)a[mi * lda + k];
    vacc = _mm512_add_epi32(vacc, _mm512_set1_epi32(128 * asum));
    _mm512_storeu_si512((__m512i*)(c + mi * ldc), vacc);
  }
}


/* AMX s8*s8 panel via TDPBUSD with on-the-fly A XOR + B reformat.
 * XOR A with 0x80 for TDPBUSD (u8*s8); subtract 128*column_sum(B). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_AMX)
void ozaki_panel_i8_amx(GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const int8_t* a, GEMM_INT_TYPE lda, const int8_t* b, GEMM_INT_TYPE ldb,
  int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
  ozaki_amx_tilecfg_t cfg;
  LIBXS_ALIGNED(int32_t c_buf[BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
  LIBXS_ALIGNED(int32_t b_tile[16 * BLOCK_N], LIBXS_ALIGNMENT);
  LIBXS_ALIGNED(uint8_t a_biased[BLOCK_M * 64], LIBXS_ALIGNMENT);
  const int c_stride = BLOCK_N * (int)sizeof(int32_t);
  GEMM_INT_TYPE mi, kb;
  LIBXS_ASSERT(M <= BLOCK_M && N == BLOCK_N);

  if (0 != beta) {
    for (mi = 0; mi < M; ++mi) memcpy(c_buf + mi * BLOCK_N, c + mi * ldc, (size_t)N * sizeof(int32_t));
  }

  if (0 < (K & ~(GEMM_INT_TYPE)63)) {
    ozaki_amx_tilecfg_init(&cfg, (int)M, 64);
    _tile_loadconfig(&cfg);
    if (0 == beta) _tile_zero(OZAKI_AMX_TILE_C);
    else _tile_loadd(OZAKI_AMX_TILE_C, c_buf, c_stride);
    for (kb = 0; kb < (K & ~(GEMM_INT_TYPE)63); kb += 64) {
      int qi;
      GEMM_INT_TYPE ki;
      for (qi = 0; qi < 16; ++qi) {
        int nj;
        for (nj = 0; nj < BLOCK_N; ++nj) {
          memcpy(b_tile + qi * BLOCK_N + nj, b + (long)nj * ldb + kb + qi * 4, 4);
        }
      }
      for (mi = 0; mi < M; ++mi) {
        for (ki = 0; ki < 64; ++ki) {
          a_biased[mi * 64 + ki] = (uint8_t)((unsigned char)a[mi * lda + kb + ki] ^ 0x80u);
        }
      }
      _tile_loadd(OZAKI_AMX_TILE_A, a_biased, 64);
      _tile_loadd(OZAKI_AMX_TILE_B, b_tile, c_stride);
      _tile_dpbusd(OZAKI_AMX_TILE_C, OZAKI_AMX_TILE_A, OZAKI_AMX_TILE_B);
    }
    _tile_stored(OZAKI_AMX_TILE_C, c_buf, c_stride);
    _tile_release();
  }

  if (kb < K) {
    const __m512i bias = _mm512_set1_epi32((int32_t)0x80808080);
    for (mi = 0; mi < M; ++mi) {
      __m512i acc = (0 == kb && 0 == beta) ? _mm512_setzero_si512()
        : _mm512_loadu_si512((__m512i*)(c_buf + mi * BLOCK_N));
      { GEMM_INT_TYPE kk;
        for (kk = kb; kk < K; kk += 4) {
          int32_t bq[BLOCK_N];
          int nj;
          for (nj = 0; nj < BLOCK_N; ++nj) memcpy(bq + nj, b + (long)nj * ldb + kk, 4);
          { const __m512i va = _mm512_xor_si512(
              _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kk)), bias);
            acc = _mm512_dpbusd_epi32(acc, va, _mm512_loadu_si512((__m512i*)bq));
          }
        }
      }
      _mm512_storeu_si512((__m512i*)(c_buf + mi * BLOCK_N), acc);
    }
  }

  { __m512i bsum = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi32(0x01010101);
    GEMM_INT_TYPE kk;
    for (kk = 0; kk < K; kk += 4) {
      int32_t bq[BLOCK_N];
      int nj;
      for (nj = 0; nj < BLOCK_N; ++nj) memcpy(bq + nj, b + (long)nj * ldb + kk, 4);
      bsum = _mm512_dpbusd_epi32(bsum, ones, _mm512_loadu_si512((__m512i*)bq));
    }
    { const __m512i correction = _mm512_mullo_epi32(_mm512_set1_epi32(128), bsum);
      for (mi = 0; mi < M; ++mi) {
        __m512i vacc = _mm512_loadu_si512((__m512i*)(c_buf + mi * BLOCK_N));
        vacc = _mm512_sub_epi32(vacc, correction);
        _mm512_storeu_si512((__m512i*)(c + mi * ldc), vacc);
      }
    }
  }
}

#endif /* LIBXS_INTRINSICS_AMX && BLOCK_M==16 && BLOCK_N==16 */


/* u8*u8 -> s32 GEMM. */
LIBXS_API_INLINE void ozaki_gemm_u8u8s32(char transa, char transb, GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const uint8_t* a, GEMM_INT_TYPE lda, const uint8_t* b, GEMM_INT_TYPE ldb, int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
#if defined(LIBXS_INTRINSICS_AMX) && 16 == BLOCK_M && 16 == BLOCK_N
  if (N == BLOCK_N && 0 == (K % BLOCK_K)) {
# if (LIBXS_X86_AVX512_AMX <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_u8_amx(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
# elif (LIBXS_X86_AVX512_AMX <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512_AMX <= ozaki_target_arch) {
      ozaki_panel_u8_amx(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
# endif
  }
#endif
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
  if (N == BLOCK_N && 0 == (K % BLOCK_K)) {
# if (LIBXS_X86_AVX512_INT8 <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_u8_buud(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_u8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
# elif (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) {
      ozaki_panel_u8_buud(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
    else if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
      ozaki_panel_u8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
# elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
      ozaki_panel_u8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
# endif
  }
#endif
  OZAKI_GEMM_INT8_BODY(uint8_t, ozaki_dot_u8_sw)
}

/* s8*s8 -> s32 GEMM.  With __DNNL: delegates to dnnl_gemm_s8s8s32. */
LIBXS_API_INLINE void ozaki_gemm_s8s8s32(char transa, char transb, GEMM_INT_TYPE M, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  const int8_t* a, GEMM_INT_TYPE lda, const int8_t* b, GEMM_INT_TYPE ldb, int beta, int32_t* c, GEMM_INT_TYPE ldc)
{
#if defined(__DNNL)
  static const int32_t zero = 0;
  dnnl_gemm_s8s8s32(transa, transb, 'F', (dnnl_dim_t)M, (dnnl_dim_t)N, (dnnl_dim_t)K, 1.0f, a, (dnnl_dim_t)lda, 0, b,
    (dnnl_dim_t)ldb, 0, 0 != beta ? 1.0f : 0.0f, c, (dnnl_dim_t)ldc, &zero);
#else
# if defined(LIBXS_INTRINSICS_AMX) && 16 == BLOCK_M && 16 == BLOCK_N
  if (N == BLOCK_N && 0 == (K % BLOCK_K)) {
#   if (LIBXS_X86_AVX512_AMX <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_i8_amx(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
#   elif (LIBXS_X86_AVX512_AMX <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512_AMX <= ozaki_target_arch) {
      ozaki_panel_i8_amx(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
#   endif
  }
# endif
# if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
  if (N == BLOCK_N && 0 == (K % BLOCK_K)) {
#   if (LIBXS_X86_AVX512_INT8 <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_i8_bssd(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
#   elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    ozaki_panel_i8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
    return;
#   elif (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) {
      ozaki_panel_i8_bssd(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
    else if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
      ozaki_panel_i8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
#   elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
    if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
      ozaki_panel_i8_vnni(M, N, K, a, lda, b, ldb, beta, c, ldc);
      return;
    }
#   endif
  }
# endif
  OZAKI_GEMM_INT8_BODY(int8_t, ozaki_dot_i8_sw)
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


/** Check whether a per-call diff exceeds configured thresholds. */
LIBXS_API_INLINE int ozaki_diff_exceeds(const libxs_matdiff_t* diff)
{
  return (NULL != diff &&
    (diff->rsq < ozaki_rsq ||
    (1 == ozaki_idx && ozaki_eps < diff->linf_abs) ||
    (2 == ozaki_idx && ozaki_eps < diff->linf_rel) ||
    (3 == ozaki_idx && ozaki_eps < diff->l2_rel) ||
    ozaki_eps < libxs_matdiff_epsilon(diff)));
}


/** Run reference BLAS on c_ref, compute matdiff vs c, repair c if diff exceeds threshold. */
LIBXS_API_INLINE void ozaki_diff_reference(GEMM_ARGDECL, GEMM_REAL_TYPE* c_ref, size_t c_size, libxs_matdiff_t* diff)
{
  gemm_nozaki = 1;
  if (NULL != gemm_original) {
    gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
  }
  else {
    GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
  }
  gemm_nozaki = 0;
  libxs_matdiff(diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), *m, *n, c_ref, c, ldc, ldc);
  if (ozaki_diff_exceeds(diff)) memcpy(c, c_ref, c_size);
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
  const int rid = libxs_rid();
  int result = EXIT_SUCCESS;
  FILE* file;

  gemm_mhd_settings_t settings;
  settings.ozaki = ozaki;
  settings.ozn = ozaki_n;
  settings.ozflags = ozaki_flags;
  settings.oztrim = ozaki_trim;
  settings.ldc = *ldc;

  LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);

  if (0 > slurm) LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%u-%i-a.mhd", rid, gemm_diff.r);
  else LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%i-%u-%i-a.mhd", slurm, rid, gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    const int result_a = gemm_mhd_write(fname, a, *m, *k, *lda, *transa, alpha, ncomponents, &settings);
    if (EXIT_SUCCESS != result_a && 0 != ozaki_verbose) {
      fprintf(stderr, "ERROR: dumping A-matrix to %s failed!\n", fname);
    }
    result |= result_a;
  }
  else fclose(file);

  if (0 > slurm) LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%u-%i-b.mhd", rid, gemm_diff.r);
  else LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%i-%u-%i-b.mhd", slurm, rid, gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    const int result_b = gemm_mhd_write(fname, b, *k, *n, *ldb, *transb, beta, ncomponents, &settings);
    if (EXIT_SUCCESS != result_b && 0 != ozaki_verbose) {
      fprintf(stderr, "ERROR: dumping B-matrix to %s failed!\n", fname);
    }
    result |= result_b;
  }
  else fclose(file);

  if (0 > ozaki_dump) {
    if (0 > slurm) LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%u-%i-c.mhd", rid, gemm_diff.r);
    else LIBXS_SNPRINTF(fname, sizeof(fname), GEMM_LABEL "-%i-%u-%i-c.mhd", slurm, rid, gemm_diff.r);
    file = fopen(fname, "rb");
    if (NULL == file) { /* Never overwrite an existing file */
      const GEMM_REAL_TYPE scale[] = { 0, 0 };
      const char transc = 'N';
      const int result_c = gemm_mhd_write(fname, c, *m, *n, *ldc, transc, scale, ncomponents, &settings);
      if (EXIT_SUCCESS != result_c && 0 != ozaki_verbose) {
        fprintf(stderr, "ERROR: dumping C-matrix to %s failed!\n", fname);
      }
      result |= result_c;
    }
    else fclose(file);
  }

  if (0 < ozaki_verbose) {
    fprintf(stderr, GEMM_LABEL "[%i.%i]: ", gemm_diff.r, rid);
    print_gemm(stderr, 2 /*compact*/, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  if (EXIT_SUCCESS == result) { /* avoid repeated dumps */
    switch(ozaki_idx) {
      case 1: ozaki_eps = gemm_diff.linf_abs; break;
      case 2: ozaki_eps = gemm_diff.linf_rel; break;
      case 3: ozaki_eps = gemm_diff.l2_rel; break;
      default: ozaki_eps = libxs_matdiff_epsilon(&gemm_diff);
    }
    ozaki_rsq = gemm_diff.rsq;
  }

  LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  return result;
}


/**
 * Post-diff processing: accumulate into global diff, verbose output,
 * matrix dumps, and conditional exit. Called after a _diff kernel returns.
 */
LIBXS_API_INLINE void ozaki_post_diff(GEMM_ARGDECL, const char* label, size_t ncomponents, libxs_matdiff_t* diff)
{
  libxs_matdiff_t call_diff = *diff;
  LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  libxs_matdiff_reduce(&gemm_diff, diff);
  call_diff.r = gemm_diff.r;
  LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  if (1 < ozaki_verbose || 0 > ozaki_verbose) {
    const int nth = (0 < ozaki_verbose ? ozaki_verbose : 1);
    if (0 == (call_diff.r % nth)) {
      if (0 <= ozaki_stat) print_diff(stderr, label, 0 /*detail*/, &call_diff);
      else {
        fprintf(stderr, "%s[%i.%i]: ", label, call_diff.r, libxs_rid());
        print_gemm(stderr, LIBXS_ABS(ozaki_stat), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
    }
  }
  if (ozaki_diff_exceeds(&call_diff) || -1 > ozaki_verbose
    || (-1 > ozaki_dump && call_diff.r == -ozaki_dump)
    || (call_diff.r == ozaki_dump))
  {
    print_diff(stderr, label, 0 /*detail*/, &call_diff);
    if (0 != gemm_dump_inhibit) {
      gemm_dump_inhibit = 2;
    }
    else {
      const int result = gemm_dump_matrices(GEMM_ARGPASS, ncomponents);
      if (0 != ozaki_exit) exit(EXIT_SUCCESS == result ? EXIT_FAILURE : result);
    }
  }
}
