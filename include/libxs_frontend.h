/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#ifndef LIBXS_FRONTEND_H
#define LIBXS_FRONTEND_H

#include "libxs.h"
#include <assert.h>

/** Helper macro for GEMM argument permutation depending on storage scheme. */
#if (0 != LIBXS_COL_MAJOR)
# define LIBXS_LD(M, N) (M)
#else
# define LIBXS_LD(M, N) (N)
#endif

/** Helper macro for aligning a buffer for aligned loads/store instructions. */
#if (0 != (LIBXS_GEMM_FLAG_ALIGN_A & LIBXS_FLAGS) || 0 != (LIBXS_GEMM_FLAG_ALIGN_C & LIBXS_FLAGS))
# define LIBXS_ALIGN_LDST(POINTER) LIBXS_ALIGN2(POINTER, LIBXS_ALIGNMENT)
#else
# define LIBXS_ALIGN_LDST(POINTER) (POINTER)
#endif

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if 0 != ((LIBXS_PREFETCH) & 2) || 0 != ((LIBXS_PREFETCH) & 4)
# define LIBXS_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 8)
# define LIBXS_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0/*no scheme yet using C*/
# define LIBXS_PREFETCH_C(EXPR) (EXPR)
#endif
#if defined(LIBXS_PREFETCH_A)
# define LIBXS_NOPREFETCH_A(EXPR) NULL
#else
# define LIBXS_NOPREFETCH_A(EXPR) (EXPR)
# define LIBXS_PREFETCH_A(EXPR) NULL
#endif
#if defined(LIBXS_PREFETCH_B)
# define LIBXS_NOPREFETCH_B(EXPR) NULL
#else
# define LIBXS_NOPREFETCH_B(EXPR) (EXPR)
# define LIBXS_PREFETCH_B(EXPR) NULL
#endif
#if defined(LIBXS_PREFETCH_C)
# define LIBXS_NOPREFETCH_C(EXPR) NULL
#else
# define LIBXS_NOPREFETCH_C(EXPR) (EXPR)
# define LIBXS_PREFETCH_C(EXPR) NULL
#endif

/** Helper macro for GEMM function names (and similar functions). */
#define LIBXS_TPREFIX(REAL, FUNCTION) LIBXS_TPREFIX_##REAL(FUNCTION)
#define LIBXS_TPREFIX_double(FUNCTION) d##FUNCTION
#define LIBXS_TPREFIX_float(FUNCTION) s##FUNCTION

/** Check ILP64 configuration for sanity. */
#if (defined(MKL_ILP64) && 0 == LIBXS_ILP64)
# error "Inconsistent ILP64 configuration detected!"
#endif

/** MKL_DIRECT_CALL requires to include the MKL interface. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if (0 != LIBXS_ILP64 && !defined(MKL_ILP64))
#   error "Inconsistent ILP64 configuration detected!"
# endif
# if defined(LIBXS_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#elif (0 != LIBXS_ILP64)
/** Fallback prototype functions served by any compliant LAPACK/BLAS (ILP64). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const double*, const double*, const long long*, const double*, const long long*,
  const double*, double*, const long long*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const float*, const float*, const long long*, const float*, const long long*,
  const float*, float*, const long long*);
#else /*LP64*/
/** Fallback prototype functions served by any compliant LAPACK/BLAS (LP64). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#define LIBXS_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const char libxs_blas_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxs_blas_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  const REAL libxs_blas_xgemm_alpha_ = (REAL)(ALPHA), libxs_blas_xgemm_beta_ = (REAL)(BETA); \
  const libxs_blasint libxs_blas_xgemm_lda_ = (libxs_blasint)(0 != (LDA) ? (LDA) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
  const libxs_blasint libxs_blas_xgemm_ldb_ = (libxs_blasint)LIBXS_LD(LDB, K); \
  const libxs_blasint libxs_blas_xgemm_ldc_ = (libxs_blasint)(0 != (LDC) ? LIBXS_LD(LDC, N) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(LIBXS_LD(M, N), sizeof(REAL), LIBXS_ALIGNMENT)); \
  const libxs_blasint libxs_blas_xgemm_m_ = (libxs_blasint)LIBXS_LD(M, N); \
  const libxs_blasint libxs_blas_xgemm_n_ = (libxs_blasint)LIBXS_LD(N, libxs_blas_xgemm_lda_); \
  const libxs_blasint libxs_blas_xgemm_k_ = (libxs_blasint)LIBXS_LD(K, libxs_blas_xgemm_ldb_); \
  SYMBOL(&libxs_blas_xgemm_transa_, &libxs_blas_xgemm_transb_, \
    &libxs_blas_xgemm_m_, &libxs_blas_xgemm_n_, &libxs_blas_xgemm_k_, \
    &libxs_blas_xgemm_alpha_, (const REAL*)LIBXS_LD(A, B), &LIBXS_LD(libxs_blas_xgemm_lda_, libxs_blas_xgemm_m_), \
                                (const REAL*)LIBXS_LD(B, A), &libxs_blas_xgemm_ldb_, \
    &libxs_blas_xgemm_beta_, (REAL*)(C), &libxs_blas_xgemm_ldc_); \
}

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(float, LIBXS_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(double, LIBXS_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXS_BLAS_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXS_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Inlinable GEMM exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXS_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const REAL libxs_inline_xgemm_alpha_ = (REAL)(1 == (ALPHA) ? 1 : (-1 == (ALPHA) ? -1 : (ALPHA))); \
  const REAL libxs_inline_xgemm_beta_ = (REAL)(1 == (BETA) ? 1 : (0 == (BETA) ? 0 : (BETA))); \
  const INT libxs_inline_xgemm_lda_ = (INT)LIBXS_LD(0 != (LDA) ? (LDA) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT), N); \
  const INT libxs_inline_xgemm_ldc_ = (INT)(0 != (LDC) ? LIBXS_LD(LDC, N) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(LIBXS_LD(M, N), sizeof(REAL), LIBXS_ALIGNMENT)); \
  INT libxs_inline_xgemm_i_, libxs_inline_xgemm_j_, libxs_inline_xgemm_k_; \
  assert(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_j_ = 0; libxs_inline_xgemm_j_ < ((INT)LIBXS_LD(M, N)); ++libxs_inline_xgemm_j_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_MAX_K, LIBXS_AVG_K) \
    for (libxs_inline_xgemm_k_ = 0; libxs_inline_xgemm_k_ < (K); ++libxs_inline_xgemm_k_) { \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_inline_xgemm_i_ = 0; libxs_inline_xgemm_i_ < ((INT)LIBXS_LD(N, M)); ++libxs_inline_xgemm_i_) { \
        ((REAL*)(C))[libxs_inline_xgemm_i_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_j_] \
          = ((const REAL*)LIBXS_LD(B, A))[libxs_inline_xgemm_i_*(INT)LIBXS_LD(LDB, K)+libxs_inline_xgemm_k_] * \
           (((const REAL*)LIBXS_LD(A, B))[libxs_inline_xgemm_k_*libxs_inline_xgemm_lda_+libxs_inline_xgemm_j_] * libxs_inline_xgemm_alpha_) \
          + ((const REAL*)(C))[libxs_inline_xgemm_i_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_j_] * libxs_inline_xgemm_beta_; \
      } \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXS_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(float, libxs_blasint, LIBXS_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXS_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(double, libxs_blasint, LIBXS_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation. */
#define LIBXS_INLINE_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXS_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Fallback code paths: LIBXS_FALLBACK0, and LIBXS_FALLBACK1 (template). */
#if defined(LIBXS_FALLBACK_INLINE_GEMM)
# define LIBXS_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXS_FALLBACK1(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXS_MMCALL_ABC(FN, A, B, C) FN(LIBXS_LD(A, B), LIBXS_LD(B, A), C)
#define LIBXS_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXS_NOPREFETCH_A(LIBXS_UNUSED(LIBXS_LD(PA, PB))); \
  LIBXS_NOPREFETCH_B(LIBXS_UNUSED(LIBXS_LD(PB, PA))); \
  LIBXS_NOPREFETCH_C(LIBXS_UNUSED(PC)); \
  FN(LIBXS_LD(A, B), LIBXS_LD(B, A), C, \
    LIBXS_PREFETCH_A(LIBXS_LD(PA, PB)), \
    LIBXS_PREFETCH_B(LIBXS_LD(PB, PA)), \
    LIBXS_PREFETCH_C(PC)); \
}

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (template).
 * LIBXS_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXS_MAX_MNK)) >= \
     (((unsigned long long)(M)) * \
      ((unsigned long long)(N)) * \
      ((unsigned long long)(K)))) \
  { \
    const int libxs_xgemm_flags_ = (int)(FLAGS), libxs_xgemm_ldb_ = (int)(LDB); \
    const int libxs_xgemm_lda_ = (int)(0 != (LDA) ? (LDA) \
      /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
      : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
    const int libxs_xgemm_ldc_ = (int)(0 != (LDC) ? (LDC) \
      /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
      : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
    const REAL libxs_xgemm_alpha_ = (REAL)(ALPHA), libxs_xgemm_beta_ = (REAL)(BETA); \
    int libxs_xgemm_fallback_ = 0; \
    if (LIBXS_PREFETCH_NONE == LIBXS_PREFETCH) { \
      const LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, mmfunction)) libxs_xgemm_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, mmdispatch))((int)(M), (int)(N), (int)(K), \
          &libxs_xgemm_lda_, &libxs_xgemm_ldb_, &libxs_xgemm_ldc_, \
          &libxs_xgemm_alpha_, &libxs_xgemm_beta_, \
          &libxs_xgemm_flags_, 0); \
      if (0 != libxs_xgemm_function_) { \
        LIBXS_MMCALL_ABC(libxs_xgemm_function_, (const REAL*)(A), (const REAL*)(B), (REAL*)(C)); \
      } \
      else { \
        libxs_xgemm_fallback_ = 1; \
      } \
    } \
    else { \
      const int libxs_xgemm_prefetch_ = (LIBXS_PREFETCH); \
      const LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, mmfunction)) libxs_xgemm_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, mmdispatch))((int)(M), (int)(N), (int)(K), \
          &libxs_xgemm_lda_, &libxs_xgemm_ldb_, &libxs_xgemm_ldc_, \
          &libxs_xgemm_alpha_, &libxs_xgemm_beta_, \
          &libxs_xgemm_flags_, &libxs_xgemm_prefetch_); \
      if (0 != libxs_xgemm_function_) { \
        LIBXS_MMCALL_PRF(libxs_xgemm_function_, (const REAL*)(A), (const REAL*)(B), (REAL*)(C), \
          ((const REAL*)(A)) + libxs_xgemm_lda_ * (K), ((const REAL*)(B)) + libxs_xgemm_ldb_ * (N), \
          ((const REAL*)(C)) + libxs_xgemm_ldc_ * (N)); \
      } \
      else { \
        libxs_xgemm_fallback_ = 1; \
      } \
    } \
    if (0 != libxs_xgemm_fallback_) { \
      LIBXS_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXS_FALLBACK1(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Dispatched general dense matrix multiplication (single-precision). */
#define LIBXS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_XGEMM(float, libxs_blasint, LIBXS_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication (double-precision). */
#define LIBXS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_XGEMM(double, libxs_blasint, LIBXS_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication. */
#define LIBXS_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

#endif /*LIBXS_FRONTEND_H*/
