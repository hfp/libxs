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

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXS_ILP64)
typedef long long libxs_blasint;
#else
typedef int libxs_blasint;
#endif

/** Helper macro for GEMM argument permutation depending on storage scheme. */
#if (0 != LIBXS_ROW_MAJOR)
# define LIBXS_LD(M, N) (N)
#else
# define LIBXS_LD(M, N) (M)
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
#if !defined(LIBXS_PREFETCH_A)
# define LIBXS_PREFETCH_A(EXPR) 0
#endif
#if !defined(LIBXS_PREFETCH_B)
# define LIBXS_PREFETCH_B(EXPR) 0
#endif
#if !defined(LIBXS_PREFETCH_C)
# define LIBXS_PREFETCH_C(EXPR) 0
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
#else
/** Fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const double*, const double*, const libxs_blasint*, const double*, const libxs_blasint*,
  const double*, double*, const libxs_blasint*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const float*, const float*, const libxs_blasint*, const float*, const libxs_blasint*,
  const float*, float*, const libxs_blasint*);
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#define LIBXS_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const char libxs_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxs_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  const libxs_blasint libxs_m_ = (libxs_blasint)(M), libxs_n_ = (libxs_blasint)(N), libxs_k_ = (libxs_blasint)(K); \
  const libxs_blasint libxs_lda_ = (libxs_blasint)(0 != (LDA) ? LIBXS_MAX/*BLAS-conformance*/(LDA, M) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)), libxs_ldb_ = (libxs_blasint)(LDB); \
  const libxs_blasint libxs_ldc_ = (libxs_blasint)(0 != (LDC) ? LIBXS_MAX/*BLAS-conformance*/(LDC, M) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
  const REAL libxs_alpha_ = (REAL)(ALPHA), libxs_beta_ = (REAL)(BETA); \
  LIBXS_FSYMBOL(LIBXS_TPREFIX(REAL, gemm))(&libxs_transa_, &libxs_transb_, \
    &libxs_m_, &libxs_n_, &libxs_k_, &libxs_alpha_, \
    A, &libxs_lda_, B, &libxs_ldb_, \
    &libxs_beta_, C, &libxs_ldc_); \
}

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BSGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BXGEMM(float, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BDGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BXGEMM(double, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXS_BGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_BDGEMM(FLAGS, M, N, K, \
      (double)(ALPHA), (const double*)(A), LDA, (const double*)(B), LDB, \
      (double) (BETA), (double*)(C), LDC); \
  } \
  else {\
    LIBXS_BSGEMM(FLAGS, M, N, K, \
      (float)(ALPHA), (const float*)(A), LDA, (const float*)(B), LDB, \
      (float) (BETA), (float*)(C), LDC); \
  } \
}

/** Inlinable GEMM exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXS_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const REAL *const libxs_a_ = (const REAL*)(B), *const libxs_b_ = (const REAL*)(A); \
  const INT libxs_m_ = (INT)(M), libxs_n_ = (INT)(N); \
  const INT libxs_lda_ = (INT)(0 != (LDA) ? LIBXS_MAX/*BLAS-conformance*/(LDA, M) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
  const INT libxs_ldc_ = (INT)(0 != (LDC) ? LIBXS_MAX/*BLAS-conformance*/(LDC, M) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
    : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
  INT libxs_i_, libxs_j_, libxs_k_; \
  REAL *const libxs_c_ = (C); \
  assert(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXS_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
  for (libxs_j_ = 0; libxs_j_ < libxs_m_; ++libxs_j_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_LD(LIBXS_MAX_N, LIBXS_MAX_M), LIBXS_LD(LIBXS_AVG_N, LIBXS_AVG_M)) \
    for (libxs_i_ = 0; libxs_i_ < libxs_n_; ++libxs_i_) { \
      const INT libxs_index_ = libxs_i_ * libxs_ldc_ + libxs_j_; \
      REAL libxs_r_ = libxs_c_[libxs_index_] * (BETA); \
      LIBXS_PRAGMA_SIMD_REDUCTION(+:libxs_r_) \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_k_ = 0; libxs_k_ < (K); ++libxs_k_) { \
        libxs_r_ += libxs_a_[libxs_i_*(LDB)+libxs_k_] * (ALPHA) \
                    * libxs_b_[libxs_k_*libxs_lda_+libxs_j_]; \
      } \
      libxs_c_[libxs_index_] = libxs_r_; \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXS_ISGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_IXGEMM(float, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXS_IDGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_IXGEMM(double, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation. */
#define LIBXS_IGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_IDGEMM(FLAGS, M, N, K, \
      (double)(ALPHA), (const double*)(A), LDA, (const double*)(B), LDB, \
      (double) (BETA), (double*)(C), LDC); \
  } \
  else {\
    LIBXS_ISGEMM(FLAGS, M, N, K, \
      (float)(ALPHA), (const float*)(A), LDA, (const float*)(B), LDB, \
      (float) (BETA), (float*)(C), LDC); \
  } \
}

/** Fallback code paths: LIBXS_FALLBACK0, and LIBXS_FALLBACK1 (template). */
#if defined(LIBXS_FALLBACK_IGEMM)
# define LIBXS_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXS_FALLBACK1(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (template).
 * LIBXS_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_GEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXS_MAX_MNK)) >= \
     (((unsigned long long)(M)) * \
      ((unsigned long long)(N)) * \
      ((unsigned long long)(K)))) \
  { \
    const int libxs_flags_ = (int)(FLAGS), libxs_ldb_ = (int)(LDB); \
    const int libxs_lda_ = (int)(0 != (LDA) ? LIBXS_MAX/*BLAS-conformance*/(LDA, M) \
      /* if the value of LDA was zero: make LDA a multiple of LIBXS_ALIGNMENT */ \
      : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
    const int libxs_ldc_ = (int)(0 != (LDC) ? LIBXS_MAX/*BLAS-conformance*/(LDC, M) \
      /* if the value of LDC was zero: make LDC a multiple of LIBXS_ALIGNMENT */ \
      : LIBXS_ALIGN_VALUE(M, sizeof(REAL), LIBXS_ALIGNMENT)); \
    const REAL libxs_alpha_ = (REAL)(ALPHA), libxs_beta_ = (REAL)(BETA); \
    int libxs_fallback_ = 0; \
    if (LIBXS_PREFETCH_NONE == LIBXS_PREFETCH) { \
      const LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, function)) libxs_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, dispatch))((int)(M), (int)(N), (int)(K), \
          &libxs_lda_, &libxs_ldb_, &libxs_ldc_, &libxs_alpha_, &libxs_beta_, &libxs_flags_, 0); \
      if (0 != libxs_function_) { \
        libxs_function_(A, B, C); \
      } \
      else { \
        libxs_fallback_ = 1; \
      } \
    } \
    else { \
      const int libxs_prefetch_ = (LIBXS_PREFETCH); \
      const LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, function)) libxs_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(REAL, dispatch))((int)(M), (int)(N), (int)(K), \
          &libxs_lda_, &libxs_ldb_, &libxs_ldc_, &libxs_alpha_, &libxs_beta_, &libxs_flags_, &libxs_prefetch_); \
      if (0 != libxs_function_) { \
        libxs_function_(A, B, C, \
          0 != LIBXS_PREFETCH_A(1) ? (((const REAL*)(A)) + (libxs_lda_) * (K)) : ((const REAL*)(A)), \
          0 != LIBXS_PREFETCH_B(1) ? (((const REAL*)(B)) + (libxs_ldb_) * (N)) : ((const REAL*)(B)), \
          0 != LIBXS_PREFETCH_C(1) ? (((const REAL*)(C)) + (libxs_ldc_) * (N)) : ((const REAL*)(C))); \
      } \
      else { \
        libxs_fallback_ = 1; \
      } \
    } \
    if (0 != libxs_fallback_) { \
      LIBXS_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXS_FALLBACK1(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

#endif /*LIBXS_FRONTEND_H*/
