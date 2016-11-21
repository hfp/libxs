/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include "libxs_macros.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <assert.h> /* intentionally here */
#include <stdint.h>
#include "libxs_generator.h"
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Helper macro for GEMM argument permutation depending on storage scheme. */
#define LIBXS_LD(M, N) (M)

/** Used to sanitize GEMM arguments (LDx vs. M/N/K). */
#if defined(LIBXS_SANITIZE_GEMM)
# define LIBXS_MAX2(A, B) LIBXS_MAX(A, B)
#else /* Argument B is not considered; pass-through A. */
# define LIBXS_MAX2(A, B) (A)
#endif

/** Helper macro for aligning a buffer for aligned loads/store instructions. */
#if (0 != (4/*LIBXS_GEMM_FLAG_ALIGN_A*/ & LIBXS_FLAGS) || 0 != (8/*LIBXS_GEMM_FLAG_ALIGN_C*/ & LIBXS_FLAGS))
# define LIBXS_ALIGN_LDST(POINTER) LIBXS_ALIGN2(POINTER, LIBXS_ALIGNMENT)
#else
# define LIBXS_ALIGN_LDST(POINTER) (POINTER)
#endif

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if !defined(_WIN32) /* disable prefetch due to issues with the calling convention */
#if 0 != ((LIBXS_PREFETCH) & 2/*AL2*/) || 0 != ((LIBXS_PREFETCH) & 4/*AL2_JPST*/)
# define LIBXS_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 8/*BL2_VIA_C*/)
# define LIBXS_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 32/*CL2*/)
# define LIBXS_PREFETCH_C(EXPR) (EXPR)
#endif
#endif
/** Secondary helper macros derived from the above group. */
#if defined(LIBXS_PREFETCH_A)
# define LIBXS_NOPREFETCH_A(EXPR)
#else
# define LIBXS_NOPREFETCH_A(EXPR) EXPR
# define LIBXS_PREFETCH_A(EXPR) 0
#endif
#if defined(LIBXS_PREFETCH_B)
# define LIBXS_NOPREFETCH_B(EXPR)
#else
# define LIBXS_NOPREFETCH_B(EXPR) EXPR
# define LIBXS_PREFETCH_B(EXPR) 0
#endif
#if defined(LIBXS_PREFETCH_C)
# define LIBXS_NOPREFETCH_C(EXPR)
#else
# define LIBXS_NOPREFETCH_C(EXPR) EXPR
# define LIBXS_PREFETCH_C(EXPR) 0
#endif

/** Helper macro for BLAS-style prefixes. */
#define LIBXS_TPREFIX_NAME(TYPE) LIBXS_CONCATENATE(LIBXS_TPREFIX_, TYPE)
#define LIBXS_TPREFIX(TYPE, SYMBOL) LIBXS_CONCATENATE(LIBXS_TPREFIX_NAME(TYPE), SYMBOL)
#define LIBXS_TPREFIX_double d
#define LIBXS_TPREFIX_float s

/** Helper macro for type postfixes. */
#define LIBXS_TPOSTFIX_NAME(TYPE) LIBXS_CONCATENATE(LIBXS_TPOSTFIX_, TYPE)
#define LIBXS_TPOSTFIX(TYPE, SYMBOL) LIBXS_CONCATENATE(SYMBOL, LIBXS_TPOSTFIX_NAME(TYPE))
#define LIBXS_TPOSTFIX_double F64
#define LIBXS_TPOSTFIX_float F32

/** Helper macro for comparing types. */
#define LIBXS_EQUAL(T1, T2, R) LIBXS_CONCATENATE(LIBXS_CONCATENATE(LIBXS_EQUAL_, T1), T2)(R)
#define LIBXS_EQUAL_doubledouble(R) R
#define LIBXS_EQUAL_floatfloat(R) R

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
#endif
#if (0 != LIBXS_ILP64)
/** Fallback prototype functions served by any compliant LAPACK/BLAS (ILP64). */
typedef LIBXS_RETARGETABLE void (*libxs_sgemm_function)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const float*, const float*, const long long*, const float*, const long long*,
  const float*, float*, const long long*);
typedef LIBXS_RETARGETABLE void (*libxs_dgemm_function)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const double*, const double*, const long long*, const double*, const long long*,
  const double*, double*, const long long*);
# else /*LP64*/
/** Fallback prototype functions served by any compliant LAPACK/BLAS (LP64). */
typedef LIBXS_RETARGETABLE void (*libxs_sgemm_function)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
typedef LIBXS_RETARGETABLE void (*libxs_dgemm_function)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
#endif

#if defined(LIBXS_BUILD_EXT)
# define LIBXS_WEAK
# define LIBXS_EXT_WEAK LIBXS_ATTRIBUTE_WEAK
#else
# define LIBXS_WEAK LIBXS_ATTRIBUTE_WEAK
# define LIBXS_EXT_WEAK
#endif
#if defined(LIBXS_BUILD) && defined(__STATIC) /*&& defined(LIBXS_GEMM_WRAP)*/
# define LIBXS_GEMM_WEAK LIBXS_WEAK
# define LIBXS_EXT_GEMM_WEAK LIBXS_EXT_WEAK
#else
# define LIBXS_GEMM_WEAK
# define LIBXS_EXT_GEMM_WEAK
#endif

/** The original GEMM functions (SGEMM and DGEMM). */
LIBXS_API LIBXS_GEMM_WEAK libxs_sgemm_function libxs_original_sgemm(const void* caller);
LIBXS_API LIBXS_GEMM_WEAK libxs_dgemm_function libxs_original_dgemm(const void* caller);

/** Construct symbol name from a given real type name (float or double). */
#define LIBXS_GEMM_TYPEFLAG(TYPE)     LIBXS_CONCATENATE(LIBXS_TPOSTFIX(TYPE, LIBXS_GEMM_FLAG_), PREC)
#define LIBXS_ORIGINAL_GEMM(TYPE)     LIBXS_CONCATENATE(libxs_original_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_BLAS_GEMM_SYMBOL(TYPE)  LIBXS_ORIGINAL_GEMM(TYPE)(LIBXS_CALLER)
#define LIBXS_GEMMFUNCTION_TYPE(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm_function))
#define LIBXS_MMFUNCTION_TYPE(TYPE)   LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmfunction))
#define LIBXS_MMDISPATCH_SYMBOL(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmdispatch))
#define LIBXS_XBLAS_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_blas_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_XGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_YGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(LIBXS_XGEMM_SYMBOL(TYPE), _omp)

/** Helper macro consolidating the applicable GEMM arguments into LIBXS's flags. */
#define LIBXS_GEMM_DECLARE_FLAGS(FLAGS, TRANSA, TRANSB) \
  int FLAGS = (0 != (TRANSA) \
    ? (('N' == *(TRANSA) || 'n' == *(TRANSA)) ? (LIBXS_FLAGS & ~LIBXS_GEMM_FLAG_TRANS_A) \
                                              : (LIBXS_FLAGS |  LIBXS_GEMM_FLAG_TRANS_A)) \
    : LIBXS_FLAGS); \
  FLAGS = (0 != (TRANSB) \
    ? (('N' == *(TRANSB) || 'n' == *(TRANSB)) ? ((FLAGS) & ~LIBXS_GEMM_FLAG_TRANS_B) \
                                              : ((FLAGS) |  LIBXS_GEMM_FLAG_TRANS_B)) \
    : (FLAGS)); \

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#if !defined(__BLAS) || (0 != __BLAS)
# define LIBXS_BLAS_XGEMM(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
    const char libxs_blas_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
    const char libxs_blas_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
    const TYPE libxs_blas_xgemm_alpha_ = (TYPE)(ALPHA), libxs_blas_xgemm_beta_ = (TYPE)(BETA); \
    const libxs_blasint libxs_blas_xgemm_lda_ = (libxs_blasint)LIBXS_MAX2(LIBXS_LD(LDA, LDB), LIBXS_LD(MM, NN)); \
    const libxs_blasint libxs_blas_xgemm_ldb_ = (libxs_blasint)LIBXS_MAX2(LIBXS_LD(LDB, LDA), KK); \
    const libxs_blasint libxs_blas_xgemm_ldc_ = (libxs_blasint)LIBXS_MAX2(LDC, LIBXS_LD(MM, NN)); \
    const libxs_blasint libxs_blas_xgemm_m_ = (libxs_blasint)LIBXS_LD(MM, NN); \
    const libxs_blasint libxs_blas_xgemm_n_ = (libxs_blasint)LIBXS_LD(NN, MM); \
    const libxs_blasint libxs_blas_xgemm_k_ = (libxs_blasint)(KK); \
    assert(0 != ((uintptr_t)LIBXS_BLAS_GEMM_SYMBOL(TYPE))); \
    LIBXS_BLAS_GEMM_SYMBOL(TYPE)(&libxs_blas_xgemm_transa_, &libxs_blas_xgemm_transb_, \
      &libxs_blas_xgemm_m_, &libxs_blas_xgemm_n_, &libxs_blas_xgemm_k_, \
      &libxs_blas_xgemm_alpha_, (const TYPE*)LIBXS_LD(A, B), &libxs_blas_xgemm_lda_, \
                                  (const TYPE*)LIBXS_LD(B, A), &libxs_blas_xgemm_ldb_, \
      &libxs_blas_xgemm_beta_, (TYPE*)(C), &libxs_blas_xgemm_ldc_); \
  }
#else
# define LIBXS_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_UNUSED(LDA); LIBXS_UNUSED(LDB); LIBXS_UNUSED(LDC); \
    LIBXS_UNUSED(M); LIBXS_UNUSED(N); LIBXS_UNUSED(K); \
    LIBXS_UNUSED(A); LIBXS_UNUSED(B); LIBXS_UNUSED(C); \
    LIBXS_UNUSED(ALPHA); LIBXS_UNUSED(BETA)
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(float, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(double, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
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
# define LIBXS_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const TYPE libxs_inline_xgemm_alpha_ = (TYPE)(ALPHA), libxs_inline_xgemm_beta_ = (TYPE)(BETA); \
  INT libxs_inline_xgemm_i_, libxs_inline_xgemm_j_, libxs_inline_xgemm_k_; \
  assert(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  /* TODO: remove/adjust precondition if anything other than NN is supported */ \
  assert(LIBXS_LD(M, N) <= LIBXS_LD(LDA, LDB) && (K) <= LIBXS_LD(LDB, LDA) && LIBXS_LD(M, N) <= (LDC)); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_j_ = 0; libxs_inline_xgemm_j_ < ((INT)LIBXS_LD(M, N)); ++libxs_inline_xgemm_j_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_MAX_K, LIBXS_AVG_K) \
    for (libxs_inline_xgemm_k_ = 0; libxs_inline_xgemm_k_ < (K); ++libxs_inline_xgemm_k_) { \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_inline_xgemm_i_ = 0; libxs_inline_xgemm_i_ < ((INT)LIBXS_LD(N, M)); ++libxs_inline_xgemm_i_) { \
        ((TYPE*)(C))[libxs_inline_xgemm_i_*((INT)(LDC))+libxs_inline_xgemm_j_] \
          = ((const TYPE*)LIBXS_LD(B, A))[libxs_inline_xgemm_i_*((INT)LIBXS_LD(LDB, LDA))+libxs_inline_xgemm_k_] * \
           (((const TYPE*)LIBXS_LD(A, B))[libxs_inline_xgemm_k_*((INT)LIBXS_LD(LDA, LDB))+libxs_inline_xgemm_j_] * libxs_inline_xgemm_alpha_) \
          + ((const TYPE*)(C))[libxs_inline_xgemm_i_*((INT)(LDC))+libxs_inline_xgemm_j_] * libxs_inline_xgemm_beta_; \
      } \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXS_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(float, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXS_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(double, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
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
# define LIBXS_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif defined(LIBXS_FALLBACK_OMPS)
# define LIBXS_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_OMPS_GEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#if defined(LIBXS_FALLBACK_OMPS)
# define LIBXS_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_OMPS_GEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXS_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

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

#if (0/*LIBXS_PREFETCH_NONE*/ == LIBXS_PREFETCH)
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_PRF(FN, A, B, C, (A) + (LDA) * (K), (B) + (LDB) * (N), (C) + (LDC) * (N))
#endif
#define LIBXS_MMCALL(FN, A, B, C, M, N, K) \
  LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LIBXS_LD(M, N), K, LIBXS_LD(M, N))

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXS_MNK_SIZE(M, N, K) (((unsigned long long)(M)) * ((unsigned long long)(N)) * ((unsigned long long)(K)))

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (template).
 * LIBXS_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXS_MAX_MNK)) >= LIBXS_MNK_SIZE(M, N, K)) { \
    const int libxs_xgemm_flags_ = (int)(FLAGS); \
    const int libxs_xgemm_lda_ = (int)(LDA), libxs_xgemm_ldb_ = (int)(LDB), libxs_xgemm_ldc_ = (int)(LDC); \
    const TYPE libxs_xgemm_alpha_ = (TYPE)(ALPHA), libxs_xgemm_beta_ = (TYPE)(BETA); \
    const LIBXS_MMFUNCTION_TYPE(TYPE) libxs_mmfunction_ = LIBXS_MMDISPATCH_SYMBOL(TYPE)( \
      (int)(M), (int)(N), (int)(K), &libxs_xgemm_lda_, &libxs_xgemm_ldb_, &libxs_xgemm_ldc_, \
      &libxs_xgemm_alpha_, &libxs_xgemm_beta_, &libxs_xgemm_flags_, 0); \
    if (0 != libxs_mmfunction_) { \
      LIBXS_MMCALL_LDX(libxs_mmfunction_, (const TYPE*)(A), (const TYPE*)(B), (TYPE*)(C), M, N, K, LDA, LDB, LDC); \
    } \
    else { \
      LIBXS_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXS_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Dispatched general dense matrix multiplication (single-precision). */
#define LIBXS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_XGEMM(float, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication (double-precision). */
#define LIBXS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_XGEMM(double, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
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
