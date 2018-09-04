/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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

#include "libxs_typedefs.h"

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if !defined(_WIN32) && !defined(__CYGWIN__) /* TODO: fully support calling convention */
#if 0 != ((LIBXS_PREFETCH) & 2/*AL2*/) \
 || 0 != ((LIBXS_PREFETCH) & 4/*AL2_JPST*/) \
 || 0 != ((LIBXS_PREFETCH) & 16/*AL2_AHEAD*/) \
 || 0 != ((LIBXS_PREFETCH) & 32/*AL1*/)
# define LIBXS_GEMM_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 8/*BL2_VIA_C*/) \
 || 0 != ((LIBXS_PREFETCH) & 64/*BL1*/)
# define LIBXS_GEMM_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 128/*CL1*/)
# define LIBXS_GEMM_PREFETCH_C(EXPR) (EXPR)
#endif
#endif
/** Secondary helper macros derived from the above group. */
#if defined(LIBXS_GEMM_PREFETCH_A)
# define LIBXS_NOPREFETCH_A(EXPR)
#else
# define LIBXS_NOPREFETCH_A(EXPR) EXPR
# define LIBXS_GEMM_PREFETCH_A(EXPR) 0
#endif
#if defined(LIBXS_GEMM_PREFETCH_B)
# define LIBXS_NOPREFETCH_B(EXPR)
#else
# define LIBXS_NOPREFETCH_B(EXPR) EXPR
# define LIBXS_GEMM_PREFETCH_B(EXPR) 0
#endif
#if defined(LIBXS_GEMM_PREFETCH_C)
# define LIBXS_NOPREFETCH_C(EXPR)
#else
# define LIBXS_NOPREFETCH_C(EXPR) EXPR
# define LIBXS_GEMM_PREFETCH_C(EXPR) 0
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

/** Automatically select a prefetch-strategy (libxs_get_gemm_xprefetch, etc.). */
#define LIBXS_PREFETCH_AUTO -1

/** Helper macro for BLAS-style prefixes. */
#define LIBXS_TPREFIX_NAME(TYPE) LIBXS_CONCATENATE(LIBXS_TPREFIX_, TYPE)
#define LIBXS_TPREFIX(TYPE, FUNCTION) LIBXS_CONCATENATE(LIBXS_TPREFIX_NAME(TYPE), FUNCTION)
#define LIBXS_TPREFIX_doubledouble d
#define LIBXS_TPREFIX_floatfloat s
#define LIBXS_TPREFIX_shortfloat ws
#define LIBXS_TPREFIX_shortint wi
/** Defaults if only the input type is specified. */
#define LIBXS_TPREFIX_double LIBXS_TPREFIX_doubledouble
#define LIBXS_TPREFIX_float LIBXS_TPREFIX_floatfloat
#define LIBXS_TPREFIX_short LIBXS_TPREFIX_shortint

/** Construct symbol name from a given real type name (float, double and short). */
#define LIBXS_GEMM_SYMBOL(TYPE)       LIBXS_FSYMBOL(LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_GEMV_SYMBOL(TYPE)       LIBXS_FSYMBOL(LIBXS_TPREFIX(TYPE, gemv))
#define LIBXS_GEMMFUNCTION_TYPE(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm_function))
#define LIBXS_GEMVFUNCTION_TYPE(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemv_function))
#define LIBXS_MMFUNCTION_TYPE(TYPE)   LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmfunction))
#define LIBXS_MMDISPATCH_SYMBOL(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmdispatch))
#define LIBXS_XBLAS_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_blas_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_XGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_YGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(LIBXS_XGEMM_SYMBOL(TYPE), _omp)

/* Construct prefix names, function type or dispatch function from given input and output types. */
#define LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE)    LIBXS_MMFUNCTION_TYPE(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)  LIBXS_MMDISPATCH_SYMBOL(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_TPREFIX_NAME2(ITYPE, OTYPE)       LIBXS_TPREFIX_NAME(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_TPREFIX2(ITYPE, OTYPE, FUNCTION)  LIBXS_TPREFIX(LIBXS_CONCATENATE(ITYPE, OTYPE), FUNCTION)

/** Helper macro for comparing selected types. */
#define LIBXS_EQUAL(T1, T2) LIBXS_CONCATENATE2(LIBXS_EQUAL_, T1, T2)
#define LIBXS_EQUAL_floatfloat 1
#define LIBXS_EQUAL_doubledouble 1
#define LIBXS_EQUAL_floatdouble 0
#define LIBXS_EQUAL_doublefloat 0

#if defined(LIBXS_GEMM_CONST)
# undef LIBXS_GEMM_CONST
# define LIBXS_GEMM_CONST const
#elif defined(LIBXS_GEMM_NONCONST) || defined(__OPENBLAS)
# define LIBXS_GEMM_CONST
#else
# define LIBXS_GEMM_CONST const
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

#if !defined(LIBXS_NO_BLAS)
# if !defined(__BLAS) || (0 != __BLAS)
#   define LIBXS_NO_BLAS 0
# else
#   define LIBXS_NO_BLAS 1
# endif
#endif

#if defined(LIBXS_BUILD)
# if defined(LIBXS_BUILD_EXT) && !defined(__STATIC)
#   define LIBXS_GEMM_SYMBOL_VISIBILITY LIBXS_APIEXT
# elif defined(LIBXS_NO_BLAS) && (1 == LIBXS_NO_BLAS)
#   define LIBXS_GEMM_SYMBOL_VISIBILITY LIBXS_API
# endif
#endif
#if !defined(LIBXS_GEMM_SYMBOL_VISIBILITY)
# define LIBXS_GEMM_SYMBOL_VISIBILITY LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE
#endif

#define LIBXS_GEMM_SYMBOL_BLAS(CONST, TYPE) LIBXS_GEMM_SYMBOL_VISIBILITY \
  void LIBXS_GEMM_SYMBOL(TYPE)(CONST char*, CONST char*, \
    CONST libxs_blasint*, CONST libxs_blasint*, CONST libxs_blasint*, \
    CONST TYPE*, CONST TYPE*, CONST libxs_blasint*, \
    CONST TYPE*, CONST libxs_blasint*, \
    CONST TYPE*, TYPE*, CONST libxs_blasint*);
#if (!defined(__BLAS) || (0 != __BLAS)) /* BLAS available */
# define LIBXS_GEMM_SYMBOL_DECL LIBXS_GEMM_SYMBOL_BLAS
#else
# define LIBXS_GEMM_SYMBOL_DECL(CONST, TYPE)
#endif

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXS_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_B))

/** Helper macro allowing NULL-requests (transposes) supplied by some default. */
#define LIBXS_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXS_GEMM_FLAGS( \
  NULL != ((const void*)(TRANSA)) ? (*(const char*)(TRANSA)) : (0 == (LIBXS_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'n' : 't'), \
  NULL != ((const void*)(TRANSB)) ? (*(const char*)(TRANSB)) : (0 == (LIBXS_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'n' : 't')) \
  | (~(LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** Inlinable GEMM exercising the compiler's code generation (macro template). TODO: only NN is supported and SP/DP matrices. */
#define LIBXS_INLINE_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxs_inline_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_A & LIBXS_FLAGS) ? 'n' : 't')); \
  const char libxs_inline_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_B & LIBXS_FLAGS) ? 'n' : 't')); \
  const libxs_blasint libxs_inline_xgemm_m_ = *(const libxs_blasint*)(M); /* must be specified */ \
  const libxs_blasint libxs_inline_xgemm_k_ = (NULL != ((void*)(K)) ? (*(const libxs_blasint*)(K)) : libxs_inline_xgemm_m_); \
  const libxs_blasint libxs_inline_xgemm_n_ = (NULL != ((void*)(N)) ? (*(const libxs_blasint*)(N)) : libxs_inline_xgemm_k_); \
  const libxs_blasint libxs_inline_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (*(const libxs_blasint*)(LDA)) : \
    (('n' == libxs_inline_xgemm_transa_ || *"N" == libxs_inline_xgemm_transa_) ? libxs_inline_xgemm_m_ : libxs_inline_xgemm_k_)); \
  const libxs_blasint libxs_inline_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (*(const libxs_blasint*)(LDB)) : \
    (('n' == libxs_inline_xgemm_transb_ || *"N" == libxs_inline_xgemm_transb_) ? libxs_inline_xgemm_k_ : libxs_inline_xgemm_n_)); \
  const libxs_blasint libxs_inline_xgemm_ldc_ = (NULL != ((void*)(LDC)) ? (*(const libxs_blasint*)(LDC)) : libxs_inline_xgemm_m_); \
  const OTYPE libxs_inline_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
  const OTYPE libxs_inline_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
  libxs_blasint libxs_inline_xgemm_ni_, libxs_inline_xgemm_mi_, libxs_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transa_ || *"N" == libxs_inline_xgemm_transa_); \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transb_ || *"N" == libxs_inline_xgemm_transb_); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_mi_ = 0; libxs_inline_xgemm_mi_ < libxs_inline_xgemm_m_; ++libxs_inline_xgemm_mi_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_MAX_K, LIBXS_AVG_K) \
    for (libxs_inline_xgemm_ki_ = 0; libxs_inline_xgemm_ki_ < libxs_inline_xgemm_k_; ++libxs_inline_xgemm_ki_) { \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_inline_xgemm_ni_ = 0; libxs_inline_xgemm_ni_ < libxs_inline_xgemm_n_; ++libxs_inline_xgemm_ni_) { \
        ((OTYPE*)(C))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_mi_] \
          = ((const ITYPE*)(B))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldb_+libxs_inline_xgemm_ki_] * \
           (((const ITYPE*)(A))[libxs_inline_xgemm_ki_*libxs_inline_xgemm_lda_+libxs_inline_xgemm_mi_] * libxs_inline_xgemm_alpha_) \
          + ((const OTYPE*)(C))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_mi_] * libxs_inline_xgemm_beta_; \
      } \
    } \
  } \
}

/** Map to appropriate BLAS function (or fall-back). The mapping is used e.g., inside of LIBXS_BLAS_XGEMM. */
#define LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, FUNCTION) LIBXS_CONCATENATE(LIBXS_BLAS_FUNCTION_, LIBXS_TPREFIX2(ITYPE, OTYPE, FUNCTION))
#if !defined(__BLAS) || (0 != __BLAS)
# define LIBXS_BLAS_FUNCTION_dgemm libxs_original_dgemm()
# define LIBXS_BLAS_FUNCTION_sgemm libxs_original_sgemm()
#else /* no BLAS */
# define LIBXS_BLAS_FUNCTION_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_INLINE_XGEMM(double, double, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
# define LIBXS_BLAS_FUNCTION_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_INLINE_XGEMM(float, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXS_BLAS_FUNCTION_wigemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(short, int, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define LIBXS_BLAS_FUNCTION_wsgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(short, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (macro template). */
#define LIBXS_BLAS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxs_blas_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_A & LIBXS_FLAGS) ? 'n' : 't')); \
  const char libxs_blas_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_B & LIBXS_FLAGS) ? 'n' : 't')); \
  const libxs_blasint *const libxs_blas_xgemm_k_ = (NULL != ((void*)(K)) ? (K) : (M)); \
  const libxs_blasint *const libxs_blas_xgemm_n_ = (NULL != ((void*)(N)) ? (N) : libxs_blas_xgemm_k_); \
  const libxs_blasint *const libxs_blas_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (LDA) : \
    (('n' == libxs_blas_xgemm_transa_ || *"N" == libxs_blas_xgemm_transa_) ? (M) : libxs_blas_xgemm_k_)); \
  const libxs_blasint *const libxs_blas_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (LDB) : \
    (('n' == libxs_blas_xgemm_transb_ || *"N" == libxs_blas_xgemm_transb_) ? libxs_blas_xgemm_k_ : libxs_blas_xgemm_n_)); \
  const libxs_blasint *const libxs_blas_xgemm_ldc_ = (NULL != ((void*)(LDC)) ? (LDC) : (M)); \
  const OTYPE libxs_blas_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
  const OTYPE libxs_blas_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
  LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(&libxs_blas_xgemm_transa_, &libxs_blas_xgemm_transb_, \
    M, libxs_blas_xgemm_n_, libxs_blas_xgemm_k_, \
    &libxs_blas_xgemm_alpha_, (const ITYPE*)(A), libxs_blas_xgemm_lda_, \
                                (const ITYPE*)(B), libxs_blas_xgemm_ldb_, \
     &libxs_blas_xgemm_beta_,       (ITYPE*)(C), libxs_blas_xgemm_ldc_); \
}

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXS_MMCALL_ABC(FN, A, B, C) \
  LIBXS_ASSERT(FN); FN(A, B, C)
#define LIBXS_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXS_NOPREFETCH_A(LIBXS_UNUSED(PA)); \
  LIBXS_NOPREFETCH_B(LIBXS_UNUSED(PB)); \
  LIBXS_NOPREFETCH_C(LIBXS_UNUSED(PC)); \
  LIBXS_ASSERT(FN); FN(A, B, C, \
    LIBXS_GEMM_PREFETCH_A(PA), \
    LIBXS_GEMM_PREFETCH_B(PB), \
    LIBXS_GEMM_PREFETCH_C(PC)); \
}

#if (0/*LIBXS_GEMM_PREFETCH_NONE*/ == LIBXS_PREFETCH)
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_PRF(FN, A, B, C, (A) + ((size_t)LDA) * (K), (B) + ((size_t)LDB) * (N), (C) + ((size_t)LDC) * (N))
#endif
#define LIBXS_MMCALL(FN, A, B, C, M, N, K) LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, M, K, M)

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXS_MNK_SIZE(M, N, K) (((unsigned long long)(M)) * ((unsigned long long)(N)) * ((unsigned long long)(K)))

/** Fall-back code paths: LIBXS_XGEMM_FALLBACK0, and LIBXS_XGEMM_FALLBACK1 (macro template). */
#if !defined(LIBXS_XGEMM_FALLBACK0)
# define LIBXS_XGEMM_FALLBACK0(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#if !defined(LIBXS_XGEMM_FALLBACK1)
# define LIBXS_XGEMM_FALLBACK1(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

/**
 * Execute a specialized function, or use a fall-back code path depending on threshold (macro template).
 * LIBXS_XGEMM_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_XGEMM_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const int libxs_xgemm_flags_ = LIBXS_GEMM_PFLAGS(TRANSA, TRANSB, LIBXS_FLAGS); \
  const libxs_blasint *const libxs_xgemm_k_ = (NULL != (K) ? (K) : (M)); \
  const libxs_blasint *const libxs_xgemm_n_ = (NULL != (N) ? (N) : libxs_xgemm_k_); \
  const libxs_blasint *const libxs_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (LDA) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? (M) : libxs_xgemm_k_)); \
  const libxs_blasint *const libxs_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (LDB) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? libxs_xgemm_k_ : libxs_xgemm_n_)); \
  const libxs_blasint *const libxs_xgemm_ldc_ = (NULL != (LDC) ? (LDC) : (M)); \
  if (((unsigned long long)(LIBXS_MAX_MNK)) >= LIBXS_MNK_SIZE(*(M), *libxs_xgemm_n_, *libxs_xgemm_k_)) { \
    const LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) libxs_mmfunction_ = LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)( \
      *(M), *libxs_xgemm_n_, *libxs_xgemm_k_, libxs_xgemm_lda_, libxs_xgemm_ldb_, libxs_xgemm_ldc_, \
      (const OTYPE*)(ALPHA), (const OTYPE*)(BETA), &libxs_xgemm_flags_, NULL); \
    if (NULL != libxs_mmfunction_) { \
      LIBXS_MMCALL_LDX(libxs_mmfunction_, (const ITYPE*)(A), (const ITYPE*)(B), (OTYPE*)(C), \
        *(M), *libxs_xgemm_n_, *libxs_xgemm_k_, *libxs_xgemm_lda_, *libxs_xgemm_ldb_, *libxs_xgemm_ldc_); \
    } \
    else { \
      const char libxs_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? 'n' : 't'); \
      const char libxs_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? 'n' : 't'); \
      const OTYPE libxs_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
      const OTYPE libxs_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
      LIBXS_XGEMM_FALLBACK0(ITYPE, OTYPE, &libxs_xgemm_transa_, &libxs_xgemm_transb_, \
        M, libxs_xgemm_n_, libxs_xgemm_k_, \
        &libxs_xgemm_alpha_, A, libxs_xgemm_lda_, \
                               B, libxs_xgemm_ldb_, \
         &libxs_xgemm_beta_, C, libxs_xgemm_ldc_); \
    } \
  } \
  else { \
    const char libxs_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? 'n' : 't'); \
    const char libxs_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? 'n' : 't'); \
    const OTYPE libxs_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
    const OTYPE libxs_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
    LIBXS_XGEMM_FALLBACK1(ITYPE, OTYPE, &libxs_xgemm_transa_, &libxs_xgemm_transb_, \
      M, libxs_xgemm_n_, libxs_xgemm_k_, \
      &libxs_xgemm_alpha_, A, libxs_xgemm_lda_, \
                             B, libxs_xgemm_ldb_, \
       &libxs_xgemm_beta_, C, libxs_xgemm_ldc_); \
  } \
}

/** Helper macro to setup a matrix with some initial values. */
#define LIBXS_MATRNG_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) { \
  /*const*/ double libxs_matrng_seed_ = (double)SEED; /* avoid constant conditional */ \
  const double libxs_matrng_scale_ = (SCALE) * libxs_matrng_seed_ + (SCALE); \
  const libxs_blasint libxs_matrng_ld_ = (libxs_blasint)LD; \
  libxs_blasint libxs_matrng_i_, libxs_matrng_j_; \
  if (0 != libxs_matrng_seed_) { \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < ((libxs_blasint)NCOLS); ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < ((libxs_blasint)NROWS); ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        (DST)[libxs_matrng_k_] = (TYPE)(libxs_matrng_scale_ / (1.0 + libxs_matrng_k_)); \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        (DST)[libxs_matrng_k_] = (TYPE)SEED; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const unsigned int libxs_matrng_maxval_ = ((unsigned int)NCOLS) * ((unsigned int)libxs_matrng_ld_); \
    const TYPE libxs_matrng_maxval2_ = (TYPE)(libxs_matrng_maxval_ / 2), libxs_matrng_inv_ = (TYPE)((SCALE) / libxs_matrng_maxval2_); \
    const size_t libxs_matrng_shuffle_ = libxs_shuffle(libxs_matrng_maxval_); \
    LIBXS_OMP_VAR(libxs_matrng_j_); OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < ((libxs_blasint)NCOLS); ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        (DST)[libxs_matrng_k_] = libxs_matrng_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
    } \
  } \
}

#define LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_ELIDE, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_SEQ(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_OMP(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_PRAGMA_OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)

/** Call libxs_gemm_print using LIBXS's GEMM-flags. */
#define LIBXS_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  LIBXS_GEMM_PRINT2(OSTREAM, PRECISION, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)
#define LIBXS_GEMM_PRINT2(OSTREAM, IPREC, OPREC, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  libxs_gemm_dprint2(OSTREAM, (libxs_gemm_precision)(IPREC), (libxs_gemm_precision)(OPREC), \
    /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'n' : 't'), \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'n' : 't'), \
    M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using e.g.,
 * ITK-SNAP or ParaView.
 */
LIBXS_API void libxs_gemm_print(void* ostream,
  libxs_gemm_precision precision, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);
LIBXS_API void libxs_gemm_print2(void* ostream,
  libxs_gemm_precision iprec, libxs_gemm_precision oprec, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);
LIBXS_API void libxs_gemm_dprint(void* ostream,
  libxs_gemm_precision precision, char transa, char transb,
  libxs_blasint m, libxs_blasint n, libxs_blasint k,
  double dalpha, const void* a, libxs_blasint lda,
  const void* b, libxs_blasint ldb,
  double dbeta, void* c, libxs_blasint ldc);
LIBXS_API void libxs_gemm_dprint2(void* ostream,
  libxs_gemm_precision iprec, libxs_gemm_precision oprec, char transa, char transb,
  libxs_blasint m, libxs_blasint n, libxs_blasint k,
  double dalpha, const void* a, libxs_blasint lda,
  const void* b, libxs_blasint ldb,
  double dbeta, void* c, libxs_blasint ldc);
LIBXS_API void libxs_gemm_xprint(void* ostream,
  libxs_xmmfunction kernel, const void* a, const void* b, void* c);

/** GEMM: fall-back prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_sgemm_function)(
  const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const float*, const float*, const libxs_blasint*, const float*, const libxs_blasint*,
  const float*, float*, const libxs_blasint*);
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_dgemm_function)(
  const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const double*, const double*, const libxs_blasint*, const double*, const libxs_blasint*,
  const double*, double*, const libxs_blasint*);

/** GEMV: fall-back prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_sgemv_function)(
  const char*, const libxs_blasint*, const libxs_blasint*,
  const float*, const float*, const libxs_blasint*, const float*, const libxs_blasint*,
  const float*, float*, const libxs_blasint*);
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_dgemv_function)(
  const char*, const libxs_blasint*, const libxs_blasint*,
  const double*, const double*, const libxs_blasint*, const double*, const libxs_blasint*,
  const double*, double*, const libxs_blasint*);

/** The original GEMM functions (SGEMM and DGEMM). */
LIBXS_API LIBXS_GEMM_WEAK libxs_dgemm_function libxs_original_dgemm(void);
LIBXS_API LIBXS_GEMM_WEAK libxs_sgemm_function libxs_original_sgemm(void);

/**
 * General dense matrix multiplication, which re-exposes LAPACK/BLAS
 * but allows to rely on LIBXS's defaults (libxs_config.h)
 * when supplying NULL-arguments in certain places.
 */
LIBXS_API void libxs_blas_xgemm(libxs_gemm_precision iprec, libxs_gemm_precision oprec,
  const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);

#define libxs_blas_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_blas_xgemm(LIBXS_GEMM_PRECISION_F64, LIBXS_GEMM_PRECISION_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxs_blas_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_blas_xgemm(LIBXS_GEMM_PRECISION_F32, LIBXS_GEMM_PRECISION_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

#define libxs_dgemm_omp(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_xgemm_omp(LIBXS_GEMM_PRECISION_F64, LIBXS_GEMM_PRECISION_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxs_sgemm_omp(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_xgemm_omp(LIBXS_GEMM_PRECISION_F32, LIBXS_GEMM_PRECISION_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Translates GEMM prefetch request into prefetch-enumeration (incl. FE's auto-prefetch). */
LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_xprefetch(const int* prefetch);
LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_prefetch(int prefetch);

#endif /*LIBXS_FRONTEND_H*/
