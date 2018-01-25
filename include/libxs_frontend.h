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
#if 0 != ((LIBXS_PREFETCH) & 2/*AL2*/) \
 || 0 != ((LIBXS_PREFETCH) & 4/*AL2_JPST*/) \
 || 0 != ((LIBXS_PREFETCH) & 16/*AL2_AHEAD*/) \
 || 0 != ((LIBXS_PREFETCH) & 32/*AL1*/)
# define LIBXS_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 8/*BL2_VIA_C*/) \
 || 0 != ((LIBXS_PREFETCH) & 64/*BL1*/)
# define LIBXS_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 128/*CL1*/)
# define LIBXS_PREFETCH_C(EXPR) (EXPR)
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
#define LIBXS_TPREFIX_short w

/** Helper macro for type postfixes. */
#define LIBXS_TPOSTFIX_NAME(TYPE) LIBXS_CONCATENATE(LIBXS_TPOSTFIX_, TYPE)
#define LIBXS_TPOSTFIX(TYPE, SYMBOL) LIBXS_CONCATENATE(SYMBOL, LIBXS_TPOSTFIX_NAME(TYPE))
#define LIBXS_TPOSTFIX_double F64
#define LIBXS_TPOSTFIX_float F32
#define LIBXS_TPOSTFIX_int I32
#define LIBXS_TPOSTFIX_short I16

/** Helper macro for comparing types. */
#define LIBXS_EQUAL_CHECK(...) LIBXS_SELECT_HEAD(__VA_ARGS__, 0)
#define LIBXS_EQUAL(T1, T2) LIBXS_EQUAL_CHECK(LIBXS_CONCATENATE(LIBXS_CONCATENATE(LIBXS_EQUAL_, T1), T2))
#define LIBXS_EQUAL_floatfloat 1
#define LIBXS_EQUAL_doubledouble 1
#define LIBXS_EQUAL_floatdouble 0
#define LIBXS_EQUAL_doublefloat 0

/** Check ILP64 configuration for sanity. */
#if !defined(LIBXS_ILP64) || (0 == LIBXS_ILP64 && defined(MKL_ILP64))
# error "Inconsistent ILP64 configuration detected!"
#elif (0 != LIBXS_ILP64 && !defined(MKL_ILP64))
# define MKL_ILP64
#endif
#if (0 != LIBXS_ILP64)
# define LIBXS_BLASINT_NBITS 64
# define LIBXS_BLASINT long long
#else /* LP64 */
# define LIBXS_BLASINT_NBITS 32
# define LIBXS_BLASINT int
#endif

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
typedef LIBXS_BLASINT libxs_blasint;

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
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void(*libxs_sgemv_function)(
  const char*, const libxs_blasint*, const libxs_blasint*,
  const float*, const float*, const libxs_blasint*, const float*, const libxs_blasint*,
  const float*, float*, const libxs_blasint*);
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void(*libxs_dgemv_function)(
  const char*, const libxs_blasint*, const libxs_blasint*,
  const double*, const double*, const libxs_blasint*, const double*, const libxs_blasint*,
  const double*, double*, const libxs_blasint*);

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
LIBXS_API LIBXS_GEMM_WEAK libxs_sgemm_function libxs_original_sgemm(const char* caller);
LIBXS_API LIBXS_GEMM_WEAK libxs_dgemm_function libxs_original_dgemm(const char* caller);

/** Construct symbol name from a given real type name (float, double and short). */
#define LIBXS_DATATYPE(TYPE)          LIBXS_TPOSTFIX(TYPE, LIBXS_DATATYPE_)
#define LIBXS_GEMM_PRECISION(TYPE)    LIBXS_TPOSTFIX(TYPE, LIBXS_GEMM_PRECISION_)
#define LIBXS_GEMM_SYMBOL(TYPE)       LIBXS_FSYMBOL(LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_GEMV_SYMBOL(TYPE)       LIBXS_FSYMBOL(LIBXS_TPREFIX(TYPE, gemv))
#define LIBXS_GEMMFUNCTION_TYPE(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm_function))
#define LIBXS_GEMVFUNCTION_TYPE(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemv_function))
#define LIBXS_ORIGINAL_GEMM(TYPE)     LIBXS_CONCATENATE(libxs_original_, LIBXS_TPREFIX(TYPE, gemm))(LIBXS_CALLER)
#define LIBXS_MMFUNCTION_TYPE(TYPE)   LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmfunction))
#define LIBXS_MMDISPATCH_SYMBOL(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmdispatch))
#define LIBXS_XBLAS_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_blas_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_XGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_YGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(LIBXS_XGEMM_SYMBOL(TYPE), _omp)

#if defined(LIBXS_GEMM_CONST)
# undef LIBXS_GEMM_CONST
# define LIBXS_GEMM_CONST const
#elif defined(LIBXS_GEMM_NONCONST) || defined(__OPENBLAS)
# define LIBXS_GEMM_CONST
#else
# define LIBXS_GEMM_CONST const
#endif

#define LIBXS_GEMM_SYMBOL_DECL(CONST, TYPE) \
  LIBXS_API_EXTERN void LIBXS_GEMM_SYMBOL(TYPE)(CONST char*, CONST char*, \
    CONST libxs_blasint*, CONST libxs_blasint*, CONST libxs_blasint*, \
    CONST TYPE*, CONST TYPE*, CONST libxs_blasint*, \
    CONST TYPE*, CONST libxs_blasint*, \
    CONST TYPE*, TYPE*, CONST libxs_blasint*)

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXS_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('N' == (TRANSA) || 'n' == (TRANSA)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_A) \
  | (('N' == (TRANSB) || 'n' == (TRANSB)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_B))

/** Helper macro allowing NULL-requests (transposes) supplied by some default. */
#define LIBXS_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXS_GEMM_FLAGS( \
  0 != ((const void*)(TRANSA)) ? *((const char*)(TRANSA)) : (0 == (LIBXS_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'N' : 'T'), \
  0 != ((const void*)(TRANSB)) ? *((const char*)(TRANSB)) : (0 == (LIBXS_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'N' : 'T')) \
  | (~(LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#if !defined(__BLAS) || (0 != __BLAS)
# define LIBXS_BLAS_XGEMM(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
    const char libxs_blas_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
    const char libxs_blas_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
    const TYPE libxs_blas_xgemm_alpha_ = (TYPE)(ALPHA), libxs_blas_xgemm_beta_ = (TYPE)(BETA); \
    const libxs_blasint libxs_blas_xgemm_lda_ = (libxs_blasint)(LDA); \
    const libxs_blasint libxs_blas_xgemm_ldb_ = (libxs_blasint)(LDB); \
    const libxs_blasint libxs_blas_xgemm_ldc_ = (libxs_blasint)(LDC); \
    const libxs_blasint libxs_blas_xgemm_m_ = (libxs_blasint)(MM); \
    const libxs_blasint libxs_blas_xgemm_n_ = (libxs_blasint)(NN); \
    const libxs_blasint libxs_blas_xgemm_k_ = (libxs_blasint)(KK); \
    LIBXS_ASSERT(0 != ((uintptr_t)LIBXS_ORIGINAL_GEMM(TYPE))); \
    LIBXS_ORIGINAL_GEMM(TYPE)(&libxs_blas_xgemm_transa_, &libxs_blas_xgemm_transb_, \
      &libxs_blas_xgemm_m_, &libxs_blas_xgemm_n_, &libxs_blas_xgemm_k_, \
      &libxs_blas_xgemm_alpha_, (const TYPE*)(A), &libxs_blas_xgemm_lda_, \
                                  (const TYPE*)(B), &libxs_blas_xgemm_ldb_, \
       &libxs_blas_xgemm_beta_, (TYPE*)(C), &libxs_blas_xgemm_ldc_); \
  }
#else
# define LIBXS_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXS_UNUSED(LDA); LIBXS_UNUSED(LDB); LIBXS_UNUSED(LDC); \
    LIBXS_UNUSED(M); LIBXS_UNUSED(N); LIBXS_UNUSED(K); \
    LIBXS_UNUSED(A); LIBXS_UNUSED(B); LIBXS_UNUSED(C); \
    LIBXS_UNUSED(ALPHA); LIBXS_UNUSED(BETA); \
    LIBXS_UNUSED(FLAGS)
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(float, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXS_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_BLAS_XGEMM(double, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXS_BLAS_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXS_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
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
  LIBXS_UNUSED(FLAGS); /* TODO: remove/adjust precondition if anything other than NN is supported */ \
  LIBXS_ASSERT(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXS_ASSERT((M) <= (LDA) && (K) <= (LDB) && (M) <= (LDC)); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_j_ = 0; libxs_inline_xgemm_j_ < ((INT)(M)); ++libxs_inline_xgemm_j_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_MAX_K, LIBXS_AVG_K) \
    for (libxs_inline_xgemm_k_ = 0; libxs_inline_xgemm_k_ < (K); ++libxs_inline_xgemm_k_) { \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_inline_xgemm_i_ = 0; libxs_inline_xgemm_i_ < ((INT)(N)); ++libxs_inline_xgemm_i_) { \
        ((TYPE*)(C))[libxs_inline_xgemm_i_*((INT)(LDC))+libxs_inline_xgemm_j_] \
          = ((const TYPE*)(B))[libxs_inline_xgemm_i_*((INT)(LDB))+libxs_inline_xgemm_k_] * \
           (((const TYPE*)(A))[libxs_inline_xgemm_k_*((INT)(LDA))+libxs_inline_xgemm_j_] * libxs_inline_xgemm_alpha_) \
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
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXS_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
    LIBXS_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Fall-back code paths: LIBXS_FALLBACK0, and LIBXS_FALLBACK1 (template). */
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

#if defined(__cplusplus) /** Fall-back from JIT inside of libxs_mmfunction (C++). */ \
  && (defined(LIBXS_FALLBACK_MMFUNCTION) || !defined(NDEBUG)/*debug code*/) \
  /* there are no individual fall-back requests for single or double-precision */ \
  && !defined(LIBXS_FALLBACK_SMMFUNCTION) && !defined(LIBXS_FALLBACK_DMMFUNCTION) \
  /* there is no request to avoid falling back */ \
  && !defined(LIBXS_FALLBACK_MMFUNCTION_NONE)
# define LIBXS_FALLBACK_SMMFUNCTION
# define LIBXS_FALLBACK_DMMFUNCTION
#endif

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXS_MMCALL_ABC(FN, A, B, C) \
  LIBXS_ASSERT(FN); \
  FN(A, B, C)
#define LIBXS_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXS_NOPREFETCH_A(LIBXS_UNUSED(PA)); \
  LIBXS_NOPREFETCH_B(LIBXS_UNUSED(PB)); \
  LIBXS_NOPREFETCH_C(LIBXS_UNUSED(PC)); \
  LIBXS_ASSERT(FN); \
  FN(A, B, C, \
    LIBXS_PREFETCH_A(PA), \
    LIBXS_PREFETCH_B(PB), \
    LIBXS_PREFETCH_C(PC)); \
}

#if (0/*LIBXS_PREFETCH_NONE*/ == LIBXS_PREFETCH)
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_PRF(FN, A, B, C, (A) + (LDA) * (K), (B) + (LDB) * (N), (C) + (LDC) * (N))
#endif
#define LIBXS_MMCALL(FN, A, B, C, M, N, K) LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, M, K, M)

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXS_MNK_SIZE(M, N, K) (((unsigned long long)(M)) * ((unsigned long long)(N)) * ((unsigned long long)(K)))

/**
 * Execute a specialized function, or use a fall-back code path depending on threshold (template).
 * LIBXS_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXS_MAX_MNK)) >= LIBXS_MNK_SIZE(M, N, K)) { \
    const int libxs_xgemm_flags_ = (int)(FLAGS); \
    const INT libxs_xgemm_lda_ = (INT)(LDA), libxs_xgemm_ldb_ = (INT)(LDB), libxs_xgemm_ldc_ = (INT)(LDC); \
    const TYPE libxs_xgemm_alpha_ = (TYPE)(ALPHA), libxs_xgemm_beta_ = (TYPE)(BETA); \
    const LIBXS_MMFUNCTION_TYPE(TYPE) libxs_mmfunction_ = LIBXS_MMDISPATCH_SYMBOL(TYPE)( \
      (INT)(M), (INT)(N), (INT)(K), &libxs_xgemm_lda_, &libxs_xgemm_ldb_, &libxs_xgemm_ldc_, \
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
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
    LIBXS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Call libxs_gemm_print using LIBXS's GEMM-flags. */
#define LIBXS_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, PM, PN, PK, PALPHA, A, PLDA, B, PLDB, PBETA, C, PLDC) { \
  const char libxs_gemm_print_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxs_gemm_print_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  libxs_gemm_print(OSTREAM, PRECISION, &libxs_gemm_print_transa_, &libxs_gemm_print_transb_, \
    PM, PN, PK, PALPHA, A, PLDA, B, PLDB, PBETA, C, PLDC); \
}

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

/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_matdiff_info {
  /** One-norm */         double norm1_abs, norm1_rel;
  /** Infinity-norm */    double normi_abs, normi_rel;
  /** Froebenius-norm */  double normf_rel;
  /** L1-norm and L2-norm of differences. */
  double l2_abs, l2_rel, l1_ref, l1_tst;
  /** Maximum absolute and relative error. */
  double linf_abs, linf_rel;
  /** Location of maximum error (m, n). */
  libxs_blasint linf_abs_m, linf_abs_n;
} libxs_matdiff_info;

/** Utility function to calculate the difference between two matrices. */
LIBXS_API int libxs_matdiff(libxs_datatype datatype, libxs_blasint m, libxs_blasint n,
  const void* ref, const void* tst, const libxs_blasint* ldref, const libxs_blasint* ldtst,
  libxs_matdiff_info* info);

LIBXS_API_INLINE void libxs_matdiff_reduce(libxs_matdiff_info* output, const libxs_matdiff_info* input) {
  LIBXS_ASSERT(0 != output && 0 != input);
  if (output->normf_rel < input->normf_rel) {
    output->linf_abs_m = input->linf_abs_m;
    output->linf_abs_n = input->linf_abs_n;
    output->norm1_abs = input->norm1_abs;
    output->norm1_rel = input->norm1_rel;
    output->normi_abs = input->normi_abs;
    output->normi_rel = input->normi_rel;
    output->normf_rel = input->normf_rel;
    output->linf_abs = input->linf_abs;
    output->linf_rel = input->linf_rel;
    output->l2_abs = input->l2_abs;
    output->l2_rel = input->l2_rel;
    output->l1_ref = input->l1_ref;
    output->l1_tst = input->l1_tst;
  }
}

/* Implementation is taken from an anonymous GitHub Gist. */
LIBXS_API_INLINE unsigned int libxs_cbrt_u64(unsigned long long n) {
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; s >= 0; s -= 3) {
    y += y; b = 3 * y * ((unsigned long long)y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

LIBXS_API_INLINE unsigned int libxs_cbrt_u32(unsigned int n) {
  unsigned int b; unsigned int y = 0; int s;
  for (s = 31; s >= 0; s -= 3) {
    y += y; b = 3 * y * (y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

#endif /*LIBXS_FRONTEND_H*/
