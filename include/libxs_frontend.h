/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_FRONTEND_H
#define LIBXS_FRONTEND_H

#include "libxs_typedefs.h"

#if !defined(LIBXS_DESCRIPTION)
# define LIBXS_DESCRIPTION "Library for specialized dense and sparse matrix operations, and deep learning primitives."
#endif

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if !defined(_WIN32) && !defined(__CYGWIN__) /* TODO: fully support calling convention */
#if 0 != ((LIBXS_PREFETCH) & 2/*AL2*/) \
 || 0 != ((LIBXS_PREFETCH) & 8/*AL2_AHEAD*/)
# define LIBXS_GEMM_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXS_PREFETCH) & 4/*BL2_VIA_C*/) \
 || 0 != ((LIBXS_PREFETCH) & 16/*BL1*/)
# define LIBXS_GEMM_PREFETCH_B(EXPR) (EXPR)
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
#if (defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL) || \
    (defined(__MKL) && !defined(LIBXS_BUILD) && \
    (!defined(__BLAS) || (0 != __BLAS)))) && \
    (defined(LIBXS_PLATFORM_X86))
# if (0 != LIBXS_ILP64 && !defined(MKL_ILP64))
#   error "Inconsistent ILP64 configuration detected!"
# endif
# include <mkl.h>
#endif
/** __INTEL_MKL__ is needed later to fix some NOTHROW issue. */
#if defined(__MKL) && !defined(__INTEL_MKL__) && defined(LIBXS_PLATFORM_X86) && \
    defined(NOTHROW)
# include <mkl_version.h>
#endif

/** Unfortunately calculation of INTEL_MKL_VERSION is not stable over time. */
#if defined(__INTEL_MKL__) && defined(__INTEL_MKL_MINOR__) && defined(__INTEL_MKL_UPDATE__)
# define LIBXS_MKL_VERSION3 LIBXS_VERSION3(__INTEL_MKL__, __INTEL_MKL_MINOR__, __INTEL_MKL_UPDATE__)
# define LIBXS_MKL_VERSION2 LIBXS_VERSION2(__INTEL_MKL__, __INTEL_MKL_MINOR__)
#endif

/** Automatically select a prefetch-strategy (libxs_get_gemm_xprefetch, etc.). */
#define LIBXS_PREFETCH_AUTO -1

/** Append "_omp" postfix to the given symbol. */
#define LIBXS_USEOMP(FUNCTION) LIBXS_CONCATENATE(FUNCTION, _omp)

/** Helper macro for BLAS-style prefixes. */
#define LIBXS_TPREFIX_NAME(TYPE) LIBXS_CONCATENATE(LIBXS_TPREFIX_, TYPE)
#define LIBXS_TPREFIX(TYPE, FUNCTION) LIBXS_CONCATENATE(LIBXS_TPREFIX_NAME(TYPE), FUNCTION)
#define LIBXS_TPREFIX_doubledouble d
#define LIBXS_TPREFIX_floatfloat s
/** Defaults if only the input type is specified. */
#define LIBXS_TPREFIX_double LIBXS_TPREFIX_doubledouble
#define LIBXS_TPREFIX_float LIBXS_TPREFIX_floatfloat

#define LIBXS_GEMM_XFLAGS(ITYPE, OTYPE) LIBXS_CONCATENATE(LIBXS_GEMM_XFLAGS_, ITYPE) /* ignore OTYPE for now */
#define LIBXS_GEMM_XFLAGS_double 0
#define LIBXS_GEMM_XFLAGS_float 0

/** Construct symbol name from a given real type name (float, double and short). */
#define LIBXS_BLAS_FNTYPE(TYPE, KIND) LIBXS_CONCATENATE3(libxs_, LIBXS_TPREFIX(TYPE, KIND), _function)
#define LIBXS_MMFUNCTION_TYPE(TYPE)   LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmfunction))
#define LIBXS_MMDISPATCH_SYMBOL(TYPE) LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, mmdispatch_v2))
#define LIBXS_BATCH_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm_batch))
#define LIBXS_XGEMM_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_XBLAS_SYMBOL(TYPE)      LIBXS_CONCATENATE(libxs_blas_, LIBXS_TPREFIX(TYPE, gemm))
#define LIBXS_BLAS_SYMBOL(TYPE, KIND) LIBXS_FSYMBOL(LIBXS_TPREFIX(TYPE, KIND))
#define LIBXS_CBLAS_SYMBOL            LIBXS_TPREFIX

#define LIBXS_BLAS_DECL(TYPE, KIND, DECL) LIBXS_CONCATENATE(LIBXS_BLAS_, LIBXS_TPREFIX(TYPE, KIND))(DECL)
#if !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
# define LIBXS_BLAS_dgemm(DECL) DECL;
# define LIBXS_BLAS_sgemm(DECL) DECL;
# define LIBXS_BLAS_dgemv(DECL) DECL;
# define LIBXS_BLAS_sgemv(DECL) DECL;
#else
# define LIBXS_BLAS_dgemm
# define LIBXS_BLAS_sgemm
# define LIBXS_BLAS_dgemv
# define LIBXS_BLAS_sgemv
#endif

/* Construct prefix names, function type or dispatch function from given input and output types. */
#define LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE)    LIBXS_MMFUNCTION_TYPE(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)  LIBXS_MMDISPATCH_SYMBOL(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_TPREFIX_NAME2(ITYPE, OTYPE)       LIBXS_TPREFIX_NAME(LIBXS_CONCATENATE(ITYPE, OTYPE))
#define LIBXS_TPREFIX2(ITYPE, OTYPE, FUNCTION)  LIBXS_TPREFIX(LIBXS_CONCATENATE(ITYPE, OTYPE), FUNCTION)

/** Helper macro for comparing selected types. */
#define LIBXS_EQUAL(T1, T2) LIBXS_CONCATENATE3(LIBXS_EQUAL_, T1, T2)
#define LIBXS_EQUAL_floatfloat 1
#define LIBXS_EQUAL_doubledouble 1
#define LIBXS_EQUAL_floatdouble 0
#define LIBXS_EQUAL_doublefloat 0

#if defined(LIBXS_BLAS_CONST)
# undef LIBXS_BLAS_CONST
# define LIBXS_BLAS_CONST const
#elif defined(OPENBLAS_CONST)
# define LIBXS_BLAS_CONST OPENBLAS_CONST
#elif (defined(LIBXS_BLAS_NONCONST) || defined(__OPENBLAS) || defined(__OPENBLAS77)) \
   && !defined(LIBXS_BUILD)
# define LIBXS_BLAS_CONST
#else
# define LIBXS_BLAS_CONST const
#endif

#if !defined(LIBXS_NO_BLAS)
# if (!defined(__BLAS) || (0 != __BLAS))
#   define LIBXS_NO_BLAS 0
#   define LIBXS_BLAS 1
# else
#   define LIBXS_NO_BLAS 1
#   define LIBXS_BLAS 0
# endif
#endif

#if defined(__BLAS) && (1 == __BLAS)
# if defined(__OPENBLAS)
    LIBXS_EXTERN void openblas_set_num_threads(int num_threads);
#   define LIBXS_BLAS_INIT openblas_set_num_threads(1);
# endif
#endif
#if !defined(LIBXS_BLAS_INIT)
# define LIBXS_BLAS_INIT
#endif

#if defined(LIBXS_BUILD)
# if defined(LIBXS_BUILD_EXT) && defined(_WINDLL) && \
    (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__))
#   define LIBXS_BLAS_SYMBOL_VISIBILITY LIBXS_APIEXT
# elif defined(LIBXS_NO_BLAS) && (1 == LIBXS_NO_BLAS)
#   define LIBXS_BLAS_SYMBOL_VISIBILITY LIBXS_API
# endif
#endif
#if !defined(LIBXS_BLAS_SYMBOL_VISIBILITY)
# define LIBXS_BLAS_SYMBOL_VISIBILITY LIBXS_EXTERN LIBXS_VISIBILITY_IMPORT
#endif

#if defined(NOTHROW)
# define LIBXS_BLAS_NOEXCEPT_AUX NOTHROW
#else
# define LIBXS_BLAS_NOEXCEPT_AUX LIBXS_NOEXCEPT
#endif
#define LIBXS_BLAS_NOEXCEPT(KIND) LIBXS_CONCATENATE(LIBXS_BLAS_NOEXCEPT_, KIND)
#if defined(LIBXS_MKL_VERSION3) && (LIBXS_VERSION3(2020, 0, 2) <= LIBXS_MKL_VERSION3)
# define LIBXS_BLAS_NOEXCEPT_gemm_batch_strided LIBXS_BLAS_NOEXCEPT_AUX
# define LIBXS_BLAS_NOEXCEPT_gemm_batch LIBXS_BLAS_NOEXCEPT_AUX
#else
# define LIBXS_BLAS_NOEXCEPT_gemm_batch_strided
# define LIBXS_BLAS_NOEXCEPT_gemm_batch
#endif
#define LIBXS_BLAS_NOEXCEPT_gemm LIBXS_BLAS_NOEXCEPT_AUX
#define LIBXS_BLAS_NOEXCEPT_gemv LIBXS_BLAS_NOEXCEPT_AUX

#define LIBXS_BLAS_SYMBOL_SIGNATURE_gemm_batch_strided(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxs_blasint CONST_STAR /*m*/, libxs_blasint CONST_STAR /*n*/, libxs_blasint CONST_STAR /*k*/, \
  TYPE CONST_STAR /*alpha*/, TYPE CONST_STAR /*a*/, libxs_blasint CONST_STAR /*lda*/, libxs_blasint CONST_STAR /*stride_a*/, \
                             TYPE CONST_STAR /*b*/, libxs_blasint CONST_STAR /*ldb*/, libxs_blasint CONST_STAR /*stride_b*/, \
  TYPE CONST_STAR /*beta*/,  TYPE       STAR /*c*/, libxs_blasint CONST_STAR /*ldc*/, libxs_blasint CONST_STAR /*stride_c*/, \
  libxs_blasint CONST_STAR /*batchsize*/
#define LIBXS_BLAS_SYMBOL_SIGNATURE_gemm_batch(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE CONST_STAR STAR, libxs_blasint CONST_STAR, TYPE CONST_STAR STAR, libxs_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE STAR STAR, libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, libxs_blasint CONST_STAR
#define LIBXS_BLAS_SYMBOL_SIGNATURE_gemm(CONST_STAR, STAR, TYPE) char CONST_STAR /*transa*/, char CONST_STAR /*transb*/, \
  libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, TYPE CONST_STAR, TYPE CONST_STAR, libxs_blasint CONST_STAR, \
  TYPE CONST_STAR, libxs_blasint CONST_STAR, TYPE CONST_STAR, TYPE STAR, libxs_blasint CONST_STAR
#define LIBXS_BLAS_SYMBOL_SIGNATURE_gemv(CONST_STAR, STAR, TYPE) char CONST_STAR, libxs_blasint CONST_STAR, libxs_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE CONST_STAR, libxs_blasint CONST_STAR, TYPE CONST_STAR, libxs_blasint CONST_STAR, \
  TYPE CONST_STAR, TYPE STAR, libxs_blasint CONST_STAR
#define LIBXS_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND) LIBXS_CONCATENATE(LIBXS_BLAS_SYMBOL_SIGNATURE_, KIND)(CONST_STAR, STAR, TYPE)
#define LIBXS_BLAS_SYMBOL_FDECL(CONST_STAR, STAR, TYPE, KIND) LIBXS_BLAS_SYMBOL_VISIBILITY \
  void LIBXS_BLAS_SYMBOL(TYPE, KIND)(LIBXS_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND)) LIBXS_BLAS_NOEXCEPT(KIND)
#define LIBXS_BLAS_SYMBOL_CDECL(CONST_STAR, STAR, TYPE, KIND) LIBXS_BLAS_SYMBOL_VISIBILITY \
  void LIBXS_CBLAS_SYMBOL(TYPE, KIND)(LIBXS_BLAS_SYMBOL_SIGNATURE(CONST_STAR, STAR, TYPE, KIND)) LIBXS_BLAS_NOEXCEPT(KIND)

#if (0 != LIBXS_BLAS) /* BLAS available */
# define LIBXS_BLAS_SYMBOL_DECL(TYPE, KIND) LIBXS_BLAS_DECL(TYPE, KIND, LIBXS_BLAS_SYMBOL_FDECL(LIBXS_BLAS_CONST*, *, TYPE, KIND))
#else
# define LIBXS_BLAS_SYMBOL_DECL(TYPE, KIND)
#endif

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXS_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_B))

/** Helper macro consolidating CBLAS transpose requests into a set of flags. */
#define LIBXS_GEMM_CFLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((CblasNoTrans == (TRANSA) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_A) \
  | (CblasNoTrans == (TRANSB) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_B))

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXS_GEMM_VNNI_FLAGS(TRANSA, TRANSB, VNNIA, VNNIB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('n' == (TRANSA) || *"N" == (TRANSA)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_A) \
  | (('n' == (TRANSB) || *"N" == (TRANSB)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_TRANS_B) \
  | (('n' == (VNNIA) || *"N" == (VNNIA)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_VNNI_A) \
  | (('n' == (VNNIB) || *"N" == (VNNIB)) ? LIBXS_GEMM_FLAG_NONE : LIBXS_GEMM_FLAG_VNNI_B))

/** Helper macro allowing NULL-requests (transposes) supplied by some default. */
#define LIBXS_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXS_GEMM_FLAGS( \
  NULL != ((const void*)(TRANSA)) ? (*(const char*)(TRANSA)) : (0 == (LIBXS_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'n' : 't'), \
  NULL != ((const void*)(TRANSB)) ? (*(const char*)(TRANSB)) : (0 == (LIBXS_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'n' : 't')) \
  | (~(LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** Inlinable GEMM exercising the compiler's code generation (macro template). TODO: only NN is supported and SP/DP matrices. */
#define LIBXS_INLINE_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
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
  libxs_blasint libxs_inline_xgemm_ni_, libxs_inline_xgemm_mi_ = 0, libxs_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transa_ || *"N" == libxs_inline_xgemm_transa_); \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transb_ || *"N" == libxs_inline_xgemm_transb_); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_mi_ = 0; libxs_inline_xgemm_mi_ < libxs_inline_xgemm_m_; ++libxs_inline_xgemm_mi_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_CONFIG_MAX_DIM, LIBXS_CONFIG_AVG_DIM) \
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
} while(0)

#if (defined(LIBXS_INIT) || defined(LIBXS_CTOR))
# undef LIBXS_INIT
# define LIBXS_INIT LIBXS_ASSERT_MSG(1 < libxs_ninit, "LIBXS is not initialized");
# define LIBXS_INIT_COMPLETED
#else
# define LIBXS_INIT if (2 > libxs_ninit) libxs_init();
#endif

/** Map to appropriate BLAS function (or fallback). The mapping is used, e.g., inside of LIBXS_BLAS_XGEMM. */
#define LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, FUNCTION) LIBXS_CONCATENATE(LIBXS_BLAS_FUNCTION_, LIBXS_TPREFIX2(ITYPE, OTYPE, FUNCTION))
#if (0 != LIBXS_BLAS) /* Helper macro to eventually (if defined) call libxs_init */
# if defined(LIBXS_INIT_COMPLETED)
#   define LIBXS_BLAS_FUNCTION_dgemm_batch_strided libxs_original_dgemm_batch_strided_function
#   define LIBXS_BLAS_FUNCTION_sgemm_batch_strided libxs_original_sgemm_batch_strided_function
#   define LIBXS_BLAS_FUNCTION_dgemm_batch libxs_original_dgemm_batch_function
#   define LIBXS_BLAS_FUNCTION_sgemm_batch libxs_original_sgemm_batch_function
#   define LIBXS_BLAS_FUNCTION_dgemm libxs_original_dgemm_function
#   define LIBXS_BLAS_FUNCTION_sgemm libxs_original_sgemm_function
#   define LIBXS_BLAS_FUNCTION_dgemv libxs_original_dgemv_function
#   define LIBXS_BLAS_FUNCTION_sgemv libxs_original_sgemv_function
# else
#   define LIBXS_BLAS_FUNCTION_dgemm_batch_strided libxs_original_dgemm_batch_strided()
#   define LIBXS_BLAS_FUNCTION_sgemm_batch_strided libxs_original_sgemm_batch_strided()
#   define LIBXS_BLAS_FUNCTION_dgemm_batch libxs_original_dgemm_batch()
#   define LIBXS_BLAS_FUNCTION_sgemm_batch libxs_original_sgemm_batch()
#   define LIBXS_BLAS_FUNCTION_dgemm libxs_original_dgemm()
#   define LIBXS_BLAS_FUNCTION_sgemm libxs_original_sgemm()
#   define LIBXS_BLAS_FUNCTION_dgemv libxs_original_dgemv()
#   define LIBXS_BLAS_FUNCTION_sgemv libxs_original_sgemv()
# endif
#else /* no BLAS */
# define LIBXS_BLAS_FUNCTION_dgemm_batch_strided libxs_blas_error("dgemm_batch_strided")
# define LIBXS_BLAS_FUNCTION_sgemm_batch_strided libxs_blas_error("sgemm_batch_strided")
# define LIBXS_BLAS_FUNCTION_dgemm_batch libxs_blas_error("dgemm_batch")
# define LIBXS_BLAS_FUNCTION_sgemm_batch libxs_blas_error("sgemm_batch")
# define LIBXS_BLAS_FUNCTION_dgemm libxs_blas_error("dgemm")
# define LIBXS_BLAS_FUNCTION_sgemm libxs_blas_error("sgemm")
# define LIBXS_BLAS_FUNCTION_dgemv libxs_blas_error("dgemv")
# define LIBXS_BLAS_FUNCTION_sgemv libxs_blas_error("sgemv")
#endif
/** Low-precision (BLAS-like) function symbols. */
#define LIBXS_BLAS_FUNCTION_wigemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(short, int, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define LIBXS_BLAS_FUNCTION_bsgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM(libxs_bfloat16, float, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Short-cut macros to construct desired BLAS function symbol. */
#define LIBXS_BLAS_FUNCTION1(TYPE, FUNCTION) LIBXS_BLAS_FUNCTION(TYPE, TYPE, FUNCTION)
#define LIBXS_GEMM_BATCH_STRIDED_SYMBOL(TYPE) LIBXS_BLAS_FUNCTION1(TYPE, gemm_batch_strided)
#define LIBXS_GEMM_BATCH_SYMBOL(TYPE) LIBXS_BLAS_FUNCTION1(TYPE, gemm_batch)
#define LIBXS_GEMM_SYMBOL(TYPE) LIBXS_BLAS_FUNCTION1(TYPE, gemm)
#define LIBXS_GEMV_SYMBOL(TYPE) LIBXS_BLAS_FUNCTION1(TYPE, gemv)

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (macro template). */
#define LIBXS_BLAS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxs_blas_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_A & LIBXS_FLAGS) ? 'n' : 't')); \
  const char libxs_blas_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_B & LIBXS_FLAGS) ? 'n' : 't')); \
  const libxs_blasint *const libxs_blas_xgemm_k_ = (NULL != ((void*)(K)) ? (K) : (M)); \
  const libxs_blasint *const libxs_blas_xgemm_n_ = (NULL != ((void*)(N)) ? (N) : libxs_blas_xgemm_k_); \
  const libxs_blasint libxs_blas_xgemm_lda_ = LIBXS_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(('n' == libxs_blas_xgemm_transa_ || *"N" == libxs_blas_xgemm_transa_) ? (M) : libxs_blas_xgemm_k_), 1); \
  const libxs_blasint libxs_blas_xgemm_ldb_ = LIBXS_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(('n' == libxs_blas_xgemm_transb_ || *"N" == libxs_blas_xgemm_transb_) ? libxs_blas_xgemm_k_ : libxs_blas_xgemm_n_), 1); \
  const libxs_blasint libxs_blas_xgemm_ldc_ = LIBXS_MAX(NULL != ((void*)(LDC)) ? *(LDC) : *(M), 1); \
  const OTYPE libxs_blas_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
  const OTYPE libxs_blas_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
  LIBXS_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(&libxs_blas_xgemm_transa_, &libxs_blas_xgemm_transb_, \
    M, libxs_blas_xgemm_n_, libxs_blas_xgemm_k_, \
    &libxs_blas_xgemm_alpha_, (const ITYPE*)(A), &libxs_blas_xgemm_lda_, \
                                (const ITYPE*)(B), &libxs_blas_xgemm_ldb_, \
     &libxs_blas_xgemm_beta_,       (ITYPE*)(C), &libxs_blas_xgemm_ldc_); \
} while(0)

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXS_MMCALL_ABC(FN, A, B, C) \
  LIBXS_ASSERT(FN); FN(A, B, C)
/* TODO: fix prefetch */
#define LIBXS_MMCALL_PRF(FN, A, B, C, PA, PB, PC) do { \
  LIBXS_NOPREFETCH_A(LIBXS_UNUSED(PA)); \
  LIBXS_NOPREFETCH_B(LIBXS_UNUSED(PB)); \
  LIBXS_NOPREFETCH_C(LIBXS_UNUSED(PC)); \
  LIBXS_ASSERT(FN); FN(A, B, C); \
} while(0)

#if (0/*LIBXS_GEMM_PREFETCH_NONE*/ == LIBXS_PREFETCH)
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXS_MMCALL_PRF(FN, A, B, C, (A) + ((size_t)LDA) * (K), (B) + ((size_t)LDB) * (N), (C) + ((size_t)LDC) * (N))
#endif
#define LIBXS_MMCALL(FN, A, B, C, M, N, K) LIBXS_MMCALL_LDX(FN, A, B, C, M, N, K, M, K, M)

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXS_MNK_SIZE(M, N, K) (((size_t)(M)) * ((size_t)(N)) * ((size_t)(K)))
/** Calculate total number of matrix-elements; matrices A, B, C are given per M, N, K, and emphasize (S) the C-size. */
#define LIBXS_SIZE(M, N, K, S) \
    (((size_t)(M) * (size_t)(K)) + ((size_t)(K) * (size_t)(N)) + \
    (((size_t)(S) * (size_t)(M) * (size_t)(N))))
/** Condition based on arithmetic intensity (AI) */
#define LIBXS_SMM_AI(M, N, K, S, TYPESIZE) \
    ((LIBXS_MNK_SIZE(M, N, K) * 2) <= ((size_t)(TYPESIZE) * 4/*AI*/ * LIBXS_SIZE(M, N, K, S)))
/** Determine whether an SMM is suitable, i.e., small enough. */
#if !defined(LIBXS_THRESHOLD_AI) /* traditional MNK-threshold */
# define LIBXS_SMM(M, N, K, S, TYPESIZE) (LIBXS_MNK_SIZE(M, N, K) <= (LIBXS_MAX_MNK))
#else /* threshold based on arithmetic intensity */
# define LIBXS_SMM LIBXS_SMM_AI
#endif

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
 * Execute a specialized function, or use a fallback code path depending on threshold (macro template).
 * LIBXS_XGEMM_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_XGEMM_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  const OTYPE libxs_xgemm_beta_ = (NULL != ((void*)(BETA)) ? (*(const OTYPE*)(BETA)) : ((OTYPE)LIBXS_BETA)); \
  const int libxs_xgemm_flags_ = LIBXS_GEMM_PFLAGS(TRANSA, TRANSB, LIBXS_FLAGS) | \
    LIBXS_GEMM_XFLAGS(ITYPE, OTYPE) | (LIBXS_NEQ(0, libxs_xgemm_beta_) ? 0 : LIBXS_GEMM_FLAG_BETA_0); \
  const libxs_blasint *const libxs_xgemm_k_ = (NULL != (K) ? (K) : (M)); \
  const libxs_blasint *const libxs_xgemm_n_ = (NULL != (N) ? (N) : libxs_xgemm_k_); \
  const libxs_blasint libxs_xgemm_lda_ = LIBXS_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? (M) : libxs_xgemm_k_), 1); \
  const libxs_blasint libxs_xgemm_ldb_ = LIBXS_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? libxs_xgemm_k_ : libxs_xgemm_n_), 1); \
  const libxs_blasint libxs_xgemm_ldc_ = LIBXS_MAX(NULL != (LDC) ? *(LDC) : *(M), 1); \
  if (LIBXS_SMM(*(M), *libxs_xgemm_n_, *libxs_xgemm_k_, 2/*RFO*/, sizeof(OTYPE))) { \
    const LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) libxs_mmfunction_ = LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)( \
      *(M), *libxs_xgemm_n_, *libxs_xgemm_k_, &libxs_xgemm_lda_, &libxs_xgemm_ldb_, &libxs_xgemm_ldc_, \
      &libxs_xgemm_flags_); \
    if (NULL != libxs_mmfunction_) { \
      LIBXS_MMCALL_LDX(libxs_mmfunction_, (const ITYPE*)(A), (const ITYPE*)(B), (OTYPE*)(C), \
        *(M), *libxs_xgemm_n_, *libxs_xgemm_k_, libxs_xgemm_lda_, libxs_xgemm_ldb_, libxs_xgemm_ldc_); \
    } \
    else { \
      const char libxs_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? 'n' : 't'); \
      const char libxs_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? 'n' : 't'); \
      const OTYPE libxs_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
      LIBXS_XGEMM_FALLBACK0(ITYPE, OTYPE, &libxs_xgemm_transa_, &libxs_xgemm_transb_, \
        M, libxs_xgemm_n_, libxs_xgemm_k_, \
        &libxs_xgemm_alpha_, A, &libxs_xgemm_lda_, \
                               B, &libxs_xgemm_ldb_, \
         &libxs_xgemm_beta_, C, &libxs_xgemm_ldc_); \
    } \
  } \
  else { \
    const char libxs_xgemm_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & libxs_xgemm_flags_) ? 'n' : 't'); \
    const char libxs_xgemm_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & libxs_xgemm_flags_) ? 'n' : 't'); \
    const OTYPE libxs_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
    LIBXS_XGEMM_FALLBACK1(ITYPE, OTYPE, &libxs_xgemm_transa_, &libxs_xgemm_transb_, \
      M, libxs_xgemm_n_, libxs_xgemm_k_, \
      &libxs_xgemm_alpha_, A, &libxs_xgemm_lda_, \
                             B, &libxs_xgemm_ldb_, \
       &libxs_xgemm_beta_, C, &libxs_xgemm_ldc_); \
  } \
} while(0)

/** Helper macro to setup a matrix with some initial values. */
#define LIBXS_MATRNG_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) do { \
  /*const*/ libxs_blasint libxs_matrng_seed_ = SEED; /* avoid constant conditional */ \
  const double libxs_matrng_scale_ = (SCALE) * libxs_matrng_seed_ + (SCALE); \
  const libxs_blasint libxs_matrng_nrows_ = (libxs_blasint)NROWS; \
  const libxs_blasint libxs_matrng_ld_ = (libxs_blasint)LD; \
  libxs_blasint libxs_matrng_i_ = 0, libxs_matrng_j_ = 0; \
  LIBXS_OMP_VAR(libxs_matrng_i_); LIBXS_OMP_VAR(libxs_matrng_j_); \
  if (0 != libxs_matrng_seed_) { \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < ((libxs_blasint)NCOLS); ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = (TYPE)(libxs_matrng_scale_ * (1.0 + \
          (double)libxs_matrng_i_ * libxs_matrng_nrows_ + libxs_matrng_j_)); \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = (TYPE)libxs_matrng_seed_; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const unsigned int libxs_matrng_maxval_ = ((unsigned int)NCOLS) * ((unsigned int)libxs_matrng_ld_); \
    const TYPE libxs_matrng_maxval2_ = (TYPE)((unsigned int)LIBXS_UPDIV(libxs_matrng_maxval_, 2)); /* non-zero */ \
    const TYPE libxs_matrng_inv_ = ((TYPE)(SCALE)) / libxs_matrng_maxval2_; \
    const size_t libxs_matrng_shuffle_ = libxs_coprime2(libxs_matrng_maxval_); \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < ((libxs_blasint)NCOLS); ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
    } \
  } \
} while(0)

#define LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_ELIDE, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_SEQ(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_OMP(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_PRAGMA_OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)

/**
 * Print the command line arguments of the current process, and get the number of written
 * characters including the prefix, the postfix, but not the terminating NULL character.
 * If zero is returned, nothing was printed (no prefix, no postfix).
 */
LIBXS_API int libxs_print_cmdline(FILE* stream, const char* prefix, const char* postfix);

/** Call libxs_gemm_print using LIBXS's GEMM-flags. */
#define LIBXS_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  LIBXS_GEMM_PRINT2(OSTREAM, PRECISION, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)
#define LIBXS_GEMM_PRINT2(OSTREAM, IPREC, OPREC, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  libxs_gemm_dprint2(OSTREAM, (libxs_datatype)(IPREC), (libxs_datatype)(OPREC), \
    /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'n' : 't'), \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'n' : 't'), \
    M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using,
 * e.g., ITK-SNAP or ParaView.
 */
LIBXS_API void libxs_gemm_print(void* ostream,
  libxs_datatype precision, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);
LIBXS_API void libxs_gemm_print2(void* ostream,
  libxs_datatype iprec, libxs_datatype oprec, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);
LIBXS_API void libxs_gemm_dprint(void* ostream,
  libxs_datatype precision, char transa, char transb,
  libxs_blasint m, libxs_blasint n, libxs_blasint k,
  double dalpha, const void* a, libxs_blasint lda,
  const void* b, libxs_blasint ldb,
  double dbeta, void* c, libxs_blasint ldc);
LIBXS_API void libxs_gemm_dprint2(void* ostream,
  libxs_datatype iprec, libxs_datatype oprec, char transa, char transb,
  libxs_blasint m, libxs_blasint n, libxs_blasint k,
  double dalpha, const void* a, libxs_blasint lda,
  const void* b, libxs_blasint ldb,
  double dbeta, void* c, libxs_blasint ldc);
LIBXS_API void libxs_gemm_xprint(void* ostream,
  libxs_xmmfunction kernel, const void* a, const void* b, void* c);

/** GEMM_BATCH_STRIDED: fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef void (*libxs_dgemm_batch_strided_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch_strided));
LIBXS_EXTERN_C typedef void (*libxs_sgemm_batch_strided_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch_strided));
/** GEMM_BATCH: fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef void (*libxs_dgemm_batch_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch));
LIBXS_EXTERN_C typedef void (*libxs_sgemm_batch_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch));
/** GEMM: fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef void (*libxs_dgemm_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm));
LIBXS_EXTERN_C typedef void (*libxs_sgemm_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, float,  gemm));
/** GEMV: fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXS_EXTERN_C typedef void (*libxs_dgemv_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemv));
LIBXS_EXTERN_C typedef void (*libxs_sgemv_function)(LIBXS_BLAS_SYMBOL_SIGNATURE(const*, *, float,  gemv));
/** Helper function to consume arguments when called. */
LIBXS_EXTERN_C typedef void (*libxs_sink_function)(const void*, ...);

/** The original BLAS functions. */
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_dgemm_batch_strided_function libxs_original_dgemm_batch_strided_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_sgemm_batch_strided_function libxs_original_sgemm_batch_strided_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_dgemm_batch_function libxs_original_dgemm_batch_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_sgemm_batch_function libxs_original_sgemm_batch_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_dgemm_function libxs_original_dgemm_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_sgemm_function libxs_original_sgemm_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_dgemv_function libxs_original_dgemv_function);
LIBXS_APIVAR_PUBLIC(/*volatile*/libxs_sgemv_function libxs_original_sgemv_function);
LIBXS_API libxs_dgemm_batch_strided_function libxs_original_dgemm_batch_strided(void);
LIBXS_API libxs_sgemm_batch_strided_function libxs_original_sgemm_batch_strided(void);
LIBXS_API libxs_dgemm_batch_function libxs_original_dgemm_batch(void);
LIBXS_API libxs_sgemm_batch_function libxs_original_sgemm_batch(void);
LIBXS_API libxs_dgemm_function libxs_original_dgemm(void);
LIBXS_API libxs_sgemm_function libxs_original_sgemm(void);
LIBXS_API libxs_dgemv_function libxs_original_dgemv(void);
LIBXS_API libxs_sgemv_function libxs_original_sgemv(void);
LIBXS_API libxs_sink_function libxs_blas_error(const char* symbol);
LIBXS_API void libxs_sink(const void* arg, ...);

#define libxs_blas_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_blas_gemm(LIBXS_DATATYPE_F64, LIBXS_DATATYPE_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxs_blas_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxs_blas_gemm(LIBXS_DATATYPE_F32, LIBXS_DATATYPE_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Translates GEMM prefetch request into prefetch-enumeration (incl. FE's auto-prefetch). */
LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_xprefetch(const int* prefetch);
LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_prefetch(int prefetch);

/** Determines the given value in double-precision (EXIT_SUCCESS if value is NULL). */
LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

#endif /*LIBXS_FRONTEND_H*/
