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
/** Fallback prototype functions served by any compliant BLAS. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const double*, const double*, const long long*, const double*, const long long*,
  const double*, double*, const long long*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const float*, const float*, const long long*, const float*, const long long*,
  const float*, float*, const long long*);
#else
/** Fallback prototype functions served by any compliant BLAS. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

/** BLAS based implementation with simplified interface. */
#define LIBXS_BLASMM(REAL, INT, FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  /*const*/char libxs_transa_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  /*const*/char libxs_transb_ = (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  /*const*/INT libxs_m_ = (INT)LIBXS_LD(M, N), libxs_n_ = (INT)LIBXS_LD(N, M), libxs_k_ = (INT)(K); \
  /*const*/INT libxs_lda_ = 0 == (LIBXS_GEMM_FLAG_ALIGN_A & (FLAGS)) ? libxs_m_ : \
    LIBXS_ALIGN_VALUE(libxs_m_, sizeof(REAL), LIBXS_ALIGNMENT); \
  /*const*/INT libxs_ldc_ = 0 == (LIBXS_GEMM_FLAG_ALIGN_C & (FLAGS)) ? libxs_m_ : \
    LIBXS_ALIGN_VALUE(libxs_m_, sizeof(REAL), LIBXS_ALIGNMENT); \
  /*const*/REAL libxs_alpha_ = 0 == (ALPHA) ? ((REAL)LIBXS_ALPHA) : *((const REAL*)(ALPHA)); \
  /*const*/REAL libxs_beta_  = 0 == (BETA)  ? ((REAL)LIBXS_BETA)  : *((const REAL*)(BETA)); \
  LIBXS_FSYMBOL(LIBXS_BLASPREC(REAL, gemm))(&libxs_transa_, &libxs_transb_, \
    &libxs_m_, &libxs_n_, &libxs_k_, &libxs_alpha_, \
    (REAL*)LIBXS_LD(A, B), &libxs_lda_, \
    (REAL*)LIBXS_LD(B, A), &libxs_k_, \
    &libxs_beta_, C, &libxs_ldc_); \
}

/** Inlinable implementation exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXS_IMM(REAL, INT, FLAGS, M, N, K, A, B, C, ALPHA, BETA) LIBXS_BLASMM(REAL, libxs_int, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#else
# define LIBXS_IMM(REAL, INT, FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  const REAL *const libxs_a_ = LIBXS_LD(B, A), *const libxs_b_ = LIBXS_LD(A, B); \
  const REAL libxs_alpha_ = 0 == (ALPHA) ? ((REAL)LIBXS_ALPHA) : (((REAL)1) == *((const REAL*)(ALPHA)) ? ((REAL)1) : (((REAL)-1) == *((const REAL*)(ALPHA)) ? ((REAL)-1) : *((const REAL*)(ALPHA)))); \
  const REAL libxs_beta_  = 0 == (BETA)  ? ((REAL)LIBXS_BETA)  : (((REAL)1) == *((const REAL*)(BETA))  ? ((REAL)1) : (((REAL) 0) == *((const REAL*)(BETA))  ? ((REAL) 0) : *((const REAL*)(BETA)))); \
  const INT libxs_m_ = (INT)LIBXS_LD(M, N), libxs_n_ = (INT)LIBXS_LD(N, M); \
  const INT libxs_lda_ = 0 == (LIBXS_GEMM_FLAG_ALIGN_A & (FLAGS)) ? libxs_m_ : \
    LIBXS_ALIGN_VALUE(libxs_m_, sizeof(REAL), LIBXS_ALIGNMENT); \
  const INT libxs_ldc_ = 0 == (LIBXS_GEMM_FLAG_ALIGN_C & (FLAGS)) ? libxs_m_ : \
    LIBXS_ALIGN_VALUE(libxs_m_, sizeof(REAL), LIBXS_ALIGNMENT); \
  INT libxs_i_, libxs_j_, libxs_k_; \
  REAL *const libxs_c_ = (C); \
  assert(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXS_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
  for (libxs_j_ = 0; libxs_j_ < libxs_m_; ++libxs_j_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_LD(LIBXS_MAX_N, LIBXS_MAX_M), LIBXS_LD(LIBXS_AVG_N, LIBXS_AVG_M)) \
    for (libxs_i_ = 0; libxs_i_ < libxs_n_; ++libxs_i_) { \
      const INT libxs_index_ = libxs_i_ * libxs_ldc_ + libxs_j_; \
      REAL libxs_r_ = libxs_c_[libxs_index_] * libxs_beta_; \
      LIBXS_PRAGMA_SIMD_REDUCTION(+:libxs_r_) \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_k_ = 0; libxs_k_ < (K); ++libxs_k_) { \
        libxs_r_ += libxs_a_[libxs_i_*(K)+libxs_k_] * libxs_alpha_ \
                    * libxs_b_[libxs_k_*libxs_lda_+libxs_j_]; \
      } \
      libxs_c_[libxs_index_] = libxs_r_; \
    } \
  } \
}
#endif

/** Inlinable implementation exercising the compiler's code generation (single-precision). */
#define LIBXS_SIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXS_IMM(float, int/*no need for ILP64*/, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
/** Inlinable implementation exercising the compiler's code generation (double-precision). */
#define LIBXS_DIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXS_IMM(double, int/*no need for ILP64*/, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
/** Inlinable implementation exercising the compiler's code generation. */
#define LIBXS_XIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXS_DIMM(FLAGS, M, N, K, (const double*)(A), (const double*)(B), (double*)(C), (const double*)(ALPHA), (const double*)(BETA)); \
  } \
  else {\
    LIBXS_SIMM(FLAGS, M, N, K, (const float*)(A), (const float*)(B), (float*)(C), (const float*)(ALPHA), (const float*)(BETA)); \
  } \
}

/** Fallback code paths: LIBXS_FALLBACK0, and LIBXS_FALLBACK1. */
#if defined(LIBXS_FALLBACK_IMM)
# define LIBXS_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
    LIBXS_IMM(REAL, int/*no need for ILP64*/, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#else
# define LIBXS_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
    LIBXS_BLASMM(REAL, libxs_blasint, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#endif
#define LIBXS_FALLBACK1(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXS_BLASMM(REAL, libxs_blasint, FLAGS, M, N, K, A, B, C, ALPHA, BETA)

/**
 * Execute a specialized function, or use a fallback code path depending on threshold.
 * LIBXS_FALLBACK0 or specialized function: below LIBXS_MAX_MNK
 * LIBXS_FALLBACK1: above LIBXS_MAX_MNK
 */
#define LIBXS_MM(REAL, FLAGS, M, N, K, A, B, C, PA, PB, PC, ALPHA, BETA) { \
  if (LIBXS_MAX_MNK >= ((M) * (N) * (K))) { \
    int libxs_fallback_ = 0; \
    if (0 != (PA) || 0 != (PB) || 0 != (PC)) { \
      const LIBXS_CONCATENATE(libxs_, LIBXS_BLASPREC(REAL, xfunction)) libxs_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_BLASPREC(REAL, xdispatch))(FLAGS, M, N, K, \
          LIBXS_LD(M, N), K, LIBXS_LD(M, N), ALPHA, BETA, LIBXS_PREFETCH); \
      if (0 != libxs_function_) { \
        const REAL *const libxs_pa_ = ((0 == (LIBXS_PREFETCH & LIBXS_PREFETCH_AL2) && 0 == (LIBXS_PREFETCH & LIBXS_PREFETCH_AL2_JPST)) \
          || 0 == (PA)) ? (A) : (PA); \
        const REAL *const libxs_pb_ = ((0 == (LIBXS_PREFETCH & LIBXS_PREFETCH_BL2_VIA_C)) \
          || 0 == (PB)) ? (B) : (PB); \
        const REAL *const libxs_pc_ = (0 == (PC)) ? (C) : (PC); \
        libxs_function_(A, B, C, libxs_pa_, libxs_pb_, libxs_pc_); \
      } \
      else { \
        libxs_fallback_ = 1; \
      } \
    } \
    else { \
      const LIBXS_CONCATENATE(libxs_, LIBXS_BLASPREC(REAL, function)) libxs_function_ = \
        LIBXS_CONCATENATE(libxs_, LIBXS_BLASPREC(REAL, dispatch))(FLAGS, M, N, K, \
          LIBXS_LD(M, N), K, LIBXS_LD(M, N), ALPHA, BETA); \
      if (0 != libxs_function_) { \
        libxs_function_(A, B, C); \
      } \
      else { \
        libxs_fallback_ = 1; \
      } \
    } \
    if (0 != libxs_fallback_) { \
      LIBXS_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA); \
    } \
  } \
  else { \
    LIBXS_FALLBACK1(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA); \
  } \
}

#endif /*LIBXS_FRONTEND_H*/
