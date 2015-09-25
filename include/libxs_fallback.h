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
#ifndef LIBXS_FALLBACK_H
#define LIBXS_FALLBACK_H

#include "libxs_macros.h"

#if (0 != LIBXS_ROW_MAJOR)
# define LIBXS_LD(M, N) (N)
#else
# define LIBXS_LD(M, N) (M)
#endif
#if (1 < LIBXS_ALIGNED_STORES)
# define LIBXS_ASSUME_ALIGNED_STORES(A) LIBXS_ASSUME_ALIGNED(A, LIBXS_ALIGNED_STORES)
# define LIBXS_ALIGN_STORES(N, TYPESIZE) LIBXS_ALIGN_VALUE(N, TYPESIZE, LIBXS_ALIGNED_STORES)
#else
# define LIBXS_ASSUME_ALIGNED_STORES(A)
# define LIBXS_ALIGN_STORES(N, TYPESIZE) (N)
#endif
#if (1 < LIBXS_ALIGNED_LOADS)
# define LIBXS_ASSUME_ALIGNED_LOADS(A) LIBXS_ASSUME_ALIGNED(A, LIBXS_ALIGNED_LOADS)
# define LIBXS_ALIGN_LOADS(N, TYPESIZE) LIBXS_ALIGN_VALUE(N, TYPESIZE, LIBXS_ALIGNED_LOADS)
#else
# define LIBXS_ASSUME_ALIGNED_LOADS(A)
# define LIBXS_ALIGN_LOADS(N, TYPESIZE) (N)
#endif

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(LIBXS_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#else
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

#define LIBXS_BLASMM(REAL, M, N, K, A, B, C) { \
  int libxs_m_ = LIBXS_LD(M, N), libxs_n_ = LIBXS_LD(N, M), libxs_k_ = (K); \
  int libxs_ldc_ = LIBXS_ALIGN_STORES(LIBXS_LD(M, N), sizeof(REAL)); \
  REAL libxs_alpha_ = 1, libxs_beta_ = 1; \
  char libxs_trans_ = 'N'; \
  LIBXS_FSYMBOL(LIBXS_BLASPREC(, REAL, gemm))(&libxs_trans_, &libxs_trans_, \
    &libxs_m_, &libxs_n_, &libxs_k_, &libxs_alpha_, \
    (REAL*)LIBXS_LD(A, B), &libxs_m_, \
    (REAL*)LIBXS_LD(B, A), &libxs_k_, \
    &libxs_beta_, (C), &libxs_ldc_); \
}

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXS_IMM(REAL, UINT, M, N, K, A, B, C, PA, PB, PC) LIBXS_BLASMM(REAL, M, N, K, A, B, C)
#else
# define LIBXS_IMM(REAL, UINT, M, N, K, A, B, C, PA, PB, PC) { \
    const REAL *const libxs_a_ = LIBXS_LD(B, A), *const libxs_b_ = LIBXS_LD(A, B); \
    const UINT libxs_ldc_ = LIBXS_ALIGN_STORES(LIBXS_LD(M, N), sizeof(REAL)); \
    UINT libxs_i_, libxs_j_, libxs_k_; \
    REAL *const libxs_c_ = (C); \
    LIBXS_UNUSED(PA); LIBXS_UNUSED(PB); LIBXS_UNUSED(PC); /*TODO: prefetching*/ \
    LIBXS_ASSUME_ALIGNED_STORES(libxs_c_); \
    /*TODO: LIBXS_ASSUME_ALIGNED_LOADS(libxs_a_);*/ \
    /*TODO: LIBXS_ASSUME_ALIGNED_LOADS(libxs_b_);*/ \
    LIBXS_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
    for (libxs_j_ = 0; libxs_j_ < LIBXS_LD(M, N); ++libxs_j_) { \
      LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_LD(LIBXS_MAX_N, LIBXS_MAX_M), LIBXS_LD(LIBXS_AVG_N, LIBXS_AVG_M)) \
      for (libxs_i_ = 0; libxs_i_ < LIBXS_LD(N, M); ++libxs_i_) { \
        const UINT libxs_index_ = libxs_i_ * libxs_ldc_ + libxs_j_; \
        REAL libxs_r_ = libxs_c_[libxs_index_]; \
        LIBXS_PRAGMA_SIMD_REDUCTION(+:libxs_r_) \
        LIBXS_PRAGMA_UNROLL \
        for (libxs_k_ = 0; libxs_k_ < (K); ++libxs_k_) { \
          libxs_r_ += libxs_a_[libxs_i_*(K)+libxs_k_] * libxs_b_[libxs_k_*LIBXS_LD(M,N)+libxs_j_]; \
        } \
        libxs_c_[libxs_index_] = libxs_r_; \
      } \
    } \
  }
#endif

/**
 * Execute a generated function, inlined code, or fall back to the linked LAPACK implementation.
 * If M, N, and K does not change for multiple calls, it is more efficient to query and reuse
 * the function pointer (libxs_?mm_dispatch).
 */
#define LIBXS_MM(REAL, M, N, K, A, B, C, PA, PB, PC) \
  if ((LIBXS_MAX_MNK) >= ((M) * (N) * (K))) { \
    const LIBXS_BLASPREC(libxs_, REAL, mm_function) libxs_mm_function_ = \
      LIBXS_BLASPREC(libxs_, REAL, mm_dispatch)(M, N, K); \
    if (libxs_mm_function_) { \
      libxs_mm_function_(A, B, C LIBXS_PREFETCH_ARGA(PA) LIBXS_PREFETCH_ARGB(PB) LIBXS_PREFETCH_ARGC(PC)); \
    } \
    else { \
      LIBXS_IMM(REAL, int, M, N, K, A, B, C, PA, PB, PC); \
    } \
  } \
  else { \
    LIBXS_BLASMM(REAL, M, N, K, A, B, C); \
  }

#endif /*LIBXS_FALLBACK_H*/
