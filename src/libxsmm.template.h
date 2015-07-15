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
#ifndef LIBXS_H
#define LIBXS_H

#include "libxs_macros.h"

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(LIBXS_OFFLOAD)
#   pragma offload_attribute(push,target(mic))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#else
LIBXS_EXTERN_C LIBXS_TARGET(mic) void LIBXS_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXS_EXTERN_C LIBXS_TARGET(mic) void LIBXS_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

/** Parameters the library was built for. */
#define LIBXS_ALIGNMENT $ALIGNMENT
#define LIBXS_ALIGNED_STORES $ALIGNED_STORES
#define LIBXS_ALIGNED_LOADS $ALIGNED_LOADS
#define LIBXS_ALIGNED_MAX $ALIGNED_MAX
#define LIBXS_ROW_MAJOR $ROW_MAJOR
#define LIBXS_COL_MAJOR $COL_MAJOR
#define LIBXS_MAX_MNK $MAX_MNK
#define LIBXS_MAX_M $MAX_M
#define LIBXS_MAX_N $MAX_N
#define LIBXS_MAX_K $MAX_K
#define LIBXS_AVG_M $AVG_M
#define LIBXS_AVG_N $AVG_N
#define LIBXS_AVG_K $AVG_K

#if (0 != LIBXS_ROW_MAJOR)
# define LIBXS_LD(M, N) N
#else
# define LIBXS_LD(M, N) M
#endif
#if (1 < LIBXS_ALIGNED_STORES)
# define LIBXS_ASSUME_ALIGNED_STORES(A) LIBXS_ASSUME_ALIGNED(A, LIBXS_ALIGNED_STORES)
# define LIBXS_LDC(REAL, UINT, M, N) LIBXS_ALIGN_VALUE(UINT, REAL, LIBXS_LD(M, N), LIBXS_ALIGNED_STORES)
#else
# define LIBXS_ASSUME_ALIGNED_STORES(A)
# define LIBXS_LDC(REAL, UINT, M, N) LIBXS_LD(M, N)
#endif
#if (1 < LIBXS_ALIGNED_LOADS)
# define LIBXS_ASSUME_ALIGNED_LOADS(A) LIBXS_ASSUME_ALIGNED(A, LIBXS_ALIGNED_LOADS)
#else
# define LIBXS_ASSUME_ALIGNED_LOADS(A)
#endif

#define LIBXS_MAX_SIMD LIBXS_MAX(LIBXS_DIV2(LIBXS_ALIGNED_MAX, sizeof(float)), 1)
#define LIBXS_MAX_SIZE LIBXS_MAX(LIBXS_MAX( \
  LIBXS_LD(LIBXS_MAX_M, LIBXS_MAX_K) * LIBXS_UP2(LIBXS_LD(LIBXS_MAX_K, LIBXS_MAX_M), LIBXS_MAX_SIMD),  \
  LIBXS_LD(LIBXS_MAX_K, LIBXS_MAX_N) * LIBXS_UP2(LIBXS_LD(LIBXS_MAX_N, LIBXS_MAX_K), LIBXS_MAX_SIMD)), \
  LIBXS_LD(LIBXS_MAX_M, LIBXS_MAX_N) * LIBXS_UP2(LIBXS_LD(LIBXS_MAX_N, LIBXS_MAX_M), LIBXS_MAX_SIMD))

#define LIBXS_BLASMM(REAL, M, N, K, A, B, C) { \
  int libxs_m_ = LIBXS_LD(M, N), libxs_n_ = LIBXS_LD(N, M), libxs_k_ = (K); \
  int libxs_ldc_ = LIBXS_LDC(REAL, int, M, N); \
  REAL libxs_alpha_ = 1, libxs_beta_ = 1; \
  char libxs_trans_ = 'N'; \
  LIBXS_FSYMBOL(LIBXS_BLASPREC(, REAL, gemm))(&libxs_trans_, &libxs_trans_, \
    &libxs_m_, &libxs_n_, &libxs_k_, \
    &libxs_alpha_, (REAL*)LIBXS_LD(A, B), &libxs_m_, (REAL*)LIBXS_LD(B, A), &libxs_k_, \
    &libxs_beta_, (C), &libxs_ldc_); \
}

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXS_IMM(REAL, UINT, M, N, K, A, B, C) LIBXS_BLASMM(REAL, M, N, K, A, B, C)
#else
# define LIBXS_IMM(REAL, UINT, M, N, K, A, B, C) { \
    const REAL *const libxs_a_ = LIBXS_LD(B, A), *const libxs_b_ = LIBXS_LD(A, B); \
    const UINT libxs_ldc_ = LIBXS_LDC(REAL, UINT, M, N); \
    UINT libxs_i_, libxs_j_, libxs_k_; \
    REAL *const libxs_c_ = (C); \
    LIBXS_ASSUME_ALIGNED_STORES(libxs_c_); \
    /*TODO: LIBXS_ASSUME_ALIGNED_LOADS(libxs_a_);*/ \
    /*TODO: LIBXS_ASSUME_ALIGNED_LOADS(libxs_b_);*/ \
    LIBXS_PRAGMA_SIMD_COLLAPSE(2) \
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
#define LIBXS_MM(REAL, M, N, K, A, B, C) \
  if ((LIBXS_MAX_MNK) >= ((M) * (N) * (K))) { \
    const LIBXS_BLASPREC(libxs_, REAL, mm_function) libxs_mm_function_ = \
      LIBXS_BLASPREC(libxs_, REAL, mm_dispatch)((M), (N), (K)); \
    if (libxs_mm_function_) { \
      libxs_mm_function_((A), (B), (C)); \
    } \
    else { \
      LIBXS_IMM(REAL, int, M, N, K, A, B, C); \
    } \
  } \
  else { \
    LIBXS_BLASMM(REAL, M, N, K, A, B, C); \
  }

/** Type of a function generated for a specific M, N, and K. */
typedef LIBXS_TARGET(mic) void (*libxs_smm_function)(const float*, const float*, float*);
typedef LIBXS_TARGET(mic) void (*libxs_dmm_function)(const double*, const double*, double*);

/** Query the pointer of a generated function; zero if it does not exist. */
LIBXS_EXTERN_C LIBXS_TARGET(mic) libxs_smm_function libxs_smm_dispatch(int m, int n, int k);
LIBXS_EXTERN_C LIBXS_TARGET(mic) libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k);

/** Dispatched matrix-matrix multiplication; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_smm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_MM(float, m, n, k, a, b, c);
}

/** Dispatched matrix-matrix multiplication; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_MM(double, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_simm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_IMM(float, int, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dimm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_IMM(double, int, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_sblasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(float, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dblasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(double, m, n, k, a, b, c);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Dispatched matrix-matrix multiplication. */
LIBXS_TARGET(mic) inline void libxs_mm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)        { libxs_smm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_mm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)     { libxs_dmm(m, n, k, a, b, c); }

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_TARGET(mic) inline void libxs_imm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)       { libxs_simm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_imm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)    { libxs_dimm(m, n, k, a, b, c); }

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_TARGET(mic) inline void libxs_blasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)    { libxs_sblasmm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_blasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) { libxs_dblasmm(m, n, k, a, b, c); }

/** Call libxs_smm_dispatch, or libxs_dmm_dispatch depending on T. */
template<typename T> class LIBXS_TARGET(mic) libxs_mm_dispatch { typedef void function_type; };

template<> class LIBXS_TARGET(mic) libxs_mm_dispatch<float> {
  typedef libxs_smm_function function_type;
  mutable/*target:mic*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_smm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

template<> class LIBXS_TARGET(mic) libxs_mm_dispatch<double> {
  typedef libxs_dmm_function function_type;
  mutable/*target:mic*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_dmm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
