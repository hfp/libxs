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

/** Parameters the library was built for. */
#define LIBXS_ALIGNMENT $ALIGNMENT
#define LIBXS_ALIGNED_STORES $ALIGNED_STORES
#define LIBXS_ALIGNED_LOADS $ALIGNED_LOADS
#define LIBXS_ALIGNED_MAX $ALIGNED_MAX
#define LIBXS_PREFETCH $PREFETCH
#define LIBXS_ROW_MAJOR $ROW_MAJOR
#define LIBXS_COL_MAJOR $COL_MAJOR
#define LIBXS_MAX_MNK $MAX_MNK
#define LIBXS_MAX_M $MAX_M
#define LIBXS_MAX_N $MAX_N
#define LIBXS_MAX_K $MAX_K
#define LIBXS_AVG_M $AVG_M
#define LIBXS_AVG_N $AVG_N
#define LIBXS_AVG_K $AVG_K
#define LIBXS_ALPHA $ALPHA
#define LIBXS_BETA $BETA
#define LIBXS_JIT $JIT

#include "libxs_typedefs.h"
#include "libxs_prefetch.h"
#include "libxs_fallback.h"


/** Structure providing the actual/extended arguments of an SGEMM call. */
typedef struct libxs_sgemm_xargs {
  /** The Alpha and Beta arguments. */
  float alpha, beta;
  /** The prefetch arguments. */
  LIBXS_PREFETCH_DECL(const float* pa)
  LIBXS_PREFETCH_DECL(const float* pb)
  LIBXS_PREFETCH_DECL(const float* pc)
} libxs_sgemm_xargs;

/** Structure providing the actual/extended arguments of a DGEMM call. */
typedef struct libxs_dgemm_xargs {
  /** The Alpha and Beta arguments. */
  double alpha, beta;
  /** The prefetch arguments. */
  LIBXS_PREFETCH_DECL(const double* pa)
  LIBXS_PREFETCH_DECL(const double* pb)
  LIBXS_PREFETCH_DECL(const double* pc)
} libxs_dgemm_xargs;

/** Generic type of a function. */
typedef LIBXS_RETARGETABLE void (*libxs_sfunction)(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c, const libxs_sgemm_xargs* xargs);
typedef LIBXS_RETARGETABLE void (*libxs_dfunction)(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c, const libxs_dgemm_xargs* xargs);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_init(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_sfunction libxs_sdispatch(int m, int n, int k, float alpha, float beta,
  int lda, int ldb, int ldc, int flags, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dfunction libxs_ddispatch(int m, int n, int k, double alpha, double beta,
  int lda, int ldb, int ldc, int flags, int prefetch);

/** Dispatched matrix-matrix multiplication (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_smm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const libxs_sgemm_xargs* xargs)
{
  LIBXS_MM(float, m, n, k, a, b, c, xargs);
}

/** Dispatched matrix-matrix multiplication (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dmm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const libxs_dgemm_xargs* xargs)
{
  LIBXS_MM(double, m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_sblasmm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const libxs_sgemm_xargs* xargs)
{
  LIBXS_BLASMM(float, m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dblasmm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const libxs_dgemm_xargs* xargs)
{
  LIBXS_BLASMM(double, m, n, k, a, b, c, xargs);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Function type depending on T. */
template<typename T> struct LIBXS_RETARGETABLE libxs_function { typedef void type; };
template<> struct LIBXS_RETARGETABLE libxs_function<float>    { typedef libxs_sfunction type; };
template<> struct LIBXS_RETARGETABLE libxs_function<double>   { typedef libxs_dfunction type; };

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported. */
template<typename T> class LIBXS_RETARGETABLE libxs_dispatch {};
template<> class LIBXS_RETARGETABLE libxs_dispatch<float> {
  mutable/*retargetable*/ libxs_sfunction m_function;
public:
  libxs_dispatch(): m_function(0) {}
  libxs_dispatch(int m, int n, int k,
    float alpha = LIBXS_ALPHA, float beta = LIBXS_BETA,
    int lda = 0, int ldb = 0, int ldc = 0,
    int flags = LIBXS_GEMM_FLAG_DEFAULT,
    int prefetch = LIBXS_PREFETCH)  : m_function(libxs_sdispatch(m, n, k, alpha, beta, lda, ldb, ldc, flags, prefetch))
  {}
  operator libxs_sfunction() const {
    return m_function;
  }
  void operator()(const float a[], const float b[], float c[], const libxs_sgemm_xargs* xargs = 0) const {
    m_function(a, b, c, xargs);
  }
  void operator()(const float a[], const float b[], float c[], const libxs_sgemm_xargs& xargs) const {
    m_function(a, b, c, &xargs);
  }
};
template<> class LIBXS_RETARGETABLE libxs_dispatch<double> {
  mutable/*retargetable*/ libxs_dfunction m_function;
public:
  libxs_dispatch(): m_function(0) {}
  libxs_dispatch(int m, int n, int k,
    double alpha = LIBXS_ALPHA, double beta = LIBXS_BETA,
    int lda = 0, int ldb = 0, int ldc = 0,
    int flags = LIBXS_GEMM_FLAG_DEFAULT,
    int prefetch = LIBXS_PREFETCH)  : m_function(libxs_ddispatch(m, n, k, alpha, beta, lda, ldb, ldc, flags, prefetch))
  {}
  operator libxs_dfunction() const {
    return m_function;
  }
  void operator()(const double a[], const double b[], double c[], const libxs_dgemm_xargs* xargs = 0) const {
    m_function(a, b, c, xargs);
  }
  void operator()(const double a[], const double b[], double c[], const libxs_dgemm_xargs& xargs) const {
    m_function(a, b, c, &xargs);
  }
};

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const libxs_sgemm_xargs* xargs = 0)
{
  libxs_smm(m, n, k, a, b, c, xargs);
}

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const libxs_dgemm_xargs* xargs = 0)
{
  libxs_dmm(m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const libxs_sgemm_xargs* xargs = 0)
{
  libxs_sblasmm(m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const libxs_dgemm_xargs* xargs = 0)
{
  libxs_dblasmm(m, n, k, a, b, c, xargs);
}

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
