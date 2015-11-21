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

#define LIBXS_VERSION "$VERSION"

/** Parameters the library and static kernels were built for. */
#define LIBXS_ALIGNMENT $ALIGNMENT
#define LIBXS_ROW_MAJOR $ROW_MAJOR
#define LIBXS_COL_MAJOR $COL_MAJOR
#define LIBXS_PREFETCH $PREFETCH
#define LIBXS_MAX_MNK $MAX_MNK
#define LIBXS_MAX_M $MAX_M
#define LIBXS_MAX_N $MAX_N
#define LIBXS_MAX_K $MAX_K
#define LIBXS_AVG_M $AVG_M
#define LIBXS_AVG_N $AVG_N
#define LIBXS_AVG_K $AVG_K
#define LIBXS_FLAGS $FLAGS
#define LIBXS_ILP64 $ILP64
#define LIBXS_ALPHA $ALPHA
#define LIBXS_BETA $BETA
#define LIBXS_JIT $JIT

#include "libxs_typedefs.h"
#include "libxs_frontend.h"


/** Integer type impacting the BLAS interface (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXS_ILP64)
typedef long long libxs_blasint;
#else
typedef int libxs_blasint;
#endif

/** Specialized function with fused alpha and beta arguments (single-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_sfunction)(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c);
/** Specialized function with fused alpha and beta arguments (double-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_dfunction)(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c);

/** Specialized function with alpha, beta, and prefetch arguments (single-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_sxfunction)(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc);
/** Specialized function with alpha, beta, and prefetch arguments (double-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_dxfunction)(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_init(void);
/** Uninitialize the library and free internal memory (optional). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_finalize(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_sfunction libxs_sdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dfunction libxs_ddispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_sxfunction libxs_sxdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dxfunction libxs_dxdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta, int prefetch);

/** Dispatched matrix multiplication (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_smm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc, const float* alpha, const float* beta)
{
  LIBXS_MM(float, flags, m, n, k, a, b, c, pa, pb, pc, alpha, beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dmm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc, const double* alpha, const double* beta)
{
  LIBXS_MM(double, flags, m, n, k, a, b, c, pa, pb, pc, alpha, beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_sblasmm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* alpha, const float* beta)
{
  LIBXS_BLASMM(float, libxs_blasint, flags, m, n, k, a, b, c, alpha, beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dblasmm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* alpha, const double* beta)
{
  LIBXS_BLASMM(double, libxs_blasint, flags, m, n, k, a, b, c, alpha, beta);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Construct and execute a specialized function. */
template<typename T> class LIBXS_RETARGETABLE libxs_function {};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXS_RETARGETABLE libxs_function<float> {
  typedef LIBXS_RETARGETABLE void (*type)(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c, ...);
  mutable/*retargetable*/ type m_function;
public:
  libxs_function(): m_function(0) {}
  libxs_function(int m, int n, int k, int flags = LIBXS_FLAGS)
    : m_function(reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXS_FLAGS)
    : m_function(reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int flags, int m, int n, int k, int prefetch)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int m, int n, int k, float alpha, float beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxs_function(int flags, int m, int n, int k, float alpha, float beta, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxs_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
  libxs_function(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_sxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_sdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
public:
  operator type() const {
    return m_function;
  }
  void operator()(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) const {
    m_function(a, b, c);
  }
  void operator()(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
    const float* pa, const float* pb, const float* pc) const
  {
    /* TODO: transition prefetch interface to xargs */
    m_function(a, b, c, pa, pb, pc);
  }
  /* TODO: support arbitrary Alpha and Beta in the backend
  void operator()(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
    const float* pa, const float* pb, const float* pc,
    float alpha, float beta) const
  {
    TODO: build xargs here
    m_function(a, b, c, xargs);
  }*/
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXS_RETARGETABLE libxs_function<double> {
  typedef LIBXS_RETARGETABLE void (*type)(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c, ...);
  mutable/*retargetable*/ type m_function;
public:
  libxs_function(): m_function(0) {}
  libxs_function(int m, int n, int k, int flags = LIBXS_FLAGS)
    : m_function(reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXS_FLAGS)
    : m_function(reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int flags, int m, int n, int k, int prefetch)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxs_function(int m, int n, int k, double alpha, double beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxs_function(int flags, int m, int n, int k, double alpha, double beta, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxs_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
  libxs_function(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(LIBXS_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxs_dxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxs_ddispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
public:
  operator type() const {
    return m_function;
  }
  void operator()(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) const {
    m_function(a, b, c);
  }
  void operator()(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
    const double* pa, const double* pb, const double* pc) const
  {
    /* TODO: transition prefetch interface to xargs */
    m_function(a, b, c, pa, pb, pc);
  }
  /* TODO: support arbitrary Alpha and Beta in the backend
  void operator()(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
    const double* pa, const double* pb, const double* pc,
    double alpha, double beta) const
  {
    TODO: build xargs here
    m_function(a, b, c, pa, pb, pc, alpha, beta);
  }*/
};

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  int flags = LIBXS_FLAGS)
{
  libxs_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  int flags = LIBXS_FLAGS)
{
  libxs_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  libxs_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  libxs_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  int flags = LIBXS_FLAGS)
{
  libxs_smm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  int flags = LIBXS_FLAGS)
{
  libxs_dmm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc)
{
  libxs_smm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc)
{
  libxs_dmm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  float alpha, float beta, int flags = LIBXS_FLAGS)
{
  libxs_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  double alpha, double beta, int flags = LIBXS_FLAGS)
{
  libxs_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  float alpha, float beta)
{
  libxs_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  double alpha, double beta)
{
  libxs_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  float alpha, float beta, int flags = LIBXS_FLAGS)
{
  libxs_smm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  double alpha, double beta, int flags = LIBXS_FLAGS)
{
  libxs_dmm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  float alpha, float beta)
{
  libxs_smm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXS_RETARGETABLE inline void libxs_mm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  double alpha, double beta)
{
  libxs_dmm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  int flags = LIBXS_FLAGS)
{
  libxs_sblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  int flags = LIBXS_FLAGS)
{
  libxs_dblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  libxs_sblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  libxs_dblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  float alpha, float beta, int flags = LIBXS_FLAGS)
{
  libxs_sblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  double alpha, double beta, int flags = LIBXS_FLAGS)
{
  libxs_dblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int flags, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c,
  float alpha, float beta)
{
  libxs_sblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXS_RETARGETABLE inline void libxs_blasmm(int flags, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c,
  double alpha, double beta)
{
  libxs_dblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
