/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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

/** Name of the version (stringized set of version numbers). */
#define LIBXS_VERSION "$VERSION"
/** Name of the branch of which the version is derived from. */
#define LIBXS_BRANCH  "$BRANCH"
/** Major version based on the last reachable tag under RCS. */
#define LIBXS_VERSION_MAJOR $MAJOR
/** Minor version based on the last reachable tag of the RCS. */
#define LIBXS_VERSION_MINOR $MINOR
/** Update number based on the last reachable tag under RCS. */
#define LIBXS_VERSION_UPDATE $UPDATE
/** Patch number counting commits since the last version stamp. */
#define LIBXS_VERSION_PATCH $PATCH

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
#define LIBXS_SYNC $SYNC
#define LIBXS_JIT $JIT
$LIBXS_OFFLOAD_BUILD
#include "libxs_frontend.h"


/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXS_ILP64)
typedef long long libxs_blasint;
#else
typedef int libxs_blasint;
#endif

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_smmfunction)(const float* a, const float* b, float* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
typedef LIBXS_RETARGETABLE void (*libxs_dmmfunction)(const double* a, const double* b, double* c, ...);

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (weak-typed). */
typedef union LIBXS_RETARGETABLE libxs_xmmfunction {
  libxs_smmfunction smm;
  libxs_dmmfunction dmm;
} libxs_xmmfunction;

/** Initialize the library; pay for setup cost at a specific point. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_init(void);
/** Uninitialize the library and free internal memory (optional). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_finalize(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (descriptor form). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smmfunction libxs_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch);

/** Dispatched general dense matrix multiplication (single-precision); can be called from F77 code. */
LIBXS_INLINE_EXPORT LIBXS_RETARGETABLE void libxs_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
#if defined(LIBXS_BUILD)
;
#else
{ LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXS_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}
#endif

/** Dispatched general dense matrix multiplication (double-precision); can be called from F77 code. */
LIBXS_INLINE_EXPORT LIBXS_RETARGETABLE void libxs_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
#if defined(LIBXS_BUILD)
;
#else
{ LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXS_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}
#endif

/** Dispatched general dense matrix multiplication (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_omps_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc);

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_omps_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc);
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Construct and execute a specialized function. */
template<typename T> class LIBXS_RETARGETABLE libxs_mmfunction {};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXS_RETARGETABLE libxs_mmfunction<float> {
  mutable/*retargetable*/ libxs_smmfunction m_function;
public:
  libxs_mmfunction(): m_function(0) {}
  libxs_mmfunction(int m, int n, int k, int flags = LIBXS_FLAGS)
    : m_function(libxs_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxs_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXS_FLAGS)
    : m_function(libxs_smmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxs_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_smmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxs_mmfunction(int m, int n, int k, float alpha, float beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, float alpha, float beta, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxs_smmfunction() const {
    return m_function;
  }
  void operator()(const float* a, const float* b, float* c) const {
    m_function(LIBXS_LD(a, b), LIBXS_LD(b, a), c);
  }
  void operator()(const float* a, const float* b, float* c,
    const float* pa, const float* pb, const float* pc) const
  {
    m_function(LIBXS_LD(a, b), LIBXS_LD(b, a), c,
      LIBXS_LD(pa, pb), LIBXS_LD(pb, pa), pc);
  }
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXS_RETARGETABLE libxs_mmfunction<double> {
  mutable/*retargetable*/ libxs_dmmfunction m_function;
public:
  libxs_mmfunction(): m_function(0) {}
  libxs_mmfunction(int m, int n, int k, int flags = LIBXS_FLAGS)
    : m_function(libxs_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxs_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXS_FLAGS)
    : m_function(libxs_dmmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxs_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_dmmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxs_mmfunction(int m, int n, int k, double alpha, double beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, double alpha, double beta, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxs_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = LIBXS_FLAGS, int prefetch = LIBXS_PREFETCH)
    : m_function(libxs_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxs_dmmfunction() const {
    return m_function;
  }
  void operator()(const double* a, const double* b, double* c) const {
    LIBXS_MMCALL_ABC(m_function, a, b, c);
  }
  void operator()(const double* a, const double* b, double* c,
    const double* pa, const double* pb, const double* pc) const
  {
    LIBXS_MMCALL_PRF(m_function, a, b, c, pa, pb, pc);
  }
};

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXS_RETARGETABLE void libxs_sgemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXS_RETARGETABLE void libxs_dgemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_sgemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_blas_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_dgemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_blas_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda, const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  libxs_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda, const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  libxs_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
