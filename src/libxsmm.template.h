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
#define LIBXS_BETA $BETA
#define LIBXS_JIT $JIT

#include "libxs_typedefs.h"
#include "libxs_prefetch.h"
#include "libxs_fallback.h"


/** Explicitly initializes the library; can be used to pay for setup cost at a specific point. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_init(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not permitted (single-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smm_function libxs_smm_dispatch(float alpha, float beta, int m, int n, int k,
  int lda, int ldb, int ldc, int flags, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not permitted (double-precision). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmm_function libxs_dmm_dispatch(double alpha, double beta, int m, int n, int k,
  int lda, int ldb, int ldc, int flags, int prefetch);

/** Dispatched matrix-matrix multiplication (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_smm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float*, pa)
  LIBXS_PREFETCH_DECL(const float*, pb)
  LIBXS_PREFETCH_DECL(const float*, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_MM(float, alpha, beta, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Dispatched matrix-matrix multiplication (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double*, pa)
  LIBXS_PREFETCH_DECL(const double*, pb)
  LIBXS_PREFETCH_DECL(const double*, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_MM(double, alpha, beta, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_simm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  LIBXS_IMM(float, int, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dimm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  LIBXS_IMM(double, int, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (single-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_sblasmm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  LIBXS_BLASMM(float, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (double-precision). */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dblasmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  LIBXS_BLASMM(double, alpha, beta, m, n, k, a, b, c);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Function type depending on T. */
template<typename T> struct LIBXS_RETARGETABLE libxs_function { typedef void type; };
template<> struct LIBXS_RETARGETABLE libxs_function<float>    { typedef libxs_smm_function type; };
template<> struct LIBXS_RETARGETABLE libxs_function<double>   { typedef libxs_dmm_function type; };

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not permitted. */
LIBXS_RETARGETABLE inline libxs_smm_function libxs_mm_dispatch(float alpha, float beta, int m, int n, int k,
  int lda = 0, int ldb = 0, int ldc = 0, int flags = LIBXS_GEMM_FLAG_DEFAULT, int prefetch = LIBXS_PREFETCH)
{
  return libxs_smm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch);
}

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not permitted. */
LIBXS_RETARGETABLE inline libxs_dmm_function libxs_mm_dispatch(double alpha, double beta, int m, int n, int k,
  int lda = 0, int ldb = 0, int ldc = 0, int flags = LIBXS_GEMM_FLAG_DEFAULT, int prefetch = LIBXS_PREFETCH)
{
  return libxs_dmm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch);
}

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float*, pa)
  LIBXS_PREFETCH_DECL(const float*, pb)
  LIBXS_PREFETCH_DECL(const float*, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_smm(alpha, beta, m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double*, pa)
  LIBXS_PREFETCH_DECL(const double*, pb)
  LIBXS_PREFETCH_DECL(const double*, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_dmm(alpha, beta, m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_RETARGETABLE inline void libxs_imm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  libxs_simm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_RETARGETABLE inline void libxs_imm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  libxs_dimm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(float alpha, float beta, int m, int n, int k,
  const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  libxs_sblasmm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  libxs_dblasmm(alpha, beta, m, n, k, a, b, c);
}

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
