/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxs_gemm_ext.h"


#if defined(LIBXS_GEMM_EXTWRAP)

LIBXS_API LIBXS_ATTRIBUTE(weak) void LIBXS_GEMM_EXTWRAP_SGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  assert(LIBXS_GEMM_EXTWRAP_SGEMM != libxs_internal_sgemm);
  LIBXS_XGEMM(float, libxs_blasint, flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}


LIBXS_API LIBXS_ATTRIBUTE(weak) void LIBXS_GEMM_EXTWRAP_DGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  assert(LIBXS_GEMM_EXTWRAP_DGEMM != libxs_internal_dgemm);
  LIBXS_XGEMM(double, libxs_blasint, flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}

#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
