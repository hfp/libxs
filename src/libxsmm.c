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
#include <libxs.h>


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float *LIBXS_RESTRICT a, const libxs_blasint* lda,
  const float *LIBXS_RESTRICT b, const libxs_blasint* ldb,
  const float* beta, float *LIBXS_RESTRICT c, const libxs_blasint* ldc)
{
  libxs_blasint ilda, ildb;
  int flags = LIBXS_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXS_GEMM(float, libxs_blasint, flags, *LIBXS_LD(m, n), *LIBXS_LD(n, m), *k,
    0 != alpha ? (1.f == *alpha ? 1.f : (-1.f == *alpha ? -1.f : *alpha)) : ((float)LIBXS_ALPHA),
    LIBXS_LD(a, b), LIBXS_LD(ilda, ildb), LIBXS_LD(b, a), LIBXS_LD(ildb, ilda),
    0 != beta  ? (1.f == *beta  ? 1.f : ( 0.f == *beta  ?  0.f : *beta))  : ((float)LIBXS_BETA),
    c, ldc ? *ldc : *LIBXS_LD(m, n));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double *LIBXS_RESTRICT a, const libxs_blasint* lda,
  const double *LIBXS_RESTRICT b, const libxs_blasint* ldb,
  const double* beta, double *LIBXS_RESTRICT c, const libxs_blasint* ldc)
{
  libxs_blasint ilda, ildb;
  int flags = LIBXS_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXS_GEMM(double, libxs_blasint, flags, *LIBXS_LD(m, n), *LIBXS_LD(n, m), *k,
    0 != alpha ? (1.0 == *alpha ? 1.0 : (-1.0 == *alpha ? -1.0 : *alpha)) : ((double)LIBXS_ALPHA),
    LIBXS_LD(a, b), LIBXS_LD(ilda, ildb), LIBXS_LD(b, a), LIBXS_LD(ildb, ilda),
    0 != beta  ? (1.0 == *beta  ? 1.0 : ( 0.0 == *beta  ?  0.0 : *beta))  : ((double)LIBXS_BETA),
    c, ldc ? *ldc : *LIBXS_LD(m, n));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float *LIBXS_RESTRICT a, const libxs_blasint* lda,
  const float *LIBXS_RESTRICT b, const libxs_blasint* ldb,
  const float* beta, float *LIBXS_RESTRICT c, const libxs_blasint* ldc)
{
  libxs_blasint ilda, ildb;
  int flags = LIBXS_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXS_BSGEMM(flags, *LIBXS_LD(m, n), *LIBXS_LD(n, m), *k,
    0 != alpha ? (1.f == *alpha ? 1.f : (-1.f == *alpha ? -1.f : *alpha)) : ((float)LIBXS_ALPHA),
    LIBXS_LD(a, b), LIBXS_LD(ilda, ildb), LIBXS_LD(b, a), LIBXS_LD(ildb, ilda),
    0 != beta  ? (1.f == *beta  ? 1.f : ( 0.f == *beta  ?  0.f : *beta))  : ((float)LIBXS_BETA),
    c, ldc ? *ldc : *LIBXS_LD(m, n));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double *LIBXS_RESTRICT a, const libxs_blasint* lda,
  const double *LIBXS_RESTRICT b, const libxs_blasint* ldb,
  const double* beta, double *LIBXS_RESTRICT c, const libxs_blasint* ldc)
{
  libxs_blasint ilda, ildb;
  int flags = LIBXS_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXS_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXS_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXS_BDGEMM(flags, *LIBXS_LD(m, n), *LIBXS_LD(n, m), *k,
    0 != alpha ? (1.0 == *alpha ? 1.0 : (-1.0 == *alpha ? -1.0 : *alpha)) : ((double)LIBXS_ALPHA),
    LIBXS_LD(a, b), LIBXS_LD(ilda, ildb), LIBXS_LD(b, a), LIBXS_LD(ildb, ilda),
    0 != beta  ? (1.0 == *beta  ? 1.0 : ( 0.0 == *beta  ?  0.0 : *beta))  : ((double)LIBXS_BETA),
    c, ldc ? *ldc : *LIBXS_LD(m, n));
}
