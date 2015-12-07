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

#if !defined(LIBXS_WRAP_XGEMM) && defined(__STATIC) && defined(__GNUC__) && !defined(__CYGWIN__)
# define LIBXS_WRAP_XGEMM
#endif


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
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
  LIBXS_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
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
  LIBXS_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
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
  LIBXS_BLAS_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_blas_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
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
  LIBXS_BLAS_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}


#if defined(LIBXS_WRAP_XGEMM)
LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_sgemm)(
  const char*, const char*,
  const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const float*, const float*, const libxs_blasint*,
  const float* b, const libxs_blasint*,
  const float* beta, float*, const libxs_blasint*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
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
  LIBXS_XGEMM(float, libxs_blasint, LIBXS_FSYMBOL(__real_sgemm), flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_dgemm)(
  const char*, const char*,
  const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const double*, const double*, const libxs_blasint*,
  const double* b, const libxs_blasint*,
  const double* beta, double*, const libxs_blasint*);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
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
  LIBXS_XGEMM(double, libxs_blasint, LIBXS_FSYMBOL(__real_dgemm), flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : m));
}
#endif /*defined(LIBXS_WRAP_XGEMM)*/
