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
#include "libxs_gemm_extomp.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(LIBXS_GEMM_EXTWRAP) && !defined(__STATIC)
# include <stdlib.h>
# include <dlfcn.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


#if defined(LIBXS_GEMM_EXTWRAP) && !defined(__STATIC)

/* implementation variant for non-static linkage; overrides weak libxs_gemm_init in libxs_gemm.c */
LIBXS_API_DEFINITION int libxs_gemm_init(int archid, int prefetch)
{
  union { const void* pv; libxs_sgemm_function pf; } fn_sgemm = { NULL };
  union { const void* pv; libxs_dgemm_function pf; } fn_dgemm = { NULL };
#if !defined(__BLAS) || (0 != __BLAS)
  fn_sgemm.pv = dlsym(RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(sgemm)));
  fn_dgemm.pv = dlsym(RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dgemm)));
#endif

  /* internal pre-initialization step */
  libxs_gemm_configure(archid, prefetch, fn_sgemm.pf, fn_dgemm.pf);

  { /* behaviour of libxs_omp_?gemm routines or LD_PRELOAD ?GEMM routines
     * 0: sequential below-threshold routine (no OpenMP); may fall-back to BLAS,
     * 1: OpenMP-parallelized but without internal parallel region,
     * 2: OpenMP-parallelized with internal parallel region" )
     */
    const char *const env = getenv("LIBXS_GEMM");
    if (0 != env && 0 != *env) {
      internal_gemm = atoi(env);
    }
  }

#if defined(LIBXS_GEMM_EXTOMP_TASKS)
  { /* consider user input about using (OpenMP-)tasks; this code must be here
    * because maybe only this translation unit is compiled with OpenMP support
    */
    const char *const env_tasks = getenv("LIBXS_TASKS");
    if (0 != env_tasks && 0 != *env_tasks) {
      internal_gemm_tasks = atoi(env_tasks);
    }
  }
#endif
#if !defined(__BLAS) || (0 != __BLAS)
  return (NULL != *libxs_original_sgemm()
       && NULL != *libxs_original_dgemm())
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
#else
  return EXIT_SUCCESS;
#endif
}


LIBXS_API_DEFINITION void libxs_gemm_finalize(void)
{
}

#endif /*defined(LIBXS_GEMM_EXTWRAP) && !defined(__STATIC)*/

LIBXS_API_DEFINITION void libxs_omp_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  const int tm = internal_gemm_tile[1/*SP*/][0/*M*/];
  const int tn = internal_gemm_tile[1/*SP*/][1/*N*/];
  const int tk = internal_gemm_tile[1/*SP*/][2/*K*/];
  const int nt = internal_gemm_nt;
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if !defined(_OPENMP)
  LIBXS_UNUSED(nt);
#endif
  if (0 == LIBXS_DIV2(internal_gemm, 2)) { /* enable internal parallelization */
    if (0 == internal_gemm_tasks) {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_FOR_INIT, LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL,
        LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXS_GEMM_EXTOMP_FOR_LOOP_END,
        float, flags | LIBXS_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
    else {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_TSK_INIT, LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL,
        LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXS_GEMM_EXTOMP_TSK_LOOP_END,
        float, flags | LIBXS_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
  else { /* potentially sequential or externally parallelized */
    if (0 == internal_gemm_tasks) {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_FOR_INIT, LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN,
        LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXS_GEMM_EXTOMP_FOR_LOOP_END,
        float, flags | LIBXS_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
    else {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_TSK_INIT, LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN,
        LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXS_GEMM_EXTOMP_TSK_LOOP_END,
        float, flags | LIBXS_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
}


LIBXS_API_DEFINITION void libxs_omp_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  const int tm = internal_gemm_tile[0/*DP*/][0/*M*/];
  const int tn = internal_gemm_tile[0/*DP*/][1/*N*/];
  const int tk = internal_gemm_tile[0/*DP*/][2/*K*/];
  const int nt = internal_gemm_nt;
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if !defined(_OPENMP)
  LIBXS_UNUSED(nt);
#endif
  if (0 == LIBXS_DIV2(internal_gemm, 2)) { /* enable internal parallelization */
    if (0 == internal_gemm_tasks) {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_FOR_INIT, LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL,
        LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXS_GEMM_EXTOMP_FOR_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
    else {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_TSK_INIT, LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL,
        LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXS_GEMM_EXTOMP_TSK_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
  else { /* potentially sequential or externally parallelized */
    if (0 == internal_gemm_tasks) {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_FOR_INIT, LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN,
        LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXS_GEMM_EXTOMP_FOR_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
    else {
      LIBXS_GEMM_EXTOMP_XGEMM(LIBXS_GEMM_EXTOMP_TSK_INIT, LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN,
        LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXS_GEMM_EXTOMP_TSK_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
}


#if defined(LIBXS_GEMM_EXTWRAP)

LIBXS_EXTERN LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_SGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  assert(LIBXS_GEMM_EXTWRAP_SGEMM != *libxs_original_sgemm());
  switch (internal_gemm) {
    case 0: { /* below-THRESHOLD xGEMM */
      libxs_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: { /* tiled xGEMM */
      libxs_omp_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
}


LIBXS_EXTERN LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_DGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  assert(LIBXS_GEMM_EXTWRAP_DGEMM != *libxs_original_dgemm());
  switch (internal_gemm) {
    case 0: { /* below-THRESHOLD xGEMM */
      libxs_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: { /* tiled xGEMM */
      libxs_omp_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
}

#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
