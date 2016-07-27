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

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_GEMM_EXTWRAP) && !defined(__STATIC)
# include <stdlib.h>
# include <dlfcn.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXS_GEMM_EXTOMP_SUFFICIENT_SIZE(M, N, K) (((LIBXS_MAX_M < (M)) || (LIBXS_MAX_N < (N)) || (LIBXS_MAX_K < (K))) ? 1 : 0)
#if defined(_OPENMP)
# if !defined(LIBXS_GEMM_EXTOMP_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXS_GEMM_EXTOMP_TASKS
# endif
# define LIBXS_GEMM_EXTOMP_MIN_NTASKS(NT) LIBXS_MAX(7/*arbitrary factor*/ * omp_get_num_threads() / (NT), 1)
# define LIBXS_GEMM_EXTOMP_OVERHEAD(NT) (/*arbitrary factor*/NT)
# define LIBXS_GEMM_EXTOMP_FOR_INIT
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN LIBXS_PRAGMA(omp for schedule(dynamic) LIBXS_OPENMP_COLLAPSE(LIBXS_GEMM_EXTOMP_COLLAPSE))
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL LIBXS_PRAGMA(omp parallel for schedule(dynamic) LIBXS_OPENMP_COLLAPSE(LIBXS_GEMM_EXTOMP_COLLAPSE))
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY(...)
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_END
#else
# define LIBXS_GEMM_EXTOMP_MIN_NTASKS(NT) 1
# define LIBXS_GEMM_EXTOMP_OVERHEAD(NT) 0
# define LIBXS_GEMM_EXTOMP_FOR_INIT
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY(...)
# define LIBXS_GEMM_EXTOMP_FOR_LOOP_END
#endif

#if defined(LIBXS_GEMM_EXTOMP_TASKS)
# define LIBXS_GEMM_EXTOMP_COLLAPSE 2
# define LIBXS_GEMM_EXTOMP_TSK_INIT LIBXS_PRAGMA(omp single nowait)
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL LIBXS_PRAGMA(omp parallel) LIBXS_PRAGMA(omp single nowait)
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY(...) LIBXS_PRAGMA(omp task firstprivate(__VA_ARGS__))
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_END LIBXS_PRAGMA(omp taskwait)
#else
# define LIBXS_GEMM_EXTOMP_COLLAPSE 2
# define LIBXS_GEMM_EXTOMP_TSK_INIT LIBXS_GEMM_EXTOMP_FOR_INIT
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL LIBXS_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_BODY LIBXS_GEMM_EXTOMP_FOR_LOOP_BODY
# define LIBXS_GEMM_EXTOMP_TSK_LOOP_END LIBXS_GEMM_EXTOMP_FOR_LOOP_END
#endif

#define LIBXS_GEMM_EXTOMP_KERNEL(REAL, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxs_blasint mm = LIBXS_MIN(TILE_M, (M) - (POS_H)), nn = LIBXS_MIN(TILE_N, (N) - (POS_I)), ic = (POS_I) * (LDC) + (POS_H); \
  libxs_blasint j = 0, j_next = TILE_K; \
  if (((TILE_M) == mm) && ((TILE_N) == nn)) { \
    for (; j < max_j; j = j_next) { \
      LIBXS_MMCALL_PRF(xmm.LIBXS_TPREFIX(REAL,mm), \
        (A) + (POS_H) + (LDA) * j, \
        (B) + (POS_I) * (LDB) + j, \
        (C) + ic, \
        (A) + (POS_H) + (LDA) * j_next, \
        (B) + (POS_I) * (LDB) + j_next, \
        (C) + ic); \
      j_next = j + (TILE_K); \
    } \
  } \
  for (; j < (K); j = j_next) { /* remainder */ \
    LIBXS_XGEMM(REAL, libxs_blasint, FLAGS, mm, nn, LIBXS_MIN(TILE_K, (K) - j), \
      ALPHA, (A) + j * (LDA) + (POS_H), LDA, (B) + (POS_I) * (LDB) + j, LDB, BETA, (C) + ic, LDC); \
    j_next = j + (TILE_K); \
  } \
}

#define LIBXS_GEMM_EXTOMP_XGEMM(INIT, LOOP_BEGIN, LOOP_BODY, LOOP_END, \
  REAL, FLAGS, NT, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) INIT \
{ \
  const signed char scalpha = (signed char)(ALPHA), scbeta = (signed char)(BETA); \
  const int sufficient_size = LIBXS_GEMM_EXTOMP_SUFFICIENT_SIZE(M, N, K); \
  libxs_blasint tile_m = 0, tile_n = 0, tile_k = 0, num_m = 0, num_n = 0, num_k = 0; \
  libxs_xmmfunction xmm = { 0 }; \
  if (0 != sufficient_size \
    /*TODO: not supported*/&& 0 == ((FLAGS) & (LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B)) \
    /*TODO: not supported*/&& 1 == scalpha && (1 == scbeta || 0 == scbeta)) \
  { \
    tile_m = LIBXS_MAX(TILE_M, 2); tile_n = LIBXS_MAX(TILE_N, 2); tile_k = LIBXS_MAX(TILE_K, 2); \
    num_m = ((M) + tile_m - 1) / tile_m; num_n = ((N) + tile_n - 1) / tile_n; num_k = ((K) + tile_k - 1) / tile_k; \
    { /* opening scope for additional variable declarations */ \
      const libxs_blasint num_t = (LIBXS_GEMM_EXTOMP_OVERHEAD(NT) <= num_k && 1 < LIBXS_GEMM_EXTOMP_COLLAPSE) \
        ? (num_m * num_n) : (num_n <= num_m ? num_m : num_n); \
      const libxs_blasint min_ntasks = LIBXS_GEMM_EXTOMP_MIN_NTASKS(NT); \
      libxs_gemm_descriptor desc; \
      if (min_ntasks < num_t) { /* ensure enough parallel slack */ \
        tile_m = (M) / num_m; tile_n = (N) / num_n; \
      } \
      else if ((LIBXS_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { \
        const double ratio = sqrt(((double)min_ntasks) / num_t); \
        tile_n = (int)(num_n * ratio /*+ 0.5*/); \
        tile_m = (min_ntasks + tile_n - 1) / tile_n; \
      } \
      else if (num_n <= num_m) { \
        tile_m = ((M) + min_ntasks - 1) / min_ntasks; \
      } \
      else { \
        tile_n = ((N) + min_ntasks - 1) / min_ntasks; \
      } \
      { /* adjust for non-square operand shapes */ \
        float rm = 1.f, rn = ((float)(N)) / M, rk = ((float)(K)) / M; \
        if (1.f < rn) { rm /= rn; rn = 1.f; rk /= rn; } \
        if (1.f < rk) { rm /= rk; rn /= rk; rk = 1.f; } \
        tile_m = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_LOG2(tile_m * rm /*+ 0.5*/)), 8), M); \
        tile_n = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_LOG2(tile_n * rn /*+ 0.5*/)), 8), N); \
        tile_k = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_LOG2(tile_k * rk /*+ 0.5*/)), 8), K); \
      } \
      LIBXS_GEMM_DESCRIPTOR(desc, LIBXS_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, \
        LDA, LDB, LDC, scalpha, scbeta, internal_gemm_prefetch); \
      xmm = libxs_xmmdispatch(&desc); \
    } \
  } \
  if (0 != xmm.dmm) { \
    const libxs_blasint max_j = ((K) / tile_k) * tile_k; \
    libxs_blasint h = 0, i = 0; \
    if ((LIBXS_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { /* amortize overhead */ \
      LOOP_BEGIN \
      for (h = 0; h < (M); h += tile_m) { \
        for (i = 0; i < (N); i += tile_n) { \
          LOOP_BODY(h, i) \
          LIBXS_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
    else if (num_n <= num_m) { \
      LOOP_BEGIN \
      for (h = 0; h < (M); h += tile_m) { \
        LOOP_BODY(h) \
        for (i = 0; i < (N); i += tile_n) { \
          LIBXS_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
    else { \
      LOOP_BEGIN \
      for (i = 0; i < (N); i += tile_n) { \
        LOOP_BODY(i) \
        for (h = 0; h < (M); h += tile_m) { \
          LIBXS_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
  } \
  else if (0 != sufficient_size) { /* BLAS fallback */ \
    LIBXS_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { /* small problem size */ \
    LIBXS_XGEMM(REAL, libxs_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


#if defined(LIBXS_GEMM_EXTWRAP) && !defined(__STATIC)

/* implementation variant for non-static linkage; overrides weak libxs_gemm_init in libxs_gemm.c */
LIBXS_API_DEFINITION int libxs_gemm_init(int archid, int prefetch)
{
  union { const void* pv; libxs_sgemm_function pf; } fn_sgemm = { NULL };
  union { const void* pv; libxs_dgemm_function pf; } fn_dgemm = { NULL };
  fn_sgemm.pv = dlsym(RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(sgemm)));
  fn_dgemm.pv = dlsym(RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dgemm)));

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
  else { /* default: potentially sequential or externally parallelized */
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
  else { /* default: potentially sequential or externally parallelized */
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
      LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXS_XGEMM(float, libxs_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
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
      LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXS_XGEMM(double, libxs_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    } break;
    default: { /* tiled xGEMM */
      libxs_omp_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
}

#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
