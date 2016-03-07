/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(_OPENMP)
# if !defined(LIBXS_GEMM_EXTOMP_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXS_GEMM_EXTOMP_TASKS
# endif
# define LIBXS_GEMM_EXTOMP_MIN_NTASKS(NT) (40 * omp_get_num_threads() / (NT))
# define LIBXS_GEMM_EXTOMP_OVERHEAD(NT) (4 * (NT))
# if defined(LIBXS_GEMM_EXTOMP_TASKS)
#   define LIBXS_GEMM_EXTOMP_START LIBXS_PRAGMA(omp single nowait)
#   define LIBXS_GEMM_EXTOMP_TASK_SYNC LIBXS_PRAGMA(omp taskwait)
#   define LIBXS_GEMM_EXTOMP_TASK(...) LIBXS_PRAGMA(omp task firstprivate(__VA_ARGS__))
#   define LIBXS_GEMM_EXTOMP_FOR(N)
# else
#   define LIBXS_GEMM_EXTOMP_START
#   define LIBXS_GEMM_EXTOMP_TASK_SYNC
#   define LIBXS_GEMM_EXTOMP_TASK(...)
#   define LIBXS_GEMM_EXTOMP_FOR(N) /*LIBXS_PRAGMA(omp for LIBXS_OPENMP_COLLAPSE(N) schedule(dynamic))*/
# endif
#else
# define LIBXS_GEMM_EXTOMP_MIN_NTASKS(NT) 1
# define LIBXS_GEMM_EXTOMP_OVERHEAD(NT) 0
# define LIBXS_GEMM_EXTOMP_START
# define LIBXS_GEMM_EXTOMP_TASK_SYNC
# define LIBXS_GEMM_EXTOMP_TASK(...)
# define LIBXS_GEMM_EXTOMP_FOR(N)
#endif

#define LIBXS_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxs_blasint mm = LIBXS_MIN(tile_m, (M) - (POS_H)); \
  const libxs_blasint nn = LIBXS_MIN(tile_n, (N) - (POS_I)); \
  const libxs_blasint ic = (POS_I) * (LDC) + (POS_H); \
  libxs_blasint j = 0; \
  if ((tile_m == mm) && (tile_n == nn)) { \
    for (; j < max_j; j += tile_k) { \
      LIBXS_MMCALL(xmm.LIBXS_TPREFIX(REAL,mm), (A) + j * (LDA) + (POS_H), (B) + (POS_I) * (LDB) + j, (C) + ic, M, N, K, LDA, LDB, LDC); \
    } \
  } \
  for (; j < (K); j += tile_k) { /* remainder */ \
    LIBXS_XGEMM(REAL, libxs_blasint, FLAGS, mm, nn, LIBXS_MIN(tile_k, (K) - j), \
      ALPHA, (A) + j * (LDA) + (POS_H), LDA, (B) + (POS_I) * (LDB) + j, LDB, BETA, (C) + ic, LDC); \
  } \
}

#define LIBXS_GEMM_EXTOMP_XGEMM(REAL, FLAGS, NT, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  libxs_blasint tile_m = LIBXS_MAX(TILE_M, 2), tile_n = LIBXS_MAX(TILE_N, 2), tile_k = LIBXS_MAX(TILE_K, 2); \
  const libxs_blasint num_m = ((M) + tile_m - 1) / tile_m, num_n = ((N) + tile_n - 1) / tile_n, num_k = ((K) + tile_k - 1) / tile_k; \
  libxs_xmmfunction xmm; \
  LIBXS_GEMM_EXTOMP_START \
  { \
    const libxs_blasint num_t = (LIBXS_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k ? (num_m * num_n) : (num_n <= num_m ? num_m : num_n); \
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
      tile_m = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_NBITS(tile_m * rm /*+ 0.5*/)),  8), M); \
      tile_n = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_NBITS(tile_n * rn /*+ 0.5*/)),  8), N); \
      tile_k = LIBXS_MIN(LIBXS_MAX((libxs_blasint)(1 << LIBXS_NBITS(tile_k * rk /*+ 0.5*/)), 32), K); \
    } \
    LIBXS_GEMM_DESCRIPTOR(desc, LIBXS_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, LDA, LDB, LDC, ALPHA, BETA, LIBXS_PREFETCH); \
    xmm = libxs_xmmdispatch(&desc); \
  } \
  if (0 != xmm.dmm) { \
    LIBXS_GEMM_EXTOMP_START \
    { \
      const libxs_blasint max_j = ((K) / tile_k) * tile_k; \
      libxs_blasint h = 0, i = 0; \
      if ((LIBXS_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { /* amortize overhead */ \
        LIBXS_GEMM_EXTOMP_FOR(2) \
        for (; h < (M); h += tile_m) { \
          for (; i < (N); i += tile_n) { \
            LIBXS_GEMM_EXTOMP_TASK(h, i) \
            LIBXS_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXS_GEMM_EXTOMP_TASK_SYNC \
      } \
      else if (num_n <= num_m) { \
        LIBXS_GEMM_EXTOMP_FOR(2) \
        for (; h < (M); h += tile_m) { \
          LIBXS_GEMM_EXTOMP_TASK(h) \
          for (; i < (N); i += tile_n) { \
            LIBXS_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXS_GEMM_EXTOMP_TASK_SYNC \
      } \
      else { \
        LIBXS_GEMM_EXTOMP_FOR(2) \
        for (; i < (N); i += tile_n) { \
          LIBXS_GEMM_EXTOMP_TASK(i) \
          for (; h < (M); h += tile_m) { \
            LIBXS_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXS_GEMM_EXTOMP_TASK_SYNC \
      } \
    } \
  } \
  else { /* fallback */ \
    LIBXS_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_omps_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXS_GEMM_EXTOMP_XGEMM(float, flags | LIBXS_GEMM_FLAG_F32PREC, libxs_internal_num_nt,
    libxs_internal_tile_size[1/*SP*/][0/*M*/],
    libxs_internal_tile_size[1/*SP*/][1/*N*/],
    libxs_internal_tile_size[1/*SP*/][2/*K*/], *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_omps_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXS_GEMM_EXTOMP_XGEMM(double, flags, libxs_internal_num_nt,
    libxs_internal_tile_size[0/*DP*/][0/*M*/],
    libxs_internal_tile_size[0/*DP*/][1/*N*/],
    libxs_internal_tile_size[0/*DP*/][2/*K*/], *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
    a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXS_BETA),
    c, *(ldc ? ldc : LIBXS_LD(m, n)));
}


#if defined(LIBXS_GEMM_EXTWRAP)

LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_SGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  assert(LIBXS_GEMM_EXTWRAP_SGEMM != libxs_internal_sgemm);
  switch (libxs_internal_gemm) {
    case 1: {
      libxs_omps_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case 2: {
#if defined(_OPENMP)
#     pragma omp parallel
#     pragma omp single
#endif
      libxs_omps_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: {
      LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXS_XGEMM(float, libxs_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_DGEMM(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  assert(LIBXS_GEMM_EXTWRAP_DGEMM != libxs_internal_dgemm);
  switch (libxs_internal_gemm) {
    case 1: {
      libxs_omps_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case 2: {
#if defined(_OPENMP)
#     pragma omp parallel
#     pragma omp single
#endif
      libxs_omps_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: {
      LIBXS_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXS_XGEMM(double, libxs_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXS_ALPHA),
        a, *(lda ? lda : LIBXS_LD(m, k)), b, *(ldb ? ldb : LIBXS_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXS_BETA),
        c, *(ldc ? ldc : LIBXS_LD(m, n)));
    }
  }
}

#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
