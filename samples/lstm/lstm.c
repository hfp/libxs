/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxs.h>
#include <math.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(__MKL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS evailable */ \
  (LIBXS_EQUAL(REAL_TYPE, float) || LIBXS_EQUAL(REAL_TYPE, double))
# define CHECK
#endif


LIBXS_INLINE LIBXS_RETARGETABLE void init(int seed, REAL_TYPE *LIBXS_RESTRICT dst,
  libxs_blasint nrows, libxs_blasint ncols, libxs_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxs_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)seed;
    }
  }
}

void matrix_add(libxs_blasint size, REAL_TYPE *a, REAL_TYPE *b, REAL_TYPE *c)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXS_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

void matrix_sigmoid(libxs_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxs_blasint i;
  REAL_TYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXS_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    exp_value = (REAL_TYPE)exp( -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}

void matrix_relu(libxs_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  LIBXS_PRAGMA_SIMD
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : -src[i];
  }
}

int main(int argc, char* argv[])
{
  const libxs_blasint m = (1 < argc ? atoi(argv[1]) : 1024);
  const libxs_blasint k = (3 < argc ? atoi(argv[3]) : m);
  const libxs_blasint n = (2 < argc ? atoi(argv[2]) : k);
  const libxs_blasint t = (4 < argc ? atoi(argv[4]) : 3);
  const libxs_blasint bm = (5 < argc ? atoi(argv[5]) : 32);
  const libxs_blasint bk = (7 < argc ? atoi(argv[7]) : bm);
  const libxs_blasint bn = (6 < argc ? atoi(argv[6]) : bk);
  const libxs_bgemm_order order = (libxs_bgemm_order)(8 < argc ? atoi(argv[8]) : 0);
  const int nrepeat = (9 < argc ? atoi(argv[9]) : 100);
  const libxs_blasint b_m1 = (10 < argc ? atoi(argv[10]) : 1);
  const libxs_blasint b_n1  = (11 < argc ? atoi(argv[11]) : 1);
  const libxs_blasint b_k1 = (12 < argc ? atoi(argv[12]) : 1);
  const libxs_blasint b_k2 = (13 < argc ? atoi(argv[13]) : 1);
  const libxs_blasint b_m2 = (14 < argc ? atoi(argv[14]) : 1);
  const libxs_blasint ldw = (15 < argc ? atoi(argv[15]) : m);
  const libxs_blasint ldx = (16 < argc ? atoi(argv[16]) : k);
  const libxs_blasint ldz = (17 < argc ? atoi(argv[17]) : m);
  const libxs_blasint ldu = (18 < argc ? atoi(argv[18]) : m);
  const libxs_blasint ldh = (19 < argc ? atoi(argv[19]) : m);
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  int result = EXIT_SUCCESS;
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./lstm [M] [N] [K] [time_steps] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [b_m2]\n\n");
    return result;
  }

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* wgold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* xgoldt = (REAL_TYPE*)libxs_malloc(ldx * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* ugold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* hgold = (REAL_TYPE*)libxs_malloc(ldh * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* z2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* zgold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* w = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* xt = (REAL_TYPE*)libxs_malloc(k * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* u = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* h = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* z2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    LIBXS_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
    LIBXS_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
    LIBXS_VLA_DECL(2, REAL_TYPE, z1, z1t, m * n);
    libxs_bgemm_handle* handlewx = 0;
    libxs_bgemm_handle* handleuh = 0;
    libxs_bgemm_handle* handlett = 0;
    unsigned long long start;
    double duration;
#if defined(CHECK)
    const char *const env_check = getenv("CHECK");
    const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
    const libxs_gemm_prefetch_type strategy = LIBXS_PREFETCH_AUTO;
    handlewx = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);

    if (0 != handlewx && 0 != handleuh) {
      init(42, wgold, m, k, ldw, 1.0);
      int it;
      for (it = 0; it < t; ++it) {
        init(24, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
      }
      init(42, ugold, m, m, ldu, 1.0);
      init(24, hgold, m, n, ldh, 1.0);
      init( 0, z1gold, m, n, ldz, 1.0);
      init( 0, z2gold, m, n, ldz, 1.0);
      init( 0, zgold, m, n, ldz, 1.0);
      libxs_bgemm_copyin_a(handlewx, wgold, &ldw, w);
      for (it = 0; it < t; ++it) {
        libxs_bgemm_copyin_b(handlewx, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXS_VLA_ACCESS(2, x, it, 0, k * n));
      }
      libxs_bgemm_copyin_a(handleuh, ugold, &ldu, u);
      libxs_bgemm_copyin_b(handleuh, hgold, &ldh, h);
      for (it = 0; it < t; ++it) {
        libxs_bgemm_copyin_c(handlewx, z1gold, &ldz, &LIBXS_VLA_ACCESS(2, z1, it, 0, m * n));
      }
      libxs_bgemm_copyin_c(handleuh, z2gold, &ldz, z2);
      libxs_bgemm_copyin_c(handlewx, zgold, &ldz, z);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warmup OpenMP (populate thread pool) */
      libxs_bgemm_omp(handlewx, w, x, &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), 1);
#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) {
        LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, z1gold, &ldz);
      }
#endif
      libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &k, &alpha, w, &ldw, x, &ldx, &beta, &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), &ldz);
      fprintf(stdout, "\n\n");
      /* warmup OpenMP (populate thread pool) */
      libxs_bgemm_omp(handleuh, u, h, z2, 1);
#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) {
        LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
      }
#endif
      libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
        &transa, &transb, &m, &n, &m, &alpha, u, &ldu, h, &ldh, &beta, z2, &ldz);
      fprintf(stdout, "\n\n");

      int s;
      int i;
      start = libxs_timer_tick();
      for (s = 0; s < nrepeat; ++s) {
        /* The following loop may be absorbed into libxs_lstm_omp */
        libxs_bgemm_omp(handlett, w, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), 1/*nrepeat*/);
        for (i = 0; i < t-1; ++i) {
          libxs_bgemm_omp(handleuh, u, h, z2, 1/*nrepeat*/);
          matrix_add(m*n, &LIBXS_VLA_ACCESS(2, z1, i, 0, m * n), z2, z);
          matrix_relu(m*n, z, h);
        }
        libxs_bgemm_omp(handleuh, u, h, z2, 1/*nrepeat*/);
        matrix_add(m*n, &LIBXS_VLA_ACCESS(2, z1, t-1, 0, m * n), z2, z);
      }
      duration = libxs_timer_duration(start, libxs_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        REAL_TYPE* ztest = 0;
        int i;
        start = libxs_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (i = 0; i < t-1; ++i) {
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, i, 0, k * n), &ldx, &beta, z1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
            matrix_add(m*n, z1gold, z2gold, zgold);
            matrix_relu(m*n, zgold, hgold);
          }
          LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, t-1, 0, k * n), &ldx, &beta, z1gold, &ldz);
          LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
          matrix_add(m*n, z1gold, z2gold, zgold);
        }
        duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxs_free(wgold); wgold = 0;
        libxs_free(xgoldt); xgoldt = 0;
        libxs_free(ugold); ugold = 0;
        libxs_free(hgold); hgold = 0;
        libxs_free(z1gold); z1gold = 0;
        libxs_free(z2gold); z2gold = 0;
        libxs_free(w); w = 0;
        libxs_free(xt); xt = 0;
        libxs_free(u); u = 0;
        libxs_free(h); h = 0;
        libxs_free(z1t); z1t = 0;
        libxs_free(z2); z2 = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ztest = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
        if (0 != ztest) {
          libxs_matdiff_info diff;
          libxs_bgemm_copyout_c(handleuh, z, &ldz, ztest);
          if (EXIT_SUCCESS == libxs_matdiff(LIBXS_DATATYPE(REAL_TYPE), m, n, zgold, ztest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxs_free(ztest);
        }
      }
#endif
      libxs_bgemm_handle_destroy(handlewx);
      libxs_bgemm_handle_destroy(handleuh);
      libxs_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXS_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxs_free(wgold);
    libxs_free(xgoldt);
    libxs_free(ugold);
    libxs_free(hgold);
    libxs_free(z1gold);
    libxs_free(z2gold);
    libxs_free(zgold);
    libxs_free(w);
    libxs_free(xt);
    libxs_free(u);
    libxs_free(h);
    libxs_free(z1t);
    libxs_free(z2);
    libxs_free(z);
  }
  fprintf(stdout, "Finished\n");

  return result;
}

