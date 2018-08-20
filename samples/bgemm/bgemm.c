/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__MKL)
# include <mkl_service.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(ITYPE)
# define ITYPE float
#endif

#if !defined(CHECK) && (LIBXS_EQUAL(ITYPE, float) || LIBXS_EQUAL(ITYPE, double))
LIBXS_GEMM_SYMBOL_DECL(LIBXS_GEMM_CONST, ITYPE)
# define CHECK
#endif

#define MYASSERT(x) if(!(x)) { printf("Assertion %s failed...\n", #x); exit(1);}


int main(int argc, char* argv[])
{
  LIBXS_GEMM_CONST libxs_blasint m = (1 < argc ? atoi(argv[1]) : 1024);
  LIBXS_GEMM_CONST libxs_blasint k = (3 < argc ? atoi(argv[3]) : m);
  LIBXS_GEMM_CONST libxs_blasint n = (2 < argc ? atoi(argv[2]) : k);
  const libxs_blasint bm = (4 < argc ? atoi(argv[4]) : 32);
  const libxs_blasint bk = (6 < argc ? atoi(argv[6]) : bm);
  const libxs_blasint bn = (5 < argc ? atoi(argv[5]) : bk);
  const libxs_bgemm_order order = (libxs_bgemm_order)(7 < argc ? atoi(argv[7]) : 0);
  const int nrepeat = (8 < argc ? atoi(argv[8]) : 100);
  const libxs_blasint b_m1 = (9 < argc ? atoi(argv[9]) : 1);
  const libxs_blasint b_n1  = (10 < argc ? atoi(argv[10]) : 1);
  const libxs_blasint b_k1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxs_blasint b_k2 = (12 < argc ? atoi(argv[12]) : 1);
  const int ab = (13 < argc ? atoi(argv[13]) : 0);
  LIBXS_GEMM_CONST libxs_blasint lda = (14 < argc ? atoi(argv[13]) : m);
  LIBXS_GEMM_CONST libxs_blasint ldb = (15 < argc ? atoi(argv[14]) : k);
  LIBXS_GEMM_CONST libxs_blasint ldc = (16 < argc ? atoi(argv[15]) : m);
  LIBXS_GEMM_CONST char transa = 'N', transb = 'N'; /* no transposes */
  LIBXS_GEMM_CONST ITYPE alpha = 1, beta = 1;
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const double gflops = 2.0 * m * n * k * 1E-9;
  int result = EXIT_SUCCESS;
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./bgemm [M] [N] [K] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [verbose]\n\n");
    return result;
  }

  MYASSERT(m % b_m1 == 0);
  MYASSERT(n % b_n1 == 0);
  MYASSERT(k % b_k1 == 0);
  MYASSERT(m/b_m1 % bm == 0);
  MYASSERT(n/b_n1 % bn == 0);
  MYASSERT(k/b_k1/b_k2 % bk == 0);

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
  {
    ITYPE* agold = (ITYPE*)libxs_malloc((size_t)(lda * k * sizeof(ITYPE)));
    ITYPE* bgold = (ITYPE*)libxs_malloc((size_t)(ldb * n * sizeof(ITYPE)));
    ITYPE* cgold = (ITYPE*)libxs_malloc((size_t)(ldc * n * sizeof(ITYPE)));
    ITYPE* a = (ITYPE*)libxs_malloc((size_t)(m * k * sizeof(ITYPE)));
    ITYPE* b = (ITYPE*)libxs_malloc((size_t)(k * n * sizeof(ITYPE)));
    ITYPE* c = (ITYPE*)libxs_malloc((size_t)(m * n * sizeof(ITYPE)));
    libxs_bgemm_handle* handle = 0;
    unsigned long long start;
    double duration;
#if defined(_OPENMP)
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    handle = libxs_bgemm_handle_create(nthreads,
      LIBXS_GEMM_PRECISION(ITYPE), LIBXS_GEMM_PRECISION(ITYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, NULL/*auto-prefetch*/, &order);

    if (0 != handle) {
      LIBXS_MATRNG(ITYPE, 42, agold, m, k, lda, 1.0);
      LIBXS_MATRNG(ITYPE, 24, bgold, k, n, ldb, 1.0);
      LIBXS_MATRNG(ITYPE,  0, cgold, m, n, ldc, 1.0);
      libxs_bgemm_copyin_a(handle, agold, &lda, a);
      libxs_bgemm_copyin_b(handle, bgold, &ldb, b);
      libxs_bgemm_copyin_c(handle, cgold, &ldc, c);
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      /* warm-up OpenMP (populate thread pool) */
      libxs_bgemm_omp(handle, a, b, c, 1);
#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) {
        LIBXS_GEMM_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
      }
#endif
      if (!ab) {
      libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(ITYPE),
        &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      fprintf(stdout, "\n\n");
      }
      start = libxs_timer_tick();
      libxs_bgemm_omp(handle, a, b, c, nrepeat);
      duration = libxs_timer_duration(start, libxs_timer_tick());
      if (0 < duration) {
        if (ab) {
          fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s | %lli,%lli,%lli,%lli,%lli,%lli,%i,%lli,%lli,%lli,%lli\n",
            gflops * nrepeat / duration, (long long)m, (long long)n, (long long)k, (long long)bm, (long long)bn, (long long)bk,
            (int)order, (long long)b_m1, (long long)b_n1, (long long)b_k1, (long long)b_k2);
        } else {
          fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
      }
#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        ITYPE* ctest = 0;
        int i;
        start = libxs_timer_tick();
        for (i = 0; i < nrepeat; ++i) {
          LIBXS_GEMM_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, agold, &lda, bgold, &ldb, &beta, cgold, &ldc);
        }
        duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxs_free(agold); agold = 0;
        libxs_free(bgold); bgold = 0;
        libxs_free(a); a = 0;
        libxs_free(b); b = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ctest = (ITYPE*)libxs_malloc((size_t)(ldc * n * sizeof(ITYPE)));
        if (0 != ctest) {
          libxs_matdiff_info diff;
          libxs_bgemm_copyout_c(handle, c, &ldc, ctest);
          result = libxs_matdiff(LIBXS_DATATYPE(ITYPE), m, n, cgold, ctest, &ldc, &ldc, &diff);
          if (EXIT_SUCCESS == result) {
            fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxs_free(ctest);
        }
      }
#endif
      libxs_bgemm_handle_destroy(handle);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXS_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxs_free(agold);
    libxs_free(bgold);
    libxs_free(cgold);
    libxs_free(a);
    libxs_free(b);
    libxs_free(c);
  }
  if(!ab) {
    fprintf(stdout, "Finished\n");
  }
  return result;
}

