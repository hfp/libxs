/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif

#if !defined(MAX_SIZE)
# define MAX_SIZE ((LIBXS_MAX_M) * (LIBXS_MAX_N))
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


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    typedef REAL_TYPE T;
    const int m = 1 < argc ? std::atoi(argv[1]) : 23;
    const int n = 2 < argc ? std::atoi(argv[2]) : m;
    const int k = 3 < argc ? std::atoi(argv[3]) : m;

    const int asize = m * k, bsize = k * n, csize = m * n, aspace = LIBXS_ALIGNMENT / sizeof(T);
    const int s = (2ULL << 30) / ((asize + bsize + csize) * sizeof(T)); // 2 GByte
    const size_t bwsize_batched = (asize/*load*/ + bsize/*load*/ + 2 * csize/*RFO*/) * sizeof(T); // batched
    const size_t bwsize = (asize/*load*/ + bsize/*load*/) * sizeof(T); // streamed, skipping C since it is just in cache
    const double gflops = 2.0 * s * m * n * k * 1E-9, scale = 1.0 / s;

    struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
      T *a, *b, *c;
      raii(int asize_, int bsize_, int csize_): a(new T[asize_]), b(new T[bsize_]), c(new T[csize_]) {}
      ~raii() { delete[] a; delete[] b; delete[] c; }
    } buffer(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1);
    T *const a = LIBXS_ALIGN(buffer.a, LIBXS_ALIGNMENT);
    T *const b = LIBXS_ALIGN(buffer.b, LIBXS_ALIGNMENT);
    T *c = LIBXS_ALIGN(buffer.c, LIBXS_ALIGNMENT);

#if defined(_OPENMP)
#   pragma omp parallel for
#endif
    for (int i = 0; i < s; ++i) {
      init(42 + i, a + i * asize, m, k, m, scale);
      init(24 + i, b + i * bsize, k, n, k, scale);
      init(22 + i, c + i * csize, m, n, m, scale);
    }

#if defined(LIBXS_OFFLOAD_TARGET)
#   pragma offload target(LIBXS_OFFLOAD_TARGET) in(a: length(s * asize)) in(b: length(s * bsize)) inout(c: length(s * csize))
#endif
    {
#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXS
      libxs_init();

      // eventually JIT-compile the requested kernel
      libxs_mmfunction<T>(m, n, k);

      fprintf(stdout, "m=%i n=%i k=%i size=%i memory=%.1f MB (%s)\n\n", m, n, k, s,
        1.0 * (s * (asize + bsize + csize) * sizeof(T)) / (1 << 20), 8 == sizeof(T) ? "DP" : "SP");

      { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        unsigned long long x = libxs_timer_xtick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          libxs_gemm(0/*transa*/, 0/*transb*/, m, n, k,
            0/*alpha*/, a + i * asize, 0/*lda*/, b + i * bsize, 0/*ldb*/,
            0/*beta*/, c + i * csize, 0/*ldc*/);
        }
        x = std::max(libxs_timer_xtick(), x) - x;
        const double duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration && 0 != x) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (s * (2.0 * m * n * k - m * n)) / x);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize_batched / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      { // streaming A and C
        fprintf(stdout, "Streamed (A,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        unsigned long long x = libxs_timer_xtick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          libxs_gemm(0/*transa*/, 0/*transb*/, m, n, k,
            0/*alpha*/, a + i * asize, 0/*lda*/, b, 0/*ldb*/,
            0/*beta*/, c + i * csize, 0/*ldc*/);
        }
        x = std::max(libxs_timer_xtick(), x) - x;
        const double duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration && 0 != x) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (s * (2.0 * m * n * k - m * n)) / x);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      { // streaming B and C
        fprintf(stdout, "Streamed (B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        unsigned long long x = libxs_timer_xtick();
#if defined(_OPENMP)
#       pragma omp parallel for
#endif
        for (int i = 0; i < s; ++i) {
          libxs_gemm(0/*transa*/, 0/*transb*/, m, n, k,
            0/*alpha*/, a, 0/*lda*/, b + i * bsize, 0/*ldb*/,
            0/*beta*/, c + i * csize, 0/*ldc*/);
        }
        x = std::max(libxs_timer_xtick(), x) - x;
        const double duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration && 0 != x) {
          fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (s * (2.0 * m * n * k - m * n)) / x);
          fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      }

      if ((MAX_SIZE) >= csize) {
        { // streaming A and B
          fprintf(stdout, "Streamed (A,B)...\n");
          const unsigned long long start = libxs_timer_tick();
          unsigned long long x = libxs_timer_xtick();
#if defined(_OPENMP)
#         pragma omp parallel for
#endif
          for (int i = 0; i < s; ++i) {
            T tmp[MAX_SIZE]; // make sure that stacksize is covering the problem size
            // do nothing else with tmp; just a benchmark
            libxs_gemm(0/*transa*/, 0/*transb*/, m, n, k,
              0/*alpha*/, a + i * asize, 0/*lda*/, b + i * bsize, 0/*ldb*/,
              0/*beta*/, tmp, 0/*ldc*/);
          }
          x = std::max(libxs_timer_xtick(), x) - x;
          const double duration = libxs_timer_duration(start, libxs_timer_tick());
          if (0 < duration && 0 != x) {
            fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (s * (2.0 * m * n * k - m * n)) / x);
            fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
            fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1 << 30)));
          }
          fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        }

        { // cached
          fprintf(stdout, "Cached...\n");
          const unsigned long long start = libxs_timer_tick();
          unsigned long long x = libxs_timer_xtick();
#if defined(_OPENMP)
#         pragma omp parallel for
#endif
          for (int i = 0; i < s; ++i) {
            T tmp[MAX_SIZE]; // make sure that stacksize is covering the problem size
            // do nothing else with tmp; just a benchmark
            libxs_gemm(0/*transa*/, 0/*transb*/, m, n, k,
              0/*alpha*/, a, 0/*lda*/, b, 0/*ldb*/,
              0/*beta*/, tmp, 0/*ldc*/);
          }
          x = std::max(libxs_timer_xtick(), x) - x;
          const double duration = libxs_timer_duration(start, libxs_timer_tick());
          if (0 < duration && 0 != x) {
            fprintf(stdout, "\tpseudo-perf.: %.1f FLOPS/cycle\n", (s * (2.0 * m * n * k - m * n)) / x);
            fprintf(stdout, "\tperformance: %.1f GFLOPS/s\n", gflops / duration);
          }
          fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        }
      }
      else {
        fprintf(stderr, "Warning: size M x N is exceeding MAX_SIZE!\n");
      }

      // finalize LIBXS
      libxs_finalize();
      fprintf(stdout, "Finished\n");
    }
  }
  catch(const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}
