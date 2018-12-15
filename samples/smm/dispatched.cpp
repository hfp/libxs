/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 0 /* enable padding on a per-matrix basis */
# define PAD(TYPE, VALUE) (LIBXS_UP2((VALUE) * sizeof(TYPE), LIBXS_ALIGNMENT) / sizeof(TYPE))
#else
# define PAD(TYPE, VALUE) (VALUE)
#endif

#if !defined(RANDOMIZED) && 0
# define RANDOMIZED
#endif

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
    const libxs_blasint benchmark = 1 < argc ? std::atoi(argv[1]) : 0;
    const libxs_blasint m = (2 < argc ? std::atoi(argv[2]) : 23);
    const libxs_blasint k = (4 < argc ? std::atoi(argv[4]) : m);
    const libxs_blasint n = (3 < argc ? std::atoi(argv[3]) : k);
    const libxs_blasint q = (5 < argc ? std::atoi(argv[5]) : 0/*auto*/);
    const libxs_blasint nrepeat = (6 < argc ? std::atoi(argv[6]) : (0 >= q ? 13 : 1));

    const libxs_blasint lda = m, ldb = k, ldc = m;
    const char transa = 'N', transb = 'N';
    const OTYPE alpha = 1, beta = 1;

    const libxs_blasint asize = PAD(ITYPE, lda * k), bsize = PAD(ITYPE, ldb * n), csize = PAD(OTYPE, ldc * n);
    const libxs_blasint max_size = ((2ULL << 30/*2 GB*/) / ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE)));
    const libxs_blasint s = LIBXS_MIN(0 < q ? q : max_size, max_size);
    const libxs_blasint aspace = LIBXS_ALIGNMENT / sizeof(ITYPE);
    const size_t bwsize = (static_cast<size_t>(asize)/*load*/ + static_cast<size_t>(bsize)/*load*/) * sizeof(ITYPE)
                        + (sizeof(OTYPE) * static_cast<size_t>(csize) * 2/*RFO*/);
    const double gflops = 2E-9 * s * m * n * k;
#if LIBXS_TYPEINFO(ITYPE, FP)
    const char *const ops = "FLOPS";
    const double scale = 1.0 / s;
#else
    const char *const ops = "OPS";
    const double scale = 1;
#endif
#if !defined(_DEBUG)
    const char *const env_check = getenv("CHECK");
    const int check = (0 == env_check ? 0 : atoi(env_check));
#else
    /*const*/ int check = 1;
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
#   pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
    {
#if defined(_OPENMP)
      const libxs_blasint chunksize = s / omp_get_max_threads();
#endif
      struct raii { // avoid std::vector (first-touch init. causes NUMA issue)
        ITYPE *a, *b;
        OTYPE *c;
        size_t m_size, m_shuffle;
        raii(libxs_blasint asize_, libxs_blasint bsize_, libxs_blasint csize_, libxs_blasint size_)
          : a(new ITYPE[static_cast<size_t>(asize_)]), b(new ITYPE[static_cast<size_t>(bsize_)])
          , c(new OTYPE[static_cast<size_t>(csize_)])
          , m_size(static_cast<size_t>(size_)), m_shuffle(libxs_shuffle(static_cast<unsigned int>(size_)))
        {}
        ~raii() { delete[] a; delete[] b; delete[] c; }
#if defined(RANDOMIZED)
        libxs_blasint shuffle(libxs_blasint i) const { return (i * m_shuffle) % m_size; }
#else
        libxs_blasint shuffle(libxs_blasint i) const { return i; }
#endif
      } helper(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1, s);

      ITYPE *const a = LIBXS_ALIGN(helper.a, LIBXS_ALIGNMENT);
      ITYPE *const b = LIBXS_ALIGN(helper.b, LIBXS_ALIGNMENT);
      OTYPE *const c = LIBXS_ALIGN(helper.c, LIBXS_ALIGNMENT);
#if defined(_OPENMP)
#     pragma omp parallel for schedule(static)
#endif
      for (libxs_blasint i = 0; i < s; ++i) {
        LIBXS_MATRNG(ITYPE, 42 + helper.shuffle(i), a + static_cast<size_t>(asize) * helper.shuffle(i), m, k, lda, scale);
        LIBXS_MATRNG(ITYPE, 24 + helper.shuffle(i), b + static_cast<size_t>(bsize) * helper.shuffle(i), k, n, ldb, scale);
        LIBXS_MATRNG(OTYPE, 22 + i, c + static_cast<size_t>(csize) * i, m, n, ldc, scale);
      }

#if defined(MKL_ENABLE_AVX512)
      mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
      // initialize LIBXS
      libxs_init();

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (input=%s output=%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE))) / (1ULL << 20),
        LIBXS_TYPENAME(ITYPE), LIBXS_TYPENAME(OTYPE));

      // eventually JIT-compile the requested kernel
      libxs_mmfunction<ITYPE,OTYPE>(LIBXS_GEMM_FLAGS(transa, transb), m, n, k, lda, ldb, ldc, alpha, beta);

      switch (benchmark) {
      case 0: { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            libxs_gemm(&transa, &transb, m, n, k,
              &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
               &beta, c + static_cast<size_t>(csize) * i, &ldc);
          }
        }
        const unsigned long long ncycles = libxs_timer_diff(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;

      case 1: { // streaming A and C
        fprintf(stdout, "Streamed (A,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            libxs_gemm(&transa, &transb, m, n, k,
              &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b, &ldb,
               &beta, c + static_cast<size_t>(csize) * i, &ldc);
          }
        }
        const unsigned long long ncycles = libxs_timer_diff(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;

      case 2: { // streaming B and C
        fprintf(stdout, "Streamed (B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            libxs_gemm(&transa, &transb, m, n, k,
              &alpha, a, &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
               &beta, c + static_cast<size_t>(csize) * i, &ldc);
          }
        }
        const unsigned long long ncycles = libxs_timer_diff(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;

      case 3: { // streaming A and B
        fprintf(stdout, "Streamed (A,B)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxs_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxs_blasint j = 0;
#endif
            libxs_gemm(&transa, &transb, m, n, k,
              &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
               &beta, c + j, &ldc);
          }
        }
        const unsigned long long ncycles = libxs_timer_diff(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;

      case 4: { // cached
        fprintf(stdout, "Cached...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            const libxs_blasint j = omp_get_thread_num() * chunksize * csize;
#else
            const libxs_blasint j = 0;
#endif
            libxs_gemm(&transa, &transb, m, n, k,
              &alpha, a, &lda, b, &ldb,
               &beta, c + j, &ldc);
          }
        }
        const unsigned long long ncycles = libxs_timer_diff(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } break;
      default: throw "invalid case selected!";
      } /*switch*/
      if (0 != check) {
        libxs_matdiff_info diff;
        result = libxs_matdiff(&diff, LIBXS_DATATYPE(OTYPE), m, n, c, NULL, &ldc, &ldc);
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "\tcheck: %f\n", diff.l1_ref);
        }
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
  catch(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    result = EXIT_FAILURE;
  }
  catch(...) {
    fprintf(stderr, "Error: unknown exception caught!\n");
    result = EXIT_FAILURE;
  }

  return result;
}

