/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
#include <vector>
#include <cmath>
#if defined(_OPENMP)
# define USEOMP(FUNCTION) LIBXS_USEOMP(FUNCTION)
# include <omp.h>
#else
# define USEOMP(FUNCTION) (FUNCTION)
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
    const libxs_blasint benchmark = (1 < argc ? std::atoi(argv[1]) : 0);
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
    const char ops[] = "FLOPS";
    const double scale = 1.0 / s;
#else
    const char ops[] = "OPS";
    const double scale = 1;
#endif
#if !defined(_DEBUG)
    const char *const env_check = getenv("CHECK");
    const int check = (NULL == env_check ? 0 : atoi(env_check));
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
        OTYPE *c, *d;
        size_t m_size, m_shuffle;
        raii(libxs_blasint asize_, libxs_blasint bsize_, libxs_blasint csize_, libxs_blasint size_)
          : a(new ITYPE[static_cast<size_t>(asize_)]), b(new ITYPE[static_cast<size_t>(bsize_)])
          , c(new OTYPE[static_cast<size_t>(csize_)]), d(new OTYPE[static_cast<size_t>(csize_)])
          , m_size(static_cast<size_t>(size_)), m_shuffle(libxs_shuffle(static_cast<unsigned int>(size_)))
        {}
        ~raii() { delete[] a; delete[] b; delete[] c; delete[] d; }
#if defined(RANDOMIZED)
        libxs_blasint shuffle(libxs_blasint i) const { return (i * m_shuffle) % m_size; }
#else
        libxs_blasint shuffle(libxs_blasint i) const { return i; }
#endif
      } helper(s * asize + aspace - 1, s * bsize + aspace - 1, s * csize + aspace - 1, s);

      ITYPE *const a = LIBXS_ALIGN(helper.a, LIBXS_ALIGNMENT);
      ITYPE *const b = LIBXS_ALIGN(helper.b, LIBXS_ALIGNMENT);
      OTYPE *const c = LIBXS_ALIGN(helper.c, LIBXS_ALIGNMENT);
      OTYPE *const d = LIBXS_ALIGN(helper.d, LIBXS_ALIGNMENT);
#if defined(_OPENMP)
      const int nthreads = omp_get_max_threads();
#     pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
      for (libxs_blasint i = 0; i < s; ++i) {
        LIBXS_MATRNG(ITYPE, 42 + helper.shuffle(i), a + static_cast<size_t>(asize) * helper.shuffle(i), m, k, lda, scale);
        LIBXS_MATRNG(ITYPE, 24 + helper.shuffle(i), b + static_cast<size_t>(bsize) * helper.shuffle(i), k, n, ldb, scale);
        LIBXS_MATRNG(OTYPE, 22 + i, c + static_cast<size_t>(csize) * i, m, n, ldc, scale);
        LIBXS_MATRNG(OTYPE, 22 + i, d + static_cast<size_t>(csize) * i, m, n, ldc, scale);
      }

      // initialize LIBXS
      libxs_init();

      fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (input=%s output=%s)\n\n",
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
        1.0 * (s * ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE))) / (1ULL << 20),
        LIBXS_TYPENAME(ITYPE), LIBXS_TYPENAME(OTYPE));

      const libxs_mmfunction<ITYPE,OTYPE> xmm(LIBXS_GEMM_FLAGS(transa, transb),
        m, n, k, lda, ldb, ldc, alpha, beta, LIBXS_PREFETCH);
      if (!xmm) throw "no specialized routine found!";

      // arrays needed for the batch interface (indirect)
      std::vector<const ITYPE*> va_array(static_cast<size_t>(s)), vb_array(static_cast<size_t>(s));
      std::vector<OTYPE*> vc_array(static_cast<size_t>(s));
      const ITYPE* *const a_array = &va_array[0];
      const ITYPE* *const b_array = &vb_array[0];
      OTYPE* *const c_array = &vc_array[0];

      switch (benchmark) {
      case 0: { // batched
        fprintf(stdout, "Batched (A,B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
#if (0 != LIBXS_PREFETCH)
            xmm(ai, bi, ci,
              LIBXS_GEMM_PREFETCH_A(ai + asize),
              LIBXS_GEMM_PREFETCH_B(bi + bsize),
              LIBXS_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(ai, bi, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */
      case 1: { // batched/indirect
        fprintf(stdout, "Indirect (A,B,C)...\n");
        for (libxs_blasint i = 0; i < s; ++i) {
          a_array[i] = a + static_cast<size_t>(asize) * helper.shuffle(i);
          b_array[i] = b + static_cast<size_t>(bsize) * helper.shuffle(i);
          c_array[i] = d + static_cast<size_t>(csize) * i;
        }
        const libxs_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
          USEOMP(libxs_gemm_batch)(LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), &transa, &transb, m, n, k,
            &alpha, &a_array[0], &lda, &ptrsize, &b_array[0], &ldb, &ptrsize, &beta, &c_array[0], &ldc, &ptrsize,
            0/*index_stride*/, 0/*index_base*/, s, 0/*batchcheck*/);
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == (benchmark & 1) && 0 != check) { /* Gold result is available */
          libxs_matdiff_info diff;
          libxs_matdiff_clear(&diff);
          for (libxs_blasint h = 0; h < s; ++h) {
            const OTYPE *const u = c + static_cast<size_t>(csize) * h, *const v = c_array[h];
            libxs_matdiff_info dv;
            result = libxs_matdiff(&dv, LIBXS_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc);
            if (EXIT_SUCCESS == result) libxs_matdiff_reduce(&diff, &dv);
          }
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
            result = EXIT_FAILURE;
          }
        }
      } break;

      case 2: { // streaming A and C
        fprintf(stdout, "Streamed (A,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
#if (0 != LIBXS_PREFETCH)
            xmm(ai, b, ci,
              LIBXS_GEMM_PREFETCH_A(ai + asize), LIBXS_GEMM_PREFETCH_B(b),
              LIBXS_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(ai, b, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */
      case 3: { // indirect A and C
        fprintf(stdout, "Indirect (A,C)...\n");
        for (libxs_blasint i = 0; i < s; ++i) {
          a_array[i] = a + static_cast<size_t>(asize) * helper.shuffle(i);
          b_array[i] = b;
          c_array[i] = d + static_cast<size_t>(csize) * i;
        }
        const libxs_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
          USEOMP(libxs_gemm_batch)(LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), &transa, &transb, m, n, k,
            &alpha, &a_array[0], &lda, &ptrsize, &b_array[0], &ldb, &ptrsize, &beta, &c_array[0], &ldc, &ptrsize,
            0/*index_stride*/, 0/*index_base*/, s, 0/*batchcheck*/);
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == (benchmark & 1) && 0 != check) { /* Gold result is available */
          libxs_matdiff_info diff;
          libxs_matdiff_clear(&diff);
          for (libxs_blasint h = 0; h < s; ++h) {
            const OTYPE *const u = c + static_cast<size_t>(csize) * h, *const v = c_array[h];
            libxs_matdiff_info dv;
            result = libxs_matdiff(&dv, LIBXS_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc);
            if (EXIT_SUCCESS == result) libxs_matdiff_reduce(&diff, &dv);
          }
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
            result = EXIT_FAILURE;
          }
        }
      } break;

      case 4: { // streaming B and C
        fprintf(stdout, "Streamed (B,C)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            const ITYPE *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
            OTYPE *const ci = c + static_cast<size_t>(csize) * i;
#if (0 != LIBXS_PREFETCH)
            xmm(a, bi, ci,
              LIBXS_GEMM_PREFETCH_A(a), LIBXS_GEMM_PREFETCH_B(bi + bsize),
              LIBXS_GEMM_PREFETCH_C(ci + csize));
#else
            xmm(a, bi, ci);
#endif
          }
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */
      case 5: { // indirect B and C
        fprintf(stdout, "Indirect (B,C)...\n");
        for (libxs_blasint i = 0; i < s; ++i) {
          a_array[i] = a;
          b_array[i] = b + static_cast<size_t>(bsize) * helper.shuffle(i);
          c_array[i] = d + static_cast<size_t>(csize) * i;
        }
        const libxs_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
          USEOMP(libxs_gemm_batch)(LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), &transa, &transb, m, n, k,
            &alpha, &a_array[0], &lda, &ptrsize, &b_array[0], &ldb, &ptrsize, &beta, &c_array[0], &ldc, &ptrsize,
            0/*index_stride*/, 0/*index_base*/, s, 0/*batchcheck*/);
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == (benchmark & 1) && 0 != check) { /* Gold result is available */
          libxs_matdiff_info diff;
          libxs_matdiff_clear(&diff);
          for (libxs_blasint h = 0; h < s; ++h) {
            const OTYPE *const u = c + static_cast<size_t>(csize) * h, *const v = c_array[h];
            libxs_matdiff_info dv;
            result = libxs_matdiff(&dv, LIBXS_DATATYPE(OTYPE), m, n, u, v, &ldc, &ldc);
            if (EXIT_SUCCESS == result) libxs_matdiff_reduce(&diff, &dv);
          }
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
            result = EXIT_FAILURE;
          }
        }
      } break;

      case 6: { // streaming A and B
        fprintf(stdout, "Streamed (A,B)...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            libxs_blasint j = 0;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            if (0 == check) j = omp_get_thread_num() * chunksize * csize;
#endif
            const ITYPE *const ai = a + static_cast<size_t>(asize) * helper.shuffle(i), *const bi = b + static_cast<size_t>(bsize) * helper.shuffle(i);
#if (0 != LIBXS_PREFETCH)
            xmm(ai, bi, c + j,
              LIBXS_GEMM_PREFETCH_A(ai + asize),
              LIBXS_GEMM_PREFETCH_B(bi + bsize),
              LIBXS_GEMM_PREFETCH_C(c + j));
#else
            xmm(ai, bi, c + j);
#endif
          }
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */
      case 7: { // indirect A and B
        fprintf(stdout, "Indirect (A,B)...\n");
#if defined(_OPENMP)
#       pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          a_array[i] = a + static_cast<size_t>(asize) * helper.shuffle(i);
          b_array[i] = b + static_cast<size_t>(bsize) * helper.shuffle(i);
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          if (0 == check) {
            c_array[i] = d + static_cast<size_t>(csize) * chunksize * omp_get_thread_num();
          }
          else
#endif
          c_array[i] = d;
        }
        const libxs_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
          USEOMP(libxs_gemm_batch)(LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), &transa, &transb, m, n, k,
            &alpha, &a_array[0], &lda, &ptrsize, &b_array[0], &ldb, &ptrsize, &beta, &c_array[0], &ldc, &ptrsize,
            0/*index_stride*/, 0/*index_base*/, 0 == check ? -s : s, 0/*batchcheck*/);
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
          fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == (benchmark & 1) && 0 != check) { /* Gold result is available */
          libxs_matdiff_info diff;
          result = libxs_matdiff(&diff, LIBXS_DATATYPE(OTYPE), m, n, c, d, &ldc, &ldc);
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
            result = EXIT_FAILURE;
          }
        }
      } break;

      case 8: { // cached
        fprintf(stdout, "Cached...\n");
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
          for (libxs_blasint i = 0; i < s; ++i) {
            libxs_blasint j = 0;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
            if (0 == check) j = omp_get_thread_num() * chunksize * csize;
#endif
#if (0 != LIBXS_PREFETCH)
            xmm(a, b, c + j,
              LIBXS_GEMM_PREFETCH_A(a),
              LIBXS_GEMM_PREFETCH_B(b),
              LIBXS_GEMM_PREFETCH_C(c + j));
#else
            xmm(a, b, c + j);
#endif
          }
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
      } /* fallthrough */
      case 9: { // indirect cached
        fprintf(stdout, "Indirect cached...\n");
#if defined(_OPENMP)
#       pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          a_array[i] = a; b_array[i] = b;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          if (0 == check) {
            c_array[i] = d + static_cast<size_t>(csize) * chunksize * omp_get_thread_num();
          }
          else
#endif
          c_array[i] = d;
        }
        const libxs_blasint ptrsize = sizeof(void*);
        const unsigned long long start = libxs_timer_tick();
        for (libxs_blasint r = 0; r < nrepeat; ++r) {
          USEOMP(libxs_gemm_batch)(LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), &transa, &transb, m, n, k,
            &alpha, &a_array[0], &lda, &ptrsize, &b_array[0], &ldb, &ptrsize, &beta, &c_array[0], &ldc, &ptrsize,
            0/*index_stride*/, 0/*index_base*/, 0 == check ? -s : s, 0/*batchcheck*/);
        }
        const unsigned long long ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
        const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
        if (0 < duration && 0 != ncycles) {
          fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
          fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        }
        fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
        if (0 == (benchmark & 1) && 0 != check) { /* Gold result is available */
          libxs_matdiff_info diff;
          result = libxs_matdiff(&diff, LIBXS_DATATYPE(OTYPE), m, n, c, d, &ldc, &ldc);
          fprintf(stdout, "\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
          if (check < diff.l2_rel) {
            fprintf(stderr, "FAILED.\n");
            result = EXIT_FAILURE;
          }
        }
      } break;
      default: throw "invalid case selected!";
      } /*switch*/
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

