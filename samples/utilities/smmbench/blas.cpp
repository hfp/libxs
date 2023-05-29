/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
    (defined(LIBXS_PLATFORM_X86))
# include <mkl_service.h>
# include <mkl.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
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

#if (LIBXS_EQUAL(ITYPE, float) || LIBXS_EQUAL(ITYPE, double)) \
  && !defined(MKL_DIRECT_CALL_SEQ) && !defined(MKL_DIRECT_CALL)
LIBXS_BLAS_SYMBOL_DECL(ITYPE, gemm)
#endif


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  try {
#if defined(__BLAS) && (0 == __BLAS)
    LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
    throw "LAPACK/BLAS library must be available for this sample code!";
#else /* BLAS available */
    const libxs_blasint benchmark = (1 < argc ? std::atoi(argv[1]) : 0);
    LIBXS_BLAS_CONST libxs_blasint m = (2 < argc ? std::atoi(argv[2]) : 23);
    LIBXS_BLAS_CONST libxs_blasint k = (4 < argc ? std::atoi(argv[4]) : m);
    LIBXS_BLAS_CONST libxs_blasint n = (3 < argc ? std::atoi(argv[3]) : k);
    const libxs_blasint q = (5 < argc ? std::atoi(argv[5]) : 0/*auto*/);
    const libxs_blasint nrepeat = ((6 < argc && 0 < std::atoi(argv[6]))
      ? std::atoi(argv[6]) : (0 >= q ? 13 : 1));

    LIBXS_BLAS_CONST libxs_blasint lda = m, ldb = k, ldc = m;
    LIBXS_BLAS_CONST char transa = 'N', transb = 'N';
    LIBXS_BLAS_CONST OTYPE alpha = 1, beta = 1;

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
#if defined(_OPENMP) && !defined(_DEBUG)
    const char *const env_check = getenv("CHECK");
    const int check = (NULL == env_check ? 0 : atoi(env_check));
#elif (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
      (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
    /*const*/ int check = 1;
#else
    const int check = 0;
#endif
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
        , m_size(static_cast<size_t>(size_)), m_shuffle(libxs_coprime2(static_cast<unsigned int>(size_)))
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

#if defined(MKL_ENABLE_AVX512)
    mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    // initialize LIBXS
    libxs_init();

    fprintf(stdout, "m=%lli n=%lli k=%lli size=%lli memory=%.1f MB (input=%s output=%s)\n\n",
      static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(s),
      1.0 * (s * ((static_cast<size_t>(asize) + bsize) * sizeof(ITYPE) + csize * sizeof(OTYPE))) / (1ULL << 20),
      LIBXS_TYPENAME(ITYPE), LIBXS_TYPENAME(OTYPE));

    // LAPACK/BLAS3 (warm-up BLAS Library)
#if defined(_OPENMP)
#     pragma omp parallel for schedule(static)
#endif
    for (libxs_blasint i = 0; i < s; ++i) {
      LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
        &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
          &beta, c + static_cast<size_t>(csize) * i, &ldc);
    }

#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
    std::vector<const ITYPE*> va_array(static_cast<size_t>(s)), vb_array(static_cast<size_t>(s));
    std::vector<OTYPE*> vc_array(static_cast<size_t>(s));
    const ITYPE* *const a_array = &va_array[0];
    const ITYPE* *const b_array = &vb_array[0];
    OTYPE* *const c_array = &vc_array[0];
    const libxs_blasint group_count = 1;
    for (libxs_blasint i = 0; i < s; ++i) { // setup batched (A,B,C)
      a_array[i] = a + static_cast<size_t>(asize) * helper.shuffle(i);
      b_array[i] = b + static_cast<size_t>(bsize) * helper.shuffle(i);
      c_array[i] = d + static_cast<size_t>(csize) * i;
    }
    // additional warm-up (also to eventually match the Gold result)
    LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
      &alpha, &a_array[0], &lda, &b_array[0], &ldb,
        &beta, &c_array[0], &ldc, &group_count, &s);
#endif

    switch (benchmark) {
    case 0: { // batched
      fprintf(stdout, "Batched (A,B,C)...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
            &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
              &beta, c + static_cast<size_t>(csize) * i, &ldc);
        }
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
      const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
      if (0 < duration && 0 != ncycles) {
        fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * bwsize / (duration * (1ULL << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
    } /* fallthrough */
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
    case 1: { // batched indirect
      fprintf(stdout, "Indirect (A,B,C)...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
        LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
          &alpha, &a_array[0], &lda, &b_array[0], &ldb,
            &beta, &c_array[0], &ldc, &group_count, &s);
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
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
    }
#endif
    break;
    case 2: { // streaming A and C
      fprintf(stdout, "Streamed (A,C)...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
            &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b, &ldb,
              &beta, c + static_cast<size_t>(csize) * i, &ldc);
        }
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
      const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
      if (0 < duration && 0 != ncycles) {
        fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - bsize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
    } /* fallthrough */
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
    case 3: { // indirect A and C
      fprintf(stdout, "Indirect (A,C)...\n");
      for (libxs_blasint i = 0; i < s; ++i) {
        a_array[i] = a + static_cast<size_t>(asize) * helper.shuffle(i);
        b_array[i] = b;
        c_array[i] = d + static_cast<size_t>(csize) * i;
      }
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
        LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
          &alpha, &a_array[0], &lda, &b_array[0], &ldb,
            &beta, &c_array[0], &ldc, &group_count, &s);
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
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
    }
#endif
    break;
    case 4: { // streaming B and C
      fprintf(stdout, "Streamed (B,C)...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
            &alpha, a, &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
              &beta, c + static_cast<size_t>(csize) * i, &ldc);
        }
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
      const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
      if (0 < duration && 0 != ncycles) {
        fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - asize * sizeof(ITYPE)) / (duration * (1ULL << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
    } /* fallthrough */
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
    case 5: { // indirect B and C
      fprintf(stdout, "Indirect (B,C)...\n");
      for (libxs_blasint i = 0; i < s; ++i) {
        a_array[i] = a;
        b_array[i] = b + static_cast<size_t>(bsize) * helper.shuffle(i);
        c_array[i] = d + static_cast<size_t>(csize) * i;
      }
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
        LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
          &alpha, &a_array[0], &lda, &b_array[0], &ldb,
            &beta, &c_array[0], &ldc, &group_count, &s);
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
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
    }
#endif
    break;
    case 6: { // streaming A and B
      fprintf(stdout, "Streamed (A,B)...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          libxs_blasint j = 0;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          if (0 == check) j = omp_get_thread_num() * chunksize * csize;
#endif
          LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
            &alpha, a + static_cast<size_t>(asize) * helper.shuffle(i), &lda, b + static_cast<size_t>(bsize) * helper.shuffle(i), &ldb,
              &beta, c + j, &ldc);
        }
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
      const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
      if (0 < duration && 0 != ncycles) {
        fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
        fprintf(stdout, "\tbandwidth: %.1f GB/s\n", s * (bwsize - sizeof(OTYPE) * csize * 2) / (duration * (1ULL << 30)));
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
    } /* fallthrough */
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
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
#if defined(_OPENMP)
      omp_set_num_threads(0 == check ? nthreads : 1);
#endif
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
        LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
          &alpha, &a_array[0], &lda, &b_array[0], &ldb,
            &beta, &c_array[0], &ldc, &group_count, &s);
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
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
    }
#endif
    break;
    case 8: { // cached
      fprintf(stdout, "Cached...\n");
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#         pragma omp parallel for num_threads(0 == check ? nthreads : 1) schedule(static)
#endif
        for (libxs_blasint i = 0; i < s; ++i) {
          libxs_blasint j = 0;
#if defined(_OPENMP) /* attempt to write to disjunct cachelines */
          if (0 == check) j = omp_get_thread_num() * chunksize * csize;
#endif
          LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm)(&transa, &transb, &m, &n, &k,
            &alpha, a, &lda, b, &ldb, &beta, c + j, &ldc);
        }
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
      const double duration = libxs_timer_duration(0, ncycles) / nrepeat;
      if (0 < duration && 0 != ncycles) {
        fprintf(stdout, "\tpseudo-perf.: %.1f %s/cycle\n", (2.0 * k - 1.0) * (static_cast<double>(s) * m * n) / ncycles, ops);
        fprintf(stdout, "\tperformance: %.1f G%s/s\n", gflops / duration, ops);
      }
      fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
    } /* fallthrough */
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
  (defined(LIBXS_MKL_VERSION2) && (LIBXS_VERSION2(11, 3) <= LIBXS_MKL_VERSION2))
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
#if defined(_OPENMP)
      omp_set_num_threads(0 == check ? nthreads : 1);
#endif
      const libxs_timer_tickint start = libxs_timer_tick();
      for (libxs_blasint r = 0; r < nrepeat; ++r) {
        LIBXS_BLAS_FUNCTION(ITYPE,OTYPE,gemm_batch)(&transa, &transb, &m, &n, &k,
          &alpha, &a_array[0], &lda, &b_array[0], &ldb,
            &beta, &c_array[0], &ldc, &group_count, &s);
      }
      const libxs_timer_tickint ncycles = libxs_timer_ncycles(start, libxs_timer_tick());
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
    }
#endif
    break;
    default: throw "invalid case selected!";
    } /*switch*/
    // finalize LIBXS
    libxs_finalize();
    fprintf(stdout, "Finished\n");
#endif
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
