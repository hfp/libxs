/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs/libxs_timer.h>
#include <libxs/libxs_rng.h>
#if defined(__LIBXSTREAM)
# include <libxstream/libxstream_opencl.h>
#endif

/**
 * Weak references: gemm-blas.x links without the Ozaki library,
 * so these symbols may be undefined. CHECK should not be used
 * with gemm-blas.x (the variables resolve to zero-address).
 */
LIBXS_PRAGMA_WEAK(gemm_original)
LIBXS_PRAGMA_WEAK(ozaki_verbose)
LIBXS_PRAGMA_WEAK(gemm_diff)
LIBXS_PRAGMA_WEAK(GEMM_REAL)
LIBXS_PRAGMA_WEAK(zgemm_reference)


static double gemm_duration(double* times, int nrepeat, double total);
static void* gemm_host_malloc(size_t nbytes, int hostmem);
static void gemm_host_free(void* ptr, int hostmem);


int main(int argc, char* argv[])
{
  const char* const nrepeat_env = getenv("NREPEAT");
  /* GEMM_HOSTMEM=1 uses the offload library's page-locked allocator instead
   * of malloc, separating the device-reach cost from the pinned-memory cost.
   * Default is 0: a real dgemm caller cannot choose the allocator, so 0
   * matches a drop-in replacement and 1 is the achievable ceiling. */
  const char* const env_hostmem = getenv("GEMM_HOSTMEM");
  const int hostmem = (NULL != env_hostmem && 0 != *env_hostmem) ? atoi(env_hostmem) : 0;
  const char* const env_check = getenv("CHECK");
  const char* const env_evil = getenv("EVIL");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const int evil_raw = (NULL != env_evil && 0 != *env_evil) ? atoi(env_evil) : 0;
  const int evil = evil_raw < 0 ? -evil_raw : evil_raw;
  const int evil_perelement = (evil_raw < 0);
  /**
   * TAME is EVIL's sibling on the other axis and with the opposite sense: EVIL widens the
   * exponent range, which is what makes a decomposition need more of everything, while TAME
   * narrows the significand, which is what makes it need less. TAME=n keeps n mantissa bits
   * and clears the rest, so TAME=24 is data promoted from single precision, TAME=1 is
   * powers of two, and dyadic fractions and few-significant-digit measurements sit between.
   * Zero (the default) leaves the operands untouched.
   *
   * It is a property of real inputs rather than a way to cheat: clearing bits that are
   * already zero is what an implementation is entitled to exploit, so a run at TAME=n and a
   * run at TAME=0 answer different questions and both are exact.
   */
  const char* const env_tame = getenv("TAME");
  const int tame = (NULL != env_tame && 0 != *env_tame) ? atoi(env_tame) : 0;
  /**
   * GRADE applies the componentwise criterion of the graded BLAS accuracy tests,
   * |fl(AB) - AB| <= f(n) * u * (|alpha||A||B| + |beta||C|), with f(n) linear in n.
   * It answers a different question than CHECK: CHECK compares one scalar against a
   * fixed threshold (1e-10 for double), which is four to six orders looser than what
   * these schemes deliver and therefore passes results that are lossy but not broken.
   * The bound here scales with the data, so it needs no per-case tuning, and rsq cannot
   * substitute for either: it is 1 - SS_res/SS_tot and saturates at 1 unless the output
   * degenerates. Costs one more reference GEMM and one m-by-n buffer.
   */
  const char* const env_grade = getenv("GRADE");
  /* Negative reports the grade without gating: a deliberately trimmed run gives up precision
   * on purpose, so the criterion it would be held to is not the one it claims. */
  const int grade = (NULL != env_grade && 0 != *env_grade) ? atoi(env_grade) : 0;
  double grade_max = -1.0;
  const int nrep = (NULL == nrepeat_env ? 3 : atoi(nrepeat_env));
  const int nrepeat = (0 < nrep ? nrep : 1);
  GEMM_INT_TYPE m = (1 < argc ? atoi(argv[1]) : 257);
  GEMM_INT_TYPE n = (2 < argc ? atoi(argv[2]) : m);
  GEMM_INT_TYPE k = (3 < argc ? atoi(argv[3]) : m);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  GEMM_REAL_TYPE alpha = (6 < argc ? atof(argv[6]) : 1);
  GEMM_REAL_TYPE beta = (7 < argc ? atof(argv[7]) : 1);
  GEMM_INT_TYPE lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? m : k));
  GEMM_INT_TYPE ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? k : n));
  GEMM_INT_TYPE ldc = (10 < argc ? atoi(argv[10]) : m);
  char transa = (0 == ta ? 'N' : 'T'), transb = (0 == tb ? 'N' : 'T');
  const GEMM_REAL_TYPE scale = (1 < nrepeat ? (1.0 / nrepeat) : 1);
  int result = EXIT_SUCCESS, file_input = 0, i;
#if defined(GEMM_COMPLEX)
  int complex_input = 1;
#else
  int complex_input = 0;
#endif
  GEMM_REAL_TYPE complex_alpha[2] = { 0 }, complex_beta[2] = { 0 };
  GEMM_REAL_TYPE *a = NULL, *b = NULL, *c = NULL, *c_ref = NULL, *c_bnd = NULL;
  GEMM_INT_TYPE a_rows, a_cols, b_rows, b_cols;
  size_t nc = 1;
  libxs_matdiff_t diff;

  libxs_init();
  libxs_matdiff_clear(&diff); /* diff.r reports whether the reference ran */

#if defined(GEMM_COMPLEX)
  /* Complex mode: alpha and beta are [real, imag] pairs */
  complex_alpha[0] = alpha;
  complex_alpha[1] = 0.0;
  complex_beta[0] = beta;
  complex_beta[1] = 0.0;
#endif

  if (2 < argc && 0 == m) { /* Indicate filename(s) */
    GEMM_REAL_TYPE scalar[2] = { 0 };
    GEMM_INT_TYPE dim0, dim1;
    size_t ncomp = 0;
    gemm_mhd_settings_t settings_a;
    if (EXIT_SUCCESS == gemm_mhd_read(argv[1], &dim0, &dim1, &transa, &lda, scalar, &ncomp, &settings_a, NULL)) {
      /* MHD stores physical layout: trans='N' is (m,k), trans='C'/'T' is (k,m) */
      if ('N' == transa || 'n' == transa) {
        m = dim0;
        if (3 >= argc) k = dim1;
        else k = atoi(argv[3]);
      }
      else {
        m = dim1;
        if (3 >= argc) k = dim0;
        else k = atoi(argv[3]);
      }
      if (4 >= argc) { /*transa from file*/
      }
      else transa = (0 == ta ? 'N' : 'T');
      if (6 >= argc) alpha = scalar[0];
      else alpha = atof(argv[6]);
      if (8 >= argc) { /*lda from file*/
      }
      else lda = atoi(argv[8]);
      if (10 >= argc) {
        ldc = (0 < settings_a.ldc) ? settings_a.ldc : m;
      }
      if (2 == ncomp) {
        complex_alpha[0] = scalar[0];
        complex_alpha[1] = scalar[1];
      }
      complex_input = (2 == ncomp);
      file_input |= 0x1;
    }
    if (0 == n) {
      size_t ncomp_b = 0;
      const int b_read = gemm_mhd_read(argv[2], &dim0, &dim1, &transb, &ldb, scalar, &ncomp_b, NULL, NULL);
      /* MHD stores physical layout: transb='N' is (k,n), transb='C'/'T' is (n,k) */
      if (EXIT_SUCCESS == b_read) {
        const GEMM_INT_TYPE bk = ('N' == transb || 'n' == transb) ? dim0 : dim1;
        const GEMM_INT_TYPE bn = ('N' == transb || 'n' == transb) ? dim1 : dim0;
        if (k == bk && ncomp_b == ncomp) {
          n = bn;
          if (5 >= argc) { /*transb from file*/
          }
          else transb = (0 == tb ? 'N' : 'T');
          if (7 >= argc) beta = scalar[0];
          else beta = atof(argv[7]);
          if (9 >= argc) { /*ldb from file*/
          }
          else ldb = atoi(argv[9]);
          if (2 == ncomp_b) {
            complex_beta[0] = scalar[0];
            complex_beta[1] = scalar[1];
          }
          file_input |= 0x2;
        }
        else {
          fprintf(stderr, "Mismatched files: A implies k=%i but B has k=%i\n", (int)k, (int)bk);
        }
      }
    }
  }

  /* Compute physical (stored) matrix dimensions. */
  a_rows = ('N' == transa || 'n' == transa) ? m : k;
  a_cols = ('N' == transa || 'n' == transa) ? k : m;
  b_rows = ('N' == transb || 'n' == transb) ? k : n;
  b_cols = ('N' == transb || 'n' == transb) ? n : k;

  if (1 > m || 1 > n || 1 > k || lda < a_rows || ldb < b_rows || ldc < m) {
    fprintf(stderr, "Invalid dimensions: m=%i n=%i k=%i lda=%i(>=%i) ldb=%i(>=%i) ldc=%i(>=%i)\n", (int)m, (int)n, (int)k, (int)lda,
      (int)a_rows, (int)ldb, (int)b_rows, (int)ldc, (int)m);
    result = EXIT_FAILURE;
  }

  /* Reals per element: the generators below must cover the whole buffer */
  nc = (0 != complex_input ? 2 : 1);

  if (EXIT_SUCCESS == result) { /* Allocate matrices */
    a = (GEMM_REAL_TYPE*)gemm_host_malloc(sizeof(GEMM_REAL_TYPE) * nc * lda * a_cols, hostmem);
    b = (GEMM_REAL_TYPE*)gemm_host_malloc(sizeof(GEMM_REAL_TYPE) * nc * ldb * b_cols, hostmem);
    c = (GEMM_REAL_TYPE*)gemm_host_malloc(sizeof(GEMM_REAL_TYPE) * nc * ldc * n, hostmem);
    c_ref = (GEMM_REAL_TYPE*)gemm_host_malloc(sizeof(GEMM_REAL_TYPE) * nc * ldc * n, hostmem);
    if (NULL != a && NULL != b && NULL != c && NULL != c_ref) {
      if (0 == file_input || 0 == beta) {
        LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, c,
          (GEMM_INT_TYPE)(nc * m), n, (GEMM_INT_TYPE)(nc * ldc), scale);
      }
      else memset(c, 0, sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
      memcpy(c_ref, c, sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
      /* |C| has to be captured here: the reference GEMM overwrites c_ref and c. */
      if (0 != grade && 0 == complex_input) {
        c_bnd = (GEMM_REAL_TYPE*)gemm_host_malloc(sizeof(GEMM_REAL_TYPE) * ldc * n, hostmem);
        if (NULL != c_bnd) {
          size_t ti;
          for (ti = 0; ti < (size_t)ldc * n; ++ti) {
            c_bnd[ti] = (GEMM_REAL_TYPE)fabs((double)c[ti]);
          }
        }
      }
    }
    else result = EXIT_FAILURE;
  }

  /* Print requested GEMM arguments (regardless of result code) */
  print_gemm(stdout, 0, &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

  /**
   * Per-element exponent spread degrades componentwise accuracy steeply, and far below the
   * significand width: measured at n=512, the grade is 233 at EVIL=-8 and 8960 at EVIL=-16,
   * against a bound of f(n)=n. The cause is the accumulated sum of partially aligned terms
   * rather than any single element leaving the window. Worth saying out loud because eps does
   * not show it, being dominated by the largest entries: at EVIL=-52 eps reads 6.9e-16 while
   * l2_rel is 3.4e-04 and the grade 4.7e+09. The exact boundary rises with n, so GRADE
   * decides the case; this only warns that the question is live.
   */
  if (0 != evil_perelement && 8 < evil) {
    fprintf(stderr, "WARNING: EVIL=-%i spreads exponents per element;"
                    " componentwise accuracy degrades steeply (check GRADE, not eps)\n", evil);
  }

  if (EXIT_SUCCESS == result) { /* Initialize A-matrix */
    if (0x1 & file_input) {
      result = gemm_mhd_read(argv[1], NULL, NULL, NULL, NULL, NULL, NULL, NULL, a);
    }
    else if (0 != evil && 0 != evil_perelement) {
      const int abs_evil = evil < 0 ? -evil : evil;
      const int sign_evil = evil < 0 ? -1 : 1;
      const size_t nelem = (size_t)lda * (size_t)a_cols;
      const size_t coprime = libxs_coprime2(nelem);
      GEMM_INT_TYPE ci, ri;
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, a,
        (GEMM_INT_TYPE)(nc * a_rows), a_cols, (GEMM_INT_TYPE)(nc * lda), scale);
      for (ci = 0; ci < a_cols; ++ci) {
        for (ri = 0; ri < a_rows; ++ri) {
          const size_t idx = (size_t)ci * lda + ri;
          const int e = sign_evil * (int)(abs_evil * (coprime * idx % nelem) / nelem);
          a[idx] = (GEMM_REAL_TYPE)ldexp((double)a[idx], e);
        }
      }
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, evil, a,
        (GEMM_INT_TYPE)(nc * a_rows), a_cols, (GEMM_INT_TYPE)(nc * lda), scale);
    }
  }

  if (EXIT_SUCCESS == result) { /* Initialize B-matrix */
    if (0x2 & file_input) {
      result = gemm_mhd_read(argv[2], NULL, NULL, NULL, NULL, NULL, NULL, NULL, b);
    }
    else if (0 != evil && 0 != evil_perelement) {
      const int abs_evil = evil < 0 ? -evil : evil;
      const int sign_evil = evil < 0 ? 1 : -1;
      const size_t nelem = (size_t)ldb * (size_t)b_cols;
      const size_t coprime = libxs_coprime2(nelem);
      GEMM_INT_TYPE ci, ri;
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, b,
        (GEMM_INT_TYPE)(nc * b_rows), b_cols, (GEMM_INT_TYPE)(nc * ldb), scale);
      for (ci = 0; ci < b_cols; ++ci) {
        for (ri = 0; ri < b_rows; ++ri) {
          const size_t idx = (size_t)ci * ldb + ri;
          const int e = sign_evil * (int)(abs_evil * (coprime * idx % nelem) / nelem);
          b[idx] = (GEMM_REAL_TYPE)ldexp((double)b[idx], e);
        }
      }
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, -evil, b,
        (GEMM_INT_TYPE)(nc * b_rows), b_cols, (GEMM_INT_TYPE)(nc * ldb), scale);
    }
  }

  /* Applied after both operands exist, so file input and every EVIL variant see it alike. */
  if (EXIT_SUCCESS == result && 0 < tame) {
    const int mant = (int)(sizeof(GEMM_REAL_TYPE) == sizeof(double) ? 53 : 24);
    const int drop = (tame < mant) ? (mant - tame) : 0;
    if (0 < drop) {
      const size_t na = (size_t)nc * lda * a_cols, nb = (size_t)nc * ldb * b_cols;
      size_t ti;
      if (sizeof(GEMM_REAL_TYPE) == sizeof(double)) {
        const unsigned long long mask = ~((1ULL << drop) - 1ULL);
        union { double d; unsigned long long u; } v;
        for (ti = 0; ti < na; ++ti) { v.d = (double)a[ti]; v.u &= mask; a[ti] = (GEMM_REAL_TYPE)v.d; }
        for (ti = 0; ti < nb; ++ti) { v.d = (double)b[ti]; v.u &= mask; b[ti] = (GEMM_REAL_TYPE)v.d; }
      }
      else {
        const unsigned int mask = ~((1U << drop) - 1U);
        union { float f; unsigned int u; } v;
        for (ti = 0; ti < na; ++ti) { v.f = (float)a[ti]; v.u &= mask; a[ti] = (GEMM_REAL_TYPE)v.f; }
        for (ti = 0; ti < nb; ++ti) { v.f = (float)b[ti]; v.u &= mask; b[ti] = (GEMM_REAL_TYPE)v.f; }
      }
    }
  }

  /* Stamped on every run: rows from different generators are otherwise indistinguishable. */
  if (EXIT_SUCCESS == result) {
    fprintf(stderr, "DATA: matrng=%i evil=%i tame=%i\n", LIBXS_MATRNG_VERSION, evil_raw, tame);
  }

  if (EXIT_SUCCESS == result) { /* Call GEMM */
    const GEMM_REAL_TYPE* const ga = (0 != complex_input) ? complex_alpha : &alpha;
    const GEMM_REAL_TYPE* const gb = (0 != complex_input) ? complex_beta : &beta;
    const double gflops = (0 != complex_input ? 8.0 : 2.0) * m * n * k * 1E-9;
    double* const times = (double*)malloc((size_t)nrepeat * sizeof(double));
    libxs_timer_tick_t start;
    double duration;
    /* Warmup: untimed call to trigger lazy initialization (JIT, etc.) */
    if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    start = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      const libxs_timer_tick_t tick = libxs_timer_tick();
      if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      if (NULL != times) times[i] = libxs_timer_duration(tick, libxs_timer_tick());
    }
    duration = gemm_duration(times, nrepeat, libxs_timer_duration(start, libxs_timer_tick()));
    printf("OZAKI GEMM: %.3f ms (%.1f GFLOPS/s)", 1E3 * duration, gflops / duration);
    /* The spread is the point of the median: it is what an unpinned run shows. */
    if (NULL != times && 1 < nrepeat) {
      printf(" [%i calls %.3f-%.3f ms]", nrepeat, 1E3 * times[0], 1E3 * times[nrepeat-1]);
    }
    printf("\n");
    free(times);
  }

  if (EXIT_SUCCESS == result) { /* Reference BLAS GEMM + diff */
    const GEMM_REAL_TYPE* const ga = (0 != complex_input) ? complex_alpha : &alpha;
    const GEMM_REAL_TYPE* const gb = (0 != complex_input) ? complex_beta : &beta;
    /* gemm_original: resolved via dlsym (LD_PRELOAD); GEMM_REAL: static --wrap */
    const gemm_function_t ref_gemm = (NULL != &gemm_original && NULL != gemm_original) ? gemm_original
                                                                                       : (NULL != &GEMM_REAL ? GEMM_REAL : NULL);
    /* ZGEMM is intercepted, so the complex reference has to be asked for by name */
    const gemm_function_t ref = (0 == complex_input) ? ref_gemm
                                                     : (NULL != &zgemm_reference ? zgemm_reference : NULL);
    if (NULL != ref) {
      const double gflops = (0 != complex_input ? 8.0 : 2.0) * m * n * k * 1E-9;
      double* const times = (double*)malloc((size_t)nrepeat * sizeof(double));
      libxs_timer_tick_t start;
      double duration;
      /* Warmup */
      ref(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
      start = libxs_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        const libxs_timer_tick_t tick = libxs_timer_tick();
        ref(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
        if (NULL != times) times[i] = libxs_timer_duration(tick, libxs_timer_tick());
      }
      /* Same statistic on both sides, or the ratio is not a comparison. */
      duration = gemm_duration(times, nrepeat, libxs_timer_duration(start, libxs_timer_tick()));
      printf("BLAS GEMM:  %.3f ms (%.1f GFLOPS/s)", 1E3 * duration, gflops / duration);
      if (NULL != times && 1 < nrepeat) {
        printf(" [%i calls %.3f-%.3f ms]", nrepeat, 1E3 * times[0], 1E3 * times[nrepeat-1]);
      }
      printf("\n");
      free(times);
      {
        const libxs_data_t dt = (0 != complex_input) ? (GEMM_IS_DOUBLE ? LIBXS_DATATYPE_C64 : LIBXS_DATATYPE_C32)
                                                     : LIBXS_DATATYPE(GEMM_REAL_TYPE);
        /**
         * One driver or the other, never both: libxs_matdiff_grade fills what libxs_matdiff
         * fills, and a second call would clear the info again, including the reference count
         * set below. A and B are dead once the reference ran, so |A| and |B| are formed in
         * place to bound the exact result by |alpha||A||B| + |beta||C|.
         */
        if (NULL != c_bnd) {
          const GEMM_REAL_TYPE absa = (GEMM_REAL_TYPE)fabs((double)alpha);
          const GEMM_REAL_TYPE absb = (GEMM_REAL_TYPE)fabs((double)beta);
          const size_t na = (size_t)lda * a_cols, nb = (size_t)ldb * b_cols;
          size_t ti;
          for (ti = 0; ti < na; ++ti) a[ti] = (GEMM_REAL_TYPE)fabs((double)a[ti]);
          for (ti = 0; ti < nb; ++ti) b[ti] = (GEMM_REAL_TYPE)fabs((double)b[ti]);
          ref(&transa, &transb, &m, &n, &k, &absa, a, &lda, b, &ldb, &absb, c_bnd, &ldc);
          result = libxs_matdiff_grade(&diff, dt, m, n, c_ref, c, c_bnd, &ldc, &ldc, &ldc);
          if (EXIT_SUCCESS == result) grade_max = diff.grade;
        }
        else result = libxs_matdiff(&diff, dt, m, n, c_ref, c, &ldc, &ldc);
      }
      if (EXIT_SUCCESS == result) {
        diff.r = nrepeat;
        print_diff(stdout, (0 != complex_input ? ZGEMM_LABEL : GEMM_LABEL), 0 /*detail*/, &diff);
      }
    }
    else { /* fallback: checksum only (no reference GEMM available) */
      const libxs_data_t dt = (0 != complex_input) ? (GEMM_IS_DOUBLE ? LIBXS_DATATYPE_C64 : LIBXS_DATATYPE_C32)
                                                   : LIBXS_DATATYPE(GEMM_REAL_TYPE);
      result = libxs_matdiff(&diff, dt, m, n, NULL /*ref*/, c /*tst*/, NULL /*ldref*/, &ldc);
      if (EXIT_SUCCESS == result) {
        printf("l1_tst=%f ncalls=%i\n", diff.l1_tst, nrepeat);
      }
    }
  }

  if (EXIT_SUCCESS == result && 0 != check) { /* Accuracy validation */
    /**
     * The outer diff is end-to-end and costs nothing: the reference it needs is
     * already run to time BLAS. The inner gemm_diff is per-call and finer, but
     * exists only under OZAKI_VERBOSE, which puts a reference GEMM inside the
     * timed loop. Whichever ran decides, and both is the stricter answer.
     */
    const int outer = (0 < diff.r), inner = (NULL != &gemm_diff && 0 < gemm_diff.r);
    const double eps_outer = (0 != outer ? libxs_matdiff_epsilon(&diff) : 0);
    const double eps_inner = (0 != inner ? libxs_matdiff_epsilon(&gemm_diff) : 0);
    const double epsilon = LIBXS_MAX(eps_outer, eps_inner);
    const double threshold = (0 < check) ? check : (sizeof(double) == sizeof(GEMM_REAL_TYPE) ? 1.0E-10 : 1.0E-3);
    if (0 == outer && 0 == inner) { /* a check with nothing to measure is not a pass */
      fprintf(stderr, "CHECK: no reference available\n");
      result = EXIT_FAILURE;
    }
    else if (threshold < epsilon) {
      fprintf(stderr, "CHECK: eps=%g exceeds threshold=%g\n", epsilon, threshold);
      result = EXIT_FAILURE;
    }
    else {
      fprintf(stderr, "CHECK: eps=%g (threshold=%g)\n", epsilon, threshold);
    }
  }

  /* Linear growth is the most a componentwise-stable O(n^3) product may show. The grade is
   * reported even when CHECK already failed, since that is where it says the most. */
  if (0 <= grade_max) {
    /**
     * f counts the summed terms, so it is K: the output width says nothing about how much
     * rounding a dot product accumulated, and the two coincide only for a square GEMM. The
     * constant matters as much as the slope here, because a decomposition rounds a
     * size-independent number of times on top of the accumulation: measured at TRIM=0 the
     * grade is 70 to 270 at every size (fp32 Scheme 1: 255 at K=512, 102 at K=2048, 74 at
     * K=8192, rising as 1/sqrt(K) while K falls), so a bound of K alone leaves no margin
     * below K=512 and would fail a correct result. Grade A constrains the growth, not the
     * constant, so the floor carries the part that does not grow. The constant is sized for
     * the device path: the host reconstruction grades a flat 35 to 42 at every size, while
     * the hierarchical one on the GPU is several times that, and the smallest failure worth
     * catching (a trimmed or broken result) has been thousands.
     */
    const double fn = (double)k + 1024.0;
    const int graded = (grade_max <= fn);
    fprintf(stderr, "GRADE: a=%g f(n)=%g (%s)\n", grade_max, fn,
      0 != graded ? "pass" : (0 < grade ? "FAIL" : "advisory"));
    if (0 == graded && 0 < grade) result = EXIT_FAILURE;
  }

  libxs_finalize();
  gemm_host_free(c_bnd, hostmem);
  gemm_host_free(c_ref, hostmem);
  gemm_host_free(c, hostmem);
  gemm_host_free(b, hostmem);
  gemm_host_free(a, hostmem);

  return result;
}


static void* gemm_host_malloc(size_t nbytes, int hostmem)
{
  void* result = NULL;
#if defined(__LIBXSTREAM)
  if (0 != hostmem) {
    if (0 != nbytes && EXIT_SUCCESS == libxstream_init()) {
      if (EXIT_SUCCESS != libxstream_mem_host_allocate(&result, nbytes, NULL)) result = NULL;
    }
  }
  else {
    result = malloc(nbytes);
    /* Declare the operand: the library cannot discover a caller's pointer on
     * its own, so pinning is the caller's contract (LIBXSTREAM_PIN decides
     * what happens with the range; doing nothing is a valid answer). No
     * libxstream_init here: a CPU-only run must not bring up a device. */
    if (NULL != result) {
      LIBXS_EXPECT(EXIT_SUCCESS == libxstream_mem_host_pin(result, nbytes));
    }
  }
#else
  /* Requested but unavailable is an error, not a fallback: silently using
   * malloc here would be recorded as page-locked when it is not, and the two
   * differ by an order of magnitude on a PCIe part. */
  if (0 != hostmem) {
    fprintf(stderr, "ERROR: GEMM_HOSTMEM=%i needs a LIBXSTREAM-enabled build.\n", hostmem);
  }
  else result = malloc(nbytes);
#endif
  return result;
}


static void gemm_host_free(void* ptr, int hostmem)
{
#if defined(__LIBXSTREAM)
  if (0 != hostmem) { if (NULL != ptr) LIBXS_EXPECT(EXIT_SUCCESS == libxstream_mem_host_deallocate(ptr, NULL)); }
  else {
    if (NULL != ptr) LIBXS_EXPECT(EXIT_SUCCESS == libxstream_mem_host_unpin(ptr));
    free(ptr);
  }
#else
  LIBXS_UNUSED(hostmem);
  free(ptr);
#endif
}


/**
 * Time of one call out of nrepeat: the median of the per-call durations rather
 * than the mean of their sum. A host GEMM's spread is dominated by thread
 * placement, so a single migrated call moves a mean that is then read as a rate,
 * and the figure stops being comparable with the device side - which reports a
 * per-kernel median of its own. The median needs the samples, which is why the
 * caller collects them; they are sorted in place, so times[0] and
 * times[nrepeat-1] are the extremes afterwards. A NULL times (allocation
 * failed) leaves the mean of total as the only figure available.
 */
static double gemm_duration(double* times, int nrepeat, double total)
{
  double result;
  if (NULL != times && 0 < nrepeat) {
    int i;
    for (i = 1; i < nrepeat; ++i) { /* insertion sort: nrepeat is a repetition count, i.e. tens */
      const double t = times[i];
      int j = i;
      for (; 0 < j && times[j-1] > t; --j) times[j] = times[j-1];
      times[j] = t;
    }
    result = (0 == (nrepeat % 2)) ? (0.5 * (times[nrepeat/2-1] + times[nrepeat/2])) : times[nrepeat/2];
  }
  else result = total / (0 < nrepeat ? nrepeat : 1);
  return result;
}
