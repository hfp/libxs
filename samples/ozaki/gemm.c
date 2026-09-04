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
  GEMM_REAL_TYPE *a = NULL, *b = NULL, *c = NULL, *c_ref = NULL;
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
    }
    else result = EXIT_FAILURE;
  }

  /* Print requested GEMM arguments (regardless of result code) */
  print_gemm(stdout, 0, &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

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
        result = libxs_matdiff(&diff, dt, m, n, c_ref, c, &ldc, &ldc);
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

  libxs_finalize();
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
