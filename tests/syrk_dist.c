/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if !defined(INTERNAL_LIBXS_MPI_CHUNK)
# define INTERNAL_LIBXS_MPI_CHUNK 4096 /* small messages: exercise the splitting */
#endif
#if !defined(LIBXS_SYRK_WIDTH)
# define LIBXS_SYRK_WIDTH 32 /* several panels, tiles and windows */
#endif
#include <libxs/libxs_source.h>
#if defined(__MPI)
# include "syrk_device.h"
# include <mpi.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif
#define TEST_CHECK(CONDITION) do { \
    if (!(CONDITION)) { \
      FPRINTF(stderr, "ERROR line #%i: %s\n", __LINE__, #CONDITION); \
      result = EXIT_FAILURE; \
    } \
  } while(0)

#define N 101
#define MB 7
#define NB 5


#if defined(__MPI)
/* settings (arguments): size and blocks of C, k of a rank, ranks left out of the grid, threads, MPI level */
static int test_n = N, test_mb = MB, test_nb = NB, test_kbase = 37, test_nidle = 0, test_nthreads = 4;
static int test_level = MPI_THREAD_MULTIPLE, test_serialized;


/* Element (i, k) of a rank's share of A or B: reproducible on every rank (no communication). */
static double value(int which, int rank, int i, int k)
{
  unsigned int h = (unsigned int)which * 15485863U + (unsigned int)rank * 7919U
    + (unsigned int)i * 104729U + (unsigned int)k * 1299709U + 1U;
  h ^= h >> 13;
  h *= 0x5bd1e995U;
  h ^= h >> 15;
  return (double)(h & 0xFFFF) / 32768.0 - 1.0;
}


static int klocal(int rank)
{
  return ((2 == rank || 0 > test_kbase) ? 0 : (test_kbase + 3 * rank));
}


static double cinit(int i, int j)
{
  return 0.25 * (double)(i - j) / LIBXS_MAX(test_n, 1) + 0.125;
}


static double not_a_number(void)
{
  static volatile double zero = 0;
  return zero / zero;
}


/* Global index of a local index (as ScaLAPACK's INDXL2G). */
static int l2g(int il, int nb, int iproc, int isrc, int nprocs)
{
  return ((il / nb) * nprocs + (nprocs + iproc - isrc) % nprocs) * nb + il % nb;
}


static void put(void* x, libxs_data_t datatype, size_t i, double v)
{
  if (LIBXS_DATATYPE_F64 == datatype) ((double*)x)[i] = v;
  else ((float*)x)[i] = (float)v;
}


static double get(const void* x, libxs_data_t datatype, size_t i)
{
  return (LIBXS_DATATYPE_F64 == datatype ? ((const double*)x)[i] : (double)((const float*)x)[i]);
}


static double cast(libxs_data_t datatype, double v)
{
  return (LIBXS_DATATYPE_F64 == datatype ? v : (double)(float)v);
}


/**
 * Ways to run the tasks of one update: as a team, more tasks than threads, fewer tasks than
 * threads, and one thread running all tasks in reverse. Any of them must give the same bits.
 */
static int run(int how, const libxs_syrk_t* plan, double alpha, const void* a, const void* b,
  double beta, void* c)
{
  const int nt = test_nthreads;
  int ntasks = 1, nthreads = 1, result = EXIT_SUCCESS, i;
  if (1 == how) ntasks = nthreads = nt;
  else if (2 == how) {
    ntasks = 3 * nt + 1; nthreads = nt;
  }
  else if (3 == how && 0 != test_serialized) { /* otherwise the main thread could miss out */
    ntasks = LIBXS_MAX(nt / 2, 1); nthreads = nt;
  }
  else if (4 == how) ntasks = 7;
  if (0 == how) result = libxs_syrk(plan, alpha, a, b, beta, c);
  else {
#if defined(_OPENMP)
#   pragma omp parallel for schedule(dynamic) num_threads(nthreads)
#endif
    for (i = 0; i < ntasks; ++i) {
      const int tid = ntasks - 1 - i;
      if (EXIT_SUCCESS != libxs_syrk_task(plan, alpha, a, b, beta, c, tid, ntasks)) {
#if defined(_OPENMP)
#       pragma omp critical
#endif
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


/**
 * One variant (0-15) on every rank. From 16 on, the even ranks have a device, which declines every
 * other panel from 32 on: a device may differ between ranks.
 */
static int check(int variant, const libxs_pdist_t* dist, int rank, int nranks, int myprow, int mypcol)
{
  const int base = variant % 16, device = (16 <= variant && 0 == rank % 2);
  const libxs_data_t datatype = (8 > base ? LIBXS_DATATYPE_F64 : LIBXS_DATATYPE_F32);
  const int upper = (0 != (base & 1)), notrans = (0 == (base & 2)), twok = (0 != (base & 4));
  const double alpha = 1.5, beta = (0 != ((base + (base >> 2)) & 1) ? 0.5 : 0.0);
  const int n = test_n, k = klocal(rank);
  const int rows = (0 != notrans ? n : k), cols = (0 != notrans ? k : n);
  const int lda = rows + 3, ldb = rows + 5;
  const int lrows = (0 <= myprow
    ? internal_libxs_syrk_numroc(n, test_mb, myprow, dist->rsrc, dist->nprow) : 0);
  const int lcols = (0 <= mypcol
    ? internal_libxs_syrk_numroc(n, test_nb, mypcol, dist->csrc, dist->npcol) : 0);
  const int ldc = LIBXS_MAX(lrows, 1);
  const double tol = (LIBXS_DATATYPE_F64 == datatype ? 1E-12 : 1E-4) * (4.0 * nranks);
  const size_t es = LIBXS_TYPESIZE(datatype), nc = (size_t)ldc * LIBXS_MAX(lcols, 1);
  void *const a = malloc(((size_t)lda * cols + 1) * es), *const b = malloc(((size_t)ldb * cols + 1) * es);
  void *const c = calloc(nc, es), *const c2 = calloc(nc, es);
  const void *const pa = (0 != k ? a : NULL), *const pb = ((0 != k && 0 != twok) ? b : NULL);
  const int empty = (0 == lrows || 0 == lcols);
  libxs_syrk_shape_t shape;
  libxs_syrk_device_t ops;
  test_device_t dev;
  libxs_syrk_t plan;
  int result = EXIT_SUCCESS, i, j, x, r, how;
  test_device_init(&dev, &ops, 32 <= variant);
  TEST_CHECK(NULL != a && NULL != b && NULL != c && NULL != c2);
  for (j = 0; j < cols && EXIT_SUCCESS == result; ++j) {
    for (i = 0; i < lda; ++i) { /* padding must not be read */
      put(a, datatype, i + (size_t)j * lda, i < rows
        ? (0 != notrans ? value(1, rank, i, j) : value(1, rank, j, i)) : 1E30);
    }
    for (i = 0; i < ldb; ++i) {
      put(b, datatype, i + (size_t)j * ldb, i < rows
        ? (0 != notrans ? value(2, rank, i, j) : value(2, rank, j, i)) : 1E30);
    }
  }
  LIBXS_MEMZERO(&shape);
  shape.dist = dist;
  shape.datatype = datatype;
  shape.uplo = (char)(0 != upper ? 'U' : 'L');
  shape.trans = (char)(0 != notrans ? 'N' : 'T');
  shape.n = n; shape.k = k;
  shape.lda = lda; shape.ldb = (0 != twok ? ldb : 0); shape.ldc = ldc;
  if (0 != device) shape.device = &ops;
  if (EXIT_SUCCESS != result || 0 == libxs_syrk_dispatch(&plan, &shape)) {
    FPRINTF(stderr, "ERROR line #%i: dispatch\n", __LINE__);
    result = EXIT_FAILURE;
  }
  else {
    for (how = 0; how < 5; ++how) { /* all ranks run all of it, whatever failed */
      void *const cc = (0 == how ? c : c2);
      for (j = 0; j < lcols; ++j) {
        const int jg = l2g(j, test_nb, mypcol, dist->csrc, dist->npcol);
        for (i = 0; i < lrows; ++i) {
          const int ig = l2g(i, test_mb, myprow, dist->rsrc, dist->nprow);
          const int inside = (0 != upper ? (ig <= jg) : (ig >= jg));
          put(cc, datatype, i + (size_t)j * ldc, (0 != inside && 0 == beta) ? not_a_number() : cinit(ig, jg));
        }
      }
      { const int npanels = (0 != k ? LIBXS_UPDIV(n, plan.width) : 0), nbefore = dev.nsubmit + dev.ndecline;
        TEST_CHECK(EXIT_SUCCESS == run(how, &plan, alpha, pa, pb, beta, 0 != empty ? NULL : cc));
        if (0 < how) TEST_CHECK(0 == memcmp(c, c2, nc * es));
        if (0 != device) { /* every panel went to the device, which was used as specified */
          TEST_CHECK(npanels == dev.nsubmit + dev.ndecline - nbefore);
          TEST_CHECK(0 == dev.wrong && 0 == dev.inflight && dev.nsubmit == dev.nwait);
        }
      }
    }
    for (j = 0; j < lcols && EXIT_SUCCESS == result; ++j) {
      const int jg = l2g(j, test_nb, mypcol, dist->csrc, dist->npcol);
      for (i = 0; i < lrows && EXIT_SUCCESS == result; ++i) {
        const int ig = l2g(i, test_mb, myprow, dist->rsrc, dist->nprow);
        const int inside = (0 != upper ? (ig <= jg) : (ig >= jg));
        const double got = get(c, datatype, i + (size_t)j * ldc);
        double ref = cast(datatype, cinit(ig, jg));
        if (0 != inside) {
          double sum = 0;
          for (r = 0; r < nranks; ++r) {
            for (x = 0; x < klocal(r); ++x) {
              const double ai = cast(datatype, value(1, r, ig, x)), aj = cast(datatype, value(1, r, jg, x));
              if (0 == twok) sum += ai * aj;
              else sum += ai * cast(datatype, value(2, r, jg, x)) + cast(datatype, value(2, r, ig, x)) * aj;
            }
          }
          ref = (0 != beta ? (beta * ref) : 0.0) + alpha * sum;
        }
        if ((0 != inside && !(LIBXS_DELTA(got, ref) <= tol * (1.0 + LIBXS_ABS(ref))))
          || (0 == inside && got != ref))
        {
          FPRINTF(stderr, "ERROR line #%i: C(%i,%i) = %g instead of %g (variant %i)\n",
            __LINE__, ig, jg, got, ref, variant);
          result = EXIT_FAILURE;
        }
      }
    }
    libxs_syrk_release(&plan);
    libxs_syrk_release(&plan);
  }
  free(a); free(b); free(c); free(c2);
  return result;
}


/* The ranks agree to refuse if one of them disagrees, and an error of one leaves nobody waiting. */
static int check_verdict(const libxs_pdist_t* dist, int rank, int nranks)
{
  const int n = LIBXS_MAX(test_n, 2), ngrid = dist->nprow * dist->npcol;
  int *const ranks = (int*)malloc(sizeof(int) * ngrid);
  double *const a = (double*)calloc(n, sizeof(double)), *const c = (double*)calloc((size_t)n * n, sizeof(double));
  int result = EXIT_SUCCESS, what, i;
  TEST_CHECK(NULL != ranks && NULL != a && NULL != c);
  for (what = 0; what < 11 && EXIT_SUCCESS == result; ++what) {
    const int odd = (0 == rank); /* the rank that disagrees */
    libxs_syrk_shape_t shape;
    libxs_pdist_t layout = *dist;
    libxs_syrk_t plan;
    int differs = 1, populated;
    LIBXS_MEMZERO(&shape);
    shape.dist = &layout;
    shape.datatype = LIBXS_DATATYPE_F64;
    shape.uplo = 'L'; shape.trans = 'N';
    shape.n = n; shape.k = 1;
    shape.lda = n; shape.ldb = 0; shape.ldc = n;
    for (i = 0; i < ngrid; ++i) ranks[i] = dist->ranks[i];
    layout.ranks = ranks;
    if (1 == what && 0 != odd) shape.ldc = 0;
    else if (2 == what && 0 != odd) shape.uplo = 'U';
    else if (3 == what && 0 != odd) shape.trans = 'T';
    else if (4 == what && 0 != odd) shape.n = n - 1;
    else if (5 == what && 0 != odd) shape.datatype = LIBXS_DATATYPE_F32;
    else if (6 == what && 0 != odd) shape.ldb = n;
    else if (7 == what && 0 != odd) layout.mb = dist->mb + 1;
    else if (8 == what && 0 != odd) layout.ranks = NULL;
    else if (9 == what && 0 != odd && 1 < ngrid) {
      ranks[0] = dist->ranks[1]; ranks[1] = dist->ranks[0];
    }
    if (0 == what || 10 == what || (1 == nranks && 1 != what && 8 != what) || (9 == what && 1 == ngrid)) {
      differs = 0;
    }
    populated = libxs_syrk_dispatch(&plan, &shape);
    TEST_CHECK((0 == differs) == (0 != populated));
    if (0 != populated && (0 == what || 10 == what)) { /* a single rank also populates the others */
      if (10 == what) { /* one rank lacks A: it reports, and nobody hangs */
        const int status = libxs_syrk(&plan, 1.0, 0 != odd ? NULL : a, NULL, 0.0, c);
        TEST_CHECK((0 != odd) == (EXIT_SUCCESS != status));
      }
      TEST_CHECK(EXIT_SUCCESS == libxs_syrk(&plan, 1.0, a, NULL, 0.0, c)); /* also after an error */
    }
    libxs_syrk_release(&plan);
  }
  free(ranks); free(a); free(c);
  return result;
}
#endif


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  libxs_init();
#if defined(__MPI)
  { int provided = MPI_THREAD_SINGLE;
    if (1 < argc) test_n = atoi(argv[1]);
    if (2 < argc) test_mb = atoi(argv[2]);
    if (3 < argc) test_nb = atoi(argv[3]);
    if (4 < argc) test_kbase = atoi(argv[4]);
    if (5 < argc) test_nidle = atoi(argv[5]);
    if (6 < argc) test_nthreads = atoi(argv[6]);
    if (7 < argc && 0 == atoi(argv[7])) test_level = MPI_THREAD_FUNNELED;
    if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, test_level, &provided)) result = EXIT_FAILURE;
    test_serialized = (MPI_THREAD_SERIALIZED <= provided);
  }
  if (EXIT_SUCCESS == result) {
    int rank = 0, nranks = 1, ngrid, nprow, g, myprow = -1, mypcol = -1, variant;
    int *ranks;
    libxs_pdist_t dist;
#if defined(_OPENMP)
    omp_set_num_threads(1); /* bits are compared: no threaded BLAS in the serial calls */
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    ngrid = LIBXS_MAX(nranks - LIBXS_MAX(test_nidle, 0), 1); /* the other ranks own no part of C */
    for (nprow = 1; (nprow + 1) * (nprow + 1) <= ngrid; ++nprow);
    for (; 0 != ngrid % nprow; --nprow);
    ranks = (int*)malloc(sizeof(int) * ngrid);
    if (NULL == ranks) result = EXIT_FAILURE;
    else {
      for (g = 0; g < ngrid; ++g) { /* permuted: grid order is not rank order */
        ranks[g] = nranks - 1 - g;
        if (rank == ranks[g]) {
          myprow = g % nprow;
          mypcol = g / nprow;
        }
      }
      dist.fcomm = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
      dist.mb = test_mb;
      dist.nb = test_nb;
      dist.nprow = nprow;
      dist.npcol = ngrid / nprow;
      dist.rsrc = 1 % dist.nprow;
      dist.csrc = 1 % dist.npcol;
      dist.ranks = ranks;
      for (variant = 0; variant < 48; ++variant) { /* every rank runs every case */
        if (EXIT_SUCCESS != check(variant, &dist, rank, nranks, myprow, mypcol)) result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS != check_verdict(&dist, rank, nranks)) result = EXIT_FAILURE;
      free(ranks);
    }
    if (MPI_SUCCESS != MPI_Finalize()) result = EXIT_FAILURE;
  }
#else
  { const int ranks[] = { 0 };
    libxs_syrk_shape_t shape;
    libxs_pdist_t dist;
    libxs_syrk_t plan;
    LIBXS_MEMZERO(&shape);
    shape.dist = &dist;
    shape.datatype = LIBXS_DATATYPE_F64;
    shape.uplo = 'L'; shape.trans = 'N';
    shape.n = N; shape.k = 1;
    shape.lda = N; shape.ldb = 0; shape.ldc = N;
    dist.fcomm = 0;
    dist.mb = MB; dist.nb = NB;
    dist.nprow = dist.npcol = 1;
    dist.rsrc = dist.csrc = 0;
    dist.ranks = ranks;
    LIBXS_UNUSED(argc);
    LIBXS_UNUSED(argv);
    TEST_CHECK(0 == libxs_syrk_dispatch(&plan, &shape)); /* no MPI */
  }
#endif
  return result;
}
