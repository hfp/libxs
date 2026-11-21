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
#if !defined(LIBXS_SYRK_DIST_NP)
# define LIBXS_SYRK_DIST_NP 16 /* several panels */
#endif
#include <libxs/libxs_source.h>
#if defined(__MPI)
# include <mpi.h>
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#define N 101
#define MB 7
#define NB 5


#if defined(__MPI)
/* settings (arguments): size and blocks of C, k of a rank, ranks left out of the grid of C */
static int test_n = N, test_mb = MB, test_nb = NB, test_kbase = 37, test_nidle = 0;


/* Element (i, k) of the rank's share of A: reproducible on every rank (no communication). */
static double value(int rank, int i, int k)
{
  unsigned int h = (unsigned int)(rank * 7919 + i * 104729 + k * 1299709 + 1);
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
  return 0.25 * (double)(i - j) / test_n + 0.125;
}


/* Global index of a local index (as ScaLAPACK's INDXL2G). */
static int l2g(int il, int nb, int iproc, int isrc, int nprocs)
{
  return ((il / nb) * nprocs + (nprocs + iproc - isrc) % nprocs) * nb + il % nb;
}


static int check(libxs_data_t datatype, char uplo, double alpha, double beta, int fcomm,
  const libxs_pdist_t* cdist, int rank, int nranks, int myprow, int mypcol, int lrows, int lcols)
{
  const int upper = ('U' == uplo), lda = test_n + 3, k = klocal(rank), ldc = LIBXS_MAX(lrows, 1);
  const double tol = (LIBXS_DATATYPE_F64 == datatype ? 1E-12 : 1E-4);
  const size_t es = LIBXS_TYPESIZE(datatype), nc = (size_t)ldc * LIBXS_MAX(lcols, 1);
  void *const a = malloc((size_t)lda * LIBXS_MAX(k, 1) * es);
  void *const c = malloc(nc * es), *const c2 = malloc(nc * es);
  libxs_syrk_dist_t dist;
  int result = ((NULL != a && NULL != c && NULL != c2) ? EXIT_SUCCESS : EXIT_FAILURE);
  int i, j, x, r, run;
  if (EXIT_SUCCESS == result) { /* a rank without a part of C still compares its (unused) element */
    memset(c, 0, nc * es);
    memset(c2, 0, nc * es);
  }
  for (j = 0; j < k && EXIT_SUCCESS == result; ++j) {
    for (i = 0; i < lda; ++i) {
      const double v = (i < test_n ? value(rank, i, j) : 1E30); /* padding must not be read */
      if (LIBXS_DATATYPE_F64 == datatype) ((double*)a)[i + (size_t)j * lda] = v;
      else ((float*)a)[i + (size_t)j * lda] = (float)v;
    }
  }
  if (EXIT_SUCCESS == result
    && 0 == libxs_syrk_dispatch_dist(&dist, datatype, test_n, k, lda, ldc, cdist, fcomm, NULL))
  {
    FPRINTF(stderr, "ERROR line #%i: dispatch\n", __LINE__);
    result = EXIT_FAILURE;
  }
  for (run = 0; run < 2 && EXIT_SUCCESS == result; ++run) {
    void *const cc = (0 == run ? c : c2);
    for (j = 0; j < lcols; ++j) {
      const int jg = l2g(j, test_nb, mypcol, cdist->csrc, cdist->npcol);
      for (i = 0; i < lrows; ++i) {
        const int ig = l2g(i, test_mb, myprow, cdist->rsrc, cdist->nprow);
        const int inside = (0 != upper ? (ig <= jg) : (ig >= jg));
        const double v = ((0 != inside && 0 == beta) ? (0.0 / 0.0) : cinit(ig, jg)); /* beta=0: not read */
        if (LIBXS_DATATYPE_F64 == datatype) ((double*)cc)[i + (size_t)j * ldc] = v;
        else ((float*)cc)[i + (size_t)j * ldc] = (float)v;
      }
    }
    libxs_syrk(&dist.config, uplo, alpha, beta, a, cc);
  }
  for (j = 0; j < lcols && EXIT_SUCCESS == result; ++j) {
    const int jg = l2g(j, test_nb, mypcol, cdist->csrc, cdist->npcol);
    for (i = 0; i < lrows && EXIT_SUCCESS == result; ++i) {
      const int ig = l2g(i, test_mb, myprow, cdist->rsrc, cdist->nprow);
      const int inside = (0 != upper ? (ig <= jg) : (ig >= jg));
      const size_t off = i + (size_t)j * ldc;
      const double got = (LIBXS_DATATYPE_F64 == datatype ? ((double*)c)[off] : ((float*)c)[off]);
      double ref = cinit(ig, jg);
      if (0 != inside) {
        double sum = 0;
        for (r = 0; r < nranks; ++r) {
          for (x = 0; x < klocal(r); ++x) {
            if (LIBXS_DATATYPE_F64 == datatype) sum += value(r, ig, x) * value(r, jg, x);
            else sum += (double)(float)value(r, ig, x) * (double)(float)value(r, jg, x);
          }
        }
        ref = (0 != beta ? (beta * ref) : 0.0) + alpha * sum;
      }
      else if (LIBXS_DATATYPE_F32 == datatype) ref = (float)ref;
      if ((0 != inside && !(LIBXS_DELTA(got, ref) <= tol * (1.0 + LIBXS_ABS(ref))))
        || (0 == inside && got != ref))
      {
        FPRINTF(stderr, "ERROR line #%i: C(%i,%i) = %g instead of %g (uplo=%c beta=%g)\n",
          __LINE__, ig, jg, got, ref, uplo, beta);
        result = EXIT_FAILURE;
      }
    }
  }
  if (EXIT_SUCCESS == result && 0 != memcmp(c, c2, nc * es)) {
    FPRINTF(stderr, "ERROR line #%i: not reproducible\n", __LINE__);
    result = EXIT_FAILURE;
  }
  free(a);
  free(c);
  free(c2);
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
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const int fcomm = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
    const char uplos[] = { 'U', 'L', 'U', 'L', 'U' };
    const double betas[] = { 0.0, 0.0, 0.5, 0.5, 0.5 };
    int rank = 0, nranks = 1, ngrid, nprow, g, myprow = -1, mypcol = -1, t;
    int *ranks;
    libxs_pdist_t cdist;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (1 < argc) test_n = atoi(argv[1]);
    if (2 < argc) test_mb = atoi(argv[2]);
    if (3 < argc) test_nb = atoi(argv[3]);
    if (4 < argc) test_kbase = atoi(argv[4]);
    if (5 < argc) test_nidle = atoi(argv[5]);
    ngrid = LIBXS_MAX(nranks - LIBXS_MAX(test_nidle, 0), 1); /* the other ranks own no part of C */
    for (nprow = 1; (nprow + 1) * (nprow + 1) <= ngrid; ++nprow);
    for (; 0 != ngrid % nprow; --nprow);
    ranks = (int*)malloc(sizeof(int) * ngrid);
    if (NULL == ranks) result = EXIT_FAILURE;
    for (g = 0; g < ngrid && EXIT_SUCCESS == result; ++g) { /* permuted: grid order is not rank order */
      ranks[g] = nranks - 1 - g;
      if (rank == ranks[g]) {
        myprow = g % nprow;
        mypcol = g / nprow;
      }
    }
    cdist.m = cdist.n = test_n;
    cdist.mb = test_mb;
    cdist.nb = test_nb;
    cdist.nprow = nprow;
    cdist.npcol = ngrid / nprow;
    cdist.rsrc = 1 % cdist.nprow;
    cdist.csrc = 1 % cdist.npcol;
    cdist.ranks = ranks;
    /* every rank runs every case: a rank that stops early leaves the others in a collective call */
    for (t = 0; t < (int)sizeof(uplos) && NULL != ranks; ++t) {
      const libxs_data_t datatype = (4 == t ? LIBXS_DATATYPE_F32 : LIBXS_DATATYPE_F64);
      const int lrows = (0 <= myprow
        ? internal_libxs_syrk_numroc(test_n, test_mb, myprow, cdist.rsrc, cdist.nprow) : 0);
      const int lcols = (0 <= mypcol
        ? internal_libxs_syrk_numroc(test_n, test_nb, mypcol, cdist.csrc, cdist.npcol) : 0);
      if (EXIT_SUCCESS != check(datatype, uplos[t], 1.5, betas[t], fcomm, &cdist, rank, nranks,
        myprow, mypcol, lrows, lcols))
      {
        result = EXIT_FAILURE;
      }
    }
    if (NULL != ranks) { /* refused by all ranks if one rank's arguments are invalid (ldc) */
      libxs_syrk_dist_t dist;
      if (0 != libxs_syrk_dispatch_dist(&dist, LIBXS_DATATYPE_F64, test_n, klocal(rank),
        test_n + 3, (0 == rank ? 0 : test_n), &cdist, fcomm, NULL))
      {
        FPRINTF(stderr, "ERROR line #%i: dispatch not refused\n", __LINE__);
        result = EXIT_FAILURE;
      }
    }
    free(ranks);
    if (MPI_SUCCESS != MPI_Finalize()) result = EXIT_FAILURE;
  }
#else
  { const int ranks[] = { 0 };
    libxs_pdist_t cdist;
    libxs_syrk_dist_t dist;
    cdist.m = cdist.n = N;
    cdist.mb = MB;
    cdist.nb = NB;
    cdist.nprow = cdist.npcol = 1;
    cdist.rsrc = cdist.csrc = 0;
    cdist.ranks = ranks;
    LIBXS_UNUSED(argc);
    LIBXS_UNUSED(argv);
    if (0 != libxs_syrk_dispatch_dist(&dist, LIBXS_DATATYPE_F64, N, 1, N, N, &cdist, 0, NULL)) {
      FPRINTF(stderr, "ERROR line #%i: dispatch without MPI\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
#endif
  return result;
}
