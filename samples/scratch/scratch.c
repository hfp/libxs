/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_malloc.h>
#include <libxs_timer.h>

#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__TBB)
# include <tbb/scalable_allocator.h>
#endif

#if defined(__TBB)
# define MALLOC(SIZE) scalable_malloc(SIZE)
# define FREE(PTR) scalable_free(PTR)
#elif defined(_OPENMP) && defined(LIBXS_INTEL_COMPILER) && (1901 > LIBXS_INTEL_COMPILER) && 0
# define MALLOC(SIZE) kmp_malloc(SIZE)
# define FREE(PTR) kmp_free(PTR)
#elif defined(LIBXS_PLATFORM_X86) && 0
# define MALLOC(SIZE) _mm_malloc(SIZE, LIBXS_ALIGNMENT)
# define FREE(PTR) _mm_free(PTR)
#elif 1
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(PTR) free(PTR)
#endif

#if !defined(MAX_MALLOC_MB)
# define MAX_MALLOC_MB 100
#endif
#if !defined(MAX_MALLOC_N)
# define MAX_MALLOC_N 24
#endif


int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int ncycles = LIBXS_MAX(1 < argc ? atoi(argv[1]) : 100, 1);
  const int max_nactive = LIBXS_CLMP(2 < argc ? atoi(argv[2]) : 4, 1, MAX_MALLOC_N);
  const int nthreads = LIBXS_CLMP(3 < argc ? atoi(argv[3]) : 1, 1, max_nthreads);
  const char *const env_check = getenv("CHECK");
  const int check = (NULL == env_check || 0 == *env_check) ? 0 : atoi(env_check);
  unsigned int nallocs = 0, nerrors0 = 0, nerrors1 = 0;
  libxs_timer_tick_t d0 = 0, d1 = 0;
  libxs_malloc_pool_info_t info;
  libxs_malloc_pool_t *pool;
  int r[MAX_MALLOC_N], i;
  int max_size = 0;
  int scratch = 0;

  /* generate set of random numbers for parallel region */
  for (i = 0; i < (MAX_MALLOC_N); ++i) r[i] = rand();

  libxs_init();

  /* count number of calls according to randomized scheme */
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i%(MAX_MALLOC_N)] % max_nactive + 1;
    int mbytes = 0, j;
    for (j = 0; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      mbytes += (r[k] % (MAX_MALLOC_MB) + 1);
    }
    max_size = LIBXS_MAX(max_size, mbytes);
    nallocs += count;
  }
  assert(0 != nallocs);

  fprintf(stdout, "Running %i cycles with max. %i malloc+free (%u calls) using %i thread%s...\n",
    ncycles, max_nactive, nallocs, 1 >= nthreads ? 1 : nthreads, 1 >= nthreads ? "" : "s");

  pool = libxs_malloc_pool(NULL, NULL);

  /* warm-up: one untimed cycle to fault pages and ramp CPU frequency */
  { const int count = r[0] % max_nactive + 1;
    void* p[MAX_MALLOC_N];
    int j;
    for (j = 0; j < count; ++j) {
      const int k = j % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      p[j] = libxs_malloc(pool, nbytes, 0/*auto*/);
    }
    for (j = 0; j < count; ++j) libxs_free(p[j]);
#if (defined(MALLOC) && defined(FREE))
    for (j = 0; j < count; ++j) {
      const int k = j % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      p[j] = MALLOC(nbytes);
    }
    for (j = 0; j < count; ++j) FREE(p[j]);
#endif
  }

#if (defined(MALLOC) && defined(FREE))
  /* benchmark stdlib's malloc+free (run first to avoid warm-page bias) */
#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i) reduction(+:d0,nerrors0)
#endif
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i % (MAX_MALLOC_N)] % max_nactive + 1;
    void* p[MAX_MALLOC_N];
    int j;
    assert(count <= MAX_MALLOC_N);
    for (j = 0; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      const libxs_timer_tick_t t1 = libxs_timer_tick();
      p[j] = MALLOC(nbytes);
      d0 += libxs_timer_ncycles(t1, libxs_timer_tick());
      if (NULL == p[j]) {
        ++nerrors0;
      }
      else if (0 != check) {
        memset(p[j], j, nbytes);
      }
    }
    for (j = 0; j < count; ++j) {
      const libxs_timer_tick_t t1 = libxs_timer_tick();
      FREE(p[j]);
      d0 += libxs_timer_ncycles(t1, libxs_timer_tick());
    }
  }
#endif /*(defined(MALLOC) && defined(FREE))*/

  /* benchmark libxs scratch pool malloc+free */
#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i) reduction(+:d1,nerrors1)
#endif
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i%(MAX_MALLOC_N)] % max_nactive + 1;
    void* p[MAX_MALLOC_N];
    int j = 0;
    assert(count <= max_nactive);
    for (; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      const libxs_timer_tick_t t1 = libxs_timer_tick();
      p[j] = libxs_malloc(pool, nbytes, 0/*auto*/);
      d1 += libxs_timer_ncycles(t1, libxs_timer_tick());
      if (NULL == p[j]) {
        ++nerrors1;
      }
      else if (0 != check) {
        memset(p[j], j, nbytes);
      }
    }
    for (j = 0; j < count; ++j) {
      const libxs_timer_tick_t t1 = libxs_timer_tick();
      libxs_free(p[j]);
      d1 += libxs_timer_ncycles(t1, libxs_timer_tick());
    }
  }

  if (EXIT_SUCCESS == libxs_malloc_pool_info(pool, &info) && 0 < info.peak) {
    scratch = (int)LIBXS_UPDIV(info.peak, (size_t)1 << 20);
    fprintf(stdout, "\nScratch: peak_mb=%i mallocs=%lu\n",
      scratch, (unsigned long int)info.nmallocs);
  }
  libxs_free_pool(pool);
  pool = NULL;

  if (0 != d0 && 0 != d1 && 0 < nallocs) {
    const double dcalls = libxs_timer_duration(0, d0);
    const double dalloc = libxs_timer_duration(0, d1);
    const double scratch_freq = 1E-3 * nallocs / dalloc;
    const double malloc_freq = 1E-3 * nallocs / dcalls;
    const double speedup = scratch_freq / malloc_freq;
    fprintf(stdout, "\tlibxs malloc+free calls/s: %.1f kHz\n", scratch_freq);
    fprintf(stdout, "Malloc: %i MB\n", max_size);
    fprintf(stdout, "\tstd.malloc+free calls/s: %.1f kHz\n", malloc_freq);
    if (0 < scratch) {
      fprintf(stdout, "Fair (size vs. speed): %.1fx\n", max_size * speedup / scratch);
    }
    fprintf(stdout, "Scratch Speedup: %.1fx\n", speedup);
  }

  libxs_finalize();

  if (0 != nerrors0 || 0 != nerrors1) {
    fprintf(stdout, "FAILED (errors: malloc=%u libxs=%u)\n", nerrors0, nerrors1);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
