/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(MAX_MALLOC_MB)
# define MAX_MALLOC_MB 100
#endif
#if !defined(MAX_MALLOC_N)
# define MAX_MALLOC_N 24
#endif


void* malloc_offsite(size_t size);


int main(int argc, char* argv[])
{
  const int ncalls = 1000000;
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int ncycles = LIBXS_MAX(1 < argc ? atoi(argv[1]) : 100, 1);
  const int max_nallocs = LIBXS_CLMP(2 < argc ? atoi(argv[2]) : 4, 1, MAX_MALLOC_N);
  const int nthreads = LIBXS_CLMP(3 < argc ? atoi(argv[3]) : 1, 1, max_nthreads);
  unsigned int nallocs = 0, nerrors = 0;
  int r[MAX_MALLOC_N], i;

  /* generate set of random number for parallel region */
  for (i = 0; i < (MAX_MALLOC_N); ++i) r[i] = rand();

  /* count number of calls according to randomized scheme */
  for (i = 0; i < ncycles; ++i) {
    nallocs += r[i%(MAX_MALLOC_N)] % max_nallocs + 1;
  }
  assert(0 != nallocs);

  fprintf(stdout, "Running %i cycles with max. %i malloc+free (%u calls) using %i thread%s...\n",
    ncycles, max_nallocs, nallocs, 1 >= nthreads ? 1 : nthreads, 1 >= nthreads ? "" : "s");

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
  {
    const char *const longlife_env = getenv("LONGLIFE");
    const int enable_longlife = ((0 == longlife_env || 0 == *longlife_env) ? 0 : atoi(longlife_env));
    void *const longlife = (0 == enable_longlife ? 0 : malloc_offsite((MAX_MALLOC_MB) << 20));
    unsigned long long d0, d1 = 0;
    libxs_scratch_info info;

    /* run non-inline function to measure call overhead of an "empty" function */
    const unsigned long long t0 = libxs_timer_tick();
    for (i = 0; i < ncalls; ++i) {
      libxs_init(); /* subsequent calls are not doing any work */
    }
    d0 = libxs_timer_diff(t0, libxs_timer_tick());

#if defined(_OPENMP)
#   pragma omp parallel for num_threads(nthreads) private(i) default(none) shared(r) reduction(+:d1,nerrors)
#endif
    for (i = 0; i < ncycles; ++i) {
      const int count = r[i%(MAX_MALLOC_N)] % max_nallocs + 1;
      void* p[MAX_MALLOC_N];
      int j;
      assert(count <= MAX_MALLOC_N);
      for (j = 0; j < count; ++j) {
        const int k = (i * count + j) % (MAX_MALLOC_N);
        const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
        const unsigned long long t1 = libxs_timer_tick();
        p[j] = libxs_aligned_scratch(nbytes, 0/*auto*/);
        d1 += libxs_timer_diff(t1, libxs_timer_tick());
        if (0 != p[j]) {
          memset(p[j], j, nbytes);
        }
        else {
          ++nerrors;
        }
      }
      for (j = 0; j < count; ++j) {
        libxs_free(p[j]);
      }
    }
    libxs_free(longlife);

    if (0 != d0 && 0 != d1 && 0 < nallocs) {
      const double dcalls = libxs_timer_duration(0, d0);
      const double dalloc = libxs_timer_duration(0, d1);
      const double alloc_freq = 1E-3 * nallocs / dalloc;
      const double empty_freq = 1E-3 * ncalls / dcalls;
      fprintf(stdout, "\tallocation+free calls/s: %.1f kHz\n", alloc_freq);
      fprintf(stdout, "\tempty calls/s: %.1f MHz\n", 1E-3 * empty_freq);
      fprintf(stdout, "\toverhead: %.1fx\n", empty_freq / alloc_freq);
    }

    if (EXIT_SUCCESS == libxs_get_scratch_info(&info) && 0 < info.size) {
      fprintf(stdout, "\nScratch: %.f MB (mallocs=%lu, pools=%u",
        1.0 * info.size / (1ULL << 20), (unsigned long int)info.nmallocs, info.npools);
      if (1 < nthreads) fprintf(stdout, ", threads=%i)\n", nthreads); else fprintf(stdout, ")\n");
      libxs_release_scratch(); /* suppress LIBXS's termination message about scratch */
    }
  }

  if (0 == nerrors) {
    fprintf(stdout, "Finished\n");
    return EXIT_SUCCESS;
  }
  else {
    fprintf(stdout, "FAILED (%u errors)\n", nerrors);
    return EXIT_FAILURE;
  }
}


void* malloc_offsite(size_t size) { return libxs_aligned_scratch(size, 0/*auto*/); }

