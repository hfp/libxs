/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>

#if defined(_OPENMP)
# include <omp.h>
#endif


int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int request = LIBXS_MAX(1 < argc ? atoi(argv[1]) : max_nthreads, 1);
  const int nrepeat = LIBXS_MAX(2 < argc ? atoi(argv[2]) : 100000, 1);
  /* the barrier is written by every task at every rendezvous, so it is given a
     line of its own rather than sharing one with the counters below */
  LIBXS_ALIGNED(libxs_barrier_t barrier, LIBXS_ALIGNMENT);
  libxs_timer_tick_t tickw = 0, tickb = 0;
  int nthreads = 1, result = EXIT_SUCCESS;
  int* stamp = NULL;
  if (3 < argc) {
    fprintf(stderr, "Usage:\n  %s [nthreads] [nrepeat]\n", argv[0]);
  }
  else {
    /**
     * The team is asked how large it is rather than told: the barrier waits for
     * exactly the number it was initialized with, so a runtime that grants fewer
     * threads than requested would leave it waiting for a task that never comes.
     */
#if defined(_OPENMP)
#   pragma omp parallel num_threads(request)
    { if (0 == omp_get_thread_num()) nthreads = omp_get_num_threads(); }
#endif
    stamp = (int*)malloc((size_t)nthreads * sizeof(int));
    if (NULL == stamp) {
      fprintf(stderr, "Error: failed to allocate %i stamps\n", nthreads);
      result = EXIT_FAILURE;
    }
    else {
      int i, wrong = 0, stale = 0;
      for (i = 0; i < nthreads; ++i) stamp[i] = -1;
      libxs_barrier_init(&barrier, nthreads);
      printf("LIBXS: barrier over nthreads=%i nrepeat=%i\n\n",
        nthreads, nrepeat);
      tickw = libxs_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel num_threads(nthreads) reduction(+:stale)
#endif
      { /* each task stamps its own slot and then reads every slot: a task that
           left the rendezvous early finds a stamp from the round before */
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        int r, t;
        for (r = 0; r < nrepeat; ++r) {
          stamp[tid] = r;
          libxs_barrier_wait(&barrier);
          for (t = 0; t < nthreads; ++t) {
            if (stamp[t] != r) ++stale;
          }
          /* a second rendezvous before the next stamp, or a task racing ahead
             would overwrite a slot another task has not read yet */
          libxs_barrier_wait(&barrier);
        }
      }
      tickw = libxs_timer_ncycles(tickw, libxs_timer_tick());
      tickb = libxs_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel num_threads(nthreads) reduction(+:wrong)
#endif
      { /* the root publishes a value that changes every round, so a task that
           reads the slot of the round before differs rather than coincides */
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        int r;
        for (r = 0; r < nrepeat; ++r) {
          if (libxs_barrier_bcast(&barrier, tid, 0, r * 7 + 1) != r * 7 + 1) {
            ++wrong;
          }
        }
      }
      tickb = libxs_timer_ncycles(tickb, libxs_timer_tick());
      { const double nw = 2.0 * nrepeat, nb = (double)nrepeat;
        printf("\twait:  %.0f ns (%.0f cycles) per rendezvous\n",
          libxs_timer_duration(0, tickw) / nw * 1e9, (double)tickw / nw);
        printf("\tbcast: %.0f ns (%.0f cycles) per rendezvous\n",
          libxs_timer_duration(0, tickb) / nb * 1e9, (double)tickb / nb);
      }
      /* a rendezvous that lets a task through early is fast and worthless, so
         the timing is only reported beside what it was measured on */
      if (0 != stale || 0 != wrong) {
        fprintf(stderr, "Error: %i stale stamps and %i wrong broadcasts\n",
          stale, wrong);
        result = EXIT_FAILURE;
      }
      free(stamp);
    }
  }
  return result;
}
