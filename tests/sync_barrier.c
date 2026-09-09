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

/** Rounds of the concurrent checks. Enough to give the scheduler occasion to
 *  interleave the tasks, few enough that the test stays a test. */
#if !defined(NREPEAT)
# define NREPEAT 2000
#endif


int main(void)
{
  libxs_barrier_t barrier, single;
  int nthreads = 1, result = EXIT_SUCCESS;
  /**
   * A team that cannot arrive must not be waited for, or a mistake in the
   * caller's task count becomes a hang rather than an error. Both NULL and a
   * count of zero are therefore answered, and a broadcast still yields the
   * value the caller carried in.
   */
  libxs_barrier_init(NULL, 4);
  libxs_barrier_wait(NULL);
  if (7 != libxs_barrier_bcast(NULL, 0, 0, 7)) result = EXIT_FAILURE;
  libxs_barrier_init(&single, 0);
  if (1 != single.ntasks) result = EXIT_FAILURE;
  libxs_barrier_wait(&single);
  if (5 != libxs_barrier_bcast(&single, 0, 0, 5)) result = EXIT_FAILURE;
#if defined(_OPENMP)
  /* the team says how large it is: a barrier told a number the runtime does not
     grant would wait for a task that never arrives */
# pragma omp parallel
  { if (0 == omp_get_thread_num()) nthreads = omp_get_num_threads(); }
#endif
  if (EXIT_SUCCESS == result && 1 < nthreads) {
    int* stamp = (int*)malloc((size_t)nthreads * sizeof(int));
    if (NULL != stamp) {
      int stale = 0, wrong = 0, i;
      for (i = 0; i < nthreads; ++i) stamp[i] = -1;
      libxs_barrier_init(&barrier, nthreads);
      /**
       * What a rendezvous is for: every task stamps its own slot with the round
       * it is in, and after the rendezvous every task reads every slot. A task
       * released before the last arrival reads a stamp from the round before,
       * which no amount of luck turns into the right number.
       */
#     pragma omp parallel num_threads(nthreads) reduction(+:stale)
      { const int tid = omp_get_thread_num();
        int r, t;
        for (r = 0; r < NREPEAT; ++r) {
          stamp[tid] = r;
          libxs_barrier_wait(&barrier);
          for (t = 0; t < nthreads; ++t) {
            if (stamp[t] != r) ++stale;
          }
          /* the second rendezvous keeps a task from stamping the next round
             over a slot another task has not read yet */
          libxs_barrier_wait(&barrier);
        }
      }
      /** The broadcast carries a value that changes every round, so a task
       *  reading the publication of the round before differs rather than
       *  coincides with the one it should have read. */
#     pragma omp parallel num_threads(nthreads) reduction(+:wrong)
      { const int tid = omp_get_thread_num();
        int r;
        for (r = 0; r < NREPEAT; ++r) {
          if (libxs_barrier_bcast(&barrier, tid, 0, r * 7 + 1) != r * 7 + 1) {
            ++wrong;
          }
        }
      }
      if (0 != stale || 0 != wrong) result = EXIT_FAILURE;
      free(stamp);
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 1 < nthreads) {
    /**
     * Consecutive broadcasts must land in DIFFERENT slots, which is what lets a
     * task be released and read late without reading the next publication. The
     * slots are inspected rather than raced against, because a race that has to
     * be lost to be observed is not a test: the property is that publication n
     * and n+1 do not share storage, and that is decidable by looking.
     */
    libxs_barrier_init(&barrier, nthreads);
#   pragma omp parallel num_threads(nthreads)
    { const int tid = omp_get_thread_num();
      libxs_barrier_bcast(&barrier, tid, 0, 101);
      libxs_barrier_bcast(&barrier, tid, 0, 202);
    }
    if (101 != barrier.value[0] || 202 != barrier.value[1]) {
      result = EXIT_FAILURE;
    }
  }
  return result;
}
