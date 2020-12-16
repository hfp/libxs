/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs_sync.h>
#include <math.h>

#if defined(_OPENMP)
# include <omp.h>
#endif


int main(int argc, char* argv[])
{
  int num_cores, threads_per_core, num_threads, num_iterations;
  libxs_timer_tickint start;
  libxs_barrier* barrier;

  if (4 < argc) {
    fprintf(stderr, "Usage:\n  %s <cores> <threads-per-core> [<iterations>]\n", argv[0]);
    return EXIT_SUCCESS;
  }

  /* parse the command line and set up the test parameters */
#if defined(_OPENMP)
  num_cores = (1 < argc ? atoi(argv[1]) : 2);
  assert(num_cores >= 1);
  threads_per_core = (2 < argc ? atoi(argv[2]) : 2);
  assert(threads_per_core >= 1);
#else
  threads_per_core = 1;
  num_cores = 1;
#endif

  num_iterations = (1 < argc ? atoi(argv[3]) : 50000);
  assert(num_iterations > 0);

  /* create a new barrier */
  barrier = libxs_barrier_create(num_cores, threads_per_core);
  assert(NULL != barrier);

  /* each thread must initialize with the barrier */
  num_threads = num_cores * threads_per_core;
#if defined(_OPENMP)
# pragma omp parallel num_threads(num_threads)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    libxs_barrier_init(barrier, tid);
  }

  start = libxs_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel num_threads(num_threads)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    int i;
    for (i = 0; i < num_iterations; ++i) {
      libxs_barrier_wait(barrier, tid);
    }
  }

  printf("libxs_barrier_wait(): %llu cycles (%d threads)\n",
    libxs_timer_ncycles(start, libxs_timer_tick()) / num_iterations,
    num_threads);

  libxs_barrier_destroy(barrier);

  return EXIT_SUCCESS;
}

