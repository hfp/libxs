/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
#include <stdio.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(LOCK_KIND)
# define LOCK_KIND LIBXS_LOCK_SPINLOCK
#endif


libxs_timer_tickint work(libxs_timer_tickint start, libxs_timer_tickint amount);
libxs_timer_tickint work(libxs_timer_tickint start, libxs_timer_tickint amount)
{
  const libxs_timer_tickint t1 = start + amount;
  libxs_timer_tickint t0 = start;
  do {
    libxs_timer_tickint i, s = 0;
    for (i = 0; i < ((t1 - t0) / 4); ++i) s += i;
    t0 = libxs_timer_tick();
  }
  while(t0 < t1);
  return t0;
}


int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int nthreads = LIBXS_MAX(1 < argc ? atoi(argv[1]) : max_nthreads, 1);
  const int wratioperc = LIBXS_CLMP(2 < argc ? atoi(argv[2]) : 5, 0, 100);
  const int work_r = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 50, 1);
  const int work_w = LIBXS_MAX(4 < argc ? atoi(argv[4]) : work_r, 1);
  const int nrepeat = LIBXS_MAX(5 < argc ? atoi(argv[5]) : 100000, 1);
  const int nw = 0 < wratioperc ? (100 / wratioperc) : (nrepeat + 1);
  libxs_timer_tickint duration = 0;

  /* declare attribute and lock */
  LIBXS_LOCK_ATTR_TYPE(LOCK_KIND) attr;
  LIBXS_LOCK_TYPE(LOCK_KIND) lock;

  /* initialize attribute and lock */
  LIBXS_LOCK_ATTR_INIT(LOCK_KIND, &attr);
  LIBXS_LOCK_INIT(LOCK_KIND, &lock, &attr);
  LIBXS_LOCK_ATTR_DESTROY(LOCK_KIND, &attr);

#if defined(_OPENMP)
# pragma omp parallel num_threads(nthreads)
#endif
  {
    int n, nn;
    libxs_timer_tickint t1, t2, d = 0;
    const libxs_timer_tickint t0 = libxs_timer_tick();
    for (n = 0; n < nrepeat; n = nn) {
      nn = n + 1;
      if (0 != (nn % nw)) { /* read */
        LIBXS_LOCK_ACQREAD(LOCK_KIND, &lock);
        t1 = libxs_timer_tick();
        t2 = work(t1, work_r);
        LIBXS_LOCK_RELREAD(LOCK_KIND, &lock);
        d += libxs_timer_diff(t1, t2);
      }
      else { /* write */
        LIBXS_LOCK_ACQUIRE(LOCK_KIND, &lock);
        t1 = libxs_timer_tick();
        t2 = work(t1, work_w);
        LIBXS_LOCK_RELEASE(LOCK_KIND, &lock);
        d += libxs_timer_diff(t1, t2);
      }
    }
    t1 = libxs_timer_diff(t0, libxs_timer_tick());
#if defined(_OPENMP)
#   pragma omp atomic
#endif
    duration += t1 - d;
  }
  LIBXS_LOCK_DESTROY(LOCK_KIND, &lock);

  assert(0 < (nthreads * nrepeat));
  printf("Duration of lock/unlock operation: %.0f us (%i threads)\n",
    libxs_timer_duration(0, duration) * 1e6 / (nrepeat * nthreads), nthreads);

  return EXIT_SUCCESS;
}

