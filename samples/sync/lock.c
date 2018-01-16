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


libxs_timer_tickint work(libxs_timer_tickint start, libxs_timer_tickint duration);
libxs_timer_tickint work(libxs_timer_tickint start, libxs_timer_tickint duration)
{
  const libxs_timer_tickint end = start + duration;
  libxs_timer_tickint tick = start;
  do {
    libxs_timer_tickint i, s = 0;
    for (i = 0; i < ((end - tick) / 4); ++i) s += i;
    tick = libxs_timer_tick();
  }
  while(tick < end);
  return tick;
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
  const int work_r = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 100, 1);
  const int work_w = LIBXS_MAX(4 < argc ? atoi(argv[4]) : (10 * work_r), 1);
  const int nrepeat = LIBXS_MAX(5 < argc ? atoi(argv[5]) : 10000, 1);
  const int nw = 0 < wratioperc ? (100 / wratioperc) : (nrepeat + 1);

  /* declare attribute and lock */
  LIBXS_LOCK_ATTR_TYPE(LOCK_KIND) attr;
  LIBXS_LOCK_TYPE(LOCK_KIND) lock;

  /* initialize attribute and lock */
  LIBXS_LOCK_ATTR_INIT(LOCK_KIND, &attr);
  LIBXS_LOCK_INIT(LOCK_KIND, &lock, &attr);
  LIBXS_LOCK_ATTR_DESTROY(LOCK_KIND, &attr);

  assert(0 < (nthreads * nrepeat));
  fprintf(stdout, "Latency and throughput for nthreads=%i wratio=%i%% work_r=%i work_w=%i nrepeat=%i\n",
    nthreads, wratioperc, work_r, work_w, nrepeat);

  { /* measure non-contended latency of RO-lock */
    libxs_timer_tickint latency = 0;
    double duration;
    int i;
    for (i = 0; i < nrepeat / 4; ++i) {
      const libxs_timer_tickint tick = libxs_timer_tick();
      LIBXS_LOCK_ACQREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_RELREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_RELREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_RELREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQREAD(LOCK_KIND, &lock);
      LIBXS_LOCK_RELREAD(LOCK_KIND, &lock);
      latency += libxs_timer_diff(tick, libxs_timer_tick());
    }
    duration = libxs_timer_duration(0, latency);
    if (0 < duration) {
      printf("\tro-latency: %.0f ns (%.0f MHz)\n",
        duration * 1e9 / nrepeat,
        nrepeat / (1e6 * duration));
    }
  }

  { /* measure non-contended latency of RW-lock */
    libxs_timer_tickint latency = 0;
    double duration;
    int i;
    for (i = 0; i < nrepeat / 4; ++i) {
      const libxs_timer_tickint tick = libxs_timer_tick();
      LIBXS_LOCK_ACQUIRE(LOCK_KIND, &lock);
      LIBXS_LOCK_RELEASE(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQUIRE(LOCK_KIND, &lock);
      LIBXS_LOCK_RELEASE(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQUIRE(LOCK_KIND, &lock);
      LIBXS_LOCK_RELEASE(LOCK_KIND, &lock);
      LIBXS_LOCK_ACQUIRE(LOCK_KIND, &lock);
      LIBXS_LOCK_RELEASE(LOCK_KIND, &lock);
      latency += libxs_timer_diff(tick, libxs_timer_tick());
    }
    duration = libxs_timer_duration(0, latency);
    if (0 < duration) {
      printf("\trw-latency: %.0f ns (%.0f MHz)\n",
        duration * 1e9 / nrepeat,
        nrepeat / (1e6 * duration));
    }
  }

  { /* measure throughput */
    libxs_timer_tickint throughput = 0;
    double duration;
#if defined(_OPENMP)
#   pragma omp parallel num_threads(nthreads)
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
#     pragma omp atomic
#endif
      throughput += t1 - d;
    }
    duration = libxs_timer_duration(0, throughput);
    if (0 < duration) {
      printf("\tthroughput: %.0f us (%.0f kHz)\n",
        duration * 1e6 / (nrepeat * nthreads),
        (nrepeat * nthreads) / (1e3 * duration));
    }
  }

  LIBXS_LOCK_DESTROY(LOCK_KIND, &lock);

  return EXIT_SUCCESS;
}

