/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/* measure non-contended latency of RO-lock */
#define MEASURE_LATENCY_RO(LOCK_KIND, LOCKPTR, NREPEAT, NR) do { \
  libxs_timer_tick_t latency = 0; \
  double duration; \
  int i; \
  for (i = 0; i < (NREPEAT) / 4; ++i) { \
    const libxs_timer_tick_t tick = libxs_timer_tick(); \
    LIBXS_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
    latency += libxs_timer_ncycles(tick, libxs_timer_tick()); \
  } \
  duration = libxs_timer_duration(0, latency); \
  if (0 < duration) { \
    printf("\tro-latency: %.0f ns (call/s %.0f MHz, %.0f cycles)\n", \
      duration * (NR) * 1e9, (NREPEAT) / (1e6 * duration), latency * (NR)); \
  } \
} while(0)

/* measure non-contended latency of RW-lock */
#define MEASURE_LATENCY_RW(LOCK_KIND, LOCKPTR, NREPEAT, NR) do { \
  libxs_timer_tick_t latency = 0; \
  double duration; \
  int i; \
  for (i = 0; i < (NREPEAT) / 4; ++i) { \
    const libxs_timer_tick_t tick = libxs_timer_tick(); \
    LIBXS_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
    LIBXS_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
    latency += libxs_timer_ncycles(tick, libxs_timer_tick()); \
  } \
  duration = libxs_timer_duration(0, latency); \
  if (0 < duration) { \
    printf("\trw-latency: %.0f ns (call/s %.0f MHz, %.0f cycles)\n", \
      duration * (NR) * 1e9, (NREPEAT) / (1e6 * duration), latency * (NR)); \
  } \
} while(0)

#if defined(_OPENMP)
# define MEASURE_THROUGHPUT_PARALLEL(NTHREADS) LIBXS_PRAGMA(omp parallel num_threads(NTHREADS))
# define MEASURE_THROUGHPUT_ATOMIC LIBXS_PRAGMA(omp atomic)
#else
# define MEASURE_THROUGHPUT_PARALLEL(NTHREADS)
# define MEASURE_THROUGHPUT_ATOMIC
#endif

#define MEASURE_THROUGHPUT(LOCK_KIND, LOCKPTR, NREPEAT, NTHREADS, WORK_R, WORK_W, NW, NT) do { \
  libxs_timer_tick_t throughput = 0; \
  double duration; \
  MEASURE_THROUGHPUT_PARALLEL(NTHREADS) \
  { \
    int n, nn; \
    libxs_timer_tick_t t1, t2, d = 0; \
    const libxs_timer_tick_t t0 = libxs_timer_tick(); \
    for (n = 0; n < (NREPEAT); n = nn) { \
      nn = n + 1; \
      if (0 != (nn % (NW))) { /* read */ \
        LIBXS_LOCK_ACQREAD(LOCK_KIND, LOCKPTR); \
        t1 = libxs_timer_tick(); \
        t2 = work(t1, WORK_R); \
        LIBXS_LOCK_RELREAD(LOCK_KIND, LOCKPTR); \
        d += libxs_timer_ncycles(t1, t2); \
      } \
      else { /* write */ \
        LIBXS_LOCK_ACQUIRE(LOCK_KIND, LOCKPTR); \
        t1 = libxs_timer_tick(); \
        t2 = work(t1, WORK_W); \
        LIBXS_LOCK_RELEASE(LOCK_KIND, LOCKPTR); \
        d += libxs_timer_ncycles(t1, t2); \
      } \
    } \
    t1 = libxs_timer_ncycles(t0, libxs_timer_tick()); \
    MEASURE_THROUGHPUT_ATOMIC \
    throughput += t1 - d; \
  } \
  duration = libxs_timer_duration(0, throughput); \
  if (0 < duration) { \
    const double r = 1.0 / (NT); \
    printf("\tthroughput: %.0f us (call/s %.0f kHz, %.0f cycles)\n", \
      duration * r * 1e6, (NT) / (1e3 * duration), throughput * r); \
  } \
} while(0)

#define BENCHMARK(LOCK_KIND, IMPL, NTHREADS, WORK_R, WORK_W, WRATIOPERC, NREPEAT_LAT, NREPEAT_TPT) do { \
  const int nw = 0 < (WRATIOPERC) ? (100 / (WRATIOPERC)) : ((NREPEAT_TPT) + 1); \
  const int nt = (NREPEAT_TPT) * (NTHREADS); \
  const double nr = 1.0 / (NREPEAT_LAT); \
  LIBXS_LOCK_ATTR_TYPE(LOCK_KIND) attr; \
  LIBXS_LOCK_TYPE(LOCK_KIND) lock; \
  LIBXS_ASSERT(0 < nt); \
  printf("Latency and throughput of \"%s\" (%s) for nthreads=%i wratio=%i%% work_r=%i work_w=%i nlat=%i ntpt=%i\n", \
    LIBXS_STRINGIFY(LOCK_KIND), IMPL, NTHREADS, WRATIOPERC, WORK_R, WORK_W, NREPEAT_LAT, NREPEAT_TPT); \
  LIBXS_LOCK_ATTR_INIT(LOCK_KIND, &attr); \
  LIBXS_LOCK_INIT(LOCK_KIND, &lock, &attr); \
  LIBXS_LOCK_ATTR_DESTROY(LOCK_KIND, &attr); \
  MEASURE_LATENCY_RO(LOCK_KIND, &lock, NREPEAT_LAT, nr); \
  MEASURE_LATENCY_RW(LOCK_KIND, &lock, NREPEAT_LAT, nr); \
  MEASURE_THROUGHPUT(LOCK_KIND, &lock, NREPEAT_TPT, NTHREADS, WORK_R, WORK_W, nw, nt); \
  LIBXS_LOCK_DESTROY(LOCK_KIND, &lock); \
} while(0)


libxs_timer_tick_t work(libxs_timer_tick_t start, libxs_timer_tick_t duration);
libxs_timer_tick_t work(libxs_timer_tick_t start, libxs_timer_tick_t duration)
{
  const libxs_timer_tick_t end = start + duration;
  libxs_timer_tick_t tick = start;
  do {
    libxs_timer_tick_t i, s = 0;
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
  const int nlat = LIBXS_MAX(5 < argc ? atoi(argv[5]) : 2000000, 1);
  const int ntpt = LIBXS_MAX(6 < argc ? atoi(argv[6]) : 10000, 1);

  libxs_init();
  printf("LIBXS: default lock-kind \"%s\" (%s)\n\n", LIBXS_STRINGIFY(LIBXS_LOCK_DEFAULT),
#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
    "OS-native");
#else
    "Other");
#endif

#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
  BENCHMARK(LIBXS_LOCK_SPINLOCK, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXS_LOCK_SPINLOCK, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif
#if defined(LIBXS_LOCK_SYSTEM_MUTEX)
  BENCHMARK(LIBXS_LOCK_MUTEX, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXS_LOCK_MUTEX, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif
#if defined(LIBXS_LOCK_SYSTEM_RWLOCK)
  BENCHMARK(LIBXS_LOCK_RWLOCK, "OS-native", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#else
  BENCHMARK(LIBXS_LOCK_RWLOCK, "Other", nthreads, work_r, work_w, wratioperc, nlat, ntpt);
#endif

  return EXIT_SUCCESS;
}

