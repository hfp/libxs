/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs_intrinsics_x86.h>
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(_WIN32)
# include <Windows.h>
#elif defined(__GNUC__) || defined(__PGI) || defined(_CRAYC)
# include <sys/time.h>
# include <time.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(__powerpc64__)
# include <sys/platform/ppc.h>
#endif

#if defined(__powerpc64__)
# define LIBXS_TIMER_RDTSC(CYCLE) { \
    CYCLE = __ppc_get_timebase(); \
  }
#elif ((defined(__GNUC__) || defined(LIBXS_INTEL_COMPILER) || defined(__PGI)) && (64 <= (LIBXS_BITS)))
# define LIBXS_TIMER_RDTSC(CYCLE) { libxs_timer_tickint libxs_timer_rdtsc_hi_; \
    __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxs_timer_rdtsc_hi_)); \
    CYCLE |= libxs_timer_rdtsc_hi_ << 32; \
  }
#elif (defined(_rdtsc) || defined(_WIN32))
# define LIBXS_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
#endif


LIBXS_APIVAR(int internal_timer_init_rtc);


LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_rtc(void)
{
  libxs_timer_tickint result;
#if !(defined(__PGI) && defined(__cplusplus))
  int dummy = 0;
#endif
#if defined(_WIN32)
# if 1
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  result = (libxs_timer_tickint)t.QuadPart;
# else /* low resolution */
  result = (libxs_timer_tickint)GetTickCount();
# endif
#elif defined(CLOCK_MONOTONIC)
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  result = 1000000000ULL * t.tv_sec + t.tv_nsec;
#else
  struct timeval t;
  gettimeofday(&t, 0);
  result = 1000000ULL * t.tv_sec + t.tv_usec;
#endif
#if !(defined(__PGI) && defined(__cplusplus))
  LIBXS_UNUSED(dummy);
  dummy =
#endif
  LIBXS_ATOMIC_ADD_FETCH(&internal_timer_init_rtc, 1, LIBXS_ATOMIC_RELAXED);
  return result;
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
libxs_timer_tickint libxs_timer_tick(void)
{
  libxs_timer_tickint result;
#if defined(LIBXS_TIMER_RDTSC)
  LIBXS_TIMER_RDTSC(result);
#else
  result = libxs_timer_tick_rtc();
#endif
  return result;
}


LIBXS_API double libxs_timer_duration(libxs_timer_tickint tick0, libxs_timer_tickint tick1)
{
  double result = (double)LIBXS_DELTA(tick0, tick1);
#if defined(LIBXS_TIMER_RDTSC)
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT_MSG(0 != internal_timer_init_rtc, "LIBXS is not initialized");
# else
  if (0 == internal_timer_init_rtc) libxs_init();
# endif
  if (0 < libxs_timer_scale) {
    result *= libxs_timer_scale;
  }
  else
#endif
  {
#if defined(_WIN32)
# if 1
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    result /= (double)frequency.QuadPart;
# else /* low resolution */
    result *= 1E-3;
# endif
#elif defined(CLOCK_MONOTONIC)
    result *= 1E-9;
#else
    result *= 1E-6;
#endif
  }
  return result;
}


#if defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_timer_ncycles)(libxs_timer_tickint* /*ncycles*/, const libxs_timer_tickint* /*tick0*/, const libxs_timer_tickint* /*tick1*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_timer_ncycles)(libxs_timer_tickint* ncycles, const libxs_timer_tickint* tick0, const libxs_timer_tickint* tick1)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != ncycles && NULL != tick0 && NULL != tick1)
#endif
  {
    *ncycles = libxs_timer_ncycles(*tick0, *tick1);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_timer_ncycles specified!\n");
  }
#endif
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
