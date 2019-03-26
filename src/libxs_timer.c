/******************************************************************************
** Copyright (c) 2009-2019, Intel Corporation                                **
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

#if (defined(__GNUC__) || defined(LIBXS_INTEL_COMPILER)) && (64 <= (LIBXS_BITS))
# define LIBXS_TIMER_RDTSC(CYCLE) { libxs_timer_tickint libxs_timer_rdtsc_hi_; \
    __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxs_timer_rdtsc_hi_)); \
    CYCLE |= libxs_timer_rdtsc_hi_ << 32; \
  }
#elif defined(_rdtsc) || defined(_WIN32)
# define LIBXS_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
#endif


LIBXS_APIVAR(int internal_timer_init_rtc);


LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_rtc(void)
{
  libxs_timer_tickint result;
#if defined(_WIN32)
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  result = (libxs_timer_tickint)t.QuadPart;
#elif defined(CLOCK_MONOTONIC)
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  result = 1000000000ULL * t.tv_sec + t.tv_nsec;
#else
  struct timeval t;
  gettimeofday(&t, 0);
  result = 1000000ULL * t.tv_sec + t.tv_usec;
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


LIBXS_API libxs_timer_tickint libxs_timer_cycles(libxs_timer_tickint tick0, libxs_timer_tickint tick1)
{
  return LIBXS_DIFF(tick0, tick1);
}


LIBXS_API double libxs_timer_duration(libxs_timer_tickint tick0, libxs_timer_tickint tick1)
{
  double result = (double)LIBXS_DIFF(tick0, tick1);
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
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    result /= (double)frequency.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    result *= 1E-9;
#else
    result *= 1E-6;
#endif
  }
  return result;
}

