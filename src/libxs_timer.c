/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include "libxs_main.h"

#if !defined(LIBXS_TIMER_VERBOSE) && !defined(NDEBUG)
# if !defined(LIBXS_PLATFORM_AARCH64) || !defined(__APPLE__)
#   define LIBXS_TIMER_VERBOSE
# endif
#endif


LIBXS_API int libxs_timer_info(libxs_timer_info_t* info)
{
  int result;
  if (NULL != info) {
#if defined(LIBXS_TIMER_RDTSC)
    if (0 < libxs_timer_scale) {
      info->tsc = 1;
    }
# if !defined(LIBXS_INIT_COMPLETED)
    else if (2 > libxs_ninit) {
      libxs_init();
      if (0 < libxs_timer_scale) {
        info->tsc = 1;
      }
      else {
        info->tsc = 0;
      }
    }
# endif
    else {
      info->tsc = 0;
    }
#else
    info->tsc = 0;
#endif
    result = EXIT_SUCCESS;
  }
  else {
#if defined(LIBXS_TIMER_VERBOSE)
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument for libxs_timer_info specified!\n");
    }
#endif
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API libxs_timer_tick_t libxs_timer_tick(void)
{
  libxs_timer_tick_t result;
#if defined(LIBXS_TIMER_RDTSC)
  if (0 < libxs_timer_scale) {
    LIBXS_TIMER_RDTSC(result);
  }
# if !defined(LIBXS_INIT_COMPLETED)
  else if (2 > libxs_ninit) {
    libxs_init();
    if (0 < libxs_timer_scale) {
      LIBXS_TIMER_RDTSC(result);
    }
    else {
      result = libxs_timer_tick_rtc();
    }
  }
# endif
  else {
    result = libxs_timer_tick_rtc();
  }
#else
  result = libxs_timer_tick_rtc();
#endif
  return result;
}


LIBXS_API double libxs_timer_duration(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1)
{
  double result;
#if defined(LIBXS_TIMER_RDTSC)
  if (0 < libxs_timer_scale) {
    result = (double)LIBXS_DELTA(tick0, tick1) * libxs_timer_scale;
  }
  else
#endif
  {
    result = libxs_timer_duration_rtc(tick0, tick1);
  }
  return result;
}
