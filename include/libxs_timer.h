/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_TIMER_H
#define LIBXS_TIMER_H

#include "libxs.h"


/** Integer type representing a tick from a high-resolution timer. */
typedef unsigned long long libxs_timer_tick_t;

/** Timer properties filled by libxs_timer_info. */
LIBXS_EXTERN_C typedef struct libxs_timer_info_t {
  /** Non-zero if a calibrated TSC (RDTSC/CNTVCT_EL0) is available, zero if the RTC fallback is used. */
  int tsc;
} libxs_timer_info_t;

/**
 * Query timer properties (returns EXIT_SUCCESS/EXIT_FAILURE).
 * Triggers lazy initialization of the library if not yet initialized.
 */
LIBXS_API int libxs_timer_info(libxs_timer_info_t* info);

/**
 * Returns the current tick of a monotonic timer source.
 * Uses a calibrated TSC (RDTSC on x86, CNTVCT_EL0 on AArch64, MFTB on PPC64)
 * when available, otherwise falls back to the OS real-time clock
 * (clock_gettime CLOCK_MONOTONIC, QueryPerformanceCounter, or gettimeofday).
 * The tick unit is platform-specific; use libxs_timer_duration to convert
 * a pair of ticks to seconds, or libxs_timer_ncycles for raw deltas.
 */
LIBXS_API libxs_timer_tick_t libxs_timer_tick(void);

/**
 * Returns the unsigned difference between two timer ticks (tick1 - tick0).
 * Handles wrap-around safely; use this instead of plain subtraction.
 */
LIBXS_API_INLINE libxs_timer_tick_t libxs_timer_ncycles(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1) {
  return LIBXS_DELTA(tick0, tick1);
}

/**
 * Converts a pair of ticks obtained from libxs_timer_tick into elapsed
 * wall-clock time in seconds (double precision).
 */
LIBXS_API double libxs_timer_duration(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1);

#endif /*LIBXS_TIMER_H*/
