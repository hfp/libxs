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


/** Integer type used to represent tick of a high-resolution timer. */
typedef unsigned long long libxs_timer_tickint;

LIBXS_EXTERN_C typedef struct libxs_timer_info {
  int tsc;
} libxs_timer_info;

/** Query timer properties. */
LIBXS_API int libxs_get_timer_info(libxs_timer_info* info);

/**
 * Returns the current clock tick of a monotonic timer source with
 * platform-specific resolution (not necessarily CPU cycles).
 */
LIBXS_API libxs_timer_tickint libxs_timer_tick(void);

/** Returns the difference between two timer ticks (cycles); avoids potential side-effects/assumptions of LIBXS_DIFF. */
LIBXS_API_INLINE libxs_timer_tickint libxs_timer_ncycles(libxs_timer_tickint tick0, libxs_timer_tickint tick1) {
  return LIBXS_DELTA(tick0, tick1);
}

/** Returns the duration (in seconds) between two values received by libxs_timer_tick. */
LIBXS_API double libxs_timer_duration(libxs_timer_tickint tick0, libxs_timer_tickint tick1);

#endif /*LIBXS_TIMER_H*/
