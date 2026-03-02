/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>


LIBXS_EXTERN_C int headeronly_aux(void);


int main(void)
{
  int result = EXIT_SUCCESS;
  /* exercise functions from this (C) translation unit */
  { const size_t g = libxs_gcd(12, 8);
    const size_t l = libxs_lcm(12, 8);
    if (4 != g || 24 != l) result = EXIT_FAILURE;
  }
  { const unsigned int h = libxs_hash32(0x12345678);
    if (0 == h) result = EXIT_FAILURE;
  }
  { libxs_timer_tick_t t0, t1;
    double dt;
    t0 = libxs_timer_tick();
    t1 = libxs_timer_tick();
    dt = libxs_timer_duration(t0, t1);
    if (0 > dt) result = EXIT_FAILURE;
  }
  /* exercise functions from the aux (potentially C++) translation unit */
  if (EXIT_SUCCESS != headeronly_aux()) result = EXIT_FAILURE;
  return result;
}
