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
LIBXS_EXTERN_C int headeronly_aux(void)
{
  int result = EXIT_SUCCESS;
  /* exercise hash, RNG, and math from this (potentially C++) TU */
  { const unsigned long long hs = libxs_hash_string("headeronly");
    if (0 == hs) result = EXIT_FAILURE;
  }
  { libxs_rng_set_seed(42);
    { const unsigned int r = libxs_rng_u32(100);
      if (100 <= r) result = EXIT_FAILURE;
    }
    { const double f = libxs_rng_f64();
      if (0 > f || 1 < f) result = EXIT_FAILURE;
    }
  }
  { const unsigned int s = libxs_isqrt_u32(49);
    if (7 != s) result = EXIT_FAILURE;
  }
  { const unsigned char d = libxs_diff("AB", "AB", 2);
    if (0 != d) result = EXIT_FAILURE;
  }
  return result;
}
