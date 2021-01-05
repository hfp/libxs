/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>


int main(/*int argc, char* argv[]*/)
{
  char item[LIBXS_DESCRIPTOR_MAXSIZE];
  const libxs_blasint isize = sizeof(item);
  const libxs_blasint size = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)isize * size);
  libxs_blasint i, j, k, s;

  if (NULL != libxs_stristr("ends with b", "Begins with b")) return EXIT_FAILURE;
  if (NULL == libxs_stristr("in between of", "BeTwEEn")) return EXIT_FAILURE;
  if (NULL == libxs_stristr("spr", "SPR")) return EXIT_FAILURE;
  if (NULL != libxs_stristr(NULL, "bb")) return EXIT_FAILURE;
  if (NULL != libxs_stristr("aa", NULL)) return EXIT_FAILURE;
  if (NULL != libxs_stristr(NULL, NULL)) return EXIT_FAILURE;

  if (NULL == data) return EXIT_FAILURE;
  libxs_rng_seq(data, isize * size);

  for (i = 0; i < ntests; ++i) {
    j = (libxs_blasint)libxs_rng_u32(size);
    s = libxs_rng_u32(isize) + 1;
    libxs_rng_seq(item, s);
    for (k = s; k < isize; ++k) item[k] = 0;
    LIBXS_MEMCPY127(data + (j * isize), item, isize);
    k = libxs_diff_n(item, data,
      (unsigned char)s, (unsigned char)isize,
      0, size);
    while (k < j) {
      k = libxs_diff_n(item, data,
        (unsigned char)s, (unsigned char)isize,
        k + 1, size);
    }
    if (k == j) {
      continue;
    }
    else {
      free(data); return EXIT_FAILURE;
    }
  }
  free(data);

  return EXIT_SUCCESS;
}

