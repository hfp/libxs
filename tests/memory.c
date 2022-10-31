/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>


int main(int argc, char* argv[])
{
  char item[LIBXS_DESCRIPTOR_MAXSIZE];
  const libxs_blasint elemsize = sizeof(item);
  const libxs_blasint count = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)elemsize * count);
  char *const shuf = (char*)malloc((size_t)elemsize * count);
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  /* check if buffers are allocated (prerequisite) */
  if (EXIT_SUCCESS == result && NULL == data) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == shuf) result = EXIT_FAILURE;

  /* check libxs_stristr */
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("ends with b", "Begins with b")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("in between of", "BeTwEEn")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("spr", "SPR")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr(NULL, "bb")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("aa", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr(NULL, NULL)) result = EXIT_FAILURE;

  /* check LIBXS_MEMCPY127 and libxs_diff_n */
  if (EXIT_SUCCESS == result) {
    libxs_blasint i = 0;
    libxs_rng_seq(data, elemsize * count);

    for (; i < ntests; ++i) {
      const libxs_blasint j = (libxs_blasint)libxs_rng_u32(count);
      const libxs_blasint s = libxs_rng_u32(elemsize) + 1;
      libxs_blasint k = s;
      libxs_rng_seq(item, s);
      for (; k < elemsize; ++k) item[k] = 0;
      LIBXS_MEMCPY127(data + (j * elemsize), item, elemsize);
      k = libxs_diff_n(item, data,
        (unsigned char)s, (unsigned char)elemsize,
        0, count);
      while (k < j) {
        k = libxs_diff_n(item, data,
          (unsigned char)s, (unsigned char)elemsize,
          k + 1, count);
      }
      if (k == j) {
        continue;
      }
      else {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  /* check libxs_shuffle2 */
  if (EXIT_SUCCESS == result) {
    libxs_blasint i = 0;
    libxs_shuffle2(shuf, data, elemsize, count);
    for (; i < count; ++i) {
      const unsigned int j = libxs_diff_n(&data[i*elemsize], shuf,
        LIBXS_CAST_UCHAR(elemsize), LIBXS_CAST_UCHAR(elemsize),
        (i + count / 2) % count, count);
      if ((size_t)count <= j) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  /* check libxs_shuffle */
  if (EXIT_SUCCESS == result) {
    libxs_blasint i = 0;
    libxs_shuffle(shuf, elemsize, count);
    for (; i < count; ++i) {
      if (0 != libxs_diff(&data[i*elemsize], &shuf[i*elemsize], LIBXS_CAST_UCHAR(elemsize))) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  free(data);
  free(shuf);

  return result;
}
