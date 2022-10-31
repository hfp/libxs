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
  const libxs_blasint isize = sizeof(item);
  const libxs_blasint size = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)isize * size);
  char* const shuf = (char*)malloc((size_t)isize * size);
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
    libxs_rng_seq(data, isize * size);

    for (; i < ntests; ++i) {
      const libxs_blasint j = (libxs_blasint)libxs_rng_u32(size);
      const libxs_blasint s = libxs_rng_u32(isize) + 1;
      libxs_blasint k = s;
      libxs_rng_seq(item, s);
      for (; k < isize; ++k) item[k] = 0;
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
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  /* check libxs_shuffle2 */
  if (EXIT_SUCCESS == result) {
    libxs_blasint i = 0;
    const char *const src = (const char*)data;
    libxs_shuffle2(shuf, src, isize, size);
    for (; i < size; ++i) {
      const size_t j = libxs_diff_n(&src[i*isize], shuf,
        LIBXS_CAST_UCHAR(isize), LIBXS_CAST_UCHAR(isize),
        (i + size / 2) % size, size);
      if ((size_t)size <= j) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }
#if 0
  /* check libxs_shuffle */
  if (EXIT_SUCCESS == result) {
    libxs_blasint i = 0;
    const char *const src = (const char*)data, *const dst = (const char*)shuf;
    libxs_shuffle(shuf, isize, size);
    for (; i < size; ++i) {
      if (0 != libxs_diff(&src[i*isize], &dst[i*isize], LIBXS_CAST_UCHAR(isize))) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }
#endif
  free(data);
  free(shuf);

  return result;
}

