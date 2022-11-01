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
  const char init[] = "The quick brown fox jumps over the lazy dog";
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  /* check if buffers are allocated (prerequisite) */
  if (EXIT_SUCCESS == result && NULL == data) result = EXIT_FAILURE;

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

  /* check libxs_shuffle */
  if (EXIT_SUCCESS == result) {
    char a[sizeof(init)], b[sizeof(init)];
    const size_t size = sizeof(init);
    size_t s = 1;
    for (; s < size; ++s) {
      size_t i = 0;
      memcpy(a, init, s); a[s] = '\0';
      memset(b, 0, s + 1);
      result = EXIT_FAILURE;
      for (; i < s; ++i) {
        libxs_shuffle(b, a, 1, s);
        if (0 != i || 2 > s || 0 != memcmp(b, init, s)) {
          libxs_shuffle(a, b, 1, s);
          if (0 == memcmp(a, init, s)) {
            result = EXIT_SUCCESS;
            break;
          }
        }
        else break;
      }
      if (EXIT_SUCCESS != result) break;
    }
  }
  free(data);

  return result;
}
