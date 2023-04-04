/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE int
#endif


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main(void)
{
  const unsigned int seed = 1975, size = 2507;
  const unsigned int n512 = 512 / (8 * sizeof(ELEM_TYPE));
  unsigned int s = LIBXS_UP(size, n512), i, h1, h2;
  int result = EXIT_SUCCESS;
  const ELEM_TYPE* value;

  ELEM_TYPE *const data = (ELEM_TYPE*)libxs_malloc(sizeof(ELEM_TYPE) * s);
  if (NULL == data) s = 0;
  for (i = 0; i < s; ++i) data[i] = (ELEM_TYPE)(rand() - ((RAND_MAX) >> 1));

  h1 = libxs_crc32_u64(seed, data);
  h2 = libxs_crc32_u32(seed, data);
  h2 = libxs_crc32_u32(h2, (unsigned int*)data + 1);
  if (h1 != h2) {
    FPRINTF(stderr, "crc32_u32 or crc32_u64 is wrong\n");
    result = EXIT_FAILURE;
  }

  h1 = libxs_crc32(seed, data, sizeof(ELEM_TYPE) * s);
  h2 = seed; value = data;
  for (i = 0; i < s; i += n512) {
    h2 = libxs_crc32_u512(h2, value + i);
  }
  if (h1 != h2) {
    FPRINTF(stderr, "(crc32=%u) != (crc32_sw=%u)\n", h1, h2);
    result = EXIT_FAILURE;
  }

  h2 = h1 >> 16;
  if ((libxs_crc32_u16(h2, &h1) & 0xFFFF) !=
      (libxs_crc32_u16(h1 & 0xFFFF, &h2) & 0xFFFF))
  {
    result = EXIT_FAILURE;
  }

  h2 = libxs_crc32_u16(h2, &h1) & 0xFFFF;
  if (h2 != libxs_hash16(h1)) {
    result = EXIT_FAILURE;
  }

  if (seed != libxs_hash(NULL/*data*/, 0/*size*/, seed)) {
    result = EXIT_FAILURE;
  }

  if (0 != libxs_hash_string(NULL/*string*/)) {
    result = EXIT_FAILURE;
  }

  libxs_free(data);

  return result;
}

