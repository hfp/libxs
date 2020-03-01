/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs.h>


int main(/*int argc, char* argv[]*/)
{
  int result = EXIT_SUCCESS, i;
  struct { int x, y, z; } key[] = {
    { 0, 0, 0 },
    { 0, 0, 1 },
    { 0, 1, 0 },
    { 0, 1, 1 },
    { 1, 0, 0 },
    { 1, 0, 1 },
    { 1, 1, 0 },
    { 1, 1, 1 }
  };
  const size_t key_size = sizeof(*key);
  const int n = (int)sizeof(key) / (int)key_size;
  /*const*/ char* value[] = {
    "hello", "world", "libxs",
    "hello world", "hello libxs",
    "value", "next", "last"
  };

  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_xregister(key, /*too large*/LIBXS_DESCRIPTOR_MAXSIZE + 1,
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_xregister(NULL, 16, /* invalid combination */
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_xregister(NULL, 0, /* invalid combination */
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_xregister(key, key_size, NULL, 0) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
#if (0 != LIBXS_JIT) /* registry service only with JIT */
  if (EXIT_SUCCESS == result) { /* same key but some payload (value=NULL: initialized later) */
    result = (NULL != libxs_xregister(key, key_size, NULL, strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* re-register same key with larger payload */
    result = (NULL == libxs_xregister(key, key_size,
      value[0], strlen(value[3]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release registered value */
    libxs_xrelease(key, key_size);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    result = (NULL != libxs_xregister(key + i, key_size, value[i], strlen(value[i]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const char *const v = (char*)libxs_xdispatch(key + i, key_size);
    libxs_kernel_info info;
    result = libxs_get_kernel_info(v, &info);
    if (EXIT_SUCCESS == result) {
      result = (LIBXS_KERNEL_KIND_USER == info.kind ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    if (EXIT_SUCCESS == result) {
      result = strcmp(v, value[i]);
    }
    libxs_release_kernel(v);
  }
#endif
  return result;
}

