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

  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    result = libxs_xregister(key + i, key_size, value[i], strlen(value[i]) + 1);
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
  }

  return result;
}

