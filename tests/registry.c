/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_reg.h>


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  typedef int key_type;
  const key_type key[] = {
    0, 0, 0,
    0, 0, 1,
    0, 1, 0,
    0, 1, 1,
    1, 0, 0,
    1, 0, 1,
    1, 1, 0,
    1, 1, 1
  };
  const char* value[] = {
    "hello", "world", "libxs",
    "hello world", "hello libxs",
    "value", "next", "last"
  };
  const size_t key_size = sizeof(key_type) * 3;
  libxs_registry_t* registry = NULL;
  const int small_key = 0, n = (int)sizeof(key) / (int)key_size;
  const char string[] = "payload";
  int i;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
  libxs_registry_create(&registry);
  if (NULL == registry) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_registry_set(registry, key, /*too large*/LIBXS_DESCRIPTOR_MAXSIZE + 1,
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_registry_set(registry, NULL, 16, /* invalid combination */
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_registry_set(registry, NULL, 0, /* invalid combination */
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxs_registry_set(registry, key, key_size,
      0, NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* register and initialize value later */
    char *const v = (char*)libxs_registry_set(registry, key, key_size, strlen(value[0]) + 1, NULL);
    strcpy(v, value[0]); /* initialize value after registration */
    result = (NULL != v ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* retrieve previously registered value */
    const char *const v = (const char*)libxs_registry_get(registry, key, key_size);
    result = ((NULL != v && 0 == strcmp(v, value[0])) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* re-register with same size of payload */
    const size_t samesize = strlen(value[0]);
    char *const v = (char*)libxs_registry_set(registry, key, key_size, samesize + 1, value[5]);
    if (NULL != v) {
      v[samesize] = '\0';
      result = (0 == strncmp(v, value[5], samesize) ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* re-register with larger payload (failure) */
    result = (NULL == libxs_registry_set(registry, key, key_size,
      strlen(value[3]) + 1, value[3]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* retrieve previously registered value */
    const char *const v = (const char*)libxs_registry_get(registry, key, key_size);
    result = ((NULL != v && 0 == strcmp(v, value[5])) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release entry (enabled for user-data) */
    libxs_registry_free(registry, key, key_size);
  }
  if (EXIT_SUCCESS == result) { /* re-register with larger payload */
    result = (NULL != libxs_registry_set(registry, key, key_size,
      strlen(value[3]) + 1, value[3]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* retrieve previously registered value */
    const char *const v = (const char*)libxs_registry_get(registry, key, key_size);
    result = ((NULL != v && 0 == strcmp(v, value[3])) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release entry (enabled for user-data) */
    libxs_registry_free(registry, key, key_size);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) { /* register all entries */
    const key_type *const ikey = key + i * 3;
    assert(0 == LIBXS_MOD2((uintptr_t)ikey, sizeof(key_type)));
    result = (NULL != libxs_registry_set(registry, ikey, key_size,
      strlen(value[i]) + 1, value[i]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) {
    const void* regkey = NULL;
    const void* regentry = libxs_registry_begin(registry, &regkey);
    /*assert(0 == LIBXS_MOD2((uintptr_t)regkey, sizeof(key_type)) || NULL == regentry);*/
    for (; NULL != regentry; regentry = libxs_registry_next(registry, regentry, &regkey)) {
      const key_type *const ikey = (const key_type*)regkey;
      const char *const ivalue = (const char*)regentry;
      /*assert(0 == LIBXS_MOD2((uintptr_t)regkey, sizeof(key_type)));*/
      result = EXIT_FAILURE;
      for (i = 0; i < n; ++i) {
        if (ikey[0/*x*/] == key[i*3+0/*x*/] && ikey[1/*y*/] == key[i*3+1/*y*/] && ikey[2/*z*/] == key[i*3+2/*z*/]) {
          result = (0 == strcmp(ivalue, value[i]) ? EXIT_SUCCESS : EXIT_FAILURE);
          break;
        }
      }
      if (EXIT_SUCCESS != result) break;
    }
  }
  if (EXIT_SUCCESS == result) { /* register small key */
    result = (NULL != libxs_registry_set(registry, &small_key, sizeof(small_key),
      sizeof(string), string) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const char *const v = (char*)libxs_registry_get(registry, key + i * 3, key_size);
    if (NULL != v) {
#if 0
      libxs_kernel_info info;
      result = libxs_get_kernel_info(v, &info);
#endif
      if (EXIT_SUCCESS == result) {
        result = strcmp(v, value[i]);
      }
      /*libxs_release_kernel(v);*/
    }
    else result = EXIT_FAILURE;
  }
#if 0
  if (EXIT_SUCCESS == result) { /* release user-entries for the sake of testing it */
    const void* regentry = libxs_registry_begin(registry, NULL);
    for (; NULL != regentry; regentry = libxs_registry_next(registry, regentry, NULL)) {
      libxs_release_kernel(regentry);
    }
  }
#endif
  libxs_registry_destroy(registry);
  return result;
}
