/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_reg.h>

#include <stdio.h>
#include <string.h>


#define TEST_CHECK(EXPR) do { \
  if (!(EXPR)) { \
    fprintf(stderr, "FAIL: %s:%i (%s)\n", __FILE__, __LINE__, #EXPR); \
    return EXIT_FAILURE; \
  } \
} while(0)


/** Padded struct key: must be memset + element-wise init (as documented). */
typedef struct test_struct_key_t {
  int x;
  char tag;
  /* padding expected between tag and y on most ABIs */
  double y;
} test_struct_key_t;


static int test_null_args(void)
{ /* NULL and invalid arguments must not crash and must return NULL / failure */
  libxs_registry_t* registry = NULL;
  libxs_registry_info_t info;
  const int key = 42;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  /* set: NULL key */
  TEST_CHECK(NULL == libxs_registry_set(registry, NULL, sizeof(key), 4, "abc"));
  /* set: zero key_size */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, 0, 4, "abc"));
  /* set: key_size exceeds maximum */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, LIBXS_REGKEY_MAXSIZE + 1, 4, "abc"));
  /* set: zero value_size */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, sizeof(key), 0, NULL));
  /* set: NULL registry */
  TEST_CHECK(NULL == libxs_registry_set(NULL, &key, sizeof(key), 4, "abc"));

  /* get: NULL registry */
  TEST_CHECK(NULL == libxs_registry_get(NULL, &key, sizeof(key)));
  /* get: NULL key */
  TEST_CHECK(NULL == libxs_registry_get(registry, NULL, sizeof(key)));
  /* get: zero key_size */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, 0));
  /* get: key_size exceeds maximum */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, LIBXS_REGKEY_MAXSIZE + 1));

  /* free: NULL registry / NULL key (must not crash) */
  libxs_registry_free(NULL, &key, sizeof(key));
  libxs_registry_free(registry, NULL, sizeof(key));
  libxs_registry_free(registry, &key, 0);

  /* begin/next: NULL registry */
  TEST_CHECK(NULL == libxs_registry_begin(NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_next(NULL, NULL, NULL));

  /* info: NULL args */
  TEST_CHECK(EXIT_SUCCESS != libxs_registry_info(NULL, &info));
  TEST_CHECK(EXIT_SUCCESS != libxs_registry_info(registry, NULL));

  libxs_registry_destroy(registry);
  /* destroy NULL is safe */
  libxs_registry_destroy(NULL);
  return EXIT_SUCCESS;
}


static int test_set_get_basic(void)
{ /* register with deferred init, retrieve, re-register same size, overwrite rejection */
  libxs_registry_t* registry = NULL;
  const int key = 1;
  const char hello[] = "hello";
  const char world[] = "world";
  const char toolarge[] = "this is a much larger payload";
  char* v;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  /* deferred init: register without value, then fill in */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), sizeof(hello), NULL);
  TEST_CHECK(NULL != v);
  memcpy(v, hello, sizeof(hello));

  /* retrieve: must match the deferred value */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key));
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, hello));

  /* re-register with same-size value: overwrites in-place */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), sizeof(world), world);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, world));

  /* re-register with LARGER value: must fail (returns NULL) */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, sizeof(key), sizeof(toolarge), toolarge));

  /* original value must be unchanged after failed overwrite */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key));
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, world));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_free_and_reregister(void)
{ /* free removes entry, get returns NULL, re-register with larger value succeeds */
  libxs_registry_t* registry = NULL;
  const int key = 7;
  const char small[] = "ab";
  const char large[] = "abcdef";
  char* v;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  v = (char*)libxs_registry_set(registry, &key, sizeof(key), sizeof(small), small);
  TEST_CHECK(NULL != v);

  libxs_registry_free(registry, &key, sizeof(key));

  /* get after free must return NULL */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, sizeof(key)));

  /* double-free must not crash */
  libxs_registry_free(registry, &key, sizeof(key));

  /* re-register with larger payload succeeds (tombstone reused) */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), sizeof(large), large);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, large));

  /* retrieve confirms re-registration */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key));
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, large));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_iteration(void)
{ /* iterate over populated registry and empty registry */
  libxs_registry_t* registry = NULL;
  typedef int key_type;
  const key_type keys[] = { 10, 20, 30, 40, 50 };
  const int n = (int)(sizeof(keys) / sizeof(keys[0]));
  int visited[5];
  int i, count;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  /* empty registry: begin returns NULL */
  TEST_CHECK(NULL == libxs_registry_begin(registry, NULL));

  /* populate */
  for (i = 0; i < n; ++i) {
    int* v = (int*)libxs_registry_set(registry, &keys[i], sizeof(keys[0]),
      sizeof(int), &keys[i]);
    TEST_CHECK(NULL != v && *v == keys[i]);
  }

  /* iterate and count, verify each key appears exactly once */
  memset(visited, 0, sizeof(visited));
  { const void* regkey = NULL;
    const void* entry = libxs_registry_begin(registry, &regkey);
    count = 0;
    for (; NULL != entry; entry = libxs_registry_next(registry, entry, &regkey)) {
      const key_type k = *(const key_type*)regkey;
      int found = 0;
      for (i = 0; i < n; ++i) {
        if (keys[i] == k) { visited[i]++; found = 1; break; }
      }
      TEST_CHECK(0 != found);
      ++count;
    }
  }
  TEST_CHECK(count == n);
  for (i = 0; i < n; ++i) TEST_CHECK(1 == visited[i]);

  /* begin with NULL key-out pointer also works */
  TEST_CHECK(NULL != libxs_registry_begin(registry, NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_info(void)
{ /* check info before and after inserts, and after free */
  libxs_registry_t* registry = NULL;
  libxs_registry_info_t info;
  const int key1 = 1, key2 = 2;
  const char val[] = "data";
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(0 == info.size);
  TEST_CHECK(0 < info.capacity);
  TEST_CHECK(LIBXS_ISPOT(info.capacity));

  TEST_CHECK(NULL != libxs_registry_set(registry, &key1, sizeof(key1), sizeof(val), val));
  TEST_CHECK(NULL != libxs_registry_set(registry, &key2, sizeof(key2), sizeof(val), val));
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(2 == info.size);
  TEST_CHECK(0 < info.nbytes);

  libxs_registry_free(registry, &key1, sizeof(key1));
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(1 == info.size);

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_growth(void)
{ /* insert enough entries to trigger at least one table growth */
  libxs_registry_t* registry = NULL;
  libxs_registry_info_t info;
  const int count = LIBXS_REGISTRY_NBUCKETS * 2; /* well beyond 75% load */
  int i;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  { const size_t initial_cap = info.capacity;
    for (i = 0; i < count; ++i) {
      int* v = (int*)libxs_registry_set(registry, &i, sizeof(i), sizeof(int), &i);
      TEST_CHECK(NULL != v && *v == i);
    }
    TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
    TEST_CHECK(info.size == (size_t)count);
    TEST_CHECK(info.capacity > initial_cap); /* must have grown */
    TEST_CHECK(LIBXS_ISPOT(info.capacity));

    /* verify all entries survive the growth/rehash */
    for (i = 0; i < count; ++i) {
      const int* v = (const int*)libxs_registry_get(registry, &i, sizeof(i));
      TEST_CHECK(NULL != v && *v == i);
    }
  }
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_struct_key(void)
{ /* padded struct key: must memset then element-wise init (documented requirement) */
  libxs_registry_t* registry = NULL;
  test_struct_key_t k1, k2;
  double val = 3.14;
  double* v;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  /* correct initialization: memset + element-wise */
  memset(&k1, 0, sizeof(k1));
  k1.x = 42; k1.tag = 'A'; k1.y = 1.0;

  v = (double*)libxs_registry_set(registry, &k1, sizeof(k1), sizeof(val), &val);
  TEST_CHECK(NULL != v && *v == val);

  /* same logical key, same binary init */
  memset(&k2, 0, sizeof(k2));
  k2.x = 42; k2.tag = 'A'; k2.y = 1.0;

  v = (double*)libxs_registry_get(registry, &k2, sizeof(k2));
  TEST_CHECK(NULL != v && *v == val);

  /* different key */
  memset(&k2, 0, sizeof(k2));
  k2.x = 42; k2.tag = 'B'; k2.y = 1.0;
  TEST_CHECK(NULL == libxs_registry_get(registry, &k2, sizeof(k2)));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_tls_cache(void)
{ /* repeated get should be served from TLS cache; free invalidates cache */
  libxs_registry_t* registry = NULL;
  const int key = 99;
  const char val[] = "cached";
  char* v;
  int i;
  libxs_registry_create(&registry);
  TEST_CHECK(NULL != registry);

  v = (char*)libxs_registry_set(registry, &key, sizeof(key), sizeof(val), val);
  TEST_CHECK(NULL != v);

  /* first get populates TLS cache, second get hits it (both must return same pointer) */
  { const char* v1 = (const char*)libxs_registry_get(registry, &key, sizeof(key));
    const char* v2 = (const char*)libxs_registry_get(registry, &key, sizeof(key));
    TEST_CHECK(NULL != v1 && NULL != v2);
    TEST_CHECK(v1 == v2); /* same pointer */
    TEST_CHECK(0 == strcmp(v1, val));
  }

  /* many repeated gets must all succeed (hammer cache path) */
  for (i = 0; i < 1000; ++i) {
    TEST_CHECK(NULL != libxs_registry_get(registry, &key, sizeof(key)));
  }

  /* free invalidates cache; subsequent get must return NULL */
  libxs_registry_free(registry, &key, sizeof(key));
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, sizeof(key)));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_multiple_registries(void)
{ /* two independent registries with same keys must not interfere */
  libxs_registry_t *r1 = NULL, *r2 = NULL;
  const int key = 1;
  const int v1 = 100, v2 = 200;
  int* p;
  libxs_registry_create(&r1);
  libxs_registry_create(&r2);
  TEST_CHECK(NULL != r1 && NULL != r2);

  p = (int*)libxs_registry_set(r1, &key, sizeof(key), sizeof(int), &v1);
  TEST_CHECK(NULL != p && *p == v1);
  p = (int*)libxs_registry_set(r2, &key, sizeof(key), sizeof(int), &v2);
  TEST_CHECK(NULL != p && *p == v2);

  /* get from each registry returns its own value */
  p = (int*)libxs_registry_get(r1, &key, sizeof(key));
  TEST_CHECK(NULL != p && *p == v1);
  p = (int*)libxs_registry_get(r2, &key, sizeof(key));
  TEST_CHECK(NULL != p && *p == v2);

  /* destroy one, other is unaffected */
  libxs_registry_destroy(r1);
  p = (int*)libxs_registry_get(r2, &key, sizeof(key));
  TEST_CHECK(NULL != p && *p == v2);

  libxs_registry_destroy(r2);
  return EXIT_SUCCESS;
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  if (EXIT_SUCCESS == result) result = test_null_args();
  if (EXIT_SUCCESS == result) result = test_set_get_basic();
  if (EXIT_SUCCESS == result) result = test_free_and_reregister();
  if (EXIT_SUCCESS == result) result = test_iteration();
  if (EXIT_SUCCESS == result) result = test_info();
  if (EXIT_SUCCESS == result) result = test_growth();
  if (EXIT_SUCCESS == result) result = test_struct_key();
  if (EXIT_SUCCESS == result) result = test_tls_cache();
  if (EXIT_SUCCESS == result) result = test_multiple_registries();

  return result;
}
