/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_reg.h>

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

static size_t test_fixup_nvisits;
static int test_fixup_keysizes;


static int test_null_args(void)
{ /* NULL and invalid arguments must not crash and must return NULL / failure */
  libxs_registry_t* registry;
  libxs_registry_info_t info;
  const int key = 42;
  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  /* set: NULL key */
  TEST_CHECK(NULL == libxs_registry_set(registry, NULL, sizeof(key), "abc", 4, NULL));
  /* set: zero key_size */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, 0, "abc", 4, NULL));
  /* set: key_size exceeds maximum */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, LIBXS_REGKEY_MAXSIZE + 1, "abc", 4, NULL));
  /* set: zero value_size */
  TEST_CHECK(NULL == libxs_registry_set(registry, &key, sizeof(key), NULL, 0, NULL));
  /* set: NULL registry */
  TEST_CHECK(NULL == libxs_registry_set(NULL, &key, sizeof(key), "abc", 4, NULL));

  /* get: NULL registry */
  TEST_CHECK(NULL == libxs_registry_get(NULL, &key, sizeof(key), NULL));
  /* get: NULL key */
  TEST_CHECK(NULL == libxs_registry_get(registry, NULL, sizeof(key), NULL));
  /* get: zero key_size */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, 0, NULL));
  /* get: key_size exceeds maximum */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, LIBXS_REGKEY_MAXSIZE + 1, NULL));

  /* hash: rejected arguments yield zero rather than reading the key */
  TEST_CHECK(0 == libxs_registry_hash(NULL, &key, sizeof(key)));
  TEST_CHECK(0 == libxs_registry_hash(registry, NULL, sizeof(key)));
  TEST_CHECK(0 == libxs_registry_hash(registry, &key, 0));
  TEST_CHECK(0 == libxs_registry_hash(registry, &key, LIBXS_REGKEY_MAXSIZE + 1));

  /* free: NULL registry / NULL key (must not crash) */
  libxs_registry_remove(NULL, &key, sizeof(key), NULL);
  libxs_registry_remove(registry, NULL, sizeof(key), NULL);
  libxs_registry_remove(registry, &key, 0, NULL);

  /* begin/next: NULL registry */
  TEST_CHECK(NULL == libxs_registry_begin(NULL, NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_next(NULL, NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_begin_length(NULL, NULL, NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_next_length(NULL, NULL, NULL, NULL));

  /* info: NULL args */
  TEST_CHECK(EXIT_SUCCESS != libxs_registry_info(NULL, &info));
  TEST_CHECK(EXIT_SUCCESS != libxs_registry_info(registry, NULL));

  libxs_registry_destroy(registry);
  /* destroy NULL is safe */
  libxs_registry_destroy(NULL);
  return EXIT_SUCCESS;
}


static int test_set_get_basic(void)
{ /* register with deferred init, retrieve, re-register same size, auto-realloc larger */
  const int key = 1;
  const char hello[] = "hello";
  const char world[] = "world";
  const char toolarge[] = "this is a much larger payload";
  char* v;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  /* deferred init: register without value, then fill in */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), NULL, sizeof(hello), NULL);
  TEST_CHECK(NULL != v);
  memcpy(v, hello, sizeof(hello));

  /* retrieve: must match the deferred value */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, hello));

  /* re-register with same-size value: overwrites in-place */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), world, sizeof(world), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, world));

  /* re-register with LARGER value: auto-realloc succeeds */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), toolarge, sizeof(toolarge), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, toolarge));

  /* retrieve confirms the larger value is stored */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, toolarge));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_free_and_reregister(void)
{ /* free removes entry, get returns NULL, re-register with larger value succeeds */
  const int key = 7;
  const char small[] = "ab";
  const char large[] = "abcdef";
  char* v;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  v = (char*)libxs_registry_set(registry, &key, sizeof(key), small, sizeof(small), NULL);
  TEST_CHECK(NULL != v);

  libxs_registry_remove(registry, &key, sizeof(key), NULL);

  /* get after free must return NULL */
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, sizeof(key), NULL));

  /* double-free must not crash */
  libxs_registry_remove(registry, &key, sizeof(key), NULL);

  /* re-register with larger payload succeeds (tombstone reused) */
  v = (char*)libxs_registry_set(registry, &key, sizeof(key), large, sizeof(large), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, large));

  /* retrieve confirms re-registration */
  v = (char*)libxs_registry_get(registry, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != v);
  TEST_CHECK(0 == strcmp(v, large));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_iteration(void)
{ /* iterate over populated registry and empty registry */
  typedef int key_type;
  const key_type keys[] = { 10, 20, 30, 40, 50 };
  const int n = (int)(sizeof(keys) / sizeof(keys[0]));
  int visited[5];
  int i, count;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  /* empty registry: begin returns NULL */
  TEST_CHECK(NULL == libxs_registry_begin(registry, NULL, NULL));

  /* populate */
  for (i = 0; i < n; ++i) {
    int* v = (int*)libxs_registry_set(registry, &keys[i], sizeof(keys[0]),
      &keys[i], sizeof(int), NULL);
    TEST_CHECK(NULL != v && *v == keys[i]);
  }

  /* iterate and count, verify each key appears exactly once */
  memset(visited, 0, sizeof(visited));
  { const void* regkey = NULL;
    size_t cursor = 0;
    const void* entry = libxs_registry_begin(registry, &regkey, &cursor);
    count = 0;
    for (; NULL != entry; entry = libxs_registry_next(registry, &regkey, &cursor)) {
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
  TEST_CHECK(NULL != libxs_registry_begin(registry, NULL, NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_info(void)
{ /* check info before and after inserts, and after free */
  libxs_registry_info_t info;
  const int key1 = 1, key2 = 2;
  const char val[] = "data";
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(0 == info.size);
  TEST_CHECK(0 < info.capacity);
  TEST_CHECK(LIBXS_ISPOT(info.capacity));

  TEST_CHECK(NULL != libxs_registry_set(registry, &key1, sizeof(key1), val, sizeof(val), NULL));
  TEST_CHECK(NULL != libxs_registry_set(registry, &key2, sizeof(key2), val, sizeof(val), NULL));
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(2 == info.size);
  TEST_CHECK(0 < info.nbytes);

  libxs_registry_remove(registry, &key1, sizeof(key1), NULL);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK(1 == info.size);

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_growth(void)
{ /* insert enough entries to trigger at least one table growth */
  libxs_registry_info_t info;
  const int count = LIBXS_REGISTRY_NBUCKETS * 2; /* well beyond 75% load */
  int i;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  { const size_t initial_cap = info.capacity;
    for (i = 0; i < count; ++i) {
      int* v = (int*)libxs_registry_set(registry, &i, sizeof(i), &i, sizeof(int), NULL);
      TEST_CHECK(NULL != v && *v == i);
    }
    TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
    TEST_CHECK(info.size == (size_t)count);
    TEST_CHECK(info.capacity > initial_cap); /* must have grown */
    TEST_CHECK(LIBXS_ISPOT(info.capacity));

    /* verify all entries survive the growth/rehash */
    for (i = 0; i < count; ++i) {
      const int* v = (const int*)libxs_registry_get(registry, &i, sizeof(i), NULL);
      TEST_CHECK(NULL != v && *v == i);
    }
  }
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_struct_key(void)
{ /* padded struct key: must memset then element-wise init (documented requirement) */
  test_struct_key_t k1, k2;
  double val = 3.14;
  double* v;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  /* correct initialization: memset + element-wise */
  memset(&k1, 0, sizeof(k1));
  k1.x = 42; k1.tag = 'A'; k1.y = 1.0;

  v = (double*)libxs_registry_set(registry, &k1, sizeof(k1), &val, sizeof(val), NULL);
  TEST_CHECK(NULL != v && *v == val);

  /* same logical key, same binary init */
  memset(&k2, 0, sizeof(k2));
  k2.x = 42; k2.tag = 'A'; k2.y = 1.0;

  v = (double*)libxs_registry_get(registry, &k2, sizeof(k2), NULL);
  TEST_CHECK(NULL != v && *v == val);

  /* different key */
  memset(&k2, 0, sizeof(k2));
  k2.x = 42; k2.tag = 'B'; k2.y = 1.0;
  TEST_CHECK(NULL == libxs_registry_get(registry, &k2, sizeof(k2), NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static void test_mixed_fill_key(unsigned char* key, int key_size, int index)
{ /* deterministic content, unique per (key_size, index) */
  int i;
  key[0] = (unsigned char)index;
  if (1 < key_size) key[1] = (unsigned char)key_size;
  for (i = 2; i < key_size; ++i) key[i] = (unsigned char)(7 * i + 1);
}


static int test_mixed_key_sizes(void)
{ /* keys of different length coexist: a shorter key is not the prefix of a */
  /* longer one, and every operation is length-specific */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  libxs_registry_info_t info;
  int i, n;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  for (i = 0; i < LIBXS_REGKEY_MAXSIZE; ++i) key[i] = (unsigned char)(i + 1);

  /* one buffer registered at every length: each length is a distinct key */
  for (n = 1; n <= LIBXS_REGKEY_MAXSIZE; ++n) {
    const int* v = (const int*)libxs_registry_set(registry, key, (size_t)n,
      &n, sizeof(n), NULL);
    TEST_CHECK(NULL != v && *v == n);
  }
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK((size_t)LIBXS_REGKEY_MAXSIZE == info.size);

  /* every length resolves to the value stored for exactly that length */
  for (n = 1; n <= LIBXS_REGKEY_MAXSIZE; ++n) {
    const int* v = (const int*)libxs_registry_get(registry, key, (size_t)n, NULL);
    TEST_CHECK(NULL != v && *v == n);
    TEST_CHECK(0 != libxs_registry_has(registry, key, (size_t)n, NULL));
    TEST_CHECK(sizeof(int) == libxs_registry_value_size(registry, key, (size_t)n, NULL));
  }

  /* same length but differing last byte: another distinct key */
  key[3] = 0xFF;
  { const int val = -1;
    const int* v = (const int*)libxs_registry_set(registry, key, 4, &val, sizeof(val), NULL);
    TEST_CHECK(NULL != v && *v == val);
    /* the shorter prefix and the original 4-Byte key are unaffected */
    v = (const int*)libxs_registry_get(registry, key, 3, NULL);
    TEST_CHECK(NULL != v && 3 == *v);
    key[3] = 4;
    v = (const int*)libxs_registry_get(registry, key, 4, NULL);
    TEST_CHECK(NULL != v && 4 == *v);
  }
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  TEST_CHECK((size_t)LIBXS_REGKEY_MAXSIZE + 1 == info.size);

  /* get_copy honors the key length */
  { int out = 0;
    TEST_CHECK(0 != libxs_registry_get_copy(registry, key, 1, &out, sizeof(out), NULL));
    TEST_CHECK(1 == out);
    TEST_CHECK(0 != libxs_registry_get_copy(registry, key, 7, &out, sizeof(out), NULL));
    TEST_CHECK(7 == out);
  }

  /* extract removes only the entry matching that length */
  { int out = 0;
    TEST_CHECK(0 != libxs_registry_extract(registry, key, 5, &out, sizeof(out), NULL));
    TEST_CHECK(5 == out);
    TEST_CHECK(0 == libxs_registry_has(registry, key, 5, NULL));
    TEST_CHECK(0 != libxs_registry_has(registry, key, 4, NULL));
    TEST_CHECK(0 != libxs_registry_has(registry, key, 6, NULL));
  }

  /* remove is length-specific as well */
  libxs_registry_remove(registry, key, 1, NULL);
  TEST_CHECK(0 == libxs_registry_has(registry, key, 1, NULL));
  TEST_CHECK(0 != libxs_registry_has(registry, key, 2, NULL));

  /* re-registering one length does not disturb the neighbouring lengths */
  { const int val = 4242;
    TEST_CHECK(NULL != libxs_registry_set(registry, key, 8, &val, sizeof(val), NULL));
    for (n = 2; n <= LIBXS_REGKEY_MAXSIZE; ++n) {
      const int* v = (const int*)libxs_registry_get(registry, key, (size_t)n, NULL);
      if (5 == n) {
        TEST_CHECK(NULL == v);
      }
      else {
        TEST_CHECK(NULL != v && *v == (8 == n ? val : n));
      }
    }
  }

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_mixed_key_sizes_growth(void)
{ /* rehash re-probes with the per-entry key length: no entry may be lost */
  /* or aliased when lengths are mixed across a growing table */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  libxs_registry_info_t info;
  int s, j, count = 0;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  memset(key, 0, sizeof(key));

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
  { const size_t initial_cap = info.capacity;
    for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
      const int m = (1 < s) ? 20 : 8; /* 1-Byte keys: only 256 exist */
      for (j = 0; j < m; ++j) {
        const int val = s * 1000 + j;
        const int* v;
        test_mixed_fill_key(key, s, j);
        v = (const int*)libxs_registry_set(registry, key, (size_t)s,
          &val, sizeof(val), NULL);
        TEST_CHECK(NULL != v && *v == val);
        ++count;
      }
    }
    TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(registry, &info));
    TEST_CHECK(info.size == (size_t)count); /* no key aliased another */
    TEST_CHECK(info.capacity > initial_cap); /* table has grown */
  }

  /* every entry survives the rehash with its own length and value */
  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    const int m = (1 < s) ? 20 : 8;
    for (j = 0; j < m; ++j) {
      const int* v;
      test_mixed_fill_key(key, s, j);
      v = (const int*)libxs_registry_get(registry, key, (size_t)s, NULL);
      TEST_CHECK(NULL != v && *v == (s * 1000 + j));
    }
  }

  /* a truncated view of a longer key must not resolve to that key */
  for (s = 3; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    const int* v;
    test_mixed_fill_key(key, s, 0);
    v = (const int*)libxs_registry_get(registry, key, (size_t)s - 1, NULL);
    /* bytes match the (s-1)-length key only if its length byte agrees */
    TEST_CHECK(NULL == v);
  }

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_mixed_key_sizes_tombstone(void)
{ /* tombstones left by one key length must not shadow or alias entries */
  /* probed with a different length */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  const int nsize = 32, nidx = 8;
  int s, j;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  memset(key, 0, sizeof(key));

  for (s = 1; s <= nsize; ++s) {
    for (j = 0; j < nidx; ++j) {
      const int val = s * 1000 + j;
      test_mixed_fill_key(key, s, j);
      TEST_CHECK(NULL != libxs_registry_set(registry, key, (size_t)s,
        &val, sizeof(val), NULL));
    }
  }

  /* remove all entries of even length: leaves tombstones throughout */
  for (s = 2; s <= nsize; s += 2) {
    for (j = 0; j < nidx; ++j) {
      test_mixed_fill_key(key, s, j);
      libxs_registry_remove(registry, key, (size_t)s, NULL);
    }
  }

  /* odd-length entries are untouched, even-length ones are gone */
  for (s = 1; s <= nsize; ++s) {
    for (j = 0; j < nidx; ++j) {
      const int* v;
      test_mixed_fill_key(key, s, j);
      v = (const int*)libxs_registry_get(registry, key, (size_t)s, NULL);
      if (0 == (s % 2)) {
        TEST_CHECK(NULL == v);
      }
      else {
        TEST_CHECK(NULL != v && *v == (s * 1000 + j));
      }
    }
  }

  /* insert fresh even-length keys: probing must traverse the tombstones */
  for (s = 2; s <= nsize; s += 2) {
    for (j = 100; j < 100 + nidx; ++j) {
      const int val = s * 1000 + j;
      const int* v;
      test_mixed_fill_key(key, s, j);
      v = (const int*)libxs_registry_set(registry, key, (size_t)s,
        &val, sizeof(val), NULL);
      TEST_CHECK(NULL != v && *v == val);
    }
  }

  /* final state: odd 0..7, even 100..107, and nothing else */
  for (s = 1; s <= nsize; ++s) {
    for (j = 0; j < nidx; ++j) {
      const int* v;
      test_mixed_fill_key(key, s, j);
      v = (const int*)libxs_registry_get(registry, key, (size_t)s, NULL);
      TEST_CHECK((0 == (s % 2)) ? (NULL == v) : (NULL != v && *v == (s * 1000 + j)));
      test_mixed_fill_key(key, s, j + 100);
      v = (const int*)libxs_registry_get(registry, key, (size_t)s, NULL);
      TEST_CHECK((0 == (s % 2)) ? (NULL != v && *v == (s * 1000 + j + 100)) : (NULL == v));
    }
  }

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_mixed_key_sizes_cache(void)
{ /* the TLS cache keys on the key length too: a cached entry must never be */
  /* served for the same bytes read at a different length */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  int i, n;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  for (i = 0; i < LIBXS_REGKEY_MAXSIZE; ++i) key[i] = (unsigned char)(i + 1);
  for (n = 1; n <= LIBXS_REGKEY_MAXSIZE; ++n) {
    TEST_CHECK(NULL != libxs_registry_set(registry, key, (size_t)n,
      &n, sizeof(n), NULL));
  }

  /* hammer alternating lengths: more distinct keys than cache entries,
     so the cache both hits and evicts throughout */
  for (i = 0; i < 100; ++i) {
    for (n = 1; n <= LIBXS_REGKEY_MAXSIZE; ++n) {
      const int* v = (const int*)libxs_registry_get(registry, key, (size_t)n, NULL);
      TEST_CHECK(NULL != v && *v == n);
    }
  }

  /* immediate neighbours: a hit for length n must not answer length n+1 */
  for (n = 1; n < LIBXS_REGKEY_MAXSIZE; ++n) {
    const int* a = (const int*)libxs_registry_get(registry, key, (size_t)n, NULL);
    const int* b = (const int*)libxs_registry_get(registry, key, (size_t)n + 1, NULL);
    TEST_CHECK(NULL != a && *a == n);
    TEST_CHECK(NULL != b && *b == (n + 1));
    TEST_CHECK(a != b);
  }

  /* invalidation is length-specific: removing one length must not drop
     the cached values of the surrounding lengths */
  libxs_registry_remove(registry, key, 3, NULL);
  TEST_CHECK(NULL == libxs_registry_get(registry, key, 3, NULL));
  { const int* v = (const int*)libxs_registry_get(registry, key, 2, NULL);
    TEST_CHECK(NULL != v && 2 == *v);
    v = (const int*)libxs_registry_get(registry, key, 4, NULL);
    TEST_CHECK(NULL != v && 4 == *v);
  }

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_mixed_key_sizes_iteration(void)
{ /* iteration yields each entry's own key size, so a mixed-length registry */
  /* can be enumerated without knowing the lengths up front */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  int visited[LIBXS_REGKEY_MAXSIZE + 1];
  int s, count = 0;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  memset(key, 0, sizeof(key));
  memset(visited, 0, sizeof(visited));

  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    test_mixed_fill_key(key, s, 0);
    TEST_CHECK(NULL != libxs_registry_set(registry, key, (size_t)s,
      &s, sizeof(s), NULL));
  }

  { const void* regkey = NULL;
    size_t regkey_size = 0, cursor = 0;
    const void* entry = libxs_registry_begin_length(
      registry, &regkey, &regkey_size, &cursor);
    for (; NULL != entry;
         entry = libxs_registry_next_length(registry, &regkey, &regkey_size, &cursor))
    {
      /* the reported size identifies the entry, and re-reading the key at
         exactly that size must round-trip back to the very same value */
      const int val = *(const int*)entry;
      TEST_CHECK(0 < regkey_size && regkey_size <= LIBXS_REGKEY_MAXSIZE);
      TEST_CHECK(val == (int)regkey_size);
      TEST_CHECK(0 == visited[regkey_size]);
      visited[regkey_size] = 1;
      { const int* v = (const int*)libxs_registry_get(
          registry, regkey, regkey_size, NULL);
        TEST_CHECK(NULL != v && *v == val);
      }
      ++count;
    }
  }
  TEST_CHECK(LIBXS_REGKEY_MAXSIZE == count);
  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) TEST_CHECK(1 == visited[s]);

  /* key_size argument is optional, and the plain flavor still works */
  { const void* regkey = NULL;
    size_t cursor = 0;
    TEST_CHECK(NULL != libxs_registry_begin_length(registry, &regkey, NULL, &cursor));
    TEST_CHECK(NULL != libxs_registry_begin_length(registry, NULL, NULL, &cursor));
    TEST_CHECK(NULL != libxs_registry_begin(registry, &regkey, &cursor));
  }

  /* empty registry: key size is cleared, not left stale */
  { const void* regkey = &key;
    size_t regkey_size = 123, cursor = 0;
    libxs_registry_t* empty = libxs_registry_create();
    TEST_CHECK(NULL != empty);
    TEST_CHECK(NULL == libxs_registry_begin_length(empty, &regkey, &regkey_size, &cursor));
    TEST_CHECK(NULL == regkey);
    TEST_CHECK(0 == regkey_size);
    /* NULL registry must not crash either */
    TEST_CHECK(NULL == libxs_registry_begin_length(NULL, NULL, NULL, NULL));
    TEST_CHECK(NULL == libxs_registry_next_length(NULL, NULL, NULL, NULL));
    libxs_registry_destroy(empty);
  }

  /* iteration past the last entry clears the reported key size */
  { const void* regkey = NULL;
    size_t regkey_size = 0, cursor = 0;
    const void* entry = libxs_registry_begin_length(
      registry, &regkey, &regkey_size, &cursor);
    while (NULL != entry) {
      entry = libxs_registry_next_length(registry, &regkey, &regkey_size, &cursor);
    }
    TEST_CHECK(NULL == regkey);
    TEST_CHECK(0 == regkey_size);
  }

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_tls_cache(void)
{ /* repeated get should be served from TLS cache; free invalidates cache */
  const int key = 99;
  const char val[] = "cached";
  char* v;
  int i;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  v = (char*)libxs_registry_set(registry, &key, sizeof(key), val, sizeof(val), NULL);
  TEST_CHECK(NULL != v);

  { /* first get populates TLS cache, second get hits it (both must return same pointer) */
    const char* v1 = (const char*)libxs_registry_get(registry, &key, sizeof(key), NULL);
    const char* v2 = (const char*)libxs_registry_get(registry, &key, sizeof(key), NULL);
    TEST_CHECK(NULL != v1 && NULL != v2);
    TEST_CHECK(v1 == v2); /* same pointer */
    TEST_CHECK(0 == strcmp(v1, val));
  }

  /* many repeated gets must all succeed (hammer cache path) */
  for (i = 0; i < 1000; ++i) {
    TEST_CHECK(NULL != libxs_registry_get(registry, &key, sizeof(key), NULL));
  }

  /* free invalidates cache; subsequent get must return NULL */
  libxs_registry_remove(registry, &key, sizeof(key), NULL);
  TEST_CHECK(NULL == libxs_registry_get(registry, &key, sizeof(key), NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_tls_cache_growth(void)
{ /* growth rehashes and frees the entry table: cached pointers to inline */
  /* values pointed into it, so a stale hit would be use-after-free */
  const int hot = 999;
  double* p;
  unsigned int i;
  double one = 1.0;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  TEST_CHECK(sizeof(one) <= sizeof(void*)); /* value must be stored inline */

  TEST_CHECK(NULL != libxs_registry_set(registry, &hot, sizeof(hot),
    &one, sizeof(one), NULL));
  p = (double*)libxs_registry_get(registry, &hot, sizeof(hot), NULL);
  TEST_CHECK(NULL != p); /* seeds the TLS cache */

  /* insert other keys one at a time (crossing several growth thresholds) and
     re-read the cached key after each: the entry moves, so every get must
     return the CURRENT location, never the freed one */
  for (i = 1; i < 4000; ++i) {
    const unsigned int k = 100000 + i;
    double v = 1.0;
    TEST_CHECK(NULL != libxs_registry_set(registry, &k, sizeof(k),
      &v, sizeof(v), NULL));
    p = (double*)libxs_registry_get(registry, &hot, sizeof(hot), NULL);
    TEST_CHECK(NULL != p);
    *p += 1.0;
  }
  /* no increment was lost to a stale pointer into the old table */
  p = (double*)libxs_registry_get(registry, &hot, sizeof(hot), NULL);
  TEST_CHECK(NULL != p);
  TEST_CHECK(4000.0 == *p);

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_multiple_registries(void)
{ /* two independent registries with same keys must not interfere */
  const int key = 1;
  const int v1 = 100, v2 = 200;
  int* p;
  libxs_registry_t *r1 = libxs_registry_create();
  libxs_registry_t *r2 = libxs_registry_create();
  TEST_CHECK(NULL != r1 && NULL != r2);

  p = (int*)libxs_registry_set(r1, &key, sizeof(key), &v1, sizeof(int), NULL);
  TEST_CHECK(NULL != p && *p == v1);
  p = (int*)libxs_registry_set(r2, &key, sizeof(key), &v2, sizeof(int), NULL);
  TEST_CHECK(NULL != p && *p == v2);

  /* get from each registry returns its own value */
  p = (int*)libxs_registry_get(r1, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != p && *p == v1);
  p = (int*)libxs_registry_get(r2, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != p && *p == v2);

  /* destroy one, other is unaffected */
  libxs_registry_destroy(r1);
  p = (int*)libxs_registry_get(r2, &key, sizeof(key), NULL);
  TEST_CHECK(NULL != p && *p == v2);

  libxs_registry_destroy(r2);
  return EXIT_SUCCESS;
}


static int test_has(void)
{ /* _has returns non-zero for existing keys, zero for missing */
  const int key = 42, missing = 99;
  const double val = 3.14;
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  TEST_CHECK(0 == libxs_registry_has(registry, &key, sizeof(key), NULL));
  TEST_CHECK(NULL != libxs_registry_set(registry, &key, sizeof(key), &val, sizeof(val), NULL));
  TEST_CHECK(0 != libxs_registry_has(registry, &key, sizeof(key), NULL));
  TEST_CHECK(0 == libxs_registry_has(registry, &missing, sizeof(missing), NULL));

  /* NULL / invalid args */
  TEST_CHECK(0 == libxs_registry_has(NULL, &key, sizeof(key), NULL));
  TEST_CHECK(0 == libxs_registry_has(registry, NULL, sizeof(key), NULL));
  TEST_CHECK(0 == libxs_registry_has(registry, &key, 0, NULL));

  /* remove -> no longer found */
  libxs_registry_remove(registry, &key, sizeof(key), NULL);
  TEST_CHECK(0 == libxs_registry_has(registry, &key, sizeof(key), NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_value_size(void)
{ /* _value_size returns stored size, 0 for missing keys */
  const int key = 7;
  const char small[] = "ab";
  const char large[] = "abcdef";
  libxs_registry_t* registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);

  TEST_CHECK(0 == libxs_registry_value_size(registry, &key, sizeof(key), NULL));

  TEST_CHECK(NULL != libxs_registry_set(registry, &key, sizeof(key), small, sizeof(small), NULL));
  TEST_CHECK(sizeof(small) == libxs_registry_value_size(registry, &key, sizeof(key), NULL));

  /* auto-realloc to larger -> value_size grows */
  TEST_CHECK(NULL != libxs_registry_set(registry, &key, sizeof(key), large, sizeof(large), NULL));
  TEST_CHECK(sizeof(large) == libxs_registry_value_size(registry, &key, sizeof(key), NULL));

  /* NULL / invalid args */
  TEST_CHECK(0 == libxs_registry_value_size(NULL, &key, sizeof(key), NULL));
  TEST_CHECK(0 == libxs_registry_value_size(registry, NULL, sizeof(key), NULL));
  TEST_CHECK(0 == libxs_registry_value_size(registry, &key, 0, NULL));

  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_save_load(void)
{ /* save registry to buffer, load from buffer, verify all entries survive */
  const int keys[] = { 1, 2, 3, 4, 5 };
  const char* vals[] = { "alpha", "beta", "gamma", "delta", "epsilon" };
  const int n = (int)(sizeof(keys) / sizeof(keys[0]));
  int i;
  size_t buf_size = 0;
  void* buf;
  libxs_registry_t* registry = libxs_registry_create();
  libxs_registry_t* loaded;
  TEST_CHECK(NULL != registry);

  for (i = 0; i < n; ++i) {
    TEST_CHECK(NULL != libxs_registry_set(registry, &keys[i], sizeof(keys[0]),
      vals[i], strlen(vals[i]) + 1, NULL));
  }

  /* query required size */
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, NULL, &buf_size));
  TEST_CHECK(0 < buf_size);

  buf = malloc(buf_size);
  TEST_CHECK(NULL != buf);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, buf, &buf_size));

  /* load from buffer */
  loaded = libxs_registry_load(buf, buf_size, NULL, NULL);
  TEST_CHECK(NULL != loaded);

  /* verify all entries */
  { libxs_registry_info_t info;
    TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(loaded, &info));
    TEST_CHECK(info.size == (size_t)n);
  }
  for (i = 0; i < n; ++i) {
    const char* v = (const char*)libxs_registry_get(loaded, &keys[i], sizeof(keys[0]), NULL);
    TEST_CHECK(NULL != v);
    TEST_CHECK(0 == strcmp(v, vals[i]));
  }

  /* overwrite a loaded entry (transitions from ext to owned) */
  { const char* newval = "replaced value that is longer";
    char* v = (char*)libxs_registry_set(loaded, &keys[0], sizeof(keys[0]),
      newval, strlen(newval) + 1, NULL);
    TEST_CHECK(NULL != v);
    TEST_CHECK(0 == strcmp(v, newval));
  }

  /* remove a loaded entry (ext pointer must not be freed) */
  libxs_registry_remove(loaded, &keys[1], sizeof(keys[0]), NULL);
  TEST_CHECK(NULL == libxs_registry_get(loaded, &keys[1], sizeof(keys[0]), NULL));

  libxs_registry_destroy(loaded);
  free(buf);
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_save_load_inline(void)
{ /* values small enough for inline storage round-trip correctly */
  const int key = 42;
  const int val = 12345;
  size_t buf_size = 0;
  void* buf;
  libxs_registry_t* registry = libxs_registry_create();
  libxs_registry_t* loaded;
  TEST_CHECK(NULL != registry);

  TEST_CHECK(NULL != libxs_registry_set(registry, &key, sizeof(key), &val, sizeof(val), NULL));

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, NULL, &buf_size));
  buf = malloc(buf_size);
  TEST_CHECK(NULL != buf);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, buf, &buf_size));

  loaded = libxs_registry_load(buf, buf_size, NULL, NULL);
  TEST_CHECK(NULL != loaded);

  { const int* v = (const int*)libxs_registry_get(loaded, &key, sizeof(key), NULL);
    TEST_CHECK(NULL != v && *v == val);
  }

  libxs_registry_destroy(loaded);
  free(buf);
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_save_load_empty(void)
{ /* save/load an empty registry */
  size_t buf_size = 0;
  void* buf;
  libxs_registry_t* registry = libxs_registry_create();
  libxs_registry_t* loaded;
  libxs_registry_info_t info;
  TEST_CHECK(NULL != registry);

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, NULL, &buf_size));
  buf = malloc(buf_size);
  TEST_CHECK(NULL != buf);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, buf, &buf_size));

  loaded = libxs_registry_load(buf, buf_size, NULL, NULL);
  TEST_CHECK(NULL != loaded);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(loaded, &info));
  TEST_CHECK(0 == info.size);

  libxs_registry_destroy(loaded);
  free(buf);
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static void test_fixup_noop(void* value, const void* key, size_t key_size,
  size_t value_size, void* udata)
{ /* record that fixup sees the per-entry key length, not a fixed one */
  const unsigned char* k = (const unsigned char*)key;
  LIBXS_UNUSED(value); LIBXS_UNUSED(value_size); LIBXS_UNUSED(udata);
  ++test_fixup_nvisits;
  /* test_mixed_fill_key encodes the key length in the second Byte */
  if (1 < key_size && k[1] != (unsigned char)key_size) test_fixup_keysizes = 1;
}


static int test_mixed_key_sizes_save_load(void)
{ /* the serialized keys section is length-prefixed, hence a per-entry */
  /* stride: a fixed-stride assumption would misplace the values section */
  unsigned char key[LIBXS_REGKEY_MAXSIZE];
  char payload[128];
  libxs_registry_info_t info;
  size_t buf_size = 0;
  void* buf;
  int s, i;
  libxs_registry_t* registry = libxs_registry_create();
  libxs_registry_t* loaded;
  TEST_CHECK(NULL != registry);
  memset(key, 0, sizeof(key));
  for (i = 0; i < (int)sizeof(payload); ++i) payload[i] = (char)(i + 1);

  /* mix key lengths AND value sizes so both inline and heap values occur */
  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    const size_t vsize = (0 == (s % 3)) ? sizeof(payload) : sizeof(int);
    test_mixed_fill_key(key, s, 0);
    if (sizeof(int) == vsize) {
      TEST_CHECK(NULL != libxs_registry_set(registry, key, (size_t)s,
        &s, sizeof(int), NULL));
    }
    else { /* first Byte encodes the key length to identify the entry */
      payload[0] = (char)s;
      TEST_CHECK(NULL != libxs_registry_set(registry, key, (size_t)s,
        payload, vsize, NULL));
    }
  }

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, NULL, &buf_size));
  TEST_CHECK(0 < buf_size);
  buf = malloc(buf_size);
  TEST_CHECK(NULL != buf);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, buf, &buf_size));

  loaded = libxs_registry_load(buf, buf_size, NULL, NULL);
  TEST_CHECK(NULL != loaded);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(loaded, &info));
  TEST_CHECK((size_t)LIBXS_REGKEY_MAXSIZE == info.size);

  /* every key length round-trips with its own length and value */
  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    const size_t vsize = (0 == (s % 3)) ? sizeof(payload) : sizeof(int);
    const void* v;
    test_mixed_fill_key(key, s, 0);
    v = libxs_registry_get(loaded, key, (size_t)s, NULL);
    TEST_CHECK(NULL != v);
    TEST_CHECK(vsize == libxs_registry_value_size(loaded, key, (size_t)s, NULL));
    if (sizeof(int) == vsize) {
      TEST_CHECK(*(const int*)v == s);
    }
    else {
      TEST_CHECK((char)s == *(const char*)v);
      TEST_CHECK(0 == memcmp((const char*)v + 1, payload + 1, vsize - 1));
    }
  }

  /* a length that was never registered must not resolve */
  test_mixed_fill_key(key, LIBXS_REGKEY_MAXSIZE, 1);
  TEST_CHECK(NULL == libxs_registry_get(loaded, key, LIBXS_REGKEY_MAXSIZE, NULL));

  libxs_registry_destroy(loaded);

  /* same round-trip through the fixup path (heap-allocates every entry) */
  test_fixup_nvisits = 0;
  test_fixup_keysizes = 0;
  loaded = libxs_registry_load(buf, buf_size, test_fixup_noop, NULL);
  TEST_CHECK(NULL != loaded);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_info(loaded, &info));
  TEST_CHECK((size_t)LIBXS_REGKEY_MAXSIZE == info.size);
  TEST_CHECK((size_t)LIBXS_REGKEY_MAXSIZE == test_fixup_nvisits);
  TEST_CHECK(0 == test_fixup_keysizes); /* fixup saw each entry's own length */
  for (s = 1; s <= LIBXS_REGKEY_MAXSIZE; ++s) {
    test_mixed_fill_key(key, s, 0);
    TEST_CHECK(NULL != libxs_registry_get(loaded, key, (size_t)s, NULL));
  }

  libxs_registry_destroy(loaded);
  free(buf);
  libxs_registry_destroy(registry);
  return EXIT_SUCCESS;
}


static int test_load_invalid(void)
{ /* invalid buffers must return NULL */
  const char garbage[] = "not a registry";
  TEST_CHECK(NULL == libxs_registry_load(NULL, 100, NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_load(garbage, sizeof(garbage), NULL, NULL));
  TEST_CHECK(NULL == libxs_registry_load(garbage, 0, NULL, NULL));
  return EXIT_SUCCESS;
}


typedef struct test_fixup_entry_t {
  int data;
  void (*callback)(int);
} test_fixup_entry_t;

static int test_fixup_counter;

static void test_fixup_cb(int v)
{
  test_fixup_counter += v;
}

static void test_fixup_fn(void* value, const void* key, size_t key_size,
  size_t value_size, void* udata)
{
  test_fixup_entry_t* e = (test_fixup_entry_t*)value;
  LIBXS_UNUSED(key); LIBXS_UNUSED(key_size);
  LIBXS_UNUSED(value_size); LIBXS_UNUSED(udata);
  e->callback = test_fixup_cb;
}

static int test_save_load_fixup(void)
{ /* fixup callback restores function pointers after load */
  const int key = 1;
  test_fixup_entry_t entry;
  size_t buf_size = 0;
  void* buf;
  libxs_registry_t* registry = libxs_registry_create();
  libxs_registry_t* loaded;
  TEST_CHECK(NULL != registry);

  memset(&entry, 0, sizeof(entry));
  entry.data = 42;
  entry.callback = test_fixup_cb;
  TEST_CHECK(NULL != libxs_registry_set(registry, &key, sizeof(key),
    &entry, sizeof(entry), NULL));

  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, NULL, &buf_size));
  buf = malloc(buf_size);
  TEST_CHECK(NULL != buf);
  TEST_CHECK(EXIT_SUCCESS == libxs_registry_save(registry, buf, &buf_size));

  loaded = libxs_registry_load(buf, buf_size, test_fixup_fn, NULL);
  TEST_CHECK(NULL != loaded);

  { const test_fixup_entry_t* v = (const test_fixup_entry_t*)libxs_registry_get(
      loaded, &key, sizeof(key), NULL);
    TEST_CHECK(NULL != v);
    TEST_CHECK(42 == v->data);
    TEST_CHECK(test_fixup_cb == v->callback);
    test_fixup_counter = 0;
    v->callback(10);
    TEST_CHECK(10 == test_fixup_counter);
  }

  libxs_registry_destroy(loaded);
  free(buf);
  libxs_registry_destroy(registry);
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
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes();
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes_growth();
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes_tombstone();
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes_cache();
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes_iteration();
  if (EXIT_SUCCESS == result) result = test_tls_cache();
  if (EXIT_SUCCESS == result) result = test_tls_cache_growth();
  if (EXIT_SUCCESS == result) result = test_multiple_registries();
  if (EXIT_SUCCESS == result) result = test_has();
  if (EXIT_SUCCESS == result) result = test_value_size();
  if (EXIT_SUCCESS == result) result = test_save_load();
  if (EXIT_SUCCESS == result) result = test_save_load_inline();
  if (EXIT_SUCCESS == result) result = test_save_load_empty();
  if (EXIT_SUCCESS == result) result = test_mixed_key_sizes_save_load();
  if (EXIT_SUCCESS == result) result = test_load_invalid();
  if (EXIT_SUCCESS == result) result = test_save_load_fixup();

  return result;
}
