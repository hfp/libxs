/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(int argc, char* argv[])
{
  char item[LIBXS_DESCRIPTOR_MAXSIZE];
  const size_t elemsize = sizeof(item);
  const int ndiffs = 1000, ntests = 1000;
  const char init[] = "The quick brown fox jumps over the lazy dog", *delims = NULL;
  int result = EXIT_SUCCESS, count = 0;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  /* check libxs_stristr */
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("ends with b", "Begins with b")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("in between of", "BeTwEEn")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("spr", "SPR")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr(NULL, "bb")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("aa", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxs_stristr(NULL, NULL)) result = EXIT_FAILURE;

  /* check libxs_strimatch */
  if (EXIT_SUCCESS == result && (2 != libxs_strimatch("Co Product A", "Corp Prod B",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (2 != libxs_strimatch("Corp Prod B", "Co Product A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Co Product A", "Corp Prod AA",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Corp Prod AA", "Co Product A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Corp Prod AA", "Co Product A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Co Product A", "Corp Prod AA",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Corp Prod A", "Co Product A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Co Product A", "Corp Prod A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("C Product A", "Cor Prod AA",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (3 != libxs_strimatch("Cor Prod AA", "C Product A",
      delims, &count) || 3 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (1 != libxs_strimatch("aaaa", "A A A A",
      delims, &count) || 4 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && (1 != libxs_strimatch("A A A A", "aaaa",
      delims, &count) || 4 != count)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const char *const sample[] = {
      "The quick red squirrel jumps over the low fence",
      "The slow green frog jumps over the lazy dog",
      "The lazy brown dog jumps over the quick fox", /* match */
      "The hazy fog crawls over the lazy crocodile"
    };
    int match = 0, i = 0;
#if defined(PRINT)
    int j = 0;
#endif
    for (; i < ((int)sizeof(sample) / (int)sizeof(*sample)); ++i) {
      const int score = libxs_strimatch(init, sample[i], delims, &count);
      if (match < score) {
        match = score;
#if defined(PRINT)
        j = i;
#endif
      }
      else if (0 > score) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      int self = 0;
      if (0 < match) {
        self = libxs_strimatch(init, init, delims, &count);
        FPRINTF(stdout, "orig (%i): %s\n", self, init);
        FPRINTF(stdout, "best (%i): %s\n", match, sample[j]);
      }
      if (9 != self || 9 != match) result = EXIT_FAILURE; /* test */
    }
  }

  /* check libxs_offset */
  if (EXIT_SUCCESS == result) {
    const size_t shape[] = { 17, 13, 64, 4 }, ndims = sizeof(shape) / sizeof(*shape);
    size_t size1 = 0, n;
    for (n = 0; n < ndims && EXIT_SUCCESS == result; ++n) {
      if (0 != libxs_offset(n, NULL, shape, NULL)) result = EXIT_FAILURE;
    }
    for (n = 0; n < ndims && EXIT_SUCCESS == result; ++n) {
      const size_t offset1 = libxs_offset(n, shape, shape, &size1);
      if (offset1 != size1) result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_MEMCPY127 and libxs_diff_n */
  if (EXIT_SUCCESS == result) {
    char *const data = (char*)malloc(elemsize * ndiffs);
    if (NULL != data) { /* check if buffer was allocated */
      int i = 0;
      libxs_rng_seq(data, elemsize * ndiffs);

      for (; i < ntests; ++i) {
        const size_t j = libxs_rng_u32((unsigned int)ndiffs);
        const size_t s = libxs_rng_u32((unsigned int)elemsize) + 1;
        size_t k = s;
        libxs_rng_seq(item, s);
        for (; k < elemsize; ++k) item[k] = 0;
        LIBXS_MEMCPY(data + elemsize * j, item, elemsize);
        k = libxs_diff_n(item, data,
          (unsigned char)s, (unsigned char)elemsize,
          0, ndiffs);
        while (k < j) {
          k = libxs_diff_n(item, data,
            (unsigned char)s, (unsigned char)elemsize,
            LIBXS_CAST_UINT(k + 1), ndiffs);
        }
        if (k == j) continue;
        else {
          result = EXIT_FAILURE;
          break;
        }
      }
      free(data);
    }
    else result = EXIT_FAILURE;
  }

  /* check LIBXS_MEMSWP127 */
  if (EXIT_SUCCESS == result) {
    char a[sizeof(init)] = { 0 };
    const size_t size = sizeof(init);
    size_t i, j, k;
    memcpy(a, init, size);

    for (k = 1; k <= 8; ++k) {
      const size_t s = (size - 1) / k;
      for (j = 0; j < s; ++j) {
        for (i = 0; i < (s - 1); ++i) {
          LIBXS_MEMSWP(a + k * i, a + k * i + k, k);
        }
      }
      if (0 != strcmp(a, init)) {
        FPRINTF(stderr, "LIBXS_MEMSWP127: incorrect result!\n");
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  if (EXIT_SUCCESS == result) { /* check libxs_shuffle */
    char a[sizeof(init)] = { 0 }, b[sizeof(init)] = { 0 };
    const size_t size = sizeof(init);
    size_t i = 1, j;
    memcpy(a, init, size);
    for (; i < size; ++i) {
      LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle(a, 1, size - i, NULL, NULL));
      LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(b, init, 1, size - i, NULL, NULL));
      if (0 == strncmp(a, b, size - i)) {
        const size_t r = libxs_unshuffle(size - i, NULL);
        for (j = 0; j < r; ++j) {
          libxs_shuffle(a, 1, size - i, NULL, NULL);
        }
        if (0 != strcmp(a, init)) {
          FPRINTF(stderr, "libxs_shuffle: data not restored!\n");
          result = EXIT_FAILURE; break;
        }
      }
      else {
        FPRINTF(stderr, "libxs_shuffle: result does not match libxs_shuffle2!\n");
        result = EXIT_FAILURE; break;
      }
    }
  }

  if (EXIT_SUCCESS == result) { /* check libxs_shuffle2 */
    char a[sizeof(init)], b[sizeof(init)];
    const size_t size = sizeof(init);
    size_t s = 0, i;
    for (; s < size; ++s) {
      const size_t shuffle = libxs_coprime2(s);
      const size_t gcd = libxs_gcd(shuffle, s);
      if (1 == gcd) {
        const size_t r = libxs_unshuffle(s, &shuffle);
        int cmp;
        memset(a, 0, size); /* clear */
        LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(a, init, 1, s, &shuffle, NULL));
        cmp = memcmp(a, init, s);
        if ((1 >= s || 0 == cmp) && (1 < s || 0 != cmp)) {
          FPRINTF(stderr, "libxs_shuffle2: data not shuffled or copy failed!\n");
          result = EXIT_FAILURE; break;
        }
        /* shuffle restores initial input */
        for (i = 0; i < r; ++i) {
          memset(b, 0, size); /* clear */
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(b, a, 1, s, &shuffle, NULL));
          /* every shuffle is different from input */
          if (1 < s && 0 == memcmp(a, b, s)) {
            FPRINTF(stderr, "libxs_shuffle2: data not shuffled!\n");
            result = EXIT_FAILURE; break;
          }
          if (0 == memcmp(b, init, s)) break; /* restored */
          else if (r == (i + 1)) {
            FPRINTF(stderr, "libxs_shuffle2: data not restored!\n");
            result = EXIT_FAILURE;
          }
          memcpy(a, b, s);
        }
        if (EXIT_SUCCESS == result) {
          memset(a, 0, size); /* clear */
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(a, init, 1, s, &shuffle, NULL));
          memset(b, 0, size); /* clear */
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(b, a, 1, s, &shuffle, &r));
          if (0 != memcmp(b, init, s)) {
            FPRINTF(stderr, "libxs_shuffle2: data not restored!\n");
            result = EXIT_FAILURE;
            break;
          }
        }
        else break; /* previous error */
      }
      else {
        FPRINTF(stderr, "libxs_shuffle2: shuffle argument not coprime!\n");
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  return result;
}
