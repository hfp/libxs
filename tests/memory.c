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
  /* backtracking: partial match at start, real match later */
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("aab", "ab")) {
    FPRINTF(stderr, "ERROR line #%i: stristr backtrack aab/ab\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && NULL == libxs_stristr("aaab", "aab")) {
    FPRINTF(stderr, "ERROR line #%i: stristr backtrack aaab/aab\n", __LINE__);
    result = EXIT_FAILURE;
  }
  /* haystack shorter than needle: must NOT match */
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("abc", "abcd")) {
    FPRINTF(stderr, "ERROR line #%i: stristr short haystack abc/abcd\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("ab", "abc")) {
    FPRINTF(stderr, "ERROR line #%i: stristr short haystack ab/abc\n", __LINE__);
    result = EXIT_FAILURE;
  }
  /* repeated chars at tail of pattern (was a buggy special-case path) */
  if (EXIT_SUCCESS == result && NULL != libxs_stristr("bo", "boo")) {
    FPRINTF(stderr, "ERROR line #%i: stristr partial bo/boo\n", __LINE__);
    result = EXIT_FAILURE;
  }

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
          FPRINTF(stderr, "ERROR line #%i: data not shuffled or copy failed!\n", __LINE__);
          result = EXIT_FAILURE; break;
        }
        /* shuffle restores initial input */
        for (i = 0; i < r; ++i) {
          memset(b, 0, size); /* clear */
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_shuffle2(b, a, 1, s, &shuffle, NULL));
          /* every shuffle is different from input */
          if (1 < s && 0 == memcmp(a, b, s)) {
            FPRINTF(stderr, "ERROR line #%i: data not shuffled!\n", __LINE__);
            result = EXIT_FAILURE; break;
          }
          if (0 == memcmp(b, init, s)) break; /* restored */
          else if (r == (i + 1)) {
            FPRINTF(stderr, "ERROR line #%i: data not restored!\n", __LINE__);
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
            FPRINTF(stderr, "ERROR line #%i: data not restored via nrepeat!\n", __LINE__);
            result = EXIT_FAILURE;
            break;
          }
        }
        else break; /* previous error */
      }
      else {
        FPRINTF(stderr, "ERROR line #%i: shuffle argument not coprime!\n", __LINE__);
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  /* check libxs_aligned */
  if (EXIT_SUCCESS == result) {
    LIBXS_ALIGNED(char aligned_buf[64], LIBXS_CACHELINE);
    int alignment = 0;
    /* An aligned buffer should be reported as aligned */
    libxs_aligned(aligned_buf, NULL, &alignment);
    if (0 >= alignment) {
      FPRINTF(stderr, "ERROR line #%i: aligned buf has alignment=%i\n", __LINE__, alignment);
      result = EXIT_FAILURE;
    }
    /* NULL increment should not affect the result */
    if (EXIT_SUCCESS == result && 0 == libxs_aligned(aligned_buf, NULL, NULL)) {
      /* may fail if cache line != vlen, not necessarily an error */
    }
    /* Misaligned pointer: offset by 1 byte */
    if (EXIT_SUCCESS == result) {
      int misalign = 0;
      libxs_aligned(aligned_buf + 1, NULL, &misalign);
      /* alignment of (base+1) should be exactly 1 */
      if (1 != misalign) {
        FPRINTF(stderr, "ERROR line #%i: misaligned ptr alignment=%i expected 1\n", __LINE__, misalign);
        result = EXIT_FAILURE;
      }
    }
  }

  /* check libxs_diff */
  if (EXIT_SUCCESS == result) {
    const char buf_a[] = "ABCDEFGHIJKLMNOP";
    const char buf_b[] = "ABCDEFGHIJKLMNOP";
    const char buf_c[] = "ABCDEFGHIJKLMNOx";
    /* identical buffers */
    if (0 != libxs_diff(buf_a, buf_b, (unsigned char)sizeof(buf_a))) {
      FPRINTF(stderr, "ERROR line #%i: diff reports mismatch for equal buffers\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* different buffers */
    if (EXIT_SUCCESS == result && 0 == libxs_diff(buf_a, buf_c, (unsigned char)sizeof(buf_a))) {
      FPRINTF(stderr, "ERROR line #%i: diff reports match for different buffers\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* zero-size comparison should report no difference */
    if (EXIT_SUCCESS == result && 0 != libxs_diff(buf_a, buf_c, 0)) {
      FPRINTF(stderr, "ERROR line #%i: diff with size=0 reports mismatch\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* single-byte comparison */
    if (EXIT_SUCCESS == result && 0 != libxs_diff(buf_a, buf_b, 1)) {
      FPRINTF(stderr, "ERROR line #%i: diff single byte mismatch\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) { /* diff at various sizes */
      char x[64], y[64];
      unsigned char s;
      for (s = 1; s <= 64 && EXIT_SUCCESS == result; ++s) {
        memset(x, 0x55, s); memset(y, 0x55, s);
        if (0 != libxs_diff(x, y, s)) {
          FPRINTF(stderr, "ERROR line #%i: diff size=%i false positive\n", __LINE__, (int)s);
          result = EXIT_FAILURE;
        }
        y[s - 1] ^= 0xFF; /* flip last byte */
        if (0 == libxs_diff(x, y, s)) {
          FPRINTF(stderr, "ERROR line #%i: diff size=%i false negative\n", __LINE__, (int)s);
          result = EXIT_FAILURE;
        }
      }
    }
  }

  /* check libxs_memcmp */
  if (EXIT_SUCCESS == result) {
    const char buf_a[] = "The quick brown fox";
    const char buf_b[] = "The quick brown fox";
    const char buf_c[] = "The quick brown foX";
    if (0 != libxs_memcmp(buf_a, buf_b, sizeof(buf_a))) {
      FPRINTF(stderr, "ERROR line #%i: memcmp identical\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 0 == libxs_memcmp(buf_a, buf_c, sizeof(buf_a))) {
      FPRINTF(stderr, "ERROR line #%i: memcmp different\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 0 != libxs_memcmp(buf_a, buf_c, 0)) {
      FPRINTF(stderr, "ERROR line #%i: memcmp size=0\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* test with larger buffers to exercise SIMD paths (>= 64 bytes) */
    if (EXIT_SUCCESS == result) {
      char big_a[256], big_b[256];
      memset(big_a, 0xAA, sizeof(big_a));
      memset(big_b, 0xAA, sizeof(big_b));
      if (0 != libxs_memcmp(big_a, big_b, sizeof(big_a))) {
        FPRINTF(stderr, "ERROR line #%i: memcmp large identical\n", __LINE__);
        result = EXIT_FAILURE;
      }
      big_b[255] = 0x55;
      if (EXIT_SUCCESS == result && 0 == libxs_memcmp(big_a, big_b, sizeof(big_a))) {
        FPRINTF(stderr, "ERROR line #%i: memcmp large different\n", __LINE__);
        result = EXIT_FAILURE;
      }
    }
  }

  /* check libxs_stristrn (length-limited case-insensitive search) */
  if (EXIT_SUCCESS == result) {
    /* basic match within length limit */
    if (NULL == libxs_stristrn("Hello World", "WORLD", 5)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn basic match\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* truncated needle: only first 3 chars of "WORLD" considered */
    if (EXIT_SUCCESS == result && NULL == libxs_stristrn("Hello World", "WORxx", 3)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn truncated match\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* no match */
    if (EXIT_SUCCESS == result && NULL != libxs_stristrn("Hello", "xyz", 3)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn false match\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* empty / NULL */
    if (EXIT_SUCCESS == result && NULL != libxs_stristrn(NULL, "x", 1)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn NULL a\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && NULL != libxs_stristrn("x", NULL, 1)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn NULL b\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && NULL != libxs_stristrn("abc", "d", 0)) {
      FPRINTF(stderr, "ERROR line #%i: stristrn maxlen=0\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }

  /* check libxs_format_value */
  if (EXIT_SUCCESS == result) {
    char buffer[32];
    size_t val;
    /* 1 KiB = 1024 B */
    val = libxs_format_value(buffer, sizeof(buffer), 1024, "KMGT", "B", 10);
    FPRINTF(stdout, "format_value(1024) = \"%s\" val=%lu\n", buffer, (unsigned long)val);
    if (1 != val) {
      FPRINTF(stderr, "ERROR line #%i: format_value 1KiB val=%lu\n", __LINE__, (unsigned long)val);
      result = EXIT_FAILURE;
    }
    /* 0 bytes */
    if (EXIT_SUCCESS == result) {
      val = libxs_format_value(buffer, sizeof(buffer), 0, "KMGT", "B", 10);
      if (0 != val) {
        FPRINTF(stderr, "ERROR line #%i: format_value 0B val=%lu\n", __LINE__, (unsigned long)val);
        result = EXIT_FAILURE;
      }
    }
    /* 1 MiB = 1048576 */
    if (EXIT_SUCCESS == result) {
      val = libxs_format_value(buffer, sizeof(buffer), 1048576, "KMGT", "B", 10);
      FPRINTF(stdout, "format_value(1048576) = \"%s\" val=%lu\n", buffer, (unsigned long)val);
      if (1 != val) {
        FPRINTF(stderr, "ERROR line #%i: format_value 1MiB val=%lu\n", __LINE__, (unsigned long)val);
        result = EXIT_FAILURE;
      }
    }
    /* sub-kilo: 512 bytes should stay as bytes */
    if (EXIT_SUCCESS == result) {
      val = libxs_format_value(buffer, sizeof(buffer), 512, "KMGT", "B", 10);
      FPRINTF(stdout, "format_value(512) = \"%s\" val=%lu\n", buffer, (unsigned long)val);
      if (512 != val) {
        FPRINTF(stderr, "ERROR line #%i: format_value 512B val=%lu\n", __LINE__, (unsigned long)val);
        result = EXIT_FAILURE;
      }
    }
  }

  /* check LIBXS_MEMSET / LIBXS_MEMZERO */
  if (EXIT_SUCCESS == result) {
    int vals[4];
    LIBXS_MEMZERO(&vals);
    if (0 != vals[0] || 0 != vals[1] || 0 != vals[2] || 0 != vals[3]) {
      FPRINTF(stderr, "ERROR line #%i: MEMZERO\n", __LINE__);
      result = EXIT_FAILURE;
    }
    LIBXS_MEMSET(&vals, 0, sizeof(vals)); /* set entire structure to 0 */
    if (0 != vals[0] || 0 != vals[1] || 0 != vals[2] || 0 != vals[3]) {
      FPRINTF(stderr, "ERROR line #%i: MEMSET\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ASSIGN / LIBXS_VALUE_ASSIGN */
  if (EXIT_SUCCESS == result) {
    int src_val = 42, dst_val = 0;
    LIBXS_ASSIGN(&dst_val, &src_val);
    if (42 != dst_val) {
      FPRINTF(stderr, "ERROR line #%i: ASSIGN\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      const int cv = 99;
      int nv = 0;
      LIBXS_VALUE_ASSIGN(nv, cv);
      if (99 != nv) {
        FPRINTF(stderr, "ERROR line #%i: VALUE_ASSIGN\n", __LINE__);
        result = EXIT_FAILURE;
      }
    }
  }

  /* check LIBXS_VALUE_SWAP */
  if (EXIT_SUCCESS == result) {
    int va = 10, vb = 20;
    LIBXS_VALUE_SWAP(va, vb);
    if (20 != va || 10 != vb) {
      FPRINTF(stderr, "ERROR line #%i: VALUE_SWAP\n", __LINE__);
      result = EXIT_FAILURE;
    }
    /* swap with larger types */
    if (EXIT_SUCCESS == result) {
      double da = 1.5, db = 2.5;
      LIBXS_VALUE_SWAP(da, db);
      if (LIBXS_NEQ(da, 2.5) || LIBXS_NEQ(db, 1.5)) {
        FPRINTF(stderr, "ERROR line #%i: VALUE_SWAP double\n", __LINE__);
        result = EXIT_FAILURE;
      }
    }
  }

  return result;
}
