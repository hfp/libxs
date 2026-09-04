/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>


static int check_decode(const char* what, const unsigned char* text,
  size_t size, unsigned long expect_cp, int expect_width)
{
  int result = EXIT_SUCCESS;
  int width = 0;
  const unsigned long cp = libxs_utf8_decode(text, size, &width);
  if (cp != expect_cp || width != expect_width) {
    fprintf(stderr, "decode %s: cp=U+%04lX width=%d, expected U+%04lX/%d\n",
      what, cp, width, expect_cp, expect_width);
    result = EXIT_FAILURE;
  }
  return result;
}


/** Well-formed sequences of every length must decode to the right code point. */
static int check_wellformed(void)
{
  int result = EXIT_SUCCESS;
  { const unsigned char a[] = { 'A' };
    result = check_decode("ascii", a, sizeof(a), 0x41, 1);
  }
  if (EXIT_SUCCESS == result) { /* U+00E4 a-umlaut */
    const unsigned char a[] = { 0xC3u, 0xA4u };
    result = check_decode("umlaut", a, sizeof(a), 0xE4, 2);
  }
  if (EXIT_SUCCESS == result) { /* U+2019 right single quote */
    const unsigned char a[] = { 0xE2u, 0x80u, 0x99u };
    result = check_decode("apostrophe", a, sizeof(a), 0x2019, 3);
  }
  if (EXIT_SUCCESS == result) { /* U+1F600 */
    const unsigned char a[] = { 0xF0u, 0x9Fu, 0x98u, 0x80u };
    result = check_decode("emoji", a, sizeof(a), 0x1F600, 4);
  }
  return result;
}


/**
 * The strict contract: a truncated or corrupt sequence yields the LEAD BYTE and
 * width 1. Returning a code point assembled from bytes that do not belong to it
 * would hand a caller a value it then tests properties of - and width 1 is what
 * keeps a scan advancing instead of standing still on the bad byte.
 */
static int check_malformed(void)
{
  int result = EXIT_SUCCESS;
  { const unsigned char a[] = { 0xC3u }; /* truncated 2-byte lead */
    result = check_decode("truncated", a, sizeof(a), 0xC3, 1);
  }
  if (EXIT_SUCCESS == result) { /* lead followed by a non-continuation byte */
    const unsigned char a[] = { 0xC3u, 'x' };
    result = check_decode("bad-continuation", a, sizeof(a), 0xC3, 1);
  }
  if (EXIT_SUCCESS == result) { /* stray continuation byte */
    const unsigned char a[] = { 0xA4u };
    result = check_decode("stray", a, sizeof(a), 0xA4, 1);
  }
  if (EXIT_SUCCESS == result) { /* 3-byte lead with only 2 bytes present */
    const unsigned char a[] = { 0xE2u, 0x80u };
    result = check_decode("short-3byte", a, sizeof(a), 0xE2, 1);
  }
  if (EXIT_SUCCESS == result) {
    int width = 0;
    if (0 != libxs_utf8_decode(NULL, 4, &width) || 1 != width) {
      fprintf(stderr, "decode NULL did not report width 1\n");
      result = EXIT_FAILURE;
    }
  }
  return result;
}


/**
 * libxs_utf8_size is the LENIENT form: it reports the width the lead byte claims,
 * clamped to what remains. It deliberately differs from decode on malformed input
 * - asserted here so the difference cannot drift into an accident.
 */
static int check_size_is_lenient(void)
{
  int result = EXIT_SUCCESS;
  { const unsigned char a[] = { 0xC3u, 'x' }; /* claims 2, bytes are bad */
    if (2 != libxs_utf8_size(a, sizeof(a), 0)) {
      fprintf(stderr, "size did not honor the claimed width\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) { /* claims 4 but only 2 remain: clamp */
    const unsigned char a[] = { 0xF0u, 0x9Fu };
    if (2 != libxs_utf8_size(a, sizeof(a), 0)) {
      fprintf(stderr, "size did not clamp to the remaining bytes\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) { /* never zero, or a scan would not advance */
    const unsigned char a[] = { 0xF0u };
    if (1 > libxs_utf8_size(a, sizeof(a), 0)) {
      fprintf(stderr, "size reported less than one byte\n");
      result = EXIT_FAILURE;
    }
  }
  return result;
}


/**
 * On WELL-FORMED text the two entry points must agree, and a scan by either must
 * cover the string exactly once - the property every caller iterating text
 * depends on.
 */
static int check_agreement_and_coverage(void)
{
  /* "Maedchen ueber cafe" with real umlauts, plus a typographic apostrophe */
  static const unsigned char text[] = {
    'M', 0xC3u, 0xA4u, 'd', 'c', 'h', 'e', 'n', ' ',
    0xC3u, 0xBCu, 'b', 'e', 'r', ' ', 'c', 'a', 'f', 0xC3u, 0xA9u,
    0xE2u, 0x80u, 0x99u, 's'
  };
  const size_t size = sizeof(text);
  int result = EXIT_SUCCESS;
  size_t pos = 0;
  int ncp = 0;
  while (pos < size && EXIT_SUCCESS == result) {
    int width = 0;
    const size_t step = libxs_utf8_size(text, size, pos);
    libxs_utf8_decode(text + pos, size - pos, &width);
    if (step != (size_t)width) {
      fprintf(stderr, "size and decode disagree at %lu: %lu vs %d\n",
        (unsigned long)pos, (unsigned long)step, width);
      result = EXIT_FAILURE;
    }
    else if (0 == step) {
      fprintf(stderr, "zero step at %lu\n", (unsigned long)pos);
      result = EXIT_FAILURE;
    }
    else {
      pos += step;
      ++ncp;
    }
  }
  if (EXIT_SUCCESS == result && pos != size) {
    fprintf(stderr, "scan overran: %lu of %lu bytes\n",
      (unsigned long)pos, (unsigned long)size);
    result = EXIT_FAILURE;
  }
  /* 24 bytes, 4 of which are 2-byte letters and one a 3-byte quote */
  if (EXIT_SUCCESS == result && 19 != ncp) {
    fprintf(stderr, "counted %d code points, expected 19\n", ncp);
    result = EXIT_FAILURE;
  }
  return result;
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
  if (EXIT_SUCCESS == result) result = check_wellformed();
  if (EXIT_SUCCESS == result) result = check_malformed();
  if (EXIT_SUCCESS == result) result = check_size_is_lenient();
  if (EXIT_SUCCESS == result) result = check_agreement_and_coverage();
  return result;
}
