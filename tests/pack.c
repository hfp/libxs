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
#include <math.h>

#define NMAX 1000
#define SENTINEL 0x5A5A5A5A5A5A5A5AULL


static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}


/* Random integers of nbits bits (signed: sign-extended), stored in the type. */
static void fill(void* values, libxs_data_t type, size_t n, int nbits, int zeros)
{
  const uint64_t mask = (64 > nbits ? ((((uint64_t)1) << nbits) - 1) : ~((uint64_t)0));
  const uint64_t sign = (0 == (1 & LIBXS_TYPEORDER(type)) ? (((uint64_t)1) << (nbits - 1)) : 0);
  size_t i;
  for (i = 0; i < n; ++i) {
    uint64_t v = rng_next() & mask;
    if (0 != zeros && 0 != ((i / 16) & 1)) v = 0; /* every other group of 16 is zero */
    else if (0 != zeros) v >>= (int)(rng_next() % (uint64_t)nbits); /* mixed magnitudes */
    if (0 != sign) v = (v ^ sign) - sign;
    switch (LIBXS_TYPESIZE(type)) {
      case 1: ((uint8_t*)values)[i] = (uint8_t)v; break;
      case 2: ((uint16_t*)values)[i] = (uint16_t)v; break;
      case 4: ((uint32_t*)values)[i] = (uint32_t)v; break;
      default: ((uint64_t*)values)[i] = v;
    }
  }
}


static int check_fixed(libxs_data_t type)
{
  static const size_t counts[] = { 0, 1, 63, 64, 65, 200, NMAX };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  uint64_t values[NMAX], back[NMAX], packed[NMAX + 2];
  int result = EXIT_SUCCESS, nbits;
  size_t ic, i;
  for (nbits = 1; nbits <= 8 * typesize && EXIT_SUCCESS == result; ++nbits) {
    for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
      const size_t n = counts[ic], nwords = (n * (size_t)nbits + 63) / 64;
      fill(values, type, n, nbits, 0);
      for (i = 0; i < NMAX + 2; ++i) packed[i] = SENTINEL;
      memset(back, 0xFF, sizeof(back));
      if (EXIT_SUCCESS != libxs_pack(packed, values, type, n, nbits)
        || EXIT_SUCCESS != libxs_unpack(back, packed, type, n, nbits))
      {
        fprintf(stderr, "fixed: call failed (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
      else if (SENTINEL != packed[nwords]) {
        fprintf(stderr, "fixed: overrun (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
      else if (0 != memcmp(values, back, n * (size_t)typesize)) {
        fprintf(stderr, "fixed: mismatch (type %i, nbits %i, n %i)\n", (int)type, nbits, (int)n);
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


/** Every level (below 1: the default, above the last: the last), counts around groups of 4 and 8, zeros or not. */
static int check_variable(libxs_data_t type)
{
  static const size_t counts[] = { 1, 3, 4, 5, 15, 16, 17, 333, NMAX };
  static const int levels[] = { -1, 0, 1, 2, 3, 9 };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  uint64_t values[NMAX], back[NMAX], packed[2 * NMAX];
  int result = EXIT_SUCCESS, maxbits, il, zeros;
  size_t ic, i;
  for (maxbits = 1; maxbits <= 8 * typesize && EXIT_SUCCESS == result; maxbits += 3) {
    for (il = 0; il < (int)(sizeof(levels) / sizeof(*levels)) && EXIT_SUCCESS == result; ++il) {
      for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
        for (zeros = 0; zeros < 2 && EXIT_SUCCESS == result; ++zeros) {
          const size_t n = counts[ic];
          const size_t bound = libxs_vpack_bound(n, maxbits);
          size_t nwords, nread;
          fill(values, type, n, maxbits, zeros);
          for (i = 0; i < 2 * NMAX; ++i) packed[i] = SENTINEL;
          memset(back, 0xFF, sizeof(back));
          nwords = libxs_vpack(packed, values, type, n, maxbits, levels[il]);
          nread = libxs_vunpack(back, packed, nwords, type, n);
          if (0 == nwords || nwords > bound || nread != nwords) {
            fprintf(stderr, "variable: %i words (bound %i, read %i) for type %i, maxbits %i, level %i, n %i\n",
              (int)nwords, (int)bound, (int)nread, (int)type, maxbits, levels[il], (int)n);
            result = EXIT_FAILURE;
          }
          else if (SENTINEL != packed[nwords]) {
            fprintf(stderr, "variable: overrun (type %i, maxbits %i, level %i, n %i)\n",
              (int)type, maxbits, levels[il], (int)n);
            result = EXIT_FAILURE;
          }
          else if (0 != memcmp(values, back, n * (size_t)typesize)) {
            fprintf(stderr, "variable: mismatch (type %i, maxbits %i, level %i, n %i)\n",
              (int)type, maxbits, levels[il], (int)n);
            result = EXIT_FAILURE;
          }
        }
      }
    }
  }
  return result;
}


static void store(void* values, libxs_data_t type, size_t i, uint64_t v)
{
  switch (LIBXS_TYPESIZE(type)) {
    case 1: ((uint8_t*)values)[i] = (uint8_t)v; break;
    case 2: ((uint16_t*)values)[i] = (uint16_t)v; break;
    case 4: ((uint32_t*)values)[i] = (uint32_t)v; break;
    default: ((uint64_t*)values)[i] = v;
  }
}


/** Groups of 8 with a few values at the full width among small ones (zeros included). */
static int check_outliers(libxs_data_t type, int level)
{
  static const int small[] = { 0, 1, 3, 7 };
  const int typesize = (int)LIBXS_TYPESIZE(type), maxbits = 8 * typesize;
  const uint64_t sign = (0 == (1 & LIBXS_TYPEORDER(type)) ? (((uint64_t)1) << (maxbits - 1)) : 0);
  const uint64_t mask = (64 > maxbits ? ((((uint64_t)1) << maxbits) - 1) : ~((uint64_t)0));
  uint64_t values[NMAX], back[NMAX], packed[2 * NMAX];
  int result = EXIT_SUCCESS, is, k;
  for (is = 0; is < (int)(sizeof(small) / sizeof(*small)) && EXIT_SUCCESS == result; ++is) {
    for (k = 1; k <= 3 && EXIT_SUCCESS == result; ++k) { /* values at the full width per group */
      const size_t n = NMAX - (size_t)k; /* a partial last group */
      const size_t bound = libxs_vpack_bound(n, maxbits);
      size_t i, nwords, nread;
      for (i = 0; i < n; ++i) {
        const size_t lane = (i + (i / 8) * 3) & 7; /* where the wide values are moves per group */
        uint64_t v = (lane < (size_t)k ? (rng_next() & mask) | (sign >> 1)
          : (rng_next() & ((((uint64_t)1) << small[is]) - 1)));
        if (0 != sign && 0 != (rng_next() & 1)) v = (0 - v) & mask; /* either sign */
        store(values, type, i, v);
      }
      for (i = 0; i < 2 * NMAX; ++i) packed[i] = SENTINEL;
      memset(back, 0xFF, sizeof(back));
      nwords = libxs_vpack(packed, values, type, n, maxbits, level);
      nread = libxs_vunpack(back, packed, nwords, type, n);
      if (0 == nwords || nwords > bound || nread != nwords) {
        fprintf(stderr, "outliers: %i words (bound %i, read %i) for type %i, small %i, wide %i\n",
          (int)nwords, (int)bound, (int)nread, (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
      else if (SENTINEL != packed[nwords]) {
        fprintf(stderr, "outliers: overrun (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
      else if (0 != memcmp(values, back, n * (size_t)typesize)) {
        fprintf(stderr, "outliers: mismatch (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
      else if (LIBXS_PACK_QUADS == level && LIBXS_PACK_QUADS != (int)((packed[0] >> 8) & 0xFF)) {
        fprintf(stderr, "outliers: groups of 4 not taken (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


/** All-zero values cost only the header; invalid arguments and empty streams are rejected. */
static int check_edges(void)
{
  uint64_t values[64] = { 0 }, packed[64], back[1] = { SENTINEL };
  int result = EXIT_SUCCESS;
  if (1 != libxs_vpack(packed, values, LIBXS_DATATYPE_I64, 32, 40, LIBXS_PACK_WORDS)) {
    fprintf(stderr, "edges: zeros are not header-only\n");
    result = EXIT_FAILURE;
  }
  else if (0 != libxs_vunpack(back, packed, 0, LIBXS_DATATYPE_I64, 1) || SENTINEL != back[0]) {
    fprintf(stderr, "edges: empty stream not rejected\n");
    result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_F64, 64, 8)
    || EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_I32, 64, 33)
    || EXIT_SUCCESS == libxs_pack(packed, values, LIBXS_DATATYPE_U8, 64, 0)
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_C64, 64, 8, 0)
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_I16, 64, 0, 0))
  {
    fprintf(stderr, "edges: invalid arguments accepted\n");
    result = EXIT_FAILURE;
  }
  return result;
}


/** The level LIBXS_PACK_CODES: runs of zero groups and blocks of nonzero ones, zeros within groups, every type and
 * width; nothing written beyond the words returned. */
static int check_lanes(libxs_data_t type)
{
  static const size_t counts[] = { 1, 7, 8, 9, 63, 64, 65, 333, 1024, 2047, 2048, 2049, 8193, 20005 };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  const size_t nmax = 20005;
  uint64_t *const values = (uint64_t*)malloc(nmax * sizeof(uint64_t));
  uint64_t *const back = (uint64_t*)malloc(nmax * sizeof(uint64_t));
  uint64_t *const packed = (uint64_t*)malloc((2 * nmax + 64) * sizeof(uint64_t));
  int result = (NULL != values && NULL != back && NULL != packed) ? EXIT_SUCCESS : EXIT_FAILURE;
  int maxbits, zeros, nlanes = 0, nlarge = 0;
  size_t ic, i;
  for (maxbits = 1; maxbits <= 8 * typesize && EXIT_SUCCESS == result; maxbits += 5) {
    for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
      for (zeros = 0; zeros < 2 && EXIT_SUCCESS == result; ++zeros) {
        const size_t n = counts[ic], bound = libxs_vpack_bound(n, maxbits);
        size_t nwords, nread;
        fill(values, type, n, maxbits, zeros);
        for (i = 0; i < 2 * nmax + 64; ++i) packed[i] = SENTINEL;
        memset(back, 0xFF, nmax * sizeof(uint64_t));
        nwords = libxs_vpack(packed, values, type, n, maxbits, LIBXS_PACK_CODES);
        nread = libxs_vunpack(back, packed, nwords, type, n);
        if (0 == nwords || nwords > bound || nread != nwords) {
          fprintf(stderr, "lanes: %i words (bound %i, read %i) for type %i, maxbits %i, n %i\n",
            (int)nwords, (int)bound, (int)nread, (int)type, maxbits, (int)n);
          result = EXIT_FAILURE;
        }
        else if (SENTINEL != packed[nwords]) {
          fprintf(stderr, "lanes: overrun (type %i, maxbits %i, n %i)\n", (int)type, maxbits, (int)n);
          result = EXIT_FAILURE;
        }
        else if (0 != memcmp(values, back, n * (size_t)typesize)) {
          fprintf(stderr, "lanes: mismatch (type %i, maxbits %i, n %i)\n", (int)type, maxbits, (int)n);
          result = EXIT_FAILURE;
        }
        else if (0 == ((const unsigned char*)packed)[0]) { /* the level, not the fallback */
          ++nlanes;
          if (LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK < n) ++nlarge; /* more than a block */
        }
      }
    }
  }
  if (EXIT_SUCCESS == result && (0 == nlanes || 0 == nlarge)) {
    fprintf(stderr, "lanes: format not taken (type %i: %i calls, %i beyond a block)\n", (int)type, nlanes, nlarge);
    result = EXIT_FAILURE;
  }
  free(values);
  free(back);
  free(packed);
  return result;
}


/** 1 for Inf, 2 for NaN, else 0: by bits, as a fast floating-point model may assume neither exists. */
static int nonfinite(double v)
{
  uint64_t bits;
  memcpy(&bits, &v, sizeof(bits));
  bits &= 0x7FFFFFFFFFFFFFFFULL;
  return (0x7FF0000000000000ULL > bits ? 0 : (0x7FF0000000000000ULL == bits ? 1 : 2));
}


/** Floating-point data: lossless (bit-exact), relative (maxbits leading bits, rounded), absolute (2^maxbits),
 * with integral values, zeros of either sign, Inf, NaN, subnormals, and large values among them. */
static int check_reals(libxs_data_t type)
{
  static const size_t counts[] = { 1, 7, 8, 9, 63, 333, 1024, 4099 };
  static const int modes64[] = { 64, 40, 20, 5, 0, -10, -30, -60, -2000 };
  static const int modes32[] = { 32, 20, 12, 3, 0, -5, -20, -200 };
  const int f64 = (LIBXS_DATATYPE_F64 == type), width = (0 != f64 ? 64 : 32), expbits = (0 != f64 ? 11 : 8);
  const int* const modes = (0 != f64 ? modes64 : modes32);
  const int nmodes = (0 != f64 ? (int)(sizeof(modes64) / sizeof(*modes64)) : (int)(sizeof(modes32) / sizeof(*modes32)));
  const size_t nmax = 4099;
  double *const x = (double*)malloc(nmax * sizeof(double)), *const y = (double*)malloc(nmax * sizeof(double));
  float *const xf = (float*)malloc(nmax * sizeof(float)), *const yf = (float*)malloc(nmax * sizeof(float));
  uint64_t *const packed = (uint64_t*)malloc((2 * nmax + 64) * sizeof(uint64_t));
  int result = (NULL != x && NULL != y && NULL != xf && NULL != yf && NULL != packed) ? EXIT_SUCCESS : EXIT_FAILURE;
  int kind, im, level, nzruns = 0;
  size_t ic, i;
  for (kind = 0; kind < 5 && EXIT_SUCCESS == result; ++kind) { /* full mantissas, integral, specials among them, */
    for (i = 0; i < nmax; ++i) { /* zeros in runs (leading and trailing ones), almost only zeros */
      const double r = (double)(rng_next() >> 11) / 9007199254740992.0 - 0.5;
      x[i] = ((0 == kind || 3 <= kind) ? ldexp(r, (int)(rng_next() % 40) - 20)
        : (double)((int64_t)(rng_next() % 2001) - 1000));
      if (3 > kind ? (0 == (rng_next() % 5)) : (3 == kind ? (i < 5 || 4090 <= i || 2 > (i / 3) % 5)
        : (0 != (rng_next() % 50))))
      {
        x[i] = 0;
      }
    }
    if (2 == kind) { /* specials */
      const uint64_t inf = 0x7FF0000000000000ULL, nan = 0x7FF8000000000001ULL;
      memcpy(x + 3, &inf, sizeof(double));
      x[5] = -x[3];
      memcpy(x + 6, &nan, sizeof(double));
      x[1] = -0.0;
      x[10] = 4.9e-324;
      x[12] = 1e300;
      x[20] = -1e-310;
    }
    for (i = 0; i < nmax; ++i) xf[i] = (float)x[i];
    for (im = 0; im < nmodes && EXIT_SUCCESS == result; ++im) {
      const int maxbits = modes[im], keep = LIBXS_MIN(LIBXS_MAX(maxbits, 2 + expbits), width);
      const int mbits = keep - 1 - expbits; /* mantissa bits kept */
      for (level = 1; level <= 3 && EXIT_SUCCESS == result; ++level) {
        for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
          const size_t n = counts[ic], bound = libxs_vpack_bound(n, maxbits);
          size_t nwords, nread, k;
          int exact = (width <= maxbits), special = 0;
          for (i = 0; i < 2 * nmax + 64; ++i) packed[i] = SENTINEL;
          memset(y, 0xFF, nmax * sizeof(double));
          memset(yf, 0xFF, nmax * sizeof(float));
          nwords = libxs_vpack(packed, 0 != f64 ? (const void*)x : (const void*)xf, type, n, maxbits, level);
          if (0 != nwords && 0 != ((packed[0] >> 48) & 1)) ++nzruns; /* zeros as runs */
          nread = libxs_vunpack(0 != f64 ? (void*)y : (void*)yf, packed, nwords, type, n);
          for (k = 0; k < n; ++k) {
            const double v = (0 != f64 ? x[k] : (double)xf[k]);
            if (0 != nonfinite(v)) special = 1;
            if (0 != f64 && 0 != v && fabs(v) * ldexp(1.0, -(LIBXS_MAX(maxbits, -1022) + 1)) >= ldexp(1.0, 62)) {
              special = 1;
            }
          }
          if (0 >= maxbits && (0 != special || (0 != f64 ? -1022 : -126) > maxbits)) exact = 1; /* lossless */
          if (0 == nwords || nwords > bound || nread != nwords) {
            fprintf(stderr, "reals: %i words (bound %i, read %i) for type %i, kind %i, maxbits %i, level %i, n %i\n",
              (int)nwords, (int)bound, (int)nread, (int)type, kind, maxbits, level, (int)n);
            result = EXIT_FAILURE;
          }
          else if (SENTINEL != packed[nwords]) {
            fprintf(stderr, "reals: overrun (type %i, kind %i, maxbits %i, level %i, n %i)\n", (int)type, kind, maxbits,
              level, (int)n);
            result = EXIT_FAILURE;
          }
          for (k = 0; k < n && EXIT_SUCCESS == result; ++k) {
            const double v = (0 != f64 ? x[k] : (double)xf[k]), w = (0 != f64 ? y[k] : (double)yf[k]);
            int ok;
            if (0 != exact) {
              ok = (0 != f64 ? (0 == memcmp(x + k, y + k, sizeof(double)))
                : (0 == memcmp(xf + k, yf + k, sizeof(float))));
            }
            else if (2 == nonfinite(v)) ok = (2 == nonfinite(w)); /* NaN stays NaN */
            else if (0 != nonfinite(v)) ok = (0 == memcmp(&v, &w, sizeof(v))); /* Inf stays Inf */
            else if (0 >= maxbits) {
              ok = (fabs(w - v) <= ldexp(1.0, maxbits));
            }
            else { /* relative, or absolute below the normal numbers */
              const double tiny = ldexp(1.0, (0 != f64 ? -1022 : -126) - mbits);
              ok = (fabs(w - v) <= LIBXS_MAX(ldexp(fabs(v), -mbits), tiny));
            }
            if (0 == ok) {
              fprintf(stderr, "reals: value %i is %.17g, not %.17g (type %i, kind %i, maxbits %i, level %i, n %i)\n",
                (int)k, w, v, (int)type, kind, maxbits, level, (int)n);
              result = EXIT_FAILURE;
            }
          }
        }
      }
    }
  }
  if (EXIT_SUCCESS == result && 0 == nzruns) {
    fprintf(stderr, "reals: zeros never as runs (type %i)\n", (int)type);
    result = EXIT_FAILURE;
  }
  free(x);
  free(y);
  free(xf);
  free(yf);
  free(packed);
  return result;
}


/** libxs_vpack_maxbits: the bits for integer magnitudes, the bits within an error, and the error kept. */
static int check_maxbits(void)
{
  static const struct { libxs_data_t type; double value; int relative, expected; } cases[] = {
    { LIBXS_DATATYPE_I8, 127, 0, 8 }, { LIBXS_DATATYPE_I8, 1000, 0, 8 }, { LIBXS_DATATYPE_U8, 255, 0, 8 },
    { LIBXS_DATATYPE_I32, 0, 0, 1 }, { LIBXS_DATATYPE_U32, 0, 0, 1 }, { LIBXS_DATATYPE_I16, -1000, 0, 11 },
    { LIBXS_DATATYPE_U64, 2.5, 0, 2 }, { LIBXS_DATATYPE_I64, 1e30, 0, 64 }, { LIBXS_DATATYPE_U16, 1024, 1, 11 },
    { LIBXS_DATATYPE_F64, 0, 0, 64 }, { LIBXS_DATATYPE_F32, -1, 1, 32 }, { LIBXS_DATATYPE_F64, 5e-12, 0, -38 },
    { LIBXS_DATATYPE_F64, 1.0, 0, 0 }, { LIBXS_DATATYPE_F64, 4.0, 0, 0 }, { LIBXS_DATATYPE_F64, 4.9e-324, 0, -1074 },
    { LIBXS_DATATYPE_F64, 0.5, 1, 13 }, { LIBXS_DATATYPE_F64, 1.0 / 1048576, 1, 31 },
    { LIBXS_DATATYPE_F64, 1.5 / 1048576, 1, 31 }, { LIBXS_DATATYPE_F32, 1.0 / 1048576, 1, 28 },
    { LIBXS_DATATYPE_F64, 1e-30, 1, 64 }, { LIBXS_DATATYPE_F64, 2.0, 1, 13 }, { LIBXS_DATATYPE_C64, 1, 0, 0 }
  };
  enum { N = 1000 };
  double x[N], y[N];
  uint64_t packed[2 * N + 64];
  int result = EXIT_SUCCESS, i, kind;
  for (i = 0; i < (int)(sizeof(cases) / sizeof(*cases)) && EXIT_SUCCESS == result; ++i) {
    const int got = libxs_vpack_maxbits(cases[i].type, cases[i].value, cases[i].relative);
    if (cases[i].expected != got) {
      fprintf(stderr, "maxbits: %i, not %i (type %i, value %g, relative %i)\n", got, cases[i].expected,
        (int)cases[i].type, cases[i].value, cases[i].relative);
      result = EXIT_FAILURE;
    }
  }
  for (i = 0; i < N; ++i) {
    x[i] = ldexp((double)(rng_next() >> 11) / 9007199254740992.0 - 0.5, (int)(rng_next() % 30) - 10);
  }
  for (kind = 0; kind < 4 && EXIT_SUCCESS == result; ++kind) { /* relative and absolute errors kept */
    static const double errors[] = { 1e-6, 3e-14, 1e-9, 0.25 };
    const int relative = (2 > kind), maxbits = libxs_vpack_maxbits(LIBXS_DATATYPE_F64, errors[kind], relative);
    const size_t nwords = libxs_vpack(packed, x, LIBXS_DATATYPE_F64, N, maxbits, 0);
    if (0 == nwords || nwords != libxs_vunpack(y, packed, nwords, LIBXS_DATATYPE_F64, N)) {
      fprintf(stderr, "maxbits: no round trip (error %g, relative %i)\n", errors[kind], relative);
      result = EXIT_FAILURE;
    }
    for (i = 0; i < N && EXIT_SUCCESS == result; ++i) {
      if (fabs(y[i] - x[i]) > errors[kind] * (0 != relative ? fabs(x[i]) : 1.0)) {
        fprintf(stderr, "maxbits: %.17g, not %.17g (error %g, relative %i)\n", y[i], x[i], errors[kind], relative);
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


int main(int argc, char* argv[])
{
  static const libxs_data_t types[] = {
    LIBXS_DATATYPE_I8, LIBXS_DATATYPE_U8, LIBXS_DATATYPE_I16, LIBXS_DATATYPE_U16,
    LIBXS_DATATYPE_I32, LIBXS_DATATYPE_U32, LIBXS_DATATYPE_I64, LIBXS_DATATYPE_U64
  };
  int result, level, i;
  LIBXS_UNUSED(argc);
  LIBXS_UNUSED(argv);
  libxs_init();
  result = check_edges();
  for (i = 0; i < (int)(sizeof(types) / sizeof(*types)) && EXIT_SUCCESS == result; ++i) {
    result = check_fixed(types[i]);
    if (EXIT_SUCCESS == result) result = check_variable(types[i]);
    for (level = LIBXS_PACK_WORDS; level <= LIBXS_PACK_CODES && EXIT_SUCCESS == result; ++level) {
      result = check_outliers(types[i], level);
    }
    if (EXIT_SUCCESS == result) result = check_lanes(types[i]);
  }
  if (EXIT_SUCCESS == result) result = check_maxbits();
  for (i = 0; i < 2 && EXIT_SUCCESS == result; ++i) { /* twice: other values (sizes near the bound) */
    result = check_reals(LIBXS_DATATYPE_F64);
    if (EXIT_SUCCESS == result) result = check_reals(LIBXS_DATATYPE_F32);
  }
  libxs_finalize();
  return result;
}
