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


static int check_variable(libxs_data_t type)
{
  static const size_t counts[] = { 1, 15, 16, 17, 333, NMAX };
  static const int groups[] = { 0, 1, 7, 64, 255 };
  const int typesize = (int)LIBXS_TYPESIZE(type);
  uint64_t values[NMAX], back[NMAX], packed[2 * NMAX];
  int result = EXIT_SUCCESS, maxbits, ig, zeros;
  size_t ic, i;
  for (maxbits = 1; maxbits <= 8 * typesize && EXIT_SUCCESS == result; maxbits += 3) {
    for (ig = 0; ig < (int)(sizeof(groups) / sizeof(*groups)) && EXIT_SUCCESS == result; ++ig) {
      for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
        for (zeros = 0; zeros < 2 && EXIT_SUCCESS == result; ++zeros) {
          const size_t n = counts[ic];
          const size_t bound = libxs_vpack_bound(n, maxbits, groups[ig]);
          size_t nwords, nread;
          fill(values, type, n, maxbits, zeros);
          for (i = 0; i < 2 * NMAX; ++i) packed[i] = SENTINEL;
          memset(back, 0xFF, sizeof(back));
          nwords = libxs_vpack(packed, values, type, n, maxbits, groups[ig]);
          nread = libxs_vunpack(back, packed, nwords, type, n);
          if (0 == nwords || nwords > bound || nread != nwords) {
            fprintf(stderr, "variable: %i words (bound %i, read %i) for type %i, maxbits %i, group %i, n %i\n",
              (int)nwords, (int)bound, (int)nread, (int)type, maxbits, groups[ig], (int)n);
            result = EXIT_FAILURE;
          }
          else if (SENTINEL != packed[nwords]) {
            fprintf(stderr, "variable: overrun (type %i, maxbits %i, group %i, n %i)\n",
              (int)type, maxbits, groups[ig], (int)n);
            result = EXIT_FAILURE;
          }
          else if (0 != memcmp(values, back, n * (size_t)typesize)) {
            fprintf(stderr, "variable: mismatch (type %i, maxbits %i, group %i, n %i)\n",
              (int)type, maxbits, groups[ig], (int)n);
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


/** Groups of 8 with a few values at the full width among small ones (zeros included): exceptions. */
static int check_exceptions(libxs_data_t type)
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
      const size_t bound = libxs_vpack_bound(n, maxbits, 8), ngroups = (n + 7) / 8;
      size_t i, nwords, nread, nexc = 0;
      for (i = 0; i < n; ++i) {
        const size_t lane = (i + (i / 8) * 3) & 7; /* where the wide values are moves per group */
        uint64_t v = (lane < (size_t)k ? (rng_next() & mask) | (sign >> 1)
          : (rng_next() & ((((uint64_t)1) << small[is]) - 1)));
        if (0 != sign && 0 != (rng_next() & 1)) v = (0 - v) & mask; /* either sign */
        store(values, type, i, v);
      }
      for (i = 0; i < 2 * NMAX; ++i) packed[i] = SENTINEL;
      memset(back, 0xFF, sizeof(back));
      nwords = libxs_vpack(packed, values, type, n, maxbits, 8);
      nread = libxs_vunpack(back, packed, nwords, type, n);
      for (i = 0; i < ngroups; ++i) nexc += (0 != (((const unsigned char*)packed)[1 + i] & 0x80) ? 1 : 0);
      if (0 == nwords || nwords > bound || nread != nwords) {
        fprintf(stderr, "exceptions: %i words (bound %i, read %i) for type %i, small %i, wide %i\n",
          (int)nwords, (int)bound, (int)nread, (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
      else if (SENTINEL != packed[nwords]) {
        fprintf(stderr, "exceptions: overrun (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
      else if (0 != memcmp(values, back, n * (size_t)typesize)) {
        fprintf(stderr, "exceptions: mismatch (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
#if !defined(LIBXS_PACK_SW)
      else if (8 == typesize && 0 == nexc) {
        fprintf(stderr, "exceptions: none used (type %i, small %i, wide %i)\n", (int)type, small[is], k);
        result = EXIT_FAILURE;
      }
#endif
      LIBXS_UNUSED(nexc);
    }
  }
  return result;
}


#if !defined(LIBXS_PACK_SW) && (defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2))
/** The AVX-512 and AVX2 encoders of a group of 8 write the header and bytes of the scalar one. */
static int check_exc_simd(void)
{
  const int arch = libxs_cpuid(NULL);
  int result = EXIT_SUCCESS, t;
  for (t = 0; t < 100000 && EXIT_SUCCESS == result; ++t) {
    unsigned char ha = 0, hb = 0, outa[80], outb[80];
    uint64_t in[8], any = 0;
    int i, w;
    for (i = 0; i < 8; ++i) { /* widths of 0 to 64 */
      const int b = (int)(rng_next() % 65);
      in[i] = (64 > b ? (rng_next() & ((((uint64_t)1) << b) - 1)) : rng_next());
      any |= in[i];
    }
    w = (int)LIBXS_NBITS(any);
    if (0 < w) {
      size_t na, nb;
      memset(outa, 0, sizeof(outa));
      na = internal_libxs_vpack8_exc(&ha, outa, in, w);
# if defined(LIBXS_PACK_AVX512)
      if (LIBXS_X86_AVX512 <= arch) {
        memset(outb, 0, sizeof(outb));
        nb = internal_libxs_vpack8_exc_avx512(&hb, outb, in, w);
        if (na != nb || ha != hb || 0 != memcmp(outa, outb, sizeof(outa))) {
          fprintf(stderr, "exceptions: AVX-512 encoder differs (width %i)\n", w);
          result = EXIT_FAILURE;
        }
      }
# endif
# if defined(LIBXS_PACK_AVX2)
      if (LIBXS_X86_AVX2 <= arch && EXIT_SUCCESS == result) {
        memset(outb, 0, sizeof(outb));
        nb = internal_libxs_vpack8_exc_avx2(&hb, outb, in, w);
        if (na != nb || ha != hb || 0 != memcmp(outa, outb, sizeof(outa))) {
          fprintf(stderr, "exceptions: AVX2 encoder differs (width %i)\n", w);
          result = EXIT_FAILURE;
        }
      }
# endif
      LIBXS_UNUSED(nb);
    }
  }
  LIBXS_UNUSED(arch);
  return result;
}
#endif


/** All-zero values cost only the header; invalid arguments and empty streams are rejected. */
static int check_edges(void)
{
  uint64_t values[64] = { 0 }, packed[64], back[1] = { SENTINEL };
  int result = EXIT_SUCCESS;
  if (1 != libxs_vpack(packed, values, LIBXS_DATATYPE_I64, 32, 40, 16)) {
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
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_F32, 64, 8, 0)
    || 0 != libxs_vpack(packed, values, LIBXS_DATATYPE_I16, 64, 8, 256))
  {
    fprintf(stderr, "edges: invalid arguments accepted\n");
    result = EXIT_FAILURE;
  }
  return result;
}


/** The format of LIBXS_VPACK=1 (groups of 8), forced: two streams from LIBXS_PACK_LANES_SPLIT values, zeros in runs and
 * within groups, every type and width; nothing written beyond the words returned. */
static int check_lanes(libxs_data_t type)
{
  static const size_t counts[] = { 1, 7, 8, 9, 63, 64, 65, 333, 1024, 2047, 2048, 2049, 8193, 20005 };
  const int typesize = (int)LIBXS_TYPESIZE(type), format = internal_libxs_vpack_format;
  const size_t nmax = 20005;
  uint64_t *const values = (uint64_t*)malloc(nmax * sizeof(uint64_t));
  uint64_t *const back = (uint64_t*)malloc(nmax * sizeof(uint64_t));
  uint64_t *const packed = (uint64_t*)malloc((2 * nmax + 64) * sizeof(uint64_t));
  int result = (NULL != values && NULL != back && NULL != packed) ? EXIT_SUCCESS : EXIT_FAILURE;
  int maxbits, zeros, nlanes = 0, nlarge = 0;
  size_t ic, i;
  internal_libxs_vpack_format = 1;
  for (maxbits = 1; maxbits <= 8 * typesize && EXIT_SUCCESS == result; maxbits += 5) {
    for (ic = 0; ic < sizeof(counts) / sizeof(*counts) && EXIT_SUCCESS == result; ++ic) {
      for (zeros = 0; zeros < 2 && EXIT_SUCCESS == result; ++zeros) {
        const size_t n = counts[ic], bound = libxs_vpack_bound(n, maxbits, 8);
        size_t nwords, nread;
        fill(values, type, n, maxbits, zeros);
        for (i = 0; i < 2 * nmax + 64; ++i) packed[i] = SENTINEL;
        memset(back, 0xFF, nmax * sizeof(uint64_t));
        nwords = libxs_vpack(packed, values, type, n, maxbits, 8);
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
        else if (0 == ((const unsigned char*)packed)[0]) { /* the format, not the fallback */
          ++nlanes;
          if (LIBXS_PACK_LANES_SPLIT <= n && 2 == ((const unsigned char*)packed)[1]) ++nlarge;
        }
      }
    }
  }
  if (EXIT_SUCCESS == result && (0 == nlanes || 0 == nlarge)) {
    fprintf(stderr, "lanes: format not taken (type %i: %i calls, %i with two streams)\n", (int)type, nlanes, nlarge);
    result = EXIT_FAILURE;
  }
  internal_libxs_vpack_format = format;
  free(values);
  free(back);
  free(packed);
  return result;
}


int main(int argc, char* argv[])
{
  static const libxs_data_t types[] = {
    LIBXS_DATATYPE_I8, LIBXS_DATATYPE_U8, LIBXS_DATATYPE_I16, LIBXS_DATATYPE_U16,
    LIBXS_DATATYPE_I32, LIBXS_DATATYPE_U32, LIBXS_DATATYPE_I64, LIBXS_DATATYPE_U64
  };
  int result, i;
  LIBXS_UNUSED(argc);
  LIBXS_UNUSED(argv);
  libxs_init(); /* reads LIBXS_VPACK before check_lanes forces the format */
  result = check_edges();
  for (i = 0; i < (int)(sizeof(types) / sizeof(*types)) && EXIT_SUCCESS == result; ++i) {
    result = check_fixed(types[i]);
    if (EXIT_SUCCESS == result) result = check_variable(types[i]);
    if (EXIT_SUCCESS == result) result = check_exceptions(types[i]);
    if (EXIT_SUCCESS == result) result = check_lanes(types[i]);
  }
#if !defined(LIBXS_PACK_SW) && (defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2))
  if (EXIT_SUCCESS == result) result = check_exc_simd();
#endif
  libxs_finalize();
  return result;
}
