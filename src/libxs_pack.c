/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_pack.h>
#include <libxs/libxs_utils.h>
#include "libxs_main.h"

/* plain code on 64-bit words (no intrinsics, no exceptions), group sizes that are a multiple of 8
 * as 8 interleaved streams (value 8*c+l in stream l, all at the same bit position): every shift is
 * uniform over lanes */
#if !defined(LIBXS_PACK_SW) && 0
# define LIBXS_PACK_SW
#endif
/* AVX2 decoder (8 permutes per group of 8): measured no faster than the scalar decoder */
#if !defined(LIBXS_PACK_AVX2_UNPACK) && 0
# define LIBXS_PACK_AVX2_UNPACK
#endif
/* values per group of the exceptions format: a bitmap byte (and a vector lane) per value */
#define LIBXS_PACK_EXC_GROUP 8
/* format of LIBXS_VPACK=1: lanes per group, parameter and escape of the codes, values from which a call
 * takes two streams, zero bytes behind the streams (reads of 8 bytes stay inside), streams at most */
#define LIBXS_PACK_LANES 8
#define LIBXS_PACK_LANES_K 2
#define LIBXS_PACK_LANES_ESC 24
#define LIBXS_PACK_LANES_SPLIT 2048
#define LIBXS_PACK_LANES_SLACK 7
#define LIBXS_PACK_LANES_NSUB (2 * (LIBXS_PACK_LANES + 1))
/* groups between two checks of the streams against their capacity (a group writes up to 2 words each) */
#define LIBXS_PACK_LANES_CHECK 32

#define LIBXS_PACK_MASK(NBITS) (64 > (NBITS) \
  ? ((((uint64_t)1) << (NBITS)) - 1) : ~((uint64_t)0))
/* expands M(1) to M(64): a literal width lets the compiler specialize per case */
#define LIBXS_PACK_CASE8(M, B) M(B + 1) M(B + 2) M(B + 3) M(B + 4) \
  M(B + 5) M(B + 6) M(B + 7) M(B + 8)
#define LIBXS_PACK_CASES(M) LIBXS_PACK_CASE8(M, 0) LIBXS_PACK_CASE8(M, 8) \
  LIBXS_PACK_CASE8(M, 16) LIBXS_PACK_CASE8(M, 24) LIBXS_PACK_CASE8(M, 32) \
  LIBXS_PACK_CASE8(M, 40) LIBXS_PACK_CASE8(M, 48) LIBXS_PACK_CASE8(M, 56)
#define LIBXS_PACK_CASE_PACK(N) case N: internal_libxs_pack64(out, in, N); break;
#define LIBXS_PACK_CASE_UNPACK(N) case N: internal_libxs_unpack64(out, in, N); break;
/* Decoding paths compiled (AVX-512, AVX2), and the one the static target covers already:
 * 3 = AVX-512, 2 = AVX2, 1 = scalar. A runtime check is needed only above the static one. */
#if defined(LIBXS_INTRINSICS_INCLUDE) && !defined(__NO_INTRINSICS) && !defined(LIBXS_PACK_SW)
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define LIBXS_PACK_AVX512
# endif
# if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#   define LIBXS_PACK_AVX2
# endif
#endif
#if defined(LIBXS_PACK_AVX512) && (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_PACK_STATIC 3
#elif defined(LIBXS_PACK_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_PACK_STATIC 2
#else
# define LIBXS_PACK_STATIC 1
#endif
#if defined(LIBXS_PACK_AVX512) && (3 > LIBXS_PACK_STATIC)
# define LIBXS_PACK_RUNTIME
#elif defined(LIBXS_PACK_AVX2) && (2 > LIBXS_PACK_STATIC)
# define LIBXS_PACK_RUNTIME
#endif
#define LIBXS_PACK_CASE_VPACK(N) case N: internal_libxs_vpack8(out, in, N); break;
#define LIBXS_PACK_CASE_VUNPACK(N) case N: internal_libxs_vunpack8(out, in, N); break;


typedef struct internal_libxs_pack_writer_t {
  uint64_t* out;
  size_t nwords;
  uint64_t acc;
  int fill;
} internal_libxs_pack_writer_t;

/* A stream of the format of LIBXS_VPACK=1. */
typedef struct internal_libxs_lanes_writer_t {
  uint64_t* out;
  uint64_t acc;
  int fill;
} internal_libxs_lanes_writer_t;

#if defined(LIBXS_PACK_AVX2)
/* Decoder state of four lanes. */
typedef struct internal_libxs_lanes_avx2_t {
  __m256i bit, p1, p2;
} internal_libxs_lanes_avx2_t;
#endif

#if defined(LIBXS_PACK_AVX512)
/* Decoder state of a stream of eight lanes. */
typedef struct internal_libxs_lanes_avx512_t {
  __m512i bit, p1, p2;
  size_t cbit, g, ngroups, run, n;
  int zero;
} internal_libxs_lanes_avx512_t;
#endif


#if defined(LIBXS_PACK_RUNTIME)
/* 0: not determined yet, else as LIBXS_PACK_STATIC */
LIBXS_APIVAR_DEFINE(int internal_libxs_pack_kind);
#endif


/* Size of the type in bytes if it is an integer type, zero otherwise. */
LIBXS_API_INLINE int internal_libxs_pack_typesize(libxs_data_t type)
{
  const int order = LIBXS_TYPEORDER(type);
  return (4 <= order && 11 >= order) ? (int)LIBXS_TYPESIZE(type) : 0;
}


LIBXS_API_INLINE int internal_libxs_pack_signed(libxs_data_t type)
{
  return 0 == (1 & LIBXS_TYPEORDER(type));
}


/* Widens n values starting at offset into dst (signed types sign-extended). */
LIBXS_API_INLINE void internal_libxs_pack_load(uint64_t* LIBXS_RESTRICT dst,
  const void* LIBXS_RESTRICT src, libxs_data_t type, size_t offset, size_t n)
{
  size_t i;
  switch (type) {
    case LIBXS_DATATYPE_I32: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int32_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U32: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint32_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_I16: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int16_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U16: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint16_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_I8: {
      for (i = 0; i < n; ++i) dst[i] = (uint64_t)(int64_t)((const int8_t*)src)[offset + i];
    } break;
    case LIBXS_DATATYPE_U8: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint8_t*)src)[offset + i];
    } break;
    default: {
      for (i = 0; i < n; ++i) dst[i] = ((const uint64_t*)src)[offset + i];
    }
  }
}


/* Narrows n values into dst starting at offset (two's complement truncation). */
LIBXS_API_INLINE void internal_libxs_pack_store(void* LIBXS_RESTRICT dst,
  const uint64_t* LIBXS_RESTRICT src, libxs_data_t type, size_t offset, size_t n)
{
  size_t i;
  switch (LIBXS_TYPESIZE(type)) {
    case 4: {
      for (i = 0; i < n; ++i) ((uint32_t*)dst)[offset + i] = (uint32_t)src[i];
    } break;
    case 2: {
      for (i = 0; i < n; ++i) ((uint16_t*)dst)[offset + i] = (uint16_t)src[i];
    } break;
    case 1: {
      for (i = 0; i < n; ++i) ((uint8_t*)dst)[offset + i] = (uint8_t)src[i];
    } break;
    default: {
      for (i = 0; i < n; ++i) ((uint64_t*)dst)[offset + i] = src[i];
    }
  }
}


/* 64 values into nbits words: each word is assembled in a register and stored once. */
LIBXS_INLINE_ALWAYS void internal_libxs_pack64(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  const uint64_t mask = LIBXS_PACK_MASK(nbits);
  uint64_t acc = 0;
  int fill = 0, nwords = 0, i;
  LIBXS_PRAGMA_UNROLL_N(64)
  for (i = 0; i < 64; ++i) {
    const uint64_t v = in[i] & mask;
    acc |= v << fill;
    fill += nbits;
    if (64 <= fill) {
      out[nwords] = acc;
      ++nwords;
      fill -= 64;
      acc = (0 != fill ? (v >> (nbits - fill)) : 0);
    }
  }
}


LIBXS_INLINE_ALWAYS void internal_libxs_unpack64(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  const uint64_t mask = LIBXS_PACK_MASK(nbits);
  int i;
  LIBXS_PRAGMA_UNROLL_N(64)
  for (i = 0; i < 64; ++i) {
    const int bit = i * nbits, off = bit & 63;
    uint64_t v = in[bit >> 6] >> off;
    if (64 < off + nbits) v |= in[(bit >> 6) + 1] << (64 - off);
    out[i] = v & mask;
  }
}


LIBXS_API_INLINE void internal_libxs_pack_group(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  switch (nbits) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_PACK)
    default:;
  }
}


LIBXS_API_INLINE void internal_libxs_unpack_group(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int nbits)
{
  switch (nbits) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_UNPACK)
    default:;
  }
}


/* 8 values of w bits (no higher bits set) into exactly w bytes. */
LIBXS_INLINE_ALWAYS void internal_libxs_vpack8(unsigned char* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int w)
{
  uint64_t tmp[8], acc = 0;
  int fill = 0, nwords = 0, i;
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const uint64_t v = in[i];
    acc |= v << fill;
    fill += w;
    if (64 <= fill) {
      tmp[nwords] = acc;
      ++nwords;
      fill -= 64;
      acc = (0 != fill ? (v >> (w - fill)) : 0);
    }
  }
  if (0 != fill) tmp[nwords] = acc;
  memcpy(out, tmp, w);
}


LIBXS_INLINE_ALWAYS void internal_libxs_vunpack8(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  const uint64_t mask = LIBXS_PACK_MASK(w);
  uint64_t tmp[8];
  int i;
  tmp[(w - 1) >> 3] = 0; /* bytes beyond w in the last word */
  memcpy(tmp, in, w);
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const int bit = i * w, off = bit & 63;
    uint64_t v = tmp[bit >> 6] >> off;
    if (64 < off + w) v |= tmp[(bit >> 6) + 1] << (64 - off);
    out[i] = v & mask;
  }
}


/* 8 values of w bits (w <= 56) by unaligned loads; reads up to 7 bytes past the w bytes. */
LIBXS_API_INLINE void internal_libxs_vunpack8_load(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  const uint64_t mask = LIBXS_PACK_MASK(w);
  int i;
  LIBXS_PRAGMA_UNROLL_N(8)
  for (i = 0; i < 8; ++i) {
    const int bit = i * w;
    uint64_t x;
    memcpy(&x, in + (bit >> 3), sizeof(x));
    out[i] = (x >> (bit & 7)) & mask;
  }
}


LIBXS_API_INLINE void internal_libxs_vpack8_group(unsigned char* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, int w)
{
  switch (w) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_VPACK)
    default:;
  }
}


LIBXS_API_INLINE void internal_libxs_vunpack8_group(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT in, int w)
{
  switch (w) {
    LIBXS_PACK_CASES(LIBXS_PACK_CASE_VUNPACK)
    default: {
      int i;
      for (i = 0; i < 8; ++i) out[i] = 0;
    }
  }
}


/* Appends the w low bits of v (1 <= w <= 64, no higher bits set). */
LIBXS_API_INLINE void internal_libxs_pack_put(internal_libxs_pack_writer_t* wr,
  uint64_t v, int w)
{
  wr->acc |= v << wr->fill;
  wr->fill += w;
  if (64 <= wr->fill) {
    wr->out[wr->nwords] = wr->acc;
    ++wr->nwords;
    wr->fill -= 64;
    wr->acc = (0 != wr->fill ? (v >> (w - wr->fill)) : 0);
  }
}


LIBXS_API_INLINE void internal_libxs_pack_flush(internal_libxs_pack_writer_t* wr)
{
  if (0 != wr->fill) {
    wr->out[wr->nwords] = wr->acc;
    ++wr->nwords;
    wr->acc = 0;
    wr->fill = 0;
  }
}


/* Reads w bits (1 <= w <= 64) at the given bit position. */
LIBXS_API_INLINE uint64_t internal_libxs_pack_get(const uint64_t* in,
  size_t bit, int w)
{
  const int off = (int)(bit & 63);
  uint64_t v = in[bit >> 6] >> off;
  if (64 < off + w) v |= in[(bit >> 6) + 1] << (64 - off);
  return v & LIBXS_PACK_MASK(w);
}


#if defined(LIBXS_PACK_SW)
/* Appends a column (8 values of w bits, 0 < w) to the 8 interleaved streams at word j and bit off. */
LIBXS_API_INLINE void internal_libxs_vpack8_col(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t* j, int* off, int w)
{
  uint64_t *const lo = out + 8 * *j;
  const int o = *off;
  int l;
  if (0 != o) {
    for (l = 0; l < 8; ++l) lo[l] |= in[l] << o;
  }
  else {
    for (l = 0; l < 8; ++l) lo[l] = in[l];
  }
  if (64 < o + w) { /* spills into the next word of every stream */
    for (l = 0; l < 8; ++l) lo[8 + l] = in[l] >> (64 - o);
  }
  *off = o + w;
  if (64 <= *off) {
    *off -= 64;
    ++*j;
  }
}


/* Reads a column (8 values of w bits) from the 8 interleaved streams at word j and bit off. */
LIBXS_API_INLINE void internal_libxs_vunpack8_col(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t* j, int* off, int w)
{
  const uint64_t *const lo = in + 8 * *j;
  const uint64_t mask = LIBXS_PACK_MASK(w);
  const int o = *off;
  int l;
  if (64 < o + w) {
    for (l = 0; l < 8; ++l) out[l] = ((lo[l] >> o) | (lo[8 + l] << (64 - o))) & mask;
  }
  else if (0 < w) {
    for (l = 0; l < 8; ++l) out[l] = (lo[l] >> o) & mask;
  }
  else {
    for (l = 0; l < 8; ++l) out[l] = 0;
  }
  *off = o + w;
  if (64 <= *off) {
    *off -= 64;
    ++*j;
  }
}
#endif


/* Payload bytes of a byte-aligned stream, rounded to words: unaligned loads stay within. */
LIBXS_API_INLINE size_t internal_libxs_vpack_nbytes(const unsigned char* widths, size_t n, int g)
{
  const size_t ngroups = (n + g - 1) / g;
  size_t i, nbytes = 0;
  for (i = 0; i < ngroups; ++i) nbytes += widths[i] * ((LIBXS_MIN((size_t)g, n - i * g) + 7) / 8);
  return (nbytes + 7) & ~(size_t)7;
}


/* Decodes 8 values (sign unfolded) from a byte-aligned group; fast reads where safe. */
LIBXS_API_INLINE void internal_libxs_vunpack8_any(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT data, size_t pos, size_t nbytes, int w, int is_signed)
{
  int i;
  if (0 < w && 56 >= w && pos + (size_t)w + 7 <= nbytes) internal_libxs_vunpack8_load(out, data + pos, w);
  else internal_libxs_vunpack8_group(out, data + pos, w);
  if (0 != is_signed) {
    for (i = 0; i < 8; ++i) out[i] = (out[i] >> 1) ^ (0 - (out[i] & 1));
  }
}


/* Decodes a byte-aligned stream into 64-bit integers; returns the bits consumed. */
LIBXS_API_INLINE size_t internal_libxs_vunpack64_sw(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g, nbytes = internal_libxs_vpack_nbytes(widths, n, g);
  uint64_t tmp[8];
  size_t i, k, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i];
    for (k = 0; k < m; k += 8) {
      if (k + 8 <= m) internal_libxs_vunpack8_any(out + base + k, data, pos, nbytes, w, is_signed);
      else {
        internal_libxs_vunpack8_any(tmp, data, pos, nbytes, w, is_signed);
        memcpy(out + base + k, tmp, (m - k) * sizeof(*tmp));
      }
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}


#if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_vunpack64_sw, a group of 8 per vector: the w bytes are loaded (masked,
 * hence no read past them), and each lane funnel-shifts its two words into place. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vunpack64_avx512(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g;
  const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i word = _mm512_set1_epi64(63), one = _mm512_set1_epi64(1);
  const __m512i sixtyfour = _mm512_set1_epi64(64);
  size_t i, k, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i];
    const __m512i bits = _mm512_mullo_epi64(lane, _mm512_set1_epi64(w));
    const __m512i lo = _mm512_srli_epi64(bits, 6), hi = _mm512_add_epi64(lo, one);
    const __m512i shift = _mm512_and_si512(bits, word), back = _mm512_sub_epi64(sixtyfour, shift);
    const __m512i mask = _mm512_set1_epi64((long long)LIBXS_PACK_MASK(w));
    const __mmask64 bytes = (64 > w ? ((((__mmask64)1) << w) - 1) : ~((__mmask64)0));
    for (k = 0; k < m; k += 8) {
      const __m512i src = _mm512_maskz_loadu_epi8(bytes, data + pos);
      __m512i x = _mm512_or_si512( /* a shift by 64 yields zero */
        _mm512_srlv_epi64(_mm512_permutexvar_epi64(lo, src), shift),
        _mm512_sllv_epi64(_mm512_permutexvar_epi64(hi, src), back));
      x = _mm512_and_si512(x, mask);
      if (0 != is_signed) {
        x = _mm512_xor_si512(_mm512_srli_epi64(x, 1),
          _mm512_sub_epi64(_mm512_setzero_si512(), _mm512_and_si512(x, one)));
      }
      if (k + 8 <= m) _mm512_storeu_si512((void*)(out + base + k), x);
      else _mm512_mask_storeu_epi64(out + base + k, (__mmask8)((1u << (m - k)) - 1), x);
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}
#endif


#if defined(LIBXS_PACK_AVX2) && defined(LIBXS_PACK_AVX2_UNPACK)
/* As internal_libxs_vunpack64_sw, a group of 8 as two halves: qword loads (masked to the
 * w bytes, rounded up to whole qwords), and a permute across two registers per word. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
size_t internal_libxs_vunpack64_avx2(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g, nbytes = internal_libxs_vpack_nbytes(widths, n, g);
  const __m256i qlane = _mm256_set_epi64x(3, 2, 1, 0), one = _mm256_set1_epi64x(1);
  const __m256i three = _mm256_set1_epi64x(3), six = _mm256_set1_epi64x(6);
  const __m256i odd = _mm256_set1_epi64x(((long long)1) << 32);
  const __m256i word = _mm256_set1_epi64x(63), sixtyfour = _mm256_set1_epi64x(64);
  __m256i plo[2], phi[2], mlo[2], mhi[2], shift[2], back[2];
  uint64_t tmp[8];
  size_t i, k, pos = 0;
  int h;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i], nq = (w + 7) >> 3;
    const __m256i wide = _mm256_set1_epi64x(w);
    const __m256i mask = _mm256_set1_epi64x((long long)LIBXS_PACK_MASK(w));
    const __m256i load0 = _mm256_cmpgt_epi64(_mm256_set1_epi64x(nq), qlane);
    const __m256i load1 = _mm256_cmpgt_epi64(_mm256_set1_epi64x(nq - 4), qlane);
    for (h = 0; h < 2; ++h) { /* per half: word indexes as dword pairs, and which register */
      const __m256i b = _mm256_mul_epu32(_mm256_add_epi64(qlane, _mm256_set1_epi64x(4 * h)), wide);
      const __m256i lo = _mm256_srli_epi64(b, 6), hi = _mm256_add_epi64(lo, one);
      const __m256i tlo = _mm256_and_si256(_mm256_slli_epi64(lo, 1), six); /* even dword */
      const __m256i thi = _mm256_and_si256(_mm256_slli_epi64(hi, 1), six);
      plo[h] = _mm256_or_si256(_mm256_or_si256(tlo, _mm256_slli_epi64(tlo, 32)), odd);
      phi[h] = _mm256_or_si256(_mm256_or_si256(thi, _mm256_slli_epi64(thi, 32)), odd);
      mlo[h] = _mm256_cmpgt_epi64(lo, three);
      mhi[h] = _mm256_cmpgt_epi64(hi, three);
      shift[h] = _mm256_and_si256(b, word);
      back[h] = _mm256_sub_epi64(sixtyfour, shift[h]);
    }
    for (k = 0; k < m; k += 8) {
      if (k + 8 <= m && 0 < w && pos + (size_t)w + 7 <= nbytes) {
        const __m256i q0 = _mm256_maskload_epi64((const long long*)(data + pos), load0);
        const __m256i q1 = _mm256_maskload_epi64((const long long*)(data + pos + 32), load1);
        for (h = 0; h < 2; ++h) {
          const __m256i a = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(q0, plo[h]),
            _mm256_permutevar8x32_epi32(q1, plo[h]), mlo[h]);
          const __m256i c = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(q0, phi[h]),
            _mm256_permutevar8x32_epi32(q1, phi[h]), mhi[h]);
          __m256i x = _mm256_or_si256( /* a shift by 64 yields zero */
            _mm256_srlv_epi64(a, shift[h]), _mm256_sllv_epi64(c, back[h]));
          x = _mm256_and_si256(x, mask);
          if (0 != is_signed) {
            x = _mm256_xor_si256(_mm256_srli_epi64(x, 1),
              _mm256_sub_epi64(_mm256_setzero_si256(), _mm256_and_si256(x, one)));
          }
          _mm256_storeu_si256((__m256i*)(out + base + k + 4 * h), x);
        }
      }
      else if (k + 8 <= m) internal_libxs_vunpack8_any(out + base + k, data, pos, nbytes, w, is_signed);
      else {
        internal_libxs_vunpack8_any(tmp, data, pos, nbytes, w, is_signed);
        memcpy(out + base + k, tmp, (m - k) * sizeof(*tmp));
      }
      pos += (size_t)w;
    }
  }
  return 8 * pos;
}
#endif


/* Decoding path to use (as LIBXS_PACK_STATIC), determined once if needed at runtime. */
LIBXS_API_INLINE int internal_libxs_pack_target(void)
{
#if defined(LIBXS_PACK_RUNTIME)
  int kind = internal_libxs_pack_kind;
  if (0 == kind) {
    const int target_arch = libxs_cpuid(NULL);
    kind = LIBXS_PACK_STATIC;
# if defined(LIBXS_PACK_AVX2)
    if (LIBXS_X86_AVX2 <= target_arch) kind = 2;
# endif
# if defined(LIBXS_PACK_AVX512)
    if (LIBXS_X86_AVX512 <= target_arch) kind = 3;
# endif
    internal_libxs_pack_kind = kind;
  }
#else
  const int kind = LIBXS_PACK_STATIC;
#endif
  return kind;
}


/* Groups of 8 with exceptions, the format unless LIBXS_PACK_SW: a regular width v and the values
 * above it at the full width w (bitmap), whenever this takes fewer bytes than w bits each. */
#if !defined(LIBXS_PACK_SW)
/* Bits set in a byte. */
LIBXS_API_INLINE int internal_libxs_pack_popcount8(unsigned int m)
{
  m = m - ((m >> 1) & 0x55u);
  m = (m & 0x33u) + ((m >> 2) & 0x33u);
  return (int)((m + (m >> 4)) & 0x0Fu);
}


/* Lanes of a group of 8 in order, at v bits or at w bits above v (nb), into words (no branch on
 * the data); writes header byte 0x80|v, bitmap and w-v. */
LIBXS_API_INLINE void internal_libxs_vpack8_exc_put(uint64_t tmp[9], unsigned char* LIBXS_RESTRICT head,
  unsigned char* LIBXS_RESTRICT out, const uint64_t* LIBXS_RESTRICT in, const long long nb[8], int v, int w)
{
  uint64_t acc = 0;
  unsigned int mask = 0;
  size_t k = 0;
  int fill = 0, i;
  for (i = 0; i < 8; ++i) { /* the word being filled stays in a register, stores only */
    const int x = (v < nb[i] ? 1 : 0), f = fill + (0 != x ? w : v), full = (64 <= f ? 1 : 0);
    const uint64_t lo = acc | (in[i] << fill), hi = (in[i] >> 1) >> (63 - fill); /* hi: bits beyond */
    tmp[k] = lo;
    k += (size_t)full;
    acc = (0 != full ? hi : lo);
    fill = f & 63;
    mask |= (unsigned int)x << i;
  }
  tmp[k] = acc;
  *head = (unsigned char)(0x80 | v);
  out[0] = (unsigned char)mask;
  out[1] = (unsigned char)(w - v);
}


/* Group of 8 (zig-zag, w bits): a regular width v (one of the widths) with the values above it at w
 * bits (header byte 0x80|v, then bitmap and w-v), or w bits each, whichever takes fewer bytes. */
LIBXS_API_INLINE size_t internal_libxs_vpack8_exc(unsigned char* LIBXS_RESTRICT head,
  unsigned char* LIBXS_RESTRICT out, const uint64_t* LIBXS_RESTRICT in, int w)
{
  size_t result = (size_t)w;
  long long nb[8];
  int i, j, v = w;
  for (i = 0; i < 8; ++i) nb[i] = (long long)LIBXS_NBITS(in[i]);
  for (j = 0; j < 8; ++j) { /* no branch on the data */
    const int c = (int)nb[j];
    size_t nbytes;
    int e = 0;
    for (i = 0; i < 8; ++i) e += (c < nb[i] ? 1 : 0);
    nbytes = (c < w ? 2 + (size_t)((8 * c + e * (w - c) + 7) >> 3) : (size_t)w);
    v = (nbytes < result ? c : v);
    result = LIBXS_MIN(nbytes, result);
  }
  if (v < w) {
    uint64_t tmp[9];
    internal_libxs_vpack8_exc_put(tmp, head, out, in, nb, v, w);
    memcpy(out + 2, tmp, result - 2);
  }
  else {
    *head = (unsigned char)w;
    internal_libxs_vpack8_group(out, in, w);
  }
  return result;
}


# if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_vpack8_exc (same result): the widths by leading zeros, then the bytes for every
 * candidate v (a lane's width) at once, the exceptions counted over the lanes rotated by 1 to 7. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vpack8_exc_avx512(unsigned char* LIBXS_RESTRICT head,
  unsigned char* LIBXS_RESTRICT out, const uint64_t* LIBXS_RESTRICT in, int w)
{
  const __m512i one = _mm512_set1_epi64(1), wv = _mm512_set1_epi64(w);
  const __m512i nbv = _mm512_sub_epi64(_mm512_set1_epi64(64), _mm512_lzcnt_epi64(_mm512_loadu_si512(in)));
  __m512i e, cost;
  long long nb[8];
  size_t result;
  /* rotations by 1 to 7 (independent): lane l compares with lane l+r */
  e = _mm512_maskz_mov_epi64(_mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 1), nbv), one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 2), nbv), e, one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 3), nbv), e, one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 4), nbv), e, one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 5), nbv), e, one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 6), nbv), e, one);
  e = _mm512_mask_add_epi64(e, _mm512_cmpgt_epi64_mask(_mm512_alignr_epi64(nbv, nbv, 7), nbv), e, one);
  cost = _mm512_add_epi64(_mm512_slli_epi64(nbv, 3), _mm512_mullo_epi64(e, _mm512_sub_epi64(wv, nbv)));
  cost = _mm512_add_epi64(_mm512_set1_epi64(2), _mm512_srli_epi64(_mm512_add_epi64(cost, _mm512_set1_epi64(7)), 3));
  cost = _mm512_mask_blend_epi64(_mm512_cmplt_epi64_mask(nbv, wv), wv, cost);
  result = (size_t)_mm512_reduce_min_epi64(cost);
  _mm512_storeu_si512((void*)nb, nbv);
  if (result < (size_t)w) {
    const int j = (int)LIBXS_INTRINSICS_BITSCANFWD32(
      (unsigned int)_mm512_cmpeq_epi64_mask(cost, _mm512_set1_epi64((long long)result)));
    uint64_t tmp[9];
    internal_libxs_vpack8_exc_put(tmp, head, out, in, nb, (int)nb[j], w);
    _mm512_mask_storeu_epi8(out + 2, (((__mmask64)1) << (result - 2)) - 1, _mm512_loadu_si512(tmp));
  }
  else {
    *head = (unsigned char)w;
    internal_libxs_vpack8_group(out, in, w);
  }
  return result;
}
# endif


# if defined(LIBXS_PACK_AVX2)
/* As internal_libxs_vpack8_exc (same result): the widths as 32-bit lanes, the exceptions counted
 * over the lanes rotated by 1 to 7, then the bytes for every candidate v at once. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
size_t internal_libxs_vpack8_exc_avx2(unsigned char* LIBXS_RESTRICT head,
  unsigned char* LIBXS_RESTRICT out, const uint64_t* LIBXS_RESTRICT in, int w)
{
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), seven = _mm256_set1_epi32(7);
  const __m256i wv = _mm256_set1_epi32(w);
  __m256i nbv, e = _mm256_setzero_si256(), cost, m;
  long long nb[8];
  int nb32[8], i;
  size_t result;
  for (i = 0; i < 8; ++i) {
    nb[i] = (long long)LIBXS_NBITS(in[i]);
    nb32[i] = (int)nb[i];
  }
  nbv = _mm256_loadu_si256((const __m256i*)nb32);
  for (i = 1; i < 8; ++i) { /* lane l compares with lane l+i (independent rotations) */
    const __m256i idx = _mm256_and_si256(_mm256_add_epi32(lane, _mm256_set1_epi32(i)), seven);
    e = _mm256_sub_epi32(e, _mm256_cmpgt_epi32(_mm256_permutevar8x32_epi32(nbv, idx), nbv));
  }
  cost = _mm256_add_epi32(_mm256_slli_epi32(nbv, 3), _mm256_mullo_epi32(e, _mm256_sub_epi32(wv, nbv)));
  cost = _mm256_add_epi32(_mm256_set1_epi32(2), _mm256_srli_epi32(_mm256_add_epi32(cost, seven), 3));
  cost = _mm256_blendv_epi8(wv, cost, _mm256_cmpgt_epi32(wv, nbv));
  m = _mm256_min_epi32(cost, _mm256_permute2x128_si256(cost, cost, 1)); /* the minimum in every lane */
  m = _mm256_min_epi32(m, _mm256_shuffle_epi32(m, 0x4E));
  m = _mm256_min_epi32(m, _mm256_shuffle_epi32(m, 0xB1));
  result = (size_t)_mm256_cvtsi256_si32(m);
  if (result < (size_t)w) {
    const int j = (int)LIBXS_INTRINSICS_BITSCANFWD32(
      (unsigned int)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(cost, m))));
    uint64_t tmp[9];
    internal_libxs_vpack8_exc_put(tmp, head, out, in, nb, (int)nb[j], w);
    memcpy(out + 2, tmp, result - 2);
  }
  else {
    *head = (unsigned char)w;
    internal_libxs_vpack8_group(out, in, w);
  }
  return result;
}
# endif


/* Decodes a group of 8 (sign unfolded) at byte position pos (stream of nbytes); returns its bytes. */
LIBXS_API_INLINE size_t internal_libxs_vunpack8_exc(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT data, size_t pos, size_t nbytes, int hb, int is_signed)
{
  size_t result;
  if (0 == (hb & 0x80)) {
    internal_libxs_vunpack8_any(out, data, pos, nbytes, hb, is_signed);
    result = (size_t)hb;
  }
  else {
    const unsigned int mask = data[pos];
    const int v = hb & 0x7F, w = v + data[pos + 1];
    const size_t nb = (size_t)((8 * v + internal_libxs_pack_popcount8(mask) * (w - v) + 7) / 8);
    size_t bit = 0;
    int i;
    if (57 >= w && pos + 2 + nb + 7 <= nbytes) { /* a load of 8 bytes per value, no branch */
      const unsigned char *const src = data + pos + 2;
      const uint64_t mv = LIBXS_PACK_MASK(v), mw = LIBXS_PACK_MASK(w), ones = 0x0101010101010101ULL;
      /* a byte per lane: its bit of the mask, then the exceptions before it (prefix by multiplying) */
      const uint64_t lanes = ((((uint64_t)mask * ones) & 0x8040201008040201ULL) + 0x7F7F7F7F7F7F7F7FULL) >> 7 & ones;
      const uint64_t before = lanes * ones - lanes;
      for (i = 0; i < 8; ++i) {
        const int x = (int)((lanes >> (8 * i)) & 1);
        const size_t at = (size_t)i * (size_t)v + (size_t)((before >> (8 * i)) & 0xFF) * (size_t)(w - v);
        uint64_t q;
        memcpy(&q, src + (at >> 3), sizeof(q));
        out[i] = (q >> (at & 7)) & (0 != x ? mw : mv);
      }
    }
    else {
      uint64_t tmp[8];
      memset(tmp, 0, sizeof(tmp));
      memcpy(tmp, data + pos + 2, nb);
      for (i = 0; i < 8; ++i) {
        const int wl = (0 != (mask & (1u << i)) ? w : v);
        out[i] = (0 < wl ? internal_libxs_pack_get(tmp, bit, wl) : 0);
        bit += (size_t)wl;
      }
    }
    if (0 != is_signed) {
      for (i = 0; i < 8; ++i) out[i] = (out[i] >> 1) ^ (0 - (out[i] & 1));
    }
    result = 2 + nb;
  }
  return result;
}


LIBXS_API_INLINE size_t internal_libxs_vunpack64_exc_sw(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, size_t nbytes, int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_EXC_GROUP - 1) / LIBXS_PACK_EXC_GROUP;
  uint64_t tmp[LIBXS_PACK_EXC_GROUP];
  size_t i, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = LIBXS_PACK_EXC_GROUP * i;
    const size_t m = LIBXS_MIN((size_t)LIBXS_PACK_EXC_GROUP, n - base);
    if (LIBXS_PACK_EXC_GROUP == m) {
      pos += internal_libxs_vunpack8_exc(out + base, data, pos, nbytes, widths[i], is_signed);
    }
    else {
      pos += internal_libxs_vunpack8_exc(tmp, data, pos, nbytes, widths[i], is_signed);
      memcpy(out + base, tmp, m * sizeof(*tmp));
    }
  }
  return 8 * pos;
}


# if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_vunpack64_exc_sw: the bit position per lane is the prefix sum of the lane
 * widths (v, or w for an exception), then as internal_libxs_vunpack64_avx512. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vunpack64_exc_avx512(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_EXC_GROUP - 1) / LIBXS_PACK_EXC_GROUP;
  const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i word = _mm512_set1_epi64(63), one = _mm512_set1_epi64(1);
  const __m512i sixtyfour = _mm512_set1_epi64(64), ones = _mm512_set1_epi64(-1);
  const __m512i zero = _mm512_setzero_si512();
  size_t i, pos = 0;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = LIBXS_PACK_EXC_GROUP * i;
    const size_t m = LIBXS_MIN((size_t)LIBXS_PACK_EXC_GROUP, n - base);
    const int hb = widths[i], v = hb & 0x7F;
    __m512i wl, bits, lo, hi, shift, back, x;
    __mmask64 bytes;
    size_t nb;
    if (0 == (hb & 0x80)) {
      wl = _mm512_set1_epi64(v);
      bits = _mm512_mullo_epi64(lane, wl);
      nb = (size_t)v;
    }
    else {
      const unsigned int mask = data[pos];
      const int w = v + data[pos + 1];
      __m512i incl;
      wl = _mm512_mask_blend_epi64((__mmask8)mask, _mm512_set1_epi64(v), _mm512_set1_epi64(w));
      incl = _mm512_add_epi64(wl, _mm512_alignr_epi64(wl, zero, 7)); /* lanes up by 1, 2, 4 */
      incl = _mm512_add_epi64(incl, _mm512_alignr_epi64(incl, zero, 6));
      incl = _mm512_add_epi64(incl, _mm512_alignr_epi64(incl, zero, 4));
      bits = _mm512_sub_epi64(incl, wl);
      nb = (size_t)((8 * v + internal_libxs_pack_popcount8(mask) * (w - v) + 7) / 8);
      pos += 2;
    }
    lo = _mm512_srli_epi64(bits, 6);
    hi = _mm512_add_epi64(lo, one);
    shift = _mm512_and_si512(bits, word);
    back = _mm512_sub_epi64(sixtyfour, shift);
    bytes = (64 > nb ? ((((__mmask64)1) << nb) - 1) : ~((__mmask64)0));
    x = _mm512_maskz_loadu_epi8(bytes, data + pos);
    x = _mm512_or_si512( /* a shift by 64 yields zero */
      _mm512_srlv_epi64(_mm512_permutexvar_epi64(lo, x), shift),
      _mm512_sllv_epi64(_mm512_permutexvar_epi64(hi, x), back));
    x = _mm512_and_si512(x, _mm512_srlv_epi64(ones, _mm512_sub_epi64(sixtyfour, wl)));
    if (0 != is_signed) {
      x = _mm512_xor_si512(_mm512_srli_epi64(x, 1), _mm512_sub_epi64(zero, _mm512_and_si512(x, one)));
    }
    if (LIBXS_PACK_EXC_GROUP == m) _mm512_storeu_si512((void*)(out + base), x);
    else _mm512_mask_storeu_epi64(out + base, (__mmask8)((1u << m) - 1), x);
    pos += nb;
  }
  return 8 * pos;
}
# endif


LIBXS_API_INLINE size_t internal_libxs_vunpack64_exc(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, size_t nbytes, int is_signed)
{
  const int kind = internal_libxs_pack_target();
  size_t result;
# if defined(LIBXS_PACK_AVX512)
  if (3 == kind) result = internal_libxs_vunpack64_exc_avx512(out, widths, data, n, is_signed);
  else
# endif
  result = internal_libxs_vunpack64_exc_sw(out, widths, data, n, nbytes, is_signed);
  LIBXS_UNUSED(kind);
  return result;
}
#endif


/* Decodes a byte-aligned stream into 64-bit integers with the best path available. */
LIBXS_API_INLINE size_t internal_libxs_vunpack64(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const unsigned char* LIBXS_RESTRICT data,
  size_t n, int g, int is_signed)
{
  const int kind = internal_libxs_pack_target();
  size_t result;
#if defined(LIBXS_PACK_AVX512)
  if (3 == kind) result = internal_libxs_vunpack64_avx512(out, widths, data, n, g, is_signed);
  else
#endif
#if defined(LIBXS_PACK_AVX2) && defined(LIBXS_PACK_AVX2_UNPACK)
  /* 8 permutes per group (no qword permute across registers) amortize from 16 values on */
  if (2 <= kind && 16 <= g) result = internal_libxs_vunpack64_avx2(out, widths, data, n, g, is_signed);
  else
#endif
  result = internal_libxs_vunpack64_sw(out, widths, data, n, g, is_signed);
  LIBXS_UNUSED(kind);
  return result;
}


/* Appends n bits (at most 64) to a stream, the word being filled stays in a register (no branch). */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_put(internal_libxs_lanes_writer_t* bw, uint64_t v, int n)
{
  const int next = bw->fill + n;
  const uint64_t acc = bw->acc | (v << bw->fill), hi = (v >> 1) >> (63 - bw->fill);
  *bw->out = acc;
  bw->out += next >> 6;
  bw->acc = (64 <= next ? hi : acc);
  bw->fill = next & 63;
}


/* Code s (an escape holds s in 8 bits) and nb bits of x. */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_value(internal_libxs_lanes_writer_t* bw, unsigned int s, uint64_t x, int nb)
{
  const unsigned int m = s >> LIBXS_PACK_LANES_K;
  if (m < LIBXS_PACK_LANES_ESC) {
    const int nc = (int)m + 1 + LIBXS_PACK_LANES_K;
    const uint64_t code = ((((uint64_t)1) << m) - 1)
      | (((uint64_t)(s & ((1U << LIBXS_PACK_LANES_K) - 1))) << (m + 1));
    if (nc + nb <= 64) internal_libxs_lanes_put(bw, code | (x << nc), nc + nb);
    else {
      internal_libxs_lanes_put(bw, code, nc);
      internal_libxs_lanes_put(bw, x, nb);
    }
  }
  else {
    internal_libxs_lanes_put(bw, (((uint64_t)1) << LIBXS_PACK_LANES_ESC) - 1, LIBXS_PACK_LANES_ESC + 1);
    internal_libxs_lanes_put(bw, s, 8);
    internal_libxs_lanes_put(bw, x, nb);
  }
}


/* Length of a run of groups. */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_run(internal_libxs_lanes_writer_t* bw, size_t run)
{
  const int b = (int)LIBXS_NBITS(run + 1);
  internal_libxs_lanes_put(bw, (((uint64_t)1) << (b - 1)) - 1, b);
  internal_libxs_lanes_put(bw, (run + 1) & LIBXS_PACK_MASK(b - 1), b - 1);
}


/* Values of stream s when n values are split into split streams of whole groups. */
LIBXS_API_INLINE size_t internal_libxs_lanes_part(size_t n, int split, int s)
{
  const size_t chunk = (n + (size_t)(LIBXS_PACK_LANES * split) - 1)
    / (size_t)(LIBXS_PACK_LANES * split) * LIBXS_PACK_LANES;
  const size_t start = LIBXS_MIN(n, (size_t)s * chunk);
  return (split - 1 == s ? n : LIBXS_MIN(n, start + chunk)) - start;
}


/* Codes len values from base into the 8 lanes and the runs (bw[0..8]); returns zero once a stream
 * passes its limit (checked every LIBXS_PACK_LANES_CHECK groups). */
LIBXS_API_INLINE int internal_libxs_lanes_code(internal_libxs_lanes_writer_t* bw, uint64_t* const limit[],
  const void* values, libxs_data_t type, size_t base, size_t len, uint64_t mask, uint64_t sign)
{
  const size_t ngroups = (len + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  const int direct = (8 == internal_libxs_pack_typesize(type)); /* 64-bit values are read in place */
  int p1[LIBXS_PACK_LANES], p2[LIBXS_PACK_LANES], l, zero = 1, result = 1;
  size_t i, run = 0;
  for (l = 0; l < LIBXS_PACK_LANES; ++l) p1[l] = p2[l] = 0;
  for (i = 0; i < ngroups && 0 != result; ++i) {
    const size_t m = LIBXS_MIN((size_t)LIBXS_PACK_LANES, len - LIBXS_PACK_LANES * i);
    uint64_t z[LIBXS_PACK_LANES], any = 0;
    const uint64_t* src = z;
    if (0 != direct && LIBXS_PACK_LANES == m) src = (const uint64_t*)values + base + LIBXS_PACK_LANES * i;
    else internal_libxs_pack_load(z, values, type, base + LIBXS_PACK_LANES * i, m);
    for (l = 0; l < LIBXS_PACK_LANES; ++l) {
      uint64_t v = ((size_t)l < m ? (src[l] & mask) : 0);
      if (0 != sign) { /* sign-extend from maxbits, then fold the sign (zig-zag) */
        v = (v ^ sign) - sign;
        v = (v << 1) ^ (0 - (v >> 63));
      }
      z[l] = v;
      any |= v;
    }
    if ((0 == any) != (0 != zero)) {
      internal_libxs_lanes_run(bw + LIBXS_PACK_LANES, run);
      zero = !zero;
      run = 0;
    }
    ++run;
    if (0 != any) {
      for (l = 0; l < LIBXS_PACK_LANES; ++l) {
        if (0 != z[l]) {
          const int b = (int)LIBXS_NBITS(z[l]), d = b - ((p1[l] + p2[l] + 1) >> 1);
          internal_libxs_lanes_value(bw + l, (unsigned int)(0 <= d ? 2 * d : -2 * d - 1) + 1,
            z[l] ^ (((uint64_t)1) << (b - 1)), b - 1);
          p2[l] = p1[l];
          p1[l] = b;
        }
        else internal_libxs_lanes_value(bw + l, 0, 0, 0);
      }
    }
    if (LIBXS_PACK_LANES_CHECK - 1 == (i % LIBXS_PACK_LANES_CHECK)) {
      int over = 0;
      for (l = 0; l <= LIBXS_PACK_LANES; ++l) over |= (bw[l].out >= limit[l] ? 1 : 0);
      if (0 != over) result = 0;
    }
  }
  if (0 != result) internal_libxs_lanes_run(bw + LIBXS_PACK_LANES, run);
  return result;
}


/* Groups of 8 as streams per lane (LIBXS_VPACK=1): each stream into scratch (a share of the bound), then
 * all copied behind a header of their lengths. Writes only the words it returns; zero if the result would
 * not fit into nwords (or on big-endian, or without scratch), so the caller falls back. */
LIBXS_API_INLINE size_t internal_libxs_vpack_lanes(void* packed, const void* values, libxs_data_t type,
  size_t n, int maxbits, size_t nwords)
{
  const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
  const int split = (LIBXS_PACK_LANES_SPLIT <= n ? 2 : 1), nsub = split * (LIBXS_PACK_LANES + 1);
  const uint64_t mask = LIBXS_PACK_MASK(maxbits);
  const uint64_t sign = (0 != internal_libxs_pack_signed(type) ? (((uint64_t)1) << (maxbits - 1)) : 0);
  /* words per stream: a quarter more than an even share of the bound, and the most written between checks */
  const size_t margin = 2 * LIBXS_PACK_LANES_CHECK + 2;
  const size_t cap = (nwords + (size_t)nsub - 1) / (size_t)nsub * 5 / 4 + margin + 2;
  size_t result = 0;
  int pool = 0;
  uint64_t *const tmp = (0 != probe.c[0]
    ? (uint64_t*)internal_libxs_scratch_malloc((size_t)nsub * cap * sizeof(uint64_t), &pool) : NULL);
  if (NULL != tmp) {
    unsigned char *const out = (unsigned char*)packed;
    internal_libxs_lanes_writer_t bw[LIBXS_PACK_LANES_NSUB];
    uint64_t* limit[LIBXS_PACK_LANES_NSUB];
    size_t nbytes[LIBXS_PACK_LANES_NSUB], maxlen = 1, at = 0;
    int s, k, ok = 1;
    for (k = 0; k < nsub; ++k) {
      bw[k].out = tmp + (size_t)k * cap;
      bw[k].acc = 0;
      bw[k].fill = 0;
      limit[k] = bw[k].out + cap - margin;
    }
    for (s = 0; s < split && 0 != ok; ++s) {
      const size_t len = internal_libxs_lanes_part(n, split, s);
      ok = internal_libxs_lanes_code(bw + (LIBXS_PACK_LANES + 1) * s, limit + (LIBXS_PACK_LANES + 1) * s,
        values, type, at, len, mask, sign);
      at += len;
    }
    if (0 != ok) {
      size_t hdr, total, dst, i;
      uint64_t acc = 0;
      int nb, fill = 0;
      for (k = 0; k < nsub; ++k) {
        if (0 != bw[k].fill) *bw[k].out = bw[k].acc;
        nbytes[k] = 8 * (size_t)(bw[k].out - (tmp + (size_t)k * cap)) + (size_t)((bw[k].fill + 7) / 8);
        maxlen = LIBXS_MAX(maxlen, nbytes[k]);
      }
      nb = (int)LIBXS_NBITS(maxlen);
      hdr = 3 + ((size_t)nsub * (size_t)nb + 7) / 8;
      total = hdr + LIBXS_PACK_LANES_SLACK;
      for (k = 0; k < nsub; ++k) total += nbytes[k];
      if ((total + 7) / 8 <= nwords) {
        out[0] = 0; /* no group size: this format */
        out[1] = (unsigned char)split;
        out[2] = (unsigned char)nb;
        for (k = 0, i = 3; k < nsub; ++k) {
          acc |= (uint64_t)nbytes[k] << fill;
          fill += nb;
          for (; 8 <= fill; fill -= 8, acc >>= 8) out[i++] = (unsigned char)acc;
        }
        if (0 < fill) out[i] = (unsigned char)acc;
        for (k = 0, dst = hdr; k < nsub; ++k) {
          memcpy(out + dst, tmp + (size_t)k * cap, nbytes[k]);
          dst += nbytes[k];
        }
        for (; dst < 8 * ((total + 7) / 8); ++dst) out[dst] = 0;
        result = (total + 7) / 8;
      }
    }
    internal_libxs_scratch_free(tmp, pool);
  }
  return result;
}


/* Length of the next run of groups. */
LIBXS_INLINE_ALWAYS size_t internal_libxs_lanes_runlen(const unsigned char* in, size_t* bit)
{
  uint64_t v;
  int b;
  memcpy(&v, in + (*bit >> 3), sizeof(v));
  v >>= *bit & 7;
  b = (int)LIBXS_INTRINSICS_BITSCANFWD64(~v) + 1;
  *bit += (size_t)(2 * b - 1);
  return (size_t)((((uint64_t)1) << (b - 1)) | ((v >> b) & LIBXS_PACK_MASK(b - 1))) - 1;
}


/* Next value of a lane (sign still folded). */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_lanes_get(const unsigned char* in, size_t* bit, uint64_t* p1, uint64_t* p2)
{
  uint64_t v, result = 0;
  unsigned int s;
  int m;
  memcpy(&v, in + (*bit >> 3), sizeof(v));
  v >>= *bit & 7;
  m = (int)LIBXS_INTRINSICS_BITSCANFWD64(~v);
  if (m < LIBXS_PACK_LANES_ESC) {
    s = ((unsigned int)m << LIBXS_PACK_LANES_K) | ((unsigned int)(v >> (m + 1)) & ((1U << LIBXS_PACK_LANES_K) - 1));
    *bit += (size_t)m + 1 + LIBXS_PACK_LANES_K;
  }
  else {
    s = (unsigned int)(v >> (LIBXS_PACK_LANES_ESC + 1)) & 0xFF;
    *bit += LIBXS_PACK_LANES_ESC + 9;
  }
  if (0 != s) {
    const unsigned int u = s - 1;
    const int d = (int)(u >> 1) ^ -(int)(u & 1), b = (int)((*p1 + *p2 + 1) >> 1) + d, nb = b - 1;
    const int lo = LIBXS_MIN(nb, 32);
    uint64_t hi = 0;
    memcpy(&v, in + (*bit >> 3), sizeof(v));
    v = (v >> (*bit & 7)) & LIBXS_PACK_MASK(lo);
    *bit += (size_t)lo;
    if (lo < nb) {
      memcpy(&hi, in + (*bit >> 3), sizeof(hi));
      hi = (hi >> (*bit & 7)) & LIBXS_PACK_MASK(nb - lo);
      *bit += (size_t)(nb - lo);
    }
    result = v | (hi << lo) | (((uint64_t)1) << nb);
    *p2 = *p1;
    *p1 = (uint64_t)b;
  }
  return result;
}


/* Unfolds the sign of a value if is_signed (0 or 1) without a branch. */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_lanes_sign(uint64_t w, int is_signed)
{
  return (w >> is_signed) ^ (0 - (w & (uint64_t)is_signed));
}


/* One stream of n values on the general path, nothing written beyond n. */
LIBXS_API_INLINE void internal_libxs_lanes_decode(uint64_t* q, size_t n, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  uint64_t p1[LIBXS_PACK_LANES], p2[LIBXS_PACK_LANES];
  size_t bit[LIBXS_PACK_LANES], cbit = 8 * offs[LIBXS_PACK_LANES], g = 0;
  int l, zero = 1;
  for (l = 0; l < LIBXS_PACK_LANES; ++l) {
    bit[l] = 8 * offs[l];
    p1[l] = p2[l] = 0;
  }
  while (g < ngroups) {
    const size_t run = internal_libxs_lanes_runlen(in, &cbit);
    if (0 != zero) {
      const size_t i0 = LIBXS_PACK_LANES * g, i1 = LIBXS_MIN(n, LIBXS_PACK_LANES * (g + run));
      memset(q + i0, 0, (i1 - i0) * sizeof(*q));
    }
    else {
      size_t k;
      for (k = 0; k < run * LIBXS_PACK_LANES; ++k) {
        const size_t i = LIBXS_PACK_LANES * g + k;
        const uint64_t v = internal_libxs_lanes_get(in, bit + k % LIBXS_PACK_LANES,
          p1 + k % LIBXS_PACK_LANES, p2 + k % LIBXS_PACK_LANES);
        if (i < n) q[i] = internal_libxs_lanes_sign(v, is_signed);
      }
    }
    g += run;
    zero = !zero;
  }
}


#if defined(LIBXS_PACK_AVX2)
/* Four lanes of a group; a code or a value out of reach takes the general path. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_lanes_step_avx2(internal_libxs_lanes_avx2_t* st, uint64_t* q, const unsigned char* in,
  int is_signed)
{
  const __m256i lut = _mm256_setr_epi8(0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
    0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4);
  const __m256i one = _mm256_set1_epi64x(1), three = _mm256_set1_epi64x(3), four = _mm256_set1_epi64x(4);
  const __m256i nibble = _mm256_set1_epi64x(15), zeros = _mm256_setzero_si256();
  const __m256i sign = _mm256_set1_epi64x(is_signed);
  const __m256i v = _mm256_srlv_epi64(_mm256_i64gather_epi64((const long long*)in, _mm256_srli_epi64(st->bit, 3), 1),
    _mm256_and_si256(st->bit, _mm256_set1_epi64x(7)));
  const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, nibble));
  const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi64(v, 4), nibble));
  const __m256i m = _mm256_add_epi64(lo, _mm256_and_si256(_mm256_cmpeq_epi64(lo, four), hi));
  const __m256i adv = _mm256_add_epi64(m, three);
  const __m256i s = _mm256_or_si256(_mm256_slli_epi64(m, LIBXS_PACK_LANES_K),
    _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_add_epi64(m, one)), three));
  const __m256i zero = _mm256_cmpeq_epi64(s, zeros);
  const __m256i u = _mm256_sub_epi64(s, one);
  const __m256i d = _mm256_xor_si256(_mm256_srli_epi64(u, 1), _mm256_sub_epi64(zeros, _mm256_and_si256(u, one)));
  const __m256i b = _mm256_andnot_si256(zero, _mm256_add_epi64(d,
    _mm256_srli_epi64(_mm256_add_epi64(_mm256_add_epi64(st->p1, st->p2), one), 1)));
  const __m256i nb = _mm256_andnot_si256(zero, _mm256_sub_epi64(b, one));
  const __m256i used = _mm256_add_epi64(adv, nb);
  if (0 == _mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi64(m, _mm256_set1_epi64x(8)),
    _mm256_cmpgt_epi64(used, _mm256_set1_epi64x(57)))))
  {
    const __m256i top = _mm256_sllv_epi64(one, nb);
    const __m256i x = _mm256_andnot_si256(zero, _mm256_or_si256(top,
      _mm256_and_si256(_mm256_srlv_epi64(v, adv), _mm256_sub_epi64(top, one))));
    const __m256i y = _mm256_xor_si256(_mm256_srlv_epi64(x, sign), _mm256_sub_epi64(zeros, _mm256_and_si256(x, sign)));
    _mm256_storeu_si256((__m256i*)q, y);
    st->bit = _mm256_add_epi64(st->bit, used);
    st->p2 = _mm256_blendv_epi8(st->p1, st->p2, zero);
    st->p1 = _mm256_blendv_epi8(b, st->p1, zero);
  }
  else {
    uint64_t bits[4], p1[4], p2[4];
    int l;
    _mm256_storeu_si256((__m256i*)bits, st->bit);
    _mm256_storeu_si256((__m256i*)p1, st->p1);
    _mm256_storeu_si256((__m256i*)p2, st->p2);
    for (l = 0; l < 4; ++l) {
      size_t bit = (size_t)bits[l];
      const uint64_t w = internal_libxs_lanes_get(in, &bit, p1 + l, p2 + l);
      q[l] = internal_libxs_lanes_sign(w, is_signed);
      bits[l] = bit;
    }
    st->bit = _mm256_loadu_si256((const __m256i*)bits);
    st->p1 = _mm256_loadu_si256((const __m256i*)p1);
    st->p2 = _mm256_loadu_si256((const __m256i*)p2);
  }
}


/* As internal_libxs_lanes_decode, a run at a time without calls (the state stays in registers). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_lanes_decode_avx2(uint64_t* q, size_t n, const unsigned char* in, const size_t* offs,
  int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  const size_t tail = n % LIBXS_PACK_LANES;
  internal_libxs_lanes_avx2_t a, b;
  size_t cbit = 8 * offs[LIBXS_PACK_LANES], g = 0;
  int zero = 1;
  a.bit = _mm256_set_epi64x((long long)(8 * offs[3]), (long long)(8 * offs[2]), (long long)(8 * offs[1]),
    (long long)(8 * offs[0]));
  b.bit = _mm256_set_epi64x((long long)(8 * offs[7]), (long long)(8 * offs[6]), (long long)(8 * offs[5]),
    (long long)(8 * offs[4]));
  a.p1 = a.p2 = b.p1 = b.p2 = _mm256_setzero_si256();
  while (g < ngroups) {
    const size_t run = internal_libxs_lanes_runlen(in, &cbit);
    if (0 != zero) {
      const size_t i0 = LIBXS_PACK_LANES * g, i1 = LIBXS_MIN(n, LIBXS_PACK_LANES * (g + run));
      memset(q + i0, 0, (i1 - i0) * sizeof(*q));
    }
    else {
      const size_t k = run - ((g + run == ngroups && 0 != tail) ? 1 : 0);
      size_t j;
      for (j = 0; j < k; ++j) {
        uint64_t *const p = q + LIBXS_PACK_LANES * (g + j);
        internal_libxs_lanes_step_avx2(&a, p, in, is_signed);
        internal_libxs_lanes_step_avx2(&b, p + 4, in, is_signed);
      }
      if (k < run) { /* the last group is incomplete */
        uint64_t t[LIBXS_PACK_LANES];
        internal_libxs_lanes_step_avx2(&a, t, in, is_signed);
        internal_libxs_lanes_step_avx2(&b, t + 4, in, is_signed);
        memcpy(q + LIBXS_PACK_LANES * (g + k), t, tail * sizeof(*t));
      }
    }
    g += run;
    zero = !zero;
  }
}
#endif


#if defined(LIBXS_PACK_AVX512)
/* A group of eight lanes, then the next group; a code or a value out of reach takes the general path. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_step_avx512(internal_libxs_lanes_avx512_t* st, uint64_t* p, const unsigned char* in,
  int is_signed)
{
  const __m512i one = _mm512_set1_epi64(1), three = _mm512_set1_epi64(3), seven = _mm512_set1_epi64(7);
  const __m512i high = _mm512_set1_epi64(63), esc = _mm512_set1_epi64(LIBXS_PACK_LANES_ESC);
  const __m512i reach = _mm512_set1_epi64(57), zeros = _mm512_setzero_si512();
  const __m512i sign = _mm512_set1_epi64(is_signed);
  const __m512i v = _mm512_srlv_epi64(_mm512_i64gather_epi64(_mm512_srli_epi64(st->bit, 3), in, 1),
    _mm512_and_si512(st->bit, seven));
  const __m512i ones = _mm512_andnot_si512(v, _mm512_add_epi64(v, one));
  const __m512i m = _mm512_sub_epi64(high, _mm512_lzcnt_epi64(ones));
  const __m512i s = _mm512_or_si512(_mm512_slli_epi64(m, LIBXS_PACK_LANES_K),
    _mm512_and_si512(_mm512_srlv_epi64(v, _mm512_add_epi64(m, one)), three));
  const __mmask8 nz = _mm512_test_epi64_mask(s, s);
  const __m512i u = _mm512_sub_epi64(s, one);
  const __m512i d = _mm512_xor_si512(_mm512_srli_epi64(u, 1), _mm512_sub_epi64(zeros, _mm512_and_si512(u, one)));
  const __m512i b = _mm512_maskz_add_epi64(nz, d,
    _mm512_srli_epi64(_mm512_add_epi64(_mm512_add_epi64(st->p1, st->p2), one), 1));
  const __m512i nb = _mm512_maskz_sub_epi64(nz, b, one);
  const __m512i adv = _mm512_add_epi64(m, three), used = _mm512_add_epi64(adv, nb);
  if (0 == (_mm512_cmpge_epu64_mask(m, esc) | _mm512_cmpgt_epi64_mask(used, reach))) {
    const __m512i top = _mm512_sllv_epi64(one, nb);
    const __m512i x = _mm512_maskz_or_epi64(nz, top,
      _mm512_and_si512(_mm512_srlv_epi64(v, adv), _mm512_sub_epi64(top, one)));
    const __m512i y = _mm512_xor_si512(_mm512_srlv_epi64(x, sign), _mm512_sub_epi64(zeros, _mm512_and_si512(x, sign)));
    _mm512_storeu_si512(p, y);
    st->bit = _mm512_add_epi64(st->bit, used);
    st->p2 = _mm512_mask_mov_epi64(st->p2, nz, st->p1);
    st->p1 = _mm512_mask_mov_epi64(st->p1, nz, b);
  }
  else {
    uint64_t bits[8], a1[8], a2[8];
    int l;
    _mm512_storeu_si512(bits, st->bit);
    _mm512_storeu_si512(a1, st->p1);
    _mm512_storeu_si512(a2, st->p2);
    for (l = 0; l < 8; ++l) {
      size_t bit = (size_t)bits[l];
      const uint64_t w = internal_libxs_lanes_get(in, &bit, a1 + l, a2 + l);
      p[l] = internal_libxs_lanes_sign(w, is_signed);
      bits[l] = bit;
    }
    st->bit = _mm512_loadu_si512(bits);
    st->p1 = _mm512_loadu_si512(a1);
    st->p2 = _mm512_loadu_si512(a2);
  }
  --st->run;
  ++st->g;
}


LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_init_avx512(internal_libxs_lanes_avx512_t* st, size_t n, const size_t* offs)
{
  st->bit = _mm512_set_epi64((long long)(8 * offs[7]), (long long)(8 * offs[6]), (long long)(8 * offs[5]),
    (long long)(8 * offs[4]), (long long)(8 * offs[3]), (long long)(8 * offs[2]), (long long)(8 * offs[1]),
    (long long)(8 * offs[0]));
  st->p1 = st->p2 = _mm512_setzero_si512();
  st->cbit = 8 * offs[LIBXS_PACK_LANES];
  st->g = st->run = 0;
  st->n = n;
  st->ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  st->zero = 1;
}


/* Reaches the next group of a nonzero run, filling the zero runs on the way; returns 0 at the end. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int internal_libxs_lanes_next_avx512(internal_libxs_lanes_avx512_t* st, uint64_t* q, const unsigned char* in)
{
  while (0 == st->run && st->g < st->ngroups) {
    const size_t run = internal_libxs_lanes_runlen(in, &st->cbit);
    if (0 != st->zero) {
      const size_t i0 = LIBXS_PACK_LANES * st->g, i1 = LIBXS_MIN(st->n, LIBXS_PACK_LANES * (st->g + run));
      memset(q + i0, 0, (i1 - i0) * sizeof(*q));
      st->g += run;
    }
    else st->run = run;
    st->zero = !st->zero;
  }
  return (0 != st->run ? 1 : 0);
}


/* Groups of the current run that can be stored in place: all but an incomplete last group. */
LIBXS_INLINE_ALWAYS size_t internal_libxs_lanes_full_avx512(const internal_libxs_lanes_avx512_t* st)
{
  return st->run - ((st->g + st->run == st->ngroups && 0 != (st->n % LIBXS_PACK_LANES)) ? 1 : 0);
}


/* The rest of a stream, a run at a time without calls (the state stays in registers). */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_finish_avx512(internal_libxs_lanes_avx512_t* st, uint64_t* q, const unsigned char* in,
  int is_signed)
{
  while (0 != internal_libxs_lanes_next_avx512(st, q, in)) {
    const size_t k = internal_libxs_lanes_full_avx512(st);
    size_t j;
    for (j = 0; j < k; ++j) internal_libxs_lanes_step_avx512(st, q + LIBXS_PACK_LANES * st->g, in, is_signed);
    if (0 != st->run) { /* the last group is incomplete */
      uint64_t t[LIBXS_PACK_LANES];
      const size_t i = LIBXS_PACK_LANES * st->g;
      internal_libxs_lanes_step_avx512(st, t, in, is_signed);
      memcpy(q + i, t, (st->n - i) * sizeof(*t));
    }
  }
}


/* One or two streams (the second at q + n[0]); the steps of the two do not depend on each other. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_decode_avx512(uint64_t* q, const size_t n[], int split, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  internal_libxs_lanes_avx512_t a, b;
  internal_libxs_lanes_init_avx512(&a, n[0], offs);
  if (2 == split) {
    uint64_t *const q1 = q + n[0];
    internal_libxs_lanes_init_avx512(&b, n[1], offs + LIBXS_PACK_LANES + 1);
    while (0 != internal_libxs_lanes_next_avx512(&a, q, in) && 0 != internal_libxs_lanes_next_avx512(&b, q1, in)) {
      const size_t k = LIBXS_MIN(internal_libxs_lanes_full_avx512(&a), internal_libxs_lanes_full_avx512(&b));
      size_t j;
      if (0 == k) break; /* an incomplete last group: the rest one by one */
      for (j = 0; j < k; ++j) {
        internal_libxs_lanes_step_avx512(&a, q + LIBXS_PACK_LANES * a.g, in, is_signed);
        internal_libxs_lanes_step_avx512(&b, q1 + LIBXS_PACK_LANES * b.g, in, is_signed);
      }
    }
    internal_libxs_lanes_finish_avx512(&a, q, in, is_signed);
    internal_libxs_lanes_finish_avx512(&b, q1, in, is_signed);
  }
  else internal_libxs_lanes_finish_avx512(&a, q, in, is_signed);
}
#endif


/* Decodes the format of internal_libxs_vpack_lanes; returns the words consumed, zero if invalid. */
LIBXS_API_INLINE size_t internal_libxs_vunpack_lanes(void* values, const void* packed, libxs_data_t type,
  size_t n, int is_signed)
{
  const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
  const unsigned char *const in = (const unsigned char*)packed;
  const int split = in[1], nb = in[2], nsub = split * (LIBXS_PACK_LANES + 1);
  size_t result = 0;
  if (0 != probe.c[0] && 1 <= split && 2 >= split && 0 < nb && 56 >= nb) {
    const size_t hdr = 3 + ((size_t)nsub * (size_t)nb + 7) / 8;
    const int typesize = internal_libxs_pack_typesize(type);
    size_t offs[LIBXS_PACK_LANES_NSUB + 1], len[2] = { 0, 0 };
    uint64_t* q = (uint64_t*)values;
    int pool = 0, k, s;
    offs[0] = hdr;
    for (k = 0; k < nsub; ++k) {
      const size_t bit = 24 + (size_t)k * (size_t)nb;
      uint64_t v;
      memcpy(&v, in + (bit >> 3), sizeof(v));
      offs[k + 1] = offs[k] + (size_t)((v >> (bit & 7)) & LIBXS_PACK_MASK(nb));
    }
    for (s = 0; s < split; ++s) len[s] = internal_libxs_lanes_part(n, split, s);
    if (8 != typesize) q = (uint64_t*)internal_libxs_scratch_malloc(n * sizeof(uint64_t), &pool);
    if (NULL != q) {
#if defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2)
      const int kind = internal_libxs_pack_target();
#endif
#if defined(LIBXS_PACK_AVX512)
      if (3 == kind) internal_libxs_lanes_decode_avx512(q, len, split, in, offs, is_signed);
      else
#endif
      {
        size_t at = 0;
        for (s = 0; s < split; ++s) {
          const size_t* const of = offs + (LIBXS_PACK_LANES + 1) * s;
#if defined(LIBXS_PACK_AVX2)
          if (2 <= kind) internal_libxs_lanes_decode_avx2(q + at, len[s], in, of, is_signed);
          else
#endif
          internal_libxs_lanes_decode(q + at, len[s], in, of, is_signed);
          at += len[s];
        }
      }
      if (8 != typesize) {
        internal_libxs_pack_store(values, q, type, 0, n);
        internal_libxs_scratch_free(q, pool);
      }
      result = (offs[nsub] + LIBXS_PACK_LANES_SLACK + 7) / 8;
    }
  }
  return result;
}


LIBXS_API int libxs_pack(void* packed, const void* values,
  libxs_data_t type, size_t n, int nbits)
{
  const int typesize = internal_libxs_pack_typesize(type);
  int result = EXIT_FAILURE;
  if (0 < typesize && 0 < nbits && nbits <= 8 * typesize
    && ((NULL != packed && NULL != values) || 0 == n))
  {
    uint64_t *const out = (uint64_t*)packed;
    const size_t ngroups = n / 64, rest = n - 64 * ngroups;
    uint64_t buf[64];
    size_t g;
    for (g = 0; g < ngroups; ++g) {
      const uint64_t* in = buf;
      if (8 == typesize) in = (const uint64_t*)values + 64 * g;
      else internal_libxs_pack_load(buf, values, type, 64 * g, 64);
      internal_libxs_pack_group(out + (size_t)nbits * g, in, nbits);
    }
    if (0 != rest) {
      internal_libxs_pack_writer_t wr;
      const uint64_t mask = LIBXS_PACK_MASK(nbits);
      size_t i;
      internal_libxs_pack_load(buf, values, type, 64 * ngroups, rest);
      wr.out = out + (size_t)nbits * ngroups;
      wr.nwords = 0;
      wr.acc = 0;
      wr.fill = 0;
      for (i = 0; i < rest; ++i) internal_libxs_pack_put(&wr, buf[i] & mask, nbits);
      internal_libxs_pack_flush(&wr);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API int libxs_unpack(void* values, const void* packed,
  libxs_data_t type, size_t n, int nbits)
{
  const int typesize = internal_libxs_pack_typesize(type);
  int result = EXIT_FAILURE;
  if (0 < typesize && 0 < nbits && nbits <= 8 * typesize
    && ((NULL != packed && NULL != values) || 0 == n))
  {
    const uint64_t *const in = (const uint64_t*)packed;
    const uint64_t sign = (0 != internal_libxs_pack_signed(type) && 64 > nbits)
      ? (((uint64_t)1) << (nbits - 1)) : 0;
    const size_t ngroups = n / 64, rest = n - 64 * ngroups;
    uint64_t buf[64];
    size_t g, i;
    for (g = 0; g < ngroups; ++g) {
      uint64_t *const out = (8 == typesize ? (uint64_t*)values + 64 * g : buf);
      internal_libxs_unpack_group(out, in + (size_t)nbits * g, nbits);
      if (0 != sign) {
        for (i = 0; i < 64; ++i) out[i] = (out[i] ^ sign) - sign;
      }
      if (8 != typesize) internal_libxs_pack_store(values, buf, type, 64 * g, 64);
    }
    if (0 != rest) {
      const uint64_t *const tail = in + (size_t)nbits * ngroups;
      for (i = 0; i < rest; ++i) {
        const uint64_t v = internal_libxs_pack_get(tail, i * (size_t)nbits, nbits);
        buf[i] = (v ^ sign) - sign;
      }
      internal_libxs_pack_store(values, buf, type, 64 * ngroups, rest);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API size_t libxs_vpack_bound(size_t n, int maxbits, int group)
{
  const int g = (0 == group ? LIBXS_VPACK_GROUP : group);
  size_t result = 0;
  if (0 < n && 0 < maxbits && 64 >= maxbits && 0 < g && 255 >= g) {
    const size_t ngroups = (n + g - 1) / g;
#if defined(LIBXS_PACK_SW) /* 8 interleaved streams, each in whole words */
    result = (1 + ngroups + 7) / 8 + 8 * ((((n + 7) / 8) * (size_t)maxbits + 63) / 64);
#else
    result = (1 + ngroups + 7) / 8 + (((n + 7) / 8) * (size_t)maxbits + 7) / 8;
#endif
  }
  return result;
}


LIBXS_API size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int group)
{
  const int typesize = internal_libxs_pack_typesize(type);
  const int g = (0 == group ? LIBXS_VPACK_GROUP : group);
  const int valid = (0 < typesize && 0 < maxbits && maxbits <= 8 * typesize && 0 < g && 255 >= g
    && 0 < n && NULL != packed && NULL != values);
  size_t result = ((0 != valid && LIBXS_PACK_EXC_GROUP == g && 1 == internal_libxs_vpack_format)
    ? internal_libxs_vpack_lanes(packed, values, type, n, maxbits, libxs_vpack_bound(n, maxbits, g)) : 0);
  if (0 != valid && 0 == result) { /* otherwise, or if it does not fit */
    const size_t ngroups = (n + g - 1) / g, nhead = (1 + ngroups + 7) / 8;
    const uint64_t mask = LIBXS_PACK_MASK(maxbits);
    const uint64_t sign = (0 != internal_libxs_pack_signed(type)
      ? (((uint64_t)1) << (maxbits - 1)) : 0);
    unsigned char *const head = (unsigned char*)packed;
    unsigned char *const data = (unsigned char*)((uint64_t*)packed + nhead);
    internal_libxs_pack_writer_t wr;
    uint64_t buf[255 + 7];
    size_t i, k, nbytes = 0;
#if defined(LIBXS_PACK_SW)
    size_t j = 0;
    int off = 0;
#endif
#if defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2)
    const int kind = internal_libxs_pack_target();
#endif
    head[0] = (unsigned char)g;
    for (i = 1 + ngroups; i < 8 * nhead; ++i) head[i] = 0;
    wr.out = (uint64_t*)packed + nhead;
    wr.nwords = 0;
    wr.acc = 0;
    wr.fill = 0;
    for (i = 0; i < ngroups; ++i) {
      const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
      uint64_t any = 0;
      int w;
      const uint64_t* src = buf;
      if (8 == typesize) src = (const uint64_t*)values + base;
      else internal_libxs_pack_load(buf, values, type, base, m);
      for (k = 0; k < m; ++k) {
        uint64_t v = src[k] & mask;
        if (0 != sign) { /* sign-extend from maxbits, then fold the sign (zig-zag) */
          v = (v ^ sign) - sign;
          v = (v << 1) ^ (0 - (v >> 63));
        }
        buf[k] = v;
        any |= v;
      }
      w = (int)LIBXS_NBITS(any);
      head[1 + i] = (unsigned char)w;
      if (0 < w && 0 == (g & 7)) { /* byte-aligned: 8 values take w bytes */
        for (k = m; k < ((m + 7) & ~(size_t)7); ++k) buf[k] = 0;
#if !defined(LIBXS_PACK_SW)
        if (LIBXS_PACK_EXC_GROUP == g) {
# if defined(LIBXS_PACK_AVX512)
          if (3 == kind) nbytes += internal_libxs_vpack8_exc_avx512(head + 1 + i, data + nbytes, buf, w);
          else
# endif
# if defined(LIBXS_PACK_AVX2)
          if (2 <= kind) nbytes += internal_libxs_vpack8_exc_avx2(head + 1 + i, data + nbytes, buf, w);
          else
# endif
          nbytes += internal_libxs_vpack8_exc(head + 1 + i, data + nbytes, buf, w);
        }
        else
#endif
        for (k = 0; k < m; k += 8) {
#if defined(LIBXS_PACK_SW)
          internal_libxs_vpack8_col(wr.out, buf + k, &j, &off, w);
#else
          internal_libxs_vpack8_group(data + nbytes, buf + k, w);
          nbytes += (size_t)w;
#endif
        }
      }
      else if (0 < w) {
        for (k = 0; k < m; ++k) internal_libxs_pack_put(&wr, buf[k], w);
      }
    }
#if defined(LIBXS_PACK_SW)
    LIBXS_UNUSED(data);
    LIBXS_UNUSED(nbytes);
#endif
#if !defined(LIBXS_PACK_SW)
    if (LIBXS_PACK_EXC_GROUP == g) {
      for (; 0 != (nbytes & 7); ++nbytes) data[nbytes] = 0;
      result = nhead + nbytes / 8;
    }
    else
#endif
#if defined(LIBXS_PACK_SW)
    if (0 == (g & 7)) result = nhead + 8 * (j + (0 != off ? 1 : 0));
#else
    if (0 == (g & 7)) {
      for (; 0 != (nbytes & 7); ++nbytes) data[nbytes] = 0;
      result = nhead + nbytes / 8;
    }
#endif
    else {
      internal_libxs_pack_flush(&wr);
      result = nhead + wr.nwords;
    }
  }
  return result;
}


LIBXS_API size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  const int typesize = internal_libxs_pack_typesize(type);
  size_t result = 0;
  if (0 < typesize && 0 < n && 0 < npacked && NULL != packed && NULL != values) {
    const unsigned char *const head = (const unsigned char*)packed;
    const int g = head[0], is_signed = internal_libxs_pack_signed(type);
    if (0 < g) {
      const size_t ngroups = (n + g - 1) / g, nhead = (1 + ngroups + 7) / 8;
      const uint64_t *const in = (const uint64_t*)packed + nhead;
      const unsigned char *const data = (const unsigned char*)in;
      const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
      uint64_t buf[255 + 7];
      size_t i, k, bit = 0;
#if !defined(NDEBUG)
      { /* the stream fits: the widths are read only once the header fits */
        size_t nbits = 0, j;
        LIBXS_ASSERT(nhead <= npacked);
        for (j = 0; j < ngroups && nhead <= npacked; ++j) {
          const size_t m = LIBXS_MIN((size_t)g, n - j * g);
          nbits += (size_t)head[1 + j] * (0 == (g & 7) ? ((m + 7) & ~(size_t)7) : m);
        }
# if defined(LIBXS_PACK_SW)
        if (0 == (g & 7)) nbits = 8 * 64 * ((nbits / 8 + 63) / 64); /* 8 streams in whole words */
# endif
# if !defined(LIBXS_PACK_SW)
        if (LIBXS_PACK_EXC_GROUP == g) nbits = 0; /* depends on the data: checked after decoding */
# endif
        LIBXS_ASSERT(nhead + (nbits + 63) / 64 <= npacked);
      }
#endif
#if !defined(LIBXS_PACK_SW)
      if (LIBXS_PACK_EXC_GROUP == g) { /* exceptions */
        const size_t nbytes = (nhead < npacked ? 8 * (npacked - nhead) : 0);
        if (8 == typesize && 0 != probe.c[0]) {
          bit = internal_libxs_vunpack64_exc((uint64_t*)values, head + 1, data, n, nbytes, is_signed);
        }
        else {
          size_t pos = 0;
          for (i = 0; i < ngroups; ++i) {
            const size_t base = LIBXS_PACK_EXC_GROUP * i;
            const size_t m = LIBXS_MIN((size_t)LIBXS_PACK_EXC_GROUP, n - base);
            pos += internal_libxs_vunpack8_exc(buf, data, pos, (0 != probe.c[0] ? nbytes : 0), head[1 + i],
              is_signed);
            internal_libxs_pack_store(values, buf, type, base, m);
          }
          bit = 8 * pos;
        }
        LIBXS_ASSERT(nhead + (bit + 63) / 64 <= npacked);
      }
      else
#endif
#if defined(LIBXS_PACK_SW)
      if (0 == (g & 7)) { /* 8 interleaved streams */
        size_t j = 0;
        int off = 0;
        for (i = 0; i < ngroups; ++i) {
          const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
          uint64_t *const dst = (8 == typesize ? (uint64_t*)values + base : buf);
          const int w = head[1 + i];
          for (k = 0; k < m; k += 8) {
            if (k + 8 <= m) internal_libxs_vunpack8_col(dst + k, in, &j, &off, w);
            else {
              uint64_t tmp[8];
              internal_libxs_vunpack8_col(tmp, in, &j, &off, w);
              memcpy(dst + k, tmp, (m - k) * sizeof(*tmp));
            }
          }
          if (0 != is_signed) {
            for (k = 0; k < m; ++k) dst[k] = (dst[k] >> 1) ^ (0 - (dst[k] & 1));
          }
          if (8 != typesize) internal_libxs_pack_store(values, buf, type, base, m);
        }
        bit = 64 * 8 * (j + (0 != off ? 1 : 0));
      }
      else
#endif
      if (0 == (g & 7) && 8 == typesize && 0 != probe.c[0]) { /* byte-aligned into 64-bit */
        bit = internal_libxs_vunpack64((uint64_t*)values, head + 1, data, n, g, is_signed);
      }
      else {
        const size_t nbytes = (0 == (g & 7) ? internal_libxs_vpack_nbytes(head + 1, n, g) : 0);
        for (i = 0; i < ngroups; ++i) {
          const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
          const int w = head[1 + i];
          if (0 == (g & 7)) { /* byte-aligned: 8 values take w bytes */
            for (k = 0; k < m; k += 8) {
              internal_libxs_vunpack8_any(buf + k, data, bit / 8, (0 != probe.c[0] ? nbytes : 0), w, 0);
              bit += 8 * (size_t)w;
            }
          }
          else {
            for (k = 0; k < m; ++k) {
              buf[k] = 0;
              if (0 < w) {
                buf[k] = internal_libxs_pack_get(in, bit, w);
                bit += w;
              }
            }
          }
          if (0 != is_signed) {
            for (k = 0; k < m; ++k) buf[k] = (buf[k] >> 1) ^ (0 - (buf[k] & 1));
          }
          internal_libxs_pack_store(values, buf, type, base, m);
        }
      }
      result = nhead + (bit + 63) / 64;
    }
    else result = internal_libxs_vunpack_lanes(values, packed, type, n, is_signed);
    LIBXS_ASSERT(result <= npacked);
  }
  return result;
}
