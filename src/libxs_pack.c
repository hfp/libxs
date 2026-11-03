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

/* plain code (no intrinsics): the general paths of every format */
#if !defined(LIBXS_PACK_SW) && 0
# define LIBXS_PACK_SW
#endif
/* formats of libxs_vpack (LIBXS_PACK, 0 or unset: LIBXS_PACK_DEFAULT): group sizes that are a multiple
 * of 8 as 8 interleaved streams of words (value 8*c+l in stream l, all at the same bit position, so every
 * shift is uniform over lanes) as samples/pack/vpack.f writes them; as bytes, a group of 8 with
 * exceptions; or a group of 8 as codes of the lengths (internal_libxs_vpack_lanes, else as bytes) */
#define LIBXS_PACK_WORDS 1
#define LIBXS_PACK_BYTES 2
#define LIBXS_PACK_CODES 3
#define LIBXS_PACK_DEFAULT LIBXS_PACK_WORDS
/* AVX2 decoder (8 permutes per group of 8): measured no faster than the scalar decoder */
#if !defined(LIBXS_PACK_AVX2_UNPACK) && 0
# define LIBXS_PACK_AVX2_UNPACK
#endif
/* values per group of the exceptions format: a bitmap byte (and a vector lane) per value */
#define LIBXS_PACK_EXC_GROUP 8
/* format LIBXS_PACK_CODES: lanes per group (a predictor each), parameter and escape of the codes,
 * streams, zero bytes behind the streams (reads of 8 bytes stay inside), groups per decoded block */
#define LIBXS_PACK_LANES 8
#define LIBXS_PACK_LANES_K 2
#define LIBXS_PACK_LANES_ESC 24
#define LIBXS_PACK_LANES_NSUB 5
#define LIBXS_PACK_LANES_SLACK 7
#define LIBXS_PACK_LANES_BLOCK 64
/* groups between two checks of the streams against their capacity */
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

/* A stream of the format LIBXS_PACK_CODES. */
typedef struct internal_libxs_lanes_writer_t {
  uint64_t* out;
  uint64_t acc;
  int fill;
} internal_libxs_lanes_writer_t;

/* Unary codes ending in a byte: their values (one byte each), how many, and the ones left at its end. */
typedef struct internal_libxs_lanes_lut_t {
  uint64_t list;
  unsigned char cnt, trail;
} internal_libxs_lanes_lut_t;

/* Decoder state: the streams, bit positions in them, and the unary parts of a block. */
typedef struct internal_libxs_lanes_reader_t {
  const unsigned char* in[LIBXS_PACK_LANES_NSUB];
  size_t cbit, have, ubyte, group, ebit, pbit;
  unsigned int carry;
  unsigned char m[LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK + 16];
} internal_libxs_lanes_reader_t;


/* Byte x of a stream of unary codes (bit 0 first, a zero ends a code), see internal_libxs_lanes_lut_t. */
static const internal_libxs_lanes_lut_t internal_libxs_lanes_lut[256] = {
  { 0x0000000000000000ULL, 8, 0 }, { 0x0000000000000001ULL, 7, 0 }, { 0x0000000000000100ULL, 7, 0 },
  { 0x0000000000000002ULL, 6, 0 }, { 0x0000000000010000ULL, 7, 0 }, { 0x0000000000000101ULL, 6, 0 },
  { 0x0000000000000200ULL, 6, 0 }, { 0x0000000000000003ULL, 5, 0 }, { 0x0000000001000000ULL, 7, 0 },
  { 0x0000000000010001ULL, 6, 0 }, { 0x0000000000010100ULL, 6, 0 }, { 0x0000000000000102ULL, 5, 0 },
  { 0x0000000000020000ULL, 6, 0 }, { 0x0000000000000201ULL, 5, 0 }, { 0x0000000000000300ULL, 5, 0 },
  { 0x0000000000000004ULL, 4, 0 }, { 0x0000000100000000ULL, 7, 0 }, { 0x0000000001000001ULL, 6, 0 },
  { 0x0000000001000100ULL, 6, 0 }, { 0x0000000000010002ULL, 5, 0 }, { 0x0000000001010000ULL, 6, 0 },
  { 0x0000000000010101ULL, 5, 0 }, { 0x0000000000010200ULL, 5, 0 }, { 0x0000000000000103ULL, 4, 0 },
  { 0x0000000002000000ULL, 6, 0 }, { 0x0000000000020001ULL, 5, 0 }, { 0x0000000000020100ULL, 5, 0 },
  { 0x0000000000000202ULL, 4, 0 }, { 0x0000000000030000ULL, 5, 0 }, { 0x0000000000000301ULL, 4, 0 },
  { 0x0000000000000400ULL, 4, 0 }, { 0x0000000000000005ULL, 3, 0 }, { 0x0000010000000000ULL, 7, 0 },
  { 0x0000000100000001ULL, 6, 0 }, { 0x0000000100000100ULL, 6, 0 }, { 0x0000000001000002ULL, 5, 0 },
  { 0x0000000100010000ULL, 6, 0 }, { 0x0000000001000101ULL, 5, 0 }, { 0x0000000001000200ULL, 5, 0 },
  { 0x0000000000010003ULL, 4, 0 }, { 0x0000000101000000ULL, 6, 0 }, { 0x0000000001010001ULL, 5, 0 },
  { 0x0000000001010100ULL, 5, 0 }, { 0x0000000000010102ULL, 4, 0 }, { 0x0000000001020000ULL, 5, 0 },
  { 0x0000000000010201ULL, 4, 0 }, { 0x0000000000010300ULL, 4, 0 }, { 0x0000000000000104ULL, 3, 0 },
  { 0x0000000200000000ULL, 6, 0 }, { 0x0000000002000001ULL, 5, 0 }, { 0x0000000002000100ULL, 5, 0 },
  { 0x0000000000020002ULL, 4, 0 }, { 0x0000000002010000ULL, 5, 0 }, { 0x0000000000020101ULL, 4, 0 },
  { 0x0000000000020200ULL, 4, 0 }, { 0x0000000000000203ULL, 3, 0 }, { 0x0000000003000000ULL, 5, 0 },
  { 0x0000000000030001ULL, 4, 0 }, { 0x0000000000030100ULL, 4, 0 }, { 0x0000000000000302ULL, 3, 0 },
  { 0x0000000000040000ULL, 4, 0 }, { 0x0000000000000401ULL, 3, 0 }, { 0x0000000000000500ULL, 3, 0 },
  { 0x0000000000000006ULL, 2, 0 }, { 0x0001000000000000ULL, 7, 0 }, { 0x0000010000000001ULL, 6, 0 },
  { 0x0000010000000100ULL, 6, 0 }, { 0x0000000100000002ULL, 5, 0 }, { 0x0000010000010000ULL, 6, 0 },
  { 0x0000000100000101ULL, 5, 0 }, { 0x0000000100000200ULL, 5, 0 }, { 0x0000000001000003ULL, 4, 0 },
  { 0x0000010001000000ULL, 6, 0 }, { 0x0000000100010001ULL, 5, 0 }, { 0x0000000100010100ULL, 5, 0 },
  { 0x0000000001000102ULL, 4, 0 }, { 0x0000000100020000ULL, 5, 0 }, { 0x0000000001000201ULL, 4, 0 },
  { 0x0000000001000300ULL, 4, 0 }, { 0x0000000000010004ULL, 3, 0 }, { 0x0000010100000000ULL, 6, 0 },
  { 0x0000000101000001ULL, 5, 0 }, { 0x0000000101000100ULL, 5, 0 }, { 0x0000000001010002ULL, 4, 0 },
  { 0x0000000101010000ULL, 5, 0 }, { 0x0000000001010101ULL, 4, 0 }, { 0x0000000001010200ULL, 4, 0 },
  { 0x0000000000010103ULL, 3, 0 }, { 0x0000000102000000ULL, 5, 0 }, { 0x0000000001020001ULL, 4, 0 },
  { 0x0000000001020100ULL, 4, 0 }, { 0x0000000000010202ULL, 3, 0 }, { 0x0000000001030000ULL, 4, 0 },
  { 0x0000000000010301ULL, 3, 0 }, { 0x0000000000010400ULL, 3, 0 }, { 0x0000000000000105ULL, 2, 0 },
  { 0x0000020000000000ULL, 6, 0 }, { 0x0000000200000001ULL, 5, 0 }, { 0x0000000200000100ULL, 5, 0 },
  { 0x0000000002000002ULL, 4, 0 }, { 0x0000000200010000ULL, 5, 0 }, { 0x0000000002000101ULL, 4, 0 },
  { 0x0000000002000200ULL, 4, 0 }, { 0x0000000000020003ULL, 3, 0 }, { 0x0000000201000000ULL, 5, 0 },
  { 0x0000000002010001ULL, 4, 0 }, { 0x0000000002010100ULL, 4, 0 }, { 0x0000000000020102ULL, 3, 0 },
  { 0x0000000002020000ULL, 4, 0 }, { 0x0000000000020201ULL, 3, 0 }, { 0x0000000000020300ULL, 3, 0 },
  { 0x0000000000000204ULL, 2, 0 }, { 0x0000000300000000ULL, 5, 0 }, { 0x0000000003000001ULL, 4, 0 },
  { 0x0000000003000100ULL, 4, 0 }, { 0x0000000000030002ULL, 3, 0 }, { 0x0000000003010000ULL, 4, 0 },
  { 0x0000000000030101ULL, 3, 0 }, { 0x0000000000030200ULL, 3, 0 }, { 0x0000000000000303ULL, 2, 0 },
  { 0x0000000004000000ULL, 4, 0 }, { 0x0000000000040001ULL, 3, 0 }, { 0x0000000000040100ULL, 3, 0 },
  { 0x0000000000000402ULL, 2, 0 }, { 0x0000000000050000ULL, 3, 0 }, { 0x0000000000000501ULL, 2, 0 },
  { 0x0000000000000600ULL, 2, 0 }, { 0x0000000000000007ULL, 1, 0 }, { 0x0000000000000000ULL, 7, 1 },
  { 0x0000000000000001ULL, 6, 1 }, { 0x0000000000000100ULL, 6, 1 }, { 0x0000000000000002ULL, 5, 1 },
  { 0x0000000000010000ULL, 6, 1 }, { 0x0000000000000101ULL, 5, 1 }, { 0x0000000000000200ULL, 5, 1 },
  { 0x0000000000000003ULL, 4, 1 }, { 0x0000000001000000ULL, 6, 1 }, { 0x0000000000010001ULL, 5, 1 },
  { 0x0000000000010100ULL, 5, 1 }, { 0x0000000000000102ULL, 4, 1 }, { 0x0000000000020000ULL, 5, 1 },
  { 0x0000000000000201ULL, 4, 1 }, { 0x0000000000000300ULL, 4, 1 }, { 0x0000000000000004ULL, 3, 1 },
  { 0x0000000100000000ULL, 6, 1 }, { 0x0000000001000001ULL, 5, 1 }, { 0x0000000001000100ULL, 5, 1 },
  { 0x0000000000010002ULL, 4, 1 }, { 0x0000000001010000ULL, 5, 1 }, { 0x0000000000010101ULL, 4, 1 },
  { 0x0000000000010200ULL, 4, 1 }, { 0x0000000000000103ULL, 3, 1 }, { 0x0000000002000000ULL, 5, 1 },
  { 0x0000000000020001ULL, 4, 1 }, { 0x0000000000020100ULL, 4, 1 }, { 0x0000000000000202ULL, 3, 1 },
  { 0x0000000000030000ULL, 4, 1 }, { 0x0000000000000301ULL, 3, 1 }, { 0x0000000000000400ULL, 3, 1 },
  { 0x0000000000000005ULL, 2, 1 }, { 0x0000010000000000ULL, 6, 1 }, { 0x0000000100000001ULL, 5, 1 },
  { 0x0000000100000100ULL, 5, 1 }, { 0x0000000001000002ULL, 4, 1 }, { 0x0000000100010000ULL, 5, 1 },
  { 0x0000000001000101ULL, 4, 1 }, { 0x0000000001000200ULL, 4, 1 }, { 0x0000000000010003ULL, 3, 1 },
  { 0x0000000101000000ULL, 5, 1 }, { 0x0000000001010001ULL, 4, 1 }, { 0x0000000001010100ULL, 4, 1 },
  { 0x0000000000010102ULL, 3, 1 }, { 0x0000000001020000ULL, 4, 1 }, { 0x0000000000010201ULL, 3, 1 },
  { 0x0000000000010300ULL, 3, 1 }, { 0x0000000000000104ULL, 2, 1 }, { 0x0000000200000000ULL, 5, 1 },
  { 0x0000000002000001ULL, 4, 1 }, { 0x0000000002000100ULL, 4, 1 }, { 0x0000000000020002ULL, 3, 1 },
  { 0x0000000002010000ULL, 4, 1 }, { 0x0000000000020101ULL, 3, 1 }, { 0x0000000000020200ULL, 3, 1 },
  { 0x0000000000000203ULL, 2, 1 }, { 0x0000000003000000ULL, 4, 1 }, { 0x0000000000030001ULL, 3, 1 },
  { 0x0000000000030100ULL, 3, 1 }, { 0x0000000000000302ULL, 2, 1 }, { 0x0000000000040000ULL, 3, 1 },
  { 0x0000000000000401ULL, 2, 1 }, { 0x0000000000000500ULL, 2, 1 }, { 0x0000000000000006ULL, 1, 1 },
  { 0x0000000000000000ULL, 6, 2 }, { 0x0000000000000001ULL, 5, 2 }, { 0x0000000000000100ULL, 5, 2 },
  { 0x0000000000000002ULL, 4, 2 }, { 0x0000000000010000ULL, 5, 2 }, { 0x0000000000000101ULL, 4, 2 },
  { 0x0000000000000200ULL, 4, 2 }, { 0x0000000000000003ULL, 3, 2 }, { 0x0000000001000000ULL, 5, 2 },
  { 0x0000000000010001ULL, 4, 2 }, { 0x0000000000010100ULL, 4, 2 }, { 0x0000000000000102ULL, 3, 2 },
  { 0x0000000000020000ULL, 4, 2 }, { 0x0000000000000201ULL, 3, 2 }, { 0x0000000000000300ULL, 3, 2 },
  { 0x0000000000000004ULL, 2, 2 }, { 0x0000000100000000ULL, 5, 2 }, { 0x0000000001000001ULL, 4, 2 },
  { 0x0000000001000100ULL, 4, 2 }, { 0x0000000000010002ULL, 3, 2 }, { 0x0000000001010000ULL, 4, 2 },
  { 0x0000000000010101ULL, 3, 2 }, { 0x0000000000010200ULL, 3, 2 }, { 0x0000000000000103ULL, 2, 2 },
  { 0x0000000002000000ULL, 4, 2 }, { 0x0000000000020001ULL, 3, 2 }, { 0x0000000000020100ULL, 3, 2 },
  { 0x0000000000000202ULL, 2, 2 }, { 0x0000000000030000ULL, 3, 2 }, { 0x0000000000000301ULL, 2, 2 },
  { 0x0000000000000400ULL, 2, 2 }, { 0x0000000000000005ULL, 1, 2 }, { 0x0000000000000000ULL, 5, 3 },
  { 0x0000000000000001ULL, 4, 3 }, { 0x0000000000000100ULL, 4, 3 }, { 0x0000000000000002ULL, 3, 3 },
  { 0x0000000000010000ULL, 4, 3 }, { 0x0000000000000101ULL, 3, 3 }, { 0x0000000000000200ULL, 3, 3 },
  { 0x0000000000000003ULL, 2, 3 }, { 0x0000000001000000ULL, 4, 3 }, { 0x0000000000010001ULL, 3, 3 },
  { 0x0000000000010100ULL, 3, 3 }, { 0x0000000000000102ULL, 2, 3 }, { 0x0000000000020000ULL, 3, 3 },
  { 0x0000000000000201ULL, 2, 3 }, { 0x0000000000000300ULL, 2, 3 }, { 0x0000000000000004ULL, 1, 3 },
  { 0x0000000000000000ULL, 4, 4 }, { 0x0000000000000001ULL, 3, 4 }, { 0x0000000000000100ULL, 3, 4 },
  { 0x0000000000000002ULL, 2, 4 }, { 0x0000000000010000ULL, 3, 4 }, { 0x0000000000000101ULL, 2, 4 },
  { 0x0000000000000200ULL, 2, 4 }, { 0x0000000000000003ULL, 1, 4 }, { 0x0000000000000000ULL, 3, 5 },
  { 0x0000000000000001ULL, 2, 5 }, { 0x0000000000000100ULL, 2, 5 }, { 0x0000000000000002ULL, 1, 5 },
  { 0x0000000000000000ULL, 2, 6 }, { 0x0000000000000001ULL, 1, 6 }, { 0x0000000000000000ULL, 1, 7 },
  { 0x0000000000000000ULL, 0, 8 }
};

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


/* Format of libxs_vpack: LIBXS_PACK (1 to 3) as found by the first call, else LIBXS_PACK_DEFAULT. */
LIBXS_API_INLINE int internal_libxs_vpack_format(void)
{
  int result = internal_libxs_pack_format;
  if (0 == result) {
    const char *const env = getenv("LIBXS_PACK");
    result = (NULL != env ? atoi(env) : 0);
    if (LIBXS_PACK_WORDS > result || LIBXS_PACK_CODES < result) result = LIBXS_PACK_DEFAULT;
    internal_libxs_pack_format = result;
  }
  return result;
}


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


/* Bits set in a byte (of the bitmap of groups of 8 with exceptions, LIBXS_PACK_BYTES: a regular width v
 * and the values above it at the full width w, whenever this takes fewer bytes than w bits each). */
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


/* Reads 8 interleaved streams (LIBXS_PACK_WORDS) into 64-bit integers; returns the bits consumed.
 * Every lane of a column takes the same shift: plain code, vectorized per target by the compiler. */
LIBXS_INLINE_ALWAYS size_t internal_libxs_vunpack64_words_body(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g;
  const uint64_t sign = (uint64_t)is_signed;
  size_t i, k, j = 0;
  int off = 0, l;
  for (i = 0; i < ngroups; ++i) {
    const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
    const int w = widths[i];
    const uint64_t mask = LIBXS_PACK_MASK(w);
    for (k = 0; k < m; k += 8) {
      const uint64_t *const lo = in + 8 * j;
      uint64_t tmp[8], *const dst = (k + 8 <= m ? out + base + k : tmp);
      if (64 < off + w) { /* spills into the next word of every stream */
        LIBXS_PRAGMA_SIMD
        for (l = 0; l < 8; ++l) {
          const uint64_t v = ((lo[l] >> off) | (lo[8 + l] << (64 - off))) & mask;
          dst[l] = (v >> sign) ^ (0 - (v & sign));
        }
      }
      else if (0 < w) {
        LIBXS_PRAGMA_SIMD
        for (l = 0; l < 8; ++l) {
          const uint64_t v = (lo[l] >> off) & mask;
          dst[l] = (v >> sign) ^ (0 - (v & sign));
        }
      }
      else {
        for (l = 0; l < 8; ++l) dst[l] = 0;
      }
      if (dst == tmp) memcpy(out + base + k, tmp, (m - k) * sizeof(*tmp));
      off += w;
      if (64 <= off) {
        off -= 64;
        ++j;
      }
    }
  }
  return 64 * 8 * (j + (0 != off ? 1 : 0));
}


LIBXS_API_INLINE size_t internal_libxs_vunpack64_words_sw(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  return internal_libxs_vunpack64_words_body(out, widths, in, n, g, is_signed);
}


#if defined(LIBXS_PACK_AVX2)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
size_t internal_libxs_vunpack64_words_avx2(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  return internal_libxs_vunpack64_words_body(out, widths, in, n, g, is_signed);
}
#endif


#if defined(LIBXS_PACK_AVX512)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vunpack64_words_avx512(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  return internal_libxs_vunpack64_words_body(out, widths, in, n, g, is_signed);
}
#endif


/* Decodes 8 interleaved streams into 64-bit integers with the widest code the CPU runs. */
LIBXS_API_INLINE size_t internal_libxs_vunpack64_words(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  size_t result;
#if defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2)
  const int kind = internal_libxs_pack_target();
#endif
#if defined(LIBXS_PACK_AVX512)
  if (3 == kind) result = internal_libxs_vunpack64_words_avx512(out, widths, in, n, g, is_signed);
  else
#endif
#if defined(LIBXS_PACK_AVX2)
  if (2 == kind) result = internal_libxs_vunpack64_words_avx2(out, widths, in, n, g, is_signed);
  else
#endif
  result = internal_libxs_vunpack64_words_sw(out, widths, in, n, g, is_signed);
  return result;
}


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


/* Length of a run of groups. */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_run(internal_libxs_lanes_writer_t* bw, size_t run)
{
  const int b = (int)LIBXS_NBITS(run + 1);
  internal_libxs_lanes_put(bw, (((uint64_t)1) << (b - 1)) - 1, b);
  internal_libxs_lanes_put(bw, (run + 1) & LIBXS_PACK_MASK(b - 1), b - 1);
}


/* Codes n values into the streams (bw[0..4]); returns zero once a stream passes its limit
 * (checked every LIBXS_PACK_LANES_CHECK groups). The writers are local (registers). */
LIBXS_API_INLINE int internal_libxs_lanes_code(internal_libxs_lanes_writer_t* bw, uint64_t* const limit[],
  const void* values, libxs_data_t type, size_t n, uint64_t mask, uint64_t sign)
{
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  const int direct = (8 == internal_libxs_pack_typesize(type)); /* 64-bit values are read in place */
  internal_libxs_lanes_writer_t wc = bw[0], wu = bw[1], wr = bw[2], we = bw[3], wp = bw[4];
  int p1[LIBXS_PACK_LANES], p2[LIBXS_PACK_LANES], l, zero = 1, result = 1;
  size_t i, run = 0;
  for (l = 0; l < LIBXS_PACK_LANES; ++l) p1[l] = p2[l] = 0;
  for (i = 0; i < ngroups && 0 != result; ++i) {
    const size_t m = LIBXS_MIN((size_t)LIBXS_PACK_LANES, n - LIBXS_PACK_LANES * i);
    uint64_t z[LIBXS_PACK_LANES], any = 0;
    const uint64_t* src = z;
    if (0 != direct && LIBXS_PACK_LANES == m) src = (const uint64_t*)values + LIBXS_PACK_LANES * i;
    else internal_libxs_pack_load(z, values, type, LIBXS_PACK_LANES * i, m);
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
      internal_libxs_lanes_run(&wc, run);
      zero = !zero;
      run = 0;
    }
    ++run;
    if (0 != any) {
      uint64_t ucode = 0;
      unsigned int rem = 0;
      int ubits = 0;
      for (l = 0; l < LIBXS_PACK_LANES; ++l) {
        unsigned int s = 0, u;
        if (0 != z[l]) {
          const int b = (int)LIBXS_NBITS(z[l]), d = b - ((p1[l] + p2[l] + 1) >> 1);
          s = (unsigned int)(0 <= d ? 2 * d : -2 * d - 1) + 1;
          internal_libxs_lanes_put(&wp, z[l] ^ (((uint64_t)1) << (b - 1)), b - 1);
          p2[l] = p1[l];
          p1[l] = b;
        }
        u = s >> LIBXS_PACK_LANES_K;
        if (u < LIBXS_PACK_LANES_ESC) {
          if (64 < ubits + (int)u + 1) {
            internal_libxs_lanes_put(&wu, ucode, ubits);
            ucode = 0;
            ubits = 0;
          }
          ucode |= ((((uint64_t)1) << u) - 1) << ubits;
          ubits += (int)u + 1;
        }
        else { /* escape: the unary part stops at LIBXS_PACK_LANES_ESC, its value follows in 6 bits */
          internal_libxs_lanes_put(&wu, ucode, ubits);
          internal_libxs_lanes_put(&wu, (((uint64_t)1) << LIBXS_PACK_LANES_ESC) - 1, LIBXS_PACK_LANES_ESC + 1);
          internal_libxs_lanes_put(&we, u, 6);
          ucode = 0;
          ubits = 0;
        }
        rem |= (s & ((1U << LIBXS_PACK_LANES_K) - 1)) << (LIBXS_PACK_LANES_K * l);
      }
      internal_libxs_lanes_put(&wu, ucode, ubits);
      internal_libxs_lanes_put(&wr, rem, LIBXS_PACK_LANES_K * LIBXS_PACK_LANES);
    }
    if (LIBXS_PACK_LANES_CHECK - 1 == (i % LIBXS_PACK_LANES_CHECK)
      && (wu.out >= limit[1] || we.out >= limit[3] || wp.out >= limit[4]))
    {
      result = 0;
    }
  }
  if (0 != result) internal_libxs_lanes_run(&wc, run);
  bw[0] = wc; bw[1] = wu; bw[2] = wr; bw[3] = we; bw[4] = wp;
  return result;
}


/* Groups of 8 (LIBXS_PACK_CODES) as five streams: runs of zero and nonzero groups, the unary parts of the
 * codes of the lengths, their remainders, escapes, and the payload. Each stream goes into scratch, then
 * all are copied behind a header of their lengths. Writes only the words it returns; zero if the result
 * would not fit into nwords (or on big-endian, or without scratch), so the caller falls back. */
LIBXS_API_INLINE size_t internal_libxs_vpack_lanes(void* packed, const void* values, libxs_data_t type,
  size_t n, int maxbits, size_t nwords)
{
  const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  const uint64_t mask = LIBXS_PACK_MASK(maxbits);
  const uint64_t sign = (0 != internal_libxs_pack_signed(type) ? (((uint64_t)1) << (maxbits - 1)) : 0);
  /* the most written between two checks (a group writes up to 8 words of payload) */
  const size_t margin = 8 * LIBXS_PACK_LANES_CHECK + 8;
  /* words per stream: runs and remainders at most, the others a share of the bound */
  size_t cap[LIBXS_PACK_LANES_NSUB], size = 0, result = 0;
  int pool = 0, k;
  uint64_t* tmp = NULL;
  cap[0] = (3 * ngroups + 3 + 63) / 64 + 2;
  cap[1] = nwords / 2 + margin;
  cap[2] = (2 * LIBXS_PACK_LANES * ngroups + 63) / 64 + 2;
  cap[3] = nwords / 8 + margin;
  cap[4] = nwords + margin;
  for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) size += cap[k];
  if (0 != probe.c[0]) tmp = (uint64_t*)internal_libxs_scratch_malloc(size * sizeof(uint64_t), &pool);
  if (NULL != tmp) {
    unsigned char *const out = (unsigned char*)packed;
    internal_libxs_lanes_writer_t bw[LIBXS_PACK_LANES_NSUB];
    uint64_t *start[LIBXS_PACK_LANES_NSUB], *limit[LIBXS_PACK_LANES_NSUB];
    size_t nbytes[LIBXS_PACK_LANES_NSUB], maxlen = 1, at = 0;
    for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) {
      start[k] = bw[k].out = tmp + at;
      bw[k].acc = 0;
      bw[k].fill = 0;
      limit[k] = start[k] + cap[k] - (0 == k || 2 == k ? 0 : margin);
      at += cap[k];
    }
    if (0 != internal_libxs_lanes_code(bw, limit, values, type, n, mask, sign)) {
      size_t hdr, total, dst, i;
      uint64_t acc = 0;
      int nb, fill = 0;
      for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) {
        if (0 != bw[k].fill) *bw[k].out = bw[k].acc;
        nbytes[k] = 8 * (size_t)(bw[k].out - start[k]) + (size_t)((bw[k].fill + 7) / 8);
        maxlen = LIBXS_MAX(maxlen, nbytes[k]);
      }
      nb = (int)LIBXS_NBITS(maxlen);
      hdr = 2 + (LIBXS_PACK_LANES_NSUB * (size_t)nb + 7) / 8;
      total = hdr + LIBXS_PACK_LANES_SLACK;
      for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) total += nbytes[k];
      if ((total + 7) / 8 <= nwords) {
        out[0] = 0; /* no group size: this format */
        out[1] = (unsigned char)nb;
        for (k = 0, i = 2; k < LIBXS_PACK_LANES_NSUB; ++k) {
          acc |= (uint64_t)nbytes[k] << fill;
          fill += nb;
          for (; 8 <= fill; fill -= 8, acc >>= 8) out[i++] = (unsigned char)acc;
        }
        if (0 < fill) out[i] = (unsigned char)acc;
        for (k = 0, dst = hdr; k < LIBXS_PACK_LANES_NSUB; ++k) {
          memcpy(out + dst, start[k], nbytes[k]);
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


/* Unary parts of the codes until "need" are known (a byte at a time, up to eight codes per byte). */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_unary(internal_libxs_lanes_reader_t* rd, size_t need)
{
  const unsigned char *const in = rd->in[1];
  unsigned char *const m = rd->m;
  size_t k = rd->have, b = rd->ubyte;
  unsigned int c = rd->carry;
  while (k < need) {
    const internal_libxs_lanes_lut_t e = internal_libxs_lanes_lut[in[b++]];
    const unsigned int keep = 0U - (unsigned int)(0 == e.cnt); /* no code ends: the ones carry on */
    const uint64_t w = e.list + (c & ~keep);
    memcpy(m + k, &w, sizeof(w));
    k += e.cnt;
    c = e.trail + (c & keep);
  }
  rd->have = k;
  rd->ubyte = b;
  rd->carry = c;
}


/* Codes of the next group: one byte per lane (unary part and remainder, escapes resolved). */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_lanes_codes(internal_libxs_lanes_reader_t* rd, size_t j)
{
  uint64_t m8, r;
  uint16_t r16;
  memcpy(&m8, rd->m + j, sizeof(m8));
  memcpy(&r16, rd->in[2] + 2 * rd->group, sizeof(r16));
  ++rd->group;
  r = r16; /* spread 2-bit remainders into bytes */
  r = (r | (r << 24)) & 0x000000FF000000FFULL;
  r = (r | (r << 12)) & 0x000F000F000F000FULL;
  r = (r | (r << 6)) & 0x0303030303030303ULL;
  if (0 != ((m8 + 0x6868686868686868ULL) & 0x8080808080808080ULL)) { /* an escape (unary part 24) */
    int l;
    for (l = 0; l < LIBXS_PACK_LANES; ++l) {
      if (LIBXS_PACK_LANES_ESC == ((m8 >> (8 * l)) & 0xFF)) {
        uint64_t v;
        memcpy(&v, rd->in[3] + (rd->ebit >> 3), sizeof(v));
        m8 = (m8 & ~(((uint64_t)0xFF) << (8 * l))) | (((v >> (rd->ebit & 7)) & 63) << (8 * l));
        rd->ebit += 6;
      }
    }
  }
  return (m8 << LIBXS_PACK_LANES_K) | r;
}


/* A value of length b (zero: the value is zero) at a payload position. */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_lanes_value(const unsigned char* in, size_t pos, int b, int is_signed)
{
  const int nb = (0 != b ? b - 1 : 0);
  uint64_t v, x;
  memcpy(&v, in + (pos >> 3), sizeof(v));
  v >>= pos & 7;
  if (56 < nb) {
    uint64_t hi;
    memcpy(&hi, in + ((pos + 56) >> 3), sizeof(hi));
    v |= (hi >> ((pos + 56) & 7)) << 56;
  }
  x = (v & LIBXS_PACK_MASK(nb)) | (((uint64_t)(0 != b)) << nb);
  return (x >> is_signed) ^ (0 - (x & (uint64_t)is_signed));
}


/* The reader at the start of the streams. */
LIBXS_INLINE_ALWAYS void internal_libxs_lanes_begin(internal_libxs_lanes_reader_t* rd, const unsigned char* in,
  const size_t* offs)
{
  int k;
  for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) rd->in[k] = in + offs[k];
  rd->cbit = rd->have = rd->ubyte = rd->group = rd->ebit = rd->pbit = 0;
  rd->carry = 0;
}


/* Where a block of values goes: in place if whole and 64-bit, else the buffer (then stored). */
LIBXS_INLINE_ALWAYS uint64_t* internal_libxs_lanes_dst(void* values, int typesize, size_t at, size_t nv, size_t n,
  uint64_t* buffer)
{
  return ((8 == typesize && at + nv <= n) ? (uint64_t*)values + at : buffer);
}


LIBXS_INLINE_ALWAYS void internal_libxs_lanes_done(void* values, libxs_data_t type, size_t at, size_t nv, size_t n,
  const uint64_t* dst, const uint64_t* buffer)
{
  if (dst == buffer) internal_libxs_pack_store(values, buffer, type, at, LIBXS_MIN(nv, n - at));
}


LIBXS_INLINE_ALWAYS void internal_libxs_lanes_zeros(void* values, int typesize, size_t at, size_t nv, size_t n)
{
  memset((char*)values + at * (size_t)typesize, 0, LIBXS_MIN(nv, n - at) * (size_t)typesize);
}


LIBXS_INLINE_ALWAYS void internal_libxs_lanes_next(internal_libxs_lanes_reader_t* rd, size_t nv)
{
  memmove(rd->m, rd->m + nv, rd->have - nv);
  rd->have -= nv;
}


/* Blocks of a nonzero run: the unary parts of a block first, then lengths and payload a group at a time. */
LIBXS_API_INLINE void internal_libxs_lanes_decode(void* values, libxs_data_t type, size_t n, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  const int typesize = internal_libxs_pack_typesize(type);
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  internal_libxs_lanes_reader_t rd;
  uint64_t buffer[LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK];
  int p1[LIBXS_PACK_LANES], p2[LIBXS_PACK_LANES], l, zero = 1;
  size_t g = 0;
  internal_libxs_lanes_begin(&rd, in, offs);
  for (l = 0; l < LIBXS_PACK_LANES; ++l) p1[l] = p2[l] = 0;
  while (g < ngroups) {
    const size_t run = internal_libxs_lanes_runlen(rd.in[0], &rd.cbit);
    size_t h;
    for (h = 0; h < run; h += LIBXS_PACK_LANES_BLOCK) {
      const size_t at = LIBXS_PACK_LANES * (g + h), nv = LIBXS_PACK_LANES * LIBXS_MIN(LIBXS_PACK_LANES_BLOCK, run - h);
      if (0 == zero) {
        uint64_t *const q = internal_libxs_lanes_dst(values, typesize, at, nv, n, buffer);
        size_t j;
        internal_libxs_lanes_unary(&rd, nv);
        for (j = 0; j < nv; j += LIBXS_PACK_LANES) {
          const uint64_t s8 = internal_libxs_lanes_codes(&rd, j);
          for (l = 0; l < LIBXS_PACK_LANES; ++l) {
            const unsigned int s = (unsigned int)(s8 >> (8 * l)) & 0xFF;
            int b = 0;
            if (0 != s) {
              const unsigned int u = s - 1;
              b = ((p1[l] + p2[l] + 1) >> 1) + ((int)(u >> 1) ^ -(int)(u & 1));
              p2[l] = p1[l];
              p1[l] = b;
            }
            q[j + l] = internal_libxs_lanes_value(rd.in[4], rd.pbit, b, is_signed);
            rd.pbit += (size_t)(0 != b ? b - 1 : 0);
          }
        }
        internal_libxs_lanes_next(&rd, nv);
        internal_libxs_lanes_done(values, type, at, nv, n, q, buffer);
      }
      else internal_libxs_lanes_zeros(values, typesize, at, nv, n);
    }
    g += run;
    zero = !zero;
  }
}


#if defined(LIBXS_PACK_AVX2)
/* Lengths of a group (eight lanes in one register) from its codes. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX2)
__m256i internal_libxs_lanes_lengths_avx2(uint64_t s8, __m256i* p1, __m256i* p2)
{
  const __m256i one = _mm256_set1_epi32(1), zeros = _mm256_setzero_si256();
  const __m256i s = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)s8));
  const __m256i nz = _mm256_cmpgt_epi32(s, zeros), u = _mm256_sub_epi32(s, one);
  const __m256i d = _mm256_xor_si256(_mm256_srli_epi32(u, 1), _mm256_sub_epi32(zeros, _mm256_and_si256(u, one)));
  const __m256i b = _mm256_and_si256(nz, _mm256_add_epi32(d,
    _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(*p1, *p2), one), 1)));
  *p2 = _mm256_blendv_epi8(*p2, *p1, nz);
  *p1 = _mm256_blendv_epi8(*p1, b, nz);
  return b;
}


/* Four values of lengths b (64-bit lanes) from the payload position base on; base moves past them. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_lanes_payload_avx2(uint64_t* q, __m256i b, __m256i* base, const unsigned char* in, int is_signed)
{
  const __m256i one = _mm256_set1_epi64x(1), seven = _mm256_set1_epi64x(7), zeros = _mm256_setzero_si256();
  const __m256i sign = _mm256_set1_epi64x(is_signed);
  const __m256i nz = _mm256_cmpgt_epi64(b, zeros), nb = _mm256_and_si256(nz, _mm256_sub_epi64(b, one));
  const __m256i x1 = _mm256_add_epi64(nb, _mm256_slli_si256(nb, 8));
  const __m256i incl = _mm256_add_epi64(x1, _mm256_blend_epi32(zeros, _mm256_permute4x64_epi64(x1, 0x55), 0xF0));
  const __m256i pos = _mm256_add_epi64(*base, _mm256_sub_epi64(incl, nb));
  const __m256i sh = _mm256_and_si256(pos, seven);
  if (0 == _mm256_movemask_epi8(_mm256_cmpgt_epi64(_mm256_add_epi64(nb, sh), _mm256_set1_epi64x(64)))) {
    const __m256i v = _mm256_srlv_epi64(_mm256_i64gather_epi64((const long long*)in, _mm256_srli_epi64(pos, 3), 1), sh);
    const __m256i top = _mm256_sllv_epi64(one, nb);
    const __m256i x = _mm256_or_si256(_mm256_and_si256(v, _mm256_sub_epi64(top, one)), _mm256_and_si256(nz, top));
    _mm256_storeu_si256((__m256i*)q,
      _mm256_xor_si256(_mm256_srlv_epi64(x, sign), _mm256_sub_epi64(zeros, _mm256_and_si256(x, sign))));
  }
  else { /* a value out of reach of one load */
    uint64_t p[4], w[4];
    int l;
    _mm256_storeu_si256((__m256i*)p, pos);
    _mm256_storeu_si256((__m256i*)w, b);
    for (l = 0; l < 4; ++l) q[l] = internal_libxs_lanes_value(in, (size_t)p[l], (int)w[l], is_signed);
  }
  *base = _mm256_add_epi64(*base, _mm256_permute4x64_epi64(incl, 0xFF));
}


/* As internal_libxs_lanes_decode, the predictor and the payload in vector registers. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_lanes_decode_avx2(void* values, libxs_data_t type, size_t n, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  const int typesize = internal_libxs_pack_typesize(type);
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  internal_libxs_lanes_reader_t rd;
  uint64_t buffer[LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK];
  __m256i p1 = _mm256_setzero_si256(), p2 = _mm256_setzero_si256(), base = _mm256_setzero_si256();
  size_t g = 0;
  int zero = 1;
  internal_libxs_lanes_begin(&rd, in, offs);
  while (g < ngroups) {
    const size_t run = internal_libxs_lanes_runlen(rd.in[0], &rd.cbit);
    size_t h;
    for (h = 0; h < run; h += LIBXS_PACK_LANES_BLOCK) {
      const size_t at = LIBXS_PACK_LANES * (g + h), nv = LIBXS_PACK_LANES * LIBXS_MIN(LIBXS_PACK_LANES_BLOCK, run - h);
      if (0 == zero) {
        uint64_t *const q = internal_libxs_lanes_dst(values, typesize, at, nv, n, buffer);
        size_t j;
        internal_libxs_lanes_unary(&rd, nv);
        for (j = 0; j < nv; j += LIBXS_PACK_LANES) {
          const __m256i b = internal_libxs_lanes_lengths_avx2(internal_libxs_lanes_codes(&rd, j), &p1, &p2);
          internal_libxs_lanes_payload_avx2(q + j, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(b)), &base,
            rd.in[4], is_signed);
          internal_libxs_lanes_payload_avx2(q + j + 4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(b, 1)), &base,
            rd.in[4], is_signed);
        }
        internal_libxs_lanes_next(&rd, nv);
        internal_libxs_lanes_done(values, type, at, nv, n, q, buffer);
      }
      else internal_libxs_lanes_zeros(values, typesize, at, nv, n);
    }
    g += run;
    zero = !zero;
  }
}
#endif


#if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_lanes_decode_avx2, the lengths with masks (the payload as there: four values at once). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_decode_avx512(void* values, libxs_data_t type, size_t n, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  const __m256i one = _mm256_set1_epi32(1), zeros = _mm256_setzero_si256();
  const int typesize = internal_libxs_pack_typesize(type);
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  internal_libxs_lanes_reader_t rd;
  uint64_t buffer[LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK];
  __m256i p1 = _mm256_setzero_si256(), p2 = _mm256_setzero_si256(), base = _mm256_setzero_si256();
  size_t g = 0;
  int zero = 1;
  internal_libxs_lanes_begin(&rd, in, offs);
  while (g < ngroups) {
    const size_t run = internal_libxs_lanes_runlen(rd.in[0], &rd.cbit);
    size_t h;
    for (h = 0; h < run; h += LIBXS_PACK_LANES_BLOCK) {
      const size_t at = LIBXS_PACK_LANES * (g + h), nv = LIBXS_PACK_LANES * LIBXS_MIN(LIBXS_PACK_LANES_BLOCK, run - h);
      if (0 == zero) {
        uint64_t *const q = internal_libxs_lanes_dst(values, typesize, at, nv, n, buffer);
        size_t j;
        internal_libxs_lanes_unary(&rd, nv);
        for (j = 0; j < nv; j += LIBXS_PACK_LANES) {
          const __m256i s = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)internal_libxs_lanes_codes(&rd, j)));
          const __mmask8 nz = _mm256_test_epi32_mask(s, s);
          const __m256i u = _mm256_sub_epi32(s, one);
          const __m256i d = _mm256_xor_si256(_mm256_srli_epi32(u, 1),
            _mm256_sub_epi32(zeros, _mm256_and_si256(u, one)));
          const __m256i b = _mm256_maskz_add_epi32(nz, d,
            _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(p1, p2), one), 1));
          p2 = _mm256_mask_mov_epi32(p2, nz, p1);
          p1 = _mm256_mask_mov_epi32(p1, nz, b);
          internal_libxs_lanes_payload_avx2(q + j, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(b)), &base,
            rd.in[4], is_signed);
          internal_libxs_lanes_payload_avx2(q + j + 4, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(b, 1)), &base,
            rd.in[4], is_signed);
        }
        internal_libxs_lanes_next(&rd, nv);
        internal_libxs_lanes_done(values, type, at, nv, n, q, buffer);
      }
      else internal_libxs_lanes_zeros(values, typesize, at, nv, n);
    }
    g += run;
    zero = !zero;
  }
}
#endif


/* Decodes the format of internal_libxs_vpack_lanes; returns the words consumed, zero if invalid. */
LIBXS_API_INLINE size_t internal_libxs_vunpack_lanes(void* values, const void* packed, libxs_data_t type,
  size_t n, int is_signed)
{
  const union { uint32_t u; unsigned char c[4]; } probe = { 1 };
  const unsigned char *const in = (const unsigned char*)packed;
  const int nb = in[1];
  size_t result = 0;
  if (0 != probe.c[0] && 0 < nb && 56 >= nb) {
    size_t offs[LIBXS_PACK_LANES_NSUB + 1];
    int k;
    offs[0] = 2 + (LIBXS_PACK_LANES_NSUB * (size_t)nb + 7) / 8;
    for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) {
      const size_t bit = 16 + (size_t)k * (size_t)nb;
      uint64_t v;
      memcpy(&v, in + (bit >> 3), sizeof(v));
      offs[k + 1] = offs[k] + (size_t)((v >> (bit & 7)) & LIBXS_PACK_MASK(nb));
    }
    {
#if defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2)
      const int kind = internal_libxs_pack_target();
#endif
#if defined(LIBXS_PACK_AVX512)
      if (3 == kind) internal_libxs_lanes_decode_avx512(values, type, n, in, offs, is_signed);
      else
#endif
#if defined(LIBXS_PACK_AVX2)
      if (2 <= kind) internal_libxs_lanes_decode_avx2(values, type, n, in, offs, is_signed);
      else
#endif
      internal_libxs_lanes_decode(values, type, n, in, offs, is_signed);
    }
    result = (offs[LIBXS_PACK_LANES_NSUB] + LIBXS_PACK_LANES_SLACK + 7) / 8;
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
    if (LIBXS_PACK_WORDS == internal_libxs_vpack_format()) { /* 8 interleaved streams, each in whole words */
      result = (1 + ngroups + 7) / 8 + 8 * ((((n + 7) / 8) * (size_t)maxbits + 63) / 64);
    }
    else result = (1 + ngroups + 7) / 8 + (((n + 7) / 8) * (size_t)maxbits + 7) / 8;
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
  const int format = (0 != valid ? internal_libxs_vpack_format() : 0);
  size_t result = ((LIBXS_PACK_CODES == format && LIBXS_PACK_EXC_GROUP == g)
    ? internal_libxs_vpack_lanes(packed, values, type, n, maxbits, libxs_vpack_bound(n, maxbits, g)) : 0);
  if (0 != valid && 0 == result) { /* otherwise, or if it does not fit */
    const int words = (LIBXS_PACK_WORDS == format && 0 == (g & 7));
    const int exc = (LIBXS_PACK_WORDS != format && LIBXS_PACK_EXC_GROUP == g);
    const size_t ngroups = (n + g - 1) / g, nhead = (1 + ngroups + 7) / 8;
    const uint64_t mask = LIBXS_PACK_MASK(maxbits);
    const uint64_t sign = (0 != internal_libxs_pack_signed(type)
      ? (((uint64_t)1) << (maxbits - 1)) : 0);
    unsigned char *const head = (unsigned char*)packed;
    unsigned char *const data = (unsigned char*)((uint64_t*)packed + nhead);
    internal_libxs_pack_writer_t wr;
    uint64_t buf[255 + 7];
    size_t i, k, j = 0, nbytes = 0;
    int off = 0;
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
        if (0 != exc) {
#if defined(LIBXS_PACK_AVX512)
          if (3 == kind) nbytes += internal_libxs_vpack8_exc_avx512(head + 1 + i, data + nbytes, buf, w);
          else
#endif
#if defined(LIBXS_PACK_AVX2)
          if (2 <= kind) nbytes += internal_libxs_vpack8_exc_avx2(head + 1 + i, data + nbytes, buf, w);
          else
#endif
          nbytes += internal_libxs_vpack8_exc(head + 1 + i, data + nbytes, buf, w);
        }
        else if (0 != words) {
          for (k = 0; k < m; k += 8) internal_libxs_vpack8_col(wr.out, buf + k, &j, &off, w);
        }
        else {
          for (k = 0; k < m; k += 8) {
            internal_libxs_vpack8_group(data + nbytes, buf + k, w);
            nbytes += (size_t)w;
          }
        }
      }
      else if (0 < w) {
        for (k = 0; k < m; ++k) internal_libxs_pack_put(&wr, buf[k], w);
      }
    }
    if (0 != words) result = nhead + 8 * (j + (0 != off ? 1 : 0));
    else if (0 == (g & 7)) { /* bytes, with or without exceptions */
      for (; 0 != (nbytes & 7); ++nbytes) data[nbytes] = 0;
      result = nhead + nbytes / 8;
    }
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
    const int g = head[0], is_signed = internal_libxs_pack_signed(type), format = internal_libxs_vpack_format();
    if (0 < g) {
      const int words = (LIBXS_PACK_WORDS == format && 0 == (g & 7));
      const int exc = (LIBXS_PACK_WORDS != format && LIBXS_PACK_EXC_GROUP == g);
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
        if (0 != words) nbits = 8 * 64 * ((nbits / 8 + 63) / 64); /* 8 streams in whole words */
        if (0 != exc) nbits = 0; /* depends on the data: checked after decoding */
        LIBXS_ASSERT(nhead + (nbits + 63) / 64 <= npacked);
      }
#endif
      if (0 != exc) {
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
      else if (0 != words && 8 == typesize) {
        bit = internal_libxs_vunpack64_words((uint64_t*)values, head + 1, in, n, g, is_signed);
      }
      else if (0 != words) { /* 8 interleaved streams */
        size_t j = 0;
        int off = 0;
        for (i = 0; i < ngroups; ++i) {
          const size_t base = i * g, m = LIBXS_MIN((size_t)g, n - base);
          uint64_t *const dst = buf;
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
          internal_libxs_pack_store(values, buf, type, base, m);
        }
        bit = 64 * 8 * (j + (0 != off ? 1 : 0));
      }
      else if (0 == (g & 7) && 8 == typesize && 0 != probe.c[0]) { /* byte-aligned into 64-bit */
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
