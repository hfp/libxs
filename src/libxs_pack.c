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
/* formats of libxs_vpack (the level, 0: LIBXS_PACK or LIBXS_PACK_DEFAULT): groups of 8 (LIBXS_PACK_GROUP)
 * as 8 interleaved streams of words (value 8*c+l in stream l, all at the same bit position, so every shift
 * is uniform over lanes) as samples/pack/vpack.f writes them (byte 0 is the group size); a group of 8 as
 * two groups of 4 in words with their widths as deltas (internal_libxs_vpack_quads); or as codes of the
 * lengths (internal_libxs_vpack_lanes). The latter two start with a zero byte and the format; where they
 * would not fit, the first one is taken. */
#define LIBXS_PACK_WORDS 1
#define LIBXS_PACK_QUADS 2
#define LIBXS_PACK_CODES 3
#define LIBXS_PACK_GROUP 8
/* floating-point data: a header word (a zero byte, this id, the transform, the bits kept per value, the exponent
 * of the absolute error in 16 bits), then the values as integers at the level; the transforms: the sign at bit 0
 * and a XOR with the previous value (as is, or bit-reversed so that trailing zeros lead), integral values as
 * integers, or quantized at a step of 2^(maxbits+1) */
#define LIBXS_PACK_REALS 4
#define LIBXS_PACK_REAL_XOR 0
#define LIBXS_PACK_REAL_REV 1
#define LIBXS_PACK_REAL_INT 2
#define LIBXS_PACK_REAL_ABS 3
/* format LIBXS_PACK_QUADS: values per group, bits of a width delta (all ones: the width follows in 8 bits) */
#define LIBXS_PACK_QUAD 4
#define LIBXS_PACK_QUAD_DELTA 4
/* formats LIBXS_PACK_WORDS and LIBXS_PACK_QUADS: columns whose positions are found before they are read */
#define LIBXS_PACK_SPAN 64
#define LIBXS_PACK_DEFAULT LIBXS_PACK_QUADS
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


/* Reads 8 interleaved streams (LIBXS_PACK_WORDS) into 64-bit integers; returns the bits consumed. A span of
 * columns at a time: their positions first (a sum of widths), then the columns, which do not depend on each
 * other. Every lane of a column takes the same shift: plain code, vectorized per target by the compiler. */
LIBXS_INLINE_ALWAYS size_t internal_libxs_vunpack64_words_body(uint64_t* LIBXS_RESTRICT out,
  const unsigned char* LIBXS_RESTRICT widths, const uint64_t* LIBXS_RESTRICT in, size_t n, int g, int is_signed)
{
  const size_t ngroups = (n + g - 1) / g, ncols = (n + 7) / 8, cpg = (size_t)g / 8;
  const uint64_t sign = (uint64_t)is_signed;
  size_t i, c, gi = 0, cc = 0, bit = 0, total = 0, nrows;
  for (i = 0; i < ngroups; ++i) total += (size_t)widths[i] * ((LIBXS_MIN((size_t)g, n - i * g) + 7) / 8);
  nrows = (total + 63) / 64;
  for (c = 0; c < ncols; c += LIBXS_PACK_SPAN) {
    const size_t nc = LIBXS_MIN((size_t)LIBXS_PACK_SPAN, ncols - c);
    size_t pos[LIBXS_PACK_SPAN], k;
    int wid[LIBXS_PACK_SPAN];
    for (k = 0; k < nc; ++k) {
      wid[k] = widths[gi];
      pos[k] = bit;
      bit += (size_t)wid[k];
      if (cpg == ++cc) {
        cc = 0;
        ++gi;
      }
    }
    for (k = 0; k < nc; ++k) {
      const size_t at = 8 * (c + k), row = pos[k] >> 6;
      const int w = wid[k], off = (int)(pos[k] & 63);
      const uint64_t mask = LIBXS_PACK_MASK(w), *const lo = in + 8 * row;
      uint64_t tmp[8], *const dst = (at + 8 <= n ? out + at : tmp);
      int l;
      if (row + 1 < nrows) { /* the next word of every stream is readable: no branch on a spill */
        LIBXS_PRAGMA_SIMD
        for (l = 0; l < 8; ++l) { /* (x << 1) << (63 - off): zero if off is zero */
          const uint64_t v = ((lo[l] >> off) | ((lo[8 + l] << 1) << (63 - off))) & mask;
          dst[l] = (v >> sign) ^ (0 - (v & sign));
        }
      }
      else if (64 < off + w) { /* spills into the next word of every stream */
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
      if (dst == tmp) memcpy(out + at, tmp, (n - at) * sizeof(*tmp));
    }
  }
  return 64 * 8 * nrows;
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


/* Words of the format LIBXS_PACK_QUADS at most: the count, 4 interleaved streams, the deltas, the escapes. */
LIBXS_API_INLINE size_t internal_libxs_vpack_quads_bound(size_t n, int maxbits)
{
  const size_t ngroups = (n + LIBXS_PACK_QUAD - 1) / LIBXS_PACK_QUAD;
  return 1 + LIBXS_PACK_QUAD * ((ngroups * (size_t)maxbits + 63) / 64) + (ngroups + 15) / 16 + ngroups / 8 + 1;
}


/* A group of 8 (LIBXS_PACK_QUADS) as groups of 4: every value of a group at the group's width in 4
 * interleaved streams of words (as LIBXS_PACK_WORDS), behind them a delta of the width per group in 4 bits
 * (zig-zag, 16 per word; all ones: the delta is the next of the escapes), then the escapes (8 bits, signed,
 * 8 per word, a word more). Word 0 holds the words of the streams shifted by 8
 * (word 0 also holds the format above a zero byte). Returns the words written, zero without scratch (the
 * caller falls back). */
LIBXS_API_INLINE size_t internal_libxs_vpack_quads(void* packed, const void* values, libxs_data_t type,
  size_t n, int maxbits)
{
  const size_t ngroups = (n + LIBXS_PACK_QUAD - 1) / LIBXS_PACK_QUAD, ndelta = (ngroups + 15) / 16;
  const int direct = (8 == internal_libxs_pack_typesize(type)); /* 64-bit values are read in place */
  const uint64_t mask = LIBXS_PACK_MASK(maxbits);
  const uint64_t sign = (0 != internal_libxs_pack_signed(type) ? (((uint64_t)1) << (maxbits - 1)) : 0);
  const uint64_t fold = (0 != sign ? 1 : 0), escape = (((uint64_t)1) << LIBXS_PACK_QUAD_DELTA) - 1;
  int pool = 0; /* the deltas, then the escapes as bytes (one per group at most) */
  uint64_t *const head = (uint64_t*)internal_libxs_scratch_malloc(ndelta * sizeof(uint64_t) + ngroups + 1, &pool);
  size_t result = 0;
  if (NULL != head) { /* no branch on the data: stores go to a dummy where nothing is complete */
    uint64_t *const out = (uint64_t*)packed, *const data = out + 1, *esc;
    unsigned char *const escb = (unsigned char*)(head + ndelta);
    uint64_t row[LIBXS_PACK_QUAD], dummy[LIBXS_PACK_QUAD], dacc = 0;
    size_t i, k, j = 0, e = 0, nwords;
    int off = 0, prev = 0, l;
    for (l = 0; l < LIBXS_PACK_QUAD; ++l) row[l] = 0;
    for (i = 0; i < ngroups; ++i) {
      const size_t base = LIBXS_PACK_QUAD * i, m = LIBXS_MIN((size_t)LIBXS_PACK_QUAD, n - base);
      uint64_t z[LIBXS_PACK_QUAD], any = 0, d;
      const uint64_t* src = z;
      int w, dw, c, x;
      if (0 != direct && LIBXS_PACK_QUAD == m) src = (const uint64_t*)values + base;
      else internal_libxs_pack_load(z, values, type, base, m);
      for (l = 0; l < LIBXS_PACK_QUAD; ++l) { /* sign-extend from maxbits, then fold the sign (zig-zag) */
        const uint64_t v = ((((size_t)l < m ? src[l] : 0) & mask) ^ sign) - sign;
        z[l] = (v << fold) ^ (0 - ((v >> 63) & fold));
        any |= z[l];
      }
      w = (int)LIBXS_NBITS(any);
      dw = w - prev;
      d = (uint64_t)(0 <= dw ? 2 * dw : -2 * dw - 1);
      x = (escape <= d ? 1 : 0);
      escb[e] = (unsigned char)dw;
      e += (size_t)x;
      dacc |= (0 != x ? escape : d) << (LIBXS_PACK_QUAD_DELTA * (i & 15));
      head[i >> 4] = dacc;
      dacc &= 0 - (uint64_t)(15 != (i & 15) ? 1 : 0);
      prev = w;
      c = (off + w) >> 6; /* the row is complete */
      {
        uint64_t *const lo = (0 != c ? data + LIBXS_PACK_QUAD * j : dummy);
        for (l = 0; l < LIBXS_PACK_QUAD; ++l) {
          const uint64_t v = row[l] | (z[l] << off);
          lo[l] = v; /* (x >> 1) >> (63 - off): what spills, zero if off is zero */
          row[l] = (0 != c ? ((z[l] >> 1) >> (63 - off)) : v);
        }
      }
      j += (size_t)c;
      off = (off + w) & 63;
    }
    if (0 != off) {
      for (l = 0; l < LIBXS_PACK_QUAD; ++l) data[LIBXS_PACK_QUAD * j + l] = row[l];
    }
    nwords = LIBXS_PACK_QUAD * (j + (0 != off ? 1 : 0));
    out[0] = ((uint64_t)nwords << 16) | ((uint64_t)LIBXS_PACK_QUADS << 8);
    memcpy(data + nwords, head, ndelta * sizeof(uint64_t));
    esc = data + nwords + ndelta;
    for (k = 0; k <= e / 8; ++k) esc[k] = 0;
    for (k = 0; k < e; ++k) esc[k >> 3] |= (uint64_t)escb[k] << (8 * (k & 7));
    result = 1 + nwords + ndelta + e / 8 + 1;
    internal_libxs_scratch_free(head, pool);
  }
  return result;
}


/* Widths and positions of the ng groups from group i (LIBXS_PACK_QUADS): the deltas of a word at once, then
 * its escapes, then sums. */
LIBXS_INLINE_ALWAYS void internal_libxs_quads_span(int wid[], size_t pos[], size_t i, size_t ng,
  const uint64_t* LIBXS_RESTRICT head, const uint64_t* LIBXS_RESTRICT esc, size_t* e, int* prev, size_t* bit)
{
  const uint64_t escape = (((uint64_t)1) << LIBXS_PACK_QUAD_DELTA) - 1;
  int delta[LIBXS_PACK_SPAN], w = *prev;
  size_t k, b = *bit, x = *e;
  for (k = 0; k < ng; k += 16) {
    const uint64_t word = head[(i + k) >> 4];
    const size_t m = LIBXS_MIN((size_t)16, ng - k);
    uint64_t mask = 0;
    int l;
    for (l = 0; l < 16; ++l) {
      const int d = (int)((word >> (LIBXS_PACK_QUAD_DELTA * l)) & escape);
      delta[k + l] = (d >> 1) ^ -(d & 1);
      mask |= (uint64_t)(escape == (uint64_t)d ? 1 : 0) << l;
    }
    for (mask &= (((uint64_t)1) << m) - 1; 0 != mask; mask &= mask - 1, ++x) {
      delta[k + LIBXS_INTRINSICS_BITSCANFWD64(mask)] = (int)(signed char)((esc[x >> 3] >> (8 * (x & 7))) & 0xFF);
    }
  }
  for (k = 0; k < ng; ++k) {
    w += delta[k];
    wid[k] = w;
    pos[k] = b;
    b += (size_t)w;
  }
  *prev = w;
  *bit = b;
  *e = x;
}


/* A group of 4 (LIBXS_PACK_QUADS) of width w at bit position p of the streams into out + at. */
LIBXS_INLINE_ALWAYS void internal_libxs_quads_group(uint64_t* LIBXS_RESTRICT out, size_t n, size_t at,
  const uint64_t* LIBXS_RESTRICT data, size_t nrows, size_t p, int w, uint64_t sign)
{
  const size_t row = p >> 6;
  const int off = (int)(p & 63);
  const uint64_t mask = LIBXS_PACK_MASK(w), *const lo = data + LIBXS_PACK_QUAD * row;
  uint64_t tmp[LIBXS_PACK_QUAD], *const dst = (at + LIBXS_PACK_QUAD <= n ? out + at : tmp);
  int l;
  if (row + 1 < nrows) { /* the next word of every stream is readable: no branch on a spill */
    LIBXS_PRAGMA_SIMD
    for (l = 0; l < LIBXS_PACK_QUAD; ++l) { /* (x << 1) << (63 - off): zero if off is zero */
      const uint64_t v = ((lo[l] >> off) | ((lo[LIBXS_PACK_QUAD + l] << 1) << (63 - off))) & mask;
      dst[l] = (v >> sign) ^ (0 - (v & sign));
    }
  }
  else if (64 < off + w) { /* spills into the next word of every stream */
    LIBXS_PRAGMA_SIMD
    for (l = 0; l < LIBXS_PACK_QUAD; ++l) {
      const uint64_t v = ((lo[l] >> off) | (lo[LIBXS_PACK_QUAD + l] << (64 - off))) & mask;
      dst[l] = (v >> sign) ^ (0 - (v & sign));
    }
  }
  else if (0 < w) {
    LIBXS_PRAGMA_SIMD
    for (l = 0; l < LIBXS_PACK_QUAD; ++l) {
      const uint64_t v = (lo[l] >> off) & mask;
      dst[l] = (v >> sign) ^ (0 - (v & sign));
    }
  }
  else {
    for (l = 0; l < LIBXS_PACK_QUAD; ++l) dst[l] = 0;
  }
  if (dst == tmp) memcpy(out + at, tmp, (n - at) * sizeof(*tmp));
}


/* Reads the format LIBXS_PACK_QUADS into 64-bit integers; returns the words consumed. A span of groups at a
 * time: their widths and positions first, then the groups, which do not depend on each other. Every lane of
 * a group takes the same shift: plain code, vectorized per target by the compiler. */
LIBXS_INLINE_ALWAYS size_t internal_libxs_vunpack64_quads_body(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t n, int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_QUAD - 1) / LIBXS_PACK_QUAD, nwords = (size_t)(in[0] >> 16);
  const uint64_t *const data = in + 1, *const head = data + nwords, *const esc = head + (ngroups + 15) / 16;
  size_t i, e = 0, bit = 0;
  int prev = 0;
  for (i = 0; i < ngroups; i += LIBXS_PACK_SPAN) {
    const size_t ng = LIBXS_MIN((size_t)LIBXS_PACK_SPAN, ngroups - i);
    size_t pos[LIBXS_PACK_SPAN], k;
    int wid[LIBXS_PACK_SPAN];
    internal_libxs_quads_span(wid, pos, i, ng, head, esc, &e, &prev, &bit);
    for (k = 0; k < ng; ++k) {
      internal_libxs_quads_group(out, n, LIBXS_PACK_QUAD * (i + k), data, nwords / LIBXS_PACK_QUAD, pos[k], wid[k],
        (uint64_t)is_signed);
    }
  }
  return 1 + nwords + (ngroups + 15) / 16 + e / 8 + 1;
}


LIBXS_API_INLINE size_t internal_libxs_vunpack64_quads_sw(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t n, int is_signed)
{
  return internal_libxs_vunpack64_quads_body(out, in, n, is_signed);
}


#if defined(LIBXS_PACK_AVX2)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
size_t internal_libxs_vunpack64_quads_avx2(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t n, int is_signed)
{
  return internal_libxs_vunpack64_quads_body(out, in, n, is_signed);
}
#endif


#if defined(LIBXS_PACK_AVX512)
/* As internal_libxs_vunpack64_quads_body, two groups at once (lanes 0-3 and 4-7, a shift per lane). */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
size_t internal_libxs_vunpack64_quads_avx512(uint64_t* LIBXS_RESTRICT out,
  const uint64_t* LIBXS_RESTRICT in, size_t n, int is_signed)
{
  const size_t ngroups = (n + LIBXS_PACK_QUAD - 1) / LIBXS_PACK_QUAD, nwords = (size_t)(in[0] >> 16);
  const size_t nrows = nwords / LIBXS_PACK_QUAD;
  const uint64_t *const data = in + 1, *const head = data + nwords, *const esc = head + (ngroups + 15) / 16;
  const __m512i ones = _mm512_set1_epi64(-1), sixtythree = _mm512_set1_epi64(63);
  const __m512i sixtyfour = _mm512_set1_epi64(64), sign = _mm512_set1_epi64(is_signed);
  const __m512i zeros = _mm512_setzero_si512();
  size_t i, e = 0, bit = 0;
  int prev = 0;
  for (i = 0; i < ngroups; i += LIBXS_PACK_SPAN) {
    const size_t ng = LIBXS_MIN((size_t)LIBXS_PACK_SPAN, ngroups - i);
    size_t pos[LIBXS_PACK_SPAN], k;
    int wid[LIBXS_PACK_SPAN];
    internal_libxs_quads_span(wid, pos, i, ng, head, esc, &e, &prev, &bit);
    for (k = 0; k + 1 < ng; k += 2) {
      const size_t at = LIBXS_PACK_QUAD * (i + k), r0 = pos[k] >> 6, r1 = pos[k + 1] >> 6;
      if (r1 + 1 < nrows && at + 2 * LIBXS_PACK_QUAD <= n) { /* both groups and their next rows */
        const uint64_t *const a = data + LIBXS_PACK_QUAD * r0, *const b = data + LIBXS_PACK_QUAD * r1;
        const __m512i lo = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)a)),
          _mm256_loadu_si256((const __m256i*)b), 1);
        const __m512i hi = _mm512_inserti64x4(_mm512_castsi256_si512(
          _mm256_loadu_si256((const __m256i*)(a + LIBXS_PACK_QUAD))),
          _mm256_loadu_si256((const __m256i*)(b + LIBXS_PACK_QUAD)), 1);
        const __m512i off = _mm512_mask_blend_epi64(0xF0, _mm512_set1_epi64((long long)(pos[k] & 63)),
          _mm512_set1_epi64((long long)(pos[k + 1] & 63)));
        const __m512i w = _mm512_mask_blend_epi64(0xF0, _mm512_set1_epi64(wid[k]), _mm512_set1_epi64(wid[k + 1]));
        const __m512i mask = _mm512_srlv_epi64(ones, _mm512_sub_epi64(sixtyfour, w)); /* zero for no bits */
        const __m512i v = _mm512_and_si512(mask, _mm512_or_si512(_mm512_srlv_epi64(lo, off),
          _mm512_sllv_epi64(_mm512_slli_epi64(hi, 1), _mm512_sub_epi64(sixtythree, off))));
        _mm512_storeu_si512(out + at,
          _mm512_xor_si512(_mm512_srlv_epi64(v, sign), _mm512_sub_epi64(zeros, _mm512_and_si512(v, sign))));
      }
      else {
        internal_libxs_quads_group(out, n, at, data, nrows, pos[k], wid[k], (uint64_t)is_signed);
        internal_libxs_quads_group(out, n, at + LIBXS_PACK_QUAD, data, nrows, pos[k + 1], wid[k + 1],
          (uint64_t)is_signed);
      }
    }
    if (k < ng) {
      internal_libxs_quads_group(out, n, LIBXS_PACK_QUAD * (i + k), data, nrows, pos[k], wid[k], (uint64_t)is_signed);
    }
  }
  return 1 + nwords + (ngroups + 15) / 16 + e / 8 + 1;
}
#endif


/* Decodes the format LIBXS_PACK_QUADS with the widest code the CPU runs; returns the words consumed. */
LIBXS_API_INLINE size_t internal_libxs_vunpack_quads(void* values, const void* packed, libxs_data_t type,
  size_t n, int is_signed)
{
  const int typesize = internal_libxs_pack_typesize(type);
  uint64_t* q = (uint64_t*)values;
  size_t result = 0;
  int pool = 0;
  if (8 != typesize) q = (uint64_t*)internal_libxs_scratch_malloc(n * sizeof(uint64_t), &pool);
  if (NULL != q) {
#if defined(LIBXS_PACK_AVX512) || defined(LIBXS_PACK_AVX2)
    const int kind = internal_libxs_pack_target();
#endif
#if defined(LIBXS_PACK_AVX512)
    if (3 == kind) result = internal_libxs_vunpack64_quads_avx512(q, (const uint64_t*)packed, n, is_signed);
    else
#endif
#if defined(LIBXS_PACK_AVX2)
    if (2 == kind) result = internal_libxs_vunpack64_quads_avx2(q, (const uint64_t*)packed, n, is_signed);
    else
#endif
    result = internal_libxs_vunpack64_quads_sw(q, (const uint64_t*)packed, n, is_signed);
    if (8 != typesize) {
      internal_libxs_pack_store(values, q, type, 0, n);
      internal_libxs_scratch_free(q, pool);
    }
  }
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
      hdr = 3 + (LIBXS_PACK_LANES_NSUB * (size_t)nb + 7) / 8;
      total = hdr + LIBXS_PACK_LANES_SLACK;
      for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) total += nbytes[k];
      if ((total + 7) / 8 <= nwords) {
        out[0] = 0; /* no group size: the format follows */
        out[1] = (unsigned char)LIBXS_PACK_CODES;
        out[2] = (unsigned char)nb;
        for (k = 0, i = 3; k < LIBXS_PACK_LANES_NSUB; ++k) {
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
/* Eight values of lengths b (64-bit lanes) from the payload position base on; base moves past them. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_payload_avx512(uint64_t* q, __m512i b, __m512i* base, const unsigned char* in, int is_signed)
{
  const __m512i one = _mm512_set1_epi64(1), seven = _mm512_set1_epi64(7), zeros = _mm512_setzero_si512();
  const __m512i sign = _mm512_set1_epi64(is_signed), last = _mm512_set1_epi64(7);
  const __mmask8 nz = _mm512_test_epi64_mask(b, b);
  const __m512i nb = _mm512_maskz_sub_epi64(nz, b, one);
  const __m512i x1 = _mm512_add_epi64(nb, _mm512_alignr_epi64(nb, zeros, 7));
  const __m512i x2 = _mm512_add_epi64(x1, _mm512_alignr_epi64(x1, zeros, 6));
  const __m512i incl = _mm512_add_epi64(x2, _mm512_alignr_epi64(x2, zeros, 4));
  const __m512i pos = _mm512_add_epi64(*base, _mm512_sub_epi64(incl, nb));
  const __m512i sh = _mm512_and_si512(pos, seven);
  if (0 == _mm512_cmpgt_epi64_mask(_mm512_add_epi64(nb, sh), _mm512_set1_epi64(64))) {
    const __m512i v = _mm512_srlv_epi64(_mm512_i64gather_epi64(_mm512_srli_epi64(pos, 3), in, 1), sh);
    const __m512i top = _mm512_sllv_epi64(one, nb);
    const __m512i x = _mm512_or_si512(_mm512_and_si512(v, _mm512_sub_epi64(top, one)), _mm512_maskz_mov_epi64(nz, top));
    _mm512_storeu_si512(q,
      _mm512_xor_si512(_mm512_srlv_epi64(x, sign), _mm512_sub_epi64(zeros, _mm512_and_si512(x, sign))));
  }
  else { /* a value out of reach of one load */
    uint64_t p[8], w[8];
    int l;
    _mm512_storeu_si512(p, pos);
    _mm512_storeu_si512(w, b);
    for (l = 0; l < 8; ++l) q[l] = internal_libxs_lanes_value(in, (size_t)p[l], (int)w[l], is_signed);
  }
  *base = _mm512_add_epi64(*base, _mm512_permutexvar_epi64(last, incl));
}


/* As internal_libxs_lanes_decode_avx2, eight values of the payload at once. */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_lanes_decode_avx512(void* values, libxs_data_t type, size_t n, const unsigned char* in,
  const size_t* offs, int is_signed)
{
  const __m256i one = _mm256_set1_epi32(1), zeros = _mm256_setzero_si256();
  const int typesize = internal_libxs_pack_typesize(type);
  const size_t ngroups = (n + LIBXS_PACK_LANES - 1) / LIBXS_PACK_LANES;
  internal_libxs_lanes_reader_t rd;
  uint64_t buffer[LIBXS_PACK_LANES * LIBXS_PACK_LANES_BLOCK];
  __m256i p1 = _mm256_setzero_si256(), p2 = _mm256_setzero_si256();
  __m512i base = _mm512_setzero_si512();
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
          internal_libxs_lanes_payload_avx512(q + j, _mm512_cvtepi32_epi64(b), &base, rd.in[4], is_signed);
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
  const int nb = in[2];
  size_t result = 0;
  if (0 != probe.c[0] && 0 < nb && 56 >= nb) {
    size_t offs[LIBXS_PACK_LANES_NSUB + 1];
    int k;
    offs[0] = 3 + (LIBXS_PACK_LANES_NSUB * (size_t)nb + 7) / 8;
    for (k = 0; k < LIBXS_PACK_LANES_NSUB; ++k) {
      const size_t bit = 24 + (size_t)k * (size_t)nb;
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


LIBXS_API size_t libxs_vpack_bound(size_t n, int maxbits)
{
  size_t result = 0;
  if (0 < n) { /* 8 interleaved streams in whole words, or groups of 4; a word more for floating-point data */
    const size_t ngroups = (n + LIBXS_PACK_GROUP - 1) / LIBXS_PACK_GROUP; /* bits kept: sign, exponent, a bit */
    const int nbits = ((0 < maxbits && 64 > maxbits) ? LIBXS_MAX(maxbits, 13) : 64);
    result = 1 + LIBXS_MAX((1 + ngroups + 7) / 8 + 8 * ((ngroups * (size_t)nbits + 63) / 64),
      internal_libxs_vpack_quads_bound(n, nbits));
  }
  return result;
}


LIBXS_API_INLINE size_t internal_libxs_vpack_int(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int level)
{
  const int typesize = internal_libxs_pack_typesize(type);
  const int valid = (0 < typesize && 0 < maxbits && maxbits <= 8 * typesize
    && 0 < n && NULL != packed && NULL != values);
  const int format = (0 != valid /* a level above the last takes the last, below 1 the default */
    ? (0 < level ? LIBXS_MIN(level, LIBXS_PACK_CODES) : internal_libxs_vpack_format()) : 0);
  size_t result = 0;
  if (LIBXS_PACK_QUADS == format) result = internal_libxs_vpack_quads(packed, values, type, n, maxbits);
  else if (LIBXS_PACK_CODES == format) { /* the bound's last word is for floating-point data (a header) */
    result = internal_libxs_vpack_lanes(packed, values, type, n, maxbits, libxs_vpack_bound(n, maxbits) - 1);
  }
  if (0 != valid && 0 == result) { /* LIBXS_PACK_WORDS: also where another format does not fit */
    const size_t ngroups = (n + LIBXS_PACK_GROUP - 1) / LIBXS_PACK_GROUP, nhead = (1 + ngroups + 7) / 8;
    const uint64_t mask = LIBXS_PACK_MASK(maxbits);
    const uint64_t sign = (0 != internal_libxs_pack_signed(type)
      ? (((uint64_t)1) << (maxbits - 1)) : 0);
    unsigned char *const head = (unsigned char*)packed;
    uint64_t *const out = (uint64_t*)packed + nhead, buf[LIBXS_PACK_GROUP];
    size_t i, k, j = 0;
    int off = 0;
    head[0] = (unsigned char)LIBXS_PACK_GROUP;
    for (i = 1 + ngroups; i < 8 * nhead; ++i) head[i] = 0;
    for (i = 0; i < ngroups; ++i) {
      const size_t base = i * LIBXS_PACK_GROUP, m = LIBXS_MIN((size_t)LIBXS_PACK_GROUP, n - base);
      uint64_t any = 0;
      int w;
      const uint64_t* src = buf;
      if (8 == typesize && LIBXS_PACK_GROUP == m) src = (const uint64_t*)values + base;
      else internal_libxs_pack_load(buf, values, type, base, m);
      for (k = 0; k < LIBXS_PACK_GROUP; ++k) {
        uint64_t v = (k < m ? src[k] & mask : 0);
        if (0 != sign) { /* sign-extend from maxbits, then fold the sign (zig-zag) */
          v = (v ^ sign) - sign;
          v = (v << 1) ^ (0 - (v >> 63));
        }
        buf[k] = v;
        any |= v;
      }
      w = (int)LIBXS_NBITS(any);
      head[1 + i] = (unsigned char)w;
      if (0 < w) internal_libxs_vpack8_col(out, buf, &j, &off, w);
    }
    result = nhead + 8 * (j + (0 != off ? 1 : 0));
  }
  return result;
}


LIBXS_API_INLINE size_t internal_libxs_vunpack_int(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  const int typesize = internal_libxs_pack_typesize(type);
  size_t result = 0;
  if (0 < typesize && 0 < n && 0 < npacked && NULL != packed && NULL != values) {
    const unsigned char *const head = (const unsigned char*)packed;
    const int is_signed = internal_libxs_pack_signed(type);
    if (LIBXS_PACK_GROUP == head[0]) { /* LIBXS_PACK_WORDS */
      const size_t ngroups = (n + LIBXS_PACK_GROUP - 1) / LIBXS_PACK_GROUP, nhead = (1 + ngroups + 7) / 8;
      const uint64_t *const in = (const uint64_t*)packed + nhead;
      size_t bit;
#if !defined(NDEBUG)
      { /* the stream fits: the widths are read only once the header fits */
        size_t nbits = 0, j;
        LIBXS_ASSERT(nhead <= npacked);
        for (j = 0; j < ngroups && nhead <= npacked; ++j) nbits += (size_t)head[1 + j];
        LIBXS_ASSERT(nhead + 8 * ((nbits + 63) / 64) <= npacked);
      }
#endif
      if (8 == typesize) {
        bit = internal_libxs_vunpack64_words((uint64_t*)values, head + 1, in, n, LIBXS_PACK_GROUP, is_signed);
      }
      else { /* a group at a time into a buffer */
        uint64_t buf[LIBXS_PACK_GROUP];
        size_t i, k, j = 0;
        int off = 0;
        for (i = 0; i < ngroups; ++i) {
          const size_t base = i * LIBXS_PACK_GROUP, m = LIBXS_MIN((size_t)LIBXS_PACK_GROUP, n - base);
          internal_libxs_vunpack8_col(buf, in, &j, &off, head[1 + i]);
          if (0 != is_signed) {
            for (k = 0; k < m; ++k) buf[k] = (buf[k] >> 1) ^ (0 - (buf[k] & 1));
          }
          internal_libxs_pack_store(values, buf, type, base, m);
        }
        bit = 64 * 8 * (j + (0 != off ? 1 : 0));
      }
      result = nhead + (bit + 63) / 64;
    }
    else if (0 == head[0]) { /* the format follows */
      if (LIBXS_PACK_QUADS == (int)((((const uint64_t*)packed)[0] >> 8) & 0xFF)) {
        result = internal_libxs_vunpack_quads(values, packed, type, n, is_signed);
      }
      else if (LIBXS_PACK_CODES == head[1]) result = internal_libxs_vunpack_lanes(values, packed, type, n, is_signed);
    }
    LIBXS_ASSERT(result <= npacked);
  }
  return result;
}


/* The lowest width bits of v reversed. */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_reals_rev(uint64_t v, int width)
{
  v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
  v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
  v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8);
  v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16);
  v = (v >> 32) | (v << 32);
  return v >> (64 - width);
}


/* 2^k for a normal double (-1022 <= k <= 1023). */
LIBXS_INLINE_ALWAYS double internal_libxs_reals_pow2(int k)
{
  const uint64_t bits = ((uint64_t)(1023 + k)) << 52;
  double result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}


/* The bits of value i (F64 or F32, zero-extended). */
LIBXS_INLINE_ALWAYS uint64_t internal_libxs_reals_bits(const void* values, int f64, size_t i)
{
  uint64_t result = 0;
  if (0 != f64) memcpy(&result, (const double*)values + i, sizeof(double));
  else {
    uint32_t b;
    memcpy(&b, (const float*)values + i, sizeof(b));
    result = b;
  }
  return result;
}


/* Floating-point data (LIBXS_PACK_REALS) as integers at the level: maxbits of zero or less quantizes at a step
 * of 2^(maxbits+1) (an absolute error of at most 2^maxbits), else maxbits leading bits of a value are kept
 * (rounded, at least sign, exponent, and a bit; all of them: lossless). Values that cannot be quantized take
 * the lossless way. */
LIBXS_API_INLINE size_t internal_libxs_vpack_reals(void* packed, const void* values, libxs_data_t type,
  size_t n, int maxbits, int level)
{
  const int f64 = (LIBXS_DATATYPE_F64 == type ? 1 : 0), width = (0 != f64 ? 64 : 32), expbits = (0 != f64 ? 11 : 8);
  const uint64_t smask = ((uint64_t)1) << (width - 1);
  const uint64_t expall = (smask - 1) & ~((((uint64_t)1) << (width - 1 - expbits)) - 1); /* exponent all ones */
  int pool = 0;
  uint64_t *const u = (uint64_t*)internal_libxs_scratch_malloc(n * sizeof(uint64_t), &pool);
  size_t result = 0;
  if (NULL != u) {
    uint64_t *const out = (uint64_t*)packed;
    int transform = -1, keep = LIBXS_MIN(LIBXS_MAX(maxbits, 2 + expbits), width), t = 0; /* a mantissa bit */
    size_t i, nwords;
    if (0 >= maxbits) { /* quantized, if every value can be (a subnormal flushed to zero stays within 2^e) */
      const int lowest = (0 != f64 ? -1022 : -126), e = LIBXS_MAX(maxbits, lowest);
      const double scale = internal_libxs_reals_pow2(-(e + 1)), big = internal_libxs_reals_pow2(62);
      const double exact = internal_libxs_reals_pow2(52); /* from here on, every double is an integer */
      int ok = (lowest <= maxbits ? 1 : 0); /* else lossless: an error below the normal numbers */
      for (i = 0; i < n && 0 != ok; ++i) {
        const double x = (0 != f64 ? ((const double*)values)[i] : (double)((const float*)values)[i]), v = x * scale;
        if (expall != (internal_libxs_reals_bits(values, f64, i) & expall) && -big < v && v < big) {
          u[i] = (uint64_t)((-exact < v && v < exact) ? (int64_t)(v + (0 <= v ? 0.5 : -0.5)) : (int64_t)v);
        }
        else ok = 0;
      }
      if (0 != ok) {
        transform = LIBXS_PACK_REAL_ABS;
        t = e;
      }
      else keep = width;
    }
    if (0 > transform && width == keep) { /* integral values as integers (no -0, no loss) */
      const double limit = internal_libxs_reals_pow2(0 != f64 ? 53 : 24);
      int ok = 1;
      for (i = 0; i < n && 0 != ok; ++i) {
        const double x = (0 != f64 ? ((const double*)values)[i] : (double)((const float*)values)[i]);
        const uint64_t b = internal_libxs_reals_bits(values, f64, i);
        if ((0 == b || (0 != (b & expall) && expall != (b & expall))) /* +0 or normal: no -0, subnormal, NaN, Inf */
          && -limit <= x && x <= limit && (double)(int64_t)x == x)
        {
          u[i] = (uint64_t)(int64_t)x;
        }
        else ok = 0;
      }
      if (0 != ok) transform = LIBXS_PACK_REAL_INT;
    }
    if (0 > transform) { /* sign at bit 0, XOR with the previous value; bit-reversed if that takes less */
      const int drop = width - keep;
      uint64_t prev = 0, gx = 0, gr = 0;
      size_t costx = 0, costr = 0;
      for (i = 0; i < n; ++i) {
        const uint64_t b = internal_libxs_reals_bits(values, f64, i);
        uint64_t mag = b & (smask - 1), s, v;
        if (0 < drop) {
          if (expall != (mag & expall)) { /* finite: round, but not into Inf */
            mag += ((uint64_t)1) << (drop - 1);
            if (expall == (mag & expall)) mag = expall - 1;
          }
          else if (expall != mag) mag |= ((uint64_t)1) << drop; /* NaN stays NaN */
          mag >>= drop;
        }
        s = (mag << 1) | (b >> (width - 1));
        v = s ^ prev;
        prev = s;
        u[i] = v;
        gx |= v;
        gr |= internal_libxs_reals_rev(v, keep);
        if (LIBXS_PACK_GROUP - 1 == (i % LIBXS_PACK_GROUP) || n == i + 1) { /* estimate: the widths of the groups */
          costx += LIBXS_NBITS(gx);
          costr += LIBXS_NBITS(gr);
          gx = gr = 0;
        }
      }
      transform = LIBXS_PACK_REAL_XOR;
      if (costr < costx) {
        transform = LIBXS_PACK_REAL_REV;
        for (i = 0; i < n; ++i) u[i] = internal_libxs_reals_rev(u[i], keep);
      }
    }
    if (LIBXS_PACK_REAL_ABS == transform) {
      nwords = internal_libxs_vpack_int(out + 1, u, LIBXS_DATATYPE_I64, n, 64, level);
    }
    else if (LIBXS_PACK_REAL_INT == transform) { /* at most 2^53 (2^24) in magnitude */
      nwords = internal_libxs_vpack_int(out + 1, u, LIBXS_DATATYPE_I64, n, 0 != f64 ? 55 : 26, level);
    }
    else nwords = internal_libxs_vpack_int(out + 1, u, LIBXS_DATATYPE_U64, n, keep, level);
    if (0 != nwords) {
      out[0] = (((uint64_t)((unsigned int)t & 0xFFFF)) << 32) | (((uint64_t)keep) << 24)
        | (((uint64_t)transform) << 16) | (((uint64_t)LIBXS_PACK_REALS) << 8);
      result = 1 + nwords;
    }
    internal_libxs_scratch_free(u, pool);
  }
  return result;
}


/* Decodes the format of internal_libxs_vpack_reals; returns the words consumed, zero if invalid. */
LIBXS_API_INLINE size_t internal_libxs_vunpack_reals(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  const uint64_t head = ((const uint64_t*)packed)[0];
  const int f64 = (LIBXS_DATATYPE_F64 == type ? 1 : 0), width = (0 != f64 ? 64 : 32), expbits = (0 != f64 ? 11 : 8);
  const int transform = (int)((head >> 16) & 0xFF), keep = (int)((head >> 24) & 0xFF);
  const int t = (int)(int16_t)((head >> 32) & 0xFFFF);
  size_t result = 0;
  if (LIBXS_PACK_REALS == (int)((head >> 8) & 0xFF) && LIBXS_PACK_REAL_ABS >= transform && 1 < npacked
    && 2 + expbits <= keep && width >= keep)
  {
    const int integers = (LIBXS_PACK_REAL_ABS == transform || LIBXS_PACK_REAL_INT == transform);
    int pool = 0;
    uint64_t *const u = (uint64_t*)internal_libxs_scratch_malloc(n * sizeof(uint64_t), &pool);
    if (NULL != u) {
      const size_t nwords = internal_libxs_vunpack_int(u, (const uint64_t*)packed + 1, npacked - 1,
        0 != integers ? LIBXS_DATATYPE_I64 : LIBXS_DATATYPE_U64, n);
      if (0 != nwords) {
        const double step = (LIBXS_PACK_REAL_ABS == transform ? internal_libxs_reals_pow2(t + 1) : 1.0);
        uint64_t s = 0;
        size_t i;
        for (i = 0; i < n; ++i) {
          if (0 != integers) { /* exact: the step is a power of two */
            const double x = (double)(int64_t)u[i] * step;
            if (0 != f64) ((double*)values)[i] = x;
            else ((float*)values)[i] = (float)x;
          }
          else { /* XOR with the previous value undone, then the sign back on top */
            const uint64_t v = (LIBXS_PACK_REAL_REV == transform ? internal_libxs_reals_rev(u[i], keep) : u[i]);
            uint64_t b;
            s ^= v;
            b = ((s >> 1) << (width - keep)) | ((s & 1) << (width - 1));
            if (0 != f64) memcpy((double*)values + i, &b, sizeof(double));
            else {
              const uint32_t b32 = (uint32_t)b;
              memcpy((float*)values + i, &b32, sizeof(b32));
            }
          }
        }
        result = 1 + nwords;
      }
      internal_libxs_scratch_free(u, pool);
    }
  }
  return result;
}


/* The exponent e of a positive finite x: 2^e <= x < 2^(e+1) (exact, subnormals included). */
LIBXS_INLINE_ALWAYS int internal_libxs_reals_log2(double x)
{
  uint64_t bits;
  int result;
  memcpy(&bits, &x, sizeof(bits));
  result = (int)((bits >> 52) & 0x7FF);
  if (0 != result) result -= 1023;
  else result = -1075 + (int)LIBXS_NBITS(bits & 0x000FFFFFFFFFFFFFULL);
  return result;
}


LIBXS_API int libxs_vpack_maxbits(libxs_data_t type, double value, int relative)
{
  const int typesize = internal_libxs_pack_typesize(type);
  int result = 0;
  if (0 < typesize) { /* the bits for magnitudes up to value (a sign bit more for signed types) */
    const double v = (0 <= value ? value : -value);
    int nbits = 8 * typesize;
    if (v < 18446744073709551616.0) { /* 2^64 */
      uint64_t u = (uint64_t)v;
      if ((double)u < v) ++u; /* up */
      nbits = (int)LIBXS_NBITS(u) + internal_libxs_pack_signed(type);
    }
    result = LIBXS_MIN(LIBXS_MAX(nbits, 1), 8 * typesize);
  }
  else if (LIBXS_DATATYPE_F64 == type || LIBXS_DATATYPE_F32 == type) { /* by bits: subnormals may be flushed */
    const int width = (LIBXS_DATATYPE_F64 == type ? 64 : 32), expbits = (LIBXS_DATATYPE_F64 == type ? 11 : 8);
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (0 != (bits >> 63) || 0 == bits || 0x7FF0000000000000ULL < bits) result = width; /* <= 0 or NaN: lossless */
    else {
      const int e = (0x7FF0000000000000ULL != bits ? internal_libxs_reals_log2(value) : 1024);
      if (0 == relative) result = LIBXS_MIN(e, 0); /* the largest power of two up to value */
      else { /* m mantissa bits round to within 2^-(m+1) of a value: m = ceil(-log2(value)) - 1 */
        result = LIBXS_MIN(1 + expbits + LIBXS_MAX(-e - 1, 1), width);
      }
    }
  }
  return result;
}


LIBXS_API size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int level)
{
  size_t result;
  if (LIBXS_DATATYPE_F64 == type || LIBXS_DATATYPE_F32 == type) {
    result = ((0 < n && NULL != packed && NULL != values)
      ? internal_libxs_vpack_reals(packed, values, type, n, maxbits, level) : 0);
  }
  else result = internal_libxs_vpack_int(packed, values, type, n, maxbits, level);
  return result;
}


LIBXS_API size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  size_t result;
  if (LIBXS_DATATYPE_F64 == type || LIBXS_DATATYPE_F32 == type) {
    result = ((0 < n && 0 < npacked && NULL != packed && NULL != values)
      ? internal_libxs_vunpack_reals(values, packed, npacked, type, n) : 0);
  }
  else result = internal_libxs_vunpack_int(values, packed, npacked, type, n);
  return result;
}
