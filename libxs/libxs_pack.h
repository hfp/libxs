/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_PACK_H
#define LIBXS_PACK_H

#include "libxs.h"


/**
 * Packs n integers of the given type (I8..I64, U8..U64) at nbits each (1 to
 * 8 times the type size) into CEILING(n*nbits/64) 64-bit words, which must be
 * aligned accordingly. Bits beyond nbits are ignored. Every group of 64 values
 * fills exactly nbits words, the first value in the least significant bits.
 * Returns EXIT_SUCCESS, or EXIT_FAILURE for a type or width out of range.
 */
LIBXS_API int libxs_pack(void* packed, const void* values,
  libxs_data_t type, size_t n, int nbits);
/**
 * Unpacks n integers written by libxs_pack. Signed types are sign-extended
 * from nbits, unsigned types zero-extended.
 */
LIBXS_API int libxs_unpack(void* values, const void* packed,
  libxs_data_t type, size_t n, int nbits);

/**
 * Variable width: packs n values of the given type (I8..I64, U8..U64, F64,
 * F32) into 64-bit words whose number follows the magnitudes, so small values
 * or zeros cost little. Integers are taken as integers of maxbits bits (signed
 * types sign-extended from maxbits, unsigned types masked); small magnitudes
 * of either sign stay small. For F64 and F32, maxbits sets the accuracy: the
 * width of the type (or more) is lossless, fewer bits keep that many leading
 * bits of each value (at least sign, exponent, and a bit: a relative error),
 * and zero or less bounds the absolute error by 2^maxbits (values that cannot
 * be held to it, such as Inf or NaN, are kept exactly, as are all values for
 * an error below the smallest normal number). The level selects a
 * trade-off: 1 packs and unpacks fastest, 2 takes less space, 3 (or more) the
 * least; 0 (or less) takes the default (2, or LIBXS_PACK in the environment).
 * The stream is internal and may change between versions; only n has to be
 * known to unpack. Returns the number of 64-bit words written, at most
 * libxs_vpack_bound, or zero for invalid arguments or n of zero.
 */
LIBXS_API size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int level);
/** Number of 64-bit words libxs_vpack writes at most (any level). */
LIBXS_API size_t libxs_vpack_bound(size_t n, int maxbits);
/**
 * The maxbits of libxs_vpack for a value: for integer types the bits that
 * magnitudes up to value need (with a sign bit for signed types); for F64 and
 * F32 the bits that keep the error within value, absolute (relative = 0) or
 * relative to each value (relative != 0; not for subnormal values), a value
 * of zero or less meaning lossless. Zero for other types.
 */
LIBXS_API int libxs_vpack_maxbits(libxs_data_t type, double value, int relative);
/**
 * Unpacks n integers written by libxs_vpack from npacked 64-bit words, which
 * have to hold the stream (checked in debug code only). Returns the number of
 * 64-bit words consumed, or zero for invalid arguments, n of zero, or npacked
 * of zero (values are left untouched).
 */
LIBXS_API size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n);

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PACK_H*/
