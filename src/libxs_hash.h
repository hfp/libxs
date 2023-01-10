/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_HASH_H
#define LIBXS_HASH_H

#include <libxs_macros.h>

/** Map number of Bits to corresponding routine. */
#define LIBXS_CRC32U(N) LIBXS_CONCATENATE(libxs_crc32_u, N)
/** Calculate CRC32-value of the given pointer. */
#define LIBXS_CRCPTR(SEED, PTR) LIBXS_CRC32U(LIBXS_BITS)(SEED, &(PTR))
/** Map number of Bytes to number of bits. */
#define LIBXS_CRC32(N) LIBXS_CONCATENATE(libxs_crc32_b, N)
#define libxs_crc32_b4 libxs_crc32_u32
#define libxs_crc32_b8 libxs_crc32_u64
#define libxs_crc32_b16 libxs_crc32_u128
#define libxs_crc32_b32 libxs_crc32_u256
#define libxs_crc32_b48 libxs_crc32_u384
#define libxs_crc32_b64 libxs_crc32_u512


/** Function type representing the CRC32 functionality. */
LIBXS_EXTERN_C typedef unsigned int (*libxs_hash_function)(
  unsigned int /*seed*/, const void* /*data*/, ... /*size*/);

/** Initialize hash function module; not thread-safe. */
LIBXS_API_INTERN void libxs_hash_init(int target_arch);
LIBXS_API_INTERN void libxs_hash_finalize(void);

LIBXS_API_INTERN unsigned int libxs_crc32_u32(unsigned int seed, const void* value, ...);
LIBXS_API_INTERN unsigned int libxs_crc32_u64(unsigned int seed, const void* value, ...);
LIBXS_API_INTERN unsigned int libxs_crc32_u128(unsigned int seed, const void* value, ...);
LIBXS_API_INTERN unsigned int libxs_crc32_u256(unsigned int seed, const void* value, ...);
LIBXS_API_INTERN unsigned int libxs_crc32_u384(unsigned int seed, const void* value, ...);
LIBXS_API_INTERN unsigned int libxs_crc32_u512(unsigned int seed, const void* value, ...);

/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXS_API_INTERN unsigned int libxs_crc32(unsigned int seed, const void* data, size_t size);

#endif /*LIBXS_HASH_H*/
