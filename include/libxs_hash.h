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

#include "libxs.h"

/** Calculate a hash value for the given buffer and seed; accepts NULL-buffer. */
LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed);
LIBXS_API unsigned int libxs_hash8(unsigned int data);
LIBXS_API unsigned int libxs_hash16(unsigned int data);
LIBXS_API unsigned int libxs_hash32(unsigned long long data);

/** Calculate a CRC32 value (ISO 3309 polynomial) for the given buffer and seed. */
LIBXS_API unsigned int libxs_hash_iso3309(const void* data, unsigned int size, unsigned int seed);

/** Calculate an Adler-32 checksum for the given buffer and seed. */
LIBXS_API unsigned int libxs_adler32(const void* data, unsigned int size, unsigned int seed);

/** Calculate a 64-bit hash for the given character string; accepts NULL-string. */
LIBXS_API unsigned long long libxs_hash_string(const char string[]);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_HASH_H*/
