/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#ifndef LIBXS_HASH_H
#define LIBXS_HASH_H

#include <libxs.h>

#if !defined(LIBXS_HASH_SW) && 0
# define LIBXS_HASH_SW
#endif

#if defined(LIBXS_BUILD) && !defined(LIBXS_HASH_NOINLINE)
# define LIBXS_HASH_API LIBXS_API_INLINE
# define LIBXS_HASH_API_DEFINITION LIBXS_HASH_API LIBXS_ATTRIBUTE_UNUSED
#else
# define LIBXS_HASH_API LIBXS_API
# define LIBXS_HASH_API_DEFINITION LIBXS_API
#endif


/** Function type representing the CRC32 functionality. */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE unsigned int (*libxs_hash_function)(
  const void*, size_t, unsigned int);

/** Initialize hash function module; not thread-safe. */
LIBXS_HASH_API void libxs_hash_init(int target_arch);
LIBXS_HASH_API void libxs_hash_finalize(void);

LIBXS_HASH_API unsigned int libxs_crc32_u32(unsigned int seed, unsigned int value);
LIBXS_HASH_API unsigned int libxs_crc32_u64(unsigned int seed, unsigned long long value);

/** Dispatched implementation which may (or may not) use a SIMD extension. */
LIBXS_HASH_API unsigned int libxs_crc32(
  const void* data, size_t size, unsigned int seed);

/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXS_HASH_API unsigned int libxs_crc32_sw(
  const void* data, size_t size, unsigned int seed);

/** Similar to libxs_crc32_sw (uses CRC32 instructions available since SSE4.2). */
LIBXS_HASH_API unsigned int libxs_crc32_sse4(
  const void* data, size_t size, unsigned int seed);


#if defined(LIBXS_BUILD) && !defined(LIBXS_HASH_NOINLINE)
# include "libxs_hash.c"
#endif

#endif /*LIBXS_HASH_H*/
