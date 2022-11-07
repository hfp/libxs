/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MEM_H
#define LIBXS_MEM_H

#include "libxs_macros.h"

#define LIBXS_MEM127_LOOP(DST, SRC, SIZE, RHS, NTS) do { \
  const signed char libxs_memory127_loop_size_ = LIBXS_CAST_ICHAR(SIZE); \
  unsigned char *const LIBXS_RESTRICT libxs_memory127_loop_dst_ = (unsigned char*)(DST); \
  signed char libxs_memory127_loop_i_; \
  NTS(libxs_memory127_loop_dst_) LIBXS_PRAGMA_UNROLL \
  for (libxs_memory127_loop_i_ = 0; libxs_memory127_loop_i_ < libxs_memory127_loop_size_; \
    ++libxs_memory127_loop_i_) \
  { \
    RHS(unsigned char, libxs_memory127_loop_dst_, SRC, libxs_memory127_loop_i_); \
  } \
} while(0)
#define LIBXS_MEM127_NTS(...)

#define LIBXS_MEMSET127_RHS(TYPE, DST, SRC, IDX) \
  ((DST)[IDX] = (TYPE)(SRC))
#define LIBXS_MEMSET127(DST, SRC, SIZE) \
  LIBXS_MEM127_LOOP(DST, SRC, SIZE, \
  LIBXS_MEMSET127_RHS, LIBXS_MEM127_NTS)
#define LIBXS_MEMZERO127(DST) LIBXS_MEMSET127(DST, 0, sizeof(*(DST)))

#define LIBXS_MEMCPY127_RHS(TYPE, DST, SRC, IDX) \
  ((DST)[IDX] = ((const TYPE*)(SRC))[IDX])
#define LIBXS_MEMCPY127(DST, SRC, SIZE) \
  LIBXS_MEM127_LOOP(DST, SRC, SIZE, \
  LIBXS_MEMCPY127_RHS, LIBXS_MEM127_NTS)
#define LIBXS_ASSIGN127(DST, SRC) do { \
  LIBXS_ASSERT(sizeof(*(SRC)) <= sizeof(*(DST))); \
  LIBXS_MEMCPY127(DST, SRC, sizeof(*(SRC))); \
} while(0)

#define LIBXS_MEMSWP127_RHS(TYPE, DST, SRC, IDX) \
  LIBXS_ISWAP((DST)[IDX], ((TYPE*)(SRC))[IDX])
#define LIBXS_MEMSWP127(DST, SRC, SIZE) \
  LIBXS_MEM127_LOOP(DST, SRC, SIZE, \
  LIBXS_MEMSWP127_RHS, LIBXS_MEM127_NTS)


/**
 * Check if pointer is SIMD-aligned and optionally consider the next access (increment in Bytes).
 * Optionally calculates the alignment of the given pointer in Bytes.
 */
LIBXS_API int libxs_aligned(const void* ptr, const size_t* inc, int* alignment);

/**
 * Calculates if there is a difference between two (short) buffers.
 * Returns zero if there is no difference; otherwise non-zero.
 */
LIBXS_API unsigned char libxs_diff(const void* a, const void* b, unsigned char size);

/**
 * Calculates the "difference" between "a" and "b"; "a" is taken "count" times into account.
 * Returns the first match (index) of no difference (or "n" if "a" did not match).
 * The hint determines the initial index searching for a difference, and it must
 * be in bounds [0, count), but otherwise only impacts performance.
 */
LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char elemsize,
  unsigned char stride, unsigned int hint, unsigned int count);

/** Similar to memcmp (C standard library), but the result is conceptually only a boolean. */
LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size);

/** Calculate a hash value for the given buffer and seed; accepts NULL-buffer. */
LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed);

/** Calculate a 64-bit hash for the given character string; accepts NULL-string. */
LIBXS_API unsigned long long libxs_hash_string(const char string[]);

/** Return the pointer to the 1st match of "b" in "a", or NULL (no match). */
LIBXS_API const char* libxs_stristr(const char a[], const char b[]);

/** Out-of-place shuffling of data given by elemsize and count. */
LIBXS_API void libxs_shuffle(void* dst, const void* src, size_t elemsize, size_t count);

#endif /*LIBXS_MEM_H*/
