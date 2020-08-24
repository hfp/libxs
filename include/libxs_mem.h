/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MEM_H
#define LIBXS_MEM_H

#include "libxs_macros.h"

#if defined(__clang_analyzer__)
# define LIBXS_MEMSET127(PTRDST, VALUE, SIZE) memset((void*)(PTRDST), VALUE, SIZE)
#else
# define LIBXS_MEMSET127(PTRDST, VALUE, SIZE) { \
  char *const libxs_memset127_dst_ = (char*)(PTRDST); \
  union { size_t size; signed char size1; } libxs_memset127_; \
  signed char libxs_memset127_i_; LIBXS_ASSERT((SIZE) <= 127); \
  libxs_memset127_.size = (SIZE); \
  LIBXS_PRAGMA_UNROLL \
  for (libxs_memset127_i_ = 0; libxs_memset127_i_ < libxs_memset127_.size1; \
    ++libxs_memset127_i_) \
  { \
    libxs_memset127_dst_[libxs_memset127_i_] = (char)(VALUE); \
  } \
}
#endif
#define LIBXS_MEMZERO127(PTRDST) LIBXS_MEMSET127(PTRDST, '\0', sizeof(*(PTRDST)))

#define LIBXS_MEMCPY127_LOOP(PTRDST, PTRSRC, SIZE, NTS) { \
  const unsigned char *const libxs_memcpy127_loop_src_ = (const unsigned char*)(PTRSRC); \
  unsigned char *const libxs_memcpy127_loop_dst_ = (unsigned char*)(PTRDST); \
  signed char libxs_memcpy127_loop_i_; LIBXS_ASSERT((SIZE) <= 127); \
  NTS(libxs_memcpy127_loop_dst_) LIBXS_PRAGMA_UNROLL \
  for (libxs_memcpy127_loop_i_ = 0; libxs_memcpy127_loop_i_ < (signed char)(SIZE); \
    ++libxs_memcpy127_loop_i_) \
  { \
    libxs_memcpy127_loop_dst_[libxs_memcpy127_loop_i_] = \
    libxs_memcpy127_loop_src_[libxs_memcpy127_loop_i_]; \
  } \
}
#define LIBXS_MEMCPY127_NTS(...)
#define LIBXS_MEMCPY127(PTRDST, PTRSRC, SIZE) \
  LIBXS_MEMCPY127_LOOP(PTRDST, PTRSRC, SIZE, LIBXS_MEMCPY127_NTS)
#define LIBXS_ASSIGN127(PTRDST, PTRSRC) LIBXS_ASSERT(sizeof(*(PTRSRC)) <= sizeof(*(PTRDST))); \
  LIBXS_MEMCPY127(PTRDST, PTRSRC, sizeof(*(PTRSRC)))


/**
 * Calculates if there is a difference between two (short) buffers.
 * Returns zero if there is no difference; otherwise non-zero.
 */
LIBXS_API unsigned char libxs_diff(const void* a, const void* b, unsigned char size);

/**
 * Calculates if there is a difference between "a" and "n x b".
 * Returns the index of the first match (or "n" in case of no match).
 */
LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n);

/** Similar to memcmp (C standard library), but the result is conceptually only a boolean. */
LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size);

/** Calculate a hash value for the given buffer and seed; accepts NULL-buffer. */
LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed);

/** Calculate a 64-bit hash for the given character string; accepts NULL-string. */
LIBXS_API unsigned long long libxs_hash_string(const char* string);

/**
 * Check if pointer is SIMD-aligned and optionally consider the next access (increment in Bytes).
 * Optionally calculates the alignment of the given pointer in Bytes.
 */
LIBXS_API int libxs_aligned(const void* ptr, const size_t* inc, int* alignment);

#endif /*LIBXS_MEM_H*/
