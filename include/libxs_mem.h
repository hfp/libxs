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

#include "libxs.h"

#define LIBXS_MEM_LOOP(DST, SRC, SIZE, RHS, NTS) do { \
  const signed char libxs_memory_loop_size_ = LIBXS_CAST_ICHAR(SIZE); \
  unsigned char *const LIBXS_RESTRICT libxs_memory_loop_dst_ = (unsigned char*)(DST); \
  signed char libxs_memory_loop_i_; \
  NTS(libxs_memory_loop_dst_) LIBXS_PRAGMA_UNROLL \
  for (libxs_memory_loop_i_ = 0; libxs_memory_loop_i_ < libxs_memory_loop_size_; \
    ++libxs_memory_loop_i_) \
  { \
    RHS(unsigned char, libxs_memory_loop_dst_, SRC, libxs_memory_loop_i_); \
  } \
} while(0)
#define LIBXS_MEM_NTS(...)

#define LIBXS_MEMSET_RHS(TYPE, DST, SRC, IDX) \
  ((DST)[IDX] = (TYPE)(SRC))
#define LIBXS_MEMSET(DST, SRC, SIZE) \
  LIBXS_MEM_LOOP(DST, SRC, SIZE, LIBXS_MEMSET_RHS, LIBXS_MEM_NTS)
#define LIBXS_MEMZERO(DST) LIBXS_MEMSET(DST, 0, sizeof(*(DST)))

#define LIBXS_MEMCPY_RHS(TYPE, DST, SRC, IDX) \
  ((DST)[IDX] = ((const TYPE*)(SRC))[IDX])
#define LIBXS_MEMCPY(DST, SRC, SIZE) \
  LIBXS_MEM_LOOP(DST, SRC, SIZE, LIBXS_MEMCPY_RHS, LIBXS_MEM_NTS)
#define LIBXS_ASSIGN(DST, SRC) do { \
  LIBXS_ASSERT(sizeof(*(SRC)) <= sizeof(*(DST))); \
  LIBXS_MEMCPY(DST, SRC, sizeof(*(SRC))); \
} while(0)

#define LIBXS_MEMSWP_RHS(TYPE, PTR_A, PTR_B, IDX) \
  LIBXS_ISWAP((PTR_A)[IDX], ((TYPE*)(PTR_B))[IDX])
#define LIBXS_MEMSWP(PTR_A, PTR_B, SIZE) do { \
  LIBXS_ASSERT((PTR_A) != (PTR_B)); \
  LIBXS_MEM_LOOP(PTR_A, PTR_B, SIZE, LIBXS_MEMSWP_RHS, LIBXS_MEM_NTS); \
} while (0)

/** Assigns SRC to DST (must be L-values). Can be used to cast const-qualifiers. */
#define LIBXS_VALUE_ASSIGN(DST, SRC) LIBXS_ASSIGN(&(DST), &(SRC))
/** Swap two arbitrary-sized values (must be L-values) */
#define LIBXS_VALUE_SWAP(A, B) do { \
  LIBXS_ASSERT(sizeof(A) == sizeof(B)); \
  LIBXS_MEMSWP(&(A), &(B), sizeof(A)); \
} while (0)

/**
 * Calculate the linear offset of the n-dimensional (ndims) offset (can be NULL),
 * and the (optional) linear size of the corresponding shape.
 */
LIBXS_API size_t libxs_offset(const size_t offset[], const size_t shape[], size_t ndims, size_t* size);

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
 * be in bounds [0, count); otherwise performance is impacted.
 */
LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char elemsize,
  unsigned char stride, unsigned int hint, unsigned int count);

/** Similar to memcmp (C standard library) with the result conceptually boolean. */
LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size);

/** Calculate a hash value for the given buffer and seed; accepts NULL-buffer. */
LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed);
LIBXS_API unsigned int libxs_hash8(unsigned int data);
LIBXS_API unsigned int libxs_hash16(unsigned int data);
LIBXS_API unsigned int libxs_hash32(unsigned long long data);

/** Calculate a 64-bit hash for the given character string; accepts NULL-string. */
LIBXS_API unsigned long long libxs_hash_string(const char string[]);

/** Return the pointer to the 1st match of "b" in "a", or NULL (no match). */
LIBXS_API const char* libxs_stristrn(const char a[], const char b[], size_t maxlen);
LIBXS_API const char* libxs_stristr(const char a[], const char b[]);

/**
 * Count the number of words in A (or B) with match in B (or A) respectively (case-insensitive).
 * Can be used to score the equality of A and B on a word-basis. The result is independent of
 * A-B or B-A order (symmetry). The score cannot exceed the number of words in A or B.
 * Optional delimiters determine characters splitting words (can be NULL).
 * Optional count yields total number of words.
 */
LIBXS_API int libxs_strimatch(const char a[], const char b[], const char delims[], int* count);

/**
 * Format for instance an amount of Bytes like libxs_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 * Caution: cannot be used multiple times in a single expression!
 */
LIBXS_API size_t libxs_format_value(char buffer[32],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/** Out-of-place shuffling of data given by elemsize and count. */
LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle,
  /** If NULL, the default value is one. */
  const size_t* nrepeat);

/** Out-of-place shuffling of data given by elemsize and count. */
LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle,
  /** If NULL, the default value is one. If zero, an ordinary copy is performed. */
  const size_t* nrepeat);

/** Determines the number of calls to restore the original data (libxs_shuffle2). */
LIBXS_API size_t libxs_unshuffle(
  /** The number of elements to be unshuffled. */
  size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle);

#endif /*LIBXS_MEM_H*/
