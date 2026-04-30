/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_STR_H
#define LIBXS_STR_H

#include "libxs.h"

/** Case-insensitive character-level edit distance (Levenshtein) between two strings. */
LIBXS_API int libxs_stridist(const char a[], const char b[]);

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

/** Matching strategy for libxs_strisimilar. */
typedef enum libxs_strisimilar_t {
  LIBXS_STRISIMILAR_GREEDY = 0,
  LIBXS_STRISIMILAR_TWOOPT = 1,
  LIBXS_STRISIMILAR_DEFAULT = LIBXS_STRISIMILAR_GREEDY
} libxs_strisimilar_t;

/**
 * Compute similarity between strings A and B as a minimum-cost word matching.
 * Words are split by optional delimiters (same as strimatch). Each matched word
 * pair contributes its character-level edit distance (case-insensitive Levenshtein).
 * Unmatched words contribute their full length. The result is order-independent.
 * Optional order receives a word-order penalty (number of pairwise inversions
 * among matched words, 0 means same order).
 */
LIBXS_API int libxs_strisimilar(const char a[], const char b[],
  const char delims[], libxs_strisimilar_t kind, int* order);

/**
 * Format for instance an amount of Bytes like libxs_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 */
LIBXS_API size_t libxs_format_value(char buffer[],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_STR_H*/
