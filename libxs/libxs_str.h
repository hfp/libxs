/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_STR_H
#define LIBXS_STR_H

#include "libxs.h"


/** Matching strategy for libxs_strisimilar. */
typedef enum libxs_strisimilar_t {
  LIBXS_STRISIMILAR_GREEDY = 0,
  LIBXS_STRISIMILAR_TWOOPT = 1,
  LIBXS_STRISIMILAR_DEFAULT = LIBXS_STRISIMILAR_GREEDY
} libxs_strisimilar_t;


/**
 * Case-insensitive character-level edit distance between two strings, counting a
 * transposition of adjacent characters as one edit and not two (restricted
 * Damerau-Levenshtein). Transposition is the commonest typing error, so charging
 * it twice puts a word further from its own misspelling than from words it has
 * nothing to do with. The restricted form is not a true metric: it satisfies no
 * triangle inequality, which a threshold or a minimum does not need but an index
 * over the distance would.
 */
LIBXS_API int libxs_stridist(const char a[], const char b[]);

/**
 * Non-zero when two byte spans are equal, case-insensitively (ASCII folding).
 * Neither argument needs to be terminated, so a slice of a larger buffer can be
 * compared without copying it out. NULL equals nothing, including NULL.
 */
LIBXS_API int libxs_striequal(const char a[], size_t asize,
  const char b[], size_t bsize);

/**
 * Pointer to the 1st case-insensitive occurrence of the bsize bytes of "b"
 * within the asize bytes of "a", or NULL. Neither argument needs to be
 * terminated: a tokenizer or corpus layer holds slices, not strings, and the
 * haystack is frequently a PREFIX of a buffer whose remainder must not be
 * searched. An empty needle finds nothing.
 */
LIBXS_API const char* libxs_strimem(const char a[], size_t asize,
  const char b[], size_t bsize);

/**
 * Return the pointer to the 1st match of "b" in "a", or NULL (no match).
 * maxlen bounds the NEEDLE, not the haystack: at most maxlen leading bytes of
 * "b" have to match, and all of "a" is searched. Both must be terminated. Use
 * libxs_strimem to bound the haystack or to search without a terminator.
 */
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
 * Compute similarity between strings A and B as a minimum-cost word matching.
 * Words are split by optional delimiters (same as strimatch). Each matched word
 * pair contributes its character-level edit distance (libxs_stridist).
 * Unmatched words contribute their full length. The result is order-independent.
 * Optional order receives a word-order penalty (number of pairwise inversions
 * among matched words, 0 means same order).
 */
LIBXS_API int libxs_strisimilar(const char a[], const char b[],
  const char delims[], libxs_strisimilar_t kind, int* order);

/**
 * Return a pointer to the i-th token in a delimited string (non-destructive).
 * str: delimited string (e.g., "M,N,K" or "foo;bar;baz").
 * delims: delimiter characters (NULL defaults to ",").
 * index: 0-based token index.
 * length: if non-NULL, receives the trimmed token length.
 * Returns pointer into str, or NULL if index is out of range.
 */
LIBXS_API const char* libxs_strtoken(const char str[],
  const char delims[], int index, int* length);

/**
 * Word-level set difference: count how many words in the smaller string cannot be
 * matched (within edit distance tolerance) to any word in the larger string.
 * Matching is greedy, case-insensitive, and each word can be matched at most once.
 * Returns the number of unmatched words (0 = all words matched), or -1 on NULL input.
 * Optional count receives the total word count of the larger string.
 */
LIBXS_API int libxs_stridiff(const char a[], const char b[],
  const char delims[], int tolerance, int* count);

/**
 * Format for instance an amount of Bytes like libxs_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 */
LIBXS_API size_t libxs_format_value(char buffer[],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/**
 * Bytes spanned by the UTF-8 sequence at text[pos], taken from the LEAD BYTE
 * alone and clamped to the bytes that remain. Always at least 1, so a scan over
 * malformed input still advances instead of standing still.
 *
 * This is the lenient form, for walking text a codepoint at a time: it does not
 * inspect continuation bytes, so a truncated or corrupt sequence is stepped over
 * by its declared width rather than rejected. Use libxs_utf8_decode when the
 * codepoint VALUE is wanted, because that requires the sequence to be valid.
 */
LIBXS_API size_t libxs_utf8_size(const unsigned char text[], size_t size,
  size_t pos);

/**
 * Decode one UTF-8 sequence at text[0], writing the bytes consumed to *width
 * (may be NULL). Returns the code point.
 *
 * This is the strict form: a sequence that is truncated, or whose continuation
 * bytes are not continuation bytes, yields the LEAD BYTE as the value and a
 * width of 1. A caller testing a property of the code point (is it a vowel, is
 * it punctuation) must not be handed a value assembled from bytes that do not
 * belong to it, and width 1 keeps such a caller advancing.
 *
 * Note the deliberate difference from libxs_utf8_size, which reports the width a
 * lead byte CLAIMS. On well-formed text the two agree; on malformed text the
 * lenient form skips the claimed span while this one skips one byte.
 */
LIBXS_API unsigned long libxs_utf8_decode(const unsigned char text[],
  size_t size, int* width);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_STR_H*/
