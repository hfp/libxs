/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_TOKEN_H
#define LIBXS_TOKEN_H

#include "libxs.h"

#include <stdlib.h>

#define LIBXS_TOKEN_BYTES 8
#define LIBXS_TOKEN_PAYLOAD 7
#define LIBXS_TOKEN_MIN_COPY 3
#define LIBXS_TOKEN_MAX_COPY 7
#define LIBXS_TOKEN_MAX_DISTANCE 65535u

#define LIBXS_TOKEN_FLAG_COPY 0x80u
#define LIBXS_TOKEN_FLAG_MARKUP 0x20u
#define LIBXS_TOKEN_FLAG_SENTENCE 0x10u
#define LIBXS_TOKEN_FLAG_BREAK 0x08u
#define LIBXS_TOKEN_LEN_MASK 0x07u


LIBXS_EXTERN_C typedef struct libxs_token_t {
  unsigned char raw[LIBXS_TOKEN_BYTES];
} libxs_token_t;

LIBXS_EXTERN_C typedef struct libxs_token_stream_t {
  libxs_token_t* data;
  size_t size;
  size_t capacity;
} libxs_token_stream_t;

LIBXS_EXTERN_C typedef struct libxs_token_info_t {
  int length;
  int distance;
  int is_copy;
  int has_break;
  int is_sentence;
  int is_markup;
} libxs_token_info_t;

/** Decode all token properties into an info struct. */
LIBXS_API void libxs_token_info(libxs_token_info_t* info,
  const libxs_token_t* token);

/** Ensure the stream can hold at least capacity tokens without reallocation. */
LIBXS_API int libxs_token_stream_reserve(libxs_token_stream_t* stream,
  size_t capacity);

/** Append a token to the stream, growing capacity as needed. */
LIBXS_API int libxs_token_stream_push(libxs_token_stream_t* stream,
  const libxs_token_t* token);

/** Release all memory held by the stream (the stream struct itself is not freed). */
LIBXS_API void libxs_token_stream_destroy(libxs_token_stream_t* stream);

/**
 * Tokenize a UTF-8 byte sequence into a stream of fixed-width tokens.
 * Each token is 8 bytes: a control byte followed by 7 payload bytes.
 * Literal tokens store up to 7 bytes of content directly.
 * Copy tokens reference earlier content via a 16-bit backward distance.
 * Break and sentence-end flags are set automatically from punctuation
 * and whitespace boundaries in the input.
 */
LIBXS_API int libxs_tokenize(const unsigned char* text, size_t size,
  libxs_token_stream_t* stream);

/**
 * Decode a token stream back into a UTF-8 byte sequence.
 * Allocates *text (caller must free). Roundtrip: decode(tokenize(t)) == t.
 */
LIBXS_API int libxs_token_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size);

/** Return non-zero if the token is a copy (back-reference), zero if literal. */
LIBXS_API_INLINE int libxs_token_is_copy(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_COPY)) ? 1 : 0;
}

/** Return the number of bytes this token represents (1..7). */
LIBXS_API_INLINE size_t libxs_token_len(const libxs_token_t* token)
{
  return (NULL != token) ? (size_t)(token->raw[0] & LIBXS_TOKEN_LEN_MASK) : 0;
}

/** Return non-zero if a preferred break (word/punctuation boundary) follows this token. */
LIBXS_API_INLINE int libxs_token_has_break(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_BREAK)) ? 1 : 0;
}

/** Return non-zero if this token is structural markup (not content). */
LIBXS_API_INLINE int libxs_token_is_markup(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_MARKUP)) ? 1 : 0;
}

/** Return non-zero if this token ends a sentence (. ? ! followed by space or end). */
LIBXS_API_INLINE int libxs_token_is_sentence_end(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_SENTENCE)) ? 1 : 0;
}

/** Return the backward distance of a copy token (0 if not a copy token). */
LIBXS_API_INLINE unsigned int libxs_token_distance(const libxs_token_t* token)
{
  if (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_COPY)) {
    return (unsigned int)token->raw[1] | ((unsigned int)token->raw[2] << 8);
  }
  return 0;
}

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_TOKEN_H*/
