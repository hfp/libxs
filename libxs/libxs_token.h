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

#include "libxs_reg.h"

/** Fixed token size in bytes. */
#define LIBXS_TOKEN_BYTES 16
/** Maximum literal payload bytes per token. */
#define LIBXS_TOKEN_PAYLOAD 14
/** Minimum copy match length (bytes). */
#define LIBXS_TOKEN_MIN_COPY 3
/** Maximum copy match length (bytes). */
#define LIBXS_TOKEN_MAX_COPY 14
/** Maximum backward copy distance (32-bit, ~4GB window). */
#define LIBXS_TOKEN_MAX_DISTANCE 4294967295u

#define LIBXS_TOKEN_FLAG_COPY 0x80u
#define LIBXS_TOKEN_FLAG_MARKUP 0x20u
#define LIBXS_TOKEN_FLAG_SENTENCE 0x10u
#define LIBXS_TOKEN_FLAG_BREAK 0x08u
#define LIBXS_TOKEN_LEN_MASK 0x0Fu


/**
 * Fixed-width token (16 bytes).
 * Byte 0: flags (copy, markup, sentence-end, break).
 * Byte 1: payload length (1..14).
 * Bytes 2-15: literal content, or bytes 2-5 = 32-bit copy distance.
 */
LIBXS_EXTERN_C typedef struct libxs_token_t {
  unsigned char raw[LIBXS_TOKEN_BYTES];
} libxs_token_t;

/** Growable array of tokens. */
LIBXS_EXTERN_C typedef struct libxs_token_stream_t {
  libxs_token_t* data;
  size_t size;
  size_t capacity;
} libxs_token_stream_t;

/** Decoded token properties. */
LIBXS_EXTERN_C typedef struct libxs_token_info_t {
  size_t length;
  unsigned int distance;
  int is_copy;
  int has_break;
  int is_sentence;
  int is_markup;
} libxs_token_info_t;

/**
 * Text rule template IDs: positional match patterns.
 * Templates are compiled logic; rules are loadable data that
 * select a template and supply an argument.
 */
enum libxs_textrule_template_t {
  /** Suppress if word before position is short (<=4) and uppercase-initial. */
  LIBXS_TRULE_PREV_WORD_SHORT_UPPER = 1,
  /** Match if character after whitespace following position is uppercase. */
  LIBXS_TRULE_NEXT_CHAR_UPPER = 2,
  /** Match if character after whitespace following position is lowercase. */
  LIBXS_TRULE_NEXT_CHAR_LOWER = 3,
  /** Match if the previous token has the BREAK flag set. */
  LIBXS_TRULE_PREV_TOKEN_BREAK = 4,
  /** Match if position is inside paired delimiters (argument = open char). */
  LIBXS_TRULE_INSIDE_DELIMITERS = 5
};

/** Actions a matched rule can take. */
enum libxs_textrule_action_t {
  LIBXS_TRULE_SUPPRESS = 0,
  LIBXS_TRULE_CONFIRM = 1
};

/**
 * Fixed 16-byte rule: storable in a registry, serializable.
 * tmpl selects the match pattern, argument parameterizes it,
 * action determines the outcome when matched.
 */
LIBXS_EXTERN_C typedef struct libxs_textrule_t {
  unsigned char tmpl;
  unsigned char action;
  unsigned char reserved[2];
  unsigned int argument;
  unsigned char pad[8];
} libxs_textrule_t;

/** Context window passed to rule evaluation. */
LIBXS_EXTERN_C typedef struct libxs_textrule_ctx_t {
  const unsigned char* text;
  size_t text_size;
  int byte_pos;
  const libxs_token_t* token;
  const libxs_token_t* prev_token;
} libxs_textrule_ctx_t;


/** Decode all token properties into an info struct. */
LIBXS_API void libxs_token_info(const libxs_token_t* token,
  libxs_token_info_t* info);

/** Initialize an empty token stream. */
LIBXS_API void libxs_token_stream_init(libxs_token_stream_t* stream);

/** Ensure the stream can hold at least capacity tokens without reallocation. */
LIBXS_API int libxs_token_stream_reserve(libxs_token_stream_t* stream,
  size_t capacity);

/** Append a token to the stream, growing capacity as needed. */
LIBXS_API int libxs_token_stream_push(libxs_token_stream_t* stream,
  const libxs_token_t* token);

/** Release all memory held by the stream (the stream struct itself is not freed). */
LIBXS_API void libxs_token_stream_release(libxs_token_stream_t* stream);

/**
 * Encode a byte sequence into an initialized stream of fixed-width tokens.
 * Each token is 16 bytes: flags, length, and up to 14 payload bytes.
 * Copy tokens reference earlier content via a 32-bit backward distance.
 * Break and sentence-end flags are set from punctuation/whitespace boundaries.
 */
LIBXS_API int libxs_token_stream_encode(libxs_token_stream_t* stream,
  const unsigned char* text, size_t size);

/**
 * Decode a token stream back into a byte sequence.
 * Allocates *text (caller must free). Roundtrip after encode preserves input.
 */
LIBXS_API int libxs_token_stream_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size);

/**
 * Evaluate a ruleset against a context. Returns 1 (sentence boundary
 * confirmed) or 0 (suppressed). If the token lacks the sentence-end flag,
 * returns 0 without consulting rules. Last matching rule wins.
 */
LIBXS_API int libxs_textrule_eval(const libxs_textrule_ctx_t* ctx,
  const libxs_textrule_t* rules, int nrules);

/** Hash a short word (for use as rule argument in PREV_WORD matching). */
LIBXS_API unsigned int libxs_textrule_wordhash(
  const unsigned char* word, int len);

/** Load rules from registry (key prefix "TRULE:"). Returns count loaded. */
LIBXS_API int libxs_textrule_load(const libxs_registry_t* registry,
  libxs_textrule_t* rules, int max_rules);

/** Save rules to registry (key prefix "TRULE:"). */
LIBXS_API int libxs_textrule_save(libxs_registry_t* registry,
  const libxs_textrule_t* rules, int nrules);

/** Populate rules[] with built-in defaults. Returns count written. */
LIBXS_API int libxs_textrule_defaults(libxs_textrule_t* rules, int max_rules);

/**
 * Reflow text: replace cosmetic newlines (column-wrap artifacts) with spaces,
 * preserving structural newlines (enumerations, headings, blank lines, verse).
 * Allocates *out (caller must free). Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_text_reflow(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size);

/** Return non-zero if the token is a copy (back-reference). */
LIBXS_API_INLINE int libxs_token_is_copy(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_COPY)) ? 1 : 0;
}

/** Return the number of bytes this token represents (1..14). */
LIBXS_API_INLINE size_t libxs_token_len(const libxs_token_t* token)
{
  return (NULL != token) ? (size_t)(token->raw[1] & LIBXS_TOKEN_LEN_MASK) : 0;
}

/** Return non-zero if a preferred break (word/punctuation boundary) follows. */
LIBXS_API_INLINE int libxs_token_has_break(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_BREAK)) ? 1 : 0;
}

/** Return non-zero if this token is structural markup (not content). */
LIBXS_API_INLINE int libxs_token_is_markup(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_MARKUP)) ? 1 : 0;
}

/** Return non-zero if this token ends a sentence (raw signal, pre-rules). */
LIBXS_API_INLINE int libxs_token_is_sentence_end(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_SENTENCE)) ? 1 : 0;
}

/** Return the backward distance of a copy token (0 if not a copy). */
LIBXS_API_INLINE unsigned int libxs_token_distance(const libxs_token_t* token)
{
  if (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_FLAG_COPY)) {
    return (unsigned int)token->raw[2]
      | ((unsigned int)token->raw[3] << 8)
      | ((unsigned int)token->raw[4] << 16)
      | ((unsigned int)token->raw[5] << 24);
  }
  return 0;
}

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_TOKEN_H*/
