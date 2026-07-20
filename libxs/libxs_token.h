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
#define LIBXS_TOKEN_BYTES 8
/** Maximum normalized text bytes per token. */
#define LIBXS_TOKEN_MAXBYTES 63

#define LIBXS_TOKEN_WORD     0x0001u
#define LIBXS_TOKEN_NUMBER   0x0002u
#define LIBXS_TOKEN_PUNCT    0x0004u
#define LIBXS_TOKEN_MARKUP   0x0008u
#define LIBXS_TOKEN_SENTENCE 0x0010u
#define LIBXS_TOKEN_QUESTION 0x0020u
#define LIBXS_TOKEN_STOP     0x0040u
#define LIBXS_TOKEN_ENTITY   0x0080u
#define LIBXS_TOKEN_BREAK    0x0100u

#define LIBXS_LEXEME_BYTES LIBXS_TOKEN_BYTES
#define LIBXS_LEXEME_MAXBYTES LIBXS_TOKEN_MAXBYTES
#define LIBXS_LEXEME_WORD LIBXS_TOKEN_WORD
#define LIBXS_LEXEME_NUMBER LIBXS_TOKEN_NUMBER
#define LIBXS_LEXEME_PUNCT LIBXS_TOKEN_PUNCT
#define LIBXS_LEXEME_MARKUP LIBXS_TOKEN_MARKUP
#define LIBXS_LEXEME_SENTENCE LIBXS_TOKEN_SENTENCE
#define LIBXS_LEXEME_QUESTION LIBXS_TOKEN_QUESTION
#define LIBXS_LEXEME_STOP LIBXS_TOKEN_STOP
#define LIBXS_LEXEME_ENTITY LIBXS_TOKEN_ENTITY
#define LIBXS_LEXEME_BREAK LIBXS_TOKEN_BREAK


/** Fixed-width token ID: vocabulary id, source byte length, and class flags. */
LIBXS_EXTERN_C typedef struct libxs_token_t {
  unsigned int id;
  unsigned short length;
  unsigned short flags;
} libxs_token_t;

/** Growable array of tokens. */
LIBXS_EXTERN_C typedef struct libxs_token_stream_t {
  libxs_token_t* data;
  size_t size;
  size_t capacity;
} libxs_token_stream_t;

/** Opaque lexical vocabulary. Token id 0 is reserved for unknown. */
LIBXS_EXTERN_C typedef struct libxs_lexicon_t libxs_lexicon_t;

typedef libxs_token_t libxs_lexeme_t;
typedef libxs_token_stream_t libxs_lexeme_stream_t;

/** Decoded token properties. */
LIBXS_EXTERN_C typedef struct libxs_token_info_t {
  unsigned int id;
  size_t length;
  unsigned int flags;
  int is_word;
  int is_number;
  int is_punct;
  int has_break;
  int is_sentence;
  int is_question;
  int is_stop;
  int is_entity;
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

/** Lexical rule template IDs. */
enum libxs_lexrule_template_t {
  /** Match a normalized word by libxs_textrule_wordhash. */
  LIBXS_LRULE_WORD_HASH = 1,
  /** Match a word whose original spelling starts with uppercase. */
  LIBXS_LRULE_WORD_INITIAL_UPPER = 2,
  /** Match a punctuation token by character. */
  LIBXS_LRULE_PUNCT_CHAR = 3
};

/** Actions a matched rule can take. */
enum libxs_textrule_action_t {
  LIBXS_TRULE_SUPPRESS = 0,
  LIBXS_TRULE_CONFIRM = 1
};

/** Lexical rule actions. */
enum libxs_lexrule_action_t {
  LIBXS_LRULE_SET = 1,
  LIBXS_LRULE_CLEAR = 2
};

/**
 * Fixed-size rule: storable in a registry, serializable.
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

/** Fixed-size lexical classification rule. */
LIBXS_EXTERN_C typedef struct libxs_lexrule_t {
  unsigned char tmpl;
  unsigned char action;
  unsigned short flags;
  unsigned int argument;
  unsigned char pad[8];
} libxs_lexrule_t;

/** Context window passed to rule evaluation. */
LIBXS_EXTERN_C typedef struct libxs_textrule_ctx_t {
  const unsigned char* text;
  size_t text_size;
  int byte_pos;
  const libxs_token_t* token;
  const libxs_token_t* prev_token;
} libxs_textrule_ctx_t;

/** Context passed to lexical rule evaluation. */
LIBXS_EXTERN_C typedef struct libxs_lexrule_ctx_t {
  const unsigned char* text;
  int length;
  unsigned int flags;
  unsigned int hash;
} libxs_lexrule_ctx_t;

/** Data-only lexical normalization: map normalized `from` text to `to`. */
LIBXS_EXTERN_C typedef struct libxs_lexnorm_t {
  char from[LIBXS_LEXEME_MAXBYTES + 1];
  char to[LIBXS_LEXEME_MAXBYTES + 1];
} libxs_lexnorm_t;


/** Decode all token properties into an info struct. */
LIBXS_API void libxs_token_info(const libxs_token_t* token,
  libxs_token_info_t* info);

/**
 * Number of tokens forming the word that starts at pos, i.e. the run reaching
 * up to (excluding) the next token flagged LIBXS_TOKEN_BREAK. Returns zero
 * once pos is beyond the stream, which iterates words spanning several
 * sub-word tokens: for (p = 0; 0 != (n = word_next(t, nt, p)); p += n).
 * Granularities that do not mark boundaries yield a single run.
 */
LIBXS_API size_t libxs_token_word_next(const libxs_token_t* tokens,
  size_t ntokens, size_t pos);

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

/** Create a lexical vocabulary. */
LIBXS_API libxs_lexicon_t* libxs_lexicon_create(void);

/** Destroy a lexical vocabulary (NULL is accepted). */
LIBXS_API void libxs_lexicon_destroy(libxs_lexicon_t* lexicon);

/** Number of vocabulary entries, excluding reserved id 0. */
LIBXS_API unsigned int libxs_lexicon_size(const libxs_lexicon_t* lexicon);

/** Map normalized token text to an id; create non-zero interns missing text. */
LIBXS_API unsigned int libxs_lexicon_id(libxs_lexicon_t* lexicon,
  const char* text, int length, unsigned int flags, int create);

/** Save lexical vocabulary to a binary buffer. */
LIBXS_API int libxs_lexicon_save(const libxs_lexicon_t* lexicon,
  void* buffer, size_t* size);

/** Load a lexical vocabulary saved with libxs_lexicon_save. */
LIBXS_API libxs_lexicon_t* libxs_lexicon_load(const void* buffer, size_t size);

/** Query normalized token text and class flags by vocabulary id. */
LIBXS_API const char* libxs_lexicon_text(const libxs_lexicon_t* lexicon,
  unsigned int id, int* length, unsigned int* flags);

/** Initialize an empty lexical token stream. */
LIBXS_API void libxs_lexeme_stream_init(libxs_lexeme_stream_t* stream);

/** Ensure the stream can hold at least capacity lexical tokens. */
LIBXS_API int libxs_lexeme_stream_reserve(libxs_lexeme_stream_t* stream,
  size_t capacity);

/** Append a lexical token to the stream. */
LIBXS_API int libxs_lexeme_stream_push(libxs_lexeme_stream_t* stream,
  const libxs_lexeme_t* lexeme);

/** Release all memory held by a lexical token stream. */
LIBXS_API void libxs_lexeme_stream_release(libxs_lexeme_stream_t* stream);

/**
 * Encode text into stable lexical token ids. Words are lowercased and then can
 * be rewritten by caller-supplied normalization rules. Numbers are mapped to
 * <num>, punctuation is preserved, and class flags are assigned by built-in
 * detection followed by optional lexical rules.
 */
LIBXS_API int libxs_lexeme_stream_encode(libxs_lexicon_t* lexicon,
  libxs_lexeme_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules,
  const libxs_lexnorm_t* norms, int nnorms, int create);

/** Encode text into initialized stream of 8-byte token IDs. */
LIBXS_API int libxs_token_stream_encode(libxs_lexicon_t* lexicon,
  libxs_token_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules,
  const libxs_lexnorm_t* norms, int nnorms, int create);

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

/** Evaluate lexical classification rules and return the resulting flags. */
LIBXS_API unsigned int libxs_lexrule_eval(const libxs_lexrule_ctx_t* ctx,
  const libxs_lexrule_t* rules, int nrules);

/** Load lexical rules from registry (key prefix "LRULE:"). */
LIBXS_API int libxs_lexrule_load(const libxs_registry_t* registry,
  libxs_lexrule_t* rules, int max_rules);

/** Save lexical rules to registry (key prefix "LRULE:"). */
LIBXS_API int libxs_lexrule_save(libxs_registry_t* registry,
  const libxs_lexrule_t* rules, int nrules);

/** Populate rules[] with built-in lexical defaults. */
LIBXS_API int libxs_lexrule_defaults(libxs_lexrule_t* rules, int max_rules);

/**
 * Reflow text: replace cosmetic newlines (column-wrap artifacts) with spaces,
 * preserving structural newlines (enumerations, headings, blank lines, verse).
 * Allocates *out (caller must free). Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_text_reflow(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size);

/** Return non-zero if the token is a copy (back-reference). */
/** Return the number of bytes this token represents (1..14). */
LIBXS_API_INLINE size_t libxs_token_len(const libxs_token_t* token)
{
  return (NULL != token) ? (size_t)token->length : 0;
}

/** Return non-zero if a preferred break (word/punctuation boundary) follows. */
LIBXS_API_INLINE int libxs_token_has_break(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->flags & LIBXS_TOKEN_BREAK)) ? 1 : 0;
}

/** Return non-zero if this token is structural markup (not content). */
LIBXS_API_INLINE int libxs_token_is_markup(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->flags & LIBXS_TOKEN_MARKUP)) ? 1 : 0;
}

/** Return non-zero if this token ends a sentence (raw signal, pre-rules). */
LIBXS_API_INLINE int libxs_token_is_sentence_end(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->flags & LIBXS_TOKEN_SENTENCE)) ? 1 : 0;
}

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_TOKEN_H*/
