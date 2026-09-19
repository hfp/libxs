/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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
#define LIBXS_TOKEN_PAYLOAD_BYTES 7
#define LIBXS_TOKEN_LENGTH_MASK 0x07u
#define LIBXS_TOKEN_CONTINUED 0x08u
#define LIBXS_TOKEN_KIND_SHIFT 4
#define LIBXS_TOKEN_KIND_MASK 0x70u
#define LIBXS_TOKEN_SENTENCE 0x80u

/** Maximum normalized text bytes per token. */
#define LIBXS_LEXEME_BYTES 8
#define LIBXS_LEXEME_MAXBYTES 63

#define LIBXS_LEXEME_WORD     0x0001u
#define LIBXS_LEXEME_NUMBER   0x0002u
#define LIBXS_LEXEME_PUNCT    0x0004u
#define LIBXS_LEXEME_MARKUP   0x0008u
#define LIBXS_LEXEME_SENTENCE 0x0010u
#define LIBXS_LEXEME_QUESTION 0x0020u
#define LIBXS_LEXEME_STOP     0x0040u
#define LIBXS_LEXEME_ENTITY   0x0080u
#define LIBXS_LEXEME_BREAK    0x0100u

/**
 * Separator for a multi-token normalization target: `to` may expand one source
 * token into several tokens, e.g. "is not". Only the first emitted token
 * carries the source byte length (and the word-break flag); the continuations
 * carry length zero, so byte accounting over a stream is unchanged by
 * normalization and libxs_lexeme_word_next groups them as one word.
 */
#define LIBXS_LEXNORM_SEPARATOR ' '


/** Physical cell of a variable-length metatoken: control byte plus payload. */
LIBXS_EXTERN_C typedef struct libxs_token_t {
  unsigned char raw[LIBXS_TOKEN_BYTES];
} libxs_token_t;

/** Vocabulary-backed lexical occurrence used by interpretive consumers. */
LIBXS_EXTERN_C typedef struct libxs_lexeme_t {
  unsigned int id;
  unsigned short length;
  unsigned short flags;
} libxs_lexeme_t;

/** Growable array of tokens. */
LIBXS_EXTERN_C typedef struct libxs_token_stream_t {
  libxs_token_t* data;
  size_t size;
  size_t capacity;
} libxs_token_stream_t;

/** Growable array of lexical occurrences. */
LIBXS_EXTERN_C typedef struct libxs_lexeme_stream_t {
  libxs_lexeme_t* data;
  size_t size;
  size_t capacity;
} libxs_lexeme_stream_t;

enum libxs_token_kind_t {
  LIBXS_TOKEN_LITERAL = 0,
  LIBXS_TOKEN_TEXT = 1,
  LIBXS_TOKEN_NUMBER = 2,
  LIBXS_TOKEN_SPACE = 3,
  LIBXS_TOKEN_PUNCT = 4,
  LIBXS_TOKEN_MARKUP = 5,
  LIBXS_TOKEN_REFERENCE = 6,
  LIBXS_TOKEN_CONTROL = 7
};

enum libxs_token_granularity_t {
  LIBXS_TOKEN_GRANULARITY_NATIVE = 0,
  LIBXS_TOKEN_GRANULARITY_WORD = 1,
  LIBXS_TOKEN_GRANULARITY_SYLLABLE = 2
};

/** Decoded properties of one physical metatoken cell. */
LIBXS_EXTERN_C typedef struct libxs_token_info_t {
  size_t length;
  size_t cells;
  int kind;
  int continued;
  int is_sentence;
} libxs_token_info_t;

/** Decoded properties of one lexical occurrence. */
LIBXS_EXTERN_C typedef struct libxs_lexeme_info_t {
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
} libxs_lexeme_info_t;

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
  const libxs_lexeme_t* token;
  const libxs_lexeme_t* prev_token;
} libxs_textrule_ctx_t;

/** Context passed to lexical rule evaluation. */
LIBXS_EXTERN_C typedef struct libxs_lexrule_ctx_t {
  const unsigned char* text;
  int length;
  unsigned int flags;
  unsigned int hash;
} libxs_lexrule_ctx_t;

/** Opaque lexical vocabulary. Token id 0 is reserved for unknown. */
LIBXS_EXTERN_C typedef struct libxs_lexicon_t libxs_lexicon_t;
/** Opaque metatoken encoder configuration. */
LIBXS_EXTERN_C typedef struct libxs_tokenizer_t libxs_tokenizer_t;

/** Data-only lexical normalization: map normalized `from` text to `to`. */
LIBXS_EXTERN_C typedef struct libxs_lexnorm_t {
  char from[LIBXS_LEXEME_MAXBYTES + 1];
  char to[LIBXS_LEXEME_MAXBYTES + 1];
} libxs_lexnorm_t;


/** Decode the control byte of one physical metatoken cell. */
LIBXS_API void libxs_token_info(const libxs_token_t* token,
  libxs_token_info_t* info);

/**
 * Number of physical cells occupied by the logical metatoken at pos. Returns
 * zero for an invalid position or malformed continuation chain. payload_size
 * receives the logical payload size when non-NULL.
 */
LIBXS_API size_t libxs_token_span(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, size_t* payload_size);

/**
 * Read one logical metatoken beginning at pos. payload may be NULL to query
 * metadata only; otherwise capacity must cover info.length bytes. Physical
 * cell count, logical byte length, kind, and final sentence flag are returned
 * through info. Returns EXIT_SUCCESS only for a complete, valid chain.
 */
LIBXS_API int libxs_token_read(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, unsigned char* payload, size_t capacity,
  libxs_token_info_t* info);

/**
 * Compare the declared payload of two physical cells. Length and payload bytes
 * must match; kind, continuation, sentence, and unused trailing bytes are
 * ignored. This is the content-matching operation, not binary token equality.
 */
LIBXS_API int libxs_token_payload_equal(const libxs_token_t* lhs,
  const libxs_token_t* rhs);

/**
 * Compare two complete logical metatoken payloads without allocation. Control
 * metadata and physical cell boundaries are ignored. Invalid chains, unequal
 * logical lengths, or unequal payload bytes return zero.
 */
LIBXS_API int libxs_token_payload_match(const libxs_token_t* lhs,
  size_t nlhs, size_t lhs_pos, const libxs_token_t* rhs,
  size_t nrhs, size_t rhs_pos);

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

/** Create a metatoken encoder with one segmentation policy. */
LIBXS_API libxs_tokenizer_t* libxs_tokenizer_create(int granularity);

/** Destroy a metatoken encoder (NULL is accepted). */
LIBXS_API void libxs_tokenizer_destroy(libxs_tokenizer_t* tokenizer);

/** Change the segmentation policy used by subsequent encode calls. */
LIBXS_API int libxs_tokenizer_set_granularity(libxs_tokenizer_t* tokenizer,
  int granularity);

/** Return the configured segmentation policy, or -1 for NULL. */
LIBXS_API int libxs_tokenizer_granularity(const libxs_tokenizer_t* tokenizer);

/**
 * Encode source bytes using the tokenizer's configured segmentation policy.
 * Every policy preserves every source byte and decoding needs no tokenizer.
 */
LIBXS_API int libxs_token_stream_encode(const libxs_tokenizer_t* tokenizer,
  libxs_token_stream_t* stream, const unsigned char* text, size_t size);

/** Decode a metatoken stream to an allocated, zero-terminated byte sequence. */
LIBXS_API int libxs_token_stream_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size);

/** Decode all lexical occurrence properties into an info struct. */
LIBXS_API void libxs_lexeme_info(const libxs_lexeme_t* lexeme,
  libxs_lexeme_info_t* info);

/** Number of lexical pieces in the source word beginning at pos. */
LIBXS_API size_t libxs_lexeme_word_next(const libxs_lexeme_t* lexemes,
  size_t nlexemes, size_t pos);

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

/**
 * Load a lexical vocabulary saved with libxs_lexicon_save, or NULL. A
 * vocabulary written by a different version of the library is rejected rather
 * than misread, so NULL does not mean the buffer held nothing: a caller that
 * falls back to an empty vocabulary renumbers the ids that a corpus or a model
 * saved beside it still refers to.
 */
LIBXS_API libxs_lexicon_t* libxs_lexicon_load(const void* buffer, size_t size);

/** Query normalized token text and class flags by vocabulary id. */
LIBXS_API const char* libxs_lexicon_text(const libxs_lexicon_t* lexicon,
  unsigned int id, int* length, unsigned int* flags);

/**
 * How often the vocabulary interned the text of an id, i.e. its occurrence
 * count over everything encoded into it (0 if the id is unknown). Only the
 * interning path counts, so a lookup does not inflate a frequency.
 */
LIBXS_API unsigned int libxs_lexicon_count(const libxs_lexicon_t* lexicon,
  unsigned int id);

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
 * Whether the character at text[0] belongs to a word, and how many bytes it spans
 * (via length, which may be NULL). Handles multi-byte encodings: an encoded letter
 * such as an umlaut is a word character while encoded punctuation such as a
 * typographic apostrophe is not, so "don<U+2019>t" stays three lexemes while a
 * German word stays one. Callers that scan word spans themselves should use this
 * rather than ctype, which rejects every byte of an encoded letter.
 */
LIBXS_API int libxs_lexeme_is_word_char(const unsigned char* text, size_t size,
  int* length);

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
 *
 * As libxs_text_reflow, and additionally maps the result back to the LINES of the
 * input: *line_offsets is allocated with *nlines entries, where entry k is the
 * offset in *out at which input line k+1 begins. Reflow drops characters but never
 * reorders them, so the map is monotone and a position in the result belongs to the
 * last input line whose offset does not exceed it - which is what lets a caller
 * cite a line of the ORIGINAL text after working on the reflowed one. Both
 * out-parameters may be NULL, in which case no map is built.
 */
LIBXS_API int libxs_text_reflow_map(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size, size_t** line_offsets, size_t* nlines);

LIBXS_API int libxs_text_reflow(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size);

/** Return the payload bytes stored in this physical cell (1..7). */
LIBXS_API_INLINE size_t libxs_token_len(const libxs_token_t* token)
{
  return (NULL != token)
    ? (size_t)(token->raw[0] & LIBXS_TOKEN_LENGTH_MASK) : 0;
}

/** Return the metatoken kind carried by this physical cell. */
LIBXS_API_INLINE int libxs_token_kind(const libxs_token_t* token)
{
  return (NULL != token)
    ? (int)((token->raw[0] & LIBXS_TOKEN_KIND_MASK)
      >> LIBXS_TOKEN_KIND_SHIFT) : LIBXS_TOKEN_CONTROL;
}

/** Return non-zero when the next cell continues the same logical metatoken. */
LIBXS_API_INLINE int libxs_token_is_continued(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_CONTINUED))
    ? 1 : 0;
}

/** Return non-zero if this physical cell terminates a sentence. */
LIBXS_API_INLINE int libxs_token_is_sentence_end(const libxs_token_t* token)
{
  return (NULL != token && 0 != (token->raw[0] & LIBXS_TOKEN_SENTENCE))
    ? 1 : 0;
}

LIBXS_API_INLINE size_t libxs_lexeme_len(const libxs_lexeme_t* lexeme)
{
  return (NULL != lexeme) ? (size_t)lexeme->length : 0;
}

LIBXS_API_INLINE int libxs_lexeme_has_break(const libxs_lexeme_t* lexeme)
{
  return (NULL != lexeme && 0 != (lexeme->flags & LIBXS_LEXEME_BREAK))
    ? 1 : 0;
}

LIBXS_API_INLINE int libxs_lexeme_is_sentence_end(
  const libxs_lexeme_t* lexeme)
{
  return (NULL != lexeme && 0 != (lexeme->flags & LIBXS_LEXEME_SENTENCE))
    ? 1 : 0;
}

/* header-only: include implementation */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_TOKEN_H*/
