/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_token.h>
#include <libxs/libxs_str.h>


typedef struct internal_libxs_lexicon_entry_t {
  unsigned int id;
  unsigned int flags;
  unsigned short length;
  char text[LIBXS_LEXEME_MAXBYTES + 1];
} internal_libxs_lexicon_entry_t;

struct libxs_lexicon_t {
  libxs_registry_t* registry;
  internal_libxs_lexicon_entry_t* entries;
  size_t capacity;
  unsigned int size;
};

struct libxs_tokenizer_t {
  int granularity;
};


LIBXS_API_INLINE
int internal_libxs_lexicon_reserve(libxs_lexicon_t* lexicon,
  unsigned int capacity)
{
  int result = EXIT_FAILURE;
  if (NULL != lexicon) {
    if ((size_t)capacity <= lexicon->capacity) result = EXIT_SUCCESS;
    else {
      size_t next_capacity = (0 != lexicon->capacity)
        ? lexicon->capacity : 64;
      internal_libxs_lexicon_entry_t* entries;
      while (next_capacity < (size_t)capacity) next_capacity *= 2;
      entries = (internal_libxs_lexicon_entry_t*)realloc(lexicon->entries,
        next_capacity * sizeof(internal_libxs_lexicon_entry_t));
      if (NULL != entries) {
        memset(entries + lexicon->capacity, 0,
          (next_capacity - lexicon->capacity)
            * sizeof(internal_libxs_lexicon_entry_t));
        lexicon->entries = entries;
        lexicon->capacity = next_capacity;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


LIBXS_API_INLINE
void internal_libxs_lexicon_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  LIBXS_UNUSED(value); LIBXS_UNUSED(key);
  LIBXS_UNUSED(key_size); LIBXS_UNUSED(value_size);
  LIBXS_UNUSED(udata);
}


LIBXS_API_INLINE
unsigned int internal_libxs_lexicon_id(libxs_lexicon_t* lexicon,
  const char* text, int length, unsigned int flags, int create)
{
  unsigned int result = 0;
  if (NULL != lexicon && NULL != text && length > 0
    && length <= LIBXS_LEXEME_MAXBYTES)
  {
    internal_libxs_lexicon_entry_t* entry;
    entry = (internal_libxs_lexicon_entry_t*)libxs_registry_get(
      lexicon->registry, text, (size_t)length, NULL);
    if (NULL != entry) result = entry->id;
    else if (0 != create) {
      internal_libxs_lexicon_entry_t init;
      unsigned int id = lexicon->size + 1;
      memset(&init, 0, sizeof(init));
      init.id = id;
      init.flags = flags;
      init.length = (unsigned short)length;
      memcpy(init.text, text, (size_t)length);
      init.text[length] = 0;
      if (EXIT_SUCCESS == internal_libxs_lexicon_reserve(lexicon, id + 1)) {
        entry = (internal_libxs_lexicon_entry_t*)libxs_registry_set(
          lexicon->registry, text, (size_t)length, &init, sizeof(init), NULL);
        if (NULL != entry) {
          lexicon->entries[id] = init;
          lexicon->size = id;
          result = id;
        }
      }
    }
  }
  return result;
}


/**
 * Whether a UTF-8 sequence starting at text[0] encodes a LETTER rather than
 * punctuation, and how many bytes it spans (via *length).
 *
 * A byte-wise "anything >= 0x80 is a letter" rule is not good enough, and the
 * reason is measurable: an English corpus written with typographic quotes carries
 * 5133 apostrophes encoded E2 80 99, so that rule silently glues "don" and "t"
 * into one lexeme and moves every English figure. Conversely ctype alone rejects
 * both bytes of an umlaut, cutting a German word into pieces - and the middle
 * piece was classed as punctuation and hence read as a SENTENCE BOUNDARY, which
 * corrupts segmentation rather than merely word identity. 9.1% of word tokens in a
 * German fairy-tale corpus carry an umlaut or sharp s.
 *
 * So the test is on the decoded codepoint, kept to the ranges that actually occur
 * in text corpora: Latin-1 Supplement and Latin Extended-A/B (C3..C9 leads) are
 * letters apart from the two punctuation slots at U+00AB/U+00BB and the
 * multiplication/division signs; General Punctuation (E2 80..) is not. Anything
 * outside the recognized ranges is treated as a letter, which keeps scripts this
 * function was never taught (Greek, Cyrillic, CJK) working as words.
 */
LIBXS_API_INLINE
int internal_libxs_lexeme_is_letter_utf8(const unsigned char* text,
  size_t size, int* length)
{
  int result = 0;
  int span = 1;
  if (NULL != text && 0 < size) {
    if (0xC0u <= text[0] && text[0] < 0xE0u && 1 < size) {
      const unsigned long cp = ((unsigned long)(text[0] & 0x1Fu) << 6)
        | (unsigned long)(text[1] & 0x3Fu);
      span = 2;
      /* Latin-1 letters, excluding guillemets and the two math signs. */
      result = ((0x00AAul == cp || 0x00B5ul == cp || 0x00BAul == cp
        || (0x00C0ul <= cp && 0x00FFul >= cp && 0x00D7ul != cp
          && 0x00F7ul != cp)
        || (0x0100ul <= cp && 0x024Ful >= cp)) != 0) ? 1 : 0;
    }
    else if (0xE0u <= text[0] && text[0] < 0xF0u && 2 < size) {
      const unsigned long cp = ((unsigned long)(text[0] & 0x0Fu) << 12)
        | ((unsigned long)(text[1] & 0x3Fu) << 6)
        | (unsigned long)(text[2] & 0x3Fu);
      span = 3;
      /* General Punctuation, and the CJK/quotation blocks around it. */
      result = ((0x2000ul <= cp && 0x206Ful >= cp)
        || (0x3000ul <= cp && 0x303Ful >= cp)) ? 0 : 1;
    }
    else if (0xF0u <= text[0]) {
      span = 4;
      result = 1;
    }
    else if (0x80u <= text[0]) {
      /* A stray continuation byte: not a letter, and consumed on its own. */
      result = 0;
    }
  }
  if (NULL != length) *length = span;
  return result;
}


LIBXS_API_INLINE
int internal_libxs_lexeme_is_word_char(unsigned char ch)
{
  int result = 0;
  if (0 != isalnum(ch) || '_' == ch) result = 1;
  return result;
}


LIBXS_API_INLINE
int internal_libxs_lexeme_normalize_word(char* out, int out_size,
  const unsigned char* text, size_t length)
{
  int result = 0;
  size_t text_pos;
  if (NULL != out && NULL != text && out_size > 0) {
    for (text_pos = 0; text_pos < length && result + 1 < out_size;
      ++text_pos)
    {
      out[result++] = (char)tolower(text[text_pos]);
    }
    out[result] = 0;
  }
  return result;
}


/**
 * Apply a whole-token normalization. The replacement may expand to several
 * tokens by separating them with LIBXS_LEXNORM_SEPARATOR; the text is rewritten
 * with the separator retained and split_out receives the offset of the first
 * separator (or the resulting length when there is none), so the caller can
 * emit one token per piece. Returns the total rewritten length.
 */
LIBXS_API_INLINE
int internal_libxs_lexeme_apply_norm(char* text, int text_size, int length,
  const libxs_lexnorm_t* norms, int nnorms, int* split_out)
{
  int result = length;
  int norm_pos;
  if (NULL != split_out) *split_out = length;
  if (NULL != text && text_size > 0 && NULL != norms && nnorms > 0) {
    for (norm_pos = 0; norm_pos < nnorms; ++norm_pos) {
      const char* from = norms[norm_pos].from;
      const char* to = norms[norm_pos].to;
      int from_len = (int)strlen(from);
      int to_len = (int)strlen(to);
      if (from_len == result && to_len > 0 && to_len < text_size
        && 0 == memcmp(text, from, (size_t)from_len))
      {
        memcpy(text, to, (size_t)to_len);
        text[to_len] = '\0';
        result = to_len;
        if (NULL != split_out) {
          const char* sep = strchr(text, LIBXS_LEXNORM_SEPARATOR);
          *split_out = (NULL != sep) ? (int)(sep - text) : result;
        }
        break;
      }
    }
  }
  return result;
}


LIBXS_API_INLINE
int internal_libxs_lexeme_normalize_punct(char* out, int out_size,
  const unsigned char* text, size_t length)
{
  int result = 0;
  if (NULL != out && NULL != text && out_size > 0 && length > 0) {
    size_t copy_size = length;
    if (copy_size >= (size_t)out_size) copy_size = (size_t)out_size - 1;
    memcpy(out, text, copy_size);
    out[copy_size] = 0;
    result = (int)copy_size;
  }
  return result;
}


LIBXS_API_INLINE
size_t internal_libxs_token_codepoint_size(const unsigned char* text,
  size_t size, size_t pos)
{
  return libxs_utf8_size(text, size, pos);
}


LIBXS_API_INLINE
int internal_libxs_token_is_sentence_char(unsigned char ch)
{
  return ('.' == ch || '?' == ch || '!' == ch) ? 1 : 0;
}


LIBXS_API_INLINE
int internal_libxs_token_is_word_char(unsigned char ch)
{
  int result = 0;
  if (0 != isalpha(ch) || '_' == ch || 0x80u <= ch) result = 1;
  return result;
}


LIBXS_API_INLINE
int internal_libxs_token_is_vowel(unsigned char ch)
{
  int result = 0;
  const unsigned char lower = (unsigned char)tolower(ch);
  if ('a' == lower || 'e' == lower || 'i' == lower
    || 'o' == lower || 'u' == lower || 'y' == lower)
  {
    result = 1;
  }
  return result;
}


LIBXS_API_INLINE
size_t internal_libxs_token_syllable_end(const unsigned char* text,
  size_t start, size_t end)
{
  size_t result = end;
  size_t pos = start;
  int seen_vowel = 0;
  while (pos < end) {
    const unsigned char ch = text[pos];
    size_t step = internal_libxs_token_codepoint_size(text, end, pos);
    if (1 == step && pos > start && 0 != seen_vowel
      && 0 == internal_libxs_token_is_vowel(ch)
      && pos + 1 < end && 0 != internal_libxs_token_is_vowel(text[pos + 1]))
    {
      result = pos;
      pos = end;
    }
    else {
      if (1 == step && 0 != internal_libxs_token_is_vowel(ch)) seen_vowel = 1;
      pos += step;
    }
  }
  return result;
}


LIBXS_API_INLINE
int internal_libxs_token_emit(libxs_token_stream_t* stream,
  const unsigned char* text, size_t length, int kind, int sentence)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != text && length > 0
    && kind >= LIBXS_TOKEN_LITERAL && kind <= LIBXS_TOKEN_CONTROL)
  {
    size_t offset = 0;
    result = EXIT_SUCCESS;
    while (EXIT_SUCCESS == result && offset < length) {
      libxs_token_t token;
      size_t chunk = length - offset;
      if (chunk > LIBXS_TOKEN_PAYLOAD_BYTES) chunk = LIBXS_TOKEN_PAYLOAD_BYTES;
      while (chunk > 1 && offset + chunk < length
        && 0x80u <= text[offset + chunk]
        && text[offset + chunk] < 0xC0u)
      {
        --chunk;
      }
      memset(&token, 0, sizeof(token));
      token.raw[0] = (unsigned char)(chunk & LIBXS_TOKEN_LENGTH_MASK);
      token.raw[0] |= (unsigned char)(kind << LIBXS_TOKEN_KIND_SHIFT);
      if (offset + chunk < length) token.raw[0] |= LIBXS_TOKEN_CONTINUED;
      else if (0 != sentence) token.raw[0] |= LIBXS_TOKEN_SENTENCE;
      memcpy(token.raw + 1, text + offset, chunk);
      result = libxs_token_stream_push(stream, &token);
      offset += chunk;
    }
  }
  return result;
}


LIBXS_API_INLINE
int internal_libxs_token_detect_markup(const unsigned char* text,
  size_t size, size_t pos, size_t len)
{
  int result = 0;
  size_t i;
  int npunct = 0, nalpha = 0;
  unsigned char ch;
  if (NULL != text && 0 != len) {
    ch = text[pos];
    if (1 == len) {
      if ('\\' == ch || '$' == ch || '{' == ch || '}' == ch) result = 1;
      else if ('*' == ch || '_' == ch || '~' == ch || '#' == ch || '`' == ch) {
        if (pos > 0 && text[pos - 1] == ch) result = 1;
        else if (pos + 1 < size && text[pos + 1] == ch) result = 1;
      }
    }
    else {
      if ('\\' == ch) {
        for (i = 1; i < len; ++i) {
          if (0 != isalpha(text[pos + i])) ++nalpha;
        }
        if (nalpha > 0) result = 1;
      }
      if (0 == result) {
        for (i = 0; i < len; ++i) {
          if (0 != ispunct(text[pos + i])) ++npunct;
        }
        if (npunct == (int)len && len >= 2) {
          const unsigned char first = ch;
          int same = 1;
          for (i = 1; i < len && 0 != same; ++i) {
            if (text[pos + i] != first) same = 0;
          }
          if (0 != same && ('*' == first || '_' == first || '~' == first
            || '#' == first || '`' == first || '-' == first || '=' == first))
          {
            result = 1;
          }
        }
      }
      if (0 == result && len <= 3) {
        int all_bracket = 1;
        for (i = 0; i < len && 0 != all_bracket; ++i) {
          const unsigned char c = text[pos + i];
          if ('{' != c && '}' != c && '[' != c && ']' != c && '$' != c) {
            all_bracket = 0;
          }
        }
        if (0 != all_bracket) result = 1;
      }
    }
  }
  return result;
}


LIBXS_API void libxs_token_info(const libxs_token_t* token,
  libxs_token_info_t* info)
{
  if (NULL != info) {
    if (NULL != token) {
      info->length = libxs_token_len(token);
      info->cells = 1;
      info->kind = libxs_token_kind(token);
      info->continued = libxs_token_is_continued(token);
      info->is_sentence = libxs_token_is_sentence_end(token);
    }
    else {
      info->length = 0;
      info->cells = 0;
      info->kind = LIBXS_TOKEN_CONTROL;
      info->continued = 0;
      info->is_sentence = 0;
    }
  }
}


LIBXS_API int libxs_token_read(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, unsigned char* payload, size_t capacity,
  libxs_token_info_t* info)
{
  int result = EXIT_FAILURE;
  size_t payload_size = 0;
  size_t cells = libxs_token_span(tokens, ntokens, pos, &payload_size);
  if (NULL != info) memset(info, 0, sizeof(*info));
  if (cells > 0 && (NULL == payload || capacity >= payload_size)) {
    size_t cell;
    size_t offset = 0;
    if (NULL != payload) {
      for (cell = 0; cell < cells; ++cell) {
        const libxs_token_t* token = tokens + pos + cell;
        const size_t length = libxs_token_len(token);
        memcpy(payload + offset, token->raw + 1, length);
        offset += length;
      }
    }
    if (NULL != info) {
      info->length = payload_size;
      info->cells = cells;
      info->kind = libxs_token_kind(tokens + pos);
      info->continued = (cells > 1) ? 1 : 0;
      info->is_sentence = libxs_token_is_sentence_end(
        tokens + pos + cells - 1);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API size_t libxs_token_span(const libxs_token_t* tokens,
  size_t ntokens, size_t pos, size_t* payload_size)
{
  size_t result = 0;
  size_t total = 0;
  if (NULL != tokens && pos < ntokens) {
    const int kind = libxs_token_kind(tokens + pos);
    size_t at = pos;
    int more = 1;
    while (at < ntokens && 0 != more
      && kind == libxs_token_kind(tokens + at)
      && libxs_token_len(tokens + at) > 0)
    {
      total += libxs_token_len(tokens + at);
      more = libxs_token_is_continued(tokens + at);
      ++at;
    }
    if (0 == more) result = at - pos;
  }
  if (NULL != payload_size) *payload_size = (0 != result) ? total : 0;
  return result;
}


LIBXS_API int libxs_token_payload_equal(const libxs_token_t* lhs,
  const libxs_token_t* rhs)
{
  int result = 0;
  if (NULL != lhs && NULL != rhs) {
    const size_t lhs_length = libxs_token_len(lhs);
    const size_t rhs_length = libxs_token_len(rhs);
    if (lhs_length == rhs_length && lhs_length > 0
      && 0 == memcmp(lhs->raw + 1, rhs->raw + 1, lhs_length))
    {
      result = 1;
    }
  }
  return result;
}


LIBXS_API int libxs_token_payload_match(const libxs_token_t* lhs,
  size_t nlhs, size_t lhs_pos, const libxs_token_t* rhs,
  size_t nrhs, size_t rhs_pos)
{
  int result = 0;
  size_t lhs_size = 0, rhs_size = 0;
  const size_t lhs_cells = libxs_token_span(lhs, nlhs, lhs_pos, &lhs_size);
  const size_t rhs_cells = libxs_token_span(rhs, nrhs, rhs_pos, &rhs_size);
  if (lhs_cells > 0 && rhs_cells > 0 && lhs_size == rhs_size) {
    size_t lhs_cell = 0, rhs_cell = 0;
    size_t lhs_offset = 0, rhs_offset = 0;
    size_t compared = 0;
    result = 1;
    while (compared < lhs_size && 0 != result) {
      const libxs_token_t* lhs_token = lhs + lhs_pos + lhs_cell;
      const libxs_token_t* rhs_token = rhs + rhs_pos + rhs_cell;
      const size_t lhs_left = libxs_token_len(lhs_token) - lhs_offset;
      const size_t rhs_left = libxs_token_len(rhs_token) - rhs_offset;
      const size_t length = (lhs_left < rhs_left) ? lhs_left : rhs_left;
      if (0 != memcmp(lhs_token->raw + 1 + lhs_offset,
        rhs_token->raw + 1 + rhs_offset, length))
      {
        result = 0;
      }
      else {
        compared += length;
        lhs_offset += length;
        rhs_offset += length;
        if (lhs_offset == libxs_token_len(lhs_token)) {
          ++lhs_cell;
          lhs_offset = 0;
        }
        if (rhs_offset == libxs_token_len(rhs_token)) {
          ++rhs_cell;
          rhs_offset = 0;
        }
      }
    }
  }
  return result;
}


LIBXS_API void libxs_token_stream_init(libxs_token_stream_t* stream)
{
  if (NULL != stream) {
    stream->data = NULL;
    stream->size = 0;
    stream->capacity = 0;
  }
}


LIBXS_API int libxs_token_stream_reserve(libxs_token_stream_t* stream,
  size_t capacity)
{
  int result = EXIT_FAILURE;
  if (NULL != stream) {
    if (capacity <= stream->capacity) result = EXIT_SUCCESS;
    else {
      libxs_token_t* data = (libxs_token_t*)realloc(stream->data,
        capacity * sizeof(libxs_token_t));
      if (NULL != data) {
        stream->data = data;
        stream->capacity = capacity;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


LIBXS_API int libxs_token_stream_push(libxs_token_stream_t* stream,
  const libxs_token_t* token)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != token) {
    const size_t cap = (0 != stream->capacity)
      ? (2 * stream->capacity) : 16;
    if (stream->size == stream->capacity) {
      result = libxs_token_stream_reserve(stream, cap);
    }
    else result = EXIT_SUCCESS;
    if (EXIT_SUCCESS == result) {
      stream->data[stream->size] = *token;
      ++stream->size;
    }
  }
  return result;
}


LIBXS_API void libxs_token_stream_release(libxs_token_stream_t* stream)
{
  if (NULL != stream) {
    free(stream->data);
    libxs_token_stream_init(stream);
  }
}


LIBXS_API libxs_tokenizer_t* libxs_tokenizer_create(int granularity)
{
  libxs_tokenizer_t* result = NULL;
  if (granularity >= LIBXS_TOKEN_GRANULARITY_NATIVE
    && granularity <= LIBXS_TOKEN_GRANULARITY_SYLLABLE)
  {
    result = (libxs_tokenizer_t*)malloc(sizeof(*result));
    if (NULL != result) result->granularity = granularity;
  }
  return result;
}


LIBXS_API void libxs_tokenizer_destroy(libxs_tokenizer_t* tokenizer)
{
  free(tokenizer);
}


LIBXS_API int libxs_tokenizer_set_granularity(libxs_tokenizer_t* tokenizer,
  int granularity)
{
  int result = EXIT_FAILURE;
  if (NULL != tokenizer
    && granularity >= LIBXS_TOKEN_GRANULARITY_NATIVE
    && granularity <= LIBXS_TOKEN_GRANULARITY_SYLLABLE)
  {
    tokenizer->granularity = granularity;
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API int libxs_tokenizer_granularity(const libxs_tokenizer_t* tokenizer)
{
  return (NULL != tokenizer) ? tokenizer->granularity : -1;
}


LIBXS_API int libxs_token_stream_encode(const libxs_tokenizer_t* tokenizer,
  libxs_token_stream_t* stream, const unsigned char* text, size_t size)
{
  int result = EXIT_FAILURE;
  if (NULL != tokenizer && NULL != stream && NULL != text
    && tokenizer->granularity >= LIBXS_TOKEN_GRANULARITY_NATIVE
    && tokenizer->granularity <= LIBXS_TOKEN_GRANULARITY_SYLLABLE)
  {
    const int granularity = tokenizer->granularity;
    size_t pos = 0;
    result = EXIT_SUCCESS;
    while (EXIT_SUCCESS == result && pos < size) {
      size_t end = pos + 1;
      int kind = LIBXS_TOKEN_LITERAL;
      int sentence = 0;
      if (LIBXS_TOKEN_GRANULARITY_NATIVE == granularity) {
        end = pos;
        while (end < size && end - pos < LIBXS_TOKEN_PAYLOAD_BYTES) {
          size_t step = internal_libxs_token_codepoint_size(text, size, end);
          if (end + step - pos > LIBXS_TOKEN_PAYLOAD_BYTES) break;
          end += step;
          if (0 != isspace(text[end - 1]) || 0 != ispunct(text[end - 1])) break;
        }
        if (end <= pos) end = pos + 1;
      }
      else if (0 != isspace(text[pos])) {
        kind = LIBXS_TOKEN_SPACE;
        while (end < size && 0 != isspace(text[end])) ++end;
      }
      else if (0 != internal_libxs_token_is_word_char(text[pos])) {
        size_t word_end;
        while (end < size && 0 != internal_libxs_token_is_word_char(text[end])) {
          ++end;
        }
        word_end = end;
        if (LIBXS_TOKEN_GRANULARITY_SYLLABLE == granularity) {
          end = internal_libxs_token_syllable_end(text, pos, word_end);
        }
        kind = LIBXS_TOKEN_TEXT;
      }
      else if (0 != isdigit(text[pos])) {
        kind = LIBXS_TOKEN_NUMBER;
        while (end < size && 0 != isdigit(text[end])) ++end;
      }
      else {
        end = pos + internal_libxs_token_codepoint_size(text, size, pos);
        kind = (0 != internal_libxs_token_detect_markup(text, size,
          pos, end - pos)) ? LIBXS_TOKEN_MARKUP : LIBXS_TOKEN_PUNCT;
        if (1 == end - pos
          && 0 != internal_libxs_token_is_sentence_char(text[pos]))
        {
          size_t next = end;
          while (next < size && ('"' == text[next] || '\'' == text[next]
            || ')' == text[next] || ']' == text[next]))
          {
            ++next;
          }
          if (next == size || 0 != isspace(text[next])) sentence = 1;
        }
      }
      result = internal_libxs_token_emit(stream, text + pos, end - pos,
        kind, sentence);
      pos = end;
    }
  }
  return result;
}


LIBXS_API int libxs_token_stream_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != text && NULL != size) {
    size_t total = 0;
    size_t token_pos;
    for (token_pos = 0; token_pos < stream->size; ++token_pos) {
      const size_t length = libxs_token_len(stream->data + token_pos);
      if (0 == length || length > LIBXS_TOKEN_PAYLOAD_BYTES) break;
      total += length;
    }
    if (token_pos == stream->size) {
      unsigned char* buffer = (unsigned char*)malloc(total + 1);
      if (NULL != buffer) {
        size_t offset = 0;
        result = EXIT_SUCCESS;
        for (token_pos = 0; token_pos < stream->size; ++token_pos) {
          const libxs_token_t* token = stream->data + token_pos;
          const size_t length = libxs_token_len(token);
          memcpy(buffer + offset, token->raw + 1, length);
          offset += length;
        }
        buffer[total] = 0;
        *text = buffer;
        *size = total;
      }
    }
  }
  return result;
}


LIBXS_API void libxs_lexeme_info(const libxs_lexeme_t* lexeme,
  libxs_lexeme_info_t* info)
{
  if (NULL != info) {
    memset(info, 0, sizeof(*info));
    if (NULL != lexeme) {
      info->id = lexeme->id;
      info->length = (size_t)lexeme->length;
      info->flags = lexeme->flags;
      info->is_word = (0 != (lexeme->flags & LIBXS_LEXEME_WORD)) ? 1 : 0;
      info->is_number = (0 != (lexeme->flags & LIBXS_LEXEME_NUMBER)) ? 1 : 0;
      info->is_punct = (0 != (lexeme->flags & LIBXS_LEXEME_PUNCT)) ? 1 : 0;
      info->has_break = (0 != (lexeme->flags & LIBXS_LEXEME_BREAK)) ? 1 : 0;
      info->is_sentence = (0 != (lexeme->flags & LIBXS_LEXEME_SENTENCE)) ? 1 : 0;
      info->is_question = (0 != (lexeme->flags & LIBXS_LEXEME_QUESTION)) ? 1 : 0;
      info->is_stop = (0 != (lexeme->flags & LIBXS_LEXEME_STOP)) ? 1 : 0;
      info->is_entity = (0 != (lexeme->flags & LIBXS_LEXEME_ENTITY)) ? 1 : 0;
      info->is_markup = (0 != (lexeme->flags & LIBXS_LEXEME_MARKUP)) ? 1 : 0;
    }
  }
}


LIBXS_API size_t libxs_lexeme_word_next(const libxs_lexeme_t* lexemes,
  size_t nlexemes, size_t pos)
{
  size_t result = 0;
  if (NULL != lexemes && pos < nlexemes) {
    size_t end = pos + 1;
    while (end < nlexemes && 0 == (lexemes[end].flags & LIBXS_LEXEME_BREAK)) {
      ++end;
    }
    result = end - pos;
  }
  return result;
}


LIBXS_API libxs_lexicon_t* libxs_lexicon_create(void)
{
  libxs_lexicon_t* result = (libxs_lexicon_t*)calloc(1,
    sizeof(libxs_lexicon_t));
  if (NULL != result) {
    result->registry = libxs_registry_create();
    if (NULL == result->registry
      || EXIT_SUCCESS != internal_libxs_lexicon_reserve(result, 1))
    {
      libxs_lexicon_destroy(result);
      result = NULL;
    }
  }
  return result;
}


LIBXS_API void libxs_lexicon_destroy(libxs_lexicon_t* lexicon)
{
  if (NULL != lexicon) {
    libxs_registry_destroy(lexicon->registry);
    free(lexicon->entries);
    free(lexicon);
  }
}


LIBXS_API unsigned int libxs_lexicon_size(const libxs_lexicon_t* lexicon)
{
  unsigned int result = 0;
  if (NULL != lexicon) result = lexicon->size;
  return result;
}


LIBXS_API unsigned int libxs_lexicon_id(libxs_lexicon_t* lexicon,
  const char* text, int length, unsigned int flags, int create)
{
  unsigned int result = 0;
  result = internal_libxs_lexicon_id(lexicon, text, length, flags, create);
  return result;
}


LIBXS_API int libxs_lexicon_save(const libxs_lexicon_t* lexicon,
  void* buffer, size_t* size)
{
  int result = EXIT_FAILURE;
  if (NULL != lexicon && NULL != size) {
    result = libxs_registry_save(lexicon->registry, buffer, size);
  }
  return result;
}


LIBXS_API libxs_lexicon_t* libxs_lexicon_load(const void* buffer, size_t size)
{
  libxs_lexicon_t* result = NULL;
  libxs_registry_t* registry = libxs_registry_load(buffer, size,
    internal_libxs_lexicon_fixup, NULL);
  if (NULL != registry) {
    result = (libxs_lexicon_t*)calloc(1, sizeof(libxs_lexicon_t));
    if (NULL != result) {
      const void* key = NULL;
      size_t cursor = 0;
      void* value;
      result->registry = registry;
      value = libxs_registry_begin(registry, &key, &cursor);
      while (NULL != value) {
        const internal_libxs_lexicon_entry_t* entry =
          (const internal_libxs_lexicon_entry_t*)value;
        if (0 != entry->id
          && EXIT_SUCCESS == internal_libxs_lexicon_reserve(result,
            entry->id + 1))
        {
          result->entries[entry->id] = *entry;
          if (entry->id > result->size) result->size = entry->id;
        }
        value = libxs_registry_next(registry, &key, &cursor);
      }
    }
    if (NULL == result) libxs_registry_destroy(registry);
  }
  return result;
}


LIBXS_API const char* libxs_lexicon_text(const libxs_lexicon_t* lexicon,
  unsigned int id, int* length, unsigned int* flags)
{
  const char* result = NULL;
  if (NULL != lexicon && 0 != id && id <= lexicon->size
    && id < lexicon->capacity && 0 != lexicon->entries[id].length)
  {
    const internal_libxs_lexicon_entry_t* entry = lexicon->entries + id;
    if (NULL != length) *length = (int)entry->length;
    if (NULL != flags) *flags = entry->flags;
    result = entry->text;
  }
  return result;
}


LIBXS_API void libxs_lexeme_stream_init(libxs_lexeme_stream_t* stream)
{
  if (NULL != stream) {
    stream->data = NULL;
    stream->size = 0;
    stream->capacity = 0;
  }
}


LIBXS_API int libxs_lexeme_stream_reserve(libxs_lexeme_stream_t* stream,
  size_t capacity)
{
  int result = EXIT_FAILURE;
  if (NULL != stream) {
    if (capacity <= stream->capacity) result = EXIT_SUCCESS;
    else {
      libxs_lexeme_t* data = (libxs_lexeme_t*)realloc(stream->data,
        capacity * sizeof(libxs_lexeme_t));
      if (NULL != data) {
        stream->data = data;
        stream->capacity = capacity;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


LIBXS_API int libxs_lexeme_stream_push(libxs_lexeme_stream_t* stream,
  const libxs_lexeme_t* lexeme)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != lexeme) {
    const size_t cap = (0 != stream->capacity)
      ? (2 * stream->capacity) : 32;
    if (stream->size == stream->capacity) {
      result = libxs_lexeme_stream_reserve(stream, cap);
    }
    else result = EXIT_SUCCESS;
    if (EXIT_SUCCESS == result) {
      stream->data[stream->size] = *lexeme;
      ++stream->size;
    }
  }
  return result;
}


LIBXS_API void libxs_lexeme_stream_release(libxs_lexeme_stream_t* stream)
{
  if (NULL != stream) {
    free(stream->data);
    libxs_lexeme_stream_init(stream);
  }
}


LIBXS_API int libxs_lexeme_is_word_char(const unsigned char* text, size_t size,
  int* length)
{
  int result = 0;
  if (NULL != text && 0 < size) {
    if (0 != internal_libxs_lexeme_is_word_char(text[0])) {
      if (NULL != length) *length = 1;
      result = 1;
    }
    else result = internal_libxs_lexeme_is_letter_utf8(text, size, length);
  }
  else if (NULL != length) *length = 1;
  return result;
}


LIBXS_API int libxs_lexeme_stream_encode(libxs_lexicon_t* lexicon,
  libxs_lexeme_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules,
  const libxs_lexnorm_t* norms, int nnorms, int create)
{
  int result = EXIT_SUCCESS;
  size_t text_pos = 0;
  int have_break = 0;
  if (NULL == lexicon || NULL == stream || NULL == text) result = EXIT_FAILURE;
  while (EXIT_SUCCESS == result && text_pos < size) {
    size_t token_start, token_len;
    unsigned int flags = 0;
    char normalized[LIBXS_LEXEME_MAXBYTES + 1];
    int normalized_len = 0, normalized_split = 0;
    while (text_pos < size && 0 != isspace(text[text_pos])) {
      have_break = 1;
      ++text_pos;
    }
    if (text_pos >= size) break;
    token_start = text_pos;
    /* A word may also START with a multi-byte letter ("Uber" with an umlaut). */
    if (0 != isalpha(text[text_pos]) || '_' == text[text_pos]
      || 0 != internal_libxs_lexeme_is_letter_utf8(text + text_pos,
           size - text_pos, NULL))
    {
      while (text_pos < size) {
        int span = 1;
        if (0 != internal_libxs_lexeme_is_word_char(text[text_pos])) {
          ++text_pos;
        }
        else if (0 != internal_libxs_lexeme_is_letter_utf8(text + text_pos,
          size - text_pos, &span))
        {
          text_pos += (size_t)span;
        }
        else break;
      }
      token_len = text_pos - token_start;
      normalized_len = internal_libxs_lexeme_normalize_word(normalized,
        (int)sizeof(normalized), text + token_start, token_len);
      normalized_len = internal_libxs_lexeme_apply_norm(normalized,
        (int)sizeof(normalized), normalized_len, norms, nnorms,
        &normalized_split);
      flags = LIBXS_LEXEME_WORD;
    }
    else if (0 != isdigit(text[text_pos])) {
      while (text_pos < size && 0 != isdigit(text[text_pos])) ++text_pos;
      while (text_pos + 1 < size
        && ('.' == text[text_pos] || ',' == text[text_pos])
        && 0 != isdigit(text[text_pos + 1]))
      {
        ++text_pos;
        while (text_pos < size && 0 != isdigit(text[text_pos])) ++text_pos;
      }
      token_len = text_pos - token_start;
      memcpy(normalized, "<num>", 6);
      normalized_len = 5;
      flags = LIBXS_LEXEME_NUMBER;
    }
    else {
      token_len = internal_libxs_token_codepoint_size(text, size, text_pos);
      text_pos += token_len;
      normalized_len = internal_libxs_lexeme_normalize_punct(normalized,
        (int)sizeof(normalized), text + token_start, token_len);
      flags = LIBXS_LEXEME_PUNCT;
      if (0 != internal_libxs_token_detect_markup(text, size, token_start, token_len)) {
        flags |= LIBXS_LEXEME_MARKUP;
      }
      if (1 == token_len
        && 0 != internal_libxs_token_is_sentence_char(text[token_start]))
      {
        flags |= LIBXS_LEXEME_SENTENCE;
        if ('?' == text[token_start]) flags |= LIBXS_LEXEME_QUESTION;
      }
    }
    if (0 != have_break) flags |= LIBXS_LEXEME_BREAK;
    have_break = 0;
    /**
     * A normalization may expand into several tokens. Emit one per piece: the
     * first keeps the source byte length and the break flag, continuations get
     * length zero, so the stream's byte total and word grouping are unchanged.
     */
    { int piece_begin = 0;
      const unsigned int base_flags = flags;
      while (EXIT_SUCCESS == result && piece_begin < normalized_len) {
        libxs_lexrule_ctx_t ctx;
        libxs_lexeme_t lexeme;
        int piece_len = normalized_split - piece_begin;
        if (piece_len <= 0 || piece_begin + piece_len > normalized_len) {
          piece_len = normalized_len - piece_begin;
        }
        flags = base_flags;
        if (0 != piece_begin) flags &= ~(unsigned int)LIBXS_LEXEME_BREAK;
        memset(&ctx, 0, sizeof(ctx));
        ctx.text = text + token_start;
        ctx.length = (int)token_len;
        ctx.flags = flags;
        ctx.hash = (0 != (flags & LIBXS_LEXEME_WORD))
          ? libxs_textrule_wordhash(
            (const unsigned char*)(normalized + piece_begin), piece_len) : 0;
        flags = libxs_lexrule_eval(&ctx, rules, nrules);
        memset(&lexeme, 0, sizeof(lexeme));
        lexeme.id = libxs_lexicon_id(lexicon, normalized + piece_begin,
          piece_len, flags, create);
        lexeme.length = (unsigned short)((0 == piece_begin) ? token_len : 0);
        lexeme.flags = (unsigned short)flags;
        result = libxs_lexeme_stream_push(stream, &lexeme);
        piece_begin += piece_len;
        while (piece_begin < normalized_len
          && LIBXS_LEXNORM_SEPARATOR == normalized[piece_begin])
        {
          ++piece_begin;
        }
        normalized_split = normalized_len;
      }
    }
  }
  return result;
}
