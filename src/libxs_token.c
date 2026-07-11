/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_token.h>


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
    if (7 == result && 0 == memcmp(out, "brought", 7)) {
      memcpy(out, "bring", 6);
      result = 5;
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
  size_t result = 1;
  if (NULL != text && pos < size) {
    const unsigned char lead = text[pos];
    if (0xC0u <= lead && lead < 0xE0u) result = 2;
    else if (0xE0u <= lead && lead < 0xF0u) result = 3;
    else if (0xF0u <= lead && lead < 0xF8u) result = 4;
    if (size - pos < result) result = size - pos;
  }
  return result;
}


LIBXS_API_INLINE
int internal_libxs_token_is_sentence_char(unsigned char ch)
{
  return ('.' == ch || '?' == ch || '!' == ch) ? 1 : 0;
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
      info->id = token->id;
      info->length = (size_t)token->length;
      info->flags = token->flags;
      info->is_word = (0 != (token->flags & LIBXS_TOKEN_WORD)) ? 1 : 0;
      info->is_number = (0 != (token->flags & LIBXS_TOKEN_NUMBER)) ? 1 : 0;
      info->is_punct = (0 != (token->flags & LIBXS_TOKEN_PUNCT)) ? 1 : 0;
      info->has_break = (0 != (token->flags & LIBXS_TOKEN_BREAK)) ? 1 : 0;
      info->is_sentence = (0 != (token->flags & LIBXS_TOKEN_SENTENCE)) ? 1 : 0;
      info->is_question = (0 != (token->flags & LIBXS_TOKEN_QUESTION)) ? 1 : 0;
      info->is_stop = (0 != (token->flags & LIBXS_TOKEN_STOP)) ? 1 : 0;
      info->is_entity = (0 != (token->flags & LIBXS_TOKEN_ENTITY)) ? 1 : 0;
      info->is_markup = (0 != (token->flags & LIBXS_TOKEN_MARKUP)) ? 1 : 0;
    }
    else {
      info->id = 0;
      info->length = 0;
      info->flags = 0;
      info->is_word = 0;
      info->is_number = 0;
      info->is_punct = 0;
      info->has_break = 0;
      info->is_sentence = 0;
      info->is_question = 0;
      info->is_stop = 0;
      info->is_entity = 0;
      info->is_markup = 0;
    }
  }
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


LIBXS_API int libxs_token_stream_encode(libxs_lexicon_t* lexicon,
  libxs_token_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules, int create)
{
  int result = EXIT_SUCCESS;
  size_t text_pos = 0;
  int have_break = 0;
  if (NULL == lexicon || NULL == stream || NULL == text) result = EXIT_FAILURE;
  while (EXIT_SUCCESS == result && text_pos < size) {
    size_t token_start, token_len;
    unsigned int flags = 0;
    char normalized[LIBXS_LEXEME_MAXBYTES + 1];
    int normalized_len = 0;
    while (text_pos < size && 0 != isspace(text[text_pos])) {
      have_break = 1;
      ++text_pos;
    }
    if (text_pos >= size) break;
    token_start = text_pos;
    if (0 != isalpha(text[text_pos]) || '_' == text[text_pos]) {
      while (text_pos < size
        && 0 != internal_libxs_lexeme_is_word_char(text[text_pos]))
      {
        ++text_pos;
      }
      token_len = text_pos - token_start;
      normalized_len = internal_libxs_lexeme_normalize_word(normalized,
        (int)sizeof(normalized), text + token_start, token_len);
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
      if (0 != internal_libxs_token_detect_markup(text, size,
        token_start, token_len)) flags |= LIBXS_LEXEME_MARKUP;
      if (1 == token_len
        && 0 != internal_libxs_token_is_sentence_char(text[token_start]))
      {
        flags |= LIBXS_LEXEME_SENTENCE;
        if ('?' == text[token_start]) flags |= LIBXS_LEXEME_QUESTION;
      }
    }
    if (0 != have_break) flags |= LIBXS_LEXEME_BREAK;
    have_break = 0;
    if (normalized_len > 0) {
      libxs_lexrule_ctx_t ctx;
      libxs_lexeme_t lexeme;
      memset(&ctx, 0, sizeof(ctx));
      ctx.text = text + token_start;
      ctx.length = (int)token_len;
      ctx.flags = flags;
      ctx.hash = (0 != (flags & LIBXS_LEXEME_WORD))
        ? libxs_textrule_wordhash((const unsigned char*)normalized,
          normalized_len) : 0;
      flags = libxs_lexrule_eval(&ctx, rules, nrules);
      memset(&lexeme, 0, sizeof(lexeme));
      lexeme.id = libxs_lexicon_id(lexicon, normalized,
        normalized_len, flags, create);
      lexeme.length = (unsigned short)token_len;
      lexeme.flags = (unsigned short)flags;
      result = libxs_lexeme_stream_push(stream, &lexeme);
    }
  }
  return result;
}


LIBXS_API int libxs_lexeme_stream_encode(libxs_lexicon_t* lexicon,
  libxs_lexeme_stream_t* stream, const unsigned char* text, size_t size,
  const libxs_lexrule_t* rules, int nrules, int create)
{
  int result = libxs_token_stream_encode(lexicon, stream, text, size,
    rules, nrules, create);
  return result;
}
