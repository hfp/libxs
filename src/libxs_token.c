/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_token.h>

#include <ctype.h>
#include <string.h>


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
int internal_libxs_token_prefer_break(unsigned char prev, unsigned char next)
{
  int result = 0;
  if (0 != isspace(prev) || 0 != isspace(next)
    || 0 != ispunct(prev) || 0 != ispunct(next))
  {
    result = 1;
  }
  else if ((0 != isalpha(prev) && 0 != isdigit(next))
    || (0 != isdigit(prev) && 0 != isalpha(next)))
  {
    result = 1;
  }
  else if (0 != islower(prev) && 0 != isupper(next)) {
    result = 1;
  }
  return result;
}


LIBXS_API_INLINE
int internal_libxs_token_is_sentence_char(unsigned char ch)
{
  return ('.' == ch || '?' == ch || '!' == ch) ? 1 : 0;
}


LIBXS_API_INLINE
size_t internal_libxs_token_literal_len(const unsigned char* text,
  size_t size, size_t pos)
{
  size_t offset = 0, best = 0;
  if (NULL != text && pos < size) {
    while (offset < LIBXS_TOKEN_PAYLOAD && pos + offset < size) {
      const size_t step = internal_libxs_token_codepoint_size(text, size,
        pos + offset);
      if (offset + step > LIBXS_TOKEN_PAYLOAD) break;
      offset += step;
      best = offset;
      if (pos + offset < size && 1 == step) {
        if (0 != internal_libxs_token_prefer_break(
          text[pos + offset - 1], text[pos + offset])) break;
      }
      if (1 == step && (0 != isspace(text[pos + offset - 1])
        || 0 != ispunct(text[pos + offset - 1])))
      {
        break;
      }
    }
  }
  return best;
}


LIBXS_API_INLINE
int internal_libxs_token_find_copy(const unsigned char* text, size_t size,
  size_t pos, size_t* best_len, size_t* best_dist, int* prefer_break_flag)
{
  int result = 0;
  size_t distance, max_distance;
  if (NULL == text || NULL == best_len || NULL == best_dist
    || NULL == prefer_break_flag || pos >= size)
  {
    result = 0;
  }
  else {
    *best_len = 0;
    *best_dist = 0;
    *prefer_break_flag = 0;
    max_distance = (pos < LIBXS_TOKEN_MAX_DISTANCE)
      ? pos : LIBXS_TOKEN_MAX_DISTANCE;
    for (distance = 1; distance <= max_distance; ++distance) {
      size_t current_len = 0;
      while (current_len < LIBXS_TOKEN_MAX_COPY && pos + current_len < size
        && text[pos - distance + current_len] == text[pos + current_len])
      {
        ++current_len;
        if (pos + current_len < size
          && 1 < internal_libxs_token_codepoint_size(text, size,
            pos + current_len - 1)
          && pos + current_len < size
          && 0x80u <= text[pos + current_len]
          && text[pos + current_len] < 0xC0u)
        {
          --current_len;
          break;
        }
      }
      while (LIBXS_TOKEN_MIN_COPY <= current_len
        && pos + current_len < size
        && 0x80u <= text[pos + current_len]
        && text[pos + current_len] < 0xC0u)
      {
        --current_len;
      }
      if (current_len >= LIBXS_TOKEN_MIN_COPY && current_len > *best_len) {
        *best_len = current_len;
        *best_dist = distance;
        if (pos + current_len < size) {
          *prefer_break_flag = internal_libxs_token_prefer_break(
            text[pos + current_len - 1], text[pos + current_len]);
        }
        result = 1;
        if (LIBXS_TOKEN_MAX_COPY == current_len) break;
      }
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


LIBXS_API_INLINE
void internal_libxs_token_init_literal(libxs_token_t* token,
  const unsigned char* text, size_t length,
  int break_flag, int sentence_flag, int markup_flag)
{
  if (NULL != token) {
    memset(token->raw, 0, sizeof(token->raw));
    token->raw[0] = (unsigned char)(length & LIBXS_TOKEN_LEN_MASK);
    if (0 != break_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_BREAK;
    if (0 != sentence_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_SENTENCE;
    if (0 != markup_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_MARKUP;
    if (NULL != text && 0 != length) memcpy(token->raw + 1, text, length);
  }
}


LIBXS_API_INLINE
void internal_libxs_token_init_copy(libxs_token_t* token,
  size_t length, size_t distance,
  int break_flag, int sentence_flag, int markup_flag)
{
  if (NULL != token) {
    memset(token->raw, 0, sizeof(token->raw));
    token->raw[0] = (unsigned char)(
      LIBXS_TOKEN_FLAG_COPY | (length & LIBXS_TOKEN_LEN_MASK));
    if (0 != break_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_BREAK;
    if (0 != sentence_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_SENTENCE;
    if (0 != markup_flag) token->raw[0] |= LIBXS_TOKEN_FLAG_MARKUP;
    token->raw[1] = (unsigned char)(distance & 0xFFu);
    token->raw[2] = (unsigned char)((distance >> 8) & 0xFFu);
  }
}


LIBXS_API_INLINE
int internal_libxs_token_detect_sentence(const unsigned char* text,
  size_t size, size_t pos, size_t token_len)
{
  int result = 0;
  size_t end = pos + token_len;
  if (end > 0 && end <= size) {
    const unsigned char last = text[end - 1];
    if (0 != internal_libxs_token_is_sentence_char(last)) {
      if (end == size || 0 != isspace(text[end])
        || '"' == text[end] || '\'' == text[end])
      {
        result = 1;
      }
    }
    if (0 == result && end >= 2) {
      const unsigned char prev = text[end - 2];
      if (0 != internal_libxs_token_is_sentence_char(prev)
        && (0 != isspace(last) || '"' == last))
      {
        result = 1;
      }
    }
  }
  return result;
}


LIBXS_API void libxs_token_info(libxs_token_info_t* info,
  const libxs_token_t* token)
{
  if (NULL != info) {
    if (NULL != token) {
      const unsigned char ctrl = token->raw[0];
      info->length = (int)(ctrl & LIBXS_TOKEN_LEN_MASK);
      info->is_copy = (0 != (ctrl & LIBXS_TOKEN_FLAG_COPY)) ? 1 : 0;
      info->has_break = (0 != (ctrl & LIBXS_TOKEN_FLAG_BREAK)) ? 1 : 0;
      info->is_sentence = (0 != (ctrl & LIBXS_TOKEN_FLAG_SENTENCE)) ? 1 : 0;
      info->is_markup = (0 != (ctrl & LIBXS_TOKEN_FLAG_MARKUP)) ? 1 : 0;
      info->distance = (0 != info->is_copy)
        ? (int)((unsigned int)token->raw[1] | ((unsigned int)token->raw[2] << 8))
        : 0;
    }
    else {
      info->length = 0;
      info->distance = 0;
      info->is_copy = 0;
      info->has_break = 0;
      info->is_sentence = 0;
      info->is_markup = 0;
    }
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


LIBXS_API void libxs_token_stream_destroy(libxs_token_stream_t* stream)
{
  if (NULL != stream) {
    free(stream->data);
    stream->data = NULL;
    stream->size = 0;
    stream->capacity = 0;
  }
}


LIBXS_API int libxs_tokenize(const unsigned char* text, size_t size,
  libxs_token_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  size_t pos = 0;
  if (NULL == text || NULL == stream) result = EXIT_FAILURE;
  while (EXIT_SUCCESS == result && pos < size) {
    libxs_token_t token;
    size_t best_len = 0, best_dist = 0;
    int pbf = 0, sentence = 0, markup = 0;
    if (0 != internal_libxs_token_find_copy(text, size, pos,
      &best_len, &best_dist, &pbf))
    {
      sentence = (0 != pbf)
        ? internal_libxs_token_detect_sentence(text, size, pos, best_len) : 0;
      markup = internal_libxs_token_detect_markup(text, size, pos, best_len);
      internal_libxs_token_init_copy(&token, best_len, best_dist,
        pbf, sentence, markup);
      result = libxs_token_stream_push(stream, &token);
      pos += best_len;
    }
    else {
      best_len = internal_libxs_token_literal_len(text, size, pos);
      if (0 == best_len) best_len = 1;
      if (pos + best_len < size) {
        pbf = internal_libxs_token_prefer_break(
          text[pos + best_len - 1], text[pos + best_len]);
      }
      sentence = (0 != pbf)
        ? internal_libxs_token_detect_sentence(text, size, pos, best_len) : 0;
      markup = internal_libxs_token_detect_markup(text, size, pos, best_len);
      internal_libxs_token_init_literal(&token, text + pos, best_len,
        pbf, sentence, markup);
      result = libxs_token_stream_push(stream, &token);
      pos += best_len;
    }
  }
  return result;
}


LIBXS_API int libxs_token_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size)
{
  int result = EXIT_FAILURE;
  size_t total = 0, i;
  unsigned char* buffer;
  if (NULL != stream && NULL != text && NULL != size) {
    for (i = 0; i < stream->size; ++i) {
      total += libxs_token_len(stream->data + i);
    }
    buffer = (unsigned char*)malloc(total + 1);
    if (NULL != buffer) {
      size_t offset = 0;
      result = EXIT_SUCCESS;
      for (i = 0; i < stream->size && EXIT_SUCCESS == result; ++i) {
        const libxs_token_t* t = stream->data + i;
        const size_t length = libxs_token_len(t);
        if (0 != libxs_token_is_copy(t)) {
          const size_t distance = (size_t)t->raw[1]
            | ((size_t)t->raw[2] << 8);
          size_t j;
          if (0 == distance || distance > offset) {
            result = EXIT_FAILURE;
          }
          else {
            for (j = 0; j < length; ++j) {
              buffer[offset + j] = buffer[offset - distance + j];
            }
          }
        }
        else {
          memcpy(buffer + offset, t->raw + 1, length);
        }
        offset += length;
      }
      if (EXIT_SUCCESS == result) {
        buffer[total] = 0;
        *text = buffer;
        *size = total;
      }
      else {
        free(buffer);
      }
    }
  }
  return result;
}
