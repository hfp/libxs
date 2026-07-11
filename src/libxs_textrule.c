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


LIBXS_API unsigned int libxs_textrule_wordhash(
  const unsigned char* word, int len)
{
  unsigned int h = 5381;
  int i;
  if (NULL == word || len <= 0) return 0;
  for (i = 0; i < len; ++i) {
    h = ((h << 5) + h) ^ (unsigned int)word[i];
  }
  return h;
}


LIBXS_API_INLINE
int internal_textrule_prev_word(const unsigned char* text, int byte_pos,
  int* out_start, int* out_len)
{
  int wstart, wlen;
  if (NULL == text || byte_pos <= 0) return 0;
  wstart = byte_pos - 1;
  while (wstart > 0 && 0 == isspace(text[wstart - 1])
    && 0 == ispunct(text[wstart - 1])) --wstart;
  wlen = byte_pos - wstart;
  if (wlen <= 0) return 0;
  *out_start = wstart;
  *out_len = wlen;
  return 1;
}


LIBXS_API_INLINE
int internal_textrule_match(const libxs_textrule_t* rule,
  const libxs_textrule_ctx_t* ctx)
{
  int result = 0;
  switch ((int)rule->tmpl) {
    case LIBXS_TRULE_PREV_WORD_SHORT_UPPER: {
      int wstart = 0, wlen = 0;
      if (0 != internal_textrule_prev_word(ctx->text, ctx->byte_pos,
        &wstart, &wlen))
      {
        if (wlen >= 1 && wlen <= 4 && 0 != isupper(ctx->text[wstart])) {
          unsigned int h = libxs_textrule_wordhash(
            ctx->text + wstart, wlen);
          if (0 == rule->argument || h == rule->argument) result = 1;
        }
      }
    } break;
    case LIBXS_TRULE_NEXT_CHAR_UPPER: {
      int npos = ctx->byte_pos;
      const libxs_token_t* t = ctx->token;
      if (NULL != t) npos += (int)libxs_token_len(t);
      while (npos < (int)ctx->text_size
        && 0 != isspace(ctx->text[npos])) ++npos;
      if (npos < (int)ctx->text_size
        && 0 != isupper(ctx->text[npos])) result = 1;
    } break;
    case LIBXS_TRULE_NEXT_CHAR_LOWER: {
      int npos = ctx->byte_pos;
      const libxs_token_t* t = ctx->token;
      if (NULL != t) npos += (int)libxs_token_len(t);
      while (npos < (int)ctx->text_size
        && 0 != isspace(ctx->text[npos])) ++npos;
      if (npos < (int)ctx->text_size
        && 0 != islower(ctx->text[npos])) result = 1;
    } break;
    case LIBXS_TRULE_PREV_TOKEN_BREAK: {
      if (NULL != ctx->prev_token
        && 0 != libxs_token_has_break(ctx->prev_token)) result = 1;
    } break;
    case LIBXS_TRULE_INSIDE_DELIMITERS: {
      unsigned char open_ch = (unsigned char)(rule->argument & 0xFFu);
      unsigned char close_ch = 0;
      int i, depth = 0;
      if ('(' == open_ch) close_ch = ')';
      else if ('[' == open_ch) close_ch = ']';
      else if ('"' == open_ch) close_ch = '"';
      if (0 != close_ch) {
        for (i = 0; i < ctx->byte_pos; ++i) {
          if (ctx->text[i] == open_ch) ++depth;
          else if (ctx->text[i] == close_ch) --depth;
        }
        if (depth > 0) result = 1;
      }
    } break;
    default: break;
  }
  return result;
}


LIBXS_API int libxs_textrule_eval(const libxs_textrule_ctx_t* ctx,
  const libxs_textrule_t* rules, int nrules)
{
  int result, i;
  if (NULL == ctx || NULL == ctx->token) return 0;
  if (0 == libxs_token_is_sentence_end(ctx->token)) return 0;
  result = 1;
  if (NULL != rules) {
    for (i = 0; i < nrules; ++i) {
      if (0 != internal_textrule_match(rules + i, ctx)) {
        result = ((int)rules[i].action == LIBXS_TRULE_CONFIRM) ? 1 : 0;
      }
    }
  }
  return result;
}


LIBXS_API int libxs_textrule_load(const libxs_registry_t* registry,
  libxs_textrule_t* rules, int max_rules)
{
  int count = 0, seq;
  unsigned char key[16];
  if (NULL == registry || NULL == rules || max_rules <= 0) return 0;
  memcpy(key, "TRULE:", 6);
  for (seq = 0; seq < max_rules; ++seq) {
    size_t key_size;
    key[6] = (unsigned char)((seq >> 8) & 0xFF);
    key[7] = (unsigned char)(seq & 0xFF);
    key_size = 8;
    if (0 != libxs_registry_get_copy(registry, key, key_size,
      rules + count, sizeof(libxs_textrule_t), NULL))
    {
      ++count;
    }
    else break;
  }
  return count;
}


LIBXS_API int libxs_textrule_save(libxs_registry_t* registry,
  const libxs_textrule_t* rules, int nrules)
{
  int result = EXIT_SUCCESS;
  int i;
  unsigned char key[16];
  if (NULL == registry || NULL == rules) return EXIT_FAILURE;
  memcpy(key, "TRULE:", 6);
  for (i = 0; i < nrules && EXIT_SUCCESS == result; ++i) {
    size_t key_size;
    key[6] = (unsigned char)((i >> 8) & 0xFF);
    key[7] = (unsigned char)(i & 0xFF);
    key_size = 8;
    if (NULL == libxs_registry_set(registry, key, key_size,
      rules + i, sizeof(libxs_textrule_t), NULL))
    {
      result = EXIT_FAILURE;
    }
  }
  return result;
}


LIBXS_API int libxs_textrule_defaults(libxs_textrule_t* rules, int max_rules)
{
  int n = 0;
  if (NULL == rules || max_rules <= 0) return 0;
  if (n < max_rules) {
    memset(rules + n, 0, sizeof(libxs_textrule_t));
    rules[n].tmpl = LIBXS_TRULE_PREV_WORD_SHORT_UPPER;
    rules[n].action = LIBXS_TRULE_SUPPRESS;
    rules[n].argument = 0;
    ++n;
  }
  return n;
}


LIBXS_API_INLINE
int internal_lexrule_match(const libxs_lexrule_t* rule,
  const libxs_lexrule_ctx_t* ctx)
{
  int result = 0;
  if (NULL != rule && NULL != ctx && NULL != ctx->text) {
    switch ((int)rule->tmpl) {
      case LIBXS_LRULE_WORD_HASH: {
        if (0 != (ctx->flags & LIBXS_LEXEME_WORD)
          && ctx->hash == rule->argument) result = 1;
      } break;
      case LIBXS_LRULE_WORD_INITIAL_UPPER: {
        if (0 != (ctx->flags & LIBXS_LEXEME_WORD) && ctx->length > 0
          && 0 != isupper(ctx->text[0])) result = 1;
      } break;
      case LIBXS_LRULE_PUNCT_CHAR: {
        if (0 != (ctx->flags & LIBXS_LEXEME_PUNCT) && 1 == ctx->length
          && ctx->text[0] == (unsigned char)(rule->argument & 0xFFu))
        {
          result = 1;
        }
      } break;
      default: break;
    }
  }
  return result;
}


LIBXS_API_INLINE
int internal_lexrule_add(libxs_lexrule_t* rules, int max_rules, int count,
  unsigned char tmpl, unsigned char action, unsigned short flags,
  unsigned int argument)
{
  int result = count;
  if (NULL != rules && count < max_rules) {
    memset(rules + count, 0, sizeof(libxs_lexrule_t));
    rules[count].tmpl = tmpl;
    rules[count].action = action;
    rules[count].flags = flags;
    rules[count].argument = argument;
    result = count + 1;
  }
  return result;
}


LIBXS_API_INLINE
int internal_lexrule_add_word(libxs_lexrule_t* rules, int max_rules,
  int count, const char* word, unsigned short flags)
{
  int result = count;
  if (NULL != word) {
    result = internal_lexrule_add(rules, max_rules, count,
      LIBXS_LRULE_WORD_HASH, LIBXS_LRULE_SET, flags,
      libxs_textrule_wordhash((const unsigned char*)word, (int)strlen(word)));
  }
  return result;
}


LIBXS_API unsigned int libxs_lexrule_eval(const libxs_lexrule_ctx_t* ctx,
  const libxs_lexrule_t* rules, int nrules)
{
  unsigned int result = 0;
  int rule_pos;
  if (NULL != ctx) {
    result = ctx->flags;
    if (NULL != rules) {
      for (rule_pos = 0; rule_pos < nrules; ++rule_pos) {
        if (0 != internal_lexrule_match(rules + rule_pos, ctx)) {
          if (LIBXS_LRULE_SET == (int)rules[rule_pos].action) {
            result |= rules[rule_pos].flags;
          }
          else if (LIBXS_LRULE_CLEAR == (int)rules[rule_pos].action) {
            result &= ~((unsigned int)rules[rule_pos].flags);
          }
        }
      }
    }
  }
  return result;
}


LIBXS_API int libxs_lexrule_load(const libxs_registry_t* registry,
  libxs_lexrule_t* rules, int max_rules)
{
  int count = 0, seq;
  unsigned char key[16];
  if (NULL == registry || NULL == rules || max_rules <= 0) return 0;
  memcpy(key, "LRULE:", 6);
  for (seq = 0; seq < max_rules; ++seq) {
    size_t key_size;
    key[6] = (unsigned char)((seq >> 8) & 0xFF);
    key[7] = (unsigned char)(seq & 0xFF);
    key_size = 8;
    if (0 != libxs_registry_get_copy(registry, key, key_size,
      rules + count, sizeof(libxs_lexrule_t), NULL))
    {
      ++count;
    }
    else break;
  }
  return count;
}


LIBXS_API int libxs_lexrule_save(libxs_registry_t* registry,
  const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_SUCCESS;
  int rule_pos;
  unsigned char key[16];
  if (NULL == registry || NULL == rules) return EXIT_FAILURE;
  memcpy(key, "LRULE:", 6);
  for (rule_pos = 0; rule_pos < nrules && EXIT_SUCCESS == result; ++rule_pos) {
    size_t key_size;
    key[6] = (unsigned char)((rule_pos >> 8) & 0xFF);
    key[7] = (unsigned char)(rule_pos & 0xFF);
    key_size = 8;
    if (NULL == libxs_registry_set(registry, key, key_size,
      rules + rule_pos, sizeof(libxs_lexrule_t), NULL))
    {
      result = EXIT_FAILURE;
    }
  }
  return result;
}


LIBXS_API int libxs_lexrule_defaults(libxs_lexrule_t* rules, int max_rules)
{
  static const char* const stopwords[] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "i", "in",
    "is", "it", "me", "of", "on", "or", "our", "she", "should",
    "that", "the", "their", "them", "there", "they", "this", "to",
    "was", "we", "were", "with", "would", "you", "your"
  };
  static const char* const question_words[] = {
    "who", "what", "where", "when", "why", "how", "which", "whose",
    "whom"
  };
  int count = 0;
  int word_pos;
  if (NULL == rules || max_rules <= 0) return 0;
  count = internal_lexrule_add(rules, max_rules, count,
    LIBXS_LRULE_WORD_INITIAL_UPPER, LIBXS_LRULE_SET,
    LIBXS_LEXEME_ENTITY, 0);
  count = internal_lexrule_add(rules, max_rules, count,
    LIBXS_LRULE_PUNCT_CHAR, LIBXS_LRULE_SET,
    LIBXS_LEXEME_SENTENCE, '.');
  count = internal_lexrule_add(rules, max_rules, count,
    LIBXS_LRULE_PUNCT_CHAR, LIBXS_LRULE_SET,
    LIBXS_LEXEME_SENTENCE, '!');
  count = internal_lexrule_add(rules, max_rules, count,
    LIBXS_LRULE_PUNCT_CHAR, LIBXS_LRULE_SET,
    LIBXS_LEXEME_SENTENCE | LIBXS_LEXEME_QUESTION, '?');
  for (word_pos = 0;
    word_pos < (int)(sizeof(question_words) / sizeof(*question_words));
    ++word_pos)
  {
    count = internal_lexrule_add_word(rules, max_rules, count,
      question_words[word_pos], LIBXS_LEXEME_QUESTION | LIBXS_LEXEME_STOP);
  }
  for (word_pos = 0;
    word_pos < (int)(sizeof(stopwords) / sizeof(*stopwords));
    ++word_pos)
  {
    count = internal_lexrule_add_word(rules, max_rules, count,
      stopwords[word_pos], LIBXS_LEXEME_STOP);
  }
  return count;
}


LIBXS_API_INLINE
int internal_reflow_line_start(const unsigned char* text, size_t size,
  size_t pos)
{
  size_t i = pos;
  while (i < size && ' ' == text[i]) ++i;
  if (i >= size) return 0;
  if (0 != isdigit(text[i])) {
    size_t j = i;
    while (j < size && 0 != isdigit(text[j])) ++j;
    if (j < size && ('.' == text[j] || ')' == text[j])) return 1;
  }
  if ('-' == text[i] || '*' == text[i] || '+' == text[i]) {
    if (i + 1 < size && ' ' == text[i + 1]) return 1;
  }
  if ('(' == text[i] && i + 1 < size && 0 != isalpha(text[i + 1])
    && i + 2 < size && ')' == text[i + 2]) return 1;
  return 0;
}


LIBXS_API_INLINE
int internal_reflow_is_structural(const unsigned char* text, size_t size,
  size_t nl_pos)
{
  size_t line_start, next_start, prev_len;
  int next_indent, cur_indent;

  if (nl_pos + 1 >= size) return 1;
  if ('\n' == text[nl_pos + 1] || '\r' == text[nl_pos + 1]) return 1;

  line_start = nl_pos;
  while (line_start > 0 && '\n' != text[line_start - 1]) --line_start;
  prev_len = nl_pos - line_start;
  if (0 == prev_len) return 1;

  next_start = nl_pos + 1;
  if (next_start < size && '\r' == text[next_start]) ++next_start;

  if (0 != internal_reflow_line_start(text, size, next_start)) return 1;

  next_indent = 0;
  { size_t k = next_start;
    while (k < size && ' ' == text[k]) { ++next_indent; ++k; }
  }
  cur_indent = 0;
  { size_t k = line_start;
    while (k < nl_pos && ' ' == text[k]) { ++cur_indent; ++k; }
  }
  if (next_indent >= 4 && next_indent > cur_indent + 2) return 1;

  if (prev_len < 40) {
    int all_upper = 1, nwords = 0;
    size_t k;
    for (k = line_start; k < nl_pos; ++k) {
      if (0 != isalpha(text[k]) && 0 == isupper(text[k])) all_upper = 0;
      if (k == line_start || (0 != isspace(text[k - 1]) && 0 == isspace(text[k])))
        ++nwords;
    }
    if (0 != all_upper && nwords <= 6) return 1;
    if (nwords <= 3 && prev_len < 25) return 1;
  }

  { size_t k = next_start;
    while (k < size && ' ' == text[k]) ++k;
    if (k < size && 0 != islower(text[k]) && prev_len >= 40) return 0;
  }

  if (prev_len < 60) return 1;
  return 0;
}


LIBXS_API int libxs_text_reflow(const unsigned char* text, size_t size,
  unsigned char** out, size_t* out_size)
{
  int result = EXIT_FAILURE;
  unsigned char* buf;
  size_t i, j;
  if (NULL == text || NULL == out || NULL == out_size) return EXIT_FAILURE;
  buf = (unsigned char*)malloc(size + 1);
  if (NULL == buf) return EXIT_FAILURE;
  j = 0;
  for (i = 0; i < size; ++i) {
    if ('\n' == text[i]) {
      if (i + 1 < size && '\r' == text[i + 1]) ++i;
      if (0 != internal_reflow_is_structural(text, size, i)) {
        buf[j++] = '\n';
      }
      else {
        if (j > 0 && ' ' != buf[j - 1]) buf[j++] = ' ';
      }
    }
    else if ('\r' == text[i]) {
      size_t nl = i;
      if (i + 1 < size && '\n' == text[i + 1]) { nl = i + 1; ++i; }
      if (0 != internal_reflow_is_structural(text, size, nl)) {
        buf[j++] = '\n';
      }
      else {
        if (j > 0 && ' ' != buf[j - 1]) buf[j++] = ' ';
      }
    }
    else {
      buf[j++] = text[i];
    }
  }
  buf[j] = 0;
  *out = buf;
  *out_size = j;
  result = EXIT_SUCCESS;
  return result;
}
