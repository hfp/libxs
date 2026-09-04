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

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(int argc, char* argv[])
{
  static const unsigned char input[] = "Who is Alice? Alice saw 123.";
  static const unsigned char meta_input[] =
    "M\xC3\xA9" "tatokenization carries syllables exactly.";
  const size_t input_size = sizeof(input) - 1;
  const size_t meta_input_size = sizeof(meta_input) - 1;
  libxs_token_stream_t meta_stream;
  libxs_tokenizer_t* tokenizer = NULL;
  libxs_lexeme_stream_t stream;
  libxs_lexeme_stream_t inflect_stream;
  libxs_lexeme_stream_t plain_stream;
  libxs_lexrule_t lexrules[96];
  libxs_lexnorm_t lexnorms[4];
  libxs_lexicon_t* lexicon = NULL;
  libxs_lexicon_t* loaded_lexicon = NULL;
  void* lexicon_buffer = NULL;
  size_t lexicon_buffer_size = 0;
  size_t i;
  int saw_sentence = 0, saw_break = 0;
  int lexrule_count = 0;
  int saw_question = 0, saw_entity = 0, saw_stop = 0, saw_number = 0;
  unsigned int first_alice = 0, second_alice = 0;
  int granularity;
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  libxs_token_stream_init(&meta_stream);
  libxs_lexeme_stream_init(&stream);
  libxs_lexeme_stream_init(&inflect_stream);
  libxs_lexeme_stream_init(&plain_stream);
  if (0 != stream.size || 0 != stream.capacity || NULL != stream.data) {
    FPRINTF(stderr, "ERROR line #%i: token stream init\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result
    && sizeof(libxs_token_t) != (size_t)LIBXS_TOKEN_BYTES)
  {
    FPRINTF(stderr, "ERROR line #%i: token size\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result
    && sizeof(libxs_lexeme_t) != (size_t)LIBXS_LEXEME_BYTES)
  {
    FPRINTF(stderr, "ERROR line #%i: lexeme size\n", __LINE__);
    result = EXIT_FAILURE;
  }
  for (granularity = LIBXS_TOKEN_GRANULARITY_NATIVE;
    granularity <= LIBXS_TOKEN_GRANULARITY_SYLLABLE
      && EXIT_SUCCESS == result; ++granularity)
  {
    unsigned char* decoded = NULL;
    size_t decoded_size = 0, token_pos = 0;
    int saw_continued = 0, saw_kind = 0, saw_space = 0, saw_sentence_meta = 0;
    if (NULL == tokenizer) tokenizer = libxs_tokenizer_create(granularity);
    else result = libxs_tokenizer_set_granularity(tokenizer, granularity);
    if (NULL == tokenizer
      || granularity != libxs_tokenizer_granularity(tokenizer))
    {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxs_token_stream_encode(tokenizer, &meta_stream, meta_input,
        meta_input_size);
    }
    if (EXIT_SUCCESS == result) {
      result = libxs_token_stream_decode(&meta_stream, &decoded,
        &decoded_size);
    }
    if (EXIT_SUCCESS == result
      && (decoded_size != meta_input_size
        || 0 != memcmp(decoded, meta_input, meta_input_size)))
    {
      FPRINTF(stderr, "ERROR line #%i: metatoken round-trip"
        " (%i bytes decoded, expected %i)\n", __LINE__,
        (int)decoded_size, (int)meta_input_size);
      result = EXIT_FAILURE;
    }
    while (EXIT_SUCCESS == result && token_pos < meta_stream.size) {
      size_t payload_size = 0;
      size_t ncells = libxs_token_span(meta_stream.data, meta_stream.size,
        token_pos, &payload_size);
      const libxs_token_t* token = meta_stream.data + token_pos;
      unsigned char payload[LIBXS_LEXEME_MAXBYTES + 1];
      libxs_token_info_t info;
      if (0 == ncells || 0 == payload_size
        || EXIT_SUCCESS != libxs_token_read(meta_stream.data,
          meta_stream.size, token_pos, payload, sizeof(payload), &info)
        || info.cells != ncells || info.length != payload_size
        || info.kind != libxs_token_kind(token))
      {
        result = EXIT_FAILURE;
      }
      else {
        const int kind = libxs_token_kind(token);
        size_t cell_pos;
        for (cell_pos = 0; cell_pos < ncells; ++cell_pos) {
          const libxs_token_t* cell = token + cell_pos;
          size_t payload_pos;
          for (payload_pos = 1 + libxs_token_len(cell);
            payload_pos < LIBXS_TOKEN_BYTES; ++payload_pos)
          {
            if (0 != cell->raw[payload_pos]) result = EXIT_FAILURE;
          }
        }
        if (ncells > 1) saw_continued = 1;
        if ((LIBXS_TOKEN_GRANULARITY_WORD == granularity
          || LIBXS_TOKEN_GRANULARITY_SYLLABLE == granularity)
          && LIBXS_TOKEN_TEXT == kind)
        {
          saw_kind = 1;
        }
        if (LIBXS_TOKEN_SPACE == kind) saw_space = 1;
        if (0 != libxs_token_is_sentence_end(token + ncells - 1)) {
          saw_sentence_meta = 1;
        }
        token_pos += ncells;
      }
    }
    if (EXIT_SUCCESS == result
      && LIBXS_TOKEN_GRANULARITY_NATIVE != granularity
      && (0 == saw_kind || 0 == saw_space || 0 == saw_sentence_meta))
    {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result
      && LIBXS_TOKEN_GRANULARITY_WORD == granularity
      && 0 == saw_continued)
    {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) {
      FPRINTF(stderr, "ERROR line #%i: metatoken granularity %i\n",
        __LINE__, granularity);
    }
    free(decoded);
    libxs_token_stream_release(&meta_stream);
  }
  libxs_tokenizer_destroy(tokenizer);
  if (EXIT_SUCCESS == result) {
    libxs_token_t lhs[2], rhs[1], prefix[1], longer[1];
    memset(lhs, 0, sizeof(lhs));
    memset(rhs, 0, sizeof(rhs));
    memset(prefix, 0, sizeof(prefix));
    memset(longer, 0, sizeof(longer));
    lhs[0].raw[0] = (unsigned char)(3 | LIBXS_TOKEN_CONTINUED
      | (LIBXS_TOKEN_TEXT << LIBXS_TOKEN_KIND_SHIFT));
    lhs[1].raw[0] = (unsigned char)(3
      | (LIBXS_TOKEN_TEXT << LIBXS_TOKEN_KIND_SHIFT));
    rhs[0].raw[0] = (unsigned char)(6 | LIBXS_TOKEN_SENTENCE
      | (LIBXS_TOKEN_PUNCT << LIBXS_TOKEN_KIND_SHIFT));
    prefix[0].raw[0] = (unsigned char)(3 | LIBXS_TOKEN_SENTENCE
      | (LIBXS_TOKEN_MARKUP << LIBXS_TOKEN_KIND_SHIFT));
    longer[0].raw[0] = (unsigned char)(4
      | (LIBXS_TOKEN_TEXT << LIBXS_TOKEN_KIND_SHIFT));
    memcpy(lhs[0].raw + 1, "abc", 3);
    memcpy(lhs[1].raw + 1, "def", 3);
    memcpy(rhs[0].raw + 1, "abcdef", 6);
    memcpy(prefix[0].raw + 1, "abc", 3);
    memcpy(longer[0].raw + 1, "abc", 3);
    prefix[0].raw[7] = 0xA5u;
    if (0 == libxs_token_payload_equal(lhs, prefix)
      || 0 != libxs_token_payload_equal(lhs, longer)
      || 0 == libxs_token_payload_match(lhs, 2, 0, rhs, 1, 0)
      || 0 != libxs_token_payload_match(lhs, 1, 0, rhs, 1, 0))
    {
      FPRINTF(stderr, "ERROR line #%i: payload matching\n", __LINE__);
      result = EXIT_FAILURE;
    }
    rhs[0].raw[6] = (unsigned char)'x';
    if (EXIT_SUCCESS == result
      && 0 != libxs_token_payload_match(lhs, 2, 0, rhs, 1, 0))
    {
      FPRINTF(stderr, "ERROR line #%i: payload mismatch\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_lexeme_stream_reserve(&stream, 2);
    if (EXIT_SUCCESS != result || stream.capacity < 2) {
      FPRINTF(stderr, "ERROR line #%i: token stream reserve\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    lexicon = libxs_lexicon_create();
    lexrule_count = libxs_lexrule_defaults(lexrules, 96);
    if (NULL == lexicon || lexrule_count <= 0) {
      FPRINTF(stderr, "ERROR line #%i: lexicon/rules init\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_lexeme_stream_encode(lexicon, &stream,
      input, input_size, lexrules, lexrule_count, NULL, 0, 1);
    if (EXIT_SUCCESS != result || stream.size < 7
      || libxs_lexicon_size(lexicon) < 6)
    {
      FPRINTF(stderr, "ERROR line #%i: token stream encode\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < stream.size; ++i) {
      int text_len = 0;
      unsigned int text_flags = 0;
      const char* text = libxs_lexicon_text(lexicon, stream.data[i].id,
        &text_len, &text_flags);
      libxs_lexeme_info_t info;
      const libxs_lexeme_t* const token = stream.data + i;
      libxs_lexeme_info(token, &info);
      if (info.length != libxs_lexeme_len(token)) {
        FPRINTF(stderr, "ERROR line #%i: token info length\n", __LINE__);
        result = EXIT_FAILURE;
        break;
      }
      if (0 != info.is_question) saw_question = 1;
      if (0 != info.is_entity) saw_entity = 1;
      if (0 != info.is_stop) saw_stop = 1;
      if (0 != info.is_number) saw_number = 1;
      if (0 != info.has_break) saw_break = 1;
      if (0 != info.is_sentence) saw_sentence = 1;
      if (NULL != text && 5 == text_len && 0 == memcmp(text, "alice", 5)) {
        if (0 == first_alice) first_alice = stream.data[i].id;
        else second_alice = stream.data[i].id;
      }
      if (NULL != text && 5 == text_len && 0 == memcmp(text, "<num>", 5)) {
        if (0 == (text_flags & LIBXS_LEXEME_NUMBER)) result = EXIT_FAILURE;
      }
    }
    if (0 == saw_question || 0 == saw_entity || 0 == saw_stop
      || 0 == saw_number || 0 == saw_sentence || 0 == saw_break
      || 0 == first_alice || first_alice != second_alice)
    {
      FPRINTF(stderr, "ERROR line #%i: token flags/ids\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result
      && first_alice != libxs_lexicon_id(lexicon, "alice", 5, 0, 0))
    {
      FPRINTF(stderr, "ERROR line #%i: lexicon text-to-id\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    static const unsigned char inflect[] =
      "bring brought count counted counting stretch stretched";
    memset(lexnorms, 0, sizeof(lexnorms));
    memcpy(lexnorms[0].from, "brought", 8);
    memcpy(lexnorms[0].to, "bring", 6);
    memcpy(lexnorms[1].from, "counted", 8);
    memcpy(lexnorms[1].to, "count", 6);
    memcpy(lexnorms[2].from, "counting", 9);
    memcpy(lexnorms[2].to, "count", 6);
    memcpy(lexnorms[3].from, "stretched", 10);
    memcpy(lexnorms[3].to, "stretch", 8);
    result = libxs_lexeme_stream_encode(lexicon, &inflect_stream,
      inflect, sizeof(inflect) - 1, lexrules, lexrule_count,
      lexnorms, 4, 1);
    if (EXIT_SUCCESS != result || 7 != inflect_stream.size
      || inflect_stream.data[0].id != inflect_stream.data[1].id
      || inflect_stream.data[2].id != inflect_stream.data[3].id
      || inflect_stream.data[2].id != inflect_stream.data[4].id
      || inflect_stream.data[5].id != inflect_stream.data[6].id)
    {
      FPRINTF(stderr, "ERROR line #%i: inflection normalization\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    static const unsigned char plain[] = "stretch stretched";
    result = libxs_lexeme_stream_encode(lexicon, &plain_stream,
      plain, sizeof(plain) - 1, lexrules, lexrule_count, NULL, 0, 1);
    if (EXIT_SUCCESS != result || 2 != plain_stream.size
      || plain_stream.data[0].id == plain_stream.data[1].id)
    {
      FPRINTF(stderr, "ERROR line #%i: optional normalization\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    size_t pos = 0, ngroups = 0, ncovered = 0, n;
    libxs_lexeme_t pieces[4];
    while (0 != (n = libxs_lexeme_word_next(stream.data, stream.size, pos))) {
      size_t k;
      for (k = pos + 1; k < pos + n; ++k) {
        if (0 != (stream.data[k].flags & LIBXS_LEXEME_BREAK)) {
          FPRINTF(stderr, "ERROR line #%i: word group break\n", __LINE__);
          result = EXIT_FAILURE;
        }
      }
      ncovered += n;
      ++ngroups;
      pos += n;
    }
    if (ncovered != stream.size || ngroups < 2 || ngroups >= stream.size
      || 0 != libxs_lexeme_word_next(stream.data, stream.size, stream.size)
      || 0 != libxs_lexeme_word_next(NULL, 4, 0))
    {
      FPRINTF(stderr, "ERROR line #%i: word iteration\n", __LINE__);
      result = EXIT_FAILURE;
    }
    memset(pieces, 0, sizeof(pieces));
    pieces[0].flags = LIBXS_LEXEME_WORD | LIBXS_LEXEME_BREAK;
    pieces[1].flags = LIBXS_LEXEME_WORD;
    pieces[2].flags = LIBXS_LEXEME_WORD | LIBXS_LEXEME_BREAK;
    pieces[3].flags = LIBXS_LEXEME_WORD;
    if (EXIT_SUCCESS == result
      && (2 != libxs_lexeme_word_next(pieces, 4, 0)
        || 1 != libxs_lexeme_word_next(pieces, 4, 1)
        || 2 != libxs_lexeme_word_next(pieces, 4, 2)
        || 1 != libxs_lexeme_word_next(pieces, 1, 0)))
    {
      FPRINTF(stderr, "ERROR line #%i: sub-word grouping\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) { /* multi-token normalization */
    libxs_lexnorm_t norms[1];
    libxs_lexeme_stream_t norm_stream;
    const char* const norm_text = "who isn't fat";
    memset(norms, 0, sizeof(norms));
    strcpy(norms[0].from, "isn");
    strcpy(norms[0].to, "is not");
    libxs_lexeme_stream_init(&norm_stream);
    result = libxs_lexeme_stream_encode(lexicon, &norm_stream,
      (const unsigned char*)norm_text, strlen(norm_text),
      lexrules, lexrule_count, norms, 1, 1);
    if (EXIT_SUCCESS == result) {
      size_t k, nbytes = 0, nlen0 = 0;
      int saw_is = 0, saw_not = 0;
      for (k = 0; k < norm_stream.size; ++k) {
        int len = 0;
        const char* text = libxs_lexicon_text(lexicon,
          norm_stream.data[k].id, &len, NULL);
        nbytes += norm_stream.data[k].length;
        if (0 == norm_stream.data[k].length) ++nlen0;
        if (NULL != text && 2 == len && 0 == memcmp(text, "is", 2)) saw_is = 1;
        if (NULL != text && 3 == len && 0 == memcmp(text, "not", 3)) {
          saw_not = 1;
          /* the continuation must not start a new word */
          if (0 != (norm_stream.data[k].flags & LIBXS_LEXEME_BREAK)
            || 0 != norm_stream.data[k].length)
          {
            result = EXIT_FAILURE;
          }
        }
      }
      /* one source token expanded to two ids, byte total still the source */
      if (0 == saw_is || 0 == saw_not || 1 != nlen0
        || nbytes != strlen(norm_text) - 2 /* two spaces are not tokens */)
      {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) { /* the pair groups as ONE word */
        size_t pos = 0, n, ngroups = 0;
        while (0 != (n = libxs_lexeme_word_next(norm_stream.data,
          norm_stream.size, pos)))
        {
          ++ngroups;
          pos += n;
        }
        if (ngroups != 3) result = EXIT_FAILURE; /* who | isn't | fat */
      }
      if (EXIT_SUCCESS != result) {
        FPRINTF(stderr, "ERROR line #%i: multi-token norm\n", __LINE__);
      }
    }
    libxs_lexeme_stream_release(&norm_stream);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_lexicon_save(lexicon, NULL, &lexicon_buffer_size);
    if (EXIT_SUCCESS == result && lexicon_buffer_size > 0) {
      lexicon_buffer = malloc(lexicon_buffer_size);
      if (NULL != lexicon_buffer) {
        result = libxs_lexicon_save(lexicon, lexicon_buffer,
          &lexicon_buffer_size);
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      loaded_lexicon = libxs_lexicon_load(lexicon_buffer,
        lexicon_buffer_size);
      if (NULL == loaded_lexicon
        || libxs_lexicon_size(loaded_lexicon) != libxs_lexicon_size(lexicon))
      {
        FPRINTF(stderr, "ERROR line #%i: lexicon save/load\n", __LINE__);
        result = EXIT_FAILURE;
      }
    }
  }
  /**
   * Multi-byte letters belong to words; multi-byte punctuation does not.
   *
   * Both halves matter and they pull in opposite directions. Accepting every byte
   * >= 0x80 as a word character glues "don" and "t" together across a typographic
   * apostrophe; rejecting them all (plain ctype) cuts a German word into pieces and
   * classes the middle piece as punctuation, which is then read as a SENTENCE
   * boundary. So this asserts a word count, not merely that nothing crashed.
   */
  if (EXIT_SUCCESS == result) {
    static const unsigned char german[] =
      "Das M\xC3\xA4" "dchen und der K\xC3\xB6" "nig.";
    static const unsigned char quoted[] = "don\xE2\x80\x99t stop";
    libxs_lexeme_stream_t utf8_stream;
    libxs_lexicon_t* utf8_lexicon = libxs_lexicon_create();
    libxs_lexeme_stream_init(&utf8_stream);
    if (NULL == utf8_lexicon) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      result = libxs_lexeme_stream_encode(utf8_lexicon, &utf8_stream, german,
        sizeof(german) - 1, NULL, 0, NULL, 0, 1);
    }
    if (EXIT_SUCCESS == result) {
      size_t nwords = 0;
      for (i = 0; i < utf8_stream.size; ++i) {
        if (0 != (utf8_stream.data[i].flags & LIBXS_LEXEME_WORD)) ++nwords;
      }
      /* "Das Maedchen und der Koenig" = five words, umlauts kept whole. */
      if (5 != nwords) {
        FPRINTF(stderr, "ERROR line #%i: utf-8 letters split a word"
          " (%i words, expected 5)\n", __LINE__, (int)nwords);
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      /* encode APPENDS, so the stream is reset before the second sentence. */
      libxs_lexeme_stream_release(&utf8_stream);
      libxs_lexeme_stream_init(&utf8_stream);
      result = libxs_lexeme_stream_encode(utf8_lexicon, &utf8_stream, quoted,
        sizeof(quoted) - 1, NULL, 0, NULL, 0, 1);
    }
    if (EXIT_SUCCESS == result) {
      size_t npunct = 0;
      for (i = 0; i < utf8_stream.size; ++i) {
        if (0 != (utf8_stream.data[i].flags & LIBXS_LEXEME_PUNCT)) ++npunct;
      }
      /* The apostrophe stays punctuation, so "don" and "t" stay separate. */
      if (1 != npunct) {
        FPRINTF(stderr, "ERROR line #%i: utf-8 punctuation joined a word"
          " (%i punct, expected 1)\n", __LINE__, (int)npunct);
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      int span = 0;
      /* The public predicate reports the span so callers advance correctly. */
      if (0 == libxs_lexeme_is_word_char(german + 4, 3, &span) || 1 != span
        || 0 == libxs_lexeme_is_word_char(german + 5, 2, &span) || 2 != span
        || 0 != libxs_lexeme_is_word_char(quoted + 3, 3, &span) || 3 != span)
      {
        FPRINTF(stderr, "ERROR line #%i: libxs_lexeme_is_word_char\n",
          __LINE__);
        result = EXIT_FAILURE;
      }
    }
    libxs_lexeme_stream_release(&utf8_stream);
    libxs_lexicon_destroy(utf8_lexicon);
  }
  free(lexicon_buffer);
  libxs_lexicon_destroy(loaded_lexicon);
  libxs_lexicon_destroy(lexicon);
  libxs_lexeme_stream_release(&plain_stream);
  libxs_lexeme_stream_release(&inflect_stream);
  libxs_lexeme_stream_release(&stream);
  if (EXIT_SUCCESS == result
    && (0 != stream.size || 0 != stream.capacity || NULL != stream.data))
  {
    FPRINTF(stderr, "ERROR line #%i: token stream release\n", __LINE__);
    result = EXIT_FAILURE;
  }
  return result;
}
