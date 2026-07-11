/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_token.h>

#include <stdio.h>
#include <string.h>

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
  const size_t input_size = sizeof(input) - 1;
  libxs_token_stream_t stream;
  libxs_token_stream_t inflect_stream;
  libxs_token_stream_t plain_stream;
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
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  libxs_token_stream_init(&stream);
  libxs_token_stream_init(&inflect_stream);
  libxs_token_stream_init(&plain_stream);
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
  if (EXIT_SUCCESS == result) {
    result = libxs_token_stream_reserve(&stream, 2);
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
    result = libxs_token_stream_encode(lexicon, &stream,
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
      libxs_token_info_t info;
      const libxs_token_t* const token = stream.data + i;
      libxs_token_info(token, &info);
      if (info.length != libxs_token_len(token)) {
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
    result = libxs_token_stream_encode(lexicon, &inflect_stream,
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
    result = libxs_token_stream_encode(lexicon, &plain_stream,
      plain, sizeof(plain) - 1, lexrules, lexrule_count, NULL, 0, 1);
    if (EXIT_SUCCESS != result || 2 != plain_stream.size
      || plain_stream.data[0].id == plain_stream.data[1].id)
    {
      FPRINTF(stderr, "ERROR line #%i: optional normalization\n", __LINE__);
      result = EXIT_FAILURE;
    }
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
  free(lexicon_buffer);
  libxs_lexicon_destroy(loaded_lexicon);
  libxs_lexicon_destroy(lexicon);
  libxs_token_stream_release(&plain_stream);
  libxs_token_stream_release(&inflect_stream);
  libxs_token_stream_release(&stream);
  if (EXIT_SUCCESS == result
    && (0 != stream.size || 0 != stream.capacity || NULL != stream.data))
  {
    FPRINTF(stderr, "ERROR line #%i: token stream release\n", __LINE__);
    result = EXIT_FAILURE;
  }
  return result;
}
