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
  static const unsigned char input[] = "Hello world. Hello world. $x$";
  const size_t input_size = sizeof(input) - 1;
  libxs_token_stream_t stream;
  unsigned char* decoded = NULL;
  size_t decoded_size = 0;
  size_t i;
  int saw_copy = 0, saw_sentence = 0, saw_break = 0;
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  libxs_token_stream_init(&stream);
  if (0 != stream.size || 0 != stream.capacity || NULL != stream.data) {
    FPRINTF(stderr, "ERROR line #%i: token stream init\n", __LINE__);
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
    result = libxs_token_stream_encode(&stream, input, input_size);
    if (EXIT_SUCCESS != result || 0 == stream.size) {
      FPRINTF(stderr, "ERROR line #%i: token stream encode\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_token_stream_decode(&stream, &decoded, &decoded_size);
    if (EXIT_SUCCESS != result || decoded_size != input_size
      || 0 != memcmp(decoded, input, input_size))
    {
      FPRINTF(stderr, "ERROR line #%i: token stream decode\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < stream.size; ++i) {
      libxs_token_info_t info;
      const libxs_token_t* const token = stream.data + i;
      libxs_token_info(token, &info);
      if (info.length != libxs_token_len(token)) {
        FPRINTF(stderr, "ERROR line #%i: token info length\n", __LINE__);
        result = EXIT_FAILURE;
        break;
      }
      if (info.distance != libxs_token_distance(token)) {
        FPRINTF(stderr, "ERROR line #%i: token info distance\n", __LINE__);
        result = EXIT_FAILURE;
        break;
      }
      if (0 != info.is_copy) saw_copy = 1;
      if (0 != info.has_break) saw_break = 1;
      if (0 != info.is_sentence) saw_sentence = 1;
    }
  }
  if (EXIT_SUCCESS == result && (0 == saw_copy || 0 == saw_break || 0 == saw_sentence)) {
    FPRINTF(stderr, "ERROR line #%i: token stream flags\n", __LINE__);
    result = EXIT_FAILURE;
  }
  free(decoded);
  libxs_token_stream_release(&stream);
  if (EXIT_SUCCESS == result
    && (0 != stream.size || 0 != stream.capacity || NULL != stream.data))
  {
    FPRINTF(stderr, "ERROR line #%i: token stream release\n", __LINE__);
    result = EXIT_FAILURE;
  }
  return result;
}
