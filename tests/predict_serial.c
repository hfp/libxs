/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_malloc.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define NENTRY 800
#define NFEAT 4
#define NCLASS 3
#define DIRT 0xA5


/**
 * Same corpus every time, and one a forest has something to split on: the label
 * follows a threshold on two of the inputs, and the remaining inputs carry a
 * pattern that correlates with nothing.
 */
static void fill(double input[], double* out, int i)
{
  input[0] = (double)(i % 97);
  input[1] = (double)((i * 7) % 31);
  input[2] = (double)((i * 13) % 11);
  input[3] = (double)(i % 5);
  *out = (double)(((input[0] > 48.0) ? 1 : 0)
    + ((input[1] > 15.0) ? 1 : 0)) ;
  if (*out >= NCLASS) *out = NCLASS - 1;
}


static int build_model(libxs_predict_t* model)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], out;
    fill(input, &out, i);
    result = libxs_predict_push(NULL, model, input, &out);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(model, LIBXS_PREDICT_RF);
    result = libxs_predict_build(model, 0, 1, 0.0);
  }
  return result;
}


/** Serializes a freshly built model; the caller owns the buffer. */
static int save_model(void** buffer, size_t* size)
{
  libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
  int result = EXIT_FAILURE;
  *buffer = NULL;
  *size = 0;
  if (NULL != model) {
    if (EXIT_SUCCESS == build_model(model)
      && EXIT_SUCCESS == libxs_predict_save(model, NULL, size) && 0 < *size)
    {
      *buffer = malloc(*size);
      if (NULL != *buffer) {
        result = libxs_predict_save(model, *buffer, size);
      }
    }
    libxs_predict_destroy(model);
  }
  return result;
}


/**
 * Leaves the scratch pool holding recognizable bytes, so that the second build
 * meets a recycled buffer where the first met a fresh one. Without it the two
 * builds can see identical scratch and a field that is never initialized reads
 * the same both times, which is the case that hides the defect.
 */
static void dirty_scratch(void)
{
  static const size_t sizes[] = { 4096, 65536, 1048576 };
  void* held[3];
  int i;
  for (i = 0; i < 3; ++i) {
    held[i] = libxs_malloc(NULL, sizes[i], 0);
    if (NULL != held[i]) memset(held[i], DIRT, sizes[i]);
  }
  for (i = 0; i < 3; ++i) libxs_free(held[i]);
}


/**
 * A saved model must depend on the corpus and the settings, and on nothing else.
 * It is checked by BYTES rather than by a round trip: a field that is written
 * but never read survives a round trip unchanged and reports nothing, while an
 * uninitialized one differs here as soon as the allocator hands out a buffer
 * that has been used before.
 */
int main(void)
{
  void* first = NULL;
  void* second = NULL;
  size_t nfirst = 0, nsecond = 0;
  int result = EXIT_SUCCESS;
  if (EXIT_SUCCESS != save_model(&first, &nfirst)) {
    fprintf(stderr, "the model could not be built or saved\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    dirty_scratch();
    if (EXIT_SUCCESS != save_model(&second, &nsecond)) {
      fprintf(stderr, "the model could not be rebuilt or saved\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && nfirst != nsecond) {
    fprintf(stderr, "the same corpus saved %llu bytes and then %llu\n",
      (unsigned long long)nfirst, (unsigned long long)nsecond);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 != memcmp(first, second, nfirst)) {
    const unsigned char* a = (const unsigned char*)first;
    const unsigned char* b = (const unsigned char*)second;
    size_t i, ndiff = 0, at = nfirst;
    for (i = 0; i < nfirst; ++i) {
      if (a[i] != b[i]) {
        if (0 == ndiff) at = i;
        ++ndiff;
      }
    }
    fprintf(stderr, "the same corpus saved differently: %llu of %llu bytes,"
      " first at %llu (%02x vs %02x)\n", (unsigned long long)ndiff,
      (unsigned long long)nfirst, (unsigned long long)at, a[at], b[at]);
    result = EXIT_FAILURE;
  }
  free(second);
  free(first);
  return result;
}
