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

#define NENTRY 900
#define NINPUT 4
#define NOUTPUT 3


/**
 * Two labels of the inputs and one continuous function of them.  What is
 * asserted is that a requested count reaches the vote and survives a round
 * trip, not which count wins: that is a measurement and belongs to the corpus.
 */
static void fill(double inputs[], double outputs[], int i)
{
  unsigned int s = (unsigned int)(i * 2654435761u);
  int j;
  for (j = 0; j < NINPUT; ++j) {
    s = s * 1103515245u + 12345u;
    inputs[j] = (double)((s >> 16) & 0x3f);
  }
  outputs[0] = (double)((int)(inputs[0] + inputs[1]) % 5);
  outputs[1] = (double)((int)inputs[2] % 3);
  outputs[2] = inputs[0] * 0.5 + inputs[3];
}


static int build_model(libxs_predict_t* model)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double inputs[NINPUT], outputs[NOUTPUT];
    fill(inputs, outputs, i);
    result = libxs_predict_push(NULL, model, inputs, outputs);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, 0, 2, 0.0);
  }
  return result;
}


/** Non-zero if the two models answer the same queries the same way. */
static int agree(const libxs_predict_t* a, const libxs_predict_t* b)
{
  int i, j, result = 1;
  for (i = 0; i < 64; ++i) {
    double inputs[NINPUT], outputs[NOUTPUT];
    double pa[NOUTPUT], pb[NOUTPUT];
    fill(inputs, outputs, i * 7 + 3);
    libxs_predict_eval(NULL, a, inputs, pa, NULL, 1);
    libxs_predict_eval(NULL, b, inputs, pb, NULL, 1);
    for (j = 0; j < NOUTPUT; ++j) {
      if (pa[j] != pb[j]) result = 0;
    }
  }
  return result;
}


/** A saved model reloaded, or NULL. */
static libxs_predict_t* reload(const libxs_predict_t* model)
{
  libxs_predict_t* result = NULL;
  size_t size = 0;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && 0 < size) {
    unsigned char* buffer = (unsigned char*)malloc(size);
    if (NULL != buffer) {
      size_t written = size;
      if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &written)) {
        result = libxs_predict_load(buffer, written);
      }
      free(buffer);
    }
  }
  return result;
}


int main(void)
{
  libxs_predict_t* derived = libxs_predict_create(NINPUT, NOUTPUT);
  libxs_predict_t* pinned = libxs_predict_create(NINPUT, NOUTPUT);
  libxs_predict_t* selected = libxs_predict_create(NINPUT, NOUTPUT);
  int result = (NULL != derived && NULL != pinned && NULL != selected)
    ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == result) result = build_model(derived);
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_neighbors(pinned, 1);
    result = build_model(pinned);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_neighbors(selected, -1);
    result = build_model(selected);
  }
  /* a pinned count changes what the vote answers, or it never arrived */
  if (EXIT_SUCCESS == result && 0 != agree(derived, pinned)) {
    fprintf(stderr, "pinning the neighbour count changed nothing\n");
    result = EXIT_FAILURE;
  }
  /* every model answers the same after a round trip, counts included */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* a = reload(derived);
    libxs_predict_t* b = reload(pinned);
    libxs_predict_t* c = reload(selected);
    if (NULL == a || NULL == b || NULL == c) {
      fprintf(stderr, "a model with neighbour counts failed to reload\n");
      result = EXIT_FAILURE;
    }
    else if (0 == agree(derived, a) || 0 == agree(pinned, b)
      || 0 == agree(selected, c))
    {
      fprintf(stderr, "a reloaded model votes differently\n");
      result = EXIT_FAILURE;
    }
    libxs_predict_destroy(c);
    libxs_predict_destroy(b);
    libxs_predict_destroy(a);
  }
  /* a request larger than the cluster is clamped rather than refused */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* huge = libxs_predict_create(NINPUT, NOUTPUT);
    if (NULL != huge) {
      libxs_predict_set_neighbors(huge, 100000);
      if (EXIT_SUCCESS != build_model(huge)) {
        fprintf(stderr, "an oversized neighbour count was refused\n");
        result = EXIT_FAILURE;
      }
      libxs_predict_destroy(huge);
    }
    else result = EXIT_FAILURE;
  }
  libxs_predict_destroy(selected);
  libxs_predict_destroy(pinned);
  libxs_predict_destroy(derived);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
