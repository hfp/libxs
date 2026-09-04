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

#define NENTRY 240


/**
 * A deliberately unbalanced corpus: at every input the majority label outnumbers
 * the minority three to one, so an unweighted vote returns the majority
 * everywhere and a weight above three has to overturn it.
 */
static void fill(double* input, double* out, int i, int* minority)
{
  *input = (double)(i / 4);
  *minority = (0 == (i % 4)) ? 1 : 0;
  *out = (0 != *minority) ? 1.0 : 0.0;
}


static int build_model(libxs_predict_t* model, double minority_weight)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; ++i) {
    double input, out;
    int minority;
    fill(&input, &out, i, &minority);
    if (0 != minority) {
      result = libxs_predict_push_weighted(NULL, model, &input, &out,
        minority_weight);
    }
    else {
      result = libxs_predict_push(NULL, model, &input, &out);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, 0, 1, 0.0);
  }
  return result;
}


/**
 * Fraction of queries answered with the minority label.  Queried between stored
 * inputs on purpose: a query that coincides with a stored point is answered by
 * the exact-match shortcut, which never reaches the vote the weights act on.
 */
static double minority_rate(const libxs_predict_t* model)
{
  int i, n = 0, hit = 0;
  for (i = 0; i < NENTRY; i += 4) {
    const double input = (double)(i / 4) + 0.25;
    double predicted = 0;
    libxs_predict_eval(NULL, model, &input, &predicted, NULL, 1);
    if (0.5 < predicted) ++hit;
    ++n;
  }
  return (0 < n) ? ((double)hit / n) : 0.0;
}


int main(void)
{
  libxs_predict_t* plain = libxs_predict_create(1, 1);
  libxs_predict_t* heavy = libxs_predict_create(1, 1);
  int result = (NULL != plain && NULL != heavy) ? EXIT_SUCCESS : EXIT_FAILURE;
  double rate_plain = 0, rate_heavy = 0;
  if (EXIT_SUCCESS == result) result = build_model(plain, 1.0);
  if (EXIT_SUCCESS == result) result = build_model(heavy, 12.0);
  if (EXIT_SUCCESS == result) {
    rate_plain = minority_rate(plain);
    rate_heavy = minority_rate(heavy);
    if (!(rate_heavy > rate_plain)) {
      fprintf(stderr, "weighting did not raise minority recall:"
        " plain %.3f, weighted %.3f\n", rate_plain, rate_heavy);
      result = EXIT_FAILURE;
    }
  }
  /* the weights must survive a round trip, or a loaded model votes differently */
  if (EXIT_SUCCESS == result) {
    size_t size = 0;
    void* buffer;
    libxs_predict_save(heavy, NULL, &size);
    buffer = malloc(size);
    if (NULL != buffer) {
      if (EXIT_SUCCESS == libxs_predict_save(heavy, buffer, &size)) {
        libxs_predict_t* loaded = libxs_predict_load(buffer, size);
        if (NULL != loaded) {
          const double rate = minority_rate(loaded);
          if (rate != rate_heavy) {
            fprintf(stderr, "round trip changed the vote: %.3f vs %.3f\n",
              rate, rate_heavy);
            result = EXIT_FAILURE;
          }
          libxs_predict_destroy(loaded);
        }
        else {
          fprintf(stderr, "a weighted model failed to load\n");
          result = EXIT_FAILURE;
        }
      }
      else result = EXIT_FAILURE;
      free(buffer);
    }
    else result = EXIT_FAILURE;
  }
  /* a weight is rejected rather than reinterpreted where it cannot apply */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* series = libxs_predict_create(4, 2);
    if (NULL != series) {
      const double step = 1.0;
      libxs_predict_set_series(series, 1, 4);
      if (EXIT_SUCCESS == libxs_predict_push_weighted(NULL, series, &step,
        NULL, 3.0))
      {
        fprintf(stderr, "series mode accepted a weighted timestep\n");
        result = EXIT_FAILURE;
      }
      libxs_predict_destroy(series);
    }
  }
  if (EXIT_SUCCESS == result) {
    double input = 0, out = 0;
    if (EXIT_SUCCESS == libxs_predict_push_weighted(NULL, plain, &input,
      &out, 0.0))
    {
      fprintf(stderr, "a zero weight was accepted\n");
      result = EXIT_FAILURE;
    }
  }
  libxs_predict_destroy(heavy);
  libxs_predict_destroy(plain);
  if (EXIT_SUCCESS == result) {
    fprintf(stdout, "OK (minority rate %.3f -> %.3f)\n",
      rate_plain, rate_heavy);
  }
  return result;
}
