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
#include <stdlib.h>
#include <stdio.h>

#define NTRAIN 2000
#define NTEST 200
#define NCLUSTERS 8
#define NCLASS 3
#define NFEAT 4


/**
 * The same corpus the calibration test fits: the label follows two of the
 * inputs and every seventh entry carries a different one, so a query's
 * neighborhood disagrees often enough for the confidence to fall under the
 * blending threshold. A fixture where nothing blends would pass every
 * assertion below without exercising anything, which is why the count of
 * blended queries is itself checked.
 */
static void fill(double input[], double* out, int i)
{
  input[0] = (double)(i % 97);
  input[1] = (double)((i * 7) % 31);
  input[2] = (double)((i * 13) % 11);
  input[3] = (double)(i % 5);
  *out = (double)(((input[0] > 48.0) ? 1 : 0) + ((input[1] > 15.0) ? 1 : 0));
  if (0 == (i % 7)) *out = (double)((LIBXS_ROUNDX(int, *out) + 1) % NCLASS);
}


/* returns the cluster that served the query, -1 if the answer was blended */
static int eval1(const libxs_predict_t* model, const double input[],
  int nblend, double* out)
{
  libxs_predict_info_t info;
  double val = 0;
  libxs_predict_eval(NULL, model, input, &val, &info, nblend);
  if (NULL != out) *out = val;
  return info.cluster;
}


int main(void)
{
  libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
  int result = EXIT_SUCCESS, nblended = 0, npinned = 0, i;
  for (i = 0; i < NTRAIN && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], out;
    fill(input, &out, i);
    result = libxs_predict_push(NULL, model, input, &out);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, NCLUSTERS, 1, 0.0);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "the model could not be built\n");
    }
  }
  /**
   * nblend selects one of three regimes: a count of two or more averages that
   * many clusters whatever the confidence says, a negative count answers from
   * the nearest cluster alone, and 0 and 1 are one setting - the adaptive one -
   * because the gate that escalates the count sits on the path both take.
   */
  for (i = 0; i < NTEST && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], ref, pin = 0, aut = 0, one = 0;
    int c_pin, c_aut, c_one, c_fix;
    fill(input, &ref, NTRAIN + i * 3 + 1);
    c_pin = eval1(model, input, -1, &pin);
    c_aut = eval1(model, input, 0, &aut);
    c_one = eval1(model, input, 1, &one);
    c_fix = eval1(model, input, NCLUSTERS / 2, NULL);
    if (0 > c_pin) {
      fprintf(stderr, "query %i blended although nblend pins it\n", i);
      result = EXIT_FAILURE;
    }
    else if (aut != one || c_aut != c_one) {
      fprintf(stderr, "query %i reads nblend 0 and 1 differently\n", i);
      result = EXIT_FAILURE;
    }
    else if (0 <= c_fix) {
      fprintf(stderr, "query %i did not blend at a fixed count\n", i);
      result = EXIT_FAILURE;
    }
    else if (0 > c_aut) {
      ++nblended;
      if (pin != aut) ++npinned;
    }
    else if (pin != aut || c_pin != c_aut) {
      fprintf(stderr, "query %i was not blended, yet pinning moved it\n", i);
      result = EXIT_FAILURE;
    }
  }
  /* an unused knob would pass the loop above: the fixture has to blend, and
   * pinning has to reach the answer rather than the reported cluster alone */
  if (EXIT_SUCCESS == result && (0 == nblended || 0 == npinned)) {
    fprintf(stderr, "%i of %i queries blended, %i moved when pinned\n",
      nblended, NTEST, npinned);
    result = EXIT_FAILURE;
  }
  libxs_predict_destroy(model);
  return result;
}
