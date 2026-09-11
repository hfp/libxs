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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define NTRAIN 900
#define NCALIB 400
#define NTEST 400
#define NFEAT 4
#define NCLASS 3


/**
 * A corpus with genuine uncertainty in it: the label follows two of the inputs,
 * and every seventh entry carries a different one. Without the noise the trees
 * agree everywhere, every query reports the same confidence, and a curve fitted
 * to one value says nothing about whether the mapping works.
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


int main(void)
{
  libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
  double* cin = (double*)malloc((size_t)NCALIB * NFEAT * sizeof(double));
  double* cout = (double*)malloc((size_t)NCALIB * sizeof(double));
  int result = EXIT_SUCCESS, i;
  if (NULL == model || NULL == cin || NULL == cout) result = EXIT_FAILURE;
  for (i = 0; i < NTRAIN && EXIT_SUCCESS == result; ++i) {
    double input[NFEAT], out;
    fill(input, &out, i);
    result = libxs_predict_push(NULL, model, input, &out);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(model, LIBXS_PREDICT_RF);
    result = libxs_predict_build(model, 0, 1, 0.0);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "the model could not be built\n");
    }
  }
  for (i = 0; i < NCALIB && EXIT_SUCCESS == result; ++i) {
    fill(cin + (size_t)i * NFEAT, cout + i, NTRAIN + i);
  }
  /* an uncalibrated model has to SAY it is uncalibrated, and hand the value
     back unchanged rather than quietly report a ranking as a probability */
  if (EXIT_SUCCESS == result) {
    double p = -1.0;
    if (EXIT_SUCCESS == libxs_predict_probability(model, 0, 0.75, &p)) {
      fprintf(stderr, "an uncalibrated model claimed a probability\n");
      result = EXIT_FAILURE;
    }
    else if (0.75 != p) {
      fprintf(stderr, "the confidence was altered without a curve: %f\n", p);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_calibrate(model, cin, cout, NCALIB);
    if (EXIT_SUCCESS != result) fprintf(stderr, "the curve was not fitted\n");
  }
  if (EXIT_SUCCESS == result) {
    double p = -1.0;
    if (EXIT_SUCCESS != libxs_predict_probability(model, 0, 0.75, &p)) {
      fprintf(stderr, "a calibrated model reported no probability\n");
      result = EXIT_FAILURE;
    }
  }
  /**
   * Monotone, which is the property the mapping is safe because of: it reorders
   * no query, so a coverage stays the coverage it was and precision at matched
   * coverage is untouched. A curve that dipped would silently move queries past
   * each other and make a gate mean something other than it did.
   */
  if (EXIT_SUCCESS == result) {
    double prev = -1.0;
    for (i = 0; i <= 100; ++i) {
      double p = 0;
      libxs_predict_probability(model, 0, 0.01 * i, &p);
      if (p < prev) {
        fprintf(stderr, "the curve falls at %.2f: %f after %f\n",
          0.01 * i, p, prev);
        result = EXIT_FAILURE;
        break;
      }
      prev = p;
    }
  }
  /**
   * And it has to be honest on the rows it was fitted from: the mean probability
   * reported over them is what the accuracy over them actually was. This is the
   * whole claim - that a threshold means a rate - and an isotonic fit satisfies
   * it on its own data by construction, so a failure here is a wiring fault
   * rather than a matter of sample size.
   */
  if (EXIT_SUCCESS == result) {
    double psum = 0;
    int hit = 0;
    for (i = 0; i < NCALIB; ++i) {
      libxs_predict_info_t info;
      double p = 0;
      libxs_predict_eval(NULL, model, cin + (size_t)i * NFEAT, NULL, &info, 1);
      if (NULL == info.confidence || NULL == info.values) continue;
      libxs_predict_probability(model, 0, info.confidence[0], &p);
      psum += p;
      if (LIBXS_ROUNDX(int, info.values[0]) == LIBXS_ROUNDX(int, cout[i])) {
        ++hit;
      }
    }
    { const double mean = psum / NCALIB;
      const double rate = (double)hit / NCALIB;
      if (0.05 < LIBXS_DELTA(mean, rate)) {
        fprintf(stderr, "the curve promises %.3f and delivers %.3f\n",
          mean, rate);
        result = EXIT_FAILURE;
      }
    }
  }
  free(cout);
  free(cin);
  libxs_predict_destroy(model);
  return result;
}
