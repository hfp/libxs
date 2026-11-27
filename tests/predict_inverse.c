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

#define NENTRY 200


/* entry i: inputs (i % 10, i / 10), outputs (a class, a value) */
static void entry(int i, double x[2], double y[2])
{
  x[0] = (double)(i % 10);
  x[1] = (double)(i / 10);
  y[0] = (double)((i % 10 + i / 10) % 3);
  y[1] = 1.5 * x[0] + 0.25 * x[1] * x[1];
}


/* the target of a stored entry finds that entry, at distance zero */
static int check(int decompose, const char* what)
{
  libxs_predict_t* model = libxs_predict_create(2, 2);
  int result = (NULL != model) ? EXIT_SUCCESS : EXIT_FAILURE, i;
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(model, decompose);
    for (i = 0; i < NENTRY; ++i) {
      double x[2], y[2];
      entry(i, x, y);
      libxs_predict_push(NULL, model, x, y);
    }
    result = libxs_predict_build(model, 0, 1, 0);
  }
  for (i = 0; i < NENTRY && EXIT_SUCCESS == result; i += 37) {
    double x[2], y[2], found[2];
    libxs_predict_info_t info;
    memset(&info, 0, sizeof(info));
    entry(i, x, y);
    libxs_predict_inverse(NULL, model, y, found, &info);
    if (0 != info.distance || x[0] != found[0] || x[1] != found[1]) {
      fprintf(stderr, "%s: entry %i found (%g,%g) at %g, expected (%g,%g)\n",
        what, i, found[0], found[1], info.distance, x[0], x[1]);
      result = EXIT_FAILURE;
    }
  }
  libxs_predict_destroy(model);
  return result;
}


int main(void)
{
  int result = check(LIBXS_PREDICT_RAW, "kNN");
  /* a forest whose method is named keeps no clusters (it crashed here) */
  if (EXIT_SUCCESS == result) result = check(LIBXS_PREDICT_RF, "forest");
  return result;
}
