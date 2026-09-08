/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PREDICT_GATE_H
#define PREDICT_GATE_H

#include <libxs/libxs_predict.h>
#include <libxs/libxs_str.h>

/**
 * Precision against coverage over a range of gates, shared by the prediction
 * samples because one gate is not a comparison.
 *
 * A confidence is a ranking as well as a number, and the two are affected
 * differently by how the model forms it: a calibrated confidence and a raw
 * ensemble share can rank identically and still put a fixed gate in a different
 * place, so reading one threshold makes a relabelling look like a change in what
 * the model can tell apart. Comparing two models at one gate has the same
 * defect, and the more so where their scales differ by construction - a vote
 * share reaches 0.9 whenever the trees agree, a probability only where the
 * answer is that likely to be right. What survives both is precision at matched
 * coverage, which is what a sweep lets a reader take off.
 */

/**
 * Gate thresholds from GATE (comma-separated, ascending or not). The first
 * entry drives the single-threshold report, so a one-element list keeps the
 * historical output; more than one additionally traces precision against
 * coverage, which is what separates a better-calibrated signal from a
 * differently-scaled one.
 */
LIBXS_INLINE int gate_list(double gates[], int capacity)
{
  const char* const env = getenv("GATE");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    int len = 0;
    const char* token = libxs_strtoken(env, ",", result, &len);
    while (NULL != token && result < capacity) {
      gates[result++] = atof(token);
      token = libxs_strtoken(env, ",", result, &len);
    }
  }
  if (0 == result) {
    gates[0] = 0.9;
    result = 1;
  }
  return result;
}


LIBXS_INLINE void gate_sweep(const double gates[], int ngates, int n,
  const double lconf[], const char lok[],
  const double xconf[], const char xok[])
{
  int g;
  fprintf(stdout, "Gate sweep (%d queries):\n", n);
  fprintf(stdout, (NULL != xconf)
    ? "  gate  libxs-prec  libxs-cov    xgb-prec    xgb-cov\n"
    : "  gate  libxs-prec  libxs-cov\n");
  for (g = 0; g < ngates; ++g) {
    int lacted = 0, lcorrect = 0, xacted = 0, xcorrect = 0, i;
    for (i = 0; i < n; ++i) {
      if (lconf[i] >= gates[g]) {
        ++lacted;
        if (0 != lok[i]) ++lcorrect;
      }
      if (NULL != xconf && xconf[i] >= gates[g]) {
        ++xacted;
        if (0 != xok[i]) ++xcorrect;
      }
    }
    fprintf(stdout, "  %.2f     %6.1f%%     %6.1f%%", gates[g],
      (0 < lacted) ? 100.0 * lcorrect / lacted : 0.0,
      (0 < n) ? 100.0 * lacted / n : 0.0);
    if (NULL != xconf) {
      fprintf(stdout, "      %6.1f%%     %6.1f%%",
        (0 < xacted) ? 100.0 * xcorrect / xacted : 0.0,
        (0 < n) ? 100.0 * xacted / n : 0.0);
    }
    fprintf(stdout, "\n");
  }
}

#endif /* PREDICT_GATE_H */
