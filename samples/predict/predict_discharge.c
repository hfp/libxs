/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mem.h>

/* Two equivalent constructions of the same engineered model:
 * default (DISCHARGE_USE_API undefined): hand-built feature vector
 *   (window + diffs + day-of-year), transforms and windowing performed
 *   in the sample (fill_inputs).
 * DISCHARGE_USE_API defined: the same features expressed through the
 *   timeseries API (set_series + set_series_deriv + set_series_aux);
 *   the framework transforms the lags, appends the derivatives, and
 *   carries the auxiliary day-of-year feature. Both produce identical
 *   inputs.
 */

enum { WINDOW = 14, HORIZON = 7, NDIFFS = 3, NINPUTS = WINDOW + NDIFFS + 1 };

static int load_discharge(const char* filename, double** values, int* count);
#if !defined(DISCHARGE_USE_API)
static void fill_inputs(const double* series, int t, double* inputs);
#endif
static void evaluate_forecast(const libxs_predict_t* model,
  const double* series, int total, int train_end);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  int decompose = LIBXS_PREDICT_RAW;
  double quality = 0, consistency = 0;
  int argi, result = EXIT_FAILURE;
  double* series = NULL;
  int total = 0;
  for (argi = 3; argi < argc; ++argi) {
    if ('c' == argv[argi][0] && 'o' == argv[argi][1]
      && 'n' == argv[argi][2])
    {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p) ++p;
      consistency = ('\0' != *p) ? atof(p) : 0.9;
    }
    else if ('c' == argv[argi][0]) {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p) ++p;
      quality = ('\0' != *p) ? atof(p) : 0.9;
    }
    else if ('h' == argv[argi][0]) decompose = LIBXS_PREDICT_HKNN;
    else if ('r' == argv[argi][0]) decompose = LIBXS_PREDICT_RF;
  }
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <discharge_file> [train_fraction] [compress[Q]] [hknn|rf]\n"
      "  River discharge forecasting using sliding-window kNN.\n"
      "  Input: USGS NWIS daily discharge (tab-delimited, # comments).\n"
      "  Predicts next %d days from previous %d days + derivatives.\n"
      "  Log-transform applied for heavy-tailed data.\n"
      "  Default train_fraction: 0.8\n", argv[0], HORIZON, WINDOW);
  }
  else if (0 < load_discharge(filename, &series, &total)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WINDOW + 1);
    libxs_predict_t* model = libxs_predict_create(NINPUTS, HORIZON);
    fprintf(stdout, "Loaded %d daily discharge values from %s\n", total, filename);
    fprintf(stdout, "Window=%d (+%d diffs +day-of-year), Horizon=%d, Train=%d, Test=%d\n",
      WINDOW, NDIFFS, HORIZON, train_end - WINDOW, total - train_end);
    if (NULL != model) {
      int t;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_decompose(model, decompose);
      if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
      libxs_predict_set_transform(model, -1, LIBXS_PREDICT_LOG);
#if defined(DISCHARGE_USE_API)
      libxs_predict_set_series(model, 1, WINDOW);
      libxs_predict_set_series_deriv(model, NDIFFS);
      libxs_predict_set_series_aux(model, 1);
      for (t = 0; t < train_end; ++t) {
        double step[2];
        step[0] = series[t];
        step[1] = (double)(t % 365);
        libxs_predict_push(NULL, model, step, NULL);
      }
#else
      double inputs[NINPUTS], outputs[HORIZON];
      for (t = WINDOW; t <= train_end - HORIZON; ++t) {
        int i;
        fill_inputs(series, t, inputs);
        for (i = 0; i < HORIZON; ++i) outputs[i] = series[t + i];
        libxs_predict_push(NULL, model, inputs, outputs);
      }
#endif
      if (EXIT_SUCCESS == libxs_predict_build(model, 0, 2, quality)) {
        libxs_predict_query_t qi;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d\n",
          qi.nclusters, qi.compression, qi.order);
        evaluate_forecast(model, series, total, train_end);
        result = EXIT_SUCCESS;
      }
      libxs_predict_destroy(model);
    }
    free(series);
  }
  else {
    fprintf(stderr, "Failed to load discharge data from %s\n", filename);
  }
  return result;
}


#if !defined(DISCHARGE_USE_API)
static void fill_inputs(const double* series, int t, double* inputs)
{
  int i;
  for (i = 0; i < WINDOW; ++i) {
    inputs[i] = log(series[t - WINDOW + i] + 1.0);
  }
  for (i = 0; i < NDIFFS; ++i) {
    inputs[WINDOW + i] = log(series[t - 1 - i] + 1.0) - log(series[t - 2 - i] + 1.0);
  }
  inputs[WINDOW + NDIFFS] = (double)(t % 365);
}
#endif


static int load_discharge(const char* filename, double** values, int* count)
{
  int result = 0;
  FILE* file = fopen(filename, "r");
  if (NULL != file) {
    char line[512];
    int capacity = 16384;
    double* data = (double*)malloc((size_t)capacity * sizeof(double));
    int n = 0;
    if (NULL != data) {
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        char* endptr = NULL;
        const char* p = line;
        double val;
        int col = 0;
        if ('#' == line[0] || '\n' == line[0] || '\r' == line[0]) continue;
        while ('\0' != *p && col < 3) {
          while ('\0' != *p && '\t' != *p && '\n' != *p) ++p;
          if ('\t' == *p) ++p;
          ++col;
        }
        val = strtod(p, &endptr);
        if (endptr != p && val >= 0.0) {
          if (n >= capacity) {
            capacity *= 2;
            data = (double*)realloc(data, (size_t)capacity * sizeof(double));
            if (NULL == data) { n = 0; break; }
          }
          data[n++] = val;
        }
      }
      *values = data;
      *count = n;
      result = n;
    }
    fclose(file);
  }
  return result;
}


static void evaluate_forecast(const libxs_predict_t* model,
  const double* series, int total, int train_end)
{
  double sum_err[HORIZON] = { 0 }, max_err[HORIZON] = { 0 };
  double sum_conf = 0;
  int neval = 0, t, h;
  for (t = train_end; t <= total - HORIZON; ++t) {
    double inputs[NINPUTS], outputs[HORIZON];
    libxs_predict_info_t info;
#if defined(DISCHARGE_USE_API)
    int i;
    for (i = 0; i < WINDOW; ++i) inputs[i] = series[t - WINDOW + i];
    inputs[WINDOW] = (double)(t % 365);
#else
    fill_inputs(series, t, inputs);
#endif
    libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
    for (h = 0; h < HORIZON; ++h) {
      const double err = LIBXS_FABS(outputs[h] - series[t + h]);
      sum_err[h] += err;
      if (err > max_err[h]) max_err[h] = err;
    }
    sum_conf += info.confidence[0];
    ++neval;
  }
  if (0 < neval) {
    fprintf(stdout, "Forecast quality (%d test windows):\n", neval);
    fprintf(stdout, "  step   avg-err   max-err\n");
    for (h = 0; h < HORIZON; ++h) {
      fprintf(stdout, "  t+%-2d  %8.1f  %8.1f\n",
        h + 1, sum_err[h] / neval, max_err[h]);
    }
    fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
  }
}
