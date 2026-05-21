/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_math.h>
#include <libxs_mem.h>

enum { WINDOW = 12, HORIZON = 6, NINPUTS = WINDOW };

static int load_sunspots(const char* filename, double** values, int* count);
static void evaluate_forecast(const libxs_predict_t* model,
  const double* series, int total, int train_end);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  int result = EXIT_FAILURE;
  double* series = NULL;
  int total = 0;
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <sunspot_csv> [train_fraction]\n"
      "  Timeseries prediction using sliding-window kNN.\n"
      "  Input: SILSO monthly sunspot CSV (semicolon-delimited).\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
  else if (0 < load_sunspots(filename, &series, &total)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WINDOW + 1);
    libxs_predict_t* model = libxs_predict_create(NINPUTS, HORIZON);
    fprintf(stdout, "Loaded %d monthly sunspot values from %s\n", total, filename);
    if (NULL != model) {
      int t;
      libxs_predict_set_mode(model, LIBXS_PREDICT_EXTRAPOLATE);
      libxs_predict_set_series(model, 1, WINDOW);
      for (t = 0; t < train_end; ++t) {
        libxs_predict_push(NULL, model, &series[t], NULL);
      }
      if (EXIT_SUCCESS == libxs_predict_build(model, 0, 2)) {
        libxs_predict_query_t qi;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        fprintf(stdout, "Window=%d, Horizon=%d, Train=%d, Test=%d\n",
          WINDOW, HORIZON, qi.nentries, total - train_end);
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
    fprintf(stderr, "Failed to load sunspot data from %s\n", filename);
  }
  return result;
}


static int load_sunspots(const char* filename, double** values, int* count)
{
  int result = 0;
  FILE* file = fopen(filename, "r");
  if (NULL != file) {
    char line[256];
    int capacity = 4096;
    double* data = (double*)malloc((size_t)capacity * sizeof(double));
    int n = 0;
    if (NULL != data) {
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        double val;
        int year, month;
        if (3 == sscanf(line, "%d;%d;%*f;%lf", &year, &month, &val)) {
          if (0 <= val) {
            if (n >= capacity) {
              capacity *= 2;
              data = (double*)realloc(data, (size_t)capacity * sizeof(double));
              if (NULL == data) { n = 0; break; }
            }
            data[n++] = val;
          }
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
  double sum_err[HORIZON] = {0}, max_err[HORIZON] = {0};
  double sum_err2[HORIZON] = {0}, max_err2[HORIZON] = {0};
  double sum_conf = 0;
  int neval = 0, t, h;
  for (t = train_end; t <= total - HORIZON; ++t) {
    double inputs[NINPUTS], outputs[HORIZON], refined[HORIZON];
    double canon[NINPUTS];
    libxs_predict_info_t info;
    int i;
    for (i = 0; i < WINDOW; ++i) inputs[i] = series[t - WINDOW + i];
    libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
    libxs_predict_inverse(NULL, model, outputs, canon, NULL);
    libxs_predict_eval(NULL, model, canon, refined, NULL, 1);
    for (h = 0; h < HORIZON; ++h) {
      double err = LIBXS_FABS(outputs[h] - series[t + h]);
      double err2 = LIBXS_FABS(refined[h] - series[t + h]);
      sum_err[h] += err;
      sum_err2[h] += err2;
      if (err > max_err[h]) max_err[h] = err;
      if (err2 > max_err2[h]) max_err2[h] = err2;
    }
    sum_conf += info.confidence[0];
    ++neval;
  }
  if (0 < neval) {
    fprintf(stdout, "Forecast quality (%d test windows):\n", neval);
    fprintf(stdout, "  step   avg-err   max-err  avg-ref   max-ref\n");
    for (h = 0; h < HORIZON; ++h) {
      fprintf(stdout, "  t+%-2d  %8.2f  %8.2f  %8.2f  %8.2f\n",
        h + 1, sum_err[h] / neval, max_err[h],
        sum_err2[h] / neval, max_err2[h]);
    }
    fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
  }
}
