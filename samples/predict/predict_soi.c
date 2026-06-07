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

enum { WINDOW = 12, HORIZON = 6, NSERIES = 2, NINPUTS = WINDOW * NSERIES };

static int load_noaa_slp(const char* filename, double** values, int* count);
static void evaluate_forecast(const libxs_predict_t* model,
  const double* tahiti, const double* darwin, int total, int train_end,
  int decompose);


int main(int argc, char* argv[])
{
  const char* tahiti_file = (argc > 1) ? argv[1] : NULL;
  const char* darwin_file = (argc > 2) ? argv[2] : NULL;
  const double split = (argc > 3) ? atof(argv[3]) : 0.8;
  int result = EXIT_FAILURE;
  double *tahiti = NULL, *darwin = NULL;
  int ntahiti = 0, ndarwin = 0;
  if (NULL == tahiti_file || NULL == darwin_file) {
    fprintf(stdout,
      "Usage: %s <tahiti_file> <darwin_file> [train_fraction]\n"
      "  SOI prediction from anti-correlated Tahiti/Darwin SLP.\n"
      "  Uses SPREAD decomposition (sum/diff modes).\n"
      "  Input: NOAA CPC fixed-width monthly SLP files.\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
  else if (0 < load_noaa_slp(tahiti_file, &tahiti, &ntahiti)
    && 0 < load_noaa_slp(darwin_file, &darwin, &ndarwin))
  {
    const int total = LIBXS_MIN(ntahiti, ndarwin);
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WINDOW + 1);
    libxs_predict_t* model = libxs_predict_create(NINPUTS, HORIZON);
    fprintf(stdout, "Loaded %d monthly values (Tahiti=%d, Darwin=%d)\n",
      total, ntahiti, ndarwin);
    if (NULL != model) {
      int t;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_series(model, NSERIES, WINDOW);
      libxs_predict_set_target(model, 0);
      libxs_predict_set_decompose(model, LIBXS_PREDICT_SPREAD);
      for (t = 0; t < train_end; ++t) {
        double vals[2];
        vals[0] = tahiti[t];
        vals[1] = darwin[t];
        libxs_predict_push(NULL, model, vals, NULL);
      }
      if (EXIT_SUCCESS == libxs_predict_build(model, 0, 2)) {
        libxs_predict_query_t qi;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        fprintf(stdout, "Window=%d, Horizon=%d, Train=%d, Test=%d\n",
          WINDOW, HORIZON, qi.nentries, total - train_end);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d\n",
          qi.nclusters, qi.compression, qi.order);
        fprintf(stdout, "\n--- SPREAD decomposition (sum/diff modes) ---\n");
        evaluate_forecast(model, tahiti, darwin, total, train_end, 1);
        result = EXIT_SUCCESS;
      }
      libxs_predict_destroy(model);
      { libxs_predict_t* raw_model = libxs_predict_create(NINPUTS, HORIZON);
        if (NULL != raw_model) {
          libxs_predict_set_mode(raw_model, LIBXS_PREDICT_TEMPORAL);
          libxs_predict_set_series(raw_model, NSERIES, WINDOW);
          libxs_predict_set_target(raw_model, 0);
          for (t = 0; t < train_end; ++t) {
            double vals[2];
            vals[0] = tahiti[t];
            vals[1] = darwin[t];
            libxs_predict_push(NULL, raw_model, vals, NULL);
          }
          if (EXIT_SUCCESS == libxs_predict_build(raw_model, 0, 2)) {
            fprintf(stdout, "\n--- RAW concatenation (baseline) ---\n");
            evaluate_forecast(raw_model, tahiti, darwin, total, train_end, 0);
          }
          libxs_predict_destroy(raw_model);
        }
      }
      { libxs_predict_t* solo_model = libxs_predict_create(WINDOW, HORIZON);
        if (NULL != solo_model) {
          libxs_predict_set_mode(solo_model, LIBXS_PREDICT_TEMPORAL);
          libxs_predict_set_series(solo_model, 1, WINDOW);
          for (t = 0; t < train_end; ++t) {
            libxs_predict_push(NULL, solo_model, &tahiti[t], NULL);
          }
          if (EXIT_SUCCESS == libxs_predict_build(solo_model, 0, 2)) {
            fprintf(stdout, "\n--- Tahiti-only (single series) ---\n");
            evaluate_forecast(solo_model, tahiti, darwin, total, train_end, 0);
          }
          libxs_predict_destroy(solo_model);
        }
      }
    }
    free(tahiti);
    free(darwin);
  }
  else {
    fprintf(stderr, "Failed to load SLP data\n");
  }
  return result;
}


static int load_noaa_slp(const char* filename, double** values, int* count)
{
  int result = 0;
  FILE* file = fopen(filename, "r");
  if (NULL != file) {
    char line[256];
    int capacity = 2048;
    double* data = (double*)malloc((size_t)capacity * sizeof(double));
    int n = 0, header_lines = 0;
    if (NULL != data) {
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        if (header_lines < 4) { ++header_lines; continue; }
        { char* p = line;
          char* endptr = NULL;
          int col = 0, valid = 1;
          double row[12];
          double year_val = strtod(p, &endptr);
          if (endptr == p) continue;
          (void)year_val;
          p = endptr;
          for (col = 0; col < 12; ++col) {
            row[col] = strtod(p, &endptr);
            if (endptr == p || row[col] <= -999.0) { valid = 0; break; }
            p = endptr;
          }
          if (0 != valid) {
            for (col = 0; col < 12; ++col) {
              if (n >= capacity) {
                capacity *= 2;
                data = (double*)realloc(data, (size_t)capacity * sizeof(double));
                if (NULL == data) { n = 0; break; }
              }
              if (NULL != data) data[n++] = row[col];
            }
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
  const double* tahiti, const double* darwin, int total, int train_end,
  int decompose)
{
  double sum_err[HORIZON] = { 0 }, max_err[HORIZON] = { 0 };
  double sum_conf = 0;
  int neval = 0, t, h;
  for (t = train_end; t <= total - HORIZON; ++t) {
    double inputs[NINPUTS], outputs[HORIZON];
    libxs_predict_info_t info;
    int i;
    for (i = 0; i < WINDOW; ++i) {
      inputs[i] = tahiti[t - WINDOW + i];
      inputs[WINDOW + i] = darwin[t - WINDOW + i];
    }
    (void)decompose;
    libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
    for (h = 0; h < HORIZON; ++h) {
      const double err = LIBXS_FABS(outputs[h] - tahiti[t + h]);
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
      fprintf(stdout, "  t+%-2d  %8.2f  %8.2f\n",
        h + 1, sum_err[h] / neval, max_err[h]);
    }
    fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
  }
}
