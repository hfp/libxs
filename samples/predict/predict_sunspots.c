/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_timer.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

enum { WINDOW_DEF = 12, HORIZON = 6, WMAX = 160 };

static int load_sunspots(const char* filename, double** values, int* count);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  const char* wenv = getenv("WINDOW");
  const int window_req = (NULL != wenv) ? atoi(wenv) : LIBXS_PREDICT_AUTO_WINDOW;
  const int ninputs = (0 < window_req) ? window_req : WMAX;
  int window = window_req;
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
      "Usage: %s <sunspot_csv> [train_fraction] [compress[Q]] [hknn|rf]\n"
      "  Timeseries prediction using sliding-window kNN.\n"
      "  Input: SILSO monthly sunspot CSV (semicolon-delimited).\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
  else if (0 < load_sunspots(filename, &series, &total)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WMAX + 1);
    libxs_predict_t* model = libxs_predict_create(ninputs, HORIZON);
    fprintf(stdout, "Loaded %d monthly sunspot values from %s\n", total, filename);
    if (NULL != model) {
      libxs_timer_tick_t tick;
      double dt_build, dt_eval;
      int t, build_ok = EXIT_FAILURE;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_decompose(model, decompose);
      libxs_predict_set_series(model, 1, window_req);
      if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
      for (t = 0; t < train_end; ++t) {
        libxs_predict_push(NULL, model, &series[t], NULL);
      }
      tick = libxs_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel
      { build_ok = libxs_predict_build_task(NULL, model, 0, 2,
          quality, omp_get_thread_num(), omp_get_num_threads());
      }
#else
      build_ok = libxs_predict_build(model, 0, 2, quality);
#endif
      dt_build = libxs_timer_duration(tick, libxs_timer_tick());
      if (EXIT_SUCCESS == build_ok) {
        libxs_predict_query_t qi;
        double sum_err[HORIZON] = { 0 }, max_err[HORIZON] = { 0 };
        double sum_conf = 0;
        int neval = 0, h;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        window = qi.window;
        fprintf(stdout, "Window=%d, Horizon=%d, Train=%d, Test=%d\n",
          window, HORIZON, qi.nentries, total - train_end);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
          " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
        tick = libxs_timer_tick();
        for (t = train_end; t <= total - HORIZON; ++t) {
          double inputs[WMAX], outputs[HORIZON];
          libxs_predict_info_t info;
          int i;
          for (i = 0; i < window; ++i) inputs[i] = series[t - window + i];
          libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
          for (h = 0; h < HORIZON; ++h) {
            const double err = LIBXS_FABS(outputs[h] - series[t + h]);
            sum_err[h] += err;
            if (err > max_err[h]) max_err[h] = err;
          }
          sum_conf += info.confidence[0];
          ++neval;
        }
        dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
        if (0 < neval) {
          fprintf(stdout, "Forecast quality (%d test windows):\n", neval);
          fprintf(stdout, "  step   avg-err   max-err\n");
          for (h = 0; h < HORIZON; ++h) {
            fprintf(stdout, "  t+%-2d  %8.2f  %8.2f\n",
              h + 1, sum_err[h] / neval, max_err[h]);
          }
          fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
          fprintf(stdout, "Eval: %d queries (%.2f s)\n", neval, dt_eval);
        }
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
