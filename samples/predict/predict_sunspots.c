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
#include <libxs/libxs_timer.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

enum { WINDOW_DEF = 12, HORIZON = 6, WMAX = 160, PMAX = 400, NBANK = 2 };

static int load_sunspots(const char* filename, double** values, int* count);
static int cycle_period(const double* series, int n);
static int cycle_phase(const double* series, int n, int period,
  double* phase);


static const char* mode_name(int decompose)
{
  static const char* names[] = { "RAW", "SPREAD", "PCA", "SETDIFF", "FISHER",
    "RF", "hKNN" };
  return (0 <= decompose && 7 > decompose) ? names[decompose] : "?";
}


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  const char* wenv = getenv("WINDOW");
  const int window_req = (NULL != wenv) ? atoi(wenv) : LIBXS_PREDICT_AUTO_WINDOW;
  const int ninputs = (0 < window_req) ? window_req : WMAX;
  int window = window_req;
  int decompose = LIBXS_PREDICT_AUTO_DECOMPOSE;
  const char* penv = getenv("NOPHASE");
  const int nophase = (NULL != penv) ? atoi(penv) : 0;
  const char* benv = getenv("NOBANK");
  const int nobank = (NULL != benv) ? atoi(benv) : 0;
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
    else if ('n' == argv[argi][0]) decompose = LIBXS_PREDICT_RAW;
  }
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <sunspot_csv> [train_fraction] [compress[Q]] [hknn|rf]\n"
      "  Timeseries prediction using sliding-window kNN.\n"
      "  Input: SILSO monthly sunspot CSV (semicolon-delimited).\n"
      "  Default train_fraction: 0.8\n"
      "  NOPHASE=1 drops the solar-cycle phase input.\n"
      "  NOBANK=1 uses a single window instead of the bank.\n", argv[0]);
  }
  else if (0 < load_sunspots(filename, &series, &total)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WMAX + 1);
    const int period = (0 == nophase) ? cycle_period(series, train_end) : 0;
    double* phase = (double*)malloc((size_t)total * sizeof(double));
    const int naux = (0 < period && NULL != phase
      && EXIT_SUCCESS == cycle_phase(series, total, period, phase)) ? 1 : 0;
    libxs_predict_t* model = libxs_predict_create(ninputs + naux, HORIZON);
    fprintf(stdout, "Loaded %d monthly sunspot values from %s\n", total, filename);
    if (0 != naux) {
      fprintf(stdout, "Cycle period %d months (%.1f years) from training"
        " autocorrelation; phase carried as one auxiliary input\n",
        period, period / 12.0);
    }
    if (NULL != model) {
      libxs_timer_tick_t tick;
      double dt_build, dt_eval;
      int t, build_ok = EXIT_FAILURE;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_decompose(model, decompose);
      libxs_predict_set_series(model, 1, window_req);
      if (0 != naux) libxs_predict_set_series_aux(model, naux);
      libxs_predict_set_series_bank(model, (0 != nobank) ? 1 : NBANK);
      if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
      for (t = 0; t < train_end; ++t) {
        double step[2];
        step[0] = series[t];
        step[1] = (0 != naux) ? phase[t] : 0.0;
        libxs_predict_push(NULL, model, step, NULL);
      }
      tick = libxs_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel
      { const int br = libxs_predict_build_task(model, 0, 2,
          quality, omp_get_thread_num(), omp_get_num_threads());
        if (0 == omp_get_thread_num()) build_ok = br;
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
        fprintf(stdout, "Decomposition: %s (%s)\n",
          mode_name(qi.decompose),
          (LIBXS_PREDICT_AUTO_DECOMPOSE == decompose)
            ? "selected at build" : "requested");
        window = qi.window;
        fprintf(stdout, "Window=%d%s, Horizon=%d, Train=%d, Test=%d\n",
          window, (0 != naux) ? " (+cycle phase)" : "", HORIZON,
          qi.nentries, total - train_end);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
          " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
        if (1 < qi.nbank) {
          fprintf(stdout, "Window bank: %d views\n", qi.nbank);
        }
        tick = libxs_timer_tick();
        for (t = train_end; t <= total - HORIZON; ++t) {
          double inputs[WMAX + 1], outputs[HORIZON];
          libxs_predict_info_t info;
          int i;
          for (i = 0; i < window; ++i) inputs[i] = series[t - window + i];
          if (0 != naux) inputs[window] = phase[t - 1];
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
    free(phase);
    free(series);
  }
  else {
    fprintf(stderr, "Failed to load sunspot data from %s\n", filename);
  }
  return result;
}


/**
 * Dominant period of the series in samples, from the first local maximum of
 * the autocorrelation after its initial decay.  Measured on the training
 * slice only, so the phase feature it parameterizes carries no knowledge of
 * the held-out tail.  Returns 0 when no periodicity is found.
 */
static int cycle_period(const double* series, int n)
{
  int result = 0;
  if (2 < n) {
    double mean = 0, denom = 0;
    int i, lag, rising = 0;
    double best = 0;
    for (i = 0; i < n; ++i) mean += series[i];
    mean /= n;
    for (i = 0; i < n; ++i) {
      const double d = series[i] - mean;
      denom += d * d;
    }
    if (0 < denom) {
      const int lmax = LIBXS_MIN(PMAX, n / 2);
      double prev = 1.0;
      for (lag = 1; lag < lmax; ++lag) {
        double acc = 0;
        for (i = 0; i + lag < n; ++i) {
          acc += (series[i] - mean) * (series[i + lag] - mean);
        }
        acc /= denom;
        if (0 == rising) {
          if (acc > prev) rising = 1;
        }
        else if (acc > best) {
          best = acc;
          result = lag;
        }
        else if (0 < result) {
          lag = lmax;
        }
        prev = acc;
      }
    }
  }
  return result;
}


/**
 * Months since the most recent minimum of the smoothed series, evaluated
 * causally: every value reads only samples at or before its own index, so the
 * feature is available at eval time and cannot leak the future.  Two windows
 * are derived from the period rather than tuned - a tenth of a cycle
 * smooths the monthly noise, half a cycle bounds the search for the minimum.
 */
static int cycle_phase(const double* series, int n, int period, double* phase)
{
  int result = EXIT_FAILURE;
  double* smooth = (double*)malloc((size_t)n * sizeof(double));
  if (NULL != smooth && 0 < period) {
    const int nsmooth = LIBXS_MAX(3, period / 10);
    const int look = LIBXS_MAX(nsmooth + 1, period / 2);
    int i;
    for (i = 0; i < n; ++i) {
      const int lo = LIBXS_MAX(0, i - nsmooth + 1);
      double acc = 0;
      int k;
      for (k = lo; k <= i; ++k) acc += series[k];
      smooth[i] = acc / (i - lo + 1);
    }
    for (i = 0; i < n; ++i) {
      const int lo = LIBXS_MAX(0, i - look);
      int argmin = lo, k;
      for (k = lo; k <= i; ++k) {
        if (smooth[k] < smooth[argmin]) argmin = k;
      }
      phase[i] = (double)(i - argmin);
    }
    result = EXIT_SUCCESS;
  }
  free(smooth);
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
