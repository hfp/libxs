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
#include <libxs/libxs_perm.h>
#include <libxs/libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

enum { WINDOW = 96, HORIZON = 96 };


static int load_ett_ot(const char* filename, double** values, int* count)
{
  int result = 0;
  FILE* file = fopen(filename, "r");
  if (NULL != file) {
    char line[512];
    int capacity = 20000;
    double* data = (double*)malloc((size_t)capacity * sizeof(double));
    int n = 0;
    if (NULL != data) {
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        const char* p = line;
        double val;
        int col = 0;
        if (line[0] < '0' || line[0] > '9') {
          if (0 == n) continue;
        }
        while ('\0' != *p && col < 7) {
          if (',' == *p) ++col;
          ++p;
        }
        if (col == 7 && 1 == sscanf(p, "%lf", &val)) {
          if (n >= capacity) {
            capacity *= 2;
            data = (double*)realloc(data, (size_t)capacity * sizeof(double));
            if (NULL == data) { n = 0; break; }
          }
          data[n++] = val;
        }
      }
      if (n > 0) {
        *values = data;
        *count = n;
        result = n;
      }
      else {
        free(data);
      }
    }
    fclose(file);
  }
  return result;
}


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.661;
  int decompose = LIBXS_PREDICT_RAW;
  int stride = HORIZON;
  double quality = 0, consistency = 0, quantile = 0;
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
    else if ('q' == argv[argi][0]) {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p) ++p;
      quantile = ('\0' != *p) ? atof(p) : 0.1;
    }
    else if ('o' == argv[argi][0]) stride = 1;
  }
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <ett_csv> [train_fraction] [compress[Q]] [consist[C]]"
      " [hknn|rf] [quantile[Q]] [overlap]\n"
      "  ETT (Electricity Transformer Temperature) forecasting.\n"
      "  Input: ETTh1.csv with OT column (last column).\n"
      "  Window=%d, Horizon=%d, Stride=%d (use 'overlap' for stride=1).\n"
      "  Reports normalized MSE/MAE (z-score with training mean/std).\n"
      "  Default: train_fraction=0.661 (ETTh1 standard split)\n",
      argv[0], WINDOW, HORIZON, HORIZON);
  }
  else if (0 < load_ett_ot(filename, &series, &total)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WINDOW + 1);
    libxs_predict_t* model = libxs_predict_create(WINDOW, HORIZON);
    double train_mean = 0, train_std = 1, conformal = 1.0;
    int ti;
    for (ti = 0; ti < train_end; ++ti) train_mean += series[ti];
    train_mean /= train_end;
    { double v = 0;
      for (ti = 0; ti < train_end; ++ti) {
        const double d = series[ti] - train_mean;
        v += d * d;
      }
      train_std = sqrt(v / train_end);
    }
    fprintf(stdout, "Loaded %d hourly OT values from %s\n", total, filename);
    fprintf(stdout, "Train mean=%.2f, std=%.2f (for normalization)\n",
      train_mean, train_std);
    if (NULL != model) {
      libxs_timer_tick_t tick;
      double dt_build, dt_eval;
      int t, build_ok = EXIT_FAILURE;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_decompose(model, decompose);
      libxs_predict_set_series(model, 1, WINDOW);
      if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
      if (0.0 != quantile) libxs_predict_set_quantile(model, quantile);
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
      if (EXIT_SUCCESS == build_ok && quantile > 0) {
        const int cal_start = train_end;
        const int cal_end = LIBXS_MIN(train_end + (total - train_end) / 3,
          total - HORIZON);
        double* scores = NULL;
        int nscores = 0, score_cap = 2048;
        scores = (double*)malloc((size_t)score_cap * sizeof(double));
        if (NULL != scores) {
          for (t = cal_start; t <= cal_end; t += HORIZON) {
            double inputs[WINDOW], outputs[HORIZON];
            libxs_predict_info_t info;
            int i, h;
            for (i = 0; i < WINDOW; ++i) inputs[i] = series[t - WINDOW + i];
            libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
            if (NULL != info.lower && NULL != info.upper) {
              for (h = 0; h < HORIZON; ++h) {
                if (t + h < total) {
                  const double actual = series[t + h];
                  const double hw = (info.upper[h] - info.lower[h]) * 0.5;
                  if (hw > 0) {
                    const double err = (outputs[h] > actual)
                      ? outputs[h] - actual : actual - outputs[h];
                    if (nscores >= score_cap) {
                      score_cap *= 2;
                      scores = (double*)realloc(scores,
                        (size_t)score_cap * sizeof(double));
                    }
                    scores[nscores++] = err / hw;
                  }
                }
              }
            }
          }
          if (nscores > 1) {
            const int idx = (int)((1.0 - 2.0 * quantile) * nscores);
            libxs_sort(scores, nscores, sizeof(double),
              libxs_cmp_f64, NULL);
            conformal = scores[(idx < nscores) ? idx : nscores - 1];
            fprintf(stdout, "Conformal calibration: scale=%.2f"
              " (from %d scores, target=%.0f%%)\n",
              conformal, nscores, (1.0 - 2.0 * quantile) * 100);
          }
          free(scores);
        }
      }
      if (EXIT_SUCCESS == build_ok) {
        libxs_predict_query_t qi;
        double sum_mae = 0, sum_mse = 0;
        double sum_conf = 0;
        int covered = 0, interval_count = 0;
        int neval = 0, h;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        fprintf(stdout, "Window=%d, Horizon=%d, Stride=%d, Train=%d, Test=%d\n",
          WINDOW, HORIZON, stride, qi.nentries, total - train_end);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
          " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
        tick = libxs_timer_tick();
        for (t = train_end; t <= total - HORIZON; t += stride) {
          double inputs[WINDOW], outputs[HORIZON];
          libxs_predict_info_t info;
          int i;
          for (i = 0; i < WINDOW; ++i) inputs[i] = series[t - WINDOW + i];
          libxs_predict_eval(NULL, model, inputs, outputs, &info, 1);
          for (h = 0; h < HORIZON; ++h) {
            const double actual = series[t + h];
            const double err = outputs[h] - actual;
            sum_mae += (err >= 0) ? err : -err;
            sum_mse += err * err;
            if (NULL != info.lower && NULL != info.upper) {
              const double mid = outputs[h];
              const double lo_cal = mid - (mid - info.lower[h]) * conformal;
              const double hi_cal = mid + (info.upper[h] - mid) * conformal;
              if (actual >= lo_cal && actual <= hi_cal) {
                ++covered;
              }
              ++interval_count;
            }
          }
          if (NULL != info.confidence) sum_conf += info.confidence[0];
          ++neval;
        }
        dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
        if (0 < neval) {
          const int ntotal_pts = neval * HORIZON;
          const double norm_mse = sum_mse / ntotal_pts
            / (train_std * train_std);
          const double norm_mae = sum_mae / ntotal_pts / train_std;
          fprintf(stdout, "Forecast (%d windows, %d points, stride=%d):\n",
            neval, ntotal_pts, stride);
          fprintf(stdout, "  MSE (normalized): %.4f\n", norm_mse);
          fprintf(stdout, "  MAE (normalized): %.4f\n", norm_mae);
          fprintf(stdout, "  MSE (raw):  %.4f\n", sum_mse / ntotal_pts);
          fprintf(stdout, "  MAE (raw):  %.4f\n", sum_mae / ntotal_pts);
          if (interval_count > 0) {
            fprintf(stdout, "  Interval coverage (q=%.2f): %.1f%% (%d/%d)\n",
              quantile, 100.0 * covered / interval_count,
              covered, interval_count);
          }
          fprintf(stdout, "  Avg confidence: %.3f\n", sum_conf / neval);
          fprintf(stdout, "Eval: %d queries (%.2f s)\n", neval, dt_eval);
        }
        result = EXIT_SUCCESS;
      }
      libxs_predict_destroy(model);
    }
    free(series);
  }
  else {
    fprintf(stderr, "Failed to load OT data from %s\n", filename);
  }
  return result;
}
