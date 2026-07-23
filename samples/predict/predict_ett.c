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
#include <libxs/libxs_mem.h>
#include <math.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

enum { WINDOW_DEF = 96, HORIZON = 96, MAXCOLS = 7, WMAX = 512 };
static const char* col_names[MAXCOLS] = {
  "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"
};


static double local_corr(const double* ch, const double* tgt, int w);
static int load_ett_all(const char* filename, double** values,
  int* count, int* ncols_out);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  int nseries = (argc > 2) ? atoi(argv[2]) : 1;
  const double split = 0.661;
  const char* wenv = getenv("WINDOW");
  const int window_req = (NULL != wenv) ? atoi(wenv) : WINDOW_DEF;
  int window = (0 < window_req) ? window_req : WINDOW_DEF;
  int decompose = LIBXS_PREDICT_RAW;
  int attend = 0;
  double quality = 0.9;
  int argi, result = EXIT_FAILURE;
  double* data = NULL;
  int total = 0, ncols = 0;
  for (argi = 3; argi < argc; ++argi) {
    if ('a' == argv[argi][0]) attend = 1;
    else if ('h' == argv[argi][0]) decompose = LIBXS_PREDICT_HKNN;
    else if ('r' == argv[argi][0]) decompose = LIBXS_PREDICT_RF;
    else if ('s' == argv[argi][0]) decompose = LIBXS_PREDICT_SPREAD;
    else if ('p' == argv[argi][0]) decompose = LIBXS_PREDICT_PCA;
    else if ('n' == argv[argi][0]) quality = 0;
  }
  if (nseries < 1) nseries = 1;
  if (nseries > MAXCOLS) nseries = MAXCOLS;
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <ett_csv> [nseries=1..7]"
      " [attend|spread|pca|hknn|rf|nocompress]\n"
      "  Multivariate ETT forecasting: predict OT from nseries channels.\n"
      "  Channels (in order): HUFL,HULL,MUFL,MULL,LUFL,LULL,OT.\n"
      "  attend: per-query local-correlation channel weighting.\n"
      "  nseries=1: univariate (OT only).\n"
      "  nseries=7: all channels as co-inputs to predict OT.\n"
      "  Window=%d, Horizon=%d, split=0.661 (standard ETTh1).\n",
      argv[0], WINDOW_DEF, HORIZON);
  }
  else if (0 < load_ett_all(filename, &data, &total, &ncols)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WMAX + 1);
    const int target = nseries - 1;
    const int ninputs = nseries * ((0 < window_req) ? window_req : WMAX);
    libxs_predict_t* model = libxs_predict_create(ninputs, HORIZON);
    double train_mean = 0, train_std = 1;
    int ti;
    for (ti = 0; ti < train_end; ++ti) {
      train_mean += data[(size_t)ti * ncols + (ncols - 1)];
    }
    train_mean /= train_end;
    { double v = 0;
      for (ti = 0; ti < train_end; ++ti) {
        const double d = data[(size_t)ti * ncols + (ncols - 1)] - train_mean;
        v += d * d;
      }
      train_std = sqrt(v / train_end);
    }
    fprintf(stdout, "Loaded %d rows (%d channels) from %s\n",
      total, ncols, filename);
    fprintf(stdout, "Using %d series as input:", nseries);
    { int s;
      for (s = 0; s < nseries; ++s) {
        fprintf(stdout, " %s", col_names[ncols - nseries + s]);
      }
    }
    fprintf(stdout, " (target: %s)\n", col_names[ncols - 1]);
    fprintf(stdout, "OT train mean=%.2f, std=%.2f\n", train_mean, train_std);
    if (NULL != model) {
      libxs_timer_tick_t tick;
      double dt_build, dt_eval;
      double avg_corr[MAXCOLS];
      int t, build_ok = EXIT_FAILURE;
      int s;
      libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
      libxs_predict_set_decompose(model, decompose);
      libxs_predict_set_series(model, nseries, window_req);
      libxs_predict_set_target(model, target);
      for (s = 0; s < nseries; ++s) avg_corr[s] = 1.0;
      if (0 != attend && nseries > 1) {
        int nw = 0;
        for (s = 0; s < nseries; ++s) avg_corr[s] = 0;
        for (ti = window; ti < train_end; ti += window) {
          double tgt_buf[WMAX], ch_buf[WMAX];
          int i;
          for (i = 0; i < window; ++i) {
            tgt_buf[i] = data[(size_t)(ti - window + i) * ncols + (ncols - 1)];
          }
          for (s = 0; s < nseries; ++s) {
            for (i = 0; i < window; ++i) {
              ch_buf[i] = data[(size_t)(ti - window + i) * ncols
                + (ncols - nseries + s)];
            }
            avg_corr[s] += local_corr(ch_buf, tgt_buf, window);
          }
          ++nw;
        }
        if (nw > 0) {
          for (s = 0; s < nseries; ++s) avg_corr[s] /= nw;
        }
        { double* wfull = (double*)calloc((size_t)ninputs, sizeof(double));
          if (NULL != wfull) {
            int i;
            for (s = 0; s < nseries; ++s) {
              const double w = (avg_corr[s] > 0.01) ? avg_corr[s] : 0.01;
              for (i = 0; i < window; ++i) wfull[s * window + i] = w;
            }
            libxs_predict_set_weights(model, wfull);
            free(wfull);
          }
        }
        fprintf(stdout, "Attend correlations:");
        for (s = 0; s < nseries; ++s) {
          fprintf(stdout, " %.3f", avg_corr[s]);
        }
        fprintf(stdout, "\n");
      }
      for (t = 0; t < train_end; ++t) {
        double step[MAXCOLS];
        for (s = 0; s < nseries; ++s) {
          step[s] = data[(size_t)t * ncols + (ncols - nseries + s)];
        }
        libxs_predict_push(NULL, model, step, NULL);
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
        double sum_mae = 0, sum_mse = 0;
        int neval = 0, h;
        LIBXS_MEMZERO(&qi);
        libxs_predict_query(model, &qi);
        window = qi.window;
        fprintf(stdout, "Window=%d, Horizon=%d, Stride=%d,"
          " nseries=%d, Train=%d, Test=%d\n",
          window, HORIZON, HORIZON, nseries, qi.nentries,
          total - train_end);
        fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
          " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
        tick = libxs_timer_tick();
        for (t = train_end; t <= total - HORIZON; t += HORIZON) {
          double outputs[HORIZON];
          int i;
          double inputs[MAXCOLS * WMAX];
          for (i = 0; i < window; ++i) {
            for (s = 0; s < nseries; ++s) {
              inputs[s * window + i] =
                data[(size_t)(t - window + i) * ncols + (ncols - nseries + s)];
            }
          }
          if (0 != attend && nseries > 1) {
            double tgt_buf[WMAX], ch_buf[WMAX];
            for (i = 0; i < window; ++i) {
              tgt_buf[i] = data[(size_t)(t - window + i) * ncols + (ncols - 1)];
            }
            for (s = 0; s < nseries; ++s) {
              double w;
              for (i = 0; i < window; ++i) {
                ch_buf[i] = inputs[s * window + i];
              }
              w = local_corr(ch_buf, tgt_buf, window);
              if (w < 0.01) w = 0.01;
              if (avg_corr[s] > 0.01) {
                const double scale = w / avg_corr[s];
                for (i = 0; i < window; ++i) {
                  inputs[s * window + i] *= scale;
                }
              }
            }
          }
          libxs_predict_eval(NULL, model, inputs, outputs, NULL, 1);
          for (h = 0; h < HORIZON; ++h) {
            const double actual =
              data[(size_t)(t + h) * ncols + (ncols - 1)];
            const double err = outputs[h] - actual;
            sum_mae += (err >= 0) ? err : -err;
            sum_mse += err * err;
          }
          ++neval;
        }
        dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
        if (0 < neval) {
          const int ntotal_pts = neval * HORIZON;
          const double norm_mse = sum_mse / ntotal_pts
            / (train_std * train_std);
          const double norm_mae = sum_mae / ntotal_pts / train_std;
          fprintf(stdout,
            "Forecast (%d windows, %d points, stride=%d):\n",
            neval, ntotal_pts, HORIZON);
          fprintf(stdout, "  MSE (normalized): %.4f\n", norm_mse);
          fprintf(stdout, "  MAE (normalized): %.4f\n", norm_mae);
          fprintf(stdout, "Eval: %d queries (%.2f s)\n", neval, dt_eval);
        }
        result = EXIT_SUCCESS;
      }
      libxs_predict_destroy(model);
    }
    free(data);
  }
  else {
    fprintf(stderr, "Failed to load data from %s\n", filename);
  }
  return result;
}


static double local_corr(const double* ch, const double* tgt, int w)
{
  double sa = 0, sb = 0, sa2 = 0, sb2 = 0, sab = 0;
  double va, vb, cov, denom;
  int i;
  for (i = 0; i < w; ++i) {
    sa += ch[i]; sb += tgt[i];
    sa2 += ch[i] * ch[i]; sb2 += tgt[i] * tgt[i];
    sab += ch[i] * tgt[i];
  }
  va = sa2 - sa * sa / w;
  vb = sb2 - sb * sb / w;
  cov = sab - sa * sb / w;
  denom = sqrt(va * vb);
  return (denom > 0) ? fabs(cov / denom) : 0;
}


static int load_ett_all(const char* filename, double** values,
  int* count, int* ncols_out)
{
  int result = 0;
  FILE* file = fopen(filename, "r");
  if (NULL != file) {
    char line[1024];
    int capacity = 20000, n = 0, ncols = MAXCOLS;
    double* data = (double*)malloc(
      (size_t)capacity * (size_t)ncols * sizeof(double));
    if (NULL != data) {
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        char* p = line;
        int col = 0;
        double vals[MAXCOLS];
        if (line[0] < '0' || line[0] > '9') {
          if (0 == n) continue;
        }
        while ('\0' != *p && ',' != *p) ++p;
        if (',' == *p) ++p;
        for (col = 0; col < ncols && '\0' != *p; ++col) {
          vals[col] = strtod(p, &p);
          if (',' == *p || '\r' == *p || '\n' == *p) ++p;
        }
        if (col == ncols) {
          int c;
          if (n >= capacity) {
            capacity *= 2;
            data = (double*)realloc(data,
              (size_t)capacity * (size_t)ncols * sizeof(double));
            if (NULL == data) { n = 0; break; }
          }
          for (c = 0; c < ncols; ++c) {
            data[(size_t)n * ncols + c] = vals[c];
          }
          ++n;
        }
      }
      if (n > 0) {
        *values = data;
        *count = n;
        *ncols_out = ncols;
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
