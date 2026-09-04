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
#include <libxs/libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__XGBOOST)
# include "predict_xgb.h"
#endif
#include "predict_args.h"

enum { WINDOW_DEF = 96, HORIZON = 96, MAXCOLS = 7, WMAX = 512 };
static const char* col_names[MAXCOLS] = {
  "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"
};


static double local_corr(const double* ch, const double* tgt, int w);
static int load_ett_all(const char* filename, double** values,
  int* count, int* ncols_out);


static const char* mode_name(int decompose)
{
  static const char* names[] = { "RAW", "SPREAD", "PCA", "SETDIFF", "FISHER",
    "RF", "hKNN" };
  return (0 <= decompose && 7 > decompose) ? names[decompose] : "?";
}


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  int nseries = 1;
  const double split = 0.661;
  const char* wenv = getenv("WINDOW");
  const int window_req = (NULL != wenv) ? atoi(wenv) : LIBXS_PREDICT_AUTO_WINDOW;
  int window = (0 < window_req) ? window_req : WINDOW_DEF;
  int decompose = LIBXS_PREDICT_AUTO_DECOMPOSE;
  int attend = 0;
  double quality = 0;
  int argi, npos = 0, use_xgb = 0, bad = 0, result = EXIT_FAILURE;
  double* data = NULL;
  int total = 0, ncols = 0;
  for (argi = 2; argi < argc; ++argi) {
    const char* arg = argv[argi];
    if (0 != predict_isnum(arg)) {
      if (0 == npos) nseries = atoi(arg);
      else bad = argi;
      ++npos;
    }
    else if (0 != predict_iskey(arg, "attend")) attend = 1;
    else if (0 != predict_iskey(arg, "hknn")) decompose = LIBXS_PREDICT_HKNN;
    else if (0 != predict_iskey(arg, "rf")) decompose = LIBXS_PREDICT_RF;
    else if (0 != predict_iskey(arg, "spread")) {
      decompose = LIBXS_PREDICT_SPREAD;
    }
    else if (0 != predict_iskey(arg, "pca")) decompose = LIBXS_PREDICT_PCA;
    else if (0 != predict_iskey(arg, "none")) decompose = LIBXS_PREDICT_RAW;
    else if (0 != predict_keyval(arg, "compress", 0.9, &quality)) {
      /* the keyword that matched has already assigned its own value */
    }
    else if (0 != predict_iskey(arg, "nocompress")) quality = 0;
    else if (0 != predict_iskey(arg, "xgb")) use_xgb = 1;
    else bad = argi;
  }
  if (0 != bad) {
    fprintf(stderr, "Unrecognized argument \"%s\".\n", argv[bad]);
  }
  if (nseries < 1) nseries = 1;
  if (nseries > MAXCOLS) nseries = MAXCOLS;
  /* the comparison reads the built windows back, which compression prunes */
  if (0 != use_xgb) quality = 0;
  if (NULL == filename || 0 != bad) {
    fprintf(stdout,
      "Usage: %s <ett_csv> [nseries=1..7]"
      " [attend|spread|pca|hknn|rf|none|compress[Q]|xgb]\n"
      "  Multivariate ETT forecasting: predict OT from nseries channels.\n"
      "  Channels (in order): HUFL,HULL,MUFL,MULL,LUFL,LULL,OT.\n"
      "  attend: per-query local-correlation channel weighting.\n"
      "  none: no decomposition. Without it the mode is selected at build,\n"
      "    which costs one full build per candidate per fold on a series.\n"
      "  compress[Q]: drop redundant entries (Q: threshold, default 0.9);\n"
      "    off unless asked, as in every other sample.\n"
      "  xgb: also train XGBoost on the same windows and compare\n"
      "    (implies no compression; plain configuration only).\n"
      "  nseries=1: univariate (OT only).\n"
      "  nseries=7: all channels as co-inputs to predict OT.\n"
      "  Window=%d, Horizon=%d, split=0.661 (standard ETTh1).\n",
      argv[0], WINDOW_DEF, HORIZON);
  }
#if !defined(__XGBOOST)
  else if (0 != use_xgb) {
    fprintf(stderr, "Requested xgb but this binary was built without XGBoost:"
      " set XGBOOST_ROOT, or install the pkg-config module.\n");
  }
#else
  /**
   * The comparison needs the plain configuration, and the selected mode is not
   * known until the build.  A caller who did not ask for a decomposition gets
   * the plain one here rather than a refusal; one who did still gets the
   * refusal, because that is a request the comparison cannot honour.
   */
  else if (0 != use_xgb && LIBXS_PREDICT_AUTO_DECOMPOSE == decompose
    && 0 == attend && NULL == getenv("BANK"))
  {
    decompose = LIBXS_PREDICT_RAW;
  }
  else if (0 != use_xgb && (LIBXS_PREDICT_RAW != decompose
    || 0 != attend || NULL != getenv("BANK")))
  {
    fprintf(stderr, "xgb needs the plain configuration: a decomposition,"
      " attend, or BANK zeroes the weight of lags the distance does not read,"
      " which makes the built windows unrecoverable.\n");
  }
#endif
  else if (0 < load_ett_all(filename, &data, &total, &ncols)) {
    const int train_end = LIBXS_MAX((int)(total * split + 0.5), WMAX + 1);
    const int target = nseries - 1;
    /**
     * Window budget (upper bound for the auto sizer). Single-series gets
     * a wide cap so the grid can explore; multi-series caps at the tuned
     * WINDOW_DEF so the library's multi-series abstention returns it.
     */
    const int wcap_req = (0 < window_req) ? window_req
      : ((nseries <= 1) ? WMAX : WINDOW_DEF);
    const int ninputs = nseries * wcap_req;
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
      { const char* kenv = getenv("BANK");
        if (NULL != kenv) libxs_predict_set_series_bank(model, atoi(kenv));
      }
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
      { const int br = libxs_predict_build_task(NULL, model, 0, 2,
          quality, omp_get_thread_num(), omp_get_num_threads());
        if (0 == omp_get_thread_num()) build_ok = br;
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
        fprintf(stdout, "Decomposition: %s (%s)\n",
          mode_name(qi.decompose),
          (LIBXS_PREDICT_AUTO_DECOMPOSE == decompose)
            ? "selected at build" : "requested");
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
#if defined(__XGBOOST)
          if (0 != use_xgb) {
            const int nin = nseries * window;
            const int ncorp = qi.nentries + neval;
            libxs_predict_t* xsrc = libxs_predict_create(nin, HORIZON);
            double* xpred = (double*)malloc(
              (size_t)ncorp * HORIZON * sizeof(double));
            char* mask = (char*)calloc((size_t)ncorp, 1);
            int* classify = (int*)calloc((size_t)HORIZON, sizeof(int));
            double* in = (double*)malloc((size_t)nin * sizeof(double));
            double* out = (double*)malloc((size_t)HORIZON * sizeof(double));
            int k = 0;
            if (NULL != xsrc && NULL != xpred && NULL != mask
              && NULL != classify && NULL != in && NULL != out)
            {
              for (k = 0; k < qi.nentries; ++k) {
                libxs_predict_get(model, k, in, out);
                libxs_predict_push(NULL, xsrc, in, out);
                mask[k] = 1;
              }
              for (t = train_end; t <= total - HORIZON; t += HORIZON) {
                int i;
                for (i = 0; i < window; ++i) {
                  for (s = 0; s < nseries; ++s) {
                    in[s * window + i] = data[(size_t)(t - window + i) * ncols
                      + (ncols - nseries + s)];
                  }
                }
                for (h = 0; h < HORIZON; ++h) {
                  out[h] = data[(size_t)(t + h) * ncols + (ncols - 1)];
                }
                libxs_predict_push(NULL, xsrc, in, out);
                ++k;
              }
              tick = libxs_timer_tick();
              if (EXIT_SUCCESS == predict_xgb(xsrc, k, nin, HORIZON, mask,
                classify, xpred, NULL, NULL, "reg:squarederror"))
              {
                const double dt_xgb =
                  libxs_timer_duration(tick, libxs_timer_tick());
                const int pts = neval * HORIZON;
                double xsum_mae = 0, xsum_mse = 0;
                int idx;
                for (idx = qi.nentries; idx < k; ++idx) {
                  libxs_predict_get(xsrc, idx, NULL, out);
                  for (h = 0; h < HORIZON; ++h) {
                    const double err =
                      xpred[(size_t)idx * HORIZON + h] - out[h];
                    xsum_mae += (err >= 0) ? err : -err;
                    xsum_mse += err * err;
                  }
                }
                fprintf(stdout, "XGBoost (%d boosters, rounds=%i, depth=%i,"
                  " eta=%g):\n", HORIZON,
                  predict_xgb_geti("XGB_ROUNDS", 200),
                  predict_xgb_geti("XGB_DEPTH", 6),
                  predict_xgb_getd("XGB_ETA", 0.1));
                fprintf(stdout, "  MSE (normalized): %.4f\n",
                  xsum_mse / pts / (train_std * train_std));
                fprintf(stdout, "  MAE (normalized): %.4f\n",
                  xsum_mae / pts / train_std);
                fprintf(stdout, "Train+eval: %d windows, %d features (%.2f s)\n",
                  qi.nentries, nin, dt_xgb);
              }
            }
            free(out);
            free(in);
            free(classify);
            free(mask);
            free(xpred);
            libxs_predict_destroy(xsrc);
          }
#endif
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
