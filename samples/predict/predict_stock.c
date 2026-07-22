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

#define STOCK_MAXCOLS 16

enum { WINDOW = 20, HORIZON = 5 };

static void build_inputs(const libxs_predict_t* source,
  int t, int series, int nseries, double* inputs);
static void evaluate_forecast(
  const libxs_predict_t* const split_models[],
  const libxs_predict_t* const full_models[],
  const libxs_predict_t* source, int total, int train_end,
  int nseries, int joint, const char (*names)[32]);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const char* colspec = (argc > 2) ? argv[2] : "1,2";
  const double split = (argc > 3) ? atof(argv[3]) : 0.8;
  int decompose_arg = -1;
  double quality = 0, consistency = 0;
  int argi, result = EXIT_FAILURE;
  for (argi = 4; argi < argc; ++argi) {
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
    else if ('h' == argv[argi][0]) decompose_arg = LIBXS_PREDICT_HKNN;
    else if ('r' == argv[argi][0] && 'f' == argv[argi][1]) {
      decompose_arg = LIBXS_PREDICT_RF;
    }
  }
  if (NULL == filename) {
    fprintf(stderr,
      "Usage: %s <csv_file> [columns] [train_fraction]"
      " [compress[Q]] [hknn|rf]\n"
      "  Multi-stock timeseries prediction with auto-differencing.\n"
      "  columns: comma-separated 0-based column indices (default: 1,2).\n"
      "  Uses PCA decomposition for 3+ series, SPREAD for 2.\n"
      "  Default train_fraction=0.8\n", argv[0]);
  }
  else {
    int cols[STOCK_MAXCOLS], nseries = 0;
    char inputs_spec[256], target_spec[16];
    libxs_predict_t* source;
    int total;
    { int pos = 0;
      while (nseries < STOCK_MAXCOLS && colspec[pos] != '\0') {
        char* end = NULL;
        cols[nseries++] = (int)strtol(colspec + pos, &end, 10);
        pos = (int)(end - colspec);
        if (',' == colspec[pos]) ++pos;
      }
    }
    if (nseries < 1) {
      fprintf(stderr, "No columns specified\n");
    }
    else {
      int ninputs = nseries * WINDOW, s, t;
      int decompose = (0 <= decompose_arg) ? decompose_arg
        : ((nseries >= 2) ? LIBXS_PREDICT_SPREAD : LIBXS_PREDICT_RAW);
      { char* wp = inputs_spec;
        for (s = 0; s < nseries; ++s) {
          if (s > 0) *wp++ = ',';
          wp += sprintf(wp, "%d", cols[s]);
        }
        *wp = '\0';
      }
      LIBXS_SNPRINTF(target_spec, sizeof(target_spec), "%d", cols[0]);
      { char csv_header[1024], delim = ',';
        csv_header[0] = '\0';
        source = libxs_predict_create(nseries, 1);
        total = (NULL != source)
          ? libxs_predict_load_csv(source, filename, NULL,
              inputs_spec, target_spec,
              csv_header, (int)sizeof(csv_header), &delim)
          : 0;
      if (0 < total) {
        const int train_end = LIBXS_MAX(
          (int)(total * split + 0.5), WINDOW + 1);
        char names_buf[STOCK_MAXCOLS][32];
        const char (*names)[32] = (const char (*)[32])names_buf;
        { char sep[2];
          sep[0] = delim; sep[1] = '\0';
          for (s = 0; s < nseries; ++s) {
            int len = 0;
            const char* tok = libxs_strtoken(csv_header, sep, cols[s], &len);
            if (NULL != tok && len > 0 && len < 32) {
              memcpy(names_buf[s], tok, (size_t)len);
              names_buf[s][len] = '\0';
            }
            else {
              LIBXS_SNPRINTF(names_buf[s], 32, "%d", cols[s]);
            }
          }
        }
        fprintf(stdout, "Loaded %d rows from %s (%d series:",
          total, filename, nseries);
        for (s = 0; s < nseries; ++s) {
          fprintf(stdout, " %s", names[s]);
        }
        fprintf(stdout, ")\n");
        fprintf(stdout, "Window=%d, Horizon=%d, Train=%d, Test=%d\n",
          WINDOW, HORIZON, train_end - WINDOW, total - train_end);
        { libxs_predict_t** split_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*split_m));
          libxs_predict_t** full_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*full_m));
          int ok = 1;
          if (NULL != split_m && NULL != full_m) {
            for (s = 0; s < nseries; ++s) {
              split_m[s] = libxs_predict_create(ninputs, HORIZON);
              full_m[s] = libxs_predict_create(ninputs, HORIZON);
              if (NULL != split_m[s] && NULL != full_m[s]) {
                libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(split_m[s], 0);
                libxs_predict_set_series(split_m[s], nseries, WINDOW);
                libxs_predict_set_target(split_m[s], s);
                libxs_predict_set_decompose(split_m[s], decompose);
                if (0.0 != consistency) libxs_predict_set_consistency(split_m[s], consistency);
                libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(full_m[s], 0);
                libxs_predict_set_series(full_m[s], nseries, WINDOW);
                libxs_predict_set_target(full_m[s], s);
                libxs_predict_set_decompose(full_m[s], decompose);
                if (0.0 != consistency) libxs_predict_set_consistency(full_m[s], consistency);
                for (t = 0; t < total; ++t) {
                  double vals[STOCK_MAXCOLS];
                  libxs_predict_get(source, t, vals, NULL);
                  if (t < train_end) {
                    libxs_predict_push(NULL, split_m[s], vals, NULL);
                  }
                  libxs_predict_push(NULL, full_m[s], vals, NULL);
                }
                if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2, quality)) {
                  ok = 0;
                }
                if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2, quality)) {
                  ok = 0;
                }
              }
              else {
                ok = 0;
              }
            }
            if (0 != ok) {
              libxs_predict_query_t qi;
              LIBXS_MEMZERO(&qi);
              libxs_predict_query(split_m[0], &qi);
              fprintf(stdout, "Built: %d clusters, %.1fx compression,"
                " order=%d, diff=%d\n",
                qi.nclusters, qi.compression, qi.order, qi.diff_order);
              fprintf(stdout, "\n--- %s decomposition ---\n",
                (LIBXS_PREDICT_SPREAD == decompose) ? "SPREAD"
                : ((LIBXS_PREDICT_PCA == decompose) ? "PCA"
                : ((LIBXS_PREDICT_HKNN == decompose) ? "hKNN"
                : ((LIBXS_PREDICT_RF == decompose) ? "RF" : "RAW"))));
              evaluate_forecast(
                (const libxs_predict_t* const*)split_m,
                (const libxs_predict_t* const*)full_m,
                source, total, train_end, nseries, 1, names);
              result = EXIT_SUCCESS;
            }
            for (s = 0; s < nseries; ++s) {
              if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
              if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
            }
          }
          free(split_m);
          free(full_m);
        }
        { libxs_predict_t** split_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*split_m));
          libxs_predict_t** full_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*full_m));
          int ok = 1;
          if (NULL != split_m && NULL != full_m) {
            for (s = 0; s < nseries; ++s) {
              split_m[s] = libxs_predict_create(ninputs, HORIZON);
              full_m[s] = libxs_predict_create(ninputs, HORIZON);
              if (NULL != split_m[s] && NULL != full_m[s]) {
                libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(split_m[s], 0);
                libxs_predict_set_series(split_m[s], nseries, WINDOW);
                libxs_predict_set_target(split_m[s], s);
                if (0 <= decompose_arg) {
                  libxs_predict_set_decompose(split_m[s], decompose_arg);
                }
                libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(full_m[s], 0);
                libxs_predict_set_series(full_m[s], nseries, WINDOW);
                libxs_predict_set_target(full_m[s], s);
                if (0 <= decompose_arg) {
                  libxs_predict_set_decompose(full_m[s], decompose_arg);
                }
                for (t = 0; t < total; ++t) {
                  double vals[STOCK_MAXCOLS];
                  libxs_predict_get(source, t, vals, NULL);
                  if (t < train_end) {
                    libxs_predict_push(NULL, split_m[s], vals, NULL);
                  }
                  libxs_predict_push(NULL, full_m[s], vals, NULL);
                }
                if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2, quality)) {
                  ok = 0;
                }
                if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2, quality)) {
                  ok = 0;
                }
              }
              else {
                ok = 0;
              }
            }
            if (0 != ok) {
              fprintf(stdout, "\n--- RAW concatenation (baseline) ---\n");
              evaluate_forecast(
                (const libxs_predict_t* const*)split_m,
                (const libxs_predict_t* const*)full_m,
                source, total, train_end, nseries, 1, names);
            }
            for (s = 0; s < nseries; ++s) {
              if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
              if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
            }
          }
          free(split_m);
          free(full_m);
        }
        { libxs_predict_t** split_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*split_m));
          libxs_predict_t** full_m = (libxs_predict_t**)calloc(
            (size_t)nseries, sizeof(*full_m));
          int ok = 1;
          if (NULL != split_m && NULL != full_m) {
            for (s = 0; s < nseries; ++s) {
              split_m[s] = libxs_predict_create(WINDOW, HORIZON);
              full_m[s] = libxs_predict_create(WINDOW, HORIZON);
              if (NULL != split_m[s] && NULL != full_m[s]) {
                libxs_predict_set_mode(split_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(split_m[s], 0);
                libxs_predict_set_series(split_m[s], 1, WINDOW);
                if (0 <= decompose_arg) {
                  libxs_predict_set_decompose(split_m[s], decompose_arg);
                }
                libxs_predict_set_mode(full_m[s], LIBXS_PREDICT_TEMPORAL);
                libxs_predict_set_diff(full_m[s], 0);
                libxs_predict_set_series(full_m[s], 1, WINDOW);
                if (0 <= decompose_arg) {
                  libxs_predict_set_decompose(full_m[s], decompose_arg);
                }
                for (t = 0; t < total; ++t) {
                  double vals[STOCK_MAXCOLS];
                  libxs_predict_get(source, t, vals, NULL);
                  if (t < train_end) {
                    libxs_predict_push(NULL, split_m[s], &vals[s], NULL);
                  }
                  libxs_predict_push(NULL, full_m[s], &vals[s], NULL);
                }
                if (EXIT_SUCCESS != libxs_predict_build(split_m[s], 0, 2, quality)) {
                  ok = 0;
                }
                if (EXIT_SUCCESS != libxs_predict_build(full_m[s], 0, 2, quality)) {
                  ok = 0;
                }
              }
              else {
                ok = 0;
              }
            }
            if (0 != ok) {
              fprintf(stdout, "\n--- Single-series (independent) ---\n");
              evaluate_forecast(
                (const libxs_predict_t* const*)split_m,
                (const libxs_predict_t* const*)full_m,
                source, total, train_end, nseries, 0, names);
            }
            for (s = 0; s < nseries; ++s) {
              if (NULL != split_m[s]) libxs_predict_destroy(split_m[s]);
              if (NULL != full_m[s]) libxs_predict_destroy(full_m[s]);
            }
          }
          free(split_m);
          free(full_m);
        }
      }
      else {
        fprintf(stderr, "Failed to load data from %s\n", filename);
      }
      if (NULL != source) libxs_predict_destroy(source);
      }
    }
  }
  return result;
}


static void build_inputs(const libxs_predict_t* source,
  int t, int series, int nseries, double* inputs)
{
  int i;
  if (series < 0) {
    int si;
    for (si = 0; si < nseries; ++si) {
      for (i = 0; i < WINDOW; ++i) {
        double vals[STOCK_MAXCOLS];
        libxs_predict_get(source, t - WINDOW + i, vals, NULL);
        inputs[si * WINDOW + i] = vals[si];
      }
    }
  }
  else {
    for (i = 0; i < WINDOW; ++i) {
      double vals[STOCK_MAXCOLS];
      libxs_predict_get(source, t - WINDOW + i, vals, NULL);
      inputs[i] = vals[series];
    }
  }
}


static void evaluate_forecast(
  const libxs_predict_t* const split_models[],
  const libxs_predict_t* const full_models[],
  const libxs_predict_t* source, int total, int train_end,
  int nseries, int joint, const char (*names)[32])
{
  double sum_err[STOCK_MAXCOLS][HORIZON] = {{ 0 }};
  double max_err[STOCK_MAXCOLS][HORIZON] = {{ 0 }};
  double forecast[STOCK_MAXCOLS][HORIZON];
  double sum_conf = 0, fc_conf = 0;
  int neval = 0, h, s, t;
  for (t = train_end; t <= total - HORIZON; ++t) {
    double inputs[STOCK_MAXCOLS * WINDOW], outputs[HORIZON];
    libxs_predict_info_t info;
    for (s = 0; s < nseries; ++s) {
      build_inputs(source, t, (0 != joint) ? -1 : s, nseries, inputs);
      libxs_predict_eval(NULL, split_models[s], inputs, outputs, &info, 1);
      for (h = 0; h < HORIZON; ++h) {
        double vals[STOCK_MAXCOLS], actual;
        libxs_predict_get(source, t + h, vals, NULL);
        actual = vals[s];
        { const double err = LIBXS_FABS(outputs[h] - actual);
          sum_err[s][h] += err;
          if (err > max_err[s][h]) max_err[s][h] = err;
        }
      }
      if (0 == s) sum_conf += info.confidence[0];
    }
    ++neval;
  }
  for (s = 0; s < nseries; ++s) {
    double inputs[STOCK_MAXCOLS * WINDOW];
    libxs_predict_info_t info;
    build_inputs(source, total, (0 != joint) ? -1 : s, nseries, inputs);
    libxs_predict_eval(NULL, full_models[s], inputs, forecast[s], &info, 1);
    fc_conf = info.confidence[0];
  }
  if (0 < neval) {
    fprintf(stdout, "Quality (%d windows) and forecast (confidence %.3f):\n",
      neval, fc_conf);
    fprintf(stdout, "  step");
    for (s = 0; s < nseries; ++s) {
      fprintf(stdout, "   avg-err   max-err  %6s", names[s]);
    }
    fprintf(stdout, "\n");
    for (h = 0; h < HORIZON; ++h) {
      fprintf(stdout, "  t+%-2d", h + 1);
      for (s = 0; s < nseries; ++s) {
        fprintf(stdout, "  %8.4f  %8.4f  %8.4f",
          sum_err[s][h] / neval, max_err[s][h], forecast[s][h]);
      }
      fprintf(stdout, "\n");
    }
    fprintf(stdout, "  avg confidence: %.3f\n", sum_conf / neval);
  }
}
