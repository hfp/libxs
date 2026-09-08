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
#if defined(__XGBOOST)
# include "predict_xgb.h"
#endif
#include "predict_args.h"

static const char input_names[] = "latitude,longitude,depth";
static const char output_names[] = "mag";

enum { NINPUTS = 3, NOUTPUTS = 1 };


static const char* mode_name(int decompose)
{
  static const char* names[] = { "RAW", "SPREAD", "PCA", "SETDIFF", "FISHER",
    "RF", "hKNN" };
  return (0 <= decompose && 7 > decompose) ? names[decompose] : "?";
}


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  double split = 0.8, quality = 0, consistency = 0;
  int decompose = LIBXS_PREDICT_AUTO_DECOMPOSE;
  int argi, npos = 0, use_xgb = 0, bad = 0, result = EXIT_FAILURE;
  for (argi = 2; argi < argc; ++argi) {
    const char* arg = argv[argi];
    if (0 != predict_isnum(arg)) {
      if (0 == npos) split = atof(arg);
      else bad = argi;
      ++npos;
    }
    else if (0 != predict_keyval(arg, "consist", 0.9, &consistency)
      || 0 != predict_keyval(arg, "compress", 0.9, &quality))
    {
      /* the keyword that matched has already assigned its own value */
    }
    else if (0 != predict_iskey(arg, "hknn")) decompose = LIBXS_PREDICT_HKNN;
    else if (0 != predict_iskey(arg, "rf")) decompose = LIBXS_PREDICT_RF;
    else if (0 != predict_iskey(arg, "none")) decompose = LIBXS_PREDICT_RAW;
    else if (0 != predict_iskey(arg, "xgb")) use_xgb = 1;
    else bad = argi;
  }
  if (0 != bad) {
    fprintf(stderr, "Unrecognized argument \"%s\".\n", argv[bad]);
  }
  if (NULL == filename || 0 != bad) {
    fprintf(stdout,
      "Usage: %s <usgs_csv> [train_fraction] [compress[Q]]"
      " [hknn|rf|none] [xgb]\n"
      "  Earthquake magnitude prediction from location and depth.\n"
      "  xgb: also train XGBoost on the same split and compare.\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
#if !defined(__XGBOOST)
  else if (0 != use_xgb) {
    fprintf(stderr, "Requested xgb but this binary was built without XGBoost:"
      " set XGBOOST_ROOT, or install the pkg-config module.\n");
  }
#endif
  else {
    libxs_predict_t* source = libxs_predict_create(NINPUTS, NOUTPUTS);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        input_names, output_names, NULL, 0, NULL);
      if (0 < total) {
        const int train_end = LIBXS_MAX((int)(total * split + 0.5), 1);
        libxs_predict_t* model = libxs_predict_create(NINPUTS, NOUTPUTS);
        fprintf(stdout, "Loaded %d earthquake events from %s\n", total, filename);
        fprintf(stdout, "Inputs: latitude, longitude, depth -> Output: magnitude\n");
        fprintf(stdout, "Train=%d, Test=%d\n", train_end, total - train_end);
        if (NULL != model) {
          libxs_timer_tick_t tick;
          double inputs[NINPUTS], outputs[NOUTPUTS], dt_build;
          int i, build_ok = EXIT_FAILURE;
          libxs_predict_set_decompose(model, decompose);
          libxs_predict_set_neighbors(model, -1);
          if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
          for (i = 0; i < train_end; ++i) {
            libxs_predict_get(source, i, inputs, outputs);
            libxs_predict_push(NULL, model, inputs, outputs);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
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
            double sum_err = 0, max_err = 0, sum_conf = 0, dt_eval;
            int neval = 0;
            LIBXS_MEMZERO(&qi);
            libxs_predict_query(model, &qi);
            fprintf(stdout, "Decomposition: %s (%s)\n", mode_name(qi.decompose),
              (LIBXS_PREDICT_AUTO_DECOMPOSE == decompose)
                ? "selected at build" : "requested");
            fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
              " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
            tick = libxs_timer_tick();
            for (i = train_end; i < total; ++i) {
              double predicted[NOUTPUTS], expected[NOUTPUTS], err;
              libxs_predict_info_t info;
              libxs_predict_get(source, i, inputs, expected);
              libxs_predict_eval(NULL, model, inputs, predicted, &info, 1);
              err = LIBXS_FABS(predicted[0] - expected[0]);
              sum_err += err;
              if (err > max_err) max_err = err;
              sum_conf += info.confidence[0];
              ++neval;
            }
            dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
            if (0 < neval) {
              fprintf(stdout, "Prediction quality (%d test events):\n", neval);
              fprintf(stdout, "  avg magnitude error: %.3f\n", sum_err / neval);
              fprintf(stdout, "  max magnitude error: %.3f\n", max_err);
              fprintf(stdout, "  avg confidence:      %.3f\n", sum_conf / neval);
              fprintf(stdout, "Eval: %d queries (%.2f s)\n", neval, dt_eval);
#if defined(__XGBOOST)
              if (0 != use_xgb) {
                double* xgb_pred = (double*)malloc(
                  (size_t)total * sizeof(double));
                char* mask = (char*)calloc((size_t)total, 1);
                int classify = 0;
                if (NULL != xgb_pred && NULL != mask) {
                  for (i = 0; i < train_end; ++i) mask[i] = 1;
                  if (EXIT_SUCCESS == predict_xgb(source, total, NINPUTS,
                    NOUTPUTS, mask, &classify, xgb_pred, NULL, NULL,
                    "reg:absoluteerror", NULL))
                  {
                    double xsum_err = 0, xmax_err = 0;
                    for (i = train_end; i < total; ++i) {
                      double expected[NOUTPUTS], err;
                      libxs_predict_get(source, i, NULL, expected);
                      err = LIBXS_FABS(xgb_pred[i] - expected[0]);
                      xsum_err += err;
                      if (err > xmax_err) xmax_err = err;
                    }
                    fprintf(stdout, "XGBoost (%s, rounds=%i, depth=%i,"
                      " eta=%g):\n", predict_xgb_regobj("reg:absoluteerror"),
                      predict_xgb_geti("XGB_ROUNDS", 200),
                      predict_xgb_geti("XGB_DEPTH", 6),
                      predict_xgb_getd("XGB_ETA", 0.1));
                    fprintf(stdout, "  avg magnitude error: %.3f\n",
                      xsum_err / neval);
                    fprintf(stdout, "  max magnitude error: %.3f\n", xmax_err);
                  }
                }
                free(mask);
                free(xgb_pred);
              }
#endif
            }
            result = EXIT_SUCCESS;
          }
          libxs_predict_destroy(model);
        }
      }
      else {
        fprintf(stderr, "Failed to load earthquake data from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}
