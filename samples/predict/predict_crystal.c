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

enum { NFEAT = 37, NGATE = 16 };

static int gate_list(double gates[], int capacity);
static void gate_sweep(const double gates[], int ngates, int n,
  const double lconf[], const char lok[],
  const double xconf[], const char xok[]);


/**
 * Replace a fraction of the inputs with the absent value, deterministically per
 * row so a run is repeatable.  A quiet NaN is what the library reads as "not
 * known here"; it is produced rather than written as a literal so the file needs
 * no constant for it.
 */
static const char* mode_name(int decompose)
{
  static const char* names[] = { "RAW", "SPREAD", "PCA", "SETDIFF", "FISHER",
    "RF", "hKNN" };
  return (0 <= decompose && 7 > decompose) ? names[decompose] : "?";
}


/**
 * Save the model, load it back, and ask the loaded one for a different method.
 * The switch is what the per-cluster storage in the file pays for, so a model
 * built under a named method declines it: what comes back carries the forest
 * and no corpus. Reported rather than judged - the caller decides which of the
 * two outcomes it expected.
 */
static void switch_method(const libxs_predict_t* model, int decompose)
{
  size_t size = 0;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && 0 < size) {
    void* buffer = malloc(size);
    if (NULL != buffer
      && EXIT_SUCCESS == libxs_predict_save(model, buffer, &size))
    {
      libxs_predict_t* loaded = libxs_predict_load(buffer, size);
      if (NULL != loaded) {
        /* any method other than the one it was built with will do */
        const int other = (LIBXS_PREDICT_HKNN == decompose)
          ? LIBXS_PREDICT_RF : LIBXS_PREDICT_HKNN;
        libxs_predict_query_t ql;
        LIBXS_MEMZERO(&ql);
        libxs_predict_query(loaded, &ql);
        fprintf(stdout, "Reloaded: %d clusters, %d entries\n",
          ql.nclusters, ql.nentries);
        libxs_predict_set_decompose(loaded, other);
        if (EXIT_SUCCESS == libxs_predict_build(loaded, 0, 1, 0.0)) {
          libxs_predict_query(loaded, &ql);
          fprintf(stdout, "Method switch: %s to %s over %d entries\n",
            mode_name(decompose), mode_name(other), ql.nentries);
        }
        else {
          fprintf(stdout, "Method switch: declined by a model built as %s\n",
            mode_name(decompose));
        }
        libxs_predict_destroy(loaded);
      }
      else fprintf(stdout, "Method switch: the model did not load\n");
    }
    free(buffer);
  }
  else fprintf(stdout, "Method switch: the model did not save\n");
}


static void blank_inputs(double inputs[], int n, double fraction, unsigned int s)
{
  const volatile double zero = 0;
  const double absent = zero / zero;
  int i;
  for (i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    if (((double)((s >> 16) & 0x7fff) / 32767.0) < fraction) inputs[i] = absent;
  }
}


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  double split = 0.8, quality = 0, consistency = 0, gaps = 0;
  int order = 2, nclusters = 0;
  int decompose = LIBXS_PREDICT_AUTO_DECOMPOSE;
  int argi, npos = 0, use_xgb = 0, bad = 0, result = EXIT_FAILURE;
  for (argi = 2; argi < argc; ++argi) {
    const char* arg = argv[argi];
    if (0 != predict_isnum(arg)) {
      if (0 == npos) split = atof(arg);
      else if (1 == npos) order = atoi(arg);
      else if (2 == npos) nclusters = atoi(arg);
      else bad = argi;
      ++npos;
    }
    else if (0 != predict_keyval(arg, "consist", 0.9, &consistency)
      || 0 != predict_keyval(arg, "compress", 0.9, &quality)
      || 0 != predict_keyval(arg, "gaps", 0.1, &gaps))
    {
      /* the keyword that matched has already assigned its own value */
    }
    else if (0 != predict_iskey(arg, "fisher")) {
      decompose = LIBXS_PREDICT_FISHER;
    }
    else if (0 != predict_iskey(arg, "hknn")) decompose = LIBXS_PREDICT_HKNN;
    else if (0 != predict_iskey(arg, "setdiff")) {
      decompose = LIBXS_PREDICT_SETDIFF;
    }
    else if (0 != predict_iskey(arg, "rf")) decompose = LIBXS_PREDICT_RF;
    else if (0 != predict_iskey(arg, "none")) decompose = LIBXS_PREDICT_RAW;
    else if (0 != predict_iskey(arg, "pca")) decompose = LIBXS_PREDICT_PCA;
    else if (0 != predict_iskey(arg, "xgb")) use_xgb = 1;
    else bad = argi;
  }
  if (0 != bad) {
    fprintf(stderr, "Unrecognized argument \"%s\".\n", argv[bad]);
  }
  if (NULL == filename || 0 != bad) {
    fprintf(stdout,
      "Usage: %s <crystal_csv> [train_fraction] [order] [nclusters]"
      " [compress[Q]] [fisher|hknn|setdiff|rf|pca|none] [gaps[F]] [xgb]\n"
      "  Crystal system prediction from composition features.\n"
      "  xgb: also train XGBoost on the same split and compare.\n"
      "  Input: CSV with numeric features + crystal_system label (last col).\n"
      "  Crystal systems: 1=triclinic, 2=monoclinic, 3=orthorhombic,\n"
      "    4=tetragonal, 5=trigonal, 6=hexagonal, 7=cubic.\n"
      "  gaps[F]: blank that fraction of input values, to exercise the\n"
      "    absent-value path (libxs_predict_set_missing). Modes that cannot\n"
      "    carry a gap refuse to build rather than answer from a coordinate\n"
      "    they never had.\n"
      "  Default: the decomposition and the neighbour count are selected at\n"
      "    build from data held back for the purpose, rather than configured.\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
#if !defined(__XGBOOST)
  else if (0 != use_xgb) {
    fprintf(stderr, "Requested xgb but this binary was built without XGBoost:"
      " set XGBOOST_ROOT, or install the pkg-config module.\n");
  }
#endif
  else {
    libxs_predict_t* source = libxs_predict_create(NFEAT, 1);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        NULL, NULL, NULL, 0, NULL);
      if (0 < total) {
        const int train_end = LIBXS_MAX((int)(total * split + 0.5), 2);
        libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
        fprintf(stdout, "Loaded %d entries (%d features) from %s\n",
          total, NFEAT, filename);
        if (NULL != model) {
          libxs_timer_tick_t tick;
          int t, correct = 0, ntest = 0, gated = 0, gated_correct = 0;
          int build_ok = EXIT_FAILURE, swept = 0;
          double sum_conf = 0, dt_build, dt_eval;
          double gates[NGATE];
          const int ngates = gate_list(gates, NGATE);
          double* lconf = (double*)malloc((size_t)total * sizeof(double));
          char* lok = (char*)calloc((size_t)total, 1);
          libxs_predict_set_decompose(model, decompose);
          libxs_predict_set_neighbors(model, -1);
          if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
          if (0.0 < gaps) libxs_predict_set_missing(model, 1);
          for (t = 0; t < train_end; ++t) {
            double inputs[NFEAT], output;
            libxs_predict_get(source, t, inputs, &output);
            if (0.0 < gaps) blank_inputs(inputs, NFEAT, gaps, (unsigned int)t);
            libxs_predict_push(NULL, model, inputs, &output);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { const int br = libxs_predict_build_task(NULL, model, nclusters, order,
              quality, omp_get_thread_num(), omp_get_num_threads());
            if (0 == omp_get_thread_num()) build_ok = br;
          }
#else
          build_ok = libxs_predict_build(model, nclusters, order, quality);
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS != build_ok && 0.0 < gaps) {
            fprintf(stdout, "Refused: this decomposition cannot carry an absent"
              " input, and the corpus has %.0f%% of them\n", 100.0 * gaps);
          }
          if (EXIT_SUCCESS == build_ok && NULL != lconf && NULL != lok) {
            libxs_predict_query_t qi;
            LIBXS_MEMZERO(&qi);
            libxs_predict_query(model, &qi);
            fprintf(stdout, "Train=%d, Test=%d\n", qi.nentries, total - train_end);
            fprintf(stdout, "Decomposition: %s (%s)\n", mode_name(qi.decompose),
              (LIBXS_PREDICT_AUTO_DECOMPOSE == decompose)
                ? "selected at build" : "requested");
            fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
              " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
            tick = libxs_timer_tick();
            for (t = train_end; t < total; ++t) {
              double inputs[NFEAT], predicted;
              libxs_predict_info_t info;
              libxs_predict_get(source, t, inputs, NULL);
              libxs_predict_eval(NULL, model, inputs, &predicted, &info, 1);
              { int label, ok;
                double expected;
                const double conf = (NULL != info.confidence)
                  ? info.confidence[0] : 0.0;
                libxs_predict_get(source, t, NULL, &expected);
                label = LIBXS_ROUNDX(int, expected);
                ok = (LIBXS_ROUNDX(int, predicted) == label);
                if (0 != ok) ++correct;
                if (conf >= gates[0]) {
                  ++gated;
                  if (0 != ok) ++gated_correct;
                }
                lconf[ntest] = conf;
                lok[ntest] = (char)ok;
                sum_conf += conf;
              }
              ++ntest;
            }
            dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
            if (0 < ntest) {
              fprintf(stdout, "Accuracy: %d/%d = %.1f%%\n",
                correct, ntest, 100.0 * correct / ntest);
              fprintf(stdout, "Confidence-gated (>=%.2f): %d/%d = %.1f%%"
                " (coverage %.1f%%)\n", gates[0],
                gated_correct, gated,
                (0 < gated) ? 100.0 * gated_correct / gated : 0.0,
                100.0 * gated / ntest);
              fprintf(stdout, "Avg confidence: %.3f\n", sum_conf / ntest);
              fprintf(stdout, "Eval: %d queries (%.2f s)\n", ntest, dt_eval);
#if defined(__XGBOOST)
              if (0 != use_xgb) {
                double* xgb_pred = (double*)malloc(
                  (size_t)total * sizeof(double));
                double* xgb_conf = (double*)malloc(
                  (size_t)total * sizeof(double));
                char* mask = (char*)calloc((size_t)total, 1);
                char* xok = (char*)calloc((size_t)total, 1);
                int classify = 1, task = 0;
                if (NULL != xgb_pred && NULL != xgb_conf && NULL != mask
                  && NULL != xok)
                {
                  for (t = 0; t < train_end; ++t) mask[t] = 1;
                  if (EXIT_SUCCESS == predict_xgb(source, total, NFEAT, 1,
                    mask, &classify, xgb_pred, xgb_conf, &task, NULL))
                  {
                    int xcorrect = 0, xgated = 0, xgated_correct = 0;
                    double xsum_conf = 0;
                    for (t = train_end; t < total; ++t) {
                      double expected;
                      int label, ok;
                      libxs_predict_get(source, t, NULL, &expected);
                      label = LIBXS_ROUNDX(int, expected);
                      ok = (LIBXS_ROUNDX(int, xgb_pred[t]) == label);
                      if (0 != ok) ++xcorrect;
                      if (xgb_conf[t] >= gates[0]) {
                        ++xgated;
                        if (0 != ok) ++xgated_correct;
                      }
                      xok[t - train_end] = (char)ok;
                      xsum_conf += xgb_conf[t];
                    }
                    fprintf(stdout, "XGBoost (%d classes, rounds=%i, depth=%i,"
                      " eta=%g):\n", task,
                      predict_xgb_geti("XGB_ROUNDS", 200),
                      predict_xgb_geti("XGB_DEPTH", 6),
                      predict_xgb_getd("XGB_ETA", 0.1));
                    fprintf(stdout, "  Accuracy: %d/%d = %.1f%%\n",
                      xcorrect, ntest, 100.0 * xcorrect / ntest);
                    fprintf(stdout, "  Confidence-gated (>=%.2f): %d/%d ="
                      " %.1f%% (coverage %.1f%%)\n", gates[0],
                      xgated_correct, xgated,
                      (0 < xgated) ? 100.0 * xgated_correct / xgated : 0.0,
                      100.0 * xgated / ntest);
                    fprintf(stdout, "  Avg confidence: %.3f\n",
                      xsum_conf / ntest);
                    if (1 < ngates) {
                      gate_sweep(gates, ngates, ntest, lconf, lok,
                        xgb_conf + train_end, xok);
                      swept = 1;
                    }
                  }
                }
                free(xok);
                free(mask);
                free(xgb_conf);
                free(xgb_pred);
              }
#endif
              if (1 < ngates && 0 == swept) {
                gate_sweep(gates, ngates, ntest, lconf, lok, NULL, NULL);
              }
            }
            /**
             * A stored model can be evaluated by a method other than the one it
             * was built with, but only where the selector chose that method: a
             * caller who names it is served a model carrying the forest alone,
             * with no corpus to rebuild anything else from. Behind TEST because
             * it is a property of the format rather than something a user of
             * this sample asks it to do, and tests/predict.sh reads the verdict.
             */
            if (NULL != getenv("TEST")) {
              switch_method(model, qi.decompose);
            }
            result = EXIT_SUCCESS;
          }
          free(lok);
          free(lconf);
          libxs_predict_destroy(model);
        }
      }
      else {
        fprintf(stderr, "Failed to load crystal data from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}


/**
 * Gate thresholds from GATE (comma-separated, ascending or not).  The first
 * entry drives the single-threshold report, so a one-element list keeps the
 * historical output; more than one additionally traces precision against
 * coverage, which is what separates a better-calibrated signal from a
 * differently-scaled one.
 */
static int gate_list(double gates[], int capacity)
{
  const char* const env = getenv("GATE");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    int len = 0;
    const char* token = libxs_strtoken(env, ",", result, &len);
    while (NULL != token && result < capacity) {
      gates[result++] = atof(token);
      token = libxs_strtoken(env, ",", result, &len);
    }
  }
  if (0 == result) {
    gates[0] = 0.9;
    result = 1;
  }
  return result;
}


static void gate_sweep(const double gates[], int ngates, int n,
  const double lconf[], const char lok[],
  const double xconf[], const char xok[])
{
  int g;
  fprintf(stdout, "Gate sweep (%d queries):\n", n);
  fprintf(stdout, (NULL != xconf)
    ? "  gate  libxs-prec  libxs-cov    xgb-prec    xgb-cov\n"
    : "  gate  libxs-prec  libxs-cov\n");
  for (g = 0; g < ngates; ++g) {
    int lacted = 0, lcorrect = 0, xacted = 0, xcorrect = 0, i;
    for (i = 0; i < n; ++i) {
      if (lconf[i] >= gates[g]) {
        ++lacted;
        if (0 != lok[i]) ++lcorrect;
      }
      if (NULL != xconf && xconf[i] >= gates[g]) {
        ++xacted;
        if (0 != xok[i]) ++xcorrect;
      }
    }
    fprintf(stdout, "  %.2f     %6.1f%%     %6.1f%%", gates[g],
      (0 < lacted) ? 100.0 * lcorrect / lacted : 0.0,
      (0 < n) ? 100.0 * lacted / n : 0.0);
    if (NULL != xconf) {
      fprintf(stdout, "      %6.1f%%     %6.1f%%",
        (0 < xacted) ? 100.0 * xcorrect / xacted : 0.0,
        (0 < n) ? 100.0 * xacted / n : 0.0);
    }
    fprintf(stdout, "\n");
  }
}
