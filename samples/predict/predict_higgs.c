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

#if defined(__XGBOOST)
# include "predict_xgb.h"
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#include "predict_gate.h"

#define NGATE 16
#define NFEAT 28
#define CSVFILE "HIGGS.csv"

/* column 0 is the label, columns 1..28 are the features */
#define HIGGS_INPUTS "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," \
  "21,22,23,24,25,26,27,28"
#define HIGGS_OUTPUT "0"


int main(int argc, char* argv[])
{
  const char* filename = CSVFILE;
  int nrows = 200000, stride = 0, mode = LIBXS_PREDICT_HKNN, refine = 0;
  int nclusters = 0, order = 1, help = 0, i;
  int depth = 0, ntrees = 0, use_xgb = 0;
  double split = 0.8;
  int result = EXIT_FAILURE;
  for (i = 1; i < argc; ++i) {
    const char* const a = argv[i];
    if (0 == strcmp("-h", a) || 0 == strcmp("--help", a)) help = 1;
    else if (0 == strcmp("raw", a)) mode = LIBXS_PREDICT_RAW;
    else if (0 == strcmp("hknn", a)) mode = LIBXS_PREDICT_HKNN;
    else if (0 == strcmp("rf", a)) mode = LIBXS_PREDICT_RF;
    else if (0 == strcmp("auto", a)) mode = LIBXS_PREDICT_AUTO_DECOMPOSE;
    else if (0 == strcmp("refine", a)) refine = -1;
    else if (0 == strcmp("xgb", a)) use_xgb = 1;
    else if (0 == strncmp("rows", a, 4)) nrows = atoi(a + 4);
    else if (0 == strncmp("stride", a, 6)) stride = atoi(a + 6);
    else if (0 == strncmp("clusters", a, 8)) nclusters = atoi(a + 8);
    else if (0 == strncmp("depth", a, 5)) depth = atoi(a + 5);
    else if (0 == strncmp("trees", a, 5)) ntrees = atoi(a + 5);
    else if (0 == strncmp("order", a, 5)) order = atoi(a + 5);
    else if (0 == strncmp("split", a, 5)) split = atof(a + 5);
    else filename = a;
  }
#if !defined(__XGBOOST)
  if (0 != use_xgb) {
    fprintf(stderr, "Requested xgb but this binary was built without XGBoost:"
      " set XGBOOST_ROOT and rebuild.\n");
    use_xgb = 0;
  }
#endif
  if (0 != help) {
    fprintf(stdout, "Usage: %s [file] [rows<N>] [stride<N>] [raw|hknn|rf|auto]\n"
      "         [clusters<N>] [order<N>] [split<F>] [refine]\n"
      "  HIGGS: 11M rows, 28 features, binary label (column 0). Get it from\n"
      "    https://archive.ics.uci.edu/dataset/280/higgs and gunzip it here.\n"
      "  rows<N>: entries to load (0: the whole file). Default 200000.\n"
      "  stride<N>: take every N-th row, so a subset spans the whole file\n"
      "    instead of being its first rows. 0 or 1 reads consecutively.\n"
      "  xgb: also train XGBoost on the same split, for comparison.\n"
      "  depth<N>/trees<N>: forest depth and tree count (0: derived).\n"
      "  order<N>: polynomial order. The label is discrete, so nothing is\n"
      "    interpolated and the order is immaterial - it is pinned to 1 to\n"
      "    skip the search over it, which would rebuild the model per order.\n"
      "  refine: re-enable the confidence-gated refinement pass. It inverts\n"
      "    through the corpus, which is a scan of every entry per query, so\n"
      "    it is off here and eval cost stays with the cluster, not the\n"
      "    corpus. It also cannot discriminate on a discrete-only output.\n"
      "  Default: hknn, which partitions by Gini on the label rather than by\n"
      "    k-means, and costs one pass instead of a hundred Lloyd iterations.\n", argv[0]);
    result = EXIT_SUCCESS;
  }
  else {
    libxs_predict_t* source = libxs_predict_create(NFEAT, 1);
    if (NULL != source) {
      libxs_predict_csv_t opts;
      libxs_timer_tick_t tick = libxs_timer_tick();
      int total;
      memset(&opts, 0, sizeof(opts));
      opts.delims = ",";
      opts.inputs = HIGGS_INPUTS;
      opts.outputs = HIGGS_OUTPUT;
      opts.nrows = nrows;
      opts.stride = stride;
      total = libxs_predict_load_csv_opts(source, filename, &opts);
      if (0 < total) {
        const double dt_load = libxs_timer_duration(tick, libxs_timer_tick());
        const int train_end = LIBXS_MAX((int)(total * split + 0.5), 2);
        libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
        fprintf(stdout, "Loaded %d entries (%d features) from %s in %.2f s\n",
          total, NFEAT, filename, dt_load);
        if (NULL != model) {
          double in[NFEAT], out[1];
          libxs_predict_query_t q;
          double dt_build, dt_eval, dt_batch = 0, sum_conf = 0;
          int t, correct = 0, ntest = 0, build_ok = EXIT_FAILURE;
          int gated = 0, gated_correct = 0, swept = 0;
          double gates[NGATE];
          const int ngates = gate_list(gates, NGATE);
          /* per query, so precision can be read against coverage rather than at
             one threshold whose meaning moves with how confidence is formed */
          double* lconf = (double*)malloc((size_t)total * sizeof(double));
          double* lpred = (double*)malloc((size_t)total * sizeof(double));
          char* lok = (char*)calloc((size_t)total, 1);
          libxs_predict_set_decompose(model, mode);
          libxs_predict_set_refine(model, refine);
          if (0 != depth || 0 != ntrees) {
            libxs_predict_set_forest(model, ntrees, depth);
          }
          for (t = 0; t < train_end; ++t) {
            libxs_predict_get(source, t, in, out);
            libxs_predict_push(NULL, model, in, out);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { const int br = libxs_predict_build_task(NULL, model, nclusters,
              order, 0, omp_get_thread_num(), omp_get_num_threads());
            if (0 == omp_get_thread_num()) build_ok = br;
          }
#else
          build_ok = libxs_predict_build(model, nclusters, order, 0);
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS == build_ok) {
            libxs_predict_query(model, &q);
            tick = libxs_timer_tick();
            for (t = train_end; t < total; ++t) {
              double pred[1];
              libxs_predict_info_t info;
              libxs_predict_get(source, t, in, out);
              libxs_predict_eval(NULL, model, in, pred, &info, 0);
              { const int ok = (0.5 > LIBXS_ABS(pred[0] - out[0]));
                const double conf = (NULL != info.confidence)
                  ? info.confidence[0] : 0;
                if (0 != ok) ++correct;
                sum_conf += conf;
                /* precision over accepted predictions, not over all */
                if (gates[0] <= conf) {
                  ++gated;
                  if (0 != ok) ++gated_correct;
                }
                if (NULL != lconf && NULL != lok) {
                  lconf[ntest] = conf;
                  lok[ntest] = (char)(0 != ok);
                }
                if (NULL != lpred) lpred[ntest] = pred[0];
                ++ntest;
              }
            }
            dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
            /**
             * The same queries again through the batch form. The loop above is
             * what a caller reading confidence has to do, since the batch form
             * returns outputs alone - but it is one query at a time on one
             * thread, so it is not what compares against a library predicting a
             * whole matrix at once. Both are therefore reported.
             *
             * It also compares the two answers, which is the only place the
             * batch path is exercised against the per-query path under threads.
             */
            { double* bin = (double*)malloc((size_t)ntest * NFEAT
                * sizeof(double));
              double* bout = (double*)malloc((size_t)ntest * sizeof(double));
              if (NULL != bin && NULL != bout && 0 < ntest) {
                int differ = 0;
                for (t = 0; t < ntest; ++t) {
                  libxs_predict_get(source, train_end + t,
                    bin + (size_t)t * NFEAT, NULL);
                }
                tick = libxs_timer_tick();
#if defined(_OPENMP)
#               pragma omp parallel
                { libxs_predict_eval_batch_task(model, bin, bout, ntest, 0,
                    omp_get_thread_num(), omp_get_num_threads());
                }
#else
                libxs_predict_eval_batch(model, bin, bout, ntest, 0);
#endif
                dt_batch = libxs_timer_duration(tick, libxs_timer_tick());
                if (NULL != lpred) {
                  for (t = 0; t < ntest; ++t) {
                    if (bout[t] != lpred[t]) ++differ;
                  }
                  if (0 != differ) {
                    fprintf(stderr, "Batch differs from per-query on %d of"
                      " %d queries\n", differ, ntest);
                  }
                }
              }
              free(bout);
              free(bin);
            }
            fprintf(stdout, "Decomposition: %d, clusters=%d, order=%d\n",
              q.decompose, q.nclusters, q.order);
            fprintf(stdout, "Scan per query: %d worst, %.0f average"
              " (of %d entries)\n", q.nscan, q.escan, q.nentries);
            fprintf(stdout, "Build: %.2f s, eval: %.2f s (%.3f ms per query)\n",
              dt_build, dt_eval,
              (0 < ntest) ? (1000.0 * dt_eval / ntest) : 0.0);
            if (0 < dt_batch && 0 < ntest) {
              fprintf(stdout, "Batched eval: %.2f s (%.2f us per row over"
                " %d threads)\n", dt_batch, 1e6 * dt_batch / ntest,
#if defined(_OPENMP)
                omp_get_max_threads());
#else
                1);
#endif
            }
            fprintf(stdout, "Accuracy: %.2f%% of %d, mean confidence %.2f\n",
              (0 < ntest) ? (100.0 * correct / ntest) : 0.0, ntest,
              (0 < ntest) ? (sum_conf / ntest) : 0.0);
            if (0 < gated) {
              fprintf(stdout, "Gated (conf>=%.2f): %.2f%% precision over %.1f%%"
                " of queries\n", gates[0], 100.0 * gated_correct / gated,
                100.0 * gated / ntest);
            }
#if defined(__XGBOOST)
            if (0 != use_xgb) {
              double* xp = (double*)malloc((size_t)total * sizeof(double));
              double* xc = (double*)malloc((size_t)total * sizeof(double));
              char* mask = (char*)calloc((size_t)total, 1);
              int classify = 1, task = 0;
              predict_xgb_time_t xtime;
              if (NULL != xp && NULL != xc && NULL != mask) {
                libxs_timer_tick_t xt = libxs_timer_tick();
                for (t = 0; t < train_end; ++t) mask[t] = 1;
                if (EXIT_SUCCESS == predict_xgb(source, total, NFEAT, 1,
                  mask, &classify, xp, xc, &task, NULL, &xtime))
                {
                  const double dt_xgb =
                    libxs_timer_duration(xt, libxs_timer_tick());
                  int xok = 0, xg = 0, xgok = 0;
                  char* xokv = (char*)calloc((size_t)total, 1);
                  for (t = train_end; t < total; ++t) {
                    double expected;
                    int ok;
                    libxs_predict_get(source, t, NULL, &expected);
                    ok = (LIBXS_ROUNDX(int, xp[t])
                      == LIBXS_ROUNDX(int, expected));
                    if (0 != ok) ++xok;
                    if (gates[0] <= xc[t]) { ++xg; if (0 != ok) ++xgok; }
                    if (NULL != xokv) xokv[t - train_end] = (char)(0 != ok);
                  }
                  /* less what the per-query probe below cost, which is measured
                     inside the same call and is not part of the comparison */
                  fprintf(stdout, "XGBoost: rounds=%i depth=%i eta=%g, %.2f s"
                    " of which marshal %.2f, train %.2f, predict %.2f\n",
                    predict_xgb_geti("XGB_ROUNDS", 200),
                    predict_xgb_geti("XGB_DEPTH", 6),
                    predict_xgb_getd("XGB_ETA", 0.1), dt_xgb - xtime.query,
                    xtime.marshal, xtime.train, xtime.predict);
                  /* what compares against what: a build against the rounds it
                     is the counterpart of, not against the whole call */
                  fprintf(stdout, "Build vs train: %.2f s vs %.2f s\n",
                    dt_build, xtime.train);
                  if (0 < xtime.nquery && 0 < ntest) {
                    fprintf(stdout, "Per query: %.1f us here, %.1f us there;"
                      " batched %.2f us here, %.2f us there\n",
                      1e6 * dt_eval / ntest,
                      1e6 * xtime.query / xtime.nquery,
                      (0 < dt_batch) ? (1e6 * dt_batch / ntest) : 0.0,
                      (0 < total) ? (1e6 * xtime.predict / total) : 0.0);
                  }
                  fprintf(stdout, "XGBoost accuracy: %.2f%% of %d\n",
                    (0 < ntest) ? (100.0 * xok / ntest) : 0.0, ntest);
                  if (0 < xg) {
                    fprintf(stdout, "XGBoost gated (conf>=%.2f): %.2f%%"
                      " precision over %.1f%% of queries\n", gates[0],
                      100.0 * xgok / xg, 100.0 * xg / ntest);
                  }
                  if (1 < ngates && NULL != lconf && NULL != lok
                    && NULL != xokv)
                  {
                    gate_sweep(gates, ngates, ntest, lconf, lok,
                      xc + train_end, xokv);
                    swept = 1;
                  }
                  free(xokv);
                }
                else fprintf(stderr, "XGBoost failed\n");
              }
              free(xp); free(xc); free(mask);
            }
#endif
            if (1 < ngates && 0 == swept && NULL != lconf && NULL != lok) {
              gate_sweep(gates, ngates, ntest, lconf, lok, NULL, NULL);
            }
            result = EXIT_SUCCESS;
          }
          else {
            fprintf(stderr, "Build failed (decomposition %d)\n", mode);
          }
          free(lok);
          free(lpred);
          free(lconf);
          libxs_predict_destroy(model);
        }
      }
      else {
        fprintf(stderr, "Failed to load %s: pass a path, or see --help"
          " for where to get it.\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}
