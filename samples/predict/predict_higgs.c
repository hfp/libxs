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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

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
    else if (0 == strncmp("rows", a, 4)) nrows = atoi(a + 4);
    else if (0 == strncmp("stride", a, 6)) stride = atoi(a + 6);
    else if (0 == strncmp("clusters", a, 8)) nclusters = atoi(a + 8);
    else if (0 == strncmp("order", a, 5)) order = atoi(a + 5);
    else if (0 == strncmp("split", a, 5)) split = atof(a + 5);
    else filename = a;
  }
  if (0 != help) {
    fprintf(stdout, "Usage: %s [file] [rows<N>] [stride<N>] [raw|hknn|rf|auto]\n"
      "         [clusters<N>] [order<N>] [split<F>] [refine]\n"
      "  HIGGS: 11M rows, 28 features, binary label (column 0). Get it from\n"
      "    https://archive.ics.uci.edu/dataset/280/higgs and gunzip it here.\n"
      "  rows<N>: entries to load (0: the whole file). Default 200000.\n"
      "  stride<N>: take every N-th row, so a subset spans the whole file\n"
      "    instead of being its first rows. 0 or 1 reads consecutively.\n"
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
          double dt_build, dt_eval, sum_conf = 0;
          int t, correct = 0, ntest = 0, build_ok = EXIT_FAILURE;
          int gated = 0, gated_correct = 0;
          libxs_predict_set_decompose(model, mode);
          libxs_predict_set_refine(model, refine);
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
                /**
                 * The gate is what a caller acts on: a prediction it accepts
                 * unchecked. Accuracy over every query and precision over the
                 * accepted ones answer different questions, and only the
                 * second says whether the confidence can be trusted.
                 */
                if (0.9 <= conf) {
                  ++gated;
                  if (0 != ok) ++gated_correct;
                }
                ++ntest;
              }
            }
            dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
            fprintf(stdout, "Decomposition: %d, clusters=%d, order=%d\n",
              q.decompose, q.nclusters, q.order);
            fprintf(stdout, "Scan per query: %d worst, %.0f average"
              " (of %d entries)\n", q.nscan, q.escan, q.nentries);
            fprintf(stdout, "Build: %.2f s, eval: %.2f s (%.3f ms per query)\n",
              dt_build, dt_eval,
              (0 < ntest) ? (1000.0 * dt_eval / ntest) : 0.0);
            fprintf(stdout, "Accuracy: %.2f%% of %d, mean confidence %.2f\n",
              (0 < ntest) ? (100.0 * correct / ntest) : 0.0, ntest,
              (0 < ntest) ? (sum_conf / ntest) : 0.0);
            if (0 < gated) {
              fprintf(stdout, "Gated (conf>=0.9): %.2f%% precision over %.1f%%"
                " of queries\n", 100.0 * gated_correct / gated,
                100.0 * gated / ntest);
            }
            result = EXIT_SUCCESS;
          }
          else {
            fprintf(stderr, "Build failed (decomposition %d)\n", mode);
          }
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
