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
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_timer.h>

#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__XGBOOST)
# include "predict_xgb.h"
#endif
#include "predict_args.h"

static const char input_names[] = "M,N,K";
static const char output_names[] =
  "BS,BM,BN,BK,WS,WG,LU,NZ,AL,TB,TC,AP,AA,AB,AC,XF";

enum { NINPUTS = 3, NOUTPUTS = 16 };

static const int confidence_outputs[] = { 5, 6, 8, 12, 13 };

static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal, const char trained[],
  int use_xgb);
static int write_confidence_maps(const char* prefix, const void* buffer,
  size_t size, const libxs_predict_t* reference, int ntotal);
static double deployment_confidence(const libxs_predict_info_t* info);


/**
 * Confidence thresholds to report, from GATE (comma separated). The first
 * drives the gated table; more than one adds a sweep, because precision at one
 * threshold is a single point on a curve and two modes rarely sit at the same
 * coverage there.
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


static const char* mode_name(int decompose)
{
  static const char* names[] = { "RAW", "SPREAD", "PCA", "SETDIFF", "FISHER",
    "RF", "hKNN" };
  return (0 <= decompose && 7 > decompose) ? names[decompose] : "?";
}


int main(int argc, char* argv[])
{
  int argi = 1, mode = LIBXS_PREDICT_AUTO, use_rf = 0, use_hknn = 0;
  int order_arg = 0, shuffle_split = 0, use_xgb = 0;
  double quality = 0, smooth = 0, consistency = 0, quantile = 0;
  double eval_fraction = 0.8;
  const char *filename, *modelfile, *confidence_prefix;
  int result = EXIT_FAILURE;
  /**
   * A keyword is recognized wherever it appears and the remaining tokens keep
   * their order as file names. Position used to carry the distinction - all
   * keywords first, then the files - which reads a keyword written after a file
   * name as another file name: `... corpus.csv xgb` silently SAVED A MODEL to a
   * file called `xgb` and ran without the comparison it was asked for. Nothing
   * needed position to be told apart, because `predict_iskey` matches a whole
   * word and never reads `results.csv` as `rf`.
   *
   * A file genuinely named after a keyword is the case this gives up, and it is
   * the rarer accident of the two; `./file` names it unambiguously.
   */
  { const char* positional[3];
    int npos = 0;
    for (; argi < argc; ++argi) {
      const char* arg = argv[argi];
      if (0 != predict_isnum(arg)) eval_fraction = atof(arg);
      else if (0 != predict_iskey(arg, "auto")) mode = LIBXS_PREDICT_AUTO;
      else if (0 != predict_iskey(arg, "cat")) mode = LIBXS_PREDICT_CLASSIFY;
      else if (0 != predict_iskey(arg, "interp")) {
        mode = LIBXS_PREDICT_INTERPOLATE;
      }
      else if (0 != predict_iskey(arg, "mix")) shuffle_split = 1;
      else if (0 != predict_iskey(arg, "rf")) use_rf = 1;
      else if (0 != predict_iskey(arg, "hknn")) use_hknn = 1;
      else if (0 != predict_iskey(arg, "xgb")) use_xgb = 1;
      else if (0 != predict_keyval(arg, "compress", 0.9, &quality)
        || 0 != predict_keyval(arg, "consist", 0.9, &consistency)
        || 0 != predict_keyval(arg, "quantile", 0.1, &quantile)
        || 0 != predict_keyval(arg, "smooth", -1.0, &smooth))
      {
        /* the keyword that matched has already assigned its own value */
      }
      else if ('-' == arg[0] && '\0' != arg[1] && 0 == order_arg) {
        order_arg = atoi(arg);
      }
      else if (npos < (int)(sizeof(positional) / sizeof(*positional))) {
        positional[npos++] = arg;
      }
      else {
        fprintf(stderr, "Ignored extra argument \"%s\"\n", arg);
      }
    }
    filename = (0 < npos) ? positional[0] : NULL;
    modelfile = (1 < npos) ? positional[1] : NULL;
    confidence_prefix = (2 < npos) ? positional[2] : NULL;
  }
  { static char modelpath[512];
    if (NULL == modelfile && NULL != filename) {
      const char* sep = strrchr(filename, '/');
      const char* base = (NULL != sep) ? (sep + 1) : filename;
      const char* dot = strrchr(base, '.');
      size_t len = (NULL != dot) ? (size_t)(dot - base) : strlen(base);
      if (len >= sizeof(modelpath) - 4) len = sizeof(modelpath) - 5;
      memcpy(modelpath, base, len);
      memcpy(modelpath + len, ".bin", 5);
      modelfile = modelpath;
    }
  }
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s [fraction] [auto|cat|compress[Q]|consist[C]|interp|quantile[Q]|rf|hknn|smooth[A]|xgb]"
      " [-N] <csvfile> [modelfile [confidence-prefix]]\n"
      "  fraction: validation split 0..1 for quality report (default: 0.8)\n"
      "  auto:     auto-detect mode per output (default)\n"
      "  cat:      force categorical (kNN) for all outputs\n"
      "  compress: drop predictable entries (Q: threshold, default 0.9)\n"
      "  consist:  round-trip consistency penalty (C: 0..1, default 0.9)\n"
      "  interp:   force interpolation for all outputs\n"
      "  mix:      draw the validation split by co-prime stride rather than\n"
      "            as a prefix (a CSV sorted by shape otherwise makes the\n"
      "            held-out part systematically larger shapes)\n"
      "  quantile: prediction intervals (Q: level 0..0.5, default 0.1)\n"
      "  rf:       Random Forest classification\n"
      "  hknn:     hierarchical kNN (Fisher-guided partition)\n"
      "  smooth:   multi-cluster blending (A: radius or -1=auto, default: auto)\n"
      "  xgb:      also train XGBoost on the same split and compare\n"
      "  -N: max polynomial order (default: 0 = auto)\n"
      "  confidence-prefix: optional prefix for saved-model confidence maps\n"
      "  Trains on all entries, saves the model, and reports\n"
      "  quality on a held-out validation set.\n", argv[0]);
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
      const int ntotal = libxs_predict_load_csv(source, filename, NULL,
        input_names, output_names, NULL, 0, NULL);
      if (0 < ntotal) {
        libxs_predict_t* model = libxs_predict_create(NINPUTS, NOUTPUTS);
        fprintf(stdout, "Loaded %d entries from %s\n", ntotal, filename);
        if (NULL != model) {
          libxs_timer_tick_t tick;
          int i, build_ok = EXIT_FAILURE;
          double inputs[NINPUTS], outputs[NOUTPUTS], dt_build;
          libxs_predict_set_mode(model, mode);
          if (0 != use_rf) libxs_predict_set_decompose(model, LIBXS_PREDICT_RF);
          else if (0 != use_hknn) libxs_predict_set_decompose(model, LIBXS_PREDICT_HKNN);
          if (0.0 != smooth) libxs_predict_set_smooth(model, smooth);
          if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
          if (0.0 != quantile) libxs_predict_set_quantile(model, quantile);
          for (i = 0; i < ntotal; ++i) {
            libxs_predict_get(source, i, inputs, outputs);
            libxs_predict_push(NULL, model, inputs, outputs);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { const int br = libxs_predict_build_task(model, 0, order_arg,
              quality, omp_get_thread_num(), omp_get_num_threads());
            if (0 == omp_get_thread_num()) build_ok = br;
          }
#else
          build_ok = libxs_predict_build_task(model, 0, order_arg,
            quality, 0, 1); /* one task: no rendezvous to hold */
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS == build_ok) {
            { libxs_predict_query_t qi;
              memset(&qi, 0, sizeof(qi));
              libxs_predict_query(model, &qi);
              fprintf(stdout, "Decomposition: %s (%s)\n",
                mode_name(qi.decompose),
                (0 == use_rf && 0 == use_hknn)
                  ? "selected at build" : "requested");
              fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
                " (%d entries, %.2f s)\n", qi.nclusters, qi.compression,
                qi.order, qi.nentries, dt_build);
            }
            { const int nval = LIBXS_MAX(
                (int)(ntotal * eval_fraction + 0.5), 1);
              libxs_predict_t* val_model =
                libxs_predict_create(NINPUTS, NOUTPUTS);
              char* trained = (char*)calloc((size_t)ntotal, 1);
              if (NULL != val_model && NULL != trained) {
                double vi[NINPUTS], vo[NOUTPUTS];
                libxs_predict_set_mode(val_model, mode);
                if (0 != use_rf) {
                  libxs_predict_set_decompose(val_model, LIBXS_PREDICT_RF);
                }
                else if (0 != use_hknn) {
                  libxs_predict_set_decompose(val_model, LIBXS_PREDICT_HKNN);
                }
                if (0.0 != smooth) libxs_predict_set_smooth(val_model, smooth);
                if (0.0 != consistency) {
                  libxs_predict_set_consistency(val_model, consistency);
                }
                if (0 != shuffle_split) {
                  const size_t co = libxs_coprime2((size_t)ntotal);
                  for (i = 0; i < nval; ++i) {
                    trained[LIBXS_SHUFFLE_INDEX(i, ntotal, co, 0)] = 1;
                  }
                }
                else {
                  for (i = 0; i < nval; ++i) trained[i] = 1;
                }
                for (i = 0; i < ntotal; ++i) {
                  if (0 != trained[i]) {
                    libxs_predict_get(source, i, vi, vo);
                    libxs_predict_push(NULL, val_model, vi, vo);
                  }
                }
                if (EXIT_SUCCESS == libxs_predict_build(
                  val_model, 0, order_arg, quality))
                {
                  evaluate(val_model, source, ntotal, trained, use_xgb);
                }
              }
              libxs_predict_destroy(val_model);
              free(trained);
            }
            { size_t size = 0;
              void* buffer;
              libxs_predict_save(model, NULL, &size);
              buffer = malloc(size);
              if (NULL != buffer) {
                if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &size)) {
                  FILE* out = fopen(modelfile, "wb");
                  if (NULL != out) {
                    fwrite(buffer, 1, size, out);
                    fclose(out);
                    fprintf(stdout, "Saved model to %s (%lu bytes)\n",
                      modelfile, (unsigned long)size);
                    if (NULL == confidence_prefix || EXIT_SUCCESS ==
                      write_confidence_maps(confidence_prefix, buffer, size,
                        source, ntotal))
                    {
                      result = EXIT_SUCCESS;
                    }
                  }
                }
                free(buffer);
              }
            }
          }
          libxs_predict_destroy(model);
        }
      }
      else {
        fprintf(stderr, "Failed to load entries from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}


static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal, const char trained[],
  int use_xgb)
{
  double novel_cov[5];
  double* all_inputs = (double*)malloc((size_t)ntotal * NINPUTS * sizeof(double));
  double* all_predicted = (double*)malloc((size_t)ntotal * NOUTPUTS * sizeof(double));
  LIBXS_UNUSED(use_xgb);
  if (NULL != all_inputs && NULL != all_predicted) {
    double maxerr[NOUTPUTS] = { 0 }, sumerr[NOUTPUTS] = { 0 };
    int interp[NOUTPUTS];
    libxs_timer_tick_t tick;
    double dt_eval;
    int i, j;
    memset(interp, 0, sizeof(interp));
    for (i = 0; i < ntotal; ++i) {
      libxs_predict_get(reference, i, all_inputs + (size_t)i * NINPUTS, NULL);
    }
    tick = libxs_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel
    { const int tid = omp_get_thread_num(), ntasks = omp_get_num_threads();
      libxs_predict_eval_batch_task(model, all_inputs, all_predicted,
        ntotal, 1, tid, ntasks);
    }
#else
    libxs_predict_eval_batch(model, all_inputs, all_predicted, ntotal, 1);
#endif
    dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
    for (i = 0; i < ntotal; ++i) {
      double expected[NOUTPUTS];
      libxs_predict_get(reference, i, NULL, expected);
      for (j = 0; j < NOUTPUTS; ++j) {
        const double err = LIBXS_DELTA(
          all_predicted[(size_t)i * NOUTPUTS + j], expected[j]);
        sumerr[j] += err;
        if (err > maxerr[j]) maxerr[j] = err;
      }
    }
    fprintf(stdout, "Validation (%d samples):\n", ntotal);
    fprintf(stdout, "  param   avg-err   max-err\n");
    for (j = 0; j < NOUTPUTS; ++j) {
      int len = 0;
      const char* name = libxs_strtoken(output_names, ",", j, &len);
      fprintf(stdout, "  %-4.*s  %9.2e %9.2e\n",
        len, name,
        (0 < ntotal) ? (sumerr[j] / ntotal) : 0.0, maxerr[j]);
    }
    fprintf(stdout, "Eval: %d queries (%.2f s)\n", ntotal, dt_eval);
    /**
     * carried out of the gate block so the XGBoost comparison below can be read
     * at the coverage this model actually reached, not at the same gate value
     */
    for (j = 0; j < 5; ++j) novel_cov[j] = -1.0;
    { const int nconf = (int)(sizeof(confidence_outputs)
        / sizeof(confidence_outputs[0]));
      double gates[8];
      const int ngates = gate_list(gates, 8);
      const double threshold = gates[0];
      double* swconf = (double*)malloc((size_t)ntotal * nconf * sizeof(double));
      char* swok = (char*)malloc((size_t)ntotal * nconf);
      int gated_correct[5] = {0}, gated_wrong[5] = {0}, deferred[5] = {0};
      int split_acted[5][2], split_correct[5][2], split_n[2];
      int ci;
      memset(split_acted, 0, sizeof(split_acted));
      memset(split_correct, 0, sizeof(split_correct));
      split_n[0] = split_n[1] = 0;
      for (i = 0; i < ntotal; ++i) {
        /**
         * An entry the model was built from is recalled, not predicted.  The
         * split separates the two so a mechanism cannot be credited for moving
         * only the attested bucket.
         */
        const int novel = (0 == trained[i]) ? 1 : 0;
        double expected[NOUTPUTS];
        libxs_predict_info_t info;
        libxs_predict_eval(NULL, model,
          all_inputs + (size_t)i * NINPUTS, NULL, &info, 1);
        libxs_predict_get(reference, i, NULL, expected);
        ++split_n[novel];
        if (NULL != info.interpolated) {
          for (j = 0; j < NOUTPUTS; ++j) {
            if (0 != info.interpolated[j]) ++interp[j];
          }
        }
        for (ci = 0; ci < nconf; ++ci) {
          const int oi = confidence_outputs[ci];
          if (NULL != swconf && NULL != swok) {
            swconf[(size_t)i * nconf + ci] = (NULL != info.confidence)
              ? info.confidence[oi] : 0.0;
            swok[(size_t)i * nconf + ci] = (char)((NULL != info.values
              && LIBXS_ROUNDX(int, info.values[oi]) == (int)expected[oi])
                ? 1 : 0);
          }
          if (NULL != info.confidence && info.confidence[oi] >= threshold) {
            const int ok = (NULL != info.values
              && LIBXS_ROUNDX(int, info.values[oi]) == (int)expected[oi]);
            ++split_acted[ci][novel];
            if (0 != ok) {
              ++gated_correct[ci];
              ++split_correct[ci][novel];
            }
            else {
              ++gated_wrong[ci];
            }
          }
          else {
            ++deferred[ci];
          }
        }
      }
      fprintf(stdout, "Gated deployment (threshold=%.1f):\n", threshold);
      fprintf(stdout,
        "  param  correct  wrong  deferred  coverage  precision\n");
      for (ci = 0; ci < nconf; ++ci) {
        const int oi = confidence_outputs[ci];
        const int acted = gated_correct[ci] + gated_wrong[ci];
        int len = 0;
        const char* name = libxs_strtoken(output_names, ",", oi, &len);
        fprintf(stdout, "  %-4.*s  %5d  %5d  %5d     %5.1f%%    %5.1f%%\n",
          len, name, gated_correct[ci], gated_wrong[ci], deferred[ci],
          (ntotal > 0) ? 100.0 * acted / ntotal : 0.0,
          (acted > 0) ? 100.0 * gated_correct[ci] / acted : 100.0);
      }
      if (1 < ngates && NULL != swconf && NULL != swok) {
        int gi;
        fprintf(stdout, "Gate sweep (coverage%% / precision%%):\n");
        fprintf(stdout, "  param ");
        for (gi = 0; gi < ngates; ++gi) fprintf(stdout, "  %11.2f", gates[gi]);
        fprintf(stdout, "\n");
        for (ci = 0; ci < nconf; ++ci) {
          const int oi = confidence_outputs[ci];
          int len = 0;
          const char* name = libxs_strtoken(output_names, ",", oi, &len);
          fprintf(stdout, "  %-4.*s ", len, name);
          for (gi = 0; gi < ngates; ++gi) {
            int acted = 0, ok = 0;
            for (i = 0; i < ntotal; ++i) {
              if (swconf[(size_t)i * nconf + ci] >= gates[gi]) {
                ++acted;
                if (0 != swok[(size_t)i * nconf + ci]) ++ok;
              }
            }
            fprintf(stdout, "  %5.1f/%5.1f",
              (0 < ntotal) ? 100.0 * acted / ntotal : 0.0,
              (0 < acted) ? 100.0 * ok / acted : 100.0);
          }
          fprintf(stdout, "\n");
        }
      }
      free(swok);
      free(swconf);
      fprintf(stdout, "Attested-entry split: attested %d (%.1f%%)"
        " | novel %d (%.1f%%)\n", split_n[0],
        (ntotal > 0) ? 100.0 * split_n[0] / ntotal : 0.0, split_n[1],
        (ntotal > 0) ? 100.0 * split_n[1] / ntotal : 0.0);
      fprintf(stdout,
        "  param  attested-prec  novel-prec  novel-cov\n");
      for (ci = 0; ci < nconf; ++ci) {
        const int oi = confidence_outputs[ci];
        int len = 0;
        const char* name = libxs_strtoken(output_names, ",", oi, &len);
        fprintf(stdout, "  %-4.*s      %6.1f%%       %6.1f%%     %5.1f%%\n",
          len, name,
          (0 < split_acted[ci][0])
            ? 100.0 * split_correct[ci][0] / split_acted[ci][0] : 100.0,
          (0 < split_acted[ci][1])
            ? 100.0 * split_correct[ci][1] / split_acted[ci][1] : 100.0,
          (0 < split_n[1]) ? 100.0 * split_acted[ci][1] / split_n[1] : 0.0);
        novel_cov[ci] = (0 < split_n[1])
          ? ((double)split_acted[ci][1] / split_n[1]) : 0.0;
      }
    }
#if defined(__XGBOOST)
    if (0 != use_xgb) {
      double* xgb_predicted = (double*)malloc(
        (size_t)ntotal * NOUTPUTS * sizeof(double));
      double* xgb_conf = (double*)malloc(
        (size_t)ntotal * NOUTPUTS * sizeof(double));
      int classify[NOUTPUTS], task[NOUTPUTS];
      for (j = 0; j < NOUTPUTS; ++j) {
        classify[j] = (2 * interp[j] < ntotal) ? 1 : 0;
      }
      if (NULL != xgb_predicted && NULL != xgb_conf
        && EXIT_SUCCESS == predict_xgb(reference,
          ntotal, NINPUTS, NOUTPUTS, trained, classify, xgb_predicted,
          xgb_conf, task, NULL, NULL))
      {
        double lsum[NOUTPUTS], xsum[NOUTPUTS];
        int lhit[NOUTPUTS], xhit[NOUTPUTS], nnovel = 0;
        memset(lsum, 0, sizeof(lsum));
        memset(xsum, 0, sizeof(xsum));
        memset(lhit, 0, sizeof(lhit));
        memset(xhit, 0, sizeof(xhit));
        for (i = 0; i < ntotal; ++i) {
          if (0 == trained[i]) {
            double expected[NOUTPUTS];
            libxs_predict_get(reference, i, NULL, expected);
            ++nnovel;
            for (j = 0; j < NOUTPUTS; ++j) {
              const double lval = all_predicted[(size_t)i * NOUTPUTS + j];
              const double xval = xgb_predicted[(size_t)i * NOUTPUTS + j];
              const int label = LIBXS_ROUNDX(int, expected[j]);
              lsum[j] += LIBXS_DELTA(lval, expected[j]);
              xsum[j] += LIBXS_DELTA(xval, expected[j]);
              if (LIBXS_ROUNDX(int, lval) == label) ++lhit[j];
              if (LIBXS_ROUNDX(int, xval) == label) ++xhit[j];
            }
          }
        }
        fprintf(stdout, "XGBoost head-to-head on %d novel entries"
          " (rounds=%i, depth=%i, eta=%g):\n", nnovel,
          predict_xgb_geti("XGB_ROUNDS", 200),
          predict_xgb_geti("XGB_DEPTH", 6),
          predict_xgb_getd("XGB_ETA", 0.1));
        fprintf(stdout, "  param  values  libxs-err   xgb-err"
          "  libxs-exact  xgb-exact\n");
        for (j = 0; j < NOUTPUTS; ++j) {
          int len = 0;
          const char* name = libxs_strtoken(output_names, ",", j, &len);
          char values[16];
          if (0 < task[j]) {
            LIBXS_SNPRINTF(values, sizeof(values), "%i", task[j]);
          }
          else {
            LIBXS_SNPRINTF(values, sizeof(values), "reg");
          }
          fprintf(stdout,
            "  %-4.*s  %6s  %9.2e %9.2e      %6.1f%%     %6.1f%%\n",
            len, name, values,
            (0 < nnovel) ? lsum[j] / nnovel : 0.0,
            (0 < nnovel) ? xsum[j] / nnovel : 0.0,
            (0 < nnovel) ? 100.0 * lhit[j] / nnovel : 0.0,
            (0 < nnovel) ? 100.0 * xhit[j] / nnovel : 0.0);
        }
        fprintf(stdout, "  Exact match is a proxy: this CSV carries no GFLOPS,"
          " so a differing\n  parameter that performs identically counts as a"
          " miss for both models.\n");
        /**
         * The same split, the same gate and the same accounting applied to
         * XGBoost's own confidence. Without it the attested/novel collapse is
         * a fact about one model rather than about the corpus: a gate that
         * only recognizes what it was built from would be a property either
         * of the vote fraction or of a held-out set that is genuinely harder,
         * and one column cannot tell those apart.
         */
        { const int xnconf = (int)(sizeof(confidence_outputs)
            / sizeof(confidence_outputs[0]));
          double xgates[8];
          const double threshold = (0 < gate_list(xgates,
            (int)(sizeof(xgates) / sizeof(*xgates)))) ? xgates[0] : 0.9;
          int xacted[5][2], xcorrect[5][2], ci;
          memset(xacted, 0, sizeof(xacted));
          memset(xcorrect, 0, sizeof(xcorrect));
          for (i = 0; i < ntotal; ++i) {
            const int novel = (0 == trained[i]) ? 1 : 0;
            double expected[NOUTPUTS];
            libxs_predict_get(reference, i, NULL, expected);
            for (ci = 0; ci < xnconf; ++ci) {
              const int oi = confidence_outputs[ci];
              if (xgb_conf[(size_t)i * NOUTPUTS + oi] >= threshold) {
                const double xval = xgb_predicted[(size_t)i * NOUTPUTS + oi];
                ++xacted[ci][novel];
                if (LIBXS_ROUNDX(int, xval) == LIBXS_ROUNDX(int, expected[oi])) {
                  ++xcorrect[ci][novel];
                }
              }
            }
          }
          fprintf(stdout, "XGBoost under the same gate (>=%.2f) and split:\n",
            threshold);
          fprintf(stdout,
            "  param  attested-prec  novel-prec  novel-cov\n");
          for (ci = 0; ci < xnconf; ++ci) {
            int len = 0;
            const char* name = libxs_strtoken(output_names, ",",
              confidence_outputs[ci], &len);
            fprintf(stdout, "  %-4.*s        %6.1f%%     %6.1f%%     %6.1f%%\n",
              len, name,
              (0 < xacted[ci][0])
                ? 100.0 * xcorrect[ci][0] / xacted[ci][0] : 100.0,
              (0 < xacted[ci][1])
                ? 100.0 * xcorrect[ci][1] / xacted[ci][1] : 100.0,
              (0 < nnovel) ? 100.0 * xacted[ci][1] / nnovel : 0.0);
          }
          /**
           * The same figures read at OUR coverage instead of at our gate. Two
           * models do not put the same meaning behind the same number, so a
           * precision compared at a shared gate is a comparison of two
           * different operating points; the gate that matters is whichever one
           * makes the two act equally often. XGBoost's gate is searched for
           * that coverage per parameter rather than assumed.
           */
          fprintf(stdout, "XGBoost read at OUR novel coverage"
            " (its gate searched per param):\n");
          fprintf(stdout,
            "  param  novel-cov  xgb-gate  xgb-prec\n");
          for (ci = 0; ci < xnconf; ++ci) {
            const int oi = confidence_outputs[ci];
            double best_gate = 1.0, best_delta = 2.0, best_prec = 100.0;
            double best_cov = 0.0;
            int step;
            if (0 > novel_cov[ci]) continue;
            for (step = 0; step <= 100; ++step) {
              const double g = 0.01 * step;
              int acted = 0, correct = 0;
              for (i = 0; i < ntotal; ++i) {
                if (0 == trained[i]
                  && xgb_conf[(size_t)i * NOUTPUTS + oi] >= g)
                {
                  double expected[NOUTPUTS];
                  const double xval = xgb_predicted[(size_t)i * NOUTPUTS + oi];
                  libxs_predict_get(reference, i, NULL, expected);
                  ++acted;
                  if (LIBXS_ROUNDX(int, xval)
                    == LIBXS_ROUNDX(int, expected[oi]))
                  {
                    ++correct;
                  }
                }
              }
              { const double cov = (0 < nnovel)
                  ? ((double)acted / nnovel) : 0.0;
                const double delta = LIBXS_DELTA(cov, novel_cov[ci]);
                if (delta < best_delta) {
                  best_delta = delta;
                  best_gate = g;
                  best_cov = cov;
                  best_prec = (0 < acted) ? (100.0 * correct / acted) : 100.0;
                }
              }
            }
            { int len = 0;
              const char* name = libxs_strtoken(output_names, ",", oi, &len);
              fprintf(stdout, "  %-4.*s     %6.1f%%     %5.2f    %6.1f%%\n",
                len, name, 100.0 * best_cov, best_gate, best_prec);
            }
          }
        }
      }
      free(xgb_conf);
      free(xgb_predicted);
    }
#endif
  }
  free(all_inputs);
  free(all_predicted);
}


static int write_confidence_maps(const char* prefix, const void* buffer,
  size_t size, const libxs_predict_t* reference, int ntotal)
{
  libxs_predict_t* model = libxs_predict_load(buffer, size);
  libxs_timer_tick_t tick;
  double inputs[NINPUTS];
  float *cube = NULL, *min_k = NULL, *mean_k = NULL;
  size_t ext3[3], ext2[2], nelements, mi, ni, ki, nm, nn, nk;
  int mmin, nmin, kmin, mmax, nmax, kmax, i;
  int result = EXIT_FAILURE;
  if (NULL != model && 0 < ntotal) {
    libxs_predict_get(reference, 0, inputs, NULL);
    mmin = mmax = (int)inputs[0];
    nmin = nmax = (int)inputs[1];
    kmin = kmax = (int)inputs[2];
    for (i = 1; i < ntotal; ++i) {
      int m_value, n_value, k_value;
      libxs_predict_get(reference, i, inputs, NULL);
      m_value = (int)inputs[0];
      n_value = (int)inputs[1];
      k_value = (int)inputs[2];
      if (m_value < mmin) mmin = m_value;
      if (mmax < m_value) mmax = m_value;
      if (n_value < nmin) nmin = n_value;
      if (nmax < n_value) nmax = n_value;
      if (k_value < kmin) kmin = k_value;
      if (kmax < k_value) kmax = k_value;
    }
    nm = (size_t)(mmax - mmin + 1);
    nn = (size_t)(nmax - nmin + 1);
    nk = (size_t)(kmax - kmin + 1);
    nelements = nm * nn * nk;
    cube = (float*)malloc(nelements * sizeof(float));
    min_k = (float*)malloc(nm * nn * sizeof(float));
    mean_k = (float*)malloc(nm * nn * sizeof(float));
    if (NULL != cube && NULL != min_k && NULL != mean_k) {
      tick = libxs_timer_tick();
      for (ni = 0; ni < nn; ++ni) {
        for (mi = 0; mi < nm; ++mi) {
          const size_t index2 = ni * nm + mi;
          float min_value = 1.0f, sum_value = 0.0f;
          for (ki = 0; ki < nk; ++ki) {
            libxs_predict_info_t info;
            double confidence;
            const size_t index3 = (ki * nn + ni) * nm + mi;
            inputs[0] = (double)(mmin + (int)mi);
            inputs[1] = (double)(nmin + (int)ni);
            inputs[2] = (double)(kmin + (int)ki);
            libxs_predict_eval(NULL, model, inputs, NULL, &info, 1);
            confidence = deployment_confidence(&info);
            cube[index3] = (float)confidence;
            if (confidence < min_value) min_value = (float)confidence;
            sum_value += (float)confidence;
          }
          min_k[index2] = min_value;
          mean_k[index2] = sum_value / (float)nk;
        }
      }
      { char filename[1024];
        libxs_mhd_info_t info3 = { 3, 1, LIBXS_DATATYPE_F32, 0 };
        libxs_mhd_info_t info2 = { 2, 1, LIBXS_DATATYPE_F32, 0 };
        ext3[0] = nm;
        ext3[1] = nn;
        ext3[2] = nk;
        ext2[0] = nm;
        ext2[1] = nn;
        LIBXS_SNPRINTF(filename, sizeof(filename), "%s_cube.mhd", prefix);
        result = libxs_mhd_write(filename, NULL, ext3, ext3, &info3, cube, NULL);
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence cube to %s\n", filename);
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s_minK.mhd", prefix);
          result = libxs_mhd_write(filename, NULL, ext2, ext2, &info2,
            min_k, NULL);
        }
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence minK projection to %s\n", filename);
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s_meanK.mhd", prefix);
          info2.header_size = 0;
          result = libxs_mhd_write(filename, NULL, ext2, ext2, &info2,
            mean_k, NULL);
        }
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence meanK projection to %s\n", filename);
          fprintf(stdout, "Confidence maps: M=%d..%d N=%d..%d K=%d..%d"
            " (%lu queries, %.2f s)\n", mmin, mmax, nmin, nmax, kmin, kmax,
            (unsigned long)nelements,
            libxs_timer_duration(tick, libxs_timer_tick()));
        }
      }
    }
  }
  free(mean_k);
  free(min_k);
  free(cube);
  libxs_predict_destroy(model);
  return result;
}


static double deployment_confidence(const libxs_predict_info_t* info)
{
  double result = 0.0;
  int i;
  if (NULL != info && NULL != info->confidence) {
    result = 1.0;
    for (i = 0; i < (int)(sizeof(confidence_outputs) /
      sizeof(confidence_outputs[0])); ++i)
    {
      const int output = confidence_outputs[i];
      if (output < info->noutputs && info->confidence[output] < result) {
        result = info->confidence[output];
      }
    }
  }
  return result;
}
