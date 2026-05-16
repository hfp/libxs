/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_math.h>
#include <libxs_perm.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

static const char* input_names[] = { "M", "N", "K" };
static const char* output_names[] = {
  "BS", "BM", "BN", "BK", "WS", "WG", "LU", "NZ",
  "AL", "TB", "TC", "AP", "AA", "AB", "AC"
};

enum { NINPUTS = 3, NOUTPUTS = 15 };

typedef struct trial_ctx_t {
  libxs_predict_t* source;
  int* perm;
  int ntotal;
  int mode;
} trial_ctx_t;

static double trial_fraction(double fraction, const void* data);
static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal, const char* label);


int main(int argc, char* argv[])
{
  int argi = 1, mode = -1;
  int order_arg = 0;
  double eval_fraction = 0.8;
  const char *filename, *modelfile;
  int result = EXIT_FAILURE;
  if (argi < argc && '0' <= argv[argi][0] && '9' >= argv[argi][0]) {
    eval_fraction = atof(argv[argi]);
    ++argi;
  }
  if (argi < argc && 'a' <= argv[argi][0]) {
    if ('a' == argv[argi][0]) mode = -1;
    else if ('c' == argv[argi][0]) mode = 1;
    else if ('i' == argv[argi][0]) mode = 0;
    ++argi;
  }
  if (argi < argc && '-' == argv[argi][0] && '\0' != argv[argi][1]) {
    order_arg = atoi(argv[argi]);
    ++argi;
  }
  filename = (argi < argc) ? argv[argi] : NULL;
  modelfile = (argi + 1 < argc) ? argv[argi + 1] : NULL;
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
      "Usage: %s [fraction] [auto|cat|interp] [-N] <csvfile> [modelfile]\n"
      "  fraction: validation split 0..1 for quality report (default: 0.8)\n"
      "  auto:     auto-detect mode per output (default)\n"
      "  cat:      force categorical (kNN) for all outputs\n"
      "  interp:   force interpolation for all outputs\n"
      "  -N: max polynomial order for final build (default: 0 = auto)\n"
      "  Finds the optimal training fraction via GSS, saves the full\n"
      "  model, and reports quality on a held-out validation set.\n", argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(NINPUTS, NOUTPUTS);
    if (NULL != source) {
      const int ntotal = libxs_predict_load_csv(source, filename, NULL,
        input_names, NINPUTS, output_names, NOUTPUTS);
      if (0 < ntotal) {
        int* perm = (int*)malloc((size_t)ntotal * sizeof(int));
        fprintf(stdout, "Loaded %d entries from %s\n", ntotal, filename);
        if (NULL != perm) {
          trial_ctx_t ctx;
          double best_fraction = 1.0;
          int i, ntrain;
          libxs_predict_t* model;
          for (i = 0; i < ntotal; ++i) perm[i] = i;
          libxs_shuffle(perm, sizeof(int), (size_t)ntotal, NULL, NULL);
          ctx.source = source;
          ctx.perm = perm;
          ctx.ntotal = ntotal;
          ctx.mode = mode;
          { double unimodality = 0;
            libxs_gss_min(trial_fraction, &ctx, 0.3, 1.0, &best_fraction, 20, &unimodality);
            ntrain = LIBXS_MAX((int)(ntotal * best_fraction + 0.5), 1);
            fprintf(stdout, "Optimal fraction: %.2f (%d/%d entries, unimodality=%.2f)\n",
              best_fraction, ntrain, ntotal, unimodality);
          }
          model = libxs_predict_create(NINPUTS, NOUTPUTS);
          if (NULL != model) {
            double inputs[NINPUTS], outputs[NOUTPUTS];
            libxs_predict_set_mode(model, mode);
            for (i = 0; i < ntrain; ++i) {
              libxs_predict_get(source, perm[i], inputs, outputs);
              libxs_predict_push(NULL, model, inputs, outputs);
            }
            { int build_ok = EXIT_FAILURE;
#if defined(_OPENMP)
#             pragma omp parallel
              { const int br = libxs_predict_build_task(NULL, model, 0, order_arg,
                  omp_get_thread_num(), omp_get_num_threads());
                if (0 == omp_get_thread_num()) build_ok = br;
              }
#else
              build_ok = libxs_predict_build_task(NULL, model, 0, order_arg, 0, 1);
#endif
            if (EXIT_SUCCESS == build_ok) {
            { libxs_predict_query_t qi = {0};
              libxs_predict_query(model, &qi);
              fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d (%d iter)\n",
                qi.nclusters, qi.compression, qi.order, qi.iterations);
            }
              { const int nval = LIBXS_MAX((int)(ntotal * eval_fraction + 0.5), 1);
                libxs_predict_t* val_model = libxs_predict_create(NINPUTS, NOUTPUTS);
                if (NULL != val_model) {
                  double vi[NINPUTS], vo[NOUTPUTS];
                  libxs_predict_set_mode(val_model, mode);
                  for (i = 0; i < nval; ++i) {
                    libxs_predict_get(source, perm[i], vi, vo);
                    libxs_predict_push(NULL, val_model, vi, vo);
                  }
                  if (EXIT_SUCCESS == libxs_predict_build(val_model, 0, order_arg)) {
                    evaluate(val_model, source, ntotal, "Validation");
                  }
                  libxs_predict_destroy(val_model);
                }
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
                      result = EXIT_SUCCESS;
                    }
                  }
                  free(buffer);
                }
              }
            }
            }
            libxs_predict_destroy(model);
          }
        }
        free(perm);
      }
      else {
        fprintf(stderr, "Failed to load entries from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}


static double trial_fraction(double fraction, const void* data)
{
  const trial_ctx_t* ctx = (const trial_ctx_t*)data;
  const int ntrain = LIBXS_MAX((int)(ctx->ntotal * fraction + 0.5), 1);
  double total_err = 0;
  libxs_predict_t* model = libxs_predict_create(NINPUTS, NOUTPUTS);
  if (NULL != model) {
    int i;
    double inputs[NINPUTS], outputs[NOUTPUTS];
    libxs_predict_set_mode(model, ctx->mode);
    for (i = 0; i < ntrain; ++i) {
      libxs_predict_get(ctx->source, ctx->perm[i], inputs, outputs);
      libxs_predict_push(NULL, model, inputs, outputs);
    }
    if (EXIT_SUCCESS == libxs_predict_build(model, 0, 1)) {
      double* all_inputs = (double*)malloc((size_t)ctx->ntotal * NINPUTS * sizeof(double));
      double* all_predicted = (double*)malloc((size_t)ctx->ntotal * NOUTPUTS * sizeof(double));
      if (NULL != all_inputs && NULL != all_predicted) {
        int j;
        for (i = 0; i < ctx->ntotal; ++i) {
          libxs_predict_get(ctx->source, i, all_inputs + (size_t)i * NINPUTS, NULL);
        }
#if defined(_OPENMP)
#       pragma omp parallel
        { const int tid = omp_get_thread_num(), ntasks = omp_get_num_threads();
          libxs_predict_eval_batch_task(model, all_inputs, all_predicted,
            ctx->ntotal, 1, tid, ntasks);
        }
#else
        libxs_predict_eval_batch(model, all_inputs, all_predicted, ctx->ntotal, 1);
#endif
        for (i = 0; i < ctx->ntotal; ++i) {
          double expected[NOUTPUTS];
          libxs_predict_get(ctx->source, i, NULL, expected);
          for (j = 0; j < NOUTPUTS; ++j) {
            total_err += LIBXS_DELTA(all_predicted[(size_t)i * NOUTPUTS + j], expected[j]);
          }
        }
      }
      else total_err = 1e30;
      free(all_inputs);
      free(all_predicted);
    }
    else total_err = 1e30;
    libxs_predict_destroy(model);
  }
  else total_err = 1e30;
  return total_err;
}


static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal, const char* label)
{
  double maxerr[NOUTPUTS] = {0}, sumerr[NOUTPUTS] = {0};
  double sum_bound[NOUTPUTS] = {0};
  double* all_inputs = (double*)malloc((size_t)ntotal * NINPUTS * sizeof(double));
  double* all_predicted = (double*)malloc((size_t)ntotal * NOUTPUTS * sizeof(double));
  int ninterp[NOUTPUTS] = {0}, i, j;
  for (i = 0; i < ntotal; ++i) {
    libxs_predict_get(reference, i, all_inputs + (size_t)i * NINPUTS, NULL);
  }
#if defined(_OPENMP)
# pragma omp parallel
  { const int tid = omp_get_thread_num(), ntasks = omp_get_num_threads();
    libxs_predict_eval_batch_task(model, all_inputs, all_predicted,
      ntotal, 1, tid, ntasks);
  }
#else
  libxs_predict_eval_batch(model, all_inputs, all_predicted, ntotal, 1);
#endif
  for (i = 0; i < ntotal; ++i) {
    double expected[NOUTPUTS];
    libxs_predict_info_t info;
    libxs_predict_get(reference, i, NULL, expected);
    libxs_predict_eval(NULL, model,
      all_inputs + (size_t)i * NINPUTS, NULL, &info, 1);
    for (j = 0; j < NOUTPUTS; ++j) {
      const double err = LIBXS_DELTA(all_predicted[(size_t)i * NOUTPUTS + j], expected[j]);
      sumerr[j] += err;
      if (err > maxerr[j]) maxerr[j] = err;
      sum_bound[j] += info.error[j];
      ninterp[j] += info.interpolated[j];
    }
  }
  free(all_inputs);
  free(all_predicted);
  fprintf(stdout, "%s (%d samples):\n", label, ntotal);
  fprintf(stdout, "  param   avg-err   max-err  avg-bound\n");
  for (j = 0; j < NOUTPUTS; ++j) {
    fprintf(stdout, "  %-3s%c  %9.2e %9.2e  %9.2e\n",
      output_names[j], (0 < ninterp[j]) ? '*' : ' ',
      (0 < ntotal) ? (sumerr[j] / ntotal) : 0.0,
      maxerr[j],
      (0 < ntotal) ? (sum_bound[j] / ntotal) : 0.0);
  }
}
