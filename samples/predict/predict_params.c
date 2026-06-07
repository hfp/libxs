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

#if defined(_OPENMP)
# include <omp.h>
#endif

static const char input_names[] = "M,N,K";
static const char output_names[] =
  "BS,BM,BN,BK,WS,WG,LU,NZ,AL,TB,TC,AP,AA,AB,AC,XF";

enum { NINPUTS = 3, NOUTPUTS = 16 };

static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal);


int main(int argc, char* argv[])
{
  int argi = 1, mode = LIBXS_PREDICT_AUTO, use_rf = 0;
  int order_arg = 0;
  double eval_fraction = 0.8;
  const char *filename, *modelfile;
  int result = EXIT_FAILURE;
  if (argi < argc && '0' <= argv[argi][0] && '9' >= argv[argi][0]) {
    eval_fraction = atof(argv[argi]);
    ++argi;
  }
  if (argi < argc && 'a' <= argv[argi][0]) {
    if ('a' == argv[argi][0]) mode = LIBXS_PREDICT_AUTO;
    else if ('c' == argv[argi][0]) mode = LIBXS_PREDICT_CLASSIFY;
    else if ('i' == argv[argi][0]) mode = LIBXS_PREDICT_INTERPOLATE;
    else if ('r' == argv[argi][0]) use_rf = 1;
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
      "Usage: %s [fraction] [auto|cat|interp|rf] [-N] <csvfile> [modelfile]\n"
      "  fraction: validation split 0..1 for quality report (default: 0.8)\n"
      "  auto:     auto-detect mode per output (default)\n"
      "  cat:      force categorical (kNN) for all outputs\n"
      "  interp:   force interpolation for all outputs\n"
      "  rf:       Random Forest classification\n"
      "  -N: max polynomial order for final build (default: 0 = auto)\n"
      "  Trains on all entries, saves the model, and reports\n"
      "  quality on a held-out validation set.\n", argv[0]);
  }
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
          for (i = 0; i < ntotal; ++i) {
            libxs_predict_get(source, i, inputs, outputs);
            libxs_predict_push(NULL, model, inputs, outputs);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { const int br = libxs_predict_build_task(NULL, model, 0, order_arg,
              omp_get_thread_num(), omp_get_num_threads());
            if (0 == omp_get_thread_num()) build_ok = br;
          }
#else
          build_ok = libxs_predict_build_task(NULL, model, 0, order_arg, 0, 1);
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS == build_ok) {
            { libxs_predict_query_t qi = { 0 };
              libxs_predict_query(model, &qi);
              fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
                " (%.2f s)\n", qi.nclusters, qi.compression, qi.order, dt_build);
            }
            { const int nval = LIBXS_MAX((int)(ntotal * eval_fraction + 0.5), 1);
              libxs_predict_t* val_model = libxs_predict_create(NINPUTS, NOUTPUTS);
              if (NULL != val_model) {
                double vi[NINPUTS], vo[NOUTPUTS];
                libxs_predict_set_mode(val_model, mode);
                if (0 != use_rf) libxs_predict_set_decompose(val_model, LIBXS_PREDICT_RF);
                for (i = 0; i < nval; ++i) {
                  libxs_predict_get(source, i, vi, vo);
                  libxs_predict_push(NULL, val_model, vi, vo);
                }
                if (EXIT_SUCCESS == libxs_predict_build(val_model, 0, order_arg)) {
                  evaluate(val_model, source, ntotal);
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
  const libxs_predict_t* reference, int ntotal)
{
  libxs_timer_tick_t tick;
  double maxerr[NOUTPUTS] = { 0 }, sumerr[NOUTPUTS] = { 0 };
  double dt_eval;
  double* all_inputs = (double*)malloc((size_t)ntotal * NINPUTS * sizeof(double));
  double* all_predicted = (double*)malloc((size_t)ntotal * NOUTPUTS * sizeof(double));
  int i, j;
  if (NULL == all_inputs || NULL == all_predicted) {
    free(all_inputs);
    free(all_predicted);
    return;
  }
  for (i = 0; i < ntotal; ++i) {
    libxs_predict_get(reference, i, all_inputs + (size_t)i * NINPUTS, NULL);
  }
  tick = libxs_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel
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
  free(all_inputs);
  free(all_predicted);
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
}
