/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_timer.h>
#include <libxs_math.h>
#include <libxs_mem.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

static const char input_names[] = "latitude,longitude,depth";
static const char output_names[] = "mag";

enum { NINPUTS = 3, NOUTPUTS = 1 };


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  int result = EXIT_FAILURE;
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <usgs_csv> [train_fraction]\n"
      "  Earthquake magnitude prediction from location and depth.\n"
      "  Input: USGS earthquake catalog CSV (comma-delimited).\n"
      "  Predicts magnitude from (latitude, longitude, depth).\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(NINPUTS, NOUTPUTS);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        input_names, output_names);
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
          for (i = 0; i < train_end; ++i) {
            libxs_predict_get(source, i, inputs, outputs);
            libxs_predict_push(NULL, model, inputs, outputs);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { build_ok = libxs_predict_build_task(NULL, model, 0, 2,
              omp_get_thread_num(), omp_get_num_threads());
          }
#else
          build_ok = libxs_predict_build(model, 0, 2);
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS == build_ok) {
            libxs_predict_query_t qi;
            double sum_err = 0, max_err = 0, sum_conf = 0, dt_eval;
            int neval = 0;
            LIBXS_MEMZERO(&qi);
            libxs_predict_query(model, &qi);
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
