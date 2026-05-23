/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_mem.h>


enum { NFEAT = 37 };


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const double split = (argc > 2) ? atof(argv[2]) : 0.8;
  const int order = (argc > 3) ? atoi(argv[3]) : 2;
  const int nclusters = (argc > 4) ? atoi(argv[4]) : 0;
  int result = EXIT_FAILURE;
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <crystal_csv> [train_fraction] [order] [nclusters]\n"
      "  Crystal system prediction from composition features.\n"
      "  Input: CSV with numeric features + crystal_system label (last col).\n"
      "  Crystal systems: 1=triclinic, 2=monoclinic, 3=orthorhombic,\n"
      "    4=tetragonal, 5=trigonal, 6=hexagonal, 7=cubic.\n"
      "  Default train_fraction: 0.8\n", argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(NFEAT, 1);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        NULL, NFEAT, NULL, 1);
      if (0 < total) {
        const int train_end = LIBXS_MAX((int)(total * split + 0.5), 2);
        libxs_predict_t* model = libxs_predict_create(NFEAT, 1);
        fprintf(stdout, "Loaded %d entries (%d features) from %s\n",
          total, NFEAT, filename);
        if (NULL != model) {
          int t, correct = 0, ntest = 0, gated = 0, gated_correct = 0;
          double sum_conf = 0;
          libxs_predict_set_decompose(model, LIBXS_PREDICT_FISHER);
          for (t = 0; t < train_end; ++t) {
            double inputs[NFEAT], output;
            libxs_predict_get(source, t, inputs, &output);
            libxs_predict_push(NULL, model, inputs, &output);
          }
          if (EXIT_SUCCESS == libxs_predict_build(model, nclusters, order)) {
            libxs_predict_query_t qi;
            LIBXS_MEMZERO(&qi);
            libxs_predict_query(model, &qi);
            fprintf(stdout, "Train=%d, Test=%d\n", qi.nentries, total - train_end);
            fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d\n",
              qi.nclusters, qi.compression, qi.order);
            for (t = train_end; t < total; ++t) {
              double inputs[NFEAT], predicted;
              libxs_predict_info_t info;
              libxs_predict_get(source, t, inputs, NULL);
              libxs_predict_eval(NULL, model, inputs, &predicted, &info, 1);
              { int label;
                double expected;
                libxs_predict_get(source, t, NULL, &expected);
                label = LIBXS_ROUNDX(int, expected);
                if (LIBXS_ROUNDX(int, predicted) == label) ++correct;
                if (info.confidence[0] >= 0.9) {
                  ++gated;
                  if (LIBXS_ROUNDX(int, predicted) == label) ++gated_correct;
                }
              }
              sum_conf += info.confidence[0];
              ++ntest;
            }
            if (0 < ntest) {
              fprintf(stdout, "Accuracy: %d/%d = %.1f%%\n",
                correct, ntest, 100.0 * correct / ntest);
              fprintf(stdout, "Confidence-gated (>=0.9): %d/%d = %.1f%% (coverage %.1f%%)\n",
                gated_correct, gated, (0 < gated) ? 100.0 * gated_correct / gated : 0.0,
                100.0 * gated / ntest);
              fprintf(stdout, "Avg confidence: %.3f\n", sum_conf / ntest);
            }
            result = EXIT_SUCCESS;
          }
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


