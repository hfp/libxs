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
#include <libxs/libxs_mem.h>

#include "predict_args.h"

/**
 * Code length of a held-out stream under the model, against the code length
 * under the training distribution alone.  A point estimate says how close the
 * model gets; this says whether the inputs carry information about the output
 * at all, which is a different question and sometimes the more useful one: a
 * model can post a competitive error while knowing nothing its own marginal
 * did not already know.
 *
 * The comparison needs a control, because "no information" is also what a
 * broken measurement reports.  Run the same command with the roles of two
 * columns exchanged (see the usage note) and the same machinery separates the
 * cases.
 */

enum { MAXVALS = 65536 };

static int support_index(const double* vals, int n, double v);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const char* innames = (argc > 2) ? argv[2] : NULL;
  const char* outname = (argc > 3) ? argv[3] : NULL;
  const double split = (argc > 4) ? atof(argv[4]) : 0.8;
  const int vocabulary = (argc > 5) ? atoi(argv[5]) : 0;
  const int hknn = (argc > 6) ? atoi(argv[6]) : 0;
  static const int numeric[] = { 4, 5, 6 };
  int result = EXIT_FAILURE;
  if (NULL == filename || NULL == innames || NULL == outname
    || 0 == predict_slots_ok(argc, argv, numeric, 3)
    || 0 == predict_split_ok(split))
  {
    fprintf(stdout,
      "Usage: %s <csvfile> <innames> <outname> [fraction] [vocabulary] [hknn]\n"
      "  Held-out code length in bits per event, under the model and under\n"
      "  the training distribution of the same output.\n"
      "  innames:    comma-separated input column names.\n"
      "  outname:    single output column name (discrete values).\n"
      "  fraction:   training split (default 0.8).\n"
      "  vocabulary: distinct values the caller considers possible; 0 uses\n"
      "              the attested support plus one aggregate novel atom, in\n"
      "              which case an unattested outcome costs infinite bits.\n"
      "              Pass a count above the support size to keep it finite.\n"
      "  hknn:       non-zero selects the hierarchical partition.\n"
      "Example (USGS catalog): magnitude is not determined by location,\n"
      "  and depth is -- the second command is the control for the first.\n"
      "  %s predict_earthquakes.csv latitude,longitude,depth mag 0.8 68\n"
      "  %s predict_earthquakes.csv latitude,longitude,mag depth 0.8 9000\n",
      argv[0], argv[0], argv[0]);
  }
  else {
    int ninputs = 1;
    const char* p = innames;
    while ('\0' != *p) { if (',' == *p++) ++ninputs; }
    { libxs_predict_t* source = libxs_predict_create(ninputs, 1);
      if (NULL != source) {
        const int total = libxs_predict_load_csv(source, filename, NULL,
          innames, outname, NULL, 0, NULL);
        if (0 < total) {
          const int train_end = LIBXS_MAX((int)(total * split + 0.5), 1);
          libxs_predict_t* model = libxs_predict_create(ninputs, 1);
          double* sup = (double*)malloc(MAXVALS * sizeof(double));
          double* freq = (double*)malloc(MAXVALS * sizeof(double));
          double* inputs = (double*)malloc((size_t)ninputs * sizeof(double));
          if (NULL != model && NULL != sup && NULL != freq && NULL != inputs) {
            double outputs[1];
            int nsup = 0, i;
            if (0 != hknn) {
              libxs_predict_set_decompose(model, LIBXS_PREDICT_HKNN);
            }
            for (i = 0; i < MAXVALS; ++i) freq[i] = 0;
            for (i = 0; i < train_end; ++i) {
              int si;
              libxs_predict_get(source, i, inputs, outputs);
              libxs_predict_push(NULL, model, inputs, outputs);
              si = support_index(sup, nsup, outputs[0]);
              if (0 > si && nsup < MAXVALS) {
                sup[nsup] = outputs[0];
                si = nsup++;
              }
              if (0 <= si) freq[si] += 1.0;
            }
            fprintf(stdout, "Loaded %d entries from %s (%s -> %s)\n",
              total, filename, innames, outname);
            fprintf(stdout, "Train=%d, Test=%d, distinct outputs=%d\n",
              train_end, total - train_end, nsup);
            if (EXIT_SUCCESS == libxs_predict_build(model, 0, 2, 0.0)) {
              void* context = libxs_predict_prob_create(model);
              if (NULL != context) {
                const int nvoc = LIBXS_MAX(vocabulary, nsup);
                const double den = (double)train_end + (double)nvoc;
                double bits = 0, bits_marg = 0, bits_att = 0, bits_nov = 0;
                int nscored = 0, nattested = 0, nnovel = 0;
                for (i = train_end; i < total; ++i) {
                  libxs_predict_prob_info_t pinfo;
                  libxs_predict_get(source, i, inputs, outputs);
                  if (0 < libxs_predict_prob_observe(NULL, model, context,
                    inputs, 0, &outputs[0], NULL, NULL, 0, NULL, &pinfo,
                    vocabulary, 1))
                  {
                    const int si = support_index(sup, nsup, outputs[0]);
                    const double q = (((0 <= si) ? freq[si] : 0.0) + 1.0) / den;
                    const double b = -pinfo.logprob[0];
                    bits += b;
                    bits_marg += -LIBXS_LOG2(q);
                    if (0 != pinfo.attested[0]) {
                      bits_att += b;
                      ++nattested;
                    }
                    else {
                      bits_nov += b;
                      ++nnovel;
                    }
                    ++nscored;
                  }
                }
                if (0 < nscored) {
                  fprintf(stdout, "Scored %d of %d held-out events\n",
                    nscored, total - train_end);
                  fprintf(stdout, "  model         %8.3f bits/event\n",
                    bits / nscored);
                  fprintf(stdout, "  training dist %8.3f bits/event"
                    " (add-one over %d values)\n", bits_marg / nscored, nvoc);
                  if (0 < nattested) {
                    fprintf(stdout, "  attested      %8.3f bits (%d, %.0f%%)\n",
                      bits_att / nattested, nattested,
                      100.0 * nattested / nscored);
                  }
                  if (0 < nnovel) {
                    fprintf(stdout, "  novel         %8.3f bits (%d, %.0f%%)\n",
                      bits_nov / nnovel, nnovel, 100.0 * nnovel / nscored);
                  }
                  result = EXIT_SUCCESS;
                }
                libxs_predict_prob_destroy(context);
              }
            }
          }
          free(inputs);
          free(freq);
          free(sup);
          libxs_predict_destroy(model);
        }
        else {
          fprintf(stderr, "Failed to load %s\n", filename);
        }
        libxs_predict_destroy(source);
      }
    }
  }
  return result;
}


static int support_index(const double* vals, int n, double v)
{
  int result = -1;
  int i;
  for (i = 0; i < n && 0 > result; ++i) {
    if (vals[i] == v) result = i;
  }
  return result;
}
