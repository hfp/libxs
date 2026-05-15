/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_rng.h>
#include <libxs_math.h>


typedef struct record_t {
  double inputs[3];
  double outputs[14];
} record_t;


static int load_records(const char* filename, record_t** records, int* nrecords)
{
  const int inputs_idx[] = { 2, 3, 4 };
  const int outputs_idx[] = { 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
  FILE* file = fopen(filename, "r");
  int result = 0, cap = 0;
  *records = NULL;
  *nrecords = 0;
  if (NULL != file) {
    char line[4096];
    const char* sep = NULL;
    if (NULL != fgets(line, (int)sizeof(line), file)) {
      sep = (NULL != strchr(line, ';')) ? ";" :
            (NULL != strchr(line, ',')) ? "," :
            (NULL != strchr(line, '\t')) ? "\t" : " ";
    }
    if (NULL != sep) {
      rewind(file);
      while (NULL != fgets(line, (int)sizeof(line), file)) {
        double vals[22];
        int col = 0, filled = 0, ok = 1;
        const char* p = line;
        while ('\0' != *p && 0 != ok) {
          const char* field = p;
          char* endptr;
          double v;
          while ('\0' != *p && NULL == strchr(sep, *p)) ++p;
          v = strtod(field, &endptr);
          if (endptr != field && col < 22) {
            vals[col] = v;
            ++filled;
          }
          else if (col >= 2 && col <= 4) ok = 0;
          else if (col >= 8 && col <= 21) ok = 0;
          if ('\0' != *p) ++p;
          ++col;
        }
        if (0 != ok && filled >= 20) {
          if (*nrecords >= cap) {
            cap = (0 < cap) ? (cap * 2) : 256;
            *records = (record_t*)realloc(*records, (size_t)cap * sizeof(record_t));
            if (NULL == *records) { ok = 0; result = -1; }
          }
          if (0 != ok) {
            record_t* r = &(*records)[*nrecords];
            int i;
            for (i = 0; i < 3; ++i) r->inputs[i] = vals[inputs_idx[i]];
            for (i = 0; i < 14; ++i) r->outputs[i] = vals[outputs_idx[i]];
            ++(*nrecords);
          }
        }
      }
      result = *nrecords;
    }
    fclose(file);
  }
  return result;
}


int main(int argc, char* argv[])
{
  const char* filename = (1 < argc) ? argv[1] : NULL;
  const char* modelfile = (2 < argc) ? argv[2] : "predict.bin";
  const double fraction = (3 < argc) ? atof(argv[3]) : 1.0;
  const int ninputs = 3, noutputs = 14;
  int result = EXIT_FAILURE;
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s <csvfile> [modelfile] [fraction]\n"
      "  fraction: 0..1, training split (default: 1.0 = full model)\n"
      "            <1.0 holds out the rest for validation\n", argv[0]);
  }
  else {
    record_t* records = NULL;
    int ntotal = 0;
    const int nloaded = load_records(filename, &records, &ntotal);
    if (0 < nloaded) {
      const int ntrain = (fraction < 1.0)
        ? LIBXS_MAX((int)(ntotal * fraction + 0.5), 1) : ntotal;
      const int ntest = ntotal - ntrain;
      unsigned char* is_train = (unsigned char*)calloc((size_t)ntotal, 1);
      libxs_predict_t* model = libxs_predict_create(ninputs, noutputs);
      fprintf(stdout, "Loaded %d entries from %s\n", ntotal, filename);
      if (NULL != is_train && NULL != model) {
        int i, npushed = 0;
        if (ntrain >= ntotal) {
          memset(is_train, 1, (size_t)ntotal);
        }
        else {
          int chosen = 0;
          while (chosen < ntrain) {
            const int idx = (int)libxs_rng_u32((unsigned int)ntotal);
            if (0 == is_train[idx]) { is_train[idx] = 1; ++chosen; }
          }
        }
        for (i = 0; i < ntotal; ++i) {
          if (0 != is_train[i]) {
            if (EXIT_SUCCESS == libxs_predict_push(NULL, model,
              records[i].inputs, records[i].outputs))
            {
              ++npushed;
            }
          }
        }
        fprintf(stdout, "Training: %d, held out: %d\n", npushed, ntest);
        if (EXIT_SUCCESS == libxs_predict_build(model, 0, 0.8)) {
          int nclusters = 0;
          double compression = 0;
          libxs_predict_query(model, &nclusters, NULL, &compression);
          fprintf(stdout, "Built: %d clusters, %.1fx compression\n",
            nclusters, compression);
          {
            static const char* names[] = {
              "BM", "BN", "BK", "WS", "WG", "LU", "NZ",
              "AL", "TB", "TC", "AP", "AA", "AB", "AC"
            };
            double maxerr[14] = {0}, sumerr[14] = {0};
            double sum_bound[14] = {0}, outputs[14];
            int ninterp[14] = {0}, neval = 0, j;
            for (i = 0; i < ntotal; ++i) {
              if (0 == is_train[i] || 0 == ntest) {
                libxs_predict_info_t info;
                libxs_predict_eval(NULL, model,
                  records[i].inputs, outputs, &info, 1);
                for (j = 0; j < noutputs; ++j) {
                  const double err = LIBXS_DELTA(outputs[j], records[i].outputs[j]);
                  sumerr[j] += err;
                  if (err > maxerr[j]) maxerr[j] = err;
                  sum_bound[j] += info.error[j];
                  ninterp[j] += info.reliable[j];
                }
                ++neval;
              }
            }
            fprintf(stdout, "%s (%d samples):\n",
              (0 < ntest) ? "Validation" : "Self-evaluation", neval);
            fprintf(stdout, "  param   avg-err   max-err  avg-bound\n");
            for (j = 0; j < noutputs; ++j) {
              fprintf(stdout, "  %-3s%c  %9.2e %9.2e  %9.2e\n",
                names[j], (0 < ninterp[j]) ? '*' : ' ',
                (0 < neval) ? (sumerr[j] / neval) : 0.0,
                maxerr[j],
                (0 < neval) ? (sum_bound[j] / neval) : 0.0);
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
      free(is_train);
      libxs_predict_destroy(model);
    }
    else {
      fprintf(stderr, "Failed to load entries from %s\n", filename);
    }
    free(records);
  }
  return result;
}
