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

#define CSVFILE "unique_test.csv"
#define NROW 8
#define NUNIQUE 5


/* rows 2, 5 and 7 repeat earlier ones (-0 is 0); row 3 shares only the inputs */
static const double rows[NROW][3] = {
  { 1, 2, 10 }, { 1, 2, 10 }, { 1, 2, 11 }, { 0, 2, 10 },
  { 0, 2, 10 }, { 3, 4, 30 }, { 1, 2, 10 }, { 5, 6, 50 }
};
static const int distinct[NUNIQUE] = { 0, 2, 3, 5, 7 };


static int load(libxs_predict_t* model, int unique, int stride, int offset)
{
  libxs_predict_csv_t opts;
  memset(&opts, 0, sizeof(opts));
  opts.inputs = "a,b";
  opts.outputs = "y";
  opts.unique = unique;
  opts.stride = stride;
  opts.offset = offset;
  return libxs_predict_load_csv_opts(model, CSVFILE, &opts);
}


/* entry i of the model is row distinct[first + i * step] of the file */
static int check_rows(const libxs_predict_t* model, int n, int first, int step,
  const char* what)
{
  int i, result = EXIT_SUCCESS;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const double* expect = rows[distinct[first + i * step]];
    double inputs[2], output;
    libxs_predict_get(model, i, inputs, &output);
    if (expect[0] != inputs[0] || expect[1] != inputs[1] || expect[2] != output) {
      fprintf(stderr, "%s: entry %i is (%g,%g,%g), expected (%g,%g,%g)\n", what,
        i, inputs[0], inputs[1], output, expect[0], expect[1], expect[2]);
      result = EXIT_FAILURE;
    }
  }
  return result;
}


int main(void)
{
  FILE* out = fopen(CSVFILE, "w");
  int result = (NULL != out) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (NULL != out) {
    fprintf(out, "a,b,y\n1,2,10\n1,2,10\n1,2,11\n0,2,10\n-0,2,10\n3,4,30\n"
      "1,2,10\n5,6,50\n");
    fclose(out);
  }
  /* a zero-initialized struct keeps every row */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* model = libxs_predict_create(2, 1);
    const int n = (NULL != model) ? load(model, 0, 0, 0) : -1;
    if (NROW != n) {
      fprintf(stderr, "plain load: %i rows, expected %i\n", n, NROW);
      result = EXIT_FAILURE;
    }
    libxs_predict_destroy(model);
  }
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* model = libxs_predict_create(2, 1);
    const int n = (NULL != model) ? load(model, 1, 0, 0) : -1;
    if (NUNIQUE != n) {
      fprintf(stderr, "unique load: %i rows, expected %i\n", n, NUNIQUE);
      result = EXIT_FAILURE;
    }
    else result = check_rows(model, n, 0, 1, "unique load");
    libxs_predict_destroy(model);
  }
  /* a repeat is not admissible, so two strided parts split the distinct rows */
  if (EXIT_SUCCESS == result) {
    libxs_predict_t* even = libxs_predict_create(2, 1);
    libxs_predict_t* odd = libxs_predict_create(2, 1);
    const int neven = (NULL != even) ? load(even, 1, 2, 0) : -1;
    const int nodd = (NULL != odd) ? load(odd, 1, 2, 1) : -1;
    if (3 != neven || 2 != nodd) {
      fprintf(stderr, "strided unique load: %i and %i rows, expected 3 and 2\n",
        neven, nodd);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) result = check_rows(even, neven, 0, 2, "even part");
    if (EXIT_SUCCESS == result) result = check_rows(odd, nodd, 1, 2, "odd part");
    libxs_predict_destroy(even);
    libxs_predict_destroy(odd);
  }
  /* an absent input equals an absent one */
  if (EXIT_SUCCESS == result) {
    out = fopen(CSVFILE, "w");
    if (NULL != out) {
      libxs_predict_t* model = libxs_predict_create(2, 1);
      int n = -1;
      fprintf(out, "a,b,y\n,7,70\n,7,70\n1,7,70\n");
      fclose(out);
      if (NULL != model) {
        libxs_predict_set_missing(model, 1);
        n = load(model, 1, 0, 0);
      }
      if (2 != n) {
        fprintf(stderr, "unique load with absent inputs: %i rows, expected 2\n", n);
        result = EXIT_FAILURE;
      }
      libxs_predict_destroy(model);
    }
    else result = EXIT_FAILURE;
  }
  remove(CSVFILE);
  return result;
}
