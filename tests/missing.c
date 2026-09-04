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

#define NENTRY 64
#define CSVFILE "missing_test.csv"


static double absent(void)
{
  static const volatile double zero = 0;
  return zero / zero;
}


/** Output depends on the first input only, so the second may be absent. */
static void fill(double inputs[], double* out, int i)
{
  inputs[0] = (double)i;
  inputs[1] = (double)(i % 7);
  *out = (double)i;
}


static int check_eval(const libxs_predict_t* model, const char* what,
  double x0, double x1, double expect, double tol, int want_interp)
{
  double inputs[2], predicted = 0;
  libxs_predict_info_t info;
  int result = EXIT_SUCCESS;
  inputs[0] = x0;
  inputs[1] = x1;
  memset(&info, 0, sizeof(info));
  libxs_predict_eval(NULL, model, inputs, &predicted, &info, 1);
  if (LIBXS_ISNAN(predicted)) {
    fprintf(stderr, "%s: prediction is not a number\n", what);
    result = EXIT_FAILURE;
  }
  else if (LIBXS_FABS(predicted - expect) > tol) {
    fprintf(stderr, "%s: predicted %f, expected %f (+-%f)\n",
      what, predicted, expect, tol);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 <= want_interp
    && NULL != info.interpolated && want_interp != info.interpolated[0])
  {
    fprintf(stderr, "%s: interpolated=%i, expected %i\n",
      what, info.interpolated[0], want_interp);
    result = EXIT_FAILURE;
  }
  return result;
}


int main(void)
{
  libxs_predict_t* model = libxs_predict_create(2, 1);
  libxs_predict_t* vmodel = libxs_predict_create(2, 1);
  libxs_predict_t* rfmodel = libxs_predict_create(2, 1);
  int result = (NULL != model && NULL != vmodel && NULL != rfmodel)
    ? EXIT_SUCCESS : EXIT_FAILURE;
  int i;
  if (EXIT_SUCCESS == result) {
    /* forced so that the absent-coordinate override is what the flag reports:
       left to the fingerprint this corpus is scored categorically anyway */
    libxs_predict_set_mode(model, LIBXS_PREDICT_INTERPOLATE);
    for (i = 0; i < NENTRY; ++i) {
      double inputs[2], out;
      fill(inputs, &out, i);
      /* every eighth entry has no second coordinate */
      if (0 == (i % 8)) inputs[1] = absent();
      if (EXIT_SUCCESS != libxs_predict_push(NULL, model, inputs, &out)
        || EXIT_SUCCESS != libxs_predict_push(NULL, vmodel, inputs, &out)
        || EXIT_SUCCESS != libxs_predict_push(NULL, rfmodel, inputs, &out))
      {
        result = EXIT_FAILURE;
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_predict_build(model, 0, 2, 0.0);
    if (EXIT_SUCCESS == result) result = libxs_predict_build(vmodel, 0, 2, 0.0);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "build rejected a corpus with absent inputs\n");
    }
  }
  /**
   * Two models over one corpus: vmodel is left to the fingerprint and carries
   * the value checks, model forces INTERPOLATE so that what interpolated[]
   * reports is the absent-coordinate override and not the mode choice.  Reading
   * a value off the forced model would score the polynomial over rank rather
   * than the neighbourhood, which is a different question.
   */
  if (EXIT_SUCCESS == result) {
    result = check_eval(model, "complete query", 21.0, 0.0, 21.0, 8.0, 1);
  }
  if (EXIT_SUCCESS == result) {
    result = check_eval(vmodel, "complete query value", 21.0, 0.0, 21.0, 2.0, -1);
  }
  /**
   * An absent query coordinate must not be answered by the polynomial: that
   * path reports confidence 1.0 unconditionally, so it would advertise the
   * query that supplied less information as the most certain kind there is.
   */
  if (EXIT_SUCCESS == result) {
    result = check_eval(model, "absent query coordinate",
      21.0, absent(), 21.0, 4.0, 0);
  }
  if (EXIT_SUCCESS == result) {
    result = check_eval(vmodel, "absent query coordinate value",
      21.0, absent(), 21.0, 4.0, -1);
  }
  /**
   * Absences must not degrade the corpus as a whole.  Deliberately not asserted:
   * that a complete query finds an entry which agrees on every coordinate it
   * has.  An entry absent in one coordinate is weaker evidence by construction -
   * the rescaling makes agreement on one of two count for less than agreement on
   * two of two - so a fully comparable neighbour that agrees slightly less well
   * can and does outrank it.  Asserting otherwise would encode an expectation
   * available-case distance never offered.
   */
  if (EXIT_SUCCESS == result) {
    double err = 0;
    int n = 0;
    for (i = 1; i < NENTRY; ++i) {
      double inputs[2], predicted = 0, expect;
      fill(inputs, &expect, i);
      libxs_predict_eval(NULL, vmodel, inputs, &predicted, NULL, 1);
      err += LIBXS_FABS(predicted - expect);
      ++n;
    }
    err = (0 < n) ? (err / n) : 0;
    if (err > 3.0) {
      fprintf(stderr, "mean error %f over complete queries is too high\n", err);
      result = EXIT_FAILURE;
    }
  }
  /* a round trip re-derives what it needs: the format carries no flag */
  if (EXIT_SUCCESS == result) {
    size_t size = 0;
    void* buffer;
    libxs_predict_save(model, NULL, &size);
    buffer = malloc(size);
    if (NULL != buffer) {
      if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &size)) {
        libxs_predict_t* loaded = libxs_predict_load(buffer, size);
        if (NULL != loaded) {
          result = check_eval(loaded, "loaded, absent query",
            21.0, absent(), 21.0, 4.0, 0);
          libxs_predict_destroy(loaded);
        }
        else {
          fprintf(stderr, "a model with absent inputs failed to load\n");
          result = EXIT_FAILURE;
        }
      }
      else result = EXIT_FAILURE;
      free(buffer);
    }
    else result = EXIT_FAILURE;
  }
  /* a tree cannot record a direction for an absent coordinate: build must fail */
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_decompose(rfmodel, LIBXS_PREDICT_RF);
    if (EXIT_SUCCESS == libxs_predict_build(rfmodel, 0, 2, 0.0)) {
      fprintf(stderr, "RF accepted absent inputs instead of refusing\n");
      result = EXIT_FAILURE;
    }
  }
  /* the loader admits an empty input field only when asked to */
  if (EXIT_SUCCESS == result) {
    FILE* out = fopen(CSVFILE, "w");
    if (NULL != out) {
      fprintf(out, "a,b,y\n");
      for (i = 0; i < NENTRY; ++i) {
        if (0 == (i % 8)) fprintf(out, "%i,,%i\n", i, i);
        else fprintf(out, "%i,%i,%i\n", i, i % 7, i);
      }
      fclose(out);
      { libxs_predict_t* strict = libxs_predict_create(2, 1);
        libxs_predict_t* lenient = libxs_predict_create(2, 1);
        if (NULL != strict && NULL != lenient) {
          const int nstrict = libxs_predict_load_csv(strict, CSVFILE, NULL,
            "a,b", "y", NULL, 0, NULL);
          int nlenient;
          libxs_predict_set_missing(lenient, 1);
          nlenient = libxs_predict_load_csv(lenient, CSVFILE, NULL,
            "a,b", "y", NULL, 0, NULL);
          if (NENTRY != nlenient || nstrict >= nlenient) {
            fprintf(stderr, "loader: strict=%i lenient=%i, expected %i lenient"
              " and fewer strict\n", nstrict, nlenient, NENTRY);
            result = EXIT_FAILURE;
          }
        }
        else result = EXIT_FAILURE;
        libxs_predict_destroy(lenient);
        libxs_predict_destroy(strict);
      }
      remove(CSVFILE);
    }
    else result = EXIT_FAILURE;
  }
  libxs_predict_destroy(rfmodel);
  libxs_predict_destroy(vmodel);
  libxs_predict_destroy(model);
  if (EXIT_SUCCESS == result) fprintf(stdout, "OK\n");
  return result;
}
