/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PREDICT_XGB_H
#define PREDICT_XGB_H

#include <libxs/libxs_predict.h>
#include <libxs/libxs_timer.h>
#include <xgboost/c_api.h>

/**
 * Largest attested value set an output may carry and still be posed to XGBoost
 * as a classification.  A wider output falls back to regression, which is
 * reported rather than silently substituted: the two tasks are not comparable.
 */
#define PREDICT_XGB_MAXCLASS 64

/** Prediction configuration: whole model, inference mode, plain 2D output. */
#define PREDICT_XGB_PREDCFG "{\"type\":0,\"training\":false," \
  "\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":false}"

/** Inplace prediction over a single row: no DMatrix is built, so the call is the
 *  descent through the trees rather than the construction of a matrix to hold
 *  one query. Anything else measures the interface instead of the model. */
#define PREDICT_XGB_DENSECFG "{\"type\":0,\"iteration_begin\":0," \
  "\"iteration_end\":0,\"strict_shape\":false,\"cache_id\":0,\"missing\":NaN}"

/** One row of floats, as the array interface the inplace prediction reads. */
#define PREDICT_XGB_ARRAY "{\"data\":[%lu,true],\"shape\":[1,%i]," \
  "\"typestr\":\"<f4\",\"version\":3}"

/** Rows the single-query latency is measured over, a bound rather than a share:
 *  it is a per-call cost, so a few thousand calls settle it and every further
 *  one is charged to the comparison for nothing. XGB_LATENCY overrides it, and
 *  zero declines the measurement. */
#if !defined(PREDICT_XGB_LATENCY)
#  define PREDICT_XGB_LATENCY 4096
#endif


/**
 * What the comparison cost, in seconds. The parts are reported apart rather than
 * summed because they are not one quantity, and because only one of them is what
 * a LIBXS build time compares against: marshal is the corpus rewritten into
 * XGBoost's layout, train is the boosting rounds, predict is one batched call
 * over every row, and query is that same prediction taken one row at a time.
 *
 * That last pair is the distinction the summed figure hides. A batched inference
 * and a single-query inference are different workloads, so a batch on one side
 * against a per-query loop on the other compares the shape of two interfaces and
 * not the cost of two models.
 */
typedef struct predict_xgb_time_t {
  double marshal, train, predict, query;
  /** Queries the last figure is a total over, zero where it was not taken. */
  int nquery;
} predict_xgb_time_t;


static int predict_xgb_geti(const char* name, int fallback)
{
  const char* const env = getenv(name);
  int result = fallback;
  if (NULL != env && '\0' != *env) result = atoi(env);
  return result;
}


static double predict_xgb_getd(const char* name, double fallback)
{
  const char* const env = getenv(name);
  double result = fallback;
  if (NULL != env && '\0' != *env) result = atof(env);
  return result;
}


/**
 * Distinct values of one output over the training subset, ascending.  Returns
 * the count, or capacity+1 when the set is wider than capacity (the caller must
 * treat that as "too wide to classify" rather than as a truncated set).
 */
static int predict_xgb_support(const float labels[], int n,
  double values[], int capacity)
{
  int result = 0, i, ok = 1;
  for (i = 0; i < n && 0 != ok; ++i) {
    const double value = (double)labels[i];
    int j = 0;
    while (j < result && values[j] < value) ++j;
    if (j == result || values[j] != value) {
      if (result < capacity) {
        int k;
        for (k = result; k > j; --k) values[k] = values[k-1];
        values[j] = value;
        ++result;
      }
      else {
        result = capacity + 1;
        ok = 0;
      }
    }
  }
  return result;
}


/**
 * Regression objective: XGB_REGOBJ wins, else what the caller proposes, else
 * squared error.  A sample reporting mean absolute error should propose
 * reg:absoluteerror, because training on squared error and scoring on absolute
 * error is the same mismatch libxs_predict_set_central removes on the LIBXS side.
 */
static const char* predict_xgb_regobj(const char* proposed)
{
  const char* const env = getenv("XGB_REGOBJ");
  const char* result = "reg:squarederror";
  if (NULL != env && '\0' != *env) result = env;
  else if (NULL != proposed && '\0' != *proposed) result = proposed;
  return result;
}


static int predict_xgb_params(BoosterHandle booster, int nclass,
  const char* regobj)
{
  char buffer[64];
  int result = 0;
  if (0 < nclass) {
    result |= XGBoosterSetParam(booster, "objective", "multi:softprob");
    LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i", nclass);
    result |= XGBoosterSetParam(booster, "num_class", buffer);
  }
  else {
    result |= XGBoosterSetParam(booster, "objective", regobj);
  }
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i",
    predict_xgb_geti("XGB_DEPTH", 6));
  result |= XGBoosterSetParam(booster, "max_depth", buffer);
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%g",
    predict_xgb_getd("XGB_ETA", 0.1));
  result |= XGBoosterSetParam(booster, "eta", buffer);
  LIBXS_SNPRINTF(buffer, sizeof(buffer), "%i",
    predict_xgb_geti("XGB_NTHREAD", 0));
  result |= XGBoosterSetParam(booster, "nthread", buffer);
  result |= XGBoosterSetParam(booster, "seed", "0");
  result |= XGBoosterSetParam(booster, "verbosity", "0");
  return result;
}


/**
 * Train one booster over dtrain and write its prediction for every row of dall
 * into predicted[i*stride], confidence[i*stride] (confidence may be NULL, and
 * is the winning class probability for a classification, 0 for a regression).
 *
 * x and trained are the rows in XGBoost's layout and the mask that says which of
 * them were trained on; they are needed only for the single-query latency, which
 * is taken over rows the booster did not see, and may be NULL with dt.
 */
static int predict_xgb_output(DMatrixHandle dtrain, DMatrixHandle dall,
  int ntotal, int nclass, const double values[],
  double predicted[], double confidence[], int stride, const char* regobj,
  const float* x, int ninputs, const char trained[], predict_xgb_time_t* dt)
{
  BoosterHandle booster = NULL;
  const int nrounds = predict_xgb_geti("XGB_ROUNDS", 200);
  libxs_timer_tick_t tick;
  int result = XGBoosterCreate(&dtrain, 1, &booster);
  if (0 == result) result = predict_xgb_params(booster, nclass, regobj);
  tick = libxs_timer_tick();
  if (0 == result) {
    int i;
    for (i = 0; i < nrounds && 0 == result; ++i) {
      result = XGBoosterUpdateOneIter(booster, i, dtrain);
    }
  }
  if (NULL != dt) dt->train += libxs_timer_duration(tick, libxs_timer_tick());
  tick = libxs_timer_tick();
  if (0 == result) {
    const float* out = NULL;
    const bst_ulong* shape = NULL;
    bst_ulong ndim = 0;
    result = XGBoosterPredictFromDMatrix(booster, dall,
      PREDICT_XGB_PREDCFG, &shape, &ndim, &out);
    /* stopped before the read-out below, which is this harness picking a class
       out of the probabilities rather than XGBoost predicting anything */
    if (NULL != dt) dt->predict += libxs_timer_duration(tick, libxs_timer_tick());
    if (0 == result && NULL != out && 0 < ndim) {
      const int width = (1 < ndim) ? (int)shape[1] : 1;
      int i;
      for (i = 0; i < ntotal; ++i) {
        if (0 < nclass && 1 < width) {
          const float* const row = out + (size_t)i * width;
          int best = 0, c;
          for (c = 1; c < width; ++c) {
            if (row[c] > row[best]) best = c;
          }
          predicted[(size_t)i * stride] = values[best];
          if (NULL != confidence) {
            confidence[(size_t)i * stride] = (double)row[best];
          }
        }
        else {
          predicted[(size_t)i * stride] = (double)out[i];
          if (NULL != confidence) confidence[(size_t)i * stride] = 0.0;
        }
      }
    }
  }
  { const int nlat = predict_xgb_geti("XGB_LATENCY", PREDICT_XGB_LATENCY);
    if (0 == result && NULL != dt && NULL != x && 0 < nlat && 0 < ninputs) {
      float* const qrow = (float*)malloc((size_t)ninputs * sizeof(float));
      if (NULL != qrow) {
        const float* out = NULL;
        const bst_ulong* shape = NULL;
        bst_ulong ndim = 0;
        char array[192];
        int n = 0, i;
        /* the row the query is read from does not move, so the interface that
           names it is written once and the loop measures the prediction */
        LIBXS_SNPRINTF(array, sizeof(array), PREDICT_XGB_ARRAY,
          (unsigned long)(size_t)qrow, ninputs);
        /**
         * One row cannot occupy more than one thread, so a booster left at the
         * training thread count pays a barrier per query for nothing: measured
         * twice at the same size, the figure moved 255 to 105 us with the
         * threads left in place, and the difference was OpenMP arrival time
         * rather than anything about the model. A caller measuring latency would
         * set this, and without it the number is not reproducible.
         */
        XGBoosterSetParam(booster, "nthread", "1");
        /* the first call allocates the buffer the result is returned in, and
           charging that to a query would charge it to every query */
        result = XGBoosterPredictFromDense(booster, array, PREDICT_XGB_DENSECFG,
          NULL, &shape, &ndim, &out);
        tick = libxs_timer_tick();
        for (i = 0; i < ntotal && n < nlat && 0 == result; ++i) {
          if (NULL != trained && 0 != trained[i]) continue;
          memcpy(qrow, x + (size_t)i * ninputs,
            (size_t)ninputs * sizeof(float));
          result = XGBoosterPredictFromDense(booster, array,
            PREDICT_XGB_DENSECFG, NULL, &shape, &ndim, &out);
          ++n;
        }
        if (0 == result) {
          dt->query += libxs_timer_duration(tick, libxs_timer_tick());
          dt->nquery += n;
        }
        free(qrow);
      }
    }
  }
  XGBoosterFree(booster);
  return result;
}


/**
 * Train XGBoost on exactly the entries LIBXS was built from and predict every
 * row, so both models are scored on one split.  Handing this a mask other than
 * the one the LIBXS model saw invalidates the comparison with no symptom to
 * notice, which is why the mask is an argument rather than recomputed here.
 *
 * source:    corpus the LIBXS model was pushed from (need not be built).
 * trained:   ntotal flags, non-zero where the entry was trained on.
 * classify:  noutputs flags requesting classification over the attested value
 *            set instead of regression (NULL requests regression throughout).
 * predicted: ntotal*noutputs values written, in user space.
 * confidence: ntotal*noutputs values written (may be NULL).
 * task:      noutputs values written (may be NULL) reporting what was actually
 *            posed per output: 0 regression, 1 constant, >1 classification over
 *            that many attested values.  An output asked to classify a set
 *            wider than PREDICT_XGB_MAXCLASS reports regression instead, which
 *            is a different task and must not be compared as if it were one.
 * regobj:    regression objective proposed for this corpus (NULL for the
 *            default); XGB_REGOBJ overrides it.
 * dt:        what each phase cost (may be NULL); see predict_xgb_time_t for why
 *            the phases are not summed.
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
static int predict_xgb(const libxs_predict_t* source, int ntotal,
  int ninputs, int noutputs, const char trained[], const int classify[],
  double predicted[], double confidence[], int task[], const char* regobj,
  predict_xgb_time_t* dt)
{
  double* row = (double*)malloc((size_t)(ninputs + noutputs) * sizeof(double));
  float* x = (float*)malloc((size_t)ntotal * ninputs * sizeof(float));
  float* y = (float*)malloc((size_t)ntotal * noutputs * sizeof(float));
  float* xtrain = (float*)malloc((size_t)ntotal * ninputs * sizeof(float));
  float* ytrain = (float*)malloc((size_t)ntotal * sizeof(float));
  int result = EXIT_FAILURE;
  if (NULL != row && NULL != x && NULL != y
    && NULL != xtrain && NULL != ytrain)
  {
    DMatrixHandle dtrain = NULL, dall = NULL;
    libxs_timer_tick_t mt = libxs_timer_tick();
    int ntrain = 0, i, j, status;
    if (NULL != dt) memset(dt, 0, sizeof(*dt));
    for (i = 0; i < ntotal; ++i) {
      libxs_predict_get(source, i, row, row + ninputs);
      for (j = 0; j < ninputs; ++j) {
        x[(size_t)i * ninputs + j] = (float)row[j];
      }
      for (j = 0; j < noutputs; ++j) {
        y[(size_t)i * noutputs + j] = (float)row[ninputs+j];
      }
      if (NULL == trained || 0 != trained[i]) {
        for (j = 0; j < ninputs; ++j) {
          xtrain[(size_t)ntrain * ninputs + j] = x[(size_t)i * ninputs + j];
        }
        ++ntrain;
      }
    }
    status = XGDMatrixCreateFromMat(xtrain, (bst_ulong)ntrain,
      (bst_ulong)ninputs, -1.0f, &dtrain);
    if (0 == status) {
      status = XGDMatrixCreateFromMat(x, (bst_ulong)ntotal,
        (bst_ulong)ninputs, -1.0f, &dall);
    }
    if (NULL != dt) dt->marshal += libxs_timer_duration(mt, libxs_timer_tick());
    for (j = 0; j < noutputs && 0 == status; ++j) {
      double values[PREDICT_XGB_MAXCLASS];
      int nclass = 0, n = 0;
      mt = libxs_timer_tick();
      for (i = 0; i < ntotal; ++i) {
        if (NULL == trained || 0 != trained[i]) {
          ytrain[n++] = y[(size_t)i * noutputs + j];
        }
      }
      if (NULL != classify && 0 != classify[j]) {
        nclass = predict_xgb_support(ytrain, n, values,
          PREDICT_XGB_MAXCLASS);
        if (PREDICT_XGB_MAXCLASS < nclass) nclass = 0;
      }
      if (1 < nclass) {
        for (i = 0; i < n; ++i) {
          int k = 0;
          while (k < nclass && values[k] != (double)ytrain[i]) ++k;
          ytrain[i] = (float)k;
        }
      }
      if (1 == nclass) {
        for (i = 0; i < ntotal; ++i) {
          predicted[(size_t)i * noutputs + j] = values[0];
          if (NULL != confidence) confidence[(size_t)i * noutputs + j] = 1.0;
        }
      }
      else {
        status = XGDMatrixSetFloatInfo(dtrain, "label", ytrain,
          (bst_ulong)n);
        if (NULL != dt) {
          dt->marshal += libxs_timer_duration(mt, libxs_timer_tick());
        }
        if (0 == status) {
          status = predict_xgb_output(dtrain, dall, ntotal, nclass, values,
            predicted + j, (NULL != confidence) ? (confidence + j) : NULL,
            noutputs, predict_xgb_regobj(regobj), x, ninputs, trained, dt);
        }
      }
      if (NULL != task) task[j] = nclass;
    }
    if (0 != status) {
      fprintf(stderr, "XGBoost error: %s\n", XGBGetLastError());
    }
    else {
      result = EXIT_SUCCESS;
    }
    XGDMatrixFree(dall);
    XGDMatrixFree(dtrain);
  }
  free(ytrain);
  free(xtrain);
  free(y);
  free(x);
  free(row);
  return result;
}

#endif /*PREDICT_XGB_H*/
