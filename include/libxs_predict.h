/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_PREDICT_H
#define LIBXS_PREDICT_H

#include "libxs_math.h"
#include "libxs_sync.h"
#include "libxs_str.h"


/** Prediction mode flags (ORable). */
typedef enum libxs_predict_mode_t {
  LIBXS_PREDICT_AUTO        = 0,
  LIBXS_PREDICT_INTERPOLATE = 1,
  LIBXS_PREDICT_CLASSIFY    = 2,
  LIBXS_PREDICT_TEMPORAL    = 4
} libxs_predict_mode_t;

/** Output transform applied during push/eval. */
typedef enum libxs_predict_transform_t {
  LIBXS_PREDICT_IDENTITY = 0,
  LIBXS_PREDICT_LOG      = 1,
  LIBXS_PREDICT_SQRT     = 2
} libxs_predict_transform_t;

/** Input decomposition / feature selection mode. */
typedef enum libxs_predict_decompose_t {
  LIBXS_PREDICT_RAW     = 0,
  LIBXS_PREDICT_SPREAD  = 1,
  LIBXS_PREDICT_PCA     = 2,
  LIBXS_PREDICT_SETDIFF = 3,
  LIBXS_PREDICT_FISHER  = 4,
  LIBXS_PREDICT_RF      = 5
} libxs_predict_decompose_t;

/** Opaque prediction model type. */
LIBXS_EXTERN_C typedef struct libxs_predict_t libxs_predict_t;

/** Per-output confidence information returned by predict. */
LIBXS_EXTERN_C typedef struct libxs_predict_info_t {
  /** Predicted output values (noutputs elements). */
  const double* values;
  /** Per-output error bound from truncation (noutputs elements). */
  const double* error;
  /** Per-output confidence from kNN vote (noutputs elements, 0..1). */
  const double* confidence;
  /** Per-output variance among k nearest neighbors (noutputs elements). */
  const double* variance;
  /** Per-output mode used: non-zero if polynomial interpolation was applied. */
  const int* interpolated;
  /** Number of outputs. */
  int noutputs;
  /** Cluster index assigned to the query (-1 if blended). */
  int cluster;
  /** Distance to assigned cluster centroid relative to cluster radius (0..inf). */
  double distance;
} libxs_predict_info_t;

/** Model statistics returned by libxs_predict_query. */
LIBXS_EXTERN_C typedef struct libxs_predict_query_t {
  /** Compression ratio (raw size / model size). */
  double compression;
  /** Polynomial order used (after auto-optimization if order <= 0). */
  int order;
  /** Number of clusters. */
  int nclusters;
  /** Total number of pushed entries. */
  int nentries;
  /** GSS iterations performed during quality optimization (0 if quality >= 0). */
  int iterations;
  /** Auto-detected differencing order (0 if DIFF not enabled or not needed). */
  int diff_order;
} libxs_predict_query_t;


/**
 * Create a prediction model for the given input/output dimensionality.
 * ninputs:  number of input parameters (M) per entry.
 * noutputs: number of output parameters (N) per entry.
 * Returns NULL on failure (invalid arguments or allocation failure).
 */
LIBXS_API libxs_predict_t* libxs_predict_create(int ninputs, int noutputs);

/** Destroy prediction model (NULL is accepted). */
LIBXS_API void libxs_predict_destroy(libxs_predict_t* model);

/** Return pointer to the model's internal lock (for use as lock argument). */
LIBXS_API libxs_lock_t* libxs_predict_lock(libxs_predict_t* model);

/**
 * Set prediction mode (ORable flags from libxs_predict_mode_t).
 * LIBXS_PREDICT_AUTO (0): fingerprint decides per output (default).
 * LIBXS_PREDICT_INTERPOLATE: force polynomial for all outputs.
 * LIBXS_PREDICT_CLASSIFY: force kNN vote for all outputs.
 * LIBXS_PREDICT_TEMPORAL: timeseries mode -- recency weighting,
 *   continuous output (no snap), and horizon smoothing.
 */
LIBXS_API void libxs_predict_set_mode(libxs_predict_t* model, int mode);

/**
 * Set per-dimension input weights for distance computation.
 * weights: ninputs values (NULL resets to uniform weighting).
 * Larger weight = dimension contributes more to distance.
 * Must be called before libxs_predict_build.
 */
LIBXS_API void libxs_predict_set_weights(libxs_predict_t* model,
  const double weights[]);

/**
 * Set per-output transform applied transparently during push/eval.
 * output: output index (0-based), or -1 to set all outputs.
 * transform: LIBXS_PREDICT_IDENTITY (default), _LOG, or _SQRT.
 * Push applies the forward transform, eval applies the inverse.
 */
LIBXS_API void libxs_predict_set_transform(libxs_predict_t* model,
  int output, int transform);

/**
 * Set the number of forward-inverse-forward refinement iterations.
 * 0 (default): iterate only when confidence is below threshold.
 * >0: always perform this many refinement iterations per eval.
 * Refinement finds the canonical historical pattern matching the
 * prediction, then re-predicts from it to improve self-consistency.
 */
LIBXS_API void libxs_predict_set_refine(libxs_predict_t* model,
  int iterations);

/**
 * Declare timeseries structure: nseries co-observed series, each with
 * the given window size. ninputs must equal nseries * window, noutputs
 * is the forecast horizon. When set, push(lock, model, values, NULL)
 * accumulates one timestep (nseries values); build constructs sliding
 * windows internally. Must be called before push.
 */
LIBXS_API void libxs_predict_set_series(libxs_predict_t* model,
  int nseries, int window);

/**
 * Set which series index to predict (0-based, default: 0).
 * Only relevant when nseries > 1.
 */
LIBXS_API void libxs_predict_set_target(libxs_predict_t* model, int target);

/**
 * Set input decomposition / feature selection / prediction mode.
 * LIBXS_PREDICT_RAW (default): standard kNN, no input transform.
 * LIBXS_PREDICT_SPREAD: sum/diff modes (for anti-correlated pairs).
 * LIBXS_PREDICT_PCA: principal component rotation + dim. reduction.
 * LIBXS_PREDICT_SETDIFF: auto feature selection via setdiff scores.
 * LIBXS_PREDICT_FISHER: auto feature selection via Fisher criterion.
 * LIBXS_PREDICT_RF: Random Forest classification (per-output).
 */
LIBXS_API void libxs_predict_set_decompose(libxs_predict_t* model,
  int decompose);

/**
 * Enable auto-differencing for non-stationary timeseries.
 * order > 0: explicit differencing order (1 removes linear trend,
 *            2 removes quadratic trend).
 * order = 0: auto-detect from fingerprint decay of the target series.
 *            The build step determines the lowest order that makes
 *            the series approximately stationary.
 * order < 0: disabled (default).
 *
 * At build time the pushed series is differentiated d times before
 * window construction. At eval time the caller provides raw values;
 * the framework differences the query, predicts in diff space, and
 * integrates the result back to absolute values.
 * Requires set_series (timeseries structure).
 * Composes with pointwise transforms (LOG/SQRT): the pipeline is
 * push -> accumulate -> diff -> windows -> fwd transform on outputs,
 * eval -> inv transform -> integrate -> absolute values.
 */
LIBXS_API void libxs_predict_set_diff(libxs_predict_t* model, int order);

/**
 * Push one training entry (incremental).
 * inputs:  M values (input parameters).
 * outputs: N values (output parameters), or NULL for timeseries mode
 *          (when set_series was called, inputs has nseries values
 *          representing one timestep; windows are built internally).
 * May be called any number of times before libxs_predict_build.
 * The lock is optional (NULL if single-threaded).
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_predict_push(libxs_lock_t* lock,
  libxs_predict_t* model,
  const double inputs[], const double outputs[]);

/**
 * Build (finalize) the prediction model from pushed entries.
 * Performs clustering, distance-ordering, fingerprinting, and
 * polynomial fitting. Must be called before libxs_predict_eval.
 *
 * nclusters: number of clusters (0 = auto-determine).
 * order:     maximum polynomial order for interpolation.
 *            >0 = use at most this order.
 *             0 = auto-optimize via GSS.
 *            <0 = auto-optimize with |order| GSS iterations.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 * May be called again after pushing additional entries (rebuilds).
 */
LIBXS_API int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order);

/**
 * Per-thread form of libxs_predict_build. All threads must call
 * this collectively with the same model/nclusters/order.
 * tid==0 performs the build; other threads spin-wait.
 * The lock is optional (NULL is accepted).
 */
LIBXS_API int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  int tid, int ntasks);

/**
 * Predict output parameters for a given input.
 * inputs: M values (query input parameters).
 * outputs: N values written (may be NULL if only info is needed).
 * info: optional detailed result (may be NULL). If non-NULL,
 *       the info structure is valid until the next call to
 *       libxs_predict_eval on the same model with the same lock,
 *       or libxs_predict_destroy.
 * nblend: number of nearest clusters to blend (1 = nearest only,
 *         0 = auto based on distance ratios).
 * The lock is optional (NULL if single-threaded); concurrent eval
 * calls with distinct locks or NULL are safe on a built model.
 */
LIBXS_API void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend);

/**
 * Inverse prediction: find inputs that produce desired outputs.
 * target_outputs: N desired output values.
 * inputs: M values written (best-matching input parameters).
 * info: optional (may be NULL). Confidence reflects match quality.
 * Discrete (classify-mode) outputs are matched exactly as constraints;
 * continuous (interpolate-mode) outputs are matched by proximity.
 */
LIBXS_API void libxs_predict_inverse(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double target_outputs[], double inputs[],
  libxs_predict_info_t* info);

/** Query model statistics after build. */
LIBXS_API void libxs_predict_query(const libxs_predict_t* model,
  libxs_predict_query_t* info);

/**
 * Retrieve the i-th pushed entry (0-based).
 * inputs: receives M values (may be NULL).
 * outputs: receives N values (may be NULL).
 */
LIBXS_API void libxs_predict_get(const libxs_predict_t* model, int index,
  double inputs[], double outputs[]);

/**
 * Predict output parameters for a batch of inputs.
 * inputs_batch: count*M values (contiguous, row-major).
 * outputs_batch: count*N values written.
 * count: number of queries in the batch.
 * nblend: number of nearest clusters to blend per query.
 */
LIBXS_API void libxs_predict_eval_batch(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend);

/** Per-thread form of libxs_predict_eval_batch. */
LIBXS_API void libxs_predict_eval_batch_task(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend, int tid, int ntasks);

/**
 * Save built model to a binary buffer.
 * buffer: destination (may be NULL to query required size).
 * size: on input, available buffer size in bytes;
 *       on output, bytes written (or required if buffer is NULL).
 * Returns EXIT_SUCCESS or EXIT_FAILURE (model not built, buffer too small).
 */
LIBXS_API int libxs_predict_save(const libxs_predict_t* model,
  void* buffer, size_t* size);

/**
 * Load a model from a binary buffer (previously saved with libxs_predict_save).
 * Returns a ready-to-eval model, or NULL on failure (corrupt data, version mismatch).
 * The returned model does not reference the buffer after this call returns.
 */
LIBXS_API libxs_predict_t* libxs_predict_load(
  const void* buffer, size_t size);

/**
 * Load delimited text (CSV) and push entries into a prediction model.
 * filename: path to the delimited text file.
 * delims:   string of delimiter characters (NULL = auto-detect: ;,\t space).
 *           Any character in the string acts as a field separator.
 * inputs:   comma-separated column names or numeric indices for input
 *           parameters, or NULL for sequential columns 0..ninputs-1.
 * outputs:  comma-separated column names or numeric indices for output
 *           parameters, or NULL for sequential columns ninputs..ninputs+noutputs-1.
 *
 * The number of tokens in each string must match the model's ninputs/noutputs
 * respectively (as set at creation time).
 *
 * Each token is matched case-insensitively against the header line.
 * If no header match is found, the token is parsed as a numeric
 * column index (0-based).
 *
 * Rows where any selected column fails numeric parsing are skipped
 * (handles header lines and non-numeric fields automatically).
 * header: if non-NULL, receives the CSV header line (up to header_size
 *         bytes). Use libxs_strtoken(header, sep, col, &len) to extract
 *         column names by index.
 * delim_out: if non-NULL, receives the detected (or given) delimiter
 *         character.
 * Returns the number of entries successfully pushed, or -1 on I/O error.
 */
LIBXS_API int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char inputs[], const char outputs[],
  char header[], int header_size, char* delim_out);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PREDICT_H*/
