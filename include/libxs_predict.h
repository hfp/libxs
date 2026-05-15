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


/** Opaque prediction model type. */
LIBXS_EXTERN_C typedef struct libxs_predict_t libxs_predict_t;

/** Per-output confidence information returned by predict. */
LIBXS_EXTERN_C typedef struct libxs_predict_info_t {
  /** Predicted output values (noutputs elements). */
  const double* values;
  /** Per-output error bound from truncation (noutputs elements). */
  const double* error;
  /** Per-output predictability flag: non-zero if fingerprint norms decay. */
  const int* reliable;
  /** Number of outputs. */
  int noutputs;
  /** Cluster index assigned to the query (-1 if blended). */
  int cluster;
} libxs_predict_info_t;


/**
 * Create a prediction model for the given input/output dimensionality.
 * ninputs:  number of input parameters (M) per entry.
 * noutputs: number of output parameters (N) per entry.
 * Returns NULL on failure (invalid arguments or allocation failure).
 */
LIBXS_API libxs_predict_t* libxs_predict_create(int ninputs, int noutputs);

/** Destroy prediction model (NULL is accepted). */
LIBXS_API void libxs_predict_destroy(libxs_predict_t* model);

/**
 * Push one training entry (incremental).
 * inputs:  M values (input parameters).
 * outputs: N values (output parameters).
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
 * quality:   compression-vs-accuracy tradeoff in [0,1].
 *            0.0 = maximum compression (aggressive truncation).
 *            1.0 = maximum fidelity (minimal truncation).
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 * May be called again after pushing additional entries (rebuilds).
 */
LIBXS_API int libxs_predict_build(libxs_predict_t* model,
  int nclusters, double quality);

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
 * Query model statistics after build.
 * nclusters: receives the number of clusters (may be NULL).
 * nentries:  receives total number of pushed entries (may be NULL).
 * compression: receives the compression ratio (may be NULL).
 */
LIBXS_API void libxs_predict_query(const libxs_predict_t* model,
  int* nclusters, int* nentries, double* compression);

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

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PREDICT_H*/
