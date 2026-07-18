/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_NGRAM_H
#define LIBXS_NGRAM_H

#include "libxs_reg.h"

/** Maximum context order (tokens of history) a model can store. */
#define LIBXS_NGRAM_ORDER_MAX 6
/** Maximum successors retained per context (space-saving count sketch). */
#define LIBXS_NGRAM_SUCC_MAX 8


/** A single successor token and how often it followed a context. */
LIBXS_EXTERN_C typedef struct libxs_ngram_succ_t {
  unsigned int id;
  unsigned int count;
} libxs_ngram_succ_t;

/**
 * Context->successor record for one context of length n. The successor list is
 * bounded to LIBXS_NGRAM_SUCC_MAX by a space-saving policy (the least frequent
 * entry is evicted), so total may exceed the sum of the retained counts.
 */
LIBXS_EXTERN_C typedef struct libxs_ngram_entry_t {
  unsigned int total;
  unsigned int nsucc;
  libxs_ngram_succ_t succ[LIBXS_NGRAM_SUCC_MAX];
} libxs_ngram_entry_t;

/**
 * Count-based variable-order n-gram model over unsigned token ids, backed by a
 * registry keyed on the context. Callers own the struct (stack allocation is
 * fine); the internal store and backoff tables are heap-managed by create and
 * destroy. Token id 0 is reserved (treated as absent context / never stored).
 */
LIBXS_EXTERN_C typedef struct libxs_ngram_t {
  libxs_registry_t* store;
  unsigned int* unifreq;
  double unifreq_total;
  unsigned int unifreq_size;
  unsigned int backoff_ids[LIBXS_NGRAM_SUCC_MAX];
  int backoff_count;
  int maxorder;
} libxs_ngram_t;

/** Per-order footprint returned by libxs_ngram_stats (index 0 unused). */
LIBXS_EXTERN_C typedef struct libxs_ngram_stats_t {
  long keys[LIBXS_NGRAM_ORDER_MAX + 1];
  double obs[LIBXS_NGRAM_ORDER_MAX + 1];
  long saturated[LIBXS_NGRAM_ORDER_MAX + 1];
  size_t entries;
  size_t nbytes;
} libxs_ngram_stats_t;


/**
 * Initialize a model that stores contexts up to maxorder (clamped to
 * 1..LIBXS_NGRAM_ORDER_MAX). Returns EXIT_SUCCESS, or EXIT_FAILURE if the
 * backing store cannot be created.
 */
LIBXS_API int libxs_ngram_create(libxs_ngram_t* model, int maxorder);

/** Release all model state (NULL is accepted; the struct becomes empty). */
LIBXS_API void libxs_ngram_destroy(libxs_ngram_t* model);

/**
 * Observe that succ follows the history hist[0..hlen-1]: increments every
 * context length 1..min(maxorder,hlen) that ends at hist[hlen-1]. Contexts
 * containing a 0 id (or a 0 successor) are skipped.
 */
LIBXS_API void libxs_ngram_observe(libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int succ);

/**
 * Build the global unigram distribution and the most-frequent-token backoff
 * list from the order-1 counts. Call once after all observations and before
 * libxs_ngram_prob / libxs_ngram_predict. vocab is the largest token id in use.
 */
LIBXS_API void libxs_ngram_finalize(libxs_ngram_t* model, unsigned int vocab);

/** Look up the record for the length-n context ending at hist[hlen-1]. */
LIBXS_API const libxs_ngram_entry_t* libxs_ngram_lookup(
  const libxs_ngram_t* model, const unsigned int hist[], int hlen, int n);

/**
 * Interpolated backoff probability of next given the history: blends orders
 * 1..min(maxorder,hlen) with weight total/(total+1) per order over the global
 * unigram prior, and never returns below a uniform floor.
 */
LIBXS_API double libxs_ngram_prob(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int next);

/**
 * Top-k successors from the longest context that has a record (hard backoff),
 * falling back to the global most-frequent list when none matches. When order
 * is non-NULL it receives the context length that produced the result (0 for
 * the global fallback), which quantifies how well grounded the prediction is.
 * Returns the number of ids written to out_ids.
 */
LIBXS_API int libxs_ngram_predict(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int out_ids[], int k,
  int* order);

/** Collect per-order key/observation counts and live footprint. */
LIBXS_API int libxs_ngram_stats(const libxs_ngram_t* model,
  libxs_ngram_stats_t* stats);

#endif /*LIBXS_NGRAM_H*/
