/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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

/**
 * Sentinel: pass as window to libxs_predict_set_series to request
 * fingerprint-guided auto-detection at build time.
 */
#define LIBXS_PREDICT_AUTO_WINDOW 0


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
  /**
   * Sentinel: choose the mode at build by trial, see
   * libxs_predict_set_decompose.
   */
  LIBXS_PREDICT_AUTO_DECOMPOSE = -1,
  LIBXS_PREDICT_RAW     = 0,
  LIBXS_PREDICT_SPREAD  = 1,
  LIBXS_PREDICT_PCA     = 2,
  LIBXS_PREDICT_SETDIFF = 3,
  LIBXS_PREDICT_FISHER  = 4,
  LIBXS_PREDICT_RF      = 5,
  LIBXS_PREDICT_HKNN    = 6
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
  /** Per-output lower prediction interval (noutputs elements, NULL if disabled). */
  const double* lower;
  /** Per-output upper prediction interval (noutputs elements, NULL if disabled). */
  const double* upper;
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
  /**
   * Effective sliding-window size used at build (0 if series mode off).
   * Equals the caller-provided window when set_series was called with a
   * positive value; equals the auto-selected window when the sentinel
   * LIBXS_PREDICT_AUTO_WINDOW (0) was passed. Read this to size the raw
   * window buffer supplied to libxs_predict_eval.
   */
  int window;
  /** Window views actually built (see set_series_bank; 1 when disabled). */
  int nbank;
  /**
   * Entries in the LARGEST cluster, which is the number of candidates a
   * neighbour query scans in the worst case: local evidence is gathered by
   * walking a cluster, so scoring cost grows with this and not with nentries.
   *
   * Read it to know what a query costs before paying for it. A model built with
   * one cluster puts every entry here, so scoring is linear in the corpus - easy
   * to miss until a run that was affordable at one scale is not at the next, and
   * the reason this is reported rather than left to be discovered.
   */
  int nscan;
  /**
   * Candidates an AVERAGE query scans: sum over clusters of n_c^2 divided by the
   * entry count, because a query falls in cluster c with probability n_c/N and
   * then walks n_c entries.
   *
   * Read together with nscan this separates two different problems that cost the
   * same at the worst case. Near nentries/nclusters, the partition is even and
   * only an index can reduce the scan. Near nscan, a few large clusters absorb
   * most queries and rebalancing the partition is the cheaper fix.
   */
  double escan;
  /**
   * Decomposition mode in force (see set_decompose). Equals the caller's
   * choice, or the mode selected by trial when the model asked for
   * LIBXS_PREDICT_AUTO_DECOMPOSE.
   */
  int decompose;
} libxs_predict_query_t;

/** Kind of quantity libxs_predict_prob reports for an output. */
typedef enum libxs_predict_pkind_t {
  /** No result (blended or RF output, or model not scoreable). */
  LIBXS_PREDICT_PNONE = 0,
  /** Discrete output: the value is a probability mass. */
  LIBXS_PREDICT_PMASS = 1,
  /** Continuous output: the value is a probability density. */
  LIBXS_PREDICT_PDENSITY = 2
} libxs_predict_pkind_t;

/** Per-output result of libxs_predict_prob. */
LIBXS_EXTERN_C typedef struct libxs_predict_prob_info_t {
  /** P(y|x) per output: mass or density according to kind. */
  const double* prob;
  /** Base-2 log of prob. Primary output: an improbable candidate underflows
   *  prob to zero while its log stays representable. */
  const double* logprob;
  /** Density mode: (y - yhat)/h. Mass mode: 0. */
  const double* zscore;
  /** libxs_predict_pkind_t per output. */
  const int* kind;
  /** Non-zero if the candidate was attested in the local neighborhood. */
  const int* attested;
  /** Distinct attested values in the local support (mass mode). */
  const int* support;
  /** Mass reserved for values outside the local support (mass mode). */
  const double* novel;
  /**
   * Sum of logprob over outputs. This assumes the outputs are conditionally
   * independent given x; it is not a calibrated joint likelihood.
   */
  double total_logprob;
  /** Number of outputs. */
  int noutputs;
  /** Cluster serving the query (-1 if blended). */
  int cluster;
  /** Escape-weight entropy in bits (0 == committed, log2(nexperts) == prior).
   *  Read this to tell whether the escape estimate is still settling. */
  double entropy;
} libxs_predict_prob_info_t;


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
 * LIBXS_PREDICT_TEMPORAL: timeseries mode - recency weighting,
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
 * 0: off, never iterate.
 * <0 (default): iterate only when confidence is below threshold.
 * >0: always perform this many refinement iterations per eval.
 * Refinement finds the canonical historical pattern matching the
 * prediction, then re-predicts from it to improve self-consistency.
 *
 * The inverse it goes through scans every entry, so refinement makes eval
 * cost grow with the corpus rather than with the cluster a query lands in
 * (libxs_predict_query_t::nscan reports the latter and cannot see this).
 * That is affordable at the scale the refinement was measured on and is the
 * dominant per-query cost well before a corpus reaches millions of entries,
 * which is what 0 is for. It also cannot discriminate where no output is
 * interpolated: with only classify-mode outputs every entry matching the
 * predicted label scores equally, so the pattern recovered is the first such
 * entry in push order rather than the nearest.
 */
LIBXS_API void libxs_predict_set_refine(libxs_predict_t* model,
  int iterations);

/**
 * Set multi-cluster smoothing for evaluation.
 * amount=0 (default): no smoothing, only nearest cluster.
 * amount<0 (e.g., -1): auto-derive from fingerprint at build time.
 *   Outputs classified as smooth by the fingerprint get blended;
 *   categorical outputs remain unblended. The effective amount is
 *   proportional to the fraction of smooth outputs.
 * amount>0: manual blending radius (fraction of nearest-cluster
 *   distance). Only smooth-mode outputs are blended; categorical
 *   outputs are not affected.
 */
LIBXS_API void libxs_predict_set_smooth(libxs_predict_t* model,
  double amount);

/**
 * Set round-trip consistency penalty (0..1).
 * When >0 and the refinement round-trip distance exceeds the cluster
 * diameter, confidence is scaled toward the quality threshold:
 *   conf = quality + conf / (1 + amount * rt_dist/dmax)
 * 0 (default): inconsistency only skips refinement.
 * 1: full penalty (halves confidence at rt_dist == dmax).
 */
LIBXS_API void libxs_predict_set_consistency(libxs_predict_t* model,
  double amount);

/**
 * Set quantile level for prediction intervals (0..0.5, default 0).
 * When >0, info->lower and info->upper are filled with the q-th and
 * (1-q)-th weighted quantiles of the k nearest neighbors, scaled by
 * 1/confidence so sparse regions widen naturally.
 * 0 (default): intervals not computed.
 */
LIBXS_API void libxs_predict_set_quantile(libxs_predict_t* model,
  double quantile);

/**
 * Select the central tendency the kNN vote reports for a many-valued output:
 * the weighted mean, or the unweighted median of the same neighbors.
 * The median minimizes absolute error where the neighborhood is skewed, which
 * a heavy-tailed target (earthquake magnitude, discharge) generally is.
 *
 * mode == 1: median, whether or not the vote extrapolates.
 * mode == 2: mean, the historical behavior, never the median.
 * mode == 0 (default) or negative: automatic. Both aggregations are scored at
 *   build time on a held-back tail of the pushed entries and the better one is
 *   kept, per output, so a model whose outputs differ in skew does not have to
 *   settle for one choice. Negative additionally prints the decision.
 */
LIBXS_API void libxs_predict_set_central(libxs_predict_t* model, int mode);

/**
 * Declare timeseries structure: nseries co-observed series, each with
 * the given window size. noutputs is the forecast horizon. When window
 * is positive, ninputs must equal nseries * window + nderiv + naux (see
 * set_series_deriv and set_series_aux); with neither, ninputs == nseries
 * * window. When window is LIBXS_PREDICT_AUTO_WINDOW (0), the framework
 * selects the window at build time via a fingerprint-plateau search
 * bounded above by the ninputs provided at create; the effective window
 * is reported via query.window and must be used by the caller to size
 * the raw window at eval.
 *
 * A negative window selects the window by trial instead: candidates are built
 * as the caller's model and scored on held-back timesteps, minimizing mean
 * absolute error over the whole horizon rather than the next step alone. The
 * magnitude, when at least 4, is offered as one more candidate.
 *
 * Expect roughly an order of magnitude more build time (0.9 s to 6.3 s on the
 * monthly sunspot series), and read the selected window from query.window as
 * usual. Worth requesting where the window is a strong lever - a series whose
 * structure spans a good part of it, forecast a few steps ahead. Not worth it
 * on a large corpus with a long horizon: on ETTh1 at 96 steps the best window
 * beats the default one by 0.7% and costs minutes to find.
 * LIBXS_PREDICT_WINDOW_FOLDS overrides the number of held-back splits (3).
 * When set, push(lock, model, values, NULL) accumulates one timestep
 * (nseries + naux values: the series first, then the auxiliary features);
 * build constructs sliding windows internally. Must be called before push.
 * At eval time the caller supplies the raw window followed by the naux
 * auxiliary values (nseries * window + naux values); the framework applies
 * the target transform to the windowed lags, appends the derivatives, and
 * carries the auxiliary features through unchanged.
 */
LIBXS_API void libxs_predict_set_series(libxs_predict_t* model,
  int nseries, int window);

/**
 * Append nderiv terminal first-differences of the (transformed) target
 * window as additional inputs. The k-th derivative is
 * lag[w-1-k] - lag[w-2-k] in transformed space, emphasizing recent
 * slope. Default 0. Must be called before push.
 */
LIBXS_API void libxs_predict_set_series_deriv(libxs_predict_t* model,
  int nderiv);

/**
 * Declare naux exogenous per-timestep features carried alongside the
 * series. They are not windowed and not transformed: each training
 * window uses the naux values sampled at its prediction origin, and
 * push accepts nseries + naux values per timestep. Useful for calendar
 * or other covariates (e.g. day-of-year). Default 0. Must be called
 * before push.
 */
LIBXS_API void libxs_predict_set_series_aux(libxs_predict_t* model,
  int naux);

/**
 * Average the forecast over nbank views of the window, each seeing a
 * different amount of history: the first view uses the whole window, and
 * each subsequent one halves the lags of the one before it, keeping the
 * most recent. Short and long views fail on different queries, so
 * averaging them removes error neither removes alone - measured on the
 * monthly sunspot series, two views lower the six-month-ahead error 21.8
 * to 20.2 and the one-month-ahead error 17.5 to 16.9.
 *
 * The views share one corpus, one partition and one neighbor index: they
 * differ only in which lags the distance reads, so a second view costs a
 * weight vector rather than a second model. That the partition can be
 * shared is measured, not assumed - independently partitioned views
 * scored within 0.5% of shared ones - because the gain comes from the
 * views seeing different amounts of history and not from their
 * disagreeing about which entries are neighbors.
 *
 * nbank: number of views (1 or less disables, default 1). A view whose
 *   window would fall below two lags is not created, so the effective
 *   count is reported by query.nbank.
 *
 * Requires series mode. Must be called before build. Note that a view
 * zeroes the weight of the lags it does not read, which makes the stored
 * entries unrecoverable on load exactly as feature selection does: such a
 * model still predicts, but libxs_predict_inverse abstains.
 */
LIBXS_API void libxs_predict_set_series_bank(libxs_predict_t* model,
  int nbank);

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
 * LIBXS_PREDICT_HKNN: hierarchical kNN with RF-derived partition
 *   from Hilbert-stratified input space.
 *
 * LIBXS_PREDICT_AUTO_DECOMPOSE selects among the applicable modes at build,
 * by building each on part of the corpus and scoring it on a part held back.
 * The right mode moves with the corpus - a forest wins by 39 to 51% on the
 * crystal corpus and loses to hierarchical kNN on earthquakes - so a fixed
 * default costs about 22% on average against the mode a caller should have
 * picked, and there is no signal short of building that says which it is.
 *
 * Scoring costs one build per candidate (per fold on a timeseries), so this is
 * requested rather than the default. Modes that cannot apply are skipped:
 * SPREAD needs at least two series, and RF and HKNN cannot take absent inputs.
 * A window requested as LIBXS_PREDICT_AUTO_WINDOW is resolved first, under the
 * default mode, and the trial then runs at that window; searching both jointly
 * costs the product of the two and was not worth it.
 *
 * Read the resolved mode from libxs_predict_query_t::decompose.
 */
LIBXS_API void libxs_predict_set_decompose(libxs_predict_t* model,
  int decompose);

/**
 * Set the number of neighbours the vote reads (default 0: derive it).
 *
 * k > 0 pins the count, k = 0 keeps the derived one, and k < 0 selects it at
 * build by trial, per output, on entries held back from a probe build.
 *
 * The derived count is max(5, cluster/3) capped at 32, which is too large on
 * every corpus measured: k=1 takes the crystal corpus from 0.3596 to 0.2838 and
 * the tuned-parameter corpus from 0.3062 to 0.2826. The gain is not an artifact
 * of duplicate rows, which that first corpus has in abundance - a query with an
 * exact match is answered before the vote runs, and splitting the held-out rows
 * by distance to their nearest stored neighbour puts the whole gain on the half
 * that has no close match.
 *
 * There is no formula to switch to. The right count depends on the intrinsic
 * dimension and not on the cluster size: holding the cluster at 56 entries and
 * varying only the dimension of the subspace the data lies on, the best count
 * runs 18, 5, 1 at dimensions 1, 2, 3. It is not a function of dimension alone
 * either - it returns to 8 by dimension 10, where the label is noisy enough per
 * neighbour that averaging pays again - so the classical n^(4/(4+d)) scaling
 * predicts none of it. Hence the trial.
 *
 * The trial is cheap in the way the mode trial is not: k changes nothing about
 * the model, only how many neighbours vote, so one probe build serves every
 * candidate and the rest is evaluation. Choosing per output pays here (about
 * 60% of the available headroom on a 16-output corpus) where choosing the mode
 * per output does not, because the candidates form an ordered grid and a noisy
 * pick still lands near the optimum.
 *
 * The resolved counts are part of the saved model, so a loaded model votes the
 * way the built one did.
 */
LIBXS_API void libxs_predict_set_neighbors(libxs_predict_t* model, int k);

/**
 * Floor that every confidence rescaling at eval pulls toward (0..1, default 0
 * = no rescaling).
 *
 * Three rescalings read it: the cluster-coverage discount, the round-trip
 * consistency penalty (see libxs_predict_set_consistency) and the outlier
 * penalty on a prediction far from its cluster's output distribution. Each
 * replaces conf with floor + s * (conf - floor) for its own s in (0,1], so the
 * floor is the value a heavily discounted confidence approaches.
 *
 * This was previously the same number as the compression threshold passed to
 * libxs_predict_build, which meant asking for compression also moved a runtime
 * knob - and the runtime effect was the larger of the two. Confidence drives how
 * many clusters a query blends, so on a categorical output the old coupling was
 * worth 3.9 points of accuracy against 1.5 for the compression itself, with the
 * same entries dropped either way. They are separate because they are separate
 * decisions.
 *
 * Raising the floor keeps confidence high and so keeps queries out of the
 * blending path; leaving it at 0 disables the rescalings entirely.
 */
LIBXS_API void libxs_predict_set_floor(libxs_predict_t* model, double floor);

/**
 * Accept absent input values, written as NaN, instead of requiring every
 * coordinate to be present.
 *
 * enable != 0: libxs_predict_load_csv admits a row whose *input* field is empty
 *   or unparseable, storing it as absent, rather than skipping the row (an
 *   absent output still skips it, because there is nothing to learn from).
 * enable == 0 (default): such rows are skipped, exactly as before.
 *
 * The flag governs that admission only. Whether absences are *handled* is
 * decided by the values themselves: push has always copied what it was given,
 * so a NaN pushed directly is honoured whether or not this was called, and the
 * build detects it. Calling this on a model fed by push is therefore harmless
 * and redundant, and its absence is not a way to opt out of the handling below.
 *
 * Where a coordinate is absent on either side, the distance omits it and scales
 * the remainder by the number of coordinates it did read, so entries carrying
 * different absences stay comparable. Two entries sharing no present coordinate
 * are reported maximally distant rather than identical, which is what a plain
 * sum over the present terms would have made them.
 *
 * A consequence worth knowing before relying on it: an entry with an absent
 * coordinate is weaker evidence than a complete one, because that scaling makes
 * agreement on one of two coordinates count for less than agreement on two of
 * two. A complete query therefore need not be answered by an entry that agrees
 * exactly on every coordinate it does have - a fully comparable neighbour
 * agreeing slightly less well can outrank it. Absences cost accuracy where they
 * are common; they do not merely cost nothing.
 *
 * Interpolation abstains for a query with an absent coordinate, and an entry
 * with one does not enter a polynomial fit: a least-squares normal equation has
 * no way to omit a term, so admitting one would silently poison the fit for
 * every other entry in the cluster rather than only for that one.
 *
 * libxs_predict_build FAILS on absences under a tree-based decomposition
 * (LIBXS_PREDICT_RF, LIBXS_PREDICT_HKNN) rather than guessing. A tree reads one
 * coordinate per node and the serialized node has no field in which to record
 * which way an absent one should go; imputing a median instead would work at
 * build and not survive a round trip, since the medians are not recoverable
 * from what is written. Use LIBXS_PREDICT_RAW, or supply complete inputs.
 *
 * Must be called before push. The flag governs ingestion only: a loaded model
 * derives what it needs from the values themselves, so the serialization format
 * is unchanged and a model built this way loads into any released version.
 */
LIBXS_API void libxs_predict_set_missing(libxs_predict_t* model, int enable);

/**
 * Size of the forest LIBXS_PREDICT_RF grows, and how deep its trees may go.
 *
 * ntrees > 0: that many trees per output (default 100).
 * max_depth > 0: that depth for every output.
 * max_depth == 0 (default): twice the base-2 logarithm of the entry count, as
 *   before.
 * max_depth < 0: scored per output at build time. Candidate depths are tried on
 *   a held-back tail of the pushed entries and the one that misclassifies least
 *   is kept, per output.
 *
 * Scoring is opt-in rather than the default because it was measured not to pay.
 * On the shipped kernel-tuning corpus it moved exact match over the sixteen
 * outputs from 62.5% to 63.0% and made the absolute error of the three widest
 * outputs worse; on the 60k-entry crystal corpus it reproduced the derived depth
 * exactly while costing 2.6x the build time. Request it where you suspect the
 * derived depth is wrong for your data - it is a function of corpus size alone,
 * so it asks for depth 20 on 1339 entries whether they carry three features or
 * three hundred - and expect to verify rather than assume it helped.
 *
 * Must be called before build. Neither value is serialized - the stored nodes
 * already encode the depth they were grown to - so this changes what a build
 * produces and not what a file can carry.
 */
LIBXS_API void libxs_predict_set_forest(libxs_predict_t* model,
  int ntrees, int max_depth);

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
 * Push one training entry with a weight on how much say it has.
 *
 * weight > 0 scales this entry's contribution to the kNN vote, on top of the
 * distance weighting it already receives. Two entries at the same distance with
 * weights 3 and 1 count three to one. weight == 1 is exactly libxs_predict_push.
 * A non-positive or NaN weight is rejected.
 *
 * Use it where the corpus is unbalanced and the metric is not: up-weighting a
 * rare class raises its recall and *lowers* overall accuracy, because the two
 * are different questions and the corpus was already optimal for the second.
 * Decide which one is being reported before reaching for this.
 *
 * Scope: the vote only. The polynomial fit, the cluster partition and the
 * probability API (libxs_predict_prob and friends) ignore the weights. The last
 * is deliberate rather than pending - reweighted evidence yields a different
 * quantity from P(y|x), and calling it that would misreport it.
 *
 * Not available in series mode with a weight other than 1: build turns pushed
 * timesteps into overlapping windows, so one timestep contributes to many
 * entries and no entry corresponds to one push. Such a call returns failure
 * rather than picking an interpretation.
 *
 * Weights are serialized (file version 3). A model saved with weights loads with
 * them; a file written before version 3 loads with every weight at 1, which is
 * what it meant.
 */
LIBXS_API int libxs_predict_push_weighted(libxs_lock_t* lock,
  libxs_predict_t* model, const double inputs[], const double outputs[],
  double weight);

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
 * quality:   enables model compression (0..1); 0.0 (default) keeps every entry.
 *            Any positive value drops the entries that a leave-one-out query
 *            recovers *exactly* from a *unanimous* neighbourhood. It does not
 *            select among them: for a classify output unanimity pins the vote
 *            fraction at 1.0, so every positive value yields the same drop set.
 *            Only an interpolating output reads the magnitude, where it scales a
 *            residual tolerance.
 *
 *            Requiring unanimity rather than a confident majority is deliberate.
 *            Letting the threshold decide was measured and is far worse: on a
 *            60k-entry corpus with near-duplicate inputs it drops 57 to 72% of
 *            the entries and takes held-out accuracy from 0.67 to 0.25, because
 *            many entries there are recovered exactly by a neighbourhood that
 *            disagrees.
 *
 *            Compression is not free even so. Dropping about 5% of that corpus
 *            costs 1.5 to 1.7 points of held-out accuracy: the entries are
 *            redundant for a query that coincides with one of them, which is
 *            what leave-one-out asks, and not for a novel query. Measure before
 *            trading accuracy for size.
 *
 *            This no longer affects confidence at eval; see
 *            libxs_predict_set_floor.
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 * May be called again after pushing additional entries (rebuilds).
 */
LIBXS_API int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order, double quality);

/**
 * Per-thread form of libxs_predict_build. All threads must call
 * this collectively with the same model/nclusters/order/quality.
 * tid==0 performs the build; other threads spin-wait.
 * The lock is optional (NULL is accepted).
 */
LIBXS_API int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  double quality, int tid, int ntasks);

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

/**
 * Create a scoring context for one stream, or NULL if the model cannot be
 * scored (not built, or no output carries a usable discrete support). Treat
 * NULL as "cannot score" and abstain: it is a distinct situation from passing
 * NULL as the context to a scoring call, which selects frozen scoring on a
 * model that is fine.
 *
 * The context is bound to this model and this build. Rebuilding the model
 * invalidates it - the size depends on the largest support, which grows as
 * entries are pushed - and a scoring call given a stale context returns no
 * result rather than using a buffer that is too small. Create a new one after
 * any build.
 *
 * The natural stream is one run over the data: the escape weights need on the
 * order of hundreds of observations to settle, so a context per corpus entry
 * would never leave its uniform prior and would score worse than frozen mode.
 * A context per document is worth it only when documents are themselves long.
 *
 * The cost of a per-run context is that results depend on the order entries
 * are scored in. That is reproducible for a fixed corpus and fixed iteration,
 * but not comparable across shuffles, and should not be reported as if it
 * were. For a figure that is both converged and order-independent, score once
 * with a context over a warm-up split, libxs_predict_prob_commit the converged
 * weights to the model, then score the reported split with context == NULL -
 * frozen at converged weights, with no transient. The commit step is required:
 * adaptation writes only into the context, so without it the model still holds
 * its uniform prior and frozen scoring is frozen at that prior, not at what the
 * stream learned.
 */
LIBXS_API void* libxs_predict_prob_create(const libxs_predict_t* model);

/** Destroy a scoring context (NULL is accepted). */
LIBXS_API void libxs_predict_prob_destroy(void* context);

/**
 * Copy a context's converged escape weights into the model, so that frozen
 * scoring (context == NULL) and libxs_predict_save carry what the stream
 * learned instead of the uniform prior.
 *
 * This is a deliberate, caller-timed act rather than a side effect of scoring:
 * an adaptive call that wrote through to the model would make concurrent
 * streams interfere and would silently give up the read-only-while-scoring
 * guarantee the context exists to provide.
 *
 * The model is modified, so no scoring call on it may be in flight. Returns
 * EXIT_SUCCESS, or EXIT_FAILURE if the context does not belong to this model
 * and build (the same rejection scoring applies to a stale context) or the
 * model has no weights to write.
 *
 * Committing mid-stream is allowed and simply publishes the weights as they
 * stand; the context keeps adapting from where it was.
 */
LIBXS_API int libxs_predict_prob_commit(libxs_predict_t* model,
  const void* context);

/**
 * Probability of a supplied candidate output given the inputs.
 *
 * Unlike libxs_predict_eval, which reports what the model would pick, this
 * scores a value the caller supplies - including one the model would never
 * have picked. The same local evidence the kNN vote uses is read at the
 * candidate instead of at its argmax.
 *
 * candidate: N values to score (in user space, before any transform).
 * prob: N values written (may be NULL if only info is needed).
 * info: optional per-output detail (may be NULL). Valid until the next call
 *       on the same model with the same lock, or libxs_predict_destroy.
 * vocabulary: for discrete outputs, the total number of distinct values the
 *       caller considers possible. When > the attested support size, the
 *       escape mass is divided over the unattested remainder so the returned
 *       masses sum to one over the whole vocabulary. Pass 0 to normalize over
 *       the attested support plus a single aggregate novel atom (reported via
 *       info->novel).
 * nblend: as libxs_predict_eval. Blended outputs report LIBXS_PREDICT_PNONE
 *       rather than scoring against evidence the prediction did not use.
 * context: per-stream state from libxs_predict_prob_create, or NULL.
 *
 * The escape weight that mixes local evidence with the fallback prior is
 * learned per output from realized log loss, because no single value serves:
 * the best rate was measured to range 0.10..0.80 across models and to differ
 * by 4x between outputs of one model. That learning is stream state, not model
 * state, so it lives in the caller's context:
 *
 * context != NULL: adaptive. The weights adapt over the stream and the model
 *   is not modified, so one context per scoring thread scores independently
 *   and reproducibly. Contexts must not be shared between concurrent calls.
 *   The context also backs everything info points at, so info requires a
 *   context and its contents stay valid until the next call using the same
 *   context.
 *
 *   For scoring a stream, prefer libxs_predict_prob_observe: it reports the
 *   distribution and observes the outcome in one call, so the ordering that
 *   keeps the two honest cannot be got wrong. This entry point is for point
 *   queries - P(y|x) with no distribution - which it answers without
 *   enumerating the support.
 * context == NULL: frozen. The model's stored weights are used and not
 *   updated - for a model loaded with converged weights this skips the
 *   adaptation transient entirely. The model is strictly read-only, so
 *   concurrent calls are safe and the result does not depend on call order.
 *   info must be NULL in this mode; prob[] receives the values.
 *
 *   Frozen scoring of a discrete output costs O(k log n) in the neighbor count
 *   rather than O(n) in the support size: no weight moves, so the mass follows
 *   in closed form from the escape prior instead of being read out of an
 *   enumerated distribution. The result is identical to the distribution's,
 *   not an approximation of it, so this is the entry point to prefer when the
 *   support is large and only P(y|x) is wanted.
 *
 * For a discrete output the result is a mass and the masses over the support
 * sum to exactly 1.0. For a continuous output it is a density, which
 * integrates to one over the reals but must not be summed with masses -
 * check info->kind before combining outputs.
 */
LIBXS_API void libxs_predict_prob(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context, const double inputs[],
  const double candidate[], double prob[],
  libxs_predict_prob_info_t* info, int vocabulary, int nblend);

/**
 * Distribution over one discrete output, plus optional observation of an
 * outcome. This is the entry point for scoring a stream.
 *
 * The returned distribution is the one in effect *before* the observation, and
 * the escape weights are advanced only afterwards. That ordering is what makes
 * a stream figure honest, and it is enforced here rather than asked of the
 * caller: reporting a distribution shaped by weights that had already seen the
 * target yields a code length better than the truth, with no symptom to notice.
 *
 * output: which output to score.
 * candidate: the observed value to score and learn from, or NULL to report the
 *       distribution without observing anything (no weight moves, so the model
 *       or context is left exactly as it was).
 * values/probs: receive up to capacity entries, sorted by value. The return
 *       value is the support size, which may exceed capacity, in which case
 *       nothing is written. Both may be NULL to query the size alone.
 * novel: receives the mass outside the support (may be NULL).
 * info: optional (may be NULL, requires a context). Fields for the scored
 *       output are filled; other outputs report LIBXS_PREDICT_PNONE.
 *       total_logprob is the scored output's logprob.
 * Returns the support size, or 0 if the output cannot be scored.
 *
 * Because normalization enumerates the whole support anyway, reporting the
 * distribution costs no more than scoring a single candidate. vocabulary and
 * context are as for libxs_predict_prob.
 */
LIBXS_API int libxs_predict_prob_observe(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context, const double inputs[],
  int output, const double* candidate,
  double values[], double probs[], int capacity, double* novel,
  libxs_predict_prob_info_t* info, int vocabulary, int nblend);

/**
 * The attested support of a discrete output: the distinct values the model has
 * seen, sorted ascending. Returns the support size, or 0 if the output has none
 * (not built, or continuous).
 *
 * values: receives up to capacity entries (may be NULL to query the size alone).
 *       Nothing is written when the support exceeds capacity, so a caller can
 *       size a buffer from one call and fill it with a second.
 *
 * The support is fixed once the model is built, so this reads it directly rather
 * than through a scoring call. Obtaining it from libxs_predict_prob_observe
 * requires supplying inputs that are irrelevant to the answer and discarding a
 * distribution that was computed only to be thrown away.
 */
LIBXS_API int libxs_predict_prob_support(const libxs_predict_t* model,
  int output, double values[], int capacity);

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
 * Files written by any released version are accepted; fields introduced later
 * take their default (a v1.0.0 flat model has no global entry order, hence no
 * inverse and no recency weighting, exactly as under v1.0.0).
 * Returns a ready-to-eval model, or NULL on failure (corrupt data, version newer
 * than this library). The returned model does not reference the buffer after
 * this call returns.
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
 *           An empty or all-blank string is treated as NULL.
 * outputs:  comma-separated column names or numeric indices for output
 *           parameters, or NULL for sequential columns ninputs..ninputs+noutputs-1.
 *           An empty or all-blank string is treated as NULL.
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

/**
 * Options for libxs_predict_load_csv_opts. A zero-initialized struct requests
 * the same load as libxs_predict_load_csv with every argument NULL or zero.
 */
LIBXS_EXTERN_C typedef struct libxs_predict_csv_t {
  /** As libxs_predict_load_csv: delimiters, column specs, header capture. */
  const char* delims;
  const char* inputs;
  const char* outputs;
  char* header;
  int header_size;
  char* delim_out;
  /** Stop after this many entries were pushed (0: read to end of file). */
  int nrows;
  /**
   * Push every stride-th admissible row (0 or 1: every row).
   *
   * A prefix of a file is not a sample of it - a corpus sorted or grouped by
   * anything puts a different distribution in its first rows than in the file
   * as a whole - so a subset taken for a scaling study or a held-back split is
   * strided rather than truncated. Together with offset this splits one file
   * into disjoint interleaved parts, each spanning the whole of it: stride 5
   * with offset 0 and stride 5 with offset 1 share no row.
   *
   * Rows that cannot be parsed, and comment rows, are not admissible and do
   * not advance the count, so neither a header nor a damaged row shifts which
   * rows a given (stride, offset) selects.
   */
  int stride;
  /** Admissible rows to skip before the first one taken (0: none). */
  int offset;
} libxs_predict_csv_t;

/**
 * Load delimited text as libxs_predict_load_csv, taking its arguments as a
 * struct so a subset of a large file can be requested (see nrows/stride).
 * opts may be NULL, which requests every row and sequential columns.
 * Returns the number of entries pushed, or -1 on I/O error.
 */
LIBXS_API int libxs_predict_load_csv_opts(libxs_predict_t* model,
  const char filename[], const libxs_predict_csv_t* opts);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PREDICT_H*/
