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
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_gemm.h>
#include <libxs/libxs_hash.h>
#include "libxs_main.h"

#if !defined(LIBXS_PREDICT_MAXITER)
#  define LIBXS_PREDICT_MAXITER 100
#endif
#if !defined(LIBXS_PREDICT_MAGIC)
#  define LIBXS_PREDICT_MAGIC 0x58535052U /* "XSPR" */
#endif
#if !defined(LIBXS_PREDICT_MAGIC_HKNN)
#  define LIBXS_PREDICT_MAGIC_HKNN 0x58534B4EU /* "XSKN" */
#endif
/**
 * Serialization format version. Bumped whenever the on-disk layout changes at
 * a release boundary; libxs_predict_load accepts every released version down to
 * 1 (v1.0.0), so the reader gates each field on the version it appeared in.
 */
#if !defined(LIBXS_PREDICT_VERSION)
#  define LIBXS_PREDICT_VERSION 2
#endif
#if !defined(LIBXS_PREDICT_KNN)
#  define LIBXS_PREDICT_KNN 32
#endif
/**
 * Fewest neighbours a vote needs to carry information: below this a
 * neighbourhood is unanimous by construction, so its confidence is 1.0
 * whatever it holds. The neighbour-count trial refuses such a count outright
 * (see libxs_predict_select.h); compression uses this as the point at which a
 * cluster stops giving entries up.
 */
#if !defined(LIBXS_PREDICT_KMIN)
#  define LIBXS_PREDICT_KMIN 3
#endif
/* Quantile knots per input axis when the rank coordinate is in use. */
#if !defined(LIBXS_PREDICT_KNOTS)
#  define LIBXS_PREDICT_KNOTS 256
#endif
/* Outputs a window bank can average without allocating (see libxs_predict_eval). */
#if !defined(LIBXS_PREDICT_HMAX)
#  define LIBXS_PREDICT_HMAX 128
#endif
#if !defined(LIBXS_PREDICT_LSQ_NOISE)
#  define LIBXS_PREDICT_LSQ_NOISE 0.05
#endif
#if !defined(LIBXS_PREDICT_LSQ_MINRATIO)
#  define LIBXS_PREDICT_LSQ_MINRATIO 2
#endif
#if !defined(LIBXS_PREDICT_BLEND_CONF)
#  define LIBXS_PREDICT_BLEND_CONF 0.7
#endif
#if !defined(LIBXS_PREDICT_BLEND_N)
#  define LIBXS_PREDICT_BLEND_N 3
#endif

#define LIBXS_PREDICT_MALLOC(SIZE, POOL) internal_libxs_scratch_malloc(SIZE, &(POOL))
#define LIBXS_PREDICT_FREE(PTR, POOL) internal_libxs_scratch_free(PTR, POOL)

/**
 * Which evidence served an output, as reported by the internal eval. BLEND and
 * RF carry no single source cluster: a scoring rule must either handle them
 * explicitly or abstain rather than fall back to the primary cluster, which
 * would silently score against evidence the prediction did not use.
 */
#define LIBXS_PREDICT_SRC_NONE     (-1)
#define LIBXS_PREDICT_SRC_CLASSIFY 0
#define LIBXS_PREDICT_SRC_INTERP   1
#define LIBXS_PREDICT_SRC_BLEND    2
#define LIBXS_PREDICT_SRC_RF       3

/**
 * Escape-rate experts for the probability escape. The mixing weight that beats
 * the local evidence alone was measured to range 0.10..0.80 across datasets -
 * an order of magnitude, and far above any novelty frequency - so no single
 * default is defensible and no build-time estimate of it succeeded (LOO
 * under-measures novelty 6-40x, coverage bins are flat). A causal fixed-share
 * bank over candidate rates lands within 0.03 bits of the per-dataset oracle
 * rate without being told it, which is why the rate is learned per query from
 * realized log loss rather than configured.
 */
#define LIBXS_PREDICT_NESCAPE 13
#define LIBXS_PREDICT_ESCAPE_ETA 0.5
#define LIBXS_PREDICT_ESCAPE_SHARE 0.01
#define LIBXS_PREDICT_ESCAPE_RELMIN 1e-12
/**
 * Half-width of the window internal_libxs_predict_local_error reads around a
 * position to estimate how much the output varies there. The buffer it fills is
 * 2*R+1 wide, so the two must move together - they were the same constant
 * written twice, which is a way for one to be changed and the other not.
 *
 * The value is not derived from anything: it is neither tied to the neighbor
 * count nor to the cluster size beyond the tiny-cluster clamp. Points adjacent
 * in this order are adjacent in space (the entries are Hilbert-sorted), so a
 * wider window reaches further spatially - but a Hilbert curve leaves a
 * subsquare from time to time, and across such a break adjacency in the order
 * stops implying adjacency in space. A width chosen against that structure
 * would be a change in behaviour and wants its own measurement.
 */
#define LIBXS_PREDICT_LOCAL_RADIUS 4
#define LIBXS_PREDICT_CTX_MAGIC 0x58535043U /* "XSPC" */

/**
 * Bytes taken by N ints inside a carved-up block, padded so that whatever
 * follows is still aligned. An odd noutputs makes an unpadded group of three
 * leave the cursor at four modulo eight, and the pointer or double array behind
 * it is then misaligned: undefined, and silently fine on x86 until it is not.
 * Every int group is sized with this, so appending to the layout stays safe.
 */
#define INTERNAL_LIBXS_PREDICT_NBINT(N) LIBXS_UP2((size_t)(N) * sizeof(int), sizeof(double))


typedef struct internal_libxs_predict_entry_t {
  double* inputs;
  double* outputs;
  /** Relative say this entry has in a vote; 1 unless push_weighted set it. */
  double weight;
} internal_libxs_predict_entry_t;

typedef struct internal_libxs_predict_cluster_t {
  double* centroid;
  double* coeffs;
  double* errors;
  double* out_rms;
  double* kd_pts;
  double* raw_outputs;
  double* out_mean;
  double* out_var;
  int* order;
  int* interpolated;
  int* mode;
  int* ndistinct;
  /** Neighbour count per output. Replaces one k_eff chosen by a majority vote
   *  over the outputs' modes, which gave an interpolating output the formula
   *  picked for its classifying neighbours. Derived at build and again at load,
   *  so it costs nothing in the file. */
  int* k_out;
  /** Entry weights in kd order, parallel to kd_pts; NULL when all are 1. */
  double* eweight;
  int* sorted_idx;
  double* sorted_dist;
  double* tangent;
  double* kd_tan;
  double dmax;
  double fprint_sig;
  int nentries;
  int maxorder;
  int k_eff;
  int tdim;
} internal_libxs_predict_cluster_t;

/**
 * One window view: the distance reads only the most recent w lags of each of
 * s series, out of full lags stored per series. NULL or w == full means the
 * whole window, which is the only view a non-series model has.
 */
typedef struct internal_libxs_predict_view_t {
  int w, s, full;
} internal_libxs_predict_view_t;

typedef struct internal_libxs_predict_order_ctx_t {
  libxs_predict_t* model;
  int nclusters;
  int tid, ntasks;
} internal_libxs_predict_order_ctx_t;

typedef struct internal_libxs_predict_rf_node_t {
  int feature;
  double threshold;
  /** Leaf read-out. Carries the subset mean for a real-valued output and the
   *  majority label for a folded one, so a leaf serves either read-out. */
  double value;
  int left, right;
  int label;
} internal_libxs_predict_rf_node_t;

typedef struct internal_libxs_predict_rf_tree_t {
  internal_libxs_predict_rf_node_t* nodes;
  /**
   * Boosted correction per node, fitted after the tree was grown, on the rows
   * this tree's bootstrap left out. Kept beside the nodes rather than in them:
   * it is a second read-out over the same partition, read only where boosting
   * ran, and the descent should not carry it through cache. NULL where the
   * output is folded, or where the stopping rule ended the stages first.
   */
  double* incr;
  int nnodes;
} internal_libxs_predict_rf_tree_t;

typedef struct internal_libxs_predict_rf_t {
  internal_libxs_predict_rf_tree_t* trees;
  int* label_offset;
  /**
   * Per-output read-out: non-zero where the output is real-valued and the
   * forest averages leaf means, zero where it folds to a class label and the
   * forest takes a majority vote. Decided once at build from the corpus: an
   * output all of whose values are integral and whose range fits the fold is a
   * class, anything else is a quantity. Without this every output was a class,
   * which capped a continuous output at the resolution of its own rounding.
   */
  int* regress;
  /**
   * Score width per output: the number of classes a folded output votes over,
   * one for a real-valued output. It is the stride of the boosted correction,
   * and making the real-valued case a width of one leaves a single fit and a
   * single stored form rather than two of each.
   */
  int* nclass;
  /** Per-output tree depth, chosen at build. Not serialized: the stored nodes
   *  already encode the depth they were grown to, and nothing reads it again. */
  int* depth;
  int ntrees;
  int noutputs;
} internal_libxs_predict_rf_t;

typedef struct {
  double val;
  int idx;
} internal_libxs_predict_rf_pair_t;

LIBXS_EXTERN_C struct libxs_predict_t {
  internal_libxs_predict_entry_t* entries;
  internal_libxs_predict_cluster_t* clusters;
  int* assignments;
  int* hknn_assignments;
  int** hknn_po_assignments;
  int* hknn_po_nclusters;
  int* hknn_po_groups;
  int hknn_ngroups;
  internal_libxs_predict_cluster_t** hknn_po_clusters;
  double* eval_buf;
  /**
   * Every entry's inputs normalized, nentries*ninputs, built once per build.
   * The partition steps each used to normalize into a buffer of their own, so
   * a build held three copies of it at once and paid for the normalization
   * three times. It is also what lets the partition be split across tasks:
   * a task cannot own a copy of something this size (at millions of entries
   * it is gigabytes) and the assignment step only reads it.
   */
  double* norm_pts;
  /**
   * Scratch for the partition in progress, nclusters*ninputs: k-means keeps its
   * working centroids here, the hierarchical refinement its compensation terms.
   * The builder allocates it and every task tests it, which is what makes the
   * step's precondition a shared one - a task that decided on its own whether
   * to take part would leave the others waiting at a rendezvous it never
   * reaches. k-means also needs it shared on its own account, since the
   * assignment step reads centroids that only the builder moves.
   */
  double* norm_cen;
  /**
   * One block holding every entry's inputs and outputs, ninputs+noutputs
   * apart. The entry pointers address it rather than owning separate
   * allocations: a corpus of millions of rows would otherwise pay two
   * allocations per row, and a single output rounds up to the allocator's
   * minimum chunk, which costs four times what it stores. Scattered pages
   * also leave no way to place them, so first touch cannot be controlled.
   */
  double* arena;
  int arena_capacity;
  double* input_min;
  double* input_rng;
  double* input_knot;
  double* weights;
  int* transforms;
  double* ts_buf;
  double* aux_buf;
  double* decompose_mat;
  internal_libxs_predict_rf_t* rf;
  libxs_lock_t lock;
  int order;
  int ninputs, noutputs;
  int nentries, capacity;
  int nclusters;
  int hknn_nclusters;
  int built;
  int eval_mode;
  int iterations;
  int nseries, window, target, decompose;
  int naux, nderiv;
  int nts, ts_capacity;
  /** Window views: nbank requested, bank_w[] lags each view reads. */
  int nbank;
  int* bank_w;
  int diff_mode, diff_order;
  int refine;
  int tangent;
  /** Absent inputs accepted (set_missing), and whether any is actually present.
   *  The second is derived - from the entries at build, from the stored points
   *  at load - so it holds for a loaded model that never saw the setter. */
  int missing_mode, has_missing;
  /** Non-zero once any pushed entry carries a weight other than 1. */
  int has_eweight;
  /** Requested forest size and depth (0: decide at build). */
  int rf_ntrees, rf_depth;
  /**
   * Requested neighbour count (see set_neighbors): positive pins it, zero
   * derives it, negative selects it by trial. The resolved per-output counts
   * live in k_sel and are saved with the model, because the derived formula
   * would otherwise reassert itself at load.
   */
  int kreq;
  int* k_sel;
  /** Requested central tendency (0/negative: decide per output at build). */
  int central;
  /** Per-output resolved choice, noutputs entries, NULL until built. */
  int* central_out;
  double smooth;
  /** Compression threshold at build; see libxs_predict_build. */
  double quality;
  /**
   * Confidence floor at eval: every rescaling pulls confidence toward this
   * rather than toward the compression threshold. The two were one number,
   * which meant asking for compression silently moved a runtime knob that
   * decides how many clusters a query blends - worth more accuracy on a
   * categorical output than the compression itself was.
   */
  double floor;
  double consistency;
  double quantile;
  /** Per-output sorted distinct values and counts: the exact support the
   *  probability normalizes over. Derived from raw_outputs at build/load, so
   *  the serialization format is unchanged. */
  double** sup_vals;
  double** sup_freq;
  int* sup_n;
  int* sup_tot;
  /** Frozen escape weights, noutputs * LIBXS_PREDICT_NESCAPE. Written by
   *  load, read when scoring without a context. */
  double* escape_w;
  /** Incremented by every build, so a scoring context can tell that the
   *  model it was sized for is no longer the model in front of it. */
  int nbuild;
  /**
   * Rendezvous for a collective build: arrivals at the stage in progress, and
   * the number of stages completed. The epoch is what a waiting task watches,
   * rather than a flag it must reset, so the same pair serves every stage and
   * every build without a thread carrying state between them.
   *
   * This replaces a single field that meant two things at once - whether the
   * builder had finished, and whether the build was collective at all. The
   * second is a property of the call and is now a parameter, which cannot go
   * stale the way the field did after the call it described had returned.
   */
  volatile int sync_count, sync_epoch;
  /**
   * Set by any task whose slice of the assignment step moved an entry, so the
   * tasks reach the same verdict on convergence and leave the loop together.
   * Read after a rendezvous and cleared by the builder before the next one.
   */
  volatile int sync_moved;
  /**
   * The builder's verdict on a stage it ran alone. A task's own result cannot
   * serve: it never ran the allocation that may have failed, so it would judge
   * the next stage differently and enter a rendezvous the others have left.
   */
  volatile int sync_result;
  /** Per-candidate scores of a collective trial, indexed by candidate. */
  double sync_score[8];
};

/**
 * Layout of a caller-owned scoring context. Everything a call mutates lives
 * here rather than in the model, so concurrent streams cannot interfere and the
 * model stays read-only while scoring. The escape weights are per output
 * because the rate that suits one output does not suit another: within a single
 * PVC model the best rate was measured at 0.55, 0.80 and 0.20 for three of its
 * outputs, and one shared bank can only converge to a compromise none of them
 * wants.
 */
typedef struct internal_libxs_predict_ctx_t {
  uint32_t magic;
  /**
   * Which model and which build this context was sized for. The size depends
   * on the largest support, which grows when a model is rebuilt after more
   * entries are pushed, so a context outliving a rebuild is undersized rather
   * than merely stale. Stamping both lets scoring refuse it instead of
   * re-initializing into a buffer that is too small.
   */
  const void* model;
  int nbuild;
  int noutputs;
  int maxsup;
  /* followed by: escape weights [n * NESCAPE], dist p, norm scratch,
     local evidence, the per-output reporting arrays, the int arrays, then the
     per-call dispatch buffers */
} internal_libxs_predict_ctx_t;


/** Candidate rates of the escape-rate bank; see LIBXS_PREDICT_NESCAPE. */
static const double internal_libxs_predict_escape_rate[
  LIBXS_PREDICT_NESCAPE] =
{
  0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
  0.05, 0.10, 0.20, 0.35, 0.55, 0.80
};


/**
 * Wait until every task has arrived.
 *
 * Reading the epoch before arriving is what makes this safe to reuse: the last
 * arrival is the only one that advances it, and no task can advance past a
 * stage that another has not yet entered, so a task that reads the epoch late
 * still waits for the advance that follows its own arrival.
 */
LIBXS_API_INLINE void internal_libxs_predict_sync(
  libxs_predict_t* model, int ntasks)
{
  if (1 < ntasks) {
    const int epoch = (int)LIBXS_ATOMIC_LOAD(
      &model->sync_epoch, LIBXS_ATOMIC_SEQ_CST);
    if (ntasks == (int)LIBXS_ATOMIC_ADD_FETCH(
      &model->sync_count, 1, LIBXS_ATOMIC_SEQ_CST))
    {
      LIBXS_ATOMIC_STORE(&model->sync_count, 0, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ATOMIC_ADD_FETCH(&model->sync_epoch, 1, LIBXS_ATOMIC_SEQ_CST);
    }
    else {
      while (epoch == (int)LIBXS_ATOMIC_LOAD(
        &model->sync_epoch, LIBXS_ATOMIC_SEQ_CST))
      {
        LIBXS_SYNC_PAUSE;
      }
    }
  }
}


LIBXS_API_INLINE int internal_libxs_predict_support_all(libxs_predict_t* model);
LIBXS_API_INLINE void internal_libxs_predict_missing_all(libxs_predict_t* model);
LIBXS_API_INLINE void internal_libxs_predict_central_all(libxs_predict_t* model);
LIBXS_API_INLINE void internal_libxs_predict_keff_all(libxs_predict_t* model);
LIBXS_API_INLINE void internal_libxs_predict_kapply(libxs_predict_t* model);
LIBXS_API_INLINE int internal_libxs_predict_build_impl(libxs_predict_t* model,
  int nclusters, int order, double quality, int tid, int ntasks);
LIBXS_API_INLINE void internal_libxs_predict_bank_all(libxs_predict_t* model);


LIBXS_API_INLINE void internal_libxs_predict_free_clusters(libxs_predict_t* model)
{
  if (NULL != model->clusters) {
    int c;
    for (c = 0; c < model->nclusters; ++c) {
      internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      free(cl->centroid);
      free(cl->coeffs);
      free(cl->errors);
      free(cl->out_rms);
      free(cl->kd_pts);
      free(cl->raw_outputs);
      free(cl->out_mean);
      free(cl->out_var);
      free(cl->order);
      free(cl->interpolated);
      free(cl->mode);
      free(cl->ndistinct);
      free(cl->k_out);
      free(cl->eweight);
      free(cl->sorted_idx);
      free(cl->sorted_dist);
      free(cl->tangent);
      free(cl->kd_tan);
    }
    free(model->clusters);
    model->clusters = NULL;
  }
  free(model->assignments);
  model->assignments = NULL;
  free(model->eval_buf);
  model->eval_buf = NULL;
  free(model->norm_pts);
  model->norm_pts = NULL;
  /* the support cache is derived from raw_outputs and must not outlive it */
  if (NULL != model->sup_vals) {
    int j;
    for (j = 0; j < model->noutputs; ++j) {
      free(model->sup_vals[j]);
      free(model->sup_freq[j]);
    }
    free(model->sup_vals);
    free(model->sup_freq);
    free(model->sup_n);
    free(model->sup_tot);
    model->sup_vals = NULL;
    model->sup_freq = NULL;
    model->sup_n = NULL;
    model->sup_tot = NULL;
  }
  free(model->central_out);
  model->central_out = NULL;
  model->nclusters = 0;
  model->built = 0;
}


/**
 * Squared distance that omits the coordinates either side is missing, scaled by
 * the count it did read so entries with different absences stay comparable.
 *
 * The missing flag is the model's, constant for the life of a build, so a model
 * without absences takes the same call it always did: this is the hot path for
 * every query and a per-coordinate NaN test on it would be paid by everyone to
 * serve the few models that need it.
 */
LIBXS_API_INLINE double internal_libxs_predict_dist2(const double* a,
  const double* b, int m, int missing)
{
  double result;
  if (0 == missing) {
    result = libxs_dist2(a, b, m);
  }
  else {
    int i, used = 0;
    result = 0;
    for (i = 0; i < m; ++i) {
      if (LIBXS_NOTNAN(a[i]) && LIBXS_NOTNAN(b[i])) {
        const double d = a[i] - b[i];
        result += d * d;
        ++used;
      }
    }
    /* no shared coordinate is no evidence, which must not read as coincident */
    result = (0 < used) ? (result * m / used) : (double)m;
  }
  return result;
}


/**
 * A quiet NaN, standing for "no value here". C89 has no NAN macro, and a
 * literal 0.0/0.0 is a constant expression the compiler is entitled to fold or
 * diagnose, so the zero is read through volatile.
 */
LIBXS_API_INLINE double internal_libxs_predict_absent(void)
{
  static const volatile double zero = 0;
  return zero / zero;
}


/** Non-zero if any coordinate of the vector is absent. */
LIBXS_API_INLINE int internal_libxs_predict_incomplete(const double* v, int m)
{
  int i, result = 0;
  for (i = 0; i < m && 0 == result; ++i) {
    if (LIBXS_ISNAN(v[i])) result = 1;
  }
  return result;
}


/**
 * Non-zero if the mode can carry an absent input.
 *
 * The distance answers a gap by skipping the coordinate and rescaling by the
 * ones it did use, so anything that reads coordinates independently can express
 * "not known here". A rotation cannot: every output component is a combination
 * of every input, so one absent coordinate contaminates the whole vector rather
 * than one place in it, and no rescaling undoes that. A tree is refused for the
 * separate reason given at its build site.
 */
LIBXS_API_INLINE int internal_libxs_predict_gaps_ok(int mode)
{
  return (LIBXS_PREDICT_PCA != mode && LIBXS_PREDICT_SPREAD != mode
    && LIBXS_PREDICT_RF != mode && LIBXS_PREDICT_HKNN != mode) ? 1 : 0;
}


LIBXS_API_INLINE int internal_libxs_predict_cmpval(const void* a, const void* b)
{
  const double x = *(const double*)a, y = *(const double*)b;
  int result;
  if (x < y) result = -1;
  else if (x > y) result = 1;
  else result = 0;
  return result;
}


/**
 * Position of a value in its own axis distribution: the empirical CDF, sampled
 * at LIBXS_PREDICT_KNOTS uniform probabilities and interpolated between them.
 * A degenerate axis carries knot[last] == knot[0] and every caller must fall
 * back to the extent, because the position is not defined without a spread.
 */
LIBXS_API_INLINE double internal_libxs_predict_rank(const double* knot, double v)
{
  const int last = LIBXS_PREDICT_KNOTS - 1;
  double result;
  if (v <= knot[0]) result = 0;
  else if (v >= knot[last]) result = 1;
  else {
    int lo = 0, hi = last;
    while (lo + 1 < hi) {
      const int mid = (lo + hi) / 2;
      if (v < knot[mid]) hi = mid;
      else lo = mid;
    }
    result = (knot[hi] > knot[lo])
      ? ((lo + (v - knot[lo]) / (knot[hi] - knot[lo])) / last)
      : ((double)lo / last);
  }
  return result;
}


/** Inverse of internal_libxs_predict_rank. */
LIBXS_API_INLINE double internal_libxs_predict_unrank(const double* knot, double u)
{
  const int last = LIBXS_PREDICT_KNOTS - 1;
  double result;
  if (u <= 0) result = knot[0];
  else if (u >= 1) result = knot[last];
  else {
    const double t = u * last;
    const int lo = (int)t;
    const int hi = (lo < last) ? (lo + 1) : last;
    result = knot[lo] + (t - lo) * (knot[hi] - knot[lo]);
  }
  return result;
}


/** Knots of one axis, or NULL where the extent is to be used instead. */
LIBXS_API_INLINE const double* internal_libxs_predict_knot(
  const libxs_predict_t* model, int j)
{
  const double* result = NULL;
  if (NULL != model->input_knot) {
    const double* knot = model->input_knot + (size_t)j * LIBXS_PREDICT_KNOTS;
    if (knot[LIBXS_PREDICT_KNOTS - 1] > knot[0]) result = knot;
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_normalize(
  const libxs_predict_t* model, const double* inputs, double* norm)
{
  const int m = model->ninputs;
  int i;
  for (i = 0; i < m; ++i) {
    const double* knot = internal_libxs_predict_knot(model, i);
    const double v = inputs[i];
    if (NULL != knot && LIBXS_NOTNAN(v)) {
      norm[i] = internal_libxs_predict_rank(knot, v);
    }
    else {
      norm[i] = (NULL != model->input_rng && model->input_rng[i] > 0)
        ? (v - model->input_min[i]) / model->input_rng[i] : v;
    }
    if (NULL != model->weights) norm[i] *= model->weights[i];
  }
}


/**
 * Inverse of internal_libxs_predict_normalize. Kept adjacent to it so the two
 * cannot drift apart. A zero weight is not invertible (feature selection drops
 * the coordinate), which callers must check before relying on the result.
 */
LIBXS_API_INLINE void internal_libxs_predict_denormalize(
  const libxs_predict_t* model, const double* norm, double* inputs)
{
  const int m = model->ninputs;
  int i;
  for (i = 0; i < m; ++i) {
    const double* knot = internal_libxs_predict_knot(model, i);
    double v = norm[i];
    if (NULL != model->weights && 0 != model->weights[i]) v /= model->weights[i];
    if (NULL != knot && LIBXS_NOTNAN(v)) {
      inputs[i] = internal_libxs_predict_unrank(knot, v);
    }
    else {
      inputs[i] = (NULL != model->input_rng && model->input_rng[i] > 0)
        ? (v * model->input_rng[i] + model->input_min[i]) : v;
    }
  }
}


/**
 * Fits the per-axis quantile knots the rank coordinate reads. Absent values are
 * excluded rather than sorted: a NaN compares false against everything, so it
 * would land wherever the sort happens to leave it and displace every knot
 * after it. An axis without a spread is zeroed, which internal_libxs_predict_knot
 * reports as "use the extent".
 */
LIBXS_API_INLINE int internal_libxs_predict_fit_knots(libxs_predict_t* model)
{
  const int m = model->ninputs, p = model->nentries;
  const int last = LIBXS_PREDICT_KNOTS - 1;
  int scratch_pool = 0;
  double* scratch = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(double), scratch_pool);
  int result = EXIT_SUCCESS;
  free(model->input_knot);
  model->input_knot = (double*)malloc(
    (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double));
  if (NULL != scratch && NULL != model->input_knot) {
    int i, j, k;
    for (j = 0; j < m; ++j) {
      double* knot = model->input_knot + (size_t)j * LIBXS_PREDICT_KNOTS;
      int nval = 0;
      for (i = 0; i < p; ++i) {
        const double v = model->entries[i].inputs[j];
        if (LIBXS_NOTNAN(v)) scratch[nval++] = v;
      }
      if (1 < nval) {
        qsort(scratch, (size_t)nval, sizeof(double),
          internal_libxs_predict_cmpval);
      }
      if (1 < nval && scratch[nval - 1] > scratch[0]) {
        for (k = 0; k <= last; ++k) {
          const double t = (double)k * (nval - 1) / last;
          const int lo = (int)t;
          const int hi = (lo < nval - 1) ? (lo + 1) : (nval - 1);
          knot[k] = scratch[lo] + (t - lo) * (scratch[hi] - scratch[lo]);
        }
      }
      else {
        for (k = 0; k <= last; ++k) knot[k] = 0;
      }
    }
  }
  else {
    free(model->input_knot);
    model->input_knot = NULL;
    result = EXIT_FAILURE;
  }
  LIBXS_PREDICT_FREE(scratch, scratch_pool);
  return result;
}


/**
 * Normalized inputs for every entry, allocated on first use per build and
 * released with the model. Returns NULL if it cannot be had, which every
 * caller must treat as "do not partition" rather than falling back to an
 * un-normalized coordinate.
 */
LIBXS_API_INLINE double* internal_libxs_predict_normpts(libxs_predict_t* model)
{
  const int m = model->ninputs;
  const int p = model->nentries;
  if (NULL == model->norm_pts && 0 < p && 0 < m) {
    model->norm_pts = (double*)malloc((size_t)p * (size_t)m * sizeof(double));
    if (NULL != model->norm_pts) {
      int i;
      for (i = 0; i < p; ++i) {
        internal_libxs_predict_normalize(model,
          model->entries[i].inputs, model->norm_pts + (size_t)i * m);
      }
    }
  }
  return model->norm_pts;
}


/**
 * tid/ntasks: the assignment step is split across the tasks and the step that
 * moves the centroids is the builder's, because assignment is O(p*k*m) against
 * the O(p*m) of moving them - the serial remainder is a k-th of an iteration.
 * The scratch it needs is the builder's alone for the same reason; only the
 * centroids are shared, since every task reads all of them.
 */
LIBXS_API_INLINE void internal_libxs_predict_kmeans(libxs_predict_t* model,
  int nclusters, int tid, int ntasks)
{
  const int m = model->ninputs;
  const int p = model->nentries;
  const int missing = model->has_missing;
  int pool_comp = 0, pool_cnt = 0, pool_dist = 0;
  int pool_dcnt = 0;
  const double* pts;
  double* centroids;
  double* comp = NULL;
  int* counts = NULL;
  double* dists = NULL;
  int* dcounts = NULL;
  if (0 == tid) {
    /**
     * Built here, not per task: it is one buffer on the model, and tasks
     * racing to create it would each fill a copy the others then read
     */
    internal_libxs_predict_normpts(model);
    free(model->norm_cen);
    model->norm_cen = (double*)malloc(
      (size_t)nclusters * (size_t)m * sizeof(double));
    comp = (double*)LIBXS_PREDICT_MALLOC((size_t)nclusters * (size_t)m * sizeof(double), pool_comp);
    counts = (int*)LIBXS_PREDICT_MALLOC((size_t)nclusters * sizeof(int), pool_cnt);
    dists = (double*)LIBXS_PREDICT_MALLOC((size_t)p * sizeof(double), pool_dist);
    /* per-dimension counts: a centroid averages only the values actually present */
    dcounts = (0 == missing) ? NULL : (int*)LIBXS_PREDICT_MALLOC(
      (size_t)nclusters * (size_t)m * sizeof(int), pool_dcnt);
    model->sync_moved = 0;
    if (NULL == model->norm_cen || NULL == comp || NULL == counts
      || NULL == dists || (0 != missing && NULL == dcounts))
    { /* every task tests norm_cen below, so releasing it declines the step */
      free(model->norm_cen); model->norm_cen = NULL;
    }
  }
  internal_libxs_predict_sync(model, ntasks);
  pts = model->norm_pts;
  centroids = model->norm_cen;
  if (NULL != pts && NULL != centroids) {
    int c, i, j, iter;
    if (0 == tid) {
    { const size_t seed = (0 == (model->eval_mode & LIBXS_PREDICT_TEMPORAL))
        ? LIBXS_SHUFFLE_INDEX(0, (size_t)p, libxs_coprime2((size_t)p), 0)
        : 0;
      memcpy(centroids, pts + seed * m, (size_t)m * sizeof(double));
    }
    for (i = 0; i < p; ++i) dists[i] = DBL_MAX;
    for (c = 1; c < nclusters; ++c) {
      int farthest = 0;
      double maxd = 0;
      for (i = 0; i < p; ++i) {
        const double d = internal_libxs_predict_dist2(
          pts + (size_t)i * m, centroids + (size_t)(c - 1) * m, m, missing);
        /**
         * A NaN distance compares false against everything, so a point that
         * produces one keeps the initial DBL_MAX and is then the farthest point
         * from every centroid there will ever be. Seeding copies it into all of
         * them, Lloyd finds one distinct centroid, and every entry lands in the
         * same cluster: the partition collapses to one and the model still
         * reports success. Such a point is excluded from seeding instead, which
         * costs one candidate centroid and cannot cost the partition.
         */
        if (LIBXS_NOTNAN(d)) {
          if (d < dists[i]) dists[i] = d;
          if (dists[i] > maxd && dists[i] < DBL_MAX) {
            maxd = dists[i];
            farthest = i;
          }
        }
        else dists[i] = 0;
      }
      memcpy(centroids + (size_t)c * m, pts + (size_t)farthest * m,
        (size_t)m * sizeof(double));
    }
    } /* seeding is the builder's: it walks the centroids in order */
    internal_libxs_predict_sync(model, ntasks);
    /* Lloyd iterations with Kahan-compensated centroid accumulation */
    for (iter = 0; iter < LIBXS_PREDICT_MAXITER; ++iter) {
      int changed = 0;
      /**
       * Cleared here rather than after the verdict is read. Clearing it there
       * races with a task that has not read it yet: that task then reads zero,
       * concludes the partition converged, leaves the loop, and the two run
       * different stages against the same rendezvous counter.
       */
      if (0 == tid) model->sync_moved = 0;
      internal_libxs_predict_sync(model, ntasks);
      for (i = tid; i < p; i += ntasks) {
        double best = internal_libxs_predict_dist2(
          pts + (size_t)i * m, centroids, m, missing);
        int bestc = 0;
        for (c = 1; c < nclusters; ++c) {
          const double d = internal_libxs_predict_dist2(
            pts + (size_t)i * m, centroids + (size_t)c * m, m, missing);
          if (d < best) { best = d; bestc = c; }
        }
        if (model->assignments[i] != bestc) {
          model->assignments[i] = bestc;
          changed = 1;
        }
      }
      if (0 != changed) {
        LIBXS_ATOMIC_STORE(&model->sync_moved, 1, LIBXS_ATOMIC_SEQ_CST);
      }
      internal_libxs_predict_sync(model, ntasks);
      /* one verdict, read by every task after the same rendezvous */
      changed = (int)LIBXS_ATOMIC_LOAD(&model->sync_moved, LIBXS_ATOMIC_SEQ_CST);
      if (0 == changed) iter = LIBXS_PREDICT_MAXITER;
      else {
        if (0 == tid) {
        memset(centroids, 0, (size_t)nclusters * (size_t)m * sizeof(double));
        memset(comp, 0, (size_t)nclusters * (size_t)m * sizeof(double));
        memset(counts, 0, (size_t)nclusters * sizeof(int));
        if (0 != missing) {
          memset(dcounts, 0, (size_t)nclusters * (size_t)m * sizeof(int));
        }
        for (i = 0; i < p; ++i) {
          const int ci = model->assignments[i];
          double* cen = centroids + (size_t)ci * m;
          double* cmp = comp + (size_t)ci * m;
          for (j = 0; j < m; ++j) {
            const double v = pts[(size_t)i * m + j];
            if (0 == missing) {
              libxs_kahan_sum(v, &cen[j], &cmp[j]);
            }
            else if (LIBXS_NOTNAN(v)) {
              libxs_kahan_sum(v, &cen[j], &cmp[j]);
              ++dcounts[(size_t)ci * m + j];
            }
          }
          ++counts[ci];
        }
        for (c = 0; c < nclusters; ++c) {
          if (0 < counts[c]) {
            double* cen = centroids + (size_t)c * m;
            if (0 == missing) {
              for (j = 0; j < m; ++j) cen[j] /= counts[c];
            }
            else for (j = 0; j < m; ++j) {
              const int n = dcounts[(size_t)c * m + j];
              /* absent throughout the cluster: the centroid has no value here,
                 and saying so lets the distance omit it rather than read a 0 */
              cen[j] = (0 < n) ? (cen[j] / n)
                : internal_libxs_predict_absent();
            }
          }
        }
        } /* moving the centroids is the builder's */
        internal_libxs_predict_sync(model, ntasks);
      }
    }
    if (0 == tid) for (c = 0; c < nclusters; ++c) {
      memcpy(model->clusters[c].centroid, centroids + (size_t)c * m, (size_t)m * sizeof(double));
    }
  }
  if (0 == tid) {
    LIBXS_PREDICT_FREE(dcounts, pool_dcnt);
    LIBXS_PREDICT_FREE(dists, pool_dist);
    LIBXS_PREDICT_FREE(comp, pool_comp);
    LIBXS_PREDICT_FREE(counts, pool_cnt);
    free(model->norm_cen);
    model->norm_cen = NULL;
  }
  /* the partition is complete for every task, not just the one that closed it */
  internal_libxs_predict_sync(model, ntasks);
}


LIBXS_API_INLINE double internal_libxs_predict_local_error(
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  int pos, int output_j)
{
  double result = cl->errors[output_j];
  const int nc = cl->nentries;
  const int radius = LIBXS_MIN(LIBXS_PREDICT_LOCAL_RADIUS, nc / 2);
  if (radius >= 2 && NULL != model->entries && NULL != cl->sorted_idx) {
    const int lo = LIBXS_MAX(pos - radius, 0);
    const int hi = LIBXS_MIN(pos + radius, nc - 1);
    const int len = hi - lo + 1;
    if (len >= 3) {
      double local[2 * LIBXS_PREDICT_LOCAL_RADIUS + 1];
      libxs_fprint_t fp;
      const size_t shape = (size_t)len;
      int k;
      for (k = 0; k < len; ++k) {
        local[k] = model->entries[cl->sorted_idx[lo + k]].outputs[output_j];
      }
      if (EXIT_SUCCESS == libxs_fprint(&fp, LIBXS_DATATYPE_F64, local,
        1, &shape, NULL, LIBXS_MIN(2, len - 1), 0, 0, 0))
      {
        const double raw1 = libxs_fprint_raw(&fp, 1, fp.linf[1]);
        result = LIBXS_MIN(result, raw1);
      }
    }
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_position(
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  const double* inputs)
{
  const int nc = cl->nentries;
  const int m = model->ninputs;
  /* linear scan: clusters are small (sqrt(P) entries typically) */
  double best = DBL_MAX;
  int best_k = 0, k;
  for (k = 0; k < nc; ++k) {
    const double d = internal_libxs_predict_dist2(inputs,
      cl->kd_pts + (size_t)k * m, m, model->has_missing);
    if (d < best) { best = d; best_k = k; }
  }
  return (double)best_k;
}


LIBXS_API_INLINE void internal_libxs_predict_lsq_fit(
  const double* y, int nc, int order, double* coeffs, double* error)
{
  const int m = order + 1;
  double a[(LIBXS_FPRINT_MAXORDER + 1) * (LIBXS_FPRINT_MAXORDER + 1)];
  double g[LIBXS_FPRINT_MAXORDER + 1];
  int i, p, q;
  for (p = 0; p < m; ++p) {
    g[p] = 0;
    for (q = 0; q < m; ++q) a[p * m + q] = 0;
  }
  for (i = 0; i < nc; ++i) {
    double phi[LIBXS_FPRINT_MAXORDER + 1];
    for (p = 0; p < m; ++p) phi[p] = libxs_binom((double)i, p);
    for (p = 0; p < m; ++p) {
      g[p] += phi[p] * y[i];
      for (q = 0; q < m; ++q) a[p * m + q] += phi[p] * phi[q];
    }
  }
  for (p = 0; p < m; ++p) {
    int piv = p;
    for (q = p + 1; q < m; ++q) {
      if (LIBXS_FABS(a[q * m + p]) > LIBXS_FABS(a[piv * m + p])) piv = q;
    }
    if (piv != p) {
      for (q = 0; q < m; ++q) {
        const double t = a[p * m + q];
        a[p * m + q] = a[piv * m + q];
        a[piv * m + q] = t;
      }
      { const double t = g[p]; g[p] = g[piv]; g[piv] = t; }
    }
    { const double d = a[p * m + p];
      if (LIBXS_FABS(d) > 1e-300) {
        for (q = p + 1; q < m; ++q) {
          const double f = a[q * m + p] / d;
          for (i = p; i < m; ++i) a[q * m + i] -= f * a[p * m + i];
          g[q] -= f * g[p];
        }
      }
    }
  }
  for (p = m - 1; p >= 0; --p) {
    double s = g[p];
    for (q = p + 1; q < m; ++q) s -= a[p * m + q] * coeffs[q];
    coeffs[p] = (LIBXS_FABS(a[p * m + p]) > 1e-300) ? s / a[p * m + p] : 0;
  }
  { double emax = 0;
    for (i = 0; i < nc; ++i) {
      double pred = 0, r;
      int k;
      for (k = 0; k < m; ++k) pred += coeffs[k] * libxs_binom((double)i, k);
      r = pred - y[i];
      if (r < 0) r = -r;
      if (r > emax) emax = r;
    }
    *error = emax;
  }
}


LIBXS_API_INLINE void internal_libxs_predict_cluster_refit(
  internal_libxs_predict_cluster_t* cl, int n, int use_fprint)
{
  const int nc = cl->nentries;
  const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
  int j, k;
  int buf_pool = 0;
  double* buf = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nc * sizeof(double), buf_pool);
  if (NULL != buf) {
    for (j = 0; j < n; ++j) {
      int ndistinct = 0, d;
      double prev, fnoise = 0;
      for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
      libxs_sort(buf, nc, sizeof(double), libxs_cmp_f64, NULL);
      prev = buf[0]; ndistinct = 1;
      for (k = 1; k < nc; ++k) {
        if (buf[k] != prev) { ++ndistinct; prev = buf[k]; }
      }
      cl->ndistinct[j] = ndistinct;
      for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
      if (0 != use_fprint) {
        const size_t shape = (size_t)nc;
        const size_t stride = (size_t)n;
        libxs_fprint_t fp;
        int decay_order = 0;
        libxs_fprint(&fp, LIBXS_DATATYPE_F64, cl->raw_outputs + j,
          1, &shape, &stride, LIBXS_FPRINT_MAXORDER, 0, 0, 0);
        cl->order[j] = cl->maxorder;
        cl->interpolated[j] = 0;
        if (0 < fp.l2[0]) {
          for (d = 1; d <= fp.order; ++d) {
            if (fp.l2[d] < fp.l2[d - 1]) ++decay_order;
            else d = fp.order + 1;
          }
          fnoise = fp.l2[decay_order] / fp.l2[0];
        }
        if (ndistinct <= ndistinct_thresh || decay_order < 2) {
          cl->mode[j] = 1;
        }
        else {
          cl->mode[j] = 0;
          cl->interpolated[j] = 1;
          cl->order[j] = LIBXS_MIN(decay_order, cl->maxorder);
        }
      }
      else {
        if (ndistinct <= ndistinct_thresh) {
          cl->mode[j] = 1;
          cl->interpolated[j] = 0;
        }
      }
      { const int trunc_order = LIBXS_MIN(cl->order[j], cl->maxorder);
        double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
        const int use_lsq = (0 == cl->mode[j])
          && (fnoise > LIBXS_PREDICT_LSQ_NOISE)
          && (nc >= LIBXS_PREDICT_LSQ_MINRATIO * (trunc_order + 1));
        cl->order[j] = LIBXS_MIN(trunc_order, nc - 1);
        if (0 != use_lsq) {
          internal_libxs_predict_lsq_fit(buf, nc, cl->order[j], cj,
            &cl->errors[j]);
        }
        else {
          cj[0] = buf[0];
          for (d = 1; d <= cl->order[j] && d < nc; ++d) {
            for (k = 0; k < nc - d; ++k) buf[k] = buf[k + 1] - buf[k];
            cj[d] = buf[0];
          }
          if (cl->order[j] < nc - 1) {
            double emax = 0;
            for (k = 0; k < nc - cl->order[j] - 1; ++k) {
              buf[k] = buf[k + 1] - buf[k];
            }
            for (k = 0; k < nc - cl->order[j] - 1; ++k) {
              const double a = buf[k] < 0 ? -buf[k] : buf[k];
              if (a > emax) emax = a;
            }
            cl->errors[j] = emax;
          }
          else {
            cl->errors[j] = 0;
          }
        }
        if (NULL != cl->out_rms && 0 == cl->mode[j]) {
          const int od = cl->order[j];
          double sse = 0;
          int ki;
          for (ki = 0; ki < nc; ++ki) {
            double pred = 0, actual, res;
            int di;
            for (di = 0; di <= od; ++di) {
              pred += cj[di] * libxs_binom((double)ki, di);
            }
            actual = cl->raw_outputs[(size_t)ki * n + j];
            res = pred - actual;
            sse += res * res;
          }
          cl->out_rms[j] = sqrt(sse / nc);
        }
      }
    }
    LIBXS_PREDICT_FREE(buf, buf_pool);
  }
  { int nclassify = 0;
    double sig_sum = 0;
    for (j = 0; j < n; ++j) {
      nclassify += cl->mode[j];
      sig_sum += cl->errors[j];
    }
    cl->fprint_sig = (n > 0) ? sig_sum / n : 0.0;
    cl->k_eff = (nclassify > n / 2)
      ? LIBXS_MIN(LIBXS_MAX(5, nc / 3), LIBXS_PREDICT_KNN)
      : LIBXS_MIN(LIBXS_MAX(3, (int)(sqrt((double)nc) + 0.5)),
          LIBXS_PREDICT_KNN);
  }
}


LIBXS_API_INLINE double internal_libxs_predict_quantile_z(double q)
{
  const double p = 1.0 - q;
  const double t = sqrt(-2.0 * log(1.0 - p));
  const double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
  const double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
  return t - (c0 + c1 * t + c2 * t * t)
    / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
}


LIBXS_API_INLINE double internal_libxs_predict_coverage(
  int nentries, int total_entries, int nclusters)
{
  const double expected = (double)total_entries / nclusters;
  const double ratio = (expected > 0) ? (double)nentries / expected : 1.0;
  return ratio < 1.0 ? ratio : 1.0;
}


/**
 * The local evidence a query draws on for one output: the k nearest neighbors
 * within the cluster, their output values and distances, plus whether the query
 * coincides with a stored point. Every scoring rule in this file reads this
 * same object - the kNN vote reduces it to a winner, a probability reads it at
 * an arbitrary value - so the scan lives here once rather than being repeated
 * per rule, where the tangent projection, per-output group filter and recency
 * weighting would have to be kept in step by hand.
 */
/**
 * Squared distance restricted to a window view: the full distance less the
 * coordinates the view does not read. Subtracting the terms is what a zero
 * weight would have done, and it needs no second copy of the points.
 */
LIBXS_API_INLINE double internal_libxs_predict_viewdist2(const double* a,
  const double* b, int m, const internal_libxs_predict_view_t* view,
  int missing)
{
  double result = internal_libxs_predict_dist2(a, b, m, missing);
  if (NULL != view && view->w < view->full && 0 < view->s) {
    int si, li;
    for (si = 0; si < view->s; ++si) {
      for (li = 0; li < view->full - view->w; ++li) {
        const int c = si * view->full + li;
        const double d = a[c] - b[c];
        result -= d * d;
      }
    }
    if (0 > result) result = 0;
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_evidence(
  const internal_libxs_predict_cluster_t* cl,
  int m, const double* inputs, int output_j, int nouts,
  int extrapolate, int skip_local, const char* skip_set,
  const int* po_groups, int query_group,
  double* candidates, double* dists, int* out_nfound,
  int* out_exact, int* out_exact_nearest, double* out_best,
  const internal_libxs_predict_view_t* view, int missing, double* out_iw)
{
  const double* kd_pts = cl->kd_pts;
  const int nc = cl->nentries;
  /**
   * k_eff sizes the caller's candidates/dists arrays, and on a loaded model it
   * comes from a file byte, so the bound is clamped here rather than assumed.
   * The loaders reject an out-of-range value, but keeping the clamp local means
   * the writes cannot exceed the arrays whatever the caller passed.
   */
  const int kreq = (NULL != cl->k_out) ? cl->k_out[output_j] : cl->k_eff;
  const int k = (kreq < LIBXS_PREDICT_KNN)
    ? ((0 < kreq) ? kreq : 1) : LIBXS_PREDICT_KNN;
  double qtan[512];
  const double* qpts = inputs;
  const double* dpts = kd_pts;
  int dm = m;
  int nfound = 0, exact = 0, exact_nearest = 0, i, max_idx = 0;
  if (NULL != cl->tangent && NULL != cl->kd_tan
    && cl->tdim > 0 && cl->tdim <= 512)
  {
    int tj, tk;
    for (tj = 0; tj < cl->tdim; ++tj) {
      double acc = 0;
      for (tk = 0; tk < m; ++tk) acc += cl->tangent[tj * m + tk] * inputs[tk];
      qtan[tj] = acc;
    }
    qpts = qtan;
    dpts = cl->kd_tan;
    dm = cl->tdim;
  }
  /* recency weighting needs sorted_idx, which a loaded flat model lacks */
  if (NULL == cl->sorted_idx) extrapolate = 0;
  if (0 != extrapolate) {
    for (i = 0; i < nc; ++i) {
      if (cl->sorted_idx[i] > max_idx) max_idx = cl->sorted_idx[i];
    }
  }
  for (i = 0; i < nc; ++i) {
    double d2;
    if (i == skip_local) continue;
    if (NULL != skip_set && 0 != skip_set[i]) continue;
    d2 = internal_libxs_predict_dist2(qpts, dpts + (size_t)i * dm, dm,
      missing);
    if (NULL != view && view->w < view->full && qpts == inputs) {
      /**
       * A view reads only the most recent view_w lags of each series. Rather
       * than pack a second copy of the points, the older lags are removed from
       * the distance they already contributed: subtracting a coordinate's term
       * is what a zero weight would have done, and it leaves the corpus, the
       * partition and the neighbor index shared. Skipped when the tangent
       * projection is active, because there the coordinates are no longer lags.
       */
      const double* row = dpts + (size_t)i * dm;
      int si, li;
      for (si = 0; si < view->s; ++si) {
        for (li = 0; li < view->full - view->w; ++li) {
          const int c = si * view->full + li;
          const double d = qpts[c] - row[c];
          d2 -= d * d;
        }
      }
      if (0 > d2) d2 = 0;
    }
    if (0 != extrapolate && max_idx > 0) {
      const double age = 1.0 - (double)cl->sorted_idx[i] / (double)max_idx;
      d2 *= 1.0 + 0.5 * age;
    }
    if (NULL != po_groups && NULL != cl->sorted_idx && query_group >= 0
      && po_groups[cl->sorted_idx[i]] != query_group)
    {
      continue;
    }
    /**
     * An exact match is authoritative for the value wherever the scan finds
     * it, not only when it happens to be admitted first. Collapsing the
     * spread is a separate question: with duplicate input vectors (9410 in
     * the crystal set) a zero-distance neighbor is ordinary evidence, and
     * reporting zero variance for it makes the compression criterion drop
     * entries it must keep. Hence nearest, tracked separately.
     */
    if (0.0 == d2) {
      if (0 == exact) {
        *out_best = cl->raw_outputs[(size_t)i * nouts + output_j];
        exact = 1;
      }
      if (0 == nfound) exact_nearest = 1;
    }
    if (nfound < k) {
      candidates[nfound] = cl->raw_outputs[(size_t)i * nouts + output_j];
      dists[nfound] = sqrt(d2);
      if (NULL != out_iw) {
        out_iw[nfound] = (NULL != cl->eweight) ? cl->eweight[i] : 1.0;
      }
      ++nfound;
    }
    else {
      int worst = 0, wi;
      for (wi = 1; wi < nfound; ++wi) {
        if (dists[wi] > dists[worst]) worst = wi;
      }
      if (sqrt(d2) < dists[worst]) {
        candidates[worst] = cl->raw_outputs[(size_t)i * nouts + output_j];
        dists[worst] = sqrt(d2);
        if (NULL != out_iw) {
          out_iw[worst] = (NULL != cl->eweight) ? cl->eweight[i] : 1.0;
        }
      }
    }
  }
  *out_nfound = nfound;
  *out_exact = exact;
  *out_exact_nearest = exact_nearest;
}


LIBXS_API_INLINE int internal_libxs_predict_central(
  const libxs_predict_t* model, int output_j)
{
  int result = model->central;
  if (0 >= result) {
    result = (NULL != model->central_out && 0 <= output_j
      && output_j < model->noutputs) ? model->central_out[output_j] : 0;
  }
  /* 2 selects the mean explicitly; the vote only tests for non-zero. */
  return (2 != result) ? result : 0;
}


LIBXS_API_INLINE double internal_libxs_predict_classify2(
  const internal_libxs_predict_cluster_t* cl,
  int m, const double* inputs, int output_j, int nouts,
  int ndistinct, int extrapolate, int skip_local, const char* skip_set,
  const int* po_groups, int query_group,
  double* confidence, double* out_variance,
  double quantile, double* out_lower, double* out_upper,
  int central, const internal_libxs_predict_view_t* view, int missing)
{
  const int nc = cl->nentries;
  const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
  double candidates[LIBXS_PREDICT_KNN];
  double dists[LIBXS_PREDICT_KNN];
  double iw[LIBXS_PREDICT_KNN];
  double best_val = 0.0;
  int nfound = 0, exact = 0, exact_nearest = 0, i;
  if (NULL != confidence) *confidence = 0.0;
  if (NULL != out_variance) *out_variance = 0.0;
  if (NULL != out_lower) *out_lower = 0.0;
  if (NULL != out_upper) *out_upper = 0.0;
  if (nc > 0 && NULL != cl->raw_outputs) {
    best_val = cl->raw_outputs[output_j];
    internal_libxs_predict_evidence(cl, m, inputs, output_j, nouts,
      extrapolate, skip_local, skip_set, po_groups, query_group,
      candidates, dists, &nfound, &exact, &exact_nearest, &best_val, view,
      missing, iw);
    if (NULL != out_variance) {
      if (0 != exact_nearest || nfound <= 1) {
        *out_variance = 0;
      }
      else {
        double mean = 0, v = 0;
        for (i = 0; i < nfound; ++i) mean += candidates[i];
        mean /= nfound;
        for (i = 0; i < nfound; ++i) {
          const double d = candidates[i] - mean;
          v += d * d;
        }
        *out_variance = v / nfound;
      }
    }
    if (NULL != out_lower && NULL != out_upper && quantile > 0
      && nfound > 1 && 0 == exact_nearest)
    {
      /**
       * Zero-initialized because nfound arrives through a pointer from the
       * evidence scan: nothing at this point proves it positive, so a reader
       * cannot see that the first element was written. The loops below are
       * bounded by nfound in any case, but leaving the arrays indeterminate
       * makes the code depend on that proof holding, which it does not across
       * the extraction boundary.
       */
      double weights[LIBXS_PREDICT_KNN] = { 0 };
      double sorted_v[LIBXS_PREDICT_KNN] = { 0 };
      double sorted_w[LIBXS_PREDICT_KNN] = { 0 };
      const int nq = (nfound < LIBXS_PREDICT_KNN)
        ? nfound : LIBXS_PREDICT_KNN;
      double wsum = 0;
      int si, sj;
      for (i = 0; i < nq; ++i) {
        weights[i] = iw[i] * ((dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30);
        wsum += weights[i];
      }
      for (i = 0; i < nq; ++i) {
        sorted_v[i] = candidates[i];
        sorted_w[i] = weights[i] / wsum;
      }
      for (si = 0; si < nq - 1; ++si) {
        for (sj = si + 1; sj < nq; ++sj) {
          if (sorted_v[sj] < sorted_v[si]) {
            double tv = sorted_v[si], tw = sorted_w[si];
            sorted_v[si] = sorted_v[sj]; sorted_w[si] = sorted_w[sj];
            sorted_v[sj] = tv; sorted_w[sj] = tw;
          }
        }
      }
      { double cum = 0;
        *out_lower = sorted_v[0];
        for (i = 0; i < nq; ++i) {
          cum += sorted_w[i];
          if (cum >= quantile) { *out_lower = sorted_v[i]; break; }
        }
      }
      { double cum = 0;
        *out_upper = sorted_v[nq - 1];
        for (i = nq - 1; i >= 0; --i) {
          cum += sorted_w[i];
          if (cum >= quantile) { *out_upper = sorted_v[i]; break; }
        }
      }
    }
    if (0 == exact && nfound > 0) {
      if (ndistinct > ndistinct_thresh) {
        double wsum = 0, wavg = 0;
        for (i = 0; i < nfound; ++i) {
          const double wi = iw[i] * ((dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30);
          wavg += wi * candidates[i];
          wsum += wi;
        }
        wavg = (wsum > 0.0) ? wavg / wsum : candidates[0];
        /**
         * A right-skewed neighborhood pulls the weighted average off the bulk
         * of its own evidence, and absolute error is minimized by the median
         * rather than the mean. The median is unweighted on purpose: distance
         * weighting measured worse than none (earthquake MAE 0.241 vs 0.236),
         * because it re-concentrates the estimate on the few nearest neighbors
         * and gives up the robustness the median was chosen for.
         */
        if (0 != central && nfound > 1) {
          double sorted[LIBXS_PREDICT_KNN];
          int si, sj;
          for (i = 0; i < nfound; ++i) sorted[i] = candidates[i];
          for (si = 0; si < nfound - 1; ++si) {
            for (sj = si + 1; sj < nfound; ++sj) {
              if (sorted[sj] < sorted[si]) {
                const double t = sorted[si];
                sorted[si] = sorted[sj];
                sorted[sj] = t;
              }
            }
          }
          wavg = (0 == (nfound & 1))
            ? (0.5 * (sorted[nfound / 2 - 1] + sorted[nfound / 2]))
            : sorted[nfound / 2];
        }
        /**
         * The mean is snapped to the nearest value the cluster attests, because
         * an average of attested values need not be one. A median of an odd
         * count already is one, and snapping it would search the whole cluster
         * and can only move it off the neighborhood it summarizes, so the
         * median is reported as computed.
         */
        if (0 != extrapolate || (0 != central && nfound > 1)) {
          best_val = wavg;
        }
        else {
          double best_dist = DBL_MAX;
          for (i = 0; i < nc; ++i) {
            const double v = cl->raw_outputs[(size_t)i * nouts + output_j];
            const double d = (v > wavg) ? (v - wavg) : (wavg - v);
            if (d < best_dist) { best_dist = d; best_val = v; }
          }
        }
        if (NULL != confidence) {
          /**
           * A many-valued output has no vote fraction, so it reports 1.0 and
           * callers read info->variance for the neighborhood spread. Folding
           * that spread into the confidence instead was measured and removed:
           * it lowered quality wherever it applied once the blended-cluster
           * count was derived from the confidence rather than fixed.
           */
          *confidence = 1.0;
        }
      }
      else {
        double best_weight = 0;
        for (i = 0; i < nfound; ++i) {
          double ws = 0;
          int ii;
          for (ii = 0; ii < nfound; ++ii) {
            if (candidates[ii] == candidates[i]) {
              ws += iw[ii] * ((dists[ii] > 0.0) ? (1.0 / dists[ii]) : 1e30);
            }
          }
          if (ws > best_weight) { best_weight = ws; best_val = candidates[i]; }
        }
        if (NULL != confidence) {
          double total_weight = 0;
          for (i = 0; i < nfound; ++i) {
            total_weight += iw[i]
              * ((dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30);
          }
          *confidence = (total_weight > 0.0) ? best_weight / total_weight : 1.0;
        }
      }
    }
    else if (NULL != confidence) {
      *confidence = 1.0;
    }
  }
  return best_val;
}


LIBXS_API_INLINE double internal_libxs_predict_classify(
  const internal_libxs_predict_cluster_t* cl,
  int m, const double* inputs, int output_j, int nouts,
  int ndistinct, int extrapolate, int skip_local,
  double* confidence, double* out_variance, int central,
  const internal_libxs_predict_view_t* view, int missing)
{
  return internal_libxs_predict_classify2(cl, m, inputs,
    output_j, nouts, ndistinct, extrapolate, skip_local, NULL,
    NULL, -1, confidence, out_variance, 0, NULL, NULL, central, view,
    missing);
}


LIBXS_API libxs_predict_t* libxs_predict_create(int ninputs, int noutputs)
{
  libxs_predict_t* model = NULL;
  if (0 < ninputs && 0 < noutputs) {
    model = (libxs_predict_t*)calloc(1, sizeof(libxs_predict_t));
    if (NULL != model) {
      model->ninputs = ninputs;
      model->noutputs = noutputs;
      model->eval_mode = LIBXS_PREDICT_AUTO;
      model->diff_mode = -1;
      model->nbank = 1;
    }
  }
  return model;
}


LIBXS_API void libxs_predict_destroy(libxs_predict_t* model)
{
  if (NULL != model) {
    if (NULL != model->sup_vals) {
      int j;
      for (j = 0; j < model->noutputs; ++j) {
        free(model->sup_vals[j]);
        free(model->sup_freq[j]);
      }
    }
    free(model->sup_vals);
    free(model->sup_freq);
    free(model->sup_n);
    free(model->sup_tot);
    free(model->escape_w);
    free(model->central_out);
    model->central_out = NULL;
    model->sup_vals = NULL;
    model->sup_freq = NULL;
    model->sup_n = NULL;
    model->sup_tot = NULL;
    model->escape_w = NULL;
  }
  if (NULL != model) {
    internal_libxs_predict_free_clusters(model);
    /* the entries address the arena and own nothing of their own */
    free(model->entries);
    free(model->arena);
    free(model->input_min);
    free(model->input_rng);
    free(model->input_knot);
    free(model->weights);
    free(model->transforms);
    free(model->ts_buf);
    free(model->aux_buf);
    free(model->bank_w);
    free(model->k_sel);
    free(model->decompose_mat);
    free(model->hknn_assignments);
    if (NULL != model->hknn_po_assignments) {
      const int ng = (model->hknn_ngroups > 0)
        ? model->hknn_ngroups : model->noutputs;
      int gi;
      for (gi = 0; gi < ng; ++gi) {
        free(model->hknn_po_assignments[gi]);
      }
      free(model->hknn_po_assignments);
    }
    if (NULL != model->hknn_po_clusters) {
      const int ng = (model->hknn_ngroups > 0)
        ? model->hknn_ngroups : model->noutputs;
      int gi;
      for (gi = 0; gi < ng; ++gi) {
        if (NULL != model->hknn_po_clusters[gi]) {
          const int nc = (NULL != model->hknn_po_nclusters)
            ? model->hknn_po_nclusters[gi] : 0;
          int ci;
          for (ci = 0; ci < nc; ++ci) {
            internal_libxs_predict_cluster_t* cl =
              &model->hknn_po_clusters[gi][ci];
            free(cl->centroid);
            free(cl->kd_pts);
            free(cl->raw_outputs);
            free(cl->sorted_idx);
            free(cl->sorted_dist);
            free(cl->order);
            free(cl->mode);
            free(cl->ndistinct);
            free(cl->k_out);
            free(cl->eweight);
            free(cl->interpolated);
            free(cl->coeffs);
            free(cl->errors);
            free(cl->tangent);
            free(cl->kd_tan);
          }
          free(model->hknn_po_clusters[gi]);
        }
      }
      free(model->hknn_po_clusters);
    }
    free(model->hknn_po_nclusters);
    free(model->hknn_po_groups);
    if (NULL != model->rf) {
      int ti;
      const int total_trees = model->rf->ntrees * model->rf->noutputs;
      for (ti = 0; ti < total_trees; ++ti) {
        free(model->rf->trees[ti].nodes);
        free(model->rf->trees[ti].incr);
      }
      free(model->rf->trees);
      free(model->rf->label_offset);
      free(model->rf->regress);
      free(model->rf->nclass);
      free(model->rf->depth);
      free(model->rf);
    }
    free(model);
  }
}


LIBXS_API libxs_lock_t* libxs_predict_lock(libxs_predict_t* model)
{
  return (NULL != model) ? &model->lock : NULL;
}


LIBXS_API void libxs_predict_set_mode(libxs_predict_t* model, int mode)
{
  LIBXS_ASSERT(NULL != model);
  model->eval_mode = mode;
}


LIBXS_API void libxs_predict_set_central(libxs_predict_t* model, int mode)
{
  LIBXS_ASSERT(NULL != model);
  model->central = mode;
}


LIBXS_API void libxs_predict_set_refine(libxs_predict_t* model, int iterations)
{
  LIBXS_ASSERT(NULL != model);
  model->refine = iterations;
}


LIBXS_API void libxs_predict_set_smooth(libxs_predict_t* model, double amount)
{
  LIBXS_ASSERT(NULL != model);
  model->smooth = amount;
}


LIBXS_API void libxs_predict_set_consistency(
  libxs_predict_t* model, double amount)
{
  LIBXS_ASSERT(NULL != model);
  model->consistency = amount;
}


LIBXS_API void libxs_predict_set_quantile(
  libxs_predict_t* model, double quantile)
{
  LIBXS_ASSERT(NULL != model);
  model->quantile = (quantile > 0 && quantile < 0.5) ? quantile : 0;
}


LIBXS_API void libxs_predict_set_weights(libxs_predict_t* model, const double weights[])
{
  LIBXS_ASSERT(NULL != model);
  if (NULL == weights) {
    free(model->weights);
    model->weights = NULL;
  }
  else {
    const int m = model->ninputs;
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      memcpy(model->weights, weights, (size_t)m * sizeof(double));
    }
  }
}


LIBXS_API void libxs_predict_set_transform(libxs_predict_t* model, int output, int transform)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL == model->transforms) {
    model->transforms = (int*)calloc((size_t)model->noutputs, sizeof(int));
  }
  if (NULL != model->transforms) {
    if (0 > output) {
      int j;
      for (j = 0; j < model->noutputs; ++j) model->transforms[j] = transform;
    }
    else if (output < model->noutputs) {
      model->transforms[output] = transform;
    }
  }
}


LIBXS_API_INLINE double internal_libxs_predict_fwd(int transform, double v)
{
  double result = v;
  switch (transform) {
    case LIBXS_PREDICT_LOG: result = log(v + 1.0); break;
    case LIBXS_PREDICT_SQRT: result = sqrt(v > 0 ? v : 0); break;
    default: break;
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_inv(int transform, double v)
{
  double result = v;
  switch (transform) {
    case LIBXS_PREDICT_LOG: result = exp(v) - 1.0; break;
    case LIBXS_PREDICT_SQRT: result = v * v; break;
    default: break;
  }
  return result;
}


LIBXS_API void libxs_predict_set_series(libxs_predict_t* model, int nseries, int window)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 < nseries
    && (0 >= window || nseries * window <= model->ninputs))
  {
    model->nseries = nseries;
    model->window = window;
  }
}


LIBXS_API void libxs_predict_set_series_bank(libxs_predict_t* model, int nbank)
{
  LIBXS_ASSERT(NULL != model);
  model->nbank = (1 < nbank) ? nbank : 1;
}


LIBXS_API void libxs_predict_set_series_aux(libxs_predict_t* model, int naux)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 <= naux) {
    model->naux = naux;
  }
}


LIBXS_API void libxs_predict_set_series_deriv(libxs_predict_t* model, int nderiv)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 <= nderiv) {
    model->nderiv = nderiv;
  }
}


LIBXS_API void libxs_predict_set_target(libxs_predict_t* model, int target)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 <= target && target < model->nseries) {
    model->target = target;
  }
}


LIBXS_API void libxs_predict_set_decompose(libxs_predict_t* model, int decompose)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->decompose = decompose;
  }
}


LIBXS_API void libxs_predict_set_diff(libxs_predict_t* model, int order)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->diff_mode = order;
  }
}


/**
 * Derive has_missing from whatever the model actually holds. A loaded model has
 * no setter call behind it and may have kept only the clustered points, so the
 * flag is recovered from the values rather than stored, which is what keeps the
 * serialization format unchanged.
 */
LIBXS_API_INLINE void internal_libxs_predict_missing_all(libxs_predict_t* model)
{
  const int m = model->ninputs;
  int i;
  model->has_missing = 0;
  if (NULL != model->entries) {
    for (i = 0; i < model->nentries && 0 == model->has_missing; ++i) {
      if (0 != internal_libxs_predict_incomplete(model->entries[i].inputs, m)) {
        model->has_missing = 1;
      }
    }
  }
  else if (NULL != model->clusters) {
    int c;
    for (c = 0; c < model->nclusters && 0 == model->has_missing; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (NULL != cl->kd_pts) {
        for (i = 0; i < cl->nentries && 0 == model->has_missing; ++i) {
          if (0 != internal_libxs_predict_incomplete(
            cl->kd_pts + (size_t)i * m, m))
          {
            model->has_missing = 1;
          }
        }
      }
    }
  }
}


LIBXS_API void libxs_predict_set_floor(libxs_predict_t* model, double floor)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 <= floor && 1 >= floor) {
    model->floor = floor;
  }
}


LIBXS_API void libxs_predict_set_missing(libxs_predict_t* model, int enable)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->missing_mode = enable;
  }
}


LIBXS_API void libxs_predict_set_forest(libxs_predict_t* model,
  int ntrees, int max_depth)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->rf_ntrees = ntrees;
    model->rf_depth = max_depth;
  }
}


LIBXS_API void libxs_predict_set_neighbors(libxs_predict_t* model, int k)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->kreq = k;
    /* a later request replaces an earlier resolution rather than joining it */
    free(model->k_sel);
    model->k_sel = NULL;
  }
}


LIBXS_API_INLINE void internal_libxs_predict_decompose_apply(
  const libxs_predict_t* model, const double* raw, double* out)
{
  const int m = model->ninputs;
  if (NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'N', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, raw, out);
  }
  else {
    memcpy(out, raw, (size_t)m * sizeof(double));
  }
}


LIBXS_API_INLINE void internal_libxs_predict_decompose_inverse(
  const libxs_predict_t* model, const double* modes, double* raw)
{
  const int m = model->ninputs;
  if (NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'T', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, modes, raw);
  }
  else {
    memcpy(raw, modes, (size_t)m * sizeof(double));
  }
}


LIBXS_API_INLINE void internal_libxs_predict_jacobi(
  double* a, int m, double* evec, double* eval)
{
  int j, k;
  for (j = 0; j < m; ++j) {
    for (k = 0; k < m; ++k) evec[j * m + k] = (j == k) ? 1.0 : 0.0;
  }
  { int iter;
    for (iter = 0; iter < 100 * m; ++iter) {
      int pi = 0, qi = 1;
      double maxoff = 0;
      for (j = 0; j < m; ++j) {
        for (k = j + 1; k < m; ++k) {
          const double v = a[j * m + k] < 0 ? -a[j * m + k] : a[j * m + k];
          if (v > maxoff) { maxoff = v; pi = j; qi = k; }
        }
      }
      if (maxoff < 1e-12) break;
      { const double app = a[pi * m + pi], aqq = a[qi * m + qi];
        const double apq = a[pi * m + qi];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0 ? 1.0 : -1.0)
          / (LIBXS_FABS(tau) + sqrt(1.0 + tau * tau));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;
        for (k = 0; k < m; ++k) {
          const double ik = a[k * m + pi], jk = a[k * m + qi];
          a[k * m + pi] = c * ik - s * jk;
          a[k * m + qi] = s * ik + c * jk;
          a[pi * m + k] = a[k * m + pi];
          a[qi * m + k] = a[k * m + qi];
        }
        a[pi * m + pi] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[qi * m + qi] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[pi * m + qi] = 0;
        a[qi * m + pi] = 0;
        for (k = 0; k < m; ++k) {
          const double ek = evec[k * m + pi], fk = evec[k * m + qi];
          evec[k * m + pi] = c * ek - s * fk;
          evec[k * m + qi] = s * ek + c * fk;
        }
      }
    }
  }
  for (j = 0; j < m; ++j) eval[j] = a[j * m + j];
  for (j = 0; j < m - 1; ++j) {
    int best = j;
    for (k = j + 1; k < m; ++k) {
      if (eval[k] > eval[best]) best = k;
    }
    if (best != j) {
      { double tmp = eval[j]; eval[j] = eval[best]; eval[best] = tmp; }
      for (k = 0; k < m; ++k) {
        double tmp = evec[k * m + j];
        evec[k * m + j] = evec[k * m + best];
        evec[k * m + best] = tmp;
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_symeig(
  const internal_libxs_predict_entry_t* entries, const double* pts,
  int npts, int m, double* mean, double* cov, double* evec, double* eval)
{
  const size_t msz = (size_t)m * (size_t)m;
  int i, j, k;
  memset(mean, 0, (size_t)m * sizeof(double));
  memset(cov, 0, msz * sizeof(double));
  for (i = 0; i < npts; ++i) {
    const double* inp = (NULL != entries) ? entries[i].inputs : (pts + (size_t)i * m);
    for (j = 0; j < m; ++j) mean[j] += inp[j];
  }
  for (j = 0; j < m; ++j) mean[j] /= npts;
  for (i = 0; i < npts; ++i) {
    const double* inp = (NULL != entries) ? entries[i].inputs : (pts + (size_t)i * m);
    for (j = 0; j < m; ++j) {
      const double dj = inp[j] - mean[j];
      for (k = j; k < m; ++k) {
        cov[j * m + k] += dj * (inp[k] - mean[k]);
      }
    }
  }
  for (j = 0; j < m; ++j) {
    for (k = j; k < m; ++k) {
      cov[j * m + k] /= npts;
      cov[k * m + j] = cov[j * m + k];
    }
  }
  internal_libxs_predict_jacobi(cov, m, evec, eval);
}


LIBXS_API_INLINE void internal_libxs_predict_cluster_tangent(
  internal_libxs_predict_cluster_t* cl, int m, int weight_mode)
{
  const int nc = cl->nentries;
  const int margin = 2;
  int pool_mean = 0, pool_cov = 0, pool_evec = 0, pool_eval = 0;
  double* mean = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_mean);
  double* cov = (double*)LIBXS_PREDICT_MALLOC((size_t)m * (size_t)m * sizeof(double), pool_cov);
  double* evec = (double*)LIBXS_PREDICT_MALLOC((size_t)m * (size_t)m * sizeof(double), pool_evec);
  double* eval = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_eval);
  cl->tangent = NULL;
  cl->kd_tan = NULL;
  cl->tdim = 0;
  if (NULL != mean && NULL != cov && NULL != evec && NULL != eval
    && nc >= 4 && NULL != cl->kd_pts)
  {
    double total = 0, cum = 0;
    int t = m, j, k;
    internal_libxs_predict_symeig(NULL, cl->kd_pts, nc, m, mean, cov, evec, eval);
    for (j = 0; j < m; ++j) total += (eval[j] > 0 ? eval[j] : 0);
    if (total > 0) {
      t = m;
      for (j = 0; j < m; ++j) {
        cum += (eval[j] > 0 ? eval[j] : 0);
        if (cum >= 0.99 * total && t == m) t = j + 1;
      }
    }
    if (t < 1) t = 1;
    if (nc >= t + margin && t <= m) {
      double* tan = (double*)malloc((size_t)t * (size_t)m * sizeof(double));
      double* kdt = (double*)malloc((size_t)nc * (size_t)t * sizeof(double));
      if (NULL != tan && NULL != kdt) {
        const double eps = 1e-12;
        int i;
        for (j = 0; j < t; ++j) {
          const double lam = (eval[j] > 0 ? eval[j] : 0);
          double scale;
          if (2 == weight_mode) scale = 1.0 / sqrt(lam + eps);      /* whiten */
          else if (3 == weight_mode) scale = sqrt(lam);             /* emphasize */
          else scale = 1.0;                                         /* project */
          for (k = 0; k < m; ++k) tan[j * m + k] = evec[k * m + j] * scale;
        }
        for (i = 0; i < nc; ++i) {
          const double* src = cl->kd_pts + (size_t)i * m;
          double* dst = kdt + (size_t)i * t;
          for (j = 0; j < t; ++j) {
            double acc = 0;
            for (k = 0; k < m; ++k) acc += tan[j * m + k] * src[k];
            dst[j] = acc;
          }
        }
        cl->tangent = tan;
        cl->kd_tan = kdt;
        cl->tdim = t;
      }
      else {
        free(tan);
        free(kdt);
      }
    }
  }
  LIBXS_PREDICT_FREE(eval, pool_eval);
  LIBXS_PREDICT_FREE(evec, pool_evec);
  LIBXS_PREDICT_FREE(cov, pool_cov);
  LIBXS_PREDICT_FREE(mean, pool_mean);
}


LIBXS_API_INLINE void internal_libxs_predict_pca_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const size_t msz = (size_t)m * (size_t)m;
  int pool_mean = 0, pool_cov = 0, pool_evec = 0, pool_eval = 0;
  double* mean = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_mean);
  double* cov = (double*)LIBXS_PREDICT_MALLOC(msz * sizeof(double), pool_cov);
  double* evec = (double*)LIBXS_PREDICT_MALLOC(msz * sizeof(double), pool_evec);
  double* eval = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_eval);
  if (NULL != mean && NULL != cov && NULL != evec && NULL != eval) {
  internal_libxs_predict_symeig(model->entries, NULL, p, m, mean, cov, evec, eval);
  { int i, j, k;
    free(model->decompose_mat);
    model->decompose_mat = (double*)malloc(msz * sizeof(double));
    if (NULL != model->decompose_mat) {
      double total_var = 0, cum_var = 0;
      int npc = m;
      for (j = 0; j < m; ++j) total_var += (eval[j] > 0 ? eval[j] : 0);
      for (j = 0; j < m; ++j) {
        for (k = 0; k < m; ++k) {
          model->decompose_mat[j * m + k] = evec[k * m + j];
        }
      }
      for (j = 0; j < m; ++j) {
        cum_var += (eval[j] > 0 ? eval[j] : 0);
        if (cum_var >= 0.95 * total_var && npc == m) npc = j + 1;
      }
      if (npc < m && LIBXS_PREDICT_PCA == model->decompose) {
        if (NULL == model->weights) {
          model->weights = (double*)malloc((size_t)m * sizeof(double));
        }
        if (NULL != model->weights) {
          for (j = 0; j < m; ++j) model->weights[j] = (j < npc) ? 1.0 : 0.0;
        }
      }
      { int xmat_pool = 0, ymat_pool = 0;
        double* xmat = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)p * (size_t)m * sizeof(double), xmat_pool);
        double* ymat = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)p * (size_t)m * sizeof(double), ymat_pool);
        if (NULL != xmat && NULL != ymat) {
          { const double alpha = 1.0, beta = 0.0;
            const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
              LIBXS_DATATYPE_F64, 'N', 'N', m, p, m, m, m, m,
              &alpha, &beta, NULL);
            for (i = 0; i < p; ++i) {
              memcpy(xmat + (size_t)i * m, model->entries[i].inputs,
                (size_t)m * sizeof(double));
            }
            libxs_gemm_call(gemm, model->decompose_mat, xmat, ymat);
          }
          for (i = 0; i < p; ++i) {
            memcpy(model->entries[i].inputs, ymat + (size_t)i * m,
              (size_t)m * sizeof(double));
          }
        }
        LIBXS_PREDICT_FREE(ymat, ymat_pool);
        LIBXS_PREDICT_FREE(xmat, xmat_pool);
      }
    }
  }
  }
  LIBXS_PREDICT_FREE(mean, pool_mean);
  LIBXS_PREDICT_FREE(cov, pool_cov);
  LIBXS_PREDICT_FREE(evec, pool_evec);
  LIBXS_PREDICT_FREE(eval, pool_eval);
}


LIBXS_API_INLINE void internal_libxs_predict_fisher_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  int nclasses = 0, j, i, ci;
  int class_id[128], class_count[128];
  /**
   * Present values per class and per dimension. A class count alone cannot
   * normalize a mean here: missingness need not be spread evenly over the
   * classes, and dividing a partial sum by the full count would report a
   * discriminant that shrinks with how much of the column is absent.
   */
  int class_dn[128][128];
  double class_mean[128][128], class_var[128][128];
  LIBXS_ASSERT(m <= 128);
  memset(class_count, 0, sizeof(class_count));
  memset(class_dn, 0, sizeof(class_dn));
  memset(class_mean, 0, sizeof(class_mean));
  memset(class_var, 0, sizeof(class_var));
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    int found = 0;
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) { found = 1; break; }
    }
    if (0 == found && nclasses < 128) { class_id[nclasses] = label; ci = nclasses++; }
    if (ci < 128) {
      ++class_count[ci];
      for (j = 0; j < m; ++j) {
        const double v = model->entries[i].inputs[j];
        if (LIBXS_NOTNAN(v)) {
          class_mean[ci][j] += v;
          ++class_dn[ci][j];
        }
      }
    }
  }
  for (ci = 0; ci < nclasses; ++ci) {
    for (j = 0; j < m; ++j) {
      if (0 < class_dn[ci][j]) class_mean[ci][j] /= class_dn[ci][j];
    }
  }
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) {
        for (j = 0; j < m; ++j) {
          const double v = model->entries[i].inputs[j];
          if (LIBXS_NOTNAN(v)) {
            const double d = v - class_mean[ci][j];
            class_var[ci][j] += d * d;
          }
        }
        break;
      }
    }
  }
  if (nclasses >= 2 && 1 == model->noutputs) {
    double scores[128] = { 0.0 }, sorted_scores[128], thr;
    for (j = 0; j < m; ++j) {
      double between = 0, within = 0, grand_mean = 0;
      int total_n = 0;
      for (ci = 0; ci < nclasses; ++ci) {
        if (0 < class_dn[ci][j]) {
          grand_mean += class_mean[ci][j] * class_dn[ci][j];
          total_n += class_dn[ci][j];
          within += class_var[ci][j];
        }
      }
      if (0 < total_n) grand_mean /= total_n;
      for (ci = 0; ci < nclasses; ++ci) {
        if (0 < class_dn[ci][j]) {
          const double d = class_mean[ci][j] - grand_mean;
          between += class_dn[ci][j] * d * d;
        }
      }
      scores[j] = (within > 0 ? (between / within) : 0.0);
    }
    memcpy(sorted_scores, scores, (size_t)m * sizeof(double));
    libxs_sort(sorted_scores, m, sizeof(double), libxs_cmp_f64, NULL);
    thr = sorted_scores[m / 2];
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      for (j = 0; j < m; ++j) {
        model->weights[j] = (scores[j] >= thr ? sqrt(scores[j]) : 0.0);
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_setdiff_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int nclasses = 0, j, i, a, b;
  int class_id[128], class_count[128];
  double scores[128], sorted_scores[128], thr;
  LIBXS_ASSERT(m <= 128);
  memset(class_count, 0, sizeof(class_count));
  memset(scores, 0, sizeof(scores));
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    int found = 0, ci;
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) { ++class_count[ci]; found = 1; break; }
    }
    if (0 == found && nclasses < 128) {
      class_id[nclasses] = label;
      class_count[nclasses] = 1;
      ++nclasses;
    }
  }
  if (nclasses >= 2 && 1 == n) {
  for (j = 0; j < m; ++j) {
    double score = 0;
    int npairs = 0;
    { double fmin = 0, fmax = 0, frange;
      int seen = 0;
      /* the range spans the values present, so a gap widens nothing */
      for (i = 0; i < p; ++i) {
        const double v = model->entries[i].inputs[j];
        if (LIBXS_NOTNAN(v)) {
          if (0 == seen) { fmin = v; fmax = v; seen = 1; }
          else {
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
          }
        }
      }
      frange = fmax - fmin;
      if (frange <= 0) frange = 1.0;
      for (a = 0; a < nclasses; ++a) {
        for (b = a + 1; b < nclasses; ++b) {
          int ca_pool = 0, cb_pool = 0;
          double* va = (double*)LIBXS_PREDICT_MALLOC(
            (size_t)class_count[a] * sizeof(double), ca_pool);
          double* vb = (double*)LIBXS_PREDICT_MALLOC(
            (size_t)class_count[b] * sizeof(double), cb_pool);
          if (NULL != va && NULL != vb) {
            int na = 0, nb = 0;
            for (i = 0; i < p; ++i) {
              const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
              const double v = model->entries[i].inputs[j];
              /* an absent value separates nothing, so it joins neither side */
              if (LIBXS_NOTNAN(v)) {
                if (label == class_id[a]) va[na++] = v;
                else if (label == class_id[b]) vb[nb++] = v;
              }
            }
            if (0 < na && 0 < nb) {
              const int sd = libxs_setdiff(LIBXS_DATATYPE_F64,
                va, na, vb, nb, frange * 0.05);
              score += (double)sd / LIBXS_MAX(na, nb);
              ++npairs;
            }
          }
          LIBXS_PREDICT_FREE(vb, cb_pool);
          LIBXS_PREDICT_FREE(va, ca_pool);
        }
      }
    }
      scores[j] = (npairs > 0) ? score / npairs : 0.0;
    }
    memcpy(sorted_scores, scores, (size_t)m * sizeof(double));
    libxs_sort(sorted_scores, m, sizeof(double), libxs_cmp_f64, NULL);
    thr = sorted_scores[m / 2];
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      for (j = 0; j < m; ++j) {
        model->weights[j] = (scores[j] >= thr) ? scores[j] : 0.0;
      }
    }
  }
}


/**
 * Contiguous partition of count items over ntasks, balanced to within one.
 *
 * Ceiling division gives every task the same rounded-up share and leaves the
 * remainder to the last one, which is the task that then decides when everybody
 * is finished: at 100 trees over 16 tasks it hands fourteen tasks seven trees,
 * one task two, and the last none at all. Spreading the remainder over the
 * leading tasks instead costs one item of imbalance at most, keeps each task's
 * range contiguous, and leaves no task idle while another still works.
 */
LIBXS_API_INLINE void internal_libxs_predict_split(int count, int tid,
  int ntasks, int* begin, int* end)
{
  const int base = (0 < ntasks) ? (count / ntasks) : count;
  const int rem = (0 < ntasks) ? (count % ntasks) : 0;
  const int lo = tid * base + LIBXS_MIN(tid, rem);
  *begin = LIBXS_MIN(lo, count);
  *end = LIBXS_MIN(lo + base + ((tid < rem) ? 1 : 0), count);
}


#include "libxs_predict_rf.h"
#include "libxs_predict_hknn.h"


LIBXS_API_INLINE int internal_libxs_predict_grow(libxs_predict_t* model)
{
  int result = EXIT_SUCCESS;
  if (model->nentries >= model->capacity) {
    const int newcap = (0 < model->capacity) ? (model->capacity * 2) : 64;
    internal_libxs_predict_entry_t* ne = (internal_libxs_predict_entry_t*)realloc(
      model->entries, (size_t)newcap * sizeof(internal_libxs_predict_entry_t));
    if (NULL != ne) {
      memset(ne + model->capacity, 0,
        (size_t)(newcap - model->capacity) * sizeof(internal_libxs_predict_entry_t));
      model->entries = ne;
      model->capacity = newcap;
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


/**
 * Address of entry i in the arena, growing it first. Every pointer into the
 * arena is re-seated afterwards, because realloc is free to move the block and
 * the entries hold addresses into it rather than offsets.
 */
LIBXS_API_INLINE double* internal_libxs_predict_slot(
  libxs_predict_t* model, int index)
{
  const size_t stride = (size_t)model->ninputs + model->noutputs;
  double* result = NULL;
  if (0 <= index && index >= model->arena_capacity) {
    const int grown = (0 < model->arena_capacity)
      ? (model->arena_capacity * 2) : 64;
    /* doubling alone does not reach an index that was jumped to */
    const int newcap = (grown > index) ? grown : (index + 1);
    double* na = (double*)realloc(model->arena,
      (size_t)newcap * stride * sizeof(double));
    if (NULL != na) {
      int i;
      model->arena = na;
      model->arena_capacity = newcap;
      for (i = 0; i < model->nentries; ++i) {
        model->entries[i].inputs = na + (size_t)i * stride;
        model->entries[i].outputs = na + (size_t)i * stride + model->ninputs;
      }
    }
  }
  if (NULL != model->arena && index < model->arena_capacity) {
    result = model->arena + (size_t)index * stride;
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_ts_diff_order(
  const libxs_predict_t* model)
{
  const int s = model->nseries;
  const int n = model->nts;
  const int tgt = model->target;
  const size_t shape = (size_t)n;
  const size_t stride = (size_t)s;
  libxs_fprint_t fp;
  int result = 0;
  if (n >= 4 && EXIT_SUCCESS == libxs_fprint(&fp, LIBXS_DATATYPE_F64,
    model->ts_buf + tgt, 1, &shape, &stride,
    LIBXS_MIN(3, n - 1), 0, 1, 0))
  {
    const double decay = libxs_fprint_decay(&fp);
    if (decay == decay && decay < 1.0) {
      result = 1;
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_ts_assemble(
  const libxs_predict_t* model, int w, const double* lags,
  const double* aux, double* out)
{
  const int s = model->nseries;
  const int tgt = model->target;
  const int tf = (NULL != model->transforms) ? model->transforms[tgt] : 0;
  int si, i, k;
  for (si = 0; si < s; ++si) {
    const int t = (si == tgt) ? tf : 0;
    for (i = 0; i < w; ++i) {
      out[si * w + i] = internal_libxs_predict_fwd(t, lags[si * w + i]);
    }
  }
  for (k = 0; k < model->nderiv; ++k) {
    const double* tl = out + (size_t)tgt * w;
    out[s * w + k] = (w - 2 - k >= 0)
      ? (tl[w - 1 - k] - tl[w - 2 - k]) : 0;
  }
  for (i = 0; i < model->naux; ++i) {
    out[s * w + model->nderiv + i] = (NULL != aux) ? aux[i] : 0;
  }
}


LIBXS_API_INLINE int internal_libxs_predict_ts_window_cap(
  const libxs_predict_t* model, int wmax)
{
  const int s = model->nseries;
  const int n = model->nts;
  const int aux_deriv = model->naux + model->nderiv;
  int wcap = (wmax > 0) ? (wmax - aux_deriv) / (s > 0 ? s : 1) : 0;
  const int wlim = n / 4;
  if (wcap <= 0) wcap = wlim;
  else if (wcap > wlim) wcap = wlim;
  return wcap;
}


LIBXS_API_INLINE int internal_libxs_predict_ts_window_acf(
  const libxs_predict_t* model, int wcap)
{
  const int s = model->nseries;
  const int n = model->nts;
  const int tgt = model->target;
  int best_w = 0, result = 0;
  int acf_pool = 0;
  double* acf;
  int maxlag;
  if (4 <= wcap) {
    maxlag = wcap + 1;
    acf = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)(maxlag + n) * sizeof(double), acf_pool);
    if (NULL != acf) {
      double* cx = acf + maxlag;
      double mean = 0;
      int i;
      for (i = 0; i < n; ++i) mean += model->ts_buf[i * s + tgt];
      mean /= n;
      for (i = 0; i < n; ++i) cx[i] = model->ts_buf[i * s + tgt] - mean;
      internal_libxs_autocorr(cx, n, 1, acf, maxlag);
      if (0 < acf[0]) {
        const double thresh = 1.0 / 2.71828182845904523536;
        int lag, found = 0;
        for (lag = 1; 0 == found && lag < maxlag; ++lag) {
          if (acf[lag] / acf[0] < thresh) { best_w = lag; found = 1; }
        }
        if (0 == found) {
          double prev = acf[1] / acf[0];
          for (lag = 2; 0 == found && lag < maxlag; ++lag) {
            const double r = acf[lag] / acf[0];
            if (r > prev) { best_w = lag - 1; found = 1; }
            prev = r;
          }
        }
        if (0 == best_w) best_w = wcap;
      }
      LIBXS_PREDICT_FREE(acf, acf_pool);
    }
    if (best_w < 4) best_w = (wcap >= 4) ? 4 : wcap;
    else if (best_w > wcap) best_w = wcap;
    result = best_w;
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_ts_window_feat(
  const libxs_predict_t* model, int w, int t, double* raw, double* feat)
{
  const int s = model->nseries;
  int si, i;
  for (si = 0; si < s; ++si) {
    for (i = 0; i < w; ++i) {
      raw[si * w + i] = model->ts_buf[(t + i) * s + si];
    }
  }
  if (model->naux > 0 || model->nderiv > 0) {
    const double* aux = (NULL != model->aux_buf)
      ? model->aux_buf + (size_t)(t + w) * model->naux : NULL;
    internal_libxs_predict_ts_assemble(model, w, raw, aux, feat);
  }
  else {
    memcpy(feat, raw, (size_t)s * w * sizeof(double));
  }
}


LIBXS_API_INLINE double internal_libxs_predict_ts_window_score(
  const libxs_predict_t* model, int w)
{
  const int s = model->nseries;
  const int h = model->noutputs;
  const int tgt = model->target;
  const int nts = model->nts;
  const int m = s * w + model->nderiv + model->naux;
  const int nwin = nts - w - h + 1;
  double result = 1e30;
  /**
   * The proxy scores the first step only. Scoring the whole horizon instead
   * was available as a switch and is not kept: it moves the selected window
   * (14 to 20 on the monthly sunspot series) and recovers 0.1 of the 2.8
   * points that window selection costs at six steps
   * (Section 5, "What Automatic Selection Costs" in the paper), so the
   * objective was never what made the proxy prefer short windows. A switch
   * that changes the model and buys nothing is worse than no switch.
   */
  const int nsteps = 1;
  const int nval = (nwin / 5 > 256) ? 256 : (nwin / 5 > 0 ? nwin / 5 : 1);
  const int ntrain = nwin - nval;
  if (nwin >= 8 && 0 < m && ntrain >= 4) {
    int f_pool = 0, o_pool = 0, mn_pool = 0, rg_pool = 0, rw_pool = 0;
    double* feat = (double*)LIBXS_PREDICT_MALLOC((size_t)nwin * m * sizeof(double), f_pool);
    double* outs = (double*)LIBXS_PREDICT_MALLOC((size_t)nwin * h * sizeof(double), o_pool);
    double* mn = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), mn_pool);
    double* rg = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), rg_pool);
    double* raw = (double*)LIBXS_PREDICT_MALLOC((size_t)s * w * sizeof(double), rw_pool);
    if (NULL != feat && NULL != outs && NULL != mn && NULL != rg && NULL != raw) {
      const int kk = (LIBXS_PREDICT_KNN < ntrain) ? LIBXS_PREDICT_KNN : ntrain;
      double err_sum = 0;
      int t, i, j, vi;
      for (t = 0; t < nwin; ++t) {
        internal_libxs_predict_ts_window_feat(model, w, t, raw, feat + (size_t)t * m);
        for (j = 0; j < h; ++j) {
          const double o = model->ts_buf[(t + w + j) * s + tgt];
          outs[(size_t)t * h + j] = (NULL != model->transforms)
            ? internal_libxs_predict_fwd(model->transforms[j], o) : o;
        }
      }
      for (j = 0; j < m; ++j) { mn[j] = feat[j]; rg[j] = feat[j]; }
      for (i = 1; i < ntrain; ++i) {
        for (j = 0; j < m; ++j) {
          const double v = feat[(size_t)i * m + j];
          if (v < mn[j]) mn[j] = v;
          if (v > rg[j]) rg[j] = v;
        }
      }
      for (j = 0; j < m; ++j) rg[j] -= mn[j];
      for (t = 0; t < nwin; ++t) {
        double* f = feat + (size_t)t * m;
        for (j = 0; j < m; ++j) {
          f[j] = (rg[j] > 0) ? (f[j] - mn[j]) / rg[j] : f[j];
          if (NULL != model->weights) f[j] *= model->weights[j];
        }
      }
      for (vi = ntrain; vi < nwin; ++vi) {
        const double* q = feat + (size_t)vi * m;
        double dist[LIBXS_PREDICT_KNN];
        int idx[LIBXS_PREDICT_KNN];
        int nfound = 0, worst = 0, step;
        for (i = 0; i < ntrain; ++i) {
          const double d = libxs_dist2(q, feat + (size_t)i * m, m);
          if (nfound < kk) {
            dist[nfound] = d; idx[nfound] = i;
            if (d > dist[worst]) worst = nfound;
            ++nfound;
          }
          else if (d < dist[worst]) {
            int wi;
            dist[worst] = d; idx[worst] = i;
            worst = 0;
            for (wi = 1; wi < kk; ++wi) {
              if (dist[wi] > dist[worst]) worst = wi;
            }
          }
        }
        for (step = 0; step < nsteps; ++step) {
          double wsum = 0, vsum = 0, pred = 0;
          int exact = 0;
          for (i = 0; i < nfound; ++i) {
            const double ov = outs[(size_t)idx[i] * h + step];
            if (dist[i] <= 0) { pred = ov; exact = 1; break; }
            { const double wgt = 1.0 / sqrt(dist[i]);
              wsum += wgt; vsum += wgt * ov;
            }
          }
          if (0 == exact) pred = (wsum > 0) ? (vsum / wsum) : 0;
          err_sum += LIBXS_FABS(pred - outs[(size_t)vi * h + step]);
        }
      }
      result = err_sum / ((double)nval * nsteps);
    }
    LIBXS_PREDICT_FREE(raw, rw_pool);
    LIBXS_PREDICT_FREE(rg, rg_pool);
    LIBXS_PREDICT_FREE(mn, mn_pool);
    LIBXS_PREDICT_FREE(outs, o_pool);
    LIBXS_PREDICT_FREE(feat, f_pool);
  }
  return result;
}


/**
 * Score one window by building the model the caller will actually get.
 *
 * Each fold trains on a prefix of the pushed timesteps and is scored on the
 * tail that follows it, so no validation window is ever in the corpus that
 * predicts it. One contiguous tail is one stretch of the series and for a
 * long cycle that is not a sample of it - the bare sunspot model ranked four
 * lags above nine on its last two cycles and the opposite way on the held-out
 * years - so the cut point walks forward and later cuts weigh more, the model
 * being built to predict what follows the data it has. Interleaving the folds
 * instead would leave the exact window in the training set for the neighbor
 * search to find, which reads as near-perfect accuracy for every candidate.
 *
 * Returns the weighted mean absolute error over the horizon, or a large value
 * if the window cannot be scored.
 */
LIBXS_API_INLINE double internal_libxs_predict_ts_window_probe(
  const libxs_predict_t* model, int w, int nfold, int decompose)
{
  const int s = model->nseries;
  const int a = model->naux;
  const int h = model->noutputs;
  const int nts = model->nts;
  const int nwin = nts - w - h + 1;
  const int nval = (nwin / 5 > 256) ? 256 : ((nwin / 5 > 0) ? (nwin / 5) : 1);
  double result = 1e30;
  if (nwin >= 8) {
    double fold_err = 0, fold_wsum = 0;
    int f;
    for (f = 0; f < nfold; ++f) {
      const int cut = nts - (nfold - f) * nval;
      const double fw = (double)(f + 1);
      if (cut > w + h) {
        libxs_predict_t* probe = libxs_predict_create(s * w + a + model->nderiv, h);
        if (NULL != probe) {
          double* step = (double*)malloc((size_t)(s + a) * sizeof(double));
          double* x = (double*)malloc((size_t)(s * w + a + model->nderiv) * sizeof(double));
          double* y = (double*)malloc((size_t)h * sizeof(double));
          if (NULL != step && NULL != x && NULL != y) {
            double err = 0;
            int t, j, nsc = 0;
            probe->eval_mode = model->eval_mode;
            probe->decompose = decompose;
            probe->target = model->target;
            probe->diff_mode = model->diff_mode;
            probe->consistency = model->consistency;
            probe->smooth = model->smooth;
            probe->central = model->central;
            probe->nbank = model->nbank;
            probe->nderiv = model->nderiv;
            probe->naux = a;
            probe->nseries = s;
            probe->window = w;
            if (NULL != model->transforms) {
              int o;
              for (o = 0; o < h; ++o) {
                libxs_predict_set_transform(probe, o, model->transforms[o]);
              }
            }
            for (t = 0; t < cut; ++t) {
              for (j = 0; j < s; ++j) step[j] = model->ts_buf[(size_t)t * s + j];
              for (j = 0; j < a; ++j) {
                step[s + j] = model->aux_buf[(size_t)t * a + j];
              }
              libxs_predict_push(NULL, probe, step, NULL);
            }
            if (EXIT_SUCCESS == libxs_predict_build(probe, 0, 2, 0.0)) {
              libxs_predict_query_t qi;
              LIBXS_MEMZERO(&qi);
              libxs_predict_query(probe, &qi);
              for (t = cut; t + h <= cut + nval && t + h <= nts; ++t) {
                const int we = qi.window;
                if (t >= we) {
                  for (j = 0; j < we * s; ++j) {
                    x[j] = model->ts_buf[(size_t)(t - we) * s + j];
                  }
                  for (j = 0; j < a; ++j) {
                    x[we * s + j] = model->aux_buf[(size_t)(t - 1) * a + j];
                  }
                  libxs_predict_eval(NULL, probe, x, y, NULL, 1);
                  for (j = 0; j < h; ++j) {
                    const double d = y[j]
                      - model->ts_buf[(size_t)(t + j) * s + model->target];
                    err += (d < 0) ? -d : d;
                  }
                  ++nsc;
                }
              }
              if (0 < nsc) {
                const double fs = err / ((double)nsc * h);
                fold_err += fw * fs;
                fold_wsum += fw;
              }
            }
          }
          free(y); free(x); free(step);
          libxs_predict_destroy(probe);
        }
      }
    }
    if (0 < fold_wsum) result = fold_err / fold_wsum;
  }
  return result;
}


/**
 * Window selection by measurement rather than by proxy.
 *
 * The cheap path scores a flat kNN over the window features, which is biased
 * toward short windows: its error grows with the feature count for a reason the
 * built model does not share, since that one partitions and sizes its own
 * neighborhood. On the monthly sunspot series the proxy score rises
 * monotonically, so its minimum is always the shortest candidate and the choice
 * falls to a tolerance rather than to a measurement. This path removes the
 * proxy instead of correcting it, and needs no tolerance.
 *
 * It walks the same geometric grid and stops after two candidates fail to
 * improve: the measured error against window is unimodal only up to noise - it
 * wiggles by 0.07 around its minimum on one of the cases here - so two rises is
 * the weaker claim that does hold, where a bracketing search could descend into
 * the wiggle. A golden-section search was measured against this and rejected:
 * reaching unit precision over the same span costs about as many builds as the
 * grid has points, spent resolving differences smaller than the noise.
 * Bisection does not apply at all, locating a root rather than a minimum.
 *
 * Two ways to spend fewer builds were measured and only one kept. Searching
 * between the winner's neighbors added three candidates and changed the
 * selected window on none of the three series measured - the grid is already
 * finer than the curve near its minimum - so it is gone, and with it a third
 * of the cost. Abandoning a candidate that trails the incumbent on the
 * cheapest fold would save another quarter, but it selected a worse window on
 * one series of three: the fold that is cheapest to build is also the one
 * trained on the least data, and it ranked two windows in the opposite order
 * from the full set. A margin would only move the arbitrariness into the
 * margin, so all folds are scored for every candidate the walk reaches.
 *
 * The cost is one build per fold per candidate, roughly an order of magnitude
 * more build time than the proxy, which is why a negative window has to ask
 * for it.
 */
LIBXS_API_INLINE int internal_libxs_predict_ts_window_exact(
  const libxs_predict_t* model, int wcap, int guess)
{
  const char* fe = getenv("LIBXS_PREDICT_WINDOW_FOLDS");
  const int nfold = (NULL != fe) ? LIBXS_MAX(atoi(fe), 1) : 3;
  int grid[32], ngrid = 0, result = 0;
  double wf = 4.0;
  while ((int)(wf + 0.5) <= wcap && ngrid < 30) {
    const int cw = (int)(wf + 0.5);
    if (0 == ngrid || cw != grid[ngrid - 1]) grid[ngrid++] = cw;
    wf *= 1.5;
  }
  if (0 == ngrid || grid[ngrid - 1] != wcap) grid[ngrid++] = wcap;
  if (4 <= guess && guess <= wcap && ngrid < 32) {
    int k = 0;
    while (k < ngrid && grid[k] != guess) ++k;
    if (k == ngrid) grid[ngrid++] = guess;
  }
  { double best = 1e30;
    int i, nrise = 0;
    for (i = 0; i < ngrid && nrise < 2; ++i) {
      /**
       * The window is resolved before the mode is, so a model that asked for
       * both is searched one after the other rather than jointly: the joint
       * grid costs the product of the two and the window is the coarser knob.
       */
      const double score = internal_libxs_predict_ts_window_probe(
        model, grid[i], nfold, (0 > model->decompose)
          ? LIBXS_PREDICT_RAW : model->decompose);
      if (score < best) {
        best = score;
        result = grid[i];
        nrise = 0;
      }
      else if (score < 1e29) {
        ++nrise;
      }
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_ts_window(
  const libxs_predict_t* model, int wmax)
{
  const int wcap = internal_libxs_predict_ts_window_cap(model, wmax);
  int result = 0;
  if (4 <= wcap) {
    /**
     * The held-out kNN proxy is faithful only for low-dimensional
     * (single-series) feature spaces. For multi-series inputs the proxy
     * is floor-biased and the real pipeline (decomposition, per-channel
     * structure) typically needs a longer window; the library abstains
     * and returns the caller's budgeted upper bound (wcap) rather than
     * risk shrinking below the forecast-optimal window.
     */
    const int can_score = (model->nseries <= 1);
    if (0 != can_score) {
      const double eps = 0.12;
      int grid[32];
      double score[32];
      double wf = 4.0;
      int ngrid = 0, best_i = 0, i;
      score[0] = 1e30;
      while ((int)(wf + 0.5) <= wcap && ngrid < 31) {
        const int cw = (int)(wf + 0.5);
        if (0 == ngrid || cw != grid[ngrid - 1]) grid[ngrid++] = cw;
        wf *= 1.5;
      }
      if (0 == ngrid || grid[ngrid - 1] != wcap) grid[ngrid++] = wcap;
      for (i = 0; i < ngrid; ++i) {
        score[i] = internal_libxs_predict_ts_window_score(model, grid[i]);
        if (score[i] < score[best_i]) best_i = i;
      }
      if (score[best_i] < 1e29) {
        if (0 == best_i) {
          const double tol = score[0] * (1.0 + eps);
          int j = 0;
          while (j + 1 < ngrid && score[j + 1] <= tol) ++j;
          best_i = j;
        }
        if (4 <= grid[best_i]) result = grid[best_i];
      }
    }
    else {
      result = wcap;
    }
    if (result < 4) result = internal_libxs_predict_ts_window_acf(model, wcap);
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_ts_diff_apply(
  double* buf, int n, int s, int d)
{
  int dd, i, si;
  for (dd = 0; dd < d; ++dd) {
    for (i = 0; i < n - 1 - dd; ++i) {
      for (si = 0; si < s; ++si) {
        buf[i * s + si] = buf[(i + 1) * s + si] - buf[i * s + si];
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_ts_expand(libxs_predict_t* model)
{
  const int s = model->nseries;
  const int h = model->noutputs;
  int nts = model->nts, diff_d, w, m, nwindows;
  int raw_pool = 0, in_pool = 0, out_pool = 0;
  double* raw;
  double* inputs;
  double* outputs;
  int t;
  if (0 >= model->window) {
    const int guess = -model->window;
    const int wcap = internal_libxs_predict_ts_window_cap(model, model->ninputs);
    model->window = (0 == guess)
      ? internal_libxs_predict_ts_window(model, model->ninputs)
      : internal_libxs_predict_ts_window_exact(model, wcap, guess);
    if (0 >= model->window) return;
  }
  if (model->diff_mode > 0) {
    model->diff_order = model->diff_mode;
  }
  else if (0 == model->diff_mode) {
    model->diff_order = internal_libxs_predict_ts_diff_order(model);
  }
  diff_d = model->diff_order;
  if (diff_d > 0) {
    internal_libxs_predict_ts_diff_apply(model->ts_buf, nts, s, diff_d);
    nts -= diff_d;
    model->nts = nts;
  }
  w = model->window - diff_d;
  m = s * w + model->nderiv + model->naux;
  if (m != model->ninputs) {
    model->ninputs = m;
  }
  nwindows = nts - w - h + 1;
  raw = (double*)LIBXS_PREDICT_MALLOC((size_t)s * w * sizeof(double), raw_pool);
  /**
   * One pair of window buffers for the whole scan rather than a pair per
   * window: the values are copied into the entry arena, so nothing outlives an
   * iteration, and a long series would otherwise allocate twice per timestep.
   */
  inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), in_pool);
  outputs = (double*)LIBXS_PREDICT_MALLOC((size_t)h * sizeof(double), out_pool);
  if (NULL != raw && NULL != inputs && NULL != outputs && nwindows > 0) {
    for (t = 0; t < nwindows; ++t) {
      int si, i;
      for (si = 0; si < s; ++si) {
        for (i = 0; i < w; ++i) {
          raw[si * w + i] = model->ts_buf[(t + i) * s + si];
        }
      }
      if (model->naux > 0 || model->nderiv > 0) {
        const double* aux = (NULL != model->aux_buf)
          ? model->aux_buf + (size_t)(t + w) * model->naux : NULL;
        internal_libxs_predict_ts_assemble(model, w, raw, aux, inputs);
      }
      else if (LIBXS_PREDICT_RAW != model->decompose && s >= 2) {
        internal_libxs_predict_decompose_apply(model, raw, inputs);
      }
      else {
        memcpy(inputs, raw, (size_t)m * sizeof(double));
      }
      for (i = 0; i < h; ++i) {
        outputs[i] = model->ts_buf[(t + w + i) * s + model->target];
      }
      if (EXIT_SUCCESS == internal_libxs_predict_grow(model)) {
        double* slot = internal_libxs_predict_slot(model, model->nentries);
        internal_libxs_predict_entry_t* e = &model->entries[model->nentries];
        if (NULL != model->transforms) {
          int j;
          for (j = 0; j < h; ++j) {
            outputs[j] = internal_libxs_predict_fwd(model->transforms[j], outputs[j]);
          }
        }
        if (NULL != slot) {
          e->inputs = slot;
          e->outputs = slot + m;
          memcpy(e->inputs, inputs, (size_t)m * sizeof(double));
          memcpy(e->outputs, outputs, (size_t)h * sizeof(double));
          ++model->nentries;
        }
      }
    }
  }
  LIBXS_PREDICT_FREE(outputs, out_pool);
  LIBXS_PREDICT_FREE(inputs, in_pool);
  LIBXS_PREDICT_FREE(raw, raw_pool);
}


/**
 * The weight travels as an argument rather than as model state: push takes an
 * optional lock so that threads may push concurrently, and a field set around
 * the call would be written outside it.
 */
LIBXS_API_INLINE int internal_libxs_predict_push_impl(
  libxs_lock_t* lock, libxs_predict_t* model, const double inputs[],
  const double outputs[], double weight)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || NULL == inputs) {
    result = EXIT_FAILURE;
  }
  else if (NULL == outputs && 0 < model->nseries) {
    const int s = model->nseries;
    const int a = model->naux;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (model->nts >= model->ts_capacity) {
      const int newcap = (0 < model->ts_capacity) ? (model->ts_capacity * 2) : 256;
      double* nb = (double*)realloc(model->ts_buf, (size_t)newcap * (size_t)s * sizeof(double));
      double* na = (a > 0)
        ? (double*)realloc(model->aux_buf, (size_t)newcap * (size_t)a * sizeof(double))
        : model->aux_buf;
      if (NULL != nb && (a <= 0 || NULL != na)) {
        model->ts_buf = nb;
        model->aux_buf = na;
        model->ts_capacity = newcap;
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      memcpy(model->ts_buf + (size_t)model->nts * s, inputs, (size_t)s * sizeof(double));
      if (a > 0) {
        memcpy(model->aux_buf + (size_t)model->nts * a, inputs + s,
          (size_t)a * sizeof(double));
      }
      ++model->nts;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  else if (NULL == outputs) {
    result = EXIT_FAILURE;
  }
  else {
    const int m = model->ninputs, n = model->noutputs;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    result = internal_libxs_predict_grow(model);
    if (EXIT_SUCCESS == result) {
      double* slot = internal_libxs_predict_slot(model, model->nentries);
      internal_libxs_predict_entry_t* e = &model->entries[model->nentries];
      e->inputs = slot;
      e->outputs = (NULL != slot) ? (slot + m) : NULL;
      if (NULL != e->inputs && NULL != e->outputs) {
        e->weight = weight;
        if (1.0 != weight) model->has_eweight = 1;
        memcpy(e->inputs, inputs, (size_t)m * sizeof(double));
        if (NULL != model->transforms) {
          int j;
          for (j = 0; j < n; ++j) {
            e->outputs[j] = internal_libxs_predict_fwd(model->transforms[j], outputs[j]);
          }
        }
        else {
          memcpy(e->outputs, outputs, (size_t)n * sizeof(double));
        }
        ++model->nentries;
      }
      else {
        e->inputs = NULL;
        e->outputs = NULL;
        result = EXIT_FAILURE;
      }
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  return result;
}


LIBXS_API int libxs_predict_push(
  libxs_lock_t* lock, libxs_predict_t* model, const double inputs[],
  const double outputs[])
{
  return internal_libxs_predict_push_impl(lock, model, inputs, outputs, 1.0);
}


LIBXS_API int libxs_predict_push_weighted(
  libxs_lock_t* lock, libxs_predict_t* model, const double inputs[],
  const double outputs[], double weight)
{
  int result = EXIT_FAILURE;
  if (NULL != model && 0 < weight && weight == weight) {
    /**
     * A weighted timestep has nothing to attach to: series mode turns the
     * pushed timesteps into overlapping windows at build, so one timestep
     * contributes to many entries and no entry corresponds to one push.
     */
    if (1.0 == weight || 0 >= model->nseries) {
      result = internal_libxs_predict_push_impl(lock, model, inputs, outputs,
        weight);
    }
  }
  return result;
}


/* whether any cluster fits any output with a polynomial */
LIBXS_API_INLINE int internal_libxs_predict_interpolates(
  const libxs_predict_t* model)
{
  int result = 0, c;
  for (c = 0; c < model->nclusters && 0 == result; ++c) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    if (NULL != cl->interpolated) {
      int j;
      for (j = 0; j < model->noutputs && 0 == result; ++j) {
        if (0 != cl->interpolated[j]) result = 1;
      }
    }
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_order_fn(
  double x, const void* data)
{
  const internal_libxs_predict_order_ctx_t* ctx =
    (const internal_libxs_predict_order_ctx_t*)data;
  const int ord = LIBXS_MAX(LIBXS_ROUNDX(int, x), 1);
  double total_err = 1e30;
  if (EXIT_SUCCESS == internal_libxs_predict_build_impl(ctx->model,
    ctx->nclusters, ord, 0, ctx->tid, ctx->ntasks))
  {
    const int p = ctx->model->nentries;
    const int n = ctx->model->noutputs;
    const int saved_decompose = ctx->model->decompose;
    int i, j;
    ctx->model->decompose = LIBXS_PREDICT_RAW;
    total_err = 0;
    for (i = 0; i < p; ++i) {
      double outputs[128];
      libxs_predict_eval(NULL, ctx->model,
        ctx->model->entries[i].inputs, outputs, NULL, 1);
      for (j = 0; j < n; ++j) {
        total_err += LIBXS_DELTA(outputs[j], ctx->model->entries[i].outputs[j]);
      }
    }
    ctx->model->decompose = saved_decompose;
  }
  return total_err;
}


#include "libxs_predict_select.h"
#include "libxs_predict_compress.h"


/**
 * collective: non-zero when the caller entered through libxs_predict_build_task
 * and the forest is therefore built by the tasks afterwards rather than here.
 * It travels with the call, including into the order search's rebuilds, so no
 * part of the build has to consult state left behind by an earlier one.
 */
LIBXS_API_INLINE int internal_libxs_predict_build_impl(libxs_predict_t* model,
  int nclusters, int order, double quality, int tid, int ntasks)
{
  int result = EXIT_SUCCESS;
  if (NULL != model) {
    const char* tenv = getenv("LIBXS_PREDICT_TANGENT");
    const char* renv = getenv("LIBXS_PREDICT_REFINE");
    if (NULL != tenv) model->tangent = atoi(tenv);
    if (NULL != renv) model->refine = atoi(renv);
  }
  /**
   * Everything from here to the partition is the builder's. It rewrites the
   * corpus - an expanded series, a rotation - and resolves what every later
   * stage reads: the mode, the absences, the feature weights. Each test below
   * also reads exactly what its own stage writes, so none of them may carry a
   * rendezvous; the single one after the region can, and the verdict is
   * published because a task that ran none of this cannot form its own.
   */
  if (0 == tid) {
    if (NULL != model && 0 < model->nts && 0 == model->nentries) {
      internal_libxs_predict_ts_expand(model);
    }
    if (NULL != model && 0 < model->nentries) {
      internal_libxs_predict_missing_all(model);
      /**
       * Absent inputs are known by now and the window, if it was requested by
       * sentinel, is resolved, so a mode that cannot apply can be ruled out
       * before any of them is built.
       */
      if (0 > model->decompose) {
        const char* fenv = getenv("LIBXS_PREDICT_DECOMPOSE_FOLDS");
        model->decompose = internal_libxs_predict_decompose_select(model,
          (NULL != fenv) ? atoi(fenv) : 0);
      }
      /**
       * A tree reads one coordinate per node and has nowhere to record which way
       * an absent one should go, so it would sort NaN into an arbitrary side and
       * build a tree on it. Imputing instead would work at build and not survive
       * a round trip, because the medians are not recoverable from what is
       * serialized. A rotation is refused for a different reason, given at
       * internal_libxs_predict_gaps_ok. Refusing is the only option here that
       * cannot mislead: the alternative is a model that silently answers from a
       * coordinate it never had.
       */
      if (0 != model->has_missing
        && 0 == internal_libxs_predict_gaps_ok(model->decompose))
      {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result
      && NULL != model && 0 < model->nentries && NULL == model->decompose_mat
      && (LIBXS_PREDICT_PCA == model->decompose
        || (LIBXS_PREDICT_SPREAD == model->decompose && model->nseries >= 2)))
    {
      internal_libxs_predict_pca_build(model);
    }
    /**
     * The scan above ran before the decomposition rewrote the entries, so it
     * cannot speak for what they hold now. A mode that reached this point is one
     * that carries gaps, and a gap it was given survives as a gap; a NaN that
     * appears here instead was manufactured, and the distance must be told either
     * way rather than treating it as a coordinate.
     */
    if (EXIT_SUCCESS == result && NULL != model && 0 < model->nentries) {
      internal_libxs_predict_missing_all(model);
    }
    if (NULL != model && 0 < model->nentries && NULL == model->weights) {
      if (LIBXS_PREDICT_SETDIFF == model->decompose) {
        internal_libxs_predict_setdiff_build(model);
      }
      else if (LIBXS_PREDICT_FISHER == model->decompose) {
        internal_libxs_predict_fisher_build(model);
      }
    }
    if (NULL != model) model->sync_result = result;
  } /* end of the builder's preparation of the corpus */
  if (NULL != model) {
    internal_libxs_predict_sync(model, ntasks);
    result = (int)model->sync_result;
  }
  if (EXIT_SUCCESS == result && NULL != model && 0 < model->nentries
    && LIBXS_PREDICT_HKNN == model->decompose
    && NULL == model->hknn_assignments)
  {
    /**
     * Deriving the hierarchy is the builder's: every task would otherwise
     * derive its own over the same corpus and race on the result
     */
    if (0 == tid) {
      model->hknn_assignments = (int*)calloc((size_t)model->nentries, sizeof(int));
      if (NULL != model->hknn_assignments) {
        model->hknn_nclusters = 0;
        internal_libxs_predict_hknn_partition(model, &model->hknn_nclusters);
        if (model->hknn_nclusters < 1) model->hknn_nclusters = 1;
      }
    }
  }
  /**
   * Outside the test, not inside: the test reads hknn_assignments and the
   * builder fills it, so a task arriving late reads it as done, skips the
   * stage, and leaves the builder waiting for an arrival that never comes.
   * A rendezvous may only be conditional on state no stage of it writes.
   */
  internal_libxs_predict_sync(model, ntasks);
  if (EXIT_SUCCESS == result && NULL != model && 0 < model->nentries
    && LIBXS_PREDICT_RF == model->decompose && NULL == model->rf)
  {
    if (0 == tid) {
      internal_libxs_predict_rf_build(model);
      if (1 >= ntasks) {
        internal_libxs_predict_rf_build_tasks(model, 0, 1);
        internal_libxs_predict_rf_boost(model);
      }
    }
  }
  /* as above: the test reads model->rf, which the stage itself sets */
  internal_libxs_predict_sync(model, ntasks);
  if (EXIT_SUCCESS != result || NULL == model || 0 >= model->nentries) {
    result = EXIT_FAILURE;
  }
  else if (order <= 0) {
    internal_libxs_predict_order_ctx_t ctx;
    const int max_ord = (order < 0) ? -order : LIBXS_FPRINT_MAXORDER;
    int best_ord = 1, ord;
    double best_err = 1e30;
    /**
     * The search is the builder's, and each candidate it scores is a serial
     * build: a collective one here would nest rendezvous inside the rendezvous
     * this stage already is. The order it settles on is published, so the build
     * that keeps it is the collective one and the tasks are idle only for the
     * search itself.
     */
    ctx.model = model;
    ctx.nclusters = nclusters;
    ctx.tid = 0;
    ctx.ntasks = 1;
    if (0 != tid) {
      internal_libxs_predict_sync(model, ntasks);
      best_ord = (int)model->sync_result;
    }
    else {
    ord = 1;
    best_err = internal_libxs_predict_order_fn((double)ord, &ctx);
    /**
     * The order is the degree of the polynomial an interpolate-mode output is
     * fitted with, so a corpus where every output classifies cannot be told
     * apart by it: the remaining candidates would rebuild the model and score
     * an identical answer. Trying order 1 first makes that decidable after one
     * build rather than eight, which is the difference between one build and
     * nine on any corpus carrying a discrete label - and the search selected
     * order 1 there anyway, so nothing is given up.
     */
    if (0 != internal_libxs_predict_interpolates(model)) {
      for (ord = 2; ord <= max_ord; ++ord) {
        const double err = internal_libxs_predict_order_fn((double)ord, &ctx);
        if (err < best_err) { best_err = err; best_ord = ord; }
      }
      ord = max_ord;
    }
    model->iterations = ord;
    model->sync_result = best_ord;
    internal_libxs_predict_sync(model, ntasks);
    }
    result = internal_libxs_predict_build_impl(model, nclusters, best_ord,
      quality, tid, ntasks);
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    int c, i;
    /* entries bucketed by cluster, so collecting a cluster's members is a
       lookup rather than a scan over all entries (that scan made build
       O(p*nclusters), hence superlinear at the default nclusters = sqrt(p)) */
    int pool_bucket = 0, pool_cbegin = 0;
    int* bucket = NULL;
    int* cbegin = NULL;
    /**
     * Laying out the corpus is the builder's: it allocates what the other
     * tasks then read. They wait at the rendezvous below and take its verdict
     * rather than their own, which is why it is published rather than returned.
     */
    if (0 == tid) {
      if (order > LIBXS_FPRINT_MAXORDER) order = LIBXS_FPRINT_MAXORDER;
      model->order = order;
      model->quality = quality;
      internal_libxs_predict_free_clusters(model);
      free(model->input_min); free(model->input_rng);
      free(model->input_knot); model->input_knot = NULL;
      /* the normalization these define is about to change, so what was
         normalized under the previous one cannot be carried over */
      free(model->norm_pts); model->norm_pts = NULL;
      model->input_min = (double*)malloc((size_t)m * sizeof(double));
      model->input_rng = (double*)malloc((size_t)m * sizeof(double));
      if (NULL != model->input_min && NULL != model->input_rng) {
        int j;
        /**
         * An absent value must not seed the extent: it compares false against
         * everything, so a NaN in the first entry would leave the whole dimension
         * NaN and silently un-normalize every later comparison on it. A
         * dimension that is absent throughout keeps a zero range, which
         * internal_libxs_predict_normalize already treats as "do not scale".
         */
        for (j = 0; j < m; ++j) {
          model->input_min[j] = 0;
          model->input_rng[j] = 0;
        }
        for (j = 0; j < m; ++j) {
          int seeded = 0;
          for (i = 0; i < p; ++i) {
            const double v = model->entries[i].inputs[j];
            if (LIBXS_NOTNAN(v)) {
              if (0 == seeded) {
                model->input_min[j] = v;
                model->input_rng[j] = v;
                seeded = 1;
              }
              else {
                if (v < model->input_min[j]) model->input_min[j] = v;
                if (v > model->input_rng[j]) model->input_rng[j] = v;
              }
            }
          }
        }
        for (j = 0; j < m; ++j) {
          model->input_rng[j] -= model->input_min[j];
        }
        /**
         * The rank coordinate is for axes that measure different quantities. A
         * window feeds lags of one series, which share a scale by construction,
         * and ranking each lag on its own distribution discards exactly the level
         * relationship that makes two windows comparable.
         */
        if (0 >= model->nseries) internal_libxs_predict_fit_knots(model);
        else {
          free(model->input_knot);
          model->input_knot = NULL;
        }
      }
      if (0 >= nclusters) {
        nclusters = (int)(sqrt((double)p) + 0.5);
        if (nclusters < 1) nclusters = 1;
      }
      if (nclusters > p) nclusters = p;
      model->assignments = (int*)calloc((size_t)p, sizeof(int));
      model->eval_buf = (double*)malloc((size_t)n * 6 * sizeof(double) + (size_t)n * sizeof(int));
      if (NULL == model->assignments || NULL == model->eval_buf) {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result && LIBXS_PREDICT_HKNN == model->decompose
        && NULL != model->hknn_assignments && model->hknn_nclusters > 0)
      {
        memcpy(model->assignments, model->hknn_assignments,
          (size_t)p * sizeof(int));
        nclusters = model->hknn_nclusters;
      }
      model->clusters = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nclusters, sizeof(internal_libxs_predict_cluster_t));
      if (NULL == model->clusters) {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        model->nclusters = nclusters;
        for (c = 0; c < nclusters && EXIT_SUCCESS == result; ++c) {
          model->clusters[c].centroid = (double*)malloc((size_t)m * sizeof(double));
          if (NULL == model->clusters[c].centroid) result = EXIT_FAILURE;
        }
      }
      model->sync_result = result;
    } /* end of the builder's layout */
    internal_libxs_predict_sync(model, ntasks);
    result = (int)model->sync_result;
    if (EXIT_SUCCESS == result) {
      /* the count the builder settled on, which the local one need not match */
      const int pnc = model->nclusters;
      if (LIBXS_PREDICT_HKNN == model->decompose) {
        if (0 == tid) internal_libxs_predict_hknn_centroids(model, pnc);
        internal_libxs_predict_sync(model, ntasks);
        internal_libxs_predict_hknn_refine(model, pnc, tid, ntasks);
      }
      else {
        internal_libxs_predict_kmeans(model, pnc, tid, ntasks);
      }
    }
    internal_libxs_predict_sync(model, ntasks);
    if (0 == tid && EXIT_SUCCESS == result) {
      for (i = 0; i < p; ++i) {
        ++model->clusters[model->assignments[i]].nentries;
      }
      { int dst = 0, has_empty = 0;
        for (c = 0; c < nclusters; ++c) {
          if (0 >= model->clusters[c].nentries) { has_empty = 1; break; }
        }
        if (0 != has_empty) {
          /* gaps preceding each cluster, so renumbering stays linear in p */
          int pool_gap = 0;
          int* gaps = (int*)LIBXS_PREDICT_MALLOC(
            (size_t)nclusters * sizeof(int), pool_gap);
          if (NULL == gaps) result = EXIT_FAILURE;
          else {
            int gap = 0;
            for (c = 0; c < nclusters; ++c) {
              gaps[c] = gap;
              if (0 >= model->clusters[c].nentries) ++gap;
            }
            for (i = 0; i < p; ++i) {
              model->assignments[i] -= gaps[model->assignments[i]];
            }
            LIBXS_PREDICT_FREE(gaps, pool_gap);
          }
          for (c = 0; c < nclusters; ++c) {
            if (model->clusters[c].nentries > 0) {
              if (dst != c) {
                model->clusters[dst] = model->clusters[c];
                memset(&model->clusters[c], 0,
                  sizeof(internal_libxs_predict_cluster_t));
              }
              ++dst;
            }
            else {
              free(model->clusters[c].centroid);
              memset(&model->clusters[c], 0,
                sizeof(internal_libxs_predict_cluster_t));
            }
          }
          nclusters = dst;
          model->nclusters = nclusters;
        }
      }
      if (EXIT_SUCCESS == result) {
        bucket = (int*)LIBXS_PREDICT_MALLOC(
          (size_t)p * sizeof(int), pool_bucket);
        cbegin = (int*)LIBXS_PREDICT_MALLOC(
          (size_t)(nclusters + 1) * sizeof(int), pool_cbegin);
        if (NULL == bucket || NULL == cbegin) result = EXIT_FAILURE;
        else { /* counting sort keeps each cluster in ascending entry order */
          int at = 0;
          for (c = 0; c < nclusters; ++c) {
            cbegin[c] = at;
            at += model->clusters[c].nentries;
          }
          cbegin[nclusters] = at;
          /* cbegin doubles as the fill cursor, then the starts are restored */
          for (i = 0; i < p; ++i) {
            bucket[cbegin[model->assignments[i]]++] = i;
          }
          for (c = 0; c < nclusters; ++c) {
            cbegin[c] -= model->clusters[c].nentries;
          }
        }
      }
      for (c = 0; c < nclusters && EXIT_SUCCESS == result; ++c) {
        internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        const int nc = cl->nentries;
        int j, k, maxorder;
        if (0 >= nc) continue;
        cl->sorted_idx = (int*)malloc((size_t)nc * sizeof(int));
        cl->sorted_dist = (double*)malloc((size_t)nc * sizeof(double));
        cl->order = (int*)malloc((size_t)n * sizeof(int));
        cl->interpolated = (int*)malloc((size_t)n * sizeof(int));
        cl->mode = (int*)malloc((size_t)n * sizeof(int));
        cl->ndistinct = (int*)malloc((size_t)n * sizeof(int));
        if (NULL == cl->sorted_idx || NULL == cl->sorted_dist
          || NULL == cl->order || NULL == cl->interpolated
          || NULL == cl->mode || NULL == cl->ndistinct)
        {
          result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result) {
          int pool_inmat = 0, pool_perm = 0;
          double *const inmat = (double*)LIBXS_PREDICT_MALLOC((size_t)nc * (size_t)m * sizeof(double), pool_inmat);
          int *const sort_perm = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), pool_perm);
          const int *const entry_map = bucket + cbegin[c];
          if (NULL == inmat || NULL == sort_perm) {
            result = EXIT_FAILURE;
          }
          else {
            int ki;
            for (ki = 0; ki < nc; ++ki) {
              const double *const src = model->entries[entry_map[ki]].inputs;
              for (k = 0; k < m; ++k) {
                inmat[(size_t)k * nc + ki] = src[k];
              }
            }
            libxs_sort_smooth(LIBXS_SORT_HILBERT, nc, m, inmat, nc,
              LIBXS_DATATYPE_F64, sort_perm);
            for (k = 0; k < nc; ++k) {
              cl->sorted_idx[k] = entry_map[sort_perm[k]];
              cl->sorted_dist[k] = sqrt(internal_libxs_predict_dist2(
                model->entries[cl->sorted_idx[k]].inputs,
                cl->centroid, m, model->has_missing));
            }
            cl->dmax = 0;
            for (k = 0; k < nc; ++k) {
              if (cl->sorted_dist[k] > cl->dmax) cl->dmax = cl->sorted_dist[k];
            }
            if (cl->dmax <= 0.0) cl->dmax = 1.0;
            cl->kd_pts = (double*)malloc((size_t)nc * (size_t)m * sizeof(double));
            if (0 != model->has_eweight && NULL == cl->eweight) {
              cl->eweight = (double*)malloc((size_t)nc * sizeof(double));
            }
            if (NULL != cl->kd_pts) {
              for (k = 0; k < nc; ++k) {
                internal_libxs_predict_normalize(model,
                  model->entries[cl->sorted_idx[k]].inputs,
                  cl->kd_pts + (size_t)k * m);
                if (NULL != cl->eweight) {
                  cl->eweight[k] = model->entries[cl->sorted_idx[k]].weight;
                }
              }
              if (0 != model->tangent) {
                internal_libxs_predict_cluster_tangent(cl, m, model->tangent);
              }
            }
          }
          LIBXS_PREDICT_FREE(sort_perm, pool_perm);
          LIBXS_PREDICT_FREE(inmat, pool_inmat);
        }
        if (EXIT_SUCCESS == result) {
          maxorder = LIBXS_MIN(nc - 1, order);
          maxorder = LIBXS_MIN(maxorder, LIBXS_FPRINT_MAXORDER);
          if (maxorder < 1) maxorder = 1;
          cl->maxorder = maxorder;
          cl->coeffs = (double*)calloc((size_t)n * (size_t)(maxorder + 1), sizeof(double));
          cl->errors = (double*)calloc((size_t)n, sizeof(double));
          cl->out_rms = (double*)calloc((size_t)n, sizeof(double));
          cl->raw_outputs = (double*)malloc((size_t)nc * (size_t)n * sizeof(double));
          cl->out_mean = (double*)calloc((size_t)n, sizeof(double));
          cl->out_var = (double*)calloc((size_t)n, sizeof(double));
          if (NULL == cl->coeffs || NULL == cl->errors || NULL == cl->out_rms
            || NULL == cl->raw_outputs
            || NULL == cl->out_mean || NULL == cl->out_var)
          {
            result = EXIT_FAILURE;
          }
          else {
            for (k = 0; k < nc; ++k) {
              for (j = 0; j < n; ++j) {
                cl->raw_outputs[(size_t)k * n + j] =
                  model->entries[cl->sorted_idx[k]].outputs[j];
                cl->out_mean[j] += model->entries[cl->sorted_idx[k]].outputs[j];
              }
            }
            for (j = 0; j < n; ++j) cl->out_mean[j] /= nc;
            for (k = 0; k < nc; ++k) {
              for (j = 0; j < n; ++j) {
                const double d = cl->raw_outputs[(size_t)k * n + j] - cl->out_mean[j];
                cl->out_var[j] += d * d;
              }
            }
            for (j = 0; j < n; ++j) cl->out_var[j] /= nc;
          }
        }
        if (EXIT_SUCCESS == result) {
          internal_libxs_predict_cluster_refit(cl, n, 1);
        }
        if (EXIT_SUCCESS == result && nc > 2 && NULL != cl->out_rms
          && model->quantile > 0) {
          for (j = 0; j < n; ++j) {
            double sse = 0;
            for (k = 0; k < nc; ++k) {
              const double actual = cl->raw_outputs[(size_t)k * n + j];
              const double pred = internal_libxs_predict_classify(
                cl, m, cl->kd_pts + (size_t)k * m,
                j, n, cl->ndistinct[j], 0, k, NULL, NULL, 0, NULL,
                model->has_missing);
              const double res = pred - actual;
              sse += res * res;
            }
            cl->out_rms[j] = sqrt(sse / nc);
          }
        }
      }
      LIBXS_PREDICT_FREE(cbegin, pool_cbegin);
      LIBXS_PREDICT_FREE(bucket, pool_bucket);
      if (EXIT_SUCCESS == result) {
        model->built = 1;
        ++model->nbuild;
        /**
         * The probability support is built here rather than on first use: a lazy
         * cache would make the first scoring call in every stream a write to
         * shared state, which is exactly what the context exists to avoid.
         */
        internal_libxs_predict_support_all(model);
        internal_libxs_predict_keff_all(model);
        /**
         * The trial runs once and its result is kept: an order search rebuilds
         * this model many times, and re-resolving the count on every pass would
         * pay for it again without asking a different question.
         */
        if (0 != model->kreq && NULL == model->k_sel) {
          if (0 > model->kreq) {
            internal_libxs_predict_neighbors_select(model);
          }
          else {
            /**
             * A pinned count is resolved here too, rather than applied on the
             * way past, so that it reaches the file: a loaded model derives its
             * own counts and would otherwise quietly discard the caller's.
             */
            model->k_sel = (int*)malloc((size_t)n * sizeof(int));
            if (NULL != model->k_sel) {
              int kj;
              for (kj = 0; kj < n; ++kj) model->k_sel[kj] = model->kreq;
            }
          }
        }
        internal_libxs_predict_kapply(model);
        if (0 >= model->central) internal_libxs_predict_central_all(model);
        internal_libxs_predict_bank_all(model);
        if (model->smooth < 0) {
          int nsmooth = 0, ntotal_modes = 0, j;
          for (c = 0; c < nclusters; ++c) {
            const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
            if (NULL != cl->mode) {
              for (j = 0; j < n; ++j) {
                if (0 == cl->mode[j]) ++nsmooth;
                ++ntotal_modes;
              }
            }
          }
          model->smooth = (ntotal_modes > 0)
            ? 0.5 * (double)nsmooth / ntotal_modes : 0.0;
        }
        if (LIBXS_PREDICT_HKNN == model->decompose && n > 1
          && NULL == model->hknn_po_clusters)
        {
          internal_libxs_predict_hknn_build_po(model);
        }
        if (quality > 0 && NULL == model->rf
          && NULL != model->entries && NULL != model->assignments)
        {
          internal_libxs_predict_compress(model, order, quality);
        }
      }
      else {
        internal_libxs_predict_free_clusters(model);
      }
    } /* end of the builder's assembly: only the partition above is shared */
  }
  return result;
}


LIBXS_API int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order, double quality)
{
  return internal_libxs_predict_build_impl(model, nclusters, order, quality, 0, 1);
}


LIBXS_API int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  double quality, int tid, int ntasks)
{
  int result = EXIT_SUCCESS;
  LIBXS_ASSERT(NULL != model);
  /**
   * The build is a sequence of collective stages rather than one serial block
   * with a parallel tail. A stage is either the builder's alone or split
   * across the tasks, and every stage ends at the same rendezvous, so adding
   * one costs a barrier rather than another meaning for a shared word.
   */
  if (0 == tid) { /* the corpus has to exist before a candidate can be scored */
    if (0 < model->nts && 0 == model->nentries) {
      internal_libxs_predict_ts_expand(model);
    }
    if (0 < model->nentries) internal_libxs_predict_missing_all(model);
  }
  internal_libxs_predict_sync(model, ntasks);
  if (0 > model->decompose && 0 < model->nentries) {
    const char* fenv = getenv("LIBXS_PREDICT_DECOMPOSE_FOLDS");
    internal_libxs_predict_decompose_score(model,
      (NULL != fenv) ? atoi(fenv) : 0, tid, ntasks, model->sync_score);
    internal_libxs_predict_sync(model, ntasks);
    if (0 == tid) {
      model->decompose =
        internal_libxs_predict_decompose_reduce(model, model->sync_score);
    }
    internal_libxs_predict_sync(model, ntasks);
  }
  /**
   * Every task enters, because the partition inside is split across them. The
   * stages that are the builder's are guarded there rather than here, and each
   * branch is taken on shared state - the corpus, the mode, the order - so the
   * tasks cannot part company at a rendezvous. Only tid varies.
   */
  result = internal_libxs_predict_build_impl(model, nclusters, order,
    quality, tid, ntasks);
  internal_libxs_predict_sync(model, ntasks);
  if (0 != tid) result = (0 != model->built) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != model->rf) {
    internal_libxs_predict_rf_build_tasks(model, tid, ntasks);
    /**
     * The stages are sequential by construction - each fits what the previous
     * one left - so this is the builder's alone, and it needs every tree to
     * exist before the first residual can be taken.
     */
    internal_libxs_predict_sync(model, ntasks);
    if (0 == tid) internal_libxs_predict_rf_boost(model);
    internal_libxs_predict_sync(model, ntasks);
  }
  LIBXS_UNUSED(lock);
  return result;
}


/**
 * Evaluation with optional reporting of which evidence served each output.
 * Deciding that is what most of this function does - flat versus per-output
 * hKNN cluster, classify versus interpolate, blended or not - and a scoring
 * rule that needs the same decision must not re-derive it. src[j] receives the
 * cluster the value came from and src_mode[j] whether it was a kNN vote, so a
 * probability can read the identical evidence this dispatch selected. Both are
 * optional; passing NULL yields exactly the public eval behavior.
 */
LIBXS_API_INLINE void internal_libxs_predict_eval_ex(libxs_lock_t* lock,
  const libxs_predict_t* model, const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend,
  const internal_libxs_predict_cluster_t** src, int* src_mode,
  int* src_out, int* src_nout, const internal_libxs_predict_view_t* view)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != inputs);
  {
    const int m = model->ninputs, n = model->noutputs;
    const int mode = model->eval_mode;
    const int diff_d = (model->diff_mode >= 0) ? model->diff_order : 0;
    /**
     * A query missing a coordinate takes the kNN vote even where the output
     * interpolates. The polynomial is fitted over rank rather than over the
     * coordinates, so it would return a finite value - but it reports confidence
     * 1.0 unconditionally, which would advertise a query that supplied less
     * information as the most certain kind there is. The vote reads the same
     * neighbours through a distance that omits the absent coordinate and reports
     * what the neighbourhood actually supports, so it is the honest path here
     * and overrides a caller's INTERPOLATE for that reason alone.
     */
    const int incomplete = (0 != model->has_missing)
      && (0 != internal_libxs_predict_incomplete(inputs, model->ninputs));
    const int force_classify =
      (0 != incomplete || 0 != (mode & LIBXS_PREDICT_CLASSIFY)) ? 1 : 0;
    const int force_interp = (0 == incomplete
      && 0 != (mode & LIBXS_PREDICT_INTERPOLATE)) ? 1 : 0;
    const int extrapolate_mode = (0 != (mode & LIBXS_PREDICT_TEMPORAL)) ? 1 : 0;
    const double* raw_inputs = inputs;
    int extrapolate = 0;
    int norm_pool = 0, decomp_pool = 0, diff_pool = 0;
    double* decomp_inputs = NULL;
    double* diff_inputs = NULL;
    double* norm_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), norm_pool);
    double local_buf[256];
    double *vals, *errs, *conf, *var, *lo, *hi, best_dist;
    int *rels, c, j, best_c = 0;
    if (NULL != src || NULL != src_mode || NULL != src_out
      || NULL != src_nout)
    {
      for (j = 0; j < n; ++j) {
        if (NULL != src) src[j] = NULL;
        if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_NONE;
        if (NULL != src_out) src_out[j] = j;
        if (NULL != src_nout) src_nout[j] = n;
      }
    }
    if ((model->naux > 0 || model->nderiv > 0) && model->nseries > 0) {
      const int w = model->window;
      const double* aux = (model->naux > 0)
        ? inputs + (size_t)model->nseries * w : NULL;
      diff_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), diff_pool);
      if (NULL != diff_inputs) {
        internal_libxs_predict_ts_assemble(model, w, inputs, aux, diff_inputs);
        inputs = diff_inputs;
      }
    }
    else if (diff_d > 0 && model->nseries > 0) {
      const int raw_w = model->window;
      const int s = model->nseries;
      const int raw_m = s * raw_w;
      const int dw = raw_w - diff_d;
      int i, si, dd;
      diff_inputs = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)raw_m * sizeof(double), diff_pool);
      if (NULL != diff_inputs) {
        memcpy(diff_inputs, inputs, (size_t)raw_m * sizeof(double));
        for (dd = 0; dd < diff_d; ++dd) {
          const int len = raw_w - dd;
          for (si = 0; si < s; ++si) {
            for (i = 0; i < len - 1; ++i) {
              diff_inputs[si * raw_w + i] =
                diff_inputs[si * raw_w + i + 1] - diff_inputs[si * raw_w + i];
            }
          }
        }
        for (si = 1; si < s; ++si) {
          for (i = 0; i < dw; ++i) {
            diff_inputs[si * dw + i] = diff_inputs[si * raw_w + i];
          }
        }
        inputs = diff_inputs;
      }
    }
    if (LIBXS_PREDICT_RAW != model->decompose
      && (model->nseries >= 2 || NULL != model->decompose_mat)) {
      decomp_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), decomp_pool);
      internal_libxs_predict_decompose_apply(model, inputs, decomp_inputs);
      inputs = decomp_inputs;
    }
    if (0 != extrapolate_mode) {
      extrapolate = 1;
    }
    else if (model->nseries > 0
      && NULL != model->input_min && NULL != model->input_rng)
    {
      for (j = 0; j < m && 0 == extrapolate; ++j) {
        if (inputs[j] < model->input_min[j]
          || (model->input_rng[j] > 0
            && inputs[j] > model->input_min[j] + model->input_rng[j]))
        {
          extrapolate = 1;
        }
      }
    }
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    internal_libxs_predict_normalize(model, inputs, norm_inputs);
    if (NULL == lock && NULL == info && NULL != outputs && n * 6 + n <= 256) {
      vals = local_buf;
    }
    else {
      vals = model->eval_buf;
    }
    errs = vals + n;
    conf = errs + n;
    var = conf + n;
    lo = var + n;
    hi = lo + n;
    rels = (int*)(hi + n);
    for (j = 0; j < n; ++j) { lo[j] = 0; hi[j] = 0; }
    if (nblend < 0) nblend = 0;
    if (nblend > model->nclusters) nblend = model->nclusters;
    /**
     * A view routes the query by the lags it reads, not by the whole window:
     * the partition is shared, but which cluster serves a short view is its own
     * decision, and forcing the full-window choice on it was measured to give
     * up most of the bank's gain (sunspots, six months ahead: 21.2 against
     * 20.2). internal_libxs_predict_viewdist2 removes the older lags from the
     * distance exactly as the neighbor scan does.
     */
    best_dist = internal_libxs_predict_viewdist2(norm_inputs,
      model->clusters[0].centroid, m, view, model->has_missing);
    for (c = 1; c < model->nclusters; ++c) {
      const double d = internal_libxs_predict_viewdist2(norm_inputs,
        model->clusters[c].centroid, m, view, model->has_missing);
      if (d < best_dist && model->clusters[c].nentries > 0) {
        best_dist = d; best_c = c;
      }
    }
    if (model->clusters[best_c].nentries <= 0) {
      for (c = 0; c < model->nclusters; ++c) {
        if (model->clusters[c].nentries > 0) { best_c = c; break; }
      }
    }
    if (model->clusters[best_c].nentries <= 0) {
      nblend = 0;
    }
    else if (NULL != model->rf) {
      for (j = 0; j < n; ++j) {
        double rf_conf = 0, rf_var = 0;
        vals[j] = internal_libxs_predict_rf_eval_output(
          model->rf, j, inputs, &rf_conf, &rf_var);
        conf[j] = rf_conf;
        var[j] = rf_var;
        errs[j] = 0;
        rels[j] = 0;
        if (NULL != src) src[j] = NULL;
        if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_RF;
      }
      if (NULL != info) {
        info->cluster = -1;
        info->distance = 0;
      }
      nblend = 0;
    }
    else if (nblend <= 1) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[best_c];
      const int nearest = (int)internal_libxs_predict_position(model, cl, norm_inputs);
      const double qi = (NULL != info && model->quantile > 0) ? model->quantile : 0;
      for (j = 0; j < n; ++j) {
        { const int use_classify = (0 != force_classify)
            ? 1 : ((0 != force_interp) ? 0 : cl->mode[j]);
          if (0 != use_classify && NULL != model->hknn_po_clusters
            && NULL != model->hknn_po_assignments)
          {
            const int pg = (NULL != model->hknn_po_groups)
              ? model->hknn_po_groups[j] : j;
            if (pg < model->hknn_ngroups && NULL != cl->sorted_idx
              && NULL != model->hknn_po_clusters[pg]
              && NULL != model->hknn_po_assignments[pg])
            {
              const int nn_entry = cl->sorted_idx[
                (nearest < cl->nentries) ? nearest : 0];
              const int po_c = (nn_entry >= 0 && nn_entry < model->nentries)
                ? model->hknn_po_assignments[pg][nn_entry] : 0;
              const internal_libxs_predict_cluster_t* pcl =
                &model->hknn_po_clusters[pg][po_c];
              int gsz = 0, lj = 0, oi;
              for (oi = 0; oi < n; ++oi) {
                if ((NULL != model->hknn_po_groups
                  && model->hknn_po_groups[oi] == pg)
                  || (NULL == model->hknn_po_groups && oi == pg))
                {
                  if (oi == j) lj = gsz;
                  ++gsz;
                }
              }
              if (gsz <= 0) gsz = 1;
              { double po_conf = 0, po_var = 0;
                if (pcl->nentries > 0 && NULL != pcl->kd_pts) {
                  vals[j] = internal_libxs_predict_classify2(
                    pcl, m, norm_inputs,
                    lj, gsz, pcl->ndistinct[lj], extrapolate, -1, NULL,
                    NULL, -1,
                    &po_conf, &po_var, qi, &lo[j], &hi[j],
                    internal_libxs_predict_central(model, j), view,
                    model->has_missing);
                  if (NULL != src) src[j] = pcl;
                  if (NULL != src_out) src_out[j] = lj;
                  if (NULL != src_nout) src_nout[j] = gsz;
                }
                else {
                  vals[j] = internal_libxs_predict_classify2(
                    cl, m, norm_inputs, j, n,
                    cl->ndistinct[j], extrapolate, -1, NULL, NULL, -1,
                    &po_conf, &po_var, qi, &lo[j], &hi[j],
                    internal_libxs_predict_central(model, j), view,
                    model->has_missing);
                  if (NULL != src) src[j] = cl;
                  if (NULL != src_out) src_out[j] = j;
                  if (NULL != src_nout) src_nout[j] = n;
                }
                if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_CLASSIFY;
              }
              internal_libxs_predict_classify(
                cl, m, norm_inputs, j, n,
                cl->ndistinct[j], extrapolate, -1, &conf[j], &var[j],
                internal_libxs_predict_central(model, j), view,
                model->has_missing);
              errs[j] = 0;
              rels[j] = 0;
            }
            else {
              vals[j] = internal_libxs_predict_classify2(
                cl, m, norm_inputs, j, n,
                cl->ndistinct[j], extrapolate, -1, NULL, NULL, -1,
                &conf[j], &var[j], qi, &lo[j], &hi[j],
                internal_libxs_predict_central(model, j), view,
                model->has_missing);
              errs[j] = 0;
              rels[j] = 0;
              if (NULL != src) src[j] = cl;
              if (NULL != src_out) src_out[j] = j;
              if (NULL != src_nout) src_nout[j] = n;
              if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_CLASSIFY;
            }
          }
          else if (0 != use_classify) {
            vals[j] = internal_libxs_predict_classify2(
              cl, m, norm_inputs, j, n,
              cl->ndistinct[j], extrapolate, -1, NULL, NULL, -1,
              &conf[j], &var[j], qi, &lo[j], &hi[j],
              internal_libxs_predict_central(model, j), view,
              model->has_missing);
            errs[j] = 0;
            rels[j] = 0;
            if (NULL != src) src[j] = cl;
            if (NULL != src_out) src_out[j] = j;
            if (NULL != src_nout) src_nout[j] = n;
            if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_CLASSIFY;
          }
          else {
            const double t = (0 != extrapolate)
              ? (double)cl->nentries : (double)nearest;
            const int d = cl->order[j];
            const double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
            double val = 0;
            int k;
            for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
            vals[j] = val;
            errs[j] = (NULL != info)
              ? internal_libxs_predict_local_error(model, cl, nearest, j)
              : cl->errors[j];
            conf[j] = 1.0;
            var[j] = 0;
            rels[j] = 1;
            if (NULL != src) src[j] = cl;
            if (NULL != src_out) src_out[j] = j;
            if (NULL != src_nout) src_nout[j] = n;
            if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_INTERP;
          }
        }
      }
      if (model->floor > 0) {
        const double cov_inter = internal_libxs_predict_coverage(
          cl->nentries, model->nentries, model->nclusters);
        const double d_rel = sqrt(best_dist) / cl->dmax;
        const double cov_intra = 1.0 / (1.0 + d_rel);
        const double cov = cov_inter * cov_intra;
        for (j = 0; j < n; ++j) {
          conf[j] = model->floor + cov * (conf[j] - model->floor);
        }
      }
      if (model->nclusters > 1) {
        double avg_conf = 0;
        for (j = 0; j < n; ++j) avg_conf += conf[j];
        avg_conf /= n;
        /**
         * How many clusters to average is read from the confidence rather than
         * fixed: the vote's own agreement says how far the evidence reaches.
         * A committed vote needs one cluster, and the count grows as agreement
         * falls, because a query whose neighborhood disagrees is one whose
         * regime the partition split. Measured on the earthquake case, where
         * confidence averages 0.45: three clusters give MAE 0.256 and the
         * curve only flattens near 0.243, so the former fixed 3 was leaving
         * most of the blending gain unclaimed.
         */
        if (avg_conf < LIBXS_PREDICT_BLEND_CONF) {
          /**
           * Only a many-valued output benefits from reaching further: its
           * estimate is an average, so more clusters average more evidence.
           * A few-valued output reports the winning vote fraction instead, and
           * additional clusters dilute the argmax rather than sharpen it -
           * measured on the GPU-tuning table, where growing the count cost AL
           * 1.5 points of gated precision. The distinction is the one the
           * vote itself already makes.
           */
          int nmany = 0;
          for (j = 0; j < n; ++j) {
            const int thresh = (int)(sqrt((double)cl->nentries) + 0.5);
            if (cl->ndistinct[j] > thresh) ++nmany;
          }
          if (nmany > n / 2) {
            const double reach = (LIBXS_PREDICT_BLEND_CONF - avg_conf)
              / LIBXS_PREDICT_BLEND_CONF;
            const int nb = LIBXS_PREDICT_BLEND_N
              + (int)(reach * reach * model->nclusters + 0.5);
            nblend = LIBXS_MIN(nb, model->nclusters);
          }
          else {
            nblend = LIBXS_MIN(LIBXS_PREDICT_BLEND_N, model->nclusters);
          }
        }
        else if (model->smooth > 0) {
          const double radius = sqrt(best_dist) * (1.0 + model->smooth);
          int nb = 1;
          for (c = 0; c < model->nclusters; ++c) {
            if (c != best_c) {
              const double d = sqrt(internal_libxs_predict_dist2(
                norm_inputs, model->clusters[c].centroid, m,
                model->has_missing));
              if (d <= radius) ++nb;
            }
          }
          nblend = LIBXS_MIN(nb, model->nclusters);
        }
      }
      if (nblend <= 1 && NULL != info) {
        info->cluster = best_c;
        info->distance = (cl->dmax > 0.0)
          ? sqrt(best_dist) / cl->dmax : 0.0;
      }
    }
    if (nblend > 1) {
      typedef struct { double dist; int idx; } dc_t;
      const double conf_thr = 0.7;
      int dc_pool = 0;
      dc_t* dists = (dc_t*)LIBXS_PREDICT_MALLOC(
        (size_t)model->nclusters * sizeof(dc_t), dc_pool);
      int b;
      LIBXS_ASSERT(NULL != dists);
      for (c = 0; c < model->nclusters; ++c) {
        dists[c].dist = sqrt(internal_libxs_predict_dist2(norm_inputs,
          model->clusters[c].centroid, m, model->has_missing));
        dists[c].idx = c;
      }
      for (b = 0; b < nblend; ++b) {
        int minj = b;
        for (c = b + 1; c < model->nclusters; ++c) {
          if (dists[c].dist < dists[minj].dist) minj = c;
        }
        if (minj != b) { dc_t tmp = dists[b]; dists[b] = dists[minj]; dists[minj] = tmp; }
      }
      { const double qi = (NULL != info && model->quantile > 0)
          ? model->quantile : 0;
        for (j = 0; j < n; ++j) {
          { const internal_libxs_predict_cluster_t* cl_primary = &model->clusters[dists[0].idx];
            const int use_classify = (0 != force_classify)
              ? 1 : ((0 != force_interp) ? 0 : cl_primary->mode[j]);
            double blend_val = 0, blend_conf = 0, blend_var = 0, blend_err = 0;
            double blend_lo = 0, blend_hi = 0, wsum = 0;
            int blend_rel = 0;
            if (conf[j] >= conf_thr && (0.0 >= model->smooth
              || 0 != use_classify)) continue;
            for (b = 0; b < nblend; ++b) {
              const int ci = dists[b].idx;
              const internal_libxs_predict_cluster_t* cl2 = &model->clusters[ci];
              double w = (dists[b].dist > 0) ? (1.0 / dists[b].dist) : 1e30;
              if (0 != extrapolate && cl_primary->fprint_sig > 0) {
                const double sim = 1.0 / (1.0
                  + LIBXS_FABS(cl2->fprint_sig - cl_primary->fprint_sig)
                  / cl_primary->fprint_sig);
                w *= sim;
              }
              if (0 != use_classify) {
                double cj_conf = 1.0, cj_var = 0, cj_lo = 0, cj_hi = 0;
                const double v = internal_libxs_predict_classify2(
                  cl2, m, norm_inputs, j, n,
                  cl2->ndistinct[j], extrapolate, -1, NULL, NULL, -1,
                  &cj_conf, &cj_var, qi, &cj_lo, &cj_hi,
                  internal_libxs_predict_central(model, j), view,
                  model->has_missing);
                blend_val += w * v;
                blend_conf += w * cj_conf;
                blend_var += w * cj_var;
                blend_lo += w * cj_lo;
                blend_hi += w * cj_hi;
              }
              else {
                const int nearest2 = (int)internal_libxs_predict_position(model, cl2, norm_inputs);
                const double t = (0 != extrapolate)
                  ? (double)cl2->nentries : (double)nearest2;
                const int d = cl2->order[j];
                const double* cj = cl2->coeffs + (size_t)j * (cl2->maxorder + 1);
                double val = 0;
                int k;
                for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
                blend_val += w * val;
                blend_err += w * cl2->errors[j];
                blend_conf += w;
                blend_rel = 1;
              }
              wsum += w;
            }
            if (wsum > 0) {
              if (NULL != src) src[j] = NULL;
              if (NULL != src_mode) src_mode[j] = LIBXS_PREDICT_SRC_BLEND;
              vals[j] = blend_val / wsum;
              conf[j] = blend_conf / wsum;
              var[j] = blend_var / wsum;
              errs[j] = blend_err / wsum;
              rels[j] = blend_rel;
              lo[j] = blend_lo / wsum;
              hi[j] = blend_hi / wsum;
            }
          }
        }
      }
      if (NULL != info) {
        const internal_libxs_predict_cluster_t* cl0 = &model->clusters[dists[0].idx];
        info->cluster = -1;
        info->distance = (cl0->dmax > 0.0)
          ? dists[0].dist / cl0->dmax : 0.0;
      }
      LIBXS_PREDICT_FREE(dists, dc_pool);
    }
    if (0 != extrapolate && n > 2) {
      int k;
      for (k = 1; k < n - 1; ++k) {
        const double avg = 0.5 * (vals[k - 1] + vals[k + 1]);
        vals[k] = 0.75 * vals[k] + 0.25 * avg;
      }
    }
    if (0 == extrapolate) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[best_c];
      for (j = 0; j < n; ++j) {
        if (var[j] > 0 && 0 != cl->mode[j]) {
          double mean = 0, global_var = 0;
          int k;
          for (k = 0; k < cl->nentries; ++k) {
            mean += cl->raw_outputs[(size_t)k * n + j];
          }
          mean /= cl->nentries;
          for (k = 0; k < cl->nentries; ++k) {
            const double d = cl->raw_outputs[(size_t)k * n + j] - mean;
            global_var += d * d;
          }
          global_var /= cl->nentries;
          if (global_var > 0) {
            const double ratio = var[j] / global_var;
            if (ratio > 1.5) {
              const double alpha = LIBXS_MIN((ratio - 1.5) * 0.1, 0.3);
              vals[j] = (1.0 - alpha) * vals[j] + alpha * mean;
            }
          }
        }
      }
    }
    { double min_conf = 1.0;
      /**
       * The consistency penalty is computed from the round trip this pass
       * makes, so a caller that asked for one still gets the pass even though
       * refinement itself is off by default: the alternative silently turns
       * set_consistency into dead code.
       */
      const int gated = (0 > model->refine)
        || (0 == model->refine && 0 < model->consistency);
      int iter_count = 0, max_iter = (0 < model->refine)
        ? model->refine : ((0 != gated) ? 1 : 0);
      for (j = 0; j < n; ++j) {
        if (conf[j] < min_conf) min_conf = conf[j];
      }
      if (0 != gated && min_conf >= 0.9) max_iter = 0;
      /**
       * A forest answers from the raw inputs, and this pass would replace that
       * answer with a cluster's on a comparison between a vote over ntrees and
       * a vote over k neighbours. The second reaches 1.0 whenever the neighbors
       * agree, which a hundred trees over seven classes rarely does, so the
       * substitution is close to unconditional: it cost the crystal corpus 2.7
       * points (82.3% against 79.6%) and made the gated precision worse too.
       * It also made a forest depend on the input coordinate, which it reads
       * none of, because the point inverted here is re-normalized to find it.
       */
      if (NULL != model->rf) max_iter = 0;
      for (; iter_count < max_iter && NULL != model->entries; ++iter_count) {
        double target[128];
        int canon_pool = 0;
        double* canon = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)m * sizeof(double), canon_pool);
        if (NULL != canon) {
          for (j = 0; j < n; ++j) {
            /**
             * The inverse reads the whole output vector to recover one set of
             * inputs, so whatever it is given for output j reaches every other
             * output's re-prediction. It is therefore given the vote's mean
             * even where the model reports the median: the median is the
             * better answer for j, but substituting it here moves the inputs
             * that j's neighbors are refined from, which measured as a real
             * loss on them (GPU-tuning AL: 99.4% gated precision to 97.9%)
             * for no gain on j. What the caller receives is unaffected -
             * only the point the refinement pass inverts through.
             */
            const double vj = (0 == internal_libxs_predict_central(model, j))
              ? vals[j]
              : internal_libxs_predict_classify(&model->clusters[best_c], m, norm_inputs, j, n,
                  model->clusters[best_c].ndistinct[j], extrapolate, -1,
                  NULL, NULL, 0, view, model->has_missing);
            target[j] = (NULL != model->transforms)
              ? internal_libxs_predict_inv(model->transforms[j], vj)
              : vj;
          }
          libxs_predict_inverse(NULL, model, target, canon, NULL);
          { double refined[128], rconf[128];
            int rpool = 0;
            double* rnorm = (double*)LIBXS_PREDICT_MALLOC(
              (size_t)m * sizeof(double), rpool);
            if (NULL != rnorm) {
              int decomp2_pool = 0;
              double* dcinp = NULL;
              const double* eval_inp = canon;
              if (LIBXS_PREDICT_RAW != model->decompose
                && (model->nseries >= 2 || NULL != model->decompose_mat))
              {
                dcinp = (double*)LIBXS_PREDICT_MALLOC(
                  (size_t)m * sizeof(double), decomp2_pool);
                if (NULL != dcinp) {
                  internal_libxs_predict_decompose_apply(model, canon, dcinp);
                  eval_inp = dcinp;
                }
              }
              internal_libxs_predict_normalize(model, eval_inp, rnorm);
              { const int rc = best_c;
                const internal_libxs_predict_cluster_t* rcl = &model->clusters[rc];
                const double rt_dist = sqrt(internal_libxs_predict_dist2(
                  norm_inputs, rnorm, m, model->has_missing));
                if (rt_dist <= rcl->dmax) {
                  for (j = 0; j < n; ++j) {
                    if (conf[j] >= 0.9) continue;
                    { const int use_classify = (0 != force_classify)
                        ? 1 : ((0 != force_interp) ? 0 : rcl->mode[j]);
                      if (0 != use_classify) {
                        double rc_conf = 0;
                        refined[j] = internal_libxs_predict_classify(
                          rcl, m, rnorm, j, n,
                          rcl->ndistinct[j], extrapolate, -1, &rc_conf, NULL,
                          internal_libxs_predict_central(model, j), view,
                          model->has_missing);
                        rconf[j] = rc_conf;
                      }
                      else {
                        refined[j] = vals[j];
                        rconf[j] = conf[j];
                      }
                    }
                  }
                  for (j = 0; j < n; ++j) {
                    if (conf[j] >= 0.9) continue;
                    if (rconf[j] > conf[j]) {
                      vals[j] = refined[j];
                      conf[j] = rconf[j];
                    }
                  }
                }
                else if (model->consistency > 0) {
                  const double q = model->floor;
                  const double s = 1.0
                    / (1.0 + model->consistency * rt_dist / rcl->dmax);
                  for (j = 0; j < n; ++j) {
                    conf[j] = q + s * (conf[j] - q);
                  }
                }
              }
              if (NULL != dcinp) LIBXS_PREDICT_FREE(dcinp, decomp2_pool);
              LIBXS_PREDICT_FREE(rnorm, rpool);
            }
          }
          LIBXS_PREDICT_FREE(canon, canon_pool);
        }
      }
    }
    if (model->floor > 0 && 0 == extrapolate) {
      const internal_libxs_predict_cluster_t* mcl = &model->clusters[best_c];
      if (NULL != mcl->out_var && mcl->nentries > 1) {
        double maha = 0;
        int nv = 0;
        for (j = 0; j < n; ++j) {
          if (mcl->out_var[j] > 0 && 0 != mcl->mode[j]) {
            const double d = vals[j] - mcl->out_mean[j];
            maha += d * d / mcl->out_var[j];
            ++nv;
          }
        }
        if (nv > 0) {
          const double maha_norm = sqrt(maha / nv);
          if (maha_norm > 1.5) {
            const double penalty = 1.5 / maha_norm;
            for (j = 0; j < n; ++j) {
              conf[j] = model->floor
                + penalty * (conf[j] - model->floor);
            }
          }
        }
      }
    }
    if (NULL != model->transforms) {
      for (j = 0; j < n; ++j) {
        vals[j] = internal_libxs_predict_inv(model->transforms[j], vals[j]);
        if (model->quantile > 0 && (lo[j] != 0 || hi[j] != 0)) {
          lo[j] = internal_libxs_predict_inv(model->transforms[j], lo[j]);
          hi[j] = internal_libxs_predict_inv(model->transforms[j], hi[j]);
        }
      }
    }
    if (diff_d > 0 && model->nseries > 0) {
      const int tgt = model->target;
      const int raw_w = model->window;
      int dd;
      for (dd = diff_d - 1; dd >= 0; --dd) {
        double base = raw_inputs[tgt * raw_w + raw_w - 1];
        double base_lo, base_hi;
        int k;
        for (k = 0; k < dd; ++k) {
          base = base - raw_inputs[tgt * raw_w + raw_w - 2 - k];
        }
        base_lo = base; base_hi = base;
        for (j = 0; j < n; ++j) {
          base += vals[j];
          vals[j] = base;
          if (model->quantile > 0 && (lo[j] != 0 || hi[j] != 0)) {
            base_lo += lo[j]; lo[j] = base_lo;
            base_hi += hi[j]; hi[j] = base_hi;
          }
        }
      }
    }
    if (NULL != outputs) {
      memcpy(outputs, vals, (size_t)n * sizeof(double));
    }
    if (NULL != info) {
      if (model->quantile > 0) {
        const double z = internal_libxs_predict_quantile_z(model->quantile);
        const internal_libxs_predict_cluster_t* icl = &model->clusters[best_c];
        for (j = 0; j < n; ++j) {
          if (lo[j] != 0 || hi[j] != 0) {
            const double c_inv = (conf[j] > 0) ? (1.0 / conf[j]) : 1.0;
            const double mid = vals[j];
            double hw_lo = (mid - lo[j]) * c_inv;
            double hw_hi = (hi[j] - mid) * c_inv;
            if (NULL != icl->out_rms && icl->out_rms[j] > 0) {
              const double cal_hw = icl->out_rms[j] * z * c_inv;
              if (cal_hw > hw_lo) hw_lo = cal_hw;
              if (cal_hw > hw_hi) hw_hi = cal_hw;
            }
            lo[j] = mid - hw_lo;
            hi[j] = mid + hw_hi;
          }
          else if (0 != rels[j] && NULL != icl->out_rms
            && icl->out_rms[j] > 0)
          {
            const double c_inv = (conf[j] > 0) ? (1.0 / conf[j]) : 1.0;
            const double sigma = (NULL != icl->out_var && icl->out_var[j] > 0)
              ? sqrt(icl->out_var[j]) : icl->out_rms[j];
            const double hw = sigma * z * c_inv;
            lo[j] = vals[j] - hw;
            hi[j] = vals[j] + hw;
          }
          else if (0 != rels[j] && errs[j] > 0) {
            const double c_inv = (conf[j] > 0) ? (1.0 / conf[j]) : 1.0;
            lo[j] = vals[j] - errs[j] * c_inv;
            hi[j] = vals[j] + errs[j] * c_inv;
          }
        }
        info->lower = lo;
        info->upper = hi;
      }
      else {
        info->lower = NULL;
        info->upper = NULL;
      }
      info->values = vals;
      info->error = errs;
      info->confidence = conf;
      info->variance = var;
      info->interpolated = rels;
      info->noutputs = n;
    }
    LIBXS_PREDICT_FREE(norm_inputs, norm_pool);
    if (NULL != decomp_inputs) LIBXS_PREDICT_FREE(decomp_inputs, decomp_pool);
    if (NULL != diff_inputs) LIBXS_PREDICT_FREE(diff_inputs, diff_pool);
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model, const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend)
{
  LIBXS_ASSERT(NULL != model);
  if (1 >= model->nbank || NULL == model->bank_w || NULL == outputs) {
    internal_libxs_predict_eval_ex(lock, model, inputs, outputs, info, nblend,
      NULL, NULL, NULL, NULL, NULL);
  }
  else {
    /**
     * Every view queries the same corpus, the same partition and the same
     * neighbor index; they differ only in how many of the most recent lags the
     * distance reads. info describes the first view, the one that reads the
     * whole window, so a caller reading confidence sees the primary model
     * rather than an average of incomparable numbers.
     */
    const int n = model->noutputs;
    double acc[LIBXS_PREDICT_HMAX];
    int b, j, nacc = 0;
    if (n > LIBXS_PREDICT_HMAX) {
      internal_libxs_predict_eval_ex(lock, model, inputs, outputs, info, nblend,
        NULL, NULL, NULL, NULL, NULL);
      return;
    }
    for (j = 0; j < n; ++j) acc[j] = 0;
    for (b = 0; b < model->nbank; ++b) {
      internal_libxs_predict_view_t view;
      view.w = model->bank_w[b];
      view.s = model->nseries;
      view.full = model->bank_w[0];
      internal_libxs_predict_eval_ex(lock, model, inputs, outputs,
        (0 == b) ? info : NULL, nblend, NULL, NULL, NULL, NULL,
        (0 == b) ? NULL : &view);
      for (j = 0; j < n; ++j) acc[j] += outputs[j];
      ++nacc;
    }
    for (j = 0; j < n; ++j) outputs[j] = acc[j] / nacc;
  }
}


LIBXS_API void libxs_predict_eval_batch_task(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend, int tid, int ntasks)
{
  const int m = model->ninputs, n = model->noutputs;
  int begin, end, i;
  internal_libxs_predict_split(count, tid, ntasks, &begin, &end);
  LIBXS_ASSERT(NULL != model && 0 != model->built);
  LIBXS_ASSERT(NULL != inputs_batch && NULL != outputs_batch);
  for (i = begin; i < end; ++i) {
    libxs_predict_eval(NULL, model,
      inputs_batch + (size_t)i * m,
      outputs_batch + (size_t)i * n,
      NULL, nblend);
  }
}


LIBXS_API void libxs_predict_eval_batch(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend)
{
  libxs_predict_eval_batch_task(model, inputs_batch, outputs_batch,
    count, nblend, 0, 1);
}


LIBXS_API void libxs_predict_inverse(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double target_outputs[], double inputs[],
  libxs_predict_info_t* info)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != target_outputs && NULL != inputs);
  if (NULL == model->entries) {
    memset(inputs, 0, (size_t)model->ninputs * sizeof(double));
    if (NULL != info) {
      info->noutputs = model->noutputs;
      info->cluster = -1;
      info->distance = DBL_MAX;
      info->values = NULL;
      info->error = NULL;
      info->confidence = NULL;
      info->lower = NULL;
      info->upper = NULL;
      info->interpolated = NULL;
    }
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    double best_score = DBL_MAX;
    int best_i = 0, i, j;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    for (i = 0; i < p; ++i) {
      const double* eout = model->entries[i].outputs;
      double score = 0;
      int disqualified = 0;
      for (j = 0; j < n && 0 == disqualified; ++j) {
        double target = target_outputs[j];
        double actual = eout[j];
        if (NULL != model->transforms) {
          target = internal_libxs_predict_fwd(model->transforms[j], target);
        }
        { const int c = model->assignments[i];
          const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
          if (0 != cl->mode[j]) {
            if (target != actual) disqualified = 1;
          }
          else {
            const double d = target - actual;
            score += d * d;
          }
        }
      }
      if (0 == disqualified && score < best_score) {
        best_score = score;
        best_i = i;
      }
    }
    if (best_score >= DBL_MAX) {
      for (i = 0; i < p; ++i) {
        const double* eout = model->entries[i].outputs;
        double score = 0;
        for (j = 0; j < n; ++j) {
          double target = target_outputs[j];
          double actual = eout[j];
          double d;
          if (NULL != model->transforms) {
            target = internal_libxs_predict_fwd(model->transforms[j], target);
          }
          d = target - actual;
          score += d * d;
        }
        if (score < best_score) { best_score = score; best_i = i; }
      }
    }
    if (LIBXS_PREDICT_RAW != model->decompose
      && (model->nseries >= 2 || NULL != model->decompose_mat)) {
      int inv_pool = 0;
      double* raw = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), inv_pool);
      internal_libxs_predict_decompose_inverse(model, model->entries[best_i].inputs, raw);
      memcpy(inputs, raw, (size_t)m * sizeof(double));
      LIBXS_PREDICT_FREE(raw, inv_pool);
    }
    else {
      memcpy(inputs, model->entries[best_i].inputs, (size_t)m * sizeof(double));
    }
    if (NULL != info) {
      info->noutputs = n;
      info->cluster = (NULL != model->assignments) ? model->assignments[best_i] : -1;
      info->distance = sqrt(best_score);
      info->values = NULL;
      info->error = NULL;
      info->confidence = NULL;
      info->lower = NULL;
      info->upper = NULL;
      info->interpolated = NULL;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


/**
 * Sorted distinct values and counts for one output, built from the clusters'
 * raw_outputs so a loaded model works without entries. This is the support the
 * probability normalizes over: exact values, never tolerance balls, because
 * independently placed balls can overlap or leave gaps and the masses would
 * then not sum to one. Derived state only - the serialized format is
 * unchanged.
 */
LIBXS_API_INLINE int internal_libxs_predict_support(libxs_predict_t* model,
  int j)
{
  int result = EXIT_SUCCESS;
  const int n = model->noutputs;
  if (NULL == model->sup_vals) {
    model->sup_vals = (double**)calloc((size_t)n, sizeof(double*));
    model->sup_freq = (double**)calloc((size_t)n, sizeof(double*));
    model->sup_n = (int*)calloc((size_t)n, sizeof(int));
    model->sup_tot = (int*)calloc((size_t)n, sizeof(int));
    if (NULL == model->sup_vals || NULL == model->sup_freq
      || NULL == model->sup_n || NULL == model->sup_tot)
    {
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && NULL == model->sup_vals[j]) {
    int total = 0, c;
    for (c = 0; c < model->nclusters; ++c) {
      if (NULL != model->clusters[c].raw_outputs) {
        total += model->clusters[c].nentries;
      }
    }
    if (0 < total) {
      double* all = (double*)malloc((size_t)total * sizeof(double));
      if (NULL != all) {
        int at = 0, i, nd = 1;
        for (c = 0; c < model->nclusters; ++c) {
          const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
          if (NULL != cl->raw_outputs) {
            for (i = 0; i < cl->nentries; ++i) {
              all[at++] = cl->raw_outputs[(size_t)i * n + j];
            }
          }
        }
        libxs_sort(all, at, sizeof(double), libxs_cmp_f64, NULL);
        for (i = 1; i < at; ++i) {
          if (all[i] != all[i - 1]) ++nd;
        }
        model->sup_vals[j] = (double*)malloc((size_t)nd * sizeof(double));
        model->sup_freq[j] = (double*)calloc((size_t)nd, sizeof(double));
        if (NULL != model->sup_vals[j] && NULL != model->sup_freq[j]) {
          int k = 0;
          model->sup_vals[j][0] = all[0];
          model->sup_freq[j][0] = 1;
          for (i = 1; i < at; ++i) {
            if (all[i] != all[i - 1]) model->sup_vals[j][++k] = all[i];
            model->sup_freq[j][k] += 1;
          }
          for (i = 0; i < nd; ++i) model->sup_freq[j][i] /= at;
          model->sup_n[j] = nd;
          model->sup_tot[j] = at;
        }
        else result = EXIT_FAILURE;
        free(all);
      }
      else result = EXIT_FAILURE;
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_support_index(
  const double vals[], int n, double v)
{
  int lo = 0, hi = n - 1, result = -1;
  while (lo <= hi && 0 > result) {
    const int mid = lo + (hi - lo) / 2;
    if (vals[mid] < v) lo = mid + 1;
    else if (vals[mid] > v) hi = mid - 1;
    else result = mid;
  }
  return result;
}


/**
 * Normalize to sum exactly 1.0. Dividing by a compensated total is not
 * sufficient: the quotients round individually. The residual is placed on the
 * largest element, where the relative perturbation is smallest, and then
 * corrected once more in the same order the verification sums, so the element
 * absorbs precisely the error the accumulation makes.
 */
LIBXS_API_INLINE void internal_libxs_predict_prob_norm(double p[], int n,
  double scratch[])
{
  if (0 < n) {
    const double total = libxs_sum2(p, n);
    int i, imax = 0;
    if (0 < total) {
      for (i = 0; i < n; ++i) p[i] /= total;
    }
    else {
      for (i = 0; i < n; ++i) p[i] = 1.0 / n;
    }
    for (i = 1; i < n; ++i) {
      if (p[i] > p[imax]) imax = i;
    }
    for (i = 0; i < n; ++i) {
      if (i != imax) scratch[i - (i > imax ? 1 : 0)] = p[i];
    }
    { const double rest = (1 < n) ? libxs_sum2(scratch, n - 1) : 0.0;
      double dev;
      p[imax] = 1.0 - rest;
      dev = libxs_sum2(p, n) - 1.0;
      if (0 != dev) p[imax] -= dev;
    }
  }
}


/**
 * One causal fixed-share step over the escape-rate experts: multiplicative
 * log-loss update toward the experts that beat the mixture, then a uniform
 * share redistributed so an expert that was wrong for a stretch can recover.
 * Scored strictly after the reported probability is committed, so no target
 * information enters it. The ratio is floored only where exactly zero, which
 * would otherwise zero an expert permanently - the uniform recovery term only
 * reaches slots that still hold mass.
 */
LIBXS_API_INLINE void internal_libxs_predict_escape_update(double weight[],
  const double plik[], double mixture)
{
  double total = 0;
  int i;
  for (i = 0; i < LIBXS_PREDICT_NESCAPE; ++i) {
    double relative = (mixture > 0) ? (plik[i] / mixture) : 1.0;
    if (!(relative > 0.0)) relative = LIBXS_PREDICT_ESCAPE_RELMIN;
    weight[i] *= pow(relative, LIBXS_PREDICT_ESCAPE_ETA);
    total += weight[i];
  }
  if (0 < total) {
    const double uniform = 1.0 / LIBXS_PREDICT_NESCAPE;
    for (i = 0; i < LIBXS_PREDICT_NESCAPE; ++i) {
      weight[i] = (1.0 - LIBXS_PREDICT_ESCAPE_SHARE) * weight[i] / total
        + LIBXS_PREDICT_ESCAPE_SHARE * uniform;
    }
  }
}


/**
 * Distribution over one discrete output at the query. Each escape-rate expert
 * mixes the local kNN evidence with the global frequency prior at its own rate;
 * the reported distribution is the bank-weighted average of the experts, and the
 * bank weights are then updated from how well each expert scored the value the
 * caller actually asked about. A single default rate is not available: the
 * best rate was measured to range 0.10..0.80 across datasets.
 *
 * p receives ns+1 entries - the support followed by the aggregate mass of
 * everything outside it. With vocabulary > ns that trailing mass is what each
 * unattested value shares; with vocabulary == 0 it is reported as-is via
 * out_novel and the support masses sum to 1 - novel.
 */
LIBXS_API_INLINE int internal_libxs_predict_dist(
  const libxs_predict_t* model, const double* weight,
  const internal_libxs_predict_cluster_t* cl, const double* norm_inputs,
  int j, int out_j, int nouts, int extrapolate, int vocabulary,
  double* p, double* scratch, int stride, double* out_entropy)
{
  const int ns = model->sup_n[j];
  const double* sv = model->sup_vals[j];
  const double* sf = model->sup_freq[j];
  double candidates[LIBXS_PREDICT_KNN];
  double dists[LIBXS_PREDICT_KNN];
  double* local = scratch + stride;
  double best = 0, wsum, ent = 0;
  int nfound = 0, exact = 0, exact_nearest = 0, i, e;
  int result = EXIT_SUCCESS;
  const int nvoc = (vocabulary > ns) ? vocabulary : ns;
  const int outside = nvoc - ns;
  internal_libxs_predict_evidence(cl,
    model->ninputs, norm_inputs, out_j, nouts, extrapolate, -1, NULL,
    NULL, -1, candidates, dists, &nfound, &exact, &exact_nearest, &best, NULL,
    model->has_missing, NULL);
  for (i = 0; i < ns; ++i) local[i] = 0;
  for (i = 0; i < nfound; ++i) {
    const int si = internal_libxs_predict_support_index(sv, ns, candidates[i]);
    if (0 <= si) local[si] += (dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30;
  }
  wsum = libxs_sum2(local, ns);
  if (0 < wsum) {
    for (i = 0; i < ns; ++i) local[i] /= wsum;
  }
  else { /* no local evidence: the prior is all there is */
    for (i = 0; i < ns; ++i) local[i] = sf[i];
  }
  /**
   * An expert with rate r puts (1-r) on the local evidence and r on the escape.
   * The escape is the frequency prior smoothed over the declared vocabulary by
   * add-one, so every value the caller considers possible keeps positive mass:
   * an attested value the neighborhood happens to miss must not be scored
   * impossible, and neither must an unattested one. Without a vocabulary the
   * escape is the prior over the attested support and the novel atom stays
   * empty, because mass for values the caller has not enumerated would make the
   * total meaningless.
   */
  for (i = 0; i <= ns; ++i) p[i] = 0;
  { const double tot = (double)model->sup_tot[j];
    const double den = tot + (double)nvoc;
    for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
      const double w = weight[e];
      const double r = internal_libxs_predict_escape_rate[e];
      for (i = 0; i < ns; ++i) {
        const double esc = (0 < outside)
          ? ((sf[i] * tot + 1.0) / den) : sf[i];
        p[i] += w * ((1.0 - r) * local[i] + r * esc);
      }
      if (0 < outside) p[ns] += w * r * (double)outside / den;
      if (0 < w) ent -= w * log(w) / log(2.0);
    }
  }
  internal_libxs_predict_prob_norm(p, ns + 1, scratch);
  if (NULL != out_entropy) *out_entropy = ent;
  if (!(libxs_sum2(p, ns + 1) > 0)) result = EXIT_FAILURE;
  return result;
}


/**
 * Mass of ONE value, without enumerating the support.
 *
 * The distribution routine costs NESCAPE * ns per call because it writes every
 * support entry. It does not have to: local evidence reaches at most KNN
 * entries, so all but those carry pure escape mass, which is a closed-form
 * function of the frequency prior. Writing
 *
 *   A = sum_e w_e (1 - r_e),  B = sum_e w_e r_e
 *
 * every entry is A * local[i] + B * esc(i), and the normalizer follows from
 * sum_i esc(i) == (tot + ns) / den, because the stored frequencies sum to one.
 * So the total needs the KNN-bounded evidence set explicitly and the remaining
 * ns - |set| entries in closed form.
 *
 * This is EXACT, not an approximation: no mass is truncated and no tail is
 * folded into the escape. It is the same quantity the distribution reports,
 * computed in O(KNN log ns) instead of O(NESCAPE * ns). A top-k truncation was
 * the plan until the KNN bound on local evidence made an exact form available;
 * an approximation would have needed its own error budget and a separate name,
 * and this needs neither.
 *
 * Returns EXIT_SUCCESS and writes *out_prob (the mass at value v, or the
 * per-value share of the novel mass when v is outside the support) plus
 * *out_novel (the aggregate novel mass).
 */
LIBXS_API_INLINE int internal_libxs_predict_point(
  const libxs_predict_t* model, const double* weight,
  const internal_libxs_predict_cluster_t* cl, const double* norm_inputs,
  int j, int out_j, int nouts, int extrapolate, int vocabulary, double v,
  double* out_prob, double* out_novel, int* out_attested, double* out_local)
{
  const int ns = model->sup_n[j];
  const double* sv = model->sup_vals[j];
  const double* sf = model->sup_freq[j];
  double candidates[LIBXS_PREDICT_KNN];
  double dists[LIBXS_PREDICT_KNN];
  double local[LIBXS_PREDICT_KNN];
  int index[LIBXS_PREDICT_KNN];
  double best = 0, wsum = 0, a = 0, b = 0, esc_sum = 0, mass, novel;
  int nfound = 0, exact = 0, exact_nearest = 0, nlocal = 0, i, e;
  int result = EXIT_SUCCESS;
  const int nvoc = (vocabulary > ns) ? vocabulary : ns;
  const int outside = nvoc - ns;
  const double tot = (double)model->sup_tot[j];
  const double den = tot + (double)nvoc;
  const int si = internal_libxs_predict_support_index(sv, ns, v);
  internal_libxs_predict_evidence(cl,
    model->ninputs, norm_inputs, out_j, nouts, extrapolate, -1, NULL,
    NULL, -1, candidates, dists, &nfound, &exact, &exact_nearest, &best, NULL,
    model->has_missing, NULL);
  /* Accumulate evidence per DISTINCT support entry, as the dense path does by
     indexing into local[]; a value can be returned by several neighbors. */
  for (i = 0; i < nfound; ++i) {
    const int k = internal_libxs_predict_support_index(sv, ns, candidates[i]);
    if (0 <= k) {
      const double d = (dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30;
      int at = -1, q;
      for (q = 0; q < nlocal && 0 > at; ++q) {
        if (index[q] == k) at = q;
      }
      if (0 <= at) local[at] += d;
      else if (nlocal < LIBXS_PREDICT_KNN) {
        index[nlocal] = k;
        local[nlocal] = d;
        ++nlocal;
      }
      wsum += d;
    }
  }
  for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
    const double r = internal_libxs_predict_escape_rate[e];
    a += weight[e] * (1.0 - r);
    b += weight[e] * r;
  }
  /**
   * With no local evidence the dense path falls back to the prior over the whole
   * support, which is not sparse - but then local[i] == sf[i], so the entry
   * value collapses to (a + b * scale) * sf[i] and the total is still closed.
   */
  for (i = 0; i < nlocal; ++i) {
    const int k = index[i];
    esc_sum += (0 < outside) ? ((sf[k] * tot + 1.0) / den) : sf[k];
  }
  { /* mass at v, and the totals needed to normalize */
    const double all_esc = (0 < outside) ? ((tot + (double)ns) / den) : 1.0;
    const double tail = b * (all_esc - esc_sum);
    double head = 0, at_v = 0;
    if (0 < wsum) {
      for (i = 0; i < nlocal; ++i) {
        const int k = index[i];
        const double le = local[i] / wsum;
        const double es = (0 < outside) ? ((sf[k] * tot + 1.0) / den) : sf[k];
        const double pk = a * le + b * es;
        head += pk;
        if (k == si) at_v = pk;
      }
      if (0 <= si && 0 == at_v) { /* attested, but no neighbor voted for it */
        at_v = b * ((0 < outside) ? ((sf[si] * tot + 1.0) / den) : sf[si]);
      }
    }
    else { /* prior-only: every entry is (a + b*scale) * sf[i] */
      head = 0;
      if (0 <= si) {
        const double es = (0 < outside)
          ? ((sf[si] * tot + 1.0) / den) : sf[si];
        at_v = a * sf[si] + b * es;
      }
    }
    novel = (0 < outside) ? (b * (double)outside / den) : 0.0;
    { /* the same total the dense path divides by */
      const double total = (0 < wsum)
        ? (head + tail + novel)
        : (a + b * ((0 < outside) ? (tot + (double)ns) / den : 1.0) + novel);
      if (total > 0) {
        mass = (0 <= si) ? (at_v / total)
          : ((0 < outside) ? ((novel / total) / (double)outside) : 0.0);
        novel /= total;
      }
      else {
        mass = 0;
        novel = 0;
        result = EXIT_FAILURE;
      }
    }
  }
  if (NULL != out_prob) *out_prob = mass;
  if (NULL != out_novel) *out_novel = novel;
  if (NULL != out_attested) {
    int a_flag = 0;
    for (i = 0; i < nlocal && 0 == a_flag; ++i) {
      if (index[i] == si && local[i] > 0) a_flag = 1;
    }
    *out_attested = a_flag;
  }
  /**
   * The normalized local evidence AT THE OBSERVED VALUE, which is the only
   * element of the dense local[] array the bank update ever reads. Returning
   * it lets an adaptive stream skip the dense pass as well: the update needs one
   * scalar, not a distribution.
   */
  if (NULL != out_local) {
    double le = 0;
    if (0 <= si) {
      if (0 < wsum) {
        for (i = 0; i < nlocal; ++i) {
          if (index[i] == si) le = local[i] / wsum;
        }
      }
      else le = sf[si]; /* prior-only fallback, as the dense path */
    }
    *out_local = le;
  }
  return result;
}


/**
 * Score the observed value under every expert and advance the bank. Kept apart
 * from the distribution so the reported probability is fully committed before
 * any weight moves: the update is causal, and a caller that scores the same
 * query twice must get the same answer the first time.
 */
/**
 * The bank update given the local evidence AT THE TRUTH only. The dense form
 * below reads local[truth] and nothing else, so this is the same update with the
 * array replaced by the one value it uses - which is what lets the sparse
 * scoring path adapt without materializing a distribution.
 */
LIBXS_API_INLINE void internal_libxs_predict_dist_learn_at(
  const libxs_predict_t* model, double* weight, int j, double local_truth,
  int truth, int vocabulary)
{
  const int ns = model->sup_n[j];
  const double* sf = model->sup_freq[j];
  const int outside = ((vocabulary > ns) ? vocabulary : ns) - ns;
  double plik[LIBXS_PREDICT_NESCAPE];
  double mix = 0;
  int e;
  { const int nvoc = (vocabulary > ns) ? vocabulary : ns;
    const double tot = (double)model->sup_tot[j];
    const double den = tot + (double)nvoc;
    for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
      const double r = internal_libxs_predict_escape_rate[e];
      if (0 <= truth && truth < ns) {
        const double esc = (0 < outside)
          ? ((sf[truth] * tot + 1.0) / den) : sf[truth];
        plik[e] = (1.0 - r) * local_truth + r * esc;
      }
      else {
        plik[e] = (0 < outside) ? (r / den) : 0.0;
      }
      mix += weight[e] * plik[e];
    }
  }
  internal_libxs_predict_escape_update(weight, plik, mix);
}


LIBXS_API_INLINE void internal_libxs_predict_dist_learn(
  const libxs_predict_t* model, double* weight, int j, const double* local,
  int truth, int vocabulary)
{
  const int ns = model->sup_n[j];
  const double* sf = model->sup_freq[j];
  const int outside = ((vocabulary > ns) ? vocabulary : ns) - ns;
  double plik[LIBXS_PREDICT_NESCAPE];
  double mix = 0;
  int e;
  { const int nvoc = (vocabulary > ns) ? vocabulary : ns;
    const double tot = (double)model->sup_tot[j];
    const double den = tot + (double)nvoc;
    for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
      const double r = internal_libxs_predict_escape_rate[e];
      if (0 <= truth && truth < ns) {
        const double esc = (0 < outside)
          ? ((sf[truth] * tot + 1.0) / den) : sf[truth];
        plik[e] = (1.0 - r) * local[truth] + r * esc;
      }
      else {
        plik[e] = (0 < outside) ? (r / den) : 0.0;
      }
      mix += weight[e] * plik[e];
    }
  }
  internal_libxs_predict_escape_update(weight, plik, mix);
}


/**
 * Layout of one context. Within a region the pointer and double arrays precede
 * the int arrays, hence no member relies on the padding NBINT adds to be
 * aligned.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_ctx_size(int n, int maxsup)
{
  const size_t stride = (size_t)maxsup + 2;
  size_t bytes = sizeof(internal_libxs_predict_ctx_t);
  bytes += (size_t)n * LIBXS_PREDICT_NESCAPE * sizeof(double);
  bytes += 3 * stride * sizeof(double);  /* p, norm scratch, local */
  bytes += 4 * (size_t)n * sizeof(double); /* prob, logprob, zscore, novel */
  bytes += INTERNAL_LIBXS_PREDICT_NBINT(3 * n); /* kind, attested, support */
  /**
   * The dispatch buffers eval_ex fills, and the values it writes, live here
   * too: they are noutputs-sized with a lifetime of one call, so allocating
   * them per call would put a malloc/free pair on every scored position.
   */
  bytes += (size_t)n * sizeof(void*);   /* src */
  bytes += (size_t)n * sizeof(double);  /* vals */
  bytes += INTERNAL_LIBXS_PREDICT_NBINT(3 * n); /* smode, sout, snout */
  return bytes;
}


/**
 * Resolve the window views once the effective window is known. Each view
 * halves the lags of the one before it and keeps the most recent, stopping
 * before a view would read fewer than two lags. Only the lag count is stored:
 * a view is applied by zeroing the weight of the lags it does not read, so the
 * corpus, the partition and the neighbor index are shared.
 */
LIBXS_API_INLINE void internal_libxs_predict_bank_all(libxs_predict_t* model)
{
  const int nreq = model->nbank;
  model->nbank = 1;
  /**
   * A view drops the oldest lags of each series, which presumes the inputs are
   * still lags. A rotation (PCA) or a mode decomposition (SPREAD) replaces them
   * with combinations spanning the whole window, where dropping a coordinate
   * removes a mode rather than a span of history: measured on the SOI pair,
   * views raised the error at every horizon (0.58 to 0.63 at six months). Such
   * a model keeps the single view it would have had.
   */
  if (1 < nreq && 0 < model->nseries && 0 < model->window
    && (LIBXS_PREDICT_RAW == model->decompose
      || LIBXS_PREDICT_HKNN == model->decompose))
  {
    const int w = model->window - ((0 < model->diff_order) ? model->diff_order : 0);
    if (3 < w) {
      int* wv = (int*)malloc((size_t)nreq * sizeof(int));
      if (NULL != wv) {
        int i = 0;
        wv[0] = w;
        for (i = 1; i < nreq; ++i) {
          const int half = (wv[i - 1] + 1) / 2;
          if (half < 2) break;
          wv[i] = half;
        }
        if (1 < i) {
          free(model->bank_w);
          model->bank_w = wv;
          model->nbank = i;
        }
        else {
          free(wv);
        }
      }
    }
  }
}


/**
 * Decide per output whether the vote reports the mean or the median, by
 * scoring both against the entries the model was built from. Each entry is
 * predicted with itself excluded (skip_local), so the comparison is not the
 * fit but the error the aggregation would have made on data it did not see.
 * Absolute error is the criterion because that is what the median optimizes;
 * a tie keeps the mean, which is the historical behavior.
 */
/** The two formulas the single k_eff chose between, now applied per output. */
LIBXS_API_INLINE int internal_libxs_predict_keff_mode(int nc, int classify)
{
  const int k = (0 != classify)
    ? LIBXS_MAX(5, nc / 3)
    : LIBXS_MAX(3, (int)(sqrt((double)nc) + 0.5));
  return LIBXS_MIN(LIBXS_MIN(k, nc), LIBXS_PREDICT_KNN);
}


/**
 * Per-output neighbour count from each output's own mode. Runs at build and at
 * load so a loaded model behaves as the built one did, which is why the derived
 * form reaches no file. A count the caller pinned or the trial resolved does
 * reach one, because this formula would otherwise overwrite it at load; see
 * internal_libxs_predict_kapply, which runs straight after.
 */
LIBXS_API_INLINE void internal_libxs_predict_keff_all(libxs_predict_t* model)
{
  const int n = model->noutputs;
  int c;
  if (NULL != model->clusters) for (c = 0; c < model->nclusters; ++c) {
    internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    if (NULL == cl->k_out) {
      cl->k_out = (int*)malloc((size_t)n * sizeof(int));
    }
    if (NULL != cl->k_out) {
      int j;
      for (j = 0; j < n; ++j) {
        const int classify = (NULL != cl->mode) ? cl->mode[j] : 1;
        cl->k_out[j] = internal_libxs_predict_keff_mode(cl->nentries, classify);
      }
    }
  }
}


/**
 * Override the derived neighbour count with the caller's, or with the one the
 * trial resolved. Runs after the formula rather than instead of it, so a
 * cluster too small for the request still votes with what it has.
 */
LIBXS_API_INLINE void internal_libxs_predict_kapply(libxs_predict_t* model)
{
  const int n = model->noutputs;
  int c;
  if (NULL != model->clusters && (NULL != model->k_sel || 0 < model->kreq)) {
    for (c = 0; c < model->nclusters; ++c) {
      internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (NULL != cl->k_out) {
        int j;
        for (j = 0; j < n; ++j) {
          const int k = (NULL != model->k_sel)
            ? model->k_sel[j] : model->kreq;
          if (0 < k) {
            cl->k_out[j] = LIBXS_MIN(LIBXS_MIN(k, cl->nentries),
              LIBXS_PREDICT_KNN);
          }
        }
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_central_all(libxs_predict_t* model)
{
  const int n = model->noutputs;
  if (NULL == model->central_out) {
    model->central_out = (int*)malloc((size_t)n * sizeof(int));
    if (NULL != model->central_out) {
      int i;
      for (i = 0; i < n; ++i) model->central_out[i] = 0;
    }
  }
  if (NULL != model->central_out && NULL != model->clusters) {
    const int m = model->ninputs;
    int j;
    for (j = 0; j < n; ++j) {
      double err_avg = 0, err_med = 0;
      int c, nscored = 0;
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        const int nc = cl->nentries;
        if (nc > 2 && NULL != cl->kd_pts && NULL != cl->raw_outputs
          && NULL != cl->ndistinct)
        {
          int k;
          for (k = 0; k < nc; ++k) {
            const double* x = cl->kd_pts + (size_t)k * m;
            const double actual = cl->raw_outputs[(size_t)k * n + j];
            const double a = internal_libxs_predict_classify(cl, m, x, j, n,
              cl->ndistinct[j], 0, k, NULL, NULL, 0, NULL, model->has_missing);
            const double d = internal_libxs_predict_classify(cl, m, x, j, n,
              cl->ndistinct[j], 0, k, NULL, NULL, 1, NULL, model->has_missing);
            err_avg += LIBXS_FABS(a - actual);
            err_med += LIBXS_FABS(d - actual);
            ++nscored;
          }
        }
      }
      model->central_out[j] = (0 < nscored && err_med < err_avg) ? 1 : 0;
      if (0 > model->central) {
        fprintf(stderr, "LIBXS PREDICT: output %i uses the %s"
          " (mean %.4f, median %.4f over %i entries)\n", j,
          (0 != model->central_out[j]) ? "median" : "mean",
          (0 < nscored) ? (err_avg / nscored) : 0.0,
          (0 < nscored) ? (err_med / nscored) : 0.0, nscored);
      }
    }
  }
}


/**
 * Build the support cache for every output. Called at build and load so that
 * scoring only reads it: a lazily-built cache would be a write to shared state
 * on the first call of every stream.
 */
LIBXS_API_INLINE int internal_libxs_predict_support_all(libxs_predict_t* model)
{
  int result = EXIT_SUCCESS;
  int j;
  for (j = 0; j < model->noutputs && EXIT_SUCCESS == result; ++j) {
    result = internal_libxs_predict_support(model, j);
  }
  if (EXIT_SUCCESS == result && NULL == model->escape_w) {
    const size_t nw = (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE;
    model->escape_w = (double*)malloc(nw * sizeof(double));
    if (NULL != model->escape_w) {
      size_t i;
      for (i = 0; i < nw; ++i) {
        model->escape_w[i] = 1.0 / LIBXS_PREDICT_NESCAPE;
      }
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_maxsup(const libxs_predict_t* model)
{
  int j, maxsup = 0;
  if (NULL != model->sup_n) {
    for (j = 0; j < model->noutputs; ++j) {
      if (model->sup_n[j] > maxsup) maxsup = model->sup_n[j];
    }
  }
  return maxsup;
}


/**
 * Zero means the model cannot be scored, and must be reachable for that to be a
 * usable signal: the support build allocates per output and can fail for some
 * outputs while leaving the array itself non-NULL, and the size formula is
 * positive even when every support is empty. Reporting a plausible size for
 * such a model would let a caller allocate, score, and receive PNONE for every
 * output - a build failure indistinguishable from a model that simply has no
 * discrete outputs. So every output is required to carry a usable support, and
 * the weights must exist.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_ctx_bytes(
  const libxs_predict_t* model)
{
  size_t result = 0;
  if (NULL != model && 0 != model->built && NULL != model->sup_n
    && NULL != model->sup_vals && NULL != model->sup_freq
    && NULL != model->sup_tot && NULL != model->escape_w)
  {
    const int maxsup = internal_libxs_predict_maxsup(model);
    int j, usable = (0 < maxsup) ? 1 : 0;
    for (j = 0; j < model->noutputs && 0 != usable; ++j) {
      if (0 >= model->sup_n[j] || NULL == model->sup_vals[j]
        || NULL == model->sup_freq[j] || 0 >= model->sup_tot[j])
      {
        usable = 0;
      }
    }
    if (0 != usable) {
      result = internal_libxs_predict_ctx_size(model->noutputs, maxsup);
    }
  }
  return result;
}


/**
 * Validate a caller-supplied context against the model in front of it. A
 * context that was not produced by libxs_predict_prob_create for this model and
 * build is rejected rather than adapted: its buffers may be smaller than this
 * model needs, so re-initializing in place would overrun them.
 */
LIBXS_API_INLINE int internal_libxs_predict_ctx_valid(
  const libxs_predict_t* model, const void* context)
{
  const internal_libxs_predict_ctx_t* ctx =
    (const internal_libxs_predict_ctx_t*)context;
  return (NULL != ctx && LIBXS_PREDICT_CTX_MAGIC == ctx->magic
    && ctx->model == (const void*)model
    && ctx->nbuild == model->nbuild
    && ctx->noutputs == model->noutputs) ? 1 : 0;
}


/**
 * A context that was supplied but does not belong to this model and build must
 * make the call fail, not quietly fall back to frozen scoring: the caller asked
 * for an adapting stream, and silently giving them a different estimator is the
 * kind of substitution that produces a plausible wrong number with no symptom.
 * Passing NULL deliberately is the only way to select frozen mode.
 */
LIBXS_API_INLINE internal_libxs_predict_ctx_t* internal_libxs_predict_ctx(
  const libxs_predict_t* model, void* context)
{
  return (0 != internal_libxs_predict_ctx_valid(model, context))
    ? (internal_libxs_predict_ctx_t*)context : NULL;
}


LIBXS_API_INLINE double* internal_libxs_predict_ctx_weights(
  internal_libxs_predict_ctx_t* ctx, int j)
{
  return (double*)((unsigned char*)ctx
    + sizeof(internal_libxs_predict_ctx_t))
    + (size_t)j * LIBXS_PREDICT_NESCAPE;
}


LIBXS_API_INLINE double* internal_libxs_predict_ctx_scratch(
  internal_libxs_predict_ctx_t* ctx)
{
  return (double*)((unsigned char*)ctx
    + sizeof(internal_libxs_predict_ctx_t))
    + (size_t)ctx->noutputs * LIBXS_PREDICT_NESCAPE;
}


/** Start of the per-call dispatch buffers, past the reporting arrays. */
LIBXS_API_INLINE void* internal_libxs_predict_ctx_disp(
  internal_libxs_predict_ctx_t* ctx)
{
  const size_t stride = (size_t)ctx->maxsup + 2;
  double* base = internal_libxs_predict_ctx_scratch(ctx);
  return (void*)((unsigned char*)(base + 3 * stride + 4 * ctx->noutputs)
    + INTERNAL_LIBXS_PREDICT_NBINT(3 * ctx->noutputs));
}


LIBXS_API void* libxs_predict_prob_create(const libxs_predict_t* model)
{
  void* result = NULL;
  const size_t size = internal_libxs_predict_ctx_bytes(model);
  if (0 < size) {
    result = libxs_malloc(internal_libxs_default_pool, size,
      LIBXS_MALLOC_AUTO);
    if (NULL != result) {
      internal_libxs_predict_ctx_t* ctx =
        (internal_libxs_predict_ctx_t*)result;
      const size_t nw = (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE;
      double* w = (double*)((unsigned char*)ctx
        + sizeof(internal_libxs_predict_ctx_t));
      size_t i;
      memset(result, 0, size);
      ctx->magic = LIBXS_PREDICT_CTX_MAGIC;
      ctx->model = (const void*)model;
      ctx->nbuild = model->nbuild;
      ctx->noutputs = model->noutputs;
      ctx->maxsup = internal_libxs_predict_maxsup(model);
      /* seed from the model's weights so a loaded, converged model does not
         re-pay the adaptation transient on every fresh stream */
      for (i = 0; i < nw; ++i) {
        w[i] = (NULL != model->escape_w)
          ? model->escape_w[i] : (1.0 / LIBXS_PREDICT_NESCAPE);
      }
    }
  }
  return result;
}


LIBXS_API void libxs_predict_prob_destroy(void* context)
{
  if (NULL != context) {
    internal_libxs_predict_ctx_t* ctx =
      (internal_libxs_predict_ctx_t*)context;
    if (LIBXS_PREDICT_CTX_MAGIC == ctx->magic) {
      ctx->magic = 0;
      libxs_free(context);
    }
  }
}


LIBXS_API int libxs_predict_prob_commit(libxs_predict_t* model,
  const void* context)
{
  int result = EXIT_FAILURE;
  if (NULL != model && NULL != model->escape_w
    && 0 != internal_libxs_predict_ctx_valid(model, context))
  {
    /* the context is only READ here, so it stays const: casting it away would
       discard exactly the guarantee this entry point wants to make */
    const internal_libxs_predict_ctx_t* ctx =
      (const internal_libxs_predict_ctx_t*)context;
    const double* base = (const double*)((const unsigned char*)ctx
      + sizeof(internal_libxs_predict_ctx_t));
    int j;
    result = EXIT_SUCCESS;
    for (j = 0; j < model->noutputs; ++j) {
      const double* w = base + (size_t)j * LIBXS_PREDICT_NESCAPE;
      double* dst = model->escape_w + (size_t)j * LIBXS_PREDICT_NESCAPE;
      int e;
      for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) dst[e] = w[e];
    }
  }
  return result;
}


LIBXS_API void libxs_predict_prob(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context, const double inputs[],
  const double candidate[], double prob[],
  libxs_predict_prob_info_t* info, int vocabulary, int nblend)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != inputs
    && NULL != candidate);
  /* info aliases the context, so reporting requires one */
  LIBXS_ASSERT(NULL == info || NULL != context);
  if (NULL != model->sup_n && NULL != model->sup_vals
    && (NULL == context
      || 0 != internal_libxs_predict_ctx_valid(model, context)))
  {
    const int n = model->noutputs;
    internal_libxs_predict_ctx_t* ctx =
      internal_libxs_predict_ctx(model, context);
    const int maxsup = internal_libxs_predict_maxsup(model);
    const int stride = maxsup + 2;
    double dflt[LIBXS_PREDICT_NESCAPE];
    /**
     * Adaptive scoring takes every buffer from the context, so the path is
     * allocation-free. Frozen scoring has no context and allocates once from
     * the library pool rather than repeatedly from the system allocator.
     */
    int scratch_pool = 0;
    const internal_libxs_predict_cluster_t** src = NULL;
    int *smode, *sout, *snout;
    double *vals, *local = NULL;
    if (NULL != ctx) {
      unsigned char* d = (unsigned char*)internal_libxs_predict_ctx_disp(ctx);
      src = (const internal_libxs_predict_cluster_t**)d;
      d += (size_t)n * sizeof(void*);
      vals = (double*)d; d += (size_t)n * sizeof(double);
      smode = (int*)d; d += (size_t)n * sizeof(int);
      sout = (int*)d; d += (size_t)n * sizeof(int);
      snout = (int*)d;
    }
    else {
      const size_t nb = (size_t)n * sizeof(void*)
        + (size_t)n * sizeof(double)
        + (size_t)3 * stride * sizeof(double)
        + INTERNAL_LIBXS_PREDICT_NBINT(3 * n);
      unsigned char* d = (unsigned char*)LIBXS_PREDICT_MALLOC(nb,
        scratch_pool);
      src = (const internal_libxs_predict_cluster_t**)d;
      if (NULL != d) {
        d += (size_t)n * sizeof(void*);
        vals = (double*)d; d += (size_t)n * sizeof(double);
        local = (double*)d; d += (size_t)3 * stride * sizeof(double);
        smode = (int*)d; d += (size_t)n * sizeof(int);
        sout = (int*)d; d += (size_t)n * sizeof(int);
        snout = (int*)d;
      }
      else { smode = NULL; sout = NULL; snout = NULL; vals = NULL; }
    }
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (NULL != src && NULL != smode && NULL != sout && NULL != snout
      && NULL != vals && (NULL != ctx || NULL != local))
    {
      double* base = (NULL != ctx)
        ? internal_libxs_predict_ctx_scratch(ctx) : local;
      double* p = base;
      double* scratch = p + stride;
      double* pr = (NULL != ctx) ? (base + 3 * stride) : NULL;
      double* lp = (NULL != pr) ? (pr + n) : NULL;
      double* zs = (NULL != lp) ? (lp + n) : NULL;
      double* nv = (NULL != zs) ? (zs + n) : NULL;
      int* kind = (NULL != nv) ? (int*)(nv + n) : NULL;
      int* att = (NULL != kind) ? (kind + n) : NULL;
      int* sup = (NULL != att) ? (att + n) : NULL;
      double total = 0, ent = 0, entsum = 0;
      int j, nent = 0;
      if (NULL == ctx) {
        int e;
        for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
          dflt[e] = 1.0 / LIBXS_PREDICT_NESCAPE;
        }
      }
      internal_libxs_predict_eval_ex(NULL, model, inputs, vals, NULL,
        nblend, src, smode, sout, snout, NULL);
      for (j = 0; j < n; ++j) {
        const int ns = model->sup_n[j];
        /* frozen mode reads the model's weights and never writes them */
        double* w = (NULL != ctx)
          ? internal_libxs_predict_ctx_weights(ctx, j)
          : ((NULL != model->escape_w)
            ? (model->escape_w + (size_t)j * LIBXS_PREDICT_NESCAPE) : dflt);
        double cand = candidate[j];
        double pj = 0, lpj = -DBL_MAX, zj = 0, nvj = 0;
        int kj = LIBXS_PREDICT_PNONE, aj = 0;
        if (NULL != model->transforms) {
          cand = internal_libxs_predict_fwd(model->transforms[j], cand);
        }
        if (LIBXS_PREDICT_SRC_CLASSIFY == smode[j] && NULL != src[j]
          && 0 < ns && NULL == ctx)
        {
          /**
           * Frozen point query: no weight moves, so the local evidence the dense
           * path leaves in scratch is not needed afterwards and the mass can be
           * had in closed form. Same number, O(KNN log ns) instead of
           * O(NESCAPE * ns) - which is what this entry point already documents.
           * Entropy is a property of the distribution and is not reported here.
           */
          if (EXIT_SUCCESS == internal_libxs_predict_point(model, w, src[j],
            inputs, j, sout[j], snout[j], 0, vocabulary, cand, &pj, &nvj, &aj,
            NULL))
          {
            kj = LIBXS_PREDICT_PMASS;
            lpj = (pj > 0) ? (log(pj) / log(2.0)) : -DBL_MAX;
          }
        }
        else if (LIBXS_PREDICT_SRC_CLASSIFY == smode[j] && NULL != src[j]
          && 0 < ns)
        {
          if (EXIT_SUCCESS == internal_libxs_predict_dist(model, w, src[j],
            inputs, j, sout[j], snout[j], 0, vocabulary, p, scratch, stride,
            &ent))
          {
            const int si = internal_libxs_predict_support_index(
              model->sup_vals[j], ns, cand);
            const int outside = ((vocabulary > ns) ? vocabulary : ns) - ns;
            const double* ev = scratch + stride;
            entsum += ent;
            ++nent;
            kj = LIBXS_PREDICT_PMASS;
            nvj = p[ns];
            if (0 <= si) {
              pj = p[si];
              aj = (0 != ev[si]) ? 1 : 0;
            }
            else if (0 < outside) pj = p[ns] / outside;
            lpj = (pj > 0) ? (log(pj) / log(2.0)) : -DBL_MAX;
            if (NULL != ctx) {
              internal_libxs_predict_dist_learn(model, w, j, ev, si,
                vocabulary);
            }
          }
        }
        else if (LIBXS_PREDICT_SRC_INTERP == smode[j] && NULL != src[j]) {
          /**
           * A continuous target has no mass at a point. The bandwidth is
           * floored by the stored fit residual so the density cannot claim
           * more precision than the fit actually has.
           */
          const internal_libxs_predict_cluster_t* cl = src[j];
          const double rms = (NULL != cl->out_rms && cl->out_rms[j] > 0)
            ? cl->out_rms[j] : 0;
          const double sd = (NULL != cl->out_var && cl->out_var[j] > 0)
            ? sqrt(cl->out_var[j]) : 0;
          const double h = (rms > 0) ? rms : ((sd > 0) ? sd : 1.0);
          const double d = (cand - vals[j]) / h;
          const double norm = h * sqrt(2.0 * M_PI);
          kj = LIBXS_PREDICT_PDENSITY;
          zj = d;
          pj = exp(-0.5 * d * d) / norm;
          lpj = (-0.5 * d * d - log(norm)) / log(2.0);
          aj = 1;
        }
        if (NULL != prob) prob[j] = pj;
        if (NULL != pr) {
          pr[j] = pj; lp[j] = lpj; zs[j] = zj; nv[j] = nvj;
          kind[j] = kj; att[j] = aj; sup[j] = ns;
        }
        if (-DBL_MAX != lpj) total += lpj;
      }
      if (NULL != info) {
        info->prob = pr;
        info->logprob = lp;
        info->zscore = zs;
        info->kind = kind;
        info->attested = att;
        info->support = sup;
        info->novel = nv;
        info->total_logprob = total;
        info->noutputs = n;
        info->cluster = -1;
        /* mean over the outputs that carry a bank: a single output's value
           would depend on which one happened to be scored last */
        info->entropy = (0 < nent) ? (entsum / nent) : 0.0;
      }
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
    if (NULL == ctx) LIBXS_PREDICT_FREE((void*)src, scratch_pool);
  }
}


/**
 * Report the distribution for one output and, optionally, observe an outcome -
 * in that order, inside one call. Splitting these across two entry points made
 * the ordering a caller obligation with no failure symptom: observing first
 * yields a distribution shaped by weights that already saw the target, which is
 * better than the truth and looks entirely plausible. Here the guarantee is
 * structural: the masses are copied out before any weight moves.
 */
LIBXS_API int libxs_predict_prob_observe(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context, const double inputs[],
  int output, const double* candidate,
  double values[], double probs[], int capacity, double* novel,
  libxs_predict_prob_info_t* info, int vocabulary, int nblend)
{
  int result = 0;
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != inputs);
  /* info aliases the context, and observing requires somewhere to learn into */
  LIBXS_ASSERT(NULL == info || NULL != context);
  if (NULL != model->sup_n && NULL != model->sup_vals
    && 0 <= output && output < model->noutputs
    && (NULL == context
      || 0 != internal_libxs_predict_ctx_valid(model, context)))
  {
    const int n = model->noutputs;
    internal_libxs_predict_ctx_t* ctx =
      internal_libxs_predict_ctx(model, context);
    const int stride = internal_libxs_predict_maxsup(model) + 2;
    double dflt[LIBXS_PREDICT_NESCAPE];
    int scratch_pool = 0;
    const internal_libxs_predict_cluster_t** src = NULL;
    int *smode, *sout, *snout;
    double* local = NULL;
    if (NULL != ctx) {
      unsigned char* d = (unsigned char*)internal_libxs_predict_ctx_disp(ctx);
      src = (const internal_libxs_predict_cluster_t**)d;
      /* the values slot is part of the layout even though nothing is reported */
      d += (size_t)n * sizeof(void*) + (size_t)n * sizeof(double);
      smode = (int*)d; d += (size_t)n * sizeof(int);
      sout = (int*)d; d += (size_t)n * sizeof(int);
      snout = (int*)d;
    }
    else {
      const size_t nb = (size_t)n * sizeof(void*)
        + (size_t)3 * stride * sizeof(double)
        + INTERNAL_LIBXS_PREDICT_NBINT(3 * n);
      unsigned char* d = (unsigned char*)LIBXS_PREDICT_MALLOC(nb,
        scratch_pool);
      src = (const internal_libxs_predict_cluster_t**)d;
      if (NULL != d) {
        d += (size_t)n * sizeof(void*);
        local = (double*)d; d += (size_t)3 * stride * sizeof(double);
        smode = (int*)d; d += (size_t)n * sizeof(int);
        sout = (int*)d; d += (size_t)n * sizeof(int);
        snout = (int*)d;
      }
      else { smode = NULL; sout = NULL; snout = NULL; }
    }
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (NULL != src && NULL != smode && NULL != sout && NULL != snout
      && (NULL != ctx || NULL != local))
    {
      double* p = (NULL != ctx)
        ? internal_libxs_predict_ctx_scratch(ctx) : local;
      double* scratch = p + stride;
      double* w = (NULL != ctx)
        ? internal_libxs_predict_ctx_weights(ctx, output)
        : ((NULL != model->escape_w)
          ? (model->escape_w + (size_t)output * LIBXS_PREDICT_NESCAPE)
          : dflt);
      double ent = 0;
      if (NULL == ctx) {
        int e;
        for (e = 0; e < LIBXS_PREDICT_NESCAPE; ++e) {
          dflt[e] = 1.0 / LIBXS_PREDICT_NESCAPE;
        }
      }
      internal_libxs_predict_eval_ex(NULL, model, inputs, NULL, NULL,
        nblend, src, smode, sout, snout, NULL);
      /**
       * Nothing to report but the outcome to learn from: the caller asked for no
       * values, no probs, no novel and no info, so the distribution is never
       * read and only the bank update needs anything. That update reads the
       * local evidence at ONE index, so the sparse form computes the same
       * weights in O(k log n) instead of O(NESCAPE * n). This is the warm-up
       * shape - converge the bank over a training split - which at a large
       * support was the dominant cost of getting an order-independent figure.
       */
      if (LIBXS_PREDICT_SRC_CLASSIFY == smode[output] && NULL != src[output]
        && NULL != ctx && NULL != candidate && NULL == values && NULL == probs
        && NULL == novel && NULL == info)
      {
        const int ns = model->sup_n[output];
        const double cand = (NULL != model->transforms)
          ? internal_libxs_predict_fwd(model->transforms[output], *candidate)
          : *candidate;
        double local_truth = 0;
        if (EXIT_SUCCESS == internal_libxs_predict_point(model, w, src[output],
          inputs, output, sout[output], snout[output], 0, vocabulary, cand,
          NULL, NULL, NULL, &local_truth))
        {
          const int si = internal_libxs_predict_support_index(
            model->sup_vals[output], ns, cand);
          result = ns;
          internal_libxs_predict_dist_learn_at(model, w, output, local_truth,
            si, vocabulary);
        }
      }
      else if (LIBXS_PREDICT_SRC_CLASSIFY == smode[output] && NULL != src[output]
        && EXIT_SUCCESS == internal_libxs_predict_dist(model, w, src[output],
          inputs, output, sout[output], snout[output], 0, vocabulary, p,
          scratch, stride, &ent))
      {
        const int ns = model->sup_n[output];
        const int outside = ((vocabulary > ns) ? vocabulary : ns) - ns;
        const double* ev = scratch + stride;
        double cand = 0;
        int si = -1;
        result = ns;
        if (NULL != candidate) {
          cand = (NULL != model->transforms)
            ? internal_libxs_predict_fwd(model->transforms[output], *candidate)
            : *candidate;
          si = internal_libxs_predict_support_index(
            model->sup_vals[output], ns, cand);
        }
        /* everything reported is taken before the bank is touched */
        if (ns <= capacity && NULL != values && NULL != probs) {
          memcpy(values, model->sup_vals[output],
            (size_t)ns * sizeof(double));
          memcpy(probs, p, (size_t)ns * sizeof(double));
        }
        if (NULL != novel) *novel = p[ns];
        LIBXS_ASSERT(1.0 == libxs_sum2(p, ns + 1));
        if (NULL != info) {
          double* pr = internal_libxs_predict_ctx_scratch(ctx) + 3 * stride;
          double* lp = pr + n;
          double* zs = lp + n;
          double* nv = zs + n;
          int* kind = (int*)(nv + n);
          int* att = kind + n;
          int* sup = att + n;
          int j;
          for (j = 0; j < n; ++j) {
            pr[j] = 0; lp[j] = -DBL_MAX; zs[j] = 0; nv[j] = 0;
            kind[j] = LIBXS_PREDICT_PNONE; att[j] = 0;
            sup[j] = model->sup_n[j];
          }
          nv[output] = p[ns];
          kind[output] = LIBXS_PREDICT_PMASS;
          if (NULL != candidate) {
            if (0 <= si) {
              pr[output] = p[si];
              att[output] = (0 != ev[si]) ? 1 : 0;
            }
            else if (0 < outside) pr[output] = p[ns] / outside;
            lp[output] = (pr[output] > 0)
              ? (log(pr[output]) / log(2.0)) : -DBL_MAX;
          }
          info->prob = pr;
          info->logprob = lp;
          info->zscore = zs;
          info->kind = kind;
          info->attested = att;
          info->support = sup;
          info->novel = nv;
          info->total_logprob = lp[output];
          info->noutputs = n;
          info->cluster = -1;
          info->entropy = ent;
        }
        /* only now may the weights move */
        if (NULL != candidate && NULL != ctx) {
          internal_libxs_predict_dist_learn(model, w, output, ev, si,
            vocabulary);
        }
      }
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
    if (NULL == ctx) LIBXS_PREDICT_FREE((void*)src, scratch_pool);
  }
  return result;
}


LIBXS_API int libxs_predict_prob_support(const libxs_predict_t* model,
  int output, double values[], int capacity)
{
  int result = 0;
  if (NULL != model && 0 != model->built && 0 <= output
    && output < model->noutputs && NULL != model->sup_n
    && NULL != model->sup_vals && NULL != model->sup_vals[output])
  {
    result = model->sup_n[output];
    if (NULL != values && result <= capacity) {
      memcpy(values, model->sup_vals[output],
        (size_t)result * sizeof(double));
    }
  }
  return result;
}


LIBXS_API void libxs_predict_query(
  const libxs_predict_t* model, libxs_predict_query_t* info)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != info);
  { const double raw = (double)model->nentries * (model->ninputs + model->noutputs);
    double compressed = 0;
    int c;
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      compressed += model->ninputs;
      /**
       * Per-output polynomials exist only where the model kind has them: an
       * hkNN model predicts from neighbours, so its clusters carry no order
       * array at all (nor does its serialized form store one). Clusters left
       * empty by the builder are likewise not populated. Counting only the
       * centroid for those keeps the ratio meaningful instead of dereferencing
       * a NULL that a polynomial model would always have provided.
       */
      if (NULL != cl->order) {
        int j;
        for (j = 0; j < model->noutputs; ++j) {
          compressed += cl->order[j] + 1;
        }
      }
    }
    info->compression = (compressed > 0) ? (raw / compressed) : 0;
  }
  info->order = model->order;
  info->nclusters = model->nclusters;
  info->nentries = model->nentries;
  info->iterations = model->iterations;
  info->diff_order = model->diff_order;
  info->window = (model->nseries > 0) ? model->window : 0;
  info->nbank = (0 < model->nbank) ? model->nbank : 1;
  info->decompose = model->decompose;
  { double sqsum = 0;
    int c;
    info->nscan = 0;
    for (c = 0; c < model->nclusters; ++c) {
      const double n = (double)model->clusters[c].nentries;
      if (info->nscan < model->clusters[c].nentries) {
        info->nscan = model->clusters[c].nentries;
      }
      sqsum += n * n;
    }
    info->escan = (0 < model->nentries)
      ? (sqsum / (double)model->nentries) : 0.0;
  }
}


LIBXS_API void libxs_predict_get(const libxs_predict_t* model, int index,
  double inputs[], double outputs[])
{
  LIBXS_ASSERT(NULL != model && 0 <= index && index < model->nentries);
  if (NULL != model->entries) {
    if (NULL != inputs) {
      memcpy(inputs, model->entries[index].inputs, (size_t)model->ninputs * sizeof(double));
    }
    if (NULL != outputs) {
      memcpy(outputs, model->entries[index].outputs, (size_t)model->noutputs * sizeof(double));
    }
  }
  else {
    /**
     * The index is the position the entry was pushed at, and the partition
     * does not preserve that order: sorted_idx carries it, so the entry is
     * looked up through it rather than by counting cluster sizes. Counting
     * returned whichever entry sat at that position in cluster order, which is
     * a different entry on any model whose clusters reordered the corpus.
     * Where sorted_idx is absent the position cannot be recovered at all, and
     * cluster order is the only thing left to answer with.
     */
    int c, offset = 0, found = 0;
    for (c = 0; c < model->nclusters && 0 == found; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      int local = -1;
      if (NULL != cl->sorted_idx) {
        int k;
        for (k = 0; k < cl->nentries; ++k) {
          if (cl->sorted_idx[k] == index) { local = k; break; }
        }
      }
      else if (index < offset + cl->nentries) {
        local = index - offset;
      }
      if (0 <= local) {
        found = 1;
        if (NULL != inputs) {
          const double* pt = cl->kd_pts + (size_t)local * model->ninputs;
          internal_libxs_predict_denormalize(model, pt, inputs);
          if (NULL != model->decompose_mat) {
            int tmp_pool = 0;
            double* tmp = (double*)LIBXS_PREDICT_MALLOC(
              (size_t)model->ninputs * sizeof(double), tmp_pool);
            if (NULL != tmp) {
              memcpy(tmp, inputs, (size_t)model->ninputs * sizeof(double));
              internal_libxs_predict_decompose_inverse(model, tmp, inputs);
              LIBXS_PREDICT_FREE(tmp, tmp_pool);
            }
          }
        }
        if (NULL != outputs) {
          memcpy(outputs, cl->raw_outputs + (size_t)local * model->noutputs,
            (size_t)model->noutputs * sizeof(double));
        }
      }
      offset += cl->nentries;
    }
  }
}


#include "libxs_predict_serial.h"


#include "libxs_predict_csv.h"
