/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#if !defined(LIBXS_PREDICT_HKNN_MINLEAF)
#  define LIBXS_PREDICT_HKNN_MINLEAF 0
#endif
/**
 * Entries a node needs before its features are scored by the team rather than
 * by the builder alone, each such node costing two rendezvous. The small nodes
 * are not negligible: every level of the tree holds the whole corpus, so the
 * levels below a threshold of 1024 left the crystal build at 0.93 s where 256
 * reaches 0.53 s. Lower gained nothing, the leaves rarely being much smaller.
 */
#if !defined(LIBXS_PREDICT_HKNN_PARNODE)
#  define LIBXS_PREDICT_HKNN_PARNODE 256
#endif


/** One node as its features are scored: everything but the feature itself. */
LIBXS_EXTERN_C typedef struct internal_libxs_predict_hknn_node_t {
  const int* idx;
  const int* output_groups;
  double classes[128];
  int count, band_lo, band_hi, use_gini, nclasses;
  /** Output the classes are of, and the fewest entries a purity split leaves on a side. */
  int gini_out, gini_min;
  int oi_lo, oi_hi, target_group;
} internal_libxs_predict_hknn_node_t;

/**
 * The team scoring the nodes of a partition: the node the builder published,
 * and per task its scratch and the best split among the features it took.
 * Workers read the node between a release and the collect that follows, and
 * the builder writes it only between a collect and the next release.
 */
LIBXS_EXTERN_C typedef struct internal_libxs_predict_hknn_team_t {
  internal_libxs_predict_hknn_node_t node;
  internal_libxs_predict_rf_pair_t** pairs;
  double** sums;
  double* score;
  int* feat;
  int* pos;
  /* how each task's pairs and sums were allocated, pairs first */
  int* pool;
  int nwork;
  volatile int op;
} internal_libxs_predict_hknn_team_t;

LIBXS_EXTERN_C typedef struct internal_libxs_predict_hknn_split_ctx_t {
  libxs_predict_t* model;
  internal_libxs_predict_rf_pair_t* pairs;
  double* sums;
  libxs_barrier_t* barrier;
  internal_libxs_predict_hknn_team_t* team;
  int target_nc;
  int ntotal;
  int target_output;
  int target_group;
  const int* output_groups;
} internal_libxs_predict_hknn_split_ctx_t;


/**
 * Score one feature of a node: sort the node by it and scan every boundary
 * between distinct values. The best boundary is the first of the highest score,
 * as a strict comparison over ascending positions keeps it. sums holds four
 * accumulators per output for the Fisher criterion.
 */
LIBXS_API_INLINE void internal_libxs_predict_hknn_feature(
  const libxs_predict_t* model, const internal_libxs_predict_hknn_node_t* node,
  int j, internal_libxs_predict_rf_pair_t* pairs, double* sums,
  double* out_score, int* out_pos)
{
  const int n = model->noutputs;
  const int count = node->count;
  const int* idx = node->idx;
  double best_score = -1;
  int best_pos = -1, i;
  for (i = 0; i < count; ++i) {
    pairs[i].val = model->entries[idx[i]].inputs[j];
    pairs[i].idx = idx[i];
  }
  libxs_sort(pairs, count, sizeof(pairs[0]),
    internal_libxs_predict_rf_pair_cmp, NULL);
  if (0 != node->use_gini) {
    const int nclasses = node->nclasses;
    int cnt_all[128], cnt_left[128], ci;
    double gini_parent = 1.0;
    memset(cnt_all, 0, (size_t)nclasses * sizeof(int));
    for (i = 0; i < count; ++i) {
      const double v = model->entries[pairs[i].idx].outputs[node->gini_out];
      for (ci = 0; ci < nclasses; ++ci) {
        if (node->classes[ci] == v) { ++cnt_all[ci]; break; }
      }
    }
    for (ci = 0; ci < nclasses; ++ci) {
      const double p_k = (double)cnt_all[ci] / count;
      gini_parent -= p_k * p_k;
    }
    memset(cnt_left, 0, (size_t)nclasses * sizeof(int));
    for (i = 0; i < count - 1; ++i) {
      const int nleft = i + 1, nright = count - nleft;
      const double v = model->entries[pairs[i].idx].outputs[node->gini_out];
      for (ci = 0; ci < nclasses; ++ci) {
        if (node->classes[ci] == v) { ++cnt_left[ci]; break; }
      }
      if (pairs[i].val != pairs[i + 1].val
        && nleft >= node->gini_min && nright >= node->gini_min)
      {
        double gini_l = 1.0, gini_r = 1.0, gain;
        for (ci = 0; ci < nclasses; ++ci) {
          const double pl = (double)cnt_left[ci] / nleft;
          const double pr = (double)(cnt_all[ci] - cnt_left[ci]) / nright;
          gini_l -= pl * pl;
          gini_r -= pr * pr;
        }
        gain = gini_parent
          - (double)nleft / count * gini_l
          - (double)nright / count * gini_r;
        { const double score = gain * (double)nclasses;
          if (score > best_score) {
            best_score = score;
            best_pos = i;
          }
        }
      }
    }
  }
  else {
    double* sum_all = sums;
    double* sum2_all = sums + n;
    double* sum_left = sums + 2 * n;
    double* sum2_left = sums + 3 * n;
    int oi;
    memset(sum_all, 0, (size_t)n * sizeof(double));
    memset(sum2_all, 0, (size_t)n * sizeof(double));
    for (i = 0; i < count; ++i) {
      for (oi = node->oi_lo; oi < node->oi_hi; ++oi) {
        if (NULL != node->output_groups
          && node->output_groups[oi] != node->target_group)
        {
          continue;
        }
        { const double v = model->entries[pairs[i].idx].outputs[oi];
          sum_all[oi] += v;
          sum2_all[oi] += v * v;
        }
      }
    }
    memset(sum_left, 0, (size_t)n * sizeof(double));
    memset(sum2_left, 0, (size_t)n * sizeof(double));
    for (i = 0; i < count - 1; ++i) {
      const int nleft = i + 1, nright = count - nleft;
      for (oi = node->oi_lo; oi < node->oi_hi; ++oi) {
        if (NULL != node->output_groups
          && node->output_groups[oi] != node->target_group)
        {
          continue;
        }
        { const double v = model->entries[pairs[i].idx].outputs[oi];
          sum_left[oi] += v;
          sum2_left[oi] += v * v;
        }
      }
      if (pairs[i].val != pairs[i + 1].val
        && nleft >= node->band_lo && nleft <= node->band_hi)
      {
        double fisher = 0;
        const double penalty =
          1.0 + 4.0 * LIBXS_FABS((double)nleft / count - 0.5);
        for (oi = node->oi_lo; oi < node->oi_hi; ++oi) {
          if (NULL != node->output_groups
            && node->output_groups[oi] != node->target_group)
          {
            continue;
          }
          { const double ml = sum_left[oi] / nleft;
            const double mr = (sum_all[oi] - sum_left[oi]) / nright;
            const double vl = sum2_left[oi] / nleft - ml * ml;
            const double vr = (sum2_all[oi] - sum2_left[oi]) / nright
              - mr * mr;
            const double within = vl * nleft + vr * nright;
            const double between = (double)nleft * nright
              * (ml - mr) * (ml - mr) / count;
            if (within > 0) fisher += between / within;
          }
        }
        { const double score = fisher / penalty;
          if (score > best_score) {
            best_score = score;
            best_pos = i;
          }
        }
      }
    }
  }
  *out_score = best_score;
  *out_pos = best_pos;
}


/**
 * The features a task scores, taken strided so that each task visits its own in
 * ascending order and keeps the first of its best, as the serial loop did.
 */
LIBXS_API_INLINE void internal_libxs_predict_hknn_features(
  const libxs_predict_t* model, const internal_libxs_predict_hknn_node_t* node,
  int first, int stride, internal_libxs_predict_rf_pair_t* pairs, double* sums,
  double* out_score, int* out_feat, int* out_pos)
{
  double best_score = -1;
  int best_feat = -1, best_pos = -1, j;
  for (j = first; j < model->ninputs; j += stride) {
    double score;
    int pos;
    internal_libxs_predict_hknn_feature(model, node, j, pairs, sums,
      &score, &pos);
    if (0 <= pos && score > best_score) {
      best_score = score;
      best_feat = j;
      best_pos = pos;
    }
  }
  *out_score = best_score;
  *out_feat = best_feat;
  *out_pos = best_pos;
}


/**
 * A worker's part of a partition: score the features it takes of every node the
 * builder publishes, until the builder calls the partition done.
 */
LIBXS_API_INLINE void internal_libxs_predict_hknn_work(
  libxs_barrier_t* barrier, libxs_predict_t* model, int tid)
{
  internal_libxs_predict_hknn_team_t* team = model->sync_hknn;
  int op;
  do {
    libxs_barrier_wait(barrier);
    op = (int)LIBXS_ATOMIC_LOAD(&team->op, LIBXS_ATOMIC_SEQ_CST);
    if (1 == op) {
      if (tid < team->nwork) {
        internal_libxs_predict_hknn_features(model, &team->node, tid,
          team->nwork, team->pairs[tid], team->sums[tid],
          team->score + tid, team->feat + tid, team->pos + tid);
      }
      libxs_barrier_wait(barrier);
    }
  } while (1 == op);
}


LIBXS_API_INLINE int internal_libxs_predict_hknn_split(
  int* dim, int* pos, double* threshold, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx)
{
  internal_libxs_predict_hknn_split_ctx_t* state =
    (internal_libxs_predict_hknn_split_ctx_t*)ctx;
  const libxs_predict_t* model = state->model;
  const int n = model->noutputs;
  internal_libxs_predict_rf_pair_t* pairs = state->pairs;
  internal_libxs_predict_hknn_team_t* team = state->team;
  const double denom = (double)count * (1 << (depth < 20 ? depth : 20));
  const double imbal = LIBXS_MAX((double)state->ntotal / denom, 1.0);
  const int ideal_half = count / 2;
  const double allowed_dev = 0.22 / LIBXS_MAX(imbal, 0.5);
  const int min_leaf = LIBXS_MAX(
    state->ntotal * 2 / (state->target_nc * 3), 3);
  const double progress = (double)nleaves / state->target_nc;
  const double score_floor = (progress > 0.8)
    ? (progress - 0.8) * (progress - 0.8) * 25.0 : 0.0;
  internal_libxs_predict_hknn_node_t node_local;
  internal_libxs_predict_hknn_node_t* node = (NULL != team)
    ? &team->node : &node_local;
  double best_score = -1;
  int best_feat = -1, best_pos = -1, i, result = 1;
  LIBXS_UNUSED(pts);
  node->idx = idx;
  node->count = count;
  node->output_groups = state->output_groups;
  node->target_group = state->target_group;
  node->band_lo = LIBXS_MAX((int)(ideal_half - count * allowed_dev), min_leaf);
  node->band_hi = LIBXS_MIN((int)(ideal_half + count * allowed_dev),
    count - min_leaf);
  node->use_gini = 0;
  node->nclasses = 0;
  node->gini_out = 0;
  node->gini_min = 1;
  if (state->target_output >= 0) {
    node->oi_lo = state->target_output;
    node->oi_hi = state->target_output + 1;
  }
  else {
    node->oi_lo = 0;
    node->oi_hi = n;
  }
  if (node->band_lo <= node->band_hi) {
    /**
     * Purity where the output is a category: a single output's classes, or the
     * one discrete output a tree of an output is grown for. That tree splits
     * where the output changes, wherever in the node that is, rather than near
     * the median as the balanced criterion must; a tile size changes value at a
     * few shapes, and that matched 0.6 points more of the tuned parameters of
     * GPU kernels than the balanced split.
     */
    int go = (1 == n) ? 0 : -1;
    if (1 < n && 0 <= state->target_group && NULL != state->output_groups
      && NULL != model->discrete)
    {
      int oi, nin = 0;
      for (oi = 0; oi < n; ++oi) {
        if (state->output_groups[oi] == state->target_group) { go = oi; ++nin; }
      }
      if (1 != nin || 0 == model->discrete[go]) go = -1;
    }
    if (0 <= go) {
      for (i = 0; i < count && node->nclasses < 128; ++i) {
        const double v = model->entries[idx[i]].outputs[go];
        int found = 0, ci;
        for (ci = 0; ci < node->nclasses; ++ci) {
          if (node->classes[ci] == v) { found = 1; break; }
        }
        if (0 == found) node->classes[node->nclasses++] = v;
      }
      node->gini_out = go;
      if (1 == n) {
        if (node->nclasses > 1 && count > node->nclasses) node->use_gini = 1;
      }
      else if (node->nclasses > 1) {
        node->use_gini = 1;
        node->gini_min = min_leaf;
      }
    }
    if (NULL != team && LIBXS_PREDICT_HKNN_PARNODE <= count) {
      int t;
      LIBXS_ATOMIC_STORE(&team->op, 1, LIBXS_ATOMIC_SEQ_CST);
      libxs_barrier_wait(state->barrier);
      internal_libxs_predict_hknn_features(model, node, 0, team->nwork,
        team->pairs[0], team->sums[0], team->score, team->feat, team->pos);
      libxs_barrier_wait(state->barrier);
      /* the first of the highest, and the lowest feature among equals */
      for (t = 0; t < team->nwork; ++t) {
        if (0 <= team->feat[t] && (team->score[t] > best_score
          || (team->score[t] == best_score && team->feat[t] < best_feat)))
        {
          best_score = team->score[t];
          best_feat = team->feat[t];
          best_pos = team->pos[t];
        }
      }
    }
    else {
      internal_libxs_predict_hknn_features(model, node, 0, 1, pairs,
        state->sums, &best_score, &best_feat, &best_pos);
    }
    /* a purity split of a tree of an output must purify: no gain, no split */
    if (best_feat >= 0 && ((0 != node->use_gini)
      ? (1 == n || 0 < best_score) : (best_score >= score_floor)))
    {
      for (i = 0; i < count; ++i) {
        pairs[i].val = model->entries[idx[i]].inputs[best_feat];
        pairs[i].idx = idx[i];
      }
      libxs_sort(pairs, count, sizeof(pairs[0]),
        internal_libxs_predict_rf_pair_cmp, NULL);
      for (i = 0; i < count; ++i) idx[i] = pairs[i].idx;
      *dim = best_feat;
      *pos = best_pos + 1;
      /* midway between the two values the boundary separates (inputs, not pts) */
      *threshold = 0.5 * (model->entries[idx[best_pos]].inputs[best_feat]
        + model->entries[idx[best_pos + 1]].inputs[best_feat]);
      result = 0;
    }
  }
  return result;
}


/**
 * The team's scratch, one set per task that scores: a pair buffer as large as
 * the corpus, because the root is a node too, and the Fisher accumulators. At
 * most one task per feature takes part, since a feature is the unit of work,
 * which also bounds what this costs.
 */
LIBXS_API_INLINE int internal_libxs_predict_hknn_team_create(
  libxs_predict_t* model, int ntasks)
{
  const int nwork = LIBXS_MIN(ntasks, model->ninputs);
  internal_libxs_predict_hknn_team_t* team =
    (internal_libxs_predict_hknn_team_t*)calloc(1, sizeof(*team));
  int ok = (NULL != team && 1 < nwork) ? 1 : 0, t;
  if (0 != ok) {
    team->nwork = nwork;
    team->pairs = (internal_libxs_predict_rf_pair_t**)calloc(
      (size_t)nwork, sizeof(*team->pairs));
    team->sums = (double**)calloc((size_t)nwork, sizeof(*team->sums));
    team->score = (double*)malloc((size_t)nwork * sizeof(double));
    team->feat = (int*)malloc((size_t)nwork * sizeof(int));
    team->pos = (int*)malloc((size_t)nwork * sizeof(int));
    team->pool = (int*)calloc((size_t)2 * nwork, sizeof(int));
    ok = (NULL != team->pairs && NULL != team->sums && NULL != team->score
      && NULL != team->feat && NULL != team->pos && NULL != team->pool)
      ? 1 : 0;
    for (t = 0; t < nwork && 0 != ok; ++t) {
      team->pairs[t] = (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
        (size_t)model->nentries * sizeof(internal_libxs_predict_rf_pair_t),
        team->pool[t]);
      team->sums[t] = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)4 * model->noutputs * sizeof(double), team->pool[nwork + t]);
      if (NULL == team->pairs[t] || NULL == team->sums[t]) ok = 0;
    }
  }
  model->sync_hknn = team;
  if (0 == ok) internal_libxs_predict_hknn_team_free(model);
  return ok;
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_team_free(
  libxs_predict_t* model)
{
  internal_libxs_predict_hknn_team_t* team = model->sync_hknn;
  if (NULL != team) {
    int t;
    if (NULL != team->pool) {
      for (t = 0; t < team->nwork; ++t) {
        if (NULL != team->pairs) {
          LIBXS_PREDICT_FREE(team->pairs[t], team->pool[t]);
        }
        if (NULL != team->sums) {
          LIBXS_PREDICT_FREE(team->sums[t], team->pool[team->nwork + t]);
        }
      }
    }
    free(team->pool);
    free(team->pairs);
    free(team->sums);
    free(team->score);
    free(team->feat);
    free(team->pos);
    free(team);
    model->sync_hknn = NULL;
  }
}


/**
 * Partition the corpus by kd-tree, splitting each node on the feature and
 * boundary that best separate its outputs. Called by the builder; where the
 * model carries a team, the other tasks score the features of the large nodes
 * with it and the barrier is theirs.
 */
LIBXS_API_INLINE void internal_libxs_predict_hknn_partition(
  libxs_predict_t* model, int* nclusters_out, libxs_barrier_t* barrier)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  const int target_nc = (int)(sqrt((double)p) + 0.5);
  const int min_leaf = (0 < LIBXS_PREDICT_HKNN_MINLEAF)
    ? LIBXS_PREDICT_HKNN_MINLEAF
    : ((1 == n) ? 1 : LIBXS_MAX(p * 2 / (target_nc * 3), 3));
  int* const out_assign = model->hknn_assignments;
  int pairs_pool = 0, order_pool = 0, pts_pool = 0, sums_pool = 0;
  double* sums = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)4 * n * sizeof(double), sums_pool);
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  int* order = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(int), order_pool);
  double* pts = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * (size_t)m * sizeof(double), pts_pool);
  int nleaves = 0;
  if (NULL != pairs && NULL != order && NULL != pts && NULL != sums) {
    internal_libxs_predict_hknn_split_ctx_t state;
    libxs_kdtree_config_t config;
    int i, j;
    for (i = 0; i < p; ++i) {
      for (j = 0; j < m; ++j) {
        pts[(size_t)i * m + j] = model->entries[i].inputs[j];
      }
    }
    state.model = model;
    state.pairs = pairs;
    state.sums = sums;
    state.barrier = barrier;
    state.team = model->sync_hknn;
    state.target_nc = target_nc;
    state.ntotal = p;
    state.target_output = -1;
    state.target_group = -1;
    state.output_groups = NULL;
    config.min_leaf = min_leaf;
    config.split = internal_libxs_predict_hknn_split;
    config.ctx = &state;
    config.nodes = NULL;
    if (n > 1) {
      /**
       * One tree per output. Grouping outputs whose Cramer's V reached 0.5 read
       * an output with more values than its 64-entry table as category 0, which
       * put a real-valued throughput into the tree of a tile size; corrected, it
       * still matched 0.5 points fewer of the tuned parameters of GPU kernels
       * than a tree per output.
       */
      int* groups = (int*)calloc((size_t)n, sizeof(int));
      int ngroups = n, g;
      if (NULL != groups) {
        int gi;
        for (gi = 0; gi < n; ++gi) groups[gi] = gi;
        model->hknn_po_groups = groups;
        model->hknn_ngroups = ngroups;
      }
      else {
        ngroups = n;
      }
      model->hknn_po_assignments = (int**)calloc(
        (size_t)ngroups, sizeof(int*));
      model->hknn_po_nclusters = (int*)calloc(
        (size_t)ngroups, sizeof(int));
      model->hknn_po_trees = (internal_libxs_predict_hknn_tree_t*)calloc(
        (size_t)ngroups, sizeof(internal_libxs_predict_hknn_tree_t));
      if (NULL != model->hknn_po_assignments
        && NULL != model->hknn_po_nclusters)
      {
        state.output_groups = groups;
        for (g = 0; g < ngroups; ++g) {
          model->hknn_po_assignments[g] = (int*)calloc(
            (size_t)p, sizeof(int));
          if (NULL != model->hknn_po_assignments[g]) {
            /* the partition writes the tree it grows, at most 2p-1 nodes; a
             * group without one routes by its nearest entry */
            internal_libxs_predict_hknn_tree_t* const tree = (NULL != model->hknn_po_trees)
              ? (model->hknn_po_trees + g) : NULL;
            if (NULL != tree && 0 < p) {
              tree->nodes = (libxs_kdtree_node_t*)malloc(
                (size_t)(2 * p - 1) * sizeof(libxs_kdtree_node_t));
            }
            state.target_output = -1;
            state.target_group = g;
            for (i = 0; i < p; ++i) order[i] = i;
            config.nodes = (NULL != tree) ? tree->nodes : NULL;
            model->hknn_po_nclusters[g] = libxs_kdtree_partition(
              pts, order, p, m, m,
              model->hknn_po_assignments[g], &config);
            if (NULL != tree && NULL != tree->nodes) {
              tree->nnodes = LIBXS_MAX(2 * model->hknn_po_nclusters[g] - 1, 0);
            }
          }
        }
        config.nodes = NULL;
        state.output_groups = NULL;
        state.target_group = -1;
      }
      state.target_output = -1;
    }
    for (i = 0; i < p; ++i) order[i] = i;
    nleaves = libxs_kdtree_partition(pts, order, p, m, m,
      out_assign, &config);
  }
  *nclusters_out = LIBXS_MAX(nleaves, 1);
  LIBXS_PREDICT_FREE(pts, pts_pool);
  LIBXS_PREDICT_FREE(order, order_pool);
  LIBXS_PREDICT_FREE(pairs, pairs_pool);
  LIBXS_PREDICT_FREE(sums, sums_pool);
}


/*
 * tid/ntasks as internal_libxs_predict_kmeans: the assignment step is split
 * across the tasks, moving the centroids is the builder's
 */
LIBXS_API_INLINE void internal_libxs_predict_hknn_refine(libxs_barrier_t* barrier,
  libxs_predict_t* model, int nclusters, int tid, int ntasks)
{
  const int p = model->nentries;
  const int m = model->ncoord; /* coordinates, not inputs */
  const int max_iter = LIBXS_MIN(LIBXS_PREDICT_MAXITER, 10);
  int cnt_pool = 0, cen_pool = 0;
  const double* pts;
  double* comp;
  int* counts = NULL;
  if (0 == tid) {
    /* see internal_libxs_predict_kmeans: one buffer, built by the builder */
    internal_libxs_predict_normpts(model);
    LIBXS_ASSERT(NULL == model->norm_cen);
    model->norm_cen = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)nclusters * (size_t)m * sizeof(double), cen_pool);
    counts = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)nclusters * sizeof(int), cnt_pool);
    model->sync_moved = 0;
    if (NULL == counts) {
      LIBXS_PREDICT_FREE(model->norm_cen, cen_pool);
      model->norm_cen = NULL;
    }
  }
  libxs_barrier_wait(barrier);
  /**
   * The builder's scratch also tells every task whether the step can run:
   * the condition has to be shared, or the tasks part company at a barrier
   */
  pts = model->norm_pts;
  comp = model->norm_cen;
  if (NULL != pts && NULL != comp) {
    int i, c, j, iter;
    for (iter = 0; iter < max_iter; ++iter) {
      int changed = 0;
      /* no bounds here: this starts near converged, too few passes to amortize */
      if (0 == tid) model->sync_moved = 0;
      libxs_barrier_wait(barrier);
      for (i = tid; i < p; i += ntasks) {
        double best = libxs_dist2(
          pts + (size_t)i * m, model->clusters[0].centroid, m);
        int bestc = 0;
        for (c = 1; c < nclusters; ++c) {
          const double d = libxs_dist2(
            pts + (size_t)i * m, model->clusters[c].centroid, m);
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
      libxs_barrier_wait(barrier);
      changed = (int)LIBXS_ATOMIC_LOAD(&model->sync_moved, LIBXS_ATOMIC_SEQ_CST);
      if (0 == changed) iter = max_iter;
      else {
        if (0 == tid) {
          memset(comp, 0, (size_t)nclusters * (size_t)m * sizeof(double));
          memset(counts, 0, (size_t)nclusters * sizeof(int));
          for (c = 0; c < nclusters; ++c) {
            memset(model->clusters[c].centroid, 0, (size_t)m * sizeof(double));
          }
          for (i = 0; i < p; ++i) {
            const int ci = model->assignments[i];
            double* cen = model->clusters[ci].centroid;
            double* cmp = comp + (size_t)ci * m;
            for (j = 0; j < m; ++j) {
              libxs_kahan_sum(pts[(size_t)i * m + j], &cen[j], &cmp[j]);
            }
            ++counts[ci];
          }
          for (c = 0; c < nclusters; ++c) {
            if (counts[c] > 0) {
              for (j = 0; j < m; ++j) {
                model->clusters[c].centroid[j] /= counts[c];
              }
            }
          }
        } /* moving the centroids is the builder's */
        libxs_barrier_wait(barrier);
      }
    }
  }
  if (0 == tid) {
    LIBXS_PREDICT_FREE(counts, cnt_pool);
    LIBXS_PREDICT_FREE(model->norm_cen, cen_pool);
    model->norm_cen = NULL;
  }
  /* the partition is complete for every task, not just the one that closed it */
  libxs_barrier_wait(barrier);
}


LIBXS_API_INLINE int internal_libxs_predict_hknn_build_po(
  libxs_predict_t* model, int order)
{
  const int p = model->nentries;
  const int m = model->ncoord; /* coordinates, not inputs */
  const int n = model->noutputs;
  const int ngroups = (model->hknn_ngroups > 0) ? model->hknn_ngroups : n;
  int result = EXIT_SUCCESS;
  if (NULL == model->hknn_po_assignments
    || NULL == model->hknn_po_nclusters || n <= 1)
  {
    result = EXIT_SUCCESS;
  }
  else {
    model->hknn_po_clusters = (internal_libxs_predict_cluster_t**)calloc(
      (size_t)ngroups, sizeof(internal_libxs_predict_cluster_t*));
    if (NULL == model->hknn_po_clusters) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && NULL != model->hknn_po_clusters) {
    int gi;
    for (gi = 0; gi < ngroups && EXIT_SUCCESS == result; ++gi) {
      const int* assign = model->hknn_po_assignments[gi];
      const int nc = model->hknn_po_nclusters[gi];
      int outs_pool = 0, gsz = 0, oi;
      int* const outs = (int*)LIBXS_PREDICT_MALLOC((size_t)n * sizeof(int), outs_pool);
      internal_libxs_predict_cluster_t* cls;
      int c, i, k;
      /* the outputs of the group, the columns its clusters hold */
      for (oi = 0; oi < n && NULL != outs; ++oi) {
        if ((NULL != model->hknn_po_groups && model->hknn_po_groups[oi] == gi)
          || (NULL == model->hknn_po_groups && oi == gi))
        {
          outs[gsz++] = oi;
        }
      }
      if (NULL == outs) { result = EXIT_FAILURE; break; }
      if (NULL == assign || nc < 1 || 0 >= gsz) {
        LIBXS_PREDICT_FREE(outs, outs_pool);
        continue;
      }
      cls = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nc, sizeof(internal_libxs_predict_cluster_t));
      if (NULL == cls) {
        LIBXS_PREDICT_FREE(outs, outs_pool);
        result = EXIT_FAILURE;
        break;
      }
      model->hknn_po_clusters[gi] = cls;
      for (c = 0; c < nc && EXIT_SUCCESS == result; ++c) {
        cls[c].centroid = (double*)calloc((size_t)m, sizeof(double));
        if (NULL == cls[c].centroid) result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        int norm_pool = 0, counts_pool = 0;
        double* norm = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)m * sizeof(double), norm_pool);
        int* counts = (int*)LIBXS_PREDICT_MALLOC(
          (size_t)nc * sizeof(int), counts_pool);
        if (NULL != norm && NULL != counts) {
          memset(counts, 0, (size_t)nc * sizeof(int));
          for (i = 0; i < p; ++i) {
            const int ci = assign[i];
            internal_libxs_predict_normalize(model,
              model->entries[i].inputs, norm);
            for (k = 0; k < m; ++k) cls[ci].centroid[k] += norm[k];
            ++counts[ci];
          }
          for (c = 0; c < nc; ++c) {
            cls[c].nentries = counts[c];
            if (counts[c] > 0) {
              for (k = 0; k < m; ++k) cls[c].centroid[k] /= counts[c];
            }
          }
        }
        /* the clusters would otherwise go on empty, as if nothing were assigned */
        else result = EXIT_FAILURE;
        LIBXS_PREDICT_FREE(counts, counts_pool);
        LIBXS_PREDICT_FREE(norm, norm_pool);
      }
      for (c = 0; c < nc && EXIT_SUCCESS == result; ++c) {
        const int nce = cls[c].nentries;
        int ki = 0;
        if (0 >= nce) continue;
        cls[c].sorted_idx = (int*)malloc((size_t)nce * sizeof(int));
        if (NULL == cls[c].sorted_idx) result = EXIT_FAILURE;
        for (i = 0; i < p && EXIT_SUCCESS == result; ++i) {
          if (assign[i] == c) cls[c].sorted_idx[ki++] = i;
        }
        if (EXIT_SUCCESS == result) {
          result = internal_libxs_predict_cluster_assemble(model, cls + c,
            order, outs, gsz);
        }
      }
      LIBXS_PREDICT_FREE(outs, outs_pool);
    }
  }
  return result;
}


/**
 * The per-output cluster of group g a query belongs to: its leaf in the group's
 * tree. Routing it by the cluster of its nearest entry instead put a query near
 * a boundary into a cell its own neighbours were not in, and descending the tree
 * matched 2.4 points more of the tuned parameters of GPU kernels. That routing
 * remains where no tree is known, where the descent reads an absent input, and
 * for inputs NULL: a temporal model asks the future of thresholds learned on the
 * past, and descending them raised the error of a forecast by 2 to 7%.
 */
LIBXS_API_INLINE int internal_libxs_predict_hknn_route(const libxs_predict_t* model,
  int g, const double inputs[], int nearest_entry)
{
  int result = (0 <= nearest_entry && nearest_entry < model->nentries)
    ? model->hknn_po_assignments[g][nearest_entry] : 0;
  if (NULL != inputs && NULL != model->hknn_po_trees
    && 0 < model->hknn_po_trees[g].nnodes)
  {
    const libxs_kdtree_node_t* const nodes = model->hknn_po_trees[g].nodes;
    int node = 0, done = 0;
    /* libxs_kdtree_descend would send an absent input right; it routes instead */
    while (0 == done) {
      const int f = nodes[node].dim;
      if (0 > f) {
        result = nodes[node].next;
        done = 1;
      }
      else if (LIBXS_NOTNAN(inputs[f])) {
        node = (inputs[f] < nodes[node].threshold) ? (node + 1) : nodes[node].next;
      }
      else done = 1;
    }
  }
  return result;
}
