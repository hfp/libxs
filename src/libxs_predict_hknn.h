#if !defined(LIBXS_PREDICT_HKNN_MINLEAF)
#  define LIBXS_PREDICT_HKNN_MINLEAF 0
#endif


LIBXS_EXTERN_C typedef struct internal_libxs_predict_hknn_split_ctx_t {
  libxs_predict_t* model;
  internal_libxs_predict_rf_pair_t* pairs;
  int target_nc;
  int ntotal;
  int target_output;
} internal_libxs_predict_hknn_split_ctx_t;


LIBXS_API_INLINE int internal_libxs_predict_hknn_split(
  int* dim, int* pos, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx)
{
  internal_libxs_predict_hknn_split_ctx_t* state =
    (internal_libxs_predict_hknn_split_ctx_t*)ctx;
  const libxs_predict_t* model = state->model;
  const int m = model->ninputs;
  const int n = model->noutputs;
  internal_libxs_predict_rf_pair_t* pairs = state->pairs;
  const double denom = (double)count * (1 << (depth < 20 ? depth : 20));
  const double imbal = LIBXS_MAX((double)state->ntotal / denom, 1.0);
  const int ideal_half = count / 2;
  const double allowed_dev = 0.22 / LIBXS_MAX(imbal, 0.5);
  const int min_leaf = LIBXS_MAX(
    state->ntotal * 2 / (state->target_nc * 3), 3);
  const int band_lo = LIBXS_MAX(
    (int)(ideal_half - count * allowed_dev), min_leaf);
  const int band_hi = LIBXS_MIN(
    (int)(ideal_half + count * allowed_dev), count - min_leaf);
  const double progress = (double)nleaves / state->target_nc;
  const double score_floor = (progress > 0.8)
    ? (progress - 0.8) * (progress - 0.8) * 25.0 : 0.0;
  int best_feat = -1, best_pos = -1, i, j, use_gini = 0;
  int nclasses = 0, result = 1;
  double classes[128];
  double best_score = -1;
  (void)pts;
  if (band_lo <= band_hi) {
    if (1 == n) {
      for (i = 0; i < count && nclasses < 128; ++i) {
        const double v = model->entries[idx[i]].outputs[0];
        int found = 0, ci;
        for (ci = 0; ci < nclasses; ++ci) {
          if (classes[ci] == v) { found = 1; break; }
        }
        if (0 == found) classes[nclasses++] = v;
      }
      if (nclasses > 1 && count > nclasses) use_gini = 1;
    }
    if (0 != use_gini) {
      int cnt_all[128], cnt_left[128];
      const int gini_lo = 1;
      const int gini_hi = count - 1;
      for (j = 0; j < m; ++j) {
        double gini_parent = 1.0;
        int ci;
        for (i = 0; i < count; ++i) {
          pairs[i].val = model->entries[idx[i]].inputs[j];
          pairs[i].idx = idx[i];
        }
        libxs_sort(pairs, count, sizeof(pairs[0]),
          internal_libxs_predict_rf_pair_cmp, NULL);
        memset(cnt_all, 0, (size_t)nclasses * sizeof(int));
        for (i = 0; i < count; ++i) {
          const double v = model->entries[pairs[i].idx].outputs[0];
          for (ci = 0; ci < nclasses; ++ci) {
            if (classes[ci] == v) { ++cnt_all[ci]; break; }
          }
        }
        for (ci = 0; ci < nclasses; ++ci) {
          const double p_k = (double)cnt_all[ci] / count;
          gini_parent -= p_k * p_k;
        }
        memset(cnt_left, 0, (size_t)nclasses * sizeof(int));
        for (i = 0; i < count - 1; ++i) {
          const int nleft = i + 1, nright = count - nleft;
          const double v = model->entries[pairs[i].idx].outputs[0];
          for (ci = 0; ci < nclasses; ++ci) {
            if (classes[ci] == v) { ++cnt_left[ci]; break; }
          }
          if (pairs[i].val != pairs[i + 1].val
            && nleft >= gini_lo && nleft <= gini_hi)
          {
            double gini_l = 1.0, gini_r = 1.0, gain;
            for (ci = 0; ci < nclasses; ++ci) {
              const double pl = (double)cnt_left[ci] / nleft;
              const double pr =
                (double)(cnt_all[ci] - cnt_left[ci]) / nright;
              gini_l -= pl * pl;
              gini_r -= pr * pr;
            }
            gain = gini_parent
              - (double)nleft / count * gini_l
              - (double)nright / count * gini_r;
            { const double score = gain * (double)nclasses;
              if (score > best_score) {
                best_score = score;
                best_feat = j;
                best_pos = i;
              }
            }
          }
        }
      }
    }
    else {
      const int oi_lo = (state->target_output >= 0) ? state->target_output : 0;
      const int oi_hi = (state->target_output >= 0) ? state->target_output + 1 : n;
      for (j = 0; j < m; ++j) {
        double sum_all[128], sum2_all[128];
        double sum_left[128], sum2_left[128];
        int oi;
        for (i = 0; i < count; ++i) {
          pairs[i].val = model->entries[idx[i]].inputs[j];
          pairs[i].idx = idx[i];
        }
        libxs_sort(pairs, count, sizeof(pairs[0]),
          internal_libxs_predict_rf_pair_cmp, NULL);
        memset(sum_all, 0, (size_t)n * sizeof(double));
        memset(sum2_all, 0, (size_t)n * sizeof(double));
        for (i = 0; i < count; ++i) {
          for (oi = oi_lo; oi < oi_hi; ++oi) {
            const double v = model->entries[pairs[i].idx].outputs[oi];
            sum_all[oi] += v;
            sum2_all[oi] += v * v;
          }
        }
        memset(sum_left, 0, (size_t)n * sizeof(double));
        memset(sum2_left, 0, (size_t)n * sizeof(double));
        for (i = 0; i < count - 1; ++i) {
          const int nleft = i + 1, nright = count - nleft;
          for (oi = oi_lo; oi < oi_hi; ++oi) {
            const double v = model->entries[pairs[i].idx].outputs[oi];
            sum_left[oi] += v;
            sum2_left[oi] += v * v;
          }
          if (pairs[i].val != pairs[i + 1].val
            && nleft >= band_lo && nleft <= band_hi)
          {
            double fisher = 0;
            const double penalty =
              1.0 + 4.0 * LIBXS_FABS((double)nleft / count - 0.5);
            for (oi = oi_lo; oi < oi_hi; ++oi) {
              const double ml = sum_left[oi] / nleft;
              const double mr = (sum_all[oi] - sum_left[oi]) / nright;
              const double vl = sum2_left[oi] / nleft - ml * ml;
              const double vr = (sum2_all[oi] - sum2_left[oi]) / nright
                - mr * mr;
              const double within = vl * nleft + vr * nright;
              const double between = (double)nleft * nright
                * (ml - mr) * (ml - mr) / count;
              if (within > 0) fisher += between / within;
            }
            { const double score = fisher / penalty;
              if (score > best_score) {
                best_score = score;
                best_feat = j;
                best_pos = i;
              }
            }
          }
        }
      }
    }
    if (best_feat >= 0 && (0 != use_gini || best_score >= score_floor)) {
      for (i = 0; i < count; ++i) {
        pairs[i].val = model->entries[idx[i]].inputs[best_feat];
        pairs[i].idx = idx[i];
      }
      libxs_sort(pairs, count, sizeof(pairs[0]),
        internal_libxs_predict_rf_pair_cmp, NULL);
      for (i = 0; i < count; ++i) idx[i] = pairs[i].idx;
      *dim = best_feat;
      *pos = best_pos + 1;
      result = 0;
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_partition(
  libxs_predict_t* model, int* nclusters_out)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  const int target_nc = (int)(sqrt((double)p) + 0.5);
  const int min_leaf = (0 < LIBXS_PREDICT_HKNN_MINLEAF)
    ? LIBXS_PREDICT_HKNN_MINLEAF
    : ((1 == n) ? 1 : LIBXS_MAX(p * 2 / (target_nc * 3), 3));
  int* const out_assign = model->hknn_assignments;
  int pairs_pool = 0, order_pool = 0, pts_pool = 0;
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  int* order = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(int), order_pool);
  double* pts = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * (size_t)m * sizeof(double), pts_pool);
  int nleaves = 0;
  if (NULL != pairs && NULL != order && NULL != pts) {
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
    state.target_nc = target_nc;
    state.ntotal = p;
    state.target_output = -1;
    config.min_leaf = min_leaf;
    config.split = internal_libxs_predict_hknn_split;
    config.ctx = &state;
    if (n > 1) {
      model->hknn_po_assignments = (int**)calloc(
        (size_t)n, sizeof(int*));
      model->hknn_po_nclusters = (int*)calloc(
        (size_t)n, sizeof(int));
      if (NULL != model->hknn_po_assignments
        && NULL != model->hknn_po_nclusters)
      {
        for (j = 0; j < n; ++j) {
          model->hknn_po_assignments[j] = (int*)calloc(
            (size_t)p, sizeof(int));
          if (NULL != model->hknn_po_assignments[j]) {
            state.target_output = j;
            for (i = 0; i < p; ++i) order[i] = i;
            model->hknn_po_nclusters[j] = libxs_kdtree_partition(
              pts, order, p, m, m,
              model->hknn_po_assignments[j], &config);
          }
        }
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
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_refine(
  libxs_predict_t* model, int nclusters)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int max_iter = LIBXS_MIN(LIBXS_PREDICT_MAXITER, 10);
  int pts_pool = 0, comp_pool = 0, cnt_pool = 0;
  double* pts = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * (size_t)m * sizeof(double), pts_pool);
  double* comp = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * (size_t)m * sizeof(double), comp_pool);
  int* counts = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * sizeof(int), cnt_pool);
  if (NULL != pts && NULL != comp && NULL != counts) {
    int i, c, j, iter;
    for (i = 0; i < p; ++i) {
      internal_libxs_predict_normalize(model,
        model->entries[i].inputs, pts + (size_t)i * m);
    }
    for (iter = 0; iter < max_iter; ++iter) {
      int changed = 0;
      for (i = 0; i < p; ++i) {
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
      if (0 == changed) iter = max_iter;
      else {
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
      }
    }
  }
  LIBXS_PREDICT_FREE(counts, cnt_pool);
  LIBXS_PREDICT_FREE(comp, comp_pool);
  LIBXS_PREDICT_FREE(pts, pts_pool);
}


LIBXS_API_INLINE void internal_libxs_predict_hknn_centroids(
  libxs_predict_t* model, int nclusters)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  int counts_pool = 0, norm_pool = 0, i, c, j;
  int* counts = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)nclusters * sizeof(int), counts_pool);
  double* norm = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)m * sizeof(double), norm_pool);
  if (NULL != counts && NULL != norm) {
    memset(counts, 0, (size_t)nclusters * sizeof(int));
    for (c = 0; c < nclusters; ++c) {
      memset(model->clusters[c].centroid, 0, (size_t)m * sizeof(double));
    }
    for (i = 0; i < p; ++i) {
      const int ci = model->assignments[i];
      internal_libxs_predict_normalize(model,
        model->entries[i].inputs, norm);
      for (j = 0; j < m; ++j) {
        model->clusters[ci].centroid[j] += norm[j];
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
  }
  LIBXS_PREDICT_FREE(norm, norm_pool);
  LIBXS_PREDICT_FREE(counts, counts_pool);
}


LIBXS_API_INLINE int internal_libxs_predict_hknn_build_po(
  libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int result = EXIT_SUCCESS;
  if (NULL == model->hknn_po_assignments
    || NULL == model->hknn_po_nclusters || n <= 1)
  {
    result = EXIT_SUCCESS;
  }
  else {
    model->hknn_po_clusters = (internal_libxs_predict_cluster_t**)calloc(
      (size_t)n, sizeof(internal_libxs_predict_cluster_t*));
    if (NULL == model->hknn_po_clusters) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && NULL != model->hknn_po_clusters) {
    int oi;
    for (oi = 0; oi < n && EXIT_SUCCESS == result; ++oi) {
      const int* assign = model->hknn_po_assignments[oi];
      const int nc = model->hknn_po_nclusters[oi];
      internal_libxs_predict_cluster_t* cls;
      int c, i, k;
      if (NULL == assign || nc < 1) continue;
      cls = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nc, sizeof(internal_libxs_predict_cluster_t));
      if (NULL == cls) { result = EXIT_FAILURE; break; }
      model->hknn_po_clusters[oi] = cls;
      for (c = 0; c < nc && EXIT_SUCCESS == result; ++c) {
        cls[c].centroid = (double*)calloc((size_t)m, sizeof(double));
        if (NULL == cls[c].centroid) result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        int norm_pool = 0;
        double* norm = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)m * sizeof(double), norm_pool);
        int* counts = (int*)calloc((size_t)nc, sizeof(int));
        if (NULL != norm && NULL != counts) {
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
        free(counts);
        LIBXS_PREDICT_FREE(norm, norm_pool);
      }
      for (c = 0; c < nc && EXIT_SUCCESS == result; ++c) {
        const int nce = cls[c].nentries;
        int ki, nd;
        double prev;
        if (0 >= nce) continue;
        cls[c].sorted_idx = (int*)malloc((size_t)nce * sizeof(int));
        cls[c].kd_pts = (double*)malloc(
          (size_t)nce * (size_t)m * sizeof(double));
        cls[c].raw_outputs = (double*)malloc((size_t)nce * sizeof(double));
        cls[c].mode = (int*)malloc(sizeof(int));
        cls[c].ndistinct = (int*)malloc(sizeof(int));
        if (NULL == cls[c].sorted_idx || NULL == cls[c].kd_pts
          || NULL == cls[c].raw_outputs
          || NULL == cls[c].mode || NULL == cls[c].ndistinct)
        {
          result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result) {
          ki = 0;
          for (i = 0; i < p; ++i) {
            if (assign[i] == c) {
              cls[c].sorted_idx[ki] = i;
              internal_libxs_predict_normalize(model,
                model->entries[i].inputs,
                cls[c].kd_pts + (size_t)ki * m);
              cls[c].raw_outputs[ki] =
                model->entries[i].outputs[oi];
              ++ki;
            }
          }
          cls[c].dmax = 0;
          for (k = 0; k < nce; ++k) {
            const double d = sqrt(
              libxs_dist2(cls[c].kd_pts + (size_t)k * m,
                cls[c].centroid, m));
            if (d > cls[c].dmax) cls[c].dmax = d;
          }
          if (cls[c].dmax <= 0.0) cls[c].dmax = 1.0;
          nd = 0;
          prev = cls[c].raw_outputs[0];
          for (k = 1; k < nce; ++k) {
            if (cls[c].raw_outputs[k] != prev) {
              ++nd; prev = cls[c].raw_outputs[k];
            }
          }
          cls[c].ndistinct[0] = nd + 1;
          cls[c].mode[0] = (cls[c].ndistinct[0] <= LIBXS_PREDICT_KNN)
            ? 1 : 0;
          cls[c].k_eff = (0 != cls[c].mode[0])
            ? LIBXS_MIN(LIBXS_MAX(5, nce / 3), LIBXS_PREDICT_KNN)
            : LIBXS_MIN(LIBXS_MAX(3, (int)(sqrt((double)nce) + 0.5)),
                LIBXS_PREDICT_KNN);
        }
      }
    }
  }
  return result;
}


