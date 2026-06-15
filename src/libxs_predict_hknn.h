#if !defined(LIBXS_PREDICT_HKNN_MINLEAF)
#  define LIBXS_PREDICT_HKNN_MINLEAF 0
#endif


LIBXS_EXTERN_C typedef struct internal_libxs_predict_hknn_split_ctx_t {
  libxs_predict_t* model;
  internal_libxs_predict_rf_pair_t* pairs;
  int target_nc;
  int ntotal;
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
      if (nclasses > 1 && nclasses <= count / 4) use_gini = 1;
    }
    if (0 != use_gini) {
      int cnt_all[128], cnt_left[128];
      const int gini_lo = LIBXS_MAX(min_leaf, 1);
      const int gini_hi = count - gini_lo;
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
          for (oi = 0; oi < n; ++oi) {
            const double v = model->entries[pairs[i].idx].outputs[oi];
            sum_all[oi] += v;
            sum2_all[oi] += v * v;
          }
        }
        memset(sum_left, 0, (size_t)n * sizeof(double));
        memset(sum2_left, 0, (size_t)n * sizeof(double));
        for (i = 0; i < count - 1; ++i) {
          const int nleft = i + 1, nright = count - nleft;
          for (oi = 0; oi < n; ++oi) {
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
            for (oi = 0; oi < n; ++oi) {
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
    if (best_feat >= 0 && best_score >= score_floor) {
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
  const int target_nc = (int)(sqrt((double)p) + 0.5);
  const int min_leaf = (0 < LIBXS_PREDICT_HKNN_MINLEAF)
    ? LIBXS_PREDICT_HKNN_MINLEAF
    : LIBXS_MAX(p * 2 / (target_nc * 3), 3);
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
      order[i] = i;
      for (j = 0; j < m; ++j) {
        pts[(size_t)i * m + j] = model->entries[i].inputs[j];
      }
    }
    state.model = model;
    state.pairs = pairs;
    state.target_nc = target_nc;
    state.ntotal = p;
    config.min_leaf = min_leaf;
    config.split = internal_libxs_predict_hknn_split;
    config.ctx = &state;
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
