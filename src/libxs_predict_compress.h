LIBXS_API_INLINE void internal_libxs_predict_compress(
  libxs_predict_t* model, int nclusters, int order, double quality)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int local_idx_pool = 0;
  int* local_idx = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(int), local_idx_pool);
  if (NULL != local_idx) {
    int keep_pool = 0;
    int* keep = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), keep_pool);
    if (NULL != keep) {
      int i, c, nkeep = 0;
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        int k;
        for (k = 0; k < cl->nentries; ++k) {
          local_idx[cl->sorted_idx[k]] = k;
        }
      }
      for (i = 0; i < p; ++i) {
        const int ci = model->assignments[i];
        const internal_libxs_predict_cluster_t* cl = &model->clusters[ci];
        const int li = local_idx[i];
        double min_conf = 1.0;
        int j, mismatch = 0, nchecked = 0;
        for (j = 0; j < n && 0 == mismatch; ++j) {
          const int use_classify =
            (0 != (model->eval_mode & LIBXS_PREDICT_CLASSIFY))
              ? 1 : ((0 != (model->eval_mode & LIBXS_PREDICT_INTERPOLATE))
                ? 0 : cl->mode[j]);
          if (0 != use_classify) {
            double conf = 0, var = 0;
            const double actual = cl->raw_outputs[(size_t)li * n + j];
            const double predicted = internal_libxs_predict_classify(
              cl, cl->kd_pts, cl->nentries, m,
              cl->kd_pts + (size_t)li * m, j, n,
              cl->ndistinct[j], 0, li, &conf, &var);
            if (predicted != actual || var > 0) {
              mismatch = 1;
            }
            else if (conf < min_conf) {
              min_conf = conf;
            }
          }
          else {
            const double actual = cl->raw_outputs[(size_t)li * n + j];
            const int d = cl->order[j];
            const double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
            const double t = (double)li;
            double val = 0, residual;
            int k;
            for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
            residual = (val > actual) ? (val - actual) : (actual - val);
            if (residual > cl->errors[j] * (1.0 - quality)) mismatch = 1;
          }
          ++nchecked;
        }
        keep[i] = (0 == nchecked || 0 != mismatch || min_conf < quality)
          ? 1 : 0;
      }
      for (i = 0; i < p; ++i) nkeep += keep[i];
      if (nkeep > 0 && nkeep < p) {
        int remap_pool = 0;
        int* remap = (int*)LIBXS_PREDICT_MALLOC(
          (size_t)p * sizeof(int), remap_pool);
        for (c = 0; c < model->nclusters; ++c) {
          internal_libxs_predict_cluster_t* cl = &model->clusters[c];
          const int nc = cl->nentries;
          int dst = 0, k;
          for (k = 0; k < nc; ++k) {
            const int gi = cl->sorted_idx[k];
            if (0 != keep[gi]) {
              if (dst != k) {
                memcpy(cl->kd_pts + (size_t)dst * m,
                  cl->kd_pts + (size_t)k * m, (size_t)m * sizeof(double));
                memcpy(cl->raw_outputs + (size_t)dst * n,
                  cl->raw_outputs + (size_t)k * n, (size_t)n * sizeof(double));
                cl->sorted_idx[dst] = gi;
              }
              ++dst;
            }
          }
          cl->nentries = dst;
          if (dst > 0 && dst < nc) {
            const int maxord = LIBXS_MIN(dst - 1,
              order > 0 ? order : cl->maxorder);
            cl->maxorder = (maxord < 1) ? 1 : maxord;
            internal_libxs_predict_cluster_refit(cl, n, 0);
          }
        }
        { internal_libxs_predict_entry_t* old_entries = model->entries;
          const int old_p = model->nentries;
          int dst = 0;
          for (i = 0; i < old_p; ++i) {
            if (0 != keep[i]) {
              if (NULL != remap) remap[i] = dst;
              if (dst != i) {
                old_entries[dst] = old_entries[i];
                model->assignments[dst] = model->assignments[i];
              }
              ++dst;
            }
            else {
              if (NULL != remap) remap[i] = -1;
              free(old_entries[i].inputs);
              free(old_entries[i].outputs);
            }
          }
          model->nentries = dst;
        }
        if (NULL != remap) {
          for (c = 0; c < model->nclusters; ++c) {
            internal_libxs_predict_cluster_t* cl = &model->clusters[c];
            int k;
            for (k = 0; k < cl->nentries; ++k) {
              cl->sorted_idx[k] = remap[cl->sorted_idx[k]];
            }
          }
          if (NULL != model->hknn_po_assignments) {
            int j2, new_p = model->nentries;
            for (j2 = 0; j2 < n; ++j2) {
              if (NULL != model->hknn_po_assignments[j2]) {
                int* old_po = model->hknn_po_assignments[j2];
                int* new_po = (int*)malloc((size_t)new_p * sizeof(int));
                if (NULL != new_po) {
                  for (i = 0; i < p; ++i) {
                    if (remap[i] >= 0) new_po[remap[i]] = old_po[i];
                  }
                  free(old_po);
                  model->hknn_po_assignments[j2] = new_po;
                }
              }
            }
          }
        }
        LIBXS_PREDICT_FREE(remap, remap_pool);
      }
      LIBXS_PREDICT_FREE(keep, keep_pool);
    }
    LIBXS_PREDICT_FREE(local_idx, local_idx_pool);
  }
}
