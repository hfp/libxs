LIBXS_API_INLINE void internal_libxs_predict_compress(
  libxs_predict_t* model, int order, double quality)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int maxn = 0, ci;
  int dropped_pool = 0;
  char* dropped;
  for (ci = 0; ci < model->nclusters; ++ci) {
    if (model->clusters[ci].nentries > maxn) maxn = model->clusters[ci].nentries;
  }
  dropped = (char*)LIBXS_PREDICT_MALLOC((size_t)(0 < maxn ? maxn : 1),
    dropped_pool);
  if (NULL != dropped) {
    int keep_pool = 0;
    int* keep = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), keep_pool);
    if (NULL != keep) {
      int i, c, nkeep = 0;
      /**
       * Entries are dropped as the scan proceeds, and every later test reads
       * the set that is left. Judging each entry against the whole corpus and
       * removing the flagged ones together was the defect: redundancy is a
       * property of a neighbourhood, so entries that are individually
       * recoverable are jointly load-bearing, and removing them all destroys
       * the density that made each one redundant. It measured as a retained
       * set worse than a random subset of the same size (crystal: 41.1%
       * against 58.3% at ~7200 entries).
       */
      for (i = 0; i < p; ++i) keep[i] = 1;
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        int nsurv = cl->nentries, li;
        for (li = 0; li < cl->nentries; ++li) dropped[li] = 0;
        for (li = 0; li < cl->nentries; ++li) {
        const int gi = cl->sorted_idx[li];
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
            const double predicted = internal_libxs_predict_classify2(
              cl, m, cl->kd_pts + (size_t)li * m, j, n,
              cl->ndistinct[j], 0, li, dropped, NULL, -1, &conf, &var,
              0, NULL, NULL,
              internal_libxs_predict_central(model, j), NULL,
              model->has_missing);
            /**
             * Unanimity is required, not merely a confident majority, and that
             * is load-bearing rather than redundant: it also pins the vote
             * fraction at 1.0, so `quality` cannot select among classify
             * outputs and the drop set is the same whatever is asked for.
             * Letting the threshold decide instead was measured and is far
             * worse - on the crystal corpus it drops 57 to 72% of the entries
             * and takes held-out accuracy from 0.67 to 0.25, because a corpus
             * with near-duplicate inputs has many entries that a *disagreeing*
             * neighbourhood still recovers exactly.  The threshold's documented
             * meaning is the wrong rule; this is the right one.
             */
            /**
             * The test needs a neighbourhood that can disagree. One neighbour
             * cannot: it has no variance to report and no vote to be short of,
             * so var and conf read 0 and 1 whatever it holds, and every entry
             * whose nearest neighbour shares its label looks redundant. That
             * measured as 80% of the entries dropped and held-out accuracy from
             * 0.68 to 0.30. The trial no longer selects such a count, so this
             * guards a count a caller pinned; declining to drop is the safe
             * answer, leaving compression a no-op rather than destructive.
             */
            const int keff = (NULL != cl->k_out) ? cl->k_out[j] : cl->k_eff;
            if (2 > keff || predicted != actual || var > 0) {
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
        /**
         * A cluster keeps enough entries to answer with: below the vote floor
         * the neighbourhood degenerates exactly as a one-neighbour vote does,
         * and the test that decides the next drop stops meaning anything.
         */
        if (0 == nchecked || 0 != mismatch || min_conf < quality
          || LIBXS_PREDICT_KMIN >= nsurv)
        {
          keep[gi] = 1;
        }
        else {
          keep[gi] = 0;
          dropped[li] = 1;
          --nsurv;
        }
        }
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
          /**
           * The values move with the entry rather than the entry keeping a
           * pointer to where it used to be: the entries address one arena, so
           * leaving slot i occupied by entry dst would let the next push write
           * over a slot that is still read.
           */
          const size_t stride = (size_t)m + n;
          for (i = 0; i < old_p; ++i) {
            if (0 != keep[i]) {
              if (NULL != remap) remap[i] = dst;
              if (dst != i) {
                double* to = model->arena + (size_t)dst * stride;
                memmove(to, model->arena + (size_t)i * stride,
                  stride * sizeof(double));
                old_entries[dst].weight = old_entries[i].weight;
                old_entries[dst].inputs = to;
                old_entries[dst].outputs = to + m;
                model->assignments[dst] = model->assignments[i];
              }
              ++dst;
            }
            else if (NULL != remap) remap[i] = -1;
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
            const int ng = (model->hknn_ngroups > 0)
              ? model->hknn_ngroups : n;
            int j2, new_p = model->nentries;
            for (j2 = 0; j2 < ng; ++j2) {
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
    LIBXS_PREDICT_FREE(dropped, dropped_pool);
  }
}
