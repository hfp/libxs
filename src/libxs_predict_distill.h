LIBXS_API_INLINE void internal_libxs_predict_distill(
  libxs_predict_t* model, int nclusters, int order)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int ok = 0, predicted_pool = 0;
  double* predicted = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * (size_t)n * sizeof(double), predicted_pool);
  if (NULL != predicted) {
    if (LIBXS_PREDICT_RF == model->decompose) {
      const int nfolds = (model->distill_folds > 0)
        ? LIBXS_MIN(model->distill_folds, p) : p;
      const int fold_size = (p + nfolds - 1) / nfolds;
      int fold;
      ok = 1;
#if defined(_OPENMP)
#     pragma omp parallel for schedule(dynamic) private(fold)
#endif
      for (fold = 0; fold < nfolds; ++fold) {
        const int fold_begin = fold * fold_size;
        const int fold_end = LIBXS_MIN(fold_begin + fold_size, p);
        libxs_predict_t* fm = libxs_predict_create(m, n);
        if (NULL != fm) {
          int fi;
          fm->eval_mode = model->eval_mode;
          fm->decompose = model->decompose;
          fm->diff_mode = -1;
          for (fi = 0; fi < p; ++fi) {
            if (fi >= fold_begin && fi < fold_end) continue;
            libxs_predict_push(NULL, fm,
              model->entries[fi].inputs, model->entries[fi].outputs);
          }
          if (EXIT_SUCCESS == libxs_predict_build(fm, nclusters, order)) {
            for (fi = fold_begin; fi < fold_end; ++fi) {
              libxs_predict_eval(NULL, fm, model->entries[fi].inputs,
                predicted + (size_t)fi * n, NULL, 1);
            }
          }
          else {
#if defined(_OPENMP)
#           pragma omp atomic write
#endif
            ok = 0;
          }
          libxs_predict_destroy(fm);
        }
        else {
#if defined(_OPENMP)
#         pragma omp atomic write
#endif
          ok = 0;
        }
      }
    }
    else {
      int local_idx_pool = 0;
      int* local_idx = (int*)LIBXS_PREDICT_MALLOC(
        (size_t)p * sizeof(int), local_idx_pool);
      if (NULL != local_idx) {
        model->distill_folds = -1;
        if (EXIT_SUCCESS == libxs_predict_build(model, nclusters, order)) {
          int i, c;
          for (c = 0; c < model->nclusters; ++c) {
            const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
            int k;
            for (k = 0; k < cl->nentries; ++k) {
              local_idx[cl->sorted_idx[k]] = k;
            }
          }
#if defined(_OPENMP)
#         pragma omp parallel for schedule(dynamic)
#endif
          for (i = 0; i < p; ++i) {
            const int ci = model->assignments[i];
            const internal_libxs_predict_cluster_t* cl = &model->clusters[ci];
            const int li = local_idx[i];
            int j;
            for (j = 0; j < n; ++j) {
              const int use_classify =
                (0 != (model->eval_mode & LIBXS_PREDICT_CLASSIFY))
                  ? 1 : ((0 != (model->eval_mode & LIBXS_PREDICT_INTERPOLATE))
                    ? 0 : cl->mode[j]);
              if (0 != use_classify) {
                double conf = 0, var = 0;
                predicted[(size_t)i * n + j] =
                  internal_libxs_predict_classify(
                    cl, cl->kd_pts, cl->nentries, m,
                    cl->kd_pts + (size_t)li * m, j, n,
                    cl->ndistinct[j], 0, li, &conf, &var);
              }
              else {
                const int nearest = (li > 0) ? li - 1
                  : (li + 1 < cl->nentries ? li + 1 : 0);
                const double t = (double)nearest;
                const int d = cl->order[j];
                const double* cj = cl->coeffs
                  + (size_t)j * (cl->maxorder + 1);
                double val = 0;
                int k;
                for (k = 0; k <= d; ++k) {
                  val += cj[k] * libxs_binom(t, k);
                }
                predicted[(size_t)i * n + j] = val;
              }
            }
          }
          ok = 1;
          internal_libxs_predict_free_clusters(model);
        }
        LIBXS_PREDICT_FREE(local_idx, local_idx_pool);
      }
    }
  }
  if (0 != ok) {
    int i;
    for (i = 0; i < p; ++i) {
      memcpy(model->entries[i].outputs, predicted + (size_t)i * n,
        (size_t)n * sizeof(double));
    }
  }
  LIBXS_PREDICT_FREE(predicted, predicted_pool);
  model->distill_folds = -1;
}
