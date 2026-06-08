LIBXS_API_INLINE void internal_libxs_predict_distill(
  libxs_predict_t* model, int nclusters, int order)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  const int nfolds = (model->distill_folds > 0)
    ? LIBXS_MIN(model->distill_folds, p) : p;
  const int fold_size = (p + nfolds - 1) / nfolds;
  double* predicted = (double*)malloc((size_t)p * (size_t)n * sizeof(double));
  if (NULL != predicted && fold_size > 0) {
    int fold, i, ok = 1;
#if defined(_OPENMP)
#   pragma omp parallel for schedule(dynamic) private(fold)
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
#         pragma omp atomic write
#endif
          ok = 0;
        }
        libxs_predict_destroy(fm);
      }
      else {
#if defined(_OPENMP)
#       pragma omp atomic write
#endif
        ok = 0;
      }
    }
    if (0 != ok) {
      for (i = 0; i < p; ++i) {
        memcpy(model->entries[i].outputs, predicted + (size_t)i * n,
          (size_t)n * sizeof(double));
      }
    }
  }
  free(predicted);
  model->distill_folds = -1;
}
