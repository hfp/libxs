/**
 * Selection of the decomposition mode by trial.
 *
 * The mode decides more than any other build-time knob and nothing but a build
 * says which one to take: a forest wins by 39 to 51% on the crystal corpus and
 * loses to hierarchical kNN on earthquakes, so a fixed default costs about 22%
 * on average against the mode a caller should have picked. Every candidate is
 * built on part of the corpus and scored on a part held back, which is the same
 * regime for all of them and the only one in which they are comparable.
 *
 * Leave-one-out over stored points would be cheaper and is not admissible here.
 * It is optimistic by an amount that varies with how much near-duplicate
 * structure a corpus has, and the modes differ in exactly how they exploit that
 * structure, so the bias does not cancel between them.
 *
 * More candidates turned out to be safer rather than riskier. The wrong picks
 * are all near-ties, where the validation slice cannot separate two modes and
 * the arbitrary choice costs nearly nothing; adding a mode that wins by a real
 * margin gives the slice something it can resolve. Shortlisting therefore
 * guards a risk that is only large where it does not matter, at the price of
 * excluding the mode that would have won.
 */

#if !defined(LIBXS_PREDICT_NDECOMPOSE)
# define LIBXS_PREDICT_NDECOMPOSE 7
#endif
#if !defined(LIBXS_PREDICT_NNEIGHBORS)
# define LIBXS_PREDICT_NNEIGHBORS 9
#endif
/**
 * Entries a candidate is fitted on while it is being ranked.
 *
 * The trial does not need the model the caller asked for, only the ORDER of the
 * candidates, and an order survives on far less data than an error does: on a
 * 60k-row corpus every cap from 1000 up selects the same mode as the whole of
 * it, at a hundred and sixty-ninth of the cost.
 *
 * The cap is nevertheless the largest of those rather than the cheapest, because
 * the saving is not free where the candidates are close. On a corpus whose
 * modes sit within 7% of each other the ranking only settled at 8000; below that
 * the trial named a different winner at every size, which is a subsample
 * resolving noise rather than a margin. Promoting only the candidates that are
 * still close would beat a fixed cap on both counts and is not implemented here.
 */
#if !defined(LIBXS_PREDICT_TRIALCAP)
# define LIBXS_PREDICT_TRIALCAP 8192
#endif


/** Candidate by index, ordered so that a prefix is a sensible shortlist. */
LIBXS_API_INLINE int internal_libxs_predict_decompose_cand(int i)
{
  int result;
  switch (i) {
    case 0: result = LIBXS_PREDICT_RAW; break;
    case 1: result = LIBXS_PREDICT_RF; break;
    case 2: result = LIBXS_PREDICT_FISHER; break;
    case 3: result = LIBXS_PREDICT_HKNN; break;
    case 4: result = LIBXS_PREDICT_PCA; break;
    case 5: result = LIBXS_PREDICT_SETDIFF; break;
    default: result = LIBXS_PREDICT_SPREAD;
  }
  return result;
}


/**
 * Non-zero if the mode can apply to this model at all. SPREAD without a second
 * series is RAW under another name, and a mode that cannot carry a gap has
 * nothing to say about a corpus that has one, so building either would spend a
 * build to learn what the model already knows.
 */
LIBXS_API_INLINE int internal_libxs_predict_decompose_ok(
  const libxs_predict_t* model, int mode)
{
  int result = 1;
  if (LIBXS_PREDICT_SPREAD == mode && 2 > model->nseries) {
    result = 0;
  }
  /**
   * Both weighting modes derive their scores from a single output's classes and
   * assign no weights at all unless there is exactly one output, which leaves
   * the model bit-identical to RAW. Building them anyway is how four of seven
   * candidates came to produce the same score to four decimals on every tuning
   * corpus: the trial was not measuring a near-tie, it was measuring one model
   * four times.
   */
  else if ((LIBXS_PREDICT_FISHER == mode || LIBXS_PREDICT_SETDIFF == mode)
    && 1 != model->noutputs)
  {
    result = 0;
  }
  else if (0 != model->has_missing
    && 0 == internal_libxs_predict_gaps_ok(mode))
  {
    result = 0;
  }
  return result;
}


/**
 * Per-output error kind and scale, taken from the fit slice alone.
 *
 * Outputs are scored together and do not share a unit, so each contributes a
 * dimensionless number: a discrete output its miss rate, a continuous one its
 * absolute error over its own mean absolute deviation. The threshold on
 * distinct values is the one the cluster refit uses to decide the same question.
 */
LIBXS_API_INLINE void internal_libxs_predict_decompose_kind(
  const libxs_predict_t* model, const char role[], int kind[], double mad[],
  double* buf)
{
  const int n = model->noutputs;
  int j;
  for (j = 0; j < n; ++j) {
    double sum = 0;
    int i, nfit = 0, ndistinct = 1;
    for (i = 0; i < model->nentries; ++i) {
      if (0 == role[i]) {
        buf[nfit++] = model->entries[i].outputs[j];
        sum += model->entries[i].outputs[j];
      }
    }
    kind[j] = 0;
    mad[j] = 1.0;
    if (0 < nfit) {
      const double mean = sum / nfit;
      libxs_sort(buf, nfit, sizeof(double), libxs_cmp_f64, NULL);
      for (i = 1; i < nfit; ++i) {
        if (buf[i] != buf[i - 1]) ++ndistinct;
      }
      kind[j] = (ndistinct <= (int)(sqrt((double)nfit) + 0.5)) ? 1 : 0;
      sum = 0;
      for (i = 0; i < model->nentries; ++i) {
        if (0 == role[i]) {
          sum += LIBXS_FABS(model->entries[i].outputs[j] - mean);
        }
      }
      if (0 < sum) mad[j] = sum / nfit;
    }
  }
}


/**
 * Score one mode on entries held back from the build.
 *
 * The probe carries the settings that change what a prediction is, and none of
 * the timeseries state: a series model reaches this through the window probe
 * instead, because its bank of window views is itself mode-dependent and would
 * not be reproduced by a model fed the expanded entries. That path is also why
 * the trial cap is not applied there: a series cannot be thinned without
 * changing what the next step means.
 *
 * Returns a large value if the mode cannot be built, which is how a candidate
 * that fails on this corpus takes itself out of the running.
 */
LIBXS_API_INLINE double internal_libxs_predict_decompose_probe(
  const libxs_predict_t* model, int mode, const char role[], const int kind[],
  const double mad[])
{
  const int m = model->ninputs, n = model->noutputs;
  libxs_predict_t* probe = libxs_predict_create(m, n);
  double result = 1e30;
  if (NULL != probe) {
    double* pred = (double*)malloc((size_t)n * sizeof(double));
    int i, j;
    probe->eval_mode = model->eval_mode;
    probe->decompose = mode;
    probe->central = model->central;
    probe->consistency = model->consistency;
    probe->smooth = model->smooth;
    probe->floor = model->floor;
    probe->refine = model->refine;
    probe->tangent = model->tangent;
    probe->missing_mode = model->missing_mode;
    probe->rf_ntrees = model->rf_ntrees;
    probe->rf_depth = model->rf_depth;
    /**
     * The neighbour count reaches the probe too. Without it every count-bearing
     * mode was scored at the derived count while the model it stands for
     * resolves its own, and a forest, having no count, was the only candidate
     * measured as it would be built.
     */
    probe->kreq = model->kreq;
    { int nfit = 0, stride, seen = 0;
      for (i = 0; i < model->nentries; ++i) {
        if (0 == role[i]) ++nfit;
      }
      /**
       * Every stride-th entry rather than a prefix: a corpus may be ordered by
       * anything at all, and the prefix of one sorted by problem size is a
       * different distribution rather than a smaller sample of the same one.
       */
      stride = (LIBXS_PREDICT_TRIALCAP < nfit)
        ? ((nfit + LIBXS_PREDICT_TRIALCAP - 1) / LIBXS_PREDICT_TRIALCAP) : 1;
      for (i = 0; i < model->nentries; ++i) {
        if (0 == role[i]) {
          if (0 == (seen % stride)) {
            libxs_predict_push(NULL, probe, model->entries[i].inputs,
              model->entries[i].outputs);
          }
          ++seen;
        }
      }
    }
    if (NULL != pred && 0 < probe->nentries
      && EXIT_SUCCESS == libxs_predict_build(probe, 0, 2, 0.0))
    {
      double err = 0;
      int nval = 0;
      for (i = 0; i < model->nentries; ++i) {
        if (1 == role[i]) {
          libxs_predict_eval(NULL, probe, model->entries[i].inputs, pred,
            NULL, 1);
          for (j = 0; j < n; ++j) {
            const double actual = model->entries[i].outputs[j];
            err += (0 != kind[j])
              ? ((LIBXS_ROUNDX(int, pred[j]) == LIBXS_ROUNDX(int, actual))
                ? 0.0 : 1.0)
              : (LIBXS_FABS(pred[j] - actual) / mad[j]);
          }
          ++nval;
        }
      }
      if (0 < nval) result = err / ((double)nval * n);
    }
    free(pred);
    libxs_predict_destroy(probe);
  }
  return result;
}


/**
 * Score the candidates this task owns, accumulating into total[].
 *
 * A candidate is one build and shares nothing with the others, which makes the
 * trial the most parallel stage of a build and, until it was distributed, the
 * least parallel: entered collectively it ran wholly on the builder while every
 * other task waited for it. Tasks take candidates round-robin and write only
 * their own slots, so the result does not depend on who finishes first.
 *
 * folds: number of folds to score, or zero to take the default for the kind.
 */
LIBXS_API_INLINE void internal_libxs_predict_decompose_score(
  const libxs_predict_t* model, int folds, int tid, int ntasks, double total[])
{
  const int series = (0 < model->nts && 0 < model->nseries) ? 1 : 0;
  const int nfold = (0 < folds) ? folds : ((0 != series) ? 3 : 1);
  int c;
  for (c = tid; c < LIBXS_PREDICT_NDECOMPOSE; c += ntasks) total[c] = 1e30;
  if (0 != series) {
    for (c = tid; c < LIBXS_PREDICT_NDECOMPOSE; c += ntasks) {
      const int mode = internal_libxs_predict_decompose_cand(c);
      if (0 != internal_libxs_predict_decompose_ok(model, mode)) {
        total[c] = internal_libxs_predict_ts_window_probe(
          model, model->window, nfold, mode);
      }
    }
  }
  else if (0 < model->nentries) {
    const int p = model->nentries;
    const int n = model->noutputs;
    int role_pool = 0, kind_pool = 0, mad_pool = 0, buf_pool = 0;
    char* role = (char*)LIBXS_PREDICT_MALLOC((size_t)p, role_pool);
    int* kind = (int*)LIBXS_PREDICT_MALLOC((size_t)n * sizeof(int), kind_pool);
    double* mad = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)n * sizeof(double), mad_pool);
    double* buf = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(double), buf_pool);
    if (NULL != role && NULL != kind && NULL != mad && NULL != buf) {
      const size_t co = libxs_coprime2((size_t)p);
      const int nfit = (int)(p * 0.8 + 0.5);
      int f, i;
      for (c = tid; c < LIBXS_PREDICT_NDECOMPOSE; c += ntasks) total[c] = 0;
      /**
       * Every task derives the same split rather than sharing one, which costs
       * a scan of the corpus per task and removes the only thing they would
       * otherwise have to agree about beyond their own slots.
       */
      for (f = 0; f < nfold; ++f) {
        for (i = 0; i < p; ++i) {
          role[LIBXS_SHUFFLE_INDEX(i, p, co, (unsigned)f)] =
            (char)((i < nfit) ? 0 : 1);
        }
        internal_libxs_predict_decompose_kind(model, role, kind, mad, buf);
        for (c = tid; c < LIBXS_PREDICT_NDECOMPOSE; c += ntasks) {
          const int mode = internal_libxs_predict_decompose_cand(c);
          total[c] += (0 != internal_libxs_predict_decompose_ok(model, mode))
            ? internal_libxs_predict_decompose_probe(model, mode, role, kind,
              mad)
            : 1e30;
        }
      }
    }
    LIBXS_PREDICT_FREE(buf, buf_pool);
    LIBXS_PREDICT_FREE(mad, mad_pool);
    LIBXS_PREDICT_FREE(kind, kind_pool);
    LIBXS_PREDICT_FREE(role, role_pool);
  }
}


/**
 * The mode with the lowest total, or the default where nothing was scoreable.
 * Taken in candidate order by one task, so the answer does not depend on how
 * the scoring was distributed.
 */
LIBXS_API_INLINE int internal_libxs_predict_decompose_reduce(
  const libxs_predict_t* model, const double total[])
{
  double best = 1e30;
  int c, result = LIBXS_PREDICT_RAW;
  for (c = 0; c < LIBXS_PREDICT_NDECOMPOSE; ++c) {
    if (total[c] < best) {
      best = total[c];
      result = internal_libxs_predict_decompose_cand(c);
    }
  }
  LIBXS_UNUSED(model);
  return result;
}


/**
 * Choose the mode, and fall back to the default rather than to a mode no
 * measurement supported: an empty or unscoreable corpus has to leave the caller
 * where a caller who never asked would have been.
 *
 * A timeseries is scored on rolling cuts and a table on one split. The cut
 * walks forward for the same reason the window trial's does, and there is more
 * than one of them because a single held-out tail was measured to reverse the
 * sign of a distance-scaling result on the discharge corpus. A table has no
 * such direction, and one shuffled split of it costs one build per candidate
 * instead of three.
 *
 * Serial form: one task scores everything, then reduces.
 *
 * folds: number of folds to score, or zero to take the default for the kind.
 */
LIBXS_API_INLINE int internal_libxs_predict_decompose_select(
  const libxs_predict_t* model, int folds)
{
  double total[LIBXS_PREDICT_NDECOMPOSE];
  internal_libxs_predict_decompose_score(model, folds, 0, 1, total);
  return internal_libxs_predict_decompose_reduce(model, total);
}


/** Neighbour counts to try, ordered; the cap at 32 makes the grid exhaustive. */
LIBXS_API_INLINE int internal_libxs_predict_neighbors_cand(int i)
{
  int result;
  switch (i) {
    case 0: result = 1; break;
    case 1: result = 2; break;
    case 2: result = 3; break;
    case 3: result = 5; break;
    case 4: result = 8; break;
    case 5: result = 12; break;
    case 6: result = 18; break;
    case 7: result = 25; break;
    default: result = LIBXS_PREDICT_KNN;
  }
  return result;
}


/* release what prep allocated, whichever part of it exists */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_free(
  internal_libxs_predict_ktrial_t* trial)
{
  if (NULL != trial) {
    libxs_predict_destroy(trial->probe);
    free(trial->part);
    free(trial->held);
    free(trial->mad);
    free(trial->kind);
    free(trial);
  }
}


/**
 * Resolve one neighbour count per output, writing model->k_sel. The trial runs
 * in three stages - prep and finish are the builder's, the scoring between
 * them the team's - and its state lives on the model while it runs.
 *
 * The count changes nothing about the model, only how many neighbours the vote
 * reads, so one model serves the whole grid. That is what makes choosing this
 * per output affordable where choosing the mode per output was not, and the grid
 * being ordered is what makes it work: a pick one step off the optimum is one
 * step off, not a different model.
 *
 * The model scored is the one being built, every entry predicted with itself
 * held out. The header of this file refuses leave-one-out for the mode, whose
 * candidates exploit near-duplicates differently; the count does not, and it
 * measured so. Against a probe fitted on four fifths and scored on the rest it
 * picked the same count on every configuration of the crystal and earthquake
 * corpora where the count reaches the answer, with identical accuracy, and its
 * error curve came out flatter rather than tilted toward small counts: an exact
 * duplicate is answered by the match at every count and so favours none. It
 * saves a second build, partition included - about half of a large one.
 *
 * A timeseries keeps the probe, fitted on the leading four fifths and scored on
 * the tail. Overlapping windows share timesteps, so a window held out alone would
 * leave its own history in the corpus that predicts it, and every candidate
 * would look equally good.
 */
LIBXS_API_INLINE internal_libxs_predict_ktrial_t*
internal_libxs_predict_neighbors_prep(libxs_predict_t* model, int ntasks)
{
  const int p = model->nentries;
  const int n = model->noutputs;
  const int nfit = (int)(p * 0.8 + 0.5);
  internal_libxs_predict_ktrial_t* trial = NULL;
  int role_pool = 0, buf_pool = 0;
  char* role = (char*)LIBXS_PREDICT_MALLOC((size_t)p, role_pool);
  double* buf = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(double), buf_pool);
  if (NULL != role && NULL != buf && 8 < p && NULL == model->k_sel) {
    trial = (internal_libxs_predict_ktrial_t*)calloc(1, sizeof(*trial));
  }
  if (NULL != trial) {
    const int series = (0 < model->nts && 0 < model->nseries) ? 1 : 0;
    const size_t npart = (size_t)3 * LIBXS_PREDICT_NNEIGHBORS * n;
    int i, ok;
    trial->kind = (int*)malloc((size_t)n * sizeof(int));
    trial->mad = (double*)malloc((size_t)n * sizeof(double));
    trial->part = (double*)malloc(npart * ntasks * sizeof(double));
    if (0 != series) {
      trial->probe = libxs_predict_create(model->ninputs, n);
      trial->held = (int*)malloc((size_t)p * sizeof(int));
    }
    ok = (NULL != trial->kind && NULL != trial->mad && NULL != trial->part
      && (0 == series || (NULL != trial->probe && NULL != trial->held)))
      ? 1 : 0;
    if (0 != ok) {
      /* every entry is scored without a probe, so every entry sets the scale */
      for (i = 0; i < p; ++i) {
        role[i] = (char)((0 != series && i >= nfit) ? 1 : 0);
      }
      internal_libxs_predict_decompose_kind(model, role, trial->kind,
        trial->mad, buf);
    }
    if (0 != ok && 0 != series) {
      libxs_predict_t* probe = trial->probe;
      probe->eval_mode = model->eval_mode;
      probe->decompose = model->decompose;
      probe->central = model->central;
      probe->consistency = model->consistency;
      probe->smooth = model->smooth;
      probe->floor = model->floor;
      probe->refine = model->refine;
      probe->tangent = model->tangent;
      probe->missing_mode = model->missing_mode;
      probe->rf_ntrees = model->rf_ntrees;
      probe->rf_depth = model->rf_depth;
      /* ascending, so each task's strided share keeps the scan's order */
      for (i = 0; i < p; ++i) {
        if (0 == role[i]) {
          libxs_predict_push(NULL, probe, model->entries[i].inputs,
            model->entries[i].outputs);
        }
        else trial->held[trial->nheld++] = i;
      }
      if (0 >= probe->nentries) ok = 0;
    }
    if (0 == ok) {
      internal_libxs_predict_neighbors_free(trial);
      trial = NULL;
    }
  }
  LIBXS_PREDICT_FREE(buf, buf_pool);
  LIBXS_PREDICT_FREE(role, role_pool);
  return trial;
}


/**
 * One task's share of the scoring without a probe: the clusters it takes, every
 * entry held out in turn. Each entry's neighbourhood is gathered once, to the
 * widest count on the grid, and every smaller count reads the nearest prefix of
 * it, so the grid costs one scan per entry rather than one per count. Ties in
 * distance are broken by position in the cluster, which is the order the scan
 * admits them in. An exact duplicate answers by the match whatever the count,
 * and it is never settled here: the entry is held out, not queried.
 *
 * The mean is scored and not the median, because the choice between the two is
 * made after the count and reads it. The probe made that choice for itself.
 */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_loo(
  const libxs_predict_t* model, const internal_libxs_predict_ktrial_t* trial,
  int tid, int ntasks, double* err, double* cmin, double* cmax)
{
  const int n = model->noutputs;
  int c;
  for (c = tid; c < model->nclusters; c += ntasks) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    if (0 != internal_libxs_predict_loo_ok(cl)) {
      int e;
      for (e = 0; e < cl->nentries; ++e) {
        internal_libxs_predict_scan_t scan;
        double cand[LIBXS_PREDICT_KNN], dist[LIBXS_PREDICT_KNN];
        double iw[LIBXS_PREDICT_KNN];
        int order[LIBXS_PREDICT_KNN], t, u, j, ci;
        internal_libxs_predict_loo_gather(model, cl, e, 0, LIBXS_PREDICT_KNN,
          &scan);
        for (t = 0; t < scan.nfound; ++t) {
          for (u = t; 0 < u && (scan.dists[order[u - 1]] > scan.dists[t]
            || (scan.dists[order[u - 1]] == scan.dists[t]
              && scan.idx[order[u - 1]] > scan.idx[t])); --u)
          {
            order[u] = order[u - 1];
          }
          order[u] = t;
        }
        for (t = 0; t < scan.nfound; ++t) {
          dist[t] = scan.dists[order[t]];
          iw[t] = scan.iw[order[t]];
        }
        for (j = 0; j < n; ++j) {
          const double actual = cl->raw_outputs[(size_t)e * n + j];
          const double best = (0 <= scan.exact_idx)
            ? cl->raw_outputs[(size_t)scan.exact_idx * n + j]
            : cl->raw_outputs[j];
          for (t = 0; t < scan.nfound; ++t) {
            cand[t] = cl->raw_outputs[(size_t)scan.idx[order[t]] * n + j];
          }
          for (ci = 0; ci < LIBXS_PREDICT_NNEIGHBORS; ++ci) {
            const int kc = LIBXS_MIN(internal_libxs_predict_neighbors_cand(ci),
              scan.nfound);
            const size_t at = (size_t)ci * n + j;
            double conf = 1.0;
            const double pred = internal_libxs_predict_vote(cl, j, n,
              cl->ndistinct[j], 0, 0, cand, dist, iw, kc, scan.exact, 0,
              best, &conf, NULL, 0, NULL, NULL);
            if (1e30 <= err[at]) err[at] = 0;
            if (conf < cmin[at]) cmin[at] = conf;
            if (conf > cmax[at]) cmax[at] = conf;
            err[at] += (0 != trial->kind[j])
              ? ((LIBXS_ROUNDX(int, pred) == LIBXS_ROUNDX(int, actual))
                ? 0.0 : 1.0)
              : (LIBXS_FABS(pred - actual) / trial->mad[j]);
          }
        }
      }
    }
  }
}


/**
 * One task's share of the scoring with a probe. The probe is built by the whole
 * team, and each candidate count is then applied by the builder and scored by
 * the team, every task taking a strided share of the held-back entries into its
 * own partials and its own evaluation buffer - the model's is shared, and info
 * points into it.
 */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_probe(
  libxs_barrier_t* barrier, const libxs_predict_t* model,
  const internal_libxs_predict_ktrial_t* trial, int tid, int ntasks,
  double* err, double* cmin, double* cmax)
{
  libxs_predict_t* probe = trial->probe;
  const int n = model->noutputs;
  int eval_pool = 0, pred_pool = 0;
  double* evalbuf = (double*)LIBXS_PREDICT_MALLOC(
    INTERNAL_LIBXS_PREDICT_EVALBYTES(n), eval_pool);
  double* pred = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)n * sizeof(double), pred_pool);
  int c;
  /* collective, so every task sees the same verdict in probe->built */
  libxs_predict_build_task(probe, 0, 2, 0.0, tid, ntasks);
  for (c = 0; c < LIBXS_PREDICT_NNEIGHBORS; ++c) {
    if (0 == tid && 0 != probe->built) {
      probe->kreq = internal_libxs_predict_neighbors_cand(c);
      internal_libxs_predict_kapply(probe);
    }
    libxs_barrier_wait(barrier);
    if (0 != probe->built && NULL != evalbuf && NULL != pred
      && 0 < trial->nheld)
    {
      int t, j;
      for (j = 0; j < n; ++j) err[c * n + j] = 0;
      for (t = tid; t < trial->nheld; t += ntasks) {
        const int i = trial->held[t];
        libxs_predict_info_t info;
        memset(&info, 0, sizeof(info));
        internal_libxs_predict_eval_scratch(NULL, probe,
          model->entries[i].inputs, pred, &info, 1, evalbuf);
        for (j = 0; j < n; ++j) {
          const double actual = model->entries[i].outputs[j];
          const double cf = (NULL != info.confidence)
            ? info.confidence[j] : 1.0;
          if (cf < cmin[c * n + j]) cmin[c * n + j] = cf;
          if (cf > cmax[c * n + j]) cmax[c * n + j] = cf;
          err[c * n + j] += (0 != trial->kind[j])
            ? ((LIBXS_ROUNDX(int, pred[j])
              == LIBXS_ROUNDX(int, actual)) ? 0.0 : 1.0)
            : (LIBXS_FABS(pred[j] - actual) / trial->mad[j]);
        }
      }
    }
    /* the builder moves the count only once every task has scored this one */
    libxs_barrier_wait(barrier);
  }
  LIBXS_PREDICT_FREE(pred, pred_pool);
  LIBXS_PREDICT_FREE(evalbuf, eval_pool);
}


/**
 * One task's share of the trial, into its own partials. With one task the sums
 * run in the order the serial trial took; across tasks the reduction reorders
 * the error of a many-valued output, so a count within rounding of the next can
 * fall the other way. A miss count is an integer and cannot.
 */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_task(
  libxs_barrier_t* barrier, libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_ktrial_t* trial = model->sync_ktrial;
  const size_t nn = (size_t)LIBXS_PREDICT_NNEIGHBORS * model->noutputs;
  double* err = trial->part + (size_t)tid * 3 * nn;
  double* cmin = err + nn;
  double* cmax = cmin + nn;
  size_t k;
  /**
   * Every slot starts unreachable, because a candidate that finds nothing to
   * score leaves its slots untouched, and a zero there reads as a perfect score
   * that would win the reduction outright.
   */
  for (k = 0; k < nn; ++k) {
    err[k] = 1e30;
    cmin[k] = 1e30;
    cmax[k] = -1e30;
  }
  if (NULL == trial->probe) {
    internal_libxs_predict_neighbors_loo(model, trial, tid, ntasks,
      err, cmin, cmax);
  }
  else {
    internal_libxs_predict_neighbors_probe(barrier, model, trial, tid, ntasks,
      err, cmin, cmax);
  }
  /* the builder reduces what every task wrote */
  libxs_barrier_wait(barrier);
}


/**
 * The builder's close of the trial: sum the tasks' errors, take the extremes of
 * their confidences, and settle one count per output.
 */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_finish(
  libxs_predict_t* model, int ntasks)
{
  internal_libxs_predict_ktrial_t* trial = model->sync_ktrial;
  const int n = model->noutputs;
  const size_t nn = (size_t)LIBXS_PREDICT_NNEIGHBORS * n;
  if (NULL == trial->probe || 0 != trial->probe->built) {
    double* err = trial->part;
    double* cmin = err + nn;
    double* cmax = cmin + nn;
    size_t k;
    int t, c, j;
    for (t = 1; t < ntasks; ++t) {
      const double* terr = trial->part + (size_t)t * 3 * nn;
      const double* tmin = terr + nn;
      const double* tmax = tmin + nn;
      for (k = 0; k < nn; ++k) {
        /* a candidate no task scored stays unreachable rather than summing */
        if (1e30 > terr[k]) err[k] = (1e30 > err[k]) ? (err[k] + terr[k]) : terr[k];
        if (tmin[k] < cmin[k]) cmin[k] = tmin[k];
        if (tmax[k] > cmax[k]) cmax[k] = tmax[k];
      }
    }
    model->k_sel = (int*)malloc((size_t)n * sizeof(int));
    if (NULL != model->k_sel) {
      /**
       * A count whose confidence never moves is refused rather than traded
       * against: one neighbour votes unanimously whatever it holds, so the
       * confidence is 1.0 everywhere and carries no information, and a gate
       * reading it selects every query. Scoring the confidence instead (a
       * Brier score over the reported value) was measured and is worse in
       * the other direction, taking the crystal corpus to the widest count
       * in the grid and 57.2% where the miss rate alone reaches 68.2%:
       * calibration improves with a wide neighbourhood and accuracy does
       * not. Excluding the degenerate end costs nothing that carries
       * information.
       */
      for (c = 0; c < LIBXS_PREDICT_NNEIGHBORS; ++c) {
        for (j = 0; j < n; ++j) {
          if (0 != trial->kind[j] && cmax[c * n + j] <= cmin[c * n + j]) {
            err[c * n + j] = 1e30;
          }
        }
      }
      for (j = 0; j < n; ++j) {
        int best = 0;
        /**
         * A tie goes to the larger count. A strict comparison kept the
         * first candidate, which is one neighbour, and a corpus of
         * near-duplicate inputs ties often enough that the grid order
         * decided the count rather than the evidence.
         */
        for (c = 1; c < LIBXS_PREDICT_NNEIGHBORS; ++c) {
          if (err[c * n + j] <= err[best * n + j]) best = c;
        }
        model->k_sel[j] = internal_libxs_predict_neighbors_cand(best);
        /**
         * A discrete output answers by vote, and a vote of one is
         * unanimous whatever the neighbourhood holds: it pins the
         * confidence at 1.0, which leaves a gate nothing to select on,
         * and it makes the compression test vacuous (see
         * libxs_predict_compress.h). Three is the fewest that leaves
         * room for a minority. The count is still chosen by the trial;
         * this only refuses the degenerate end of the grid.
         */
      }
    }
  }
  internal_libxs_predict_neighbors_free(trial);
  model->sync_ktrial = NULL;
}
