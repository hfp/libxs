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


/**
 * Resolve one neighbour count per output, writing model->k_sel.
 *
 * A single probe build serves the whole grid, because the count changes nothing
 * about the model and only how many neighbours the vote reads. That is what
 * makes choosing this per output affordable where choosing the mode per output
 * was not, and the grid being ordered is what makes it work: a pick one step off
 * the optimum is one step off, not a different model.
 *
 * The held-back entries are a contiguous tail for a timeseries and a shuffled
 * fifth otherwise. Overlapping windows share timesteps, so a shuffled split of
 * them would leave a validation window's own history in the corpus that predicts
 * it, and every candidate would look equally good.
 */
LIBXS_API_INLINE void internal_libxs_predict_neighbors_select(
  libxs_predict_t* model)
{
  const int p = model->nentries;
  const int n = model->noutputs;
  const int nfit = (int)(p * 0.8 + 0.5);
  int role_pool = 0, kind_pool = 0, mad_pool = 0, buf_pool = 0;
  char* role = (char*)LIBXS_PREDICT_MALLOC((size_t)p, role_pool);
  int* kind = (int*)LIBXS_PREDICT_MALLOC((size_t)n * sizeof(int), kind_pool);
  double* mad = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)n * sizeof(double), mad_pool);
  double* buf = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)p * sizeof(double), buf_pool);
  if (NULL != role && NULL != kind && NULL != mad && NULL != buf
    && 8 < p && NULL == model->k_sel)
  {
    libxs_predict_t* probe = libxs_predict_create(model->ninputs, n);
    const int series = (0 < model->nts && 0 < model->nseries) ? 1 : 0;
    int i;
    if (0 != series) {
      for (i = 0; i < p; ++i) role[i] = (char)((i < nfit) ? 0 : 1);
    }
    else {
      const size_t co = libxs_coprime2((size_t)p);
      for (i = 0; i < p; ++i) {
        role[LIBXS_SHUFFLE_INDEX(i, p, co, 0)] = (char)((i < nfit) ? 0 : 1);
      }
    }
    internal_libxs_predict_decompose_kind(model, role, kind, mad, buf);
    if (NULL != probe) {
      double* pred = (double*)malloc((size_t)n * sizeof(double));
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
      for (i = 0; i < p; ++i) {
        if (0 == role[i]) {
          libxs_predict_push(NULL, probe, model->entries[i].inputs,
            model->entries[i].outputs);
        }
      }
      if (NULL != pred && 0 < probe->nentries
        && EXIT_SUCCESS == libxs_predict_build(probe, 0, 2, 0.0))
      {
        double* err = (double*)malloc(
          (size_t)LIBXS_PREDICT_NNEIGHBORS * (size_t)n * 3 * sizeof(double));
        double* cmin = err + (size_t)LIBXS_PREDICT_NNEIGHBORS * n;
        double* cmax = cmin + (size_t)LIBXS_PREDICT_NNEIGHBORS * n;
        if (NULL != err) {
          int c, j;
          /**
           * Every slot starts unreachable, because the scan below abandons the
           * grid as soon as a candidate finds nothing to score and leaves the
           * remaining slots untouched. A zero there reads as a perfect score
           * and would win the reduction outright.
           */
          for (c = 0; c < LIBXS_PREDICT_NNEIGHBORS * n; ++c) {
            err[c] = 1e30;
            cmin[c] = 1e30;
            cmax[c] = -1e30;
          }
          for (c = 0; c < LIBXS_PREDICT_NNEIGHBORS; ++c) {
            int nval = 0;
            probe->kreq = internal_libxs_predict_neighbors_cand(c);
            internal_libxs_predict_kapply(probe);
            for (j = 0; j < n; ++j) err[c * n + j] = 0;
            for (i = 0; i < p; ++i) {
              if (1 == role[i]) {
                libxs_predict_info_t info;
                memset(&info, 0, sizeof(info));
                libxs_predict_eval(NULL, probe, model->entries[i].inputs,
                  pred, &info, 1);
                for (j = 0; j < n; ++j) {
                  const double actual = model->entries[i].outputs[j];
                  const double cf = (NULL != info.confidence)
                    ? info.confidence[j] : 1.0;
                  if (cf < cmin[c * n + j]) cmin[c * n + j] = cf;
                  if (cf > cmax[c * n + j]) cmax[c * n + j] = cf;
                  err[c * n + j] += (0 != kind[j])
                    ? ((LIBXS_ROUNDX(int, pred[j])
                      == LIBXS_ROUNDX(int, actual)) ? 0.0 : 1.0)
                    : (LIBXS_FABS(pred[j] - actual) / mad[j]);
                }
                ++nval;
              }
            }
            if (0 >= nval) c = LIBXS_PREDICT_NNEIGHBORS;
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
              if (0 != kind[j] && cmax[c * n + j] <= cmin[c * n + j]) {
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
          free(err);
        }
      }
      free(pred);
      libxs_predict_destroy(probe);
    }
  }
  LIBXS_PREDICT_FREE(buf, buf_pool);
  LIBXS_PREDICT_FREE(mad, mad_pool);
  LIBXS_PREDICT_FREE(kind, kind_pool);
  LIBXS_PREDICT_FREE(role, role_pool);
}
