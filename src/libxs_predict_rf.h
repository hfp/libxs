/* bins per input for split finding; one byte holds the index */
#if !defined(LIBXS_PREDICT_RF_NBINS)
#  define LIBXS_PREDICT_RF_NBINS 256
#endif
/**
 * Rows a node must hold before its split is found over the bins instead of over
 * the sorted column. It is a crossover and not a preference: the histogram costs
 * one gather over the subset plus a scan of NBINS candidates, and the sorted
 * search costs a sort plus a scan of as many candidates as the subset has
 * distinct values, so below a few rows per bin the fixed scan is the larger of
 * the two and the bins are mostly empty besides.
 */
#if !defined(LIBXS_PREDICT_RF_BINMIN)
#  define LIBXS_PREDICT_RF_BINMIN 1024
#endif
/**
 * Rows a corpus must hold before it is binned at all. The node crossover above is
 * about which search is cheaper; this is about whether to approximate a search at
 * all, and below this size the answer is no: the sorted search over a corpus this
 * small is seconds of the build, so there is nothing to buy, and an approximation
 * that buys nothing can still cost.
 *
 * Binning EVERY node was tried before and withdrawn for costing accuracy, and the
 * node crossover above is what answers the cause: the bins are global, so a DEEP
 * node spanning a few of them has a few candidate thresholds where the sorted
 * search over its own rows had one per distinct value it holds. Neither edge
 * placement nor bin count accounted for the loss.
 *
 * So this is the size regime the histogram is FOR rather than a measured
 * crossover. Every corpus tuned before the bins existed is under it and is split
 * exactly at every node, which is what keeps them answering as they did.
 */
#if !defined(LIBXS_PREDICT_RF_BINROWS)
#  define LIBXS_PREDICT_RF_BINROWS 262144
#endif
/**
 * Rows the bin edges are placed from. Hundreds per bin is ample for a quantile,
 * and a bound rather than a share keeps the sort off the corpus size.
 */
#if !defined(LIBXS_PREDICT_RF_SKETCH)
#  define LIBXS_PREDICT_RF_SKETCH 65536
#endif
/**
 * Bytes of histogram one split may hold. It buys the width of the accumulating
 * pass, so it wants to be a cache the pass stays inside of.
 */
#if !defined(LIBXS_PREDICT_RF_HISTMAX)
#  define LIBXS_PREDICT_RF_HISTMAX 65536
#endif
/**
 * Shrinkage applied to each boosted stage. Regression uses it by default. A
 * folded vote share is already unbiased, so folded outputs attempt corrections
 * only when the RF_RATE environment variable is set explicitly.
 */
#if !defined(LIBXS_PREDICT_RF_RATE)
#  define LIBXS_PREDICT_RF_RATE 0.1
#endif
/**
 * Consecutive stages allowed not to improve before boosting stops. Each stage
 * scores on its own tree's out-of-bag rows, a different subset every time, so
 * a single stage that fails to improve is noise rather than a trend.
 */
#if !defined(LIBXS_PREDICT_RF_PATIENCE)
#  define LIBXS_PREDICT_RF_PATIENCE 3
#endif
/**
 * One row in this many is held back from every stage's leaf means. Half select
 * corrections; half calibrate and train the other folds of ordinary trees.
 */
#if !defined(LIBXS_PREDICT_RF_HOLD)
#  define LIBXS_PREDICT_RF_HOLD 5
#endif
#if !defined(LIBXS_PREDICT_RF_SEED)
#  define LIBXS_PREDICT_RF_SEED 1013
#endif
/* smallest parent worth splitting; a finer one buys capacity, and costs it */
#if !defined(LIBXS_PREDICT_RF_MINLEAF)
#  define LIBXS_PREDICT_RF_MINLEAF 3
#endif
/**
 * Nodes a tree may hold. This is a memory bound and nothing else: a task builds
 * one tree at a time into a scratch of this many nodes, so the peak is one such
 * scratch per task. It is fixed rather than derived from the machine, so that a
 * corpus yields the same forest whatever the thread count.
 *
 * It used to be 32767 because a saved node index was a signed 16-bit number, and
 * it kept that value after the index was widened. That mattered more than a stale
 * constant usually does, because the budget is what sets leaf_floor
 * (2*nentries/MAXNODES): a fixed budget forces coarser trees as the corpus grows,
 * which is why accuracy stopped improving with data. A finer budget than this one
 * still buys a little accuracy, and stops being worth the memory.
 *
 * The budget raises the leaf floor rather than truncating growth: growth is
 * depth-first, so hitting the ceiling leaves the first subtree grown and every
 * later one a stub, worth 9 points on a million rows.
 */
#if !defined(LIBXS_PREDICT_RF_MAXNODES)
#  define LIBXS_PREDICT_RF_MAXNODES 524287
#endif
#if !defined(LIBXS_PREDICT_RF_NTREES)
#  define LIBXS_PREDICT_RF_NTREES 100
#endif
/** Concurrent tree working sets when caller tasks outnumber trees. */
#if !defined(LIBXS_PREDICT_RF_BUILD_TEAMS)
#  define LIBXS_PREDICT_RF_BUILD_TEAMS 100
#endif
/**
 * Trees per candidate while scoring depth: enough to average out the
 * bootstrap, few enough that trying four depths is not four full builds.
 */
#if !defined(LIBXS_PREDICT_RF_PROBE)
#  define LIBXS_PREDICT_RF_PROBE 12
#endif
/**
 * Bins over native RF confidence, each carrying its empirical correctness.
 * Few enough that every bin is populated on a modest corpus.
 */
#if !defined(LIBXS_PREDICT_RF_CALIB)
#  define LIBXS_PREDICT_RF_CALIB 16
#endif
/**
 * Rows used to fit the automatic OOB curve. Hundreds per bin are enough, and
 * a fixed bound keeps calibration independent of corpus size.
 */
#if !defined(LIBXS_PREDICT_RF_CALIB_SAMPLE)
#  define LIBXS_PREDICT_RF_CALIB_SAMPLE 8192
#endif
/** Calibration rows are divided over this many disjoint groups of trees. */
#if !defined(LIBXS_PREDICT_RF_CALIB_FOLDS)
#  define LIBXS_PREDICT_RF_CALIB_FOLDS 4
#endif
/** Queries advanced together through one packed tree. */
#if !defined(LIBXS_PREDICT_RF_PACKET)
#  define LIBXS_PREDICT_RF_PACKET 16
#endif


LIBXS_API_INLINE int internal_libxs_predict_rf_pair_cmp(
  const void* a, const void* b, void* ctx)
{
  const double va = ((const internal_libxs_predict_rf_pair_t*)a)->val;
  const double vb = ((const internal_libxs_predict_rf_pair_t*)b)->val;
  LIBXS_UNUSED(ctx);
  return (va > vb) - (va < vb);
}


/**
 * Split of a node found over the sorted column: every distinct value the subset
 *  holds is a candidate, and the subset is sorted once per candidate feature.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_split_sort(
  const internal_libxs_predict_entry_t* entries,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_build_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass,
  double* values_scratch, int* index_scratch)
{
  /**
   * The fold is as wide as the corpus has classes, not as wide as it can be:
   * sweeping all 128 at every candidate was most of the split's own cost
   */
  const int ncls = (0 < nclass && 128 >= nclass) ? nclass : 128;
  int result = 0;
  /** Negative until a candidate is seen: unlike impurity, a sum of squares has
   *  no upper bound that could serve as the initial best. */
  double best_score = -1.0;
  double mu = 0;
  int trial, i;
  int keys_pool = 0, ord_pool = 0;
  const size_t feat_coprime = libxs_coprime2((size_t)nfeat);
  /* sorting an order over one column reaches the radix path in libxs_sort,
   * which a value/index pair cannot: its comparator is not recognized */
  double* keys = values_scratch;
  int* ord = index_scratch;
  if (NULL == keys) {
    keys = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)nsub * sizeof(double), keys_pool);
  }
  if (NULL == ord) {
    ord = (int*)LIBXS_PREDICT_MALLOC((size_t)nsub * sizeof(int), ord_pool);
  }
  node->feature = -1;
  node->label = -1;
  if (NULL != keys && NULL != ord) {
    /**
     * Deviations are taken about the subset mean rather than about zero: the
     * sums of squares of an output that is large and narrow differ in their
     * trailing digits only, and a split's improvement is lost in the
     * cancellation. Subtracting the mean first puts that improvement in the
     * leading digits.
     */
    if (0 != regress && 0 < nsub) {
      for (i = 0; i < nsub; ++i) mu += entries[subset[i]].outputs[output_idx];
      mu /= nsub;
    }
    for (trial = 0; trial < nfeatsub; ++trial) {
      const int f = (int)(LIBXS_SHUFFLE_INDEX(
        (size_t)trial, (size_t)nfeat, feat_coprime, seed) % (size_t)nfeat);
      int nleft, nright;
      for (i = 0; i < nsub; ++i) {
        keys[i] = entries[subset[i]].inputs[f];
        ord[i] = i;
      }
      libxs_sort(ord, nsub, sizeof(*ord), libxs_cmp_f64_idx, keys);
      if (0 != regress) {
        double sum_l = 0, sqr_l = 0, sum_t = 0, sqr_t = 0;
        for (i = 0; i < nsub; ++i) {
          const double d = entries[subset[ord[i]]].outputs[output_idx] - mu;
          sum_t += d;
          sqr_t += d * d;
        }
        nright = nsub; nleft = 0;
        for (i = 0; i < nsub - 1; ++i) {
          const double d = entries[subset[ord[i]]].outputs[output_idx] - mu;
          sum_l += d; sqr_l += d * d; ++nleft;
          --nright;
          if (keys[ord[i]] == keys[ord[i + 1]]) continue;
          /** A leaf below the floor is what makes the node count unbounded: the
           *  floor is otherwise only a reason not to split a parent, so a parent
           *  just above it splits off a single entry and the tree grows a leaf per
           *  entry. Honouring it on both sides is what makes 2*nsub/min_leaf the
           *  bound the caller sizes the node budget from.
           */
          if (nleft < min_leaf || nright < min_leaf) continue;
          /** The right side is the total less the left rather than a second
           *  running sum: subtracting each element in turn would accumulate the
           *  cancellation of every step, and the right side ends near zero. */
          { const double sum_r = sum_t - sum_l;
            const double sqr_r = sqr_t - sqr_l;
            const double sse = (sqr_l - sum_l * sum_l / nleft)
              + (sqr_r - sum_r * sum_r / nright);
            if (0 > best_score || sse < best_score) {
              best_score = sse;
              node->feature = f;
              node->threshold = 0.5 * (keys[ord[i]] + keys[ord[i + 1]]);
            }
          }
        }
      }
      else {
        int left_counts[128], right_counts[128];
        int k;
        memset(right_counts, 0, (size_t)ncls * sizeof(int));
        nright = nsub; nleft = 0;
        for (i = 0; i < nsub; ++i) {
          int lab = (LIBXS_ROUNDX(int, entries[subset[ord[i]]].outputs[output_idx]) + label_off) & 127;
          if (lab >= ncls) lab = ncls - 1;
          ++right_counts[lab];
        }
        memset(left_counts, 0, (size_t)ncls * sizeof(int));
        for (i = 0; i < nsub - 1; ++i) {
          int label = (LIBXS_ROUNDX(int, entries[subset[ord[i]]].outputs[output_idx]) + label_off) & 127;
          if (label >= ncls) label = ncls - 1;
          ++left_counts[label]; ++nleft;
          --right_counts[label]; --nright;
          if (keys[ord[i]] == keys[ord[i + 1]]) continue;
          if (nleft < min_leaf || nright < min_leaf) continue;
          { double gini_l = 1.0, gini_r = 1.0, gini;
            for (k = 0; k < ncls; ++k) {
              if (left_counts[k] > 0) {
                double p = (double)left_counts[k] / nleft;
                gini_l -= p * p;
              }
              if (right_counts[k] > 0) {
                double p = (double)right_counts[k] / nright;
                gini_r -= p * p;
              }
            }
            gini = ((double)nleft * gini_l + (double)nright * gini_r) / nsub;
            if (0 > best_score || gini < best_score) {
              best_score = gini;
              node->feature = f;
              node->threshold = 0.5 * (keys[ord[i]] + keys[ord[i + 1]]);
            }
          }
        }
      }
    }
  }
  if (NULL == index_scratch) LIBXS_PREDICT_FREE(ord, ord_pool);
  if (NULL == values_scratch) LIBXS_PREDICT_FREE(keys, keys_pool);
  result = (node->feature >= 0) ? 1 : 0;
  return result;
}


/**
 * Split of a node found over the binned inputs: one pass over the subset
 * accumulates a histogram per candidate feature, and the candidates are then the
 * bin edges rather than every distinct value the subset holds. No sort, and a
 * scan whose length is the number of bins instead of the size of the subset.
 *
 * The partition it scores is exactly the one the caller re-forms from the raw
 * column, because a row is binned into the first bin whose upper edge holds it:
 * "bin at most b" and "value at most edge[b+1]" are then the same set, and
 * edge[b+1] is what the threshold is set to. Nothing here may loosen that.
 *
 * The features of a group are accumulated in one pass rather than one at a time.
 * The subset is scattered over the corpus, so the pass is a gather and its cost
 * is the walk, not the arithmetic: one walk feeding every histogram of the group
 * costs what one walk feeding a single histogram costs. The group is as wide as
 * the histogram budget allows - every candidate feature of a corpus that folds to
 * few classes, one feature of a corpus that folds to many.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_split_hist(
  const internal_libxs_predict_entry_t* entries,
  const unsigned char* bins, const double* bin_edge, int nbins,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_build_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass,
  double* values_scratch, int* index_scratch)
{
  /** As in the sorted search, the fold is as wide as the corpus has classes. */
  const int ncls = (0 != regress) ? 1
    : ((0 < nclass && 128 >= nclass) ? nclass : 128);
  /** Count, sum and sum of squares per bin for a quantity, a count per class for
   *  a label: one array serves both read-outs at two widths. */
  const int width = (0 != regress) ? 3 : ncls;
  const size_t per = (size_t)nbins * width;
  const size_t feat_coprime = libxs_coprime2((size_t)nfeat);
  double best_score = -1.0, mu = 0;
  int nfused = (int)(LIBXS_PREDICT_RF_HISTMAX / (per * sizeof(double)));
  int acc_pool = 0, fsel_pool = 0;
  double* acc = values_scratch;
  int* fsel = index_scratch;
  int base, i, j, k, b, result;
  if (1 > nfused) nfused = 1;
  if (nfeatsub < nfused) nfused = nfeatsub;
  if (NULL == acc) {
    acc = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)nfused * per * sizeof(double), acc_pool);
  }
  if (NULL == fsel) {
    fsel = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)nfeatsub * sizeof(int), fsel_pool);
  }
  node->feature = -1;
  node->label = -1;
  if (NULL != acc && NULL != fsel) {
    /* the same draw the sorted search makes, so the two paths differ in the
     * resolution of the candidates and in nothing else */
    for (i = 0; i < nfeatsub; ++i) {
      fsel[i] = (int)(LIBXS_SHUFFLE_INDEX((size_t)i, (size_t)nfeat,
        feat_coprime, seed) % (size_t)nfeat);
    }
    if (0 != regress) {
      for (i = 0; i < nsub; ++i) mu += entries[subset[i]].outputs[output_idx];
      if (0 < nsub) mu /= nsub;
    }
    for (base = 0; base < nfeatsub; base += nfused) {
      const int nf = LIBXS_MIN(nfused, nfeatsub - base);
      memset(acc, 0, (size_t)nf * per * sizeof(double));
      for (i = 0; i < nsub; ++i) {
        const int row = subset[i];
        const unsigned char* const bin = bins + (size_t)row * nfeat;
        if (0 != regress) {
          const double d = entries[row].outputs[output_idx] - mu;
          for (j = 0; j < nf; ++j) {
            double* const h = acc
              + ((size_t)j * nbins + bin[fsel[base + j]]) * width;
            h[0] += 1.0;
            h[1] += d;
            h[2] += d * d;
          }
        }
        else {
          int lab = (LIBXS_ROUNDX(int,
            entries[row].outputs[output_idx]) + label_off) & 127;
          if (lab >= ncls) lab = ncls - 1;
          for (j = 0; j < nf; ++j) {
            acc[((size_t)j * nbins + bin[fsel[base + j]]) * width + lab] += 1.0;
          }
        }
      }
      for (j = 0; j < nf; ++j) {
        const double* const hist = acc + (size_t)j * per;
        const double* const edge = bin_edge
          + (size_t)fsel[base + j] * (nbins + 1);
        if (0 != regress) {
          double tot_n = 0, tot_s = 0, tot_q = 0, nl = 0, sl = 0, ql = 0;
          for (b = 0; b < nbins; ++b) {
            const double* const h = hist + (size_t)b * width;
            tot_n += h[0];
            tot_s += h[1];
            tot_q += h[2];
          }
          for (b = 0; b < nbins - 1; ++b) {
            const double* const h = hist + (size_t)b * width;
            /* an empty bin would score the partition its predecessor scored */
            if (0 >= h[0]) continue;
            nl += h[0];
            sl += h[1];
            ql += h[2];
            { const double nr = tot_n - nl;
              if (nl < min_leaf || nr < min_leaf) continue;
              /** The right side is the total less the left, as in the sorted
               *  search: a second running sum ends near zero and collects the
               *  cancellation of every step it took to get there. */
              { const double sr = tot_s - sl, qr = tot_q - ql;
                const double sse = (ql - sl * sl / nl) + (qr - sr * sr / nr);
                if (0 > best_score || sse < best_score) {
                  best_score = sse;
                  node->feature = fsel[base + j];
                  node->threshold = edge[b + 1];
                }
              }
            }
          }
        }
        else {
          double tot[128], cl[128];
          double tot_n = 0, nl = 0;
          for (k = 0; k < ncls; ++k) {
            tot[k] = 0;
            cl[k] = 0;
          }
          for (b = 0; b < nbins; ++b) {
            const double* const h = hist + (size_t)b * width;
            for (k = 0; k < ncls; ++k) tot[k] += h[k];
          }
          for (k = 0; k < ncls; ++k) tot_n += tot[k];
          for (b = 0; b < nbins - 1; ++b) {
            const double* const h = hist + (size_t)b * width;
            double nb = 0;
            for (k = 0; k < ncls; ++k) {
              cl[k] += h[k];
              nb += h[k];
            }
            if (0 >= nb) continue;
            nl += nb;
            { const double nr = tot_n - nl;
              if (nl < min_leaf || nr < min_leaf) continue;
              { double gini_l = 1.0, gini_r = 1.0, gini;
                for (k = 0; k < ncls; ++k) {
                  const double cr = tot[k] - cl[k];
                  if (0 < cl[k]) {
                    const double q = cl[k] / nl;
                    gini_l -= q * q;
                  }
                  if (0 < cr) {
                    const double q = cr / nr;
                    gini_r -= q * q;
                  }
                }
                gini = (nl * gini_l + nr * gini_r) / nsub;
                if (0 > best_score || gini < best_score) {
                  best_score = gini;
                  node->feature = fsel[base + j];
                  node->threshold = edge[b + 1];
                }
              }
            }
          }
        }
      }
    }
  }
  if (NULL == index_scratch) LIBXS_PREDICT_FREE(fsel, fsel_pool);
  if (NULL == values_scratch) LIBXS_PREDICT_FREE(acc, acc_pool);
  result = (node->feature >= 0) ? 1 : 0;
  return result;
}


/**
 * A wide node is split over the bins and a narrow one over the sorted column,
 * because each is the cheaper of the two where it is used. The crossover is what
 * confines the approximation to the top of a tree: the coarse splits, where 256
 * quantiles resolve more than the split needs, and where the sort the histogram
 * replaces was costing the whole subset. Deeper down nothing changes, and a
 * corpus under BINROWS rows is never binned, so it is split exactly throughout.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_split(
  const internal_libxs_predict_entry_t* entries,
  const unsigned char* bins, const double* bin_edge, int nbins,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_build_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass,
  double* values_scratch, int* index_scratch)
{
  int result;
  if (NULL != bins && NULL != bin_edge && 0 < nbins
    && LIBXS_PREDICT_RF_BINMIN <= nsub)
  {
    result = internal_libxs_predict_rf_split_hist(entries, bins, bin_edge,
      nbins, subset, nsub, nfeat, nfeatsub, node, seed, output_idx, label_off,
      regress, min_leaf, nclass, values_scratch, index_scratch);
  }
  else {
    result = internal_libxs_predict_rf_split_sort(entries, subset, nsub, nfeat,
      nfeatsub, node, seed, output_idx, label_off, regress, min_leaf, nclass,
      values_scratch, index_scratch);
  }
  return result;
}


/**
 * Features a split samples: the square root of what there is, which is the choice
 * that makes a forest a forest. Here rather than at each caller so they agree.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_nfeatsub(int nfeat)
{
  int result = (int)(sqrt((double)nfeat) + 0.5);
  if (1 > result) result = 1;
  return result;
}


/**
 * Grows a subtree from one unit of work into `nodes`, whose entry 0 is that unit's
 * own node. The unit is `si0`, `nc0` and `depth0`: the rows it covers as a range
 * into `subset`, and how deep it already sits. Growth is depth-first, as it has
 * always been.
 *
 * `frontier` bounds the pending stack. Reaching it stops the loop and leaves the
 * pending units in `fr_*`, which is how several tasks take work from one tree: a
 * pending unit is a node that exists and has not been grown, and the rows of any
 * two of them are disjoint ranges of `subset`, so growing them shares nothing.
 * Zero grows the subtree whole and writes no frontier, which is what a caller that
 * wants one tree in one task passes.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_build_part(
  const internal_libxs_predict_rf_grow_t* g, int* subset,
  int si0, int nc0, int depth0,
  internal_libxs_predict_rf_build_node_t* nodes, int max_nodes,
  int frontier, int* fr_si, int* fr_nc, int* fr_depth, int* fr_node,
  int* fr_count)
{
  const internal_libxs_predict_entry_t* entries = g->entries;
  const unsigned char* bins = g->bins;
  const double* bin_edge = g->bin_edge;
  const int nbins = g->nbins, nfeat = g->nfeat, nfeatsub = g->nfeatsub;
  const int max_depth = g->max_depth, min_leaf = g->min_leaf;
  const int leaf_floor = g->leaf_floor, output_idx = g->output_idx;
  const int label_off = g->label_off, regress = g->regress, nclass = g->nclass;
  int stack_subset[64], stack_count[64], stack_depth[64], stack_node[64];
  int sp = 0, nnodes = 0;
  stack_subset[0] = si0;
  stack_count[0] = nc0;
  stack_depth[0] = depth0;
  stack_node[0] = nnodes++;
  nodes[0].feature = -1;
  nodes[0].left = -1;
  nodes[0].right = -1;
  nodes[0].label = 0;
  nodes[0].value = 0;
  nodes[0].threshold = 0;
  nodes[0].leafp = 0.f;
  sp = 1;
  while (sp > 0 && nnodes < max_nodes - 2 && (0 == frontier || sp < frontier)) {
    const int si = stack_subset[--sp];
    const int nc = stack_count[sp];
    const int depth = stack_depth[sp];
    const int ni = stack_node[sp];
    int best_label = 0, best_count = 0, pure = 0, k;
    double mean = 0, dev = 0;
    internal_libxs_predict_rf_build_node_t split;
    LIBXS_MEMZERO(&split);
    if (0 != regress) {
      for (k = 0; k < nc; ++k) {
        mean += entries[subset[si + k]].outputs[output_idx];
      }
      if (0 < nc) mean /= nc;
      for (k = 0; k < nc; ++k) {
        const double d = entries[subset[si + k]].outputs[output_idx] - mean;
        dev += d * d;
      }
      /** A constant subset has nothing left to split on; without this the
       *  best split of no variance is still taken and grows dead nodes. */
      pure = (0 == dev) ? 1 : 0;
    }
    else {
      int counts[128] = { 0 };
      for (k = 0; k < nc; ++k) {
        ++counts[(LIBXS_ROUNDX(int, entries[subset[si + k]].outputs[output_idx]) + label_off) & 127];
      }
      for (k = 0; k < 128; ++k) {
        if (counts[k] > best_count) { best_count = counts[k]; best_label = k; }
      }
      mean = (double)best_label;
      pure = (best_count == nc) ? 1 : 0;
    }
    nodes[ni].label = best_label;
    nodes[ni].value = mean;
    /* what this read-out would be worth if the node ends as a leaf; a folded
     * output only, since a real-valued one reports no share to begin with */
    nodes[ni].leafp = (0 == regress && 0 < nc)
      ? (float)((best_count + 1.0) / (nc + ((0 < nclass) ? nclass : 1)))
      : 0.f;
    if (depth >= max_depth || nc <= min_leaf || 0 != pure
      || 0 == internal_libxs_predict_rf_split(entries, bins, bin_edge, nbins,
        subset + si, nc, nfeat, nfeatsub, &split,
        (size_t)si * 2654435761u + (size_t)nc, output_idx,
        label_off, regress, leaf_floor, nclass, g->values_scratch,
        g->index_scratch))
    {
      nodes[ni].feature = -1;
      continue;
    }
    { int* sub = subset + si;
      int i, nleft = 0, nright = 0;
      nodes[ni].feature = split.feature;
      nodes[ni].threshold = split.threshold;
      for (i = 0; i < nc; ++i) {
        if (entries[sub[i]].inputs[split.feature] <= split.threshold) ++nleft;
      }
      nright = nc - nleft;
      if (0 == nleft || 0 == nright) { nodes[ni].feature = -1; continue; }
      { int part_pool = 0;
        int* part = g->index_scratch;
        if (NULL == part) {
          part = (int*)LIBXS_PREDICT_MALLOC(
            (size_t)nc * sizeof(int), part_pool);
        }
        if (NULL != part) {
          int li = 0, ri = 0;
          for (i = 0; i < nc; ++i) {
            if (entries[sub[i]].inputs[split.feature] <= split.threshold) {
              part[li++] = sub[i];
            }
            else {
              part[nleft + ri++] = sub[i];
            }
          }
          memcpy(sub, part, (size_t)nc * sizeof(int));
          if (NULL == g->index_scratch) {
            LIBXS_PREDICT_FREE(part, part_pool);
          }
        }
        else { nodes[ni].feature = -1; continue; }
      }
      /**
       * A child is created before it is known whether the stack has room to
       * process it. Its read-out must be initialized here: when sp saturates
       * the node is never popped, and eval reads it unconditionally at a leaf.
       * Left uninitialized it takes whatever the scratch allocator returned,
       * which varies with allocation history and thread count - the cause of
       * run-to-run differences in RF results. The parent's own read-out is
       * the correct fallback, being what a leaf at this point would predict.
       */
      nodes[ni].left = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      nodes[nnodes].label = best_label;
      nodes[nnodes].value = mean;
      /* a node that stays a leaf is never given one, and it is written out */
      nodes[nnodes].threshold = 0;
      nodes[nnodes].leafp = nodes[ni].leafp;
      if (sp < 64) {
        stack_subset[sp] = si;
        stack_count[sp] = nleft;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
      nodes[ni].right = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      nodes[nnodes].label = best_label;
      nodes[nnodes].value = mean;
      /* a node that stays a leaf is never given one, and it is written out */
      nodes[nnodes].threshold = 0;
      nodes[nnodes].leafp = nodes[ni].leafp;
      if (sp < 64) {
        stack_subset[sp] = si + nleft;
        stack_count[sp] = nright;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
    }
  }
  /* what is left pending is the frontier: nodes that exist and are not grown */
  if (NULL != fr_count) {
    int k;
    for (k = 0; k < sp; ++k) {
      fr_si[k] = stack_subset[k];
      fr_nc[k] = stack_count[k];
      fr_depth[k] = stack_depth[k];
      fr_node[k] = stack_node[k];
    }
    *fr_count = sp;
  }
  return nnodes;
}


/**
 * Whole tree in one task: the unit is the root, covering every row, and nothing is
 * left pending. Kept as its own entry point because that is what most callers want
 * and it is the shape the depth probe needs.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree(
  const internal_libxs_predict_rf_grow_t* g, int* subset, int nsub,
  internal_libxs_predict_rf_build_node_t* nodes, int max_nodes)
{
  return internal_libxs_predict_rf_build_part(g, subset, 0, nsub, 0,
    nodes, max_nodes, 0, NULL, NULL, NULL, NULL, NULL);
}


/**
 * Renumbers a grown tree into the order one task growing it alone would produce.
 * The numbering follows the SHAPE and nothing else - a split takes both children
 * at once, then the right is descended first - and the shape does not depend on
 * who grew which part of it. That is what lets a tree assembled from separately
 * grown subtrees serialize byte for byte like a tree grown in one piece. Nodes the
 * assembly left unreachable are dropped, so it also compacts.
 *
 * `map` is scratch of `nsrc` entries. Returns the count written to `dst`, or zero
 * if the tree is deeper than the traversal holds, which is the bound growth has.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_relabel(
  const internal_libxs_predict_rf_build_node_t* src, int nsrc, int root,
  int* map, internal_libxs_predict_rf_build_node_t* dst)
{
  int stack[64], sp = 0, next = 1, i, result = 0;
  for (i = 0; i < nsrc; ++i) map[i] = -1;
  if (0 < nsrc && 0 <= root && root < nsrc) {
    int ok = 1;
    map[root] = 0;
    stack[sp++] = root;
    while (0 < sp && 0 != ok) {
      const int ni = stack[--sp];
      if (0 <= src[ni].feature && 0 <= src[ni].left && 0 <= src[ni].right) {
        map[src[ni].left] = next++;
        map[src[ni].right] = next++;
        /* both children are taken before either is descended, so the pending
         * count grows by one per split exactly as it does while growing */
        if (62 >= sp) {
          stack[sp++] = src[ni].left;
          stack[sp++] = src[ni].right;
        }
        else ok = 0;
      }
    }
    if (0 != ok) {
      for (i = 0; i < nsrc; ++i) {
        if (0 <= map[i]) {
          dst[map[i]] = src[i];
          if (0 <= src[i].feature && 0 <= src[i].left && 0 <= src[i].right) {
            dst[map[i]].left = map[src[i].left];
            dst[map[i]].right = map[src[i].right];
          }
        }
      }
      result = next;
    }
  }
  return result;
}


/**
 * Grows a whole tree through the frontier decomposition: the top is grown until
 * `frontier` units are pending, each pending unit is grown on its own, and the
 * result is renumbered into the tree a single pass would have built.
 *
 * The units cover DISJOINT ranges of `subset`, which is what will let them go to
 * different tasks. Here they are grown in order, so that the decomposition can be
 * verified against the undecomposed build before any task holds one.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree_parts(
  const internal_libxs_predict_rf_grow_t* g, int* subset, int nsub,
  internal_libxs_predict_rf_build_node_t* nodes, int max_nodes, int frontier)
{
  int fr_si[64], fr_nc[64], fr_depth[64], fr_node[64], fc = 0;
  int result = internal_libxs_predict_rf_build_part(g, subset, 0, nsub, 0,
    nodes, max_nodes, frontier, fr_si, fr_nc, fr_depth, fr_node, &fc);
  int i;
  for (i = 0; i < fc && 0 < result; ++i) {
    const int base = result;
    const int room = max_nodes - base;
    if (2 < room) {
      const int np = internal_libxs_predict_rf_build_part(g, subset,
        fr_si[i], fr_nc[i], fr_depth[i], nodes + base, room,
        0, NULL, NULL, NULL, NULL, NULL);
      int j;
      /* the unit grew into its own range and numbered from zero within it */
      for (j = 0; j < np; ++j) {
        if (0 <= nodes[base + j].left) nodes[base + j].left += base;
        if (0 <= nodes[base + j].right) nodes[base + j].right += base;
      }
      /* the unit's node already exists in the top, so its grown form replaces it
       * and the copy at `base` is left for the renumbering to drop */
      nodes[fr_node[i]] = nodes[base];
      result = base + np;
    }
    else result = 0;
  }
  if (0 < fc && 0 < result) {
    int map_pool = 0, dst_pool = 0;
    int* map = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)result * sizeof(int), map_pool);
    internal_libxs_predict_rf_build_node_t* dst =
      (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
        (size_t)result * sizeof(internal_libxs_predict_rf_build_node_t),
        dst_pool);
    if (NULL != map && NULL != dst) {
      const int n = internal_libxs_predict_rf_relabel(nodes, result, 0, map, dst);
      if (0 < n) {
        memcpy(nodes, dst,
          (size_t)n * sizeof(internal_libxs_predict_rf_build_node_t));
        result = n;
      }
      else result = 0;
    }
    else result = 0;
    if (NULL != dst) LIBXS_PREDICT_FREE(dst, dst_pool);
    if (NULL != map) LIBXS_PREDICT_FREE(map, map_pool);
  }
  return result;
}


/** Calibration fold of one row, or -1 where the row fits corrections. */
LIBXS_API_INLINE int internal_libxs_predict_rf_calib_fold(
  int row, int p, size_t hold_inv)
{
  int result = -1;
  if (0 < hold_inv && 0 <= row && row < p) {
    const size_t h = LIBXS_UNSHUFFLE_INDEX((size_t)row, (size_t)p,
      hold_inv, LIBXS_PREDICT_RF_SEED);
    if (h < (size_t)(p / LIBXS_PREDICT_RF_HOLD) && 0 != (h & 1)) {
      result = (int)((h / 2) % LIBXS_PREDICT_RF_CALIB_FOLDS);
    }
  }
  return result;
}


/** The row a tree's i-th bootstrap draw lands on. */
LIBXS_API_INLINE int internal_libxs_predict_rf_draw(size_t i, size_t boot_n,
  size_t coprime, size_t seed, int p)
{
  return (int)(LIBXS_SHUFFLE_INDEX(i, boot_n, coprime, seed) % (size_t)p);
}


/**
 * Error of a small forest grown to max_depth over the first ntrain entries,
 * measured on the rest: the misclassification rate of a folded output, the mean
 * absolute error of a real-valued one. The two are never compared against each
 * other, only across the depth candidates of one output. Depth is scored
 * rather than derived because the derived 2*log2(p) is a function of the corpus
 * size alone: it says 20 for a corpus of 1339 whether that corpus has three
 * features or three hundred, and a tree that deep on three features is fitting
 * the sample.
 */
LIBXS_API_INLINE double internal_libxs_predict_rf_score(
  const internal_libxs_predict_entry_t* entries,
  int p, int m,
  int output_idx, int label_off, int max_depth, int min_leaf, int ntrain,
  int regress, int nclass)
{
  const int nt = LIBXS_PREDICT_RF_PROBE;
  /* the probe holds every one of its trees at once, so it is bounded by the
   * same budget divided among them rather than by one of its own */
  const int max_nodes = LIBXS_MIN(ntrain / min_leaf * 2 + 1,
    LIBXS_MAX(LIBXS_PREDICT_RF_MAXNODES / LIBXS_PREDICT_RF_PROBE, 1));
  int nodes_pool = 0, boot_pool = 0, nn_pool = 0;
  int values_pool = 0, index_pool = 0;
  internal_libxs_predict_rf_build_node_t* nodes =
    (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nt * (size_t)max_nodes
        * sizeof(internal_libxs_predict_rf_build_node_t), nodes_pool);
  int* bootstrap = (int*)LIBXS_PREDICT_MALLOC((size_t)ntrain * sizeof(int),
    boot_pool);
  int* nn = (int*)LIBXS_PREDICT_MALLOC((size_t)nt * sizeof(int), nn_pool);
  double* values_scratch = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)ntrain * sizeof(double), values_pool);
  int* index_scratch = (int*)LIBXS_PREDICT_MALLOC(
    (size_t)ntrain * sizeof(int), index_pool);
  double result = 1.0;
  if (NULL != nodes && NULL != bootstrap && NULL != nn
    && NULL != values_scratch && NULL != index_scratch)
  {
    const size_t boot_n = (size_t)ntrain * 2 + 1;
    const size_t boot_coprime = libxs_coprime2(boot_n);
    int t, i, wrong = 0, scored = 0;
    double err = 0;
    for (t = 0; t < nt; ++t) {
      for (i = 0; i < ntrain; ++i) {
        bootstrap[i] = (int)(LIBXS_SHUFFLE_INDEX(i, boot_n, boot_coprime,
          (size_t)t * 7 + 13) % (size_t)ntrain);
      }
      /* the probe splits exactly: it runs before the bins are filled, and it
       * ranks depths against each other rather than reporting an error */
      { internal_libxs_predict_rf_grow_t g;
        g.entries = entries; g.bins = NULL; g.bin_edge = NULL; g.nbins = 0;
        g.values_scratch = values_scratch; g.index_scratch = index_scratch;
        g.nfeat = m; g.nfeatsub = internal_libxs_predict_rf_nfeatsub(m);
        g.max_depth = max_depth; g.min_leaf = min_leaf; g.leaf_floor = min_leaf;
        g.output_idx = output_idx; g.label_off = label_off;
        g.regress = regress; g.nclass = nclass;
        nn[t] = internal_libxs_predict_rf_build_tree(&g, bootstrap, ntrain,
          nodes + (size_t)t * max_nodes, max_nodes);
      }
    }
    for (i = ntrain; i < p; ++i) {
      const double* inputs = entries[i].inputs;
      const int label =
        (LIBXS_ROUNDX(int, entries[i].outputs[output_idx]) + label_off) & 127;
      int votes[128], best_label = 0, best_count = 0, k, nvalid = 0;
      double sum = 0;
      memset(votes, 0, sizeof(votes));
      for (t = 0; t < nt; ++t) {
        const internal_libxs_predict_rf_build_node_t* tn =
          nodes + (size_t)t * max_nodes;
        int ni = 0;
        if (0 >= nn[t]) continue;
        while (ni >= 0 && ni < nn[t] && tn[ni].feature >= 0) {
          ni = (inputs[tn[ni].feature] <= tn[ni].threshold)
            ? tn[ni].left : tn[ni].right;
        }
        if (ni >= 0 && ni < nn[t]) {
          if (0 != regress) { sum += tn[ni].value; ++nvalid; }
          else ++votes[tn[ni].label & 127];
        }
      }
      if (0 != regress) {
        if (0 < nvalid) {
          err += LIBXS_FABS(sum / nvalid - entries[i].outputs[output_idx]);
        }
      }
      else {
        for (k = 0; k < 128; ++k) {
          if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
        }
        if (best_label != label) ++wrong;
      }
      ++scored;
    }
    if (0 < scored) {
      result = (0 != regress) ? (err / scored) : ((double)wrong / scored);
    }
  }
  LIBXS_PREDICT_FREE(index_scratch, index_pool);
  LIBXS_PREDICT_FREE(values_scratch, values_pool);
  LIBXS_PREDICT_FREE(nn, nn_pool);
  LIBXS_PREDICT_FREE(bootstrap, boot_pool);
  LIBXS_PREDICT_FREE(nodes, nodes_pool);
  return result;
}


/**
 * Places the bin edges at quantiles of each input and allocates the bins the
 * inputs are about to be sorted into. The quantiles are taken from a strided
 * sample rather than from the whole corpus: 65536 rows put hundreds in each of
 * 256 bins, which is what a quantile needs, and it keeps the sort off a size that
 * grows. Nothing is binned here - that is one binary search per value and the
 * corpus holds nentries*ninputs of them, so it is a task stage of its own.
 *
 * Declines where no node can reach the width that reads the bins, so a corpus
 * split exactly at every node does not allocate a byte per value to prove it.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_edges(libxs_predict_t* model)
{
  internal_libxs_predict_rf_t* const rf = model->rf;
  const int p = model->nentries;
  const int m = model->ninputs;
  if (NULL != rf && LIBXS_PREDICT_RF_BINROWS <= p && 0 < m) {
    const int nb = LIBXS_PREDICT_RF_NBINS;
    int bins_pool = 0, edge_pool = 0;
    unsigned char* const bins = (unsigned char*)LIBXS_PREDICT_MALLOC(
      (size_t)p * (size_t)m, bins_pool);
    double* const edge = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)m * (size_t)(nb + 1) * sizeof(double), edge_pool);
    if (NULL != bins && NULL != edge) {
      const int nsamp = LIBXS_MIN(p, LIBXS_PREDICT_RF_SKETCH);
      const int step = LIBXS_MAX(p / nsamp, 1);
      int spool = 0;
      double* sv = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)nsamp * sizeof(double), spool);
      int i, j, k;
      for (j = 0; j < m && NULL != sv; ++j) {
        double* const ej = edge + (size_t)j * (nb + 1);
        int ns = 0;
        for (i = 0; i < p && ns < nsamp; i += step) {
          const double v = model->entries[i].inputs[j];
          if (0 != LIBXS_NOTNAN(v)) sv[ns++] = v;
        }
        if (0 == ns) { /* an input with no value to sort has one bin */
          for (k = 0; k <= nb; ++k) ej[k] = 0;
          continue;
        }
        libxs_sort(sv, ns, sizeof(*sv), libxs_cmp_f64, NULL);
        for (k = 0; k <= nb; ++k) {
          int at = (int)((size_t)k * ns / nb);
          if (at >= ns) at = ns - 1;
          ej[k] = sv[at];
        }
      }
      if (NULL != sv) { /* nothing was placed if the sample could not be held */
        rf->bins = bins;
        rf->bin_edge = edge;
        rf->bins_pool = bins_pool;
        rf->edge_pool = edge_pool;
        rf->nbins = nb;
      }
      LIBXS_PREDICT_FREE(sv, spool);
    }
    if (0 >= rf->nbins) { /* the sorted search needs none of it */
      LIBXS_PREDICT_FREE(bins, bins_pool);
      LIBXS_PREDICT_FREE(edge, edge_pool);
    }
  }
}


/**
 * Sorts every input of every entry into its bin. Split across the tasks because
 * it is one search per value, and read by every task afterwards, so it has to be
 * complete before the first tree rather than filled as the trees need it.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_bins_tasks(
  libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_rf_t* const rf = model->rf;
  if (NULL != rf && NULL != rf->bins && NULL != rf->bin_edge && 0 < rf->nbins) {
    const int m = model->ninputs;
    const int nb = rf->nbins;
    int begin, end, i, j;
    internal_libxs_predict_split(model->nentries, tid, ntasks, &begin, &end);
    for (i = begin; i < end; ++i) {
      const double* const inputs = model->entries[i].inputs;
      unsigned char* const bin = rf->bins + (size_t)i * m;
      for (j = 0; j < m; ++j) {
        const double* const edge = rf->bin_edge + (size_t)j * (nb + 1);
        const double v = inputs[j];
        int lo = 0, hi = nb - 1;
        /** A value that is not a number is not ordered against the edges, and
         *  the raw comparison the tree re-forms the partition with sends it
         *  right. The last bin is where the histogram says the same thing. */
        if (0 == LIBXS_NOTNAN(v)) lo = nb - 1;
        else while (lo < hi) { /* first bin whose upper edge holds v */
          const int mid = (lo + hi) / 2;
          if (v <= edge[mid + 1]) hi = mid; else lo = mid + 1;
        }
        bin[j] = (unsigned char)lo;
      }
    }
  }
}


/**
 * Releases the bins once the forest is grown: split finding is what read them,
 * and boosting and the calibration descend the raw inputs.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_bins_free(libxs_predict_t* model)
{
  if (NULL != model->rf) {
    LIBXS_PREDICT_FREE(model->rf->bins, model->rf->bins_pool);
    LIBXS_PREDICT_FREE(model->rf->bin_edge, model->rf->edge_pool);
    model->rf->bins = NULL;
    model->rf->bin_edge = NULL;
    model->rf->bins_pool = 0;
    model->rf->edge_pool = 0;
    model->rf->nbins = 0;
  }
}


LIBXS_API_INLINE void internal_libxs_predict_rf_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int n = model->noutputs;
  const int ntrees = (0 < model->rf_ntrees)
    ? model->rf_ntrees : LIBXS_PREDICT_RF_NTREES;
  internal_libxs_predict_rf_t* rf =
    (internal_libxs_predict_rf_t*)calloc(1, sizeof(internal_libxs_predict_rf_t));
  if (NULL != rf) {
    rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
      (size_t)ntrees * (size_t)n, sizeof(internal_libxs_predict_rf_tree_t));
    rf->label_offset = (int*)malloc((size_t)n * sizeof(int));
    rf->regress = (int*)malloc((size_t)n * sizeof(int));
    rf->nclass = (int*)malloc((size_t)n * sizeof(int));
    rf->depth = (int*)malloc((size_t)n * sizeof(int));
    rf->ntrees = ntrees;
    rf->noutputs = n;
    if (NULL != rf->trees && NULL != rf->label_offset && NULL != rf->depth
      && NULL != rf->regress && NULL != rf->nclass)
    {
      const int derived = (int)(2.0 * log((double)p) / log(2.0));
      const int min_leaf = 5;
      const int ntrain = (int)(p * 0.8 + 0.5);
      int oi, i;
      /**
       * An output is a class only if it is written like one: every value
       * integral, and the whole range representable in the 128 labels the
       * folding leaves. Everything else is a quantity, split on variance and
       * read out as a mean. Both halves of the test matter. Without the
       * first, a magnitude of 4.5 is answered as 5 and the forest cannot beat
       * its own rounding. Without the second, an integral output spanning more
       * than the fold wraps two distant values onto one label and the forest is
       * left predicting a class it invented.
       */
      for (oi = 0; oi < n; ++oi) {
        double vmin = model->entries[0].outputs[oi];
        double vmax = vmin;
        int integral = 1;
        for (i = 0; i < p; ++i) {
          const double v = model->entries[i].outputs[oi];
          if (0 != LIBXS_NOTNAN(v)) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
          /** Rounding is only defined over the range of the integer it goes
           *  through; beyond it the value is a quantity in any case. */
          if (0 != integral && (0 == LIBXS_NOTNAN(v)
            || 2147483647.0 < LIBXS_FABS(v)
            || 0.0 != v - (double)LIBXS_ROUNDX(long long, v)))
          {
            integral = 0;
          }
        }
        rf->regress[oi] = (0 == integral || 127.0 < vmax - vmin) ? 1 : 0;
        /** Only a class has an offset, and only a class is in range for it. */
        rf->label_offset[oi] = (0 == rf->regress[oi])
          ? -LIBXS_ROUNDX(int, vmin) : 0;
        rf->nclass[oi] = (0 == rf->regress[oi])
          ? (LIBXS_ROUNDX(int, vmax) - LIBXS_ROUNDX(int, vmin) + 1) : 1;
      }
      if (0 < model->rf_depth) {
        for (oi = 0; oi < n; ++oi) rf->depth[oi] = model->rf_depth;
      }
      /**
       * Scoring is opt-in (a negative request), not the default, because it was
       * measured not to pay: on the shipped tuning corpus it moved exact match
       * over sixteen outputs by half a point and made the absolute error of the
       * widest three outputs worse, while costing several times the build.
       * It is kept because it is the only way to find out for a corpus
       * where the derived depth is wrong, and because there was previously no
       * way to ask at all.
       */
      else if (0 == model->rf_depth || ntrain <= min_leaf || ntrain >= p) {
        for (oi = 0; oi < n; ++oi) rf->depth[oi] = derived;
      }
      else for (oi = 0; oi < n; ++oi) {
        /**
         * Scored per output, not once for the model: outputs of one corpus
         * differ in how much structure there is to fit, and the shallow
         * candidates exist because the derived depth is the overfitting end of
         * the range on a small corpus with few features.
         */
        int cand[4];
        double best_err = -1.0;
        int best = derived, ci;
        cand[0] = 3;
        cand[1] = 6;
        cand[2] = derived / 2;
        cand[3] = derived;
        for (ci = 0; ci < 4; ++ci) {
          const int d = (3 > cand[ci]) ? 3 : cand[ci];
          if (0 < ci && d == ((3 > cand[ci-1]) ? 3 : cand[ci-1])) continue;
          { const double err = internal_libxs_predict_rf_score(model->entries,
              p, model->ninputs, oi, rf->label_offset[oi], d, min_leaf, ntrain,
              rf->regress[oi], rf->nclass[oi]);
            if (0 > best_err || err < best_err) { best_err = err; best = d; }
          }
        }
        rf->depth[oi] = best;
      }
      model->rf = rf;
      rf->calib_fold = (unsigned char*)LIBXS_PREDICT_MALLOC(
        (size_t)p, rf->fold_pool);
      if (NULL != rf->calib_fold) {
        const int ncalib = (p / LIBXS_PREDICT_RF_HOLD) / 2;
        const int nsample = LIBXS_MIN(ncalib, LIBXS_PREDICT_RF_CALIB_SAMPLE);
        const int step = LIBXS_MAX(
          (ncalib + nsample - 1) / LIBXS_MAX(nsample, 1), 1);
        const size_t hold_coprime = libxs_coprime2((size_t)p);
        const size_t hold_inv = (1 < p)
          ? libxs_mod_inverse(hold_coprime, (size_t)p) : 0;
        int ci;
        memset(rf->calib_fold, 0, (size_t)p);
        for (ci = 0; ci < ncalib; ci += step) {
          const int h = ci * 2 + 1;
          const int row = (int)LIBXS_SHUFFLE_INDEX((size_t)h, (size_t)p,
            hold_coprime, LIBXS_PREDICT_RF_SEED);
          rf->calib_fold[row] = (unsigned char)(1
            + internal_libxs_predict_rf_calib_fold(row, p, hold_inv));
        }
      }
      /* last, and after the depth probe rather than before it: the probe splits
       * exactly, and the bins are read by the trees the tasks grow */
      internal_libxs_predict_rf_edges(model);
    }
    else {
      free(rf->trees);
      free(rf->label_offset);
      free(rf->regress);
      free(rf->nclass);
      free(rf->depth);
      free(rf);
    }
  }
}


LIBXS_API_INLINE int internal_libxs_predict_rf_pack_part(
  const internal_libxs_predict_rf_build_node_t src[], int nsrc, int si,
  internal_libxs_predict_rf_node_t dst[], int* next)
{
  int result = EXIT_FAILURE;
  if (0 <= si && si < nsrc && *next < nsrc) {
    const internal_libxs_predict_rf_build_node_t* const sn = src + si;
    const int di = (*next)++;
    internal_libxs_predict_rf_node_t* const dn = dst + di;
    if (UINT16_MAX <= sn->feature) result = EXIT_FAILURE;
    else {
      dn->feature = (0 <= sn->feature) ? (uint16_t)sn->feature : UINT16_MAX;
      dn->value = (0 <= sn->feature) ? sn->threshold : sn->value;
      dn->label = (uint8_t)sn->label;
      if (0 > sn->feature) {
        dn->data.leafp = sn->leafp;
        result = EXIT_SUCCESS;
      }
      else if (0 <= sn->left && sn->left < nsrc
        && 0 <= sn->right && sn->right < nsrc)
      {
        dn->data.right = 0;
        result = internal_libxs_predict_rf_pack_part(
          src, nsrc, sn->left, dst, next);
        if (EXIT_SUCCESS == result) {
          dn->data.right = *next - di;
          result = internal_libxs_predict_rf_pack_part(
            src, nsrc, sn->right, dst, next);
        }
      }
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_pack_tree(
  const internal_libxs_predict_rf_build_node_t src[], int nsrc,
  internal_libxs_predict_rf_node_t** packed)
{
  int result = 0;
  *packed = NULL;
  if (0 < nsrc) {
    internal_libxs_predict_rf_node_t* const dst =
      (internal_libxs_predict_rf_node_t*)malloc(
        (size_t)nsrc * sizeof(internal_libxs_predict_rf_node_t));
    if (NULL != dst) {
      int next = 0;
      if (EXIT_SUCCESS == internal_libxs_predict_rf_pack_part(
        src, nsrc, 0, dst, &next))
      {
        *packed = dst;
        result = next;
      }
      else free(dst);
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_rf_team_bounds(
  int ntasks, int nteams, int team, int* begin, int* end)
{
  internal_libxs_predict_split(ntasks, team, nteams, begin, end);
}


LIBXS_API_INLINE int internal_libxs_predict_rf_team_prepare(
  libxs_predict_t* model, int ntasks)
{
  internal_libxs_predict_rf_t* const rf = model->rf;
  int result = EXIT_SUCCESS;
  if (NULL != rf) {
    const int total_trees = rf->ntrees * rf->noutputs;
    const int nteams = LIBXS_MIN(total_trees, LIBXS_PREDICT_RF_BUILD_TEAMS);
    if (ntasks > nteams && 0 < nteams
      && INTERNAL_LIBXS_PREDICT_RF_TEAM_MAX * nteams >= ntasks)
    {
      int team;
      rf->build_team = (internal_libxs_predict_rf_team_t*)calloc(
        (size_t)nteams, sizeof(internal_libxs_predict_rf_team_t));
      if (NULL != rf->build_team) {
        rf->build_nteams = nteams;
        for (team = 0; team < nteams; ++team) {
          int begin, end;
          internal_libxs_predict_rf_team_bounds(
            ntasks, nteams, team, &begin, &end);
          libxs_barrier_init(&rf->build_team[team].barrier, end - begin);
        }
      }
      else result = EXIT_FAILURE;
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_rf_team_release(
  libxs_predict_t* model)
{
  if (NULL != model->rf) {
    free(model->rf->build_team);
    model->rf->build_team = NULL;
    model->rf->build_nteams = 0;
  }
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tasks_independent(
  libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_rf_t* rf = model->rf;
  int result = EXIT_FAILURE;
  if (NULL != rf) {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = rf->noutputs;
    const int ntrees = rf->ntrees;
    const int total_trees = ntrees * n;
    const int min_leaf = LIBXS_PREDICT_RF_MINLEAF;
    /**
     * Two floors, because they answer different questions. min_leaf is the
     * parent too small to be worth splitting. leaf_floor is what the node
     * budget requires of a child, and only where the budget binds: constraining
     * a child where it does not costs a small corpus accuracy, and leaving it
     * unconstrained where it does costs a large one much more.
     */
    const int leaf_floor = (LIBXS_PREDICT_RF_MAXNODES < p * 2 / min_leaf)
      ? LIBXS_MAX(1, (p * 2 + LIBXS_PREDICT_RF_MAXNODES - 2)
        / (LIBXS_PREDICT_RF_MAXNODES - 1)) : 1;
    const int max_nodes = LIBXS_MIN(p / leaf_floor * 2 + 1,
      LIBXS_PREDICT_RF_MAXNODES);
    const size_t boot_n = (size_t)p * 2 + 1;
    const size_t boot_coprime = libxs_coprime2(boot_n);
    size_t values_count = (NULL != rf->bins)
      ? LIBXS_PREDICT_RF_BINMIN : (size_t)p;
    int begin, end, bootstrap_pool = 0, values_pool = 0, index_pool = 0;
    int* bootstrap = NULL;
    double* values_scratch = NULL;
    int* index_scratch = NULL;
    int output;
    if (NULL != rf->bins) {
      for (output = 0; output < n; ++output) {
        const int ncls = (0 != rf->regress[output]) ? 1
          : ((0 < rf->nclass[output] && 128 >= rf->nclass[output])
            ? rf->nclass[output] : 128);
        const int width = (0 != rf->regress[output]) ? 3 : ncls;
        const size_t per = (size_t)rf->nbins * width;
        int nfused = (int)(LIBXS_PREDICT_RF_HISTMAX
          / (per * sizeof(double)));
        size_t needed;
        if (1 > nfused) nfused = 1;
        nfused = LIBXS_MIN(nfused,
          internal_libxs_predict_rf_nfeatsub(m));
        needed = (size_t)nfused * per;
        if (values_count < needed) values_count = needed;
      }
    }
    internal_libxs_predict_split(total_trees, tid, ntasks, &begin, &end);
    result = EXIT_SUCCESS;
    if (begin < end) {
      bootstrap = (int*)LIBXS_PREDICT_MALLOC(
        (size_t)p * sizeof(int), bootstrap_pool);
      values_scratch = (double*)LIBXS_PREDICT_MALLOC(
        values_count * sizeof(double), values_pool);
      index_scratch = (int*)LIBXS_PREDICT_MALLOC(
        (size_t)p * sizeof(int), index_pool);
      if (NULL == bootstrap || NULL == values_scratch
        || NULL == index_scratch)
      {
        result = EXIT_FAILURE;
      }
    }
    if (NULL != bootstrap && NULL != values_scratch && NULL != index_scratch) {
      int ti;
      for (ti = begin; ti < end && EXIT_SUCCESS == result; ++ti) {
        const int oi = ti / ntrees;
        const int max_depth = rf->depth[oi];
        int nodes_pool = 0;
        internal_libxs_predict_rf_build_node_t* nodes;
        int i, nn;
        if (NULL != rf->trees[ti].nodes) continue;
        nodes = (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
            (size_t)max_nodes
              * sizeof(internal_libxs_predict_rf_build_node_t), nodes_pool);
        for (i = 0; i < p; ++i) {
          bootstrap[i] = internal_libxs_predict_rf_draw((size_t)i, boot_n,
            boot_coprime, (size_t)ti * 7 + 13, p);
          if (NULL != rf->calib_fold) {
            const int fold = ti % LIBXS_PREDICT_RF_CALIB_FOLDS;
            while (1 + fold == rf->calib_fold[bootstrap[i]]) {
              bootstrap[i] = (bootstrap[i] + 1 < p)
                ? (bootstrap[i] + 1) : 0;
            }
          }
        }
        if (NULL != nodes) {
          internal_libxs_predict_rf_grow_t g;
          g.entries = model->entries;
          g.bins = rf->bins; g.bin_edge = rf->bin_edge; g.nbins = rf->nbins;
          g.values_scratch = values_scratch; g.index_scratch = index_scratch;
          g.nfeat = m; g.nfeatsub = internal_libxs_predict_rf_nfeatsub(m);
          g.max_depth = max_depth; g.min_leaf = min_leaf;
          g.leaf_floor = leaf_floor; g.output_idx = oi;
          g.label_off = rf->label_offset[oi]; g.regress = rf->regress[oi];
          g.nclass = rf->nclass[oi];
          /* the decomposition must answer as the single pass does, so it is
           * selectable and off by default until a task actually holds a unit */
          { const char* fenv = getenv("LIBXS_PREDICT_RF_FRONTIER");
            const int fr = (NULL != fenv) ? atoi(fenv) : 0;
            nn = (0 < fr && 63 > max_depth)
              ? internal_libxs_predict_rf_build_tree_parts(&g, bootstrap, p,
                  nodes, max_nodes, LIBXS_MIN(fr, 64))
              : internal_libxs_predict_rf_build_tree(&g, bootstrap, p,
                  nodes, max_nodes);
          }
          rf->trees[ti].nnodes = internal_libxs_predict_rf_pack_tree(
            nodes, nn, &rf->trees[ti].nodes);
          if (0 >= rf->trees[ti].nnodes) result = EXIT_FAILURE;
          LIBXS_PREDICT_FREE(nodes, nodes_pool);
        }
        else result = EXIT_FAILURE;
      }
    }
    LIBXS_PREDICT_FREE(index_scratch, index_pool);
    LIBXS_PREDICT_FREE(values_scratch, values_pool);
    LIBXS_PREDICT_FREE(bootstrap, bootstrap_pool);
  }
  return result;
}


LIBXS_API_INLINE size_t internal_libxs_predict_rf_values_count(
  const internal_libxs_predict_rf_t* rf, int output, int nfeat, int nrows)
{
  size_t result = (NULL != rf->bins)
    ? LIBXS_PREDICT_RF_BINMIN : (size_t)nrows;
  if (NULL != rf->bins) {
    const int ncls = (0 != rf->regress[output]) ? 1
      : ((0 < rf->nclass[output] && 128 >= rf->nclass[output])
        ? rf->nclass[output] : 128);
    const int width = (0 != rf->regress[output]) ? 3 : ncls;
    const size_t per = (size_t)rf->nbins * width;
    int nfused = (int)(LIBXS_PREDICT_RF_HISTMAX
      / (per * sizeof(double)));
    const int nfeatsub = internal_libxs_predict_rf_nfeatsub(nfeat);
    size_t needed;
    if (1 > nfused) nfused = 1;
    nfused = LIBXS_MIN(nfused, nfeatsub);
    needed = (size_t)nfused * per;
    if (result < needed) result = needed;
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree_team(
  libxs_predict_t* model, int tree, int rank, int team_size,
  internal_libxs_predict_rf_team_t* ctx)
{
  internal_libxs_predict_rf_t* const rf = model->rf;
  const int p = model->nentries;
  const int m = model->ninputs;
  const int min_leaf = LIBXS_PREDICT_RF_MINLEAF;
  const int leaf_floor = (LIBXS_PREDICT_RF_MAXNODES < p * 2 / min_leaf)
    ? LIBXS_MAX(1, (p * 2 + LIBXS_PREDICT_RF_MAXNODES - 2)
      / (LIBXS_PREDICT_RF_MAXNODES - 1)) : 1;
  const int max_nodes = LIBXS_MIN(p / leaf_floor * 2 + 1,
    LIBXS_PREDICT_RF_MAXNODES);
  const size_t boot_n = (size_t)p * 2 + 1;
  const size_t boot_coprime = libxs_coprime2(boot_n);
  const int output = tree / rf->ntrees;
  int result;
  internal_libxs_predict_rf_build_node_t* part = NULL;
  double* values_scratch = NULL;
  int* index_scratch = NULL;
  /* this rank's scratch, then the team's, which rank zero owns */
  int part_pool = 0, values_pool = 0, index_pool = 0;
  int boot_pool = 0, tvalues_pool = 0, tindex_pool = 0, nodes_pool = 0;
  if (0 == rank) {
    const size_t values_count = internal_libxs_predict_rf_values_count(
      rf, output, m, p);
    int i;
    ctx->failed = 0;
    ctx->top_n = 0;
    ctx->nfrontier = 0;
    for (i = 0; i < INTERNAL_LIBXS_PREDICT_RF_TEAM_MAX; ++i) {
      ctx->part[i] = NULL;
      ctx->part_n[i] = 0;
    }
    ctx->bootstrap = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), boot_pool);
    ctx->values_scratch = (double*)LIBXS_PREDICT_MALLOC(
      values_count * sizeof(double), tvalues_pool);
    ctx->index_scratch = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), tindex_pool);
    ctx->nodes = (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)(max_nodes + INTERNAL_LIBXS_PREDICT_RF_TEAM_MAX)
        * sizeof(internal_libxs_predict_rf_build_node_t), nodes_pool);
    if (NULL == ctx->bootstrap || NULL == ctx->values_scratch
      || NULL == ctx->index_scratch || NULL == ctx->nodes)
    {
      LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
    }
  }
  libxs_barrier_wait(&ctx->barrier);
  if (0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST)) {
    int begin, end, i;
    internal_libxs_predict_split(p, rank, team_size, &begin, &end);
    for (i = begin; i < end; ++i) {
      int probed = 0;
      ctx->bootstrap[i] = internal_libxs_predict_rf_draw((size_t)i, boot_n,
        boot_coprime, (size_t)tree * 7 + 13, p);
      if (NULL != rf->calib_fold) {
        const int fold = tree % LIBXS_PREDICT_RF_CALIB_FOLDS;
        while (probed < p
          && 1 + fold == rf->calib_fold[ctx->bootstrap[i]])
        {
          ctx->bootstrap[i] = (ctx->bootstrap[i] + 1 < p)
            ? (ctx->bootstrap[i] + 1) : 0;
          ++probed;
        }
        if (p <= probed) {
          LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
        }
      }
    }
  }
  libxs_barrier_wait(&ctx->barrier);
  if (0 == rank
    && 0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST))
  {
    internal_libxs_predict_rf_grow_t g;
    g.entries = model->entries;
    g.bins = rf->bins; g.bin_edge = rf->bin_edge; g.nbins = rf->nbins;
    g.values_scratch = ctx->values_scratch;
    g.index_scratch = ctx->index_scratch;
    g.nfeat = m; g.nfeatsub = internal_libxs_predict_rf_nfeatsub(m);
    g.max_depth = rf->depth[output]; g.min_leaf = min_leaf;
    g.leaf_floor = leaf_floor; g.output_idx = output;
    g.label_off = rf->label_offset[output]; g.regress = rf->regress[output];
    g.nclass = rf->nclass[output];
    ctx->top_n = internal_libxs_predict_rf_build_part(&g, ctx->bootstrap,
      0, p, 0, ctx->nodes, max_nodes, team_size,
      ctx->fr_si, ctx->fr_nc, ctx->fr_depth, ctx->fr_node,
      &ctx->nfrontier);
    if (0 >= ctx->top_n || max_nodes < ctx->top_n
      || INTERNAL_LIBXS_PREDICT_RF_TEAM_MAX < ctx->nfrontier)
    {
      LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
    }
  }
  libxs_barrier_wait(&ctx->barrier);
  if (rank < ctx->nfrontier
    && 0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST))
  {
    const int nc = ctx->fr_nc[rank];
    const int part_cap = LIBXS_MIN(
      LIBXS_MAX(nc / leaf_floor * 2 + 1, 3), max_nodes);
    const size_t values_count = internal_libxs_predict_rf_values_count(
      rf, output, m, nc);
    internal_libxs_predict_rf_grow_t g;
    part = (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)part_cap * sizeof(internal_libxs_predict_rf_build_node_t),
      part_pool);
    values_scratch = (double*)LIBXS_PREDICT_MALLOC(
      values_count * sizeof(double), values_pool);
    index_scratch = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)nc * sizeof(int), index_pool);
    if (NULL != part && NULL != values_scratch && NULL != index_scratch) {
      g.entries = model->entries;
      g.bins = rf->bins; g.bin_edge = rf->bin_edge; g.nbins = rf->nbins;
      g.values_scratch = values_scratch; g.index_scratch = index_scratch;
      g.nfeat = m; g.nfeatsub = internal_libxs_predict_rf_nfeatsub(m);
      g.max_depth = rf->depth[output]; g.min_leaf = min_leaf;
      g.leaf_floor = leaf_floor; g.output_idx = output;
      g.label_off = rf->label_offset[output];
      g.regress = rf->regress[output]; g.nclass = rf->nclass[output];
      ctx->part_n[rank] = internal_libxs_predict_rf_build_part(&g,
        ctx->bootstrap, ctx->fr_si[rank], nc, ctx->fr_depth[rank],
        part, part_cap, 0, NULL, NULL, NULL, NULL, NULL);
      ctx->part[rank] = part;
      if (0 >= ctx->part_n[rank]) {
        LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
      }
    }
    else LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
  }
  libxs_barrier_wait(&ctx->barrier);
  if (0 == rank
    && 0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST))
  {
    int assembled = ctx->top_n;
    int frontier;
    for (frontier = 0; frontier < ctx->nfrontier; ++frontier) {
      const int base = assembled;
      const int np = ctx->part_n[frontier];
      int j;
      if (max_nodes + INTERNAL_LIBXS_PREDICT_RF_TEAM_MAX < assembled + np) {
        LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
        break;
      }
      memcpy(ctx->nodes + base, ctx->part[frontier],
        (size_t)np * sizeof(internal_libxs_predict_rf_build_node_t));
      for (j = 0; j < np; ++j) {
        if (0 <= ctx->nodes[base + j].left) {
          ctx->nodes[base + j].left += base;
        }
        if (0 <= ctx->nodes[base + j].right) {
          ctx->nodes[base + j].right += base;
        }
      }
      ctx->nodes[ctx->fr_node[frontier]] = ctx->nodes[base];
      assembled += np;
    }
    if (0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST)) {
      rf->trees[tree].nnodes = internal_libxs_predict_rf_pack_tree(
        ctx->nodes, assembled, &rf->trees[tree].nodes);
      if (0 >= rf->trees[tree].nnodes) {
        LIBXS_ATOMIC_STORE(&ctx->failed, 1, LIBXS_ATOMIC_SEQ_CST);
      }
    }
  }
  libxs_barrier_wait(&ctx->barrier);
  result = (0 == LIBXS_ATOMIC_LOAD(&ctx->failed, LIBXS_ATOMIC_SEQ_CST))
    ? EXIT_SUCCESS : EXIT_FAILURE;
  LIBXS_PREDICT_FREE(index_scratch, index_pool);
  LIBXS_PREDICT_FREE(values_scratch, values_pool);
  LIBXS_PREDICT_FREE(part, part_pool);
  if (0 == rank) {
    LIBXS_PREDICT_FREE(ctx->nodes, nodes_pool);
    LIBXS_PREDICT_FREE(ctx->index_scratch, tindex_pool);
    LIBXS_PREDICT_FREE(ctx->values_scratch, tvalues_pool);
    LIBXS_PREDICT_FREE(ctx->bootstrap, boot_pool);
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tasks_team(
  libxs_predict_t* model, int tid, int ntasks)
{
  internal_libxs_predict_rf_t* const rf = model->rf;
  const int total_trees = rf->ntrees * rf->noutputs;
  int team = 0, task_begin = 0, task_end = 0;
  int rank, team_size, tree, result = EXIT_SUCCESS;
  while (team < rf->build_nteams) {
    internal_libxs_predict_rf_team_bounds(ntasks, rf->build_nteams,
      team, &task_begin, &task_end);
    if (task_begin <= tid && tid < task_end) break;
    ++team;
  }
  if (rf->build_nteams <= team) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    rank = tid - task_begin;
    team_size = task_end - task_begin;
    for (tree = team; tree < total_trees && EXIT_SUCCESS == result;
      tree += rf->build_nteams)
    {
      result = internal_libxs_predict_rf_build_tree_team(model, tree,
        rank, team_size, rf->build_team + team);
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tasks(
  libxs_predict_t* model, int tid, int ntasks)
{
  int result;
  if (NULL != model->rf && NULL != model->rf->build_team
    && 0 < model->rf->build_nteams)
  {
    result = internal_libxs_predict_rf_build_tasks_team(model, tid, ntasks);
  }
  else {
    result = internal_libxs_predict_rf_build_tasks_independent(
      model, tid, ntasks);
  }
  return result;
}


/**
 * Index of the leaf the inputs descend to, or negative if the tree is empty
 * or a relative jump leaves the packed preorder array.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_leafof(
  const internal_libxs_predict_rf_tree_t* tree, const double* inputs)
{
  int result = 0;
  if (NULL == tree->nodes || 0 == tree->nnodes) {
    result = -1;
  }
  else {
    while (0 <= result && result < tree->nnodes
      && UINT16_MAX != tree->nodes[result].feature)
    {
      const internal_libxs_predict_rf_node_t* nd = &tree->nodes[result];
      result += (inputs[nd->feature] <= nd->value) ? 1 : nd->data.right;
    }
    if (result >= tree->nnodes) result = -1;
  }
  return result;
}


/**
 * Held-back score of the read-out as it currently stands: how many rows it gets
 * wrong, and how far its scores sit from the truth. The count is what the
 * read-out promises, so it decides; the distance breaks its ties, so a stage
 * that sharpens the scores without yet flipping any row still counts as
 * progress. A real-valued output has no rows to get wrong and is judged on the
 * distance alone, which is its absolute error.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_hold_score(
  const internal_libxs_predict_entry_t* entries, int p, int output_idx,
  const unsigned char* hold, const double* pred, int nclass, int label_off,
  int regress, int* miss, double* dist)
{
  int i, n = 0;
  *miss = 0;
  *dist = 0;
  for (i = 0; i < p; ++i) {
    if (1 == hold[i]) {
      if (0 != regress) {
        *dist += LIBXS_FABS(entries[i].outputs[output_idx] - pred[i]);
      }
      else {
        const int lab = (LIBXS_ROUNDX(int,
          entries[i].outputs[output_idx]) + label_off) & 127;
        const double* row = pred + (size_t)i * nclass;
        int best = 0, c;
        for (c = 1; c < nclass; ++c) {
          if (row[c] > row[best]) best = c;
        }
        if (best != lab) ++*miss;
        for (c = 0; c < nclass; ++c) {
          const double d = ((c == lab) ? 1.0 : 0.0) - row[c];
          *dist += d * d;
        }
      }
      ++n;
    }
  }
  if (0 < n) *dist /= n;
}


/**
 * Measure what native RF confidence is worth as empirical correctness.
 *
 * The score is read out-of-bag: a row is voted on only by trees whose bootstrap
 * left it out, exactly as the boosting stages are judged. That makes the score
 * a few trees' worth noisier than the one eval computes over the whole forest,
 * so the bins are wide enough to absorb it rather than model the noise.
 *
 * The curve is forced non-decreasing. It is a statement about evidence - more
 * trees agreeing cannot mean a worse answer - and a bin that dips below its
 * predecessor is reading a sampling accident, which is what pooling it away
 * says. Bins nothing landed in inherit the value below them for the same
 * reason: they carry no evidence of their own.
 *
 * The bin a score falls in, as one rule: the curve is filled, read and refitted
 * in separate places and they have to agree on where a value belongs.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_calib_bin(double score, int nbin)
{
  int result = (int)(score * nbin);
  if (result >= nbin) result = nbin - 1;
  if (0 > result) result = 0;
  return result;
}


/**
 * Pool adjacent violators: where a bin scores below the one under it, the two
 * are merged and the merged block re-checked against what is under IT, so a dip
 * is averaged away against the evidence that contradicts it. Clamping the dip up
 * to its predecessor instead looks like the same thing and is not - one thinly
 * populated low bin that happens to score well then propagates its value through
 * every bin above, which flattens the curve to a single number and reports one
 * confidence for every query.
 *
 * The result is monotone, which is what makes the mapping safe to apply after
 * the fact: it reorders no query, so a coverage stays the coverage it was.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_isotonic(
  const double hit[], const double cnt[], int nbin, double curve[])
{
  double wsum[LIBXS_PREDICT_RF_CALIB], vsum[LIBXS_PREDICT_RF_CALIB];
  int at[LIBXS_PREDICT_RF_CALIB], nblock = 0, b, k;
  LIBXS_ASSERT(0 < nbin && LIBXS_PREDICT_RF_CALIB >= nbin);
  for (b = 0; b < nbin; ++b) {
    if (0 >= cnt[b]) continue; /* no evidence of its own */
    wsum[nblock] = cnt[b];
    vsum[nblock] = hit[b];
    at[nblock] = b;
    ++nblock;
    while (1 < nblock && vsum[nblock - 1] / wsum[nblock - 1]
      < vsum[nblock - 2] / wsum[nblock - 2])
    {
      wsum[nblock - 2] += wsum[nblock - 1];
      vsum[nblock - 2] += vsum[nblock - 1];
      --nblock;
    }
  }
  { /* every bin takes the block that covers it, and a bin below the first block
     * or above the last takes the nearest one */
    double prev = (0 < nblock) ? (vsum[0] / wsum[0]) : 0.0;
    k = 0;
    for (b = 0; b < nbin; ++b) {
      while (k + 1 < nblock && at[k + 1] <= b) ++k;
      if (0 < nblock && at[k] <= b) prev = vsum[k] / wsum[k];
      curve[b] = prev;
    }
  }
}


/**
 * Fits the additive read-out over the partitions the forest already grew.
 *
 * The two read-outs combine rather than compete: eval answers with the bagged
 * read-out plus the sum of the corrections, so the stages correct a
 * variance-reduced base instead of rebuilding it. Nothing here grows a tree,
 * and eval pays one array lookup per descent it was making anyway.
 *
 * Real-valued and folded outputs are the same procedure at two widths. A
 * real-valued output carries one score, the leaf mean, and its correction is
 * the mean residual. A folded one carries a score per class, the share of the
 * trees voting for it, and its correction is the mean residual of each class
 * indicator. Setting nclass to one for the former makes the second case the
 * general one and the first its degenerate width, so there is a single fit, a
 * single stopping rule, and a single stored correction.
 *
 * One choice carries the honesty of the whole fit: what the residual is taken
 * against. It has to be the out-of-bag forest read-out, averaging each row
 * over only the trees whose bootstrap left that row out. The tempting
 * alternative is the full forest read-out, on the grounds that it is exactly
 * what eval starts from and the corrections ought to be fitted against the base
 * they will be added to. That is wrong, and measurably so: on a training row
 * the forest is nearly unbiased because most of its trees memorized that row,
 * so the leaf means come out as noise rather than as bias, and summing a
 * hundred stages of noise is a random walk that degrades the read-out in
 * proportion to the learning rate. The out-of-bag read-out is a few trees'
 * worth noisier than the one eval uses but carries the same bias, and bias is
 * the only thing the stages can correct.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_boost(libxs_predict_t* model)
{
  internal_libxs_predict_rf_t* rf = model->rf;
  if (NULL != rf && 0 < model->nentries && NULL != rf->regress
    && NULL != rf->nclass)
  {
    const internal_libxs_predict_entry_t* entries = model->entries;
    const int p = model->nentries;
    const int ntrees = rf->ntrees;
    const char* renv = getenv("LIBXS_PREDICT_RF_RATE");
    const double configured_rate = (NULL != renv)
      ? atof(renv) : LIBXS_PREDICT_RF_RATE;
    int maxn = 0, ncmax = 1, enabled = (NULL != renv) ? 1 : 0, ti;
    for (ti = 0; ti < ntrees * rf->noutputs; ++ti) {
      if (maxn < rf->trees[ti].nnodes) maxn = rf->trees[ti].nnodes;
    }
    for (ti = 0; ti < rf->noutputs; ++ti) {
      if (ncmax < rf->nclass[ti]) ncmax = rf->nclass[ti];
      if (0 != rf->regress[ti]) enabled = 1;
    }
    if (0 < maxn && 0 < configured_rate && 0 != enabled) {
      int pred_pool = 0, sum_pool = 0, cnt_pool = 0;
      int oob_pool = 0, cover_pool = 0, hold_pool = 0;
      double* pred = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)p * ncmax * sizeof(double), pred_pool);
      double* sum = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)maxn * ncmax * sizeof(double), sum_pool);
      int* cnt = (int*)LIBXS_PREDICT_MALLOC(
        (size_t)maxn * ncmax * sizeof(int), cnt_pool);
      unsigned char* oob = (unsigned char*)LIBXS_PREDICT_MALLOC(
        (size_t)p, oob_pool);
      int* cover = (int*)LIBXS_PREDICT_MALLOC(
        (size_t)p * sizeof(int), cover_pool);
      unsigned char* hold = (unsigned char*)LIBXS_PREDICT_MALLOC(
        (size_t)p, hold_pool);
      if (NULL != pred && NULL != sum && NULL != cnt && NULL != oob
        && NULL != cover && NULL != hold)
      {
        const size_t boot_n = (size_t)p * 2 + 1;
        const size_t boot_coprime = libxs_coprime2(boot_n);
        const size_t hold_coprime = libxs_coprime2((size_t)p);
        int oi, h;
        /** Spread over the corpus by the shuffle rather than taken as a block.
         *  Even positions select corrections; odd positions calibrate later.
         *  Neither role fits a correction, while both still train RF trees. */
        memset(hold, 0, (size_t)p);
        for (h = 0; h < p / LIBXS_PREDICT_RF_HOLD; ++h) {
          hold[(int)LIBXS_SHUFFLE_INDEX((size_t)h, (size_t)p, hold_coprime,
            LIBXS_PREDICT_RF_SEED)] = (unsigned char)(1 + (h & 1));
        }
        for (oi = 0; oi < rf->noutputs; ++oi) {
          const int tbase = oi * ntrees;
          const int nc = rf->nclass[oi];
          const int reg = rf->regress[oi];
          const int loff = rf->label_offset[oi];
          const double rate = (NULL != renv || 0 != reg)
            ? configured_rate : 0;
          int stale = 0, miss0 = 0, miss1 = 0, t, i, k, c;
          double dist0 = 0, dist1 = 0;
          if (1 > nc || 0 >= rate) continue;
          memset(pred, 0, (size_t)p * nc * sizeof(double));
          for (i = 0; i < p; ++i) cover[i] = 0;
          for (t = 0; t < ntrees; ++t) {
            const internal_libxs_predict_rf_tree_t* tr = &rf->trees[tbase + t];
            if (NULL == tr->nodes || 0 == tr->nnodes) continue;
            memset(oob, 1, (size_t)p);
            for (i = 0; i < p; ++i) {
              int row = internal_libxs_predict_rf_draw((size_t)i, boot_n,
                boot_coprime, (size_t)(tbase + t) * 7 + 13, p);
              if (NULL != rf->calib_fold) {
                const int fold = (tbase + t) % LIBXS_PREDICT_RF_CALIB_FOLDS;
                while (1 + fold == rf->calib_fold[row]) {
                  row = (row + 1 < p) ? (row + 1) : 0;
                }
              }
              oob[row] = 0;
            }
            for (i = 0; i < p; ++i) {
              if (0 != oob[i]) {
                const int li = internal_libxs_predict_rf_leafof(tr,
                  entries[i].inputs);
                if (0 <= li) {
                  if (0 != reg) pred[i] += tr->nodes[li].value;
                  else {
                    const int lc = tr->nodes[li].label & 127;
                    if (lc < nc) pred[(size_t)i * nc + lc] += 1.0;
                  }
                  ++cover[i];
                }
              }
            }
          }
          for (i = 0; i < p; ++i) {
            if (0 < cover[i]) {
              for (c = 0; c < nc; ++c) pred[(size_t)i * nc + c] /= cover[i];
            }
          }
          for (t = 0; t < ntrees; ++t) {
            internal_libxs_predict_rf_tree_t* tr = &rf->trees[tbase + t];
            if (NULL == tr->nodes || 0 == tr->nnodes) continue;
            /** The bootstrap is regenerated rather than stored: it is a pure
             *  function of the tree index, and holding p indices per tree to
             *  avoid recomputing them would cost more than the forest. */
            memset(oob, 1, (size_t)p);
            for (i = 0; i < p; ++i) {
              int row = internal_libxs_predict_rf_draw((size_t)i, boot_n,
                boot_coprime, (size_t)(tbase + t) * 7 + 13, p);
              if (NULL != rf->calib_fold) {
                const int fold = (tbase + t) % LIBXS_PREDICT_RF_CALIB_FOLDS;
                while (1 + fold == rf->calib_fold[row]) {
                  row = (row + 1 < p) ? (row + 1) : 0;
                }
              }
              oob[row] = 0;
            }
            /**
             * Judged on the held-back rows alone. Judging on this tree's
             * out-of-bag rows instead looks natural and is worthless: those
             * rows were fitted by every earlier stage that had them out of
             * bag, roughly a third of them, so the score falls whether or not
             * the stages generalize and the rule never fires.
             */
            internal_libxs_predict_rf_hold_score(entries, p, oi, hold, pred,
              nc, loff, reg, &miss0, &dist0);
            memset(sum, 0, (size_t)tr->nnodes * nc * sizeof(double));
            memset(cnt, 0, (size_t)tr->nnodes * nc * sizeof(int));
            for (i = 0; i < p; ++i) {
              if (0 != oob[i] && 0 == hold[i]) {
                const int li = internal_libxs_predict_rf_leafof(tr,
                  entries[i].inputs);
                if (0 <= li) {
                  if (0 != reg) {
                    sum[li] += entries[i].outputs[oi] - pred[i];
                    ++cnt[li];
                  }
                  else {
                    const int lab = (LIBXS_ROUNDX(int,
                      entries[i].outputs[oi]) + loff) & 127;
                    for (c = 0; c < nc; ++c) {
                      sum[(size_t)li * nc + c] += ((c == lab) ? 1.0 : 0.0)
                        - pred[(size_t)i * nc + c];
                      ++cnt[(size_t)li * nc + c];
                    }
                  }
                }
              }
            }
            tr->incr = (double*)calloc((size_t)tr->nnodes * nc, sizeof(double));
            if (NULL == tr->incr) break;
            for (k = 0; k < tr->nnodes * nc; ++k) {
              if (0 < cnt[k]) tr->incr[k] = rate * sum[k] / cnt[k];
            }
            /** Carried into the score for every row, not only the ones that
             *  fitted it: the next stage sees what this one actually left. */
            for (i = 0; i < p; ++i) {
              const int li = internal_libxs_predict_rf_leafof(tr,
                entries[i].inputs);
              if (0 <= li) {
                for (c = 0; c < nc; ++c) {
                  pred[(size_t)i * nc + c] += tr->incr[(size_t)li * nc + c];
                }
              }
            }
            internal_libxs_predict_rf_hold_score(entries, p, oi, hold, pred,
              nc, loff, reg, &miss1, &dist1);
            /**
             * A stage is fitted before it is judged, and discarded unless it
             * improved the held-back rows. Deciding in advance cannot work -
             * whether a correction generalizes is not knowable until it has
             * been made - and letting an unhelpful stage stand while waiting
             * for a later one to redeem it is how the read-out came to degrade
             * in proportion to the learning rate. Discarding instead makes the
             * combined read-out no worse than the bagged one it corrects.
             */
            if (miss1 < miss0 || (miss1 == miss0 && dist1 < dist0)) {
              stale = 0;
            }
            else {
              for (i = 0; i < p; ++i) {
                const int li = internal_libxs_predict_rf_leafof(tr,
                  entries[i].inputs);
                if (0 <= li) {
                  for (c = 0; c < nc; ++c) {
                    pred[(size_t)i * nc + c] -= tr->incr[(size_t)li * nc + c];
                  }
                }
              }
              free(tr->incr);
              tr->incr = NULL;
              if (LIBXS_PREDICT_RF_PATIENCE <= ++stale) break;
            }
          }
        }
      }
      LIBXS_PREDICT_FREE(hold, hold_pool);
      LIBXS_PREDICT_FREE(cover, cover_pool);
      LIBXS_PREDICT_FREE(oob, oob_pool);
      LIBXS_PREDICT_FREE(cnt, cnt_pool);
      LIBXS_PREDICT_FREE(sum, sum_pool);
      LIBXS_PREDICT_FREE(pred, pred_pool);
    }
  }
}


/**
 * Forest read-out for one output: the mean of the leaves reached for a
 * real-valued output, the majority of the labels reached for a folded one.
 * Both descend the same trees, so the read-out is a choice of what to
 * accumulate rather than a second traversal, and the boosted correction rides
 * along on the same descent again.
 */
LIBXS_API_INLINE double internal_libxs_predict_rf_eval_output_impl(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence, double* variance,
  int calib_fold, int* evidence)
{
  const int regress = (NULL != rf->regress) ? rf->regress[output_idx] : 0;
  const int nc = (NULL != rf->nclass) ? rf->nclass[output_idx] : 1;
  const int base = output_idx * rf->ntrees;
  int votes[128];
  double bscore[128], lscore[128];
  int best_label = 0, best_count = 0, nvalid = 0, t, k;
  double sum = 0, sqr = 0, boost = 0, result, scale = 1.0;
  if (0 == regress) {
    memset(votes, 0, sizeof(votes));
    memset(bscore, 0, sizeof(bscore));
    memset(lscore, 0, sizeof(lscore));
  }
  for (t = 0; t < rf->ntrees; ++t) {
    const internal_libxs_predict_rf_tree_t* tree = &rf->trees[base + t];
    int ni;
    if (0 <= calib_fold
      && calib_fold != ((base + t) % LIBXS_PREDICT_RF_CALIB_FOLDS))
    {
      continue;
    }
    ni = internal_libxs_predict_rf_leafof(tree, inputs);
    if (0 <= ni) {
      if (0 != regress) {
        const double v = tree->nodes[ni].value;
        sum += v;
        sqr += v * v;
        ++nvalid;
        if (NULL != tree->incr) boost += tree->incr[ni];
      }
      else {
        const int lab = tree->nodes[ni].label & 127;
        ++votes[lab];
        lscore[lab] += tree->nodes[ni].data.leafp;
        ++nvalid;
        if (NULL != tree->incr) {
          for (k = 0; k < nc && k < 128; ++k) {
            bscore[k] += tree->incr[(size_t)ni * nc + k];
          }
        }
      }
    }
  }
  if (0 <= calib_fold && 0 < nvalid) scale = (double)rf->ntrees / nvalid;
  if (0 != regress) {
    const double mean = (0 < nvalid) ? (sum / nvalid) : 0.0;
    result = mean + scale * boost;
    if (NULL != variance) {
      /**
       * The spread of the trees about their own mean, not about the boosted
       * result: the correction is one deterministic quantity added to every
       * tree alike, so it shifts the read-out without telling us anything
       * about how much the trees disagreed. Clamped because the closed form
       * can fall below zero by rounding when they agree closely.
       */
      const double v = (0 < nvalid) ? (sqr / nvalid - mean * mean) : 0.0;
      *variance = (0 < v) ? v : 0.0;
    }
    /**
     * The spread across trees is reported as variance and not folded into the
     * confidence: a dispersion-derived confidence was measured to be worse than
     * a pinned one, and callers gate on the variance instead.
     */
    if (NULL != confidence) *confidence = 1.0;
  }
  else {
    /**
     * The class score is the share of the trees that voted for it plus the
     * corrections the stages fitted. With no stages the corrections are zero
     * and the shares rank exactly as the raw counts do, so the answer is the
     * majority vote unchanged.
     */
    double best_score = 0;
    for (k = 0; k < 128; ++k) {
      const double s = ((0 < nvalid) ? ((double)votes[k] / nvalid) : 0.0)
        + scale * ((k < nc) ? bscore[k] : 0.0);
      if (0 == k || s > best_score) {
        best_score = s;
        best_label = k;
      }
    }
    best_count = votes[best_label];
    if (NULL != confidence) {
      /**
       * What the trees that voted for the answer were worth, rather than how
       * many of them there were. The count alone cannot separate the queries
       * every tree agrees on - a third of a corpus arrives there and is handed
       * one number - and those are exactly the queries a high gate keeps. The
       * per-leaf estimate does separate them, because it reads how many rows
       * stood behind each of those agreeing leaves.
       *
       * Bounded above by the share, since no leaf is worth more than one vote,
       * and it falls back TO the share where the estimates are absent (a model
       * saved before they were recorded), so an older model keeps answering as
       * it did.
       *
       * A RANKING still, on its own scale: libxs_predict_recalibrate is what makes
       * it a rate. The decision above is untouched - it is the same majority
       * vote - so this changes what is reported and not what is answered.
       */
      const double share = (0 < nvalid)
        ? (double)best_count / nvalid : 0.0;
      const double soft = (0 < nvalid)
        ? (lscore[best_label] / nvalid) : 0.0;
      *confidence = (0 < soft) ? soft : share;
    }
    if (NULL != variance) *variance = 0;
    result = (double)(best_label - rf->label_offset[output_idx]);
  }
  if (NULL != evidence) *evidence = nvalid;
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_rf_eval_output(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence, double* variance)
{
  return internal_libxs_predict_rf_eval_output_impl(rf, output_idx, inputs,
    confidence, variance, -1, NULL);
}


LIBXS_API_INLINE int internal_libxs_predict_rf_batchable(
  const internal_libxs_predict_rf_t* rf)
{
  int result = (NULL != rf && NULL != rf->regress) ? 1 : 0;
  int i;
  for (i = 0; 0 != result && i < rf->noutputs; ++i) {
    if (0 != rf->regress[i]) result = 0;
  }
  for (i = 0; 0 != result && i < rf->ntrees * rf->noutputs; ++i) {
    if (NULL == rf->trees[i].nodes || 0 >= rf->trees[i].nnodes
      || NULL != rf->trees[i].incr)
    {
      result = 0;
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_rf_eval_batch_folded(
  const internal_libxs_predict_rf_t* rf, const double inputs[], int ninputs,
  double outputs[], int noutputs, int begin, int end)
{
  int first;
  for (first = begin; first < end; first += LIBXS_PREDICT_RF_PACKET) {
    const int lanes = LIBXS_MIN(LIBXS_PREDICT_RF_PACKET, end - first);
    int output;
    for (output = 0; output < noutputs; ++output) {
      int votes[LIBXS_PREDICT_RF_PACKET][128];
      const int base = output * rf->ntrees;
      int tree, lane;
      memset(votes, 0, sizeof(votes));
      for (tree = 0; tree < rf->ntrees; ++tree) {
        const internal_libxs_predict_rf_tree_t* const tr =
          rf->trees + base + tree;
        int node[LIBXS_PREDICT_RF_PACKET];
        int active = lanes;
        for (lane = 0; lane < lanes; ++lane) node[lane] = 0;
        while (0 < active) {
          active = 0;
          LIBXS_PRAGMA_SIMD_REDUCTION(+:active)
          for (lane = 0; lane < lanes; ++lane) {
            const int ni = node[lane];
            if (0 <= ni && ni < tr->nnodes
              && UINT16_MAX != tr->nodes[ni].feature)
            {
              const internal_libxs_predict_rf_node_t* const nd =
                tr->nodes + ni;
              const double* const input = inputs
                + (size_t)(first + lane) * ninputs;
              node[lane] = ni
                + ((input[nd->feature] <= nd->value) ? 1 : nd->data.right);
              ++active;
            }
          }
        }
        for (lane = 0; lane < lanes; ++lane) {
          const int ni = node[lane];
          if (0 <= ni && ni < tr->nnodes) {
            ++votes[lane][tr->nodes[ni].label & 127];
          }
        }
      }
      for (lane = 0; lane < lanes; ++lane) {
        int best = 0, count = 0, label;
        for (label = 0; label < 128; ++label) {
          if (count < votes[lane][label]) {
            count = votes[lane][label];
            best = label;
          }
        }
        outputs[(size_t)(first + lane) * noutputs + output] =
          (double)(best - rf->label_offset[output]);
      }
    }
  }
}


/**
 * Fit P(final hybrid prediction is correct | native RF confidence) from rows
 * excluded from correction fitting and selection. Every such row still trains
 * ordinary RF trees; its score uses only trees whose bootstrap omitted it and
 * rescales their additive sum to the full forest that deployment evaluates.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_calibrate_oob(
  libxs_predict_t* model)
{
  internal_libxs_predict_rf_t* rf = model->rf;
  const int p = model->nentries;
  const int nhold = p / LIBXS_PREDICT_RF_HOLD;
  const int ncalib = nhold / 2;
  const int nsample = LIBXS_MIN(ncalib, LIBXS_PREDICT_RF_CALIB_SAMPLE);
  if (NULL != rf && NULL != rf->calib_fold
    && 0 < nsample && 0 < rf->ntrees)
  {
    const int nbin = LIBXS_PREDICT_RF_CALIB;
    const int n = rf->noutputs;
    const size_t hold_coprime = libxs_coprime2((size_t)p);
    const size_t hold_inv = (1 < p)
      ? libxs_mod_inverse(hold_coprime, (size_t)p) : 0;
    const int step = LIBXS_MAX((ncalib + nsample - 1) / nsample, 1);
    int hit_pool = 0, cnt_pool = 0;
    double* hit = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)n * nbin * sizeof(double), hit_pool);
    double* cnt = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)n * nbin * sizeof(double), cnt_pool);
    /* kept: it becomes the forest's curve */
    double* curve = (double*)malloc((size_t)n * nbin * sizeof(double));
    if (NULL != hit && NULL != cnt && NULL != curve) {
      int ci, oi, any = 0;
      memset(hit, 0, (size_t)n * nbin * sizeof(double));
      memset(cnt, 0, (size_t)n * nbin * sizeof(double));
      for (ci = 0; ci < ncalib; ci += step) {
        const int h = ci * 2 + 1;
        const int row = (int)LIBXS_SHUFFLE_INDEX((size_t)h, (size_t)p,
          hold_coprime, LIBXS_PREDICT_RF_SEED);
        const int fold = internal_libxs_predict_rf_calib_fold(
          row, p, hold_inv);
        for (oi = 0; oi < n; ++oi) {
          if (0 == rf->regress[oi] && 1 < rf->nclass[oi]
            && 128 >= rf->nclass[oi])
          {
            double confidence = 0;
            int evidence = 0;
            const double predicted = internal_libxs_predict_rf_eval_output_impl(
              rf, oi, model->entries[row].inputs, &confidence, NULL,
              fold, &evidence);
            if (evidence >= LIBXS_MAX(rf->ntrees / 10, 3)) {
              const int b = internal_libxs_predict_rf_calib_bin(
                confidence, nbin);
              ++cnt[(size_t)oi * nbin + b];
              if (LIBXS_ROUNDX(int, predicted) == LIBXS_ROUNDX(int,
                model->entries[row].outputs[oi]))
              {
                ++hit[(size_t)oi * nbin + b];
              }
            }
          }
        }
      }
      for (oi = 0; oi < n; ++oi) {
        double total = 0;
        int b;
        for (b = 0; b < nbin; ++b) total += cnt[(size_t)oi * nbin + b];
        if (5 * nbin <= total) {
          internal_libxs_predict_rf_isotonic(hit + (size_t)oi * nbin,
            cnt + (size_t)oi * nbin, nbin, curve + (size_t)oi * nbin);
          any = 1;
        }
        else for (b = 0; b < nbin; ++b) {
          curve[(size_t)oi * nbin + b] = -1.0;
        }
      }
      if (0 != any) {
        free(rf->calib);
        rf->calib = curve;
        curve = NULL;
      }
    }
    free(curve);
    LIBXS_PREDICT_FREE(cnt, cnt_pool);
    LIBXS_PREDICT_FREE(hit, hit_pool);
  }
  if (NULL != rf) {
    LIBXS_PREDICT_FREE(rf->calib_fold, rf->fold_pool);
    rf->calib_fold = NULL;
    rf->fold_pool = 0;
  }
}
