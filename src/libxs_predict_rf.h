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
 * It cost two points once. Binning EVERY node was measured before and withdrawn:
 * it was 3.6x on HIGGS at unchanged accuracy, and 5.6x on the crystal corpus at
 * 82.5% -> 80.3%. The cause was found then and is what the node crossover above
 * answers - the bins are global, so a DEEP node spanning three of them has three
 * candidate thresholds where the sorted search over its own rows had twenty-nine.
 * Edge placement was not the cause (quantile against equal-width moved 0.2, which
 * is noise) and neither was bin count (32 against 256 moved 0.7).
 *
 * So this is the size regime the histogram is FOR rather than a measured crossover:
 * a build of a million rows was the complaint. Every corpus tuned before the bins
 * existed is under it and is split exactly at every node, which is what keeps them
 * answering as they did.
 */
#if !defined(LIBXS_PREDICT_RF_BINROWS)
#  define LIBXS_PREDICT_RF_BINROWS 262144
#endif
/** Rows the bin edges are placed from. Hundreds per bin is ample for a quantile,
 *  and a bound rather than a share keeps the sort off the corpus size. */
#if !defined(LIBXS_PREDICT_RF_SKETCH)
#  define LIBXS_PREDICT_RF_SKETCH 65536
#endif
/** Bytes of histogram one split may hold. It buys the width of the accumulating
 *  pass, so it wants to be a cache the pass stays inside of. */
#if !defined(LIBXS_PREDICT_RF_HISTMAX)
#  define LIBXS_PREDICT_RF_HISTMAX 65536
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
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass)
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
     which a value/index pair cannot: its comparator is not recognized */
  double* keys = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nsub * sizeof(double), keys_pool);
  int* ord = (int*)LIBXS_PREDICT_MALLOC((size_t)nsub * sizeof(int), ord_pool);
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
  LIBXS_PREDICT_FREE(ord, ord_pool);
  LIBXS_PREDICT_FREE(keys, keys_pool);
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
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass)
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
  double* acc;
  int* fsel;
  int base, i, j, k, b, result;
  if (1 > nfused) nfused = 1;
  if (nfeatsub < nfused) nfused = nfeatsub;
  acc = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nfused * per * sizeof(double), acc_pool);
  fsel = (int*)LIBXS_PREDICT_MALLOC((size_t)nfeatsub * sizeof(int), fsel_pool);
  node->feature = -1;
  node->label = -1;
  if (NULL != acc && NULL != fsel) {
    /* the same draw the sorted search makes, so the two paths differ in the
       resolution of the candidates and in nothing else */
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
  LIBXS_PREDICT_FREE(fsel, fsel_pool);
  LIBXS_PREDICT_FREE(acc, acc_pool);
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
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off, int regress, int min_leaf, int nclass)
{
  int result;
  if (NULL != bins && NULL != bin_edge && 0 < nbins
    && LIBXS_PREDICT_RF_BINMIN <= nsub)
  {
    result = internal_libxs_predict_rf_split_hist(entries, bins, bin_edge,
      nbins, subset, nsub, nfeat, nfeatsub, node, seed, output_idx, label_off,
      regress, min_leaf, nclass);
  }
  else {
    result = internal_libxs_predict_rf_split_sort(entries, subset, nsub, nfeat,
      nfeatsub, node, seed, output_idx, label_off, regress, min_leaf, nclass);
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree(
  const internal_libxs_predict_entry_t* entries,
  const unsigned char* bins, const double* bin_edge, int nbins,
  int* subset, int nsub, int nfeat, int max_depth, int min_leaf,
  internal_libxs_predict_rf_node_t* nodes, int max_nodes,
  int output_idx, int label_off, int regress, int nclass, int leaf_floor)
{
  int stack_subset[64], stack_count[64], stack_depth[64], stack_node[64];
  int sp = 0, nnodes = 0;
  int nfeatsub = (int)(sqrt((double)nfeat) + 0.5);
  if (nfeatsub < 1) nfeatsub = 1;
  stack_subset[0] = 0;
  stack_count[0] = nsub;
  stack_depth[0] = 0;
  stack_node[0] = nnodes++;
  nodes[0].feature = -1;
  nodes[0].left = -1;
  nodes[0].right = -1;
  nodes[0].label = 0;
  nodes[0].value = 0;
  sp = 1;
  while (sp > 0 && nnodes < max_nodes - 2) {
    const int si = stack_subset[--sp];
    const int nc = stack_count[sp];
    const int depth = stack_depth[sp];
    const int ni = stack_node[sp];
    int best_label = 0, best_count = 0, pure = 0, k;
    double mean = 0, dev = 0;
    internal_libxs_predict_rf_node_t split;
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
    if (depth >= max_depth || nc <= min_leaf || 0 != pure
      || 0 == internal_libxs_predict_rf_split(entries, bins, bin_edge, nbins,
        subset + si, nc, nfeat, nfeatsub, &split, (size_t)ni, output_idx,
        label_off, regress, leaf_floor, nclass))
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
        int* part = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), part_pool);
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
          LIBXS_PREDICT_FREE(part, part_pool);
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
  return nnodes;
}


/**
 * Nodes one tree may hold. It raises the leaf floor rather than truncating
 * growth: growth is depth-first, so hitting the ceiling leaves the first
 * subtree grown and every later one a stub, worth 9 points on a million rows.
 */
/* smallest parent worth splitting; a finer one buys capacity, and costs it */
#if !defined(LIBXS_PREDICT_RF_MINLEAF)
#  define LIBXS_PREDICT_RF_MINLEAF 3
#endif
/**
 * Nodes a tree may hold. This is a memory bound and nothing else: a task builds
 * one tree at a time into a scratch of this many nodes at 40 bytes each, so the
 * peak is ntasks * MAXNODES * 40 - 21 MB per task here, which is 8 GB across 384
 * of them and 168 MB across eight. It is fixed rather than derived from the
 * machine so that a corpus yields the same forest whatever the thread count.
 *
 * It used to be 32767 because a saved node index was a signed 16-bit number, and
 * it kept that value after the index was widened. That mattered more than a
 * stale constant usually does, because the budget is what sets leaf_floor
 * (2*nentries/MAXNODES): a fixed budget forces coarser trees as the corpus
 * grows, which is why accuracy stopped improving with data. On HIGGS at 4.4M
 * rows the leaf floor was 269 and the forest underfitted; at this value it is 17
 * and accuracy rises 73.14% to 74.40%, past XGBoost on the same split. Finer
 * still (floor 5) buys 0.22 more points for 3.3x the build, which is where the
 * returns stop being worth the memory.
 */
#if !defined(LIBXS_PREDICT_RF_MAXNODES)
#  define LIBXS_PREDICT_RF_MAXNODES 524287
#endif
#if !defined(LIBXS_PREDICT_RF_NTREES)
#  define LIBXS_PREDICT_RF_NTREES 100
#endif
/** Trees per candidate while scoring depth: enough to average out the
 *  bootstrap, few enough that trying four depths is not four full builds. */
#if !defined(LIBXS_PREDICT_RF_PROBE)
#  define LIBXS_PREDICT_RF_PROBE 12
#endif
/** Bins over the share of trees agreeing, each carrying what that share was
 *  worth. Few enough that every bin is populated on a small corpus. */
#if !defined(LIBXS_PREDICT_RF_CALIB)
#  define LIBXS_PREDICT_RF_CALIB 16
#endif
/**
 * Rows the calibration measures on, and the switch that asks for it at all.
 *
 * ZERO BY DEFAULT, so the reported confidence is the share of the trees that
 * agree: a ranking, on a scale of its own. That is what a caller taking the most
 * confident fraction of its queries needs, and it costs nothing.
 *
 * A caller reading the confidence as a PROBABILITY - gating at 0.9 and expecting
 * nine in ten to be right - needs the curve, and pays for it: the rows are
 * withheld from every tree, which measured 0.2 to 0.4 points of accuracy on the
 * crystal corpus and 0.17 on HIGGS at 800k rows. Set this to the number of rows
 * to measure on (65536 is ample; the curve wants hundreds per bin) either here
 * or in the environment under the same name.
 *
 * What it buys is comparability and nothing else. The mapping is monotone, so it
 * reorders nothing: accuracy is unchanged and precision at matched coverage is
 * identical. What changes is that a threshold means the same thing across
 * corpora, across tree granularities, and against another library - where the
 * bare share admitted 43% of HIGGS queries at a gate of 0.9 and returned 86.6%.
 */
#if !defined(LIBXS_PREDICT_RF_CALIB_ROWS)
#  define LIBXS_PREDICT_RF_CALIB_ROWS 0
#endif
/** Most of the corpus this many rows may be withheld, so that a small corpus
 *  gives up a share of itself rather than a fixed count of its rows. */
#if !defined(LIBXS_PREDICT_RF_HOLDOUT)
#  define LIBXS_PREDICT_RF_HOLDOUT 50
#endif


/**
 * Which rows the calibration withholds, as a stride: every hstep-th row up to
 * hrows of them. A rule rather than a stored set, because the bootstrap has to
 * agree with it in four places - where it is drawn, and the three that
 * reconstruct membership from it - and a rule cannot fall out of step with
 * itself. hstep of zero withholds nothing, which is the default.
 *
 * Withheld from EVERY tree, not from a fold of them. Spreading the rows over
 * folds so that each tree omits only a tenth of them looks like it buys the
 * accuracy back for nothing, and cannot: a row omitted by a tenth of the trees
 * is a row only a tenth of them can judge, and how much of the forest omits a
 * row is the same quantity as how much of it can score that row.
 *
 * Moving the query off the row instead - far enough to leave the leaf that
 * memorized it - costs no rows and was measured to fail differently: accuracy
 * against the borrowed label FALLS as the trees agree more (0.930 at a share of
 * 0.7, 0.797 at 0.9 on the crystal corpus), because a query that has crossed a
 * boundary is confidently right about where it now is while the label still
 * belongs to the row it came from. The artifact sits exactly where a gate reads.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_holdout(int p,
  int* hstep, int* hrows)
{
  const char* renv = getenv("LIBXS_PREDICT_RF_CALIB_ROWS");
  const int rows = (NULL != renv) ? atoi(renv) : LIBXS_PREDICT_RF_CALIB_ROWS;
  const int want = LIBXS_MIN(rows, p / LIBXS_PREDICT_RF_HOLDOUT);
  if (0 < rows && 0 < want && 4 <= p) {
    *hstep = p / want;
    *hrows = want;
    /* a stride of one would withhold the corpus; nothing is measurable then */
    if (2 > *hstep) { *hstep = 0; *hrows = 0; }
  }
  else {
    *hstep = 0;
    *hrows = 0;
  }
}


/**
 * The row a tree's i-th bootstrap draw lands on, skipping any row withheld for
 * the calibration. Row 1 is the fallback because a stride of at least two never
 * withholds it, where wrapping to row 0 would land on a withheld row again.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_draw(size_t i, size_t boot_n,
  size_t coprime, size_t seed, int p, int hstep, int hrows)
{
  int j = (int)(LIBXS_SHUFFLE_INDEX(i, boot_n, coprime, seed) % (size_t)p);
  if (0 < hstep && 0 == (j % hstep) && (j / hstep) < hrows) {
    j = (j + 1 < p) ? (j + 1) : 1;
  }
  return j;
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
     same budget divided among them rather than by one of its own */
  const int max_nodes = LIBXS_MIN(ntrain / min_leaf * 2 + 1,
    LIBXS_MAX(LIBXS_PREDICT_RF_MAXNODES / LIBXS_PREDICT_RF_PROBE, 1));
  int nodes_pool = 0, boot_pool = 0, nn_pool = 0;
  internal_libxs_predict_rf_node_t* nodes =
    (internal_libxs_predict_rf_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nt * (size_t)max_nodes
        * sizeof(internal_libxs_predict_rf_node_t), nodes_pool);
  int* bootstrap = (int*)LIBXS_PREDICT_MALLOC((size_t)ntrain * sizeof(int),
    boot_pool);
  int* nn = (int*)LIBXS_PREDICT_MALLOC((size_t)nt * sizeof(int), nn_pool);
  double result = 1.0;
  if (NULL != nodes && NULL != bootstrap && NULL != nn) {
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
         ranks depths against each other rather than reporting an error */
      nn[t] = internal_libxs_predict_rf_build_tree(entries, NULL, NULL, 0,
        bootstrap, ntrain,
        m, max_depth, min_leaf, nodes + (size_t)t * max_nodes, max_nodes,
        output_idx, label_off, regress, nclass, min_leaf);
    }
    for (i = ntrain; i < p; ++i) {
      const double* inputs = entries[i].inputs;
      const int label =
        (LIBXS_ROUNDX(int, entries[i].outputs[output_idx]) + label_off) & 127;
      int votes[128], best_label = 0, best_count = 0, k, nvalid = 0;
      double sum = 0;
      memset(votes, 0, sizeof(votes));
      for (t = 0; t < nt; ++t) {
        const internal_libxs_predict_rf_node_t* tn = nodes + (size_t)t * max_nodes;
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
    unsigned char* const bins = (unsigned char*)malloc((size_t)p * (size_t)m);
    double* const edge = (double*)malloc(
      (size_t)m * (size_t)(nb + 1) * sizeof(double));
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
        rf->nbins = nb;
      }
      LIBXS_PREDICT_FREE(sv, spool);
    }
    if (0 >= rf->nbins) { /* the sorted search needs none of it */
      free(bins);
      free(edge);
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


/** Releases the bins once the forest is grown: split finding is what read them,
 *  and boosting and the calibration descend the raw inputs. */
LIBXS_API_INLINE void internal_libxs_predict_rf_bins_free(libxs_predict_t* model)
{
  if (NULL != model->rf) {
    free(model->rf->bins);
    free(model->rf->bin_edge);
    model->rf->bins = NULL;
    model->rf->bin_edge = NULL;
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
       * widest three outputs worse, while costing the crystal corpus 2.6x its
       * build. It is kept because it is the only way to find out for a corpus
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
      /* last, and after the depth probe rather than before it: the probe splits
         exactly, and the bins are read by the trees the tasks grow */
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


LIBXS_API_INLINE void internal_libxs_predict_rf_build_tasks(
  libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_rf_t* rf = model->rf;
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
     * a child where it does not cost the crystal corpus 1.9 points, and leaving
     * it unconstrained where it does cost a million rows 9.5.
     */
    const int leaf_floor = (LIBXS_PREDICT_RF_MAXNODES < p * 2 / min_leaf)
      ? LIBXS_MAX(1, (p * 2 + LIBXS_PREDICT_RF_MAXNODES - 2)
        / (LIBXS_PREDICT_RF_MAXNODES - 1)) : 1;
    const int max_nodes = LIBXS_MIN(p / leaf_floor * 2 + 1,
      LIBXS_PREDICT_RF_MAXNODES);
    int begin, end, bootstrap_pool = 0, hstep = 0, hrows = 0;
    int* bootstrap = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), bootstrap_pool);
    /* a withheld row is in no tree, so the draw skips it; none by default */
    internal_libxs_predict_rf_holdout(p, &hstep, &hrows);
    internal_libxs_predict_split(total_trees, tid, ntasks, &begin, &end);
    if (NULL != bootstrap) {
      int ti;
      for (ti = begin; ti < end; ++ti) {
        const int oi = ti / ntrees;
        const int t = ti % ntrees;
        const int max_depth = rf->depth[oi];
        const size_t boot_n = (size_t)p * 2 + 1;
        const size_t boot_coprime = libxs_coprime2(boot_n);
        int nodes_pool = 0;
        internal_libxs_predict_rf_node_t* nodes;
        int i, nn;
        if (NULL != rf->trees[ti].nodes) continue;
        nodes = (internal_libxs_predict_rf_node_t*)LIBXS_PREDICT_MALLOC(
            (size_t)max_nodes * sizeof(internal_libxs_predict_rf_node_t),
            nodes_pool);
        for (i = 0; i < p; ++i) {
          bootstrap[i] = internal_libxs_predict_rf_draw((size_t)i, boot_n,
            boot_coprime, (size_t)(oi * ntrees + t) * 7 + 13, p, hstep, hrows);
        }
        if (NULL != nodes) {
          nn = internal_libxs_predict_rf_build_tree(
            model->entries, rf->bins, rf->bin_edge, rf->nbins,
            bootstrap, p, m, max_depth, min_leaf,
            nodes, max_nodes, oi, rf->label_offset[oi], rf->regress[oi],
            rf->nclass[oi], leaf_floor);
          rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
            (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
          if (NULL != rf->trees[ti].nodes) {
            memcpy(rf->trees[ti].nodes, nodes,
              (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
            rf->trees[ti].nnodes = nn;
          }
          LIBXS_PREDICT_FREE(nodes, nodes_pool);
        }
      }
      LIBXS_PREDICT_FREE(bootstrap, bootstrap_pool);
    }
  }
}


/** Index of the leaf the inputs descend to, or negative if the tree is empty
 *  or its links leave the node array. */
LIBXS_API_INLINE int internal_libxs_predict_rf_leafof(
  const internal_libxs_predict_rf_tree_t* tree, const double* inputs)
{
  int result = 0;
  if (NULL == tree->nodes || 0 == tree->nnodes) {
    result = -1;
  }
  else {
    while (0 <= result && result < tree->nnodes
      && 0 <= tree->nodes[result].feature)
    {
      const internal_libxs_predict_rf_node_t* nd = &tree->nodes[result];
      result = (inputs[nd->feature] <= nd->threshold) ? nd->left : nd->right;
    }
    if (result >= tree->nnodes) result = -1;
  }
  return result;
}


/** Shrinkage applied to each boosted stage. The leaf basis is large enough
 *  that an unshrunk read-out over it fits the sample rather than the signal. */
#if !defined(LIBXS_PREDICT_RF_RATE)
#  define LIBXS_PREDICT_RF_RATE 0.1
#endif
/** Consecutive stages allowed not to improve before boosting stops. Each stage
 *  scores on its own tree's out-of-bag rows, a different subset every time, so
 *  a single stage that fails to improve is noise rather than a trend. */
#if !defined(LIBXS_PREDICT_RF_PATIENCE)
#  define LIBXS_PREDICT_RF_PATIENCE 3
#endif
/** One row in this many is held back from every stage's leaf means, to be the
 *  only honest witness of whether the stages are still generalizing. */
#if !defined(LIBXS_PREDICT_RF_HOLD)
#  define LIBXS_PREDICT_RF_HOLD 5
#endif
#if !defined(LIBXS_PREDICT_RF_SEED)
#  define LIBXS_PREDICT_RF_SEED 1013
#endif


/**
 * Fits the additive read-out over the partitions the forest already grew.
 *
 * The two read-outs combine rather than compete: eval answers with the bagged
 * mean plus the sum of the corrections, so the stages correct a
 * variance-reduced base instead of rebuilding it. Nothing here grows a tree,
 * and eval pays one array lookup per descent it was making anyway.
 *
 * One choice carries the honesty of the whole fit: what the residual is taken
 * against. It has to be the out-of-bag forest mean, averaging each row over
 * only the trees whose bootstrap left that row out. The tempting alternative
 * is the full forest mean, on the grounds that it is exactly what eval starts
 * from and the corrections ought to be fitted against the base they will be
 * added to. That is wrong, and measurably so: on a training row the forest is
 * nearly unbiased because most of its trees memorized that row, so the leaf
 * means come out as noise rather than as bias, and summing a hundred stages of
 * noise is a random walk that degrades the read-out in proportion to the
 * learning rate. The out-of-bag mean is a few trees' worth noisier than the
 * one eval uses but carries the same bias, and bias is the only thing the
 * stages can correct.
 *
 * Applies to real-valued outputs alone. A folded output answers with a class,
 * and a class plus a real correction is not a class; boosting one needs a
 * correction per class per leaf, which is a different structure.
 */
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
    if (0 != hold[i]) {
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
/**
 * Measure what a share of the trees agreeing is worth, so that the reported
 * confidence is a probability rather than an ensemble statistic.
 *
 * The share is read out-of-bag: a row is voted on only by the trees whose
 * bootstrap left it out, exactly as the boosting stages are judged, and for the
 * same reason - on a training row most trees memorized the answer and the share
 * says nothing. That makes the share a few trees' worth noisier than the one
 * eval computes over the whole forest, so the bins are wide enough to absorb it
 * rather than narrow enough to model it.
 *
 * The curve is forced non-decreasing. It is a statement about evidence - more
 * trees agreeing cannot mean a worse answer - and a bin that dips below its
 * predecessor is reading a sampling accident, which is what pooling it away
 * says. Bins nothing landed in inherit the value below them for the same
 * reason: they carry no evidence of their own.
 */
LIBXS_API_INLINE void internal_libxs_predict_rf_calibrate(libxs_predict_t* model)
{
  internal_libxs_predict_rf_t* rf = model->rf;
  const int nbin = LIBXS_PREDICT_RF_CALIB;
  int hstep = 0, hrows = 0;
  if (0 < nbin && NULL != rf && NULL != model->entries
    && NULL != rf->regress && NULL != rf->nclass && NULL == rf->calib)
  {
    internal_libxs_predict_rf_holdout(model->nentries, &hstep, &hrows);
  }
  /* nothing withheld is nothing to measure on: the share is reported unchanged */
  if (0 < hstep && 0 < hrows) {
    const internal_libxs_predict_entry_t* entries = model->entries;
    const int ntrees = rf->ntrees;
    int hit_pool = 0, cnt_pool = 0, oi;
    double* hit = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)nbin * sizeof(double), hit_pool);
    double* cnt = (double*)LIBXS_PREDICT_MALLOC(
      (size_t)nbin * sizeof(double), cnt_pool);
    rf->calib = (double*)malloc(
      (size_t)rf->noutputs * (size_t)nbin * sizeof(double));
    if (NULL != hit && NULL != cnt && NULL != rf->calib) {
      for (oi = 0; oi < rf->noutputs; ++oi) {
        double* curve = rf->calib + (size_t)oi * nbin;
        const int tbase = oi * ntrees;
        const int nc = rf->nclass[oi];
        int b, r;
        for (b = 0; b < nbin; ++b) { hit[b] = 0; cnt[b] = 0; }
        /* a real-valued output reports no share to calibrate, see
           internal_libxs_predict_rf_eval_output */
        if (0 == rf->regress[oi] && 1 < nc && 128 >= nc) {
          for (r = 0; r < hrows; ++r) {
            const int i = r * hstep;
            int votes[128], nvote = 0, best = 0, bcount = 0, k, t;
            if (i >= model->nentries) break;
            memset(votes, 0, (size_t)nc * sizeof(int));
            /* every tree, which is the vote eval computes, and honestly so
               because the row is in none of their bootstraps */
            for (t = 0; t < ntrees; ++t) {
              const internal_libxs_predict_rf_tree_t* tr = &rf->trees[tbase + t];
              int ni;
              if (NULL == tr->nodes || 0 == tr->nnodes) continue;
              ni = internal_libxs_predict_rf_leafof(tr, entries[i].inputs);
              if (0 <= ni) {
                const int lc = tr->nodes[ni].label & 127;
                if (lc < nc) { ++votes[lc]; ++nvote; }
              }
            }
            if (0 < nvote) {
              const int label = (LIBXS_ROUNDX(int,
                entries[i].outputs[oi]) + rf->label_offset[oi]) & 127;
              for (k = 0; k < nc; ++k) {
                if (votes[k] > bcount) { bcount = votes[k]; best = k; }
              }
              /* binned by the share over the WHOLE forest, which is what eval
                 hands to the curve, rather than over the trees that voted */
              b = (int)((double)bcount / ntrees * nbin);
              if (b >= nbin) b = nbin - 1;
              if (0 > b) b = 0;
              cnt[b] += 1.0;
              if (best == label) hit[b] += 1.0;
            }
          }
        }
        { /**
           * Pool adjacent violators: where a bin scores below the one under it,
           * the two are merged and the merged block re-checked against what is
           * under IT, so a dip is averaged away against the evidence that
           * contradicts it. Clamping the dip up to its predecessor instead looks
           * like the same thing and is not - one thinly populated low bin that
           * happens to score well then propagates its value through every bin
           * above, which flattens the curve to a single number and reports one
           * confidence for every query.
           */
          double wsum[LIBXS_PREDICT_RF_CALIB], vsum[LIBXS_PREDICT_RF_CALIB];
          int at[LIBXS_PREDICT_RF_CALIB], nblock = 0, k;
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
          { /* every bin takes the block that covers it, and a bin below the
               first block or above the last takes the nearest one */
            double prev = (0 < nblock) ? (vsum[0] / wsum[0]) : 0.0;
            k = 0;
            for (b = 0; b < nbin; ++b) {
              while (k + 1 < nblock && at[k + 1] <= b) ++k;
              if (0 < nblock && at[k] <= b) prev = vsum[k] / wsum[k];
              curve[b] = prev;
            }
          }
        }
      }
    }
    else { /* without the whole measurement the share is reported unchanged */
      free(rf->calib);
      rf->calib = NULL;
    }
    LIBXS_PREDICT_FREE(cnt, cnt_pool);
    LIBXS_PREDICT_FREE(hit, hit_pool);
  }
}


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
    const double rate = (NULL != renv) ? atof(renv) : LIBXS_PREDICT_RF_RATE;
    int maxn = 0, ncmax = 1, ti, hstep = 0, hrows = 0;
    /* the same rule the bootstrap was drawn under, or the rows reconstructed as
       out-of-bag are not the rows the tree actually left out */
    internal_libxs_predict_rf_holdout(p, &hstep, &hrows);
    for (ti = 0; ti < ntrees * rf->noutputs; ++ti) {
      if (maxn < rf->trees[ti].nnodes) maxn = rf->trees[ti].nnodes;
    }
    for (ti = 0; ti < rf->noutputs; ++ti) {
      if (ncmax < rf->nclass[ti]) ncmax = rf->nclass[ti];
    }
    if (0 < maxn && 0 < rate) {
      double* pred = (double*)malloc((size_t)p * ncmax * sizeof(double));
      double* sum = (double*)malloc((size_t)maxn * ncmax * sizeof(double));
      int* cnt = (int*)malloc((size_t)maxn * ncmax * sizeof(int));
      unsigned char* oob = (unsigned char*)malloc((size_t)p);
      int* cover = (int*)malloc((size_t)p * sizeof(int));
      unsigned char* hold = (unsigned char*)malloc((size_t)p);
      if (NULL != pred && NULL != sum && NULL != cnt && NULL != oob
        && NULL != cover && NULL != hold)
      {
        const size_t boot_n = (size_t)p * 2 + 1;
        const size_t boot_coprime = libxs_coprime2(boot_n);
        const size_t hold_coprime = libxs_coprime2((size_t)p);
        int oi, h;
        /** Spread over the corpus by the shuffle rather than taken as a block,
         *  so a held-back row exists in every region the trees partition. */
        memset(hold, 0, (size_t)p);
        for (h = 0; h < p / LIBXS_PREDICT_RF_HOLD; ++h) {
          hold[(int)LIBXS_SHUFFLE_INDEX((size_t)h, (size_t)p, hold_coprime,
            LIBXS_PREDICT_RF_SEED)] = 1;
        }
        for (oi = 0; oi < rf->noutputs; ++oi) {
          const int tbase = oi * ntrees;
          const int nc = rf->nclass[oi];
          const int reg = rf->regress[oi];
          const int loff = rf->label_offset[oi];
          int stale = 0, miss0 = 0, miss1 = 0, t, i, k, c;
          double dist0 = 0, dist1 = 0;
          if (1 > nc) continue;
          memset(pred, 0, (size_t)p * nc * sizeof(double));
          for (i = 0; i < p; ++i) cover[i] = 0;
          for (t = 0; t < ntrees; ++t) {
            const internal_libxs_predict_rf_tree_t* tr = &rf->trees[tbase + t];
            if (NULL == tr->nodes || 0 == tr->nnodes) continue;
            memset(oob, 1, (size_t)p);
            for (i = 0; i < p; ++i) {
              oob[internal_libxs_predict_rf_draw((size_t)i, boot_n,
                boot_coprime, (size_t)(tbase + t) * 7 + 13,
                p, hstep, hrows)] = 0;
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
              oob[internal_libxs_predict_rf_draw((size_t)i, boot_n,
                boot_coprime, (size_t)(tbase + t) * 7 + 13,
                p, hstep, hrows)] = 0;
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
             * combined read-out no worse than the bagged one it corrects,
             * which is the property that lets it be on by default.
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
      free(hold);
      free(cover);
      free(oob);
      free(cnt);
      free(sum);
      free(pred);
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
LIBXS_API_INLINE double internal_libxs_predict_rf_eval_output(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence, double* variance)
{
  const int regress = (NULL != rf->regress) ? rf->regress[output_idx] : 0;
  const int nc = (NULL != rf->nclass) ? rf->nclass[output_idx] : 1;
  const int base = output_idx * rf->ntrees;
  int votes[128];
  double bscore[128];
  int best_label = 0, best_count = 0, nvalid = 0, t, k;
  double sum = 0, sqr = 0, boost = 0, result;
  if (0 == regress) {
    memset(votes, 0, sizeof(votes));
    memset(bscore, 0, sizeof(bscore));
  }
  for (t = 0; t < rf->ntrees; ++t) {
    const internal_libxs_predict_rf_tree_t* tree = &rf->trees[base + t];
    const int ni = internal_libxs_predict_rf_leafof(tree, inputs);
    if (0 <= ni) {
      if (0 != regress) {
        const double v = tree->nodes[ni].value;
        sum += v;
        sqr += v * v;
        ++nvalid;
        if (NULL != tree->incr) boost += tree->incr[ni];
      }
      else {
        ++votes[tree->nodes[ni].label & 127];
        ++nvalid;
        if (NULL != tree->incr) {
          for (k = 0; k < nc && k < 128; ++k) {
            bscore[k] += tree->incr[(size_t)ni * nc + k];
          }
        }
      }
    }
  }
  if (0 != regress) {
    const double mean = (0 < nvalid) ? (sum / nvalid) : 0.0;
    result = mean + boost;
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
        + ((k < nc) ? bscore[k] : 0.0);
      if (0 == k || s > best_score) {
        best_score = s;
        best_label = k;
      }
    }
    best_count = votes[best_label];
    if (NULL != confidence) {
      const double share = (rf->ntrees > 0)
        ? (double)best_count / rf->ntrees : 0.0;
      if (NULL != rf->calib) {
        /* what that share was measured to be worth, see
           internal_libxs_predict_rf_calibrate */
        int b = (int)(share * LIBXS_PREDICT_RF_CALIB);
        if (b >= LIBXS_PREDICT_RF_CALIB) b = LIBXS_PREDICT_RF_CALIB - 1;
        if (0 > b) b = 0;
        *confidence = rf->calib[(size_t)output_idx * LIBXS_PREDICT_RF_CALIB + b];
      }
      else *confidence = share;
    }
    if (NULL != variance) *variance = 0;
    result = (double)(best_label - rf->label_offset[output_idx]);
  }
  return result;
}
