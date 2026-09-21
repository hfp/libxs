/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_hist.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_timer.h>

#include "predict_args.h"

#define PROB_KNN 32
#define PROB_LOO 2000
#define PROB_NBUCKET 12
#define PROB_MINSAMPLE 12

/**
 * Escape-rate experts for the causal bank.  A fixed escape cannot be right
 * everywhere: leave-one-out measures it 6-40x too low because dropping one
 * entry leaves its neighbors intact, and a coverage-binned estimate was
 * measured flat.  Rather than estimate the rate at build time, the bank carries
 * one expert per candidate rate and reweights them per query by realized log
 * loss, which is the mechanism the converse paper's per-position order
 * selection uses to beat its own best fixed component.
 */
#define PROB_BANK_SHARE 0.01
#define PROB_BANK_RELMIN 1e-12


enum {
  PROB_FLAT = 0, PROB_FREQ = 1, PROB_DX = 2, PROB_CAL = 3, PROB_BANK = 4,
  PROB_BLEND = 5, PROB_NVARIANT = 6
};

enum { PROB_NEXPERT = 13 };

typedef struct prob_support_t {
  double* vals;
  double* freq;
  double* prior;
  double* dx;
  double* wlocal;
  double* work;
  double* mix;
  double* scratch;
  int n;
} prob_support_t;

typedef struct prob_score_t {
  double nll_all;
  double nll_attested;
  double nll_novel;
  double rr_novel;
  double maxdev;
  double pnovel_sum;
  int n_all;
  int n_attested;
  int n_novel;
  int nexact;
} prob_score_t;

/**
 * Leave-one-out novelty calibration.  The escape mass is measured rather than
 * chosen: LOO asks how often the held-out true value was absent from its own
 * local support, binned by a coverage proxy, so P(novel | coverage) is read off
 * a histogram instead of a hand-picked constant or an assumed functional form.
 */
typedef struct prob_calib_t {
  double rate[PROB_NBUCKET];
  int nsample[PROB_NBUCKET];
  double lo, hi;
  double global;
} prob_calib_t;


static const char* const prob_variant_name[] = {
  "flat", "freq", "dx", "cal", "bank", "blend"
};

/**
 * Candidate escape rates, geometric so the sweep resolves the small-rate end
 * where the optimum was measured to sit.  These double as the bank's experts.
 */
static const double prob_expert_rate[PROB_NEXPERT] = {
  0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
  0.05, 0.10, 0.20, 0.35, 0.55, 0.80
};


static void prob_normalize_inputs(const double* raw, const double* min,
  const double* rng, int m, double* norm);
static int prob_support_index(const double vals[], int n, double v);
static int prob_support_build(const double* outputs, int ntrain, int nout,
  int j, prob_support_t* sup);
static void prob_support_free(prob_support_t* sup);
static double prob_query_scan(const double* tnorm, const double* touts,
  int ntrain, int m, int nout, int j, const double* qnorm, int skip,
  prob_support_t* sup);
static double prob_coverage(const prob_support_t* sup, double dknn);
static int prob_calib_bucket(const prob_calib_t* cal, double coverage);
static double prob_calib_novel(const prob_calib_t* cal, double coverage);
static double prob_normalize_exact(double p[], int n, double scratch[]);
static void prob_backoff(int variant, const prob_support_t* sup, double dscale,
  double b[]);
static double prob_expert_eval(const prob_support_t* sup, double rate,
  int truth, double p[], double scratch[], int* nexact);
static void prob_bank_update(double weight[], const double plik[],
  double mixture, double eta);
static void prob_score_query(const prob_support_t* sup, int variant,
  double lambda, double dscale, int truth, prob_score_t* score, int novel);


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const int ninputs = (argc > 2) ? atoi(argv[2]) : 37;
  const int noutputs = (argc > 3) ? atoi(argv[3]) : 1;
  const int target = (argc > 4) ? atoi(argv[4]) : 0;
  const double split = (argc > 5) ? atof(argv[5]) : 0.8;
  const int ntest_cap = (argc > 6) ? atoi(argv[6]) : 2000;
  const char* innames = (argc > 7) ? argv[7] : NULL;
  const char* outnames = (argc > 8) ? argv[8] : NULL;
  const double eta = (argc > 9) ? atof(argv[9]) : 0.5;
  static const int numeric[] = { 2, 3, 4, 5, 6, 9 };
  int result = EXIT_FAILURE;
  if (NULL == filename || 0 == predict_slots_ok(argc, argv, numeric, 6)
    || 0 >= ninputs || 0 >= noutputs
    || 0 > target || target >= noutputs || 0 == predict_split_ok(split))
  {
    fprintf(stdout,
      "Usage: %s <csvfile> [ninputs] [noutputs] [output] [fraction]"
      " [ntest] [innames] [outnames]\n"
      "  Stage-1 probability harness: scores P(y|x) for an arbitrary\n"
      "  candidate y over the exact distinct-value support of one output,\n"
      "  using only the public prediction API.  Reports held-out NLL split\n"
      "  by locally-attested vs novel, and compares three backoff shapes\n"
      "  (flat, global frequency, input-distance dx) on the novel bucket.\n"
      "  Discrete (mass) outputs only; density mode is not covered here.\n"
      "  ntest: cap on scored queries (co-prime stride), 0 = all\n"
      "  eta:   bank learning rate (default 0.5)\n"
      "  gamma: weight-trajectory extrapolation (default 1.0)\n",
      argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(ninputs, noutputs);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        innames, outnames, NULL, 0, NULL);
      if (0 < total) {
        const int ntrain = LIBXS_MAX((int)(total * split + 0.5), 2);
        const int m = ninputs, n = noutputs;
        double* tnorm = (double*)malloc((size_t)ntrain * m * sizeof(double));
        double* touts = (double*)malloc((size_t)ntrain * n * sizeof(double));
        double* imin = (double*)malloc((size_t)m * sizeof(double));
        double* irng = (double*)malloc((size_t)m * sizeof(double));
        double* qnorm = (double*)malloc((size_t)m * sizeof(double));
        double* qraw = (double*)malloc((size_t)m * sizeof(double));
        fprintf(stdout, "Loaded %d entries from %s (M=%d, N=%d, output=%d)\n",
          total, filename, m, n, target);
        if (NULL != tnorm && NULL != touts && NULL != imin && NULL != irng
          && NULL != qnorm && NULL != qraw)
        {
          prob_support_t sup;
          libxs_timer_tick_t tick;
          int i, j;
          memset(&sup, 0, sizeof(sup));
          for (i = 0; i < ntrain; ++i) {
            libxs_predict_get(source, i, tnorm + (size_t)i * m,
              touts + (size_t)i * n);
          }
          for (j = 0; j < m; ++j) {
            double lo = tnorm[j], hi = tnorm[j];
            for (i = 1; i < ntrain; ++i) {
              const double v = tnorm[(size_t)i * m + j];
              if (v < lo) lo = v;
              if (hi < v) hi = v;
            }
            imin[j] = lo;
            irng[j] = hi - lo;
          }
          for (i = 0; i < ntrain; ++i) {
            double* row = tnorm + (size_t)i * m;
            for (j = 0; j < m; ++j) {
              row[j] = (irng[j] > 0) ? ((row[j] - imin[j]) / irng[j]) : row[j];
            }
          }
          if (EXIT_SUCCESS == prob_support_build(touts, ntrain, n, target,
            &sup))
          {
            const int nloo = LIBXS_MIN(PROB_LOO, ntrain);
            const size_t loco = libxs_coprime2((size_t)ntrain);
            libxs_hist_t* hist = libxs_hist_create(PROB_NBUCKET, 2, NULL, NULL, NULL);
            prob_calib_t cal;
            double lambda, dscale_sum = 0;
            int nnovel_loo = 0;
            memset(&cal, 0, sizeof(cal));
            tick = libxs_timer_tick();
            for (i = 0; i < nloo; ++i) {
              const int ei = (int)LIBXS_SHUFFLE_INDEX((size_t)i,
                (size_t)ntrain, loco, 0);
              const double dknn = prob_query_scan(tnorm, touts, ntrain, m, n,
                target, tnorm + (size_t)ei * m, ei, &sup);
              const int si = prob_support_index(sup.vals, sup.n,
                touts[(size_t)ei * n + target]);
              const int isnovel = (0 > si || 0 == sup.wlocal[si]) ? 1 : 0;
              double sample[2];
              sample[0] = prob_coverage(&sup, dknn);
              sample[1] = (double)isnovel;
              if (NULL != hist) libxs_hist_push(NULL, hist, sample);
              nnovel_loo += isnovel;
              dscale_sum += dknn;
            }
            lambda = 1.0 - (double)nnovel_loo / nloo;
            cal.global = (double)nnovel_loo / nloo;
            if (NULL != hist) {
              libxs_hist_info_t hi;
              memset(&hi, 0, sizeof(hi));
              libxs_hist_query(NULL, hist, &hi);
              cal.lo = hi.range[0];
              cal.hi = hi.range[1];
              for (i = 0; i < hi.nbuckets && i < PROB_NBUCKET; ++i) {
                cal.nsample[i] = hi.buckets[i];
                cal.rate[i] = (NULL != hi.vals)
                  ? hi.vals[(size_t)i * hi.nvals + 1] : cal.global;
              }
            }
            fprintf(stdout, "Support: %d distinct values | train=%d\n",
              sup.n, ntrain);
            fprintf(stdout, "LOO novelty rate: %.4f (%d/%d) -> lambda=%.4f"
              " (%.2f s)\n", (double)nnovel_loo / nloo, nnovel_loo, nloo,
              lambda, libxs_timer_duration(tick, libxs_timer_tick()));
            fprintf(stdout, "Novelty calibration (coverage %.4f..%.4f):\n",
              cal.lo, cal.hi);
            for (i = 0; i < PROB_NBUCKET; ++i) {
              if (0 < cal.nsample[i]) {
                fprintf(stdout, "  bucket %2d  n=%5d  P(novel)=%.4f%s\n",
                  i, cal.nsample[i], cal.rate[i],
                  (PROB_MINSAMPLE > cal.nsample[i]) ? "  (thin)" : "");
              }
            }
            libxs_hist_destroy(hist);
            { const int ntest_all = total - ntrain;
              const int ntest = (0 < ntest_cap && ntest_cap < ntest_all)
                ? ntest_cap : ntest_all;
              const size_t tco = (0 < ntest_all)
                ? libxs_coprime2((size_t)ntest_all) : 1;
              prob_score_t score[PROB_NVARIANT];
              double bweight[PROB_NEXPERT], bplik[PROB_NEXPERT];
              double bwacc[PROB_NEXPERT], bnll[PROB_NEXPERT];
              double bnll_att[PROB_NEXPERT], bnll_novel[PROB_NEXPERT];
              double pool_nll = 0, ent_sum = 0;
              double early_bank = 0, late_bank = 0;
              double early_pool = 0, late_pool = 0;
              int v, top1 = 0, nseen = 0;
              memset(score, 0, sizeof(score));
              memset(bwacc, 0, sizeof(bwacc));
              memset(bnll, 0, sizeof(bnll));
              memset(bnll_att, 0, sizeof(bnll_att));
              memset(bnll_novel, 0, sizeof(bnll_novel));
              for (v = 0; v < PROB_NEXPERT; ++v) {
                bweight[v] = 1.0 / PROB_NEXPERT;
              }
              tick = libxs_timer_tick();
              for (i = 0; i < ntest; ++i) {
                const int qi = ntrain + (int)LIBXS_SHUFFLE_INDEX((size_t)i,
                  (size_t)ntest_all, tco, 0);
                double qout[64];
                libxs_predict_get(source, qi, qraw, NULL);
                libxs_predict_get(source, qi, NULL, qout);
                prob_normalize_inputs(qraw, imin, irng, m, qnorm);
                { const double dknn = prob_query_scan(tnorm, touts, ntrain, m,
                    n, target, qnorm, -1, &sup);
                  const double dscale = (dknn > 0)
                    ? dknn : (dscale_sum / nloo + 1e-12);
                  const int truth = prob_support_index(sup.vals, sup.n,
                    qout[target]);
                  const int novel = (0 > truth || 0 == sup.wlocal[truth])
                    ? 1 : 0;
                  double best = -1.0;
                  int bi = -1;
                  for (j = 0; j < sup.n; ++j) {
                    if (sup.wlocal[j] > best) { best = sup.wlocal[j]; bi = j; }
                  }
                  if (0 <= bi && bi == truth) ++top1;
                  { double mix = 0, pool = 0, ent = 0;
                    int e;
                    for (e = 0; e < PROB_NEXPERT; ++e) {
                      bplik[e] = prob_expert_eval(&sup, prob_expert_rate[e],
                        truth, sup.work, sup.scratch,
                        &score[PROB_BANK].nexact);
                      mix += bweight[e] * bplik[e];
                      pool += bplik[e] / PROB_NEXPERT;
                      if (bweight[e] > 0) {
                        ent -= bweight[e] * log(bweight[e]) / log(2.0);
                      }
                      bwacc[e] += bweight[e];
                      /**
                       * Each expert is also scored standalone: a mixture that
                       * cannot beat its own best fixed component is only
                       * discovering a better constant, not adapting.
                       */
                      if (0 <= truth && truth < sup.n) {
                        const double eb = -(log(bplik[e] > 0
                          ? bplik[e] : 1e-300) / log(2.0));
                        bnll[e] += eb;
                        if (0 != novel) bnll_novel[e] += eb;
                        else bnll_att[e] += eb;
                      }
                    }
                    if (0 <= truth && truth < sup.n) {
                      /**
                       * The uniform pool is the same expert set with weights
                       * frozen at 1/N: it pays no adaptation transient, so
                       * comparing it against the bank isolates what the
                       * reweighting actually buys from what it costs.
                       */
                      const double bl = 0.5 * mix + 0.5 * pool;
                      const double bits = -(log(mix > 0 ? mix : 1e-300)
                        / log(2.0));
                      const double pbits = -(log(pool > 0 ? pool : 1e-300)
                        / log(2.0));
                      const double blbits = -(log(bl > 0 ? bl : 1e-300)
                        / log(2.0));
                      prob_score_t* s = &score[PROB_BANK];
                      prob_score_t* sb = &score[PROB_BLEND];
                      s->nll_all += bits;
                      ++s->n_all;
                      sb->nll_all += blbits;
                      ++sb->n_all;
                      pool_nll += pbits;
                      if (nseen < ntest / 2) early_bank += bits;
                      else late_bank += bits;
                      if (nseen < ntest / 2) early_pool += pbits;
                      else late_pool += pbits;
                      ent_sum += ent;
                      ++nseen;
                      if (0 != novel) {
                        s->nll_novel += bits;
                        ++s->n_novel;
                        sb->nll_novel += blbits;
                        ++sb->n_novel;
                      }
                      else {
                        s->nll_attested += bits;
                        ++s->n_attested;
                        sb->nll_attested += blbits;
                        ++sb->n_attested;
                      }
                    }
                    prob_bank_update(bweight, bplik, mix, eta);
                  }
                  for (v = 0; v < PROB_BANK; ++v) {
                    /**
                     * Only the calibrated variant gets a per-query escape; the
                     * others keep the single global rate, so the comparison
                     * isolates x-conditioning of the mass from the shape.
                     */
                    const double lam = (PROB_CAL == v)
                      ? (1.0 - prob_calib_novel(&cal,
                          prob_coverage(&sup, dknn)))
                      : lambda;
                    score[v].pnovel_sum += 1.0 - lam;
                    prob_score_query(&sup, v, lam, dscale, truth,
                      &score[v], novel);
                  }
                }
              }
              fprintf(stdout, "Scored %d held-out queries (%.2f s)\n", ntest,
                libxs_timer_duration(tick, libxs_timer_tick()));
              fprintf(stdout, "Local top-1: %d/%d = %.1f%%\n", top1, ntest,
                (0 < ntest) ? 100.0 * top1 / ntest : 0.0);
              fprintf(stdout, "Locally-attested split: attested %d (%.1f%%)"
                " | novel %d (%.1f%%)\n",
                score[0].n_attested,
                (0 < score[0].n_all)
                  ? 100.0 * score[0].n_attested / score[0].n_all : 0.0,
                score[0].n_novel,
                (0 < score[0].n_all)
                  ? 100.0 * score[0].n_novel / score[0].n_all : 0.0);
              fprintf(stdout,
                "  backoff   NLL-all  NLL-attested  NLL-novel  MRR-novel"
                "  P(novel)  observed\n");
              for (v = 0; v < PROB_NVARIANT; ++v) {
                const prob_score_t* s = &score[v];
                fprintf(stdout, "  %-8s %8.4f  %12.4f  %9.4f  %9.4f"
                  "  %8.4f  %8.4f\n",
                  prob_variant_name[v],
                  (0 < s->n_all) ? s->nll_all / s->n_all : 0.0,
                  (0 < s->n_attested)
                    ? s->nll_attested / s->n_attested : 0.0,
                  (0 < s->n_novel) ? s->nll_novel / s->n_novel : 0.0,
                  (0 < s->n_novel) ? s->rr_novel / s->n_novel : 0.0,
                  (0 < ntest) ? s->pnovel_sum / ntest : 0.0,
                  (0 < s->n_all) ? (double)s->n_novel / s->n_all : 0.0);
              }
              { const int nh = nseen / 2, nl = nseen - nseen / 2;
                fprintf(stdout, "Bank diagnostics: mean weight entropy %.3f"
                  " of %.3f max bits\n", (0 < nseen) ? ent_sum / nseen : 0.0,
                  log((double)PROB_NEXPERT) / log(2.0));
                fprintf(stdout, "  uniform pool NLL-all %.4f | bank early"
                  " %.4f late %.4f | pool early %.4f late %.4f\n",
                  (0 < nseen) ? pool_nll / nseen : 0.0,
                  (0 < nh) ? early_bank / nh : 0.0,
                  (0 < nl) ? late_bank / nl : 0.0,
                  (0 < nh) ? early_pool / nh : 0.0,
                  (0 < nl) ? late_pool / nl : 0.0);
                fprintf(stdout, "  eta=%.2f\n", eta);
              }
              { const prob_score_t* s = &score[PROB_BANK];
                fprintf(stdout, "Escape-rate experts (standalone) vs bank:\n");
                fprintf(stdout, "  rate     NLL-all  NLL-attested"
                  "  NLL-novel  mean-weight\n");
                for (v = 0; v < PROB_NEXPERT; ++v) {
                  fprintf(stdout, "  %-6.3f %8.4f  %12.4f  %9.4f  %11.4f\n",
                    prob_expert_rate[v],
                    (0 < s->n_all) ? bnll[v] / s->n_all : 0.0,
                    (0 < s->n_attested) ? bnll_att[v] / s->n_attested : 0.0,
                    (0 < s->n_novel) ? bnll_novel[v] / s->n_novel : 0.0,
                    (0 < ntest) ? bwacc[v] / ntest : 0.0);
                }
              }
              { int nexact = 0;
                double maxdev = 0;
                for (v = 0; v < PROB_NVARIANT; ++v) {
                  nexact += score[v].nexact;
                  if (score[v].maxdev > maxdev) maxdev = score[v].maxdev;
                }
                fprintf(stdout, "Normalization: %d/%d exact (sum==1.0),"
                  " max deviation %.3e\n", nexact,
                  (PROB_NVARIANT - 1 + PROB_NEXPERT) * ntest, maxdev);
              }
              result = EXIT_SUCCESS;
            }
          }
          prob_support_free(&sup);
        }
        free(qraw);
        free(qnorm);
        free(irng);
        free(imin);
        free(touts);
        free(tnorm);
      }
      else {
        fprintf(stderr, "Failed to load entries from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}


static void prob_normalize_inputs(const double* raw, const double* min,
  const double* rng, int m, double* norm)
{
  int i;
  for (i = 0; i < m; ++i) {
    norm[i] = (rng[i] > 0) ? ((raw[i] - min[i]) / rng[i]) : raw[i];
  }
}


static int prob_support_index(const double vals[], int n, double v)
{
  int lo = 0, hi = n - 1, result = -1;
  while (lo <= hi && 0 > result) {
    const int mid = lo + (hi - lo) / 2;
    if (vals[mid] < v) lo = mid + 1;
    else if (vals[mid] > v) hi = mid - 1;
    else result = mid;
  }
  return result;
}


/**
 * The support is the set of exact distinct values, not a set of tolerance
 * balls: overlapping or gapped buckets would make the masses over the support
 * sum to something other than one, which is the invariant under test here.
 */
static int prob_support_build(const double* outputs, int ntrain, int nout,
  int j, prob_support_t* sup)
{
  int result = EXIT_FAILURE;
  double* all = (double*)malloc((size_t)ntrain * sizeof(double));
  if (NULL != all) {
    int i, ndistinct = 1;
    for (i = 0; i < ntrain; ++i) all[i] = outputs[(size_t)i * nout + j];
    libxs_sort(all, ntrain, sizeof(double), libxs_cmp_f64, NULL);
    for (i = 1; i < ntrain; ++i) {
      if (all[i] != all[i - 1]) ++ndistinct;
    }
    sup->vals = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->freq = (double*)calloc((size_t)ndistinct, sizeof(double));
    sup->prior = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->dx = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->wlocal = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->work = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->mix = (double*)malloc((size_t)ndistinct * sizeof(double));
    sup->scratch = (double*)malloc((size_t)ndistinct * sizeof(double));
    if (NULL != sup->vals && NULL != sup->freq && NULL != sup->prior
      && NULL != sup->dx
      && NULL != sup->wlocal && NULL != sup->work && NULL != sup->mix
      && NULL != sup->scratch)
    {
      int k = 0;
      sup->vals[0] = all[0];
      sup->freq[0] = 1;
      for (i = 1; i < ntrain; ++i) {
        if (all[i] != all[i - 1]) sup->vals[++k] = all[i];
        sup->freq[k] += 1;
      }
      sup->n = ndistinct;
      for (i = 0; i < ndistinct; ++i) {
        sup->prior[i] = sup->freq[i] / ntrain;
      }
      result = EXIT_SUCCESS;
    }
    free(all);
  }
  return result;
}


static void prob_support_free(prob_support_t* sup)
{
  free(sup->vals);
  free(sup->freq);
  free(sup->prior);
  free(sup->dx);
  free(sup->wlocal);
  free(sup->work);
  free(sup->mix);
  free(sup->scratch);
  memset(sup, 0, sizeof(*sup));
}


/**
 * One pass over the training entries fills the local neighbor weights and the
 * per-value minimum input distance.  dx lost the ranking experiment against a
 * global frequency prior on the novel bucket, so it is retained only as a
 * reported diagnostic and no longer shapes any escape mass.
 */
static double prob_query_scan(const double* tnorm, const double* touts,
  int ntrain, int m, int nout, int j, const double* qnorm, int skip,
  prob_support_t* sup)
{
  double nd[PROB_KNN];
  int ni[PROB_KNN];
  int i, nfound = 0;
  double dknn = 0;
  for (i = 0; i < sup->n; ++i) {
    sup->dx[i] = -1.0;
    sup->wlocal[i] = 0.0;
  }
  for (i = 0; i < ntrain; ++i) {
    if (i != skip) {
      const double d = sqrt(libxs_dist2(qnorm, tnorm + (size_t)i * m, m));
      const int si = prob_support_index(sup->n > 0 ? sup->vals : NULL, sup->n,
        touts[(size_t)i * nout + j]);
      if (0 <= si && (0 > sup->dx[si] || d < sup->dx[si])) sup->dx[si] = d;
      if (nfound < PROB_KNN) {
        nd[nfound] = d;
        ni[nfound] = i;
        ++nfound;
      }
      else {
        int worst = 0, wi;
        for (wi = 1; wi < nfound; ++wi) {
          if (nd[wi] > nd[worst]) worst = wi;
        }
        if (d < nd[worst]) {
          nd[worst] = d;
          ni[worst] = i;
        }
      }
    }
  }
  for (i = 0; i < nfound; ++i) {
    const int si = prob_support_index(sup->vals, sup->n,
      touts[(size_t)ni[i] * nout + j]);
    if (0 <= si) sup->wlocal[si] += (nd[i] > 0) ? (1.0 / nd[i]) : 1e30;
    if (nd[i] > dknn) dknn = nd[i];
  }
  return dknn;
}


/**
 * Coverage proxy in the same spirit as the paper's alpha_inter * alpha_intra:
 * how concentrated the local evidence is (winning weight share) attenuated by
 * how far the neighborhood reaches (kNN radius).  A query deep inside a dense,
 * agreeing region scores near 1; a query whose neighbors are distant and
 * disagree scores near 0.
 */
static double prob_coverage(const prob_support_t* sup, double dknn)
{
  double wsum = 0, wbest = 0;
  int i;
  for (i = 0; i < sup->n; ++i) {
    wsum += sup->wlocal[i];
    if (sup->wlocal[i] > wbest) wbest = sup->wlocal[i];
  }
  return (wsum > 0) ? ((wbest / wsum) / (1.0 + dknn)) : 0.0;
}


static int prob_calib_bucket(const prob_calib_t* cal, double coverage)
{
  int result = 0;
  if (cal->hi > cal->lo) {
    const double t = (coverage - cal->lo) / (cal->hi - cal->lo);
    result = (int)(t * PROB_NBUCKET);
    if (0 > result) result = 0;
    if (PROB_NBUCKET <= result) result = PROB_NBUCKET - 1;
  }
  return result;
}


/**
 * Novelty probability for a query.  Thin buckets fall back to the global LOO
 * rate rather than trusting a rate estimated from a handful of samples.  The
 * Krichevsky-Trofimov floor (k+1/2)/(n+1) is not cosmetic: a bucket in which
 * LOO happened to observe no novel case would otherwise assert that novelty is
 * impossible there, and a single novel query then costs unbounded bits.
 */
static double prob_calib_novel(const prob_calib_t* cal, double coverage)
{
  const int b = prob_calib_bucket(cal, coverage);
  const int n = cal->nsample[b];
  double result = cal->global;
  if (PROB_MINSAMPLE <= n) {
    const double k = cal->rate[b] * n;
    result = (k + 0.5) / (n + 1.0);
  }
  return result;
}


/**
 * Exact normalization: the residual goes onto the largest mass, where the
 * relative perturbation is smallest, and the deviation before the fixup is
 * returned so the caller can check the correction stayed at rounding level
 * rather than masking an inconsistency.
 */
static double prob_normalize_exact(double p[], int n, double scratch[])
{
  double result = 0;
  if (0 < n) {
    const double total = libxs_sum2(p, n);
    int i, imax = 0;
    if (0 < total) {
      for (i = 0; i < n; ++i) p[i] /= total;
    }
    else {
      for (i = 0; i < n; ++i) p[i] = 1.0 / n;
    }
    for (i = 1; i < n; ++i) {
      if (p[i] > p[imax]) imax = i;
    }
    for (i = 0; i < n; ++i) {
      if (i != imax) scratch[i - (i > imax ? 1 : 0)] = p[i];
    }
    /**
     * The residual lands on the largest mass, where the relative perturbation
     * is smallest.  Assigning 1.0 - rest is not sufficient: that subtraction
     * rounds unless rest is at least 0.5, and re-running it does not converge.
     * Instead the correction is applied in the same order the check sums, so
     * the fixed-up element absorbs exactly the error the accumulation makes.
     */
    { const double rest = (1 < n) ? libxs_sum2(scratch, n - 1) : 0.0;
      double dev;
      result = LIBXS_FABS(p[imax] - (1.0 - rest));
      p[imax] = 1.0 - rest;
      dev = libxs_sum2(p, n) - 1.0;
      if (0 != dev) p[imax] -= dev;
    }
  }
  return result;
}


static void prob_backoff(int variant, const prob_support_t* sup, double dscale,
  double b[])
{
  int i;
  for (i = 0; i < sup->n; ++i) {
    if (PROB_FREQ == variant || PROB_CAL == variant) b[i] = sup->freq[i];
    else if (PROB_DX == variant) {
      b[i] = (0 <= sup->dx[i])
        ? (1.0 / (1.0 + sup->dx[i] / dscale)) : 0.0;
    }
    else b[i] = 1.0;
  }
}


/**
 * Probability the observed value receives under one escape rate.  The local
 * evidence and the frequency backoff are both proper distributions over the
 * support, so mixing them with weight (1-rate) on the escape is itself proper;
 * the value asked about is always inside the support here, so no atom for
 * never-seen values is needed at this stage.
 */
static double prob_expert_eval(const prob_support_t* sup, double rate,
  int truth, double p[], double scratch[], int* nexact)
{
  int i;
  for (i = 0; i < sup->n; ++i) p[i] = sup->wlocal[i];
  prob_normalize_exact(p, sup->n, scratch);
  /**
   * freq holds raw counts, so it must be normalized before mixing: otherwise
   * the effective escape is rate * ntrain rather than rate, and the sweep
   * measures a rescaled axis whose optimum drifts with the training size.
   */
  for (i = 0; i < sup->n; ++i) {
    p[i] = (1.0 - rate) * p[i] + rate * sup->prior[i];
  }
  prob_normalize_exact(p, sup->n, scratch);
  if (NULL != nexact && 1.0 == libxs_sum2(p, sup->n)) ++(*nexact);
  return (0 <= truth && truth < sup->n) ? p[truth] : 0.0;
}


/**
 * One causal fixed-share step over the escape-rate experts: multiplicative
 * log-loss update toward the experts that beat the mixture, then a uniform
 * share redistributed so an expert that was wrong for a stretch can recover.
 * The update is scored strictly after the prediction is committed, so no target
 * information enters the reported figure.
 */
static void prob_bank_update(double weight[], const double plik[],
  double mixture, double eta)
{
  double total = 0;
  int i;
  for (i = 0; i < PROB_NEXPERT; ++i) {
    double relative = (mixture > 0) ? (plik[i] / mixture) : 1.0;
    if (!(relative > 0.0)) relative = PROB_BANK_RELMIN;
    weight[i] *= pow(relative, eta);
    total += weight[i];
  }
  if (total > 0) {
    const double uniform = 1.0 / PROB_NEXPERT;
    for (i = 0; i < PROB_NEXPERT; ++i) {
      weight[i] = (1.0 - PROB_BANK_SHARE) * weight[i] / total
        + PROB_BANK_SHARE * uniform;
    }
  }
}


static void prob_score_query(const prob_support_t* sup, int variant,
  double lambda, double dscale, int truth, prob_score_t* score, int novel)
{
  double* p = sup->work;
  double* mix = sup->mix;
  int i, rank = 1;
  prob_backoff(variant, sup, dscale, mix);
  prob_normalize_exact(mix, sup->n, sup->scratch);
  for (i = 0; i < sup->n; ++i) p[i] = sup->wlocal[i];
  prob_normalize_exact(p, sup->n, sup->scratch);
  for (i = 0; i < sup->n; ++i) {
    p[i] = lambda * p[i] + (1.0 - lambda) * mix[i];
  }
  prob_normalize_exact(p, sup->n, sup->scratch);
  { const double sum = libxs_sum2(p, sup->n);
    const double dev = LIBXS_FABS(sum - 1.0);
    if (1.0 == sum) ++score->nexact;
    if (dev > score->maxdev) score->maxdev = dev;
  }
  if (0 <= truth && truth < sup->n) {
    const double bits = -(log(p[truth] > 0 ? p[truth] : 1e-300) / log(2.0));
    score->nll_all += bits;
    ++score->n_all;
    if (0 != novel) {
      score->nll_novel += bits;
      ++score->n_novel;
      /**
       * Ties count as half a rank each: a flat backoff ranks every value
       * equally, and crediting it rank 1 would report a perfect MRR for a
       * shape that carries no ordering information at all.
       */
      { int nties = 0;
        for (i = 0; i < sup->n; ++i) {
          if (mix[i] > mix[truth]) ++rank;
          else if (i != truth && mix[i] == mix[truth]) ++nties;
        }
        score->rr_novel += 1.0 / (rank + 0.5 * nties);
      }
    }
    else {
      score->nll_attested += bits;
      ++score->n_attested;
    }
  }
}
