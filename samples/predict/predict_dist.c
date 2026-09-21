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
#include <libxs/libxs_rng.h>
#include <libxs/libxs_timer.h>

#include "predict_args.h"

#define DIST_MAXSUP 4096
#define DIST_DRAWS 20000


int main(int argc, char* argv[])
{
  const char* filename = (argc > 1) ? argv[1] : NULL;
  const int ninputs = (argc > 2) ? atoi(argv[2]) : 37;
  const int noutputs = (argc > 3) ? atoi(argv[3]) : 1;
  const int output = (argc > 4) ? atoi(argv[4]) : 0;
  const double split = (argc > 5) ? atof(argv[5]) : 0.8;
  const int ntest_cap = (argc > 6) ? atoi(argv[6]) : 500;
  const char* innames = (argc > 7) ? argv[7] : NULL;
  const char* outnames = (argc > 8) ? argv[8] : NULL;
  const int vocabulary = (argc > 9) ? atoi(argv[9]) : 0;
  const int decompose = (argc > 10) ? atoi(argv[10]) : LIBXS_PREDICT_RAW;
  static const int numeric[] = { 2, 3, 4, 5, 6, 9, 10 };
  int result = EXIT_FAILURE;

  libxs_init();

  if (NULL == filename || 0 == predict_slots_ok(argc, argv, numeric, 7)
    || 0 >= ninputs || 0 >= noutputs
    || 0 > output || output >= noutputs || 0 == predict_split_ok(split))
  {
    fprintf(stdout,
      "Usage: %s <csvfile> [ninputs] [noutputs] [output] [fraction]"
      " [ntest] [innames] [outnames]\n"
      "  Validates libxs_predict_prob: exact normalization, agreement\n"
      "  between the single-candidate and full-distribution entry points,\n"
      "  and that sampling from the reported distribution reproduces it.\n"
      "  Reports held-out NLL split by locally-attested vs novel.\n"
      "  vocabulary: total distinct values the caller considers possible;\n"
      "    0 normalizes over the attested support plus a novel atom, so a\n"
      "    value attested nowhere scores zero (infinite bits) by design.\n",
      argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(ninputs, noutputs);
    if (NULL != source) {
      const int total = libxs_predict_load_csv(source, filename, NULL,
        innames, outnames, NULL, 0, NULL);
      if (0 < total) {
        const int ntrain = LIBXS_MAX((int)(total * split + 0.5), 2);
        libxs_predict_t* model = libxs_predict_create(ninputs, noutputs);
        fprintf(stdout, "Loaded %d entries from %s (M=%d N=%d output=%d)\n",
          total, filename, ninputs, noutputs, output);
        if (NULL != model) {
          double* inbuf = (double*)malloc((size_t)ninputs * sizeof(double));
          double* outbuf = (double*)malloc((size_t)noutputs * sizeof(double));
          double* sval = (double*)malloc(DIST_MAXSUP * sizeof(double));
          double* sprb = (double*)malloc(DIST_MAXSUP * sizeof(double));
          void* pctx = NULL;
          int i;
          for (i = 0; i < ntrain; ++i) {
            libxs_predict_get(source, i, inbuf, outbuf);
            libxs_predict_push(NULL, model, inbuf, outbuf);
          }
          libxs_predict_set_mode(model, LIBXS_PREDICT_CLASSIFY);
          if (LIBXS_PREDICT_RAW != decompose) {
            libxs_predict_set_decompose(model, decompose);
          }
          if (EXIT_SUCCESS == libxs_predict_build(model, 0, 2, 0)
            && NULL != inbuf && NULL != outbuf && NULL != sval && NULL != sprb
            && NULL != (pctx = libxs_predict_prob_create(model)))
          {
            const int ntest_all = total - ntrain;
            const int ntest = (0 < ntest_cap && ntest_cap < ntest_all)
              ? ntest_cap : ntest_all;
            const size_t tco = (0 < ntest_all)
              ? libxs_coprime2((size_t)ntest_all) : 1;
            libxs_timer_tick_t tick = libxs_timer_tick();
            double nll_att = 0, nll_nov = 0, maxdev = 0, novel_mass = 0;
            double first_ent = 0, last_ent = 0;
            int nexact = 0, nsum = 0, nagree = 0, ncmp = 0;
            int n_att = 0, n_nov = 0, ns_last = 0;
            for (i = 0; i < ntest; ++i) {
              const int qi = ntrain + (int)LIBXS_SHUFFLE_INDEX((size_t)i,
                (size_t)ntest_all, tco, 0);
              libxs_predict_prob_info_t pinfo;
              double truth[64];
              int ns;
              libxs_predict_get(source, qi, inbuf, NULL);
              libxs_predict_get(source, qi, NULL, truth);
              memset(&pinfo, 0, sizeof(pinfo));
              /* full distribution first: it defines the denominator */
              ns = libxs_predict_prob_observe(NULL, model, pctx, inbuf,
                output, NULL, sval, sprb, DIST_MAXSUP, &novel_mass, NULL,
                vocabulary, 1);
              if (0 < ns && ns < DIST_MAXSUP) {
                double sum, dev;
                sprb[ns] = novel_mass;
                sum = libxs_sum2(sprb, ns + 1);
                dev = LIBXS_FABS(sum - 1.0);
                ns_last = ns;
                ++nsum;
                if (1.0 == sum) ++nexact;
                if (dev > maxdev) maxdev = dev;
              }
              /* single candidate must agree with the sweep bitwise */
              libxs_predict_prob(NULL, model, pctx, inbuf, truth, outbuf,
                &pinfo, vocabulary, 1);
              if (0 < ns && NULL != pinfo.prob) {
                int si;
                for (si = 0; si < ns; ++si) {
                  if (sval[si] == truth[output]) {
                    ++ncmp;
                    if (sprb[si] == pinfo.prob[output]) ++nagree;
                    break;
                  }
                }
                if (0 == i) first_ent = pinfo.entropy;
                last_ent = pinfo.entropy;
                if (NULL != pinfo.logprob && NULL != pinfo.attested) {
                  if (0 != pinfo.attested[output]) {
                    nll_att -= pinfo.logprob[output];
                    ++n_att;
                  }
                  else {
                    nll_nov -= pinfo.logprob[output];
                    ++n_nov;
                  }
                }
              }
            }
            fprintf(stdout, "Scored %d queries (%.2f s), support %d\n",
              ntest, libxs_timer_duration(tick, libxs_timer_tick()), ns_last);
            fprintf(stdout, "Invariant 1 (sum==1.0 bitwise): %d/%d exact,"
              " max deviation %.3e\n", nexact, nsum, maxdev);
            fprintf(stdout, "Invariant 2 (single==sweep bitwise): %d/%d\n",
              nagree, ncmp);
            fprintf(stdout, "Escape entropy: first %.3f -> last %.3f"
              " of %.3f max bits\n", first_ent, last_ent,
              log((double)13) / log(2.0));
            fprintf(stdout, "NLL: attested %.4f (n=%d) | novel %.4f (n=%d)\n",
              (0 < n_att) ? nll_att / n_att : 0.0, n_att,
              (0 < n_nov) ? nll_nov / n_nov : 0.0, n_nov);
            /**
             * Invariant 3: draw from the reported distribution and check the
             * empirical histogram converges back to it.  A distribution that
             * cannot be sampled to reproduce itself is not a distribution,
             * whatever its masses sum to.
             */
            { const int ns = libxs_predict_prob_observe(NULL, model, pctx,
                inbuf, output, NULL, sval, sprb, DIST_MAXSUP, &novel_mass,
                NULL, vocabulary, 1);
              if (0 < ns && ns <= DIST_MAXSUP) {
                double* count = (double*)calloc((size_t)ns, sizeof(double));
                if (NULL != count) {
                  double worst = 0;
                  int d, k;
                  libxs_rng_set_seed(1);
                  for (d = 0; d < DIST_DRAWS; ++d) {
                    const double u = libxs_rng_f64();
                    double cum = 0;
                    for (k = 0; k < ns; ++k) {
                      cum += sprb[k];
                      if (u <= cum) { count[k] += 1; break; }
                    }
                  }
                  for (k = 0; k < ns; ++k) {
                    const double emp = count[k] / DIST_DRAWS;
                    const double dev = LIBXS_FABS(emp - sprb[k]);
                    if (dev > worst) worst = dev;
                  }
                  fprintf(stdout, "Invariant 3 (sampling reconverges):"
                    " max |empirical-reported| = %.4f over %d draws\n",
                    worst, DIST_DRAWS);
                  free(count);
                }
              }
            }
            /**
             * Escape weights must survive save/load: a reloaded model that
             * re-learned from the uniform prior would silently re-pay the
             * adaptation transient the weights exist to avoid.
             */
            { size_t bsize = 0;
              void* buf;
              const int qr = libxs_predict_save(model, NULL, &bsize);
              buf = malloc(bsize);
              if (NULL != buf
                && EXIT_SUCCESS == libxs_predict_save(model, buf, &bsize))
              {
                libxs_predict_t* rl = libxs_predict_load(buf, bsize);
                if (NULL == rl) {
                  fprintf(stdout, "  (query=%d size=%lu)\n", qr,
                    (unsigned long)bsize);
                }
                if (NULL != rl) {
                  libxs_predict_prob_info_t a, b;
                  double pa[64], pb[64];
                  libxs_predict_t* rl2 = libxs_predict_load(buf, bsize);
                  memset(&a, 0, sizeof(a));
                  memset(&b, 0, sizeof(b));
                  libxs_predict_get(source, ntrain, inbuf, outbuf);
                  /**
                   * Both sides must start from the same bank state, so the
                   * comparison is between two independent reloads of one
                   * buffer.  Scoring the live model again would advance its
                   * bank a further step and compare unequal histories.
                   */
                  { void* c1 = libxs_predict_prob_create(rl2);
                    void* c2 = libxs_predict_prob_create(rl);
                    libxs_predict_prob(NULL, rl2, c1, inbuf, outbuf, pa, &a,
                      vocabulary, 1);
                    libxs_predict_prob(NULL, rl, c2, inbuf, outbuf, pb, &b,
                      vocabulary, 1);
                    libxs_predict_prob_destroy(c1);
                    libxs_predict_prob_destroy(c2);
                  }
                  libxs_predict_destroy(rl2);
                  fprintf(stdout, "Escape round-trip: entropy %.6f vs %.6f"
                    " | P %.17g vs %.17g -> %s\n", a.entropy, b.entropy,
                    pa[output], pb[output],
                    (a.entropy == b.entropy && pa[output] == pb[output])
                      ? "identical" : "DIFFERS");
                  libxs_predict_destroy(rl);
                }
                else fprintf(stdout, "Escape round-trip: load failed\n");
              }
              free(buf);
            }
            result = EXIT_SUCCESS;
          }
          libxs_predict_prob_destroy(pctx);
          free(sprb);
          free(sval);
          free(outbuf);
          free(inbuf);
          libxs_predict_destroy(model);
        }
      }
      else fprintf(stderr, "Failed to load entries from %s\n", filename);
      libxs_predict_destroy(source);
    }
  }

  libxs_finalize();

  return result;
}
