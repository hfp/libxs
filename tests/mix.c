/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>

#define NSLOT 4


static double weight_sum(const libxs_mix_t* mix)
{
  double total = 0.0;
  int i;
  for (i = 0; i < mix->nslot; ++i) total += mix->weight[i];
  return total;
}


/** A uniform prior must pool to the plain mean of the experts. */
static int check_uniform_pool(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t mix;
  if (EXIT_SUCCESS == libxs_mix_create(&mix, NSLOT, 0.15, 0.005, 1e-4)) {
    double prob[NSLOT];
    double pooled;
    int i;
    for (i = 0; i < NSLOT; ++i) prob[i] = 0.1 * (i + 1);
    pooled = libxs_mix_pool(&mix, prob, NULL);
    if (fabs(pooled - 0.25) > 1e-12) { /* mean of .1 .2 .3 .4 */
      fprintf(stderr, "uniform pool %.17g, expected 0.25\n", pooled);
    }
    else if (fabs(weight_sum(&mix) - 1.0) > 1e-12) {
      fprintf(stderr, "initial weights sum to %.17g\n", weight_sum(&mix));
    }
    else result = EXIT_SUCCESS;
    libxs_mix_destroy(&mix);
  }
  return result;
}


/** Weights must stay normalized, and pooling must not move them. */
static int check_pool_is_pure(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t mix;
  if (EXIT_SUCCESS == libxs_mix_create(&mix, NSLOT, 0.15, 0.005, 1e-4)) {
    double prob[NSLOT], before[NSLOT];
    int i, moved = 0;
    for (i = 0; i < NSLOT; ++i) { prob[i] = 0.1 * (i + 1); before[i] = mix.weight[i]; }
    libxs_mix_pool(&mix, prob, NULL);
    libxs_mix_pool(&mix, prob, NULL);
    for (i = 0; i < NSLOT; ++i) {
      if (before[i] != mix.weight[i]) moved = 1;
    }
    if (0 != moved) fprintf(stderr, "pool moved the weights\n");
    else result = EXIT_SUCCESS;
    libxs_mix_destroy(&mix);
  }
  return result;
}


/**
 * The expert that is consistently right must gain weight, and the total must
 * stay normalized over many steps - the property the whole mechanism rests on.
 */
static int check_learning(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t mix;
  if (EXIT_SUCCESS == libxs_mix_create(&mix, NSLOT, 0.15, 0.005, 1e-4)) {
    double prob[NSLOT];
    int i, step;
    for (i = 0; i < NSLOT; ++i) prob[i] = 0.01;
    prob[2] = 0.9; /* slot 2 is the good expert */
    for (step = 0; step < 500; ++step) libxs_mix_observe(&mix, prob, NULL);
    if (fabs(weight_sum(&mix) - 1.0) > 1e-9) {
      fprintf(stderr, "weights drifted: sum %.17g\n", weight_sum(&mix));
    }
    else if (!(mix.weight[2] > 0.9)) {
      fprintf(stderr, "good expert only reached %.6f\n", mix.weight[2]);
    }
    else result = EXIT_SUCCESS;
    libxs_mix_destroy(&mix);
  }
  return result;
}


/**
 * Abstention is not a probability of zero: a silent expert must neither drag
 * the pool down nor lose weight for staying silent. This is the distinction that
 * a naive implementation gets wrong by treating an absent opinion as p=0.
 */
static int check_abstention(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t mix;
  if (EXIT_SUCCESS == libxs_mix_create(&mix, NSLOT, 0.15, 0.005, 1e-4)) {
    double prob[NSLOT];
    int active[NSLOT];
    double pooled;
    int i;
    for (i = 0; i < NSLOT; ++i) { prob[i] = 0.5; active[i] = 1; }
    prob[3] = 0.0; active[3] = 0; /* abstains */
    pooled = libxs_mix_pool(&mix, prob, active);
    if (fabs(pooled - 0.5) > 1e-12) {
      fprintf(stderr, "abstention diluted the pool to %.17g\n", pooled);
    }
    else result = EXIT_SUCCESS;
    libxs_mix_destroy(&mix);
  }
  return result;
}


/**
 * The ratio floor: an expert that gives the outcome no mass at all must not be
 * multiplied by exactly zero, or the share term can never revive it (the share
 * only reaches slots that still hold mass). With relmin disabled the expert is
 * expected to die - asserting BOTH directions is what makes the floor's purpose
 * explicit rather than incidental.
 */
static int check_relmin(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t floored, bare;
  if (EXIT_SUCCESS == libxs_mix_create(&floored, NSLOT, 0.5, 0.005, 1e-4)
    && EXIT_SUCCESS == libxs_mix_create(&bare, NSLOT, 0.5, 0.005, 0.0))
  {
    double prob[NSLOT];
    int i;
    for (i = 0; i < NSLOT; ++i) prob[i] = 0.5;
    prob[1] = 0.0; /* slot 1 gives the outcome nothing */
    libxs_mix_observe(&floored, prob, NULL);
    libxs_mix_observe(&bare, prob, NULL);
    if (!(floored.weight[1] > 0.0)) {
      fprintf(stderr, "floored bank still zeroed the slot\n");
    }
    else if (0.0 != bare.weight[1]) {
      fprintf(stderr, "unfloored bank kept %.17g, expected 0\n",
        bare.weight[1]);
    }
    else {
      /* and the floored slot must recover once it is right again */
      for (i = 0; i < NSLOT; ++i) prob[i] = 0.01;
      prob[1] = 0.9;
      for (i = 0; i < 200; ++i) libxs_mix_observe(&floored, prob, NULL);
      if (!(floored.weight[1] > 0.5)) {
        fprintf(stderr, "floored slot did not recover: %.6f\n",
          floored.weight[1]);
      }
      else result = EXIT_SUCCESS;
    }
  }
  libxs_mix_destroy(&floored);
  libxs_mix_destroy(&bare);
  return result;
}


/** reset(active) must leave inactive slots at zero permanently. */
static int check_reset_mask(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t mix;
  if (EXIT_SUCCESS == libxs_mix_create(&mix, NSLOT, 0.15, 0.005, 1e-4)) {
    int active[NSLOT];
    double prob[NSLOT];
    int i;
    for (i = 0; i < NSLOT; ++i) { active[i] = 1; prob[i] = 0.5; }
    active[0] = 0;
    libxs_mix_reset(&mix, active);
    for (i = 0; i < 50; ++i) libxs_mix_observe(&mix, prob, NULL);
    if (0.0 != mix.weight[0]) {
      fprintf(stderr, "disabled slot gained weight %.17g\n", mix.weight[0]);
    }
    else if (fabs(weight_sum(&mix) - 1.0) > 1e-9) {
      fprintf(stderr, "masked weights sum to %.17g\n", weight_sum(&mix));
    }
    else result = EXIT_SUCCESS;
    libxs_mix_destroy(&mix);
  }
  return result;
}


/** observe must report the pool taken BEFORE the update, not after. */
static int check_causal_order(void)
{
  int result = EXIT_FAILURE;
  libxs_mix_t a, b;
  if (EXIT_SUCCESS == libxs_mix_create(&a, NSLOT, 0.5, 0.005, 1e-4)
    && EXIT_SUCCESS == libxs_mix_create(&b, NSLOT, 0.5, 0.005, 1e-4))
  {
    double prob[NSLOT];
    double reported, pre;
    int i;
    for (i = 0; i < NSLOT; ++i) prob[i] = 0.1 * (i + 1);
    pre = libxs_mix_pool(&b, prob, NULL);
    reported = libxs_mix_observe(&a, prob, NULL);
    if (fabs(reported - pre) > 1e-15) {
      fprintf(stderr, "observe reported a post-update pool: %.17g vs %.17g\n",
        reported, pre);
    }
    else result = EXIT_SUCCESS;
  }
  libxs_mix_destroy(&a);
  libxs_mix_destroy(&b);
  return result;
}


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
  if (EXIT_SUCCESS == result) result = check_uniform_pool();
  if (EXIT_SUCCESS == result) result = check_pool_is_pure();
  if (EXIT_SUCCESS == result) result = check_learning();
  if (EXIT_SUCCESS == result) result = check_abstention();
  if (EXIT_SUCCESS == result) result = check_relmin();
  if (EXIT_SUCCESS == result) result = check_reset_mask();
  if (EXIT_SUCCESS == result) result = check_causal_order();
  return result;
}
