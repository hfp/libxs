/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_ngram.h>


static int check_mass(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int vocab)
{
  int result = EXIT_SUCCESS;
  unsigned int id;
  double mass = 0.0;
  for (id = 1; id <= vocab; ++id) {
    mass += libxs_ngram_prob(model, hist, hlen, id);
  }
  if (fabs(mass - 1.0) > 1e-12) {
    fprintf(stderr, "ngram probability mass %.17g for history length %d\n",
      mass, hlen);
    result = EXIT_FAILURE;
  }
  return result;
}


int main(int argc, char* argv[])
{
  enum { VOCAB = 12 };
  libxs_ngram_t model;
  unsigned int hist1[1];
  unsigned int hist2[2];
  int result = EXIT_SUCCESS;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);
  hist1[0] = 1;
  hist2[0] = 1;
  hist2[1] = 2;
  if (EXIT_SUCCESS != libxs_ngram_create(&model, 2)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    libxs_ngram_observe(&model, hist1, 1, 2);
    libxs_ngram_observe(&model, hist1, 1, 2);
    libxs_ngram_observe(&model, hist1, 1, 3);
    libxs_ngram_observe(&model, hist2, 2, 3);
    libxs_ngram_observe(&model, hist2, 2, 4);
    { unsigned int id;
      hist1[0] = 5;
      for (id = 1; id <= VOCAB; ++id) {
        libxs_ngram_observe(&model, hist1, 1, id);
      }
    }
    libxs_ngram_finalize(&model, VOCAB);
    hist1[0] = 1;
    result = check_mass(&model, hist1, 1, VOCAB);
  }
  if (EXIT_SUCCESS == result) {
    hist1[0] = 5;
    result = check_mass(&model, hist1, 1, VOCAB);
  }
  if (EXIT_SUCCESS == result) {
    result = check_mass(&model, hist2, 2, VOCAB);
  }
  libxs_ngram_destroy(&model);
  return result;
}
