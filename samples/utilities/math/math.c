/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_utils.h>
#include <libxs.h>


int main(int argc, char* argv[])
{
  const int insize = (1 < argc ? atoi(argv[1]) : 0);
  const int niters = (2 < argc ? atoi(argv[2]) : 1);
  const size_t n = (0 >= insize ? (((size_t)2 << 30/*2 GB*/) / sizeof(float)) : ((size_t)insize));
  float *inp, *out, *gold;
  unsigned char* low;
  size_t size, nrpt;
  int result;

  if (0 < niters) {
    nrpt = niters;
    size = n;
  }
  else {
    nrpt = n;
    size = LIBXS_MAX(LIBXS_ABS(niters), 1);
  }

  gold = (float*)(malloc(sizeof(float) * size));
  out = (float*)(malloc(sizeof(float) * size));
  inp = (float*)(malloc(sizeof(float) * size));
  low = (unsigned char*)(malloc(size));

  if (NULL != gold && NULL != out && NULL != inp && NULL != low) {
    libxs_timer_tickint start;
    libxs_matdiff_info diff;
    size_t i, j;

    /* initialize the input data */
    libxs_rng_set_seed(25071975);
    libxs_rng_f32_seq(inp, (libxs_blasint)size);
    for (i = 0; i < size; ++i) {
      low[i]= (unsigned char)(255.f * inp[i]);
    }

    /* collect gold data for exp2 function */
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)LIBXS_EXP2(inp[i]);
        }
      }
      printf("standard exp2:\t%.3f s\t\tgold\n", libxs_timer_duration(start, libxs_timer_tick()));
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = LIBXS_EXP2F(inp[i]);
        }
      }
      printf("standard exp2f:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxs_sexp2(inp[i]);
        }
      }
      printf("libxs_sexp2:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    /* collect gold data for limited-range exp2 function */
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)LIBXS_EXP2(low[i]);
        }
      }
      printf("low-range exp2:\t%.3f s\t\tgold\n", libxs_timer_duration(start, libxs_timer_tick()));
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxs_sexp2_u8(low[i]);
        }
      }
      printf("libxs_sexp2:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    printf("\n"); /* separate exp and sqrt output */
    /* collect gold data for sqrt function */
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)sqrt(inp[i]);
        }
      }
      printf("standard sqrt:\t%.3f s\t\tgold\n", libxs_timer_duration(start, libxs_timer_tick()));
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = (float)libxs_dsqrt(inp[i]);
        }
      }
      printf("libxs_dsqrt:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = LIBXS_SQRTF(inp[i]);
        }
      }
      printf("standard sqrtf:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }
    { start = libxs_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxs_ssqrt(inp[i]);
        }
      }
      printf("libxs_ssqrt:\t%.3f s", libxs_timer_duration(start, libxs_timer_tick()));
      if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE_F32, 1/*m*/,
        (libxs_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }

  free(gold);
  free(out);
  free(inp);
  free(low);

  return result;
}

