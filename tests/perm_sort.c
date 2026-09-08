/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_perm.h>

/**
 * libxs_sort across the sizes that select its implementation. The comparison
 * sort and the radix path answer the same question and are chosen by n, so a
 * test at one size says nothing about the other: the sizes below straddle the
 * threshold, and each is run for every kind of comparator. What is checked is
 * not only the order but the contract around it, because ctx means the source
 * to read from for a value comparator and the keys to compare by for the
 * indirect one, and getting that backwards sorted a buffer nothing had written.
 */

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

/* an independent oracle: what the result must equal, not merely that it rises */
static int cmp_qsort(const void* a, const void* b) {
  const double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

/* an unknown comparator, which must take the generic path */
static int cmp_descend(const void* a, const void* b, void* ctx) {
  const double x = *(const double*)a, y = *(const double*)b;
  LIBXS_UNUSED(ctx);
  return (x < y) - (x > y);
}

/* spread, signed, and with duplicates so equal keys are exercised */
static double value_of(int i) {
  const double v = (double)(int)((unsigned int)i * 2654435761u % 1000003u);
  return ((0 == (i % 7)) ? -v : v) / 8.0;
}

int main(void) {
  /* around LIBXS_SORT_RADIX_MIN, and one size well above it */
  const int sizes[] = { 1, 2, 3, 39, 40, 41, 511, 512, 513, 4096 };
  const int nsizes = (int)(sizeof(sizes) / sizeof(*sizes));
  int result = EXIT_SUCCESS;
  int si;
  for (si = 0; si < nsizes && EXIT_SUCCESS == result; ++si) {
    const int n = sizes[si];
    double* data = (double*)malloc((size_t)n * sizeof(double));
    double* want = (double*)malloc((size_t)n * sizeof(double));
    double* src = (double*)malloc((size_t)n * sizeof(double));
    double* keys = (double*)malloc((size_t)n * sizeof(double));
    int* perm = (int*)malloc((size_t)n * sizeof(int));
    char* seen = (char*)calloc((size_t)n, 1);
    if (NULL == data || NULL == want || NULL == src || NULL == keys
      || NULL == perm || NULL == seen)
    {
      result = EXIT_FAILURE;
    }
    else {
      int i;
      for (i = 0; i < n; ++i) {
        data[i] = value_of(i);
        src[i] = value_of(i);
        keys[i] = value_of(i);
        perm[i] = i;
        want[i] = value_of(i);
      }
      qsort(want, (size_t)n, sizeof(double), cmp_qsort);
      libxs_sort(data, n, sizeof(double), libxs_cmp_f64, NULL);
      for (i = 0; i < n; ++i) {
        if (data[i] != want[i]) {
          FPRINTF(stderr, "f64 n=%i differs at %i\n", n, i);
          result = EXIT_FAILURE;
        }
      }
      /* ctx is the source, so base is written and the source is not */
      for (i = 0; i < n; ++i) data[i] = 0;
      libxs_sort(data, n, sizeof(double), libxs_cmp_f64, src);
      /* against the oracle, because zeros left unwritten are ordered too */
      for (i = 0; i < n; ++i) {
        if (data[i] != want[i]) {
          FPRINTF(stderr, "f64 out-of-place n=%i differs at %i\n", n, i);
          result = EXIT_FAILURE;
        }
      }
      for (i = 0; i < n; ++i) {
        if (src[i] != value_of(i)) {
          FPRINTF(stderr, "f64 out-of-place n=%i wrote the source\n", n);
          result = EXIT_FAILURE;
        }
      }
      /* the indirect kind orders the indices and must leave the keys alone */
      libxs_sort(perm, n, sizeof(int), libxs_cmp_f64_idx, keys);
      for (i = 0; i < n; ++i) {
        if (keys[perm[i]] != want[i]) {
          FPRINTF(stderr, "f64_idx n=%i differs at %i\n", n, i);
          result = EXIT_FAILURE;
        }
      }
      for (i = 0; i < n; ++i) {
        if (keys[i] != value_of(i)) {
          FPRINTF(stderr, "f64_idx n=%i moved the keys\n", n);
          result = EXIT_FAILURE;
        }
      }
      /* every index once: a permutation, not merely something ordered */
      for (i = 0; i < n; ++i) {
        if (0 > perm[i] || n <= perm[i] || 0 != seen[perm[i]]) {
          FPRINTF(stderr, "f64_idx n=%i is not a permutation\n", n);
          result = EXIT_FAILURE;
        }
        else seen[perm[i]] = 1;
      }
      for (i = 0; i < n; ++i) data[i] = value_of(i);
      libxs_sort(data, n, sizeof(double), cmp_descend, NULL);
      for (i = 0; i < n; ++i) {
        if (data[i] != want[n - 1 - i]) {
          FPRINTF(stderr, "custom n=%i differs at %i\n", n, i);
          result = EXIT_FAILURE;
        }
      }
    }
    free(seen); free(perm); free(keys); free(src); free(want); free(data);
  }
  return result;
}
