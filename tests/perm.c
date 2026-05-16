/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_perm.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


static int check_sorted_f64(const double* data, int n) {
  int i;
  for (i = 1; i < n; ++i) {
    if (data[i - 1] > data[i]) return 0;
  }
  return 1;
}


static int check_sorted_f32(const float* data, int n) {
  int i;
  for (i = 1; i < n; ++i) {
    if (data[i - 1] > data[i]) return 0;
  }
  return 1;
}


static int check_sorted_i32(const int* data, int n) {
  int i;
  for (i = 1; i < n; ++i) {
    if (data[i - 1] > data[i]) return 0;
  }
  return 1;
}


static int cmp_by_abs(const void* a, const void* b, void* ctx) {
  const double va = *(const double*)a, vb = *(const double*)b;
  const double aa = (va < 0 ? -va : va), ab = (vb < 0 ? -vb : vb);
  (void)ctx;
  return (aa > ab) - (aa < ab);
}


static int cmp_indirect_f64(const void* a, const void* b, void* ctx) {
  const double* keys = (const double*)ctx;
  const int ia = *(const int*)a, ib = *(const int*)b;
  return (keys[ia] > keys[ib]) - (keys[ia] < keys[ib]);
}


int main(void)
{
  /* direct in-place sort of doubles */
  { double data[] = {5.0, -1.0, 3.0, 0.0, 2.0, -4.0, 1.0};
    libxs_sort(data, 7, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 7)) {
      FPRINTF(stderr, "ERROR line #%i: f64 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* out-of-place sort of doubles (ctx = source) */
  { const double src[] = {9.0, 1.0, 5.0, 3.0, 7.0};
    double dst[5];
    libxs_sort(dst, 5, sizeof(double), libxs_cmp_f64, (void*)(uintptr_t)src);
    if (!check_sorted_f64(dst, 5)) {
      FPRINTF(stderr, "ERROR line #%i: f64 out-of-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (9.0 != src[0] || 1.0 != src[1]) {
      FPRINTF(stderr, "ERROR line #%i: f64 source modified\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* f32 in-place */
  { float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    libxs_sort(data, 8, sizeof(float), libxs_cmp_f32, NULL);
    if (!check_sorted_f32(data, 8)) {
      FPRINTF(stderr, "ERROR line #%i: f32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* i32 in-place */
  { int data[] = {42, -7, 0, 100, -100, 3, 3};
    libxs_sort(data, 7, sizeof(int), libxs_cmp_i32, NULL);
    if (!check_sorted_i32(data, 7)) {
      FPRINTF(stderr, "ERROR line #%i: i32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* u32 in-place */
  { unsigned int data[] = {300, 100, 200, 0, 400};
    libxs_sort(data, 5, sizeof(unsigned int), libxs_cmp_u32, NULL);
    if (data[0] != 0 || data[1] != 100 || data[2] != 200
      || data[3] != 300 || data[4] != 400)
    {
      FPRINTF(stderr, "ERROR line #%i: u32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* custom comparator: sort by absolute value */
  { double data[] = {-5.0, 1.0, -3.0, 2.0, -4.0};
    libxs_sort(data, 5, sizeof(double), cmp_by_abs, NULL);
    if (!(1.0 == data[0] && 2.0 == data[1] && -3.0 == data[2]
      && -4.0 == data[3] && -5.0 == data[4]))
    {
      FPRINTF(stderr, "ERROR line #%i: custom comparator\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* indirect sort (argsort): sort index array by key values */
  { const double keys[] = {3.0, 1.0, 4.0, 1.5, 2.0};
    int perm[] = {0, 1, 2, 3, 4};
    libxs_sort(perm, 5, sizeof(int), cmp_indirect_f64, (void*)(uintptr_t)keys);
    if (perm[0] != 1 || perm[1] != 3 || perm[2] != 4
      || perm[3] != 0 || perm[4] != 2)
    {
      FPRINTF(stderr, "ERROR line #%i: indirect sort\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* single element */
  { double data[] = {42.0};
    libxs_sort(data, 1, sizeof(double), libxs_cmp_f64, NULL);
    if (42.0 != data[0]) {
      FPRINTF(stderr, "ERROR line #%i: single element\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* already sorted */
  { double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: already sorted\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* reverse sorted */
  { double data[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: reverse sorted\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* duplicates */
  { double data[] = {2.0, 2.0, 1.0, 1.0, 3.0, 3.0};
    libxs_sort(data, 6, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 6)) {
      FPRINTF(stderr, "ERROR line #%i: duplicates\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* negative values */
  { double data[] = {-1.0, -5.0, -2.0, -4.0, -3.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: negatives\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* hilbert (ndims=2): verify locality on 4x4 grid */
  { const unsigned int n = 4;
    uint64_t codes[16];
    unsigned int order[16];
    unsigned int coords[2], x, y, idx;
    for (y = 0; y < n; ++y) {
      for (x = 0; x < n; ++x) {
        coords[0] = x; coords[1] = y;
        codes[y * n + x] = libxs_hilbert(coords, 2);
      }
    }
    /* sort by code to get curve order */
    for (idx = 0; idx < n * n; ++idx) order[idx] = idx;
    { unsigned int i, j;
      for (i = 0; i < n * n - 1; ++i) {
        for (j = i + 1; j < n * n; ++j) {
          if (codes[order[j]] < codes[order[i]]) {
            unsigned int t = order[i]; order[i] = order[j]; order[j] = t;
          }
        }
      }
    }
    /* check all distinct */
    { unsigned int i;
      for (i = 1; i < n * n; ++i) {
        if (codes[order[i]] == codes[order[i - 1]]) {
          FPRINTF(stderr, "ERROR line #%i: hilbert 2D not bijective\n", __LINE__);
          exit(EXIT_FAILURE);
        }
      }
    }
    /* check locality: consecutive curve positions are Manhattan-adjacent */
    { unsigned int i;
      for (i = 1; i < n * n; ++i) {
        const unsigned int px = order[i - 1] % n, py = order[i - 1] / n;
        const unsigned int cx = order[i] % n, cy = order[i] / n;
        const unsigned int dx = (cx > px) ? cx - px : px - cx;
        const unsigned int dy = (cy > py) ? cy - py : py - cy;
        if (dx + dy != 1) {
          FPRINTF(stderr, "ERROR line #%i: hilbert 2D locality "
            "i=%u dx=%u dy=%u\n", __LINE__, i, dx, dy);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  /* hilbert (ndims=3): verify bijectivity on 4x4x4 grid */
  { const unsigned int n = 4;
    const unsigned int total = n * n * n;
    uint64_t codes[64];
    unsigned int x, y, z, i, j, collisions = 0;
    for (z = 0; z < n; ++z) {
      for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
          unsigned int coords[3];
          coords[0] = x; coords[1] = y; coords[2] = z;
          codes[z * n * n + y * n + x] = libxs_hilbert(coords, 3);
        }
      }
    }
    for (i = 0; i < total; ++i) {
      for (j = i + 1; j < total; ++j) {
        if (codes[i] == codes[j]) ++collisions;
      }
    }
    if (0 != collisions) {
      FPRINTF(stderr, "ERROR line #%i: hilbert 3D collisions=%u\n",
        __LINE__, collisions);
      exit(EXIT_FAILURE);
    }
  }

  /* kdtree2d: basic nearest neighbor */
  { double pts[] = {0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0};
    int idx[] = {0, 1, 2, 3};
    int hit;
    libxs_kdtree2d_build(pts, idx, 4);
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 4, 0.1, 0.1, 1.0);
    if (hit != 0) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d nearest=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 4, 0.9, 0.9, 1.0);
    if (hit != 3) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d nearest=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
  }
  /* kdtree2d: used-flag consumption */
  { double pts[] = {0.0,0.0, 0.1,0.1, 5.0,5.0};
    int idx[] = {0, 1, 2};
    unsigned char used[] = {0, 0, 0};
    int h1, h2;
    libxs_kdtree2d_build(pts, idx, 3);
    h1 = libxs_kdtree2d_nearest(pts, idx, used, 3, 0.0, 0.0, 1.0);
    if (h1 != 0) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d used h1=%d\n", __LINE__, h1);
      exit(EXIT_FAILURE);
    }
    used[h1] = 1;
    h2 = libxs_kdtree2d_nearest(pts, idx, used, 3, 0.0, 0.0, 1.0);
    if (h2 != 1) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d used h2=%d\n", __LINE__, h2);
      exit(EXIT_FAILURE);
    }
  }
  /* kdtree2d: no match within radius */
  { double pts[] = {10.0,10.0, 20.0,20.0};
    int idx[] = {0, 1};
    int hit;
    libxs_kdtree2d_build(pts, idx, 2);
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 2, 0.0, 0.0, 1.0);
    if (hit != -1) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d no-match=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
