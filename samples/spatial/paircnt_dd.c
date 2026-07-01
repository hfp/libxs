/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_perm.h>
#include <libxs/libxs_timer.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

#define PAIRCNT_NDIMS 3
#define PAIRCNT_NBINS_MAX 128
#define PAIRCNT_LEAF_MAX 256


typedef struct paircnt_catalog_t {
  double* pts;
  int n;
  double boxsize;
} paircnt_catalog_t;

typedef struct paircnt_bins_t {
  double edges[PAIRCNT_NBINS_MAX + 1];
  double edges2[PAIRCNT_NBINS_MAX + 1];
  int nbins;
} paircnt_bins_t;

typedef struct paircnt_node_t {
  double lo[PAIRCNT_NDIMS];
  double hi[PAIRCNT_NDIMS];
  int begin;
  int end;
  int left;
  int right;
} paircnt_node_t;

typedef struct paircnt_tree_t {
  paircnt_node_t* nodes;
  int nnodes;
  int capacity;
  const double* pts;
  const int* idx;
  int n;
} paircnt_tree_t;


static int read_catalog_text(const char* path, paircnt_catalog_t* cat);
static int read_catalog_ff(const char* path, paircnt_catalog_t* cat);
static int read_catalog(const char* path, paircnt_catalog_t* cat);
static int make_bins_log(paircnt_bins_t* bins, double rmin,
  double rmax, int nbins);
static int tree_build(paircnt_tree_t* tree, const double* pts,
  const int* idx, int n, int min_leaf);
static double min_dist2_between_boxes(
  const double* lo_a, const double* hi_a,
  const double* lo_b, const double* hi_b);
static double max_dist2_between_boxes(
  const double* lo_a, const double* hi_a,
  const double* lo_b, const double* hi_b);
static void count_pairs_leaf(const double* pts, const int* idx,
  int begin_a, int end_a, int begin_b, int end_b,
  const paircnt_bins_t* bins, long long* counts);
static void count_pairs_leaf_self(const double* pts,
  const int* idx, int begin, int end,
  const paircnt_bins_t* bins, long long* counts);
static void count_pairs_dual(const paircnt_tree_t* tree,
  int node_a, int node_b,
  const paircnt_bins_t* bins, long long* counts);


int main(int argc, char* argv[])
{
  int result = EXIT_FAILURE;
  int argi = 1, nbins = 20, leaf = 64, i;
  double rmin = 0.1, rmax = 25.0, boxsize = 0.0;
  const char* catalog_path = NULL;
  paircnt_catalog_t cat;
  paircnt_bins_t bins;
  paircnt_tree_t tree;
  long long* counts = NULL;
  int* idx = NULL;
  libxs_timer_tick_t tick;
  double t_build, t_count;

  memset(&cat, 0, sizeof(cat));
  memset(&bins, 0, sizeof(bins));
  memset(&tree, 0, sizeof(tree));

  while (argi < argc) {
    if (0 == strcmp(argv[argi], "--rmin") && argi + 1 < argc) {
      rmin = atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "--rmax") && argi + 1 < argc) {
      rmax = atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "--nbins") && argi + 1 < argc) {
      nbins = atoi(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "--boxsize") && argi + 1 < argc) {
      boxsize = atof(argv[++argi]);
    }
    else if (0 == strcmp(argv[argi], "--leaf") && argi + 1 < argc) {
      leaf = atoi(argv[++argi]);
    }
    else if (NULL == catalog_path) {
      catalog_path = argv[argi];
    }
    ++argi;
  }

  if (NULL == catalog_path) {
    fprintf(stderr,
      "usage: paircnt_dd <catalog> [--rmin R] [--rmax R] "
      "[--nbins N] [--boxsize L]\n");
    result = EXIT_FAILURE;
  }
  else {
    result = EXIT_SUCCESS;
  }
  if (EXIT_SUCCESS == result) {
    result = read_catalog(catalog_path, &cat);
    if (EXIT_SUCCESS == result) {
      cat.boxsize = boxsize;
      fprintf(stderr, "catalog: %d points from %s\n", cat.n, catalog_path);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = make_bins_log(&bins, rmin, rmax, nbins);
  }
  if (EXIT_SUCCESS == result) {
    idx = (int*)malloc((size_t)cat.n * sizeof(int));
    counts = (long long*)calloc((size_t)nbins, sizeof(long long));
    if (NULL == idx || NULL == counts) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    libxs_kdtree_config_t cfg;
    for (i = 0; i < cat.n; ++i) idx[i] = i;
    cfg.min_leaf = leaf;
    cfg.split = NULL;
    cfg.ctx = NULL;
    tick = libxs_timer_tick();
    libxs_kdtree_build(cat.pts, idx, cat.n, PAIRCNT_NDIMS,
      PAIRCNT_NDIMS, &cfg);
    t_build = libxs_timer_duration(tick, libxs_timer_tick());
    fprintf(stderr, "kdtree: %.3f ms (%d points, leaf=%d)\n",
      t_build * 1e3, cat.n, cfg.min_leaf);
  }
  if (EXIT_SUCCESS == result) {
    result = tree_build(&tree, cat.pts, idx, cat.n, leaf);
  }
  if (EXIT_SUCCESS == result) {
    tick = libxs_timer_tick();
    count_pairs_dual(&tree, 0, 0, &bins, counts);
    t_count = libxs_timer_duration(tick, libxs_timer_tick());
    fprintf(stderr, "paircnt: %.3f ms\n", t_count * 1e3);
  }
  if (EXIT_SUCCESS == result) {
    long long total = 0;
    printf("# rmin rmax npairs\n");
    for (i = 0; i < nbins; ++i) {
      long long npairs = 2 * counts[i];
      printf("%.6e %.6e %lld\n", bins.edges[i], bins.edges[i + 1], npairs);
      total += npairs;
    }
    fprintf(stderr, "total pairs: %lld\n", total);
  }

  free(tree.nodes);
  free(counts);
  free(idx);
  free(cat.pts);
  return result;
}


static int read_catalog_text(const char* path, paircnt_catalog_t* cat)
{
  int result = EXIT_SUCCESS, capacity = 4096, n = 0;
  double* pts = NULL;
  FILE* fp = NULL;
  assert(NULL != path && NULL != cat);
  fp = fopen(path, "r");
  if (NULL == fp) {
    fprintf(stderr, "error: cannot open %s\n", path);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    pts = (double*)malloc(capacity * PAIRCNT_NDIMS * sizeof(double));
    if (NULL == pts) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    char line[256];
    while (NULL != fgets(line, sizeof(line), fp)) {
      double x, y, z;
      if ('#' == line[0] || '\n' == line[0]) continue;
      if (3 != sscanf(line, "%lf %lf %lf", &x, &y, &z)) continue;
      if (n >= capacity) {
        capacity *= 2;
        pts = (double*)realloc(pts, capacity * PAIRCNT_NDIMS * sizeof(double));
        if (NULL == pts) { result = EXIT_FAILURE; break; }
      }
      pts[n * PAIRCNT_NDIMS + 0] = x;
      pts[n * PAIRCNT_NDIMS + 1] = y;
      pts[n * PAIRCNT_NDIMS + 2] = z;
      ++n;
    }
  }
  if (EXIT_SUCCESS == result) {
    cat->pts = pts;
    cat->n = n;
  }
  else {
    free(pts);
  }
  if (NULL != fp) fclose(fp);
  return result;
}


static int read_catalog_ff(const char* path, paircnt_catalog_t* cat)
{
  int result = EXIT_SUCCESS, np = 0, idat[5];
  int marker0 = 0, marker1 = 0, elem_bytes = 0;
  double* pts = NULL;
  FILE* fp = NULL;
  assert(NULL != path && NULL != cat);
  fp = fopen(path, "rb");
  if (NULL == fp) {
    fprintf(stderr, "error: cannot open %s\n", path);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (1 != fread(&marker0, 4, 1, fp)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (marker0 != (int)sizeof(idat)) {
      fprintf(stderr, "error: unexpected idat record size %d\n", marker0);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    if (5 != fread(idat, sizeof(int), 5, fp)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (1 != fread(&marker1, 4, 1, fp)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (marker0 != marker1) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    long skip;
    np = idat[1];
    if (np <= 0) result = EXIT_FAILURE;
    skip = (4 + 9 * (long)sizeof(float) + 4) +
           (4 + (long)sizeof(float) + 4);
    if (EXIT_SUCCESS == result && 0 != fseek(fp, skip, SEEK_CUR)) {
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    if (1 != fread(&marker0, 4, 1, fp)) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      elem_bytes = marker0 / np;
      if (4 != elem_bytes && 8 != elem_bytes) {
        fprintf(stderr, "error: unexpected element size %d\n", elem_bytes);
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result && 0 != fseek(fp, -4, SEEK_CUR)) {
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    pts = (double*)malloc((size_t)np * PAIRCNT_NDIMS * sizeof(double));
    if (NULL == pts) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    int d;
    for (d = 0; d < PAIRCNT_NDIMS && EXIT_SUCCESS == result; ++d) {
      if (1 != fread(&marker0, 4, 1, fp)) { result = EXIT_FAILURE; break; }
      if (8 == elem_bytes) {
        int i;
        double* col = (double*)malloc((size_t)np * sizeof(double));
        if (NULL == col) { result = EXIT_FAILURE; break; }
        if ((size_t)np != fread(col, sizeof(double), (size_t)np, fp)) {
          free(col); result = EXIT_FAILURE; break;
        }
        for (i = 0; i < np; ++i) pts[i * PAIRCNT_NDIMS + d] = col[i];
        free(col);
      }
      else {
        int i;
        float* col = (float*)malloc((size_t)np * sizeof(float));
        if (NULL == col) { result = EXIT_FAILURE; break; }
        if ((size_t)np != fread(col, sizeof(float), (size_t)np, fp)) {
          free(col); result = EXIT_FAILURE; break;
        }
        for (i = 0; i < np; ++i) pts[i * PAIRCNT_NDIMS + d] = (double)col[i];
        free(col);
      }
      if (1 != fread(&marker1, 4, 1, fp)) { result = EXIT_FAILURE; break; }
      if (marker0 != marker1) { result = EXIT_FAILURE; break; }
    }
  }
  if (EXIT_SUCCESS == result) {
    cat->pts = pts;
    cat->n = np;
  }
  else {
    free(pts);
  }
  if (NULL != fp) fclose(fp);
  return result;
}


static int read_catalog(const char* path, paircnt_catalog_t* cat)
{
  int result;
  const char* ext;
  assert(NULL != path && NULL != cat);
  cat->pts = NULL;
  cat->n = 0;
  cat->boxsize = 0.0;
  ext = strrchr(path, '.');
  if (NULL != ext && 0 == strcmp(ext, ".ff")) {
    result = read_catalog_ff(path, cat);
  }
  else {
    result = read_catalog_text(path, cat);
  }
  return result;
}


static int make_bins_log(paircnt_bins_t* bins, double rmin, double rmax,
  int nbins)
{
  int result = EXIT_SUCCESS, i;
  double logrmin, logstep;
  assert(NULL != bins);
  if (nbins <= 0 || nbins > PAIRCNT_NBINS_MAX || rmin <= 0 || rmax <= rmin) {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    logrmin = log10(rmin);
    logstep = (log10(rmax) - logrmin) / nbins;
    for (i = 0; i <= nbins; ++i) {
      bins->edges[i] = pow(10.0, logrmin + i * logstep);
      bins->edges2[i] = bins->edges[i] * bins->edges[i];
    }
    bins->nbins = nbins;
  }
  return result;
}


static int tree_build(paircnt_tree_t* tree, const double* pts,
  const int* idx, int n, int min_leaf)
{
  int result = EXIT_SUCCESS, stack[128], sp = 0;
  assert(NULL != tree && NULL != pts && NULL != idx);
  tree->pts = pts;
  tree->idx = idx;
  tree->n = n;
  tree->nnodes = 0;
  tree->capacity = 256;
  tree->nodes = (paircnt_node_t*)malloc(
    (size_t)tree->capacity * sizeof(paircnt_node_t));
  if (NULL == tree->nodes) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    stack[sp++] = 0;
    stack[sp++] = n;
    stack[sp++] = 0;
  }
  while (sp > 0 && EXIT_SUCCESS == result) {
    int parent_slot = stack[--sp];
    int end = stack[--sp];
    int begin = stack[--sp];
    int count = end - begin;
    int mid = begin + count / 2;
    int node_id;
    paircnt_node_t* node;
    if (tree->nnodes >= tree->capacity) {
      int newcap = tree->capacity * 2;
      paircnt_node_t* tmp = (paircnt_node_t*)realloc(tree->nodes,
        (size_t)newcap * sizeof(paircnt_node_t));
      if (NULL == tmp) { result = EXIT_FAILURE; break; }
      tree->nodes = tmp;
      tree->capacity = newcap;
    }
    node_id = tree->nnodes++;
    node = &tree->nodes[node_id];
    node->begin = begin;
    node->end = end;
    node->left = -1;
    node->right = -1;
    if (parent_slot > 0) {
      int pslot = parent_slot - 1;
      if (tree->nodes[pslot / 2].left == -2) {
        tree->nodes[pslot / 2].left = node_id;
      }
      else {
        tree->nodes[pslot / 2].right = node_id;
      }
    }
    { /* compute bounding box */
      int i, d;
      for (d = 0; d < PAIRCNT_NDIMS; ++d) {
        node->lo[d] = DBL_MAX;
        node->hi[d] = -DBL_MAX;
      }
      for (i = begin; i < end; ++i) {
        int pi = idx[i];
        for (d = 0; d < PAIRCNT_NDIMS; ++d) {
          double v = pts[pi * PAIRCNT_NDIMS + d];
          if (v < node->lo[d]) node->lo[d] = v;
          if (v > node->hi[d]) node->hi[d] = v;
        }
      }
    }
    if (count > min_leaf) {
      node->left = -2;
      node->right = -2;
      if (sp + 6 > (int)(sizeof(stack) / sizeof(stack[0]))) {
        result = EXIT_FAILURE; break;
      }
      stack[sp++] = mid;
      stack[sp++] = end;
      stack[sp++] = node_id * 2 + 2;
      stack[sp++] = begin;
      stack[sp++] = mid;
      stack[sp++] = node_id * 2 + 1;
    }
  }
  return result;
}


static double min_dist2_between_boxes(
  const double* lo_a, const double* hi_a,
  const double* lo_b, const double* hi_b)
{
  double dist2 = 0.0;
  int d;
  for (d = 0; d < PAIRCNT_NDIMS; ++d) {
    double gap = 0.0;
    if (lo_a[d] > hi_b[d]) gap = lo_a[d] - hi_b[d];
    else if (lo_b[d] > hi_a[d]) gap = lo_b[d] - hi_a[d];
    dist2 += gap * gap;
  }
  return dist2;
}


static double max_dist2_between_boxes(
  const double* lo_a, const double* hi_a,
  const double* lo_b, const double* hi_b)
{
  double dist2 = 0.0;
  int d;
  for (d = 0; d < PAIRCNT_NDIMS; ++d) {
    double span = hi_a[d] - lo_b[d];
    double span2 = hi_b[d] - lo_a[d];
    if (span2 > span) span = span2;
    if (span < 0.0) span = 0.0;
    dist2 += span * span;
  }
  return dist2;
}


static void count_pairs_leaf(const double* pts, const int* idx,
  int begin_a, int end_a, int begin_b, int end_b,
  const paircnt_bins_t* bins, long long* counts)
{
  int ia, ib, nb = end_b - begin_b;
  const int nbins = bins->nbins;
  const double* edges2 = bins->edges2;
  double rmax2 = edges2[nbins];
  double rmin2 = edges2[0];
  double r2buf[PAIRCNT_LEAF_MAX];
  for (ia = begin_a; ia < end_a; ++ia) {
    int pi = idx[ia];
    double xi = pts[pi * PAIRCNT_NDIMS + 0];
    double yi = pts[pi * PAIRCNT_NDIMS + 1];
    double zi = pts[pi * PAIRCNT_NDIMS + 2];
    LIBXS_PRAGMA_SIMD
    for (ib = 0; ib < nb; ++ib) {
      int pj = idx[begin_b + ib];
      double dx = xi - pts[pj * PAIRCNT_NDIMS + 0];
      double dy = yi - pts[pj * PAIRCNT_NDIMS + 1];
      double dz = zi - pts[pj * PAIRCNT_NDIMS + 2];
      r2buf[ib] = dx * dx + dy * dy + dz * dz;
    }
    for (ib = 0; ib < nb; ++ib) {
      double r2 = r2buf[ib];
      if (r2 >= rmin2 && r2 < rmax2) {
        int k = nbins - 1;
        while (k > 0 && r2 < edges2[k]) --k;
        ++counts[k];
      }
    }
  }
}


static void count_pairs_leaf_self(const double* pts, const int* idx,
  int begin, int end,
  const paircnt_bins_t* bins, long long* counts)
{
  int ia, ib, nb;
  const int nbins = bins->nbins;
  const double* edges2 = bins->edges2;
  double rmax2 = edges2[nbins];
  double rmin2 = edges2[0];
  double r2buf[PAIRCNT_LEAF_MAX];
  for (ia = begin; ia < end; ++ia) {
    int pi = idx[ia];
    double xi = pts[pi * PAIRCNT_NDIMS + 0];
    double yi = pts[pi * PAIRCNT_NDIMS + 1];
    double zi = pts[pi * PAIRCNT_NDIMS + 2];
    nb = end - (ia + 1);
    LIBXS_PRAGMA_SIMD
    for (ib = 0; ib < nb; ++ib) {
      int pj = idx[ia + 1 + ib];
      double dx = xi - pts[pj * PAIRCNT_NDIMS + 0];
      double dy = yi - pts[pj * PAIRCNT_NDIMS + 1];
      double dz = zi - pts[pj * PAIRCNT_NDIMS + 2];
      r2buf[ib] = dx * dx + dy * dy + dz * dz;
    }
    for (ib = 0; ib < nb; ++ib) {
      double r2 = r2buf[ib];
      if (r2 >= rmin2 && r2 < rmax2) {
        int k = nbins - 1;
        while (k > 0 && r2 < edges2[k]) --k;
        ++counts[k];
      }
    }
  }
}


static void count_pairs_dual(const paircnt_tree_t* tree,
  int node_a, int node_b,
  const paircnt_bins_t* bins, long long* counts)
{
  const paircnt_node_t* a = &tree->nodes[node_a];
  const paircnt_node_t* b = &tree->nodes[node_b];
  double mindist2 = min_dist2_between_boxes(a->lo, a->hi, b->lo, b->hi);
  double maxdist2;
  if (mindist2 >= bins->edges2[bins->nbins]) return;
  maxdist2 = max_dist2_between_boxes(a->lo, a->hi, b->lo, b->hi);
  if (maxdist2 < bins->edges2[0]) return;
  if (-1 == a->left && -1 == b->left) {
    if (node_a == node_b) {
      count_pairs_leaf_self(tree->pts, tree->idx,
        a->begin, a->end, bins, counts);
    }
    else {
      count_pairs_leaf(tree->pts, tree->idx,
        a->begin, a->end, b->begin, b->end, bins, counts);
    }
    return;
  }
  if (node_a == node_b) {
    count_pairs_dual(tree, a->left, a->left, bins, counts);
    count_pairs_dual(tree, a->left, a->right, bins, counts);
    count_pairs_dual(tree, a->right, a->right, bins, counts);
  }
  else if (-1 != a->left && (-1 == b->left
    || (a->end - a->begin) >= (b->end - b->begin)))
  {
    count_pairs_dual(tree, a->left, node_b, bins, counts);
    count_pairs_dual(tree, a->right, node_b, bins, counts);
  }
  else {
    count_pairs_dual(tree, node_a, b->left, bins, counts);
    count_pairs_dual(tree, node_a, b->right, bins, counts);
  }
}
