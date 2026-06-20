/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/* Rosetta: recovering structure from opaque bytes.
 *
 * Encodes a table of harmonic-series data as a flat binary blob,
 * then applies the LIBXS hierarchical analysis to rediscover the
 * element type, record stride, and structural content -- starting
 * from the raw bytes with no metadata.
 *
 * The flat 1-D probe deliberately fails (interleaved fields of
 * different scales look noisy), demonstrating why the hierarchical
 * approach is needed. The stride sweep then discovers the record
 * boundary by testing per-column decay at each candidate stride.
 *
 * Data: for n = 1..NROWS, six f64 fields per record:
 *   [0] n          index (integer stored as double)
 *   [1] 1/n        harmonic term
 *   [2] H_n        partial sum of the harmonic series
 *   [3] ln(n)      natural logarithm
 *   [4] H_n-ln(n)  difference (converges to Euler-Mascheroni gamma)
 *   [5] gamma_n    running estimate: H_n - ln(n) - 1/(2n)
 */
#include <libxs/libxs_math.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_perm.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NROWS   64
#define NFIELDS 6
#define ROSETTA_PATH_MAX 1024
#define EULER_MASCHERONI 0.5772156649015329

typedef struct field_image_context_t {
  const double* values;
  double vmin[NFIELDS];
  double vmax[NFIELDS];
  size_t nrows;
  int stride;
} field_image_context_t;

static FILE* artifact_file = NULL;
static char artifact_dir[ROSETTA_PATH_MAX] = { 0 };

static const char* field_names[NFIELDS] = {
  "n", "1/n", "H_n", "ln(n)", "H_n-ln(n)", "gamma_est"
};


static void separator(void) {
  printf("--------------------------------------------------\n");
}


static int artifact_path(char path[ROSETTA_PATH_MAX], const char name[]);
static void artifact_open(void);
static void artifact_close(void);
static void artifact_write_matrix(const char name[], const void* data,
  libxs_data_t dtype, size_t nrows, int stride);
static void artifact_write_field_image(const char name[], const void* data,
  libxs_data_t dtype, size_t nrows, int stride);
static int field_image_handler(void* dst,
  const libxs_mhd_element_handler_info_t* dst_info, libxs_data_t src_type,
  const void* src, const void* src_min, const void* src_max,
  size_t index, void* context);
static void generate_table(double table[NROWS][NFIELDS]);
/* Level 0: flat 1-D probe -- expected to be inconclusive. */
static libxs_data_t flat_probe(const void* blob, size_t nbytes);
/* Stride sweep: for each candidate element width and record stride,
 * fingerprint down the columns and find the (width, stride) pair
 * with the smallest per-column decay. */
static int stride_sweep(const void* blob, size_t nbytes,
  libxs_data_t* out_type);
/* Shuffle stability on the discovered column structure. */
static void shuffle_test(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride);
/* Per-field decay analysis -- the payoff. */
static void field_analysis(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride);
/* GREEDY sort test. */
static void sort_test(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride);
/* Setdiff verification: compare blob against original. */
static void verify(const void* table,
  const void* blob, size_t nbytes, libxs_data_t dtype);


int main(void)
{
  double table[NROWS][NFIELDS];
  const size_t nbytes = sizeof(table);
  void* blob;
  libxs_data_t discovered;
  int stride;

  generate_table(table);
  blob = malloc(nbytes);
  if (NULL == blob) return EXIT_FAILURE;
  memcpy(blob, table, nbytes);
  artifact_open();

  printf("ROSETTA: Recovering structure from opaque bytes\n");
  separator();
  printf("Ground truth (hidden from the analysis):\n");
  printf("  %i records x %i fields = %i doubles (%i bytes)\n",
    NROWS, NFIELDS, NROWS * NFIELDS, (int)nbytes);
  printf("  Data: harmonic series H_n and derived quantities.\n");
  printf("  Field [5] converges to Euler-Mascheroni gamma.\n");
  separator();

  flat_probe(blob, nbytes);
  separator();

  stride = stride_sweep(blob, nbytes, &discovered);
  artifact_write_matrix("rosetta_original.mhd", blob, discovered,
    (size_t)(nbytes / (LIBXS_TYPESIZE(discovered) * stride)), stride);
  artifact_write_field_image("rosetta_fields.mhd", blob, discovered,
    (size_t)(nbytes / (LIBXS_TYPESIZE(discovered) * stride)), stride);
  separator();

  shuffle_test(blob, nbytes, discovered, stride);
  separator();

  field_analysis(blob, nbytes, discovered, stride);
  separator();

  sort_test(blob, nbytes, discovered, stride);
  separator();

  verify(table, blob, nbytes, discovered);
  separator();

  artifact_close();
  free(blob);
  return EXIT_SUCCESS;
}


static int artifact_path(char path[ROSETTA_PATH_MAX], const char name[])
{
  int result = EXIT_FAILURE;
  if (0 != artifact_dir[0] && NULL != name) {
    const size_t dlen = strlen(artifact_dir);
    const size_t nlen = strlen(name);
    if (dlen + nlen + 2 < ROSETTA_PATH_MAX) {
      strcpy(path, artifact_dir);
      if ('/' != path[dlen - 1]) strcat(path, "/");
      strcat(path, name);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static void artifact_open(void)
{
  const char* const outdir = getenv("ROSETTA_OUTDIR");
  char path[ROSETTA_PATH_MAX];
  if (NULL != outdir && 0 != *outdir && strlen(outdir) < sizeof(artifact_dir)) {
    strcpy(artifact_dir, outdir);
    if (EXIT_SUCCESS == artifact_path(path, "summary.csv")) {
      artifact_file = fopen(path, "w");
      if (NULL != artifact_file) {
        fprintf(artifact_file, "stage,index,name,value\n");
      }
    }
  }
}


static void artifact_close(void)
{
  if (NULL != artifact_file) {
    fclose(artifact_file);
    artifact_file = NULL;
  }
}


static void artifact_write_matrix(const char name[], const void* data,
  libxs_data_t dtype, size_t nrows, int stride)
{
  char path[ROSETTA_PATH_MAX];
  if (NULL != data && 0 < nrows && 0 < stride
    && EXIT_SUCCESS == artifact_path(path, name))
  {
    size_t size[2];
    libxs_mhd_info_t info;
    size[0] = (size_t)stride;
    size[1] = nrows;
    info.ndims = 2;
    info.ncomponents = 1;
    info.type = dtype;
    info.header_size = 0;
    libxs_mhd_write(path, NULL, size, NULL, &info, data, NULL);
  }
}


static void artifact_write_field_image(const char name[], const void* data,
  libxs_data_t dtype, size_t nrows, int stride)
{
  char path[ROSETTA_PATH_MAX];
  int result = EXIT_FAILURE;
  if (LIBXS_DATATYPE_F64 == dtype && NULL != data && 0 < nrows
    && 0 < stride && stride <= NFIELDS
    && EXIT_SUCCESS == artifact_path(path, name))
  {
    field_image_context_t context;
    libxs_mhd_element_handler_info_t handler_info;
    libxs_mhd_write_info_t write_info;
    size_t size[2];
    libxs_mhd_info_t info;
    int f;
    context.values = (const double*)data;
    context.nrows = nrows;
    context.stride = stride;
    for (f = 0; f < stride; ++f) {
      size_t r;
      context.vmin[f] = context.values[f];
      context.vmax[f] = context.values[f];
      for (r = 1; r < nrows; ++r) {
        const double value = context.values[r * (size_t)stride + (size_t)f];
        if (value < context.vmin[f]) context.vmin[f] = value;
        if (context.vmax[f] < value) context.vmax[f] = value;
      }
    }
    handler_info.type = LIBXS_DATATYPE_U8;
    handler_info.hint = LIBXS_MHD_ELEMENT_CONVERSION_DEFAULT;
    memset(&write_info, 0, sizeof(write_info));
    write_info.handler_info = &handler_info;
    write_info.handler = field_image_handler;
    write_info.handler_context = &context;
    size[0] = nrows;
    size[1] = (size_t)stride;
    info.ndims = 2;
    info.ncomponents = 1;
    info.type = dtype;
    info.header_size = 0;
    result = libxs_mhd_write(path, NULL, size, NULL, &info, data, &write_info);
  }
  if (EXIT_SUCCESS != result && NULL != artifact_file) {
    fprintf(artifact_file, "artifact,,%s,not_written\n", name);
  }
}


static int field_image_handler(void* dst,
  const libxs_mhd_element_handler_info_t* dst_info, libxs_data_t src_type,
  const void* src, const void* src_min, const void* src_max,
  size_t index, void* context)
{
  const field_image_context_t* const info = (const field_image_context_t*)context;
  int result = EXIT_FAILURE;
  LIBXS_UNUSED(dst_info); LIBXS_UNUSED(src_type); LIBXS_UNUSED(src);
  LIBXS_UNUSED(src_min); LIBXS_UNUSED(src_max);
  if (NULL != dst && NULL != info && 0 < info->nrows) {
    const size_t row = index % info->nrows;
    const int field = (int)(index / info->nrows);
    unsigned char pixel = 0;
    if (0 <= field && field < info->stride) {
      const double vmin = info->vmin[field], vmax = info->vmax[field];
      if (vmin < vmax) {
        const double value = info->values[row * (size_t)info->stride + (size_t)field];
        const double scaled = 255.0 * (value - vmin) / (vmax - vmin);
        pixel = (unsigned char)(scaled + 0.5);
      }
      *(unsigned char*)dst = pixel;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static void generate_table(double table[NROWS][NFIELDS])
{
  double hn = 0;
  int i;
  for (i = 0; i < NROWS; ++i) {
    const double n = (double)(i + 1);
    hn += 1.0 / n;
    table[i][0] = n;
    table[i][1] = 1.0 / n;
    table[i][2] = hn;
    table[i][3] = log(n);
    table[i][4] = hn - log(n);
    table[i][5] = hn - log(n) - 1.0 / (2.0 * n);
  }
}


/* Level 0: flat 1-D probe -- expected to be inconclusive. */
static libxs_data_t flat_probe(const void* blob, size_t nbytes)
{
  libxs_fprint_t fp;
  int r;
  double decay = -1;
  printf("Level 0: Flat 1-D probe (all bytes as one sequence)\n");
  printf("  Input: %i opaque bytes\n", (int)nbytes);
  r = libxs_fprint(&fp, LIBXS_DATATYPE_UNKNOWN, blob,
    1, &nbytes, NULL, 6, 0, 0, 0);
  if (EXIT_SUCCESS == r) decay = libxs_fprint_decay(&fp);
  if (NULL != artifact_file) {
    fprintf(artifact_file, "flat,,success,%i\n", EXIT_SUCCESS == r ? 1 : 0);
    fprintf(artifact_file, "flat,,decay,%.17g\n", decay);
  }
  if (EXIT_SUCCESS == r && decay < 1.0) {
    printf("  Discovered: %s, decay=%.6f\n",
      libxs_typename(fp.datatype), decay);
    if (NULL != artifact_file) {
      fprintf(artifact_file, "flat,,type,%s\n", libxs_typename(fp.datatype));
      fprintf(artifact_file, "flat,,accepted,1\n");
    }
    return fp.datatype;
  }
  printf("  No type has decay < 1 -- flat stream is not smooth.\n");
  printf("  This is expected: interleaved fields of different\n");
  printf("  scales look like noise when read sequentially.\n");
  if (NULL != artifact_file) {
    fprintf(artifact_file, "flat,,type,unknown\n");
    fprintf(artifact_file, "flat,,accepted,0\n");
  }
  return LIBXS_DATATYPE_UNKNOWN;
}


/* Stride sweep: for each candidate element width and record stride,
 * fingerprint down the columns and find the (width, stride) pair
 * with the smallest per-column decay. */
static int stride_sweep(const void* blob, size_t nbytes,
  libxs_data_t* out_type)
{
  static const libxs_data_t types[] = {
    LIBXS_DATATYPE_F64, LIBXS_DATATYPE_F32,
    LIBXS_DATATYPE_I32, LIBXS_DATATYPE_I16,
    LIBXS_DATATYPE_I8
  };
  const int ntypes = (int)(sizeof(types) / sizeof(*types));
  double best_decay = 1e30;
  libxs_data_t best_type = LIBXS_DATATYPE_F64;
  int best_stride = 1, ti, s;

  printf("Stride sweep: discovering record layout\n");
  for (ti = 0; ti < ntypes; ++ti) {
    const size_t tw = LIBXS_TYPESIZE((int)types[ti]);
    const size_t ne = nbytes / tw;
    const int max_stride = LIBXS_MIN((int)ne / 4, 64);
    if (0 != nbytes % tw || ne < 8) continue;
    for (s = 2; s <= max_stride; ++s) {
      const size_t nrows = ne / s;
      double col_decay_sum = 0;
      int col_count = 0, f;
      if (0 != ne % s || nrows < 4) continue;
      for (f = 0; f < s; ++f) {
        const size_t stride_sz = (size_t)s;
        const void* col = (const char*)blob + f * tw;
        libxs_fprint_t fp;
        if (EXIT_SUCCESS == libxs_fprint(&fp, types[ti], col,
          1, &nrows, &stride_sz, 4, 0, 0, 0))
        {
          const double d = libxs_fprint_decay(&fp);
          if (d < 1.0 && d == d) {
            col_decay_sum += d;
            ++col_count;
          }
        }
      }
      if (col_count == s) {
        const double avg = col_decay_sum / s;
        if (avg < best_decay) {
          best_decay = avg;
          best_stride = s;
          best_type = types[ti];
        }
      }
    }
  }
  printf("  Best type:   %s\n", libxs_typename(best_type));
  printf("  Best stride: %i elements (%i bytes per record)\n",
    best_stride, (int)(best_stride * LIBXS_TYPESIZE(best_type)));
  printf("  Records:     %i\n",
    (int)(nbytes / (LIBXS_TYPESIZE(best_type) * best_stride)));
  printf("  Avg decay:   %.6f\n", best_decay);
  if (NULL != artifact_file) {
    fprintf(artifact_file, "stride,,type,%s\n", libxs_typename(best_type));
    fprintf(artifact_file, "stride,,elements,%i\n", best_stride);
    fprintf(artifact_file, "stride,,records,%i\n",
      (int)(nbytes / (LIBXS_TYPESIZE(best_type) * best_stride)));
    fprintf(artifact_file, "stride,,avg_decay,%.17g\n", best_decay);
  }
  *out_type = best_type;
  return best_stride;
}


/* Shuffle stability on the discovered column structure. */
static void shuffle_test(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride)
{
  const size_t tw = LIBXS_TYPESIZE(dtype);
  const size_t ne = nbytes / tw;
  const size_t nrows = ne / stride;
  const size_t row_bytes = stride * tw;
  void* shuffled;
  printf("Shuffle stability (record-level)\n");
  shuffled = malloc(nbytes);
  if (NULL != shuffled) {
    double r_orig = 0, r_shuf = 0;
    int f, count = 0;
    libxs_shuffle2(shuffled, blob, row_bytes, nrows, NULL, 0, NULL);
    for (f = 0; f < stride; ++f) {
      const size_t stride_sz = (size_t)stride;
      libxs_fprint_t fp_o, fp_s;
      if (EXIT_SUCCESS == libxs_fprint(&fp_o, dtype,
        (const char*)blob + f * tw, 1, &nrows, &stride_sz, 4, 0, 0, 0)
        && EXIT_SUCCESS == libxs_fprint(&fp_s, dtype,
        (const char*)shuffled + f * tw, 1, &nrows, &stride_sz, 4, 0, 0, 0))
      {
        r_orig += libxs_fprint_decay(&fp_o);
        r_shuf += libxs_fprint_decay(&fp_s);
        ++count;
      }
    }
    if (0 < count) {
      r_orig /= count; r_shuf /= count;
      printf("  Avg decay (original): %.6f\n", r_orig);
      printf("  Avg decay (shuffled): %.6f\n", r_shuf);
      printf("  Ratio:                %.1fx\n",
        (0 < r_orig) ? (r_shuf / r_orig) : 0);
      printf("  -> Record order %s structure.\n",
        (r_shuf > 2 * r_orig) ? "carries" : "does not carry");
      if (NULL != artifact_file) {
        fprintf(artifact_file, "shuffle,,original_decay,%.17g\n", r_orig);
        fprintf(artifact_file, "shuffle,,shuffled_decay,%.17g\n", r_shuf);
        fprintf(artifact_file, "shuffle,,ratio,%.17g\n",
          (0 < r_orig) ? (r_shuf / r_orig) : 0);
        fprintf(artifact_file, "shuffle,,carries_structure,%i\n",
          (r_shuf > 2 * r_orig) ? 1 : 0);
      }
    }
    artifact_write_matrix("rosetta_shuffled.mhd", shuffled, dtype,
      nrows, stride);
    artifact_write_field_image("rosetta_fields_shuffled.mhd", shuffled,
      dtype, nrows, stride);
    free(shuffled);
  }
}


/* Per-field decay analysis -- the payoff. */
static void field_analysis(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride)
{
  const size_t tw = LIBXS_TYPESIZE(dtype);
  const size_t ne = nbytes / tw;
  const size_t nrows = ne / stride;
  int f, best_f = 0;
  double best_decay = 1e30;
  printf("Per-field decay analysis\n");
  for (f = 0; f < stride && f < NFIELDS; ++f) {
    const size_t stride_sz = (size_t)stride;
    const void* col = (const char*)blob + f * tw;
    libxs_fprint_t fp;
    if (EXIT_SUCCESS == libxs_fprint(&fp, dtype, col,
      1, &nrows, &stride_sz, 4, 0, 0, 0))
    {
      const double decay = libxs_fprint_decay(&fp);
      printf("  [%i] %-10s  decay=%.6f", f, field_names[f], decay);
      if (decay < best_decay) { best_decay = decay; best_f = f; }
      if (NULL != artifact_file) {
        fprintf(artifact_file, "field,%i,%s,%.17g\n",
          f, field_names[f], decay);
      }
      if (5 == f) {
        const double* vals = (const double*)col;
        printf("  (last=%.10f, gamma=%.10f)",
          vals[(nrows - 1) * stride], EULER_MASCHERONI);
      }
      printf("\n");
    }
  }
  if (0 == best_f) {
    printf("  Smoothest: [0] %s (perfect ramp, decay=0)\n",
      field_names[0]);
    printf("  -> This reveals a sequential index column.\n");
    printf("  Most interesting: [5] %s -- converges to a constant.\n",
      field_names[5]);
  }
  else {
    printf("  Smoothest: [%i] %s\n", best_f, field_names[best_f]);
  }
  if (NULL != artifact_file) {
    fprintf(artifact_file, "field,,smoothest,%i\n", best_f);
  }
}


/* GREEDY sort test. */
static void sort_test(const void* blob, size_t nbytes,
  libxs_data_t dtype, int stride)
{
  const size_t tw = LIBXS_TYPESIZE(dtype);
  const size_t ne = nbytes / tw;
  const int nrows = (int)(ne / stride);
  int* perm;
  printf("GREEDY sort test (%i rows x %i cols)\n", nrows, stride);
  perm = (int*)malloc(nrows * sizeof(int));
  if (NULL != perm) {
    double* colmaj = (double*)malloc(nbytes);
    int i, is_identity = 1;
    if (NULL != colmaj) {
      int r, c;
      const double* rowmaj = (const double*)blob;
      for (r = 0; r < nrows; ++r)
        for (c = 0; c < stride; ++c)
          colmaj[c * nrows + r] = rowmaj[r * stride + c];
      libxs_sort_smooth(LIBXS_SORT_GREEDY, nrows, stride,
        colmaj, nrows, dtype, perm);
      free(colmaj);
    }
    for (i = 0; i < nrows; ++i) {
      if (perm[i] != i) { is_identity = 0; break; }
    }
    printf("  Data is %s ordered for row smoothness.\n",
      is_identity ? "already optimally" : "NOT optimally");
    if (NULL != artifact_file) {
      fprintf(artifact_file, "sort,,identity,%i\n", is_identity);
    }
    if (0 == is_identity) {
      printf("  Permutation (first 16): ");
      for (i = 0; i < LIBXS_MIN(nrows, 16); ++i)
        printf("%i ", perm[i]);
      printf("...\n");
    }
    free(perm);
  }
}


/* Setdiff verification: compare discovered blob against original. */
static void verify(const void* table,
  const void* blob, size_t nbytes, libxs_data_t dtype)
{
  const size_t tw = LIBXS_TYPESIZE(dtype);
  const int ne = (int)(nbytes / tw);
  double tol = 0;
  int d;
  printf("Verification: setdiff(original, blob)\n");
  d = libxs_setdiff_min(dtype, table, ne, blob, ne, &tol);
  printf("  Unmatched: %i, tolerance: %.2e\n", d, tol);
  printf("  -> %s\n", (0 == d) ? "Byte-perfect recovery."
    : "Values differ (unexpected).");
  if (NULL != artifact_file) {
    fprintf(artifact_file, "verify,,unmatched,%i\n", d);
    fprintf(artifact_file, "verify,,tolerance,%.17g\n", tol);
  }
}
