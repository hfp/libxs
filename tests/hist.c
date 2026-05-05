/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_hist.h>
#include <math.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if !defined(TOLERANCE)
# define TOLERANCE 1E-6
#endif


static int test_create_destroy(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) {
    FPRINTF(stderr, "ERROR line #%i: hist_create failed\n", __LINE__);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  libxs_hist_destroy(NULL);
  return result;
}


static int test_single_value(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  const double value[] = { 42.0 };
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_push(NULL, hist, value);
  libxs_hist_query(NULL, hist, &info);
  if (1 != info.nbuckets || 1 != info.nvals || NULL == info.buckets || NULL == info.vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected get result\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(info.vals[0] - 42.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected value 42.0, got %f\n", __LINE__, info.vals[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (fabs(info.range[0] - 42.0) > TOLERANCE || fabs(info.range[1] - 42.0) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range mismatch [%f, %f]\n", __LINE__, info.range[0], info.range[1]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 1 != info.nsamples) {
    FPRINTF(stderr, "ERROR line #%i: expected nsamples=1, got %i\n", __LINE__, info.nsamples);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_fill_phase_range(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  const double v1[] = { 10.0 }, v2[] = { 20.0 }, v3[] = { 15.0 };
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_push(NULL, hist, v1);
  libxs_hist_push(NULL, hist, v2);
  libxs_hist_push(NULL, hist, v3);
  libxs_hist_query(NULL, hist, &info);
  if (fabs(info.range[0] - 10.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected min=10.0, got %f\n", __LINE__, info.range[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(info.range[1] - 20.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected max=20.0, got %f\n", __LINE__, info.range[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_bucket_distribution(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_add };
  libxs_hist_info_t info;
  int i, total;
  int result = EXIT_SUCCESS;
  const double vmin[] = { 0.0 }, vmax[] = { 100.0 };
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_push(NULL, hist, vmin);
  libxs_hist_push(NULL, hist, vmax);
  {
    const double v33[] = { 33.0 }, v66[] = { 66.0 };
    libxs_hist_push(NULL, hist, v33);
    libxs_hist_push(NULL, hist, v66);
  }
  for (i = 0; i < 40; ++i) {
    const double v[] = { 2.5 * i };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query(NULL, hist, &info);
  if (4 != info.nbuckets || NULL == info.buckets) {
    FPRINTF(stderr, "ERROR line #%i: unexpected nbuckets=%i\n", __LINE__, info.nbuckets);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < info.nbuckets; ++i) {
      if (info.buckets[i] < 0) {
        FPRINTF(stderr, "ERROR line #%i: negative bucket[%i]=%i\n", __LINE__, i, info.buckets[i]);
        result = EXIT_FAILURE;
        break;
      }
      total += info.buckets[i];
    }
    if (EXIT_SUCCESS == result && total < 40) {
      FPRINTF(stderr, "ERROR line #%i: total count %i < 40\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_update_add(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_add };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v[] = { 5.0 };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query(NULL, hist, &info);
  if (1 != info.nbuckets || NULL == info.vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state\n", __LINE__);
    libxs_hist_destroy(hist);
    return EXIT_FAILURE;
  }
  {
    const double v1[] = { 5.0 }, v2[] = { 5.0 }, v3[] = { 5.0 };
    libxs_hist_push(NULL, hist, v1);
    libxs_hist_push(NULL, hist, v2);
    libxs_hist_push(NULL, hist, v3);
  }
  libxs_hist_query(NULL, hist, &info);
  if (NULL == info.vals || NULL == info.buckets) {
    FPRINTF(stderr, "ERROR line #%i: get failed\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && info.buckets[0] < 2) {
    FPRINTF(stderr, "ERROR line #%i: expected bucket count >= 2, got %i\n", __LINE__, info.buckets[0]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_update_avg(void)
{
  double a = 10.0;
  const double b = 20.0;
  int result = EXIT_SUCCESS;
  /* Welford: mean of {10, 20} = 15.0 */
  libxs_hist_update_avg(&a, &b, 2);
  if (fabs(a - 15.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 15.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  /* Welford: mean of {10, 20, 20} = 16.667 */
  libxs_hist_update_avg(&a, &b, 3);
  if (fabs(a - 50.0 / 3) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 16.667, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_update_add_fn(void)
{
  double a = 3.0;
  const double b = 7.0;
  int result = EXIT_SUCCESS;
  libxs_hist_update_add(&a, &b, 2);
  if (fabs(a - 10.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 10.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  libxs_hist_update_add(&a, &b, 3);
  if (fabs(a - 17.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 17.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_update_min_max(void)
{
  double mn = 5.0, mx = 5.0;
  const double lo = 2.0, hi = 8.0;
  int result = EXIT_SUCCESS;
  libxs_hist_update_min(&mn, &lo, 2);
  if (fabs(mn - 2.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 2.0, got %f\n", __LINE__, mn);
    result = EXIT_FAILURE;
  }
  libxs_hist_update_min(&mn, &hi, 3);
  if (EXIT_SUCCESS == result && fabs(mn - 2.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 2.0, got %f\n", __LINE__, mn);
    result = EXIT_FAILURE;
  }
  libxs_hist_update_max(&mx, &hi, 2);
  if (EXIT_SUCCESS == result && fabs(mx - 8.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 8.0, got %f\n", __LINE__, mx);
    result = EXIT_FAILURE;
  }
  libxs_hist_update_max(&mx, &lo, 3);
  if (EXIT_SUCCESS == result && fabs(mx - 8.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 8.0, got %f\n", __LINE__, mx);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_multiple_values(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_add };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(2/*nbuckets*/, 2/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double e1[] = { 0.0, 100.0 }, e2[] = { 10.0, 200.0 };
    libxs_hist_push(NULL, hist, e1);
    libxs_hist_push(NULL, hist, e2);
  }
  libxs_hist_query(NULL, hist, &info);
  if (2 != info.nbuckets || 2 != info.nvals || NULL == info.buckets || NULL == info.vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state nbuckets=%i nvals=%i\n",
      __LINE__, info.nbuckets, info.nvals);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (fabs(info.range[0] - 0.0) > TOLERANCE || fabs(info.range[1] - 10.0) > TOLERANCE) {
      FPRINTF(stderr, "ERROR line #%i: range [%f, %f] unexpected\n", __LINE__, info.range[0], info.range[1]);
      result = EXIT_FAILURE;
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_print(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(3/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 1.0 }, v2[] = { 2.0 }, v3[] = { 3.0 };
    libxs_hist_push(NULL, hist, v1);
    libxs_hist_push(NULL, hist, v2);
    libxs_hist_push(NULL, hist, v3);
  }
  {
#if defined(_DEBUG)
    FILE *const ostream = stderr;
#elif !defined(_WIN32)
    FILE *const ostream = fopen("/dev/null", "w");
#else
    FILE *const ostream = fopen("NUL", "w");
#endif
    if (NULL != ostream) {
      const int prec[] = { 2 };
      libxs_hist_print(ostream, hist, prec, "test_print");
#if !defined(_DEBUG)
      fclose(ostream);
#endif
    }
  }
  libxs_hist_print(NULL, hist, NULL, "null_stream");
  libxs_hist_destroy(hist);
  return result;
}


static int test_set_null_hist(void)
{
  int result = EXIT_SUCCESS;
  const double value[] = { 1.0 };
  libxs_hist_push(NULL, NULL, value);
  return result;
}


static int test_get_null_hist(void)
{
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  libxs_hist_query(NULL, NULL, &info);
  if (0 != info.nbuckets || 0 != info.nvals || NULL != info.buckets || NULL != info.vals) {
    FPRINTF(stderr, "ERROR line #%i: get on NULL hist should yield empty\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (fabs(info.range[0]) > TOLERANCE || fabs(info.range[1]) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range should be [0,0] for NULL hist\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 != info.nsamples) {
    FPRINTF(stderr, "ERROR line #%i: nsamples should be 0 for NULL hist\n", __LINE__);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_many_values_bucketing(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int i, total;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(10/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    for (i = 0; i < 10; ++i) {
      const double fv[] = { 200.0 * i / 9 };
      libxs_hist_push(NULL, hist, fv);
    }
  }
  for (i = 0; i < 100; ++i) {
    const double v[] = { 2.0 * i };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query(NULL, hist, &info);
  if (10 != info.nbuckets || NULL == info.buckets) {
    FPRINTF(stderr, "ERROR line #%i: nbuckets=%i\n", __LINE__, info.nbuckets);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
    if (total < 10) {
      FPRINTF(stderr, "ERROR line #%i: total=%i < 10\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
    for (i = 0; i < info.nbuckets && EXIT_SUCCESS == result; ++i) {
      if (0 == info.buckets[i]) {
        FPRINTF(stderr, "ERROR line #%i: bucket[%i] is empty\n", __LINE__, i);
        result = EXIT_FAILURE;
      }
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_underpopulated(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int i, total;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(8/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 10.0 }, v2[] = { 50.0 }, v3[] = { 90.0 };
    libxs_hist_push(NULL, hist, v1);
    libxs_hist_push(NULL, hist, v2);
    libxs_hist_push(NULL, hist, v3);
  }
  libxs_hist_query(NULL, hist, &info);
  if (3 != info.nbuckets) {
    FPRINTF(stderr, "ERROR line #%i: expected 3 buckets, got %i\n", __LINE__, info.nbuckets);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (NULL == info.buckets || NULL == info.vals || 1 != info.nvals)) {
    FPRINTF(stderr, "ERROR line #%i: unexpected NULL or nvals=%i\n", __LINE__, info.nvals);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
    if (3 != total) {
      FPRINTF(stderr, "ERROR line #%i: total=%i (expected 3)\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && (fabs(info.range[0] - 10.0) > TOLERANCE || fabs(info.range[1] - 90.0) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range [%f, %f] unexpected\n", __LINE__, info.range[0], info.range[1]);
    result = EXIT_FAILURE;
  }
  {
    const double v4[] = { 30.0 }, v5[] = { 70.0 };
    libxs_hist_push(NULL, hist, v4);
    libxs_hist_push(NULL, hist, v5);
  }
  libxs_hist_query(NULL, hist, &info);
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
    if (5 != total) {
      FPRINTF(stderr, "ERROR line #%i: total=%i after more inserts (expected 5)\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_commit_arithmetic_avg(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  /* 1 bucket, nqueue=4: all 4 values land in the same bucket at commit.
   * Arithmetic mean of {10, 20, 30, 40} = 25.0
   */
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 10.0 }, v2[] = { 20.0 }, v3[] = { 30.0 }, v4[] = { 40.0 };
    libxs_hist_push(NULL, hist, v1);
    libxs_hist_push(NULL, hist, v2);
    libxs_hist_push(NULL, hist, v3);
    libxs_hist_push(NULL, hist, v4);
  }
  libxs_hist_query(NULL, hist, &info);
  if (1 != info.nbuckets || NULL == info.vals || NULL == info.buckets) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(info.vals[0] - 25.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 25.0 (arithmetic mean), got %f\n", __LINE__, info.vals[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 4 != info.buckets[0]) {
    FPRINTF(stderr, "ERROR line #%i: expected count=4, got %i\n", __LINE__, info.buckets[0]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_hybrid_avg_then_welford(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int result = EXIT_SUCCESS;
  /* 1 bucket, nqueue=2: commit produces arithmetic mean,
   * then subsequent inserts use Welford.
   * Queue: {10, 30} -> commit: mean=20.0
   * Welford with 40.0 (count=3): 20 + (40-20)/3 = 26.667
   */
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 10.0 }, v2[] = { 30.0 };
    libxs_hist_push(NULL, hist, v1);
    libxs_hist_push(NULL, hist, v2);
  }
  libxs_hist_query(NULL, hist, &info);
  if (EXIT_SUCCESS == result && fabs(info.vals[0] - 20.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 20.0 after commit, got %f\n", __LINE__, info.vals[0]);
    result = EXIT_FAILURE;
  }
  {
    const double v3[] = { 40.0 };
    libxs_hist_push(NULL, hist, v3);
  }
  libxs_hist_query(NULL, hist, &info);
  if (EXIT_SUCCESS == result && fabs(info.vals[0] - 80.0 / 3) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 26.667 after Welford, got %f\n", __LINE__, info.vals[0]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_nsamples(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  libxs_hist_info_t info;
  int i;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i < 20; ++i) {
    const double v[] = { (double)i };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query(NULL, hist, &info);
  if (20 != info.nsamples) {
    FPRINTF(stderr, "ERROR line #%i: expected nsamples=20, got %i\n", __LINE__, info.nsamples);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_median_uniform(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  double vals[1];
  int i;
  hist = libxs_hist_create(10/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i <= 100; ++i) {
    const double v[] = { (double)i };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query_median(NULL, hist, vals);
  if (fabs(vals[0] - 50.0) > 5.0 + TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected median ~50.0, got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  {
    double p0[1], p1[1];
    libxs_hist_query_percentile(NULL, hist, p0, 0.0);
    libxs_hist_query_percentile(NULL, hist, p1, 1.0);
    if (p0[0] > 10.0 + TOLERANCE) {
      FPRINTF(stderr, "ERROR line #%i: percentile(0)=%f too high\n", __LINE__, p0[0]);
      result = EXIT_FAILURE;
    }
    if (p1[0] < 90.0 - TOLERANCE) {
      FPRINTF(stderr, "ERROR line #%i: percentile(1)=%f too low\n", __LINE__, p1[0]);
      result = EXIT_FAILURE;
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_median_single(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  double vals[1];
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v[] = { 42.0 };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query_median(NULL, hist, vals);
  if (fabs(vals[0] - 42.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 42.0, got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_median_null(void)
{
  int result = EXIT_SUCCESS;
  double vals[1] = { -1.0 };
  libxs_hist_query_median(NULL, NULL, vals);
  if (fabs(vals[0] - (-1.0)) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected -1.0 (untouched), got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_percentile_vals(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  double vals[2];
  int i;
  hist = libxs_hist_create(4/*nbuckets*/, 2/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i <= 40; ++i) {
    const double v[] = { (double)i, 100.0 + i };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query_median(NULL, hist, vals);
  if (fabs(vals[0] - 20.0) > 5.0 + TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected median ~20.0, got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(vals[1] - 120.0) > 10.0 + TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected aux ~120.0, got %f\n", __LINE__, vals[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


int main(void)
{
  int result = EXIT_SUCCESS;

  if (EXIT_SUCCESS != test_create_destroy()) {
    FPRINTF(stderr, "FAILED: test_create_destroy\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_single_value()) {
    FPRINTF(stderr, "FAILED: test_single_value\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_fill_phase_range()) {
    FPRINTF(stderr, "FAILED: test_fill_phase_range\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_bucket_distribution()) {
    FPRINTF(stderr, "FAILED: test_bucket_distribution\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_update_add()) {
    FPRINTF(stderr, "FAILED: test_update_add\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_update_avg()) {
    FPRINTF(stderr, "FAILED: test_update_avg\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_update_add_fn()) {
    FPRINTF(stderr, "FAILED: test_update_add_fn\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_update_min_max()) {
    FPRINTF(stderr, "FAILED: test_update_min_max\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_multiple_values()) {
    FPRINTF(stderr, "FAILED: test_multiple_values\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_print()) {
    FPRINTF(stderr, "FAILED: test_print\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_set_null_hist()) {
    FPRINTF(stderr, "FAILED: test_set_null_hist\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_get_null_hist()) {
    FPRINTF(stderr, "FAILED: test_get_null_hist\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_many_values_bucketing()) {
    FPRINTF(stderr, "FAILED: test_many_values_bucketing\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_underpopulated()) {
    FPRINTF(stderr, "FAILED: test_underpopulated\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_commit_arithmetic_avg()) {
    FPRINTF(stderr, "FAILED: test_commit_arithmetic_avg\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_hybrid_avg_then_welford()) {
    FPRINTF(stderr, "FAILED: test_hybrid_avg_then_welford\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_nsamples()) {
    FPRINTF(stderr, "FAILED: test_nsamples\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_median_uniform()) {
    FPRINTF(stderr, "FAILED: test_median_uniform\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_median_single()) {
    FPRINTF(stderr, "FAILED: test_median_single\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_median_null()) {
    FPRINTF(stderr, "FAILED: test_median_null\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_percentile_vals()) {
    FPRINTF(stderr, "FAILED: test_percentile_vals\n");
    result = EXIT_FAILURE;
  }

  return result;
}
