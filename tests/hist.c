/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_hist.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_rng.h>

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
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(2/*nbuckets*/, 2/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(3/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(10/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(8/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  /**
   * 1 bucket, nqueue=4: all 4 values land in the same bucket at commit.
   * Arithmetic mean of {10, 20, 30, 40} = 25.0
   */
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  /**
   * 1 bucket, nqueue=2: commit produces arithmetic mean,
   * then subsequent inserts use Welford.
   * Queue: {10, 30} -> commit: mean=20.0
   * Welford with 40.0 (count=3): 20 + (40-20)/3 = 26.667
   */
  hist = libxs_hist_create(1/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(10/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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
  hist = libxs_hist_create(4/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
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


/**
 * Bimodal data must not yield a reading that belongs to no sample. Two tight
 * clusters far apart leave empty buckets between them; interpolating towards one
 * reads storage no sample ever wrote, and the result then varies with the bucket
 * count alone. Both clusters share a rate-forming pair (amount, duration), so the
 * queried ratio has to match one of them.
 */
static int test_query_bimodal(void)
{
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg, libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  int nbuckets;
  for (nbuckets = 2; nbuckets <= 12 && EXIT_SUCCESS == result; ++nbuckets) {
    libxs_hist_t* const hist = libxs_hist_create(nbuckets, 3/*nvals*/, update, NULL, NULL);
    double med[3] = { 0 }, mod[3] = { 0 };
    int i;
    if (NULL == hist) return EXIT_FAILURE;
    for (i = 0; i < 40; ++i) { /* alternating: 100/10 and 10/5 */
      const double lo[] = { 10.0, 10.0, 5.0 }, hi[] = { 100.0, 100.0, 10.0 };
      libxs_hist_push(NULL, hist, (0 == (i & 1)) ? hi : lo);
    }
    /* the mode always reports a populated bucket, hence an observed ratio */
    libxs_hist_query_mode(NULL, hist, mod);
    if (0 >= mod[2] || (fabs(mod[1] / mod[2] - 10.0) > TOLERANCE && fabs(mod[1] / mod[2] - 2.0) > TOLERANCE)) {
      FPRINTF(stderr, "ERROR line #%i: nbuckets=%i mode ratio %f is neither 10.0 nor 2.0\n", __LINE__, nbuckets,
        0 < mod[2] ? (mod[1] / mod[2]) : 0.0);
      result = EXIT_FAILURE;
    }
    /**
     * The median may legitimately blend when both neighbours are populated (at
     * nbuckets=2 the two clusters are adjacent), but must never mix across an
     * empty bucket: with a gap the ratio has to be one of the two observed.
     */
    libxs_hist_query_median(NULL, hist, med);
    if (EXIT_SUCCESS == result && 2 < nbuckets) {
      if (0 >= med[2] || (fabs(med[1] / med[2] - 10.0) > TOLERANCE && fabs(med[1] / med[2] - 2.0) > TOLERANCE)) {
        FPRINTF(stderr, "ERROR line #%i: nbuckets=%i median ratio %f is neither 10.0 nor 2.0\n", __LINE__, nbuckets,
          0 < med[2] ? (med[1] / med[2]) : 0.0);
        result = EXIT_FAILURE;
      }
    }
    libxs_hist_destroy(hist);
  }
  return result;
}


/** The mode reports the most populated bucket, which a skewed sample set pins. */
static int test_mode_skewed(void)
{
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg };
  libxs_hist_t* hist = NULL;
  int result = EXIT_SUCCESS;
  double vals[2] = { 0 };
  int i;
  hist = libxs_hist_create(4/*nbuckets*/, 2/*nvals*/, update, NULL, NULL);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i < 40; ++i) { /* 36 samples at 10.0, 4 at 100.0 */
    const double lo[] = { 10.0, 7.0 }, hi[] = { 100.0, 70.0 };
    libxs_hist_push(NULL, hist, (0 == (i % 10)) ? hi : lo);
  }
  libxs_hist_query_mode(NULL, hist, vals);
  if (fabs(vals[1] - 7.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected mode aux 7.0 (dominant cluster), got %f\n", __LINE__, vals[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


/** Uniform data: mode and median agree, and both are exact. */
static int test_mode_uniform(void)
{
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg };
  libxs_hist_t* hist = NULL;
  int result = EXIT_SUCCESS;
  double med[2] = { 0 }, mod[2] = { 0 };
  int i;
  hist = libxs_hist_create(4/*nbuckets*/, 2/*nvals*/, update, NULL, NULL);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i < 20; ++i) {
    const double v[] = { 42.0, 13.0 };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query_median(NULL, hist, med);
  libxs_hist_query_mode(NULL, hist, mod);
  if (fabs(mod[1] - 13.0) > TOLERANCE || fabs(med[1] - 13.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 13.0 for both, got mode %f median %f\n", __LINE__, mod[1], med[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


/** Empty and NULL histograms leave the caller's values untouched. */
static int test_mode_empty_null(void)
{
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg };
  libxs_hist_t* hist = NULL;
  int result = EXIT_SUCCESS;
  double vals[2] = { -1.0, -1.0 };
  libxs_hist_query_mode(NULL, NULL, vals);
  if (fabs(vals[0] - (-1.0)) > TOLERANCE || fabs(vals[1] - (-1.0)) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: NULL hist must leave vals untouched\n", __LINE__);
    result = EXIT_FAILURE;
  }
  hist = libxs_hist_create(4/*nbuckets*/, 2/*nvals*/, update, NULL, NULL);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_query_mode(NULL, hist, vals);
  if (EXIT_SUCCESS == result && (fabs(vals[0] - (-1.0)) > TOLERANCE || fabs(vals[1] - (-1.0)) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: empty hist must leave vals untouched\n", __LINE__);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_percentile_vals(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_avg };
  int result = EXIT_SUCCESS;
  double vals[2];
  int i;
  hist = libxs_hist_create(4/*nbuckets*/, 2/*nvals*/, update, NULL, NULL);
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


static int test_commit_out_of_order(void)
{
  /**
   * A queued sample has to survive the commit even when its bucket is not the
   * one its queue position aliases. Committing in place relocated such a sample
   * onto an index the scan had already passed: three kernel launches timed
   * 86.161, 90.349 and 89.131 ms committed as two, and every derived figure was
   * then computed from the wrong population.
   */
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  const double sample[] = { 86.161, 90.349, 89.131 };
  libxs_hist_info_t info;
  int i, total = 0;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(3/*nbuckets*/, 1/*nvals*/, update, NULL, NULL);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i < 3; ++i) {
    const double v[] = { sample[i] };
    libxs_hist_push(NULL, hist, v);
  }
  libxs_hist_query(NULL, hist, &info);
  for (i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
  if (info.nsamples != total) {
    FPRINTF(stderr, "ERROR line #%i: %i of %i samples committed\n", __LINE__, total, info.nsamples);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (3 != info.nbuckets || 2 != info.buckets[2]
    || fabs(info.vals[2] - 89.74) > 1E-3))
  {
    FPRINTF(stderr, "ERROR line #%i: bucket 3 holds %i -> %f (expected 2 -> 89.740)\n",
      __LINE__, info.buckets[2], info.vals[2]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_commit_counts_every_sample(void)
{
  /**
   * Sum of the bucket counts is the number of pushes, for any shape of input:
   * below the queue capacity (one fold) and above it (fold plus direct binning).
   * Deterministic, so a failure names the case that produced it.
   */
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int nbuckets, n, trial;
  int result = EXIT_SUCCESS;
  for (nbuckets = 1; nbuckets <= 8 && EXIT_SUCCESS == result; ++nbuckets) {
    for (n = 1; n <= 24 && EXIT_SUCCESS == result; ++n) {
      for (trial = 0; trial < 8 && EXIT_SUCCESS == result; ++trial) {
        libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
        libxs_hist_info_t info;
        int i, total = 0;
        if (NULL != hist) {
          for (i = 0; i < n; ++i) {
            const double v[] = { (double)(((i * 7 + trial * 13) % n) + 1) };
            libxs_hist_push(NULL, hist, v);
          }
          libxs_hist_query(NULL, hist, &info);
          for (i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
          if (info.nsamples != total) {
            FPRINTF(stderr, "ERROR line #%i: nbuckets=%i n=%i trial=%i: %i of %i committed\n",
              __LINE__, nbuckets, n, trial, total, info.nsamples);
            result = EXIT_FAILURE;
          }
          libxs_hist_destroy(hist);
        }
        else result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


static int test_running_sum(void)
{
  /**
   * The running total is accumulated on push, so it holds for every update
   * function - including min and max, whose buckets discard what a total would
   * have to be reconstructed from.
   */
  const libxs_hist_update_t update[] = { libxs_hist_update_avg, libxs_hist_update_add,
    libxs_hist_update_min, libxs_hist_update_max };
  const double sample[] = { 3.0, 17.0, 5.0, 11.0, 2.0, 29.0, 7.0 };
  const int nsample = (int)(sizeof(sample) / sizeof(*sample));
  libxs_hist_t* hist = NULL;
  libxs_hist_info_t info;
  double expect = 0;
  int i, k;
  int result = EXIT_SUCCESS;
  hist = libxs_hist_create(3/*nbuckets*/, 4/*nvals*/, update, NULL, NULL);
  if (NULL == hist) return EXIT_FAILURE;
  for (i = 0; i < nsample; ++i) {
    const double v[] = { sample[i], sample[i], sample[i], sample[i] };
    libxs_hist_push(NULL, hist, v);
    expect += sample[i];
  }
  libxs_hist_query(NULL, hist, &info);
  if (NULL == info.sum || nsample != info.nsamples) {
    FPRINTF(stderr, "ERROR line #%i: sum=%p nsamples=%i (expected %i)\n",
      __LINE__, (const void*)info.sum, info.nsamples, nsample);
    result = EXIT_FAILURE;
  }
  for (k = 0; k < 4 && EXIT_SUCCESS == result; ++k) {
    if (fabs(info.sum[k] - expect) > TOLERANCE) {
      FPRINTF(stderr, "ERROR line #%i: sum[%i]=%f (expected %f)\n", __LINE__, k, info.sum[k], expect);
      result = EXIT_FAILURE;
    }
  }
  /* and past the queue, where samples bin directly instead of being folded */
  for (i = 0; i < 100; ++i) {
    const double v[] = { 1.0, 1.0, 1.0, 1.0 };
    libxs_hist_push(NULL, hist, v);
    expect += 1.0;
  }
  libxs_hist_query(NULL, hist, &info);
  if (EXIT_SUCCESS == result && (NULL == info.sum || fabs(info.sum[0] - expect) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: sum[0]=%f after direct binning (expected %f)\n",
      __LINE__, NULL != info.sum ? info.sum[0] : 0.0, expect);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_median_outlier(void)
{
  /**
   * A far outlier must not push the median into empty space. Six kernel
   * launches near 0.88 ms and a first launch at 90 ms, which is what a one-time
   * page commit inside the profiled window looks like: the median has to name
   * the cluster the samples occupy, not a coordinate between the two clusters.
   * Reconstructing it from the bucket's position on the axis reported 11.3 ms,
   * a value no launch took and 13x the real one.
   *
   * From three buckets up, since with two the outlier's bucket is adjacent and
   * a blend between two populated neighbours is legitimate.
   */
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  const double sample[] = { 90.093, 0.878, 0.880, 0.877, 0.873, 0.880, 0.875 };
  const int nsample = (int)(sizeof(sample) / sizeof(*sample));
  int nbuckets;
  int result = EXIT_SUCCESS;
  for (nbuckets = 3; nbuckets <= 12 && EXIT_SUCCESS == result; ++nbuckets) {
    libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
    double med[1];
    int i;
    if (NULL != hist) {
      med[0] = 0;
      for (i = 0; i < nsample; ++i) {
        const double v[] = { sample[i] };
        libxs_hist_push(NULL, hist, v);
      }
      libxs_hist_query_median(NULL, hist, med);
      if (fabs(med[0] - 0.877) > 0.01) {
        FPRINTF(stderr, "ERROR line #%i: nbuckets=%i median=%f (expected ~0.877)\n",
          __LINE__, nbuckets, med[0]);
        result = EXIT_FAILURE;
      }
      libxs_hist_destroy(hist);
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


/**
 * Reproduce the binning the histogram applies, over samples the test retains.
 * Deliberately a second implementation rather than a call back into the library:
 * an oracle that shares code with what it checks proves nothing.
 */
static void oracle(const double sample[], int nsample, int nbuckets,
  int counts[], double means[], double range[2])
{
  double lo = sample[0], hi = sample[0], w;
  int i, k;
  for (i = 1; i < nsample; ++i) {
    if (sample[i] < lo) lo = sample[i];
    if (sample[i] > hi) hi = sample[i];
  }
  nbuckets = LIBXS_MIN(nbuckets, nsample);
  w = hi - lo;
  for (i = 0; i < nbuckets; ++i) {
    counts[i] = 0;
    means[i] = 0;
  }
  for (k = 0; k < nsample; ++k) {
    for (i = 1; i <= nbuckets; ++i) {
      const double q = lo + i * w / nbuckets;
      if (sample[k] <= q || nbuckets == i) {
        means[i - 1] += sample[k];
        ++counts[i - 1];
        break;
      }
    }
  }
  for (i = 0; i < nbuckets; ++i) {
    if (0 < counts[i]) means[i] /= counts[i];
  }
  range[0] = lo;
  range[1] = hi;
}


static int test_oracle_exact(void)
{
  /**
   * Randomized input that stays inside the queue, which is the regime a profile
   * lives in and the one the swap-and-flag commit lost samples in. The bound
   * matters: once a push triggers the fold, later samples are binned against a
   * range derived from the batch alone, and one falling outside it rebins the
   * histogram - after which no oracle over the whole input can predict a
   * per-bucket mean. Up to the capacity every sample is still queued, the fold
   * happens at query with all of them present, and counts and means are exact.
   */
  enum { MAXN = 40 };
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  double sample[MAXN], means[MAXN], range[2];
  int counts[MAXN];
  int nbuckets, nsample, trial, i;
  int result = EXIT_SUCCESS;
  libxs_rng_set_seed(12345u); /* reproducible */
  for (nbuckets = 1; nbuckets <= 8 && EXIT_SUCCESS == result; ++nbuckets) {

    const int maxn = LIBXS_MIN(MAXN, 16 * nbuckets);
    for (nsample = 1; nsample <= maxn && EXIT_SUCCESS == result; ++nsample) {
      for (trial = 0; trial < 20 && EXIT_SUCCESS == result; ++trial) {
        libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
        libxs_hist_info_t info;
        if (NULL == hist) {
          result = EXIT_FAILURE;
          break;
        }
        for (i = 0; i < nsample; ++i) {
          const double v[] = { 100.0 * libxs_rng_f64() };
          sample[i] = v[0];
          libxs_hist_push(NULL, hist, v);
        }
        oracle(sample, nsample, nbuckets, counts, means, range);
        libxs_hist_query(NULL, hist, &info);
        if (nsample != info.nsamples) {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: nsamples=%i\n",
            __LINE__, nbuckets, nsample, trial, info.nsamples);
          result = EXIT_FAILURE;
        }
        for (i = 0; i < info.nbuckets && EXIT_SUCCESS == result; ++i) {
          if (counts[i] != info.buckets[i]) {
            FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: bucket %i count %i != %i\n",
              __LINE__, nbuckets, nsample, trial, i, info.buckets[i], counts[i]);
            result = EXIT_FAILURE;
          }
          else if (0 < counts[i] && fabs(means[i] - info.vals[i * info.nvals]) > 1E-9) {
            FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: bucket %i mean %f != %f\n",
              __LINE__, nbuckets, nsample, trial, i, info.vals[i * info.nvals], means[i]);
            result = EXIT_FAILURE;
          }
        }
        if (EXIT_SUCCESS == result
          && (fabs(range[0] - info.range[0]) > 1E-9 || fabs(range[1] - info.range[1]) > 1E-9))
        {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: range [%f,%f]\n",
            __LINE__, nbuckets, nsample, trial, info.range[0], info.range[1]);
          result = EXIT_FAILURE;
        }
        libxs_hist_destroy(hist);
      }
    }
  }
  return result;
}


static int test_invariants_random(void)
{
  /**
   * Beyond the queue capacity, where rebinning moves aggregates by a bucket's
   * midpoint and individual means are no longer predictable. Two identities
   * survive that and are asserted instead: the counts total the samples, and
   * count times mean totals the sum. The latter is the strong one - it fails on
   * a dropped sample, a double-counted one, a botched Welford step and a lossy
   * rebin alike - and it is checked against a running sum accumulated by a
   * different mechanism, so agreement is evidence rather than a tautology.
   */
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int nbuckets, nsample, trial, i;
  int result = EXIT_SUCCESS;
  libxs_rng_set_seed(987654321u); /* reproducible */
  for (nbuckets = 1; nbuckets <= 8 && EXIT_SUCCESS == result; ++nbuckets) {
    for (nsample = 1; nsample <= 400 && EXIT_SUCCESS == result; nsample += 37) {
      for (trial = 0; trial < 10 && EXIT_SUCCESS == result; ++trial) {
        libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
        libxs_hist_info_t info;
        double expect = 0, product = 0;
        int total = 0;
        if (NULL == hist) {
          result = EXIT_FAILURE;
          break;
        }
        for (i = 0; i < nsample; ++i) {
          /**
           * A heavy tail every so often, which is what a first launch paying a
           * one-time cost looks like and what forces a rebin.
           */
          const double v[] = { (0 == (i % 29)) ? (1E4 * libxs_rng_f64()) : libxs_rng_f64() };
          expect += v[0];
          libxs_hist_push(NULL, hist, v);
        }
        libxs_hist_query(NULL, hist, &info);
        for (i = 0; i < info.nbuckets; ++i) {
          total += info.buckets[i];
          product += info.buckets[i] * info.vals[i * info.nvals];
        }
        if (nsample != total || nsample != info.nsamples) {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: %i counted, nsamples=%i\n",
            __LINE__, nbuckets, nsample, trial, total, info.nsamples);
          result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result && fabs(product - expect) > 1E-6 * fabs(expect)) {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: count*mean=%f, sum=%f\n",
            __LINE__, nbuckets, nsample, trial, product, expect);
          result = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == result
          && (NULL == info.sum || fabs(info.sum[0] - expect) > 1E-6 * fabs(expect)))
        {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: running sum=%f, sum=%f\n",
            __LINE__, nbuckets, nsample, trial, NULL != info.sum ? info.sum[0] : 0.0, expect);
          result = EXIT_FAILURE;
        }
        libxs_hist_destroy(hist);
      }
    }
  }
  return result;
}


static int test_oracle_multifold(void)
{
  /**
   * Past the queue capacity, so the batch is folded several times, but with the
   * extremes pushed first so that no later sample reaches outside the axis the
   * first batch established. Nothing rebins, every fold bins against the same
   * axis, and the result stays exactly predictable across all of them - which
   * the single-batch oracle cannot check and which is where a cyclic fold would
   * lose or double-count a sample.
   */
  enum { MAXN = 300 };
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  double sample[MAXN], means[16], range[2];
  int counts[16];
  int nbuckets, nsample, trial, i;
  int result = EXIT_SUCCESS;
  libxs_rng_set_seed(24680u); /* reproducible */
  for (nbuckets = 1; nbuckets <= 8 && EXIT_SUCCESS == result; ++nbuckets) {
    for (nsample = 2; nsample <= MAXN && EXIT_SUCCESS == result; nsample += 31) {
      for (trial = 0; trial < 10 && EXIT_SUCCESS == result; ++trial) {
        libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
        libxs_hist_info_t info;
        if (NULL == hist) {
          result = EXIT_FAILURE;
          break;
        }
        for (i = 0; i < nsample; ++i) {
          double v[1];
          /* the extremes first, so the first fold sees the whole range */
          if (0 == i) v[0] = 0.0;
          else if (1 == i) v[0] = 100.0;
          else v[0] = 100.0 * libxs_rng_f64();
          sample[i] = v[0];
          libxs_hist_push(NULL, hist, v);
        }
        oracle(sample, nsample, nbuckets, counts, means, range);
        libxs_hist_query(NULL, hist, &info);
        if (nsample != info.nsamples) {
          FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: nsamples=%i\n",
            __LINE__, nbuckets, nsample, trial, info.nsamples);
          result = EXIT_FAILURE;
        }
        for (i = 0; i < info.nbuckets && EXIT_SUCCESS == result; ++i) {
          if (counts[i] != info.buckets[i]) {
            FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: bucket %i count %i != %i\n",
              __LINE__, nbuckets, nsample, trial, i, info.buckets[i], counts[i]);
            result = EXIT_FAILURE;
          }
          else if (0 < counts[i] && fabs(means[i] - info.vals[i * info.nvals]) > 1E-9) {
            FPRINTF(stderr, "ERROR line #%i: nb=%i n=%i t=%i: bucket %i mean %f != %f\n",
              __LINE__, nbuckets, nsample, trial, i, info.vals[i * info.nvals], means[i]);
            result = EXIT_FAILURE;
          }
        }
        libxs_hist_destroy(hist);
      }
    }
  }
  return result;
}


static int test_query_interleaved(void)
{
  /**
   * A query folds whatever is pending, so pushing and querying in turn folds
   * partial batches. Nothing may be lost or counted twice across that, and
   * querying twice in a row must report the same thing rather than folding the
   * same samples again.
   */
  const libxs_hist_update_t update[] = { libxs_hist_update_avg };
  int nbuckets, period, i;
  int result = EXIT_SUCCESS;
  libxs_rng_set_seed(13579u); /* reproducible */
  for (nbuckets = 1; nbuckets <= 6 && EXIT_SUCCESS == result; ++nbuckets) {
    for (period = 1; period <= 20 && EXIT_SUCCESS == result; period += 3) {
      libxs_hist_t* const hist = libxs_hist_create(nbuckets, 1/*nvals*/, update, NULL, NULL);
      libxs_hist_info_t info, again;
      double expect = 0, product = 0;
      int total = 0;
      if (NULL == hist) {
        result = EXIT_FAILURE;
        break;
      }
      for (i = 0; i < 200; ++i) {
        const double v[] = { 100.0 * libxs_rng_f64() };
        expect += v[0];
        libxs_hist_push(NULL, hist, v);
        if (0 == (i % period)) libxs_hist_query(NULL, hist, &info);
      }
      libxs_hist_query(NULL, hist, &info);
      libxs_hist_query(NULL, hist, &again);
      for (i = 0; i < info.nbuckets; ++i) {
        total += info.buckets[i];
        product += info.buckets[i] * info.vals[i * info.nvals];
      }
      if (200 != total || 200 != info.nsamples || again.nsamples != info.nsamples) {
        FPRINTF(stderr, "ERROR line #%i: nb=%i p=%i: %i counted, nsamples=%i/%i\n",
          __LINE__, nbuckets, period, total, info.nsamples, again.nsamples);
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result && fabs(product - expect) > 1E-6 * fabs(expect)) {
        FPRINTF(stderr, "ERROR line #%i: nb=%i p=%i: count*mean=%f, sum=%f\n",
          __LINE__, nbuckets, period, product, expect);
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result
        && (NULL == info.sum || fabs(info.sum[0] - expect) > 1E-6 * fabs(expect)))
      {
        FPRINTF(stderr, "ERROR line #%i: nb=%i p=%i: running sum mismatch\n",
          __LINE__, nbuckets, period);
        result = EXIT_FAILURE;
      }
      libxs_hist_destroy(hist);
    }
  }
  return result;
}


/** One interval, ordered by where it begins. */
typedef struct span_t {
  double begin, end;
} span_t;


static int span_cmp(const void* a, const void* b, void* ctx)
{
  const double x = ((const span_t*)a)->begin, y = ((const span_t*)b)->begin;
  LIBXS_UNUSED(ctx);
  return (x < y) ? -1 : ((y < x) ? 1 : 0);
}


/**
 * Union of intervals, computed the obvious way over every retained sample. The
 * sort is the library's own rather than a hand-rolled one: it is not what this
 * checks, it is covered by its own tests, and it is one less loop for a compiler
 * to get wrong - icx 2026.1.1 at -O2 miscompiled the insertion sort that stood
 * here, returning one interval's length instead of the union.
 */
static double union_ref(const double begin[], const double end[], int n)
{
  span_t span[512];
  double total = 0, cb, ce;
  int i;
  for (i = 0; i < n; ++i) {
    span[i].begin = begin[i];
    span[i].end = end[i];
  }
  libxs_sort(span, n, sizeof(span_t), span_cmp, NULL /*ctx*/);
  cb = span[0].begin;
  ce = span[0].end;
  for (i = 1; i < n; ++i) {
    if (span[i].begin > ce) { /* a gap: the open segment is complete */
      total += ce - cb;
      cb = span[i].begin;
      ce = span[i].end;
    }
    else if (span[i].end > ce) ce = span[i].end;
  }
  return total + (ce - cb);
}


static int test_union_fold(void)
{
  /**
   * The union against a brute-force reference, over shapes that a sum of
   * durations cannot tell apart: disjoint, touching, nested, identical, and
   * heavily overlapping. Sequential intervals must come out equal to their sum,
   * and simultaneous ones must not be counted twice - which is the whole reason
   * the fold exists.
   */
  enum { MAXN = 400, NSEG = 8 };
  const libxs_hist_update_t update[] = { libxs_hist_update_avg,
    libxs_hist_update_avg, libxs_hist_update_avg };
  double begin[MAXN], end[MAXN];
  int shape, nsample, i;
  int result = EXIT_SUCCESS;
  libxs_rng_set_seed(777u); /* reproducible */
  for (shape = 0; shape < 6 && EXIT_SUCCESS == result; ++shape) {
    for (nsample = 1; nsample <= MAXN && EXIT_SUCCESS == result; nsample += 47) {
      libxs_span_t* const span = libxs_span_create(NSEG);
      libxs_hist_t* const hist = libxs_hist_create(4/*nbuckets*/, 3/*nvals*/, update,
        libxs_hist_fold_union, span);
      libxs_hist_info_t info;
      double got, want;
      int inexact = 0;
      if (NULL == hist || NULL == span) {
        libxs_hist_destroy(hist);
        libxs_span_destroy(span);
        result = EXIT_FAILURE;
        break;
      }
      for (i = 0; i < nsample; ++i) {
        double v[3];
        switch (shape) {
          case 0: /* strictly sequential with gaps: union == sum */
            begin[i] = 10.0 * i;
            end[i] = 10.0 * i + 4.0;
            break;
          case 1: /* back to back: union == sum, and one segment */
            begin[i] = 10.0 * i;
            end[i] = 10.0 * (i + 1);
            break;
          case 2: /* all simultaneous: union == one interval */
            begin[i] = 0.0;
            end[i] = 100.0;
            break;
          case 3: /* two streams in lockstep: union == half the sum */
            begin[i] = 10.0 * (i / 2);
            end[i] = 10.0 * (i / 2) + 10.0;
            break;
          case 4: /* advancing with jitter and overlap, as completions arrive */
            begin[i] = 5.0 * i + 2.0 * libxs_rng_f64();
            end[i] = begin[i] + 12.0 * libxs_rng_f64();
            break;
          default: /* uniformly random, which reaches back arbitrarily far */
            begin[i] = 1000.0 * libxs_rng_f64();
            end[i] = begin[i] + 20.0 * libxs_rng_f64();
            break;
        }
        v[0] = end[i] - begin[i];
        v[1] = begin[i];
        v[2] = end[i];
        libxs_hist_push(NULL, hist, v);
      }
      libxs_hist_query(NULL, hist, &info);
      got = libxs_span_total(span, &inexact);
      want = union_ref(begin, end, nsample);
      /**
       * Exact while nothing reached back past what the fold had to retire, and
       * an upper bound otherwise: an interval that overlaps retired time is
       * added whole, because that overlap can no longer be subtracted. It must
       * never come out below the true union, which is what makes a ratio of sum
       * over it a lower bound on concurrency in either case.
       */
      if (0 == inexact) {
        if (fabs(got - want) > 1E-6 * LIBXS_MAX(want, 1.0)) {
          FPRINTF(stderr, "ERROR line #%i: shape=%i n=%i: union=%f (expected %f)\n",
            __LINE__, shape, nsample, got, want);
          result = EXIT_FAILURE;
        }
      }
      else if (got < want - 1E-6 * LIBXS_MAX(want, 1.0)) {
        FPRINTF(stderr, "ERROR line #%i: shape=%i n=%i: union %f below %f with %i inexact\n",
          __LINE__, shape, nsample, got, want, inexact);
        result = EXIT_FAILURE;
      }
      /**
       * The arrival order a completion callback produces must not be the
       * inexact case: if it is, the retained window is too small to be useful.
       */
      if (EXIT_SUCCESS == result && 4 == shape && 0 != inexact) {
        FPRINTF(stderr, "ERROR line #%i: n=%i: %i inexact on in-order arrivals\n",
          __LINE__, nsample, inexact);
        result = EXIT_FAILURE;
      }
      libxs_hist_destroy(hist);
      libxs_span_destroy(span);
    }
  }
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
  if (EXIT_SUCCESS != test_query_bimodal()) {
    FPRINTF(stderr, "FAILED: test_query_bimodal\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_mode_skewed()) {
    FPRINTF(stderr, "FAILED: test_mode_skewed\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_mode_uniform()) {
    FPRINTF(stderr, "FAILED: test_mode_uniform\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_mode_empty_null()) {
    FPRINTF(stderr, "FAILED: test_mode_empty_null\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_commit_out_of_order()) {
    FPRINTF(stderr, "FAILED: test_commit_out_of_order\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_commit_counts_every_sample()) {
    FPRINTF(stderr, "FAILED: test_commit_counts_every_sample\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_running_sum()) {
    FPRINTF(stderr, "FAILED: test_running_sum\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_median_outlier()) {
    FPRINTF(stderr, "FAILED: test_median_outlier\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_oracle_exact()) {
    FPRINTF(stderr, "FAILED: test_oracle_exact\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_invariants_random()) {
    FPRINTF(stderr, "FAILED: test_invariants_random\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_oracle_multifold()) {
    FPRINTF(stderr, "FAILED: test_oracle_multifold\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_query_interleaved()) {
    FPRINTF(stderr, "FAILED: test_query_interleaved\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS != test_union_fold()) {
    FPRINTF(stderr, "FAILED: test_union_fold\n");
    result = EXIT_FAILURE;
  }

  return result;
}
