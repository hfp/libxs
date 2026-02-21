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
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  int result = EXIT_SUCCESS;
  /* basic create/destroy cycle */
  libxs_hist_create(&hist, 4/*nbuckets*/, 8/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) {
    FPRINTF(stderr, "ERROR line #%i: hist_create failed\n", __LINE__);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  /* destroy NULL should be safe */
  libxs_hist_destroy(NULL);
  return result;
}


static int test_single_value(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  const int *buckets = NULL;
  const double *vals = NULL;
  double range[2];
  int nbuckets = 0, nvals = 0;
  int result = EXIT_SUCCESS;
  const double value[] = { 42.0 };
  /* nqueue=1: a single value completes the fill phase immediately */
  libxs_hist_create(&hist, 1/*nbuckets*/, 1/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_set(NULL/*lock*/, hist, value);
  libxs_hist_get(NULL/*lock*/, hist, &buckets, &nbuckets, range, &vals, &nvals);
  if (1 != nbuckets || 1 != nvals || NULL == buckets || NULL == vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected get result\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(vals[0] - 42.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected value 42.0, got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (fabs(range[0] - 42.0) > TOLERANCE || fabs(range[1] - 42.0) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range mismatch [%f, %f]\n", __LINE__, range[0], range[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_fill_phase_range(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  double range[2];
  int result = EXIT_SUCCESS;
  const double v1[] = { 10.0 }, v2[] = { 20.0 }, v3[] = { 15.0 };
  /* nqueue=3: three values fill the queue and determine range */
  libxs_hist_create(&hist, 4/*nbuckets*/, 3/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  libxs_hist_set(NULL/*lock*/, hist, v1);
  libxs_hist_set(NULL/*lock*/, hist, v2);
  libxs_hist_set(NULL/*lock*/, hist, v3);
  libxs_hist_get(NULL/*lock*/, hist, NULL, NULL, range, NULL, NULL);
  if (fabs(range[0] - 10.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected min=10.0, got %f\n", __LINE__, range[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(range[1] - 20.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected max=20.0, got %f\n", __LINE__, range[1]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_bucket_distribution(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_add };
  const int *buckets = NULL;
  int nbuckets = 0, i, total;
  int result = EXIT_SUCCESS;
  const double vmin[] = { 0.0 }, vmax[] = { 100.0 };
  /* nqueue >= nbuckets: full bucket resolution from queued values */
  libxs_hist_create(&hist, 4/*nbuckets*/, 4/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  /* fill phase: queue values to establish range */
  libxs_hist_set(NULL, hist, vmin);
  libxs_hist_set(NULL, hist, vmax);
  { /* fill remaining queue slots */
    const double v33[] = { 33.0 }, v66[] = { 66.0 };
    libxs_hist_set(NULL, hist, v33);
    libxs_hist_set(NULL, hist, v66);
  }
  /* bucket phase: insert values across the range */
  for (i = 0; i < 40; ++i) {
    const double v[] = { 2.5 * i }; /* 0, 2.5, 5, ..., 97.5 */
    libxs_hist_set(NULL, hist, v);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, NULL, NULL);
  if (4 != nbuckets || NULL == buckets) {
    FPRINTF(stderr, "ERROR line #%i: unexpected nbuckets=%i\n", __LINE__, nbuckets);
    result = EXIT_FAILURE;
  }
  /* verify all inserted values are accounted for */
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < nbuckets; ++i) {
      if (buckets[i] < 0) {
        FPRINTF(stderr, "ERROR line #%i: negative bucket[%i]=%i\n", __LINE__, i, buckets[i]);
        result = EXIT_FAILURE;
        break;
      }
      total += buckets[i];
    }
    /* total should account for all 40 bucketed values plus the 4 fill-phase values */
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
  const libxs_hist_update_t update[] = { libxs_hist_add };
  const int *buckets = NULL;
  const double *vals = NULL;
  int nbuckets = 0, nvals = 0;
  int result = EXIT_SUCCESS;
  /* single bucket with nqueue=1 so every subsequent value maps to same bucket */
  libxs_hist_create(&hist, 1/*nbuckets*/, 1/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  { /* fill phase: first value establishes range */
    const double v[] = { 5.0 };
    libxs_hist_set(NULL, hist, v);
  }
  /* trigger get to finalize fill phase */
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, &vals, &nvals);
  if (1 != nbuckets || NULL == vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state\n", __LINE__);
    libxs_hist_destroy(hist);
    return EXIT_FAILURE;
  }
  { /* bucket phase: subsequent values should be accumulated via add */
    const double v1[] = { 5.0 }, v2[] = { 5.0 }, v3[] = { 5.0 };
    libxs_hist_set(NULL, hist, v1);
    libxs_hist_set(NULL, hist, v2);
    libxs_hist_set(NULL, hist, v3);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, &vals, &nvals);
  if (NULL == vals || NULL == buckets) {
    FPRINTF(stderr, "ERROR line #%i: get failed\n", __LINE__);
    result = EXIT_FAILURE;
  }
  /* bucket should have been hit multiple times */
  if (EXIT_SUCCESS == result && buckets[0] < 2) {
    FPRINTF(stderr, "ERROR line #%i: expected bucket count >= 2, got %i\n", __LINE__, buckets[0]);
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
  /* direct test of the avg update function */
  libxs_hist_avg(&a, &b);
  if (fabs(a - 15.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 15.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  /* applying avg again: avg(15, 20) = 17.5 */
  libxs_hist_avg(&a, &b);
  if (fabs(a - 17.5) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 17.5, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_update_add_fn(void)
{
  double a = 3.0;
  const double b = 7.0;
  int result = EXIT_SUCCESS;
  /* direct test of the add update function */
  libxs_hist_add(&a, &b);
  if (fabs(a - 10.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 10.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  libxs_hist_add(&a, &b);
  if (fabs(a - 17.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 17.0, got %f\n", __LINE__, a);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_multiple_values(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg, libxs_hist_add };
  const int *buckets = NULL;
  const double *vals = NULL;
  double range[2];
  int nbuckets = 0, nvals = 0;
  int result = EXIT_SUCCESS;
  /* nvals=2: first value is the key (for bucketing), second is auxiliary data */
  libxs_hist_create(&hist, 2/*nbuckets*/, 2/*nqueue*/, 2/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  { /* fill phase */
    const double e1[] = { 0.0, 100.0 }, e2[] = { 10.0, 200.0 };
    libxs_hist_set(NULL, hist, e1);
    libxs_hist_set(NULL, hist, e2);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, range, &vals, &nvals);
  if (2 != nbuckets || 2 != nvals || NULL == buckets || NULL == vals) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state nbuckets=%i nvals=%i\n",
      __LINE__, nbuckets, nvals);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    if (fabs(range[0] - 0.0) > TOLERANCE || fabs(range[1] - 10.0) > TOLERANCE) {
      FPRINTF(stderr, "ERROR line #%i: range [%f, %f] unexpected\n", __LINE__, range[0], range[1]);
      result = EXIT_FAILURE;
    }
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_print(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  int result = EXIT_SUCCESS;
  /* create and populate a small histogram, then print it */
  libxs_hist_create(&hist, 3/*nbuckets*/, 3/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 1.0 }, v2[] = { 2.0 }, v3[] = { 3.0 };
    libxs_hist_set(NULL, hist, v1);
    libxs_hist_set(NULL, hist, v2);
    libxs_hist_set(NULL, hist, v3);
  }
  /* print to /dev/null (or stderr under debug): just ensure no crash */
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
      libxs_hist_print(ostream, hist, "test_print", prec, NULL/*adjust*/);
#if !defined(_DEBUG)
      fclose(ostream);
#endif
    }
  }
  /* also test print with NULL stream (should not crash) */
  libxs_hist_print(NULL, hist, "null_stream", NULL, NULL);
  libxs_hist_destroy(hist);
  return result;
}


static int test_set_null_hist(void)
{
  int result = EXIT_SUCCESS;
  const double value[] = { 1.0 };
  /* hist_set with NULL hist should be safe (no-op) */
  libxs_hist_set(NULL, NULL, value);
  return result;
}


static int test_get_null_hist(void)
{
  const int *buckets = NULL;
  const double *vals = NULL;
  double range[2] = { -1.0, -1.0 };
  int nbuckets = -1, nvals = -1;
  int result = EXIT_SUCCESS;
  /* hist_get with NULL hist should return zeros/NULLs */
  libxs_hist_get(NULL, NULL, &buckets, &nbuckets, range, &vals, &nvals);
  if (0 != nbuckets || 0 != nvals || NULL != buckets || NULL != vals) {
    FPRINTF(stderr, "ERROR line #%i: get on NULL hist should yield empty\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (fabs(range[0]) > TOLERANCE || fabs(range[1]) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range should be [0,0] for NULL hist\n", __LINE__);
    result = EXIT_FAILURE;
  }
  return result;
}


static int test_many_values_bucketing(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  const int *buckets = NULL;
  int nbuckets = 0, i, total;
  int result = EXIT_SUCCESS;
  /* nqueue >= nbuckets: full resolution */
  libxs_hist_create(&hist, 10/*nbuckets*/, 10/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  { /* fill phase: 10 values spanning [0..200] */
    for (i = 0; i < 10; ++i) {
      const double fv[] = { 200.0 * i / 9 };
      libxs_hist_set(NULL, hist, fv);
    }
  }
  /* bucket phase: insert uniform samples */
  for (i = 0; i < 100; ++i) {
    const double v[] = { 2.0 * i };
    libxs_hist_set(NULL, hist, v);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, NULL, NULL);
  if (10 != nbuckets || NULL == buckets) {
    FPRINTF(stderr, "ERROR line #%i: nbuckets=%i\n", __LINE__, nbuckets);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < nbuckets; ++i) total += buckets[i];
    /* all values during bucket phase + fill-phase values */
    if (total < 10) {
      FPRINTF(stderr, "ERROR line #%i: total=%i < 10\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
    /* verify buckets are non-empty (uniform distribution over 10 buckets) */
    for (i = 0; i < nbuckets && EXIT_SUCCESS == result; ++i) {
      if (0 == buckets[i]) {
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
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  const int *buckets = NULL;
  const double *vals = NULL;
  double range[2];
  int nbuckets = 0, nvals = 0, i, total;
  int result = EXIT_SUCCESS;
  /* fewer queued values than requested buckets: get commits shorter queue,
   * nbuckets is reduced to the number of queued items.
   */
  libxs_hist_create(&hist, 8/*nbuckets*/, 3/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  { /* insert only 3 values into 8-bucket histogram */
    const double v1[] = { 10.0 }, v2[] = { 50.0 }, v3[] = { 90.0 };
    libxs_hist_set(NULL, hist, v1);
    libxs_hist_set(NULL, hist, v2);
    libxs_hist_set(NULL, hist, v3);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, range, &vals, &nvals);
  /* nbuckets reduced to n=3 (only 3 values were queued) */
  if (3 != nbuckets) {
    FPRINTF(stderr, "ERROR line #%i: expected 3 buckets, got %i\n", __LINE__, nbuckets);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (NULL == buckets || NULL == vals || 1 != nvals)) {
    FPRINTF(stderr, "ERROR line #%i: unexpected NULL or nvals=%i\n", __LINE__, nvals);
    result = EXIT_FAILURE;
  }
  /* exactly 3 values committed: total count across all buckets must be 3 */
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < nbuckets; ++i) total += buckets[i];
    if (3 != total) {
      FPRINTF(stderr, "ERROR line #%i: total=%i (expected 3)\n", __LINE__, total);
      result = EXIT_FAILURE;
    }
  }
  /* range must reflect the inserted values */
  if (EXIT_SUCCESS == result && (fabs(range[0] - 10.0) > TOLERANCE || fabs(range[1] - 90.0) > TOLERANCE)) {
    FPRINTF(stderr, "ERROR line #%i: range [%f, %f] unexpected\n", __LINE__, range[0], range[1]);
    result = EXIT_FAILURE;
  }
  /* further inserts work against the committed (reduced) bucket set */
  {
    const double v4[] = { 30.0 }, v5[] = { 70.0 };
    libxs_hist_set(NULL, hist, v4);
    libxs_hist_set(NULL, hist, v5);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, NULL, NULL);
  if (EXIT_SUCCESS == result) {
    for (total = 0, i = 0; i < nbuckets; ++i) total += buckets[i];
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
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  const int *buckets = NULL;
  const double *vals = NULL;
  int nbuckets = 0, nvals = 0;
  int result = EXIT_SUCCESS;
  /* 1 bucket, nqueue=4: all 4 values land in the same bucket at commit.
   * Arithmetic mean of {10, 20, 30, 40} = 25.0
   */
  libxs_hist_create(&hist, 1/*nbuckets*/, 4/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 10.0 }, v2[] = { 20.0 }, v3[] = { 30.0 }, v4[] = { 40.0 };
    libxs_hist_set(NULL, hist, v1);
    libxs_hist_set(NULL, hist, v2);
    libxs_hist_set(NULL, hist, v3);
    libxs_hist_set(NULL, hist, v4);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, &vals, &nvals);
  if (1 != nbuckets || NULL == vals || NULL == buckets) {
    FPRINTF(stderr, "ERROR line #%i: unexpected state\n", __LINE__);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && fabs(vals[0] - 25.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 25.0 (arithmetic mean), got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 4 != buckets[0]) {
    FPRINTF(stderr, "ERROR line #%i: expected count=4, got %i\n", __LINE__, buckets[0]);
    result = EXIT_FAILURE;
  }
  libxs_hist_destroy(hist);
  return result;
}


static int test_hybrid_avg_then_slide(void)
{
  libxs_hist_t* hist = NULL;
  const libxs_hist_update_t update[] = { libxs_hist_avg };
  const int *buckets = NULL;
  const double *vals = NULL;
  int nbuckets = 0, nvals = 0;
  int result = EXIT_SUCCESS;
  /* 1 bucket, nqueue=2: commit produces arithmetic mean,
   * then subsequent inserts slide (exponential moving average).
   * Queue: {10, 30} -> commit: mean=20.0
   * Slide with 40.0: avg(20, 40) = 30.0
   */
  libxs_hist_create(&hist, 1/*nbuckets*/, 2/*nqueue*/, 1/*nvals*/, update);
  if (NULL == hist) return EXIT_FAILURE;
  {
    const double v1[] = { 10.0 }, v2[] = { 30.0 };
    libxs_hist_set(NULL, hist, v1);
    libxs_hist_set(NULL, hist, v2);
  }
  /* trigger commit */
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, &vals, &nvals);
  if (EXIT_SUCCESS == result && fabs(vals[0] - 20.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 20.0 after commit, got %f\n", __LINE__, vals[0]);
    result = EXIT_FAILURE;
  }
  { /* bucket phase: slide */
    const double v3[] = { 40.0 };
    libxs_hist_set(NULL, hist, v3);
  }
  libxs_hist_get(NULL, hist, &buckets, &nbuckets, NULL, &vals, &nvals);
  if (EXIT_SUCCESS == result && fabs(vals[0] - 30.0) > TOLERANCE) {
    FPRINTF(stderr, "ERROR line #%i: expected 30.0 after slide, got %f\n", __LINE__, vals[0]);
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
  if (EXIT_SUCCESS != test_hybrid_avg_then_slide()) {
    FPRINTF(stderr, "FAILED: test_hybrid_avg_then_slide\n");
    result = EXIT_FAILURE;
  }

  return result;
}
