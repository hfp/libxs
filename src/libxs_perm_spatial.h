/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

typedef struct internal_spatial_pair_t {
  uint64_t code;
  void* value;
} internal_spatial_pair_t;


LIBXS_API_INLINE
int internal_spatial_cmp(const void* a, const void* b)
{
  const internal_spatial_pair_t* pa = (const internal_spatial_pair_t*)a;
  const internal_spatial_pair_t* pb = (const internal_spatial_pair_t*)b;
  return (pa->code > pb->code) - (pa->code < pb->code);
}


LIBXS_API int libxs_spatial_build_codes(libxs_spatial_t* sp,
  const uint64_t codes[], void* const values[], int n)
{
  int result = EXIT_FAILURE;
  if (NULL != sp) {
    sp->codes = NULL;
    sp->values = NULL;
    sp->n = 0;
    if (NULL != codes && NULL != values && 0 < n) {
      internal_spatial_pair_t* pairs = (internal_spatial_pair_t*)libxs_malloc(
        NULL, (size_t)n * sizeof(internal_spatial_pair_t), LIBXS_MALLOC_AUTO);
      if (NULL != pairs) {
        int i;
        for (i = 0; i < n; ++i) {
          pairs[i].code = codes[i];
          pairs[i].value = values[i];
        }
        qsort(pairs, (size_t)n, sizeof(internal_spatial_pair_t),
          internal_spatial_cmp);
        sp->codes = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
        sp->values = (void**)malloc((size_t)n * sizeof(void*));
        if (NULL != sp->codes && NULL != sp->values) {
          for (i = 0; i < n; ++i) {
            sp->codes[i] = pairs[i].code;
            sp->values[i] = pairs[i].value;
          }
          sp->n = n;
          result = EXIT_SUCCESS;
        }
        else {
          free(sp->codes);
          free(sp->values);
          sp->codes = NULL;
          sp->values = NULL;
        }
        libxs_free(pairs);
      }
    }
  }
  return result;
}


LIBXS_API int libxs_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* registry)
{
  int result = EXIT_FAILURE;
  libxs_registry_info_t info;
  if (NULL != sp && NULL != registry) {
    sp->codes = NULL;
    sp->values = NULL;
    sp->n = 0;
    libxs_registry_info(registry, &info);
    if (info.size > 0) {
      uint64_t* codes = (uint64_t*)libxs_malloc(NULL,
        info.size * sizeof(uint64_t), LIBXS_MALLOC_AUTO);
      void** values = (void**)libxs_malloc(NULL,
        info.size * sizeof(void*), LIBXS_MALLOC_AUTO);
      if (NULL != codes && NULL != values) {
        size_t cursor = 0;
        const void* key = NULL;
        void* val;
        int count = 0;
        val = libxs_registry_begin(registry, &key, &cursor);
        while (NULL != val && count < (int)info.size) {
          uint64_t code = 0;
          memcpy(&code, key, 8);
          codes[count] = code;
          values[count] = val;
          ++count;
          val = libxs_registry_next(registry, &key, &cursor);
        }
        result = libxs_spatial_build_codes(sp, codes, values, count);
      }
      libxs_free(codes);
      libxs_free(values);
    }
  }
  return result;
}


LIBXS_API int libxs_spatial_nearest(const libxs_spatial_t* sp,
  uint64_t query_code, int k, void** out_values)
{
  int count = 0, lo, hi, mid, left, right;
  if (NULL != sp && NULL != out_values && 0 != sp->n && 0 < k) {
    lo = 0;
    hi = sp->n - 1;
    while (lo <= hi) {
      mid = (lo + hi) / 2;
      if (sp->codes[mid] < query_code) lo = mid + 1;
      else if (sp->codes[mid] > query_code) hi = mid - 1;
      else { lo = mid; break; }
    }
    if (lo > sp->n - 1) lo = sp->n - 1;

    left = lo - 1;
    right = lo;
    while (count < k && (left >= 0 || right < sp->n)) {
      int pick_left = 0;
      if (left >= 0 && right < sp->n) {
        uint64_t dl = query_code - sp->codes[left];
        uint64_t dr = sp->codes[right] - query_code;
        if (query_code < sp->codes[left]) dl = sp->codes[left] - query_code;
        if (query_code > sp->codes[right]) dr = query_code - sp->codes[right];
        pick_left = (dl <= dr) ? 1 : 0;
      }
      else if (left >= 0) {
        pick_left = 1;
      }
      if (0 != pick_left) {
        out_values[count++] = sp->values[left];
        --left;
      }
      else {
        out_values[count++] = sp->values[right];
        ++right;
      }
    }
  }
  return count;
}


LIBXS_API void libxs_spatial_destroy(libxs_spatial_t* sp)
{
  if (NULL != sp) {
    free(sp->codes);
    free(sp->values);
    sp->codes = NULL;
    sp->values = NULL;
    sp->n = 0;
  }
}
