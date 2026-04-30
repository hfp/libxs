/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_str.h>
#include <libxs_utils.h>

#include <ctype.h>


LIBXS_API const char* libxs_stristrn(const char a[], const char b[], size_t maxlen)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b && 0 != maxlen) {
    do {
      if (tolower(*a) != tolower(*b)) ++a;
      else {
        const char* const start = a;
        const char* c = b;
        size_t i = 0;
        result = a;
        while ('\0' != c[++i] && i != maxlen && '\0' != *++a) {
          if (tolower(*a) != tolower(c[i])) {
            result = NULL;
            break;
          }
        }
        if ('\0' == c[i] || i == maxlen) break; /* full match */
        result = NULL;
        a = start + 1; /* backtrack past match start */
      }
    } while ('\0' != *a);
  }
  return result;
}


LIBXS_API const char* libxs_stristr(const char a[], const char b[])
{
  return libxs_stristrn(a, b, (size_t)-1);
}


LIBXS_API int libxs_strimatch(const char a[], const char b[], const char delims[], int* count)
{
  int result = 0, na = 0, nb = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char* const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char *c, *tmp;
    char s[2] = {'\0'};
    size_t m, n;
    for (;;) {
      while (*s = *b, NULL != strpbrk(s, sep)) ++b; /* left-trim */
      if ('\0' != *b && '[' != *b) ++nb; /* count words */
      else break;
      tmp = b;
      while ('\0' != *tmp && (*s = *tmp, NULL == strpbrk(s, sep))) ++tmp;
      m = tmp - b;
      c = libxs_stristrn(a, b, LIBXS_MIN(1, m));
      if (NULL != c) {
        const char* d = c;
        while ('\0' != *d && (*s = *d, NULL == strpbrk(s, sep))) ++d;
        n = d - c;
        if (1 >= n || NULL != libxs_stristrn(c, b, LIBXS_MIN(m, n))) ++result;
      }
      b = tmp;
    }
    for (;;) { /* count number of words */
      while (*s = *a, NULL != strpbrk(s, sep)) ++a; /* left-trim */
      if ('\0' != *a && '[' != *a) ++na; /* count words */
      else break;
      while ('\0' != *a && (*s = *a, NULL == strpbrk(s, sep))) ++a;
    }
    if (na < result) result = na;
  }
  else result = -1;
  if (NULL != count) *count = LIBXS_MAX(na, nb);
  return result;
}


LIBXS_API_INLINE
int internal_libxs_levenshtein(const char* a, int na, const char* b, int nb)
{
  int row[64], i, j;
  if (0 == na) return nb;
  if (0 == nb) return na;
  if (na < nb) { /* ensure nb <= na for O(min) space */
    const char* t = a; a = b; b = t;
    i = na; na = nb; nb = i;
  }
  LIBXS_ASSERT(nb <= 64);
  for (j = 0; j < nb; ++j) row[j] = j + 1;
  for (i = 0; i < na; ++i) {
    const int ca = tolower(a[i]);
    int prev = i;
    for (j = 0; j < nb; ++j) {
      const int cost = (ca != tolower(b[j]));
      int val = prev + cost; /* substitution */
      if (row[j] + 1 < val) val = row[j] + 1; /* deletion */
      if ((j > 0 ? row[j - 1] : i + 1) + 1 < val) val = (j > 0 ? row[j - 1] : i + 1) + 1; /* insertion */
      prev = row[j];
      row[j] = val;
    }
  }
  return row[nb - 1];
}


LIBXS_API int libxs_stridist(const char a[], const char b[])
{
  if (NULL != a && NULL != b) {
    return internal_libxs_levenshtein(a, (int)strlen(a), b, (int)strlen(b));
  }
  return -1;
}


LIBXS_API int libxs_strisimilar(const char a[], const char b[],
  const char delims[], libxs_strisimilar_t kind, int* order)
{
  int result = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char* const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char* wa[64]; int la[64], na = 0;
    const char* wb[64]; int lb[64], nb = 0;
    int cost[64 * 64], used_a[64], used_b[64];
    int match_ia[64], match_ib[64], nmatched = 0;
    char s[2] = {'\0'};
    int i, j, nmax;
    { /* tokenize A */
      const char* p = a;
      for (;;) {
        while (*s = *p, NULL != strpbrk(s, sep)) ++p;
        if ('\0' == *p || '[' == *p || 64 <= na) break;
        wa[na] = p;
        while ('\0' != *p && (*s = *p, NULL == strpbrk(s, sep))) ++p;
        la[na] = (int)(p - wa[na]);
        ++na;
      }
    }
    { /* tokenize B */
      const char* p = b;
      for (;;) {
        while (*s = *p, NULL != strpbrk(s, sep)) ++p;
        if ('\0' == *p || '[' == *p || 64 <= nb) break;
        wb[nb] = p;
        while ('\0' != *p && (*s = *p, NULL == strpbrk(s, sep))) ++p;
        lb[nb] = (int)(p - wb[nb]);
        ++nb;
      }
    }
    for (i = 0; i < na; ++i) {
      for (j = 0; j < nb; ++j) {
        cost[i * nb + j] = internal_libxs_levenshtein(wa[i], la[i], wb[j], lb[j]);
      }
    }
    for (i = 0; i < na; ++i) used_a[i] = 0;
    for (j = 0; j < nb; ++j) used_b[j] = 0;
    nmax = LIBXS_MIN(na, nb);
    while (nmatched < nmax) { /* greedy minimum-cost matching */
      int best_i = 0, best_j = 0, best_c = (1 << 30);
      for (i = 0; i < na; ++i) {
        if (0 != used_a[i]) continue;
        for (j = 0; j < nb; ++j) {
          if (0 != used_b[j]) continue;
          if (cost[i * nb + j] < best_c) {
            best_c = cost[i * nb + j];
            best_i = i; best_j = j;
          }
        }
      }
      used_a[best_i] = 1; used_b[best_j] = 1;
      match_ia[nmatched] = best_i;
      match_ib[nmatched] = best_j;
      result += best_c;
      ++nmatched;
    }
    if (LIBXS_STRISIMILAR_TWOOPT <= kind) { /* 2-opt refinement */
      for (;;) {
        int improved = 0;
        for (i = 0; i < nmatched; ++i) {
          for (j = i + 1; j < nmatched; ++j) {
            const int old_c = cost[match_ia[i] * nb + match_ib[i]]
                            + cost[match_ia[j] * nb + match_ib[j]];
            const int new_c = cost[match_ia[i] * nb + match_ib[j]]
                            + cost[match_ia[j] * nb + match_ib[i]];
            if (new_c < old_c) {
              const int tmp = match_ib[i];
              match_ib[i] = match_ib[j];
              match_ib[j] = tmp;
              result += new_c - old_c;
              improved = 1;
            }
          }
        }
        if (0 == improved) break;
      }
    }
    for (i = 0; i < na; ++i) { /* unmatched words in A */
      if (0 == used_a[i]) result += la[i];
    }
    for (j = 0; j < nb; ++j) { /* unmatched words in B */
      if (0 == used_b[j]) result += lb[j];
    }
    if (NULL != order) { /* count inversions among matched pairs */
      int inv = 0;
      for (i = 0; i < nmatched; ++i) {
        for (j = i + 1; j < nmatched; ++j) {
          if ((match_ia[i] < match_ia[j]) != (match_ib[i] < match_ib[j])) ++inv;
        }
      }
      *order = inv;
    }
  }
  else {
    result = -1;
    if (NULL != order) *order = 0;
  }
  return result;
}


LIBXS_API size_t libxs_format_value(char buffer[],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base)
{
  const int len = (NULL != scale ? ((int)strlen(scale)) : 0);
  const int m = LIBXS_INTRINSICS_BITSCANBWD64(nbytes) / LIBXS_MAX(base, 1), n = LIBXS_MIN(m, len);
  int i;
  buffer[0] = 0; /* clear */
  LIBXS_ASSERT(NULL != unit && 0 < base);
  for (i = 0; i < n; ++i) nbytes >>= base;
  LIBXS_SNPRINTF(buffer, buffer_size, "%lu %c%s",
    (unsigned long)nbytes, 0 < n ? scale[n-1] : *unit, 0 < n ? unit : "");
  return nbytes;
}
