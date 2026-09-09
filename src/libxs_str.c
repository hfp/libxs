/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_str.h>
#include <libxs/libxs_utils.h>


/**
 * Byte-wise ASCII case folding. The cast is not cosmetic: tolower accepts EOF or
 * a value representable as unsigned char, so passing a negative char is
 * undefined, which is what any byte above 0x7F is on a platform with signed
 * char - and text carrying UTF-8 punctuation or accents is full of them.
 */
LIBXS_API_INLINE int internal_libxs_strilower(char c)
{
  return tolower((unsigned char)c);
}


LIBXS_API int libxs_striequal(const char a[], size_t asize,
  const char b[], size_t bsize)
{
  int result = 0;
  if (NULL != a && NULL != b && asize == bsize) {
    size_t at = 0;
    while (at < asize
      && internal_libxs_strilower(a[at]) == internal_libxs_strilower(b[at]))
    {
      ++at;
    }
    result = (at == asize) ? 1 : 0;
  }
  return result;
}


LIBXS_API const char* libxs_strimem(const char a[], size_t asize,
  const char b[], size_t bsize)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && 0 != bsize && bsize <= asize) {
    const size_t last = asize - bsize;
    size_t at = 0;
    while (at <= last && NULL == result) {
      size_t i = 0;
      while (i < bsize && internal_libxs_strilower(a[at + i])
        == internal_libxs_strilower(b[i]))
      {
        ++i;
      }
      if (i == bsize) result = a + at;
      ++at;
    }
  }
  return result;
}


LIBXS_API const char* libxs_stristrn(const char a[], const char b[], size_t maxlen)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && 0 != maxlen) {
    size_t bsize = 0;
    while (bsize < maxlen && '\0' != b[bsize]) ++bsize;
    result = libxs_strimem(a, strlen(a), b, bsize);
  }
  return result;
}


LIBXS_API const char* libxs_stristr(const char a[], const char b[])
{
  return libxs_stristrn(a, b, (size_t)-1);
}


LIBXS_API const char* libxs_strtoken(const char str[],
  const char delims[], int index, int* length)
{
  const char* result = NULL;
  if (NULL != str) {
    const char* const sep = (NULL != delims && '\0' != *delims) ? delims : LIBXS_DELIMS;
    const char* p = str;
    int i = 0;
    while (i < index && '\0' != *p) {
      if (NULL != strchr(sep, *p)) ++i;
      ++p;
    }
    if (i == index && '\0' != *p) {
      const char* end;
      while (' ' == *p || '\t' == *p) ++p;
      end = p;
      while ('\0' != *end && NULL == strchr(sep, *end)) ++end;
      while (end > p && (' ' == end[-1] || '\t' == end[-1])) --end;
      result = p;
      if (NULL != length) *length = (int)(end - p);
    }
  }
  return result;
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
  int row[64], result, i, j;
  if (0 == na || 0 == nb) result = (0 == na) ? nb : na;
  else {
    if (na < nb) { /* ensure nb <= na for O(min) space */
      const char* t = a; a = b; b = t;
      i = na; na = nb; nb = i;
    }
    LIBXS_ASSERT(nb <= 64);
    for (j = 0; j < nb; ++j) row[j] = j + 1;
    for (i = 0; i < na; ++i) {
      const int ca = internal_libxs_strilower(a[i]);
      int prev = i;
      for (j = 0; j < nb; ++j) {
        const int cost = (ca != internal_libxs_strilower(b[j]));
        int val = prev + cost; /* substitution */
        if (row[j] + 1 < val) val = row[j] + 1; /* deletion */
        if ((j > 0 ? row[j - 1] : i + 1) + 1 < val) val = (j > 0 ? row[j - 1] : i + 1) + 1; /* insertion */
        prev = row[j];
        row[j] = val;
      }
    }
    result = row[nb - 1];
  }
  return result;
}


LIBXS_API int libxs_stridist(const char a[], const char b[])
{
  int result = -1;
  if (NULL != a && NULL != b) {
    result = internal_libxs_levenshtein(a, (int)strlen(a), b, (int)strlen(b));
  }
  return result;
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


LIBXS_API int libxs_stridiff(const char a[], const char b[],
  const char delims[], int tolerance, int* count)
{
  int result = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char* const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char* wa[64]; int la[64], na = 0;
    const char* wb[64]; int lb[64], nb = 0;
    int used[64];
    char s[2] = {'\0'};
    int i, j;
    const char** ws; int* ls; int ns;
    const char** wl; int* ll; int nl;
    { const char* p = a;
      for (;;) {
        while (*s = *p, NULL != strpbrk(s, sep)) ++p;
        if ('\0' == *p || 64 <= na) break;
        wa[na] = p;
        while ('\0' != *p && (*s = *p, NULL == strpbrk(s, sep))) ++p;
        la[na] = (int)(p - wa[na]);
        ++na;
      }
    }
    { const char* p = b;
      for (;;) {
        while (*s = *p, NULL != strpbrk(s, sep)) ++p;
        if ('\0' == *p || 64 <= nb) break;
        wb[nb] = p;
        while ('\0' != *p && (*s = *p, NULL == strpbrk(s, sep))) ++p;
        lb[nb] = (int)(p - wb[nb]);
        ++nb;
      }
    }
    if (na <= nb) { ws = wa; ls = la; ns = na; wl = wb; ll = lb; nl = nb; }
    else { ws = wb; ls = lb; ns = nb; wl = wa; ll = la; nl = na; }
    for (j = 0; j < nl; ++j) used[j] = 0;
    for (i = 0; i < ns; ++i) {
      int best_j = -1, best_d = (1 << 30);
      for (j = 0; j < nl; ++j) {
        int d;
        if (0 != used[j]) continue;
        d = internal_libxs_levenshtein(ws[i], ls[i], wl[j], ll[j]);
        if (d <= tolerance && d < best_d) { best_d = d; best_j = j; }
      }
      if (-1 != best_j) used[best_j] = 1;
      else ++result;
    }
    if (NULL != count) *count = LIBXS_MAX(na, nb);
  }
  else {
    result = -1;
    if (NULL != count) *count = 0;
  }
  return result;
}


LIBXS_API size_t libxs_utf8_size(const unsigned char text[], size_t size,
  size_t pos)
{
  size_t result = 1;
  if (NULL != text && pos < size) {
    const unsigned char lead = text[pos];
    if (0xC0u <= lead && lead < 0xE0u) result = 2;
    else if (0xE0u <= lead && lead < 0xF0u) result = 3;
    else if (0xF0u <= lead && lead < 0xF8u) result = 4;
    if (size - pos < result) result = size - pos;
  }
  return result;
}


LIBXS_API unsigned long libxs_utf8_decode(const unsigned char text[],
  size_t size, int* width)
{
  unsigned long result = 0;
  int span = 1;
  if (NULL != text && 0 < size) {
    result = text[0];
    if (0 == (text[0] & 0x80u)) {
      span = 1;
    }
    else if (0xC0u == (text[0] & 0xE0u) && 1 < size
      && 0x80u == (text[1] & 0xC0u))
    {
      result = ((unsigned long)(text[0] & 0x1Fu) << 6)
        | (unsigned long)(text[1] & 0x3Fu);
      span = 2;
    }
    else if (0xE0u == (text[0] & 0xF0u) && 2 < size
      && 0x80u == (text[1] & 0xC0u) && 0x80u == (text[2] & 0xC0u))
    {
      result = ((unsigned long)(text[0] & 0x0Fu) << 12)
        | ((unsigned long)(text[1] & 0x3Fu) << 6)
        | (unsigned long)(text[2] & 0x3Fu);
      span = 3;
    }
    else if (0xF0u == (text[0] & 0xF8u) && 3 < size
      && 0x80u == (text[1] & 0xC0u) && 0x80u == (text[2] & 0xC0u)
      && 0x80u == (text[3] & 0xC0u))
    {
      result = ((unsigned long)(text[0] & 0x07u) << 18)
        | ((unsigned long)(text[1] & 0x3Fu) << 12)
        | ((unsigned long)(text[2] & 0x3Fu) << 6)
        | (unsigned long)(text[3] & 0x3Fu);
      span = 4;
    }
  }
  if (NULL != width) *width = span;
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
