/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs_mem.h>
#include <libxs_rng.h>

#if !defined(MAXSIZE)
# define MAXSIZE 64
#endif
#if !defined(INSIZE)
# define INSIZE MAXSIZE
#endif


static int dbl_cmp(const void*, const void*);
static double dbl_median(double*, size_t);
static void inject_mismatch(unsigned char*, size_t, size_t, int, int);


int main(int argc, char* argv[])
{
  const int insize = (1 < argc ? atoi(argv[1]) : 0);
  const int incrmt = (2 < argc ? atoi(argv[2]) : 0);
  const int nelems = (3 < argc ? atoi(argv[3]) : 0);
  const int niters = (4 < argc ? atoi(argv[4]) : 5);
  const int elsize = (0 >= insize ? INSIZE : insize);
  const int stride = (0 >= incrmt ? LIBXS_MAX(MAXSIZE, elsize) : LIBXS_MAX(incrmt, elsize));
  const size_t size = (0 >= nelems ? (((size_t)2 << 30/*2 GB*/) / stride) : ((size_t)nelems));
  const size_t nrpt = LIBXS_MAX(niters, 1), nbytes = size * stride, shuffle = libxs_coprime2(size);
  const char *const env_strided = getenv("STRIDED"), *const env_check = getenv("CHECK");
  const char *const env_mismatch = getenv("MISMATCH"), *const env_offset = getenv("OFFSET");
  const int strided = (NULL == env_strided || 0 == *env_strided) ? 0/*default*/ : atoi(env_strided);
  const int check = (NULL == env_check || 0 == *env_check) ? 0/*default*/ : atoi(env_check);
  const int mismatch = (NULL == env_mismatch || 0 == *env_mismatch) ? 0 : atoi(env_mismatch);
  const int offset = (NULL == env_offset || 0 == *env_offset) ? 0 : LIBXS_MAX(atoi(env_offset), 0);
  double d0, *t1, *t2, *t3;
  int result = EXIT_SUCCESS;
  unsigned char *a, *b, *b_base;
  size_t i;

  libxs_init();
  a = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);
  b_base = (unsigned char*)(0 != nbytes ? malloc(nbytes + (size_t)offset) : NULL);
  b = (NULL != b_base ? b_base + offset : NULL);
  t1 = (double*)calloc(nrpt, sizeof(double));
  t2 = (double*)calloc(nrpt, sizeof(double));
  t3 = (double*)calloc(nrpt, sizeof(double));

  if (NULL != a && NULL != b && NULL != t1 && NULL != t2 && NULL != t3) {
    size_t diff = 0, j;
    if (0 != offset) {
      printf("NOTE: buffer b offset by %i byte(s)\n", offset);
    }
    if (0 != mismatch) {
      printf("NOTE: mismatch injected at %s of each element\n",
        1 == mismatch ? "first byte" : 2 == mismatch ? "middle" :
        3 == mismatch ? "last byte" : "random position");
    }
    /* warm-up (demand-page, ramp CPU frequency) */
    libxs_rng_seq(a, nbytes);
    memcpy(b, a, nbytes);
    diff = 0;

    for (i = 0; i < nrpt; ++i) {
      printf("-------------------------------------------------\n");
      /* initialize the data */
      libxs_rng_seq(a, nbytes);
      memcpy(b, a, nbytes);
      inject_mismatch(b, nbytes, stride, elsize, mismatch);
      /* benchmark libxs_diff (always strided) */
      if (elsize < 256) {
        const libxs_timer_tick_t start = libxs_timer_tick();
        if (1 >= strided) {
          for (j = 0; j < nbytes; j += stride) {
            const void *const u = a + j, *const v = b + j;
            diff += libxs_diff(u, v, (unsigned char)elsize);
          }
        }
        else {
          for (j = 0; j < size; ++j) {
            const size_t k = (shuffle * j) % size;
            const void *const u = a + k * stride, *const v = b + k * stride;
            diff += libxs_diff(u, v, (unsigned char)elsize);
          }
        }
        d0 = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < d0) printf("libxs_diff:\t\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        t1[i] = d0;
      }
      else if (0 == i) {
        printf("libxs_diff:\t\tskipped (elsize=%i >= 256)\n", elsize);
      }

      { /* benchmark libxs_memcmp */
        libxs_timer_tick_t start;
        /* reinitialize the data (flush caches) */
        libxs_rng_seq(a, nbytes);
        memcpy(b, a, nbytes);
        inject_mismatch(b, nbytes, stride, elsize, mismatch);
        start = libxs_timer_tick();
        if (stride == elsize && 0 == strided) {
          diff += libxs_memcmp(a, b, nbytes);
        }
        else if (1 == strided) {
          for (j = 0; j < nbytes; j += stride) {
            const void *const u = a + j, *const v = b + j;
            diff += libxs_memcmp(u, v, elsize);
          }
        }
        else {
          for (j = 0; j < size; ++j) {
            const size_t k = (shuffle * j) % size;
            const void *const u = a + k * stride, *const v = b + k * stride;
            diff += libxs_memcmp(u, v, elsize);
          }
        }
        d0 = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < d0) printf("libxs_memcmp:\t\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        t2[i] = d0;
      }

      { /* benchmark stdlib's memcmp */
        libxs_timer_tick_t start;
        /* reinitialize the data (flush caches) */
        libxs_rng_seq(a, nbytes);
        memcpy(b, a, nbytes);
        inject_mismatch(b, nbytes, stride, elsize, mismatch);
        start = libxs_timer_tick();
        if (stride == elsize && 0 == strided) {
          diff += (0 != memcmp(a, b, nbytes));
        }
        else if (1 == strided) {
          for (j = 0; j < nbytes; j += stride) {
            const void *const u = a + j, *const v = b + j;
#if defined(_MSC_VER)
#           pragma warning(push)
#           pragma warning(disable: 6385)
#endif
            diff += (0 != memcmp(u, v, elsize));
#if defined(_MSC_VER)
#           pragma warning(pop)
#endif
          }
        }
        else {
          for (j = 0; j < size; ++j) {
            const size_t k = (shuffle * j) % size;
            const void *const u = a + k * stride, *const v = b + k * stride;
#if defined(_MSC_VER)
#           pragma warning(push)
#           pragma warning(disable: 6385)
#endif
            diff += (0 != memcmp(u, v, elsize));
#if defined(_MSC_VER)
#           pragma warning(pop)
#endif
          }
        }
        d0 = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < d0) printf("stdlib memcmp:\t\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        t3[i] = d0;
      }
    }

    if (1 < nrpt) {
      double m1, m2, m3;
      printf("-------------------------------------------------\n");
      printf("Statistics over %llu iterations (min / median / max)\n", (unsigned long long)nrpt);
      printf("-------------------------------------------------\n");
      m1 = dbl_median(t1, nrpt); /* sorts in-place */
      m2 = dbl_median(t2, nrpt);
      m3 = dbl_median(t3, nrpt);
      if (0 < t1[0]) {
        printf("libxs_diff:\t %7i %7i %7i   (MB/s)\n",
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t1[nrpt - 1])),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * m1)),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t1[0])));
      }
      if (0 < t2[0]) {
        printf("libxs_memcmp:\t %7i %7i %7i   (MB/s)\n",
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t2[nrpt - 1])),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * m2)),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t2[0])));
      }
      if (0 < t3[0]) {
        printf("stdlib memcmp:\t %7i %7i %7i   (MB/s)\n",
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t3[nrpt - 1])),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * m3)),
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * t3[0])));
      }
    }
    if (0 < nrpt) {
      printf("-------------------------------------------------\n");
    }

    if (0 == mismatch && 0 != diff) {
      fprintf(stderr, "ERROR: benchmark detected spurious difference!\n");
      result = EXIT_FAILURE;
    }
    if (0 != check) { /* validation */
      size_t k;
      srand(42); /* deterministic seed for reproducibility */
      diff = 0; /* reset for validation */
      for (i = 0; i < nrpt; ++i) {
        for (j = 0; j < nbytes; j += stride) {
          unsigned char *const u = a + j, *const v = b + j;
          for (k = 0; k < 2; ++k) {
            const int r = rand() % elsize;
#if defined(_MSC_VER)
#           pragma warning(push)
#           pragma warning(disable: 6385)
#endif
            if (0 != memcmp(u, v, elsize)) {
              if (elsize < 256 && 0 == libxs_diff(u, v, (unsigned char)elsize)) ++diff;
              if (0 == libxs_memcmp(u, v, elsize)) ++diff;
            }
            else {
              if (elsize < 256 && 0 != libxs_diff(u, v, (unsigned char)elsize)) ++diff;
              if (0 != libxs_memcmp(u, v, elsize)) ++diff;
            }
#if defined(_MSC_VER)
#           pragma warning(pop)
#endif
            /* inject difference into a or b */
            if (0 != (rand() & 1)) {
              u[r] = (unsigned char)(rand() % 256);
            }
            else {
              v[r] = (unsigned char)(rand() % 256);
            }
          }
        }
      }
      if (0 != diff) {
        fprintf(stderr, "ERROR: errors=%i - validation failed!\n", (int)diff);
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  libxs_finalize();
  free(t3);
  free(t2);
  free(t1);
  free(b_base);
  free(a);

  return result;
}


static int dbl_cmp(const void* x, const void* y)
{
  const double a = *(const double*)x, b = *(const double*)y;
  return (a > b) - (a < b);
}


static double dbl_median(double* v, size_t n)
{
  qsort(v, n, sizeof(double), dbl_cmp);
  if (0 != (n & 1)) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}


static void inject_mismatch(unsigned char* buf, size_t nbytes,
  size_t stride, int elsize, int mismatch)
{
  size_t j;
  if (0 >= mismatch || 0 == nbytes) return;
  for (j = 0; j < nbytes; j += stride) {
    size_t pos;
    switch (mismatch) {
      case 1: pos = 0; break;
      case 2: pos = (size_t)elsize / 2; break;
      case 3: pos = (size_t)elsize - 1; break;
      default: pos = (size_t)(rand() % elsize); break;
    }
    buf[j + pos] ^= 1;
  }
}
