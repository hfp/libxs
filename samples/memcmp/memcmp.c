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
  const int strided = (NULL == env_strided || 0 == *env_strided) ? 0/*default*/ : atoi(env_strided);
  const int check = (NULL == env_check || 0 == *env_check) ? 0/*default*/ : atoi(env_check);
  double d0, d1 = 0, d2 = 0, d3 = 0;
  int result = EXIT_SUCCESS;
  unsigned char *a, *b;
  size_t i;

  libxs_init();
  a = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);
  b = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);

  if (NULL != a && NULL != b) {
    size_t diff = 0, j;
    for (i = 0; i < nrpt; ++i) {
      printf("-------------------------------------------------\n");
      /* initialize the data */
      libxs_rng_seq(a, nbytes);
      memcpy(b, a, nbytes); /* same content */
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
            const void *const u = a + k, *const v = b + k;
            diff += libxs_diff(u, v, (unsigned char)elsize);
          }
        }
        d0 = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < d0) printf("libxs_diff:\t\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
        d1 += d0;
      }

      { /* benchmark libxs_memcmp */
        libxs_timer_tick_t start;
        /* reinitialize the data (flush caches) */
        libxs_rng_seq(a, nbytes);
        memcpy(b, a, nbytes); /* same content */
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
            const void *const u = a + k, *const v = b + k;
            diff += libxs_memcmp(u, v, elsize);
          }
        }
        d0 = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < d0) printf("libxs_memcmp:\t\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
        d2 += d0;
      }

      { /* benchmark stdlib's memcmp */
        libxs_timer_tick_t start;
        /* reinitialize the data (flush caches) */
        libxs_rng_seq(a, nbytes);
        memcpy(b, a, nbytes); /* same content */
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
            const void *const u = a + k, *const v = b + k;
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
        result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
        d3 += d0;
      }
    }

    if (1 < nrpt) {
      printf("-------------------------------------------------\n");
      printf("Arithmetic average of %llu iterations\n", (unsigned long long)nrpt);
      printf("-------------------------------------------------\n");
      d1 /= nrpt; d2 /= nrpt; d3 /= nrpt;
      if (0 < d1) printf("libxs_diff:\t\t%.8f s (%i MB/s)\n", d1,
        (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
      if (0 < d2) printf("libxs_memcmp:\t\t%.8f s (%i MB/s)\n", d2,
        (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
      if (0 < d3) printf("stdlib memcmp:\t\t%.8f s (%i MB/s)\n", d3,
        (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
    }
    if (0 < nrpt) {
      printf("-------------------------------------------------\n");
    }

    if (0 != check) { /* validation */
      size_t k;
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

  free(a);
  free(b);

  return result;
}
