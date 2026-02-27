/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs_rng.h>
#include <libxs_mem.h>
#include <libxs_mhd.h>

/* Fisher-Yates shuffle */
#define SHUFFLE(INOUT, ELEMSIZE, COUNT, NSWAPS) do { \
  char *const data = (char*)(INOUT); \
  size_t i = 0; \
  for (; i < ((COUNT) - 1); ++i) { \
    const size_t j = i + libxs_rng_u32((unsigned int)((COUNT) - i)); \
    LIBXS_ASSERT(i <= j && j < (COUNT)); \
    if (i != j) { \
      LIBXS_MEMSWP( \
        data + (ELEMSIZE) * i, \
        data + (ELEMSIZE) * j, \
        ELEMSIZE); \
      ++(NSWAPS); \
     } \
  } \
} while(0)

#define REDUCE_ADD(TYPE, INPUT, M, N, LO, HI) do { \
  const TYPE *const data = (const TYPE*)(INPUT); \
  size_t i = 0; LIBXS_ASSERT((M) < (N)); \
  for (; i < (M); ++i) (LO) += data[i]; \
  for (; i < (N); ++i) (HI) += data[i]; \
} while(0)

typedef enum redop_enum {
  redop_imbalance,
  redop_mdistance
} redop_enum;

size_t shuffle(void* inout, size_t elemsize, size_t count);
/** Compares the sum of values between the left and the right partition. */
size_t uint_reduce_op(const void* input, size_t elemsize, size_t count,
  redop_enum redop, size_t split);
/** Bubble-Sort the given data and return the number of swap operations. */
size_t uint_bsort_asc(void* inout, size_t elemsize, size_t count);


int main(int argc, char* argv[])
{
  const int nelems = (1 < argc ? atoi(argv[1]) : 0);
  const int insize = (2 < argc ? atoi(argv[2]) : 0);
  const int niters = (3 < argc ? atoi(argv[3]) : 1);
  const int repeat = (4 < argc ? atoi(argv[4]) : 3);
  const int random = (NULL == getenv("RANDOM")
    ? 0 : atoi(getenv("RANDOM")));
  const int split = (NULL == getenv("SPLIT")
    ? 1 : atoi(getenv("SPLIT")));
  const int stats = (NULL == getenv("STATS")
    ? 0 : atoi(getenv("STATS")));
  const size_t elsize = (0 >= insize ? 4 : insize);
  const size_t m = (0 <= niters ? niters : 1);
  const size_t n = (0 >= nelems
    ? (((size_t)64 << 20/*64 MB*/) / elsize)
    : ((size_t)nelems));
  const size_t mm = LIBXS_MAX(m, 1);
  const size_t nbytes = n * elsize;
  void *const data1 = malloc(nbytes);
  void *const data2 = malloc(nbytes);
  int result = EXIT_SUCCESS;

  libxs_init();

  if (NULL != data1 && NULL != data2) {
    size_t a1 = 0, a2 = 0, a3 = 0, b1 = 0, b2 = 0, b3 = 0;
    size_t g1 = 0, g2 = 0, g3 = 0, h1 = 0, h2 = 0, h3 = 0;
    size_t n0 = 0, n1 = 0, n2 = 0, n3 = 0, j;
    double d0, d1 = 0, d2 = 0, d3 = 0;
    const libxs_datatype elemtypes[] = {
      LIBXS_DATATYPE_U64,
      LIBXS_DATATYPE_U32,
      LIBXS_DATATYPE_U16,
      LIBXS_DATATYPE_U8
    };
    libxs_mhd_element_handler_info_t mhdinfo = { 0 };
    libxs_mhd_info_t mhd_write_info = { 2, 1, LIBXS_DATATYPE_UNKNOWN, 0 };
    const size_t nelemtypes = sizeof(elemtypes) / sizeof(*elemtypes);
    const size_t nchannels = 1, mhdsize = n / nchannels;
    size_t shape[2], y = 0, typesize = 0;
    libxs_timer_tick_t start;
    int elemtype = -1, i;

    /* initialize the data */
    if (sizeof(size_t) < elsize) memset(data1, 0, nbytes);
    for (j = 0; j < n; ++j) {
      LIBXS_MEMCPY((char*)data1 + elsize * j, &j,
        LIBXS_MIN(elsize, sizeof(size_t)));
    }

    /* prepare writing MHD-image files */
    y = (size_t)libxs_isqrt_u64(mhdsize);
    shape[0] = mhdsize / y; shape[1] = y;
    for (j = 0; j < nelemtypes; ++j) {
      typesize = LIBXS_TYPESIZE(elemtypes[j]);
      if (elsize == typesize) {
        mhdinfo.hint = LIBXS_MHD_ELEMENT_CONVERSION_MODULUS;
        mhdinfo.type = LIBXS_DATATYPE_U8;
        mhd_write_info.type = elemtypes[j];
        elemtype = (int)j;
        break;
      }
    }
    if (0 > elemtype) {
      printf("---------------------------------------\n");
      printf("Unsupported type for writing MHD-file!\n");
    }

    for (i = 0; i <= repeat && EXIT_SUCCESS == result; ++i) {
      printf("---------------------------------------\n");

      if (EXIT_SUCCESS == result) { /* benchmark RNG-based shuffle routine */
        memcpy(data2, data1, nbytes); n0 = 0;
        start = libxs_timer_tick();
        for (j = 0; j < m; ++j) n0 += shuffle(data2, elsize, n);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (i == repeat) { /* only last iteration */
          if (0 != stats) {
            a1 = uint_reduce_op(data2, elsize, n, redop_mdistance, split);
            b1 = uint_reduce_op(data2, elsize, n, redop_mdistance, split * 2);
            g1 = uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h1 = uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
          }
          if (0 == random) {
            if (0 <= elemtype) {
              result = libxs_mhd_write("shuffle_rng.mhd", NULL/*offset*/, shape, NULL/*pitch*/,
                &mhd_write_info, data2, &mhdinfo, NULL/*handler*/, NULL/*extension_header*/,
                NULL/*extension*/, 0/*extension_size*/);
            }
          }
          else n1 = uint_bsort_asc(data2, elsize, n);
        }
        if (0 < d0) printf("RNG-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d1 += d0; /* ignore first iteration */
      }

      if (EXIT_SUCCESS == result) { /* benchmark in-place shuffle (DS1) */
        memcpy(data2, data1, nbytes);
        start = libxs_timer_tick();
        libxs_shuffle(data2, elsize, n, NULL, &m);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (i == repeat) { /* only last iteration */
          if (0 != stats) {
            a2 = uint_reduce_op(data2, elsize, n, redop_mdistance, split);
            b2 = uint_reduce_op(data2, elsize, n, redop_mdistance, split * 2);
            g2 = uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h2 = uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
          }
          if (0 == random) {
            if (0 <= elemtype) {
              result = libxs_mhd_write("shuffle_ds1.mhd", NULL/*offset*/, shape, NULL/*pitch*/,
                &mhd_write_info, data2, &mhdinfo, NULL/*handler*/, NULL/*extension_header*/,
                NULL/*extension*/, 0/*extension_size*/);
            }
          }
          else n2 = uint_bsort_asc(data2, elsize, n);
        }
        if (0 < d0) printf("DS1-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d2 += d0; /* ignore first iteration */
      }

      if (EXIT_SUCCESS == result) { /* benchmark out-of-place shuffle (DS2) */
        start = libxs_timer_tick();
        libxs_shuffle2(data2, data1, elsize, n, NULL, &m);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (i == repeat) { /* only last iteration */
          if (0 != stats) {
            a3 = uint_reduce_op(data2, elsize, n, redop_mdistance, split);
            b3 = uint_reduce_op(data2, elsize, n, redop_mdistance, split * 2);
            g3 = uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h3 = uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
          }
          if (0 == random) {
            if (0 <= elemtype) {
              result = libxs_mhd_write("shuffle_ds2.mhd", NULL/*offset*/, shape, NULL/*pitch*/,
                &mhd_write_info, data2, &mhdinfo, NULL/*handler*/, NULL/*extension_header*/,
                NULL/*extension*/, 0/*extension_size*/);
            }
          }
          else n3 = uint_bsort_asc(data2, elsize, n);
        }
        if (0 < d0) printf("DS2-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d3 += d0; /* ignore first iteration */
      }
    }

    if (1 < repeat && EXIT_SUCCESS == result) {
      const unsigned long long nn = (n * n - n + 3) / 4;
      printf("---------------------------------------\n");
      printf("Arithmetic average of %i iterations\n", repeat);
      printf("---------------------------------------\n");
      d1 /= repeat; d2 /= repeat; d3 /= repeat;
      if (0 < d1) {
        printf("RNG-shuffle: %.8f s (%i MB/s)\n", d1,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
        if (0 != stats) {
          printf("             dst%i=%llu%% dst%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b1, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a1, n));
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h1, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g1, n));
        }
        if (0 != random) {
          n0 = (nn * n - 1 + LIBXS_MIN(LIBXS_MIN(n1, nn) * n, n0 * nn)
            * 100ULL) / (nn * n);
          printf("             rand=%llu%%\n", (unsigned long long)n0);
        }
      }
      if (0 < d2) {
        printf("DS1-shuffle: %.8f s (%i MB/s)\n", d2,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
        if (0 != stats) {
          printf("             dst%i=%llu%% dst%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b2, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a2, n));
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h2, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g2, n));
        }
        if (0 != random) {
          printf("             rand=%llu%%\n",
            (unsigned long long)LIBXS_UPDIV(100ULL * LIBXS_MIN(n2, nn), nn));
        }
      }
      if (0 < d3) {
        printf("DS2-shuffle: %.8f s (%i MB/s)\n", d3,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
        if (0 != stats) {
          printf("             dst%i=%llu%% dst%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b3, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a3, n));
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h3, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g3, n));
        }
        if (0 != random) {
          printf("             rand=%llu%%\n",
            (unsigned long long)LIBXS_UPDIV(100ULL * LIBXS_MIN(n3, nn), nn));
        }
      }
    }
    if (0 < repeat) {
      printf("---------------------------------------\n");
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  libxs_finalize();
  free(data1);
  free(data2);

  return result;
}


size_t shuffle(void* inout, size_t elemsize, size_t count) {
  size_t result = 0;
  if (1 < count) {
    switch (elemsize) {
      case 8:   SHUFFLE(inout, 8, count, result); break;
      case 4:   SHUFFLE(inout, 4, count, result); break;
      case 2:   SHUFFLE(inout, 2, count, result); break;
      case 1:   SHUFFLE(inout, 1, count, result); break;
      default:  SHUFFLE(inout, elemsize, count, result);
    }
  }
  return result;
}


size_t uint_reduce_op(const void* input, size_t elemsize, size_t count,
  redop_enum redop, size_t split)
{
  const size_t inner = LIBXS_UPDIV(count, 2);
  unsigned long long lo = 0, hi = 0, result;
  if (1 < split) {
    lo = uint_reduce_op(input, elemsize, inner, redop, split - 1);
    hi = uint_reduce_op((const char*)input + elemsize * inner,
      elemsize, count - inner, redop, split - 1);
    result = LIBXS_UPDIV(lo + hi, 2); /* average */
  }
  else {
    switch (elemsize) {
      case 8: {
        REDUCE_ADD(unsigned long long, input, inner, count, lo, hi);
      } break;
      case 4: {
        REDUCE_ADD(unsigned int, input, inner, count, lo, hi);
      } break;
      case 2: {
        REDUCE_ADD(unsigned short, input, inner, count, lo, hi);
      } break;
      default: {
        REDUCE_ADD(unsigned char, input, elemsize * inner,
          elemsize * count, lo, hi);
      }
    }
    result = lo + hi;
    if (0 < split) {
      unsigned long long n;
      switch (redop) {
        case redop_imbalance: {
          n = LIBXS_UPDIV(LIBXS_DELTA(lo, hi) * count, result);
        } break;
        case redop_mdistance: {
          n = LIBXS_DELTA(result * 2, count * (count - 1)) / 2;
        } break;
        default: n = result;
      }
      /* normalize result relative to length of input */
      if (0 != result) result = LIBXS_UPDIV(n * count, result);
    }
  }
  return (size_t)result;
}


size_t uint_bsort_asc(void* inout, size_t elemsize, size_t count) {
  size_t nswaps = 0; /* count number of swaps */
  if (0 != count) {
    unsigned char *const data = (unsigned char*)inout;
    int swap = 1;
    for (; 0 != swap; --count) {
      size_t i = 0, j, k;
      assert(0 != count);
      for (swap = 0; i < elemsize * (count - 1); i += elemsize) {
        for (j = i + elemsize, k = elemsize; 0 < k; --k) {
          const unsigned char x = data[i + k - 1], y = data[j + k - 1];
          if (x != y) {
            if (x > y) {
              LIBXS_MEMSWP(data + i, data + j, elemsize);
              swap = 1; ++nswaps;
            }
            break;
          }
        }
      }
    }
  }
  return nswaps;
}
