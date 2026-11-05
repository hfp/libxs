/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_timer.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_hash.h>
#include <libxs/libxs_rng.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_gemm.h>
#include <errno.h>
#include <limits.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define WORK_DIM 16

#if !defined(BUBBLE_SORT) && 0
# define BUBBLE_SORT
#endif

/* Fisher-Yates shuffle */
#define SHUFFLE(INOUT, ELEMSIZE, COUNT, NSWAPS) do { \
  char *const data_ = (char*)(INOUT); \
  size_t i_ = 0; \
  for (; i_ < ((COUNT) - 1); ++i_) { \
    const size_t j_ = i_ + libxs_rng_u32((unsigned int)((COUNT) - i_)); \
    LIBXS_ASSERT(i_ <= j_ && j_ < (COUNT)); \
    if (i_ != j_) { \
      LIBXS_MEMSWP( \
        data_ + (ELEMSIZE) * i_, \
        data_ + (ELEMSIZE) * j_, \
        ELEMSIZE); \
      ++(NSWAPS); \
     } \
  } \
} while(0)

#define REDUCE_ADD(TYPE, INPUT, M, N, LO, HI) do { \
  const TYPE *const data_ = (const TYPE*)(INPUT); \
  size_t i_ = 0; LIBXS_ASSERT((M) < (N)); \
  for (; i_ < (M); ++i_) (LO) += data_[i_]; \
  for (; i_ < (N); ++i_) (HI) += data_[i_]; \
} while(0)


typedef enum redop_enum {
  redop_imbalance,
  redop_sumdelta
} redop_enum;

typedef struct work_t {
  size_t n, coprime, offset;
  int gemm;
  int* cost;
  size_t* rowptr;
  unsigned int* column;
  double *value, *x, *output, *reference;
  double a[WORK_DIM * WORK_DIM], b[WORK_DIM * WORK_DIM];
  libxs_gemm_config_t config;
} work_t;

typedef struct worker_t {
  uint64_t units;
  size_t tasks;
  double seconds;
} worker_t;

static const size_t energy_lags[] = { 1, 2, 3, 7, 16, 63 };
static const char* work_methods[] = {
  "blocked", "cyclic", "affine", "table", "dynamic", "guided"
};
static const void* mhd_element_hash_base;

static int shuffle_benchmark(int argc, char* argv[]);
static int work_benchmark(const char* kernel, int argc, char* argv[]);
static int work_integer(const char* name, const char* text,
  int fallback, int lower, int upper, int* value);
static int work_cost(size_t row, size_t count, int pattern,
  int low, int high, int width, int shift, int period);
static double work_item(const work_t* work, size_t row);
static int work_measure(work_t* work, int method, const unsigned int* table,
  worker_t* workers, int* nthreads, double* seconds);
static int uint_energy_report(const void* original, void* scratch,
  size_t elemsize, size_t count, size_t coprime, size_t offset, const size_t* nrepeat);
static double uint_energy(const void* input, size_t elemsize, size_t count, size_t lag);
size_t shuffle(void* inout, size_t elemsize, size_t count);
/** Compares the sum of values between the left and the right partition. */
size_t uint_reduce_op(const void* input, size_t elemsize, size_t count,
  redop_enum redop, size_t split);
/** Mean distance of each value from its position (percent of count), negative if values repeat. */
double uint_displacement(const void* input, size_t elemsize, size_t count);
#if defined(BUBBLE_SORT)
/** Bubble-Sort the given data and return the number of swap operations. */
size_t uint_bsort_asc(void* inout, size_t elemsize, size_t count);
#else
/** Count inversions via merge-sort (O(n log n)); data is sorted ascending. */
size_t uint_msort_inversions(void* inout, size_t elemsize, size_t count);
#endif

static int mhd_element_hash(void* dst,
  const libxs_mhd_element_handler_info_t* dst_info, libxs_data_t src_type,
  const void* src, const void* src_min, const void* src_max,
  size_t index, void* context);


int main(int argc, char* argv[])
{
  const char* kernel = getenv("LIBXS_SHUFFLE_WORK");
  int result;
  libxs_init();
  result = (NULL == kernel ? shuffle_benchmark(argc, argv)
    : work_benchmark(kernel, argc, argv));
  libxs_finalize();
  return result;
}


static int shuffle_benchmark(int argc, char* argv[])
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
  const double bias = (NULL == getenv("BIAS")
    ? 0.0 : atof(getenv("BIAS")));
  const size_t offset = (NULL == getenv("OFFSET")
    ? 0 : (size_t)atol(getenv("OFFSET")));
  const size_t elsize = (0 >= insize ? 4 : insize);
  const size_t m = (0 < niters ? niters : 1);
  const size_t n = (0 >= nelems
    ? (((size_t)64 << 20/*64 MB*/) / elsize)
    : ((size_t)nelems));
  const size_t coprime = libxs_coprime_bias(n, bias);
  const size_t mm = LIBXS_MAX(m, 1);
  const size_t nbytes = n * elsize;
  void *const data1 = malloc(nbytes);
  void *const data2 = malloc(nbytes);
  int result = EXIT_SUCCESS;

  if (NULL != data1 && NULL != data2) {
    size_t a1 = 0, a2 = 0, a3 = 0, b1 = 0, b2 = 0, b3 = 0;
    size_t g1 = 0, g2 = 0, g3 = 0, h1 = 0, h2 = 0, h3 = 0;
    size_t n1 = 0, n2 = 0, n3 = 0, j;
    double d0, d1 = 0, d2 = 0, d3 = 0, p1 = 0, p2 = 0, p3 = 0;
    const libxs_data_t elemtypes[] = {
      LIBXS_DATATYPE_U64,
      LIBXS_DATATYPE_U32,
      LIBXS_DATATYPE_U16,
      LIBXS_DATATYPE_U8
    };
    libxs_mhd_element_handler_info_t mhdinfo;
    libxs_mhd_write_info_t mhd_winfo;
    const size_t nelemtypes = sizeof(elemtypes) / sizeof(*elemtypes);
    const size_t nchannels = 1, mhdsize = n / nchannels;
    size_t shape[2], y = 0, typesize = 0;
    libxs_timer_tick_t start;
    int elemtype = -1, i;

    /* initialize the data */
    LIBXS_MEMZERO(&mhd_winfo);
    LIBXS_MEMZERO(&mhdinfo);
    if (sizeof(size_t) < elsize) memset(data1, 0, nbytes);
    for (j = 0; j < n; ++j) {
      LIBXS_MEMCPY((char*)data1 + elsize * j, &j,
        LIBXS_MIN(elsize, sizeof(size_t)));
    }

    /**
     * Prepare writing PNG-image files.
     * A 1D coprime shuffle has stride C ~ sqrt(N). Reshaping to a
     * sqrt(N) x sqrt(N) image creates diagonal ghost patterns because
     * the stride and image width are commensurate. To break this, we
     * apply an independent 2D coprime scatter: the linear pixel index
     * i is mapped to row = (Cr * i) mod H, col = (Cc * i) mod W with
     * Cr coprime to H and Cc coprime to W. This redistributes the 1D
     * periodicity across both image axes independently.
     */
    y = (size_t)libxs_isqrt_u64(mhdsize);
    if (0 == y) y = 1;
    shape[0] = mhdsize / y; shape[1] = y;
    for (j = 0; j < nelemtypes; ++j) {
      typesize = LIBXS_TYPESIZE(elemtypes[j]);
      if (elsize == typesize) {
        mhdinfo.hint = LIBXS_MHD_ELEMENT_CONVERSION_DEFAULT;
        mhdinfo.type = LIBXS_DATATYPE_U8;
        mhd_winfo.handler_info = &mhdinfo;
        mhd_winfo.handler = mhd_element_hash;
        elemtype = (int)j;
        break;
      }
    }
    if (0 > elemtype) {
      printf("---------------------------------------\n");
      printf("Unsupported type for writing PNG-file!\n");
    }

    if (0 != bias || 0 != offset || 0 != stats || 0 != random) {
      printf("---------------------------------------\n");
      printf("N=%llu coprime=%llu bias=%.2f offset=%llu\n",
        (unsigned long long)n, (unsigned long long)coprime, bias,
        (unsigned long long)offset);
    }

    for (i = 0; i <= repeat && EXIT_SUCCESS == result; ++i) {
      printf("---------------------------------------\n");

      if (EXIT_SUCCESS == result) { /* benchmark RNG-based shuffle routine */
        memcpy(data2, data1, nbytes);
        start = libxs_timer_tick();
        for (j = 0; j < m; ++j) shuffle(data2, elsize, n);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (0 < i) { /* skip warm-up; average RNG quality metrics */
          if (0 != stats) {
            a1 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split);
            b1 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split * 2);
            g1 += uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h1 += uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
            p1 += uint_displacement(data2, elsize, n);
          }
          if (0 != random) {
#if defined(BUBBLE_SORT)
            n1 += uint_bsort_asc(data2, elsize, n);
#else
            n1 += uint_msort_inversions(data2, elsize, n);
#endif
          }
        }
        if (i == repeat && 0 == random) {
          if (0 <= elemtype) {
            mhd_element_hash_base = data2;
            result = libxs_png_write("shuffle_rng.png", shape, NULL, nchannels,
              elemtypes[elemtype], data2, &mhd_winfo);
          }
        }
        /* bandwidth: 2*n*elsize approximation (RNG skips ~37% of swaps) */
        if (0 < d0) {
          printf("RNG-shuffle: %.8f s (%i MB/s)\n", d0,
            (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        }
        if (0 < i) d1 += d0; /* ignore first iteration */
      }

      if (EXIT_SUCCESS == result) { /* benchmark in-place shuffle (DS1) */
        memcpy(data2, data1, nbytes);
        start = libxs_timer_tick();
        libxs_shuffle(data2, elsize, n, &coprime, offset, &m);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (0 < i) {
          if (0 != stats) {
            a2 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split);
            b2 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split * 2);
            g2 += uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h2 += uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
            p2 += uint_displacement(data2, elsize, n);
          }
          if (0 != random) {
#if defined(BUBBLE_SORT)
            n2 += uint_bsort_asc(data2, elsize, n);
#else
            n2 += uint_msort_inversions(data2, elsize, n);
#endif
          }
        }
        if (i == repeat && 0 == random) {
          if (0 <= elemtype) {
            mhd_element_hash_base = data2;
            result = libxs_png_write("shuffle_ds1.png", shape, NULL, nchannels,
              elemtypes[elemtype], data2, &mhd_winfo);
          }
        }
        if (0 < d0) {
          printf("DS1-shuffle: %.8f s (%i MB/s)\n", d0,
            (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        }
        if (0 < i) d2 += d0; /* ignore first iteration */
      }

      if (EXIT_SUCCESS == result) { /* benchmark out-of-place shuffle (DS2) */
        memset(data2, 0, nbytes); /* equalize page state (cf. DS1 memcpy) */
        start = libxs_timer_tick();
        libxs_shuffle2(data2, data1, elsize, n, &coprime, offset, &m);
        d0 = libxs_timer_duration(start, libxs_timer_tick()) / mm;
        if (0 < i) {
          if (0 != stats) {
            a3 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split);
            b3 += uint_reduce_op(data2, elsize, n, redop_sumdelta, split * 2);
            g3 += uint_reduce_op(data2, elsize, n, redop_imbalance, split);
            h3 += uint_reduce_op(data2, elsize, n, redop_imbalance, split * 2);
            p3 += uint_displacement(data2, elsize, n);
          }
          if (0 != random) {
#if defined(BUBBLE_SORT)
            n3 += uint_bsort_asc(data2, elsize, n);
#else
            n3 += uint_msort_inversions(data2, elsize, n);
#endif
          }
        }
        if (i == repeat && 0 == random) {
          if (0 <= elemtype) {
            mhd_element_hash_base = data2;
            result = libxs_png_write("shuffle_ds2.png", shape, NULL, nchannels,
              elemtypes[elemtype], data2, &mhd_winfo);
          }
        }
        if (0 < d0) {
          printf("DS2-shuffle: %.8f s (%i MB/s)\n", d0,
            (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        }
        if (0 < i) d3 += d0; /* ignore first iteration */
      }
    }

    if (1 < repeat && EXIT_SUCCESS == result) {
      const unsigned long long nn = (n * n - n + 3) / 4;
      printf("---------------------------------------\n");
      printf("Arithmetic average of %i iterations\n", repeat);
      printf("---------------------------------------\n");
      d1 /= repeat; d2 /= repeat; d3 /= repeat;
      /* average accumulated quality metrics */
      if (0 != stats) {
        a1 /= repeat; b1 /= repeat; g1 /= repeat; h1 /= repeat;
        a2 /= repeat; b2 /= repeat; g2 /= repeat; h2 /= repeat;
        a3 /= repeat; b3 /= repeat; g3 /= repeat; h3 /= repeat;
        p1 /= repeat; p2 /= repeat; p3 /= repeat;
      }
      if (0 != random) { n1 /= repeat; n2 /= repeat; n3 /= repeat; }
      if (0 < d1) {
        printf("RNG-shuffle: %.8f s (%i MB/s)\n", d1,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
        if (0 != stats) {
          printf("             sum%i=%llu%% sum%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b1, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a1, n));
          if (0 <= p1) printf("             disp=%.1f%%\n", p1);
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h1, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g1, n));
        }
        if (0 != random) {
          printf("             rand=%.1f%%\n", 100.0 * (double)n1 / (double)nn);
        }
      }
      if (0 < d2) {
        printf("DS1-shuffle: %.8f s (%i MB/s)\n", d2,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
        if (0 != stats) {
          printf("             sum%i=%llu%% sum%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b2, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a2, n));
          if (0 <= p2) printf("             disp=%.1f%%\n", p2);
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h2, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g2, n));
        }
        if (0 != random) {
          printf("             rand=%.1f%%\n", 100.0 * (double)n2 / (double)nn);
        }
      }
      if (0 < d3) {
        printf("DS2-shuffle: %.8f s (%i MB/s)\n", d3,
          (int)LIBXS_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
        if (0 != stats) {
          printf("             sum%i=%llu%% sum%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * b3, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * a3, n));
          if (0 <= p3) printf("             disp=%.1f%%\n", p3);
          printf("             imb%i=%llu%% imb%i=%llu%%\n",
            split * 2, (unsigned long long)LIBXS_UPDIV(100ULL * h3, n),
            split, (unsigned long long)LIBXS_UPDIV(100ULL * g3, n));
        }
        if (0 != random) {
          printf("             rand=%.1f%%\n", 100.0 * (double)n3 / (double)nn);
        }
      }
    }
    if (0 != stats && EXIT_SUCCESS == result) {
      result = uint_energy_report(data1, data2, elsize, n, coprime, offset, &m);
    }
    if (EXIT_SUCCESS == result) {
      const size_t vmask = (elsize < sizeof(size_t)
        ? (((size_t)1 << (8 * elsize)) - 1) : ~(size_t)0);
      size_t errors = 0;
      libxs_shuffle2(data2, data1, elsize, n, &coprime, offset, NULL);
      libxs_unshuffle2(data1, data2, elsize, n, &coprime, offset, NULL);
      for (j = 0; j < n && 0 == errors; ++j) {
        size_t v = 0;
        LIBXS_MEMCPY(&v, (const char*)data1 + elsize * j,
          LIBXS_MIN(elsize, sizeof(size_t)));
        if (v != (j & vmask)) ++errors;
      }
      printf("---------------------------------------\n");
      printf("Round-trip (shuffle+unshuffle): %s\n",
        0 == errors ? "PASS" : "FAIL");
      for (j = 0; j < n; ++j) {
        LIBXS_MEMCPY((char*)data1 + elsize * j, &j,
          LIBXS_MIN(elsize, sizeof(size_t)));
      }
      errors = 0;
      libxs_shuffle2(data2, data1, elsize, n, &coprime, 42, NULL);
      libxs_unshuffle2(data1, data2, elsize, n, &coprime, 42, NULL);
      for (j = 0; j < n && 0 == errors; ++j) {
        size_t v = 0;
        LIBXS_MEMCPY(&v, (const char*)data1 + elsize * j,
          LIBXS_MIN(elsize, sizeof(size_t)));
        if (v != (j & vmask)) ++errors;
      }
      printf("Round-trip (affine+unaff,b=42): %s\n",
        0 == errors ? "PASS" : "FAIL");
      for (j = 0; j < n; ++j) {
        LIBXS_MEMCPY((char*)data1 + elsize * j, &j,
          LIBXS_MIN(elsize, sizeof(size_t)));
      }
    }
    if (0 > elemtype && 0 == random) {
      printf("NOTE: PNG output skipped (unsupported element size %i)\n",
        (int)elsize);
    }
    if (0 < repeat) {
      printf("---------------------------------------\n");
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  free(data1);
  free(data2);

  return result;
}


static int work_integer(const char* name, const char* text,
  int fallback, int lower, int upper, int* value)
{
  int result = EXIT_SUCCESS;
  long number = fallback;
  if (NULL != text) {
    char* end;
    errno = 0;
    number = strtol(text, &end, 10);
    if (ERANGE == errno || end == text || '\0' != *end) result = EXIT_FAILURE;
  }
  if (lower > number || upper < number) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) *value = (int)number;
  else fprintf(stderr, "ERROR: %s must be an integer in [%d,%d]\n", name, lower, upper);
  return result;
}


static int work_cost(size_t row, size_t count, int pattern,
  int low, int high, int width, int shift, int period)
{
  const size_t position = (row + count - (size_t)shift) % count;
  int result = low;
  if (1 == pattern) {
    result += (int)((uint64_t)(high - low) * position / (count - 1));
  }
  else if ((2 == pattern && position < (size_t)width)
    || (3 == pattern && 0 == position % (size_t)period))
  {
    result = high;
  }
  return result;
}


static double work_item(const work_t* work, size_t row)
{
  double result = 0;
  size_t j;
  if (0 != work->gemm) {
    double c[WORK_DIM * WORK_DIM];
    int repeat;
    memset(c, 0, sizeof(c));
    for (repeat = 0; repeat < work->cost[row]; ++repeat) {
      libxs_gemm_call(&work->config, work->a, work->b, c);
    }
    for (j = 0; j < WORK_DIM * WORK_DIM; ++j) result += c[j];
  }
  else {
    for (j = work->rowptr[row]; j < work->rowptr[row + 1]; ++j) {
      result += work->value[j] * work->x[work->column[j]];
    }
  }
  return result;
}


static int work_measure(work_t* work, int method, const unsigned int* table,
  worker_t* workers, int* nthreads, double* seconds)
{
  const libxs_timer_tick_t start = libxs_timer_tick();
  int result = EXIT_SUCCESS;
  size_t row;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    int tid = 0, team = 1, i;
    uint64_t units = 0;
    size_t tasks = 0;
    libxs_timer_tick_t worker_start;
#if defined(_OPENMP)
    tid = omp_get_thread_num();
    team = omp_get_num_threads();
#pragma omp barrier
#endif
    worker_start = libxs_timer_tick();
    if (0 == tid) *nthreads = team;
    if (4 == method) {
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1) nowait
#endif
      for (i = 0; i < (int)work->n; ++i) {
        work->output[i] = work_item(work, (size_t)i);
        units += work->cost[i];
        ++tasks;
      }
    }
    else if (5 == method) {
#if defined(_OPENMP)
#pragma omp for schedule(guided, 1) nowait
#endif
      for (i = 0; i < (int)work->n; ++i) {
        work->output[i] = work_item(work, (size_t)i);
        units += work->cost[i];
        ++tasks;
      }
    }
    else {
      const size_t width = work->n / (size_t)team;
      const size_t remainder = work->n % (size_t)team;
      const size_t begin = width * (size_t)tid + LIBXS_MIN((size_t)tid, remainder);
      const size_t end = begin + width + ((size_t)tid < remainder ? 1 : 0);
      size_t k = (1 == method ? (size_t)tid : begin);
      const size_t step = (1 == method ? (size_t)team : 1);
      const size_t limit = (1 == method ? work->n : end);
      for (; k < limit; k += step) {
        const size_t task = (2 == method
          ? LIBXS_SHUFFLE_INDEX(k, work->n, work->coprime, work->offset)
          : 3 == method ? table[k] : k);
        work->output[task] = work_item(work, task);
        units += work->cost[task];
        ++tasks;
      }
    }
    workers[tid].units = units;
    workers[tid].tasks = tasks;
    workers[tid].seconds = libxs_timer_duration(worker_start, libxs_timer_tick());
  }
  *seconds = libxs_timer_duration(start, libxs_timer_tick());
  for (row = 0; row < work->n && EXIT_SUCCESS == result; ++row) {
    if (work->output[row] != work->reference[row]) {
      fprintf(stderr, "ERROR: %s result mismatch at task %llu\n",
        work_methods[method], (unsigned long long)row);
      result = EXIT_FAILURE;
    }
  }
  return result;
}


static int work_benchmark(const char* kernel, int argc, char* argv[])
{
  work_t work;
  const char* pattern = getenv("LIBXS_SHUFFLE_PATTERN");
  const char* env_bias = getenv("BIAS");
  unsigned int* table = NULL;
  worker_t* workers = NULL;
  int result = EXIT_SUCCESS, n = 4096, repeat = 3, low = 1, high = 16;
  int width = 1, shift = 0, period = 64, seed = 1, offset = 0, kind = 2;
  int capacity = 1, iteration, order;
  size_t row, j;
  uint64_t total = 0;
  double bias = 0, table_seconds = 0, affine_seconds = 0, gemm_reference = 0;
  libxs_timer_tick_t start;
  LIBXS_MEMZERO(&work);
  if (NULL == pattern) pattern = "hotspot";
  work.gemm = (0 == strcmp(kernel, "gemm"));
  if (0 == strcmp(kernel, "spmv")) { low = 8; high = 128; }
  else if (0 == work.gemm) {
    fprintf(stderr, "ERROR: LIBXS_SHUFFLE_WORK must be spmv or gemm\n");
    result = EXIT_FAILURE;
  }
  if (0 == strcmp(pattern, "uniform")) kind = 0;
  else if (0 == strcmp(pattern, "ramp")) kind = 1;
  else if (0 == strcmp(pattern, "hotspot")) kind = 2;
  else if (0 == strcmp(pattern, "periodic")) kind = 3;
  else {
    fprintf(stderr, "ERROR: LIBXS_SHUFFLE_PATTERN must be uniform, ramp, hotspot, or periodic\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("nelems", 1 < argc ? argv[1] : NULL, n, 2, INT_MAX, &n);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("repeat", 4 < argc ? argv[4] : NULL, repeat, 1, 100000, &repeat);
  }
  if (EXIT_SUCCESS == result) {
    int argument;
    result = work_integer("elemsize in workload mode", 2 < argc ? argv[2] : NULL, 4, 4, 4, &argument);
    if (EXIT_SUCCESS == result) {
      result = work_integer("niters in workload mode", 3 < argc ? argv[3] : NULL, 1, 1, 1, &argument);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_LOW", getenv("LIBXS_SHUFFLE_LOW"), low, 1, 1000000, &low);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_HIGH", getenv("LIBXS_SHUFFLE_HIGH"), high, low, 1000000, &high);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_WIDTH", getenv("LIBXS_SHUFFLE_WIDTH"),
      LIBXS_MAX(n / 8, 1), 1, n, &width);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_SHIFT", getenv("LIBXS_SHUFFLE_SHIFT"), 0, 0, n - 1, &shift);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_PERIOD", getenv("LIBXS_SHUFFLE_PERIOD"),
      LIBXS_MIN(n, period), 1, n, &period);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("LIBXS_SHUFFLE_SEED", getenv("LIBXS_SHUFFLE_SEED"), 1, 0, INT_MAX, &seed);
  }
  if (EXIT_SUCCESS == result) {
    result = work_integer("OFFSET", getenv("OFFSET"), 0, 0, INT_MAX, &offset);
  }
  if (EXIT_SUCCESS == result && NULL != env_bias) {
    char* end;
    errno = 0;
    bias = strtod(env_bias, &end);
    if (ERANGE == errno || end == env_bias || '\0' != *end || !(-1 <= bias && bias <= 1)) {
      fprintf(stderr, "ERROR: BIAS must be a number in [-1,1]\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && 0 != work.gemm) {
    const char* backend = getenv("LIBXS_GEMM_BACKEND");
    if (NULL == backend || 0 != strcmp(backend, "4")) {
      fprintf(stderr, "ERROR: use LIBXS_GEMM_BACKEND=4 for the serial built-in GEMM workload\n");
      result = EXIT_FAILURE;
    }
  }
#if defined(_OPENMP)
  capacity = omp_get_max_threads();
#else
  if (EXIT_SUCCESS == result) fprintf(stderr, "NOTE: OpenMP disabled; using one worker\n");
#endif
  if (EXIT_SUCCESS == result) {
    work.n = (size_t)n;
    start = libxs_timer_tick();
    work.coprime = libxs_coprime_bias(work.n, bias);
    work.offset = (size_t)offset % work.n;
    affine_seconds = libxs_timer_duration(start, libxs_timer_tick());
    if ((size_t)-1 / sizeof(double) <= work.n
      || 0 == work.coprime || ((size_t)-1 - work.offset) / (work.n - 1) < work.coprime)
    {
      fprintf(stderr, "ERROR: workload size exceeds allocation or index arithmetic limits\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    work.cost = (int*)malloc(work.n * sizeof(int));
    work.output = (double*)malloc(work.n * sizeof(double));
    work.reference = (double*)malloc(work.n * sizeof(double));
    table = (unsigned int*)malloc(work.n * sizeof(unsigned int));
    workers = (worker_t*)calloc((size_t)capacity, sizeof(worker_t));
    if (NULL == work.cost || NULL == work.output || NULL == work.reference
      || NULL == table || NULL == workers)
    {
      fprintf(stderr, "ERROR: workload allocation failed\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    for (row = 0; row < work.n; ++row) {
      work.cost[row] = work_cost(row, work.n, kind, low, high, width, shift, period);
      total += work.cost[row];
    }
    if (0 != work.gemm) {
      libxs_gemm_config_t* cached;
      const double one = 1;
      int m, k, col;
      for (j = 0; j < WORK_DIM * WORK_DIM; ++j) {
        work.a[j] = (double)(1 + j % 7) / 8;
        work.b[j] = (double)(1 + j % 5) / 8;
      }
      for (col = 0; col < WORK_DIM; ++col) {
        for (m = 0; m < WORK_DIM; ++m) {
          for (k = 0; k < WORK_DIM; ++k) {
            gemm_reference += work.a[m + k * WORK_DIM] * work.b[k + col * WORK_DIM];
          }
        }
      }
      cached = libxs_gemm_dispatch(LIBXS_DATATYPE_F64, 'N', 'N',
        WORK_DIM, WORK_DIM, WORK_DIM, WORK_DIM, WORK_DIM, WORK_DIM, &one, &one, NULL);
      if (0 == libxs_gemm_config_cpy(&work.config, cached)) {
        fprintf(stderr, "ERROR: GEMM dispatch failed\n");
        result = EXIT_FAILURE;
      }
      for (row = 0; row < work.n; ++row) work.reference[row] = gemm_reference * work.cost[row];
    }
    else if ((uint64_t)((size_t)-1 / sizeof(double)) < total) {
      fprintf(stderr, "ERROR: CSR storage size exceeds allocation limits\n");
      result = EXIT_FAILURE;
    }
    else {
      work.rowptr = (size_t*)malloc((work.n + 1) * sizeof(size_t));
      work.column = (unsigned int*)malloc((size_t)total * sizeof(unsigned int));
      work.value = (double*)malloc((size_t)total * sizeof(double));
      work.x = (double*)malloc(work.n * sizeof(double));
      if (NULL == work.rowptr || NULL == work.column || NULL == work.value || NULL == work.x) {
        fprintf(stderr, "ERROR: CSR allocation failed\n");
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        for (row = 0; row < work.n; ++row) work.x[row] = 1 + (double)(row % 8) / 8;
        work.rowptr[0] = 0;
        for (row = 0; row < work.n; ++row) {
          double expected = 0;
          work.rowptr[row + 1] = work.rowptr[row] + (size_t)work.cost[row];
          for (j = work.rowptr[row]; j < work.rowptr[row + 1]; ++j) {
            const unsigned int column = libxs_hash32((uint64_t)j) % (unsigned int)n;
            work.column[j] = column;
            work.value[j] = 1;
            expected += work.x[column];
          }
          work.reference[row] = expected;
        }
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    libxs_rng_set_seed((unsigned int)seed);
    start = libxs_timer_tick();
    for (row = 0; row < work.n; ++row) table[row] = (unsigned int)row;
    shuffle(table, sizeof(*table), work.n);
    table_seconds = libxs_timer_duration(start, libxs_timer_tick());
    printf("Workload: kernel=%s pattern=%s N=%d C=%llu B=%llu low=%d high=%d width=%d shift=%d period=%d seed=%d\n",
      kernel, pattern, n, (unsigned long long)work.coprime, (unsigned long long)work.offset,
      low, high, width, shift, period, seed);
    printf("work,kernel,pattern,N,C,B,seed,low,high,width,shift,period,method,threads,repeat,"
      "seconds,index_setup_s,index_bytes,units,max_units,imbalance_pct,checksum\n");
    printf("worker,kernel,pattern,N,C,B,seed,low,high,width,shift,period,method,threads,"
      "repeat,tid,tasks,units,seconds\n");
    for (iteration = 0; iteration <= repeat && EXIT_SUCCESS == result; ++iteration) {
      for (order = 0; order < 6 && EXIT_SUCCESS == result; ++order) {
        const int method = (order + iteration) % 6;
        uint64_t accounted = 0, maximum = 0;
        size_t tasks = 0;
        double seconds = 0, checksum = 0;
        int team = 1, tid;
        for (row = 0; row < work.n; ++row) work.output[row] = -1;
        result = work_measure(&work, method, table, workers, &team, &seconds);
        for (tid = 0; tid < team; ++tid) {
          accounted += workers[tid].units;
          maximum = LIBXS_MAX(maximum, workers[tid].units);
          tasks += workers[tid].tasks;
        }
        if (EXIT_SUCCESS == result && (total != accounted || work.n != tasks)) {
          fprintf(stderr, "ERROR: workload coverage/accounting mismatch\n");
          result = EXIT_FAILURE;
        }
        for (row = 0; row < work.n; ++row) checksum += work.output[row];
        if (EXIT_SUCCESS == result && 0 < iteration) {
          printf("work,%s,%s,%d,%llu,%llu,%d,%d,%d,%d,%d,%d,%s,%d,%d,"
            "%.9g,%.9g,%llu,%llu,%llu,%.6f,%.17g\n",
            kernel, pattern, n, (unsigned long long)work.coprime, (unsigned long long)work.offset,
            seed, low, high, width, shift, period, work_methods[method], team, iteration, seconds,
            3 == method ? table_seconds : 2 == method ? affine_seconds : 0,
            (unsigned long long)(3 == method ? work.n * sizeof(*table) : 0),
            (unsigned long long)total, (unsigned long long)maximum,
            100 * ((double)team * (double)maximum / (double)total - 1), checksum);
          for (tid = 0; tid < team; ++tid) {
            printf("worker,%s,%s,%d,%llu,%llu,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%llu,%llu,%.9g\n",
              kernel, pattern, n, (unsigned long long)work.coprime,
              (unsigned long long)work.offset, seed, low, high, width, shift, period,
              work_methods[method], team, iteration, tid,
              (unsigned long long)workers[tid].tasks,
              (unsigned long long)workers[tid].units, workers[tid].seconds);
          }
        }
      }
    }
  }
  free(work.cost); free(work.output); free(work.reference);
  free(work.rowptr); free(work.column); free(work.value); free(work.x);
  free(table); free(workers);
  return result;
}


static double uint_energy(const void* input, size_t elemsize, size_t count, size_t lag)
{
  const char* data = (const char*)input;
  long double sum = 0;
  size_t i;
  for (i = 0; i < count; ++i) {
    size_t x = 0, y = 0;
    long double delta;
    LIBXS_MEMCPY(&x, data + elemsize * i, LIBXS_MIN(elemsize, sizeof(size_t)));
    LIBXS_MEMCPY(&y, data + elemsize * ((i + lag) % count), LIBXS_MIN(elemsize, sizeof(size_t)));
    delta = (long double)y - (long double)x;
    sum += delta * delta;
  }
  return (double)(sum / (long double)count);
}


static int uint_energy_report(const void* original, void* scratch,
  size_t elemsize, size_t count, size_t coprime, size_t offset, const size_t* nrepeat)
{
  const char* names[] = { "RNG", "DS1", "DS2" };
  const double reference = (double)count * ((double)count + 1) / 6;
  int method, result = EXIT_SUCCESS;
  size_t lag;
  if (1 >= count || (elemsize < sizeof(size_t) && count > ((size_t)1 << (CHAR_BIT * elemsize)))) {
    printf("Lag energy skipped: requires at least two distinct identity labels\n");
  }
  else {
    printf("Cyclic lag energy (Q=energy/random expectation)\n");
    for (method = 0; method < 3 && EXIT_SUCCESS == result; ++method) {
      size_t coefficient = 1, r;
      memcpy(scratch, original, elemsize * count);
      if (0 == method) shuffle(scratch, elemsize, count);
      else if (1 == method) result = libxs_shuffle(scratch, elemsize, count, &coprime, offset, nrepeat);
      else result = libxs_shuffle2(scratch, original, elemsize, count, &coprime, offset, nrepeat);
      for (r = 0; r < *nrepeat; ++r) coefficient = (size_t)((uint64_t)coefficient * coprime % count);
      for (lag = 0; lag < sizeof(energy_lags) / sizeof(*energy_lags) && EXIT_SUCCESS == result; ++lag) {
        const size_t h = energy_lags[lag];
        if (h < count) {
          const double energy = uint_energy(scratch, elemsize, count, h);
          if (0 == method) {
            printf("energy,%s,%llu,%.12g,%.12g\n",
              names[method], (unsigned long long)h, energy, energy / reference);
          } else {
            const size_t d = (size_t)((uint64_t)coefficient * h % count);
            const double expected = (double)d * (double)(count - d);
            printf("energy,%s,%llu,%.12g,%.12g,analytic=%.12g\n",
              names[method], (unsigned long long)h, energy, energy / reference, expected / reference);
            if (1E-8 * LIBXS_MAX(1.0, expected) < fabs(energy - expected)) {
              fprintf(stderr, "ERROR: %s lag-energy identity failed at h=%llu\n",
                names[method], (unsigned long long)h);
              result = EXIT_FAILURE;
            }
          }
        }
      }
    }
    if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: lag-energy evaluation failed\n");
  }
  return result;
}


size_t shuffle(void* inout, size_t elemsize, size_t count)
{
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
      unsigned long long n = result;
      switch (redop) {
        case redop_imbalance: {
          n = LIBXS_UPDIV(LIBXS_DELTA(lo, hi) * count, result);
        } break;
        case redop_sumdelta: {
          n = LIBXS_DELTA(result * 2, count * (count - 1)) / 2;
        } break;
        /*default: n = result;*/
      }
      /* normalize result relative to length of input */
      if (0 != result) result = LIBXS_UPDIV(n * count, result);
    }
  }
  return (size_t)result;
}


double uint_displacement(const void* input, size_t elemsize, size_t count)
{
  double result = -1;
  if (sizeof(size_t) <= elemsize || count <= ((size_t)1) << (8 * elemsize)) {
    const char *const data = (const char*)input;
    size_t i;
    result = 0;
    for (i = 0; i < count; ++i) {
      size_t v = 0;
      LIBXS_MEMCPY(&v, data + elemsize * i, LIBXS_MIN(elemsize, sizeof(size_t)));
      result += (double)(i < v ? (v - i) : (i - v));
    }
    result *= 100.0 / ((double)count * (double)count);
  }
  return result;
}


#if defined(BUBBLE_SORT)
size_t uint_bsort_asc(void* inout, size_t elemsize, size_t count)
{
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

#else /* merge-sort inversion counter O(n log n) */

static size_t msort_inv(unsigned char* data, unsigned char* scratch,
  size_t elemsize, size_t count)
{
  size_t result;
  size_t inv = 0, mid, il, ir, nl, nr;
  if (count <= 1) result = 0;
  else {
    mid = count / 2;
    inv += msort_inv(data, scratch, elemsize, mid);
    inv += msort_inv(data + elemsize * mid, scratch + elemsize * mid,
      elemsize, count - mid);
    /* merge two sorted halves into scratch, counting split inversions */
    nl = mid; nr = count - mid; il = 0; ir = 0;
    while (il < nl && ir < nr) {
      const unsigned char *a = data + elemsize * il;
      const unsigned char *b = data + elemsize * (mid + ir);
      size_t k = elemsize;
      int cmp = 0;
      for (; 0 < k; --k) {
        if (a[k - 1] != b[k - 1]) {
          cmp = (a[k - 1] > b[k - 1]) ? 1 : -1;
          break;
        }
      }
      if (cmp <= 0) {
        memcpy(scratch + elemsize * (il + ir), a, elemsize);
        ++il;
      }
      else {
        memcpy(scratch + elemsize * (il + ir), b, elemsize);
        ++ir;
        inv += nl - il;
      }
    }
    while (il < nl) {
      memcpy(scratch + elemsize * (il + ir), data + elemsize * il, elemsize);
      ++il;
    }
    while (ir < nr) {
      memcpy(scratch + elemsize * (il + ir),
        data + elemsize * (mid + ir), elemsize);
      ++ir;
    }
    memcpy(data, scratch, elemsize * count);
    result = inv;
  }
  return result;
}


size_t uint_msort_inversions(void* inout, size_t elemsize, size_t count)
{
  size_t result = 0;
  void* scratch = malloc(elemsize * count);
  if (NULL != scratch) {
    result = msort_inv((unsigned char*)inout,
      (unsigned char*)scratch, elemsize, count);
    free(scratch);
  }
  return result;
}
#endif


static int mhd_element_hash(void* dst,
  const libxs_mhd_element_handler_info_t* dst_info, libxs_data_t src_type,
  const void* src, const void* src_min, const void* src_max,
  size_t index, void* context)
{
  unsigned int v = 0;
  const size_t typesize = LIBXS_TYPESIZE(src_type);
  const size_t n = LIBXS_MIN(typesize, sizeof(v));
  LIBXS_UNUSED(dst_info); LIBXS_UNUSED(src_min); LIBXS_UNUSED(src_max);
  LIBXS_UNUSED(context);
  LIBXS_MEMCPY(&v, src, n);
  if (0 != index || NULL != mhd_element_hash_base) {
    const unsigned int i = (unsigned int)(NULL != mhd_element_hash_base
      ? ((const char*)src - (const char*)mhd_element_hash_base) / typesize
      : index);
    v = libxs_hash32(((unsigned long long)libxs_hash32(v) << 32) | i);
  }
  *(unsigned char*)dst = (unsigned char)(libxs_hash32(v) & 0xFF);
  return EXIT_SUCCESS;
}
