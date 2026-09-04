/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * Microbenchmark for the registry (key-value store) dispatch path.
 * Measures: registration, cold lookup, cached lookup, multi-threaded
 * reads, contended writes, and mixed read/write scenarios. The
 * single-threaded phases are swept over a range of key sizes, since
 * hashing and key comparison both scale with the size of the key.
 */
#include <libxs/libxs_reg.h>
#include <libxs/libxs_timer.h>
#include <libxs/libxs_math.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/** Largest key size taken into account by the sweep (Bytes). */
#if !defined(KEY_MAXSIZE)
# define KEY_MAXSIZE LIBXS_REGKEY_MAXSIZE
#endif
/** Key size used for the multi-threaded phases (Bytes). */
#if !defined(KEY_DEFSIZE)
# define KEY_DEFSIZE 12
#endif
/** Smallest key size carrying both the id and the tag (Bytes). */
#define KEY_MINSIZE 8


typedef struct bench_value_t {
  double data[2];
} bench_value_t;

/** Per-operation timings of the single-threaded phases. */
typedef struct bench_serial_t {
  double reg_ns, cold_ns, cached_ns;
  uintptr_t reg_cyc, cold_cyc, cached_cyc;
  size_t size, capacity, nbytes;
} bench_serial_t;


/**
 * Key sizes of interest: 8 is the malloc registry, 16 is libxs_textrule_t and
 * libxs_lexrule_t, 28 is internal_libxs_ngram_key_t and 32 the same if padded,
 * 48 is libxs_gemm_shape_t, and 64 is the largest key the registry accepts.
 */
static const size_t bench_keysizes[] = { 8, 12, 16, 28, 32, 48, 64 };


static void print_perop(const char* label, double per_op_ns, int count, uintptr_t cycles_per_op);
static int bench_serial(size_t key_size, int size_total, int nrepeat, bench_serial_t* timing);
static int bench_threaded(size_t key_size, int size_total, int nrepeat, int nthreads);
static unsigned char* bench_keys(size_t key_size, int size_total);
static bench_value_t* bench_vals(int size_total);


/**
 * CLI: registry [total] [nrepeat] [nthreads] [keysize]
 *   total    - number of unique keys to register (default: 10000)
 *   nrepeat  - number of repeat iterations for lookup phases (default: 10)
 *   nthreads - number of OpenMP threads (default: max available)
 *   keysize  - pin a single key size in Bytes (default: sweep)
 */
int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int size_total = LIBXS_MAX((1 < argc && 0 < atoi(argv[1])) ? atoi(argv[1]) : 10000, 2);
  const int nrepeat    = LIBXS_MAX((2 < argc && 0 < atoi(argv[2])) ? atoi(argv[2]) : 10, 1);
  const int nthreads   = LIBXS_CLMP((3 < argc && 0 < atoi(argv[3])) ? atoi(argv[3]) : max_nthreads, 1, max_nthreads);
  const int keysize    = (4 < argc && 0 < atoi(argv[4])) ? atoi(argv[4]) : 0;
  const int nsizes     = (int)(sizeof(bench_keysizes) / sizeof(*bench_keysizes));
  bench_serial_t timing;
  int result = EXIT_SUCCESS;
  int i;
  libxs_init();
  { /* warm up the timer */
    const libxs_timer_tick_t start = libxs_timer_tick();
    const libxs_timer_tick_t cycles = libxs_timer_ncycles(start, libxs_timer_tick());
    LIBXS_UNUSED(cycles);
  }
  printf("Registry benchmark: %d keys, %d repeat%s, %d thread%s\n",
    size_total, nrepeat, 1 < nrepeat ? "s" : "",
    nthreads, 1 < nthreads ? "s" : "");
  if (0 != keysize) { /* pinned key size: per-phase detail */
    printf("\nSingle-threaded (%d-Byte keys):\n", keysize);
    result = bench_serial((size_t)keysize, size_total, nrepeat, &timing);
    if (EXIT_SUCCESS == result) {
      print_perop("registration (write):", timing.reg_ns, size_total, timing.reg_cyc);
      print_perop("cold lookup (shuffled):", timing.cold_ns, size_total * nrepeat, timing.cold_cyc);
      print_perop("cached lookup (local):", timing.cached_ns, size_total * nrepeat, timing.cached_cyc);
      printf("\tregistry: size=%" PRIuPTR " capacity=%" PRIuPTR
        " nbytes=%" PRIuPTR "\n", (uintptr_t)timing.size,
        (uintptr_t)timing.capacity, (uintptr_t)timing.nbytes);
    }
    else {
      fprintf(stderr, "ERROR: %d-Byte keys failed\n", keysize);
    }
  }
  else { /* sweep: one row per key size */
    printf("\nSingle-threaded key-size sweep:\n");
    printf("\t         registration     cold lookup      cached lookup\n");
    printf("\tBytes    ns/op  cyc/op    ns/op  cyc/op    ns/op  cyc/op\n");
    for (i = 0; i < nsizes && EXIT_SUCCESS == result; ++i) {
      const size_t key_size = bench_keysizes[i];
      if (key_size <= (size_t)KEY_MAXSIZE && key_size <= LIBXS_REGKEY_MAXSIZE) {
        result = bench_serial(key_size, size_total, nrepeat, &timing);
        if (EXIT_SUCCESS == result) {
          printf("\t%5" PRIuPTR "  %7.1f %7" PRIuPTR "  %7.1f %7" PRIuPTR
            "  %7.1f %7" PRIuPTR "\n", (uintptr_t)key_size,
            timing.reg_ns, timing.reg_cyc, timing.cold_ns, timing.cold_cyc,
            timing.cached_ns, timing.cached_cyc);
        }
        else {
          fprintf(stderr, "ERROR: %" PRIuPTR "-Byte keys failed\n",
            (uintptr_t)key_size);
        }
      }
    }
  }
  if (EXIT_SUCCESS == result && 1 < nthreads) {
    const int key_size = (0 != keysize) ? keysize : KEY_DEFSIZE;
    printf("\nMulti-threaded (%d-Byte keys):\n", key_size);
    result = bench_threaded((size_t)key_size, size_total, nrepeat, nthreads);
  }
  libxs_finalize();
  if (EXIT_SUCCESS == result) {
    printf("Finished\n");
  }
  else {
    fprintf(stderr, "FAILED\n");
  }
  return result;
}


static void print_perop(const char* label, double per_op_ns, int count,
  uintptr_t cycles_per_op)
{
  if (1E6 < per_op_ns) {
    printf("\t%-28s %8.2f ms/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns * 1E-6, count, cycles_per_op);
  }
  else if (1E3 < per_op_ns) {
    printf("\t%-28s %8.2f us/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns * 1E-3, count, cycles_per_op);
  }
  else {
    printf("\t%-28s %8.1f ns/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns, count, cycles_per_op);
  }
}


/**
 * Build the key arena: size_total keys of key_size Bytes, the leading eight
 * Bytes carrying the id and the tag. The stride equals the key size, which
 * reproduces the alignment an array of same-sized structures would yield.
 * calloc zeroes the remainder, and that is what makes the keys binary
 * reproducible as libxs_registry_set requires.
 */
static unsigned char* bench_keys(size_t key_size, int size_total)
{
  unsigned char* result = NULL;
  if ((size_t)KEY_MINSIZE <= key_size && key_size <= LIBXS_REGKEY_MAXSIZE) {
    result = (unsigned char*)calloc((size_t)size_total, key_size);
  }
  if (NULL != result) {
    int i;
    for (i = 0; i < size_total; ++i) {
      unsigned char *const key = result + (size_t)i * key_size;
      const int id = i, tag = i ^ 0xABCD;
      memcpy(key, &id, sizeof(id));
      memcpy(key + sizeof(id), &tag, sizeof(tag));
    }
  }
  return result;
}


static bench_value_t* bench_vals(int size_total)
{
  bench_value_t *const result = (bench_value_t*)malloc(
    sizeof(bench_value_t) * (size_t)size_total);
  if (NULL != result) {
    int i;
    for (i = 0; i < size_total; ++i) {
      result[i].data[0] = (double)i;
      result[i].data[1] = (double)(i * 2);
    }
  }
  return result;
}


/**
 * Single-threaded phases, which are the ones sensitive to the key size:
 *   (1) cold registration of size_total unique keys (write),
 *   (2) cold lookup: shuffled access pattern defeating the TLS cache,
 *   (3) cached lookup: repeated access to a small set (TLS-cache-friendly).
 * A TLS cache hit takes neither the lock nor a probe, hence phase (3) is
 * the purest measure of hashing the key and comparing it.
 */
static int bench_serial(size_t key_size, int size_total, int nrepeat,
  bench_serial_t* timing)
{
  const size_t shuffle = libxs_coprime2((size_t)size_total);
  unsigned char *const keys = bench_keys(key_size, size_total);
  bench_value_t *const vals = bench_vals(size_total);
  libxs_registry_t *const registry = libxs_registry_create();
  int result = (NULL != keys && NULL != vals && NULL != registry)
    ? EXIT_SUCCESS : EXIT_FAILURE;
  memset(timing, 0, sizeof(*timing));
  if (EXIT_SUCCESS == result) { /* (1) registration */
    const libxs_timer_tick_t start = libxs_timer_tick();
    libxs_timer_tick_t stop;
    int i;
    for (i = 0; i < size_total; ++i) {
      if (NULL == libxs_registry_set(registry, keys + (size_t)i * key_size,
        key_size, vals + i, sizeof(bench_value_t), libxs_registry_lock(registry)))
      {
        result = EXIT_FAILURE;
      }
    }
    stop = libxs_timer_tick();
    timing->reg_ns = 1E9 * libxs_timer_duration(start, stop) / size_total;
    timing->reg_cyc = (uintptr_t)(libxs_timer_ncycles(start, stop)
      / (libxs_timer_tick_t)size_total);
  }
  if (EXIT_SUCCESS == result) { /* (2) cold lookup: shuffled access */
    const int count = size_total * nrepeat;
    libxs_timer_tick_t total_cycles = 0;
    int i, n;
    for (n = 0; n < nrepeat; ++n) {
      const libxs_timer_tick_t start = libxs_timer_tick();
      for (i = 0; i < size_total; ++i) {
        const size_t j = (shuffle * (size_t)i) % (size_t)size_total;
        if (NULL == libxs_registry_get(registry, keys + j * key_size,
          key_size, libxs_registry_lock(registry)))
        {
          result = EXIT_FAILURE;
        }
      }
      total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
    }
    timing->cold_ns = 1E9 * libxs_timer_duration(0, total_cycles) / count;
    timing->cold_cyc = (uintptr_t)(total_cycles / (libxs_timer_tick_t)count);
  }
  if (EXIT_SUCCESS == result) { /* (3) cached lookup: small working set */
    const int local_size = LIBXS_MIN(LIBXS_REGCACHE_NENTRIES, size_total);
    const int count = size_total * nrepeat;
    libxs_timer_tick_t total_cycles = 0;
    int i, n;
    for (n = 0; n < nrepeat; ++n) {
      const libxs_timer_tick_t start = libxs_timer_tick();
      for (i = 0; i < size_total; ++i) {
        const size_t j = (size_t)(i % local_size);
        if (NULL == libxs_registry_get(registry, keys + j * key_size,
          key_size, libxs_registry_lock(registry)))
        {
          result = EXIT_FAILURE;
        }
      }
      total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
    }
    timing->cached_ns = 1E9 * libxs_timer_duration(0, total_cycles) / count;
    timing->cached_cyc = (uintptr_t)(total_cycles / (libxs_timer_tick_t)count);
  }
  if (EXIT_SUCCESS == result) {
    libxs_registry_info_t info;
    if (EXIT_SUCCESS == libxs_registry_info(registry, &info)) {
      timing->size = info.size;
      timing->capacity = info.capacity;
      timing->nbytes = info.nbytes;
    }
    if ((size_t)size_total != timing->size) result = EXIT_FAILURE;
  }
  libxs_registry_destroy(registry);
  free(keys);
  free(vals);
  return result;
}


#if defined(_OPENMP)
static void print_duration(const char* label, double total_ns, int count,
  libxs_timer_tick_t total_cycles);
static void print_duration(const char* label, double total_ns, int count,
  libxs_timer_tick_t total_cycles)
{
  print_perop(label, total_ns / count, count,
    (uintptr_t)(total_cycles / (libxs_timer_tick_t)count));
}
#endif


/**
 * Multi-threaded phases, which measure locking and contention rather than
 * the key size:
 *   (4) parallel reads across all threads,
 *   (5) contended parallel writes (each thread writes unique keys),
 *   (6) mixed: one writer thread, remaining threads read concurrently.
 */
static int bench_threaded(size_t key_size, int size_total, int nrepeat,
  int nthreads)
{
  int result = EXIT_SUCCESS;
#if defined(_OPENMP)
  const size_t shuffle = libxs_coprime2((size_t)size_total);
  unsigned char *const keys = bench_keys(key_size, size_total);
  bench_value_t *const vals = bench_vals(size_total);
  libxs_registry_t* registry = NULL;
  if (NULL == keys || NULL == vals) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) { /* (4) parallel reads */
    libxs_timer_tick_t total_cycles = 0;
    libxs_timer_tick_t start = 0;
    int nerr = 0, i, n;
    registry = libxs_registry_create();
    if (NULL == registry) result = EXIT_FAILURE;
    for (i = 0; i < size_total && EXIT_SUCCESS == result; ++i) {
      if (NULL == libxs_registry_set(registry, keys + (size_t)i * key_size,
        key_size, vals + i, sizeof(bench_value_t), libxs_registry_lock(registry)))
      {
        result = EXIT_FAILURE;
      }
    }
    for (n = 0; n < nrepeat && EXIT_SUCCESS == result; ++n) {
#     pragma omp parallel num_threads(nthreads) private(i)
      {
#       pragma omp master
        start = libxs_timer_tick();
#       pragma omp barrier
#       pragma omp for schedule(static) reduction(+ : nerr)
        for (i = 0; i < size_total; ++i) {
          const size_t j = (shuffle * (size_t)i) % (size_t)size_total;
          if (NULL == libxs_registry_get(registry, keys + j * key_size,
            key_size, libxs_registry_lock(registry))) ++nerr;
        }
#       pragma omp master
        total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
      }
    }
    if (0 != nerr) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      print_duration("parallel read (all thr):",
        1E9 * libxs_timer_duration(0, total_cycles), size_total * nrepeat,
        total_cycles);
    }
    libxs_registry_destroy(registry);
    registry = NULL;
  }
  if (EXIT_SUCCESS == result) { /* (5) contended writes */
    libxs_timer_tick_t start, total_cycles;
    int nerr = 0, i;
    registry = libxs_registry_create();
    if (NULL == registry) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      start = libxs_timer_tick();
#     pragma omp parallel num_threads(nthreads) private(i)
      {
#       pragma omp for schedule(static) reduction(+ : nerr)
        for (i = 0; i < size_total; ++i) {
          if (NULL == libxs_registry_set(registry, keys + (size_t)i * key_size,
            key_size, vals + i, sizeof(bench_value_t),
            libxs_registry_lock(registry))) ++nerr;
        }
      }
      total_cycles = libxs_timer_ncycles(start, libxs_timer_tick());
      if (0 != nerr) result = EXIT_FAILURE;
      if (EXIT_SUCCESS == result) {
        print_duration("contended write (all thr):",
          1E9 * libxs_timer_duration(0, total_cycles), size_total, total_cycles);
      }
      for (i = 0; i < size_total && EXIT_SUCCESS == result; ++i) {
        const bench_value_t *const v = (const bench_value_t*)libxs_registry_get(
          registry, keys + (size_t)i * key_size, key_size,
          libxs_registry_lock(registry));
        if (NULL == v || v->data[0] != vals[i].data[0]) result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS != result) {
        fprintf(stderr, "ERROR: contended-write verification failed\n");
      }
    }
    libxs_registry_destroy(registry);
    registry = NULL;
  }
  if (EXIT_SUCCESS == result && 2 < nthreads) { /* (6) one writer, N readers */
    const int half = size_total / 2;
    libxs_timer_tick_t start, total_cycles;
    int i;
    registry = libxs_registry_create();
    if (NULL == registry) result = EXIT_FAILURE;
    for (i = 0; i < half && EXIT_SUCCESS == result; ++i) {
      if (NULL == libxs_registry_set(registry, keys + (size_t)i * key_size,
        key_size, vals + i, sizeof(bench_value_t), libxs_registry_lock(registry)))
      {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      start = libxs_timer_tick();
#     pragma omp parallel num_threads(nthreads)
      {
        if (0 == omp_get_thread_num()) { /* writer: register remaining keys */
          int w;
          for (w = half; w < size_total; ++w) {
            if (NULL == libxs_registry_set(registry, keys + (size_t)w * key_size,
              key_size, vals + w, sizeof(bench_value_t),
              libxs_registry_lock(registry))) result = EXIT_FAILURE;
          }
        }
        else { /* readers: look up the pre-populated keys */
          int r;
          for (r = 0; r < half; ++r) {
            const size_t j = ((size_t)r * shuffle) % (size_t)half;
            (void)libxs_registry_get(registry, keys + j * key_size, key_size,
              libxs_registry_lock(registry));
          }
        }
      }
      total_cycles = libxs_timer_ncycles(start, libxs_timer_tick());
      if (EXIT_SUCCESS == result) {
        print_duration("mixed r/w (1w + readers):",
          1E9 * libxs_timer_duration(0, total_cycles),
          (size_total - half) + half * (nthreads - 1), total_cycles);
      }
    }
    libxs_registry_destroy(registry);
  }
  free(keys);
  free(vals);
#else
  LIBXS_UNUSED(key_size); LIBXS_UNUSED(size_total);
  LIBXS_UNUSED(nrepeat); LIBXS_UNUSED(nthreads);
#endif
  return result;
}
