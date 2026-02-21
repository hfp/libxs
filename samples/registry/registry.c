/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Microbenchmark for the registry (key-value store) dispatch path.
 * Measures: registration, cold lookup, cached lookup, multi-threaded
 *           reads, contended writes, and mixed read/write scenarios. */

#include <libxs_reg.h>
#include <libxs_timer.h>
#include <libxs_math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#if defined(_OPENMP)
# include <omp.h>
#endif


#if !defined(KEY_MAXSIZE)
# define KEY_MAXSIZE 12
#endif

typedef struct bench_key_t {
  int id;
  int tag;
  int pad;
} bench_key_t;

typedef struct bench_value_t {
  double data[2];
} bench_value_t;


static void print_duration(const char* label, double total_ns, int count,
  libxs_timer_tick_t total_cycles)
{
  const double per_op_ns = total_ns / count;
  if (1E6 < per_op_ns) {
    printf("\t%-28s %8.2f ms/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns * 1E-6, count,
      (uintptr_t)(total_cycles / (libxs_timer_tick_t)count));
  }
  else if (1E3 < per_op_ns) {
    printf("\t%-28s %8.2f us/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns * 1E-3, count,
      (uintptr_t)(total_cycles / (libxs_timer_tick_t)count));
  }
  else {
    printf("\t%-28s %8.1f ns/op  (%d ops, %" PRIuPTR " cycles/op)\n",
      label, per_op_ns, count,
      (uintptr_t)(total_cycles / (libxs_timer_tick_t)count));
  }
}


/**
 * This (micro-)benchmark measures the duration needed to register and
 * look up entries in the registry. Various scenarios are measured:
 *   (1) cold registration of N unique keys (write),
 *   (2) cold lookup: shuffled access pattern defeating TLS cache,
 *   (3) cached lookup: repeated sequential access (TLS-cache-friendly),
 *   (4) multi-threaded parallel reads,
 *   (5) contended parallel writes (each thread writes unique keys),
 *   (6) mixed: one writer thread, remaining threads read concurrently.
 *
 * CLI: registry [total] [nrepeat] [nthreads]
 *   total    - number of unique keys to register (default: 10000)
 *   nrepeat  - number of repeat iterations for lookup phases (default: 10)
 *   nthreads - number of OpenMP threads (default: max available)
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
  const size_t shuffle = libxs_coprime2((size_t)size_total);

  bench_key_t* keys = NULL;
  bench_value_t* vals = NULL;
  libxs_registry_t* registry = NULL;
  libxs_timer_tick_t start, cycles;
  double duration_ns;
  int result = EXIT_SUCCESS;
  int i, n;

  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  /* allocate key/value arrays */
  keys = (bench_key_t*)calloc((size_t)size_total, sizeof(bench_key_t));
  vals = (bench_value_t*)malloc(sizeof(bench_value_t) * (size_t)size_total);
  if (NULL == keys || NULL == vals) {
    fprintf(stderr, "ERROR: allocation failed\n");
    free(keys); free(vals);
    return EXIT_FAILURE;
  }

  /* initialize keys (memset guarantees binary reproducibility) and values */
  for (i = 0; i < size_total; ++i) {
    memset(&keys[i], 0, sizeof(bench_key_t));
    keys[i].id = i;
    keys[i].tag = i ^ 0xABCD;
    keys[i].pad = 0;
    vals[i].data[0] = (double)i;
    vals[i].data[1] = (double)(i * 2);
  }

  printf("Registry benchmark: %d keys, %d repeat%s, %d thread%s\n",
    size_total, nrepeat, nrepeat > 1 ? "s" : "",
    nthreads, nthreads > 1 ? "s" : "");

  /* warm up timer */
  libxs_init();
  start = libxs_timer_tick();
  cycles = libxs_timer_ncycles(start, libxs_timer_tick());
  LIBXS_UNUSED(cycles);

  /*=========================================================================
   * (1) Registration: insert all keys (single-threaded)
   *=========================================================================*/
  libxs_registry_create(&registry);
  if (NULL == registry) { result = EXIT_FAILURE; goto cleanup; }

  start = libxs_timer_tick();
  for (i = 0; i < size_total; ++i) {
    bench_value_t* v = (bench_value_t*)libxs_registry_set(
      registry, &keys[i], sizeof(bench_key_t),
      sizeof(bench_value_t), &vals[i]);
    if (NULL == v) { result = EXIT_FAILURE; goto cleanup; }
  }
  cycles = libxs_timer_ncycles(start, libxs_timer_tick());
  duration_ns = 1E9 * libxs_timer_duration(start, libxs_timer_tick());
  print_duration("registration (write):", duration_ns, size_total, cycles);

  { /* verify info */
    libxs_registry_info_t info;
    if (EXIT_SUCCESS == libxs_registry_info(registry, &info)) {
      printf("\tregistry: size=%zu capacity=%zu nbytes=%zu\n",
        info.size, info.capacity, info.nbytes);
    }
  }

  /*=========================================================================
   * (2) Cold lookup: shuffled access (defeats TLS cache)
   *=========================================================================*/
  { libxs_timer_tick_t total_cycles = 0;
    for (n = 0; n < nrepeat; ++n) {
      start = libxs_timer_tick();
      for (i = 0; i < size_total; ++i) {
        const int j = (int)((shuffle * (size_t)i) % (size_t)size_total);
        const bench_value_t* v = (const bench_value_t*)libxs_registry_get(
          registry, &keys[j], sizeof(bench_key_t));
        if (NULL == v) { result = EXIT_FAILURE; goto cleanup; }
      }
      total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
    }
    duration_ns = 1E9 * libxs_timer_duration(0, total_cycles);
    print_duration("cold lookup (shuffled):",
      duration_ns, size_total * nrepeat, total_cycles);
  }

  /*=========================================================================
   * (3) Cached lookup: sequential repeated access (TLS-cache-friendly)
   *=========================================================================*/
  { const int local_size = LIBXS_MIN(LIBXS_REGCACHE_NENTRIES, size_total);
    libxs_timer_tick_t total_cycles = 0;
    for (n = 0; n < nrepeat; ++n) {
      start = libxs_timer_tick();
      for (i = 0; i < size_total; ++i) {
        const int j = i % local_size; /* cycle through a small set */
        const bench_value_t* v = (const bench_value_t*)libxs_registry_get(
          registry, &keys[j], sizeof(bench_key_t));
        if (NULL == v) { result = EXIT_FAILURE; goto cleanup; }
      }
      total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
    }
    duration_ns = 1E9 * libxs_timer_duration(0, total_cycles);
    print_duration("cached lookup (local):",
      duration_ns, size_total * nrepeat, total_cycles);
  }

  /*=========================================================================
   * (4) Multi-threaded parallel reads
   *=========================================================================*/
#if defined(_OPENMP)
  if (1 < nthreads) {
    libxs_timer_tick_t total_cycles = 0;
    for (n = 0; n < nrepeat; ++n) {
#     pragma omp parallel num_threads(nthreads) private(i)
      {
#       pragma omp master
        start = libxs_timer_tick();
#       pragma omp barrier
#       pragma omp for schedule(static)
        for (i = 0; i < size_total; ++i) {
          const int j = (int)((shuffle * (size_t)i) % (size_t)size_total);
          const bench_value_t* v = (const bench_value_t*)libxs_registry_get(
            registry, &keys[j], sizeof(bench_key_t));
          if (NULL == v) result = EXIT_FAILURE;
        }
#       pragma omp master
        total_cycles += libxs_timer_ncycles(start, libxs_timer_tick());
      }
    }
    duration_ns = 1E9 * libxs_timer_duration(0, total_cycles);
    print_duration("parallel read (all thr):",
      duration_ns, size_total * nrepeat, total_cycles);
  }
#endif

  libxs_registry_destroy(registry);
  registry = NULL;

  /*=========================================================================
   * (5) Contended writes: each thread writes its own key range in parallel
   *=========================================================================*/
#if defined(_OPENMP)
  if (1 < nthreads) {
    libxs_timer_tick_t total_cycles = 0;
    libxs_registry_create(&registry);
    if (NULL == registry) { result = EXIT_FAILURE; goto cleanup; }
    start = libxs_timer_tick();
#   pragma omp parallel num_threads(nthreads) private(i)
    {
#     pragma omp for schedule(static)
      for (i = 0; i < size_total; ++i) {
        bench_value_t* v = (bench_value_t*)libxs_registry_set(
          registry, &keys[i], sizeof(bench_key_t),
          sizeof(bench_value_t), &vals[i]);
        if (NULL == v) result = EXIT_FAILURE;
      }
    }
    total_cycles = libxs_timer_ncycles(start, libxs_timer_tick());
    duration_ns = 1E9 * libxs_timer_duration(0, total_cycles);
    print_duration("contended write (all thr):",
      duration_ns, size_total, total_cycles);

    /* verify */
    for (i = 0; i < size_total && EXIT_SUCCESS == result; ++i) {
      const bench_value_t* v = (const bench_value_t*)libxs_registry_get(
        registry, &keys[i], sizeof(bench_key_t));
      if (NULL == v || v->data[0] != vals[i].data[0]) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "ERROR: contended-write verification failed\n");
    }
    libxs_registry_destroy(registry);
    registry = NULL;
  }
#endif

  /*=========================================================================
   * (6) Mixed: one writer, remaining threads read concurrently
   *=========================================================================*/
#if defined(_OPENMP)
  if (2 < nthreads) {
    libxs_timer_tick_t total_cycles = 0;
    libxs_registry_create(&registry);
    if (NULL == registry) { result = EXIT_FAILURE; goto cleanup; }

    /* pre-populate half so readers have something to find */
    { const int half = size_total / 2;
      for (i = 0; i < half; ++i) {
        if (NULL == libxs_registry_set(registry, &keys[i], sizeof(bench_key_t),
          sizeof(bench_value_t), &vals[i]))
        { result = EXIT_FAILURE; goto cleanup; }
      }
      start = libxs_timer_tick();
#     pragma omp parallel num_threads(nthreads)
      {
        const int tid = omp_get_thread_num();
        if (0 == tid) { /* writer: register remaining keys */
          int w;
          for (w = half; w < size_total; ++w) {
            bench_value_t* v = (bench_value_t*)libxs_registry_set(
              registry, &keys[w], sizeof(bench_key_t),
              sizeof(bench_value_t), &vals[w]);
            if (NULL == v) result = EXIT_FAILURE;
          }
        }
        else { /* readers: look up pre-populated keys */
          int r;
          for (r = 0; r < half; ++r) {
            const int j = (int)(((size_t)r * shuffle) % (size_t)half);
            (void)libxs_registry_get(registry, &keys[j], sizeof(bench_key_t));
          }
        }
      }
      total_cycles = libxs_timer_ncycles(start, libxs_timer_tick());
      duration_ns = 1E9 * libxs_timer_duration(0, total_cycles);
      { const int total_ops = (size_total - half) + half * (nthreads - 1);
        print_duration("mixed r/w (1w + readers):",
          duration_ns, total_ops, total_cycles);
      }
    }
    libxs_registry_destroy(registry);
    registry = NULL;
  }
#endif

cleanup:
  if (NULL != registry) libxs_registry_destroy(registry);
  free(keys);
  free(vals);

  if (EXIT_SUCCESS == result) {
    printf("Finished\n");
  }
  else {
    fprintf(stderr, "FAILED\n");
  }
  return result;
}
