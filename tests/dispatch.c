#include <libxs.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif
#if !defined(NTESTS)
# define NTESTS 10000
#endif


int main(void)
{
  const int m[] = { 1, 2, 3, LIBXS_MAX_M - 1, LIBXS_MAX_M, LIBXS_MAX_M + 1,    16,    16,    16 };
  const int n[] = { 1, 2, 3, LIBXS_MAX_N - 1, LIBXS_MAX_N, LIBXS_MAX_N + 1, 65279, 65280, 65792 };
  const int k[] = { 1, 2, 3, LIBXS_MAX_K - 1, LIBXS_MAX_K, LIBXS_MAX_K + 1,    16,    16,    16 };
  const int size = sizeof(m) / sizeof(*m), flags = LIBXS_FLAGS, prefetch = LIBXS_PREFETCH;
  const REAL_TYPE alpha = LIBXS_ALPHA, beta = LIBXS_BETA;
  LIBXS_MMFUNCTION_TYPE(REAL_TYPE) f[sizeof(m)/sizeof(*m)];
  int i, nerrors = 0;

  /* initially generate a number of test kernels */
  for (i = 0; i < size; ++i) {
    f[i] = LIBXS_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i], n[i], k[i], m + i, k + i, m + i,
      &alpha, &beta, &flags, &prefetch);
  }

  /* check that the same kernels are dispatched as previously generated */
  for (i = 0; i < (NTESTS); ++i) {
    const LIBXS_MMFUNCTION_TYPE(REAL_TYPE) fi = LIBXS_MMDISPATCH_SYMBOL(REAL_TYPE)(
      m[i%size], n[i%size], k[i%size], m + (i % size), k + (i % size), m + (i % size),
      &alpha, &beta, &flags, &prefetch);

    if (fi != f[i%size]) { /* always an error even when JIT is disabled at compile-time */
#if defined(_DEBUG)
      if (0 != fi) {
        fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m[i%size], n[i%size], k[i%size]);
      }
      else { /* did not find previously generated and recorded kernel */
        fprintf(stderr, "Error: cannot find %ix%ix%i-kernel!\n", m[i%size], n[i%size], k[i%size]);
      }
#endif
      ++nerrors;
    }
  }

  return (0 == nerrors) ? EXIT_SUCCESS : EXIT_FAILURE;
}

