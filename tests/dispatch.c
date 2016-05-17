#include <libxs.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif


int main()
{
#if 0 != LIBXS_JIT
  const int m[] = { 1, 2, 3, 4, 5, 6, 7, LIBXS_MAX_M - 1, LIBXS_MAX_M, LIBXS_MAX_M + 1 };
  const int n[] = { 1, 2, 3, 4, 5, 6, 7, LIBXS_MAX_N - 1, LIBXS_MAX_N, LIBXS_MAX_N + 1 };
  const int k[] = { 1, 2, 3, 4, 5, 6, 7, LIBXS_MAX_K - 1, LIBXS_MAX_K, LIBXS_MAX_K + 1 };
  const int size = sizeof(m) / sizeof(*m), flags = LIBXS_FLAGS, prefetch = LIBXS_PREFETCH;
  const REAL_TYPE alpha = LIBXS_ALPHA, beta = LIBXS_BETA;
  int i, j = 0, nerrors = 0;

  for (i = 0; i < size; ++i) {
    const int lda = m[i], ldb = k[i], ldc = m[i];
    if (0 == LIBXS_MMDISPATCH_SYMBOL(REAL_TYPE)(m[i], n[i], k[i], &lda, &ldb, &ldc,
      &alpha, &beta, &flags, &prefetch))
    {
      if (0 == j) { /* capture first failure*/
        j = i;
      }
      ++nerrors;
    }
  }

  if (size != nerrors) {
    return size == i ? EXIT_SUCCESS : (i + 1)/*EXIT_FAILURE*/;
  }
  else if (LIBXS_X86_AVX > libxs_get_target_archid()) {
    /* potentially unsupported platforms due to not supporting AVX, due to calling convention,
     * or due to the environment variable LIBXS_JIT being set to zero.
     */
    fprintf(stderr, "JIT support is unavailable\n");
    return EXIT_SUCCESS;
  }
  else {
    return EXIT_FAILURE;
  }
#else
  fprintf(stderr, "Please rebuild LIBXS with JIT=1\n");
  return EXIT_SUCCESS;
#endif
}

