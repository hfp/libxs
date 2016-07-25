#include <libxs.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(MAX_NKERNELS)
# define MAX_NKERNELS 1000
#endif


int main()
{
  /* we do not care about the initial values */
  /*const*/ float a[23*23], b[23*23];
  libxs_smmfunction f[MAX_NKERNELS];
  int i;

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    libxs_init();
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i) shared(a, b, f)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    const libxs_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
    /* playing ping-pong with fi's cache line is not the subject */
    f[i] = libxs_smmdispatch(m, n, k,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
  }

  for (i = 0; i < MAX_NKERNELS; ++i) {
    float c[23/*m*/*23/*n*/];
    const libxs_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
    const libxs_smmfunction fi = libxs_smmdispatch(m, n, k,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);

    if (NULL != f[i]) {
      if (fi == f[i]) {
        LIBXS_MMCALL(f[i], a, b, c, m, n, k);
      }
      else {
#if defined(_DEBUG)
        fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m, n, k);
#endif
        assert(EXIT_SUCCESS != (i + 1));
        return i + 1;
      }
    }
    else {
      libxs_sgemm(NULL/*transa*/, NULL/*transb*/, &m, &n, &k,
        NULL/*alpha*/, a, NULL/*lda*/, b, NULL/*ldb*/, 
        NULL/*beta*/, c, NULL/*ldc*/);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    libxs_finalize();
  }

  return EXIT_SUCCESS;
}
