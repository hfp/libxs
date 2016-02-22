#include <libxs.h>
#include <stdlib.h>


int main()
{
  /* we do not care about the initial values */
  /*const*/ float a[23*23], b[23*23];
  int i;

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < 1000; ++i) {
    libxs_init();
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < 1000; ++i) {
    float c[23*23];
    const libxs_smmfunction f = libxs_smmdispatch(23, 23, (i / 50) % 23 + 1,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
    if (NULL != f) {
      LIBXS_MMCALL_ABC(f, a, b, c);
    }
    else {
      const libxs_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
      libxs_sgemm(NULL/*transa*/, NULL/*transb*/, &m, &n, &k,
        NULL/*alpha*/, a, NULL/*lda*/, b, NULL/*ldb*/, 
        NULL/*beta*/, c, NULL/*ldc*/);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < 1000; ++i) {
    libxs_finalize();
  }

  return EXIT_SUCCESS;
}
