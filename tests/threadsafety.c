#include <libxs.h>
#include <stdlib.h>


int main()
{
  /* we do not care about the initial values */
  const float a[23*23], b[23*23];
  int i;

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    libxs_init();
  }

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    LIBXS_ALIGNED(float c[LIBXS_ALIGN_VALUE(23,sizeof(float),LIBXS_ALIGNMENT)*23], LIBXS_ALIGNMENT);
    const libxs_sfunction f = libxs_sdispatch(
      LIBXS_FLAGS, 23, 23, 23,
      0/*lda*/, 0/*ldb*/, 0/*ldc*/,
      0/*alpha*/, 0/*beta*/);
    if (0 != f) {
      f(a, b, c);
    }
    else {
      libxs_smm(LIBXS_FLAGS,
        23, 23, 23, a, b, c,
        LIBXS_PREFETCH_A(a),
        LIBXS_PREFETCH_B(b),
        LIBXS_PREFETCH_C(c),
        0/*alpha*/, 0/*beta*/);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    libxs_finalize();
  }

  return EXIT_SUCCESS;
}
