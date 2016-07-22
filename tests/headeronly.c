#include <libxs_source.h>
#include <stdlib.h>
#include <stdio.h>


LIBXS_EXTERN libxs_dmmfunction dmmdispatch(int m, int n, int k);


int main()
{
  const int m = LIBXS_MAX_M, n = LIBXS_MAX_N, k = LIBXS_MAX_K;
  const libxs_dmmfunction fa = libxs_dmmdispatch(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
  const libxs_dmmfunction fb = dmmdispatch(m, n, k);
#if defined(_DEBUG)
  if (fa != fb) {
    union { libxs_xmmfunction xmm; void* pmm; } a, b;
    a.xmm.dmm = fa; b.xmm.dmm = fb;
    fprintf(stderr, "Error: %p != %p\n", a.pmm, b.pmm);
  }
#endif
  return fa == fb ? EXIT_SUCCESS : EXIT_FAILURE;
}

