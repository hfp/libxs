#include <libxs_source.h>


LIBXS_EXTERN libxs_dmmfunction dmmdispatch(int m, int n, int k)
{
  fprintf(stderr, "\nDEBUG: %p", internal_registry);
  return libxs_dmmdispatch(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
    NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);
}

