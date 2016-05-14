#include <libxs.h>
#include <stdlib.h>


int main()
{
  const int archid = libxs_get_target_arch();

  /* official runtime check for JIT availability */
  if (LIBXS_X86_AVX <= archid) { /* available */
#if 0 == LIBXS_JIT
    /* runtime check should have been negative */
    return EXIT_FAILURE;
#endif
    libxs_set_target_archid("0"); /* disable JIT */
    /* likely returns NULL, however should not crash */
    libxs_smmdispatch(LIBXS_MAX_M, LIBXS_MAX_N, LIBXS_MAX_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
      NULL/*alpha*/, NULL/*beta*/, NULL/*flags*/, NULL/*prefetch*/);
  }
#if 0 != LIBXS_JIT /* JIT is built-in (enabled at compile-time) */
  else { /* JIT is not available at runtime */
    /* bypass CPUID flags and setup to something supported with JIT */
    libxs_set_target_arch(LIBXS_X86_AVX);

    if (0 == libxs_dmmdispatch(LIBXS_MAX_M, LIBXS_MAX_N, LIBXS_MAX_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
      NULL/*alpha*/, NULL/*beta*/, NULL/*flags*/, NULL/*prefetch*/))
    {
      /* requested function should have been JITted */
      return EXIT_FAILURE;
    }
  }
#endif

  return EXIT_SUCCESS;
}

