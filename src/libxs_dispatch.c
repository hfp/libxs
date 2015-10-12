#include "libxs_dispatch.h"
#include "generator_extern_typedefs.h"
#include "libxs_crc32.h"
#include <libxs.h>

#define LIBXS_CACHESIZE (LIBXS_MAX_M) * (LIBXS_MAX_N) * (LIBXS_MAX_K) * 24
#define LIBXS_SEED 0


/** Filled with zeros due to C language rule. */
LIBXS_RETARGETABLE libxs_function libxs_cache[2][(LIBXS_CACHESIZE)];
LIBXS_RETARGETABLE int libxs_init = 0;


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_function libxs_dispatch(const void* key, size_t key_size, size_t cache_id, libxs_function function)
{
  const unsigned int hash = libxs_crc32(key, key_size, LIBXS_SEED), i = hash % (LIBXS_CACHESIZE);
  libxs_function *const cache = libxs_cache[cache_id%2];
  const libxs_function f = cache[i];
  cache[i] = function;
  return f;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_function libxs_lookup(const void* key, size_t key_size, size_t cache_id)
{
  return libxs_cache[cache_id%2][libxs_crc32(key, key_size, LIBXS_SEED)%(LIBXS_CACHESIZE)];
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smm_function libxs_smm_dispatch(int m, int n, int k)
{
  libxs_xgemm_descriptor LIBXS_XGEMM_DESCRIPTOR(desc, 1/*single precision*/, LIBXS_PREFETCH, 'n', 'n', 1/*alpha*/, LIBXS_BETA,
    m, n, k, m, k, LIBXS_ALIGN_STORES(m, sizeof(float)));

  if (0 == libxs_init) {
    libxs_build_static();
    libxs_init = 1;
  }

  return (libxs_smm_function)libxs_lookup(&desc, LIBXS_XGEMM_DESCRIPTOR_SIZE, 1/*single precision*/);
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k)
{
  libxs_xgemm_descriptor LIBXS_XGEMM_DESCRIPTOR(desc, 0/*double precision*/, LIBXS_PREFETCH, 'n', 'n', 1/*alpha*/, LIBXS_BETA,
    m, n, k, m, k, LIBXS_ALIGN_STORES(m, sizeof(double)));

  if (0 == libxs_init) {
    libxs_build_static();
    libxs_init = 1;
  }

  return (libxs_dmm_function)libxs_lookup(&desc, LIBXS_XGEMM_DESCRIPTOR_SIZE, 0/*double precision*/);
}
