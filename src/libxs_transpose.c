#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/* Based on cache-oblivious scheme as published by Frigo et.al. */
LIBXS_INLINE LIBXS_RETARGETABLE void inernal_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  const libxs_blasint m = m1 - m0, n = n1 - n0;
  libxs_blasint i, j;

  if (m * n * typesize <= 16384) {
    switch(typesize) {
      case 1: {
        const char *const a = (const char*)in;
        char *const b = (char*)out;
        for (i = m0; i < m1; ++i) {
          for (j = n0; j < n1; ++j) {
            b[i*ldo+j] = a[j*ld+i];
          }
        }
      } break;
      case 2: {
        const short *const a = (const short*)in;
        short *const b = (short*)out;
        for (i = m0; i < m1; ++i) {
          for (j = n0; j < n1; ++j) {
            b[i*ldo+j] = a[j*ld+i];
          }
        }
      } break;
      case 4: {
        const float *const a = (const float*)in;
        float *const b = (float*)out;
        for (i = m0; i < m1; ++i) {
          for (j = n0; j < n1; ++j) {
            b[i*ldo+j] = a[j*ld+i];
          }
        }
      } break;
      case 8: {
        const double *const a = (const double*)in;
        double *const b = (double*)out;
        for (i = m0; i < m1; ++i) {
          for (j = n0; j < n1; ++j) {
            b[i*ldo+j] = a[j*ld+i];
          }
        }
      } break;
      default: assert(0);
    }
  }
  else if (m >= n) {
    const libxs_blasint mi = (m0 + m1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
    inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
  }
  else {
    const libxs_blasint ni = (n0 + n1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
    inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  inernal_transpose_oop(out, in, typesize, 0, m, 0, n, ld, ldo);
}

