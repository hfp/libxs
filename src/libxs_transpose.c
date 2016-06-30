#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
#endif
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_TRANSPOSE_CACHESIZE)
# define LIBXS_TRANSPOSE_CACHESIZE 32768
#endif
#if !defined(LIBXS_TRANSPOSE_N)
# define LIBXS_TRANSPOSE_N 32
#endif


/* Based on cache-oblivious scheme as published by Frigo et.al. */
LIBXS_INLINE LIBXS_RETARGETABLE void inernal_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  const libxs_blasint m = m1 - m0, n = n1 - n0;
  libxs_blasint i, j;

  if (m * n * typesize <= (LIBXS_TRANSPOSE_CACHESIZE / 2)) {
    switch(typesize) {
      case 1: {
        const char *const a = (const char*)in;
        char *const b = (char*)out;
        for (i = n0; i < n1; ++i) {
#if 0 < LIBXS_TRANSPOSE_N
          LIBXS_ASSUME(m <= LIBXS_TRANSPOSE_N)
          LIBXS_PRAGMA_LOOP_COUNT(LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxs_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 2: {
        const short *const a = (const short*)in;
        short *const b = (short*)out;
        for (i = n0; i < n1; ++i) {
#if 0 < LIBXS_TRANSPOSE_N
          LIBXS_ASSUME(m <= LIBXS_TRANSPOSE_N)
          LIBXS_PRAGMA_LOOP_COUNT(LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxs_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 4: {
        const float *const a = (const float*)in;
        float *const b = (float*)out;
        for (i = n0; i < n1; ++i) {
#if 0 < LIBXS_TRANSPOSE_N
          LIBXS_ASSUME(m <= LIBXS_TRANSPOSE_N)
          LIBXS_PRAGMA_LOOP_COUNT(LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxs_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      case 8: {
        const double *const a = (const double*)in;
        double *const b = (double*)out;
        for (i = n0; i < n1; ++i) {
#if 0 < LIBXS_TRANSPOSE_N
          LIBXS_ASSUME(m <= LIBXS_TRANSPOSE_N)
          LIBXS_PRAGMA_LOOP_COUNT(LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N, LIBXS_TRANSPOSE_N)
#endif
          for (j = 0; j < m; ++j) {
            const libxs_blasint k = j + m0;
            b[i*ldo+k/*consecutive*/] = a[k*ld+i/*strided*/];
          }
        }
      } break;
      default: assert(0);
    }
  }
  else if (n >= m) {
    const libxs_blasint ni = (n0 + n1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
    inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
  }
  else {
#if 0 < LIBXS_TRANSPOSE_N
    if (LIBXS_TRANSPOSE_N < m) {
      const libxs_blasint mi = m0 + LIBXS_TRANSPOSE_N;
      inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
      inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
    }
    else
#endif
    {
      const libxs_blasint mi = (m0 + m1) / 2;
      inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
      inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
    }
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (ld < LIBXS_LD(n, m) && ldo < LIBXS_LD(m, n)) {
    fprintf(stderr, "LIBXS: the leading dimensions of the transpose are too small!\n");
  }
  else if (ld < LIBXS_LD(n, m)) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose input is too small!\n");
  }
  else if (ldo < LIBXS_LD(m, n)) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose output is too small!\n");
  }
#endif
  inernal_transpose_oop(out, in, typesize, 0, LIBXS_LD(m, n), 0, LIBXS_LD(n, m), ld, ldo);
}

