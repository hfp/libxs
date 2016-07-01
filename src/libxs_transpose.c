#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_trans.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_TRANSPOSE_CACHESIZE)
# define LIBXS_TRANSPOSE_CACHESIZE 32768
#endif
#if !defined(LIBXS_TRANSPOSE_CHUNK)
# define LIBXS_TRANSPOSE_CHUNK 32
#endif

#define INTERNAL_TRANSPOSE_OOP(TYPE, OUT, IN, M0, M1, N0, N1) { \
  const TYPE *const a = (const TYPE*)IN; \
  TYPE *const b = (TYPE*)OUT; \
  libxs_blasint i, j; \
  if (LIBXS_TRANSPOSE_CHUNK == m) { \
    for (i = N0; i < N1; ++i) { \
      LIBXS_PRAGMA_NONTEMPORAL \
      for (j = M0; j < M0 + LIBXS_TRANSPOSE_CHUNK; ++j) { \
        b[i*ldo+j/*consecutive*/] = a[j*ld+i/*strided*/]; \
      } \
    } \
  } \
  else { \
    for (i = N0; i < N1; ++i) { \
      LIBXS_PRAGMA_NONTEMPORAL \
      for (j = M0; j < M1; ++j) { \
        b[i*ldo+j/*consecutive*/] = a[j*ld+i/*strided*/]; \
      } \
    } \
  } \
}


/* Based on cache-oblivious scheme as published by Frigo et.al. */
LIBXS_INLINE LIBXS_RETARGETABLE void inernal_transpose_oop(void *LIBXS_RESTRICT out, const void *LIBXS_RESTRICT in,
  unsigned int typesize, libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  const libxs_blasint m = m1 - m0, n = n1 - n0;

  if (m * n * typesize <= (LIBXS_TRANSPOSE_CACHESIZE / 2)) {
    switch(typesize) {
      case 1: {
        INTERNAL_TRANSPOSE_OOP(char, out, in, m0, m1, n0, n1);
      } break;
      case 2: {
        INTERNAL_TRANSPOSE_OOP(short, out, in, m0, m1, n0, n1);
      } break;
      case 4: {
        INTERNAL_TRANSPOSE_OOP(float, out, in, m0, m1, n0, n1);
      } break;
      case 8: {
        INTERNAL_TRANSPOSE_OOP(double, out, in, m0, m1, n0, n1);
      } break;
      case 16: {
        typedef struct dvec2_t { double value[2]; } dvec2_t;
        INTERNAL_TRANSPOSE_OOP(dvec2_t, out, in, m0, m1, n0, n1);
      } break;
      default: {
#if !defined(NDEBUG) /* library code is expected to be mute */
        fprintf(stderr, "LIBXS: unsupported element type in transpose!\n");
#endif
        assert(0);
      }
    }
  }
  else if (n >= m) {
    const libxs_blasint ni = (n0 + n1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
    inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
  }
  else {
#if (0 < LIBXS_TRANSPOSE_CHUNK)
    if (LIBXS_TRANSPOSE_CHUNK < m) {
      const libxs_blasint mi = m0 + LIBXS_TRANSPOSE_CHUNK;
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
  if (ld < m && ldo < n) {
    fprintf(stderr, "LIBXS: the leading dimensions of the transpose are too small!\n");
  }
  else if (ld < m) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose input is too small!\n");
  }
  else if (ldo < n) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose output is too small!\n");
  }
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
  if (8 == typesize) {
    mkl_domatcopy('C', 'T', m, n, 1, (const double*)in, ld, (double*)out, ldo);
  }
  else if (4 == typesize) {
    mkl_somatcopy('C', 'T', m, n, 1, (const float*)in, ld, (float*)out, ldo);
  }
  else
#endif
  {
    inernal_transpose_oop(out, in, typesize, 0, n, 0, m, ld, ldo);
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_stranspose_oop(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_dtranspose_oop(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(double), m, n, ld, ldo);
}

