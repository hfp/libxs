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

#define INTERNAL_TRANSPOSE_OOP(TYPE, OUT, IN, M0, M1, N0, N1, N) { \
  const TYPE *const a = (const TYPE*)IN; \
  TYPE *const b = (TYPE*)OUT; \
  libxs_blasint i, j; \
  if (LIBXS_TRANSPOSE_CHUNK == N) { \
    for (i = M0; i < M1; ++i) { \
      LIBXS_PRAGMA_NONTEMPORAL \
      for (j = N0; j < N0 + LIBXS_TRANSPOSE_CHUNK; ++j) { \
        b[i*ldo+j/*consecutive*/] = a[j*ld+i/*strided*/]; \
      } \
    } \
  } \
  else { \
    for (i = M0; i < M1; ++i) { \
      LIBXS_PRAGMA_NONTEMPORAL \
      for (j = N0; j < N1; ++j) { \
        b[i*ldo+j/*consecutive*/] = a[j*ld+i/*strided*/]; \
      } \
    } \
  } \
}


/* Based on cache-oblivious scheme as published by Frigo et.al. Further optimization for compile-time bounded loops. */
LIBXS_INLINE LIBXS_RETARGETABLE void inernal_transpose_oop(void *LIBXS_RESTRICT out, const void *LIBXS_RESTRICT in,
  unsigned int typesize, libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  const libxs_blasint m = m1 - m0, n = n1 - n0;

  if (m * n * typesize <= (LIBXS_TRANSPOSE_CACHESIZE / 2)) {
    switch(typesize) {
      case 1: {
        INTERNAL_TRANSPOSE_OOP(char, out, in, m0, m1, n0, n1, n);
      } break;
      case 2: {
        INTERNAL_TRANSPOSE_OOP(short, out, in, m0, m1, n0, n1, n);
      } break;
      case 4: {
        INTERNAL_TRANSPOSE_OOP(float, out, in, m0, m1, n0, n1, n);
      } break;
      case 8: {
        INTERNAL_TRANSPOSE_OOP(double, out, in, m0, m1, n0, n1, n);
      } break;
      case 16: {
        typedef struct dvec2_t { double value[2]; } dvec2_t;
        INTERNAL_TRANSPOSE_OOP(dvec2_t, out, in, m0, m1, n0, n1, n);
      } break;
      default: {
#if !defined(NDEBUG) /* library code is expected to be mute */
        fprintf(stderr, "LIBXS: unsupported element type in transpose!\n");
#endif
        assert(0);
      }
    }
  }
  else if (m >= n) {
    const libxs_blasint mi = (m0 + m1) / 2;
    inernal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
    inernal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
  }
  else {
#if (0 < LIBXS_TRANSPOSE_CHUNK)
    if (LIBXS_TRANSPOSE_CHUNK < n) {
      const libxs_blasint ni = n0 + LIBXS_TRANSPOSE_CHUNK;
      inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
    else
#endif
    {
      const libxs_blasint ni = (n0 + n1) / 2;
      inernal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      inernal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
  }
}


LIBXS_API_DEFINITION void libxs_transpose_oop(void* out, const void* in, unsigned int typesize,
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
    mkl_domatcopy('C', 'T', m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
  }
  else if (4 == typesize) {
    mkl_somatcopy('C', 'T', m, n, 1.f, (const float*)in, ld, (float*)out, ldo);
  }
  else if (16 == typesize) {
    const MKL_Complex16 one = { 1.0/*real*/, 0.0/*imag*/ };
    mkl_zomatcopy('C', 'T', m, n, one, (const MKL_Complex16*)in, ld, (MKL_Complex16*)out, ldo);
  }
  else
#endif
  {
    inernal_transpose_oop(out, in, typesize, 0, m, 0, n, ld, ldo);
  }
}


LIBXS_API_DEFINITION void libxs_transpose_inp(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  LIBXS_UNUSED(inout); LIBXS_UNUSED(typesize); LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(ld);
  assert(0/*Not yet implemented!*/);
}


#if defined(LIBXS_BUILD)

LIBXS_API_DEFINITION void libxs_stranspose_oop(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXS_API_DEFINITION void libxs_dtranspose_oop(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(double), m, n, ld, ldo);
}


LIBXS_API_DEFINITION void libxs_stranspose_inp(float* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  libxs_transpose_inp(inout, sizeof(float), m, n, ld);
}


LIBXS_API_DEFINITION void libxs_dtranspose_inp(double* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  libxs_transpose_inp(inout, sizeof(double), m, n, ld);
}

#endif /*defined(LIBXS_BUILD)*/
