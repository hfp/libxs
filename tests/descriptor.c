#include <libxs_gemm_diff.h>
#include <libxs_cpuid_x86.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main()
{
  const char* archid;
  const int cpuid = libxs_cpuid_x86(&archid);
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  union { libxs_gemm_descriptor descriptor; char simd[LIBXS_ALIGNMENT]; } a, b;
  unsigned int i;

  LIBXS_GEMM_DESCRIPTOR(a.descriptor, LIBXS_ALIGNMENT, LIBXS_FLAGS,
    LIBXS_LD(m, n), LIBXS_LD(n, m), k,
    LIBXS_LD(lda, ldb), LIBXS_LD(ldb, lda), ldc,
    LIBXS_ALPHA, LIBXS_BETA,
    LIBXS_PREFETCH_NONE);
  LIBXS_GEMM_DESCRIPTOR(b.descriptor, LIBXS_ALIGNMENT, LIBXS_FLAGS,
    LIBXS_LD(m, n), LIBXS_LD(n, m), k,
    LIBXS_LD(lda, ldb), LIBXS_LD(ldb, lda), ldc,
    LIBXS_ALPHA, LIBXS_BETA,
    LIBXS_PREFETCH_BL2_VIA_C);

#if defined(LIBXS_GEMM_DIFF_MASK_A)
  for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = 'a'; b.simd[i] = 'b';
  }
#else
  for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(a.simd); ++i) {
    a.simd[i] = b.simd[i] = 0;
  }
#endif

  if (0 == libxs_gemm_diff_sw(&a.descriptor, &b.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 1;
  }
  else if (0 == libxs_gemm_diff_sw(&b.descriptor, &a.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 2;
  }
  else if (0 != libxs_gemm_diff_sw(&a.descriptor, &a.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 3;
  }
  else if (0 != libxs_gemm_diff_sw(&b.descriptor, &b.descriptor)) {
    fprintf(stderr, "using generic code path\n");
    return 4;
  }
  if (LIBXS_X86_SSE3 <= cpuid) {
    if (0 == libxs_gemm_diff_sse(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 5;
    }
    else if (0 == libxs_gemm_diff_sse(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 6;
    }
    else if (0 != libxs_gemm_diff_sse(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 7;
    }
    else if (0 != libxs_gemm_diff_sse(&b.descriptor, &b.descriptor)) {
      fprintf(stderr, "using SSE code path\n");
      return 8;
    }
  }
  if (LIBXS_X86_AVX <= cpuid) {
    if (0 == libxs_gemm_diff_avx(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 9;
    }
    else if (0 == libxs_gemm_diff_avx(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 10;
    }
    else if (0 != libxs_gemm_diff_avx(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 11;
    }
    else if (0 != libxs_gemm_diff_avx(&b.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX code path\n");
      return 12;
    }
  }
  if (LIBXS_X86_AVX2 <= cpuid) {
    if (0 == libxs_gemm_diff_avx2(&a.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 13;
    }
    else if (0 == libxs_gemm_diff_avx2(&b.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 14;
    }
    else if (0 != libxs_gemm_diff_avx2(&a.descriptor, &a.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 15;
    }
    else if (0 != libxs_gemm_diff_avx2(&b.descriptor, &b.descriptor)) {
      fprintf(stderr, "using AVX2 code path\n");
      return 16;
    }
  }
  if (1 != libxs_gemm_diffn(&a.descriptor, &b.descriptor, 0, 1, sizeof(libxs_gemm_descriptor))) {
    fprintf(stderr, "using dispatched code path\n");
    return 17;
  }
  else if (1 != libxs_gemm_diffn(&b.descriptor, &a.descriptor, 0, 1, sizeof(libxs_gemm_descriptor))) {
    fprintf(stderr, "using dispatched code path\n");
    return 18;
  }
  else if (0 != libxs_gemm_diffn(&a.descriptor, &a.descriptor, 0, 1, sizeof(libxs_gemm_descriptor))) {
    fprintf(stderr, "using dispatched code path\n");
    return 19;
  }
  else if (0 != libxs_gemm_diffn(&b.descriptor, &b.descriptor, 0, 1, sizeof(libxs_gemm_descriptor))) {
    fprintf(stderr, "using dispatched code path\n");
    return 20;
  }

  return EXIT_SUCCESS;
}

