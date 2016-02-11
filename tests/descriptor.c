#include <libxs_gemm_diff.h>
#include <libxs_cpuid.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main()
{
  int is_static, has_crc32;
  const char *const cpuid = libxs_cpuid(&is_static, &has_crc32);
  const libxs_gemm_diff_function diff = 0 != cpuid
    ? (/*snb*/'b' != cpuid[2] ? libxs_gemm_diff_avx2 : libxs_gemm_diff_avx)
    : (0 != has_crc32/*sse4.2*/ ? libxs_gemm_diff_sse : libxs_gemm_diff);
  const int m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  libxs_gemm_descriptor a, b, c;

  LIBXS_GEMM_DESCRIPTOR(a, LIBXS_ALIGNMENT, LIBXS_FLAGS,
    LIBXS_LD(m, n), LIBXS_LD(n, m), k,
    LIBXS_LD(lda, ldb), LIBXS_LD(ldb, lda), ldc,
    LIBXS_ALPHA, LIBXS_BETA,
    LIBXS_PREFETCH_NONE);
  LIBXS_GEMM_DESCRIPTOR(b, LIBXS_ALIGNMENT, LIBXS_FLAGS,
    LIBXS_LD(m, n), LIBXS_LD(n, m), k,
    LIBXS_LD(lda, ldb), LIBXS_LD(ldb, lda), ldc,
    LIBXS_ALPHA, LIBXS_BETA,
    LIBXS_PREFETCH_BL2_VIA_C);
  c = a;

  if (0 == libxs_gemm_diff(&a, &b)) {
    fprintf(stderr, "using static code path\n");
    return 1;
  }
  else if (0 != libxs_gemm_diff(&a, &c)) {
    fprintf(stderr, "using static code path\n");
    return 2;
  }
  else if (0 == libxs_gemm_diff(&b, &c)) {
    fprintf(stderr, "using static code path\n");
    return 3;
  }
  else if (0 == diff(&a, &b)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 4;
  }
  else if (0 != diff(&a, &c)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 5;
  }
  else if (0 == diff(&b, &c)) {
    fprintf(stderr, "using %s code path\n", cpuid
      ? cpuid : (0 != has_crc32 ? "SSE"
      : (0 != is_static ? "static" : "non-AVX")));
    return 6;
  }
  else {
    return EXIT_SUCCESS;
  }
}

