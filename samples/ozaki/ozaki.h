/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_malloc.h>
#include <libxs_mhd.h>
#include <libxs_sync.h>

#if GEMM_IS_DOUBLE
# define OZ_MANT_BITS 52
# define OZ_EXP_BIAS 1023
#else /* single-precision */
# define OZ_MANT_BITS 23
# define OZ_EXP_BIAS 127
#endif
#define OZ_BIAS_PLUS_MANT (OZ_EXP_BIAS + OZ_MANT_BITS)
#define OZ_EXP_MASK (2U * OZ_EXP_BIAS + 1U)

#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif
#if !defined(BATCH_K)
# define BATCH_K 4
#endif
#if !defined(OZ2_SIGNED)
# define OZ2_SIGNED 1
#endif

#if !defined(MAX_NSLICES)
# if GEMM_IS_DOUBLE
#   define MAX_NSLICES 16
# else
#   define MAX_NSLICES 8
# endif
#endif
#if !defined(NSLICES_DEFAULT)
# if GEMM_IS_DOUBLE
#   define NSLICES_DEFAULT 8
# else
#   define NSLICES_DEFAULT 4
# endif
#endif

/* Runtime flag-set controlling the Ozaki scheme 1 (GEMM_OZFLAGS env var).
 * Bit 0 (1): TRIANGULAR  - iterate upper triangle of slice-pair matrix
 * Bit 1 (2): SYMMETRIZE  - compute mirror D(sb,sa) for off-diagonal pairs
 * Default 3 = TRIANGULAR + SYMMETRIZE (correct, fewer loop iterations).
 *
 * The trim parameter (GEMM_OZTRIM env var) drops the T least significant
 * diagonals: pairs with sa + sb > 2*(S-1) - T are skipped. Default 0
 * means exact (all pairs). Each dropped diagonal loses ~7 bits. */
#define OZ1_TRIANGULAR   1
#define OZ1_SYMMETRIZE   2
#define OZ1_DEFAULT (OZ1_TRIANGULAR | OZ1_SYMMETRIZE)

#if GEMM_IS_DOUBLE
# if 0 != OZ2_SIGNED
#   define OZ2_NPRIMES_MAX 18
#   define OZ2_NPRIMES_DEFAULT 17
# else
#   define OZ2_NPRIMES_MAX 17
#   define OZ2_NPRIMES_DEFAULT 15
# endif
#else /* single-precision */
# if 0 != OZ2_SIGNED
#   define OZ2_NPRIMES_MAX 10
#   define OZ2_NPRIMES_DEFAULT 8
# else
#   define OZ2_NPRIMES_MAX 10
#   define OZ2_NPRIMES_DEFAULT 7
# endif
#endif

/** Implement the public gemm_ozN function: call the _diff kernel,
 *  then handle verbose output, diff accumulation, and matrix dumps.
 *  DIFF_FN is the _diff kernel (gemm_oz1_diff or gemm_oz2_diff). */
#define OZAKI_GEMM_WRAPPER(DIFF_FN) \
  if (0 == gemm_verbose) { \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, \
      0, NULL); \
  } \
  else { \
    double epsilon; \
    libxs_matdiff_info_t diff; \
    libxs_matdiff_clear(&diff); \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, \
      LIBXS_ABS(gemm_stat), &diff); \
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER); \
    libxs_matdiff_reduce(&gemm_diff, &diff); diff.r = gemm_diff.r; \
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER); \
    epsilon = libxs_matdiff_epsilon(&diff); \
    if (1 < gemm_verbose || 0 > gemm_verbose) { \
      const int nth = (0 < gemm_verbose ? gemm_verbose : 1); \
      if (0 == (diff.r % nth)) print_diff(stderr, &diff); \
    } \
    if (gemm_eps < epsilon || diff.rsq < gemm_rsq || -1 > gemm_verbose) { \
      if (0 != gemm_dump_inhibit) { \
        gemm_dump_inhibit = 2; \
      } \
      else { \
        gemm_dump_matrices(GEMM_ARGPASS, 1); \
        if (0 != gemm_exit) exit(EXIT_FAILURE); \
      } \
    } \
  }

/* Wrap/real symbol definitions for real GEMM */
#define GEMM_WRAP LIBXS_CONCATENATE(__wrap_, GEMM)
#define GEMM_REAL LIBXS_CONCATENATE(__real_, GEMM)

/* Wrap/real symbol definitions for complex GEMM */
#define ZGEMM_WRAP LIBXS_CONCATENATE(__wrap_, ZGEMM)
#define ZGEMM_REAL LIBXS_CONCATENATE(__real_, ZGEMM)

/* Precision-specific type and variable names (enable dual-precision builds).
 * These macros redirect "friendly" names used throughout the implementation
 * to unique symbols, e.g. gemm_original -> dgemm_original (double) or
 * sgemm_original (float). Both precisions can coexist in one binary. */
#define gemm_function_t     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define zgemm_function_t    LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define gemm_original       LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_original)
#define zgemm_original      LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_original)
#define gemm_lock           LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_lock)
#define gemm_ozaki          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozaki)
#define gemm_ozn            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozn)
#define gemm_ozflags        LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozflags)
#define gemm_oztrim         LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oztrim)
#define gemm_stat           LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_stat)
#define gemm_exit           LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_exit)
#define gemm_eps            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_eps)
#define gemm_rsq            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_rsq)
#define ozaki_target_arch   LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_tarch)
#define gemm_oz1            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz1)
#define gemm_oz2            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz2)
#define gemm_dump_inhibit   LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_inhibit)
#define gemm_dump_matrices  LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_mhd)
#define zgemm3m             LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm3m)
#define gemm_atexit         LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_atexit)

/** Function type for GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*gemm_function_t)(GEMM_ARGDECL);
/** Function type for complex GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*zgemm_function_t)(GEMM_ARGDECL);

/** Function prototypes for wrapped / real / public GEMM and complex GEMM. */
LIBXS_API_INTERN void GEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void GEMM_REAL(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_REAL(GEMM_ARGDECL);
LIBXS_API void ZGEMM(GEMM_ARGDECL);

/** Function prototype for GEMM using low-precision (Ozaki scheme 1). */
LIBXS_API void gemm_oz1(GEMM_ARGDECL);
/** Function prototype for GEMM using CRT modular arithmetic (Ozaki scheme 2). */
LIBXS_API void gemm_oz2(GEMM_ARGDECL);
/** Complex GEMM 3M (Karatsuba) implementation (internal). */
LIBXS_API_INTERN void zgemm3m(GEMM_ARGDECL);

LIBXS_APIVAR_PUBLIC(int gemm_ozaki);
LIBXS_APIVAR_PUBLIC(int gemm_stat);
LIBXS_APIVAR_PRIVATE(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE(zgemm_function_t zgemm_original);
LIBXS_APIVAR_PRIVATE(int ozaki_target_arch);
LIBXS_APIVAR_PRIVATE(int gemm_ozflags);
LIBXS_APIVAR_PRIVATE(int gemm_oztrim);
LIBXS_APIVAR_PRIVATE(int gemm_ozn);
LIBXS_APIVAR_PRIVATE(int gemm_exit);
extern LIBXS_TLS int gemm_dump_inhibit;
LIBXS_APIVAR_PRIVATE(double gemm_eps);
LIBXS_APIVAR_PRIVATE(double gemm_rsq);

/* ================================================================ */
/* Shared int8 dot-product infrastructure (VNNI + scalar fallback)  */
/* ================================================================ */

/* For signed int8 dot products: VPDPBUSD treats the first operand as
 * unsigned by XOR with 0x80 (+128), then subtract 128*sum(b[]) to
 * compensate.  AVX-512 VNNI exposes VPDPBUSD at all three register
 * widths (XMM/128, YMM/256, ZMM/512); we select the narrowest that
 * covers BLOCK_K bytes, falling back to scalar for other sizes. */
#if defined(LIBXS_INTRINSICS_AVX512)

# if 16 == BLOCK_K /* 128-bit: one __m128i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i va = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), bias);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  const __m128i ones = _mm_set1_epi8(1);
  __m128i dp = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
  __m128i sb = _mm_dpbusd_epi32(_mm_setzero_si128(), ones, vb);
  dp = _mm_hadd_epi32(dp, sb);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0) - 128 * _mm_extract_epi32(dp, 1);
}

# elif 32 == BLOCK_K /* 256-bit: one __m256i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i va = _mm256_xor_si256(
    _mm256_loadu_si256((const __m256i*)a), bias);
  const __m256i vb = _mm256_loadu_si256((const __m256i*)b);
  const __m256i ones = _mm256_set1_epi8(1);
  __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, vb);
  __m256i sb = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, vb);
  { const __m128i hi_dp = _mm256_extracti128_si256(dp, 1);
    const __m128i hi_sb = _mm256_extracti128_si256(sb, 1);
    __m128i lo_dp = _mm256_castsi256_si128(dp);
    __m128i lo_sb = _mm256_castsi256_si128(sb);
    lo_dp = _mm_add_epi32(lo_dp, hi_dp);
    lo_sb = _mm_add_epi32(lo_sb, hi_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_dp);
    return _mm_extract_epi32(lo_dp, 0) - 128 * _mm_extract_epi32(lo_dp, 1);
  }
}

# elif 64 == BLOCK_K /* 512-bit: one __m512i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m512i bias = _mm512_set1_epi8((char)0x80);
  const __m512i va = _mm512_xor_si512(
    _mm512_loadu_si512((const __m512i*)a), bias);
  const __m512i vb = _mm512_loadu_si512((const __m512i*)b);
  const __m512i ones = _mm512_set1_epi8(1);
  __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, vb);
  __m512i sb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, vb);
  return _mm512_reduce_add_epi32(dp)
       - 128 * _mm512_reduce_add_epi32(sb);
}

# endif /* BLOCK_K width selection */
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API_INLINE int32_t ozaki_dot_i8_sw(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* Function pointer type for int8 dot product dispatch. */
typedef int32_t (*ozaki_dot_i8_fn)(const int8_t[BLOCK_K], const int8_t[BLOCK_K]);

#if defined(LIBXS_INTRINSICS_AVX512) && \
    (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) /* VNNI guaranteed */
#   define ozaki_dot_i8_init() ozaki_dot_i8_vnni
# else /* runtime dispatch */
#   define ozaki_dot_i8_init() \
      ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_i8_vnni : ozaki_dot_i8_sw)
# endif
#else
# define ozaki_dot_i8_init() ozaki_dot_i8_sw
#endif


/** Extract IEEE-754 biased exponent and full mantissa (with implicit bit)
 *  into uint64_t.  Returns sign (+1 or -1); for zero/subnormal/NaN/Inf
 *  sets exp_biased=0 and mantissa=0, returns +1. */
LIBXS_API_INLINE int ozaki_extract_ieee(GEMM_REAL_TYPE value,
  int16_t* exp_biased, uint64_t* mantissa)
{
  const union { uint32_t raw; float value; } inf = { 0x7F800000U };
  int sign;

  if (value == (GEMM_REAL_TYPE)0 || LIBXS_ISNAN(value)
    || (float)value == inf.value || (float)value == -inf.value)
  {
    *exp_biased = 0; *mantissa = 0;
    return 1;
  }

  sign = (value < (GEMM_REAL_TYPE)0) ? -1 : 1;
  if (value < (GEMM_REAL_TYPE)0) value = -value;

  { uint64_t bits;
#if GEMM_IS_DOUBLE
    { union { double d; uint64_t u; } cvt; cvt.d = value; bits = cvt.u; }
#else
    { union { float f; uint32_t u; } cvt; cvt.f = value; bits = cvt.u; }
#endif
    { const uint64_t frac = bits & ((1ULL << OZ_MANT_BITS) - 1ULL);
      const uint16_t exp_raw = (uint16_t)(
        (bits >> OZ_MANT_BITS) & OZ_EXP_MASK);
      if (0 == exp_raw) { /* subnormal treated as zero */
        *exp_biased = 0; *mantissa = 0;
        return sign;
      }
      *exp_biased = (int16_t)exp_raw;
      *mantissa = (1ULL << OZ_MANT_BITS) | frac;
    }
  }
  return sign;
}


/** Split a (pre-aligned) mantissa into signed 7-bit digits.
 *  The mantissa is expected to be in the same format as produced
 *  by ozaki_extract_ieee (implicit bit at position OZ_MANT_BITS),
 *  but may have been right-shifted for exponent alignment. */
LIBXS_API_INLINE void ozaki_split_digits(uint64_t mantissa, int sign,
  int8_t digits[MAX_NSLICES])
{
  int s;
  if (0 == mantissa) {
    memset(digits, 0, sizeof(int8_t) * gemm_ozn);
    return;
  }
  LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
  for (s = 0; s < gemm_ozn; ++s) {
    const int high = OZ_MANT_BITS - (7 * s);
    if (high < 0) {
      digits[s] = 0;
      continue;
    }
    { const int low = high - 6;
      uint64_t chunk;
      if (low >= 0) {
        chunk = (mantissa >> low) & 0x7FULL;
      }
      else {
        const int width = high + 1;
        chunk = mantissa & ((1ULL << width) - 1ULL);
      }
      digits[s] = (int8_t)(sign * (int64_t)chunk);
    }
  }
}


/** Scale a tile of C by beta, optionally capturing the pre-scaled block.
 *  Per BLAS spec, beta=0 must zero out C unconditionally (C may hold NaN
 *  or Inf when uninitialized); a plain multiply would give NaN. */
LIBXS_API_INLINE void ozaki_scale_block_beta(GEMM_REAL_TYPE* mb, GEMM_INT_TYPE ldc,
  GEMM_INT_TYPE iblk, GEMM_INT_TYPE jblk, const GEMM_REAL_TYPE* beta,
  GEMM_REAL_TYPE* ref_blk, int capture_ref)
{
  const GEMM_REAL_TYPE b = *beta;
  GEMM_INT_TYPE mi, nj;
  for (mi = 0; mi < iblk; ++mi) {
    for (nj = 0; nj < jblk; ++nj) {
      if (0 != capture_ref) ref_blk[mi + nj * BLOCK_M] = mb[mi + nj * ldc];
      mb[mi + nj * ldc] = ((GEMM_REAL_TYPE)0 != b)
        ? mb[mi + nj * ldc] * b : (GEMM_REAL_TYPE)0;
    }
  }
}


/** Store a (reference, reconstructed) value pair into block buffers. */
LIBXS_API_INLINE void ozaki_store_block_pair(GEMM_REAL_TYPE* ref_blk,
  GEMM_REAL_TYPE* recon_blk, GEMM_INT_TYPE ld, GEMM_INT_TYPE row,
  GEMM_INT_TYPE col, GEMM_REAL_TYPE ref_val, GEMM_REAL_TYPE recon_val)
{
  recon_blk[row + col * ld] = recon_val;
  ref_blk[row + col * ld] = ref_val;
}


/** Compute matrix diff for one block and reduce into accumulator. */
LIBXS_API_INLINE void ozaki_accumulate_block_diff(libxs_matdiff_info_t* acc,
  const GEMM_REAL_TYPE* ref_blk, const GEMM_REAL_TYPE* tst_blk,
  GEMM_INT_TYPE bm, GEMM_INT_TYPE bn, GEMM_INT_TYPE ld_ref,
  GEMM_INT_TYPE ld_tst)
{
  libxs_matdiff_info_t block_diff;
  const int ild_ref = (int)ld_ref, ild_tst = (int)ld_tst;
  if (EXIT_SUCCESS == libxs_matdiff(&block_diff, LIBXS_DATATYPE(GEMM_REAL_TYPE),
    bm, bn, ref_blk, tst_blk, &ild_ref, &ild_tst))
  {
    libxs_matdiff_reduce(acc, &block_diff);
  }
}


/**
 * Dump A and B matrices as MHD files.
 * Works for both real (ncomponents=1) and complex (ncomponents=2) matrices.
 * Uses gemm_diff.r as the dump ID and updates gemm_eps/gemm_rsq thresholds
 * to avoid repeated dumps.
 */
LIBXS_API_INLINE void gemm_dump_matrices(GEMM_ARGDECL, size_t ncomponents)
{
  gemm_mhd_settings_t settings;
  char fname[64];
  int result = EXIT_SUCCESS;
  FILE *file;

  settings.ozaki = gemm_ozaki;
  settings.ozn = gemm_ozn;
  settings.ozflags = gemm_ozflags;
  settings.oztrim = gemm_oztrim;
  settings.ldc = *ldc;
  settings.eps = gemm_eps;
  settings.rsq = gemm_rsq;

  LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);

  LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%u-%i-a.mhd", libxs_pid(), gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    result |= gemm_mhd_write(fname, a, *m, *k, *lda, *transa, alpha,
      ncomponents, &settings);
  }
  else fclose(file);

  LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%u-%i-b.mhd", libxs_pid(), gemm_diff.r);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    result |= gemm_mhd_write(fname, b, *k, *n, *ldb, *transb, beta,
      ncomponents, &settings);
  }
  else fclose(file);

  if (EXIT_SUCCESS == result) {
  /* avoid repeated dumps */
  gemm_eps = libxs_matdiff_epsilon(&gemm_diff);
  gemm_rsq = gemm_diff.rsq;
  }
  else if (0 != gemm_verbose) {
    fprintf(stderr, "ERROR: dumping A and B matrix failed!\n");
  }

  LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
}
