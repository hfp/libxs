/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"

#if defined(OZAKI_TESTROOT) && defined(OZAKI_TEST_N) && GEMM_IS_DOUBLE

#define N OZAKI_TEST_N
#define OZT_EXP_RANGE 10


typedef signed char SINT8;
typedef unsigned char UINT8;
typedef int64_t SINT64;
typedef uint64_t UINT64;
typedef enum { OZT_SUCCESS, OZT_INF_NAN } ozaki_test_status_t;
typedef union { UINT64 w; UINT8 w8[8]; double f; } element64;

static element64 ozt_A[N], ozt_B[N];
static short ozt_EA[N], ozt_EB[N], ozt_ESUM[N];


static ozaki_test_status_t ozt_acc_dp(double* pres, element64* Ain, element64* Bin)
{
  int i, max_Exp_P, max_Exp_A, max_Exp_B, scale;
  SINT64 sum[10], lsum, lsum_h, ldpres;
  UINT64 lsum_l;
  double dres, dres_h;

  for (i = 0; i < N; i++) ozt_EA[i] = (short)(((Ain[i].w) >> 52) & 0x7ff);
  for (i = 0; i < N; i++) ozt_EB[i] = (short)(((Bin[i].w) >> 52) & 0x7ff);
  for (i = 0; i < N; i++) {
    ozt_ESUM[i] = (short)((ozt_EA[i] & (short)((0 - ozt_EB[i]) >> 15))
                         + (ozt_EB[i] & (short)((0 - ozt_EA[i]) >> 15)));
    ozt_ESUM[i] |= (short)((((ozt_EA[i] + 1) | (ozt_EB[i] + 1)) & 0x800) << 4);
  }
  max_Exp_P = ozt_ESUM[0];
  for (i = 1; i < N; i++) max_Exp_P = LIBXS_MAX(max_Exp_P, (int)ozt_ESUM[i]);
  if (max_Exp_P & 0x8000) return OZT_INF_NAN;
  if (!max_Exp_P) { *pres = 0; return OZT_SUCCESS; }

  max_Exp_A = max_Exp_P >> 1;
  max_Exp_B = (max_Exp_P + 1) >> 1;
  for (i = 0; i < N; i++) {
    SINT64 sgn = ((SINT64)Ain[i].w) >> 63;
    SINT64 ltmp = (Ain[i].w & 0x000fffffffffffffll) | 0x0010000000000000ll;
    int expon, shift;
    ltmp = (Ain[i].w & 0x7ff0000000000000ll) ? ltmp : 0ll;
    ltmp = ltmp << 10;
    expon = (ozt_ESUM[i] + 0) >> 1;
    shift = max_Exp_A - expon;
    ltmp = (expon == 0) ? 0 : ltmp;
    shift = LIBXS_MIN(shift, 63);
    ltmp = ltmp >> shift;
    ozt_A[i].w = (UINT64)((ltmp + sgn) ^ sgn);
  }
  for (i = 0; i < N; i++) {
    SINT64 sgn = ((SINT64)Bin[i].w) >> 63;
    SINT64 ltmp = (Bin[i].w & 0x000fffffffffffffll) | 0x0010000000000000ll;
    int expon, shift;
    ltmp = (Bin[i].w & 0x7ff0000000000000ll) ? ltmp : 0ll;
    ltmp = ltmp << 10;
    expon = (ozt_ESUM[i] + 1) >> 1;
    shift = max_Exp_B - expon;
    ltmp = (expon == 0) ? 0 : ltmp;
    shift = LIBXS_MIN(shift, 63);
    ltmp = ltmp >> shift;
    ozt_B[i].w = (UINT64)((ltmp + sgn) ^ sgn);
  }

  for (i = 0; i < 9; i++) sum[i] = 0;
  lsum = lsum_h = 0;

#define OZT_S8U8(R, X, xi, Y, yi) { int j_; (R) = 0; for (j_ = 0; j_ < N; j_++) (R) += (SINT8)X[j_].w8[xi] * (UINT8)Y[j_].w8[yi]; }
#define OZT_U8U8(R, X, xi, Y, yi) { int j_; (R) = 0; for (j_ = 0; j_ < N; j_++) (R) += (UINT8)X[j_].w8[xi] * (UINT8)Y[j_].w8[yi]; }
#define OZT_S8S8(R, X, xi, Y, yi) { int j_; (R) = 0; for (j_ = 0; j_ < N; j_++) (R) += (SINT8)X[j_].w8[xi] * (SINT8)Y[j_].w8[yi]; }

  OZT_S8U8(sum[7], ozt_A, 7, ozt_B, 0); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 0); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 1); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 2); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 3); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 4); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 2, ozt_B, 5); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 1, ozt_B, 6); sum[7] += ldpres;
  lsum += (sum[7] << 8);

  OZT_S8U8(sum[6], ozt_A, 7, ozt_B, 1); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 1); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 2); sum[6] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 3); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 4); sum[6] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 5); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 2, ozt_B, 6); sum[6] += ldpres;
  lsum += (sum[6] << 16);

  OZT_S8U8(sum[5], ozt_A, 7, ozt_B, 2); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 2); sum[5] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 3); sum[5] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 4); sum[5] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 5); sum[5] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 6); sum[5] += ldpres;
  lsum += (sum[5] << 24);
  lsum_l = lsum & 0xfffffffffffffull;
  lsum_h = lsum >> 52;

  OZT_S8U8(sum[4], ozt_A, 7, ozt_B, 3); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 3); sum[4] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 4); sum[4] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 5); sum[4] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 6); sum[4] += ldpres;
  lsum_l += ((UINT64)(sum[4] & 0xfffff) << 32);
  lsum_h += (sum[4] >> 20);

  OZT_S8U8(sum[3], ozt_A, 7, ozt_B, 4); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 4); sum[3] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 5); sum[3] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 6); sum[3] += ldpres;
  lsum_l += ((UINT64)(sum[3] & 0xfff) << 40);
  lsum_h += (sum[3] >> 12);

  OZT_S8U8(sum[2], ozt_A, 7, ozt_B, 5); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 5); sum[2] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 6); sum[2] += ldpres;
  lsum_l += ((UINT64)(sum[2] & 0xf) << 48);
  lsum_h += (sum[2] >> 4);

  OZT_S8U8(sum[1], ozt_A, 7, ozt_B, 6); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 6); sum[1] += ldpres;
  lsum_h += (sum[1] << 4);

  OZT_S8S8(sum[0], ozt_A, 7, ozt_B, 7);
  lsum_h += (sum[0] << 12);

  dres = (double)lsum_l; dres_h = (double)lsum_h;
  dres += dres_h * 0x1p+52;
  scale = max_Exp_P - 60 - 16 - 0x3ff * 2;
  *pres = ldexp(dres, scale);
  return OZT_SUCCESS;
}


static ozaki_test_status_t ozt_dp(double* pres, element64* Ain, element64* Bin)
{
  int i, max_Exp_A, max_Exp_B, min_Exp_A, min_Exp_B, scale;
  SINT64 sum[10], lsum, lsum_h, lsum_l, ldpres;
  double dres, dres_h;
  SINT64 ltmp, ltmp2, ltmp3, lr;

  ltmp2 = ltmp = Ain[0].w & 0x7ff0000000000000ll;
  for (i = 1; i < N; i++) {
    lr = Ain[i].w & 0x7ff0000000000000ll;
    ltmp = LIBXS_MAX(ltmp, lr);
    ltmp3 = LIBXS_MIN(ltmp2, lr);
    ltmp2 = (!ltmp3) ? ltmp2 : ltmp3;
  }
  max_Exp_A = (int)(ltmp >> 52); min_Exp_A = (int)(ltmp2 >> 52);

  ltmp2 = ltmp = Bin[0].w & 0x7ff0000000000000ll;
  for (i = 1; i < N; i++) {
    lr = Bin[i].w & 0x7ff0000000000000ll;
    ltmp = LIBXS_MAX(ltmp, lr);
    ltmp3 = LIBXS_MIN(ltmp2, lr);
    ltmp2 = (!ltmp3) ? ltmp2 : ltmp3;
  }
  max_Exp_B = (int)(ltmp >> 52); min_Exp_B = (int)(ltmp2 >> 52);

  if (((max_Exp_A + 1) | (max_Exp_B + 1)) & 0x800) return OZT_INF_NAN;
  if (((max_Exp_A - 1) | (max_Exp_B - 1)) & 0x800) { *pres = 0; return OZT_SUCCESS; }

  if ((max_Exp_A - min_Exp_A > OZT_EXP_RANGE) || (max_Exp_B - min_Exp_B > OZT_EXP_RANGE))
    return ozt_acc_dp(pres, Ain, Bin);

  for (i = 0; i < N; i++) {
    SINT64 sgn = ((SINT64)Ain[i].w) >> 63;
    SINT64 m = (Ain[i].w & 0x000fffffffffffffll) | 0x0010000000000000ll;
    int expon, shift;
    m = (Ain[i].w & 0x7ff0000000000000ll) ? m : 0ll;
    m = (m + sgn) ^ sgn;
    m = m << 10;
    expon = ((Ain[i].w >> 52) & 0x7ff);
    shift = max_Exp_A - expon;
    m = (expon == 0) ? 0 : m;
    shift = (expon == 0) ? 0 : shift;
    shift = LIBXS_MIN(shift, 63);
    ozt_A[i].w = (UINT64)(m >> shift);
  }
  for (i = 0; i < N; i++) {
    SINT64 sgn = ((SINT64)Bin[i].w) >> 63;
    SINT64 m = (Bin[i].w & 0x000fffffffffffffll) | 0x0010000000000000ll;
    int expon, shift;
    m = (Bin[i].w & 0x7ff0000000000000ll) ? m : 0ll;
    m = (m + sgn) ^ sgn;
    m = m << 10;
    expon = ((Bin[i].w >> 52) & 0x7ff);
    shift = max_Exp_B - expon;
    m = (expon == 0) ? 0 : m;
    shift = (expon == 0) ? 0 : shift;
    shift = LIBXS_MIN(shift, 63);
    ozt_B[i].w = (UINT64)(m >> shift);
  }

  for (i = 0; i < 8; i++) sum[i] = 0;
  lsum = lsum_h = 0;

  OZT_S8U8(sum[7], ozt_A, 7, ozt_B, 0); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 0); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 1); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 2); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 3); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 4); sum[7] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 2, ozt_B, 5); sum[7] += ldpres; OZT_U8U8(ldpres, ozt_A, 1, ozt_B, 6); sum[7] += ldpres;
  lsum += sum[7];

  OZT_S8U8(sum[6], ozt_A, 7, ozt_B, 1); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 1); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 2); sum[6] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 3); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 4); sum[6] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 5); sum[6] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 2, ozt_B, 6); sum[6] += ldpres;
  lsum += (sum[6] << 8);

  OZT_S8U8(sum[5], ozt_A, 7, ozt_B, 2); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 2); sum[5] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 3); sum[5] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 4); sum[5] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 5); sum[5] += ldpres; OZT_U8U8(ldpres, ozt_A, 3, ozt_B, 6); sum[5] += ldpres;
  lsum += (sum[5] << 16);

  OZT_S8U8(sum[4], ozt_A, 7, ozt_B, 3); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 3); sum[4] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 4); sum[4] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 5); sum[4] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 4, ozt_B, 6); sum[4] += ldpres;
  lsum += (sum[4] << 24);
  lsum_l = lsum & 0xfffffffffffffll;
  lsum_h = lsum >> 52;

  OZT_S8U8(sum[3], ozt_A, 7, ozt_B, 4); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 4); sum[3] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 5); sum[3] += ldpres; OZT_U8U8(ldpres, ozt_A, 5, ozt_B, 6); sum[3] += ldpres;
  lsum_l += ((sum[3] & 0xfffff) << 32);
  lsum_h += (sum[3] >> 20);

  OZT_S8U8(sum[2], ozt_A, 7, ozt_B, 5); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 5); sum[2] += ldpres;
  OZT_U8U8(ldpres, ozt_A, 6, ozt_B, 6); sum[2] += ldpres;
  lsum_l += ((sum[2] & 0xfff) << 40);
  lsum_h += (sum[2] >> 12);

  OZT_S8U8(sum[1], ozt_A, 7, ozt_B, 6); OZT_S8U8(ldpres, ozt_B, 7, ozt_A, 6); sum[1] += ldpres;
  lsum_l += ((sum[1] & 0xf) << 48);
  lsum_h += (sum[1] >> 4);

  OZT_S8S8(sum[0], ozt_A, 7, ozt_B, 7);
  lsum_h += (sum[0] << 4);

  dres = (double)lsum_l; dres_h = (double)lsum_h;
  dres += dres_h * 0x1p+52;
  scale = max_Exp_A + max_Exp_B - 60 - 8 - 0x3ff * 2;
  *pres = ldexp(dres, scale);
  return OZT_SUCCESS;
}

#undef OZT_S8U8
#undef OZT_U8U8
#undef OZT_S8S8


OZAKI_API_INTERN void gemm_oz_test_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
  const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  libxs_matdiff_t* diff)
{
  static element64 avec[OZAKI_TEST_N], bvec[OZAKI_TEST_N];
  const GEMM_INT_TYPE M = *m, N_ = *n, K = *k;
  const int ta = ('N' != *transa && 'n' != *transa);
  const int tb = ('N' != *transb && 'n' != *transb);
  GEMM_REAL_TYPE* c_ref = NULL;
  GEMM_REAL_TYPE* c_test = NULL;
  const size_t c_size = (size_t)(*ldc) * (size_t)N_ * sizeof(GEMM_REAL_TYPE);

  if (NULL != diff) {
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    c_test = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
  }

  gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  if (NULL != c_ref && K <= OZAKI_TEST_N) {
    GEMM_INT_TYPE mi, nj, ki;
    memcpy(c_test, c_ref, c_size);
    for (nj = 0; nj < N_; ++nj) {
      for (mi = 0; mi < M; ++mi) {
        double result = 0;
        memset(avec, 0, N * sizeof(element64));
        memset(bvec, 0, N * sizeof(element64));
        for (ki = 0; ki < K; ++ki) {
          avec[ki].f = ta ? a[(size_t)mi * (*lda) + ki] : a[(size_t)ki * (*lda) + mi];
          bvec[ki].f = tb ? b[(size_t)ki * (*ldb) + nj] : b[(size_t)nj * (*ldb) + ki];
        }
        ozt_dp(&result, avec, bvec);
        c_test[(size_t)nj * (*ldc) + mi] = (*alpha) * result + (*beta) * c_ref[(size_t)nj * (*ldc) + mi];
      }
    }
    gemm_nozaki = 1;
    if (NULL != gemm_original) gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
    else GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
    gemm_nozaki = 0;
    libxs_matdiff(diff, LIBXS_DATATYPE_F64, M, N_, c_ref, c_test, ldc, ldc);
  }
  else if (NULL != c_ref) {
    ozaki_diff_reference(GEMM_ARGPASS, c_ref, c_size, diff);
  }

  libxs_free(c_test);
  libxs_free(c_ref);
}

#undef N

#endif /* OZAKI_TESTROOT && OZAKI_TEST_N && GEMM_IS_DOUBLE */
