/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

const LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE *const real_ref = (const LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE*)ref;
const LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE *const real_tst = (const LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE*)tst;
double compf = 0, compfr = 0, compft = 0, normfr = 0, normft = 0, normr = 0, normt = 0;
double normrc = 0, normtc = 0, compr = 0, compt = 0, compd = 0;
#if defined(LIBXS_MATDIFF_SHUFFLE)
const size_t size = (size_t)mm * nn, shuffle = libxs_coprime2(size);
#endif
int ii, jj;

for (ii = 0; ii < nn; ++ii) {
  double comprj = 0, comptj = 0, compij = 0;
  double normrj = 0, normtj = 0, normij = 0;

  for (jj = 0; jj < mm; ++jj) {
#if defined(LIBXS_MATDIFF_SHUFFLE)
    const size_t index = (shuffle * (ii * mm + jj)) % size;
    const int i = (int)(index / mm), j = (int)(index % mm);
#else
    const int i = ii, j = jj;
#endif
    const double ti = (NULL != real_tst ? LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(real_tst[(size_t)i*ldt+j]) : 0);
    const double ri = LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(real_ref[(size_t)i*ldr+j]);
    const double ta = LIBXS_ABS(ti);
    const double ra = LIBXS_ABS(ri);

    /* minimum/maximum of reference set */
    if (ri < info->min_ref) info->min_ref = ri;
    if (ri > info->max_ref) info->max_ref = ri;

    if (LIBXS_NOTNAN(ti) && (pos_inf > ta || ti == ri)) {
      const double di = ((NULL != real_tst && ta != ra) ? LIBXS_DELTA(ri, ti) : 0);
      const double dri = LIBXS_MATDIFF_DIV(di, ra, ta);

      /* minimum/maximum of test set */
      if (ti < info->min_tst) info->min_tst = ti;
      if (ti > info->max_tst) info->max_tst = ti;

      /* maximum absolute error and location */
      if (info->linf_abs < di) {
        info->linf_abs = di;
        info->v_ref = ri;
        info->v_tst = ti;
        info->m = j;
        info->n = i;
      }

      /* maximum error relative to current value */
      if (info->linf_rel < dri) info->linf_rel = dri;
      /* sum of relative differences */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(dri * dri, &info->l2_rel, &compd);

      /* row-wise sum of reference values with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ra, &normrj, &comprj);

      /* row-wise sum of test values with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ta, &normtj, &comptj);

      /* row-wise sum of differences with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(di, &normij, &compij);

      /* Froebenius-norm of reference matrix with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ri * ri, &normfr, &compfr);

      /* Froebenius-norm of test matrix with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ti * ti, &normft, &compft);

      /* Froebenius-norm of differences with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(di * di, &info->l2_abs, &compf);
    }
    else { /* NaN */
      result_nan = ((LIBXS_NOTNAN(ri) && pos_inf > ra) ? 1 : 2);
      info->m = j; info->n = i;
      info->v_ref = ri;
      info->v_tst = ti;
      break;
    }
  }

  if (0 == result_nan) {
    /* summarize reference values */
    LIBXS_PRAGMA_FORCEINLINE
    libxs_kahan_sum(normrj, &info->l1_ref, &compr);

    /* summarize test values */
    LIBXS_PRAGMA_FORCEINLINE
    libxs_kahan_sum(normtj, &info->l1_tst, &compt);

    /* calculate Infinity-norm of differences */
    if (info->normi_abs < normij) info->normi_abs = normij;
    /* calculate Infinity-norm of reference/test values */
    if (normr < normrj) normr = normrj;
    if (normt < normtj) normt = normtj;
  }
  else {
    break;
  }
}

if (0 == result_nan) {
  double compr_var = 0, compt_var = 0;

  /* initial variance */
  assert(0 == info->var_ref); /* !LIBXS_ASSERT */
  assert(0 == info->var_tst); /* !LIBXS_ASSERT */
  if (0 != ntotal) { /* final average */
    info->avg_ref = info->l1_ref / ntotal;
    info->avg_tst = info->l1_tst / ntotal;
  }

  /* Infinity-norm relative to reference */
  info->normi_rel = LIBXS_MATDIFF_DIV(info->normi_abs, normr, normt);
  /* Froebenius-norm relative to reference */
  info->normf_rel = LIBXS_MATDIFF_DIV(info->l2_abs, normfr,
    LIBXS_MIN(normft * normft, info->l2_abs));

  for (jj = 0; jj < mm; ++jj) {
    double compri = 0, compti = 0, comp1 = 0;
    double normri = 0, normti = 0, norm1 = 0;

    for (ii = 0; ii < nn; ++ii) {
#if defined(LIBXS_MATDIFF_SHUFFLE)
      const size_t index = (shuffle * (ii * mm + jj)) % size;
      const int i = (int)(index / mm), j = (int)(index % mm);
#else
      const int i = ii, j = jj;
#endif
      const double ti = (NULL != real_tst ? LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(real_tst[(size_t)i * ldt + j]) : 0);
      const double ri = LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(real_ref[(size_t)i*ldr+j]);
      const double ta = LIBXS_ABS(ti), ra = LIBXS_ABS(ri);
      const double di = ((NULL != real_tst && ta != ra) ? LIBXS_DELTA(ri, ti) : 0);
      const double rd = ri - info->avg_ref, td = ti - info->avg_tst;

      /* variance of reference set with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(rd * rd, &info->var_ref, &compr_var);

      /* variance of test set with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(td * td, &info->var_tst, &compt_var);

      /* column-wise sum of reference values with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ra, &normri, &compri);

      /* column-wise sum of test values with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(ta, &normti, &compti);

      /* column-wise sum of differences with Kahan compensation */
      LIBXS_PRAGMA_FORCEINLINE
      libxs_kahan_sum(di, &norm1, &comp1);
    }

    /* calculate One-norm of differences */
    if (info->norm1_abs < norm1) info->norm1_abs = norm1;
    /* calculate One-norm of reference/test values */
    if (normrc < normri) normrc = normri;
    if (normtc < normti) normtc = normti;
  }

  /* One-norm relative to reference */
  info->norm1_rel = LIBXS_MATDIFF_DIV(info->norm1_abs, normrc, info->norm1_abs);
}
