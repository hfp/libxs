/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef SYRK_DEVICE_H
#define SYRK_DEVICE_H

#include <libxs/libxs_syrk.h>

#define TEST_DEVICE_SLOTS 16


/**
 * A device as the tests see it: it keeps what was submitted, and computes it only when waited for
 * (naive loops, hence other bits than the host). It records whether the calls overlapped, came out
 * of order, or left something in flight, and it declines every other panel if asked to.
 */
typedef struct test_device_t {
  const libxs_syrk_t* plan[TEST_DEVICE_SLOTS];
  const void *a[TEST_DEVICE_SLOTS], *b[TEST_DEVICE_SLOTS];
  libxs_syrk_panel_t panel[TEST_DEVICE_SLOTS];
  int busy[TEST_DEVICE_SLOTS];
  int decline, nsubmit, ndecline, nwait, inflight, last, inside, wrong;
} test_device_t;


static double test_device_op(const libxs_syrk_t* plan, const void* x, int ld, int i, int p)
{
  const size_t at = ('N' == plan->shape.trans ? (i + (size_t)p * ld) : (p + (size_t)i * ld));
  return (LIBXS_DATATYPE_F64 == plan->shape.datatype ? ((const double*)x)[at] : (double)((const float*)x)[at]);
}


static int test_device_submit(void* ctx, const libxs_syrk_t* plan,
  const void* a, const void* b, const libxs_syrk_panel_t* panel)
{
  test_device_t *const dev = (test_device_t*)ctx;
  int result = -1, slot;
  if (1 != LIBXS_ATOMIC_ADD_FETCH(&dev->inside, 1, LIBXS_ATOMIC_SEQ_CST)) dev->wrong = 1;
  if (0 != panel->c0 && panel->c0 <= dev->last) dev->wrong = 1; /* in order within an update */
  dev->last = panel->c0;
  if (0 != dev->decline && 0 != (panel->c0 / plan->width) % 2) ++dev->ndecline;
  else {
    for (slot = 0; slot < TEST_DEVICE_SLOTS && 0 != dev->busy[slot]; ++slot);
    if (TEST_DEVICE_SLOTS > slot) {
      dev->plan[slot] = plan;
      dev->a[slot] = a;
      dev->b[slot] = b;
      dev->panel[slot] = *panel;
      dev->busy[slot] = 1;
      ++dev->nsubmit;
      ++dev->inflight;
      result = slot;
    }
    else dev->wrong = 1;
  }
  LIBXS_ATOMIC_SUB_FETCH(&dev->inside, 1, LIBXS_ATOMIC_SEQ_CST);
  return result;
}


static int test_device_wait(void* ctx, int ticket)
{
  test_device_t *const dev = (test_device_t*)ctx;
  int result = EXIT_FAILURE;
  if (1 != LIBXS_ATOMIC_ADD_FETCH(&dev->inside, 1, LIBXS_ATOMIC_SEQ_CST)) dev->wrong = 1;
  if (0 <= ticket && TEST_DEVICE_SLOTS > ticket && 0 != dev->busy[ticket]) {
    const libxs_syrk_t *const plan = dev->plan[ticket];
    const libxs_syrk_panel_t *const panel = dev->panel + ticket;
    const int upper = ('U' == plan->shape.uplo), twok = (0 != plan->shape.ldb);
    const int lda = plan->shape.lda, ldb = plan->shape.ldb;
    int i, j, p;
    for (j = 0; j < panel->w; ++j) {
      const int gj = panel->c0 + j;
      for (i = 0; i < panel->m; ++i) {
        const int gi = panel->r0 + i;
        if (0 != upper ? (gi <= gj) : (gi >= gj)) {
          const size_t at = i + (size_t)j * panel->ldo;
          double sum = 0, v;
          for (p = 0; p < plan->shape.k; ++p) {
            const double ai = test_device_op(plan, dev->a[ticket], lda, gi, p);
            const double aj = test_device_op(plan, dev->a[ticket], lda, gj, p);
            if (0 == twok) sum += ai * aj;
            else {
              sum += ai * test_device_op(plan, dev->b[ticket], ldb, gj, p)
                + test_device_op(plan, dev->b[ticket], ldb, gi, p) * aj;
            }
          }
          if (LIBXS_DATATYPE_F64 == plan->shape.datatype) {
            double *const out = (double*)panel->out + at;
            v = panel->alpha * sum + (0 != panel->beta ? (panel->beta * *out) : 0);
            *out = v;
          }
          else {
            float *const out = (float*)panel->out + at;
            v = panel->alpha * sum + (0 != panel->beta ? (panel->beta * *out) : 0);
            *out = (float)v;
          }
        }
      }
    }
    dev->busy[ticket] = 0;
    ++dev->nwait;
    --dev->inflight;
    result = EXIT_SUCCESS;
  }
  else dev->wrong = 1;
  LIBXS_ATOMIC_SUB_FETCH(&dev->inside, 1, LIBXS_ATOMIC_SEQ_CST);
  return result;
}


static void test_device_init(test_device_t* dev, libxs_syrk_device_t* device, int decline)
{
  LIBXS_MEMZERO(dev);
  dev->decline = decline;
  dev->last = -1;
  device->submit = test_device_submit;
  device->wait = test_device_wait;
  device->ctx = dev;
}

#endif /*SYRK_DEVICE_H*/
