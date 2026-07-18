/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_mem.h>
#include <stdlib.h>
#include <math.h>


typedef struct internal_libxs_ngram_key_t {
  int n;
  unsigned int ctx[LIBXS_NGRAM_ORDER_MAX];
} internal_libxs_ngram_key_t;


static void internal_libxs_ngram_key(internal_libxs_ngram_key_t* key,
  const unsigned int hist[], int hlen, int n)
{
  int i;
  LIBXS_MEMZERO(key);
  key->n = n;
  for (i = 0; i < n; ++i) key->ctx[i] = hist[hlen - n + i];
}


static void internal_libxs_ngram_add(libxs_ngram_entry_t* entry,
  unsigned int succ_id)
{
  unsigned int slot;
  int found = 0;
  entry->total += 1;
  for (slot = 0; slot < entry->nsucc && 0 == found; ++slot) {
    if (entry->succ[slot].id == succ_id) {
      entry->succ[slot].count += 1;
      found = 1;
    }
  }
  if (0 == found) {
    if (entry->nsucc < LIBXS_NGRAM_SUCC_MAX) {
      entry->succ[entry->nsucc].id = succ_id;
      entry->succ[entry->nsucc].count = 1;
      entry->nsucc += 1;
    }
    else {
      unsigned int min_slot = 0;
      for (slot = 1; slot < entry->nsucc; ++slot) {
        if (entry->succ[slot].count < entry->succ[min_slot].count) {
          min_slot = slot;
        }
      }
      entry->succ[min_slot].id = succ_id;
      entry->succ[min_slot].count += 1;
    }
  }
}


static double internal_libxs_ngram_relfreq(const libxs_ngram_entry_t* entry,
  unsigned int succ_id)
{
  double result = 0.0;
  if (NULL != entry && entry->total > 0) {
    unsigned int slot;
    for (slot = 0; slot < entry->nsucc; ++slot) {
      if (entry->succ[slot].id == succ_id) {
        result = (double)entry->succ[slot].count / (double)entry->total;
        break;
      }
    }
  }
  return result;
}


static int internal_libxs_ngram_topk(const libxs_ngram_entry_t* entry,
  unsigned int out_ids[], int k)
{
  int result = 0;
  if (NULL != entry && NULL != out_ids && k > 0) {
    unsigned int taken[LIBXS_NGRAM_SUCC_MAX];
    unsigned int nsucc = entry->nsucc;
    unsigned int slot;
    for (slot = 0; slot < nsucc && slot < LIBXS_NGRAM_SUCC_MAX; ++slot) {
      taken[slot] = 0;
    }
    while (result < k) {
      int best = -1;
      for (slot = 0; slot < nsucc && slot < LIBXS_NGRAM_SUCC_MAX; ++slot) {
        if (0 == taken[slot]
          && (best < 0 || entry->succ[slot].count > entry->succ[best].count))
        {
          best = (int)slot;
        }
      }
      if (best < 0) break;
      taken[best] = 1;
      out_ids[result] = entry->succ[best].id;
      ++result;
    }
  }
  return result;
}


static double internal_libxs_ngram_prior(const libxs_ngram_t* model,
  unsigned int id)
{
  double result = 0.0;
  if (NULL != model->unifreq && 0 != id && id <= model->unifreq_size) {
    result = (double)model->unifreq[id] / model->unifreq_total;
  }
  return result;
}


LIBXS_API int libxs_ngram_create(libxs_ngram_t* model, int maxorder)
{
  int result = EXIT_FAILURE;
  if (NULL != model) {
    LIBXS_MEMZERO(model);
    model->unifreq_total = 1.0;
    model->maxorder = (maxorder < 1) ? 1
      : ((maxorder > LIBXS_NGRAM_ORDER_MAX) ? LIBXS_NGRAM_ORDER_MAX : maxorder);
    model->store = libxs_registry_create();
    if (NULL != model->store) result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API void libxs_ngram_destroy(libxs_ngram_t* model)
{
  if (NULL != model) {
    free(model->unifreq);
    libxs_registry_destroy(model->store);
    LIBXS_MEMZERO(model);
    model->unifreq_total = 1.0;
  }
}


LIBXS_API void libxs_ngram_observe(libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int succ)
{
  int n;
  if (NULL == model || NULL == model->store || 0 == succ) return;
  for (n = 1; n <= model->maxorder && n <= hlen; ++n) {
    internal_libxs_ngram_key_t key;
    libxs_ngram_entry_t* entry;
    int ok = 1, i;
    for (i = 0; i < n; ++i) {
      if (0 == hist[hlen - n + i]) ok = 0;
    }
    if (0 == ok) continue;
    internal_libxs_ngram_key(&key, hist, hlen, n);
    entry = (libxs_ngram_entry_t*)libxs_registry_get(model->store, &key,
      sizeof(key), NULL);
    if (NULL == entry) {
      libxs_ngram_entry_t fresh;
      LIBXS_MEMZERO(&fresh);
      fresh.total = 1;
      fresh.nsucc = 1;
      fresh.succ[0].id = succ;
      fresh.succ[0].count = 1;
      libxs_registry_set(model->store, &key, sizeof(key), &fresh,
        sizeof(fresh), NULL);
    }
    else internal_libxs_ngram_add(entry, succ);
  }
}


LIBXS_API void libxs_ngram_finalize(libxs_ngram_t* model, unsigned int vocab)
{
  free(model->unifreq);
  model->unifreq = NULL;
  model->unifreq_size = 0;
  model->unifreq_total = 1.0;
  model->backoff_count = 0;
  if (NULL != model && NULL != model->store && vocab > 0) {
    unsigned int* freq = (unsigned int*)calloc((size_t)vocab + 1,
      sizeof(*freq));
    if (NULL != freq) {
      const void* key = NULL;
      size_t cursor = 0;
      void* value = libxs_registry_begin(model->store, &key, &cursor);
      double total = 0.0;
      unsigned int picked[LIBXS_NGRAM_SUCC_MAX];
      int slot;
      while (NULL != value) {
        const libxs_ngram_entry_t* entry = (const libxs_ngram_entry_t*)value;
        const internal_libxs_ngram_key_t* entry_key =
          (const internal_libxs_ngram_key_t*)key;
        if (NULL != entry_key && 1 == entry_key->n) {
          unsigned int s;
          for (s = 0; s < entry->nsucc; ++s) {
            if (entry->succ[s].id <= vocab) {
              freq[entry->succ[s].id] += entry->succ[s].count;
              total += (double)entry->succ[s].count;
            }
          }
        }
        value = libxs_registry_next(model->store, &key, &cursor);
      }
      for (slot = 0; slot < LIBXS_NGRAM_SUCC_MAX; ++slot) {
        unsigned int id;
        unsigned int best = 0;
        unsigned int best_freq = 0;
        int prior;
        for (id = 1; id <= vocab; ++id) {
          int skip = 0;
          for (prior = 0; prior < slot; ++prior) {
            if (picked[prior] == id) skip = 1;
          }
          if (0 == skip && freq[id] > best_freq) {
            best_freq = freq[id];
            best = id;
          }
        }
        if (0 == best) break;
        model->backoff_ids[slot] = best;
        picked[slot] = best;
        ++model->backoff_count;
      }
      model->unifreq = freq;
      model->unifreq_size = vocab;
      model->unifreq_total = (total > 0.0) ? total : 1.0;
    }
  }
}


LIBXS_API const libxs_ngram_entry_t* libxs_ngram_lookup(
  const libxs_ngram_t* model, const unsigned int hist[], int hlen, int n)
{
  internal_libxs_ngram_key_t key;
  if (NULL == model || NULL == model->store || n <= 0 || n > hlen) return NULL;
  internal_libxs_ngram_key(&key, hist, hlen, n);
  return (const libxs_ngram_entry_t*)libxs_registry_get(model->store, &key,
    sizeof(key), NULL);
}


LIBXS_API double libxs_ngram_prob(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int next)
{
  double vocab, uni_floor, p_uni, p;
  int n;
  if (NULL == model) return 0.0;
  vocab = (model->unifreq_size > 0) ? (double)model->unifreq_size : 1.0;
  uni_floor = 1.0 / (vocab + 1.0);
  p_uni = internal_libxs_ngram_prior(model, next);
  p = (p_uni > 0.0) ? p_uni : uni_floor;
  for (n = 1; n <= model->maxorder && n <= hlen; ++n) {
    const libxs_ngram_entry_t* entry = libxs_ngram_lookup(model, hist, hlen, n);
    if (NULL != entry && entry->total > 0) {
      double t = (double)entry->total;
      double lambda = t / (t + 1.0);
      p = lambda * internal_libxs_ngram_relfreq(entry, next)
        + (1.0 - lambda) * p;
    }
  }
  if (p < uni_floor) p = uni_floor;
  if (p > 1.0) p = 1.0;
  return p;
}


LIBXS_API int libxs_ngram_predict(const libxs_ngram_t* model,
  const unsigned int hist[], int hlen, unsigned int out_ids[], int k,
  int* order)
{
  const libxs_ngram_entry_t* entry = NULL;
  int n, matched = 0;
  if (NULL == model) {
    if (NULL != order) *order = 0;
    return 0;
  }
  for (n = (model->maxorder < hlen) ? model->maxorder : hlen;
    n >= 1 && NULL == entry; --n)
  {
    entry = libxs_ngram_lookup(model, hist, hlen, n);
    if (NULL != entry) matched = n;
  }
  if (NULL != order) *order = matched;
  if (NULL != entry) return internal_libxs_ngram_topk(entry, out_ids, k);
  { int slot, result = 0;
    for (slot = 0; slot < k && slot < model->backoff_count; ++slot) {
      out_ids[slot] = model->backoff_ids[slot];
      ++result;
    }
    return result;
  }
}


LIBXS_API int libxs_ngram_stats(const libxs_ngram_t* model,
  libxs_ngram_stats_t* stats)
{
  int result = EXIT_FAILURE;
  if (NULL != model && NULL != model->store && NULL != stats) {
    libxs_registry_info_t info;
    const void* key = NULL;
    size_t cursor = 0;
    void* value;
    int n;
    LIBXS_MEMZERO(stats);
    value = libxs_registry_begin(model->store, &key, &cursor);
    while (NULL != value) {
      const libxs_ngram_entry_t* entry = (const libxs_ngram_entry_t*)value;
      const internal_libxs_ngram_key_t* entry_key =
        (const internal_libxs_ngram_key_t*)key;
      if (NULL != entry_key && entry_key->n >= 1
        && entry_key->n <= LIBXS_NGRAM_ORDER_MAX)
      {
        n = entry_key->n;
        stats->keys[n] += 1;
        stats->obs[n] += (double)entry->total;
        if (LIBXS_NGRAM_SUCC_MAX <= entry->nsucc) stats->saturated[n] += 1;
      }
      value = libxs_registry_next(model->store, &key, &cursor);
    }
    if (EXIT_SUCCESS == libxs_registry_info(model->store, &info)) {
      stats->entries = info.size;
      stats->nbytes = info.nbytes;
    }
    result = EXIT_SUCCESS;
  }
  return result;
}
