#include <libxs/libxs.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_mix.h>

#include "converse.h"
#include "converse_hier.h"

#define HIER_KEY_MAX (COMPOSE_MAXTEXT + 1)
#define HIER_ESCAPE_TEXT 1u
#define HIER_ESCAPE_NATIVE 2u
#define HIER_SYMBOL_FIRST 3u
#define HIER_SYLLABLE_ESCAPE 1u
#define HIER_SYLLABLE_END 2u
#define HIER_SYLLABLE_FIRST 3u
#define HIER_BYTE_END 257u
#define HIER_BYTE_VOCAB 257u
#define HIER_HISTORY_START (~0u)
#define HIER_CLOCK_WORD_BASE 0x10000000u
#define HIER_CLOCK_SYLLABLE_BASE 0x20000000u
#define HIER_CLOCK_BYTE_START 0x30000000u
#define HIER_CLOCK_RAW_MAX 4
#define HIER_RECURRENT_BASE 0x40000000u
#define HIER_RECURRENT_DIM 8
#define HIER_RECURRENT_RAW 2
#define HIER_RECURRENT_ORDER 3
/**
 * Byte-context order for the local PPM models, independent of
 * LIBXS_NGRAM_ORDER_MAX (which bounds the libxs_ngram token store). HARD LIMIT:
 * hier_ppm_key_t is a registry KEY, and libxs_registry_get/set silently reject
 * keys larger than LIBXS_REGKEY_MAXSIZE (64) by returning NULL - every lookup
 * would miss and the model would collapse to its unigram. sizeof(key) is
 * 4 + 4*order, so order 15 is the maximum that fits.
 */
#define HIER_PPM_ORDER_MAX 12
#define HIER_EXPERT_WORD (HIER_PPM_ORDER_MAX + 1)
#define HIER_EXPERT_SYLLABLE (HIER_PPM_ORDER_MAX + 2)
#define HIER_EXPERT_SYLLABLE_ROLE (HIER_PPM_ORDER_MAX + 3)
#define HIER_EXPERT_SKIP2 (HIER_PPM_ORDER_MAX + 4)
#define HIER_EXPERT_SKIP3 (HIER_PPM_ORDER_MAX + 5)
#define HIER_EXPERT_SKIP5 (HIER_PPM_ORDER_MAX + 6)
#define HIER_EXPERT_LAST HIER_EXPERT_SKIP5
#define HIER_EXPERT_MAX (HIER_EXPERT_LAST + 1)
#define HIER_SKIP_COUNT 3
#define HIER_ROLE_SINGLE 0
#define HIER_ROLE_BEGIN 1
#define HIER_ROLE_MIDDLE 2
#define HIER_ROLE_END 3


typedef struct hier_symbol_t {
  long count;
  unsigned int id;
} hier_symbol_t;

typedef struct hier_eval_t {
  long ntokens;
  long ntop1;
  long ntext_escape;
  long nnative_escape;
  long nsyllable_escape;
  long ndeep;
  long nshallow;
  double bits;
  double bytes;
  double top_bits;
  double syllable_bits;
  double byte_bits;
  double deep_bits;
  double deep_bytes;
  double shallow_bits;
  double shallow_bytes;
} hier_eval_t;

typedef struct hier_clock_eval_t {
  long nbytes;
  long nppm;
  long nraw_top1;
  long ncontext_top1;
  long nppm_top1;
  long nppm_top3;
  long nadaptive_top1;
  long nadaptive_top3;
  long nexpert_top1;
  long nexpert_top3;
  long nadaptive_expert_top1;
  long nadaptive_expert_top3;
  long ndeep;
  long nshallow;
  double raw_bits;
  double context_bits;
  double mix_bits;
  double deep_bits;
  double shallow_bits;
  double raw_ppm_bits;
  double context_ppm_bits;
  double ppm_mix_bits;
  double recurrent_bits;
  double recurrent_mix_bits;
  double frozen_interp_bits;
  double adaptive_bits;
  double expert_bits[HIER_EXPERT_MAX];
  double expert_mix_bits;
  long nexpert_attested;
  long nexpert_novel;
  double expert_attested_bits;
  double expert_novel_bits;
  double expert_weight[HIER_EXPERT_MAX];
  double adaptive_expert_bits[HIER_EXPERT_MAX];
  double adaptive_expert_mix_bits;
  double adaptive_expert_weight[HIER_EXPERT_MAX];
} hier_clock_eval_t;

typedef struct hier_ppm_key_t {
  int order;
  unsigned int context[HIER_PPM_ORDER_MAX];
} hier_ppm_key_t;

typedef struct hier_ppm_pair_t {
  hier_ppm_key_t key;
  unsigned int next;
} hier_ppm_pair_t;

typedef struct hier_ppm_stats_t {
  unsigned int total;
  unsigned int distinct;
  unsigned int seen[8];
  double backoff_norm;
} hier_ppm_stats_t;

typedef struct hier_ppm_t {
  libxs_registry_t* contexts;
  libxs_registry_t* pairs;
  unsigned int unigram[256];
  double total;
  int maxorder;
} hier_ppm_t;

struct converse_hier_t {
  libxs_registry_t* symbols;
  libxs_registry_t* syllables;
  libxs_registry_t* syllable_payloads;
  libxs_tokenizer_t* word_tokenizer;
  libxs_tokenizer_t* syllable_tokenizer;
  libxs_ngram_t word_model;
  libxs_ngram_t syllable_model;
  libxs_ngram_t byte_model;
  libxs_ngram_t stream_byte_model;
  libxs_ngram_t clock_byte_model;
  hier_ppm_t stream_ppm;
  hier_ppm_t clock_ppm;
  hier_ppm_t recurrent_ppm;
  hier_ppm_t adaptive_ppm;
  hier_ppm_t expert_ppm;
  hier_ppm_t word_clock_ppm;
  hier_ppm_t syllable_clock_ppm;
  hier_ppm_t syllable_role_ppm;
  hier_ppm_t skip_ppm[HIER_SKIP_COUNT];
  unsigned int word_vocab;
  unsigned int syllable_vocab;
  int maxorder;
  int clock_order;
  int mincount;
  int top_stride;
  int expert_order;
  double expert_rate;
  double expert_share;
  double recurrent_decay;
  int ready;
};


static const int hier_skip_distance[HIER_SKIP_COUNT] = { 2, 3, 5 };


static void hier_ppm_dist_order(const hier_ppm_t* model,
  const unsigned int history[], int history_length, int maxorder,
  double dist[]);
static void hier_ppm_dist(const hier_ppm_t* model,
  const unsigned int history[], int history_length, double dist[]);


/**
 * Escape estimator: 0 = method C (escape mass distinct/(total+distinct)),
 * 1 = method D (escape distinct/(2*total), symbol (2c-1)/(2*total)). Both are
 * normalized; D is the usual choice on text because it charges a novel symbol
 * half a count instead of a whole one. Exclusion is unaffected: the escaped mass
 * is still redistributed over symbols unseen at this order via backoff_norm.
 */
/**
 * Logarithmic (logit-domain) pooling of the expert distributions instead of the
 * linear pool. A linear pool cannot be more confident than its most confident
 * expert; a logarithmic pool compounds independent agreement, which is what the
 * word, syllable and skip experts often provide. It requires a normalizer over
 * the whole alphabet, hence the byte budget below: unlike the linear pool, which
 * is evaluated at the target symbol only, this costs 256 expert evaluations per
 * position.
 */
static int hier_ppm_logit(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_LOGIT");
    cached = (NULL != env && '0' != *env) ? 1 : 0;
  }
  return cached;
}


/** Cap on evaluated test bytes (0 = all), for affordable A/B comparisons. */
static long hier_eval_max(void)
{
  static long cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_EVAL_MAX");
    cached = (NULL != env && '\0' != *env) ? atol(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Bytes of each candidate the rescorer scores (0 = all). The query can only
 * influence the first few bytes: it leaves the byte context window after
 * expert_order positions, so averaging lift over a whole long sentence dilutes a
 * short signal with unconditioned text. A window near the context order keeps
 * the comparison to the region the query actually reaches.
 */
static int hier_rescore_window(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_RESCORE_WINDOW");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 24;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/** Whether the skip-context experts participate (1 = yes), for A/B measurement. */
static int hier_skip_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_SKIP");
    cached = (NULL != env && '0' == *env) ? 0 : 1;
  }
  return cached;
}


static int hier_ppm_escape_d(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_PPM_D");
    cached = (NULL != env && '0' != *env) ? 1 : 0;
  }
  return cached;
}


static int hier_skip_history(const unsigned int raw_history[], int raw_length,
  int distance, unsigned int history[])
{
  history[0] = raw_history[raw_length - 1];
  history[1] = (raw_length >= distance)
    ? raw_history[raw_length - distance] : HIER_CLOCK_BYTE_START;
  return 2;
}


static int hier_struct_history(unsigned int state,
  const unsigned int raw_history[], int raw_length, unsigned int history[])
{
  history[0] = state;
  history[1] = raw_history[raw_length - 1];
  return 2;
}


static unsigned int hier_recurrent_code(const double state[])
{
  unsigned int result = 0;
  int dim;
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) {
    if (state[dim] >= 0.0) result |= 1u << dim;
  }
  return HIER_RECURRENT_BASE + result;
}


static void hier_recurrent_update(double state[], unsigned int byte,
  double decay)
{
  double next[HIER_RECURRENT_DIM];
  int dim;
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) {
    const unsigned int hash = (byte + 1u) * 2654435761u
      + (unsigned int)(dim + 1) * 2246822519u;
    const double feature = (0 != (hash & 0x80000000u)) ? 1.0 : -1.0;
    next[dim] = decay * state[(dim + 1) % HIER_RECURRENT_DIM] + feature;
  }
  for (dim = 0; dim < HIER_RECURRENT_DIM; ++dim) state[dim] = next[dim];
}


static int hier_recurrent_history(const double state[],
  const unsigned int raw_history[], int raw_length, unsigned int history[])
{
  int keep = raw_length;
  int result = 1;
  int pos;
  if (keep > HIER_RECURRENT_RAW) keep = HIER_RECURRENT_RAW;
  history[0] = hier_recurrent_code(state);
  for (pos = raw_length - keep; pos < raw_length; ++pos) {
    history[result++] = raw_history[pos];
  }
  return result;
}


static int hier_ppm_create(hier_ppm_t* model, int maxorder)
{
  int result = EXIT_FAILURE;
  if (NULL != model && maxorder >= 1
    && maxorder <= HIER_PPM_ORDER_MAX)
  {
    memset(model, 0, sizeof(*model));
    model->contexts = libxs_registry_create();
    model->pairs = libxs_registry_create();
    model->maxorder = maxorder;
    if (NULL != model->contexts && NULL != model->pairs) result = EXIT_SUCCESS;
  }
  return result;
}


static void hier_ppm_destroy(hier_ppm_t* model)
{
  if (NULL != model) {
    libxs_registry_destroy(model->contexts);
    libxs_registry_destroy(model->pairs);
    memset(model, 0, sizeof(*model));
  }
}


static void hier_ppm_key(hier_ppm_key_t* key,
  const unsigned int history[], int history_length, int order)
{
  int pos;
  memset(key, 0, sizeof(*key));
  key->order = order;
  for (pos = 0; pos < order; ++pos) {
    key->context[pos] = history[history_length - order + pos];
  }
}


static void hier_ppm_observe(hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return;
  ++model->unigram[next - 1];
  model->total += 1.0;
  for (order = 1; order <= model->maxorder && order <= history_length;
    ++order)
  {
    hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    hier_ppm_stats_t* stats;
    unsigned int* count;
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL == stats) {
      hier_ppm_stats_t fresh;
      memset(&fresh, 0, sizeof(fresh));
      fresh.total = 1;
      fresh.distinct = 1;
      fresh.seen[(next - 1) / 32] |= 1u << ((next - 1) % 32);
      libxs_registry_set(model->contexts, &key, sizeof(key), &fresh,
        sizeof(fresh), NULL);
    }
    else {
      ++stats->total;
      if (NULL == count) {
        ++stats->distinct;
        stats->seen[(next - 1) / 32] |= 1u << ((next - 1) % 32);
      }
    }
    if (NULL != count) ++*count;
    else {
      const unsigned int one = 1;
      libxs_registry_set(model->pairs, &pair, sizeof(pair), &one,
        sizeof(one), NULL);
    }
  }
}


static double hier_ppm_unigram(const hier_ppm_t* model, unsigned int next)
{
  const double alpha = 0.5;
  double result = 1.0 / 256.0;
  if (NULL != model && next >= 1 && next <= 256 && model->total > 0.0) {
    result = ((double)model->unigram[next - 1] + alpha)
      / (model->total + alpha * 256.0);
  }
  return result;
}


static double hier_ppm_prob_order(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next,
  int order)
{
  double result;
  if (order <= 0) return hier_ppm_unigram(model, next);
  { hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    const hier_ppm_stats_t* stats;
    const unsigned int* count;
    const double lower = hier_ppm_prob_order(model, history, history_length,
      next, order - 1);
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (const hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (const unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL != stats && stats->total > 0) {
      const int method_d = hier_ppm_escape_d();
      const double denom = (0 != method_d)
        ? (2.0 * (double)stats->total)
        : ((double)stats->total + (double)stats->distinct);
      if (NULL != count) {
        result = (0 != method_d)
          ? ((2.0 * (double)*count - 1.0) / denom)
          : ((double)*count / denom);
      }
      else if (stats->backoff_norm > 0.0) {
        result = ((double)stats->distinct / denom)
          * lower / stats->backoff_norm;
      }
      else result = lower;
    }
    else result = lower;
  }
  return result;
}


static void hier_ppm_finalize(hier_ppm_t* model)
{
  int order;
  if (NULL == model) return;
  /**
   * Exclusion renormalizer: backoff_norm is the lower-order mass NOT already
   * attested at this order, so escaped mass can be redistributed over unseen
   * symbols only. Orders are processed low to high because order k reads the
   * order k-1 distribution. One distribution fill per context replaces up to 256
   * recursive per-symbol backoff walks.
   */
  for (order = 1; order <= model->maxorder; ++order) {
    const void* registry_key = NULL;
    size_t cursor = 0;
    void* value = libxs_registry_begin(model->contexts, &registry_key,
      &cursor);
    while (NULL != value) {
      const hier_ppm_key_t* key = (const hier_ppm_key_t*)registry_key;
      hier_ppm_stats_t* stats = (hier_ppm_stats_t*)value;
      if (key->order == order) {
        double dist[257];
        double seen_mass = 0.0;
        unsigned int id;
        hier_ppm_dist_order(model, key->context, order, order - 1, dist);
        for (id = 1; id <= 256; ++id) {
          if (0 != (stats->seen[(id - 1) / 32] & (1u << ((id - 1) % 32)))) {
            seen_mass += dist[id];
          }
        }
        stats->backoff_norm = 1.0 - seen_mass;
        if (stats->backoff_norm < 1e-15) stats->backoff_norm = 1e-15;
      }
      value = libxs_registry_next(model->contexts, &registry_key, &cursor);
    }
  }
}


static double hier_ppm_prob(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return 0.0;
  order = model->maxorder;
  if (order > history_length) order = history_length;
  return hier_ppm_prob_order(model, history, history_length, next, order);
}


static double hier_ppm_interp_prob_order(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next,
  int maxorder)
{
  double result;
  int order;
  if (NULL == model || NULL == history || next < 1 || next > 256) return 0.0;
  result = hier_ppm_unigram(model, next);
  if (maxorder > model->maxorder) maxorder = model->maxorder;
  for (order = 1; order <= maxorder && order <= history_length;
    ++order)
  {
    hier_ppm_key_t key;
    hier_ppm_pair_t pair;
    const hier_ppm_stats_t* stats;
    const unsigned int* count;
    hier_ppm_key(&key, history, history_length, order);
    pair.key = key;
    pair.next = next;
    stats = (const hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    count = (const unsigned int*)libxs_registry_get(model->pairs,
      &pair, sizeof(pair), NULL);
    if (NULL != stats && stats->total > 0) {
      const double denom = (double)stats->total + (double)stats->distinct;
      const double observed = (NULL != count) ? (double)*count : 0.0;
      result = observed / denom + (double)stats->distinct / denom * result;
    }
  }
  return result;
}


static double hier_ppm_interp_prob(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next)
{
  return hier_ppm_interp_prob_order(model, history, history_length, next,
    model->maxorder);
}


static int hier_ppm_rank(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int target,
  int interpolate)
{
  unsigned int best_id[3] = { 0, 0, 0 };
  double best_probability[3] = { -1.0, -1.0, -1.0 };
  unsigned int id;
  int result = 0;
  for (id = 1; id <= 256; ++id) {
    const double probability = (0 != interpolate)
      ? hier_ppm_interp_prob(model, history, history_length, id)
      : hier_ppm_prob(model, history, history_length, id);
    int slot;
    for (slot = 0; slot < 3; ++slot) {
      if (probability > best_probability[slot]) break;
    }
    if (slot < 3) {
      int move;
      for (move = 2; move > slot; --move) {
        best_probability[move] = best_probability[move - 1];
        best_id[move] = best_id[move - 1];
      }
      best_probability[slot] = probability;
      best_id[slot] = id;
    }
  }
  if (best_id[0] == target) result = 1;
  else if (best_id[1] == target) result = 2;
  else if (best_id[2] == target) result = 3;
  return result;
}


/**
 * Whether this position's FULL-order context recurred verbatim in training, i.e.
 * whether the deepest expert had exact evidence rather than having to escape.
 *
 * This is the byte-level counterpart of the attested-context control: aggregate
 * BPC is dominated by positions whose context was seen, so a mechanism can lower
 * it substantially while doing nothing on the positions that actually require
 * generalizing. Splitting the mixture's own bits by this flag is what separates
 * the two. A history shorter than the order cannot be full-order attested and
 * counts as novel.
 */
static int hier_ppm_attested(const hier_ppm_t* model,
  const unsigned int history[], int history_length, int order)
{
  int result = 0;
  if (NULL != model && 0 < order && history_length >= order
    && order <= model->maxorder)
  {
    hier_ppm_key_t key;
    hier_ppm_key(&key, history, history_length, order);
    if (NULL != libxs_registry_get(model->contexts, &key, sizeof(key), NULL)) {
      result = 1;
    }
  }
  return result;
}


static double hier_expert_probability(const hier_ppm_t* model,
  const unsigned int history[], int history_length, unsigned int next,
  const double weight[], int maxorder, int interpolate)
{
  double result = 0.0;
  int order;
  for (order = 0; order <= maxorder; ++order) {
    const int effective = (order < history_length) ? order : history_length;
    const double probability = (0 != interpolate)
      ? hier_ppm_interp_prob_order(model, history, history_length, next,
          effective)
      : hier_ppm_prob_order(model, history, history_length, next, effective);
    result += weight[order] * probability;
  }
  return result;
}


static int hier_expert_rank(const hier_ppm_t* model,
  const hier_ppm_t* word_model, const hier_ppm_t* syllable_model,
  const hier_ppm_t* syllable_role_model,
  const hier_ppm_t skip_model[HIER_SKIP_COUNT],
  const unsigned int history[], int history_length, unsigned int target,
  const unsigned int word_history[], const unsigned int syllable_history[],
  const unsigned int syllable_role_history[],
  unsigned int skip_history[HIER_SKIP_COUNT][2],
  const double weight[], int maxorder, int interpolate)
{
  unsigned int best_id[3] = { 0, 0, 0 };
  double best_probability[3] = { -1.0, -1.0, -1.0 };
  unsigned int id;
  int result = 0, skip, slot;
  for (id = 1; id <= 256; ++id) {
    double probability = hier_expert_probability(model, history,
      history_length, id, weight, maxorder, interpolate)
      + weight[HIER_EXPERT_WORD] * hier_ppm_prob(word_model, word_history, 2, id)
      + weight[HIER_EXPERT_SYLLABLE]
        * hier_ppm_prob(syllable_model, syllable_history, 2, id)
      + weight[HIER_EXPERT_SYLLABLE_ROLE]
        * hier_ppm_prob(syllable_role_model, syllable_role_history, 2, id);
    for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
      probability += weight[HIER_EXPERT_SKIP2 + skip]
        * hier_ppm_prob(skip_model + skip, skip_history[skip], 2, id);
    }
    for (slot = 0; slot < 3; ++slot) {
      if (probability > best_probability[slot]) break;
    }
    if (slot < 3) {
      int move;
      for (move = 2; move > slot; --move) {
        best_probability[move] = best_probability[move - 1];
        best_id[move] = best_id[move - 1];
      }
      best_probability[slot] = probability;
      best_id[slot] = id;
    }
  }
  if (best_id[0] == target) result = 1;
  else if (best_id[1] == target) result = 2;
  else if (best_id[2] == target) result = 3;
  return result;
}


static void hier_expert_update(double weight[], const double probability[],
  int maxorder, double mixture, double rate, double share)
{
  libxs_mix_t mix;
  int active[HIER_EXPERT_MAX];
  int order;
  /**
   * Only experts that were initialized participate. The expert index space is
   * sized for the deepest supported byte order, so when a shallower order is
   * configured the unused slots hold weight zero; giving them share mass would
   * spend probability on experts that never produce one. The weight>0 test the
   * shared primitive applies covers that, and the mask retires the slots past
   * the configured order.
   */
  for (order = 0; order < HIER_EXPERT_MAX; ++order) {
    active[order] = (order <= maxorder) ? 1 : 0;
  }
  mix.weight = weight;
  mix.nslot = HIER_EXPERT_MAX;
  mix.rate = rate;
  mix.share = share;
  /**
   * No ratio floor here, which preserves this consumer bit-for-bit. It is a
   * latent defect rather than a choice: an expert that once gave the outcome no
   * mass is multiplied by exactly zero and the share term cannot revive it,
   * because share only reaches slots that still hold mass. Left as it was so the
   * extraction changes nothing measurable; fixing it is a separate change with
   * its own measurement.
   */
  mix.relmin = 0.0;
  /**
   * The caller's mixture, not a recomputed one: it is accumulated over a WIDER
   * slot set than this update walks (the word and skip experts sit past
   * maxorder), so pooling here would divide by a different weight mass and
   * silently change every byte-side number.
   */
  libxs_mix_update(&mix, probability, active, mixture);
}


/**
 * Logarithmic pool over the whole alphabet: p(x) proportional to
 * prod_i q_i(x)^{w_i}, i.e. a weighted geometric mean, evaluated in the log
 * domain and normalized. Returns the pooled probability of the target symbol.
 * Weights are the same fixed-share weights the linear pool uses, so the two
 * differ only in how the experts are combined.
 *
 * Each expert contributes ONE hier_ppm_dist call (a single backoff walk over the
 * alphabet) rather than 256 per-symbol walks; without that the normalizer this
 * pool requires makes it unaffordable.
 */
static double hier_expert_logit_mix(const converse_hier_t* model,
  const double expert_weight[], const unsigned int raw_history[],
  int raw_length, unsigned int target,
  const unsigned int word_history[], int word_length,
  const unsigned int syllable_history[], int syllable_length,
  const unsigned int syllable_role_history[], int syllable_role_length,
  unsigned int skip_history[HIER_SKIP_COUNT][2])
{
  double accum[257], dist[257];
  double total = 0.0, best = 0.0;
  unsigned int id;
  int expert_order, skip;
  for (id = 1; id <= 256; ++id) accum[id] = 0.0;
  for (expert_order = 0; expert_order <= model->expert_order; ++expert_order) {
    const int effective = (expert_order < raw_length)
      ? expert_order : raw_length;
    const double weight = expert_weight[expert_order];
    if (weight <= 0.0) continue;
    hier_ppm_dist_order(&model->expert_ppm, raw_history, raw_length, effective,
      dist);
    for (id = 1; id <= 256; ++id) {
      accum[id] += weight * log(dist[id] > 1e-300 ? dist[id] : 1e-300);
    }
  }
  { const double weight = expert_weight[HIER_EXPERT_WORD];
    if (weight > 0.0) {
      hier_ppm_dist(&model->word_clock_ppm, word_history, word_length, dist);
      for (id = 1; id <= 256; ++id) {
        accum[id] += weight * log(dist[id] > 1e-300 ? dist[id] : 1e-300);
      }
    }
  }
  { const double weight = expert_weight[HIER_EXPERT_SYLLABLE];
    if (weight > 0.0) {
      hier_ppm_dist(&model->syllable_clock_ppm, syllable_history,
        syllable_length, dist);
      for (id = 1; id <= 256; ++id) {
        accum[id] += weight * log(dist[id] > 1e-300 ? dist[id] : 1e-300);
      }
    }
  }
  { const double weight = expert_weight[HIER_EXPERT_SYLLABLE_ROLE];
    if (weight > 0.0) {
      hier_ppm_dist(&model->syllable_role_ppm, syllable_role_history,
        syllable_role_length, dist);
      for (id = 1; id <= 256; ++id) {
        accum[id] += weight * log(dist[id] > 1e-300 ? dist[id] : 1e-300);
      }
    }
  }
  for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
    const double weight = expert_weight[HIER_EXPERT_SKIP2 + skip];
    const int skip_length = hier_skip_history(raw_history, raw_length,
      hier_skip_distance[skip], skip_history[skip]);
    if (weight <= 0.0) continue;
    hier_ppm_dist(&model->skip_ppm[skip], skip_history[skip], skip_length,
      dist);
    for (id = 1; id <= 256; ++id) {
      accum[id] += weight * log(dist[id] > 1e-300 ? dist[id] : 1e-300);
    }
  }
  for (id = 1; id <= 256; ++id) {
    if (1 == id || accum[id] > best) best = accum[id];
  }
  for (id = 1; id <= 256; ++id) {
    accum[id] = exp(accum[id] - best);
    total += accum[id];
  }
  return (total > 0.0) ? (accum[target] / total) : 0.0;
}


/**
 * Fill dist[1..256] with one expert's distribution over the whole alphabet.
 *
 * hier_ppm_prob_order walks the backoff chain per symbol, so scoring every
 * symbol repeats the identical chain 256 times. Here the chain is walked ONCE:
 * start from the unigram, then for each order from low to high overwrite the
 * symbols attested at that order and scale the remaining (excluded) mass by the
 * escape probability. The result matches hier_ppm_prob_order per symbol - same
 * estimator, same exclusion via backoff_norm - at one pass instead of 256.
 */
static void hier_ppm_dist_order(const hier_ppm_t* model,
  const unsigned int history[], int history_length, int maxorder,
  double dist[])
{
  int order, top;
  unsigned int id;
  if (NULL == model || NULL == dist) return;
  for (id = 1; id <= 256; ++id) dist[id] = hier_ppm_unigram(model, id);
  top = maxorder;
  if (top > model->maxorder) top = model->maxorder;
  if (top > history_length) top = history_length;
  for (order = 1; order <= top; ++order) {
    hier_ppm_key_t key;
    const hier_ppm_stats_t* stats;
    hier_ppm_key(&key, history, history_length, order);
    stats = (const hier_ppm_stats_t*)libxs_registry_get(model->contexts,
      &key, sizeof(key), NULL);
    if (NULL == stats || 0 == stats->total) continue;
    { const int method_d = hier_ppm_escape_d();
      const double denom = (0 != method_d)
        ? (2.0 * (double)stats->total)
        : ((double)stats->total + (double)stats->distinct);
      const double escape = (double)stats->distinct / denom;
      const double norm = (stats->backoff_norm > 0.0)
        ? stats->backoff_norm : 1.0;
      for (id = 1; id <= 256; ++id) {
        if (0 != (stats->seen[(id - 1) / 32] & (1u << ((id - 1) % 32)))) {
          hier_ppm_pair_t pair;
          const unsigned int* count;
          pair.key = key;
          pair.next = id;
          count = (const unsigned int*)libxs_registry_get(model->pairs,
            &pair, sizeof(pair), NULL);
          if (NULL != count) {
            dist[id] = (0 != method_d)
              ? ((2.0 * (double)*count - 1.0) / denom)
              : ((double)*count / denom);
          }
        }
        else dist[id] = escape * dist[id] / norm;
      }
    }
  }
}


static void hier_ppm_dist(const hier_ppm_t* model,
  const unsigned int history[], int history_length, double dist[])
{
  hier_ppm_dist_order(model, history, history_length, (NULL != model)
    ? model->maxorder : 0, dist);
}


static int hier_ppm_check(const hier_ppm_t* model)
{
  int result = EXIT_SUCCESS;
  const void* registry_key = NULL;
  size_t cursor = 0;
  int checked = 0;
  void* value;
  double mass = 0.0;
  unsigned int id;
  for (id = 1; id <= 256; ++id) mass += hier_ppm_unigram(model, id);
  if (fabs(mass - 1.0) > 1e-10) result = EXIT_FAILURE;
  value = libxs_registry_begin(model->contexts, &registry_key, &cursor);
  while (EXIT_SUCCESS == result && NULL != value && checked < 128) {
    const hier_ppm_key_t* key = (const hier_ppm_key_t*)registry_key;
    mass = 0.0;
    for (id = 1; id <= 256; ++id) {
      mass += hier_ppm_prob_order(model, key->context, key->order, id,
        key->order);
    }
    if (fabs(mass - 1.0) > 1e-10) result = EXIT_FAILURE;
    ++checked;
    value = libxs_registry_next(model->contexts, &registry_key, &cursor);
  }
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "PPM probability mass check failed: %.17g\n", mass);
  }
  return result;
}


static int hier_is_test(long index, int holdout, long corpus_size)
{
  int result = 0;
  const char* tail = getenv("CONVERSE_HOLDOUT_TAIL");
  if (holdout > 0) {
    if (NULL != tail && '0' != tail[0] && corpus_size > 0) {
      const long split = corpus_size - corpus_size / (long)holdout;
      result = (index >= split) ? 1 : 0;
    }
    else result = (0 == (index % (long)holdout)) ? 1 : 0;
  }
  return result;
}


static int hier_key(int kind, const unsigned char* payload, size_t length,
  unsigned char key[], size_t* key_size)
{
  int result = EXIT_FAILURE;
  if (NULL != payload && length > 0 && length + 1 <= HIER_KEY_MAX
    && NULL != key && NULL != key_size)
  {
    key[0] = (unsigned char)kind;
    memcpy(key + 1, payload, length);
    *key_size = length + 1;
    result = EXIT_SUCCESS;
  }
  return result;
}


static void hier_symbol_observe(libxs_registry_t* symbols, int kind,
  const unsigned char* payload, size_t length)
{
  unsigned char key[HIER_KEY_MAX];
  size_t key_size = 0;
  if (NULL != symbols
    && EXIT_SUCCESS == hier_key(kind, payload, length, key, &key_size))
  {
    hier_symbol_t* symbol = (hier_symbol_t*)libxs_registry_get(symbols,
      key, key_size, NULL);
    if (NULL != symbol) ++symbol->count;
    else {
      hier_symbol_t fresh;
      fresh.count = 1;
      fresh.id = 0;
      libxs_registry_set(symbols, key, key_size, &fresh, sizeof(fresh), NULL);
    }
  }
}


static unsigned int hier_symbol_find(const libxs_registry_t* symbols,
  int kind, const unsigned char* payload, size_t length)
{
  unsigned int result = 0;
  unsigned char key[HIER_KEY_MAX];
  size_t key_size = 0;
  if (NULL != symbols
    && EXIT_SUCCESS == hier_key(kind, payload, length, key, &key_size))
  {
    const hier_symbol_t* symbol = (const hier_symbol_t*)libxs_registry_get(
      symbols, key, key_size, NULL);
    if (NULL != symbol) result = symbol->id;
  }
  return result;
}


static unsigned int hier_symbol_assign(libxs_registry_t* symbols,
  int mincount, unsigned int first)
{
  unsigned int result = first - 1;
  const void* key = NULL;
  size_t cursor = 0;
  void* value = libxs_registry_begin(symbols, &key, &cursor);
  while (NULL != value) {
    hier_symbol_t* symbol = (hier_symbol_t*)value;
    if (symbol->count >= mincount) {
      ++result;
      symbol->id = result;
    }
    value = libxs_registry_next(symbols, &key, &cursor);
  }
  return result;
}


static void hier_history_push(unsigned int history[], int* length,
  int capacity, unsigned int id)
{
  if (*length < capacity) history[(*length)++] = id;
  else {
    int pos;
    for (pos = 1; pos < capacity; ++pos) history[pos - 1] = history[pos];
    history[capacity - 1] = id;
  }
}


static void hier_ngram_observe(libxs_ngram_t* model,
  unsigned int history[], int* history_length, unsigned int id)
{
  libxs_ngram_observe(model, history, *history_length, id);
  hier_history_push(history, history_length, LIBXS_NGRAM_ORDER_MAX, id);
}


static int hier_read(const libxs_token_stream_t* stream, size_t token_pos,
  unsigned char payload[], libxs_token_info_t* info)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != payload && NULL != info
    && EXIT_SUCCESS == libxs_token_read(stream->data, stream->size,
      token_pos, payload, COMPOSE_MAXTEXT, info))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


static int hier_role_kind(const libxs_token_stream_t* stream,
  size_t token_pos, const libxs_token_info_t* info, int previous_kind)
{
  int result = info->kind;
  if (LIBXS_TOKEN_TEXT == info->kind) {
    const size_t next_pos = token_pos + info->cells;
    int next_kind = LIBXS_TOKEN_CONTROL;
    int begin = (LIBXS_TOKEN_TEXT != previous_kind) ? 1 : 0;
    int end = 1;
    if (next_pos < stream->size) {
      next_kind = libxs_token_kind(stream->data + next_pos);
      if (LIBXS_TOKEN_TEXT == next_kind) end = 0;
    }
    if (0 != begin && 0 != end) result += 8 * HIER_ROLE_SINGLE;
    else if (0 != begin) result += 8 * HIER_ROLE_BEGIN;
    else if (0 != end) result += 8 * HIER_ROLE_END;
    else result += 8 * HIER_ROLE_MIDDLE;
  }
  return result;
}


static void hier_count_word(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  libxs_token_stream_t stream;
  size_t token_pos = 0;
  int previous_kind = LIBXS_TOKEN_CONTROL;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      int kind;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      kind = hier_role_kind(&stream, token_pos, &info, previous_kind);
      hier_symbol_observe(model->syllables, kind, syllable, info.length);
      hier_symbol_observe(model->syllable_payloads, info.kind, syllable,
        info.length);
      previous_kind = info.kind;
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_count_text(converse_hier_t* model,
  const char* text, int text_length)
{
  libxs_token_stream_t stream;
  size_t token_pos = 0;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      hier_symbol_observe(model->symbols, info.kind, payload, info.length);
      if (LIBXS_TOKEN_TEXT == info.kind) {
        hier_count_word(model, payload, info.length);
      }
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_train_bytes(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t pos;
  history[0] = HIER_HISTORY_START;
  for (pos = 0; pos < length; ++pos) {
    const unsigned int id = (unsigned int)payload[pos] + 1u;
    hier_ngram_observe(&model->byte_model, history, &history_length, id);
  }
  hier_ngram_observe(&model->byte_model, history, &history_length,
    HIER_BYTE_END);
}


static void hier_train_word(converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  int previous_kind = LIBXS_TOKEN_CONTROL;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      int kind;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      kind = hier_role_kind(&stream, token_pos, &info, previous_kind);
      id = hier_symbol_find(model->syllables, kind, syllable, info.length);
      if (0 == id) id = HIER_SYLLABLE_ESCAPE;
      hier_ngram_observe(&model->syllable_model, history, &history_length, id);
      hier_train_bytes(model, syllable, info.length);
      previous_kind = info.kind;
      token_pos += info.cells;
    }
    hier_ngram_observe(&model->syllable_model, history, &history_length,
      HIER_SYLLABLE_END);
  }
  libxs_token_stream_release(&stream);
}


static void hier_train_text(converse_hier_t* model,
  const char* text, int text_length)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      id = hier_symbol_find(model->symbols, info.kind, payload, info.length);
      if (0 == id) {
        id = (LIBXS_TOKEN_TEXT == info.kind)
          ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
      }
      hier_ngram_observe(&model->word_model, history, &history_length, id);
      if (LIBXS_TOKEN_TEXT == info.kind) {
        hier_train_word(model, payload, info.length);
      }
      else hier_train_bytes(model, payload, info.length);
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static unsigned int hier_clock_symbol(const converse_hier_t* model,
  const libxs_registry_t* symbols, int kind, const unsigned char* payload,
  size_t length, unsigned int base)
{
  unsigned int id = hier_symbol_find(symbols, kind, payload, length);
  if (0 == id) {
    if (symbols == model->symbols) {
      id = (LIBXS_TOKEN_TEXT == kind)
        ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
    }
    else id = HIER_SYLLABLE_ESCAPE;
  }
  return base + id;
}


static int hier_clock_states(const converse_hier_t* model,
  const char* text, int text_length, const libxs_tokenizer_t* tokenizer,
  const libxs_registry_t* symbols, unsigned int base,
  unsigned int states[], int role_aware)
{
  int result = EXIT_FAILURE;
  libxs_token_stream_t stream;
  size_t token_pos = 0, byte_pos = 0;
  unsigned int previous = base;
  int previous_kind = LIBXS_TOKEN_CONTROL;
  libxs_token_stream_init(&stream);
  if (NULL != model && NULL != text && text_length >= 0
    && NULL != tokenizer && NULL != symbols && NULL != states
    && EXIT_SUCCESS == libxs_token_stream_encode(tokenizer, &stream,
      (const unsigned char*)text, (size_t)text_length))
  {
    result = EXIT_SUCCESS;
    while (token_pos < stream.size && byte_pos < (size_t)text_length) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      size_t offset;
      unsigned int current;
      int kind;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) {
        result = EXIT_FAILURE;
        break;
      }
      kind = (0 != role_aware)
        ? hier_role_kind(&stream, token_pos, &info, previous_kind) : info.kind;
      current = hier_clock_symbol(model, symbols, kind, payload,
        info.length, base);
      for (offset = 0; offset < info.length
        && byte_pos < (size_t)text_length; ++offset)
      {
        states[byte_pos++] = previous;
      }
      previous = current;
      previous_kind = info.kind;
      token_pos += info.cells;
    }
    if (byte_pos != (size_t)text_length) result = EXIT_FAILURE;
  }
  libxs_token_stream_release(&stream);
  return result;
}


static int hier_clock_history(const unsigned int raw_history[],
  int raw_length, unsigned int word_state, unsigned int syllable_state,
  unsigned int history[])
{
  int keep = raw_length;
  int result = 2;
  int pos;
  if (keep > HIER_CLOCK_RAW_MAX) keep = HIER_CLOCK_RAW_MAX;
  history[0] = word_state;
  history[1] = syllable_state;
  for (pos = raw_length - keep; pos < raw_length; ++pos) {
    history[result++] = raw_history[pos];
  }
  return result;
}


static void hier_train_clock_text(converse_hier_t* model,
  const char* text, int text_length)
{
  unsigned int word_state[COMPOSE_MAXTEXT];
  unsigned int syllable_state[COMPOSE_MAXTEXT];
  unsigned int syllable_role_state[COMPOSE_MAXTEXT];
  unsigned int raw_history[HIER_PPM_ORDER_MAX];
  int raw_length = 1;
  int pos;
  double recurrent[HIER_RECURRENT_DIM];
  memset(recurrent, 0, sizeof(recurrent));
  raw_history[0] = HIER_CLOCK_BYTE_START;
  if (text_length > 0 && text_length <= COMPOSE_MAXTEXT
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->word_tokenizer, model->symbols, HIER_CLOCK_WORD_BASE,
      word_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllable_payloads,
      HIER_CLOCK_SYLLABLE_BASE, syllable_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllables,
      HIER_CLOCK_SYLLABLE_BASE, syllable_role_state, 1))
  {
    for (pos = 0; pos < text_length; ++pos) {
      unsigned int history[LIBXS_NGRAM_ORDER_MAX];
      unsigned int skip_history[HIER_SKIP_COUNT][2];
      const unsigned int id = (unsigned int)(unsigned char)text[pos] + 1u;
      const int history_length = hier_clock_history(raw_history, raw_length,
        word_state[pos], syllable_state[pos], history);
      unsigned int recurrent_history[HIER_RECURRENT_ORDER];
      unsigned int word_history[2], syllable_history[2];
      unsigned int syllable_role_history[2];
      const int recurrent_length = hier_recurrent_history(recurrent,
        raw_history, raw_length, recurrent_history);
      const int word_length = hier_struct_history(word_state[pos], raw_history,
        raw_length, word_history);
      const int syllable_length = hier_struct_history(syllable_state[pos],
        raw_history, raw_length, syllable_history);
      const int syllable_role_length = hier_struct_history(
        syllable_role_state[pos], raw_history, raw_length,
        syllable_role_history);
      int skip;
      libxs_ngram_observe(&model->clock_byte_model, history, history_length,
        id);
      libxs_ngram_observe(&model->stream_byte_model, raw_history, raw_length,
        id);
      hier_ppm_observe(&model->clock_ppm, history, history_length, id);
      hier_ppm_observe(&model->stream_ppm, raw_history, raw_length, id);
      hier_ppm_observe(&model->recurrent_ppm, recurrent_history,
        recurrent_length, id);
      hier_ppm_observe(&model->adaptive_ppm, raw_history, raw_length, id);
      hier_ppm_observe(&model->expert_ppm, raw_history, raw_length, id);
      hier_ppm_observe(&model->word_clock_ppm, word_history, word_length, id);
      hier_ppm_observe(&model->syllable_clock_ppm, syllable_history,
        syllable_length, id);
      hier_ppm_observe(&model->syllable_role_ppm, syllable_role_history,
        syllable_role_length, id);
      for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
        const int skip_length = hier_skip_history(raw_history, raw_length,
          hier_skip_distance[skip], skip_history[skip]);
        hier_ppm_observe(&model->skip_ppm[skip], skip_history[skip],
          skip_length, id);
      }
      hier_history_push(raw_history, &raw_length, HIER_PPM_ORDER_MAX, id);
      hier_recurrent_update(recurrent, (unsigned int)(unsigned char)text[pos],
        model->recurrent_decay);
    }
  }
}


static double hier_bits(const libxs_ngram_t* model,
  const unsigned int history[], int history_length, unsigned int id)
{
  double result = HUGE_VAL;
  const double probability = libxs_ngram_prob(model, history,
    history_length, id);
  if (probability > 0.0) result = -log(probability) / log(2.0);
  return result;
}


static double hier_score_bytes(const converse_hier_t* model,
  const unsigned char* payload, size_t length)
{
  double result = 0.0;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t pos;
  history[0] = HIER_HISTORY_START;
  for (pos = 0; pos < length; ++pos) {
    const unsigned int id = (unsigned int)payload[pos] + 1u;
    result += hier_bits(&model->byte_model, history, history_length, id);
    hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
  }
  result += hier_bits(&model->byte_model, history, history_length,
    HIER_BYTE_END);
  return result;
}


static double hier_score_word(const converse_hier_t* model,
  const unsigned char* payload, size_t length, hier_eval_t* evaluation)
{
  double result = 0.0;
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  int previous_kind = LIBXS_TOKEN_CONTROL;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->syllable_tokenizer,
    &stream, payload, length))
  {
    while (token_pos < stream.size) {
      unsigned char syllable[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      double bits;
      int kind;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, syllable, &info)) break;
      kind = hier_role_kind(&stream, token_pos, &info, previous_kind);
      id = hier_symbol_find(model->syllables, kind, syllable, info.length);
      if (0 == id) id = HIER_SYLLABLE_ESCAPE;
      bits = hier_bits(&model->syllable_model, history, history_length, id);
      result += bits;
      evaluation->syllable_bits += bits;
      if (HIER_SYLLABLE_ESCAPE == id) {
        const double byte_bits = hier_score_bytes(model, syllable, info.length);
        result += byte_bits;
        evaluation->byte_bits += byte_bits;
        ++evaluation->nsyllable_escape;
      }
      hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
      previous_kind = info.kind;
      token_pos += info.cells;
    }
    { const double bits = hier_bits(&model->syllable_model, history,
        history_length, HIER_SYLLABLE_END);
      result += bits;
      evaluation->syllable_bits += bits;
    }
  }
  libxs_token_stream_release(&stream);
  return result;
}


static void hier_score_text(const converse_hier_t* model,
  const char* text, int text_length, hier_eval_t* evaluation)
{
  libxs_token_stream_t stream;
  unsigned int history[LIBXS_NGRAM_ORDER_MAX];
  int history_length = 1;
  size_t token_pos = 0;
  history[0] = HIER_HISTORY_START;
  libxs_token_stream_init(&stream);
  if (EXIT_SUCCESS == libxs_token_stream_encode(model->word_tokenizer,
    &stream, (const unsigned char*)text, (size_t)text_length))
  {
    while (token_pos < stream.size) {
      unsigned char payload[COMPOSE_MAXTEXT];
      libxs_token_info_t info;
      unsigned int id;
      unsigned int top_ids[1];
      double bits;
      int deep;
      if (EXIT_SUCCESS != hier_read(&stream, token_pos, payload, &info)) break;
      id = hier_symbol_find(model->symbols, info.kind, payload, info.length);
      if (0 == id) {
        id = (LIBXS_TOKEN_TEXT == info.kind)
          ? HIER_ESCAPE_TEXT : HIER_ESCAPE_NATIVE;
      }
      bits = hier_bits(&model->word_model, history, history_length, id);
      evaluation->top_bits += bits;
      if (HIER_ESCAPE_TEXT == id) {
        bits += hier_score_word(model, payload, info.length, evaluation);
        ++evaluation->ntext_escape;
      }
      else if (HIER_ESCAPE_NATIVE == id) {
        const double byte_bits = hier_score_bytes(model, payload, info.length);
        bits += byte_bits;
        evaluation->byte_bits += byte_bits;
        ++evaluation->nnative_escape;
      }
      deep = (history_length >= model->maxorder
        && NULL != libxs_ngram_lookup(&model->word_model, history,
          history_length, model->maxorder)) ? 1 : 0;
      if (0 < libxs_ngram_predict(&model->word_model, history,
        history_length, top_ids, 1, NULL) && top_ids[0] == id)
      {
        ++evaluation->ntop1;
      }
      ++evaluation->ntokens;
      evaluation->bits += bits;
      evaluation->bytes += (double)info.length;
      if (0 != deep) {
        ++evaluation->ndeep;
        evaluation->deep_bits += bits;
        evaluation->deep_bytes += (double)info.length;
      }
      else {
        ++evaluation->nshallow;
        evaluation->shallow_bits += bits;
        evaluation->shallow_bytes += (double)info.length;
      }
      hier_history_push(history, &history_length, LIBXS_NGRAM_ORDER_MAX, id);
      token_pos += info.cells;
    }
  }
  libxs_token_stream_release(&stream);
}


static void hier_score_clock_text(converse_hier_t* model,
  const char* text, int text_length, double mix,
  hier_clock_eval_t* evaluation)
{
  unsigned int word_state[COMPOSE_MAXTEXT];
  unsigned int syllable_state[COMPOSE_MAXTEXT];
  unsigned int syllable_role_state[COMPOSE_MAXTEXT];
  unsigned int raw_history[HIER_PPM_ORDER_MAX];
  int raw_length = 1;
  int pos;
  double recurrent[HIER_RECURRENT_DIM];
  memset(recurrent, 0, sizeof(recurrent));
  raw_history[0] = HIER_CLOCK_BYTE_START;
  if (text_length > 0 && text_length <= COMPOSE_MAXTEXT
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->word_tokenizer, model->symbols, HIER_CLOCK_WORD_BASE,
      word_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllable_payloads,
      HIER_CLOCK_SYLLABLE_BASE, syllable_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, text, text_length,
      model->syllable_tokenizer, model->syllables,
      HIER_CLOCK_SYLLABLE_BASE, syllable_role_state, 1))
  {
    const long eval_max = hier_eval_max();
    const int eval_stop = (0 < eval_max && eval_max < (long)text_length)
      ? (int)eval_max : text_length;
    for (pos = 0; pos < eval_stop; ++pos) {
      unsigned int history[LIBXS_NGRAM_ORDER_MAX];
      unsigned int raw_ids[1], context_ids[1];
      const unsigned int id = (unsigned int)(unsigned char)text[pos] + 1u;
      const int history_length = hier_clock_history(raw_history, raw_length,
        word_state[pos], syllable_state[pos], history);
      unsigned int recurrent_history[HIER_RECURRENT_ORDER];
      unsigned int word_history[2], syllable_history[2];
      unsigned int syllable_role_history[2];
      unsigned int skip_history[HIER_SKIP_COUNT][2];
      const int recurrent_length = hier_recurrent_history(recurrent,
        raw_history, raw_length, recurrent_history);
      const int word_length = hier_struct_history(word_state[pos], raw_history,
        raw_length, word_history);
      const int syllable_length = hier_struct_history(syllable_state[pos],
        raw_history, raw_length, syllable_history);
      const int syllable_role_length = hier_struct_history(
        syllable_role_state[pos], raw_history, raw_length,
        syllable_role_history);
      const double raw_probability = libxs_ngram_prob(
        &model->stream_byte_model, raw_history, raw_length, id);
      const double context_probability = libxs_ngram_prob(
        &model->clock_byte_model, history, history_length, id);
      const double probability = mix * context_probability
        + (1.0 - mix) * raw_probability;
      const double raw_ppm_probability = hier_ppm_prob(&model->stream_ppm,
        raw_history, raw_length, id);
      const double context_ppm_probability = hier_ppm_prob(&model->clock_ppm,
        history, history_length, id);
      const double ppm_probability = mix * context_ppm_probability
        + (1.0 - mix) * raw_ppm_probability;
      const double recurrent_probability = hier_ppm_prob(
        &model->recurrent_ppm, recurrent_history, recurrent_length, id);
      const double recurrent_mix_probability = mix * recurrent_probability
        + (1.0 - mix) * raw_ppm_probability;
      const double frozen_interp_probability = hier_ppm_interp_prob(
        &model->stream_ppm, raw_history, raw_length, id);
      const double adaptive_probability = hier_ppm_interp_prob(
        &model->adaptive_ppm, raw_history, raw_length, id);
      double expert_probability[HIER_EXPERT_MAX];
      double adaptive_expert_probability[HIER_EXPERT_MAX];
      double expert_mixture = 0.0;
      double adaptive_expert_mixture = 0.0;
      int expert_order;
      int skip;
      const double raw_bits = -log(raw_probability) / log(2.0);
      const double context_bits = -log(context_probability) / log(2.0);
      const double bits = -log(probability) / log(2.0);
      const int deep = (history_length >= model->clock_order
        && NULL != libxs_ngram_lookup(&model->clock_byte_model, history,
          history_length, model->clock_order)) ? 1 : 0;
      if (0 < libxs_ngram_predict(&model->stream_byte_model, raw_history,
        raw_length, raw_ids, 1, NULL) && raw_ids[0] == id)
      {
        ++evaluation->nraw_top1;
      }
      if (0 < libxs_ngram_predict(&model->clock_byte_model, history,
        history_length, context_ids, 1, NULL) && context_ids[0] == id)
      {
        ++evaluation->ncontext_top1;
      }
      for (expert_order = 0; expert_order <= model->expert_order;
        ++expert_order)
      {
        const int effective = (expert_order < raw_length)
          ? expert_order : raw_length;
        expert_probability[expert_order] = hier_ppm_prob_order(
          &model->expert_ppm, raw_history, raw_length, id, effective);
        expert_mixture += evaluation->expert_weight[expert_order]
          * expert_probability[expert_order];
      }
      expert_probability[HIER_EXPERT_WORD] = hier_ppm_prob(
        &model->word_clock_ppm, word_history, word_length, id);
      expert_probability[HIER_EXPERT_SYLLABLE] = hier_ppm_prob(
        &model->syllable_clock_ppm, syllable_history, syllable_length, id);
      expert_probability[HIER_EXPERT_SYLLABLE_ROLE] = hier_ppm_prob(
        &model->syllable_role_ppm, syllable_role_history,
        syllable_role_length, id);
      expert_mixture += evaluation->expert_weight[HIER_EXPERT_WORD]
        * expert_probability[HIER_EXPERT_WORD]
        + evaluation->expert_weight[HIER_EXPERT_SYLLABLE]
          * expert_probability[HIER_EXPERT_SYLLABLE]
        + evaluation->expert_weight[HIER_EXPERT_SYLLABLE_ROLE]
          * expert_probability[HIER_EXPERT_SYLLABLE_ROLE];
      for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
        const int skip_length = hier_skip_history(raw_history, raw_length,
          hier_skip_distance[skip], skip_history[skip]);
        expert_probability[HIER_EXPERT_SKIP2 + skip] = hier_ppm_prob(
          &model->skip_ppm[skip], skip_history[skip], skip_length, id);
        expert_mixture += evaluation->expert_weight[HIER_EXPERT_SKIP2 + skip]
          * expert_probability[HIER_EXPERT_SKIP2 + skip];
      }
      if (0 != hier_ppm_logit()) {
        expert_mixture = hier_expert_logit_mix(model,
          evaluation->expert_weight, raw_history,
          raw_length, id, word_history, word_length, syllable_history,
          syllable_length, syllable_role_history, syllable_role_length,
          skip_history);
      }
      for (expert_order = 0; expert_order <= model->clock_order;
        ++expert_order)
      {
        const int effective = (expert_order < raw_length)
          ? expert_order : raw_length;
        adaptive_expert_probability[expert_order]
          = hier_ppm_interp_prob_order(&model->adaptive_ppm, raw_history,
            raw_length, id, effective);
        adaptive_expert_mixture
          += evaluation->adaptive_expert_weight[expert_order]
            * adaptive_expert_probability[expert_order];
      }
      if (0 == (evaluation->nbytes % model->top_stride)) {
        const int ppm_rank = hier_ppm_rank(&model->stream_ppm, raw_history,
          raw_length, id, 0);
        const int adaptive_rank = hier_ppm_rank(&model->adaptive_ppm,
          raw_history, raw_length, id, 1);
        const int expert_rank = hier_expert_rank(&model->expert_ppm,
          &model->word_clock_ppm, &model->syllable_clock_ppm,
          &model->syllable_role_ppm, model->skip_ppm,
          raw_history, raw_length, id, word_history, syllable_history,
          syllable_role_history, skip_history,
          evaluation->expert_weight, model->expert_order, 0);
        const int adaptive_expert_rank = hier_expert_rank(
          &model->adaptive_ppm, &model->word_clock_ppm,
          &model->syllable_clock_ppm, &model->syllable_role_ppm,
          model->skip_ppm, raw_history, raw_length, id, word_history,
          syllable_history, syllable_role_history, skip_history,
          evaluation->adaptive_expert_weight, model->clock_order, 1);
        ++evaluation->nppm;
        if (1 == ppm_rank) ++evaluation->nppm_top1;
        if (ppm_rank >= 1 && ppm_rank <= 3) ++evaluation->nppm_top3;
        if (1 == adaptive_rank) ++evaluation->nadaptive_top1;
        if (adaptive_rank >= 1 && adaptive_rank <= 3) {
          ++evaluation->nadaptive_top3;
        }
        if (1 == expert_rank) ++evaluation->nexpert_top1;
        if (expert_rank >= 1 && expert_rank <= 3) {
          ++evaluation->nexpert_top3;
        }
        if (1 == adaptive_expert_rank) ++evaluation->nadaptive_expert_top1;
        if (adaptive_expert_rank >= 1 && adaptive_expert_rank <= 3) {
          ++evaluation->nadaptive_expert_top3;
        }
      }
      ++evaluation->nbytes;
      evaluation->raw_bits += raw_bits;
      evaluation->context_bits += context_bits;
      evaluation->mix_bits += bits;
      evaluation->raw_ppm_bits += -log(raw_ppm_probability) / log(2.0);
      evaluation->context_ppm_bits += -log(context_ppm_probability) / log(2.0);
      evaluation->ppm_mix_bits += -log(ppm_probability) / log(2.0);
      evaluation->recurrent_bits += -log(recurrent_probability) / log(2.0);
      evaluation->recurrent_mix_bits += -log(recurrent_mix_probability)
        / log(2.0);
      evaluation->frozen_interp_bits += -log(frozen_interp_probability)
        / log(2.0);
      evaluation->adaptive_bits += -log(adaptive_probability) / log(2.0);
      for (expert_order = 0; expert_order <= model->expert_order;
        ++expert_order)
      {
        evaluation->expert_bits[expert_order]
          += -log(expert_probability[expert_order]) / log(2.0);
      }
      evaluation->expert_bits[HIER_EXPERT_WORD]
        += -log(expert_probability[HIER_EXPERT_WORD]) / log(2.0);
      evaluation->expert_bits[HIER_EXPERT_SYLLABLE]
        += -log(expert_probability[HIER_EXPERT_SYLLABLE]) / log(2.0);
      evaluation->expert_bits[HIER_EXPERT_SYLLABLE_ROLE]
        += -log(expert_probability[HIER_EXPERT_SYLLABLE_ROLE]) / log(2.0);
      for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
        evaluation->expert_bits[HIER_EXPERT_SKIP2 + skip]
          += -log(expert_probability[HIER_EXPERT_SKIP2 + skip]) / log(2.0);
      }
      evaluation->expert_mix_bits += -log(expert_mixture) / log(2.0);
      { const int attested = hier_ppm_attested(&model->expert_ppm, raw_history,
          raw_length, model->expert_order);
        const double mix_bits = -log(expert_mixture) / log(2.0);
        if (0 != attested) {
          ++evaluation->nexpert_attested;
          evaluation->expert_attested_bits += mix_bits;
        }
        else {
          ++evaluation->nexpert_novel;
          evaluation->expert_novel_bits += mix_bits;
        }
      }
      for (expert_order = 0; expert_order <= model->clock_order;
        ++expert_order)
      {
        evaluation->adaptive_expert_bits[expert_order]
          += -log(adaptive_expert_probability[expert_order]) / log(2.0);
      }
      evaluation->adaptive_expert_mix_bits
        += -log(adaptive_expert_mixture) / log(2.0);
      if (0 != deep) {
        ++evaluation->ndeep;
        evaluation->deep_bits += bits;
      }
      else {
        ++evaluation->nshallow;
        evaluation->shallow_bits += bits;
      }
      hier_ppm_observe(&model->adaptive_ppm, raw_history, raw_length, id);
      hier_expert_update(evaluation->expert_weight, expert_probability,
        HIER_EXPERT_LAST, expert_mixture, model->expert_rate,
        model->expert_share);
      hier_expert_update(evaluation->adaptive_expert_weight,
        adaptive_expert_probability, model->clock_order,
        adaptive_expert_mixture, model->expert_rate, model->expert_share);
      hier_history_push(raw_history, &raw_length, HIER_PPM_ORDER_MAX, id);
      hier_recurrent_update(recurrent, (unsigned int)(unsigned char)text[pos],
        model->recurrent_decay);
    }
  }
}


/**
 * Seed the mixture with uniform weight over the experts that actually exist:
 * the per-order experts up to expert_order, then each named expert. Seeding the
 * unused slots (the gap between expert_order and HIER_PPM_ORDER_MAX) would give
 * weight to experts whose probability is always zero and dilute the mixture.
 */
static void hier_expert_weight_init(const converse_hier_t* model,
  double weight[])
{
  const int nskip = (0 != hier_skip_on()) ? HIER_SKIP_COUNT : 0;
  const int nexperts = model->expert_order + 1 + 3 + nskip;
  const double uniform = 1.0 / (double)nexperts;
  /**
   * CONVERSE_HIER_WIDEINIT reinstates the superseded initialization for
   * attribution only: seeding the whole fixed index span gives weight to
   * per-order experts above expert_order that never produce a probability,
   * which dilutes the mixture.
   */
  const int wide = (NULL != getenv("CONVERSE_HIER_WIDEINIT")) ? 1 : 0;
  int expert_order, skip;
  if (0 != wide) {
    const double share = 1.0 / (double)HIER_EXPERT_MAX;
    int slot;
    for (slot = 0; slot < HIER_EXPERT_MAX; ++slot) weight[slot] = share;
  }
  else {
    for (expert_order = 0; expert_order <= model->expert_order;
      ++expert_order)
    {
      weight[expert_order] = uniform;
    }
    weight[HIER_EXPERT_WORD] = uniform;
    weight[HIER_EXPERT_SYLLABLE] = uniform;
    weight[HIER_EXPERT_SYLLABLE_ROLE] = uniform;
    /**
     * A disabled expert is left at weight zero rather than removed: the mixture
     * adds weight*probability, the fixed-share update already skips zero-weight
     * experts, and the logarithmic pool already ignores them, so zero weight is
     * exactly "absent" everywhere without a second code path.
     */
    for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
      weight[HIER_EXPERT_SKIP2 + skip] = (skip < nskip) ? uniform : 0.0;
    }
  }
}


/**
 * Bits the expert bank assigns to candidate[] given prefix[] as history: the
 * prefix is scored to warm the byte history and the mixture weights but does not
 * contribute to the returned total, so what comes back is a CONDITIONAL code
 * length, -log2 P(candidate | prefix). Scoring the candidate alone would instead
 * measure how ordinary it is and would prefer boilerplate to answers.
 *
 * Both texts are scored through the same per-position path the evaluator uses,
 * so this reports the bits of the model the BPC numbers describe. Returns
 * EXIT_SUCCESS and *bits when the clock states could be derived for both.
 */
static int hier_score_conditional(const converse_hier_t* model,
  const char* prefix, int prefix_length, const char* candidate,
  int candidate_length, int score_length, double* bits)
{
  int result = EXIT_FAILURE;
  char joined[COMPOSE_MAXTEXT];
  unsigned int word_state[COMPOSE_MAXTEXT];
  unsigned int syllable_state[COMPOSE_MAXTEXT];
  unsigned int syllable_role_state[COMPOSE_MAXTEXT];
  unsigned int raw_history[HIER_PPM_ORDER_MAX];
  double weight[HIER_EXPERT_MAX];
  int total_length = 0;
  if (NULL != model && 0 != model->ready
    && (NULL != prefix || 0 == prefix_length)
    && NULL != candidate && NULL != bits && prefix_length >= 0
    && candidate_length > 0 && candidate_length + 1 < COMPOSE_MAXTEXT)
  {
    /**
     * The candidate is always kept whole; an over-long prefix is trimmed from
     * the LEFT so the bytes nearest the candidate survive. Those are the only
     * ones the model can see anyway - the byte context reaches back at most
     * HIER_PPM_ORDER_MAX - so trimming the head costs nothing and lets a
     * growing generation context be scored without failing.
     */
    int keep = prefix_length;
    if (keep + candidate_length + 1 >= COMPOSE_MAXTEXT) {
      keep = COMPOSE_MAXTEXT - candidate_length - 2;
      if (keep < 0) keep = 0;
    }
    if (keep > 0) {
      memcpy(joined, prefix + prefix_length - keep, (size_t)keep);
      total_length = keep;
      joined[total_length++] = '\n';
    }
    memcpy(joined + total_length, candidate, (size_t)candidate_length);
    total_length += candidate_length;
    joined[total_length] = '\0';
  }
  if (total_length > 0
    && EXIT_SUCCESS == hier_clock_states(model, joined, total_length,
      model->word_tokenizer, model->symbols, HIER_CLOCK_WORD_BASE,
      word_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, joined, total_length,
      model->syllable_tokenizer, model->syllable_payloads,
      HIER_CLOCK_SYLLABLE_BASE, syllable_state, 0)
    && EXIT_SUCCESS == hier_clock_states(model, joined, total_length,
      model->syllable_tokenizer, model->syllables,
      HIER_CLOCK_SYLLABLE_BASE, syllable_role_state, 1))
  {
    const int scored_from = total_length - candidate_length;
    const int scored_to = (0 < score_length
      && score_length < candidate_length)
      ? (scored_from + score_length) : total_length;
    int raw_length = 1;
    int pos;
    double total_bits = 0.0;
    memset(weight, 0, sizeof(weight));
    hier_expert_weight_init(model, weight);
    raw_history[0] = HIER_CLOCK_BYTE_START;
    for (pos = 0; pos < total_length; ++pos) {
      const unsigned int id = (unsigned int)(unsigned char)joined[pos] + 1u;
      unsigned int word_history[2], syllable_history[2];
      unsigned int syllable_role_history[2];
      unsigned int skip_history[HIER_SKIP_COUNT][2];
      double probability[HIER_EXPERT_MAX];
      double mixture = 0.0;
      int expert_order, skip;
      const int word_length = hier_struct_history(word_state[pos], raw_history,
        raw_length, word_history);
      const int syllable_length = hier_struct_history(syllable_state[pos],
        raw_history, raw_length, syllable_history);
      const int syllable_role_length = hier_struct_history(
        syllable_role_state[pos], raw_history, raw_length,
        syllable_role_history);
      memset(probability, 0, sizeof(probability));
      for (expert_order = 0; expert_order <= model->expert_order;
        ++expert_order)
      {
        const int effective = (expert_order < raw_length)
          ? expert_order : raw_length;
        probability[expert_order] = hier_ppm_prob_order(&model->expert_ppm,
          raw_history, raw_length, id, effective);
        mixture += weight[expert_order] * probability[expert_order];
      }
      probability[HIER_EXPERT_WORD] = hier_ppm_prob(&model->word_clock_ppm,
        word_history, word_length, id);
      probability[HIER_EXPERT_SYLLABLE] = hier_ppm_prob(
        &model->syllable_clock_ppm, syllable_history, syllable_length, id);
      probability[HIER_EXPERT_SYLLABLE_ROLE] = hier_ppm_prob(
        &model->syllable_role_ppm, syllable_role_history,
        syllable_role_length, id);
      mixture += weight[HIER_EXPERT_WORD] * probability[HIER_EXPERT_WORD]
        + weight[HIER_EXPERT_SYLLABLE] * probability[HIER_EXPERT_SYLLABLE]
        + weight[HIER_EXPERT_SYLLABLE_ROLE]
          * probability[HIER_EXPERT_SYLLABLE_ROLE];
      for (skip = 0; skip < HIER_SKIP_COUNT; ++skip) {
        const int skip_length = hier_skip_history(raw_history, raw_length,
          hier_skip_distance[skip], skip_history[skip]);
        probability[HIER_EXPERT_SKIP2 + skip] = hier_ppm_prob(
          &model->skip_ppm[skip], skip_history[skip], skip_length, id);
        mixture += weight[HIER_EXPERT_SKIP2 + skip]
          * probability[HIER_EXPERT_SKIP2 + skip];
      }
      if (0 != hier_ppm_logit()) {
        mixture = hier_expert_logit_mix(model, weight, raw_history, raw_length,
          id, word_history, word_length, syllable_history, syllable_length,
          syllable_role_history, syllable_role_length, skip_history);
      }
      if (pos >= scored_from && pos < scored_to && mixture > 0.0) {
        total_bits += -log(mixture) / log(2.0);
      }
      hier_expert_update(weight, probability, HIER_EXPERT_LAST, mixture,
        model->expert_rate, model->expert_share);
      hier_history_push(raw_history, &raw_length, HIER_PPM_ORDER_MAX, id);
    }
    *bits = total_bits;
    result = EXIT_SUCCESS;
  }
  return result;
}


/**
 * Bits the query SAVES on each candidate: the candidate's code length scored
 * cold, minus its code length scored with the query as history. This is the lift
 * metric P(candidate | query) / P(candidate) expressed in bits, and the
 * distinction matters more than it looks.
 *
 * Ranking by the conditional length alone ranks by how ORDINARY a candidate is,
 * because that term dominates: common dialogue outscores a naming sentence
 * carrying the queried proper noun. Subtracting the unconditional length cancels
 * the candidate's own predictability and leaves only what conditioning on the
 * query contributed, which is the quantity relevance actually corresponds to.
 * Larger is better, so callers rank by descending value.
 */
int converse_hier_rescore(const converse_hier_t* model,
  const char* query, int query_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates, double bits[])
{
  int result = EXIT_FAILURE;
  if (NULL != model && 0 != model->ready && NULL != query
    && NULL != candidates && NULL != candidate_lengths && NULL != bits
    && 0 < ncandidates)
  {
    const int window = hier_rescore_window();
    int slot, nscored = 0;
    for (slot = 0; slot < ncandidates; ++slot) {
      double conditional = 0.0, unconditional = 0.0;
      const int scored = (0 < window && window < candidate_lengths[slot])
        ? window : candidate_lengths[slot];
      bits[slot] = 0.0;
      if (NULL != candidates[slot] && 0 < candidate_lengths[slot]
        && EXIT_SUCCESS == hier_score_conditional(model, query, query_length,
          candidates[slot], candidate_lengths[slot], window, &conditional)
        && EXIT_SUCCESS == hier_score_conditional(model, NULL, 0,
          candidates[slot], candidate_lengths[slot], window, &unconditional))
      {
        bits[slot] = (unconditional - conditional) / (double)scored;
        ++nscored;
      }
    }
    /**
     * All or nothing: lift is signed, so there is no value left to mark an
     * unscorable candidate with, and a partial ranking would mix two
     * incomparable orderings anyway.
     */
    if (nscored == ncandidates) result = EXIT_SUCCESS;
  }
  return result;
}


/**
 * Pick the continuation the model finds most likely: the index of the candidate
 * with the SMALLEST -log2 P(candidate | context).
 *
 * Note this is the conditional length, not the lift converse_hier_rescore uses,
 * and the difference is not an inconsistency. There the candidates were competing
 * answers and predictability was a confound to cancel; here they are competing
 * continuations of one fixed context, which is the question a language model is
 * actually built to answer, so its own preference is the signal. Returns -1 when
 * no candidate could be scored.
 */
/**
 * Conditional code length of the first score_length bytes of suffix[] given
 * prefix[] as history (0 = whole suffix), in bits per scored byte.
 *
 * This is the seam-fluency measure: with a short window it asks only "how
 * surprising is the text immediately after this join", which is what makes it a
 * judgement about the junction rather than about either side's content. Note the
 * inversion relative to converse_hier_rescore - there, ranking by conditional
 * length was wrong because it ranked by how ordinary a candidate is; here that is
 * exactly the quantity wanted.
 */
int converse_hier_seam_bits(const converse_hier_t* model,
  const char* prefix, int prefix_length, const char* suffix,
  int suffix_length, int score_length, double* bits)
{
  int result = EXIT_FAILURE;
  if (NULL != bits && 0 < suffix_length) {
    const int scored = (0 < score_length && score_length < suffix_length)
      ? score_length : suffix_length;
    double total = 0.0;
    if (EXIT_SUCCESS == hier_score_conditional(model, prefix, prefix_length,
      suffix, suffix_length, score_length, &total))
    {
      *bits = total / (double)scored;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


int converse_hier_choose(const converse_hier_t* model,
  const char* context, int context_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates)
{
  int result = -1;
  if (NULL != model && 0 != model->ready && NULL != candidates
    && NULL != candidate_lengths && 0 < ncandidates)
  {
    double best = 0.0;
    int slot;
    for (slot = 0; slot < ncandidates; ++slot) {
      double bits = 0.0;
      if (NULL != candidates[slot] && 0 < candidate_lengths[slot]
        && EXIT_SUCCESS == hier_score_conditional(model, context,
          context_length, candidates[slot], candidate_lengths[slot], 0, &bits)
        && (result < 0 || bits < best))
      {
        best = bits;
        result = slot;
      }
    }
  }
  return result;
}


converse_hier_t* converse_hier_build(const libxs_registry_t* corpus,
  int holdout, long corpus_size, int maxorder)
{
  converse_hier_t* result = NULL;
  converse_hier_t* model = (converse_hier_t*)calloc(1, sizeof(*model));
  LIBXS_UNUSED(corpus_key_from_fprint);
  if (NULL != model) {
    const char* env = getenv("CONVERSE_HIER_MINCOUNT");
    const char* clock_env = getenv("CONVERSE_HIER_CLOCK_ORDER");
    const char* decay_env = getenv("CONVERSE_HIER_STATE_DECAY");
    const char* stride_env = getenv("CONVERSE_HIER_TOP_STRIDE");
    const char* expert_env = getenv("CONVERSE_HIER_EXPERT_ORDER");
    const char* rate_env = getenv("CONVERSE_HIER_EXPERT_RATE");
    const char* share_env = getenv("CONVERSE_HIER_EXPERT_SHARE");
    corpus_entry_t scratch;
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    model->mincount = 2;
    if (NULL != env && '\0' != *env) {
      const int parsed = atoi(env);
      if (parsed > 0) model->mincount = parsed;
    }
    model->maxorder = maxorder;
    model->clock_order = 2;
    model->recurrent_decay = 0.875;
    model->top_stride = 40;
    model->expert_order = LIBXS_NGRAM_ORDER_MAX;
    model->expert_rate = 0.15;
    model->expert_share = 0.005;
    if (NULL != clock_env && '\0' != *clock_env) {
      const int parsed = atoi(clock_env);
      if (parsed >= 1 && parsed <= LIBXS_NGRAM_ORDER_MAX) {
        model->clock_order = parsed;
      }
    }
    if (NULL != decay_env && '\0' != *decay_env) {
      const double parsed = atof(decay_env);
      if (parsed >= 0.0 && parsed < 1.0) model->recurrent_decay = parsed;
    }
    if (NULL != stride_env && '\0' != *stride_env) {
      const int parsed = atoi(stride_env);
      if (parsed > 0) model->top_stride = parsed;
    }
    if (NULL != expert_env && '\0' != *expert_env) {
      const int parsed = atoi(expert_env);
      if (parsed >= 0 && parsed <= HIER_PPM_ORDER_MAX) {
        model->expert_order = parsed;
      }
    }
    if (NULL != rate_env && '\0' != *rate_env) {
      const double parsed = atof(rate_env);
      if (parsed > 0.0 && parsed <= 1.0) model->expert_rate = parsed;
    }
    if (NULL != share_env && '\0' != *share_env) {
      const double parsed = atof(share_env);
      if (parsed >= 0.0 && parsed < 1.0) model->expert_share = parsed;
    }
    model->symbols = libxs_registry_create();
    model->syllables = libxs_registry_create();
    model->syllable_payloads = libxs_registry_create();
    model->word_tokenizer = libxs_tokenizer_create(
      LIBXS_TOKEN_GRANULARITY_WORD);
    model->syllable_tokenizer = libxs_tokenizer_create(
      LIBXS_TOKEN_GRANULARITY_SYLLABLE);
    if (NULL != corpus && NULL != model->symbols && NULL != model->syllables
      && NULL != model->syllable_payloads
      && NULL != model->word_tokenizer && NULL != model->syllable_tokenizer)
    {
      value = corpus_iterx_begin(corpus, &key, &cursor);
      while (NULL != value) {
        const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
        if (0 == hier_is_test(index, holdout, corpus_size)) {
          hier_count_text(model, entry->text, entry->text_len);
        }
        ++index;
        value = corpus_iterx_next(corpus, &key, &cursor);
      }
      model->word_vocab = hier_symbol_assign(model->symbols,
        model->mincount, HIER_SYMBOL_FIRST);
      model->syllable_vocab = hier_symbol_assign(model->syllables,
        model->mincount, HIER_SYLLABLE_FIRST);
      hier_symbol_assign(model->syllable_payloads, model->mincount,
        HIER_SYLLABLE_FIRST);
      if (EXIT_SUCCESS == libxs_ngram_create(&model->word_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->syllable_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->byte_model, maxorder)
        && EXIT_SUCCESS == libxs_ngram_create(&model->stream_byte_model,
          model->clock_order)
        && EXIT_SUCCESS == libxs_ngram_create(&model->clock_byte_model,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->stream_ppm,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->clock_ppm,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->recurrent_ppm,
          HIER_RECURRENT_ORDER)
        && EXIT_SUCCESS == hier_ppm_create(&model->adaptive_ppm,
          model->clock_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->expert_ppm,
          model->expert_order)
        && EXIT_SUCCESS == hier_ppm_create(&model->word_clock_ppm, 2)
        && EXIT_SUCCESS == hier_ppm_create(&model->syllable_clock_ppm, 2)
        && EXIT_SUCCESS == hier_ppm_create(&model->syllable_role_ppm, 2))
      {
        key = NULL;
        cursor = 0;
        index = 0;
        value = corpus_iterx_begin(corpus, &key, &cursor);
        while (NULL != value) {
          const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
          if (0 == hier_is_test(index, holdout, corpus_size)) {
            hier_train_text(model, entry->text, entry->text_len);
            hier_train_clock_text(model, entry->text, entry->text_len);
          }
          ++index;
          value = corpus_iterx_next(corpus, &key, &cursor);
        }
        libxs_ngram_finalize(&model->word_model, model->word_vocab);
        libxs_ngram_finalize(&model->syllable_model, model->syllable_vocab);
        libxs_ngram_finalize(&model->byte_model, HIER_BYTE_VOCAB);
        libxs_ngram_finalize(&model->stream_byte_model, 256);
        libxs_ngram_finalize(&model->clock_byte_model, 256);
        hier_ppm_finalize(&model->stream_ppm);
        hier_ppm_finalize(&model->clock_ppm);
        hier_ppm_finalize(&model->recurrent_ppm);
        hier_ppm_finalize(&model->expert_ppm);
        hier_ppm_finalize(&model->word_clock_ppm);
        hier_ppm_finalize(&model->syllable_clock_ppm);
        hier_ppm_finalize(&model->syllable_role_ppm);
        if (EXIT_SUCCESS == hier_ppm_check(&model->stream_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->clock_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->recurrent_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->expert_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->word_clock_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->syllable_clock_ppm)
          && EXIT_SUCCESS == hier_ppm_check(&model->syllable_role_ppm))
        {
          model->ready = 1;
          fprintf(stderr, "hierarchy: word-vocab=%u syllable-vocab=%u"
            " mincount=%d order=%d clock-order=%d state-decay=%.3f"
            " experts=0..%d\n",
            model->word_vocab,
            model->syllable_vocab, model->mincount, model->maxorder,
            model->clock_order, model->recurrent_decay,
            model->expert_order);
          result = model;
        }
      }
    }
  }
  if (NULL == result) converse_hier_destroy(model);
  return result;
}


void converse_hier_destroy(converse_hier_t* model)
{
  if (NULL != model) {
    libxs_ngram_destroy(&model->word_model);
    libxs_ngram_destroy(&model->syllable_model);
    libxs_ngram_destroy(&model->byte_model);
    libxs_ngram_destroy(&model->stream_byte_model);
    libxs_ngram_destroy(&model->clock_byte_model);
    hier_ppm_destroy(&model->stream_ppm);
    hier_ppm_destroy(&model->clock_ppm);
    hier_ppm_destroy(&model->recurrent_ppm);
    hier_ppm_destroy(&model->adaptive_ppm);
    hier_ppm_destroy(&model->expert_ppm);
    hier_ppm_destroy(&model->word_clock_ppm);
    hier_ppm_destroy(&model->syllable_clock_ppm);
    hier_ppm_destroy(&model->syllable_role_ppm);
    libxs_registry_destroy(model->symbols);
    libxs_registry_destroy(model->syllables);
    libxs_registry_destroy(model->syllable_payloads);
    libxs_tokenizer_destroy(model->word_tokenizer);
    libxs_tokenizer_destroy(model->syllable_tokenizer);
    free(model);
  }
}


int converse_hier_eval(converse_hier_t* model,
  const libxs_registry_t* corpus, int holdout, long corpus_size,
  const char* label)
{
  int result = EXIT_FAILURE;
  if (NULL != model && 0 != model->ready && NULL != corpus) {
    hier_eval_t evaluation;
    hier_clock_eval_t clock_evaluation;
    corpus_entry_t scratch;
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value;
    memset(&evaluation, 0, sizeof(evaluation));
    memset(&clock_evaluation, 0, sizeof(clock_evaluation));
    hier_expert_weight_init(model, clock_evaluation.expert_weight);
    { const double uniform = 1.0 / (double)(model->clock_order + 1);
      int expert_order;
      for (expert_order = 0; expert_order <= model->clock_order;
        ++expert_order)
      {
        clock_evaluation.adaptive_expert_weight[expert_order] = uniform;
      }
    }
    value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
      if (0 == holdout || 0 != hier_is_test(index, holdout, corpus_size)) {
        hier_score_text(model, entry->text, entry->text_len, &evaluation);
        hier_score_clock_text(model, entry->text, entry->text_len, 0.5,
          &clock_evaluation);
      }
      ++index;
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
    if (evaluation.bytes > 0.0 && evaluation.ntokens > 0) {
      fprintf(stdout, "predict-hier[%s%s]: top1=%.1f%% n=%ld bpc=%.3f\n",
        (NULL != label) ? label : "hier", (holdout > 0) ? ":heldout" : "",
        100.0 * (double)evaluation.ntop1 / (double)evaluation.ntokens,
        evaluation.ntokens, evaluation.bits / evaluation.bytes);
      fprintf(stderr, "  hierarchy bits: top=%.3f syllable=%.3f byte=%.3f"
        " | escapes: text=%.1f%% native=%.1f%% syllable=%ld\n",
        evaluation.top_bits / evaluation.bytes,
        evaluation.syllable_bits / evaluation.bytes,
        evaluation.byte_bits / evaluation.bytes,
        100.0 * (double)evaluation.ntext_escape / (double)evaluation.ntokens,
        100.0 * (double)evaluation.nnative_escape / (double)evaluation.ntokens,
        evaluation.nsyllable_escape);
      fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
        " (bpc=%.3f) | novel %.1f%% (bpc=%.3f)\n",
        100.0 * (double)evaluation.ndeep / (double)evaluation.ntokens,
        (evaluation.deep_bytes > 0.0)
          ? evaluation.deep_bits / evaluation.deep_bytes : 0.0,
        100.0 * (double)evaluation.nshallow / (double)evaluation.ntokens,
        (evaluation.shallow_bytes > 0.0)
          ? evaluation.shallow_bits / evaluation.shallow_bytes : 0.0);
      if (clock_evaluation.nbytes > 0) {
        fprintf(stdout, "predict-clock[%s]: raw-top1=%.1f%%"
          " context-top1=%.1f%% n=%ld raw-bpc=%.3f context-bpc=%.3f"
          " mix-bpc=%.3f\n", (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nraw_top1
            / (double)clock_evaluation.nbytes,
          100.0 * (double)clock_evaluation.ncontext_top1
            / (double)clock_evaluation.nbytes,
          clock_evaluation.nbytes,
          clock_evaluation.raw_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.context_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.mix_bits / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-ppm[%s]: top1=%.1f%% top3=%.1f%% n=%ld"
          " (stride=%d) raw-bpc=%.3f context-bpc=%.3f mix-bpc=%.3f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nppm_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nppm_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.raw_ppm_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.context_ppm_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.ppm_mix_bits / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-recurrent[%s]: n=%ld state-bpc=%.3f"
          " mix-bpc=%.3f\n", (NULL != label) ? label : "metatoken",
          clock_evaluation.nbytes,
          clock_evaluation.recurrent_bits / (double)clock_evaluation.nbytes,
          clock_evaluation.recurrent_mix_bits
            / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-adaptive[%s]: top1=%.1f%% top3=%.1f%% n=%ld"
          " (stride=%d) frozen-interp-bpc=%.3f adaptive-bpc=%.3f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nadaptive_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nadaptive_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.frozen_interp_bits
            / (double)clock_evaluation.nbytes,
          clock_evaluation.adaptive_bits / (double)clock_evaluation.nbytes);
        fprintf(stdout, "predict-experts[%s]: top1=%.1f%% top3=%.1f%% n=%ld"
          " (stride=%d) bpc=%.3f rate=%.3f share=%.4f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nexpert_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nexpert_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.expert_mix_bits / (double)clock_evaluation.nbytes,
          model->expert_rate, model->expert_share);
        fprintf(stdout, "  expert attested split (order %d):"
          " verbatim %.1f%% (bpc=%.3f) | novel %.1f%% (bpc=%.3f)\n",
          model->expert_order,
          100.0 * (double)clock_evaluation.nexpert_attested
            / (double)clock_evaluation.nbytes,
          (0 < clock_evaluation.nexpert_attested)
            ? clock_evaluation.expert_attested_bits
              / (double)clock_evaluation.nexpert_attested : 0.0,
          100.0 * (double)clock_evaluation.nexpert_novel
            / (double)clock_evaluation.nbytes,
          (0 < clock_evaluation.nexpert_novel)
            ? clock_evaluation.expert_novel_bits
              / (double)clock_evaluation.nexpert_novel : 0.0);
        fprintf(stderr, "  expert orders:");
        { int expert_order;
          for (expert_order = 0; expert_order <= model->expert_order;
            ++expert_order)
          {
            fprintf(stderr, " %d=%.3f/%.3f", expert_order,
              clock_evaluation.expert_bits[expert_order]
                / (double)clock_evaluation.nbytes,
              clock_evaluation.expert_weight[expert_order]);
          }
        }
        fprintf(stderr, " word=%.3f/%.3f syllable=%.3f/%.3f"
          " syllable-role=%.3f/%.3f",
          clock_evaluation.expert_bits[HIER_EXPERT_WORD]
            / (double)clock_evaluation.nbytes,
          clock_evaluation.expert_weight[HIER_EXPERT_WORD],
          clock_evaluation.expert_bits[HIER_EXPERT_SYLLABLE]
            / (double)clock_evaluation.nbytes,
          clock_evaluation.expert_weight[HIER_EXPERT_SYLLABLE],
          clock_evaluation.expert_bits[HIER_EXPERT_SYLLABLE_ROLE]
            / (double)clock_evaluation.nbytes,
          clock_evaluation.expert_weight[HIER_EXPERT_SYLLABLE_ROLE]);
        fprintf(stderr, "\n");
        fprintf(stdout, "predict-adaptive-experts[%s]: top1=%.1f%%"
          " top3=%.1f%% n=%ld (stride=%d) bpc=%.3f\n",
          (NULL != label) ? label : "metatoken",
          100.0 * (double)clock_evaluation.nadaptive_expert_top1
            / (double)clock_evaluation.nppm,
          100.0 * (double)clock_evaluation.nadaptive_expert_top3
            / (double)clock_evaluation.nppm,
          clock_evaluation.nppm, model->top_stride,
          clock_evaluation.adaptive_expert_mix_bits
            / (double)clock_evaluation.nbytes);
        fprintf(stderr, "  adaptive expert orders:");
        { int expert_order;
          for (expert_order = 0; expert_order <= model->clock_order;
            ++expert_order)
          {
            fprintf(stderr, " %d=%.3f/%.3f", expert_order,
              clock_evaluation.adaptive_expert_bits[expert_order]
                / (double)clock_evaluation.nbytes,
              clock_evaluation.adaptive_expert_weight[expert_order]);
          }
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "  clock attested split: verbatim %.1f%%"
          " (bpc=%.3f) | novel %.1f%% (bpc=%.3f)\n",
          100.0 * (double)clock_evaluation.ndeep
            / (double)clock_evaluation.nbytes,
          (clock_evaluation.ndeep > 0)
            ? clock_evaluation.deep_bits / (double)clock_evaluation.ndeep
            : 0.0,
          100.0 * (double)clock_evaluation.nshallow
            / (double)clock_evaluation.nbytes,
          (clock_evaluation.nshallow > 0)
            ? clock_evaluation.shallow_bits / (double)clock_evaluation.nshallow
            : 0.0);
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}
