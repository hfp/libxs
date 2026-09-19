#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mix.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_timer.h>

#include "converse_lm.h"
#include "converse_hier.h"

#define NGRAM_SUCC_MAX LIBXS_NGRAM_SUCC_MAX
#define NGRAM_TOPK 3

/**
 * Expert-bank slots. Slots 1..NGRAM_ORDER_MAX are the fixed-order experts and
 * keep their order as their index, so the array stays order-indexed where that
 * is the natural key; slots beyond that are experts of a different KIND. An
 * extra slot has a FIXED index rather than one relative to the running
 * maxorder: a slot whose meaning shifted with the order would carry a weight
 * that was learned for a different expert.
 */
#define NGRAM_BANK_SKIP (NGRAM_ORDER_MAX + 1)
#define NGRAM_BANK_PRIOR (NGRAM_ORDER_MAX + 2)
#define NGRAM_BANK_PREDICT (NGRAM_ORDER_MAX + 3)
#define NGRAM_BANK_EMB (NGRAM_ORDER_MAX + 4)
#define NGRAM_BANK_LAST NGRAM_BANK_EMB
#define NGRAM_BANK_MAX (NGRAM_BANK_LAST + 1)
/** Ratio substituted for an expert that gave the target no mass at all. */
#define NGRAM_BANK_RELMIN 1e-4
/**
 * Escape-bank entropy above which the predict slot's estimate is still settling.
 * log2(13) == 3.700 is the uninformative prior; a figure taken while the bank is
 * near it is not trustworthy, so the run reports the mean rather than assuming
 * convergence.
 */
#define NGRAM_PREDICT_ENTROPY_MAX 3.5

#if !defined(TOKEN_PREDICT_TRAIN_MAX)
# define TOKEN_PREDICT_TRAIN_MAX 40000
#endif
#define TOKEN_PREDICT_EVAL_STRIDE 40

#define KNNLM_K 24
#define KNNLM_VOTE_MAX 4
#define KNNLM_ANN_DIMS 8
#define KNNLM_ANN_BITS 8
#define KNNLM_ANN_WINDOW 512

#define RERANK_INPUTS 9
#define RERANK_RELIABILITY 32.0
#define RERANK_LIFT_MAX 8.0


typedef libxs_ngram_succ_t ngram_succ_t;
typedef libxs_ngram_entry_t ngram_entry_t;

/**
 * One expert-bank slot at one position. ACTIVE is not "probability zero":
 * an expert that cannot speak here (the skip tier without a seen pair, a
 * retrieval store without neighbours) must be left out of the pool and out of
 * the weight update entirely. Folding abstention in as zero drives the slot's
 * weight to exactly zero through the multiplicative update, and the uniform
 * recovery share only reaches slots that still hold mass - so one abstention
 * would retire the expert permanently.
 */
typedef struct ngram_expert_t {
  double probability;
  int active;
} ngram_expert_t;

/**
 * Coverage probe for slot abstraction (does NOT predict anything).
 *
 * The question it answers: on positions whose exact context did NOT recur in
 * training - the only positions that measure generalization - would a context
 * with ONE content word replaced by a typed hole have recurred? If yes for a
 * large share, then patterns-with-holes are a funded mechanism: they let a novel
 * context match training evidence, which is the type-level sharing that a
 * transformer gets from weights and that exact-context lookup cannot express.
 * If coverage is thin, abstraction cannot reach these positions either and the
 * mechanism is retired for one run instead of one session.
 *
 * A token is a SLOT CANDIDATE when its corpus frequency is at or below
 * CONVERSE_SLOT_MAXFREQ (default 20): frequent tokens are function words that
 * carry the pattern, rare ones are the content the pattern is about. This is
 * deliberately not keyed on the relation rules, so the result is not circular
 * with hand-written vocabulary.
 *
 * "Novel" here matches the definition already used by ngram_eval's attested
 * split: the full-order CONTEXT did not recur, independently of the successor.
 * Coverage, by contrast, requires the abstracted context to have been seen with
 * the SAME successor - the abstraction has to predict the actual next token, not
 * merely exist. The asymmetry is deliberate: it makes coverage a claim about
 * useful evidence rather than about pattern frequency.
 */
#define SLOT_HOLE_ID 0xFFFFFFFEu


typedef struct slot_key_t {
  unsigned int context[NGRAM_ORDER_MAX];
  unsigned int next;
  int order;
} slot_key_t;

/** Same as slot_key_t without the successor: "was this context ever seen?" */
typedef struct slot_ctx_key_t {
  unsigned int context[NGRAM_ORDER_MAX];
  int order;
} slot_ctx_key_t;

typedef struct slot_probe_t {
  libxs_registry_t* patterns;
  libxs_registry_t* contexts;
  libxs_registry_t* freq;
  libxs_lexicon_t* lexicon;
  int order;
  int building;
  long nnovel;
  long ncovered;
  long nexact;
  long nslotless;
} slot_probe_t;


/**
 * Predict-slot diagnostics. The escape bank's entropy says whether its estimate
 * has settled; a figure taken near the uninformative prior is not trustworthy,
 * so the mean is reported rather than convergence being assumed.
 */
static double predict_entropy_sum = 0.0;
static long predict_nscored = 0;
static int predict_support_last = 0;
/**
 * Does the slot's kNN vote contribute anything, or is its value entirely the
 * learned escape over the frequency prior? The inputs are RAW TOKEN IDS, so
 * Euclidean distance between them is meaningless and the vote may be reading
 * noise. info->attested says whether the observed token received any local
 * evidence at all, which splits the slot's own bits into the part the vote
 * touched and the part it did not.
 */
static long predict_att_n = 0, predict_unatt_n = 0;
static double predict_att_bits = 0.0, predict_unatt_bits = 0.0;

static const libxs_predict_t* knnlm_cache_model = NULL;
static double* knnlm_cache_in = NULL;
static unsigned int* knnlm_cache_next = NULL;
static int knnlm_cache_size = 0;
static double* knnlm_dyn_in = NULL;
static unsigned int* knnlm_dyn_next = NULL;
static int knnlm_dyn_size = 0;
static int knnlm_dyn_cap = 0;
static uint64_t* knnlm_ann_code = NULL;
static int* knnlm_ann_order = NULL;
static int knnlm_ann_size = 0;

/**
 * Diagonal learned projection for retrieval (CONVERSE_KNNLM_PROJ=1, default off
 * and bit-exact). Retrieval wants contexts that share a next token close and
 * contexts with different next tokens far, which is Fisher's criterion: weight
 * each dimension by sqrt(between-class / within-class scatter) over next-token
 * classes. Regression onto token ids would be meaningless - ids carry no metric
 * - so the objective is discriminative, not least-squares. Fitted from the
 * TRAINING entries only (the datastore already excludes held-out text) and
 * applied to BOTH the query and the stored keys, since a distance is only
 * meaningful when both sides live in one space.
 *
 * Diagonal first by design: it tests whether reweighting dimensions by
 * discriminative power helps at all, before paying for the symmetric generalized
 * eigenproblem a full matrix needs (no such solver exists in libxs_math). A
 * rotation can only add over a diagonal if the informative directions are
 * misaligned with the axes, and the PPMI factorization already delivers
 * variance-ordered axes.
 */
static double token_proj[2 * TOKEN_EMB_DIM];
static int token_proj_ready = 0;


static const ngram_entry_t* ngramk_lookup(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int n);
static double ngram_syllable_cost(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const char* word, int wlen, unsigned long mask,
  const unsigned int hist_in[], int hlen_in, int maxorder, int* out_npiece);
static void ngram_syllable_oracle(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_registry_t* corpus, int holdout, int maxorder);
static int ngram_oracle_probe(void);
static int ngram_select_order(void);
static int ngram_bank_probe(void);
static double ngram_bank_rate(void);
static double ngram_bank_share(void);
static void ngram_bank_view(libxs_mix_t* mix, double weight[],
  const ngram_expert_t expert[], double prob[], int active[],
  double rate, double share);
static void ngram_bank_update(double weight[], const ngram_expert_t expert[],
  double mixture, double rate, double share);
static unsigned int ngram_bank_slots(void);
static int ngram_bank_enabled(unsigned int slots, int slot);
static const char* ngram_bank_slotname(int slot);
static void ngram_bank_experts(const unsigned int hist[], int hlen,
  int maxorder, unsigned int cur, unsigned int slots,
  const libxs_predict_t* store, void* context, int use_emb, int vocabulary,
  ngram_expert_t expert[]);
static int ngram_bank_geometric(void);
static int ngram_bank_support(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int ids[], int max);
static double ngram_bank_pool_geo(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int cur,
  const double weight[], const ngram_expert_t expert[]);
static double ngram_bank_pool(const double weight[],
  const ngram_expert_t expert[]);
static int predict_eval_stride(void);
static int ngram_topk(const ngram_entry_t* entry, unsigned int out_ids[],
  int k);
static double ngram_unigram_prior(unsigned int id);
static double ngram_pair_relfreq(libxs_registry_t* model, unsigned int ctx_a,
  unsigned int ctx_b, unsigned int succ_id);
static int ngram_emb_ctx(void);
static int token_input_vector(unsigned int prev2, unsigned int prev1,
  int use_emb, double inputs[]);
static int knnlm_ctxlen(void);
static double knnlm_decay(void);
static int knnlm_weights(double out[], int max);
static void knnlm_ctx_vector(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double inputs[]);
static double knnlm_cosine(const double* a, const double* b, int n);
static void knnlm_order_probe(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double* sum_inner, double* sum_cross, long* n_inner,
  long* n_cross);
static const ngram_entry_t* ngram_lookup(libxs_registry_t* model,
  unsigned int ctx_a, unsigned int ctx_b);
static int ngram_predict(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int out_ids[], int k);
static void ngram_last_context(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  unsigned int* prev2, unsigned int* prev1);
static int ngram_gen_full(void);
static int ngram_gen_embrank(void);
static int ngram_gen_embcand(void);
static int ngram_gen_join(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int n, char* out, size_t out_size);
static int ngram_gen_bank_rank(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int ids[], int n,
  unsigned int slots, const libxs_predict_t* store, void* context,
  int use_emb, int vocabulary, double weight[]);
static void gen_bucket_report(const char* label, long n, long top1,
  long inlist, long abst, double rr);
static void gen_embrank_report(const char* label, long n, long top1,
  long top10, double rr);
static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind,
  const libxs_predict_t* store, int use_emb);
static int slot_maxfreq(void);
static void slot_key_build(slot_key_t* key, const unsigned int hist[],
  int hlen, int order, unsigned int next, int hole);
static void slot_scan(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout, int want_test,
  void (*visit)(const unsigned int*, int, unsigned int, void*), void* udata);
static long slot_freq_of(const slot_probe_t* probe, unsigned int id);
static void slot_count_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata);
static void slot_train_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata);
static void slot_test_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata);
static void slot_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static int ngram_bank_frozen(void);
static int ngram_bank_warmup(libxs_predict_t* store, int vocabulary);
static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind, const libxs_predict_t* store,
  int use_emb);
static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout,
  int ctxlen);
static unsigned int token_predict_next(const libxs_predict_t* model,
  unsigned int prev2, unsigned int prev1, int use_emb);
static int token_predict_eval(const libxs_predict_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int use_emb, int holdout,
  const char* kind);
static void token_complete(const libxs_predict_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int use_emb, const char* text, int text_len);
static int ngram_candidates(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int ids[], double relfreq[],
  int provenance[], int* ctx_total, int k);
static int token_is_stop(libxs_lexicon_t* lexicon, unsigned int id);
static void rerank_features(libxs_registry_t* ngram, unsigned int prev1,
  unsigned int id, double relfreq, int rank, int provenance, int ctx_total,
  double ctx_top, libxs_lexicon_t* lexicon, double inputs[RERANK_INPUTS]);
static libxs_predict_t* rerank_build(const libxs_registry_t* corpus,
  libxs_registry_t* ngram, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order,
  const answer_predict_profile_t* profile, int holdout);
static int rerank_topk(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  unsigned int prev2, unsigned int prev1, int order, unsigned int out_ids[],
  int k);
static int rerank_eval(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind);
static void rerank_complete(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static void knnlm_cache_free(void);
static int knnlm_ann_mode(void);
static uint64_t knnlm_ann_encode(const double* in);
static int knnlm_ann_cmp(const void* a, const void* b);
static void knnlm_ann_build(void);
static void knnlm_ann_consider(const double* in, int idx,
  unsigned int near_next[], double near_dist[], int* nnear);
static void knnlm_ann_scan(const double* in, unsigned int near_next[],
  double near_dist[], int* nnear);
static void knnlm_dyn_reset(void);
static void knnlm_dyn_insert(const unsigned int hist[], int hlen, int ctxlen,
  unsigned int next);
static int knnlm_heads(void);
static void knnlm_scan_head(const double* in, const double* cin,
  const unsigned int* cnext, int count, int dim_begin, int dim_end,
  unsigned int near_next[], double near_dist[], int* nnear);
static int token_proj_mode(void);
static void token_proj_apply(double* vec);
static void token_proj_build(const libxs_predict_t* store);
static void knnlm_cache_build(const libxs_predict_t* store);
static double knnlm_temp(void);
static int knnlm_control(void);
static int knnlm_vote_control(int mode, unsigned int position,
  unsigned int vote_ids[], double vote_p[], int maxvote);
static int knnlm_vote(const libxs_predict_t* store, const unsigned int hist[],
  int hlen, int ctxlen, unsigned int vote_ids[], double vote_p[], int maxvote);
static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  const unsigned int hist[], int hlen, int ctxlen, int order,
  unsigned int out_ids[], int k, unsigned int target, double* target_prob);
static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind);
static void knnlm_complete(libxs_registry_t* ngram,
  const libxs_predict_t* store, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);


static const ngram_entry_t* ngramk_lookup(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int n)
{
  LIBXS_UNUSED(model);
  return libxs_ngram_lookup(converse_ngram_handle(), hist, hlen, n);
}

/**
 * Cost in bits of emitting one word as the pieces implied by a cut mask, scored
 * against the trained store. Bit i of mask means "cut before letter i+1".
 *
 * THE TRAP THIS AVOIDS: a candidate split invents pieces the training text never
 * contained, and an unknown piece has NO id. Scoring only the pieces that happen
 * to be known would make exotic splits look free - fewer scored positions, less
 * accumulated cost - and the oracle would "win" by producing garbage. So a
 * piece with no id is charged the unigram floor rather than skipped, which is the
 * honest price of a unit the model cannot represent.
 *
 * Returns bits, and writes the piece count. Scoring is done with create=0: the
 * oracle must not grow the lexicon it is measuring against.
 */
static double ngram_syllable_cost(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const char* word, int wlen, unsigned long mask,
  const unsigned int hist_in[], int hlen_in, int maxorder, int* out_npiece)
{
  const double inv_log2 = 1.0 / log(2.0);
  unsigned int hist[NGRAM_ORDER_MAX];
  double bits = 0.0;
  int hlen = hlen_in, npiece = 0, at = 0, i;
  for (i = 0; i < hlen && i < NGRAM_ORDER_MAX; ++i) hist[i] = hist_in[i];
  for (i = 1; i <= wlen; ++i) {
    /* a cut before letter i, or the end of the word, closes a piece */
    if (wlen == i || 0 != (mask & (1UL << (i - 1)))) {
      char buf[LIBXS_LEXEME_MAXBYTES + 1];
      int plen = i - at;
      if (0 < plen && plen <= (int)sizeof(buf) - 1) {
        unsigned int id;
        memcpy(buf, word + at, (size_t)plen);
        id = libxs_lexicon_id(lexicon, buf, plen, LIBXS_LEXEME_WORD, 0);
        if (0 != id) {
          const double p = ngramk_prob(model, hist, hlen, maxorder, id);
          bits -= log((p > 0.0) ? p : 1e-12) * inv_log2;
          ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, id);
        }
        else {
          /* unrepresentable piece: charge the floor, do not extend history */
          bits -= log(1e-12) * inv_log2;
        }
        ++npiece;
      }
      at = i;
    }
  }
  if (NULL != out_npiece) *out_npiece = npiece;
  return bits;
}

/**
 * Per-word split ORACLE: the cheapest cut set over all candidates, which bounds
 * what ANY cut rule - hand-written or learned - can achieve on this corpus.
 *
 * The decision rule was pre-committed before this was written (and is the reason
 * to write it): if the oracle is not materially better than the maximal-onset
 * repair, a learned splitter CANNOT PAY and the line is dropped. Exactly how
 * CONVERSE_NGRAM_ORACLE bounded order selection at 2.018 and closed that route.
 *
 * Words longer than ORACLE_MAXLEN are left to the heuristic: the candidate space
 * is 2^(n-1), so an exhaustive oracle is only honest where it is affordable, and
 * silently sampling a subset would report a bound that is not one.
 */
static void ngram_syllable_oracle(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_registry_t* corpus, int holdout, int maxorder)
{
  enum { ORACLE_MAXLEN = 12 };
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  double heur_bits = 0.0, oracle_bits = 0.0, single_bits = 0.0;
  long nword = 0, nskipped = 0, nheur_optimal = 0;
  long index = 0;
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    const int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    if (0 != is_test && entry->text_len > 0) {
      int pos = 0;
      while (pos < entry->text_len) {
        const char* text = entry->text;
        if (0 == ngram_is_wordchar((unsigned char)text[pos])) { ++pos; continue; }
        { int wlen = 0;
          while (pos + wlen < entry->text_len
            && 0 != ngram_is_wordchar((unsigned char)text[pos + wlen])) ++wlen;
          if (1 < wlen && ORACLE_MAXLEN >= wlen) {
            int begin[LIBXS_LEXEME_MAXBYTES], plen[LIBXS_LEXEME_MAXBYTES];
            const int np = ngram_syllable_split(text + pos, wlen, begin, plen,
              LIBXS_LEXEME_MAXBYTES);
            const unsigned long ncand = 1UL << (wlen - 1);
            unsigned long mask, hmask = 0, best_mask = 0;
            double best = -1.0, hcost;
            int k;
            /* the heuristic's own cut set, as a mask */
            for (k = 1; k < np; ++k) hmask |= 1UL << (begin[k] - 1);
            hcost = ngram_syllable_cost(model, lexicon, text + pos, wlen,
              hmask, NULL, 0, maxorder, NULL);
            for (mask = 0; mask < ncand; ++mask) {
              const double c = ngram_syllable_cost(model, lexicon, text + pos,
                wlen, mask, NULL, 0, maxorder, NULL);
              if (best < 0.0 || c < best) { best = c; best_mask = mask; }
            }
            heur_bits += hcost;
            oracle_bits += best;
            single_bits += ngram_syllable_cost(model, lexicon, text + pos,
              wlen, 0, NULL, 0, maxorder, NULL);
            if (best_mask == hmask) ++nheur_optimal;
            ++nword;
          }
          else if (1 < wlen) ++nskipped;
          pos += wlen;
        }
      }
    }
    LIBXS_UNUSED(rules); LIBXS_UNUSED(nrules);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  if (0 < nword) {
    fprintf(stderr, "syllable oracle (words 2..%d letters, n=%ld, skipped %ld"
      " longer):\n  heuristic=%.3f bits/word | ORACLE=%.3f | whole-word=%.3f"
      " | heuristic optimal for %.1f%% of words\n", (int)ORACLE_MAXLEN, nword,
      nskipped, heur_bits / (double)nword, oracle_bits / (double)nword,
      single_bits / (double)nword,
      100.0 * (double)nheur_optimal / (double)nword);
  }
}

/**
 * Report, per position, the cost of every fixed-order expert and of an oracle
 * that picks the cheapest. A measurement, not a mechanism: the oracle reads the
 * truth, so it is an upper bound on what any per-position order selection can
 * win over the single interpolated model.
 */
static int ngram_oracle_probe(void)
{
  const char* env = getenv("CONVERSE_NGRAM_ORACLE");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}

/**
 * Which fixed-order expert the achievable selector falls back to. The best
 * single order is unit-dependent (order 1 for whole words, order 2 for the
 * sub-word units), so a hardcoded 1 measures the rule on the wrong expert and
 * reports a loss where there is a gain.
 */
static int ngram_select_order(void)
{
  int result = 1;
  const char* env = getenv("CONVERSE_NGRAM_SELORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}

/**
 * Word-level expert bank: mix the fixed-order experts with causally updated
 * fixed-share weights instead of the count-derived backoff weights. The byte
 * side earned 0.5-0.8 bits cross-document this way, and the order-selection
 * probe showed the same headroom exists at word and sub-word units, so this is
 * the same mechanism at a coarser unit rather than a new one.
 *
 * The threshold rule proved the headroom is reachable from an observable
 * feature; this replaces the hard threshold with a learned per-position
 * weighting, which is what the byte side actually does.
 */
static int ngram_bank_probe(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


static double ngram_bank_rate(void)
{
  double result = 0.15;
  const char* env = getenv("CONVERSE_NGRAM_BANK_RATE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    /**
     * Zero is ACCEPTED: it means the weights never move from their uniform
     * prior, which is the control the learned weighting has to beat and the
     * only way to read the rate sweep's endpoint. Rejecting it silently
     * substituted the default, so a run labelled rate 0 was measured at 0.15
     * and looked like evidence that the rate does not matter.
     */
    if (v >= 0.0 && v <= 4.0) result = v;
  }
  return result;
}


static double ngram_bank_share(void)
{
  double result = 0.005;
  const char* env = getenv("CONVERSE_NGRAM_BANK_SHARE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    if (v >= 0.0 && v < 1.0) result = v;
  }
  return result;
}

/**
 * One causal fixed-share step over the expert slots. Multiplicative log-loss
 * update toward the experts that beat the mixture, then a share of the mass
 * redistributed uniformly so an expert that was wrong for a while can recover.
 * Only slots that are ENABLED (weight > 0) and ACTIVE (spoke at this position)
 * participate: an abstaining slot is neither updated nor renormalized, so it
 * carries its weight to the next position where it can speak. Restricting the
 * update to the slots that spoke is also what keeps the order experts bit-exact
 * to the pre-slot bank, where the loop simply stopped at min(maxorder, hlen).
 *
 * The ratio is floored only where it is EXACTLY zero, which happens when an
 * active expert assigns no mass at all to the observed target (the skip tier
 * matches a pair whose successor list omits it). pow(0, rate) is exactly zero,
 * so such a slot would lose all its mass in one step and the uniform recovery
 * term, which only reaches slots that still hold mass, could never revive it.
 * A merely tiny ratio needs no floor - it decays steeply but stays positive,
 * and clamping it would perturb the order-only weights that are otherwise
 * bit-exact to the pre-slot bank.
 */
/**
 * Spread the slot-indexed experts into the parallel arrays libxs_mix takes. Slot
 * 0 is unused here and holds weight zero, so the shared primitive skips it on
 * the same weight>0 test the local loop used.
 */
static void ngram_bank_view(libxs_mix_t* mix, double weight[],
  const ngram_expert_t expert[], double prob[], int active[],
  double rate, double share)
{
  int slot;
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) {
    prob[slot] = expert[slot].probability;
    active[slot] = (slot >= 1 && slot <= NGRAM_BANK_LAST)
      ? expert[slot].active : 0;
  }
  mix->weight = weight;
  mix->nslot = NGRAM_BANK_MAX;
  mix->rate = rate;
  mix->share = share;
  mix->relmin = NGRAM_BANK_RELMIN;
}


static void ngram_bank_update(double weight[], const ngram_expert_t expert[],
  double mixture, double rate, double share)
{
  libxs_mix_t mix;
  double prob[NGRAM_BANK_MAX];
  int active[NGRAM_BANK_MAX];
  ngram_bank_view(&mix, weight, expert, prob, active, rate, share);
  /* the caller's mixture, which carries the log-loss floor ngram_bank_pool
   * applies; recomputing it here would drop that floor */
  libxs_mix_update(&mix, prob, active, mixture);
}

/**
 * Which extra slots the bank carries beyond the fixed orders, as a mask over
 * slot indices. Empty by default: with no extra slot the bank is the order-only
 * pool, which is the control the kind-different experts have to beat. The knobs
 * are independent so one candidate can be measured without the others.
 */
static unsigned int ngram_bank_slots(void)
{
  unsigned int result = 0;
  const char* skip = getenv("CONVERSE_NGRAM_BANK_SKIP");
  const char* prior = getenv("CONVERSE_NGRAM_BANK_PRIOR");
  if (NULL != skip && '0' != skip[0] && '\0' != skip[0]) {
    result |= 1u << NGRAM_BANK_SKIP;
  }
  if (NULL != prior && '0' != prior[0] && '\0' != prior[0]) {
    result |= 1u << NGRAM_BANK_PRIOR;
  }
  { const char* prd = getenv("CONVERSE_NGRAM_BANK_PREDICT");
    if (NULL != prd && '0' != prd[0] && '\0' != prd[0]) {
      result |= 1u << NGRAM_BANK_PREDICT;
    }
  }
  { const char* emb = getenv("CONVERSE_NGRAM_BANK_EMB");
    if (NULL != emb && '0' != emb[0] && '\0' != emb[0]) {
      result |= 1u << NGRAM_BANK_EMB;
    }
  }
  return result;
}


static int ngram_bank_enabled(unsigned int slots, int slot)
{
  return (slot <= NGRAM_ORDER_MAX || 0 != (slots & (1u << slot))) ? 1 : 0;
}


static const char* ngram_bank_slotname(int slot)
{
  static char name[8];
  const char* result = name;
  if (NGRAM_BANK_SKIP == slot) result = "skip";
  else if (NGRAM_BANK_PRIOR == slot) result = "uni";
  else if (NGRAM_BANK_PREDICT == slot) result = "prd";
  else if (NGRAM_BANK_EMB == slot) result = "emb";
  else sprintf(name, "o%d", slot);
  return result;
}

/**
 * Fill the per-position expert set. The fixed-order experts are the SAME
 * interpolated model given only their last k words, so expert k is what a model
 * capped at order k would say.
 *
 * When the skip slot is enabled the order experts are asked for the SKIP-FREE
 * estimate: ngramk_prob folds the skip tier into every order at a fixed weight,
 * so leaving that in would double-count it and make the learned skip weight
 * meaningless. Without the slot they keep the folded estimate, which is what
 * holds every existing configuration bit-exact.
 *
 * An expert that cannot speak is left inactive rather than given probability
 * zero - see ngram_expert_t. Order expert k needs k words of history; the skip
 * slot needs three and a pair that was actually observed. The unigram slot is
 * TOTAL: it always speaks, which is the point of carrying it - the pool then
 * has a floor at every position, including those where counts have nothing.
 */
static void ngram_bank_experts(const unsigned int hist[], int hlen,
  int maxorder, unsigned int cur, unsigned int slots,
  const libxs_predict_t* store, void* context, int use_emb, int vocabulary,
  ngram_expert_t expert[])
{
  const int skip_slot = (0 != (slots & (1u << NGRAM_BANK_SKIP))) ? 1 : 0;
  int slot;
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) {
    expert[slot].probability = 0.0;
    expert[slot].active = 0;
  }
  for (slot = 1; slot <= maxorder && slot <= hlen; ++slot) {
    const unsigned int* sub = hist + (hlen - slot);
    expert[slot].probability = (0 != skip_slot)
      ? ngramk_prob_exact(sub, slot, cur)
      : ngramk_prob(NULL, sub, slot, slot, cur);
    expert[slot].active = 1;
  }
  if (0 != skip_slot) {
    const double ps = ngram_skip_prob(hist, hlen, cur);
    /**
     * Abstention is a property of the CONTEXT, not of the target: the tier
     * speaks when its pair was observed, whatever probability it then assigns
     * to this particular successor. Testing ps > 0.0 instead would silently
     * make the slot active only where it happens to be right, which reads the
     * target and would flatter the expert.
     */
    if (0 != ngram_skip_ready(hist, hlen)) {
      expert[NGRAM_BANK_SKIP].probability = ps;
      expert[NGRAM_BANK_SKIP].active = 1;
    }
  }
  if (0 != (slots & (1u << NGRAM_BANK_PRIOR))) {
    /* Zero history reaches the smoothed unigram prior with no backoff term. */
    expert[NGRAM_BANK_PRIOR].probability = ngramk_prob_exact(hist, 0, cur);
    expert[NGRAM_BANK_PRIOR].active = 1;
  }
  if (0 != (slots & (1u << NGRAM_BANK_PREDICT)) && NULL != store
    && hlen >= 1)
  {
    /**
     * The store is keyed on the last two tokens, so it speaks wherever one token
     * of history exists - including positions no count context attested, which
     * is the reason to carry it. The escape mass covers the unattested
     * remainder, so it is TOTAL: active whenever the output is scoreable.
     *
     * One call per position. prob_observe reports the distribution in effect
     * BEFORE the observation and advances the escape bank only afterwards, so
     * the ordering that keeps a stream figure honest is enforced by the library
     * rather than by call sequence here.
     */
    const unsigned int prev1 = hist[hlen - 1];
    const unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
    libxs_predict_prob_info_t pinfo;
    double in[2 * TOKEN_EMB_DIM];
    const double candidate = (double)cur;
    int nsupport;
    token_input_vector(prev2, prev1, use_emb, in);
    if (NULL != context) {
      nsupport = libxs_predict_prob_observe(NULL, store, context, in, 0,
        &candidate, NULL, NULL, 0, NULL, &pinfo, vocabulary, 1);
      /**
       * Mass mode only: a token id is discrete, so PDENSITY cannot arise, and an
       * unexpected PNONE must surface as abstention rather than a silent zero.
       */
      if (0 < nsupport && NULL != pinfo.kind
        && LIBXS_PREDICT_PMASS == pinfo.kind[0] && NULL != pinfo.prob)
      {
        expert[NGRAM_BANK_PREDICT].probability = pinfo.prob[0];
        expert[NGRAM_BANK_PREDICT].active = 1;
        predict_entropy_sum += pinfo.entropy;
        predict_support_last = nsupport;
        ++predict_nscored;
        { const double bits = -log((pinfo.prob[0] > 0.0)
            ? pinfo.prob[0] : 1e-12) / log(2.0);
          if (NULL != pinfo.attested && 0 != pinfo.attested[0]) {
            ++predict_att_n;
            predict_att_bits += bits;
          }
          else {
            ++predict_unatt_n;
            predict_unatt_bits += bits;
          }
        }
      }
    }
    else {
      /**
       * Frozen: the weights the warm-up committed are read and never written, so
       * the score does not depend on how many positions preceded this one. info
       * aliases the context, so the distribution cannot be reported here and the
       * point query is what is available - which is also cheaper, since it
       * answers P(y|x) without enumerating the support. The mass-mode check
       * moves to the reported probability being a usable number.
       */
      double p = 0.0;
      libxs_predict_prob(NULL, store, NULL, in, &candidate, &p, NULL,
        vocabulary, 1);
      nsupport = 0;
      if (p > 0.0 && p <= 1.0) {
        expert[NGRAM_BANK_PREDICT].probability = p;
        expert[NGRAM_BANK_PREDICT].active = 1;
        ++predict_nscored;
      }
    }
  }
  if (0 != (slots & (1u << NGRAM_BANK_EMB)) && hlen >= 1 && vocabulary > 0) {
    /**
     * TOTAL by construction and for a different reason than the predict slot:
     * that one is total because its escape mass covers an unattested remainder,
     * this one because a low-rank completion of PPMI assigns a score to pairs
     * that were never observed at all. It reads ONE token of history, so it is
     * the weakest possible context - carried anyway because the measured wall is
     * that 78% of novel-context positions have no attested successor to rank, and
     * coverage there is worth more than context depth here.
     */
    const int nctx = (ngram_emb_ctx() < hlen) ? ngram_emb_ctx() : hlen;
    const double p = token_emb_succ_prob(hist + (hlen - nctx), nctx, cur,
      (unsigned int)vocabulary, ngram_emb_temp());
    if (p > 0.0 && p <= 1.0) {
      expert[NGRAM_BANK_EMB].probability = p;
      expert[NGRAM_BANK_EMB].active = 1;
    }
  }
}

/**
 * Which pooling rule the bank uses: 0 = linear (default, bit-exact), 1 =
 * geometric (log-linear).
 *
 * The linear pool cannot beat its best member by much: it is a convex
 * combination, so a confident expert can be outvoted but never veto. Three
 * added slots were each capped near 0.008 bits, which makes the RULE a suspect
 * independent of the expert supply. A geometric pool multiplies instead of
 * averaging, so agreement sharpens and any expert can suppress a candidate.
 */
static int ngram_bank_geometric(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK_GEO");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}

/**
 * Collect the union of successor ids reachable from this history, which is the
 * only part of the vocabulary where the experts differ from a scaled prior.
 * Every expert is the same interpolated backoff over suffixes of one history,
 * so the union over all of them is the union of the successor lists at suffix
 * lengths 1..maxorder - at most maxorder * SUCC_MAX ids.
 */
static int ngram_bank_support(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int ids[], int max)
{
  int result = 0, n;
  for (n = 1; n <= maxorder && n <= hlen; ++n) {
    const ngram_entry_t* entry = ngramk_lookup(model, hist, hlen, n);
    if (NULL != entry) {
      unsigned int s;
      for (s = 0; s < entry->nsucc && result < max; ++s) {
        const unsigned int id = entry->succ[s].id;
        int seen = 0, i;
        for (i = 0; i < result; ++i) {
          if (ids[i] == id) {
            seen = 1;
            break;
          }
        }
        if (0 == seen && 0 != id) ids[result++] = id;
      }
    }
  }
  return result;
}

/**
 * Geometric (log-linear) pool, normalized EXACTLY rather than approximately.
 * An unnormalized log-linear score is not a code length, so a BPC taken from
 * one would be meaningless - the whole point of the measurement.
 *
 * The normalizer is exact at bounded cost because of the backoff structure: for
 * an id outside every successor list along the chain, each expert reduces to
 * c_k * prior(id) with c_k its total backoff mass. With the weights normalized
 * to sum to one, the product over such ids is (prod c_k^w_k) * prior(id), so
 * that entire tail of the vocabulary sums in closed form and only the support
 * needs explicit terms. c_k itself comes from each expert summing to one over
 * the vocabulary, so no probe id and no vocabulary scan is needed.
 *
 * Partial experts are excluded: an expert assigning exactly zero would drive
 * the product to zero everywhere outside its own successor list, which is the
 * known pathology of log-linear pools rather than a property of the data. Only
 * the total experts (the orders and the prior) participate.
 */
static double ngram_bank_pool_geo(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int cur,
  const double weight[], const ngram_expert_t expert[])
{
  unsigned int support[NGRAM_ORDER_MAX * NGRAM_SUCC_MAX];
  double logprod[NGRAM_ORDER_MAX * NGRAM_SUCC_MAX];
  double prior_sum = 0.0, tail_log = 0.0, wtotal = 0.0;
  double zsum = 0.0, numerator;
  const int nsupport = ngram_bank_support(model, hist, hlen, maxorder, support,
    (int)(sizeof(support) / sizeof(*support)));
  int slot, i, cur_index = -1;
  for (slot = 1; slot <= NGRAM_BANK_LAST; ++slot) {
    if (weight[slot] > 0.0 && 0 != expert[slot].active
      && expert[slot].probability > 0.0)
    {
      wtotal += weight[slot];
    }
  }
  if (!(wtotal > 0.0)) return 1e-12;
  for (i = 0; i < nsupport; ++i) {
    logprod[i] = 0.0;
    prior_sum += ngramk_prob_exact(hist, 0, support[i]);
    if (support[i] == cur) cur_index = i;
  }
  if (prior_sum >= 1.0) prior_sum = 1.0;
  for (slot = 1; slot <= NGRAM_BANK_LAST; ++slot) {
    if (weight[slot] > 0.0 && 0 != expert[slot].active
      && expert[slot].probability > 0.0)
    {
      const double w = weight[slot] / wtotal;
      const int k = (slot <= NGRAM_ORDER_MAX) ? slot : 0;
      double mass = 0.0, backoff;
      for (i = 0; i < nsupport; ++i) {
        const double pk = (k > 0)
          ? ngramk_prob_exact(hist + (hlen - k), k, support[i])
          : ngramk_prob_exact(hist, 0, support[i]);
        logprod[i] += w * log((pk > 0.0) ? pk : 1e-300);
        mass += pk;
      }
      /* Backoff mass left for the tail, from this expert summing to one. */
      backoff = (prior_sum < 1.0) ? (1.0 - mass) / (1.0 - prior_sum) : 0.0;
      tail_log += w * log((backoff > 0.0) ? backoff : 1e-300);
    }
  }
  for (i = 0; i < nsupport; ++i) zsum += exp(logprod[i]);
  zsum += exp(tail_log) * (1.0 - prior_sum);
  /**
   * The target is scored through the same partition, not added to the support:
   * the tail form is exact for any id outside it, so the normalizer stays
   * independent of which target is being scored.
   */
  numerator = (cur_index >= 0) ? exp(logprod[cur_index])
    : exp(tail_log) * ngramk_prob_exact(hist, 0, cur);
  if (!(zsum > 0.0)) return 1e-12;
  numerator /= zsum;
  return (numerator > 0.0) ? numerator : 1e-12;
}

/**
 * Score one position under the bank: a linear pool over the active slots,
 * renormalized by their weight mass so an abstaining expert costs no
 * probability mass rather than draining it.
 */
static double ngram_bank_pool(const double weight[],
  const ngram_expert_t expert[])
{
  libxs_mix_t mix;
  double copy[NGRAM_BANK_MAX];
  double prob[NGRAM_BANK_MAX];
  int active[NGRAM_BANK_MAX];
  double pooled;
  int slot;
  /* pool does not write the weights, but the view type is mutable; copying the
   * few slots keeps the const contract instead of casting it away */
  for (slot = 0; slot < NGRAM_BANK_MAX; ++slot) copy[slot] = weight[slot];
  ngram_bank_view(&mix, copy, expert, prob, active, 0.0, 0.0);
  pooled = libxs_mix_pool(&mix, prob, active);
  /* the log-loss floor stays here: the primitive reports what it computed and
   * leaves the choice of floor to the caller that takes the logarithm */
  return (pooled > 0.0) ? pooled : 1e-12;
}


static int predict_eval_stride(void)
{
  int result = TOKEN_PREDICT_EVAL_STRIDE;
  const char* env = getenv("CONVERSE_EVAL_STRIDE");
  if (NULL != env && '\0' != *env) {
    int value = atoi(env);
    if (value > 0) result = value;
  }
  return result;
}

/**
 * Rank the successors of a single record by count (converse's legacy order-2
 * adapters expose a bare entry; the library ranks internally via predict).
 */
static int ngram_topk(const ngram_entry_t* entry, unsigned int out_ids[],
  int k)
{
  int result = 0;
  if (NULL != entry && NULL != out_ids && k > 0) {
    unsigned int taken[NGRAM_SUCC_MAX];
    unsigned int nsucc = entry->nsucc;
    unsigned int slot;
    for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) taken[slot] = 0;
    while (result < k) {
      int best = -1;
      for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
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


static double ngram_unigram_prior(unsigned int id)
{
  double result = 0.0;
  if (NULL != converse_ngram_handle()->unifreq && 0 != id
    && id <= converse_ngram_handle()->unifreq_size)
  {
    result = (double)converse_ngram_handle()->unifreq[id] / converse_ngram_handle()->unifreq_total;
  }
  return result;
}


static double ngram_pair_relfreq(libxs_registry_t* model, unsigned int ctx_a,
  unsigned int ctx_b, unsigned int succ_id)
{
  double result = 0.0;
  const ngram_entry_t* entry = ngram_lookup(model, ctx_a, ctx_b);
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

/**
 * How many tokens of history the successor distribution conditions on (1 = the
 * immediate predecessor only, which is where this expert started).
 *
 * Summing the context vectors is not the retired "decayed bag": under this link
 * it is the naive-Bayes combination of independent evidence. Because
 * <sum_i u_i, v_c> = sum_i PMI(p_i, c), adding context in embedding space
 * multiplies the per-position likelihood ratios, so each history token
 * contributes its own testimony about the successor. What the flat position-
 * weighting axis retired was averaging TOPICAL vectors to form a retrieval query;
 * these vectors are successor-predictive and the sum has a defined meaning.
 *
 * The magnitude grows with the number of terms, which is correct for evidence
 * accumulation (more testimony, sharper posterior) but does interact with the
 * temperature, so the two want sweeping together rather than in isolation.
 */
static int ngram_emb_ctx(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_CTX");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 1;
    if (cached < 1) cached = 1;
    if (cached > TOKEN_CTX_MAX) cached = TOKEN_CTX_MAX;
  }
  return cached;
}


static int token_input_vector(unsigned int prev2, unsigned int prev1,
  int use_emb, double inputs[])
{
  int result;
  if (0 != use_emb) {
    const double* emb2 = token_emb_get(prev2);
    const double* emb1 = token_emb_get(prev1);
    int dim;
    for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
      inputs[dim] = emb2[dim];
      inputs[TOKEN_EMB_DIM + dim] = emb1[dim];
    }
    result = 2 * TOKEN_EMB_DIM;
  }
  else {
    inputs[0] = (double)prev2;
    inputs[1] = (double)prev1;
    result = 2;
  }
  return result;
}

/**
 * Number of context tokens summarized into a kNN-LM query vector (>=2; 2 is
 * the historical prev2/prev1 pair, bit-exact). Wider context = longer reach.
 */
static int knnlm_ctxlen(void)
{
  int result = 2;
  const char* env = getenv("CONVERSE_KNNLM_CTX");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 2 && v <= TOKEN_CTX_MAX) result = v;
  }
  return result;
}

/* Geometric decay applied to older context tokens in the summarized half. */
static double knnlm_decay(void)
{
  double result = 0.5;
  const char* env = getenv("CONVERSE_KNNLM_DECAY");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}

/**
 * Build a kNN-LM query vector from a rolling history (most-recent last). The
 * prev1 half is the embedding of the most recent token; the prev2 half is a
 * decayed, L2-normalized sum of the preceding ctxlen-1 tokens. With ctxlen==2
 * this reproduces token_input_vector(prev2, prev1, 1, .) exactly, so the
 * default path is byte-identical to the two-token model.
 */
/**
 * Per-position weights for the summarized half, replacing the geometric decay
 * when CONVERSE_KNNLM_WEIGHTS is set (comma-separated, nearest position first).
 * Returns the number parsed, or 0 to keep the geometric profile. A weight
 * profile is the scalar case of a position-specific projection: if position
 * identity carries usable signal, an explicit profile must beat the geometric
 * one, which is a far cheaper test than fitting matrices.
 */
static int knnlm_weights(double out[], int max)
{
  const char* env = getenv("CONVERSE_KNNLM_WEIGHTS");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    const char* p = env;
    while (result < max && '\0' != *p) {
      out[result++] = atof(p);
      while ('\0' != *p && ',' != *p) ++p;
      if (',' == *p) ++p;
    }
  }
  return result;
}


static void knnlm_ctx_vector(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double inputs[])
{
  const double* emb1 = token_emb_get((hlen > 0) ? hist[hlen - 1] : 0);
  double profile[TOKEN_CTX_MAX];
  int nprofile = knnlm_weights(profile, TOKEN_CTX_MAX);
  int dim, back;
  double weight = 1.0, norm = 0.0;
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
    inputs[TOKEN_EMB_DIM + dim] = emb1[dim];
    inputs[dim] = 0.0;
  }
  for (back = 2; back <= ctxlen; ++back) {
    int idx = hlen - back;
    if (idx >= 0) {
      const double* emb = token_emb_get(hist[idx]);
      const double w = (back - 2) < nprofile ? profile[back - 2] : weight;
      for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) inputs[dim] += w * emb[dim];
    }
    weight *= decay;
  }
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) norm += inputs[dim] * inputs[dim];
  if (norm > 0.0) {
    double inv = 1.0 / sqrt(norm);
    for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) inputs[dim] *= inv;
  }
}


static double knnlm_cosine(const double* a, const double* b, int n)
{
  double dot = 0.0, na = 0.0, nb = 0.0;
  int i;
  for (i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0.0 && nb > 0.0) ? (dot / (sqrt(na) * sqrt(nb))) : 0.0;
}

/**
 * Quantify how much word order survives in a query vector: for each evaluated
 * context, compare the vector against the one built from the SAME tokens in a
 * swapped order. Cosine 1.0 means order was discarded entirely; lower means the
 * representation distinguishes the permutation. Reports the mean over contexts,
 * separately for a swap inside the summarized half (positions 2 and 3) and for
 * a swap that crosses into the most-recent slot (positions 1 and 2).
 */
static void knnlm_order_probe(const unsigned int hist[], int hlen, int ctxlen,
  double decay, double* sum_inner, double* sum_cross, long* n_inner,
  long* n_cross)
{
  double base[2 * TOKEN_EMB_DIM], perm[2 * TOKEN_EMB_DIM];
  unsigned int swapped[TOKEN_CTX_MAX];
  int i;
  if (hlen < 2) return;
  for (i = 0; i < hlen && i < TOKEN_CTX_MAX; ++i) swapped[i] = hist[i];
  knnlm_ctx_vector(hist, hlen, ctxlen, decay, base);
  if (hlen >= 3 && ctxlen >= 3 && hist[hlen - 2] != hist[hlen - 3]) {
    swapped[hlen - 2] = hist[hlen - 3];
    swapped[hlen - 3] = hist[hlen - 2];
    knnlm_ctx_vector(swapped, hlen, ctxlen, decay, perm);
    *sum_inner += knnlm_cosine(base, perm, 2 * TOKEN_EMB_DIM);
    ++*n_inner;
    swapped[hlen - 2] = hist[hlen - 2];
    swapped[hlen - 3] = hist[hlen - 3];
  }
  if (hist[hlen - 1] != hist[hlen - 2]) {
    swapped[hlen - 1] = hist[hlen - 2];
    swapped[hlen - 2] = hist[hlen - 1];
    knnlm_ctx_vector(swapped, hlen, ctxlen, decay, perm);
    *sum_cross += knnlm_cosine(base, perm, 2 * TOKEN_EMB_DIM);
    ++*n_cross;
  }
}


static const ngram_entry_t* ngram_lookup(libxs_registry_t* model,
  unsigned int ctx_a, unsigned int ctx_b)
{
  const ngram_entry_t* result = NULL;
  if (NULL != model && 0 != ctx_b) {
    unsigned int hist[2];
    if (0 == ctx_a) {
      hist[0] = ctx_b;
      result = ngramk_lookup(model, hist, 1, 1);
    }
    else {
      hist[0] = ctx_a;
      hist[1] = ctx_b;
      result = ngramk_lookup(model, hist, 2, 2);
    }
  }
  return result;
}


static int ngram_predict(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int out_ids[], int k)
{
  int result = 0;
  if (NULL != model && NULL != out_ids && k > 0) {
    const ngram_entry_t* entry = NULL;
    if (order >= 2 && 0 != prev2) entry = ngram_lookup(model, prev2, prev1);
    if (NULL == entry) entry = ngram_lookup(model, 0, prev1);
    if (NULL != entry) result = ngram_topk(entry, out_ids, k);
    if (0 == result) {
      int slot;
      for (slot = 0; slot < k && slot < converse_ngram_handle()->backoff_count; ++slot) {
        out_ids[slot] = converse_ngram_handle()->backoff_ids[slot];
        ++result;
      }
    }
  }
  return result;
}


static void ngram_last_context(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  unsigned int* prev2, unsigned int* prev1)
{
  libxs_lexeme_stream_t stream;
  unsigned int p2 = 0, p1 = 0;
  libxs_lexeme_stream_init(&stream);
  if (NULL != lexicon && NULL != rules && nrules > 0 && text_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, (size_t)text_len, rules, nrules,
      converse_lexnorms(), converse_lexnorms_size(), 0))
  {
    size_t pos;
    for (pos = 0; pos < stream.size; ++pos) {
      const libxs_lexeme_t* lex = stream.data + pos;
      if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        p2 = p1;
        p1 = lex->id;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  if (NULL != prev2) *prev2 = p2;
  if (NULL != prev1) *prev1 = p1;
}

/**
 * Whether gen-eval keeps scoring after the first wrong token instead of ending
 * the sentence there.
 *
 * Off is the historical definition and stays bit-exact: mean-reproduced is the
 * length of the verbatim prefix, so the scan stops at the first miss. That
 * definition has almost no dynamic range on novel seeds - divergence is at the
 * very first position, so the bucket reads the same 0.06 for every configuration
 * measured so far, which is a property of the metric and not of the mechanism.
 * With this on, every lookahead position is visited with the TRUTH token fed
 * back as history, and the per-position accuracy and rank statistics are
 * computed over all of them. The prefix metrics keep their old definition
 * either way, so one run reports both readings - exactly so while the bank is
 * off. With the bank on, the extra positions also feed ngram_bank_update, so
 * the weights differ from a prefix-mode run and the prefix metrics are then
 * comparable in meaning but not to the digit.
 */
static int ngram_gen_full(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_FULL");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}

/**
 * Whether gen-eval also ranks the truth by the successor-side embedding, over the
 * WHOLE vocabulary and independently of what the count model proposed.
 *
 * This is the probe that decides the directed representation BEFORE any of it is
 * wired into a predictor. The count model carries the truth in its proposal on
 * only 22% of novel-context positions, so the question that matters is not
 * whether the embedding reranks better - it is whether a TOTAL scorer can rank
 * the truth at all where the partial one had nothing to offer. Costs one
 * vocabulary scan per position, which is why it is a knob and not always on.
 */
static int ngram_gen_embrank(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_EMBRANK");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}

/**
 * How many embedding-proposed successors join the candidate set per step, 0 =
 * off.
 *
 * OFF BY DEFAULT BECAUSE IT CHANGES THE PROMISE, not because it is unfinished.
 * Every other generation tier emits a successor the n-gram ATTESTED in this
 * context, which is what makes the output traceable to corpus text; the one
 * existing exception, recombination's composer, is opt-in for exactly this
 * reason. A candidate proposed by the low-rank completion was never observed
 * after this context, so with this on the path SYNTHESIZES rather than SELECTS
 * and the caller has to have asked for that.
 *
 * Secondary consequence to keep in mind when reading the numbers: a position the
 * grounding floor would have suppressed becomes scoreable once proposals exist,
 * so the count candidates below that floor are admitted alongside them and the
 * prefix metrics are not comparable across this knob.
 */
static int ngram_gen_embcand(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_EMBCAND");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
    if (cached > GEN_CAND_MAX) cached = GEN_CAND_MAX;
  }
  return cached;
}

/** Text of the first n ids, as scoring context for the byte model. */
static int ngram_gen_join(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int n, char* out, size_t out_size)
{
  size_t pos = 0;
  int slot;
  if (0 < out_size) out[0] = '\0';
  for (slot = 0; slot < n; ++slot) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[slot], &len, NULL);
    pos = ngram_render_append(out, out_size, pos, word, len, (0 < pos) ? 1 : 0);
  }
  return (int)pos;
}

/**
 * Held-out generation quality: seed each test sentence with its first GEN_SEED
 * content tokens, greedily extend, and count how many of the sentence's actual
 * remaining tokens the generator reproduces before diverging. A gradient-free,
 * generation-native counterpart to BPC (which only scores one-step prediction).
 * Also reports the mean grounding order over generated tokens.
 */
/**
 * Reorder generative candidates by the expert bank instead of by raw count
 * rank. Generation is where a slot has to earn its keep for real: BPC scores a
 * distribution against a known target, whereas here the model must put the
 * right word FIRST with no target in sight. A slot that lowers BPC but cannot
 * change the argmax has not changed what the system produces.
 *
 * Weights are carried across positions exactly as in scoring, but the update is
 * driven by the CHOSEN token, not the true one - no target information may
 * enter generation. The bank therefore adapts to its own trajectory here, which
 * is the honest online setting.
 */
static int ngram_gen_bank_rank(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int ids[], int n,
  unsigned int slots, const libxs_predict_t* store, void* context,
  int use_emb, int vocabulary, double weight[])
{
  int result = 0;
  if (1 < n) {
    double best = -1.0;
    int i;
    for (i = 0; i < n; ++i) {
      ngram_expert_t expert[NGRAM_BANK_MAX];
      double pooled;
      ngram_bank_experts(hist, hlen, maxorder, ids[i], slots, store, context,
        use_emb, vocabulary, expert);
      pooled = ngram_bank_pool(weight, expert);
      if (pooled > best) {
        best = pooled;
        result = i;
      }
    }
  }
  return result;
}

/**
 * One bucket of the per-position generation report.
 *
 * inlist is the share of scored positions where the truth appears ANYWHERE in
 * the offered candidates. It separates the two ways a generator can be wrong:
 * top1 - inlist is what better ranking could still win, and 100 - inlist is what
 * no reranking can reach because the truth was never proposed. The count model
 * offers only attested successors, so on novel context the second term is the
 * one that binds, and a mechanism aimed at the first would be misdirected.
 */
static void gen_bucket_report(const char* label, long n, long top1,
  long inlist, long abst, double rr)
{
  fprintf(stderr, "%s n=%ld top1=%.2f%% mrr=%.4f inlist=%.2f%%"
    " abstain=%.2f%%\n", label, n,
    (n > 0) ? 100.0 * (double)top1 / (double)n : 0.0,
    (n > 0) ? rr / (double)n : 0.0,
    (n > 0) ? 100.0 * (double)inlist / (double)n : 0.0,
    (0 < n + abst) ? 100.0 * (double)abst / (double)(n + abst) : 0.0);
}


static void gen_embrank_report(const char* label, long n, long top1,
  long top10, double rr)
{
  fprintf(stderr, "%s n=%ld top1=%.2f%% top10=%.2f%% mrr=%.4f\n", label, n,
    (n > 0) ? 100.0 * (double)top1 / (double)n : 0.0,
    (n > 0) ? 100.0 * (double)top10 / (double)n : 0.0,
    (n > 0) ? rr / (double)n : 0.0);
}


static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind,
  const libxs_predict_t* store, int use_emb)
{
  enum { GEN_SEED = 3, GEN_LOOK = 20 };
  int result = EXIT_FAILURE;
  long nsent = 0, sum_repro = 0, gen_tokens = 0, order_sum = 0, index = 0;
  long nsent_att = 0, sum_repro_att = 0, nsent_nov = 0, sum_repro_nov = 0;
  int maxorder = ngram_maxorder();
  int minorder = ngram_gen_minorder();
  const int gen_full = ngram_gen_full();
  const int gen_ncand = ngram_gen_ncand();
  long pos_n = 0, pos_top1 = 0, pos_abst = 0, pos_order = 0, pos_inl = 0;
  long pos_n_att = 0, pos_top1_att = 0, pos_abst_att = 0, pos_inl_att = 0;
  long pos_n_nov = 0, pos_top1_nov = 0, pos_abst_nov = 0, pos_inl_nov = 0;
  double pos_rr = 0.0, pos_rr_att = 0.0, pos_rr_nov = 0.0;
  long ctx_n_att = 0, ctx_top1_att = 0, ctx_abst_att = 0, ctx_inl_att = 0;
  long ctx_n_nov = 0, ctx_top1_nov = 0, ctx_abst_nov = 0, ctx_inl_nov = 0;
  double ctx_rr_att = 0.0, ctx_rr_nov = 0.0;
  const int gen_embrank = ngram_gen_embrank();
  const int gen_embcand = ngram_gen_embcand();
  const int gen_embctx = ngram_emb_ctx();
  /**
   * Candidate budget is FIXED at GEN_CAND_MAX, so proposals take slots from the
   * counts rather than enlarging the list. Two reasons. Without it the counts can
   * fill the list at exactly the positions proposals are for - a match at order
   * 1 still returns a full set while failing the grounding floor, so there was no
   * room left and 53249 positions kept abstaining. And a longer list would raise
   * inlist for free, which would make the comparison against the attested-only
   * baseline meaningless: at matched width, a change in inlist is a real
   * substitution.
   */
  const int gen_ncand_cnt = (0 < gen_embcand)
    ? ((GEN_CAND_MAX - gen_embcand) > 1 ? (GEN_CAND_MAX - gen_embcand) : 1)
    : 0;
  long pos_embadd = 0, pos_embpick = 0, pos_embpick_nov = 0, pos_resc = 0;
  long emb_n_att = 0, emb_top1_att = 0, emb_top10_att = 0;
  long emb_n_nov = 0, emb_top1_nov = 0, emb_top10_nov = 0;
  long emb_n_dec = 0, emb_top1_dec = 0, emb_top10_dec = 0;
  long emb_n_spk = 0, emb_top1_spk = 0, emb_top10_spk = 0;
  double emb_rr_att = 0.0, emb_rr_nov = 0.0;
  double emb_rr_dec = 0.0, emb_rr_spk = 0.0;
  long gen_ranked = 0, gen_reordered = 0;
  const int gen_bank = ngram_bank_probe();
  const unsigned int bank_slots = ngram_bank_slots();
  const int gen_vocab = (int)libxs_lexicon_size(lexicon);
  double bank_weight[NGRAM_BANK_MAX];
  void* gen_context = NULL;
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  { int k, nslot = 0;
    for (k = 1; k <= NGRAM_BANK_LAST; ++k) {
      if (0 != ngram_bank_enabled(bank_slots, k)) ++nslot;
    }
    for (k = 0; k < NGRAM_BANK_MAX; ++k) {
      bank_weight[k] = (k >= 1 && k <= NGRAM_BANK_LAST
        && 0 != ngram_bank_enabled(bank_slots, k)) ? 1.0 / (double)nslot : 0.0;
    }
  }
  if (0 != gen_bank && 0 != (bank_slots & (1u << NGRAM_BANK_PREDICT))
    && NULL != store)
  {
    gen_context = libxs_predict_prob_create(store);
  }

  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_t stream;
    const int native = ngram_native_mode();
    libxs_lexeme_stream_init(&stream);
    /**
     * The truth tokens must come from the SAME unit the model was trained on.
     * Reading word lexemes while the model holds syllable or byte-pair ids makes
     * every prediction a miss and reports a reproduction of zero, which reads as
     * a catastrophically bad generator rather than as a unit mismatch.
     */
    if (0 != is_test && entry->text_len > 0
      && (0 != native || EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
        &stream, (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0)))
    {
      unsigned int truth[GEN_SEED + GEN_LOOK];
      libxs_lexeme_t nat[COMPOSE_MAXTEXT];
      int ntruth = 0;
      int ntok, ti;
      ntok = (0 != native)
        ? ngram_native_tokens(lexicon, entry->text, entry->text_len, nat,
          NULL, COMPOSE_MAXTEXT, 0)
        : (int)stream.size;
      for (ti = 0; ti < ntok && ntruth < GEN_SEED + GEN_LOOK; ++ti) {
        if (0 != native) {
          if (0 != nat[ti].id) truth[ntruth++] = nat[ti].id;
        }
        else {
          const libxs_lexeme_t* lex = stream.data + ti;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            truth[ntruth++] = lex->id;
          }
        }
      }
      if (ntruth > GEN_SEED) {
        unsigned int hist[NGRAM_ORDER_MAX];
        const int seedcap = (GEN_SEED < maxorder) ? GEN_SEED : maxorder;
        int hlen = 0, t, repro = 0, diverged = 0, seed_attested = -1;
        int stop = 0, spoke = 0, hit = 0, abst = 0, ordsum = 0, inl = 0;
        double rrsum = 0.0;
        for (t = 0; t < GEN_SEED; ++t) hist[hlen++] = truth[t];
        for (t = GEN_SEED; t < ntruth && 0 == stop; ++t) {
          unsigned int ids[GEN_CAND_MAX];
          unsigned int embids[GEN_CAND_MAX];
          int got_order = 0;
          int ctxatt = 0, nemb = 0, cdecl;
          int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
            (0 < gen_embcand && gen_ncand > gen_ncand_cnt)
              ? gen_ncand_cnt : gen_ncand, &got_order);
          /* Proposals join BEFORE the bank so the pool arbitrates the union
           * rather than being handed a decision already made. */
          if (0 < gen_embcand) {
            const int nctx = (gen_embctx < hlen) ? gen_embctx : hlen;
            int k;
            nemb = token_emb_succ_append(hist + (hlen - nctx), nctx,
              (unsigned int)gen_vocab, ngram_emb_temp(), ids, n, GEN_CAND_MAX,
              gen_embcand);
            for (k = 0; k < nemb; ++k) embids[k] = ids[n + k];
            n += nemb;
            pos_embadd += nemb;
          }
          if (1 < n && 0 != converse_judge_active()) {
            char context[COMPOSE_MAXTEXT];
            const int context_len = ngram_gen_join(lexicon, truth, t,
              context, sizeof(context));
            const int pick = ngram_gen_select(lexicon, ids, n, context,
              context_len);
            if (0 != pick) {
              const unsigned int chosen = ids[pick];
              ids[pick] = ids[0];
              ids[0] = chosen;
            }
          }
          if (0 != gen_bank && 1 < n) {
            const int pick = ngram_gen_bank_rank(model, hist, hlen, maxorder,
              ids, n, bank_slots, store, gen_context, use_emb, gen_vocab,
              bank_weight);
            ++gen_ranked;
            if (0 != pick) ++gen_reordered;
            if (0 != pick) {
              const unsigned int chosen = ids[pick];
              ids[pick] = ids[0];
              ids[0] = chosen;
            }
            /* Adapt on the CHOSEN token: no target may enter generation. */
            { ngram_expert_t expert[NGRAM_BANK_MAX];
              double pooled;
              ngram_bank_experts(hist, hlen, maxorder, ids[0], bank_slots,
                store, gen_context, use_emb, gen_vocab, expert);
              pooled = ngram_bank_pool(bank_weight, expert);
              ngram_bank_update(bank_weight, expert, pooled,
                ngram_bank_rate(), ngram_bank_share());
            }
          }
          /**
           * Classify the sentence by its SEED context only, before any token is
           * generated, so the split cannot be influenced by how far generation
           * then runs: was the whole seed attested as a context in training?
           */
          if (seed_attested < 0) {
            seed_attested = (n > 0 && got_order >= seedcap) ? 1 : 0;
          }
          /**
           * Abstention is a position the model declined, not a wrong answer, so
           * it is counted apart from accuracy rather than folded in as a miss.
           * Under the prefix definition it also ends the sentence, which is why
           * the old scan could report a short run without saying why it stopped.
           */
          /**
           * Second split, per POSITION rather than per seed: did the FULL-ORDER
           * context of THIS position recur in training. That is the definition
           * ngram_eval's attested/novel BPC split already uses, so the two
           * instruments become readable together; the seed split classifies a
           * whole sentence by its first three words and stops discriminating
           * once true history is fed back.
           */
          ctxatt = (got_order >= maxorder) ? 1 : 0;
          /**
           * Whether the COUNTS declined, judged on the count-supplied part of the
           * list only. With no proposals nemb is zero and this is exactly the
           * historical condition, which is what keeps the knob-off path
           * bit-exact; with proposals it separates "counts had nothing" from
           * "nobody had anything".
           */
          cdecl = ((n - nemb) <= 0 || got_order < minorder) ? 1 : 0;
          /**
           * Scored at EVERY visited position, including the ones the count model
           * declined: a total scorer must be judged where the partial one was
           * silent, or the comparison quietly excludes the positions the whole
           * mechanism is for.
           */
          if (0 != gen_embrank) {
            const int nctx = (gen_embctx < hlen) ? gen_embctx : hlen;
            const int erank = token_emb_succ_rank(hist + (hlen - nctx), nctx,
              truth[t], (unsigned int)gen_vocab);
            /**
             * The control that decides whether a total scorer is worth anything:
             * its accuracy ON THE POSITIONS THE COUNT MODEL DECLINED. Averaged
             * over all positions it could earn its score entirely where counts
             * already speak, which would make it a redundant expert rather than a
             * reach into the bucket that has no evidence at all.
             */
            if (0 <= erank) {
              if (0 != cdecl) {
                ++emb_n_dec;
                if (0 == erank) ++emb_top1_dec;
                if (10 > erank) ++emb_top10_dec;
                emb_rr_dec += 1.0 / (double)(erank + 1);
              }
              else {
                ++emb_n_spk;
                if (0 == erank) ++emb_top1_spk;
                if (10 > erank) ++emb_top10_spk;
                emb_rr_spk += 1.0 / (double)(erank + 1);
              }
              if (0 != ctxatt) {
                ++emb_n_att;
                if (0 == erank) ++emb_top1_att;
                if (10 > erank) ++emb_top10_att;
                emb_rr_att += 1.0 / (double)(erank + 1);
              }
              else {
                ++emb_n_nov;
                if (0 == erank) ++emb_top1_nov;
                if (10 > erank) ++emb_top10_nov;
                emb_rr_nov += 1.0 / (double)(erank + 1);
              }
            }
          }
          if (0 != cdecl && 0 == nemb) {
            ++abst;
            if (0 != ctxatt) ++ctx_abst_att;
            else ++ctx_abst_nov;
            /**
             * The prefix definition ends the run here, so the prefix metrics
             * must stop accumulating even when the scan continues - otherwise
             * positions past an abstention extend mean-reproduced and the two
             * readings stop being comparable (measured: 7.48 -> 7.68 bpc).
             */
            diverged = 1;
            if (0 == gen_full) stop = 1;
          }
          else {
            /**
             * Rank of the truth within the offered candidates, AFTER every
             * reranking, so it scores the ranking the model actually produces.
             * It is -1 when the truth was not offered at all: the count model
             * proposes only attested successors, so on novel contexts that is
             * the common case and it must not contribute to the reciprocal rank.
             */
            int rank = -1, c;
            for (c = 0; c < n && rank < 0; ++c) {
              if (ids[c] == truth[t]) rank = c;
            }
            if (0 != cdecl) ++pos_resc;
            /* Whether what generation actually EMITTED is a successor no count
             * context attested here - the share of output that is synthesized
             * rather than selected, which is the cost side of this mode. */
            { int fromemb = 0, k;
              for (k = 0; k < nemb && 0 == fromemb; ++k) {
                if (embids[k] == ids[0]) fromemb = 1;
              }
              if (0 != fromemb) {
                ++pos_embpick;
                if (0 == ctxatt) ++pos_embpick_nov;
              }
            }
            if (0 == diverged) {
              ++gen_tokens;
              order_sum += got_order;
              if (0 == rank) ++repro;
              else diverged = 1;
            }
            ++spoke;
            ordsum += got_order;
            if (0 == rank) ++hit;
            if (0 <= rank) {
              ++inl;
              rrsum += 1.0 / (double)(rank + 1);
            }
            if (0 != ctxatt) {
              ++ctx_n_att;
              if (0 == rank) ++ctx_top1_att;
              if (0 <= rank) {
                ++ctx_inl_att;
                ctx_rr_att += 1.0 / (double)(rank + 1);
              }
            }
            else {
              ++ctx_n_nov;
              if (0 == rank) ++ctx_top1_nov;
              if (0 <= rank) {
                ++ctx_inl_nov;
                ctx_rr_nov += 1.0 / (double)(rank + 1);
              }
            }
            if (0 == gen_full) stop = diverged;
          }
          if (0 == stop) {
            if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = truth[t];
            else {
              int s;
              for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
              hist[NGRAM_ORDER_MAX - 1] = truth[t];
            }
          }
        }
        sum_repro += repro;
        ++nsent;
        pos_n += spoke;
        pos_top1 += hit;
        pos_abst += abst;
        pos_order += ordsum;
        pos_inl += inl;
        pos_rr += rrsum;
        if (1 == seed_attested) {
          sum_repro_att += repro;
          ++nsent_att;
          pos_n_att += spoke;
          pos_top1_att += hit;
          pos_abst_att += abst;
          pos_inl_att += inl;
          pos_rr_att += rrsum;
        }
        else {
          sum_repro_nov += repro;
          ++nsent_nov;
          pos_n_nov += spoke;
          pos_top1_nov += hit;
          pos_abst_nov += abst;
          pos_inl_nov += inl;
          pos_rr_nov += rrsum;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  libxs_predict_prob_destroy(gen_context);
  if (nsent > 0) {
    fprintf(stdout,
      "gen-eval[%s%s]: sentences=%ld mean-reproduced=%.2f "
      "mean-order=%.2f minorder=%d%s\n",
      kind, (holdout > 0) ? ":heldout" : "", nsent,
      (double)sum_repro / (double)nsent,
      (gen_tokens > 0) ? (double)order_sum / (double)gen_tokens : 0.0,
      minorder, (0 != gen_bank) ? " bank" : "");
    if (0 != gen_bank) {
      fprintf(stderr, "  bank rerank: %ld of %ld positions reordered"
        " (%.1f%%)\n", gen_reordered, gen_ranked,
        (gen_ranked > 0) ? 100.0 * (double)gen_reordered / (double)gen_ranked
          : 0.0);
    }
    fprintf(stderr, "  seed split: attested %.1f%% of sentences"
      " (mean-reproduced=%.2f) | novel %.1f%% (mean-reproduced=%.2f)\n",
      100.0 * (double)nsent_att / (double)nsent,
      (nsent_att > 0) ? (double)sum_repro_att / (double)nsent_att : 0.0,
      100.0 * (double)nsent_nov / (double)nsent,
      (nsent_nov > 0) ? (double)sum_repro_nov / (double)nsent_nov : 0.0);
    /**
     * mrr is the mean reciprocal rank of the truth over the offered candidates,
     * so at ncand=1 (the default) it equals top1 and carries no extra
     * information: widen CONVERSE_GEN_NCAND to give it range. ncand is printed
     * because it decides how to read the column.
     */
    fprintf(stderr, "  per-position[%s]: order=%.2f ncand=%d\n",
      (0 != gen_full) ? "full" : "prefix",
      (pos_n > 0) ? (double)pos_order / (double)pos_n : 0.0, gen_ncand);
    gen_bucket_report("    all:            ", pos_n, pos_top1, pos_inl,
      pos_abst, pos_rr);
    gen_bucket_report("    seed attested:  ", pos_n_att, pos_top1_att,
      pos_inl_att, pos_abst_att, pos_rr_att);
    gen_bucket_report("    seed novel:     ", pos_n_nov, pos_top1_nov,
      pos_inl_nov, pos_abst_nov, pos_rr_nov);
    gen_bucket_report("    ctx attested:   ", ctx_n_att, ctx_top1_att,
      ctx_inl_att, ctx_abst_att, ctx_rr_att);
    gen_bucket_report("    ctx novel:      ", ctx_n_nov, ctx_top1_nov,
      ctx_inl_nov, ctx_abst_nov, ctx_rr_nov);
    if (0 < gen_embcand) {
      fprintf(stderr, "  emb proposals[ncand=%d]: added=%ld rescued=%ld"
        " of %ld scored | EMITTED an unattested successor %.2f%% of scored"
        " (ctx-novel %.2f%%)\n", gen_embcand, pos_embadd, pos_resc, pos_n,
        (pos_n > 0) ? 100.0 * (double)pos_embpick / (double)pos_n : 0.0,
        (ctx_n_nov > 0)
          ? 100.0 * (double)pos_embpick_nov / (double)ctx_n_nov : 0.0);
    }
    if (0 != gen_embrank && 0 < emb_n_att + emb_n_nov) {
      fprintf(stderr, "  embedding rank of truth over vocab=%d"
        " (directed=%d, chance top1=%.4f%%):\n", gen_vocab,
        token_emb_directed(),
        (gen_vocab > 0) ? 100.0 / (double)gen_vocab : 0.0);
      gen_embrank_report("    ctx attested:   ", emb_n_att, emb_top1_att,
        emb_top10_att, emb_rr_att);
      gen_embrank_report("    ctx novel:      ", emb_n_nov, emb_top1_nov,
        emb_top10_nov, emb_rr_nov);
      gen_embrank_report("    counts spoke:   ", emb_n_spk, emb_top1_spk,
        emb_top10_spk, emb_rr_spk);
      gen_embrank_report("    counts DECLINED:", emb_n_dec, emb_top1_dec,
        emb_top10_dec, emb_rr_dec);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


static int slot_maxfreq(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_SLOT_MAXFREQ");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 20;
    if (cached < 1) cached = 1;
  }
  return cached;
}


static void slot_key_build(slot_key_t* key, const unsigned int hist[],
  int hlen, int order, unsigned int next, int hole)
{
  int pos;
  memset(key, 0, sizeof(*key));
  key->order = order;
  key->next = next;
  for (pos = 0; pos < order; ++pos) {
    const unsigned int id = hist[hlen - order + pos];
    key->context[pos] = (pos == hole) ? SLOT_HOLE_ID : id;
  }
}

/**
 * One pass over the corpus, calling back with every (history, successor) pair at
 * the configured order. Shared by the slot probe's train and test passes so both
 * see exactly the same tokenization the n-gram was built from.
 */
static void slot_scan(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout, int want_test,
  void (*visit)(const unsigned int*, int, unsigned int, void*), void* udata)
{
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  long index = 0;
  void* value = corpus_iterx_begin(corpus, &key, &cursor);
  const int maxorder = ngram_maxorder();
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    const int is_test = (0 != predict_is_test(index, holdout)) ? 1 : 0;
    libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    /**
     * Sentence scale only. The corpus holds each text at BOTH sentence and
     * paragraph scale, so scanning every entry sees each sentence twice - and a
     * paragraph copy of a training sentence would then make that sentence's own
     * contexts look attested, inflating the exact-match share far above what the
     * n-gram model actually holds.
     */
    if (SCALE_SENTENCE == entry->scale
      && (0 == holdout || is_test == want_test) && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
        (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
    {
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      size_t pos;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen >= maxorder) visit(hist, hlen, lex->id, udata);
          ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, lex->id);
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
}


static long slot_freq_of(const slot_probe_t* probe, unsigned int id)
{
  const long* count = (const long*)libxs_registry_get(probe->freq, &id,
    sizeof(id), NULL);
  return (NULL != count) ? *count : 0;
}


static void slot_count_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  long* count = (long*)libxs_registry_get(probe->freq, &next, sizeof(next),
    NULL);
  LIBXS_UNUSED(hist); LIBXS_UNUSED(hlen);
  if (NULL != count) ++*count;
  else {
    const long one = 1;
    libxs_registry_set(probe->freq, &next, sizeof(next), &one, sizeof(one),
      NULL);
  }
}


static void slot_train_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  const int order = probe->order;
  const char present = 1;
  int hole;
  slot_key_t key;
  if (hlen >= order) {
    slot_ctx_key_t ctx;
    int pos;
    memset(&ctx, 0, sizeof(ctx));
    ctx.order = order;
    for (pos = 0; pos < order; ++pos) ctx.context[pos] = hist[hlen - order + pos];
    libxs_registry_set(probe->contexts, &ctx, sizeof(ctx), &present,
      sizeof(present), NULL);
    slot_key_build(&key, hist, hlen, order, next, -1);
    libxs_registry_set(probe->patterns, &key, sizeof(key), &present,
      sizeof(present), NULL);
    for (hole = 0; hole < order; ++hole) {
      if (slot_freq_of(probe, hist[hlen - order + hole]) <= slot_maxfreq()) {
        slot_key_build(&key, hist, hlen, order, next, hole);
        libxs_registry_set(probe->patterns, &key, sizeof(key), &present,
          sizeof(present), NULL);
      }
    }
  }
}


static void slot_test_visit(const unsigned int hist[], int hlen,
  unsigned int next, void* udata)
{
  slot_probe_t* probe = (slot_probe_t*)udata;
  const int order = probe->order;
  slot_key_t key;
  slot_ctx_key_t ctx;
  int hole, nslots = 0, covered = 0, pos;
  if (hlen < order) return;
  memset(&ctx, 0, sizeof(ctx));
  ctx.order = order;
  for (pos = 0; pos < order; ++pos) ctx.context[pos] = hist[hlen - order + pos];
  if (NULL != libxs_registry_get(probe->contexts, &ctx, sizeof(ctx), NULL)) {
    ++probe->nexact;
    return;
  }
  ++probe->nnovel;
  for (hole = 0; hole < order; ++hole) {
    if (slot_freq_of(probe, hist[hlen - order + hole]) <= slot_maxfreq()) {
      ++nslots;
      slot_key_build(&key, hist, hlen, order, next, hole);
      if (NULL != libxs_registry_get(probe->patterns, &key, sizeof(key), NULL)) {
        covered = 1;
      }
    }
  }
  if (0 == nslots) ++probe->nslotless;
  if (0 != covered) ++probe->ncovered;
}

/**
 * Report what fraction of novel-context positions a one-hole abstraction would
 * have covered. Reported on stdout so it lands beside the other eval lines.
 */
static void slot_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  slot_probe_t probe;
  memset(&probe, 0, sizeof(probe));
  probe.patterns = libxs_registry_create();
  probe.contexts = libxs_registry_create();
  probe.freq = libxs_registry_create();
  probe.lexicon = lexicon;
  probe.order = ngram_maxorder();
  if (NULL != probe.patterns && NULL != probe.contexts
    && NULL != probe.freq)
  {
    slot_scan(corpus, lexicon, rules, nrules, holdout, 0, slot_count_visit,
      &probe);
    slot_scan(corpus, lexicon, rules, nrules, holdout, 0, slot_train_visit,
      &probe);
    slot_scan(corpus, lexicon, rules, nrules, holdout, 1, slot_test_visit,
      &probe);
    { const long total = probe.nexact + probe.nnovel;
      fprintf(stdout, "slot-probe[order=%d maxfreq=%d]: n=%ld"
        " exact=%.1f%% novel=%.1f%%"
        " | of novel: covered=%.1f%% no-slot=%.1f%%\n",
        probe.order, slot_maxfreq(), total,
        (0 < total) ? 100.0 * (double)probe.nexact / (double)total : 0.0,
        (0 < total) ? 100.0 * (double)probe.nnovel / (double)total : 0.0,
        (0 < probe.nnovel)
          ? 100.0 * (double)probe.ncovered / (double)probe.nnovel : 0.0,
        (0 < probe.nnovel)
          ? 100.0 * (double)probe.nslotless / (double)probe.nnovel : 0.0);
    }
  }
  libxs_registry_destroy(probe.patterns);
  libxs_registry_destroy(probe.contexts);
  libxs_registry_destroy(probe.freq);
}

/**
 * Which scoring mode the predict slot uses: 0 = adaptive (default), 1 = warm up
 * over the training entries, commit the converged weights, then score FROZEN.
 *
 * Adaptive scoring carries one context across the run, so the escape bank
 * converges but the figure becomes a function of the order entries are
 * iterated - reproducible here, not comparable across shuffles, and the
 * content-hash key already changed that order once. Frozen scoring is
 * order-independent by construction, but only says something once the weights
 * it freezes have converged, which is what the warm-up pass is for.
 */
static int ngram_bank_frozen(void)
{
  const char* env = getenv("CONVERSE_NGRAM_BANK_FROZEN");
  return (NULL != env && 0 != atoi(env)) ? 1 : 0;
}

/**
 * Converge the escape bank on the TRAINING data, then publish the weights so
 * frozen scoring has something converged to freeze.
 *
 * The entries pushed into the store are the training split already - pushing
 * is gated on predict_is_test in token_predict_build - so replaying them
 * reads no held-out data and needs none of the tokenization the eval loop does.
 * Passing NULL for values/probs/info still advances the bank: prob_observe
 * learns from the candidate, and the distribution is only reported if asked
 * for.
 *
 * Weights move only while this runs. Afterwards the model carries them and
 * scoring is strictly read-only, which is the property that makes the reported
 * figure independent of iteration order.
 */
static int ngram_bank_warmup(libxs_predict_t* store, int vocabulary)
{
  int result = EXIT_FAILURE;
  void* context = (NULL != store) ? libxs_predict_prob_create(store) : NULL;
  if (NULL != context) {
    libxs_predict_query_t info;
    long observed = 0;
    int i, nwarm;
    memset(&info, 0, sizeof(info));
    libxs_predict_query(store, &info);
    /**
     * How many entries the warm-up observes. The escape bank needs hundreds of
     * observations to settle, not hundreds of thousands, but each observation
     * costs a scan of every entry in the cluster - so observing the whole store
     * is quadratic in the store size while buying convergence that a prefix
     * already reached. A bound makes that trade measurable rather than assumed;
     * 0 keeps the full pass.
     */
    nwarm = info.nentries;
    { const char* env = getenv("CONVERSE_NGRAM_BANK_WARMUP");
      if (NULL != env) {
        const int n = atoi(env);
        if (0 < n && n < nwarm) nwarm = n;
      }
    }
    for (i = 0; i < nwarm; ++i) {
      /* sized for the widest input vector either profile uses */
      double in[2 * TOKEN_EMB_DIM];
      double out[1];
      out[0] = 0.0;
      libxs_predict_get(store, i, in, out);
      if (0 < libxs_predict_prob_observe(NULL, store, context, in, 0,
        out, NULL, NULL, 0, NULL, NULL, vocabulary, 1))
      {
        ++observed;
      }
    }
    result = libxs_predict_prob_commit(store, context);
    /* nscan is the candidates ONE observation walks, so the product is the real
     * cost of this pass - printed because a bound that looks small can still be
     * quadratic against a large cluster. */
    fprintf(stderr, "predict slot: warm-up observed %ld of %i entries"
      " (bound %i, %i clusters, scan max=%i avg=%.0f mean=%.0f"
      " => %.1fM pair-ops), commit %s\n",
      observed, info.nentries, nwarm, info.nclusters, info.nscan, info.escan,
      (0 < info.nclusters) ? ((double)info.nentries / info.nclusters) : 0.0,
      1e-6 * (double)observed * info.escan,
      (EXIT_SUCCESS == result) ? "ok" : "FAILED");
    libxs_predict_prob_destroy(context);
  }
  return result;
}


static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind, const libxs_predict_t* store,
  int use_emb)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, index = 0;
  long ndeep = 0, ndeep_top1 = 0, nshallow = 0, nshallow_top1 = 0;
  double deep_bits = 0.0, deep_bytes = 0.0;
  double shallow_bits = 0.0, shallow_bytes = 0.0;
  double sum_bits = 0.0, sum_bytes = 0.0;
  long novel_match[NGRAM_ORDER_MAX + 1];
  double novel_match_bits[NGRAM_ORDER_MAX + 1];
  double novel_match_bytes[NGRAM_ORDER_MAX + 1];
  long novel_short = 0;
  double novel_short_bits = 0.0;
  long novel_sat = 0;
  double novel_sat_bits = 0.0;
  long ntrunc = 0, ntrunc_top1 = 0;
  double trunc_bits = 0.0, trunc_bytes = 0.0;
  double oracle_bits = 0.0;
  /**
   * The oracle split by whether the full-order context recurred. Aggregate
   * calibration gains in this project have consistently come from the attested
   * side while the novel side stayed put, so a bound that does not separate them
   * cannot say whether reweighting has anything left to offer where generation
   * actually fails.
   */
  double oracle_deep_bits = 0.0, oracle_shallow_bits = 0.0;
  long oracle_extra = 0;
  double expert_bits[NGRAM_ORDER_MAX + 1];
  double expert_bytes[NGRAM_ORDER_MAX + 1];
  long oracle_pick[NGRAM_ORDER_MAX + 1];
  long oracle_mixed = 0;
  long sel_total[NGRAM_ORDER_MAX + 1];
  long sel_mixed[NGRAM_ORDER_MAX + 1];
  long sel_low[NGRAM_ORDER_MAX + 1];
  /**
   * How often the KIND-DIFFERENT slot is the cheapest member, bucketed by the
   * matched context length the model already reports. The oracle reads the
   * target, so its 0.188 bits are only reachable if an OBSERVABLE feature
   * separates those positions: if the share is flat across buckets, no selector
   * keyed on this feature can find them and the headroom is not reachable.
   */
  long sel_extra[NGRAM_ORDER_MAX + 1];
  double sel_bits = 0.0;
  const int sel_order = ngram_select_order();
  double entry_bytes = 0.0;
  double bank_bits = 0.0;
  double bank_weight[NGRAM_BANK_MAX];
  double bank_wsum[NGRAM_BANK_MAX];
  long bank_active[NGRAM_BANK_MAX];
  /**
   * Each slot's STANDALONE cost, over exactly the positions where it spoke.
   * Without this a collapsed weight is uninterpretable: it cannot be told apart
   * from a plumbing fault. Same per-expert denominator discipline as the order
   * experts - an abstaining slot must not be charged for positions it never
   * scored, nor credited with them.
   */
  double bank_slot_bits[NGRAM_BANK_MAX];
  double bank_slot_bytes[NGRAM_BANK_MAX];
  long bank_n = 0;
  /* The bank under the attested/novel split: two mechanisms so far had their
   * headline reversed by bucket-splitting, so every slot set reports both. */
  double bank_deep_bits = 0.0, bank_shallow_bits = 0.0;
  const int bank = ngram_bank_probe();
  const unsigned int bank_slots = ngram_bank_slots();
  const int bank_geo = ngram_bank_geometric();
  /**
   * One scoring context for the whole run. The escape bank needs hundreds of
   * observations to commit, so a per-entry stream would be all transient and no
   * convergence; a per-run stream converges but makes the figure a function of
   * ENTRY ORDER - reproducible for a fixed corpus and iteration, not comparable
   * across shuffles. A size of 0 means the model cannot be scored, in which case
   * the slot abstains rather than silently falling back to frozen weights.
   */
  void* bank_context = NULL;
  const int bank_frozen = ngram_bank_frozen();
  const int bank_vocab = (int)libxs_lexicon_size(lexicon);
  const double bank_rate = ngram_bank_rate();
  const double bank_share = ngram_bank_share();
  const int oracle = ngram_oracle_probe();
  const double inv_log2 = 1.0 / log(2.0);
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  FILE* file;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  memset(expert_bits, 0, sizeof(expert_bits));
  memset(expert_bytes, 0, sizeof(expert_bytes));
  memset(oracle_pick, 0, sizeof(oracle_pick));
  if (0 != bank && 0 != (bank_slots & (1u << NGRAM_BANK_PREDICT))
    && NULL != store && 0 == bank_frozen)
  {
    /**
     * NULL means the model cannot be scored, which is a different situation
     * from passing NULL as the context (that selects frozen scoring on a model
     * that is fine), so the slot abstains rather than silently freezing.
     */
    bank_context = libxs_predict_prob_create(store);
    if (NULL == bank_context) {
      fprintf(stderr, "predict slot: model cannot be scored, slot abstains\n");
    }
  }
  memset(bank_wsum, 0, sizeof(bank_wsum));
  memset(bank_active, 0, sizeof(bank_active));
  memset(bank_slot_bits, 0, sizeof(bank_slot_bits));
  memset(bank_slot_bytes, 0, sizeof(bank_slot_bytes));
  { int k;
    /**
     * Uniform prior over the ENABLED slots; the update is what differentiates
     * them. A disabled slot starts at zero and stays there, which is what makes
     * the order-only bank bit-exact to the pre-slot version.
     */
    int nslot = 0;
    for (k = 1; k <= NGRAM_BANK_LAST; ++k) {
      if (0 != ngram_bank_enabled(bank_slots, k)) ++nslot;
    }
    for (k = 0; k < NGRAM_BANK_MAX; ++k) {
      bank_weight[k] = (k >= 1 && k <= NGRAM_BANK_LAST
        && 0 != ngram_bank_enabled(bank_slots, k)) ? 1.0 / (double)nslot : 0.0;
    }
  }
  memset(sel_total, 0, sizeof(sel_total));
  memset(sel_mixed, 0, sizeof(sel_mixed));
  memset(sel_low, 0, sizeof(sel_low));
  memset(sel_extra, 0, sizeof(sel_extra));
  memset(novel_match, 0, sizeof(novel_match));
  memset(novel_match_bits, 0, sizeof(novel_match_bits));
  memset(novel_match_bytes, 0, sizeof(novel_match_bytes));
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    int native = ngram_native_mode();
    libxs_lexeme_stream_init(&stream);
    { const int ds = ngram_dedup_scale();
      const int want = (2 == ds) ? SCALE_PARAGRAPH : SCALE_SENTENCE;
      /**
       * Each source byte must be counted once, or BPC is not comparable across
       * granularities - which is the one job the metric exists to do. Clause
       * fragments are stored at sentence scale ALONGSIDE their parent sentence,
       * so keeping both scored ~3x the real text.
       */
      if (0 != ds && (want != entry->scale
        || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT)))
      {
        is_test = 0;
      }
    }
    if (0 != is_test && entry->text_len > 0
      && (0 != native || EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0)))
    {
      int wctx = ngram_wordctx();
      libxs_lexeme_t nat[COMPOSE_MAXTEXT];
      unsigned int word_ids[COMPOSE_MAXTEXT];
      int ntok = (0 != native)
        ? ngram_native_tokens(lexicon, entry->text, entry->text_len,
          nat, (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 0)
        : (int)stream.size;
      int ti;
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      int maxorder = ngram_maxorder();
      /* Source bytes this entry contributes, whether or not every token in it
       * is scored - the ceiling the denominator should approach. */
      entry_bytes += (double)entry->text_len;
      for (ti = 0; ti < ntok; ++ti) {
        unsigned int cur;
        unsigned short curlen;
        int is_content;
        if (0 != wctx) {
          hlen = ngram_wordctx_hist(nat, word_ids, ti, wctx, hist,
            NGRAM_ORDER_MAX);
        }
        if (0 != native) {
          cur = nat[ti].id;
          curlen = nat[ti].length;
          is_content = (0 != cur) ? 1 : 0;
        }
        else {
          const libxs_lexeme_t* lex = stream.data + ti;
          cur = lex->id;
          curlen = lex->length;
          is_content = (0 != (lex->flags
            & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER)) && 0 != cur) ? 1 : 0;
        }
        if (0 != is_content) {
          if (hlen > 0) {
            unsigned int ids[NGRAM_TOPK];
            int matched = 0;
            int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
              NGRAM_TOPK, &matched);
            double p = ngramk_prob(model, hist, hlen, maxorder, cur);
            double bits = -log(p) * inv_log2;
            int rank, top1 = 0;
            int deep = (matched >= maxorder && hlen >= maxorder) ? 1 : 0;
            /**
             * Per-position order selection, bounded from above. Each expert is
             * the SAME interpolated model given only its last k words of
             * history, so expert k is what a model capped at order k would say.
             * The oracle takes the cheapest expert per position: not achievable
             * (it reads the answer), but it bounds what ANY per-position mixture
             * of these predictors can win. The byte side found local order
             * selection worth 0.5-0.8 bits cross-document, and this asks whether
             * the same headroom exists at the word level before anything is
             * built to exploit it.
             */
            if (0 != oracle) {
              int k;
              double best = bits;
              int bestk = 0, best_extra = 0;
              const double w = (curlen > 0) ? (double)curlen : 1.0;
              for (k = 1; k <= maxorder && k <= hlen; ++k) {
                double pk = ngramk_prob(model, hist + (hlen - k), k, k, cur);
                double bk = -log((pk > 0.0) ? pk : 1e-12) * inv_log2;
                /**
                 * Per-expert byte denominator: expert k only exists where the
                 * history reaches k, so dividing its bits by the TOTAL bytes
                 * would flatter every high order by omitting the positions it
                 * cannot score. That artifact made order 6 look like the best
                 * expert when it had simply skipped the hard short-history
                 * positions.
                 */
                expert_bits[k] += bk;
                expert_bytes[k] += w;
                if (bk < best) {
                  best = bk;
                  bestk = k;
                }
              }
              /**
               * The bank: linear pool over the expert slots, with weights
               * carried across positions and updated causally (score first,
               * then update), so no target information enters the score. The
               * slots are the fixed orders plus, when enabled, experts of a
               * different KIND - the 2.018 oracle below says order selection
               * alone cannot reach the target, so mixing kinds is the only
               * direction with measured room.
               */
              if (0 != bank) {
                ngram_expert_t expert[NGRAM_BANK_MAX];
                double pooled, pbits;
                int b;
                ngram_bank_experts(hist, hlen, maxorder, cur, bank_slots,
                  store, bank_context, use_emb, bank_vocab, expert);
                pooled = (0 != bank_geo)
                  ? ngram_bank_pool_geo(model, hist, hlen, maxorder, cur,
                    bank_weight, expert)
                  : ngram_bank_pool(bank_weight, expert);
                pbits = -log(pooled) * inv_log2;
                bank_bits += pbits;
                if (0 != deep) bank_deep_bits += pbits;
                else bank_shallow_bits += pbits;
                ++bank_n;
                for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
                  bank_wsum[b] += bank_weight[b];
                  if (0 != expert[b].active) {
                    const double pb = expert[b].probability;
                    ++bank_active[b];
                    bank_slot_bits[b] += -log((pb > 0.0) ? pb : 1e-12)
                      * inv_log2;
                    bank_slot_bytes[b] += w;
                  }
                }
                /**
                 * Extend the bound to the WHOLE pool, not just the fixed orders.
                 * Any per-position reweighting can at best put all its mass on
                 * the cheapest member available at that position, so taking the
                 * minimum over every active slot - including the ones that
                 * differ in kind - bounds what ANY mixing rule over this
                 * evidence could reach. Without the extra slots the bound would
                 * flatter a pool it does not cover.
                 */
                for (b = NGRAM_ORDER_MAX + 1; b <= NGRAM_BANK_LAST; ++b) {
                  if (0 != expert[b].active) {
                    const double pb = expert[b].probability;
                    const double bb = -log((pb > 0.0) ? pb : 1e-12) * inv_log2;
                    if (bb < best) {
                      best = bb;
                      best_extra = 1;
                      ++oracle_extra;
                    }
                  }
                }
                ngram_bank_update(bank_weight, expert, pooled, bank_rate,
                  bank_share);
              }
              oracle_bits += best;
              if (0 != deep) oracle_deep_bits += best;
              else oracle_shallow_bits += best;
              /* bestk==0 means no single-order expert beat the mixture. */
              if (0 == bestk) ++oracle_mixed;
              else ++oracle_pick[bestk];
              /**
               * Can an OBSERVABLE feature separate "o1 wins" from "the mixture
               * wins"? The natural candidate costs nothing: the matched context
               * length, which the predictor already reports and which does not
               * read the target. Accumulate the oracle's choice against it, so
               * the confusion is visible rather than assumed. Secs 16/19/20 each
               * found a signal that separated classes in hindsight and died when
               * computed without the answer, so this is checked before any
               * chooser is written.
               */
              { const int mm = (matched < 0) ? 0
                  : ((matched > NGRAM_ORDER_MAX) ? NGRAM_ORDER_MAX : matched);
                ++sel_total[mm];
                if (0 != best_extra) ++sel_extra[mm];
                if (0 == bestk) ++sel_mixed[mm];
                else if (1 == bestk) ++sel_low[mm];
                /**
                 * The ACHIEVABLE rule the confusion above suggests: take the
                 * order-1 expert when a short context matched (matched >= 2),
                 * otherwise keep the mixture. Uses only the reported match
                 * length, never the target, so unlike the oracle this is a
                 * mechanism and its bits are honest.
                 */
                if (mm >= 2 && hlen >= sel_order) {
                  double ps = ngramk_prob(model, hist + (hlen - sel_order),
                    sel_order, sel_order, cur);
                  sel_bits += -log((ps > 0.0) ? ps : 1e-12) * inv_log2;
                }
                else sel_bits += bits;
              }
            }
            ++npairs;
            sum_bits += bits;
            sum_bytes += (curlen > 0) ? (double)curlen : 1.0;
            for (rank = 0; rank < n; ++rank) {
              if (ids[rank] == cur) {
                if (0 == rank) {
                  ++ntop1;
                  top1 = 1;
                }
                ++ntopk;
                break;
              }
            }
            /**
             * Split by whether the FULL-ORDER context was attested in training.
             * The store is built from non-test entries only, so a match at the
             * maximum order means this exact context recurs verbatim; those
             * positions are recall, not generalization. Reported separately so
             * the depth result cannot be read as generalization it is not.
             */
            if (0 != deep) {
              ++ndeep;
              ndeep_top1 += top1;
              deep_bits += bits;
              deep_bytes += (curlen > 0) ? (double)curlen : 1.0;
            }
            else {
              /**
               * Decompose the novel bucket by how much context DID match. The
               * predictor already reports that length, so this costs no extra
               * lookup. It bounds what a total function can buy: mass at
               * matched==0 is scored by the unigram prior alone and is the only
               * part where no count evidence exists at any order, while mass at
               * matched>0 is already carried by a shorter attested context.
               */
              const int m = (matched < 0) ? 0
                : ((matched > NGRAM_ORDER_MAX) ? NGRAM_ORDER_MAX : matched);
              ++nshallow;
              nshallow_top1 += top1;
              shallow_bits += bits;
              shallow_bytes += (curlen > 0) ? (double)curlen : 1.0;
              ++novel_match[m];
              novel_match_bits[m] += bits;
              novel_match_bytes[m] += (curlen > 0) ? (double)curlen : 1.0;
              /**
               * A position whose history is shorter than the order cannot be
               * full-order attested, so it lands in the novel bucket by sentence
               * position rather than by unattested context. Counted separately:
               * without this the short-context mass reads as weak evidence when
               * part of it is merely an early position with nothing to look up.
               */
              if (hlen < maxorder) {
                ++novel_short;
                novel_short_bits += bits;
              }
              /**
               * The distinction that decides how to read the short-context mass.
               * SATURATED means the model matched ALL the history that existed,
               * so nothing was unattested and the only remedy is more history -
               * an early sentence position. TRUNCATED means history was
               * available and the longer context was not in the store, which is
               * the genuinely unattested case. Lumping them reads a young
               * sentence as weak evidence.
               */
              { const int avail = (hlen < maxorder) ? hlen : maxorder;
                if (m >= avail) {
                  ++novel_sat;
                  novel_sat_bits += bits;
                }
                /**
                 * The REPAIRED generalization bucket: a full-order history was
                 * available AND the full-order context was not attested. This is
                 * the only set where the model was asked something its evidence
                 * could have covered and did not. The wider novel bucket mixes
                 * these with early-sentence positions that had nothing longer to
                 * look up, so a mechanism can move it without generalizing.
                 */
                else if (hlen >= maxorder) {
                  ++ntrunc;
                  ntrunc_top1 += top1;
                  trunc_bits += bits;
                  trunc_bytes += (curlen > 0) ? (double)curlen : 1.0;
                }
              }
            }
          }
          if (0 == wctx) ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, cur);
        }
        if (0 == native && 0 == wctx) {
          const libxs_lexeme_t* lex = stream.data + ti;
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-ngram[%s%s]: top1=%.1f%% top%d=%.1f%% n=%ld bpc=%.3f\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
    /**
     * The BPC denominator, printed because it is the only thing that makes the
     * number comparable ACROSS granularities: the same test text must yield the
     * same byte count whatever the unit. A zero-length token (a sub-word
     * continuation carries the whole word's bytes on its first piece) falls back
     * to 1.0, which would silently inflate the denominator and understate BPC
     * for the finer units, so the count is reported rather than trusted.
     */
    fprintf(stderr, "  bpc denominator: %.0f source bytes of %.0f in scored"
      " entries (%.1f%% covered, %.2f bytes/position)\n", sum_bytes,
      entry_bytes, (entry_bytes > 0.0) ? 100.0 * sum_bytes / entry_bytes : 0.0,
      (npairs > 0) ? sum_bytes / (double)npairs : 0.0);
    fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
      " (top1=%.1f%% bpc=%.3f) | novel %.1f%% (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ndeep / (double)npairs,
      (ndeep > 0) ? 100.0 * (double)ndeep_top1 / (double)ndeep : 0.0,
      (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
      100.0 * (double)nshallow / (double)npairs,
      (nshallow > 0) ? 100.0 * (double)nshallow_top1 / (double)nshallow : 0.0,
      (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0);
    if (0 != oracle) {
      int k;
      fprintf(stderr, "  order experts (bpc):");
      for (k = 1; k <= NGRAM_ORDER_MAX; ++k) {
        if (expert_bytes[k] > 0.0) {
          fprintf(stderr, " o%d=%.3f", k,
            expert_bits[k] / expert_bytes[k]);
        }
      }
      fprintf(stderr, " | ORACLE=%.3f (mixed=%.3f)\n",
        (sum_bytes > 0.0) ? oracle_bits / sum_bytes : 0.0,
        (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
      /**
       * The bound where it matters. A mechanism that only improves the attested
       * side cannot help generation on unseen material, and the novel column is
       * what any reweighting of this evidence could reach there.
       */
      fprintf(stderr, "  ORACLE split: attested=%.3f (of %.3f) |"
        " novel=%.3f (of %.3f) | extra-slot wins=%ld\n",
        (deep_bytes > 0.0) ? oracle_deep_bits / deep_bytes : 0.0,
        (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
        (shallow_bytes > 0.0) ? oracle_shallow_bits / shallow_bytes : 0.0,
        (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0,
        oracle_extra);
      fprintf(stderr, "  oracle picks:");
      for (k = 1; k <= NGRAM_ORDER_MAX; ++k) {
        if (oracle_pick[k] > 0) {
          fprintf(stderr, " o%d=%.1f%%", k,
            100.0 * (double)oracle_pick[k] / (double)npairs);
        }
      }
      fprintf(stderr, " mixed=%.1f%%\n",
        100.0 * (double)oracle_mixed / (double)npairs);
      fprintf(stderr, "  oracle choice vs matched context (o1-wins%%/mixed%%):");
      for (k = 0; k <= NGRAM_ORDER_MAX; ++k) {
        if (sel_total[k] > 0) {
          fprintf(stderr, " m%d=%.0f/%.0f", k,
            100.0 * (double)sel_low[k] / (double)sel_total[k],
            100.0 * (double)sel_mixed[k] / (double)sel_total[k]);
        }
      }
      /**
       * The reachability test for the oracle's remaining headroom. A share that
       * varies across buckets is a selector waiting to be written; a flat share
       * means this feature cannot tell those positions apart and the headroom
       * stays with the oracle.
       */
      if (0 < oracle_extra) {
        fprintf(stderr, "\n  kind-different slot cheapest, by matched context:");
        for (k = 0; k <= NGRAM_ORDER_MAX; ++k) {
          if (sel_total[k] > 0) {
            fprintf(stderr, " m%d=%.0f%%(n=%ld)", k,
              100.0 * (double)sel_extra[k] / (double)sel_total[k],
              sel_total[k]);
          }
        }
      }
      if (0 != bank && bank_n > 0) {
        int b, nslot = 0;
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (0 != ngram_bank_enabled(bank_slots, b)) ++nslot;
        }
        fprintf(stderr, "  BANK(%s rate=%.2f share=%.3f slots=%d)=%.3f"
          " | attested %.3f novel %.3f | mean weight (active%%):",
          (0 != bank_geo) ? "geo" : "linear", bank_rate, bank_share, nslot,
          (sum_bytes > 0.0) ? bank_bits / sum_bytes : 0.0,
          (deep_bytes > 0.0) ? bank_deep_bits / deep_bytes : 0.0,
          (shallow_bytes > 0.0) ? bank_shallow_bits / shallow_bytes : 0.0);
        /**
         * Mean weight alone cannot be read without the active share: a slot
         * that abstains often keeps its weight while contributing nothing, so a
         * healthy-looking weight can belong to an expert that almost never
         * spoke.
         */
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (bank_wsum[b] > 0.0) {
            fprintf(stderr, " %s=%.3f(%.0f%%)", ngram_bank_slotname(b),
              bank_wsum[b] / (double)bank_n,
              100.0 * (double)bank_active[b] / (double)bank_n);
          }
        }
        if (predict_nscored > 0) {
          /**
           * Frozen scoring reports no entropy: info aliases the context and
           * there is none, so the point query cannot return it. Saying FROZEN
           * rather than printing 0.000 keeps a missing diagnostic from reading
           * as a converged one.
           */
          if (0 != bank_frozen) {
            fprintf(stderr, "\n  predict slot: vocab=%d scored=%ld FROZEN"
              " (order-independent)", bank_vocab, predict_nscored);
          }
          else {
            const double ent = predict_entropy_sum / (double)predict_nscored;
            fprintf(stderr, "\n  predict slot: support=%d vocab=%d scored=%ld"
              " escape entropy=%.3f%s", predict_support_last, bank_vocab,
              predict_nscored, ent,
              (ent > NGRAM_PREDICT_ENTROPY_MAX) ? " (UNSETTLED)" : "");
          }
          /**
           * How much of the slot is the kNN vote. Bits are per position, not per
           * byte, so they compare only with each other - the question is
           * whether the positions the vote reached are cheaper than the rest and
           * how many there are.
           */
          if (0 < predict_att_n + predict_unatt_n) {
            const long tot = predict_att_n + predict_unatt_n;
            fprintf(stderr, "\n  predict slot local evidence: attested %.1f%%"
              " (%.3f bits/pos) | none %.1f%% (%.3f bits/pos)",
              100.0 * (double)predict_att_n / (double)tot,
              (0 < predict_att_n)
                ? predict_att_bits / (double)predict_att_n : 0.0,
              100.0 * (double)predict_unatt_n / (double)tot,
              (0 < predict_unatt_n)
                ? predict_unatt_bits / (double)predict_unatt_n : 0.0);
          }
        }
        fprintf(stderr, "\n  bank slots standalone (bpc where active):");
        for (b = 1; b <= NGRAM_BANK_LAST; ++b) {
          if (bank_slot_bytes[b] > 0.0) {
            fprintf(stderr, " %s=%.3f", ngram_bank_slotname(b),
              bank_slot_bits[b] / bank_slot_bytes[b]);
          }
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, " | RULE(m>=2 -> o%d)=%.3f\n", sel_order,
        (sum_bytes > 0.0) ? sel_bits / sum_bytes : 0.0);
    }
    if (nshallow > 0) {
      int m;
      fprintf(stderr, "  novel bucket by matched context:");
      for (m = 0; m <= NGRAM_ORDER_MAX; ++m) {
        if (novel_match[m] > 0) {
          fprintf(stderr, " n%d=%.1f%%/bpc=%.3f/bits=%.1f%%", m,
            100.0 * (double)novel_match[m] / (double)nshallow,
            (novel_match_bytes[m] > 0.0)
              ? novel_match_bits[m] / novel_match_bytes[m] : 0.0,
            (shallow_bits > 0.0)
              ? 100.0 * novel_match_bits[m] / shallow_bits : 0.0);
        }
      }
      fprintf(stderr, " | truncated %.1f%% of all (top1=%.1f%% bpc=%.3f)\n",
        100.0 * (double)ntrunc / (double)npairs,
        (ntrunc > 0) ? 100.0 * (double)ntrunc_top1 / (double)ntrunc : 0.0,
        (trunc_bytes > 0.0) ? trunc_bits / trunc_bytes : 0.0);
      fprintf(stderr, "  novel bucket composition:"
        " short-history %.1f%%/bits=%.1f%%"
        " saturated %.1f%%/bits=%.1f%% (n=%ld)\n",
        100.0 * (double)novel_short / (double)nshallow,
        (shallow_bits > 0.0) ? 100.0 * novel_short_bits / shallow_bits : 0.0,
        100.0 * (double)novel_sat / (double)nshallow,
        (shallow_bits > 0.0) ? 100.0 * novel_sat_bits / shallow_bits : 0.0,
        nshallow);
    }
    result = EXIT_SUCCESS;
  }
  file = fopen(converse_predict_eval_path(), "r");
  if (NULL != file) {
    long hnpairs = 0, htop1 = 0, htopk = 0;
    char line[EVAL_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      char* context;
      char* expected;
      char* sep;
      unsigned int prev2 = 0, prev1 = 0;
      unsigned int ids[NGRAM_TOPK];
      int n, rank;
      context = eval_trim(line);
      if ('\0' == *context || '#' == *context) continue;
      sep = strchr(context, '|');
      if (NULL == sep) continue;
      *sep = '\0';
      expected = eval_trim(sep + 1);
      context = eval_trim(context);
      if ('\0' == *context || '\0' == *expected) continue;
      ngram_last_context(lexicon, rules, nrules, context,
        (int)strlen(context), &prev2, &prev1);
      n = ngram_predict(model, prev2, prev1, order, ids, NGRAM_TOPK);
      ++hnpairs;
      for (rank = 0; rank < n; ++rank) {
        char word[LIBXS_LEXEME_MAXBYTES + 1];
        int len = 0;
        const char* text = libxs_lexicon_text(lexicon, ids[rank], &len, NULL);
        if (NULL != text && len > 0 && len < (int)sizeof(word)) {
          memcpy(word, text, (size_t)len);
          word[len] = '\0';
          if (0 != text_contains_word_ci(expected, (int)strlen(expected),
            word))
          {
            if (0 == rank) ++htop1;
            ++htopk;
            break;
          }
        }
      }
    }
    if (hnpairs > 0) {
      fprintf(stdout,
        "predict-ngram[%s-fixture]: top1=%.1f%% top%d=%.1f%% n=%ld\n",
        kind, 100.0 * (double)htop1 / (double)hnpairs, NGRAM_TOPK,
        100.0 * (double)htopk / (double)hnpairs, hnpairs);
    }
    fclose(file);
  }
  libxs_predict_prob_destroy(bank_context);
  return result;
}


static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout,
  int ctxlen)
{
  int ninputs = (0 != use_emb) ? 2 * TOKEN_EMB_DIM : 2;
  libxs_predict_t* model = libxs_predict_create(ninputs, 1);
  double decay = knnlm_decay();
  long pushed = 0;
  if (ctxlen < 2) ctxlen = 2;
  if (NULL != model && NULL != corpus && NULL != profile) {
    const void* key = NULL;
    size_t cursor = 0;
    corpus_entry_t scratch;
    long index = 0;
    void* value;
    libxs_predict_set_mode(model, profile->mode);
    libxs_predict_set_decompose(model, profile->decompose);
    if (0 != use_emb) {
      double weights[2 * TOKEN_EMB_DIM];
      int dim;
      for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
        weights[dim] = 1.0;
        weights[TOKEN_EMB_DIM + dim] = 2.0;
      }
      libxs_predict_set_weights(model, weights);
    }
    value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
      {
        size_t pos;
        unsigned int prev1 = 0, prev2 = 0;
        unsigned int hist[TOKEN_CTX_MAX];
        int hlen = 0;
        for (pos = 0; pos < stream.size && pushed < TOKEN_PREDICT_TRAIN_MAX;
          ++pos)
        {
          const libxs_lexeme_t* lex = stream.data + pos;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            if (0 != prev1) {
              double in[2 * TOKEN_EMB_DIM];
              double out[1];
              if (ctxlen > 2 && 0 != use_emb) {
                knnlm_ctx_vector(hist, hlen, ctxlen, decay, in);
              }
              else token_input_vector(prev2, prev1, use_emb, in);
              out[0] = (double)lex->id;
              if (EXIT_SUCCESS == libxs_predict_push(NULL, model, in, out)) {
                ++pushed;
              }
            }
            prev2 = prev1;
            prev1 = lex->id;
            ngram_hist_push(hist, &hlen, TOKEN_CTX_MAX, lex->id);
          }
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
            prev1 = 0;
            prev2 = 0;
            hlen = 0;
          }
        }
      }
      libxs_lexeme_stream_release(&stream);
      ++index;
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
    if (pushed <= 0 || EXIT_SUCCESS != libxs_predict_build(model,
      profile->clusters, profile->order, profile->quality))
    {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  else {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


static unsigned int token_predict_next(const libxs_predict_t* model,
  unsigned int prev2, unsigned int prev1, int use_emb)
{
  double in[2 * TOKEN_EMB_DIM];
  double out[1];
  token_input_vector(prev2, prev1, use_emb, in);
  out[0] = 0.0;
  libxs_predict_eval(NULL, model, in, out, NULL, 1);
  return (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
}


static int token_predict_eval(const libxs_predict_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int use_emb, int holdout,
  const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, seen = 0, index = 0;
  int stride = predict_eval_stride();
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
    {
      size_t pos;
      unsigned int prev1 = 0, prev2 = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (0 != prev1) {
            if (0 == (seen % stride)) {
              unsigned int pred = token_predict_next(model, prev2, prev1,
                use_emb);
              ++npairs;
              if (pred == lex->id) ++ntop1;
            }
            ++seen;
          }
          prev2 = prev1;
          prev1 = lex->id;
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
          prev1 = 0;
          prev2 = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout, "predict-%s%s: top1=%.1f%% n=%ld (stride=%d)\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, npairs,
      stride);
    result = EXIT_SUCCESS;
  }
  return result;
}


static void token_complete(const libxs_predict_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int use_emb, const char* text, int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int pred;
  int len = 0;
  const char* word;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  if (0 == prev1) {
    printf("(no continuation)\n");
    return;
  }
  pred = token_predict_next(model, prev2, prev1, use_emb);
  word = (0 != pred) ? libxs_lexicon_text(lexicon, pred, &len, NULL) : NULL;
  printf("next:");
  if (NULL != word && len > 0) printf(" %.*s", len, word);
  printf("\n");
  { unsigned int step2 = prev2, step1 = prev1;
    int step;
    printf("greedy:");
    for (step = 0; step < 12; ++step) {
      unsigned int p = token_predict_next(model, step2, step1, use_emb);
      int l = 0;
      const char* w = (0 != p) ? libxs_lexicon_text(lexicon, p, &l, NULL)
        : NULL;
      if (NULL == w || l <= 0) break;
      printf(" %.*s", l, w);
      step2 = step1;
      step1 = p;
    }
    printf("\n");
  }
}


static int ngram_candidates(libxs_registry_t* model, unsigned int prev2,
  unsigned int prev1, int order, unsigned int ids[], double relfreq[],
  int provenance[], int* ctx_total, int k)
{
  int result = 0;
  const ngram_entry_t* entry = NULL;
  int prov = 0;
  if (order >= 2 && 0 != prev2) {
    entry = ngram_lookup(model, prev2, prev1);
    if (NULL != entry) prov = 2;
  }
  if (NULL == entry) {
    entry = ngram_lookup(model, 0, prev1);
    if (NULL != entry) prov = 1;
  }
  if (NULL != ctx_total) *ctx_total = (NULL != entry) ? (int)entry->total : 0;
  if (NULL != entry) {
    unsigned int taken[NGRAM_SUCC_MAX];
    unsigned int nsucc = entry->nsucc;
    unsigned int slot;
    double total = (entry->total > 0) ? (double)entry->total : 1.0;
    for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
      taken[slot] = 0;
    }
    while (result < k) {
      int best = -1;
      for (slot = 0; slot < nsucc && slot < NGRAM_SUCC_MAX; ++slot) {
        if (0 == taken[slot]
          && (best < 0 || entry->succ[slot].count > entry->succ[best].count))
        {
          best = (int)slot;
        }
      }
      if (best < 0) break;
      taken[best] = 1;
      ids[result] = entry->succ[best].id;
      relfreq[result] = (double)entry->succ[best].count / total;
      provenance[result] = prov;
      ++result;
    }
  }
  else {
    int slot;
    for (slot = 0; slot < k && slot < converse_ngram_handle()->backoff_count; ++slot) {
      ids[slot] = converse_ngram_handle()->backoff_ids[slot];
      relfreq[slot] = 0.0;
      provenance[slot] = 0;
      ++result;
    }
  }
  return result;
}


static int token_is_stop(libxs_lexicon_t* lexicon, unsigned int id)
{
  unsigned int flags = 0;
  int len = 0;
  if (NULL != lexicon && 0 != id) {
    libxs_lexicon_text(lexicon, id, &len, &flags);
  }
  return (0 != (flags & LIBXS_LEXEME_STOP)) ? 1 : 0;
}


static void rerank_features(libxs_registry_t* ngram, unsigned int prev1,
  unsigned int id, double relfreq, int rank, int provenance, int ctx_total,
  double ctx_top, libxs_lexicon_t* lexicon, double inputs[RERANK_INPUTS])
{
  double prior = ngram_unigram_prior(id);
  double reliability = (double)ctx_total / RERANK_RELIABILITY;
  double lift = relfreq / (prior + 1e-6);
  if (reliability > 1.0) reliability = 1.0;
  if (lift > RERANK_LIFT_MAX) lift = RERANK_LIFT_MAX;
  inputs[0] = relfreq;
  inputs[1] = (double)rank / (double)NGRAM_SUCC_MAX;
  inputs[2] = (double)provenance / 2.0;
  inputs[3] = (double)token_is_stop(lexicon, id);
  inputs[4] = ngram_pair_relfreq(ngram, 0, prev1, id);
  inputs[5] = prior;
  inputs[6] = reliability;
  inputs[7] = ctx_top;
  inputs[8] = lift;
}


static libxs_predict_t* rerank_build(const libxs_registry_t* corpus,
  libxs_registry_t* ngram, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order,
  const answer_predict_profile_t* profile, int holdout)
{
  libxs_predict_t* model = libxs_predict_create(RERANK_INPUTS, 1);
  long pushed = 0;
  if (NULL != model && NULL != corpus && NULL != ngram && NULL != profile) {
    const void* key = NULL;
    size_t cursor = 0;
    corpus_entry_t scratch;
    long index = 0;
    void* value;
    static const double weights[RERANK_INPUTS] = {
      3.0, 0.5, 1.0, 0.4, 1.5, 0.6, 0.8, 1.0, 2.0
    };
    libxs_predict_set_mode(model, profile->mode);
    libxs_predict_set_decompose(model, profile->decompose);
    libxs_predict_set_weights(model, weights);
    if (0.0 != profile->smooth) libxs_predict_set_smooth(model,
      profile->smooth);
    value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
      {
        size_t pos;
        unsigned int prev1 = 0, prev2 = 0;
        for (pos = 0; pos < stream.size && pushed < TOKEN_PREDICT_TRAIN_MAX;
          ++pos)
        {
          const libxs_lexeme_t* lex = stream.data + pos;
          if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 != lex->id)
          {
            if (0 != prev1) {
              unsigned int ids[NGRAM_SUCC_MAX];
              double relfreq[NGRAM_SUCC_MAX];
              int prov[NGRAM_SUCC_MAX];
              int ctx_total = 0;
              int n = ngram_candidates(ngram, prev2, prev1, order, ids,
                relfreq, prov, &ctx_total, NGRAM_SUCC_MAX);
              double ctx_top = (n > 0) ? relfreq[0] : 0.0;
              int rank;
              for (rank = 0; rank < n; ++rank) {
                double in[RERANK_INPUTS];
                double out[1];
                rerank_features(ngram, prev1, ids[rank], relfreq[rank], rank,
                  prov[rank], ctx_total, ctx_top, lexicon, in);
                out[0] = (ids[rank] == lex->id) ? 1.0 : 0.0;
                libxs_predict_push(NULL, model, in, out);
                ++pushed;
              }
            }
            prev2 = prev1;
            prev1 = lex->id;
          }
          if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
            prev1 = 0;
            prev2 = 0;
          }
        }
      }
      libxs_lexeme_stream_release(&stream);
      ++index;
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
    if (pushed <= 0 || EXIT_SUCCESS != libxs_predict_build(model,
      profile->clusters, profile->order, profile->quality))
    {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  else {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


static int rerank_topk(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  unsigned int prev2, unsigned int prev1, int order, unsigned int out_ids[],
  int k)
{
  unsigned int ids[NGRAM_SUCC_MAX];
  double relfreq[NGRAM_SUCC_MAX];
  int prov[NGRAM_SUCC_MAX];
  double score[NGRAM_SUCC_MAX];
  int taken[NGRAM_SUCC_MAX];
  int ctx_total = 0;
  int n = ngram_candidates(ngram, prev2, prev1, order, ids, relfreq, prov,
    &ctx_total, NGRAM_SUCC_MAX);
  double ctx_top = (n > 0) ? relfreq[0] : 0.0;
  int result = 0;
  int rank;
  for (rank = 0; rank < n; ++rank) {
    double in[RERANK_INPUTS];
    double out[1];
    rerank_features(ngram, prev1, ids[rank], relfreq[rank], rank, prov[rank],
      ctx_total, ctx_top, lexicon, in);
    out[0] = 0.0;
    libxs_predict_eval(NULL, reranker, in, out, NULL, 1);
    score[rank] = out[0];
    taken[rank] = 0;
  }
  while (result < k && result < n) {
    int best = -1;
    for (rank = 0; rank < n; ++rank) {
      if (0 == taken[rank] && (best < 0 || score[rank] > score[best])) {
        best = rank;
      }
    }
    if (best < 0) break;
    taken[best] = 1;
    out_ids[result] = ids[best];
    ++result;
  }
  return result;
}


static int rerank_eval(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, seen = 0, index = 0;
  int stride = predict_eval_stride();
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  if (NULL == ngram || NULL == reranker || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
    {
      size_t pos;
      unsigned int prev1 = 0, prev2 = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (0 != prev1) {
            if (0 == (seen % stride)) {
              unsigned int ids[NGRAM_TOPK];
              int n = rerank_topk(ngram, reranker, lexicon, prev2, prev1,
                order, ids, NGRAM_TOPK);
              int rank;
              ++npairs;
              for (rank = 0; rank < n; ++rank) {
                if (ids[rank] == lex->id) {
                  if (0 == rank) ++ntop1;
                  ++ntopk;
                  break;
                }
              }
            }
            ++seen;
          }
          prev2 = prev1;
          prev1 = lex->id;
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) {
          prev1 = 0;
          prev2 = 0;
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-%s%s: top1=%.1f%% top%d=%.1f%% n=%ld (stride=%d)\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      stride);
    result = EXIT_SUCCESS;
  }
  return result;
}


static void rerank_complete(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int ids[NGRAM_TOPK];
  int n, i;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  n = rerank_topk(ngram, reranker, lexicon, prev2, prev1, order, ids,
    NGRAM_TOPK);
  if (n <= 0) {
    printf("(no continuation)\n");
    return;
  }
  printf("next:");
  for (i = 0; i < n; ++i) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[i], &len, NULL);
    if (NULL != word && len > 0) printf(" %.*s", len, word);
  }
  printf("\n");
  { unsigned int step2 = prev2, step1 = prev1;
    int step;
    printf("greedy:");
    for (step = 0; step < 12; ++step) {
      unsigned int step_ids[1];
      int len = 0;
      const char* word;
      if (0 == rerank_topk(ngram, reranker, lexicon, step2, step1, order,
        step_ids, 1)) break;
      word = libxs_lexicon_text(lexicon, step_ids[0], &len, NULL);
      if (NULL == word || len <= 0) break;
      printf(" %.*s", len, word);
      step2 = step1;
      step1 = step_ids[0];
    }
    printf("\n");
  }
}


static void knnlm_cache_free(void)
{
  free(knnlm_cache_in);
  free(knnlm_cache_next);
  free(knnlm_dyn_in);
  free(knnlm_dyn_next);
  free(knnlm_ann_code);
  free(knnlm_ann_order);
  knnlm_cache_in = NULL;
  knnlm_cache_next = NULL;
  knnlm_dyn_in = NULL;
  knnlm_dyn_next = NULL;
  knnlm_ann_code = NULL;
  knnlm_ann_order = NULL;
  knnlm_cache_size = 0;
  knnlm_dyn_size = 0;
  knnlm_dyn_cap = 0;
  knnlm_ann_size = 0;
  knnlm_cache_model = NULL;
}

/**
 * Approximate NN over the static datastore is enabled by CONVERSE_KNNLM_ANN;
 * default off keeps the exact brute-force scan (bit-identical results).
 */
static int knnlm_ann_mode(void)
{
  const char* env = getenv("CONVERSE_KNNLM_ANN");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}

/**
 * Quantize a context vector's leading dims to a Hilbert code: locality in the
 * embedding space becomes locality along the 1-D code so a window around the
 * query code holds its near neighbors. Values are the prev1 half of the input
 * (the more predictive token, per the 4x weighting in knnlm_scan).
 */
static uint64_t knnlm_ann_encode(const double* in)
{
  unsigned int coords[KNNLM_ANN_DIMS];
  const double* v = in + TOKEN_EMB_DIM;
  int d;
  double scale = (double)((1 << KNNLM_ANN_BITS) - 1);
  for (d = 0; d < KNNLM_ANN_DIMS; ++d) {
    double x = (d < TOKEN_EMB_DIM) ? v[d] : 0.0;
    double u = 0.5 * (x + 1.0);
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    coords[d] = (unsigned int)(u * scale + 0.5);
  }
  return libxs_hilbert_bits(coords, KNNLM_ANN_DIMS, KNNLM_ANN_BITS);
}


static int knnlm_ann_cmp(const void* a, const void* b)
{
  int ia = *(const int*)a, ib = *(const int*)b;
  uint64_t ca = knnlm_ann_code[ia], cb = knnlm_ann_code[ib];
  if (ca < cb) return -1;
  if (ca > cb) return 1;
  return 0;
}

/* Build the sorted Hilbert index over the current static cache. */
static void knnlm_ann_build(void)
{
  free(knnlm_ann_code);
  free(knnlm_ann_order);
  knnlm_ann_code = NULL;
  knnlm_ann_order = NULL;
  knnlm_ann_size = 0;
  if (knnlm_cache_size > 0) {
    knnlm_ann_code = (uint64_t*)malloc((size_t)knnlm_cache_size
      * sizeof(*knnlm_ann_code));
    knnlm_ann_order = (int*)malloc((size_t)knnlm_cache_size
      * sizeof(*knnlm_ann_order));
    if (NULL != knnlm_ann_code && NULL != knnlm_ann_order) {
      int i;
      for (i = 0; i < knnlm_cache_size; ++i) {
        knnlm_ann_code[i] = knnlm_ann_encode(knnlm_cache_in
          + (size_t)i * 2 * TOKEN_EMB_DIM);
        knnlm_ann_order[i] = i;
      }
      qsort(knnlm_ann_order, (size_t)knnlm_cache_size,
        sizeof(*knnlm_ann_order), knnlm_ann_cmp);
      knnlm_ann_size = knnlm_cache_size;
    }
    else {
      free(knnlm_ann_code);
      free(knnlm_ann_order);
      knnlm_ann_code = NULL;
      knnlm_ann_order = NULL;
    }
  }
}

/**
 * Exact-distance rerank of one candidate against the query into the running
 * top-K (same metric and heap discipline as knnlm_scan).
 */
static void knnlm_ann_consider(const double* in, int idx,
  unsigned int near_next[], double near_dist[], int* nnear)
{
  const double* entry = knnlm_cache_in + (size_t)idx * 2 * TOKEN_EMB_DIM;
  unsigned int cand = knnlm_cache_next[idx];
  double dist = 0.0;
  int dim, slot;
  if (0 == cand) return;
  for (dim = 0; dim < TOKEN_EMB_DIM; ++dim) {
    double d2 = in[dim] - entry[dim];
    double d1 = in[TOKEN_EMB_DIM + dim] - entry[TOKEN_EMB_DIM + dim];
    dist += d2 * d2 + 4.0 * d1 * d1;
  }
  for (slot = *nnear; slot > 0; --slot) {
    if (dist >= near_dist[slot - 1]) break;
  }
  if (slot < KNNLM_K) {
    int move = (*nnear < KNNLM_K) ? *nnear : (KNNLM_K - 1);
    for (; move > slot; --move) {
      near_next[move] = near_next[move - 1];
      near_dist[move] = near_dist[move - 1];
    }
    near_next[slot] = cand;
    near_dist[slot] = dist;
    if (*nnear < KNNLM_K) ++*nnear;
  }
}

/**
 * Retrieve the static-cache neighbors of the query from a window around its
 * Hilbert code, then exact-rerank them. Replaces the O(N) scan of the static
 * cache; the dynamic store is still scanned brute-force by the caller.
 */
static void knnlm_ann_scan(const double* in, unsigned int near_next[],
  double near_dist[], int* nnear)
{
  if (knnlm_ann_size > 0) {
    uint64_t q = knnlm_ann_encode(in);
    int lo = 0, hi = knnlm_ann_size - 1, mid, left, right, taken;
    while (lo <= hi) {
      mid = (lo + hi) / 2;
      if (knnlm_ann_code[knnlm_ann_order[mid]] < q) lo = mid + 1;
      else hi = mid - 1;
    }
    if (lo > knnlm_ann_size - 1) lo = knnlm_ann_size - 1;
    left = lo - 1;
    right = lo;
    taken = 0;
    while (taken < KNNLM_ANN_WINDOW && (left >= 0 || right < knnlm_ann_size)) {
      int pick_left = 0;
      if (left >= 0 && right < knnlm_ann_size) {
        uint64_t cl = knnlm_ann_code[knnlm_ann_order[left]];
        uint64_t cr = knnlm_ann_code[knnlm_ann_order[right]];
        uint64_t dl = (q > cl) ? (q - cl) : (cl - q);
        uint64_t dr = (cr > q) ? (cr - q) : (q - cr);
        pick_left = (dl <= dr) ? 1 : 0;
      }
      else if (left >= 0) pick_left = 1;
      if (0 != pick_left) {
        knnlm_ann_consider(in, knnlm_ann_order[left], near_next, near_dist,
          nnear);
        --left;
      }
      else {
        knnlm_ann_consider(in, knnlm_ann_order[right], near_next, near_dist,
          nnear);
        ++right;
      }
      ++taken;
    }
  }
}


static void knnlm_dyn_reset(void)
{
  knnlm_dyn_size = 0;
}


static void knnlm_dyn_insert(const unsigned int hist[], int hlen, int ctxlen,
  unsigned int next)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  if (0 != prev1 && 0 != next) {
    if (knnlm_dyn_size >= knnlm_dyn_cap) {
      int cap = (knnlm_dyn_cap > 0) ? (2 * knnlm_dyn_cap) : 4096;
      double* nin = (double*)realloc(knnlm_dyn_in,
        (size_t)cap * 2 * TOKEN_EMB_DIM * sizeof(double));
      unsigned int* nnext = (unsigned int*)realloc(knnlm_dyn_next,
        (size_t)cap * sizeof(unsigned int));
      if (NULL == nin || NULL == nnext) {
        free(nin); free(nnext);
        return;
      }
      knnlm_dyn_in = nin;
      knnlm_dyn_next = nnext;
      knnlm_dyn_cap = cap;
    }
    if (ctxlen > 2) {
      knnlm_ctx_vector(hist, hlen, ctxlen, knnlm_decay(),
        knnlm_dyn_in + (size_t)knnlm_dyn_size * 2 * TOKEN_EMB_DIM);
    }
    else token_input_vector(prev2, prev1, 1,
      knnlm_dyn_in + (size_t)knnlm_dyn_size * 2 * TOKEN_EMB_DIM);
    knnlm_dyn_next[knnlm_dyn_size] = next;
    ++knnlm_dyn_size;
  }
}

/**
 * Number of retrieval heads (CONVERSE_KNNLM_HEADS, default 1 = bit-exact). With
 * h heads the embedding is split into h contiguous subspaces of TOKEN_EMB_DIM/h
 * dimensions; each head retrieves its own top-K over its own subspace distance,
 * and the votes are summed. This is multi-head attention's decomposition minus
 * the learned projections: the subspaces are fixed slices of the factorization
 * rather than anything W_Q/W_K could rotate into place.
 */
static int knnlm_heads(void)
{
  int result = 1;
  const char* env = getenv("CONVERSE_KNNLM_HEADS");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= TOKEN_EMB_DIM && 0 == (TOKEN_EMB_DIM % v)) result = v;
  }
  return result;
}


static void knnlm_scan_head(const double* in, const double* cin,
  const unsigned int* cnext, int count, int dim_begin, int dim_end,
  unsigned int near_next[], double near_dist[], int* nnear)
{
  int i, dim;
  for (i = 0; i < count; ++i) {
    const double* entry = cin + (size_t)i * 2 * TOKEN_EMB_DIM;
    double dist = 0.0;
    int slot;
    if (0 == cnext[i]) continue;
    for (dim = dim_begin; dim < dim_end; ++dim) {
      double d2 = in[dim] - entry[dim];
      double d1 = in[TOKEN_EMB_DIM + dim] - entry[TOKEN_EMB_DIM + dim];
      dist += d2 * d2 + 4.0 * d1 * d1;
    }
    for (slot = *nnear; slot > 0; --slot) {
      if (dist >= near_dist[slot - 1]) break;
    }
    if (slot < KNNLM_K) {
      int move = (*nnear < KNNLM_K) ? *nnear : (KNNLM_K - 1);
      for (; move > slot; --move) {
        near_next[move] = near_next[move - 1];
        near_dist[move] = near_dist[move - 1];
      }
      near_next[slot] = cnext[i];
      near_dist[slot] = dist;
      if (*nnear < KNNLM_K) ++*nnear;
    }
  }
}


static int token_proj_mode(void)
{
  const char* env = getenv("CONVERSE_KNNLM_PROJ");
  return (NULL != env && '0' != *env) ? 1 : 0;
}


static void token_proj_apply(double* vec)
{
  if (0 != token_proj_ready) {
    int d;
    for (d = 0; d < 2 * TOKEN_EMB_DIM; ++d) vec[d] *= token_proj[d];
  }
}


static void token_proj_build(const libxs_predict_t* store)
{
  enum { PROJ_CLASS_MAX = 256, PROJ_MINCOUNT = 20 };
  const int dims = 2 * TOKEN_EMB_DIM;
  libxs_predict_query_t stats;
  token_proj_ready = 0;
  if (0 == token_proj_mode() || NULL == store) return;
  libxs_predict_query(store, &stats);
  if (stats.nentries < PROJ_CLASS_MAX) return;
  { /* two passes: class means, then scatter about them */
    static double mean[PROJ_CLASS_MAX][2 * TOKEN_EMB_DIM];
    static double grand[2 * TOKEN_EMB_DIM];
    static double between[2 * TOKEN_EMB_DIM], within[2 * TOKEN_EMB_DIM];
    unsigned int class_id[PROJ_CLASS_MAX];
    long count[PROJ_CLASS_MAX];
    long freq_total = 0;
    int nclasses = 0, ci, d, i;
    /**
     * Class granularity: a next token becomes its own class only once it recurs
     * often enough to estimate a mean; everything rarer is pooled into class 0.
     * With thousands of distinct next tokens over tens of thousands of pairs the
     * within-class scatter would otherwise be estimated from single members and
     * be degenerate.
     */
    memset(mean, 0, sizeof(mean));
    memset(grand, 0, sizeof(grand));
    memset(between, 0, sizeof(between));
    memset(within, 0, sizeof(within));
    memset(count, 0, sizeof(count));
    class_id[0] = 0; /* pooled rare-token class */
    nclasses = 1;
    for (i = 0; i < stats.nentries; ++i) {
      double in[2 * TOKEN_EMB_DIM], out[1] = { 0 };
      unsigned int next;
      long seen;
      libxs_predict_get(store, i, in, out);
      next = (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
      if (0 == next) continue;
      seen = (long)(ngram_unigram_prior(next) * (double)stats.nentries);
      ci = 0;
      if (seen >= PROJ_MINCOUNT) {
        int found = 0;
        for (ci = 1; ci < nclasses; ++ci) {
          if (class_id[ci] == next) { found = 1; break; }
        }
        if (0 == found) {
          if (nclasses < PROJ_CLASS_MAX) {
            class_id[nclasses] = next;
            ci = nclasses++;
          }
          else ci = 0;
        }
      }
      ++count[ci];
      ++freq_total;
      for (d = 0; d < dims; ++d) {
        mean[ci][d] += in[d];
        grand[d] += in[d];
      }
    }
    if (freq_total < PROJ_CLASS_MAX || nclasses < 2) return;
    for (ci = 0; ci < nclasses; ++ci) {
      if (count[ci] > 0) {
        for (d = 0; d < dims; ++d) mean[ci][d] /= (double)count[ci];
      }
    }
    for (d = 0; d < dims; ++d) grand[d] /= (double)freq_total;
    for (ci = 0; ci < nclasses; ++ci) {
      for (d = 0; d < dims; ++d) {
        const double diff = mean[ci][d] - grand[d];
        between[d] += (double)count[ci] * diff * diff;
      }
    }
    for (i = 0; i < stats.nentries; ++i) {
      double in[2 * TOKEN_EMB_DIM], out[1] = { 0 };
      unsigned int next;
      long seen;
      libxs_predict_get(store, i, in, out);
      next = (out[0] > 0.0) ? (unsigned int)(out[0] + 0.5) : 0;
      if (0 == next) continue;
      seen = (long)(ngram_unigram_prior(next) * (double)stats.nentries);
      ci = 0;
      if (seen >= PROJ_MINCOUNT) {
        for (ci = 1; ci < nclasses; ++ci) {
          if (class_id[ci] == next) break;
        }
        if (ci >= nclasses) ci = 0;
      }
      for (d = 0; d < dims; ++d) {
        const double diff = in[d] - mean[ci][d];
        within[d] += diff * diff;
      }
    }
    { double wmax = 0.0;
      for (d = 0; d < dims; ++d) {
        const double ratio = (within[d] > 0.0) ? (between[d] / within[d]) : 0.0;
        token_proj[d] = sqrt(ratio);
        if (token_proj[d] > wmax) wmax = token_proj[d];
      }
      if (wmax <= 0.0) return;
      for (d = 0; d < dims; ++d) token_proj[d] /= wmax;
      token_proj_ready = 1;
      fprintf(stderr, "projection: %d classes, weights", nclasses);
      for (d = 0; d < dims; d += TOKEN_EMB_DIM / 4) {
        fprintf(stderr, " %.2f", token_proj[d]);
      }
      fprintf(stderr, " (every %d dims)\n", TOKEN_EMB_DIM / 4);
    }
  }
}


static void knnlm_cache_build(const libxs_predict_t* store)
{
  libxs_predict_query_t stats;
  knnlm_cache_free();
  libxs_predict_query(store, &stats);
  if (stats.nentries > 0) {
    knnlm_cache_in = (double*)malloc((size_t)stats.nentries
      * 2 * TOKEN_EMB_DIM * sizeof(double));
    knnlm_cache_next = (unsigned int*)malloc((size_t)stats.nentries
      * sizeof(unsigned int));
    if (NULL != knnlm_cache_in && NULL != knnlm_cache_next) {
      int i;
      for (i = 0; i < stats.nentries; ++i) {
        double out[1] = { 0 };
        libxs_predict_get(store, i,
          knnlm_cache_in + (size_t)i * 2 * TOKEN_EMB_DIM, out);
        knnlm_cache_next[i] = (out[0] > 0.0)
          ? (unsigned int)(out[0] + 0.5) : 0;
      }
      knnlm_cache_size = stats.nentries;
      knnlm_cache_model = store;
      /* Fit the projection from the datastore, then map the keys into it; the
       * query is mapped in knnlm_vote so both sides share one space. */
      token_proj_build(store);
      if (0 != token_proj_ready) {
        for (i = 0; i < stats.nentries; ++i) {
          token_proj_apply(knnlm_cache_in + (size_t)i * 2 * TOKEN_EMB_DIM);
        }
      }
    }
    else knnlm_cache_free();
  }
}

/**
 * Control distributions replacing the retrieved vote, to test whether the
 * kNN-LM gain comes from the learned metric or merely from interpolating some
 * smoother distribution into a sparse backoff estimator.
 * 1 = global unigram prior (no context, no retrieval at all).
 * 2 = K neighbors drawn by a coprime stride over the datastore (retrieval
 *     machinery and vote arithmetic intact, metric replaced by an arbitrary
 *     but reproducible selection).
 */
/**
 * Softmax temperature for the retrieval vote (CONVERSE_KNNLM_TEMP). Zero keeps
 * the historical inverse-distance kernel 1/(eps+d), bit-exact. A positive value
 * weights neighbor i by exp(-d_i/T) after subtracting the minimum distance for
 * numerical stability, which is attention's weighting rather than an ad-hoc
 * one: the temperature then controls how peaked the vote is, exactly as it
 * does in a transformer's attention.
 */
static double knnlm_temp(void)
{
  double result = 0.0;
  const char* env = getenv("CONVERSE_KNNLM_TEMP");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v > 0.0) result = v;
  }
  return result;
}


static int knnlm_control(void)
{
  const char* env = getenv("CONVERSE_KNNLM_CONTROL");
  int result = 0;
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 0 && v <= 2) result = v;
  }
  return result;
}


static int knnlm_vote_control(int mode, unsigned int position,
  unsigned int vote_ids[], double vote_p[], int maxvote)
{
  int result = 0;
  if (1 == mode) {
    const unsigned int vocab = converse_ngram_handle()->unifreq_size;
    unsigned int id;
    double total = 0.0;
    for (id = 1; id <= vocab; ++id) {
      const double p = ngram_unigram_prior(id);
      if (p > 0.0) {
        int slot = (result < maxvote) ? result : (maxvote - 1);
        if (result < maxvote) ++result;
        else if (vote_p[slot] >= p) continue;
        while (slot > 0 && vote_p[slot - 1] < p) {
          vote_ids[slot] = vote_ids[slot - 1];
          vote_p[slot] = vote_p[slot - 1];
          --slot;
        }
        vote_ids[slot] = id;
        vote_p[slot] = p;
      }
    }
    for (id = 0; id < (unsigned int)result; ++id) total += vote_p[id];
    if (total > 0.0) {
      for (id = 0; id < (unsigned int)result; ++id) vote_p[id] /= total;
    }
  }
  else if (2 == mode && knnlm_cache_size > 0) {
    const size_t stride = libxs_coprime_bias((size_t)knnlm_cache_size, -1.0);
    unsigned int uniq_id[KNNLM_K];
    double uniq_w[KNNLM_K];
    int nuniq = 0, i, j;
    double wtotal = 0.0;
    for (i = 0; i < KNNLM_K; ++i) {
      const size_t at = (stride * (position + (unsigned int)i) + 1)
        % (size_t)knnlm_cache_size;
      const unsigned int next = knnlm_cache_next[at];
      if (0 == next) continue;
      for (j = 0; j < nuniq; ++j) {
        if (uniq_id[j] == next) break;
      }
      if (j == nuniq) {
        uniq_id[j] = next;
        uniq_w[j] = 0.0;
        ++nuniq;
      }
      uniq_w[j] += 1.0;
      wtotal += 1.0;
    }
    while (result < maxvote && result < nuniq) {
      int best = -1;
      for (j = 0; j < nuniq; ++j) {
        if (0.0 <= uniq_w[j] && (best < 0 || uniq_w[j] > uniq_w[best])) {
          best = j;
        }
      }
      if (best < 0 || uniq_w[best] <= 0.0) break;
      vote_ids[result] = uniq_id[best];
      vote_p[result] = uniq_w[best] / wtotal;
      uniq_w[best] = -1.0;
      ++result;
    }
  }
  return result;
}


static int knnlm_vote(const libxs_predict_t* store, const unsigned int hist[],
  int hlen, int ctxlen, unsigned int vote_ids[], double vote_p[], int maxvote)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  int result = 0;
  if (NULL != store && 0 != prev1) {
    double in[2 * TOKEN_EMB_DIM];
    unsigned int near_next[KNNLM_K];
    double near_dist[KNNLM_K];
    unsigned int uniq_id[KNNLM_K];
    double uniq_w[KNNLM_K];
    int nnear = 0, nuniq = 0, i, j;
    double wtotal = 0.0, dmin = 0.0;
    const double temp = knnlm_temp();
    int ann = knnlm_ann_mode();
    const int control = knnlm_control();
    if (knnlm_cache_model != store) {
      knnlm_cache_build(store);
      if (0 != ann) knnlm_ann_build();
    }
    if (0 != control) {
      result = knnlm_vote_control(control, prev1 + 31u * prev2, vote_ids,
        vote_p, maxvote);
      nnear = 0;
      nuniq = 0;
    }
    else {
      const int heads = knnlm_heads();
      const int span = TOKEN_EMB_DIM / heads;
      int head;
      if (ctxlen > 2) knnlm_ctx_vector(hist, hlen, ctxlen, knnlm_decay(), in);
      else token_input_vector(prev2, prev1, 1, in);
      token_proj_apply(in);
      /**
       * One retrieval per head over its own subspace, votes accumulated into
       * the shared distribution. With heads==1 the loop performs exactly the
       * single full-width scan it replaced (and keeps using the ANN index,
       * which is built over the full vector).
       */
      for (head = 0; head < heads; ++head) {
        nnear = 0;
        if (1 == heads && 0 != ann) {
          knnlm_ann_scan(in, near_next, near_dist, &nnear);
        }
        else {
          knnlm_scan_head(in, knnlm_cache_in, knnlm_cache_next,
            knnlm_cache_size, head * span, (head + 1) * span,
            near_next, near_dist, &nnear);
        }
        knnlm_scan_head(in, knnlm_dyn_in, knnlm_dyn_next, knnlm_dyn_size,
          head * span, (head + 1) * span, near_next, near_dist, &nnear);
        /* Softmax needs the minimum distance first (stability); the historical
         * inverse-distance kernel needs no such pass. */
        if (temp > 0.0) {
          for (i = 0; i < nnear; ++i) {
            if (i == 0 || near_dist[i] < dmin) dmin = near_dist[i];
          }
        }
        for (i = 0; i < nnear; ++i) {
          unsigned int next = near_next[i];
          double w = (temp > 0.0)
            ? exp(-(near_dist[i] - dmin) / temp)
            : (1.0 / (0.05 + near_dist[i]));
          for (j = 0; j < nuniq; ++j) {
            if (uniq_id[j] == next) break;
          }
          if (j == nuniq && nuniq < KNNLM_K) {
            uniq_id[j] = next;
            uniq_w[j] = 0.0;
            ++nuniq;
          }
          if (j < nuniq) {
            uniq_w[j] += w;
            wtotal += w;
          }
        }
      }
      while (result < maxvote && result < nuniq) {
        int best = -1;
        for (j = 0; j < nuniq; ++j) {
          if (0.0 <= uniq_w[j] && (best < 0 || uniq_w[j] > uniq_w[best])) {
            best = j;
          }
        }
        if (best < 0 || uniq_w[best] <= 0.0) break;
        vote_ids[result] = uniq_id[best];
        vote_p[result] = uniq_w[best] / wtotal;
        uniq_w[best] = -1.0;
        ++result;
      }
    }
  }
  return result;
}


static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  const unsigned int hist[], int hlen, int ctxlen, int order,
  unsigned int out_ids[], int k, unsigned int target, double* target_prob)
{
  unsigned int prev1 = (hlen > 0) ? hist[hlen - 1] : 0;
  unsigned int prev2 = (hlen > 1) ? hist[hlen - 2] : 0;
  unsigned int ids[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  double relfreq[NGRAM_SUCC_MAX];
  double score[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  int prov[NGRAM_SUCC_MAX];
  int taken[NGRAM_SUCC_MAX + KNNLM_VOTE_MAX];
  unsigned int vote_ids[KNNLM_VOTE_MAX];
  double vote_p[KNNLM_VOTE_MAX];
  int ctx_total = 0;
  int n = ngram_candidates(ngram, prev2, prev1, order, ids, relfreq, prov,
    &ctx_total, NGRAM_SUCC_MAX);
  int nvote = knnlm_vote(store, hist, hlen, ctxlen, vote_ids, vote_p,
    KNNLM_VOTE_MAX);
  double lambda;
  int result = 0, rank, v;
  const char* env = getenv("CONVERSE_KNNLM_LAMBDA");
  if (NULL != env && '\0' != *env) {
    lambda = atof(env);
  }
  else {
    double t = (double)ctx_total;
    int prov0 = (n > 0) ? prov[0] : 0;
    if (2 == prov0) lambda = t / (t + 1.0);
    else if (1 == prov0) lambda = 0.5;
    else lambda = 0.1;
  }
  if (lambda < 0.0) lambda = 0.0;
  if (lambda > 1.0) lambda = 1.0;
  for (rank = 0; rank < n; ++rank) {
    score[rank] = lambda * relfreq[rank];
    taken[rank] = 0;
  }
  for (v = 0; v < nvote; ++v) {
    for (rank = 0; rank < n; ++rank) {
      if (ids[rank] == vote_ids[v]) break;
    }
    if (rank < n) {
      score[rank] += (1.0 - lambda) * vote_p[v];
    }
    else if (n < NGRAM_SUCC_MAX + KNNLM_VOTE_MAX) {
      ids[n] = vote_ids[v];
      score[n] = (1.0 - lambda) * vote_p[v];
      taken[n] = 0;
      ++n;
    }
  }
  while (result < k && result < n) {
    int best = -1;
    for (rank = 0; rank < n; ++rank) {
      if (0 == taken[rank] && (best < 0 || score[rank] > score[best])) {
        best = rank;
      }
    }
    if (best < 0) break;
    taken[best] = 1;
    out_ids[result] = ids[best];
    ++result;
  }
  /**
   * A normalized probability for one target, needed to score this model in bits
   * rather than only by rank. The candidate scores above rank on relative
   * frequency within the retained successor list, which does not sum to one over
   * the vocabulary; the classic kNN-LM mixture does, because the n-gram term is
   * normalized and the vote mass sums to one over the neighbours. So lambda=1
   * must reproduce the n-gram baseline bit for bit, and that is the control
   * which says this is wired correctly rather than merely plausible.
   */
  if (NULL != target_prob) {
    double p = lambda * ngramk_prob(ngram, hist, hlen, ngram_maxorder(),
      target);
    for (v = 0; v < nvote; ++v) {
      if (vote_ids[v] == target) {
        p += (1.0 - lambda) * vote_p[v];
        break;
      }
    }
    *target_prob = p;
  }
  return result;
}


static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, seen = 0, index = 0;
  long nnullq = 0, nnulltgt = 0;
  long ndeep = 0, ndeep_top1 = 0, nshallow = 0, nshallow_top1 = 0;
  double deep_bits = 0.0, deep_bytes = 0.0;
  double shallow_bits = 0.0, shallow_bytes = 0.0;
  double sum_bits = 0.0, sum_bytes = 0.0;
  long ntrunc = 0, ntrunc_top1 = 0;
  double trunc_bits = 0.0, trunc_bytes = 0.0;
  const double inv_log2 = 1.0 / log(2.0);
  double ord_inner = 0.0, ord_cross = 0.0;
  long nord_inner = 0, nord_cross = 0;
  const char* ord_env = getenv("CONVERSE_KNNLM_ORDERPROBE");
  int order_probe = (NULL != ord_env && '0' != ord_env[0]) ? 1 : 0;
  int stride = predict_eval_stride();
  int ctxlen = knnlm_ctxlen();
  const char* dyn_env = getenv("CONVERSE_KNNLM_DYN");
  int dynamic = (NULL != dyn_env && '0' != dyn_env[0]) ? 1 : 0;
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  if (NULL == ngram || NULL == store || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  if (0 != dynamic) knnlm_dyn_reset();
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != ngram_dedup_scale() && (SCALE_SENTENCE != entry->scale
      || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT)))
    {
      is_test = 0;
    }
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
    {
      size_t pos;
      unsigned int hist[TOKEN_CTX_MAX];
      int hlen = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen > 0) {
            if (0 == (seen % stride)) {
              unsigned int ids[NGRAM_TOPK];
              double p = 0.0;
              int n = knnlm_topk(ngram, store, hist, hlen, ctxlen, order, ids,
                NGRAM_TOPK, lex->id, &p);
              int rank, top1 = 0;
              /**
               * Whether the FULL-order context recurred in training, by the same
               * definition ngram_eval uses, so the two models' buckets are the
               * same set of positions. Retrieval has only ever been reported as
               * a top-k delta over all positions; the project's own criterion is
               * that a mechanism moving only the verbatim bucket has achieved
               * nothing, and that criterion had never been applied here.
               */
              int matched = 0;
              double bits;
              int deep;
              ngramk_predict_order(ngram, hist, hlen, ngram_maxorder(), NULL, 0,
                &matched);
              deep = (matched >= ngram_maxorder()
                && hlen >= ngram_maxorder()) ? 1 : 0;
              if (!(p > 0.0)) p = 1e-12;
              bits = -log(p) * inv_log2;
              ++npairs;
              sum_bits += bits;
              sum_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
              if (0 != token_emb_isnull(hist[hlen - 1])) ++nnullq;
              if (0 != token_emb_isnull(lex->id)) ++nnulltgt;
              if (0 != order_probe) {
                knnlm_order_probe(hist, hlen, ctxlen, knnlm_decay(),
                  &ord_inner, &ord_cross, &nord_inner, &nord_cross);
              }
              for (rank = 0; rank < n; ++rank) {
                if (ids[rank] == lex->id) {
                  if (0 == rank) {
                    ++ntop1;
                    top1 = 1;
                  }
                  ++ntopk;
                  break;
                }
              }
              if (0 != deep) {
                ++ndeep;
                ndeep_top1 += top1;
                deep_bits += bits;
                deep_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
              }
              else {
                const int mo = ngram_maxorder();
                const int avail = (hlen < mo) ? hlen : mo;
                ++nshallow;
                nshallow_top1 += top1;
                shallow_bits += bits;
                shallow_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
                /* The repaired generalization bucket; see ngram_eval. */
                if (matched < avail && hlen >= mo) {
                  ++ntrunc;
                  ntrunc_top1 += top1;
                  trunc_bits += bits;
                  trunc_bytes += (lex->length > 0) ? (double)lex->length : 1.0;
                }
              }
            }
            ++seen;
            if (0 != dynamic) knnlm_dyn_insert(hist, hlen, ctxlen, lex->id);
          }
          ngram_hist_push(hist, &hlen, TOKEN_CTX_MAX, lex->id);
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-%s%s%s: top1=%.1f%% top%d=%.1f%% n=%ld (stride=%d) bpc=%.3f\n",
      kind, (0 != dynamic) ? ":dyn" : "", (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      stride, (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
    fprintf(stderr, "  attested-context split: verbatim %.1f%% of positions"
      " (top1=%.1f%% bpc=%.3f) | novel %.1f%% (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ndeep / (double)npairs,
      (ndeep > 0) ? 100.0 * (double)ndeep_top1 / (double)ndeep : 0.0,
      (deep_bytes > 0.0) ? deep_bits / deep_bytes : 0.0,
      100.0 * (double)nshallow / (double)npairs,
      (nshallow > 0) ? 100.0 * (double)nshallow_top1 / (double)nshallow : 0.0,
      (shallow_bytes > 0.0) ? shallow_bits / shallow_bytes : 0.0);
    fprintf(stderr, "  truncated %.1f%% of all (top1=%.1f%% bpc=%.3f)\n",
      100.0 * (double)ntrunc / (double)npairs,
      (ntrunc > 0) ? 100.0 * (double)ntrunc_top1 / (double)ntrunc : 0.0,
      (trunc_bytes > 0.0) ? trunc_bits / trunc_bytes : 0.0);
    fprintf(stderr, "embedding coverage: null query vectors %.2f%%,"
      " null targets %.2f%% (n=%ld)\n",
      100.0 * (double)nnullq / (double)npairs,
      100.0 * (double)nnulltgt / (double)npairs, npairs);
    if (0 != order_probe && nord_cross > 0) {
      fprintf(stderr, "order sensitivity (cosine of swapped context, 1.0 ="
        " order discarded): inner=%.4f (n=%ld) cross=%.4f (n=%ld)\n",
        (nord_inner > 0) ? (ord_inner / (double)nord_inner) : 1.0, nord_inner,
        ord_cross / (double)nord_cross, nord_cross);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}


static void knnlm_complete(libxs_registry_t* ngram,
  const libxs_predict_t* store, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  unsigned int prev2 = 0, prev1 = 0;
  unsigned int hist[2];
  unsigned int ids[NGRAM_TOPK];
  int n, i;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  hist[0] = prev2;
  hist[1] = prev1;
  n = knnlm_topk(ngram, store, hist, 2, 2, order, ids, NGRAM_TOPK, 0, NULL);
  if (n <= 0) {
    printf("(no continuation)\n");
    return;
  }
  printf("next:");
  for (i = 0; i < n; ++i) {
    int len = 0;
    const char* word = libxs_lexicon_text(lexicon, ids[i], &len, NULL);
    if (NULL != word && len > 0) printf(" %.*s", len, word);
  }
  printf("\n");
  { unsigned int step[2];
    int s;
    step[0] = prev2;
    step[1] = prev1;
    printf("greedy:");
    for (s = 0; s < 12; ++s) {
      unsigned int step_ids[1];
      int len = 0;
      const char* word;
      if (0 == knnlm_topk(ngram, store, step, 2, 2, order, step_ids, 1,
        0, NULL))
      {
        break;
      }
      word = libxs_lexicon_text(lexicon, step_ids[0], &len, NULL);
      if (NULL == word || len <= 0) break;
      printf(" %.*s", len, word);
      step[0] = step[1];
      step[1] = step_ids[0];
    }
    printf("\n");
  }
}


static void* converse_lm_judge_open(const libxs_registry_t* corpus,
  int maxorder)
{
  return converse_hier_build(corpus, 0, 0, maxorder);
}


static void converse_lm_judge_close(void* model)
{
  converse_hier_destroy((converse_hier_t*)model);
}


static int converse_lm_judge_rescore(const void* model, const char* query,
  int query_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates, double bits[])
{
  return converse_hier_rescore((const converse_hier_t*)model, query,
    query_length, candidates, candidate_lengths, ncandidates, bits);
}


static int converse_lm_judge_choose(const void* model, const char* context,
  int context_length, const char* const candidates[],
  const int candidate_lengths[], int ncandidates)
{
  return converse_hier_choose((const converse_hier_t*)model, context,
    context_length, candidates, candidate_lengths, ncandidates);
}


static int converse_lm_judge_seam(const void* model, const char* prefix,
  int prefix_length, const char* suffix, int suffix_length, int score_length,
  double* bits)
{
  return converse_hier_seam_bits((const converse_hier_t*)model, prefix,
    prefix_length, suffix, suffix_length, score_length, bits);
}


const converse_judge_t* converse_lm_judge(void)
{
  static const converse_judge_t judge = { converse_lm_judge_open,
    converse_lm_judge_close, converse_lm_judge_rescore,
    converse_lm_judge_choose, converse_lm_judge_seam };
  return &judge;
}


/**
 * A -T prefix supplies a fixed held-out corpus: train on all of the main corpus
 * (holdout forced 0) and evaluate on the separate test set. This keeps the test
 * set identical across training-corpus sizes, so BPC is comparable along a
 * scaling curve.
 */
static const libxs_registry_t* converse_lm_test_corpus(converse_run_t* run,
  libxs_registry_t** owned)
{
  const libxs_registry_t* result = run->corpus;
  *owned = NULL;
  if (NULL != run->test_prefix) {
    libxs_registry_t* test = libxs_registry_create();
    converse_stage_begin();
    if (NULL != test && EXIT_SUCCESS == corpus_ingest_basename(test,
      run->test_prefix, run->lexicon, run->rules, run->nrules))
    {
      libxs_registry_info_t tinfo = { 0 };
      libxs_registry_info(test, &tinfo);
      fprintf(stderr, "test corpus: %lu sentences (%s)\n",
        (unsigned long)tinfo.size, run->test_prefix);
      result = test;
      *owned = test;
      run->ngram_holdout = 0;
    }
    else {
      fprintf(stderr, "failed to load test corpus: %s\n", run->test_prefix);
      libxs_registry_destroy(test);
    }
  }
  return result;
}


/** Read prompts and print next-token continuations from the selected kind. */
static void converse_lm_complete(converse_run_t* run,
  libxs_registry_t* ngram_model, const libxs_predict_t* token_model,
  int use_predict, int use_embed, int use_rerank, int use_knnlm)
{
  char line[4096];
  printf("> ");
  fflush(stdout);
  while (NULL != fgets(line, (int)sizeof(line), stdin)) {
    size_t len = strlen(line);
    while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
    if (0 < len) {
      if (0 != use_predict || 0 != use_embed) {
        token_complete(token_model, run->lexicon, run->rules, run->nrules,
          use_embed, line, (int)len);
      }
      else if (0 != use_rerank) {
        rerank_complete(ngram_model, token_model, run->lexicon, run->rules,
          run->nrules, run->ngram_order, line, (int)len);
      }
      else if (0 != use_knnlm) {
        knnlm_complete(ngram_model, token_model, run->lexicon, run->rules,
          run->nrules, run->ngram_order, line, (int)len);
      }
      else ngram_complete(ngram_model, run->lexicon, run->rules, run->nrules,
        run->ngram_order, line, (int)len);
    }
    printf("> ");
    fflush(stdout);
  }
}


int converse_lm_run(converse_run_t* run)
{
  const int use_predict = (0 == strcmp(run->ngram_kind, "predict")) ? 1 : 0;
  const int use_embed = (0 == strcmp(run->ngram_kind, "embed")) ? 1 : 0;
  const int use_rerank = (0 == strcmp(run->ngram_kind, "rerank")) ? 1 : 0;
  const int use_knnlm = (0 == strcmp(run->ngram_kind, "knnlm")) ? 1 : 0;
  const int use_hier = (0 == strcmp(run->ngram_kind, "hier")) ? 1 : 0;
  libxs_registry_t* ngram_model = NULL;
  libxs_registry_t* test_corpus = NULL;
  libxs_predict_t* token_model = NULL;
  converse_hier_t* hier_model = NULL;
  const libxs_registry_t* eval_corpus;
  int result = EXIT_FAILURE;
  eval_corpus = converse_lm_test_corpus(run, &test_corpus);
  converse_judge_open(run->corpus);
  if (0 != use_hier) {
    hier_model = converse_hier_build(run->corpus, run->ngram_holdout,
      run->nsentences, ngram_maxorder());
  }
  else if (0 != use_predict) {
    token_model = token_predict_build(run->corpus, run->lexicon, run->rules,
      run->nrules, run->profile, 0, run->ngram_holdout, 2);
  }
  else {
    converse_bpe_prepare(run->corpus, run->ngram_holdout);
    converse_stage_end("test_ingest");
    ngram_model = ngram_build(run->corpus, run->lexicon, run->rules,
      run->nrules, run->ngram_holdout);
    ngram_backoff_build(ngram_model, run->lexicon);
    converse_stage_end("ngram_build");
    /* The embedding-rank probe reads the embedding without being a prediction
     * KIND, so it declares the dependency itself rather than forcing -K embed,
     * which the dispatch below would route away from gen-eval. Split out of
     * the kind chain so the probe does not also build a predict store: that
     * store is what gen-eval scores as the bank's kind-different slot, and
     * handing it one it would not otherwise have would change the run being
     * measured. */
    if (0 != use_embed || 0 != use_knnlm || 0 != ngram_gen_embrank()
      || 0 != ngram_gen_embcand()
      || 0 != (ngram_bank_slots() & (1u << NGRAM_BANK_EMB)))
    {
      token_emb_build(run->corpus, run->lexicon, run->rules, run->nrules,
        run->ngram_holdout);
      converse_stage_end("emb_build");
    }
    if (0 != use_embed || 0 != use_knnlm) {
      token_model = token_predict_build(run->corpus, run->lexicon, run->rules,
        run->nrules, run->profile, 1, run->ngram_holdout,
        (0 != use_knnlm) ? knnlm_ctxlen() : 2);
      converse_stage_end("token_predict");
    }
    else if (0 != use_rerank) {
      token_model = rerank_build(run->corpus, ngram_model, run->lexicon,
        run->rules, run->nrules, run->ngram_order, run->profile,
        run->ngram_holdout);
    }
    /* The expert bank can carry the predict store as a slot, which needs the
     * store in the plain n-gram path too. Gated on the slot being asked for,
     * so every other configuration stays byte-identical. */
    else if (0 != (ngram_bank_slots() & (1u << NGRAM_BANK_PREDICT))) {
      token_model = token_predict_build(run->corpus, run->lexicon, run->rules,
        run->nrules, run->profile, 0, run->ngram_holdout, 2);
      /* Separate timers: one stage covering both made a slow build
       * indistinguishable from slow scoring, and attributing the cost to the
       * wrong one of those is how a fix gets aimed at the wrong code. */
      converse_stage_end("predict_build");
      /* Converge and publish the escape weights BEFORE scoring, so frozen
       * mode freezes something converged rather than the uniform prior. The
       * model is mutable only here, which is also the only place it may be
       * written: scoring requires it read-only. */
      if (0 != ngram_bank_frozen() && NULL != token_model) {
        ngram_bank_warmup(token_model, (int)libxs_lexicon_size(run->lexicon));
        converse_stage_end("predict_warmup");
      }
    }
  }
  converse_stage_begin();
  if (0 != run->predict_eval_mode) {
    if (0 != use_hier) {
      result = converse_hier_eval(hier_model, eval_corpus, run->ngram_holdout,
        (eval_corpus == run->corpus) ? run->nsentences : 0, "metatoken");
    }
    else if (0 != use_predict || 0 != use_embed) {
      char label[64];
      sprintf(label, "%s:%s", use_embed ? "embed" : run->profile->name,
        run->profile->name);
      result = token_predict_eval(token_model, eval_corpus, run->lexicon,
        run->rules, run->nrules, use_embed, run->ngram_holdout,
        use_embed ? label : run->profile->name);
    }
    else if (0 != use_rerank) {
      char label[64];
      sprintf(label, "rerank:%s", run->profile->name);
      result = rerank_eval(ngram_model, token_model, eval_corpus, run->lexicon,
        run->rules, run->nrules, run->ngram_order, run->ngram_holdout, label);
    }
    else if (0 != use_knnlm) {
      char label[64];
      sprintf(label, "knnlm:%s", run->profile->name);
      result = knnlm_eval(ngram_model, token_model, eval_corpus, run->lexicon,
        run->rules, run->nrules, run->ngram_order, run->ngram_holdout, label);
    }
    else if (NULL != getenv("CONVERSE_SLOT_PROBE")) {
      slot_probe_run(eval_corpus, run->lexicon, run->rules, run->nrules,
        run->ngram_holdout);
      result = EXIT_SUCCESS;
    }
    else if (NULL != getenv("CONVERSE_GEN_EVAL")) {
      result = ngram_gen_eval(ngram_model, eval_corpus, run->lexicon,
        run->rules, run->nrules, run->ngram_holdout, run->ngram_kind,
        token_model, use_embed);
    }
    else {
      result = ngram_eval(ngram_model, eval_corpus, run->lexicon, run->rules,
        run->nrules, run->ngram_order, run->ngram_holdout, run->ngram_kind,
        token_model, use_embed);
    }
    if (0 == use_hier) ngram_stats(ngram_model);
    /* Bounds any cut rule against the model just trained, so it runs after
     * the store exists and reads only held-out entries. */
    if (NULL != getenv("CONVERSE_SYLLABLE_ORACLE") && NULL != ngram_model) {
      ngram_syllable_oracle(ngram_model, run->lexicon, run->rules, run->nrules,
        eval_corpus, run->ngram_holdout, ngram_maxorder());
    }
    converse_stage_end("eval");
    converse_stage_report();
  }
  else if (0 != use_hier) {
    fprintf(stderr, "hier is currently an evaluation-only model (-E)\n");
  }
  else {
    converse_lm_complete(run, ngram_model, token_model, use_predict, use_embed,
      use_rerank, use_knnlm);
    result = EXIT_SUCCESS;
  }
  converse_judge_close();
  libxs_predict_destroy(token_model);
  converse_hier_destroy(hier_model);
  libxs_registry_destroy(test_corpus);
  knnlm_cache_free();
  return result;
}
