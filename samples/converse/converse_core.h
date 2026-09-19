#ifndef CONVERSE_CORE_H
#define CONVERSE_CORE_H

#include "converse.h"

/**
 * Services converse.c provides to both halves: the corpus and its ingest, the
 * tokenizer at every granularity, the word n-gram, the successor embedding, and
 * the answer ranker that -L persists.
 *
 * The boundary is PERSISTED SHARED STATE plus the representations built from it.
 * That is what makes -L available to either binary and what decides the two
 * straddlers the census got wrong twice: the basic n-gram is here because the QA
 * half prints a continuation under an answer, and the embedding is here because
 * rule learning scores class members with it. Neither is an LM experiment.
 *
 * Declared apart from converse.h so converse_recomb.c and converse_hier.c, which
 * are models rather than halves, do not see this at all.
 */

int corpus_ingest_basename(libxs_registry_t* corpus, const char* basename,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
/**
 * Build one corpus entry from text. create interns unseen words, which only
 * ingest may do: a pass that rebuilds derived entries from stored text passes
 * zero, because every id it needs exists and counting the occurrences again
 * would inflate libxs_lexicon_count on every run.
 */
int corpus_entry_build(corpus_entry_t* entry, const unsigned char* text,
  int len, unsigned char scale, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int create);
int count_words(const unsigned char* text, int length);
int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id);

/**
 * Bytes of the heading this span opens with, 0 if it does not open with one.
 * One definition, so the ingest that captures a section and the reply path that
 * strips a captured heading back off cannot disagree about what a heading is.
 */
int corpus_title_len(const char* text, int len);
/**
 * Non-zero if the text is MARKUP rather than prose: field syntax or an entity
 * reference. One definition, so the section table that refuses to call a caption a
 * heading and the fact layers that refuse to read a proposition out of one cannot
 * disagree about what markup is.
 */
int corpus_line_markup(const char* text, int len);
/** Path an entry's source id stands for, or NULL if the run has no name for it. */
const char* corpus_source_path(unsigned int id);

/**
 * Path of the normalization table for the active corpus namespace. Loaded by
 * converse_setup as relation rules of kind "norm", so a table written here is
 * read by the next run of any binary.
 */
const char* converse_norms_path(void);
int corpus_case_forced(const char* text, int at, int heading_len);

int text_ends_sentence(const char* text, int text_len);
size_t text_closer_size(const unsigned char* text, size_t size, size_t pos);
int text_find_ci(const char* text, int text_len, const char* term);
int text_contains_word_ci(const char* text, int text_len, const char* term);
char* eval_trim(char* text);
int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text);

int answer_relation_rule_has_term(int kind, const char* text, int text_len);
/** Whether text IS a term of this kind, rather than containing one. */
int answer_relation_rule_is_term(int kind, const char* text, int text_len);
/** The first declared term of a kind, or NULL. */
const char* answer_relation_rule_first_term(int kind, int* term_len);
/** The nth declared term of a kind with its first field, for MAP-shaped classes. */
const char* answer_relation_rule_term_at(int kind, int index,
  const char** relation);
/** ASSERTED, LEARNED or PROPOSED, for a term a reply is about to rest on. */
int answer_relation_rule_provenance(int kind, const char* text, int text_len);
int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size);

const answer_predict_profile_t* answer_predict_profile_default(void);
libxs_predict_t* answer_predict_create(
  const answer_predict_profile_t* profile);
int answer_predict_build_model(libxs_predict_t* model,
  const answer_predict_profile_t* profile);
int answer_features_fill(const corpus_entry_t* entry, size_t entry_size,
  double overlap, int query_type, double inputs[ANSWER_PREDICT_INPUTS]);

void converse_stage_begin(void);
void converse_stage_end(const char* name);
void converse_stage_report(void);

int ngram_native_mode(void);
int ngram_native_tokens(libxs_lexicon_t* lexicon, const char* text,
  int text_len, libxs_lexeme_t tokens[], unsigned int word_ids[], int max,
  int create);
int ngram_wordctx(void);
int ngram_wordctx_hist(const libxs_lexeme_t nat[],
  const unsigned int word_ids[], int i, int wctx, unsigned int hist[], int cap);
int ngram_syllable_split(const char* text, int wlen, int piece_begin[],
  int piece_len[], int max);
int ngram_is_wordchar(unsigned char c);
void converse_bpe_prepare(const libxs_registry_t* corpus, int holdout);
size_t ngram_render_append(char* out, size_t out_size, size_t pos,
  const char* piece, int len, int leading);

libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon);
void ngram_stats(const libxs_registry_t* model);
int ngram_maxorder(void);
int ngram_dedup_scale(void);
double ngramk_prob(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int next);
double ngramk_prob_exact(const unsigned int hist[], int hlen,
  unsigned int next);
int ngramk_predict_order(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int out_ids[], int k, int* order);
void ngram_hist_push(unsigned int hist[], int* hlen, int cap, unsigned int id);
double ngram_skip_prob(const unsigned int hist[], int hlen,
  unsigned int succ_id);
int ngram_skip_ready(const unsigned int hist[], int hlen);
int predict_is_test(long index, int holdout);
/** Word-scale backoff probability, shaped for converse_recomb_prob_t. */
double recomb_word_prob(const unsigned int hist[], int hlen, unsigned int next);

int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out);
void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
int ngram_gen_ncand(void);
int ngram_gen_minorder(void);
int ngram_gen_select(libxs_lexicon_t* lexicon, const unsigned int ids[],
  int nids, const char* context, int context_len);

void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
const double* token_emb_get(unsigned int id);
int token_emb_isnull(unsigned int id);
int token_emb_directed(void);
double token_emb_succ_prob(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab, double temp);
int token_emb_succ_rank(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab);
int token_emb_succ_append(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp, unsigned int ids[], int n, int max,
  int want);
double ngram_emb_temp(void);

#endif /*CONVERSE_CORE_H*/
