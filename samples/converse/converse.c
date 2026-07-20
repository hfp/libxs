#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>

#include "converse.h"

#include <unistd.h>

#define RESPONSE_BUDGET 1
#define ANSWER_MAX 4
#define ANSWER_MIN_SCORE 0.35
#define LEXICON_FILE "converse.lex"
#define PREDICT_FILE "converse.prd"
#define BRIDGE_FILE "converse.bridges"
#define BRIDGE_LINE_MAX 2048
#define RELATION_FILE "converse.relations"
#define RELATION_LINE_MAX 1024
#define EVAL_FILE "converse.eval"
#define EVAL_LINE_MAX 2048
#define NGRAM_FILE "converse.ngram"
#define NGRAM_SUCC_MAX LIBXS_NGRAM_SUCC_MAX
#define NGRAM_TOPK 3
#define NGRAM_NATIVE_WIDTH 4
#define NGRAM_ORDER_MAX LIBXS_NGRAM_ORDER_MAX
#define BPE_SYMBOL_MAX 32
#define BPE_WORD_MAX 128
#define BPE_MERGES_DEFAULT 750
#define BPE_WORD_CAP 80000
#define PREDICT_EVAL_FILE "converse.predict"
#define CONVERSE_PATH_MAX 512
#define CONV_TOPIC_MAX 64
#define TOKEN_PREDICT_TRAIN_MAX 40000
#define TOKEN_PREDICT_EVAL_STRIDE 40
#define TOKEN_EMB_DIM 16
#define TOKEN_EMB_CTX 256
#define TOKEN_EMB_WINDOW 2
#define TOKEN_EMB_ITER 24
#define TOKEN_EMB_BACKFILL_MIN 3
#define TOKEN_EMB_BACKFILL_REF 5
#define KNNLM_K 24
#define KNNLM_VOTE_MAX 4
#define KNNLM_ANN_DIMS 8
#define KNNLM_ANN_BITS 8
#define KNNLM_ANN_WINDOW 512
#define RERANK_INPUTS 9
#define RERANK_RELIABILITY 32.0
#define RERANK_LIFT_MAX 8.0
#define ANSWER_PREDICT_INPUTS 10
#define CORPUS_BASENAME_PART_MAX 999


enum { PROFILE_PROSE = 0, PROFILE_MARKDOWN = 1 };

enum { GRAN_WORD = 0, GRAN_NATIVE = 1, GRAN_SYLLABLE = 2, GRAN_BPE = 3 };

enum { QUERY_GENERIC = 0, QUERY_WHO, QUERY_WHAT, QUERY_WHERE,
  QUERY_WHEN, QUERY_WHY, QUERY_HOW, QUERY_YESNO };

typedef struct answer_predict_profile_t {
  const char* name;
  int mode;
  int decompose;
  int clusters;
  int order;
  double quality;
  double smooth;
  int nseries;
  int window;
  int target;
  int diff_order;
} answer_predict_profile_t;

typedef struct answer_bridge_t {
  const char* name;
  const char* query;
  const char* evidence;
  const char* reply;
  double score;
} answer_bridge_t;

enum { RELATION_RULE_ALIAS = 1, RELATION_RULE_PERSON, RELATION_RULE_SKIP };

typedef struct answer_relation_rule_t {
  int kind;
  char relation[64];
  char term[64];
} answer_relation_rule_t;

typedef struct answer_relation_match_t {
  char answer[128];
  char relation[64];
  char actor[64];
  int answer_len;
  int relation_len;
  int actor_len;
  int plural;
  int made;
  double score;
} answer_relation_match_t;

typedef struct answer_relation_fact_t {
  char answer[128];
  char relation[64];
  char actor[64];
  char section[ENTRY_SECTION_MAX];
  int answer_len;
  int relation_len;
  int actor_len;
  int section_len;
  int plural;
  int made;
  double score;
} answer_relation_fact_t;

typedef struct answer_identity_fact_t {
  char name[64];
  char role[64];
  char section[ENTRY_SECTION_MAX];
  int name_len;
  int role_len;
  int section_len;
  double score;
} answer_identity_fact_t;

typedef struct answer_describe_fact_t {
  char role[64];
  char text[192];
  char section[ENTRY_SECTION_MAX];
  int role_len;
  int text_len;
  int section_len;
  double score;
} answer_describe_fact_t;

typedef struct answer_docdef_fact_t {
  char title[ENTRY_SECTION_MAX];
  char header[64];
  char text[COMPOSE_MAXTEXT];
  int title_len;
  int header_len;
  int text_len;
} answer_docdef_fact_t;

typedef struct ngram_key_t {
  unsigned int a;
  unsigned int b;
} ngram_key_t;

typedef libxs_ngram_succ_t ngram_succ_t;
typedef libxs_ngram_entry_t ngram_entry_t;

typedef struct bpe_symbol_t {
  int len;
  char bytes[BPE_SYMBOL_MAX];
} bpe_symbol_t;

typedef struct bpe_pair_t {
  int a;
  int b;
} bpe_pair_t;

typedef struct bpe_rank_t {
  int rank;
  int merged;
} bpe_rank_t;

typedef struct bpe_word_t {
  long count;
  int nsyms;
  int syms[BPE_WORD_MAX];
} bpe_word_t;


static const answer_predict_profile_t answer_predict_profiles[] = {
  { "raw", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "poly2", LIBXS_PREDICT_INTERPOLATE, LIBXS_PREDICT_RAW, 0, 2, 0.0, 0.0, 0, 0, 0, 0 },
  { "smooth", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 0, 0, 0, 0 },
  { "temporal", LIBXS_PREDICT_TEMPORAL, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 1, ANSWER_PREDICT_INPUTS, 0, 1 },
  { "rf", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_RF, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "fisher", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_FISHER, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "hknn", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_HKNN, 0, 1, 0.0, 0.0, 0, 0, 0, 0 }
};

static char converse_path_corpus[CONVERSE_PATH_MAX] = CORPUS_FILE;
static char converse_path_lexicon[CONVERSE_PATH_MAX] = LEXICON_FILE;
static char converse_path_predict[CONVERSE_PATH_MAX] = PREDICT_FILE;
static char converse_path_bridge[CONVERSE_PATH_MAX] = BRIDGE_FILE;
static char converse_path_relation[CONVERSE_PATH_MAX] = RELATION_FILE;
static char converse_path_eval[CONVERSE_PATH_MAX] = EVAL_FILE;
static char converse_path_predict_eval[CONVERSE_PATH_MAX] = PREDICT_EVAL_FILE;
static answer_bridge_t* answer_bridge_loaded = NULL;
static size_t answer_bridge_loaded_size = 0;
static answer_relation_rule_t* answer_relation_rules = NULL;
static size_t answer_relation_rules_size = 0;
static answer_relation_fact_t* answer_relation_facts = NULL;
static size_t answer_relation_facts_size = 0;
static answer_identity_fact_t* answer_identity_facts = NULL;
static size_t answer_identity_facts_size = 0;
static answer_describe_fact_t* answer_describe_facts = NULL;
static size_t answer_describe_facts_size = 0;
static answer_docdef_fact_t* answer_docdef_facts = NULL;
static size_t answer_docdef_facts_size = 0;
static char conv_topic[CONV_TOPIC_MAX] = "";
static int conv_topic_len = 0;
static long predict_ntotal = 0;


static void converse_namespace_init(const char* prefix);
static libxs_registry_t* corpus_load(void);
static int corpus_save(const libxs_registry_t* corpus);
static libxs_lexicon_t* converse_lexicon_load(void);
static int converse_lexicon_save(const libxs_lexicon_t* lexicon);
static libxs_predict_t* converse_predict_load(void);
static int converse_predict_save(const libxs_predict_t* model);
static libxs_predict_t* converse_predict_train(const libxs_registry_t* corpus,
  const answer_predict_profile_t* profile);
static const answer_predict_profile_t* answer_predict_profile_default(void);
static const answer_predict_profile_t* answer_predict_profile_find(
  const char* name);
static void answer_predict_profile_list(FILE* stream);
static void answer_bridge_free_loaded(void);
static size_t answer_bridge_load_file(const char* path);
static void answer_bridge_report(FILE* stream);
static void answer_relation_rules_free(void);
static size_t answer_relation_rules_load_file(const char* path);
static void answer_relation_rules_report(FILE* stream);
static int answer_relation_rule_has_term(int kind, const char* text,
  int text_len);
static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len);
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int corpus_ingest_basename(libxs_registry_t* corpus,
  const char* basename, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules);
static int corpus_profile_for_path(const char* path);
static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like);
static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like);
static int corpus_ingest_markdown(libxs_registry_t* corpus,
  const unsigned char* text, size_t text_size, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int count_words(const unsigned char* text, int length);
static size_t text_closer_size(const unsigned char* text, size_t size,
  size_t pos);
static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos);
static int text_starts_sentence(const char* text, int text_len);
static int text_ends_sentence(const char* text, int text_len);
static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id);
static int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id);
static int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text);
static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text);
static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance);
static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon);
static int query_type_prefers_sentence(int query_type);
static int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void corpus_entry_set_section(corpus_entry_t* entry,
  const char* section, int section_len);
static int corpus_title_prefix(const unsigned char* text, int len,
  char* title, int title_size);
static int corpus_entry_same_section(const corpus_entry_t* lhs,
  size_t lhs_size, const corpus_entry_t* rhs);
static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry);
static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size);
static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len);
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size, int* upper_initial);
static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size);
static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos);
static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len);
static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit);
static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match);
static int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size);
static void answer_relation_facts_free(void);
static size_t answer_relation_facts_build(const libxs_registry_t* corpus);
static void answer_relation_facts_report(FILE* stream);
static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_identity_facts_free(void);
static size_t answer_identity_facts_build(const libxs_registry_t* corpus);
static void answer_identity_facts_report(FILE* stream);
static void answer_describe_facts_free(void);
static size_t answer_describe_facts_build(const libxs_registry_t* corpus);
static void answer_describe_facts_report(FILE* stream);
static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_docdef_facts_free(void);
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus);
static void answer_docdef_facts_report(FILE* stream);
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void conv_reset(void);
static void conv_remember(const char* query_text, size_t query_len);
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size);
static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role);
static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry);
static int answer_features_fill(const corpus_entry_t* entry,
  size_t entry_size, double overlap, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static double answer_weak_label(const corpus_entry_t* entry, int query_type);
static libxs_predict_t* answer_predict_create(
  const answer_predict_profile_t* profile);
static int answer_predict_build_model(libxs_predict_t* model,
  const answer_predict_profile_t* profile);
static void answer_predict_report(const char* label,
  const libxs_predict_t* model, int ntrain,
  const answer_predict_profile_t* profile);
static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type);
static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry);
static double answer_semantic_bridge_score(const answer_bridge_t* bridge);
static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile);
static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score);
static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX]);
static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static int text_find_ci(const char* text, int text_len, const char* term);
static int text_find_word_ci(const char* text, int text_len,
  const char* term);
static int text_contains_ci(const char* text, int text_len, const char* term);
static int text_contains_word_ci(const char* text, int text_len,
  const char* term);
static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon);
static const ngram_entry_t* ngram_lookup(libxs_registry_t* model,
  unsigned int ctx_a, unsigned int ctx_b);
static double ngram_unigram_prior(unsigned int id);
static int ngram_order(void);
static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len);
static libxs_ngram_t converse_ngram;
static int converse_profile_override = -1;
static int converse_order_max = 0;
static bpe_symbol_t* bpe_symbols = NULL;
static int bpe_nsymbols = 0;
static int bpe_cap_symbols = 0;
static libxs_registry_t* bpe_merges = NULL;
static void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out);
static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind);
static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind);
static void ngram_stats(const libxs_registry_t* model);
static int ngram_gran_mode(void);
static int predict_is_test(long index, int holdout);
static int bpe_add_symbol(const char* bytes, int len);
static void bpe_free(void);
static void bpe_build(const libxs_registry_t* corpus, int holdout);
static int bpe_encode_run(const char* text, int len, libxs_token_t tokens[],
  int max, int start, libxs_lexicon_t* lexicon, int create);
static void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout);
static void token_emb_free(void);
static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  unsigned int prev2, unsigned int prev1, int order, unsigned int out_ids[],
  int k);
static void knnlm_cache_free(void);
static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind);
static void knnlm_complete(libxs_registry_t* ngram,
  const libxs_predict_t* store, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout);
static int token_predict_eval(const libxs_predict_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int use_emb, int holdout,
  const char* kind);
static void token_complete(const libxs_predict_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int use_emb, const char* text, int text_len);
static libxs_predict_t* rerank_build(const libxs_registry_t* corpus,
  libxs_registry_t* ngram, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order,
  const answer_predict_profile_t* profile, int holdout);
static int rerank_eval(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind);
static void rerank_complete(libxs_registry_t* ngram,
  const libxs_predict_t* reranker, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len);
static double converse_score(const libxs_fprint_t* candidate,
  const libxs_fprint_t* query, const libxs_fprint_t* prev);
static int entry_used(const corpus_entry_t* e,
  const corpus_entry_t* const used[], int nused);
static const corpus_entry_t* select_best(void* const candidates[],
  int ncandidates, const corpus_entry_t* const used[], int nused,
  const libxs_fprint_t* query, const libxs_fprint_t* prev,
  int require_scale, unsigned char preferred_scale);
static uint64_t query_hilbert_code(const libxs_fprint_t* fp);
static int respond(const libxs_spatial_t* spatial,
  const libxs_registry_t* corpus, const char* query_text,
  size_t query_len, const libxs_fprint_t* query, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);


int main(int argc, char* argv[])
{
  libxs_registry_t* corpus;
  libxs_lexicon_t* lexicon;
  libxs_predict_t* answer_model;
  libxs_lexrule_t rules[96];
  int i, budget = RESPONSE_BUDGET, eval_mode = 0;
  int complete_mode = 0, predict_eval_mode = 0, learn_mode = 0;
  int warm_start = 0;
  int ngram_order = 2;
  int ngram_holdout = 0;
  const char* ngram_kind = "trigram";
  libxs_registry_t* ngram_model = NULL;
  const char** basenames;
  int nbasenames = 0;
  int nrules;
  libxs_registry_info_t rinfo;
  char line[4096];
  const answer_predict_profile_t* predict_profile =
    answer_predict_profile_default();

  if (argc < 2) {
    fprintf(stderr,
      "Usage: %s [-e] [-n N] [-P PROFILE] [-b PREFIX] corpus1.txt [corpus2.txt ...]\n"
      "  Interactive conversational agent.\n"
      "  -e: run converse.eval evaluation and exit.\n"
      "  -E: run next-token prediction evaluation and exit.\n"
      "  -L: learn from the corpus, save state, and exit.\n"
      "  -c: read prompts and print next-token continuations.\n"
      "  -K KIND: next-token model "
      "(bigram|trigram|predict|embed|rerank|knnlm).\n"
      "  -H N: held-out split; train on non-test, eval 1-in-N test.\n"
      "  -n N: response sentence budget (default %d).\n"
      "  -P PROFILE: answer predictor profile.\n"
      "  -p STRUCTURE: corpus structure (prose|markdown; default by ext).\n"
      "  -x: extreme mode; deepest n-gram context (affects -c and -E).\n"
      "  -b PREFIX: ingest matching files next to PREFIX.\n"
      "Environment variables (override defaults):\n"
      "  CONVERSE_NGRAM_ORDER=N   n-gram context order 1..%d (default 2; -x=%d).\n"
      "  CONVERSE_GRAN=UNIT       token unit: word|native|syllable|bpe.\n"
      "  CONVERSE_GEN_MINORDER=N  generation grounding floor (default 2).\n"
      "  CONVERSE_BPE_MERGES=N    BPE merge budget (default 750).\n"
      "  CONVERSE_HOLDOUT_TAIL=1  held-out split is a contiguous tail.\n"
      "  CONVERSE_EVAL_STRIDE=N   evaluate 1-in-N test items (default 40).\n"
      "  CONVERSE_GEN_EVAL=1      -E reports generation reproduction.\n"
      "  CONVERSE_NGRAM_STATS=1   print per-order n-gram footprint.\n"
      "  CONVERSE_KNNLM_LAMBDA=F  fixed kNN-LM interpolation weight.\n"
      "  CONVERSE_KNNLM_DYN=1     dynamic (test-time) kNN-LM datastore.\n"
      "  CONVERSE_KNNLM_ANN=1     Hilbert-indexed kNN-LM retrieval (faster).\n"
      "  CONVERSE_EMB_PROBE=w1,w2 print nearest embedding neighbors.\n"
      "  CONVERSE_EMB_BACKFILL=0  disable rare-token vector backfill.\n",
      argv[0], RESPONSE_BUDGET, NGRAM_ORDER_MAX, NGRAM_ORDER_MAX);
    answer_predict_profile_list(stderr);
    return EXIT_FAILURE;
  }

  basenames = (const char**)malloc((size_t)argc * sizeof(*basenames));
  if (NULL == basenames) return EXIT_FAILURE;

  i = 1;
  while (i < argc && '-' == argv[i][0] && '\0' != argv[i][1]) {
    if (0 == strcmp(argv[i], "-e")) {
      eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-E")) {
      predict_eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-L")) {
      learn_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-c")) {
      complete_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-x")) {
      converse_order_max = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-H") && i + 1 < argc) {
      ngram_holdout = atoi(argv[i + 1]);
      if (ngram_holdout < 0) ngram_holdout = 0;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-K") && i + 1 < argc) {
      if (0 == strcmp(argv[i + 1], "bigram")) {
        ngram_kind = "bigram";
        ngram_order = 1;
      }
      else if (0 == strcmp(argv[i + 1], "trigram")) {
        ngram_kind = "trigram";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "predict")) {
        ngram_kind = "predict";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "embed")) {
        ngram_kind = "embed";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "rerank")) {
        ngram_kind = "rerank";
        ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "knnlm")) {
        ngram_kind = "knnlm";
        ngram_order = 2;
      }
      else {
        fprintf(stderr,
          "unknown prediction kind: %s "
          "(use bigram|trigram|predict|embed|rerank|knnlm)\n",
          argv[i + 1]);
        free(basenames);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-n") && i + 1 < argc) {
      budget = atoi(argv[i + 1]);
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-P") && i + 1 < argc) {
      predict_profile = answer_predict_profile_find(argv[i + 1]);
      if (NULL == predict_profile) {
        fprintf(stderr, "unknown predictor profile: %s\n", argv[i + 1]);
        answer_predict_profile_list(stderr);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-p") && i + 1 < argc) {
      if (0 == strcmp(argv[i + 1], "prose")) {
        converse_profile_override = PROFILE_PROSE;
      }
      else if (0 == strcmp(argv[i + 1], "markdown")) {
        converse_profile_override = PROFILE_MARKDOWN;
      }
      else {
        fprintf(stderr, "unknown structure profile: %s (use prose|markdown)\n",
          argv[i + 1]);
        free(basenames);
        return EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-b") && i + 1 < argc) {
      basenames[nbasenames] = argv[i + 1];
      ++nbasenames;
      i += 2;
    }
    else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      answer_predict_profile_list(stderr);
      free(basenames);
      return EXIT_FAILURE;
    }
  }

  if (i == argc && 0 == nbasenames) {
    fprintf(stderr, "no corpus source given\n");
    free(basenames);
    return EXIT_FAILURE;
  }

  converse_namespace_init((nbasenames > 0) ? basenames[0] : NULL);

  if (0 != eval_mode && nbasenames + argc - i < 1) {
    fprintf(stderr,
      "eval mode expects at least one corpus source and reads %s\n",
      converse_path_eval);
    free(basenames);
    return EXIT_FAILURE;
  }

  corpus = corpus_load();
  if (NULL == corpus) corpus = libxs_registry_create();
  lexicon = converse_lexicon_load();
  if (NULL == lexicon) lexicon = libxs_lexicon_create();
  answer_model = converse_predict_load();
  nrules = libxs_lexrule_defaults(rules, 96);
  if (NULL == corpus || NULL == lexicon || nrules <= 0) {
    libxs_registry_destroy(corpus);
    libxs_lexicon_destroy(lexicon);
    libxs_predict_destroy(answer_model);
    free(basenames);
    return EXIT_FAILURE;
  }

  answer_bridge_load_file(converse_path_bridge);
  answer_bridge_report(stderr);
  answer_relation_rules_load_file(converse_path_relation);
  answer_relation_rules_report(stderr);

  /* A warm start reuses the persisted corpus, lexicon, and predictor instead
     of re-ingesting and re-training: the state is complete when the corpus
     loaded non-empty, the lexicon is populated, and the predictor loaded.
     Learn mode (-L) always rebuilds. Only the cheap fact index is rebuilt each
     run below. */
  { libxs_registry_info_t warm;
    warm.size = 0;
    warm_start = (0 == learn_mode && NULL != answer_model
      && libxs_lexicon_size(lexicon) > 0
      && EXIT_SUCCESS == libxs_registry_info(corpus, &warm)
      && warm.size > 0) ? 1 : 0;
  }

  if (0 == warm_start) {
    int basename_index;
    int have_positional = (i < argc) ? 1 : 0;
    for (basename_index = 0; basename_index < nbasenames; ++basename_index) {
      if (EXIT_SUCCESS != corpus_ingest_basename(corpus,
        basenames[basename_index], lexicon, rules, nrules)
        && 0 == have_positional)
      {
        libxs_registry_destroy(corpus);
        libxs_lexicon_destroy(lexicon);
        libxs_predict_destroy(answer_model);
        answer_bridge_free_loaded();
        answer_relation_rules_free();
        answer_relation_facts_free();
        answer_identity_facts_free();
        answer_describe_facts_free();
        answer_docdef_facts_free();
        free(basenames);
        return EXIT_FAILURE;
      }
    }
    for (; i < argc; ++i) {
      corpus_ingest_file(corpus, argv[i], lexicon, rules, nrules);
    }
    corpus_save(corpus);
    converse_lexicon_save(lexicon);
    { libxs_predict_t* trained = converse_predict_train(corpus,
        predict_profile);
      if (NULL != trained) {
        libxs_predict_destroy(answer_model);
        answer_model = trained;
        converse_predict_save(answer_model);
      }
    }
  }
  else fprintf(stderr, "warm start: reusing %s\n", converse_path_corpus);
  free(basenames);
  answer_relation_facts_build(corpus);
  answer_relation_facts_report(stderr);
  answer_identity_facts_build(corpus);
  answer_identity_facts_report(stderr);
  answer_describe_facts_build(corpus);
  answer_describe_facts_report(stderr);
  answer_docdef_facts_build(corpus);
  answer_docdef_facts_report(stderr);

  libxs_registry_info(corpus, &rinfo);
  fprintf(stderr, "corpus: %lu sentences\n", (unsigned long)rinfo.size);
  predict_ntotal = (long)rinfo.size;

  if (0 != learn_mode) {
    fprintf(stderr, "learned: %s, %s, %s\n", converse_path_corpus,
      converse_path_lexicon, converse_path_predict);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return EXIT_SUCCESS;
  }

  if (0 != predict_eval_mode || 0 != complete_mode) {
    int mode_result = EXIT_FAILURE;
    int use_predict = (0 == strcmp(ngram_kind, "predict")) ? 1 : 0;
    int use_embed = (0 == strcmp(ngram_kind, "embed")) ? 1 : 0;
    int use_rerank = (0 == strcmp(ngram_kind, "rerank")) ? 1 : 0;
    int use_knnlm = (0 == strcmp(ngram_kind, "knnlm")) ? 1 : 0;
    libxs_predict_t* token_model = NULL;
    if (0 != use_predict) {
      token_model = token_predict_build(corpus, lexicon, rules, nrules,
        predict_profile, 0, ngram_holdout);
    }
    else {
      if (GRAN_BPE == ngram_gran_mode()) {
        bpe_build(corpus, ngram_holdout);
      }
      ngram_model = ngram_build(corpus, lexicon, rules, nrules, ngram_holdout);
      ngram_backoff_build(ngram_model, lexicon);
      if (0 != use_embed || 0 != use_knnlm) {
        token_emb_build(corpus, lexicon, rules, nrules, ngram_holdout);
        token_model = token_predict_build(corpus, lexicon, rules, nrules,
          predict_profile, 1, ngram_holdout);
      }
      else if (0 != use_rerank) {
        token_model = rerank_build(corpus, ngram_model, lexicon, rules,
          nrules, ngram_order, predict_profile, ngram_holdout);
      }
    }
    if (0 != predict_eval_mode) {
      if (0 != use_predict || 0 != use_embed) {
        char label[64];
        sprintf(label, "%s:%s", use_embed ? "embed" : predict_profile->name,
          predict_profile->name);
        mode_result = token_predict_eval(token_model, corpus, lexicon, rules,
          nrules, use_embed, ngram_holdout,
          use_embed ? label : predict_profile->name);
      }
      else if (0 != use_rerank) {
        char label[64];
        sprintf(label, "rerank:%s", predict_profile->name);
        mode_result = rerank_eval(ngram_model, token_model, corpus, lexicon,
          rules, nrules, ngram_order, ngram_holdout, label);
      }
      else if (0 != use_knnlm) {
        char label[64];
        sprintf(label, "knnlm:%s", predict_profile->name);
        mode_result = knnlm_eval(ngram_model, token_model, corpus, lexicon,
          rules, nrules, ngram_order, ngram_holdout, label);
      }
      else if (NULL != getenv("CONVERSE_GEN_EVAL")) {
        mode_result = ngram_gen_eval(ngram_model, corpus, lexicon, rules,
          nrules, ngram_holdout, ngram_kind);
      }
      else {
        mode_result = ngram_eval(ngram_model, corpus, lexicon, rules, nrules,
          ngram_order, ngram_holdout, ngram_kind);
      }
      ngram_stats(ngram_model);
    }
    else {
      printf("> ");
      fflush(stdout);
      while (NULL != fgets(line, (int)sizeof(line), stdin)) {
        size_t len = strlen(line);
        while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
        if (0 == len) { printf("> "); fflush(stdout); continue; }
        if (0 != use_predict || 0 != use_embed) {
          token_complete(token_model, lexicon, rules, nrules, use_embed, line,
            (int)len);
        }
        else if (0 != use_rerank) {
          rerank_complete(ngram_model, token_model, lexicon, rules, nrules,
            ngram_order, line, (int)len);
        }
        else if (0 != use_knnlm) {
          knnlm_complete(ngram_model, token_model, lexicon, rules, nrules,
            ngram_order, line, (int)len);
        }
        else ngram_complete(ngram_model, lexicon, rules, nrules,
          ngram_order, line, (int)len);
        printf("> ");
        fflush(stdout);
      }
      mode_result = EXIT_SUCCESS;
    }
    libxs_predict_destroy(token_model);
    knnlm_cache_free();
    token_emb_free();
    bpe_free();
    libxs_ngram_destroy(&converse_ngram);
    ngram_model = NULL;
    converse_lexicon_save(lexicon);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return mode_result;
  }

  if (0 != eval_mode) {
    int eval_result = eval_converse(corpus, lexicon, rules, nrules,
      answer_model, predict_profile);
    converse_lexicon_save(lexicon);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    answer_bridge_free_loaded();
    answer_relation_rules_free();
    answer_relation_facts_free();
    answer_identity_facts_free();
    answer_describe_facts_free();
    answer_docdef_facts_free();
    return eval_result;
  }

  { libxs_spatial_t spatial;
    if (EXIT_SUCCESS != libxs_spatial_build(&spatial, corpus)) {
      libxs_predict_destroy(answer_model);
      libxs_lexicon_destroy(lexicon);
      libxs_registry_destroy(corpus);
      answer_bridge_free_loaded();
      answer_relation_rules_free();
      answer_relation_facts_free();
      answer_identity_facts_free();
      answer_describe_facts_free();
      answer_docdef_facts_free();
      return EXIT_FAILURE;
    }
    ngram_model = ngram_build(corpus, lexicon, rules, nrules, 0);
    ngram_backoff_build(ngram_model, lexicon);
    conv_reset();
    printf("> ");
    fflush(stdout);
    while (NULL != fgets(line, (int)sizeof(line), stdin)) {
      size_t len = strlen(line);
      libxs_fprint_t query;
      size_t shape;
      while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
      if (0 == len) { printf("> "); fflush(stdout); continue; }
      shape = len;
      libxs_fprint(&query, LIBXS_DATATYPE_U8, line, 1,
        &shape, NULL, FPRINT_ORDER, 0, 0, 0);
      respond(&spatial, corpus, line, len, &query, budget,
        lexicon, rules, nrules, answer_model, predict_profile);
      printf("> ");
      fflush(stdout);
    }
    libxs_spatial_destroy(&spatial);
  }

  libxs_ngram_destroy(&converse_ngram);
  ngram_model = NULL;
  converse_lexicon_save(lexicon);
  libxs_predict_destroy(answer_model);
  libxs_lexicon_destroy(lexicon);
  libxs_registry_destroy(corpus);
  answer_bridge_free_loaded();
  answer_relation_rules_free();
  answer_relation_facts_free();
  answer_identity_facts_free();
  answer_describe_facts_free();
  answer_docdef_facts_free();
  return EXIT_SUCCESS;
}


static void corpus_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  LIBXS_UNUSED(value); LIBXS_UNUSED(key);
  LIBXS_UNUSED(key_size); LIBXS_UNUSED(value_size);
  LIBXS_UNUSED(udata);
}


static const answer_predict_profile_t* answer_predict_profile_default(void)
{
  return answer_predict_profiles;
}


static const answer_predict_profile_t* answer_predict_profile_find(
  const char* name)
{
  const answer_predict_profile_t* result = NULL;
  size_t nprofiles = sizeof(answer_predict_profiles)
    / sizeof(answer_predict_profiles[0]);
  size_t profile_pos;
  if (NULL != name) {
    for (profile_pos = 0; profile_pos < nprofiles && NULL == result;
      ++profile_pos)
    {
      if (0 == strcmp(name, answer_predict_profiles[profile_pos].name)) {
        result = answer_predict_profiles + profile_pos;
      }
    }
  }
  return result;
}


static void answer_predict_profile_list(FILE* stream)
{
  size_t nprofiles = sizeof(answer_predict_profiles)
    / sizeof(answer_predict_profiles[0]);
  size_t profile_pos;
  if (NULL != stream) {
    fprintf(stream, "  profiles:");
    for (profile_pos = 0; profile_pos < nprofiles; ++profile_pos) {
      fprintf(stream, " %s", answer_predict_profiles[profile_pos].name);
    }
    fprintf(stream, "\n");
  }
}


static void answer_bridge_free_const(const char* ptr)
{
  union { const char* cptr; void* ptr; } cvt;
  if (NULL != ptr) {
    cvt.cptr = ptr;
    free(cvt.ptr);
  }
}


static char* answer_bridge_copy_trim(const char* text)
{
  char* result = NULL;
  const char* begin;
  const char* end;
  size_t size;
  if (NULL != text) {
    begin = text;
    while ('\0' != *begin && 0 != isspace((unsigned char)*begin)) ++begin;
    end = begin + strlen(begin);
    while (end > begin && 0 != isspace((unsigned char)end[-1])) --end;
    size = (size_t)(end - begin);
    result = (char*)malloc(size + 1);
    if (NULL != result) {
      memcpy(result, begin, size);
      result[size] = '\0';
    }
  }
  return result;
}


static void answer_bridge_free_loaded(void)
{
  size_t bridge_pos;
  for (bridge_pos = 0; bridge_pos < answer_bridge_loaded_size; ++bridge_pos) {
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].name);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].query);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].evidence);
    answer_bridge_free_const(answer_bridge_loaded[bridge_pos].reply);
  }
  free(answer_bridge_loaded);
  answer_bridge_loaded = NULL;
  answer_bridge_loaded_size = 0;
}


static int answer_bridge_append_loaded(const char* name, const char* query,
  const char* evidence, const char* score, const char* reply)
{
  int result = EXIT_FAILURE;
  answer_bridge_t bridge;
  answer_bridge_t* bridges;
  LIBXS_MEMZERO(&bridge);
  bridge.name = answer_bridge_copy_trim(name);
  bridge.query = answer_bridge_copy_trim(query);
  bridge.evidence = answer_bridge_copy_trim(evidence);
  bridge.reply = answer_bridge_copy_trim(reply);
  bridge.score = (NULL != score) ? atof(score) : 0.0;
  if (bridge.score <= 0.0) bridge.score = 0.90;
  bridges = (answer_bridge_t*)realloc(answer_bridge_loaded,
    (answer_bridge_loaded_size + 1) * sizeof(*bridges));
  if (NULL != bridge.name && '\0' != bridge.name[0]
    && NULL != bridge.query && '\0' != bridge.query[0]
    && NULL != bridge.evidence && '\0' != bridge.evidence[0]
    && NULL != bridge.reply && '\0' != bridge.reply[0]
    && NULL != bridges)
  {
    answer_bridge_loaded = bridges;
    answer_bridge_loaded[answer_bridge_loaded_size] = bridge;
    ++answer_bridge_loaded_size;
    result = EXIT_SUCCESS;
  }
  else {
    answer_bridge_free_const(bridge.name);
    answer_bridge_free_const(bridge.query);
    answer_bridge_free_const(bridge.evidence);
    answer_bridge_free_const(bridge.reply);
  }
  return result;
}


static int answer_bridge_parse_line(char* line)
{
  int result = EXIT_FAILURE;
  char* fields[5];
  char* cursor;
  int field_pos;
  if (NULL == line) return EXIT_FAILURE;
  cursor = line;
  for (field_pos = 0; field_pos < 5; ++field_pos) fields[field_pos] = NULL;
  for (field_pos = 0; field_pos < 4 && NULL != cursor; ++field_pos) {
    char* sep = strchr(cursor, '|');
    if (NULL != sep) {
      *sep = '\0';
      fields[field_pos] = cursor;
      cursor = sep + 1;
    }
    else {
      cursor = NULL;
    }
  }
  if (NULL != cursor) {
    fields[4] = cursor;
    result = answer_bridge_append_loaded(fields[0], fields[1], fields[2],
      fields[3], fields[4]);
  }
  return result;
}


static size_t answer_bridge_load_file(const char* path)
{
  size_t result = 0;
  FILE* file;
  if (NULL == path) return 0;
  answer_bridge_free_loaded();
  file = fopen(path, "r");
  if (NULL != file) {
    char line[BRIDGE_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len = strlen(line);
      char* begin;
      while (len > 0 && ('\n' == line[len - 1] || '\r' == line[len - 1])) {
        line[--len] = '\0';
      }
      begin = line;
      while ('\0' != *begin && 0 != isspace((unsigned char)*begin)) ++begin;
      if ('\0' != *begin && '#' != *begin
        && EXIT_SUCCESS == answer_bridge_parse_line(begin))
      {
        ++result;
      }
    }
    fclose(file);
  }
  return result;
}


static void answer_bridge_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "bridges: %lu loaded\n",
      (unsigned long)answer_bridge_loaded_size);
  }
}


static void answer_relation_rules_free(void)
{
  free(answer_relation_rules);
  answer_relation_rules = NULL;
  answer_relation_rules_size = 0;
}


static char* answer_relation_rule_trim(char* text)
{
  char* result = text;
  char* end;
  if (NULL != result) {
    while ('\0' != *result && 0 != isspace((unsigned char)*result)) ++result;
    end = result + strlen(result);
    while (end > result && 0 != isspace((unsigned char)end[-1])) --end;
    *end = '\0';
  }
  return result;
}


static int answer_relation_rule_kind(const char* text)
{
  int result = 0;
  if (NULL != text) {
    if (0 == strcmp(text, "alias")) result = RELATION_RULE_ALIAS;
    else if (0 == strcmp(text, "person")) result = RELATION_RULE_PERSON;
    else if (0 == strcmp(text, "skip")) result = RELATION_RULE_SKIP;
  }
  return result;
}


static int answer_relation_rule_append(int kind, const char* relation,
  const char* term)
{
  int result = EXIT_FAILURE;
  answer_relation_rule_t* rules;
  if (kind > 0 && NULL != term && '\0' != term[0]
    && strlen(term) < sizeof(answer_relation_rules[0].term)
    && (RELATION_RULE_ALIAS != kind
      || (NULL != relation && '\0' != relation[0]
        && strlen(relation) < sizeof(answer_relation_rules[0].relation))))
  {
    rules = (answer_relation_rule_t*)realloc(answer_relation_rules,
      (answer_relation_rules_size + 1) * sizeof(*rules));
    if (NULL != rules) {
      answer_relation_rules = rules;
      LIBXS_MEMZERO(answer_relation_rules + answer_relation_rules_size);
      answer_relation_rules[answer_relation_rules_size].kind = kind;
      if (NULL != relation) {
        strcpy(answer_relation_rules[answer_relation_rules_size].relation,
          relation);
      }
      strcpy(answer_relation_rules[answer_relation_rules_size].term, term);
      ++answer_relation_rules_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_rule_parse_line(char* line)
{
  int result = EXIT_FAILURE;
  char* fields[3];
  char* cursor;
  int field_pos;
  if (NULL == line) return EXIT_FAILURE;
  cursor = answer_relation_rule_trim(line);
  for (field_pos = 0; field_pos < 3; ++field_pos) fields[field_pos] = NULL;
  for (field_pos = 0; field_pos < 3 && NULL != cursor; ++field_pos) {
    char* sep = strchr(cursor, '|');
    if (NULL != sep) {
      *sep = '\0';
      fields[field_pos] = answer_relation_rule_trim(cursor);
      cursor = sep + 1;
    }
    else {
      fields[field_pos] = answer_relation_rule_trim(cursor);
      cursor = NULL;
    }
  }
  if (NULL != fields[0] && NULL != fields[1]) {
    int kind = answer_relation_rule_kind(fields[0]);
    if (RELATION_RULE_ALIAS == kind && NULL != fields[2]) {
      result = answer_relation_rule_append(kind, fields[1], fields[2]);
    }
    else if (RELATION_RULE_ALIAS != kind) {
      result = answer_relation_rule_append(kind, NULL, fields[1]);
    }
  }
  return result;
}


static size_t answer_relation_rules_load_file(const char* path)
{
  size_t result = 0;
  FILE* file;
  if (NULL == path) return 0;
  answer_relation_rules_free();
  file = fopen(path, "r");
  if (NULL != file) {
    char line[RELATION_LINE_MAX];
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len = strlen(line);
      char* begin;
      while (len > 0 && ('\n' == line[len - 1] || '\r' == line[len - 1])) {
        line[--len] = '\0';
      }
      begin = answer_relation_rule_trim(line);
      if ('\0' != *begin && '#' != *begin
        && EXIT_SUCCESS == answer_relation_rule_parse_line(begin))
      {
        ++result;
      }
    }
    fclose(file);
  }
  return result;
}


static void answer_relation_rules_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "relation rules: %lu loaded\n",
      (unsigned long)answer_relation_rules_size);
  }
}


static int answer_relation_rule_has_term(int kind, const char* text,
  int text_len)
{
  int result = 0;
  size_t rule_pos;
  if (NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (rule->kind == kind
        && 0 != text_contains_word_ci(text, text_len, rule->term))
      {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len)
{
  int result = -1;
  size_t rule_pos;
  if (NULL != relation && NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && result < 0;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (RELATION_RULE_ALIAS == rule->kind
        && 0 != text_contains_word_ci(rule->relation,
          (int)strlen(rule->relation), relation))
      {
        result = text_find_word_ci(text, text_len, rule->term);
        if (result >= 0 && NULL != alias_len) {
          *alias_len = (int)strlen(rule->term);
        }
      }
    }
  }
  return result;
}


/* Resolve the per-corpus namespace from the prefix: state and companion files
   live in the current directory keyed by the prefix basename, so each corpus
   owns its own converse.dat/eval/relations/... A prefix of "." (the default)
   takes the working directory's own name, which reproduces the plain
   "converse.*" layout when run from the sample directory. */
static void converse_namespace_init(const char* prefix)
{
  char base[CONVERSE_PATH_MAX - 16];
  char cwd[CONVERSE_PATH_MAX];
  const char* name = NULL;
  int base_len = 0;
  if (NULL == prefix || '\0' == prefix[0] || 0 == strcmp(prefix, ".")) {
    if (NULL != getcwd(cwd, sizeof(cwd))) {
      const char* slash = strrchr(cwd, '/');
      name = (NULL != slash && '\0' != slash[1]) ? slash + 1 : cwd;
    }
  }
  else {
    const char* slash = strrchr(prefix, '/');
    name = (NULL != slash) ? slash + 1 : prefix;
  }
  if (NULL != name && '\0' != name[0]) {
    base_len = (int)strlen(name);
    if (base_len >= (int)sizeof(base)) base_len = (int)sizeof(base) - 1;
    memcpy(base, name, (size_t)base_len);
    base[base_len] = '\0';
  }
  if (base_len > 0 && 0 != strcmp(base, "converse")) {
    sprintf(converse_path_corpus, "%s.dat", base);
    sprintf(converse_path_lexicon, "%s.lex", base);
    sprintf(converse_path_predict, "%s.prd", base);
    sprintf(converse_path_bridge, "%s.bridges", base);
    sprintf(converse_path_relation, "%s.relations", base);
    sprintf(converse_path_eval, "%s.eval", base);
    sprintf(converse_path_predict_eval, "%s.predict", base);
  }
}


static libxs_registry_t* corpus_load(void)
{
  libxs_registry_t* result = NULL;
  FILE* f = fopen(converse_path_corpus, "rb");
  if (NULL != f) {
    long len;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, f) == len) {
          result = libxs_registry_load(buf, (size_t)len, corpus_fixup, NULL);
        }
        free(buf);
      }
    }
    fclose(f);
  }
  return result;
}


static int corpus_save(const libxs_registry_t* corpus)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == corpus) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_registry_save(corpus, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_registry_save(corpus, buf, &size)) {
        FILE* f = fopen(converse_path_corpus, "wb");
        if (NULL != f) {
          if (fwrite(buf, 1, size, f) == size) result = EXIT_SUCCESS;
          fclose(f);
        }
      }
      free(buf);
    }
  }
  return result;
}


static libxs_lexicon_t* converse_lexicon_load(void)
{
  libxs_lexicon_t* result = NULL;
  FILE* file = fopen(converse_path_lexicon, "rb");
  if (NULL != file) {
    long len;
    fseek(file, 0, SEEK_END);
    len = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, file) == len) {
          result = libxs_lexicon_load(buf, (size_t)len);
        }
        free(buf);
      }
    }
    fclose(file);
  }
  return result;
}


static int converse_lexicon_save(const libxs_lexicon_t* lexicon)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == lexicon) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_lexicon_save(lexicon, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_lexicon_save(lexicon, buf, &size)) {
        FILE* file = fopen(converse_path_lexicon, "wb");
        if (NULL != file) {
          if (fwrite(buf, 1, size, file) == size) result = EXIT_SUCCESS;
          fclose(file);
        }
      }
      free(buf);
    }
  }
  return result;
}


static libxs_predict_t* converse_predict_load(void)
{
  libxs_predict_t* result = NULL;
  FILE* file = fopen(converse_path_predict, "rb");
  if (NULL != file) {
    long len;
    fseek(file, 0, SEEK_END);
    len = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, file) == len) {
          result = libxs_predict_load(buf, (size_t)len);
        }
        free(buf);
      }
    }
    fclose(file);
  }
  return result;
}


static int converse_predict_save(const libxs_predict_t* model)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == model) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_predict_save(model, buf, &size)) {
        FILE* file = fopen(converse_path_predict, "wb");
        if (NULL != file) {
          if (fwrite(buf, 1, size, file) == size) result = EXIT_SUCCESS;
          fclose(file);
        }
      }
      free(buf);
    }
  }
  return result;
}


static int count_words(const unsigned char* text, int length)
{
  int n = 0, i = 0;
  while (i < length) {
    while (i < length && (0 != isspace(text[i]) || 0 != ispunct(text[i]))) ++i;
    if (i < length) {
      ++n;
      while (i < length && 0 == isspace(text[i]) && 0 == ispunct(text[i])) ++i;
    }
  }
  return n;
}


static size_t text_closer_size(const unsigned char* text, size_t size,
  size_t pos)
{
  size_t result = 0;
  if (NULL != text && pos < size) {
    unsigned char ch = text[pos];
    if ('"' == ch || '\'' == ch || ')' == ch || ']' == ch) result = 1;
    else if (pos + 2 < size && 0xe2 == text[pos]
      && 0x80 == text[pos + 1]
      && (0x99 == text[pos + 2] || 0x9d == text[pos + 2]))
    {
      result = 3;
    }
  }
  return result;
}


static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos)
{
  int result = 0;
  if (NULL != text && pos < size
    && ('.' == text[pos] || '?' == text[pos] || '!' == text[pos]))
  {
    size_t next = pos + 1;
    size_t close_size = text_closer_size(text, size, next);
    while (0 != close_size) {
      next += close_size;
      close_size = text_closer_size(text, size, next);
    }
    if (next >= size || 0 != isspace(text[next])) result = 1;
  }
  return result;
}


static int text_starts_sentence(const char* text, int text_len)
{
  int result = 0;
  int pos = 0;
  if (NULL == text || text_len <= 0) return 0;
  while (pos < text_len && 0 != isspace((unsigned char)text[pos])) ++pos;
  if (pos < text_len) {
    const unsigned char* utext = (const unsigned char*)text;
    unsigned char ch = utext[pos];
    if (',' != ch && ';' != ch && ':' != ch && ')' != ch && ']' != ch
      && 0 == (pos + 2 < text_len && 0xe2 == utext[pos]
        && 0x80 == utext[pos + 1]
        && (0x99 == utext[pos + 2] || 0x9d == utext[pos + 2])))
    {
      result = 1;
    }
  }
  return result;
}


static int text_ends_sentence(const char* text, int text_len)
{
  int result = 0;
  int end = text_len;
  if (NULL == text || text_len <= 0) return 0;
  while (end > 0 && 0 != isspace((unsigned char)text[end - 1])) --end;
  while (end > 0) {
    const unsigned char* utext = (const unsigned char*)text;
    unsigned char ch = utext[end - 1];
    if ('"' == ch || '\'' == ch || ')' == ch || ']' == ch) --end;
    else if (end >= 3 && 0xe2 == utext[end - 3]
      && 0x80 == utext[end - 2]
      && (0x99 == utext[end - 1] || 0x9d == utext[end - 1]))
    {
      end -= 3;
    }
    else break;
  }
  if (end > 0) {
    unsigned char ch = (unsigned char)text[end - 1];
    if ('.' == ch || '?' == ch || '!' == ch) result = 1;
  }
  return result;
}


static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = 0;
  libxs_lexeme_stream_t stream;
  size_t lexeme_pos;
  libxs_lexeme_stream_init(&stream);
  if (NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, length, rules, nrules,
      NULL, 0, 1))
  {
    for (lexeme_pos = 0; lexeme_pos < stream.size && 0 == result;
      ++lexeme_pos)
    {
      if (0 != (stream.data[lexeme_pos].flags & LIBXS_LEXEME_QUESTION)) {
        result = 1;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id)
{
  int result = 0;
  size_t lexeme_pos;
  if (NULL != stream && 0 != id) {
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      if (stream->data[lexeme_pos].id == id) result = 1;
    }
  }
  return result;
}


static int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id)
{
  int result = 0;
  unsigned short token_pos;
  if (NULL != entry && 0 != id) {
    for (token_pos = 0; token_pos < entry->ntokens && 0 == result;
      ++token_pos)
    {
      if (entry->token_ids[token_pos] == id) result = 1;
    }
  }
  return result;
}


static int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text)
{
  int result = 0;
  int length = 0;
  const char* stored;
  if (NULL != lexicon && NULL != lexeme && NULL != text) {
    stored = libxs_lexicon_text(lexicon, lexeme->id, &length, NULL);
    if (NULL != stored && (int)strlen(text) == length
      && 0 == memcmp(stored, text, (size_t)length)) result = 1;
  }
  return result;
}


static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text)
{
  int result = 0;
  size_t lexeme_pos;
  if (NULL != stream && NULL != lexicon && NULL != text) {
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      result = lexeme_text_is(lexicon, stream->data + lexeme_pos, text);
    }
  }
  return result;
}


static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance)
{
  int result = 0;
  size_t lexeme_pos;
  char lhs[64];
  if (NULL != stream && NULL != lexicon && NULL != text && text_len > 0
    && text_len < (int)sizeof(lhs) && tolerance >= 0)
  {
    memcpy(lhs, text, (size_t)text_len);
    lhs[text_len] = '\0';
    for (lexeme_pos = 0; lexeme_pos < stream->size && 0 == result;
      ++lexeme_pos)
    {
      const libxs_lexeme_t* lexeme = stream->data + lexeme_pos;
      if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
      {
        int rhs_len = 0;
        const char* rhs = libxs_lexicon_text(lexicon, lexeme->id,
          &rhs_len, NULL);
        if (NULL != rhs && rhs_len > 0 && rhs_len < 64) {
          char rhs_buf[64];
          memcpy(rhs_buf, rhs, (size_t)rhs_len);
          rhs_buf[rhs_len] = '\0';
          if (libxs_stridist(lhs, rhs_buf) <= tolerance) result = 1;
        }
      }
    }
  }
  return result;
}


static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon)
{
  int result = QUERY_GENERIC;
  size_t lexeme_pos;
  if (NULL != query && NULL != lexicon) {
    for (lexeme_pos = 0; lexeme_pos < query->size
      && QUERY_GENERIC == result; ++lexeme_pos)
    {
      const libxs_lexeme_t* lexeme = query->data + lexeme_pos;
      if (0 != (lexeme->flags & LIBXS_LEXEME_WORD)) {
        if (0 != lexeme_text_is(lexicon, lexeme, "who")) result = QUERY_WHO;
        else if (0 != lexeme_text_is(lexicon, lexeme, "what")) result = QUERY_WHAT;
        else if (0 != lexeme_text_is(lexicon, lexeme, "where")) result = QUERY_WHERE;
        else if (0 != lexeme_text_is(lexicon, lexeme, "when")) result = QUERY_WHEN;
        else if (0 != lexeme_text_is(lexicon, lexeme, "why")) result = QUERY_WHY;
        else if (0 != lexeme_text_is(lexicon, lexeme, "how")) result = QUERY_HOW;
        else if (0 != lexeme_text_is(lexicon, lexeme, "is")
          || 0 != lexeme_text_is(lexicon, lexeme, "are")
          || 0 != lexeme_text_is(lexicon, lexeme, "was")
          || 0 != lexeme_text_is(lexicon, lexeme, "were")
          || 0 != lexeme_text_is(lexicon, lexeme, "do")
          || 0 != lexeme_text_is(lexicon, lexeme, "does")
          || 0 != lexeme_text_is(lexicon, lexeme, "did")
          || 0 != lexeme_text_is(lexicon, lexeme, "can")
          || 0 != lexeme_text_is(lexicon, lexeme, "could")
          || 0 != lexeme_text_is(lexicon, lexeme, "has")
          || 0 != lexeme_text_is(lexicon, lexeme, "have"))
        {
          result = QUERY_YESNO;
        }
      }
    }
  }
  return result;
}


static int query_type_prefers_sentence(int query_type)
{
  int result = 0;
  if (QUERY_WHO == query_type || QUERY_WHAT == query_type
    || QUERY_WHERE == query_type || QUERY_WHEN == query_type
    || QUERY_YESNO == query_type)
  {
    result = 1;
  }
  return result;
}


static int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_FAILURE;
  const size_t shape = (size_t)len;
  libxs_lexeme_stream_t stream;
  size_t lexeme_pos;
  libxs_lexeme_stream_init(&stream);
  if (NULL == entry || NULL == text || len <= 0) return EXIT_FAILURE;
  memset(entry, 0, sizeof(*entry));
  if (EXIT_SUCCESS == libxs_fprint(&entry->fprint, LIBXS_DATATYPE_U8,
    text, 1, &shape, NULL, FPRINT_ORDER, 0, 0, 0))
  {
    entry->text_len = len;
    memcpy(entry->text, text, (size_t)len);
    entry->connector = CONN_NEWLINE;
    entry->scale = scale;
    result = EXIT_SUCCESS;
  }
  if (EXIT_SUCCESS == result && NULL != lexicon && NULL != rules
    && nrules > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, text, (size_t)len, rules, nrules,
      NULL, 0, 1))
  {
    for (lexeme_pos = 0; lexeme_pos < stream.size; ++lexeme_pos) {
      const libxs_lexeme_t* lexeme = stream.data + lexeme_pos;
      if (0 != (lexeme->flags & LIBXS_LEXEME_ENTITY)) {
        entry->lexical_flags |= ENTRY_LEX_ENTITY;
        ++entry->nentities;
      }
      if (0 != (lexeme->flags & LIBXS_LEXEME_NUMBER)) {
        entry->lexical_flags |= ENTRY_LEX_NUMBER;
        ++entry->nnumbers;
      }
      if (0 != (lexeme->flags & LIBXS_LEXEME_QUESTION)) {
        entry->lexical_flags |= ENTRY_LEX_QUESTION;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "in")
        || 0 != lexeme_text_is(lexicon, lexeme, "at")
        || 0 != lexeme_text_is(lexicon, lexeme, "near")
        || 0 != lexeme_text_is(lexicon, lexeme, "from")
        || 0 != lexeme_text_is(lexicon, lexeme, "inside")
        || 0 != lexeme_text_is(lexicon, lexeme, "outside"))
      {
        entry->lexical_flags |= ENTRY_LEX_PLACE;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "because")
        || 0 != lexeme_text_is(lexicon, lexeme, "therefore")
        || 0 != lexeme_text_is(lexicon, lexeme, "since")
        || 0 != lexeme_text_is(lexicon, lexeme, "hence")
        || 0 != lexeme_text_is(lexicon, lexeme, "thus")
        || 0 != lexeme_text_is(lexicon, lexeme, "reason")
        || 0 != lexeme_text_is(lexicon, lexeme, "result"))
      {
        entry->lexical_flags |= ENTRY_LEX_CAUSE;
      }
      if (0 != lexeme_text_is(lexicon, lexeme, "by")
        || 0 != lexeme_text_is(lexicon, lexeme, "through")
        || 0 != lexeme_text_is(lexicon, lexeme, "using")
        || 0 != lexeme_text_is(lexicon, lexeme, "with")
        || 0 != lexeme_text_is(lexicon, lexeme, "via")
        || 0 != lexeme_text_is(lexicon, lexeme, "method")
        || 0 != lexeme_text_is(lexicon, lexeme, "process"))
      {
        entry->lexical_flags |= ENTRY_LEX_METHOD;
      }
      if (entry->ntokens < ENTRY_TOKEN_MAX
        && 0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 == (lexeme->flags & LIBXS_LEXEME_STOP)
        && 0 == entry_sketch_has_id(entry, lexeme->id))
      {
        entry->token_ids[entry->ntokens] = lexeme->id;
        entry->token_flags[entry->ntokens] = lexeme->flags;
        ++entry->ntokens;
        ++entry->ncontent;
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static void corpus_entry_set_section(corpus_entry_t* entry,
  const char* section, int section_len)
{
  int copy_len;
  if (NULL == entry || NULL == section || section_len <= 0) return;
  copy_len = section_len;
  if (copy_len >= ENTRY_SECTION_MAX) copy_len = ENTRY_SECTION_MAX - 1;
  memcpy(entry->section, section, (size_t)copy_len);
  entry->section[copy_len] = '\0';
  entry->section_len = (unsigned short)copy_len;
}


static int corpus_title_prefix(const unsigned char* text, int len,
  char* title, int title_size)
{
  int result = 0;
  int pos, prefix_words = 0, in_word = 0;
  int title_begin = 0, title_end;
  if (NULL == text || NULL == title || title_size <= 0 || len <= 0) return 0;
  title[0] = '\0';
  for (pos = 0; pos < len; ++pos) {
    unsigned char ch = text[pos];
    if (0 != islower(ch) || '.' == ch || ',' == ch || ';' == ch
      || ':' == ch || '!' == ch || '?' == ch)
    {
      break;
    }
    if (0 != isupper(ch) && pos + 1 < len && 0 != islower(text[pos + 1])) {
      break;
    }
    if (0 != isupper(ch)) {
      if (0 == in_word) {
        ++prefix_words;
        in_word = 1;
      }
    }
    else if (0 != isspace(ch)) {
      in_word = 0;
    }
  }
  title_end = pos;
  if (prefix_words >= 2 && pos < len && pos > 0) {
    int prev_end = pos;
    int prev_start;
    while (prev_end > 0 && 0 != isspace(text[prev_end - 1])) --prev_end;
    prev_start = prev_end;
    while (prev_start > 0 && 0 == isspace(text[prev_start - 1])) {
      --prev_start;
    }
    if ((1 == prev_end - prev_start && 'A' == text[prev_start])
      || (2 == prev_end - prev_start && 'A' == text[prev_start]
        && 'N' == text[prev_start + 1]))
    {
      title_end = prev_start;
    }
    while (title_begin < title_end && 0 != isspace(text[title_begin])) {
      ++title_begin;
    }
    while (title_end > title_begin && 0 != isspace(text[title_end - 1])) {
      --title_end;
    }
    result = title_end - title_begin;
    if (result >= title_size) result = title_size - 1;
    if (result > 0) {
      memcpy(title, text + title_begin, (size_t)result);
      title[result] = '\0';
    }
  }
  return result;
}


static int corpus_entry_same_section(const corpus_entry_t* lhs,
  size_t lhs_size, const corpus_entry_t* rhs)
{
  int result = 0;
  if (NULL != lhs && NULL != rhs) {
    if (lhs_size < sizeof(*lhs)) result = (0 == rhs->section_len) ? 1 : 0;
    else if (lhs->section_len == rhs->section_len
      && 0 == libxs_memcmp(lhs->section, rhs->section, lhs->section_len))
    {
      result = 1;
    }
  }
  return result;
}


static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry)
{
  int result = 0;
  unsigned char key[16];
  size_t key_size = 0;
  unsigned int seq;
  int matched = 0;
  if (NULL == corpus || NULL == entry) return 0;
  corpus_key_from_fprint(&entry->fprint, key, &key_size);
  for (seq = 0; seq < 65536; ++seq) {
    void* existing;
    unsigned short seq_key = (unsigned short)seq;
    memcpy(key + 8, &seq_key, 2);
    key_size = 10;
    existing = libxs_registry_get(corpus, key, key_size, NULL);
    if (NULL == existing) {
      if (0 == matched) {
        libxs_registry_set(corpus, key, key_size,
          entry, sizeof(*entry), NULL);
        result = 1;
      }
      break;
    }
    else {
      const corpus_entry_t* old_entry = (const corpus_entry_t*)existing;
      size_t old_size = libxs_registry_value_size(corpus, key,
        key_size, NULL);
      if (old_entry->text_len == entry->text_len
        && 0 == libxs_memcmp(old_entry->text, entry->text,
          (size_t)entry->text_len)
        && 0 != corpus_entry_same_section(old_entry, old_size, entry))
      {
        matched = 1;
        if (old_size != sizeof(*entry)
          || (0 == old_entry->ntokens && entry->ntokens > 0))
        {
          libxs_registry_set(corpus, key, key_size,
            entry, sizeof(*entry), NULL);
        }
        if (old_size == sizeof(*entry) && old_entry->ntokens > 0) break;
      }
    }
  }
  return result;
}


static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size)
{
  int result = 0;
  size_t pos = 0, begin, end, comma_pos;
  int marker_pos;
  int end_pos;
  if (NULL == query_text || NULL == title || title_size <= 0) return 0;
  title[0] = '\0';
  while (pos < query_len && 0 != isspace((unsigned char)query_text[pos])) {
    ++pos;
  }
  if (pos + 3 < query_len
    && 'i' == tolower((unsigned char)query_text[pos])
    && 'n' == tolower((unsigned char)query_text[pos + 1])
    && 0 != isspace((unsigned char)query_text[pos + 2]))
  {
    begin = pos + 3;
    while (begin < query_len
      && 0 != isspace((unsigned char)query_text[begin])) ++begin;
    end = begin;
    while (end < query_len && ',' != query_text[end]
      && '?' != query_text[end] && '!' != query_text[end]) ++end;
    comma_pos = end;
    while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) {
      --end;
    }
    result = (int)(end - begin);
    if (result >= title_size) result = title_size - 1;
    if (result > 0 && comma_pos < query_len && ',' == query_text[comma_pos]) {
      memcpy(title, query_text + begin, (size_t)result);
      title[result] = '\0';
    }
    else result = 0;
  }
  if (0 == result) {
    marker_pos = text_find_ci(query_text, (int)query_len, " in ");
    if (marker_pos >= 0) {
      begin = (size_t)marker_pos + 4;
      while (begin < query_len
        && 0 != isspace((unsigned char)query_text[begin])) ++begin;
      end = query_len;
      end_pos = text_find_ci(query_text + begin, (int)(query_len - begin),
        " is ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " are ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " was ");
      if (end_pos < 0) end_pos = text_find_ci(query_text + begin,
        (int)(query_len - begin), " were ");
      if (end_pos >= 0) end = begin + (size_t)end_pos;
      else {
        while (end > begin && ('?' == query_text[end - 1]
          || '!' == query_text[end - 1] || '.' == query_text[end - 1]
          || ',' == query_text[end - 1]
          || 0 != isspace((unsigned char)query_text[end - 1]))) --end;
      }
      while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) {
        --end;
      }
      result = (int)(end - begin);
      if (result >= title_size) result = title_size - 1;
      if (result > 0) {
        memcpy(title, query_text + begin, (size_t)result);
        title[result] = '\0';
      }
    }
  }
  return result;
}


static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len)
{
  int result = 0;
  int entry_pos = 0, title_pos = 0, entry_len = 0;
  if (NULL == title || title_len <= 0) return 1;
  if (NULL != entry && entry_size >= sizeof(*entry)
    && '\0' != entry->section[0])
  {
    while (entry_len < ENTRY_SECTION_MAX && '\0' != entry->section[entry_len]) {
      ++entry_len;
    }
    while (entry_pos < entry_len && title_pos < title_len) {
      while (entry_pos < entry_len
        && 0 == isalnum((unsigned char)entry->section[entry_pos]))
      {
        ++entry_pos;
      }
      while (title_pos < title_len
        && 0 == isalnum((unsigned char)title[title_pos]))
      {
        ++title_pos;
      }
      if (entry_pos < entry_len && title_pos < title_len) {
        if (tolower((unsigned char)entry->section[entry_pos])
          != tolower((unsigned char)title[title_pos]))
        {
          break;
        }
        ++entry_pos;
        ++title_pos;
      }
    }
    while (entry_pos < entry_len
      && 0 == isalnum((unsigned char)entry->section[entry_pos])) ++entry_pos;
    while (title_pos < title_len
      && 0 == isalnum((unsigned char)title[title_pos])) ++title_pos;
    if (title_pos == title_len) {
      result = 1;
    }
    else if (title_len < ENTRY_SECTION_MAX) {
      char title_buf[ENTRY_SECTION_MAX];
      memcpy(title_buf, title, (size_t)title_len);
      title_buf[title_len] = '\0';
      if (0 == libxs_stridiff(entry->section, title_buf, NULL, 1, NULL)) {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size, int* upper_initial)
{
  static const char* const markers[] = {
    " is ", " are ", " was ", " were "
  };
  int result = 0;
  int marker_pos = -1;
  int marker_len = 0;
  int marker_index;
  size_t begin;
  size_t end;
  if (NULL == query_text || NULL == word || word_size <= 0) return 0;
  word[0] = '\0';
  if (NULL != upper_initial) *upper_initial = 0;
  for (marker_index = 0; marker_index < 4 && marker_pos < 0;
    ++marker_index)
  {
    marker_pos = text_find_ci(query_text, (int)query_len,
      markers[marker_index]);
    if (marker_pos >= 0) marker_len = (int)strlen(markers[marker_index]);
  }
  if (marker_pos >= 0) {
    begin = (size_t)marker_pos + (size_t)marker_len;
    do {
      while (begin < query_len
        && 0 != isspace((unsigned char)query_text[begin])) ++begin;
      end = begin;
      while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
        ++end;
      }
      result = (int)(end - begin);
      if (result > 0 && 0 != answer_relation_rule_has_term(RELATION_RULE_SKIP,
        query_text + begin, result))
      {
        begin = end;
        result = 0;
      }
    } while (0 == result && end < query_len);
    if (result >= word_size) result = word_size - 1;
    if (result > 0) {
      memcpy(word, query_text + begin, (size_t)result);
      word[result] = '\0';
      if (NULL != upper_initial) {
        *upper_initial = isupper((unsigned char)query_text[begin]) ? 1 : 0;
      }
    }
  }
  return result;
}


static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size)
{
  int result = 0;
  int marker_pos;
  size_t begin, end;
  if (NULL == query_text || NULL == actor || actor_size <= 0) return 0;
  actor[0] = '\0';
  marker_pos = text_find_ci(query_text, (int)query_len, " by ");
  if (marker_pos >= 0) {
    begin = (size_t)marker_pos + 4;
    while (begin < query_len
      && 0 != isspace((unsigned char)query_text[begin])) ++begin;
    end = begin;
    while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
      ++end;
    }
    while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
      ++end;
    }
    if ((3 == end - begin && 0 == strncmp(query_text + begin, "the", 3))
      || (1 == end - begin && 'a' == tolower((unsigned char)query_text[begin]))
      || (2 == end - begin && 0 == strncmp(query_text + begin, "an", 2)))
    {
      while (end < query_len
        && 0 != isspace((unsigned char)query_text[end])) ++end;
      while (end < query_len && 0 != isalnum((unsigned char)query_text[end])) {
        ++end;
      }
    }
    result = (int)(end - begin);
    if (result >= actor_size) result = actor_size - 1;
    if (result > 0) {
      memcpy(actor, query_text + begin, (size_t)result);
      actor[result] = '\0';
    }
  }
  return result;
}


static int answer_relation_copy_name(char* output, int output_size,
  const char* text, int text_len, int begin, int end)
{
  int result;
  while (begin < end && 0 == isalnum((unsigned char)text[begin])) ++begin;
  while (end > begin && 0 == isalnum((unsigned char)text[end - 1])) --end;
  result = end - begin;
  if (result >= output_size) result = output_size - 1;
  if (result > 0) {
    memcpy(output, text + begin, (size_t)result);
    output[result] = '\0';
    output[0] = (char)toupper((unsigned char)output[0]);
  }
  return result;
}


static int answer_relation_actor_has_token(const char* actor, int actor_len,
  const char* token, int token_len)
{
  int result = 0;
  char token_buf[64];
  if (NULL != actor && actor_len > 0 && NULL != token && token_len > 0
    && token_len < (int)sizeof(token_buf))
  {
    memcpy(token_buf, token, (size_t)token_len);
    token_buf[token_len] = '\0';
    result = text_contains_ci(actor, actor_len, token_buf);
  }
  return result;
}


static int answer_relation_copy_section_head(char* output, int output_size,
  const corpus_entry_t* entry)
{
  int result = 0;
  int end = 0;
  if (NULL == output || output_size <= 0 || NULL == entry
    || entry->section_len <= 0) return 0;
  while (end < entry->section_len
    && 0 != isalnum((unsigned char)entry->section[end])) ++end;
  result = answer_relation_copy_name(output, output_size, entry->section,
    entry->section_len, 0, end);
  if (3 == result && 0 != text_contains_ci(output, result, "the")) result = 0;
  if (1 == result && 0 != text_contains_ci(output, result, "a")) result = 0;
  if (2 == result && 0 != text_contains_ci(output, result, "an")) result = 0;
  if (result > 1) {
    int pos;
    for (pos = 1; pos < result; ++pos) {
      output[pos] = (char)tolower((unsigned char)output[pos]);
    }
  }
  return result;
}


static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos)
{
  int result = 0;
  int scan = 0;
  int first_word = 1;
  if (NULL != output && output_size > 0 && NULL != text && text_len > 0
    && cue_pos > 0)
  {
    output[0] = '\0';
    while (scan < cue_pos && scan < text_len) {
      int begin;
      int end;
      while (scan < cue_pos && scan < text_len
        && 0 == isalnum((unsigned char)text[scan])) ++scan;
      begin = scan;
      while (scan < cue_pos && scan < text_len
        && ('-' == text[scan] || 0 != isalnum((unsigned char)text[scan]))) {
        ++scan;
      }
      end = scan;
      if (end > begin && 0 != isupper((unsigned char)text[begin])
        && 0 == first_word
        && 0 == text_contains_ci(text + begin, end - begin, "Then")
        && 0 == text_contains_ci(text + begin, end - begin, "When")
        && 0 == text_contains_ci(text + begin, end - begin, "The")
        && 0 == text_contains_ci(text + begin, end - begin, "She")
        && 0 == text_contains_ci(text + begin, end - begin, "He"))
      {
        result = answer_relation_copy_name(output, output_size, text,
          text_len, begin, end);
        break;
      }
      if (end > begin) first_word = 0;
    }
  }
  return result;
}


static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len)
{
  int result = 0;
  int best_pos = -1;
  int best_len = 0;
  size_t rule_pos;
  if (NULL != text && text_len > 0 && limit > 0)
  {
    if (limit > text_len) limit = text_len;
    for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      int rule_len = (int)strlen(rule->term);
      int pos = 0;
      if (RELATION_RULE_PERSON == rule->kind && rule_len > 0) {
        while (pos < limit) {
          int found = text_find_word_ci(text + pos, limit - pos, rule->term);
          if (found < 0) break;
          found += pos;
          if (found > best_pos) {
            best_pos = found;
            best_len = rule_len;
          }
          pos = found + rule_len;
        }
      }
    }
    if (best_pos >= 0 && best_len > 0) {
      if (NULL != term_pos) *term_pos = best_pos;
      if (NULL != term_len) *term_len = best_len;
      result = 1;
    }
  }
  return result;
}


static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit)
{
  int result = 0;
  int term_pos = -1;
  int term_len = 0;
  if (NULL != output && output_size > 0
    && 0 != answer_relation_find_person_before(text, text_len, limit,
      &term_pos, &term_len))
  {
      result = answer_relation_copy_name(output, output_size, text,
        text_len, term_pos, term_pos + term_len);
  }
  return result;
}


static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match)
{
  int result = 0;
  char relation[64];
  char actor[64];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  if (NULL == query_text || NULL == entry || NULL == match
    || QUERY_WHO != query_type) return 0;
  memset(match, 0, sizeof(*match));
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  if (relation_len <= 0 || 0 != relation_upper) return 0;
  memcpy(match->relation, relation, (size_t)relation_len + 1);
  match->relation_len = relation_len;
  if (actor_len > 0) {
    memcpy(match->actor, actor, (size_t)actor_len + 1);
    match->actor_len = actor_len;
  }
  if (actor_len > 0)
  {
    int rel_pos = text_find_word_ci(entry->text, entry->text_len, relation);
    int alt_pos = -1;
    int alt_len = 0;
    int verb_pos;
    int verb_len;
    int actor_seen = text_contains_ci(entry->text, entry->text_len, actor);
    if (rel_pos < 0) alt_pos = answer_relation_rule_alias_pos(relation,
      entry->text, entry->text_len, &alt_len);
    verb_pos = (rel_pos >= 0) ? rel_pos : alt_pos;
    verb_len = (rel_pos >= 0) ? relation_len : alt_len;
    if (0 == actor_seen && verb_pos > 0
      && 0 != text_contains_word_ci(entry->text, verb_pos, "he"))
    {
      actor_seen = 1;
    }
    if (verb_pos >= 0 && 0 != actor_seen) {
      int obj_begin = verb_pos + ((verb_len > 0) ? verb_len : 1);
      int obj_end;
      if (verb_len <= 0) {
        while (obj_begin < entry->text_len
          && 0 != isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      }
      while (obj_begin < entry->text_len
        && 0 == isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      if (obj_begin + 2 < entry->text_len
        && 0 == strncmp(entry->text + obj_begin, "up", 2)
        && 0 == isalnum((unsigned char)entry->text[obj_begin + 2]))
      {
        obj_begin += 2;
        while (obj_begin < entry->text_len
          && 0 == isalnum((unsigned char)entry->text[obj_begin])) ++obj_begin;
      }
      obj_end = obj_begin;
      while (obj_end < entry->text_len
        && ('-' == entry->text[obj_end]
          || 0 != isalnum((unsigned char)entry->text[obj_end]))) ++obj_end;
      if ((3 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "the"))
        || (1 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "a"))
        || (2 == obj_end - obj_begin
          && 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "an")))
      {
        while (obj_end < entry->text_len
          && 0 == isalnum((unsigned char)entry->text[obj_end])) ++obj_end;
        obj_begin = obj_end;
        while (obj_end < entry->text_len
          && ('-' == entry->text[obj_end]
            || 0 != isalnum((unsigned char)entry->text[obj_end]))) ++obj_end;
      }
      if (obj_end > obj_begin
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "he")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "she")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "her")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "him")
        && 0 == text_contains_ci(entry->text + obj_begin,
          obj_end - obj_begin, "them"))
      {
        if (0 == answer_relation_actor_has_token(actor, actor_len,
          entry->text + obj_begin, obj_end - obj_begin))
        {
          match->answer_len = answer_relation_copy_name(match->answer,
            (int)sizeof(match->answer), entry->text, entry->text_len,
            obj_begin, obj_end);
        }
      }
      else if (obj_end > obj_begin
        && (0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "her")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "him")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "he")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "she")
          || 0 != text_contains_ci(entry->text + obj_begin,
            obj_end - obj_begin, "them")))
      {
        int scan = verb_pos - 1;
        while (scan > 0 && match->answer_len <= 0) {
          int end = scan + 1;
          int begin = scan;
          while (begin > 0
            && 0 != isalnum((unsigned char)entry->text[begin - 1])) --begin;
          if (end > begin
            && end - begin > 1
            && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
              entry->text + begin, end - begin)
            && 0 == text_contains_ci(entry->text + begin, end - begin,
              actor)
            && 0 == answer_relation_actor_has_token(actor, actor_len,
              entry->text + begin, end - begin)
            && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
              entry->text + begin, end - begin))
          {
            match->answer_len = answer_relation_copy_name(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              begin, end);
          }
          scan = begin - 1;
        }
        if (match->answer_len <= 0) {
          match->answer_len = answer_relation_copy_person_before(
            match->answer, (int)sizeof(match->answer), entry->text,
            entry->text_len, verb_pos);
        }
      }
      if (match->answer_len > 0) {
        if (0 != answer_relation_actor_has_token(actor, actor_len,
          match->answer, match->answer_len))
        {
          match->answer_len = 0;
        }
      }
      if (match->answer_len > 0) {
        match->score = 1.65;
        result = 1;
      }
    }
  }
  else if (actor_len <= 0 && 0 != text_contains_word_ci(entry->text,
    entry->text_len, relation))
  {
    int rel_pos = text_find_word_ci(entry->text, entry->text_len, relation);
    int made_pos = -1;
    int made_scan = 0;
    int before = rel_pos;
    int answer_begin, answer_end;
    while (rel_pos > made_scan) {
      int next_pos = text_find_word_ci(entry->text + made_scan,
        rel_pos - made_scan, "made");
      if (next_pos < 0) break;
      made_pos = made_scan + next_pos;
      made_scan = made_pos + 4;
    }
    if (rel_pos >= 0 && made_pos >= 0 && made_pos < rel_pos
      && rel_pos - made_pos <= 16)
    {
      if (made_pos >= 0 && made_pos < rel_pos) {
        match->made = 1;
        {
          int cue_pos = -1;
          int cue_len = 0;
          int possessive = 0;
          match->answer_len = answer_relation_copy_section_head(match->answer,
            (int)sizeof(match->answer), entry);
          if (0 != answer_relation_find_person_before(entry->text,
            entry->text_len, made_pos, &cue_pos, &cue_len))
          {
            int prev_end = cue_pos;
            int prev_begin;
            while (prev_end > 0
              && 0 == isalnum((unsigned char)entry->text[prev_end - 1])) {
              --prev_end;
            }
            prev_begin = prev_end;
            while (prev_begin > 0
              && 0 != isalnum((unsigned char)entry->text[prev_begin - 1])) {
              --prev_begin;
            }
            if (prev_end > prev_begin
              && 0 != answer_relation_rule_has_term(RELATION_RULE_SKIP,
                entry->text + prev_begin, prev_end - prev_begin))
            {
              possessive = 1;
            }
          }
          if (match->answer_len <= 0 && 0 == possessive && cue_pos >= 0) {
            match->answer_len = answer_relation_copy_antecedent(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              cue_pos);
          }
          if (match->answer_len <= 0 && 0 == possessive && cue_pos >= 0) {
            match->answer_len = answer_relation_copy_name(match->answer,
              (int)sizeof(match->answer), entry->text, entry->text_len,
              cue_pos, cue_pos + cue_len);
          }
        }
      }
      if (match->answer_len <= 0 && 0 == match->made) {
        while (before > 0
          && 0 == isalnum((unsigned char)entry->text[before - 1])) --before;
        answer_end = before;
        while (before > 0
          && 0 != isalnum((unsigned char)entry->text[before - 1])) --before;
        answer_begin = before;
        if (answer_end - answer_begin > 1
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "made")
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "too")
          && 0 == text_contains_ci(entry->text + answer_begin,
            answer_end - answer_begin, "the"))
        {
          match->answer_len = answer_relation_copy_name(match->answer,
            (int)sizeof(match->answer), entry->text, entry->text_len,
            answer_begin, answer_end);
        }
      }
      match->score = (match->answer_len > 0)
        ? ((0 != match->made) ? 1.85 : 1.25) : 0.0;
      result = (match->answer_len > 0) ? 1 : 0;
    }
  }
  return result;
}


static int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  size_t pos = 0;
  const char* copula;
  if (NULL == match || NULL == output || 0 == output_size
    || match->answer_len <= 0 || match->relation_len <= 0) return EXIT_FAILURE;
  copula = (0 != match->plural) ? " were " : " was ";
  if ((size_t)match->answer_len + strlen(copula)
    + (size_t)match->relation_len + (size_t)match->actor_len + 8
    < output_size)
  {
    memcpy(output + pos, match->answer, (size_t)match->answer_len);
    pos += (size_t)match->answer_len;
    if (0 != match->made) {
      static const char made_text[] = " is to be made ";
      memcpy(output + pos, made_text, sizeof(made_text) - 1);
      pos += sizeof(made_text) - 1;
    }
    else {
      memcpy(output + pos, copula, strlen(copula));
      pos += strlen(copula);
    }
    memcpy(output + pos, match->relation, (size_t)match->relation_len);
    pos += (size_t)match->relation_len;
    if (match->actor_len > 0) {
      static const char by_text[] = " by ";
      memcpy(output + pos, by_text, sizeof(by_text) - 1);
      pos += sizeof(by_text) - 1;
      memcpy(output + pos, match->actor, (size_t)match->actor_len);
      pos += (size_t)match->actor_len;
    }
    output[pos++] = '.';
    output[pos] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static int answer_relation_section_title(char* output, int output_size,
  const corpus_entry_t* entry, const answer_relation_match_t* match)
{
  int result = 0;
  if (NULL != output && output_size > 0 && NULL != entry && NULL != match
    && entry->section_len > match->answer_len && match->answer_len > 0
    && entry->section_len < output_size
    && 0 != text_contains_ci(entry->section, entry->section_len,
      match->answer))
  {
    int pos;
    int new_word = 1;
    int title_len = entry->section_len;
    for (pos = 0; pos < title_len; ++pos) {
      if ('[' == entry->section[pos] || '(' == entry->section[pos]) {
        title_len = pos;
      }
    }
    while (title_len > 0
      && 0 != isspace((unsigned char)entry->section[title_len - 1])) {
      --title_len;
    }
    memcpy(output, entry->section, (size_t)title_len);
    output[title_len] = '\0';
    for (pos = 0; pos < title_len; ++pos) {
      if (0 != isalnum((unsigned char)output[pos])) {
        output[pos] = (char)(0 != new_word
          ? toupper((unsigned char)output[pos])
          : tolower((unsigned char)output[pos]));
        new_word = 0;
      }
      else new_word = 1;
    }
    result = title_len;
  }
  return result;
}


static int answer_relation_same_answer(const char* lhs, int lhs_len,
  const char* rhs, int rhs_len)
{
  int result = 0;
  if (NULL != lhs && NULL != rhs && lhs_len > 0 && rhs_len > 0) {
    if (lhs_len == rhs_len && 0 != text_contains_ci(lhs, lhs_len, rhs)) {
      result = 1;
    }
    else if (lhs_len > rhs_len && 0 != text_contains_ci(lhs, lhs_len, rhs)) {
      result = 1;
    }
    else if (rhs_len > lhs_len && 0 != text_contains_ci(rhs, rhs_len, lhs)) {
      result = 1;
    }
  }
  return result;
}


static void answer_relation_facts_free(void)
{
  free(answer_relation_facts);
  answer_relation_facts = NULL;
  answer_relation_facts_size = 0;
}


static int answer_relation_fact_append(const corpus_entry_t* entry,
  const answer_relation_match_t* match)
{
  int result = EXIT_FAILURE;
  answer_relation_fact_t fact;
  answer_relation_fact_t* facts;
  size_t fact_pos;
  if (NULL == entry || NULL == match || match->answer_len <= 0
    || match->relation_len <= 0) return EXIT_FAILURE;
  LIBXS_MEMZERO(&fact);
  fact.answer_len = 0;
  if (match->actor_len > 0) {
    fact.answer_len = answer_relation_section_title(fact.answer,
      (int)sizeof(fact.answer), entry, match);
  }
  if (fact.answer_len <= 0) {
    fact.answer_len = match->answer_len;
    memcpy(fact.answer, match->answer, (size_t)fact.answer_len + 1);
  }
  fact.relation_len = match->relation_len;
  memcpy(fact.relation, match->relation, (size_t)fact.relation_len + 1);
  fact.actor_len = match->actor_len;
  if (fact.actor_len > 0) {
    memcpy(fact.actor, match->actor, (size_t)fact.actor_len + 1);
  }
  fact.section_len = entry->section_len;
  if (fact.section_len > 0) {
    memcpy(fact.section, entry->section, (size_t)fact.section_len);
    fact.section[fact.section_len] = '\0';
  }
  fact.plural = match->plural;
  fact.made = match->made;
  fact.score = match->score;
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    answer_relation_fact_t* old_fact = answer_relation_facts + fact_pos;
    if (old_fact->relation_len == fact.relation_len
      && old_fact->actor_len == fact.actor_len
      && old_fact->section_len == fact.section_len
      && 0 != answer_relation_same_answer(old_fact->answer,
        old_fact->answer_len, fact.answer, fact.answer_len)
      && 0 != text_contains_ci(old_fact->relation, old_fact->relation_len,
        fact.relation)
      && (0 == fact.actor_len
        || 0 != text_contains_ci(old_fact->actor, old_fact->actor_len,
          fact.actor))
      && (0 == fact.section_len
        || 0 != text_contains_ci(old_fact->section, old_fact->section_len,
          fact.section)))
    {
      if (fact.score > old_fact->score) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_relation_fact_t*)realloc(answer_relation_facts,
      (answer_relation_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_relation_facts = facts;
      answer_relation_facts[answer_relation_facts_size] = fact;
      ++answer_relation_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_fact_extract_actor(const char* text, int text_len,
  int verb_pos, int* scan_pos, char* actor, int actor_size)
{
  int result = 0;
  int begin, end;
  if (NULL == text || NULL == scan_pos || NULL == actor || actor_size <= 0) {
    return 0;
  }
  actor[0] = '\0';
  while (*scan_pos < verb_pos) {
    while (*scan_pos < verb_pos
      && 0 == isalnum((unsigned char)text[*scan_pos])) ++*scan_pos;
    begin = *scan_pos;
    while (*scan_pos < verb_pos
      && ('-' == text[*scan_pos]
        || 0 != isalnum((unsigned char)text[*scan_pos]))) ++*scan_pos;
    end = *scan_pos;
    if (end > begin && end - begin < actor_size
      && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
        text + begin, end - begin))
    {
      memcpy(actor, text + begin, (size_t)(end - begin));
      actor[end - begin] = '\0';
      result = end - begin;
      break;
    }
  }
  return result;
}


static int answer_relation_fact_extract_made(const corpus_entry_t* entry,
  int made_pos)
{
  int result = 0;
  int rel_begin, rel_end;
  char relation[64];
  char query[96];
  answer_relation_match_t match;
  if (NULL == entry || made_pos < 0 || made_pos + 4 >= entry->text_len) {
    return 0;
  }
  rel_begin = made_pos + 4;
  while (rel_begin < entry->text_len
    && 0 == isalnum((unsigned char)entry->text[rel_begin])) ++rel_begin;
  rel_end = rel_begin;
  while (rel_end < entry->text_len
    && ('-' == entry->text[rel_end]
      || 0 != isalnum((unsigned char)entry->text[rel_end]))) ++rel_end;
  if (rel_end > rel_begin && rel_end - rel_begin > 2
    && rel_end - rel_begin < (int)sizeof(relation)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
      entry->text + rel_begin, rel_end - rel_begin)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON,
      entry->text + rel_begin, rel_end - rel_begin))
  {
    memcpy(relation, entry->text + rel_begin, (size_t)(rel_end - rel_begin));
    relation[rel_end - rel_begin] = '\0';
    sprintf(query, "who is %s?", relation);
    if (0 != answer_relation_match_query(query, strlen(query), QUERY_WHO,
      entry, &match)
      && match.made != 0
      && EXIT_SUCCESS == answer_relation_fact_append(entry, &match))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_relation_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_relation_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    int made_scan = 0;
    size_t rule_pos;
    while (made_scan < entry->text_len) {
      int made_pos = text_find_word_ci(entry->text + made_scan,
        entry->text_len - made_scan, "made");
      if (made_pos < 0) break;
      made_pos += made_scan;
      if (0 != answer_relation_fact_extract_made(entry, made_pos)) ++result;
      made_scan = made_pos + 4;
    }
    for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      int alias_len = 0;
      int verb_pos;
      if (RELATION_RULE_ALIAS != rule->kind) continue;
      verb_pos = answer_relation_rule_alias_pos(rule->relation, entry->text,
        entry->text_len, &alias_len);
      if (verb_pos >= 0 && alias_len > 0) {
        int scan_pos = 0;
        char actor[64];
        while (0 != answer_relation_fact_extract_actor(entry->text,
          entry->text_len, verb_pos, &scan_pos, actor, (int)sizeof(actor)))
        {
          char query[192];
          answer_relation_match_t match;
          sprintf(query, "who was %s by %s?", rule->relation, actor);
          if (0 != answer_relation_match_query(query, strlen(query),
            QUERY_WHO, entry, &match)
            && EXIT_SUCCESS == answer_relation_fact_append(entry, &match))
          {
            ++result;
          }
        }
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_relation_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "relation facts: %lu learned\n",
      (unsigned long)answer_relation_facts_size);
  }
}


static void answer_identity_facts_free(void)
{
  free(answer_identity_facts);
  answer_identity_facts = NULL;
  answer_identity_facts_size = 0;
}


static int answer_identity_word_is_name(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word && word_len > 1
    && 0 != isupper((unsigned char)word[0])
    && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON, word, word_len)
    && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP, word, word_len))
  {
    int clean = 1;
    int pos;
    for (pos = 0; pos < word_len && 0 != clean; ++pos) {
      unsigned char c = (unsigned char)word[pos];
      if (0 == isalpha(c) && '-' != c) clean = 0;
    }
    result = clean;
  }
  return result;
}


static int answer_identity_fact_append(const char* name, int name_len,
  const char* role, int role_len, const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_identity_fact_t fact;
  answer_identity_fact_t* facts;
  size_t fact_pos;
  if (NULL == name || name_len <= 0 || name_len >= (int)sizeof(fact.name)
    || NULL == role || role_len <= 0 || role_len >= (int)sizeof(fact.role))
  {
    return EXIT_FAILURE;
  }
  LIBXS_MEMZERO(&fact);
  memcpy(fact.name, name, (size_t)name_len);
  fact.name[name_len] = '\0';
  fact.name_len = name_len;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section[entry->section_len] = '\0';
    fact.section_len = entry->section_len;
  }
  fact.score = score;
  for (fact_pos = 0; fact_pos < answer_identity_facts_size; ++fact_pos) {
    answer_identity_fact_t* old_fact = answer_identity_facts + fact_pos;
    if (old_fact->name_len == fact.name_len
      && 0 != text_contains_word_ci(old_fact->name, old_fact->name_len,
        fact.name))
    {
      if (fact.score > old_fact->score) {
        memcpy(old_fact->role, fact.role, (size_t)fact.role_len + 1);
        old_fact->role_len = fact.role_len;
        memcpy(old_fact->section, fact.section, (size_t)fact.section_len + 1);
        old_fact->section_len = fact.section_len;
        old_fact->score = fact.score;
      }
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_identity_fact_t*)realloc(answer_identity_facts,
      (answer_identity_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_identity_facts = facts;
      answer_identity_facts[answer_identity_facts_size] = fact;
      ++answer_identity_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_identity_is_connective(const char* word, int word_len)
{
  static const char* const connectives[] = {
    "is", "was", "are", "were", "be", "called", "named", "known",
    "a", "an", "the", "that", "who", "as"
  };
  int result = 0;
  size_t idx;
  for (idx = 0; idx < sizeof(connectives) / sizeof(*connectives)
    && 0 == result; ++idx)
  {
    int clen = (int)strlen(connectives[idx]);
    if (clen == word_len
      && 0 != text_contains_word_ci(word, word_len, connectives[idx]))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_identity_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  enum { IDENTITY_GAP_MAX = 3 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_identity_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    char role[64];
    int role_len = 0;
    int have_role = 0;
    int gap = 0;
    int token_index = 0;
    const char* token;
    const char* prev_end = entry->text;
    int token_len = 0;
    while (NULL != (token = libxs_strtoken(entry->text, delims,
      token_index, &token_len)))
    {
      const char* scan;
      for (scan = prev_end; scan < token; ++scan) {
        if ('.' == *scan || '!' == *scan || '?' == *scan
          || ';' == *scan || ':' == *scan) have_role = 0;
      }
      prev_end = token + token_len;
      if (token_len > 0) {
        int is_name = answer_identity_word_is_name(token, token_len);
        int is_role = (token_len < (int)sizeof(role)
          && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
            token, token_len)) ? 1 : 0;
        if (0 != have_role && 0 != is_name) {
          if (EXIT_SUCCESS == answer_identity_fact_append(token, token_len,
            role, role_len, entry, (double)(IDENTITY_GAP_MAX - gap)))
          {
            ++result;
          }
          have_role = 0;
        }
        else if (0 != is_role) {
          memcpy(role, token, (size_t)token_len);
          role[token_len] = '\0';
          role_len = token_len;
          have_role = 1;
          gap = 0;
        }
        else if (0 != have_role
          && 0 != answer_identity_is_connective(token, token_len)
          && gap + 1 < IDENTITY_GAP_MAX)
        {
          ++gap;
        }
        else have_role = 0;
      }
      ++token_index;
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_identity_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "identity facts: %lu learned\n",
      (unsigned long)answer_identity_facts_size);
  }
}


static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char name[64];
  char query_section[ENTRY_SECTION_MAX];
  int name_len;
  int name_upper = 0;
  int query_section_len;
  const answer_identity_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_identity_facts_size) return EXIT_FAILURE;
  name_len = answer_query_be_word(query_text, query_len, name,
    (int)sizeof(name), &name_upper);
  if (name_len <= 0 || 0 == name_upper) return EXIT_FAILURE;
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  for (fact_pos = 0; fact_pos < answer_identity_facts_size; ++fact_pos) {
    const answer_identity_fact_t* fact = answer_identity_facts + fact_pos;
    if (fact->name_len == name_len
      && 0 != text_contains_word_ci(fact->name, fact->name_len, name)
      && (query_section_len <= 0 || fact->section_len <= 0
        || 0 != text_contains_ci(fact->section, fact->section_len,
          query_section))
      && (NULL == best || fact->score > best->score))
    {
      best = fact;
    }
  }
  if (NULL != best) {
    result = answer_reply_role(output, output_size, best->name,
      best->name_len, best->role);
  }
  return result;
}


static void answer_describe_facts_free(void)
{
  free(answer_describe_facts);
  answer_describe_facts = NULL;
  answer_describe_facts_size = 0;
}


static int answer_describe_fact_append(const char* role, int role_len,
  const char* text, int text_len, const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_describe_fact_t fact;
  answer_describe_fact_t* facts;
  size_t fact_pos;
  if (NULL == role || role_len <= 0 || role_len >= (int)sizeof(fact.role)
    || NULL == text || text_len <= 0 || NULL == entry) return EXIT_FAILURE;
  LIBXS_MEMZERO(&fact);
  if (text_len >= (int)sizeof(fact.text)) text_len = (int)sizeof(fact.text) - 1;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  memcpy(fact.text, text, (size_t)text_len);
  fact.text[text_len] = '\0';
  fact.text_len = text_len;
  fact.section_len = entry->section_len;
  if (fact.section_len > 0) {
    memcpy(fact.section, entry->section, (size_t)fact.section_len);
    fact.section[fact.section_len] = '\0';
  }
  fact.score = score;
  for (fact_pos = 0; fact_pos < answer_describe_facts_size; ++fact_pos) {
    answer_describe_fact_t* old_fact = answer_describe_facts + fact_pos;
    if (old_fact->role_len == fact.role_len
      && 0 != text_contains_word_ci(old_fact->role, old_fact->role_len,
        fact.role)
      && old_fact->section_len == fact.section_len
      && (0 == fact.section_len
        || 0 != text_contains_ci(old_fact->section, old_fact->section_len,
          fact.section)))
    {
      if (fact.score > old_fact->score) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_describe_fact_t*)realloc(answer_describe_facts,
      (answer_describe_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_describe_facts = facts;
      answer_describe_facts[answer_describe_facts_size] = fact;
      ++answer_describe_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_describe_word_is_article(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word) {
    if (1 == word_len && ('a' == (word[0] | 32))) result = 1;
    else if (2 == word_len && ('a' == (word[0] | 32))
      && ('n' == (word[1] | 32)))
    {
      result = 1;
    }
  }
  return result;
}


static size_t answer_describe_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n.,;:!?()[]{}\"";
  enum { DESCRIBE_GAP_MAX = 3 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_describe_facts_free();
  if (NULL == corpus || 0 == answer_relation_rules_size) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale) {
      const char* article = NULL;
      int gap = 0;
      int token_index = 0;
      const char* token;
      const char* prev_end = entry->text;
      int token_len = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* scan;
        for (scan = prev_end; scan < token; ++scan) {
          if (0 == isspace((unsigned char)*scan)) article = NULL;
        }
        prev_end = token + token_len;
        if (token_len > 0) {
          int is_role = (token_len < 64
            && 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
              token, token_len)) ? 1 : 0;
          if (0 != is_role && NULL != article) {
            const char* text_end = entry->text + entry->text_len;
            const char* clause = token + token_len;
            const char* end = clause;
            double score = 1.0;
            while (clause < text_end
              && 0 != isspace((unsigned char)*clause)) ++clause;
            if (clause < text_end && ',' == *clause) {
              int rel_len = 0;
              const char* rel = clause + 1;
              while (rel < text_end
                && 0 != isspace((unsigned char)*rel)) ++rel;
              while (rel + rel_len < text_end
                && 0 != isalnum((unsigned char)rel[rel_len])) ++rel_len;
              if ((3 == rel_len && 0 == strncmp(rel, "who", 3))
                || (5 == rel_len && 0 == strncmp(rel, "which", 5)))
              {
                end = rel + rel_len;
                while (end < text_end && ',' != *end && '.' != *end
                  && ';' != *end && '!' != *end && '?' != *end) ++end;
                score = 2.0;
              }
            }
            if (EXIT_SUCCESS == answer_describe_fact_append(token, token_len,
              article, (int)(end - article), entry, score))
            {
              ++result;
            }
            article = NULL;
          }
          else if (0 != answer_describe_word_is_article(token, token_len)) {
            article = token;
            gap = 0;
          }
          else if (NULL != article && gap + 1 < DESCRIBE_GAP_MAX) ++gap;
          else article = NULL;
        }
        ++token_index;
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_describe_facts_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "describe facts: %lu learned\n",
      (unsigned long)answer_describe_facts_size);
  }
}


static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char role[64];
  char query_section[ENTRY_SECTION_MAX];
  int role_len;
  int query_section_len;
  const answer_describe_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_describe_facts_size) return EXIT_FAILURE;
  role_len = answer_query_be_word(query_text, query_len, role,
    (int)sizeof(role), NULL);
  if (role_len <= 0 || 0 == answer_relation_rule_has_term(
    RELATION_RULE_PERSON, role, role_len)) return EXIT_FAILURE;
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  for (fact_pos = 0; fact_pos < answer_describe_facts_size; ++fact_pos) {
    const answer_describe_fact_t* fact = answer_describe_facts + fact_pos;
    if (fact->role_len == role_len
      && 0 != text_contains_word_ci(fact->role, fact->role_len, role)
      && (query_section_len <= 0 || fact->section_len <= 0
        || 0 != text_contains_ci(fact->section, fact->section_len,
          query_section))
      && (NULL == best || fact->score > best->score))
    {
      best = fact;
    }
  }
  if (NULL != best && (size_t)best->text_len + 2 <= output_size) {
    memcpy(output, best->text, (size_t)best->text_len);
    output[0] = (char)toupper((unsigned char)output[0]);
    output[best->text_len] = '.';
    output[best->text_len + 1] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void answer_docdef_facts_free(void)
{
  free(answer_docdef_facts);
  answer_docdef_facts = NULL;
  answer_docdef_facts_size = 0;
}


/* Parse an optional leading "Header: `name`" line and return the byte offset
   of the definition prose that follows it (0 when absent). Fills header with
   the base module name (backticks and trailing extension stripped). */
static int answer_docdef_header(const char* text, int text_len,
  char* header, int header_size, int* header_len)
{
  int result = 0;
  *header_len = 0;
  if (text_len > 7 && 0 == strncmp(text, "Header:", 7)) {
    int i = 7;
    int begin, end;
    while (i < text_len && (0 != isspace((unsigned char)text[i])
      || '`' == text[i])) ++i;
    begin = i;
    while (i < text_len && '`' != text[i]
      && 0 == isspace((unsigned char)text[i])) ++i;
    end = i;
    while (end > begin && '.' != text[end - 1]) {
      if ('.' == text[end - 1]) break;
      --end;
    }
    if (end <= begin) end = i;
    else --end;
    if (end - begin > 0 && end - begin < header_size) {
      memcpy(header, text + begin, (size_t)(end - begin));
      header[end - begin] = '\0';
      *header_len = end - begin;
    }
    while (i < text_len && '\n' != text[i]) ++i;
    while (i < text_len && 0 != isspace((unsigned char)text[i])) ++i;
    result = i;
    for (;;) {
      int j = result;
      while (j < text_len && (0 != isalpha((unsigned char)text[j])
        || '-' == text[j] || '_' == text[j])) ++j;
      if (j > result && j < text_len && ':' == text[j]) {
        while (j < text_len && '\n' != text[j]) ++j;
        while (j < text_len && 0 != isspace((unsigned char)text[j])) ++j;
        result = j;
      }
      else break;
    }
  }
  return result;
}


/* Learn module definitions from Markdown ingest: the first paragraph under a
   heading becomes that title's definition, with the "Header:" filename kept as
   an alias. Structural only (no corpus vocabulary), so prose corpora without
   headings contribute nothing. */
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_docdef_facts_free();
  if (NULL == corpus) return 0;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_PARAGRAPH == entry->scale && entry->section_len > 0) {
      size_t fact_pos;
      int seen = 0;
      for (fact_pos = 0; fact_pos < answer_docdef_facts_size; ++fact_pos) {
        const answer_docdef_fact_t* old = answer_docdef_facts + fact_pos;
        if (old->title_len == entry->section_len
          && 0 == libxs_memcmp(old->title, entry->section,
            (size_t)entry->section_len))
        {
          seen = 1;
          break;
        }
      }
      if (0 == seen) {
        char header[64];
        int header_len = 0;
        int offset = answer_docdef_header(entry->text, entry->text_len,
          header, (int)sizeof(header), &header_len);
        int text_len = entry->text_len - offset;
        if (text_len > 0 && header_len > 0) {
          answer_docdef_fact_t* facts = (answer_docdef_fact_t*)realloc(
            answer_docdef_facts,
            (answer_docdef_facts_size + 1) * sizeof(*facts));
          if (NULL != facts) {
            answer_docdef_fact_t* fact = facts + answer_docdef_facts_size;
            answer_docdef_facts = facts;
            LIBXS_MEMZERO(fact);
            if (text_len >= (int)sizeof(fact->text)) {
              text_len = (int)sizeof(fact->text) - 1;
            }
            memcpy(fact->title, entry->section, (size_t)entry->section_len);
            fact->title[entry->section_len] = '\0';
            fact->title_len = entry->section_len;
            fact->header_len = header_len;
            if (header_len > 0) memcpy(fact->header, header,
              (size_t)header_len + 1);
            memcpy(fact->text, entry->text + offset, (size_t)text_len);
            fact->text[text_len] = '\0';
            fact->text_len = text_len;
            ++answer_docdef_facts_size;
            ++result;
          }
        }
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_docdef_facts_report(FILE* stream)
{
  if (NULL != stream && answer_docdef_facts_size > 0) {
    fprintf(stream, "docdef facts: %lu learned\n",
      (unsigned long)answer_docdef_facts_size);
  }
}


/* Extract the subject term from "what is [the] X" or "what does X do",
   skipping a leading article. Language-generic, no corpus vocabulary. */
static int answer_docdef_term(const char* query_text, size_t query_len,
  char* term, int term_size)
{
  static const char* const markers[] = { " is ", " does ", " are " };
  int result = 0;
  int marker_pos = -1;
  int marker_len = 0;
  int m;
  size_t begin, end;
  if (NULL == query_text || NULL == term || term_size <= 0) return 0;
  term[0] = '\0';
  for (m = 0; m < 3 && marker_pos < 0; ++m) {
    marker_pos = text_find_ci(query_text, (int)query_len, markers[m]);
    if (marker_pos >= 0) marker_len = (int)strlen(markers[m]);
  }
  if (marker_pos < 0) return 0;
  begin = (size_t)marker_pos + (size_t)marker_len;
  while (begin < query_len && 0 != isspace((unsigned char)query_text[begin])) {
    ++begin;
  }
  end = begin;
  while (end < query_len && '?' != query_text[end] && '.' != query_text[end]
    && '!' != query_text[end]) ++end;
  while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) --end;
  if (end - begin >= 3 && 0 != isspace((unsigned char)query_text[end - 3])
    && ('d' == (query_text[end - 2] | 32)) && ('o' == (query_text[end - 1] | 32)))
  {
    end -= 3;
  }
  while (end > begin && 0 != isspace((unsigned char)query_text[end - 1])) --end;
  if (end - begin >= 2 && 0 == strncmp(query_text + begin, "a ", 2)) begin += 2;
  else if (end - begin >= 3
    && 0 == strncmp(query_text + begin, "an ", 3)) begin += 3;
  else if (end - begin >= 4
    && 0 == strncmp(query_text + begin, "the ", 4)) begin += 4;
  while (begin < end && 0 != isspace((unsigned char)query_text[begin])) {
    ++begin;
  }
  result = (int)(end - begin);
  if (result >= term_size) result = term_size - 1;
  if (result > 0) {
    memcpy(term, query_text + begin, (size_t)result);
    term[result] = '\0';
  }
  return result;
}


/* Answer "What is X?" / "What does X do?" from a learned module definition,
   matching X against either the heading text or the Header: filename alias. */
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char term[ENTRY_SECTION_MAX];
  int term_len;
  const answer_docdef_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_docdef_facts_size) return EXIT_FAILURE;
  term_len = answer_docdef_term(query_text, query_len, term,
    (int)sizeof(term));
  if (term_len <= 0) return EXIT_FAILURE;
  for (fact_pos = 0; fact_pos < answer_docdef_facts_size && NULL == best;
    ++fact_pos)
  {
    const answer_docdef_fact_t* fact = answer_docdef_facts + fact_pos;
    if ((fact->title_len == term_len
        && 0 != text_contains_word_ci(fact->title, fact->title_len, term))
      || (fact->header_len == term_len
        && 0 != text_contains_word_ci(fact->header, fact->header_len, term)))
    {
      best = fact;
    }
  }
  if (NULL != best && (size_t)best->text_len + 1 <= output_size) {
    memcpy(output, best->text, (size_t)best->text_len);
    output[best->text_len] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void conv_reset(void)
{
  conv_topic[0] = '\0';
  conv_topic_len = 0;
}


/* Remember the subject of a successfully answered question so that a later
   follow-up can refer back to it. Uses the same subject extraction as the
   definition path; only a real subject term updates the topic. */
static void conv_remember(const char* query_text, size_t query_len)
{
  char term[CONV_TOPIC_MAX];
  int term_len = answer_docdef_term(query_text, query_len, term,
    (int)sizeof(term));
  if (term_len > 0) {
    memcpy(conv_topic, term, (size_t)term_len);
    conv_topic[term_len] = '\0';
    conv_topic_len = term_len;
  }
}


static int conv_word_is_pronoun(const char* word, int len)
{
  static const char* const pronouns[] = { "it", "its", "it's", "that",
    "this", "they", "them", "their" };
  int result = 0;
  int p;
  for (p = 0; p < (int)(sizeof(pronouns) / sizeof(*pronouns)) && 0 == result;
    ++p)
  {
    if ((int)strlen(pronouns[p]) == len) {
      int i, same = 1;
      for (i = 0; i < len && 0 != same; ++i) {
        if ((word[i] | 32) != pronouns[p][i]) same = 0;
      }
      result = same;
    }
  }
  return result;
}


/* Rewrite a follow-up against the remembered topic. Two grounded moves:
   substitute a back-reference pronoun (it/its/that/...) with the topic, and,
   when a question carries no subject of its own, append the topic so the
   answer path is scoped to it. Returns EXIT_SUCCESS only when a rewrite was
   made, leaving the original query untouched otherwise. */
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size)
{
  int result = EXIT_FAILURE;
  char own[CONV_TOPIC_MAX];
  int own_len;
  size_t pos = 0, w = 0;
  int replaced = 0;
  if (NULL == query_text || NULL == out || 0 == out_size
    || 0 == conv_topic_len) return EXIT_FAILURE;
  while (w < query_len && w + 1 < out_size) {
    size_t begin = w;
    while (w < query_len && 0 != isalpha((unsigned char)query_text[w])) ++w;
    if (w > begin && 0 != conv_word_is_pronoun(query_text + begin,
      (int)(w - begin)))
    {
      if (pos + (size_t)conv_topic_len < out_size) {
        memcpy(out + pos, conv_topic, (size_t)conv_topic_len);
        pos += (size_t)conv_topic_len;
        replaced = 1;
      }
    }
    else if (w > begin) {
      if (pos + (w - begin) < out_size) {
        memcpy(out + pos, query_text + begin, w - begin);
        pos += w - begin;
      }
    }
    if (w < query_len && 0 == isalpha((unsigned char)query_text[w])
      && pos + 1 < out_size)
    {
      out[pos++] = query_text[w++];
    }
  }
  out[pos] = '\0';
  if (0 != replaced) {
    result = EXIT_SUCCESS;
  }
  else {
    own_len = answer_docdef_term(query_text, query_len, own,
      (int)sizeof(own));
    if (own_len <= 0 && pos > 0) {
      static const char suffix[] = " of the ";
      size_t suffix_len = sizeof(suffix) - 1;
      if (pos + suffix_len + (size_t)conv_topic_len < out_size) {
        memcpy(out + pos, suffix, suffix_len);
        pos += suffix_len;
        memcpy(out + pos, conv_topic, (size_t)conv_topic_len);
        pos += (size_t)conv_topic_len;
        out[pos] = '\0';
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact)
{
  int result = 0;
  size_t rule_pos;
  if (NULL == query_relation || NULL == fact) return 0;
  if (0 != text_contains_ci(query_relation, (int)strlen(query_relation),
    fact->relation))
  {
    result = 1;
  }
  for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (RELATION_RULE_ALIAS == rule->kind
      && 0 != text_contains_ci(rule->relation, (int)strlen(rule->relation),
        fact->relation)
      && 0 != text_contains_ci(query_relation, (int)strlen(query_relation),
        rule->term))
    {
      result = 1;
    }
  }
  return result;
}


static int answer_relation_fact_actor_match(const char* query_actor,
  int query_actor_len, const answer_relation_fact_t* fact)
{
  int result = 0;
  if (NULL == fact) return 0;
  if (query_actor_len <= 0) result = (0 == fact->actor_len) ? 1 : 0;
  else if (fact->actor_len > 0) {
    if (0 != answer_relation_actor_has_token(query_actor, query_actor_len,
        fact->actor, fact->actor_len)
      || 0 != answer_relation_actor_has_token(fact->actor, fact->actor_len,
        query_actor, query_actor_len))
    {
      result = 1;
    }
  }
  return result;
}


static int answer_relation_fact_section_match(const char* query_section,
  int query_section_len, const answer_relation_fact_t* fact)
{
  int result = 1;
  if (query_section_len > 0) {
    result = 0;
    if (NULL != fact && fact->section_len > 0
      && query_section_len < ENTRY_SECTION_MAX)
    {
      char section[ENTRY_SECTION_MAX];
      memcpy(section, query_section, (size_t)query_section_len);
      section[query_section_len] = '\0';
      if (0 != text_contains_ci(fact->section, fact->section_len, section)
        || 0 == libxs_stridiff(fact->section, section, NULL, 1, NULL))
      {
        result = 1;
      }
    }
  }
  return result;
}


static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  enum { RELATION_FACT_MAX = 4 };
  int result = EXIT_FAILURE;
  char relation[64];
  char actor[64];
  char query_section[ENTRY_SECTION_MAX];
  char answers[RELATION_FACT_MAX][64];
  int answer_lens[RELATION_FACT_MAX];
  int answer_made[RELATION_FACT_MAX];
  double answer_scores[RELATION_FACT_MAX];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  int query_section_len;
  int count = 0;
  int slot;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_relation_facts_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (relation_len <= 0 || 0 != relation_upper) {
    return EXIT_FAILURE;
  }
  for (slot = 0; slot < RELATION_FACT_MAX; ++slot) {
    answers[slot][0] = '\0';
    answer_lens[slot] = 0;
    answer_made[slot] = 0;
    answer_scores[slot] = 0.0;
  }
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
    if (0 != answer_relation_fact_relation_match(relation, fact)
      && 0 != answer_relation_fact_actor_match(actor, actor_len, fact)
      && 0 != answer_relation_fact_section_match(query_section,
        query_section_len, fact))
    {
      int duplicate = 0;
      for (slot = 0; slot < count; ++slot) {
        if (0 != answer_relation_same_answer(answers[slot],
          answer_lens[slot], fact->answer, fact->answer_len))
        {
          duplicate = 1;
          if (fact->score > answer_scores[slot]) {
            memcpy(answers[slot], fact->answer,
              (size_t)fact->answer_len + 1);
            answer_lens[slot] = fact->answer_len;
            answer_made[slot] = fact->made;
            answer_scores[slot] = fact->score;
          }
        }
      }
      if (0 == duplicate && count < RELATION_FACT_MAX) {
        int insert = count;
        while (insert > 0 && fact->score > answer_scores[insert - 1]) {
          memcpy(answers[insert], answers[insert - 1],
            (size_t)answer_lens[insert - 1] + 1);
          answer_lens[insert] = answer_lens[insert - 1];
          answer_made[insert] = answer_made[insert - 1];
          answer_scores[insert] = answer_scores[insert - 1];
          --insert;
        }
        memcpy(answers[insert], fact->answer,
          (size_t)fact->answer_len + 1);
        answer_lens[insert] = fact->answer_len;
        answer_made[insert] = fact->made;
        answer_scores[insert] = fact->score;
        ++count;
      }
    }
  }
  if (count > 0) {
    size_t pos = 0;
    int item;
    output[0] = '\0';
    for (item = 0; item < count && pos + 1 < output_size; ++item) {
      if (item > 0) {
        const char* joiner = (item + 1 == count) ? " and " : ", ";
        size_t joiner_len = strlen(joiner);
        if (pos + joiner_len + 1 >= output_size) break;
        memcpy(output + pos, joiner, joiner_len);
        pos += joiner_len;
      }
      if (pos + (size_t)answer_lens[item] + 1 >= output_size) break;
      memcpy(output + pos, answers[item], (size_t)answer_lens[item]);
      pos += (size_t)answer_lens[item];
    }
    if (pos + (size_t)relation_len + (size_t)actor_len + 24 < output_size) {
      const char* copula = (count > 1) ? " were " : " was ";
      if (actor_len <= 0) copula = (count > 1) ? " are " : " is ";
      if (actor_len <= 0 && 0 != answer_made[0]) {
        static const char made_one[] = " is to be made ";
        static const char made_many[] = " are to be made ";
        const char* made_text = (count > 1) ? made_many : made_one;
        memcpy(output + pos, made_text, strlen(made_text));
        pos += strlen(made_text);
      }
      else {
        memcpy(output + pos, copula, strlen(copula));
        pos += strlen(copula);
      }
      memcpy(output + pos, relation, (size_t)relation_len);
      pos += (size_t)relation_len;
      if (actor_len > 0) {
        static const char by_text[] = " by ";
        memcpy(output + pos, by_text, sizeof(by_text) - 1);
        pos += sizeof(by_text) - 1;
        memcpy(output + pos, actor, (size_t)actor_len);
        pos += (size_t)actor_len;
      }
      output[pos++] = '.';
      output[pos] = '\0';
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size)
{
  enum { RELATION_AGG_MAX = 4 };
  int result = EXIT_FAILURE;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len;
  char answers[RELATION_AGG_MAX][64];
  int answer_lens[RELATION_AGG_MAX];
  double answer_scores[RELATION_AGG_MAX];
  int count = 0;
  char relation[64];
  char actor[64];
  int relation_upper = 0;
  int relation_len;
  int actor_len;
  if (NULL == corpus || NULL == query_text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation), &relation_upper);
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (query_section_len > 0 && relation_len > 0 && 0 == relation_upper
    && actor_len > 0)
  {
    int slot;
    for (slot = 0; slot < RELATION_AGG_MAX; ++slot) {
      answers[slot][0] = '\0';
      answer_lens[slot] = 0;
      answer_scores[slot] = 0.0;
    }
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, 10, NULL) : sizeof(*entry);
      answer_relation_match_t match;
      if ((query_section_len <= 0
          || 0 != corpus_entry_section_match(entry, entry_size,
            query_section, query_section_len))
        && 0 != answer_relation_match_query(query_text, query_len,
          QUERY_WHO, entry, &match)
        && 0 == match.made && match.actor_len > 0 && match.answer_len > 0)
      {
        char candidate[64];
        int candidate_len = answer_relation_section_title(candidate,
          (int)sizeof(candidate), entry, &match);
        double candidate_score = match.score;
        int duplicate = 0;
        if (candidate_len > 0) candidate_score += 0.10;
        else {
          candidate_len = match.answer_len;
          memcpy(candidate, match.answer, (size_t)candidate_len);
          candidate[candidate_len] = '\0';
        }
        for (slot = 0; slot < count; ++slot) {
          if (0 != answer_relation_same_answer(answers[slot],
            answer_lens[slot], candidate, candidate_len))
          {
            duplicate = 1;
            if (candidate_score > answer_scores[slot]) {
              memcpy(answers[slot], candidate, (size_t)candidate_len + 1);
              answer_lens[slot] = candidate_len;
              answer_scores[slot] = candidate_score;
            }
          }
        }
        if (0 == duplicate && count < RELATION_AGG_MAX) {
          int insert = count;
          while (insert > 0 && candidate_score > answer_scores[insert - 1]) {
            memcpy(answers[insert], answers[insert - 1],
              (size_t)answer_lens[insert - 1] + 1);
            answer_lens[insert] = answer_lens[insert - 1];
            answer_scores[insert] = answer_scores[insert - 1];
            --insert;
          }
          memcpy(answers[insert], candidate, (size_t)candidate_len + 1);
          answer_lens[insert] = candidate_len;
          answer_scores[insert] = candidate_score;
          ++count;
        }
      }
      value = libxs_registry_next(corpus, &key, &cursor);
    }
    if (count > 1) {
      size_t pos = 0;
      int item;
      output[0] = '\0';
      for (item = 0; item < count && pos + 1 < output_size; ++item) {
        if (item > 0) {
          const char* joiner = (item + 1 == count) ? " and " : ", ";
          size_t joiner_len = strlen(joiner);
          if (pos + joiner_len + 1 >= output_size) break;
          memcpy(output + pos, joiner, joiner_len);
          pos += joiner_len;
        }
        if (pos + (size_t)answer_lens[item] + 1 >= output_size) break;
        memcpy(output + pos, answers[item], (size_t)answer_lens[item]);
        pos += (size_t)answer_lens[item];
      }
      if (pos + (size_t)relation_len + (size_t)actor_len + 11 < output_size) {
        static const char were_text[] = " were ";
        static const char by_text[] = " by ";
        memcpy(output + pos, were_text, sizeof(were_text) - 1);
        pos += sizeof(were_text) - 1;
        memcpy(output + pos, relation, (size_t)relation_len);
        pos += (size_t)relation_len;
        memcpy(output + pos, by_text, sizeof(by_text) - 1);
        pos += sizeof(by_text) - 1;
        memcpy(output + pos, actor, (size_t)actor_len);
        pos += (size_t)actor_len;
        output[pos++] = '.';
        output[pos] = '\0';
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry)
{
  double result = 0.0;
  char word[64];
  int upper_initial = 0;
  int word_len;
  answer_relation_match_t relation_match;
  if (QUERY_WHO != query_type || NULL == entry) return 0.0;
  word_len = answer_query_be_word(query_text, query_len, word,
    (int)sizeof(word), &upper_initial);
  if (word_len <= 0) return 0.0;
  if (0 == upper_initial && 0 != answer_relation_match_query(query_text,
    query_len, query_type, entry, &relation_match))
  {
    result = relation_match.score;
  }
  if (0 == text_contains_word_ci(entry->text, entry->text_len, word)) {
    return result;
  }
  if (0 != upper_initial) {
    result = 0.55;
    if (0 != text_contains_ci(entry->text, entry->text_len, "called")) {
      result += 0.55;
    }
    if (0 != answer_relation_rule_has_term(RELATION_RULE_PERSON,
      entry->text, entry->text_len))
    {
      result += 0.35;
    }
  }
  else {
    result = 0.35;
    if (0 != text_contains_ci(entry->text, entry->text_len, " is ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " was ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " are ")
      || 0 != text_contains_ci(entry->text, entry->text_len, " were "))
    {
      result += 0.25;
    }
    if (0 != text_contains_ci(entry->text, entry->text_len, "made")) {
      result += 0.35;
    }
    if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) result += 0.15;
  }
  return result;
}


static int answer_features_fill(const corpus_entry_t* entry,
  size_t entry_size, double overlap, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS])
{
  int result = EXIT_FAILURE;
  int use_sketch = 0;
  int entry_words = 0;
  int input_pos;
  for (input_pos = 0; input_pos < ANSWER_PREDICT_INPUTS; ++input_pos) {
    inputs[input_pos] = 0.0;
  }
  if (NULL == entry || NULL == inputs) return EXIT_FAILURE;
  if (entry_size >= sizeof(*entry) && entry->ntokens > 0) use_sketch = 1;
  if (0 == use_sketch) return EXIT_FAILURE;
  if (overlap < 0.0) overlap = 0.0;
  if (overlap > 1.0) overlap = 1.0;
  entry_words = count_words((const unsigned char*)entry->text, entry->text_len);
  inputs[0] = overlap;
  inputs[1] = (double)query_type / (double)QUERY_YESNO;
  inputs[2] = (SCALE_SENTENCE == entry->scale) ? 1.0 : 0.0;
  inputs[3] = (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) ? 1.0 : 0.0;
  inputs[4] = (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) ? 1.0 : 0.0;
  inputs[5] = (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) ? 1.0 : 0.0;
  inputs[6] = (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) ? 1.0 : 0.0;
  inputs[7] = (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) ? 1.0 : 0.0;
  inputs[8] = (entry_words > 0) ? (double)entry->ncontent / entry_words : 0.0;
  inputs[9] = (entry->text_len > COMPOSE_MAXTEXT)
    ? 1.0 : (double)entry->text_len / (double)COMPOSE_MAXTEXT;
  result = EXIT_SUCCESS;
  return result;
}


static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS])
{
  double total = 0.0, matched = 0.0;
  size_t query_pos;
  if (NULL == query || NULL == entry || NULL == inputs) return EXIT_FAILURE;
  for (query_pos = 0; query_pos < query->size; ++query_pos) {
    const libxs_lexeme_t* lexeme = query->data + query_pos;
    if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
    {
      double weight = (lexeme->length >= 6) ? 1.5 : 1.0;
      total += weight;
      if (0 != entry_sketch_has_id(entry, lexeme->id)) matched += weight;
    }
  }
  return answer_features_fill(entry, entry_size,
    (total > 0.0) ? matched / total : 0.0, query_type, inputs);
}


static double answer_weak_label(const corpus_entry_t* entry, int query_type)
{
  double result = 0.0;
  if (NULL != entry) {
    switch (query_type) {
      case QUERY_WHO:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) ? 1.0 : 0.0;
        break;
      case QUERY_WHAT:
        result = (SCALE_SENTENCE == entry->scale) ? 0.60 : 0.35;
        break;
      case QUERY_WHERE:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) ? 1.0 : 0.0;
        break;
      case QUERY_WHEN:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) ? 0.95 : 0.0;
        break;
      case QUERY_WHY:
        result = (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) ? 1.0 : 0.0;
        break;
      case QUERY_HOW:
        if (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) result = 0.95;
        else if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) result = 0.35;
        break;
      default:
        break;
    }
  }
  return result;
}


static libxs_predict_t* answer_predict_create(
  const answer_predict_profile_t* profile)
{
  static const double weights[ANSWER_PREDICT_INPUTS] = {
    3.0, 1.0, 0.5, 1.5, 1.2, 1.4, 1.4, 1.2, 0.4, 0.3
  };
  libxs_predict_t* result = libxs_predict_create(ANSWER_PREDICT_INPUTS, 1);
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != result) {
    libxs_predict_set_mode(result, profile->mode);
    libxs_predict_set_decompose(result, profile->decompose);
    libxs_predict_set_weights(result, weights);
    if (0.0 != profile->smooth) libxs_predict_set_smooth(result,
      profile->smooth);
    if (0 < profile->nseries && 0 < profile->window) {
      libxs_predict_set_series(result, profile->nseries, profile->window);
      libxs_predict_set_target(result, profile->target);
    }
    if (0 != profile->diff_order) libxs_predict_set_diff(result,
      profile->diff_order);
  }
  return result;
}


static int answer_predict_build_model(libxs_predict_t* model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_FAILURE;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != model) {
    result = libxs_predict_build(model, profile->clusters,
      profile->order, profile->quality);
  }
  return result;
}


static void answer_predict_report(const char* label,
  const libxs_predict_t* model, int ntrain,
  const answer_predict_profile_t* profile)
{
  libxs_predict_query_t info;
  LIBXS_MEMZERO(&info);
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL != label && NULL != model) {
    libxs_predict_query(model, &info);
    fprintf(stderr,
      "predict[%s:%s]: inputs=%d pushed=%d entries=%d clusters=%d order=%d diff=%d compression=%.2fx\n",
      label, profile->name, ANSWER_PREDICT_INPUTS, ntrain,
      info.nentries, info.nclusters, info.order, info.diff_order,
      info.compression);
  }
}


static libxs_predict_t* converse_predict_train(const libxs_registry_t* corpus,
  const answer_predict_profile_t* profile)
{
  libxs_predict_t* result = NULL;
  libxs_predict_t* model = NULL;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, 10, NULL) : sizeof(*entry);
    if (entry_size >= sizeof(*entry) && entry->ntokens > 0
      && entry->text_len >= 16)
    {
      int query_type;
      for (query_type = QUERY_WHO; query_type <= QUERY_HOW; ++query_type) {
        double label = answer_weak_label(entry, query_type);
        double overlap = (label >= 0.5) ? 1.0 : 0.0;
        double inputs[ANSWER_PREDICT_INPUTS];
        if (QUERY_WHAT == query_type && label > 0.0) overlap = 0.65;
        if (EXIT_SUCCESS == answer_features_fill(entry, entry_size,
          overlap, query_type, inputs)
          && EXIT_SUCCESS == libxs_predict_push(NULL, model, inputs, &label))
        {
          ++ntrain;
        }
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (ntrain >= 8 && EXIT_SUCCESS == answer_predict_build_model(model,
    profile))
  {
    answer_predict_report("persistent", model, ntrain, profile);
    result = model;
  }
  else {
    libxs_predict_destroy(model);
  }
  return result;
}


static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile)
{
  libxs_predict_t* result = NULL;
  libxs_predict_t* model = NULL;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus || NULL == query || NULL == lexicon) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, 10, NULL) : sizeof(*entry);
    double inputs[ANSWER_PREDICT_INPUTS];
    double output;
    if (entry->text_len >= 16
      && EXIT_SUCCESS == answer_features(query, entry, entry_size,
        query_type, inputs))
    {
      output = lexical_score(query, lexicon, rules, nrules, entry,
        entry_size, query_type);
      if (EXIT_SUCCESS == libxs_predict_push(NULL, model, inputs, &output)) {
        ++ntrain;
      }
    }
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (ntrain >= 4 && EXIT_SUCCESS == answer_predict_build_model(model,
    profile))
  {
    result = model;
  }
  else {
    libxs_predict_destroy(model);
  }
  return result;
}


static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score)
{
  double result = base_score;
  if (NULL != model && NULL != inputs) {
    double output = 0.0;
    libxs_predict_info_t info;
    libxs_predict_eval(NULL, model, inputs, &output, &info, 1);
    if (NULL != info.confidence && info.confidence[0] > 0.0) {
      double confidence = info.confidence[0];
      if (confidence > 1.0) confidence = 1.0;
      if (output < 0.0) output = 0.0;
      result = 0.75 * base_score + 0.25 * confidence * output;
    }
  }
  return result;
}


static int answer_bridge_query_group_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const char* group, int group_len)
{
  int result = 0;
  int start = 0;
  while (start < group_len && 0 == result) {
    int end = start;
    char term[64];
    int term_len;
    while (end < group_len && '/' != group[end]) ++end;
    term_len = end - start;
    if (term_len > 0 && term_len < (int)sizeof(term)) {
      memcpy(term, group + start, (size_t)term_len);
      term[term_len] = '\0';
      result = lexeme_stream_has_text(query, lexicon, term);
      if (0 == result && term_len >= 5) {
        result = lexeme_stream_has_similar_text(query, lexicon, term,
          term_len, (term_len >= 7) ? 2 : 1);
      }
    }
    start = end + 1;
  }
  return result;
}


static int answer_bridge_query_match(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* spec)
{
  int result = 1;
  const char* group = spec;
  if (NULL == query || NULL == lexicon || NULL == spec) return 0;
  while ('\0' != *group && 0 != result) {
    const char* end;
    while ('\0' != *group && 0 != isspace((unsigned char)*group)) ++group;
    end = group;
    while ('\0' != *end && 0 == isspace((unsigned char)*end)) ++end;
    if (end > group) {
      result = answer_bridge_query_group_match(query, lexicon, group,
        (int)(end - group));
    }
    group = end;
  }
  return result;
}


static int answer_bridge_evidence_group_match(const char* text, int text_len,
  const char* group, int group_len)
{
  int result = 0;
  int start = 0;
  while (start < group_len && 0 == result) {
    int end = start;
    char term[64];
    int term_pos, term_len;
    while (end < group_len && '/' != group[end]) ++end;
    term_len = end - start;
    if (term_len > 0 && term_len < (int)sizeof(term)) {
      char text_buf[COMPOSE_MAXTEXT];
      memcpy(term, group + start, (size_t)term_len);
      term[term_len] = '\0';
      for (term_pos = 0; term_pos < term_len; ++term_pos) {
        if ('_' == term[term_pos]) term[term_pos] = ' ';
      }
      if (text_len > 0 && text_len < (int)sizeof(text_buf)) {
        int count = 0;
        int matches;
        memcpy(text_buf, text, (size_t)text_len);
        text_buf[text_len] = '\0';
        matches = libxs_strimatch(text_buf, term, NULL, &count);
        if (count > 0 && matches >= count) result = 1;
      }
      if (0 == result) result = text_contains_ci(text, text_len, term);
    }
    start = end + 1;
  }
  return result;
}


static int answer_bridge_evidence_match(const corpus_entry_t* entry,
  const char* spec)
{
  int result = 1;
  const char* group = spec;
  if (NULL == entry || NULL == spec) return 0;
  while ('\0' != *group && 0 != result) {
    const char* end;
    while ('\0' != *group && 0 != isspace((unsigned char)*group)) ++group;
    end = group;
    while ('\0' != *end && 0 == isspace((unsigned char)*end)) ++end;
    if (end > group) {
      result = answer_bridge_evidence_group_match(entry->text,
        entry->text_len, group, (int)(end - group));
    }
    group = end;
  }
  return result;
}


static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry)
{
  const answer_bridge_t* result = NULL;
  size_t bridge_pos;
  if (NULL != query && NULL != lexicon && NULL != entry) {
    for (bridge_pos = 0; bridge_pos < answer_bridge_loaded_size
      && NULL == result; ++bridge_pos)
    {
      const answer_bridge_t* bridge = answer_bridge_loaded + bridge_pos;
      if (0 != answer_bridge_query_match(query, lexicon, bridge->query)
        && 0 != answer_bridge_evidence_match(entry, bridge->evidence))
      {
        result = bridge;
      }
    }
  }
  return result;
}


static double answer_semantic_bridge_score(const answer_bridge_t* bridge)
{
  return (NULL != bridge) ? bridge->score : 0.0;
}


static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type)
{
  double score = 0.0, total = 0.0;
  double min_overlap = 0.75;
  int matches = 0, use_sketch = 0;
  int entry_stream_ready = 0;
  int long_terms = 0, long_matches = 0;
  int entry_words;
  double bridge_score;
  const answer_bridge_t* bridge;
  double overlap;
  libxs_lexeme_stream_t entry_stream;
  size_t query_pos;
  libxs_lexeme_stream_init(&entry_stream);
  if (NULL == query || NULL == lexicon || NULL == entry) return 0.0;
  if (entry_size >= sizeof(*entry) && entry->ntokens > 0) {
    use_sketch = 1;
  }
  else if (EXIT_SUCCESS != libxs_lexeme_stream_encode(lexicon, &entry_stream,
      (const unsigned char*)entry->text, (size_t)entry->text_len,
      rules, nrules, NULL, 0, 1))
  {
    libxs_lexeme_stream_release(&entry_stream);
    return 0.0;
  }
  else entry_stream_ready = 1;
  for (query_pos = 0; query_pos < query->size; ++query_pos) {
    const libxs_lexeme_t* lexeme = query->data + query_pos;
    if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
    {
      double weight = (lexeme->length >= 6) ? 1.5 : 1.0;
      int long_term = (lexeme->length >= 6) ? 1 : 0;
      int matched;
      total += weight;
      if (0 != long_term) ++long_terms;
      matched = ((0 != use_sketch && 0 != entry_sketch_has_id(entry,
          lexeme->id))
        || (0 == use_sketch && 0 != lexeme_stream_has_id(&entry_stream,
          lexeme->id))) ? 1 : 0;
      if (0 == matched && lexeme->length >= 5) {
        int term_len = 0;
        const char* term = libxs_lexicon_text(lexicon, lexeme->id,
          &term_len, NULL);
        if (0 == entry_stream_ready
          && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
            &entry_stream, (const unsigned char*)entry->text,
            (size_t)entry->text_len, rules, nrules, NULL, 0, 1))
        {
          entry_stream_ready = 1;
        }
        if (0 != entry_stream_ready
          && 0 != lexeme_stream_has_similar_text(&entry_stream, lexicon,
            term, term_len, (lexeme->length >= 7) ? 2 : 1))
        {
          matched = 1;
        }
      }
      if (0 != matched) {
        score += weight;
        ++matches;
        if (0 != long_term) ++long_matches;
      }
    }
  }
  bridge = answer_bridge_match(query, lexicon, entry);
  bridge_score = answer_semantic_bridge_score(bridge);
  if (bridge_score > 0.0) {
    score += bridge_score;
    total += 1.0;
    ++matches;
    if (long_terms > 0) ++long_matches;
  }
  libxs_lexeme_stream_release(&entry_stream);
  if (total <= 0.0 || 0 == matches) return 0.0;
  if (QUERY_GENERIC != query_type && long_terms > 0 && 0 == long_matches) {
    return 0.0;
  }
  if (QUERY_GENERIC != query_type) min_overlap = 0.40;
  overlap = score / total;
  if (bridge_score > 0.0 && overlap < bridge_score) overlap = bridge_score;
  if (overlap < min_overlap && !(total <= 1.5 && matches >= 1)) {
    return 0.0;
  }
  score = overlap;
  if (0 != use_sketch) {
    switch (query_type) {
      case QUERY_WHO:
        if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) score += 0.16;
        else score -= 0.06;
        break;
      case QUERY_WHERE:
        if (0 != (entry->lexical_flags & ENTRY_LEX_PLACE)) score += 0.14;
        if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) score += 0.04;
        break;
      case QUERY_WHEN:
        if (0 != (entry->lexical_flags & ENTRY_LEX_NUMBER)) score += 0.14;
        break;
      case QUERY_WHY:
        if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) score += 0.18;
        break;
      case QUERY_HOW:
        if (0 != (entry->lexical_flags & ENTRY_LEX_METHOD)) score += 0.14;
        if (0 != (entry->lexical_flags & ENTRY_LEX_CAUSE)) score += 0.06;
        break;
      case QUERY_YESNO:
        score += 0.04;
        break;
      default:
        break;
    }
  }
  entry_words = count_words((const unsigned char*)entry->text, entry->text_len);
  if (SCALE_SENTENCE == entry->scale) score += 0.10;
  score += 0.02 * matches;
  if (entry_words > 80) score -= 0.08;
  if (entry_words > 160) score -= 0.12;
  return score;
}


static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX])
{
  libxs_lexeme_stream_t query;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  int answer_count = 0;
  int slot;
  int query_type = QUERY_GENERIC;
  int result = 0;
  int limit = budget;
  const libxs_predict_t* predictor = answer_model;
  libxs_predict_t* query_predictor = NULL;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  const char* rank_query_text = query_text;
  size_t rank_query_len = query_len;
  char query_be_word[64];
  int query_be_upper = 0;
  int query_be_len = 0;
  if (query_section_len > 0) {
    size_t body_pos = 0;
    while (body_pos < query_len && ',' != query_text[body_pos]) ++body_pos;
    if (body_pos < query_len) {
      ++body_pos;
      while (body_pos < query_len
        && 0 != isspace((unsigned char)query_text[body_pos])) ++body_pos;
      if (body_pos < query_len) {
        rank_query_text = query_text + body_pos;
        rank_query_len = query_len - body_pos;
      }
    }
  }
  libxs_lexeme_stream_init(&query);
  if (limit < 1) limit = 1;
  if (limit > ANSWER_MAX) limit = ANSWER_MAX;
  for (slot = 0; slot < ANSWER_MAX; ++slot) {
    entries[slot] = NULL;
    scores[slot] = 0.0;
  }
  if (NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)rank_query_text, rank_query_len, rules, nrules,
      NULL, 0, 1))
  {
    query_type = query_type_of(&query, lexicon);
    query_be_len = answer_query_be_word(rank_query_text, rank_query_len,
      query_be_word, (int)sizeof(query_be_word), &query_be_upper);
    if (NULL == predictor) {
      query_predictor = answer_predict_build(corpus, &query, lexicon,
        rules, nrules, query_type, profile);
      predictor = query_predictor;
    }
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, 10, NULL) : sizeof(*entry);
      double base_score = 0.0;
      double score = base_score;
      double identity_score = 0.0;
      int relation_ranked = 0;
      answer_relation_match_t relation_match;
      double inputs[ANSWER_PREDICT_INPUTS];
      if (query_section_len > 0
        && 0 == corpus_entry_section_match(entry, entry_size,
          query_section, query_section_len))
      {
        value = libxs_registry_next(corpus, &key, &cursor);
        continue;
      }
      if (0 != query_type_prefers_sentence(query_type)
        && SCALE_SENTENCE == entry->scale
        && (0 == text_starts_sentence(entry->text, entry->text_len)
          || 0 == text_ends_sentence(entry->text, entry->text_len)))
      {
        value = libxs_registry_next(corpus, &key, &cursor);
        continue;
      }
      if (QUERY_WHO == query_type && query_be_len > 0
        && 0 == query_be_upper
        && 0 == answer_relation_match_query(rank_query_text, rank_query_len,
          query_type, entry, &relation_match))
      {
        value = libxs_registry_next(corpus, &key, &cursor);
        continue;
      }
      base_score = lexical_score(&query, lexicon, rules, nrules, entry,
        entry_size, query_type);
      score = base_score;
      identity_score = answer_identity_score(rank_query_text, rank_query_len,
        query_type, entry);
      if (identity_score > base_score) {
        base_score = identity_score;
        score = base_score;
        if (identity_score > 1.0) relation_ranked = 1;
      }
      if (base_score >= ANSWER_MIN_SCORE && entry->text_len >= 16) {
        if (EXIT_SUCCESS == answer_features(&query, entry, entry_size,
          query_type, inputs))
        {
          score = answer_predict_score(predictor, inputs, base_score);
          if (base_score > 1.0 && score < base_score) score = base_score;
        }
        if (0 != relation_ranked) score += base_score;
        if (0 != query_type_prefers_sentence(query_type)) {
          score += (SCALE_SENTENCE == entry->scale) ? 0.12 : -0.18;
        }
        for (slot = 0; slot < limit; ++slot) {
          if (NULL == entries[slot] || score > scores[slot]) {
            int move_slot;
            for (move_slot = limit - 1; move_slot > slot; --move_slot) {
              entries[move_slot] = entries[move_slot - 1];
              scores[move_slot] = scores[move_slot - 1];
            }
            entries[slot] = entry;
            scores[slot] = score;
            break;
          }
        }
      }
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  libxs_predict_destroy(query_predictor);
  for (slot = 0; slot < limit && NULL != entries[slot]; ++slot) {
    ++answer_count;
  }
  if (answer_count > 0) result = answer_count;
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role)
{
  static const char middle[] = " is the ";
  int result = EXIT_FAILURE;
  size_t role_len;
  size_t pos = 0;
  if (NULL == output || 0 == output_size || NULL == name || name_len <= 0
    || NULL == role) return EXIT_FAILURE;
  role_len = strlen(role);
  if ((size_t)name_len + sizeof(middle) - 1 + role_len + 2 < output_size) {
    memcpy(output + pos, name, (size_t)name_len);
    pos += (size_t)name_len;
    memcpy(output + pos, middle, sizeof(middle) - 1);
    pos += sizeof(middle) - 1;
    memcpy(output + pos, role, role_len);
    pos += role_len;
    output[pos++] = '.';
    output[pos] = '\0';
    result = EXIT_SUCCESS;
  }
  return result;
}


static void answer_strip_heading_prefix(const char** text, int* text_len)
{
  int pos;
  int prefix_words = 0;
  int in_word = 0;
  if (NULL == text || NULL == *text || NULL == text_len || *text_len <= 0) {
    return;
  }
  for (pos = 0; pos < *text_len; ++pos) {
    unsigned char ch = (unsigned char)(*text)[pos];
    if (0 != islower(ch) || '.' == ch || ',' == ch || ';' == ch
      || ':' == ch || '!' == ch || '?' == ch)
    {
      break;
    }
    if (0 != isupper(ch)) {
      if (0 == in_word) {
        ++prefix_words;
        in_word = 1;
      }
    }
    else if (0 != isspace(ch)) {
      in_word = 0;
    }
  }
  if (prefix_words >= 2 && pos < *text_len && pos > 0) {
    const char* next = *text + pos;
    int remaining = *text_len - pos;
    int prev_end = pos;
    int prev_start;
    while (prev_end > 0
      && 0 != isspace((unsigned char)(*text)[prev_end - 1]))
    {
      --prev_end;
    }
    prev_start = prev_end;
    while (prev_start > 0
      && 0 == isspace((unsigned char)(*text)[prev_start - 1]))
    {
      --prev_start;
    }
    if ((1 == prev_end - prev_start
        && 'A' == (*text)[prev_start])
      || (2 == prev_end - prev_start
        && 'A' == (*text)[prev_start] && 'N' == (*text)[prev_start + 1]))
    {
      next = *text + prev_start;
      remaining = *text_len - prev_start;
    }
    while (remaining > 0 && 0 != isspace((unsigned char)*next)) {
      ++next;
      --remaining;
    }
    if (remaining > 0) {
      *text = next;
      *text_len = remaining;
    }
  }
}


static size_t answer_append_clean(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len)
{
  int text_pos;
  int last_space = 1;
  if (NULL == output || 0 == output_size || NULL == text) return output_pos;
  if (text_len < 0) text_len = (int)strlen(text);
  for (text_pos = 0; text_pos < text_len && output_pos + 1 < output_size;
    ++text_pos)
  {
    unsigned char ch = (unsigned char)text[text_pos];
    if (0 != isspace(ch)) {
      if (0 == last_space) output[output_pos++] = ' ';
      last_space = 1;
    }
    else {
      output[output_pos++] = (char)ch;
      last_space = 0;
    }
  }
  output[output_pos] = '\0';
  return output_pos;
}


static int answer_frame_after(char* output, size_t output_size,
  size_t* output_pos, const char* text, int text_len, const char* marker)
{
  int result = EXIT_FAILURE;
  int start, end;
  if (NULL == output_pos || NULL == marker) return EXIT_FAILURE;
  start = text_find_ci(text, text_len, marker);
  if (start >= 0) {
    start += (int)strlen(marker);
    while (start < text_len && 0 != isspace((unsigned char)text[start])) {
      ++start;
    }
    end = start;
    while (end < text_len && '.' != text[end] && '!' != text[end]
      && '?' != text[end]) ++end;
    if (end > start) {
      *output_pos = answer_append_clean(output, output_size, *output_pos,
        text + start, end - start);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int answer_frame_keywords_after(char* output, size_t output_size,
  size_t* output_pos, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  const char* marker)
{
  int result = EXIT_FAILURE;
  int start, end;
  libxs_lexeme_stream_t stream;
  unsigned int used[32];
  size_t used_size = 0;
  size_t lexeme_pos;
  int wrote = 0;
  libxs_lexeme_stream_init(&stream);
  if (NULL == output_pos || NULL == lexicon || NULL == rules
    || nrules <= 0 || NULL == marker) return EXIT_FAILURE;
  start = text_find_ci(text, text_len, marker);
  if (start >= 0) {
    start += (int)strlen(marker);
    while (start < text_len && 0 != isspace((unsigned char)text[start])) {
      ++start;
    }
    end = start;
    while (end < text_len && '.' != text[end] && '!' != text[end]
      && '?' != text[end]) ++end;
    if (end > start && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, (const unsigned char*)text + start, (size_t)(end - start),
      rules, nrules, NULL, 0, 1))
    {
      for (lexeme_pos = 0; lexeme_pos < stream.size && used_size < 32;
        ++lexeme_pos)
      {
        const libxs_lexeme_t* lexeme = stream.data + lexeme_pos;
        int duplicate = 0;
        size_t used_pos;
        if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
        {
          for (used_pos = 0; used_pos < used_size && 0 == duplicate;
            ++used_pos)
          {
            if (used[used_pos] == lexeme->id) duplicate = 1;
          }
          if (0 == duplicate) {
            int token_len = 0;
            const char* token = libxs_lexicon_text(lexicon, lexeme->id,
              &token_len, NULL);
            if (NULL != token && token_len > 0) {
              if (0 != wrote) {
                *output_pos = answer_append_clean(output, output_size,
                  *output_pos, ", ", -1);
              }
              *output_pos = answer_append_clean(output, output_size,
                *output_pos, token, token_len);
              used[used_size++] = lexeme->id;
              wrote = 1;
            }
          }
        }
      }
      if (0 != wrote) result = EXIT_SUCCESS;
    }
  }
  libxs_lexeme_stream_release(&stream);
  return result;
}


static int answer_bridge_expand_reply(const answer_bridge_t* bridge,
  const char* text, int text_len, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, char* output, size_t output_size)
{
  int result = EXIT_SUCCESS;
  size_t output_pos = 0;
  const char* cursor;
  if (NULL == bridge || NULL == bridge->reply || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  cursor = bridge->reply;
  output[0] = '\0';
  while ('\0' != *cursor && output_pos + 1 < output_size
    && EXIT_SUCCESS == result)
  {
    const char* open = strchr(cursor, '{');
    if (NULL == open) {
      output_pos = answer_append_clean(output, output_size, output_pos,
        cursor, -1);
      cursor += strlen(cursor);
    }
    else {
      const char* close = strchr(open + 1, '}');
      output_pos = answer_append_clean(output, output_size, output_pos,
        cursor, (int)(open - cursor));
      if (NULL == close) {
        result = EXIT_FAILURE;
      }
      else if (0 == strncmp(open + 1, "after:", 6)) {
        char marker[128];
        int marker_len = (int)(close - (open + 7));
        if (marker_len > 0 && marker_len < (int)sizeof(marker)) {
          memcpy(marker, open + 7, (size_t)marker_len);
          marker[marker_len] = '\0';
          result = answer_frame_after(output, output_size, &output_pos,
            text, text_len, marker);
        }
        else result = EXIT_FAILURE;
      }
      else if (0 == strncmp(open + 1, "keywords-after:", 15)) {
        char marker[128];
        int marker_len = (int)(close - (open + 16));
        if (marker_len > 0 && marker_len < (int)sizeof(marker)) {
          memcpy(marker, open + 16, (size_t)marker_len);
          marker[marker_len] = '\0';
          result = answer_frame_keywords_after(output, output_size,
            &output_pos, lexicon, rules, nrules, text, text_len, marker);
        }
        else result = EXIT_FAILURE;
      }
      else {
        result = EXIT_FAILURE;
      }
      cursor = close + 1;
    }
  }
  if (EXIT_SUCCESS == result) {
    while (output_pos > 0 && 0 != isspace((unsigned char)output[output_pos - 1])) {
      --output_pos;
    }
    if (output_pos > 0 && '.' != output[output_pos - 1]
      && '?' != output[output_pos - 1] && '!' != output[output_pos - 1]
      && output_pos + 1 < output_size)
    {
      output[output_pos++] = '.';
    }
    output[output_pos] = '\0';
  }
  return result;
}


static int answer_reply_what_is(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  const char* target = NULL;
  int target_len = 0;
  size_t lexeme_pos;
  if (NULL == query || NULL == lexicon || NULL == text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  if (0 == lexeme_stream_has_text(query, lexicon, "what")
    || 0 == lexeme_stream_has_text(query, lexicon, "is"))
  {
    return EXIT_FAILURE;
  }
  for (lexeme_pos = 0; lexeme_pos < query->size; ++lexeme_pos) {
    const libxs_lexeme_t* lexeme = query->data + lexeme_pos;
    if (0 != (lexeme->flags & LIBXS_LEXEME_WORD)
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP)
      && 0 == lexeme_text_is(lexicon, lexeme, "what")
      && 0 == lexeme_text_is(lexicon, lexeme, "is"))
    {
      target = libxs_lexicon_text(lexicon, lexeme->id, &target_len, NULL);
    }
  }
  if (NULL != target && target_len > 0) {
    char target_buf[64];
    int pos;
    if (target_len < (int)sizeof(target_buf)) {
      memcpy(target_buf, target, (size_t)target_len);
      target_buf[target_len] = '\0';
      pos = text_find_ci(text, text_len, target_buf);
      if (pos >= 0) {
        int end = pos + target_len;
        while (end < text_len && ',' != text[end] && '.' != text[end]
          && '!' != text[end] && '?' != text[end]) ++end;
        if (end > pos) {
          size_t output_pos = answer_append_clean(output, output_size, 0,
            text + pos, end - pos);
          if (output_pos > 0) {
            output[0] = (char)toupper((unsigned char)output[0]);
            if ('.' != output[output_pos - 1] && '?' != output[output_pos - 1]
              && '!' != output[output_pos - 1] && output_pos + 1 < output_size)
            {
              output[output_pos++] = '.';
              output[output_pos] = '\0';
            }
            result = EXIT_SUCCESS;
          }
        }
      }
    }
  }
  return result;
}


static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  int query_type = QUERY_GENERIC;
  libxs_lexeme_stream_t query;
  const answer_bridge_t* bridge = NULL;
  const char* text;
  int text_len;
  char be_word[64];
  int be_upper = 0;
  int be_len = 0;
  answer_relation_match_t relation_match;
  libxs_lexeme_stream_init(&query);
  if (NULL == query_text || NULL == entry || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  text = entry->text;
  text_len = entry->text_len;
  while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
    ++text;
    --text_len;
  }
  while (text_len > 0 && 0 != isspace((unsigned char)text[text_len - 1])) {
    --text_len;
  }
  answer_strip_heading_prefix(&text, &text_len);
  if (NULL != lexicon && NULL != rules && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)query_text, query_len, rules, nrules,
      NULL, 0, 1))
  {
    query_type = query_type_of(&query, lexicon);
    bridge = answer_bridge_match(&query, lexicon, entry);
    be_len = answer_query_be_word(query_text, query_len, be_word,
      (int)sizeof(be_word), &be_upper);
  }
  if (NULL != bridge && NULL != bridge->reply) {
    result = answer_bridge_expand_reply(bridge, text, text_len, lexicon,
      rules, nrules, output, output_size);
  }
  else if (QUERY_WHO == query_type && be_len > 0 && 0 == be_upper
    && 0 != answer_relation_match_query(query_text, query_len, query_type,
      entry, &relation_match))
  {
    result = answer_relation_reply(&relation_match, output, output_size);
  }
  else if (QUERY_WHAT == query_type) {
    result = answer_reply_what_is(&query, lexicon, text, text_len,
      output, output_size);
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  libxs_lexeme_stream_t query;
  int best_start = -1, best_end = -1, best_score = 0;
  int sent_start = 0;
  int text_len;
  const char* text;
  libxs_lexeme_stream_init(&query);
  if (NULL == query_text || NULL == entry || NULL == lexicon || NULL == rules
    || nrules <= 0 || NULL == output || 0 == output_size) return EXIT_FAILURE;
  text = entry->text;
  text_len = entry->text_len;
  while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
    ++text;
    --text_len;
  }
  while (text_len > 0 && 0 != isspace((unsigned char)text[text_len - 1])) {
    --text_len;
  }
  answer_strip_heading_prefix(&text, &text_len);
  if (EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
    (const unsigned char*)query_text, query_len, rules, nrules,
    NULL, 0, 1))
  {
    int pos;
    for (pos = 0; pos <= text_len; ++pos) {
      if ((pos == text_len
          && 0 != text_ends_sentence(text + sent_start,
            text_len - sent_start))
        || '.' == text[pos] || '!' == text[pos]
        || '?' == text[pos])
      {
        int sent_end = (pos < text_len) ? pos + 1 : pos;
        int score = 0;
        size_t lexeme_pos;
        while (sent_end < text_len) {
          size_t close_size = text_closer_size((const unsigned char*)text,
            (size_t)text_len, (size_t)sent_end);
          if (0 == close_size) break;
          sent_end += (int)close_size;
        }
        for (lexeme_pos = 0; lexeme_pos < query.size; ++lexeme_pos) {
          const libxs_lexeme_t* lexeme = query.data + lexeme_pos;
          if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
            && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
          {
            int term_len = 0;
            const char* term = libxs_lexicon_text(lexicon, lexeme->id,
              &term_len, NULL);
            if (NULL != term && term_len > 0 && term_len < 64) {
              char term_buf[64];
              memcpy(term_buf, term, (size_t)term_len);
              term_buf[term_len] = '\0';
              if (0 != text_contains_ci(text + sent_start,
                sent_end - sent_start, term_buf)) ++score;
            }
          }
        }
        if (score > best_score) {
          best_score = score;
          best_start = sent_start;
          best_end = sent_end;
        }
        sent_start = sent_end;
        while (sent_start < text_len
          && 0 != isspace((unsigned char)text[sent_start])) ++sent_start;
      }
    }
  }
  if (best_score > 0 && best_end > best_start) {
    answer_append_clean(output, output_size, 0, text + best_start,
      best_end - best_start);
    result = EXIT_SUCCESS;
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  if (EXIT_SUCCESS == answer_relation_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_relation_aggregate_reply(corpus, query_text,
      query_len, output, output_size)
    || EXIT_SUCCESS == answer_identity_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_describe_fact_reply(query_text, query_len,
      output, output_size)
    || EXIT_SUCCESS == answer_docdef_fact_reply(query_text, query_len,
      output, output_size))
  {
    result = EXIT_SUCCESS;
  }
  return result;
}


static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  const corpus_entry_t* entries[ANSWER_MAX];
  double scores[ANSWER_MAX];
  int answer_count;
  int slot;
  char reply[COMPOSE_MAXTEXT];
  if (EXIT_SUCCESS == answer_fact_reply(corpus, query_text, query_len,
    reply, sizeof(reply)))
  {
    printf("%s\n", reply);
    return 1;
  }
  answer_count = answer_select(corpus, query_text, query_len, budget,
    lexicon, rules, nrules, answer_model, profile, entries, scores);
  if (answer_count > 0 && NULL != entries[0]
    && EXIT_SUCCESS == answer_reply(query_text, query_len, entries[0],
      lexicon, rules, nrules, reply, sizeof(reply)))
  {
    printf("%s\n", reply);
    LIBXS_UNUSED(scores);
    return 1;
  }
  for (slot = 0; slot < answer_count && NULL != entries[slot]; ++slot) {
    const char* text = entries[slot]->text;
    int text_len = entries[slot]->text_len;
    if (EXIT_SUCCESS == answer_evidence_sentence(query_text, query_len,
      entries[slot], lexicon, rules, nrules, reply, sizeof(reply)))
    {
      if (slot > 0) printf("\n");
      printf("%s\n", reply);
      continue;
    }
    while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
      ++text;
      --text_len;
    }
    while (text_len > 0
      && 0 != isspace((unsigned char)text[text_len - 1])) --text_len;
    answer_strip_heading_prefix(&text, &text_len);
    if (text_len > 0 && (SCALE_SENTENCE != entries[slot]->scale
        || (0 != text_starts_sentence(text, text_len)
          && 0 != text_ends_sentence(text, text_len))))
    {
      if (slot > 0) printf("\n");
      printf("%.*s\n", text_len, text);
    }
  }
  LIBXS_UNUSED(scores);
  return (answer_count > 0) ? 1 : 0;
}


static int text_find_ci(const char* text, int text_len, const char* term)
{
  int result = -1;
  int term_len, text_pos;
  if (NULL == text || NULL == term || text_len <= 0) return -1;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return -1;
  for (text_pos = 0; text_pos <= text_len - term_len && result < 0;
    ++text_pos)
  {
    int term_pos, match = 1;
    for (term_pos = 0; term_pos < term_len && 0 != match; ++term_pos) {
      unsigned char a = (unsigned char)text[text_pos + term_pos];
      unsigned char b = (unsigned char)term[term_pos];
      if (tolower(a) != tolower(b)) match = 0;
    }
    if (0 != match) result = text_pos;
  }
  return result;
}


static int text_contains_ci(const char* text, int text_len, const char* term)
{
  return (text_find_ci(text, text_len, term) >= 0) ? 1 : 0;
}


static int text_find_word_ci(const char* text, int text_len, const char* term)
{
  int result = -1;
  int term_len;
  int pos = 0;
  if (NULL == text || NULL == term || text_len <= 0) return -1;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return -1;
  while (pos <= text_len - term_len && result < 0) {
    int found = text_find_ci(text + pos, text_len - pos, term);
    if (found < 0) break;
    pos += found;
    if ((0 == pos || 0 == isalnum((unsigned char)text[pos - 1]))
      && (pos + term_len >= text_len
        || 0 == isalnum((unsigned char)text[pos + term_len])))
    {
      result = pos;
    }
    pos += term_len;
  }
  return result;
}


static int text_contains_word_ci(const char* text, int text_len,
  const char* term)
{
  int result = 0;
  int term_len;
  int pos = 0;
  if (NULL == text || NULL == term || text_len <= 0) return 0;
  term_len = (int)strlen(term);
  if (term_len <= 0 || term_len > text_len) return 0;
  while (pos <= text_len - term_len && 0 == result) {
    int found = text_find_ci(text + pos, text_len - pos, term);
    if (found < 0) break;
    pos += found;
    if ((0 == pos || 0 == isalnum((unsigned char)text[pos - 1]))
      && (pos + term_len >= text_len
        || 0 == isalnum((unsigned char)text[pos + term_len])))
    {
      result = 1;
    }
    pos += term_len;
  }
  return result;
}


static char* eval_trim(char* text)
{
  char* result = text;
  char* end;
  if (NULL != result) {
    while ('\0' != *result && 0 != isspace((unsigned char)*result)) ++result;
    end = result + strlen(result);
    while (end > result && 0 != isspace((unsigned char)end[-1])) --end;
    *end = '\0';
  }
  return result;
}


static int eval_parse_line(char* line, char* fields[4])
{
  int result = EXIT_FAILURE;
  char* cursor;
  int field_pos;
  if (NULL != fields) {
    for (field_pos = 0; field_pos < 4; ++field_pos) fields[field_pos] = NULL;
  }
  if (NULL != line && NULL != fields) {
    cursor = eval_trim(line);
    if ('\0' != *cursor && '#' != *cursor) {
      for (field_pos = 0; field_pos < 3 && NULL != cursor; ++field_pos) {
        char* sep = strchr(cursor, '|');
        if (NULL != sep) {
          *sep = '\0';
          fields[field_pos] = eval_trim(cursor);
          cursor = sep + 1;
        }
        else cursor = NULL;
      }
      if (NULL != cursor) fields[3] = eval_trim(cursor);
      if (NULL != fields[0] && '\0' != fields[0][0]
        && NULL != fields[1] && NULL != fields[2])
      {
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int eval_terms_empty(const char* spec)
{
  int result = 1;
  if (NULL != spec) {
    while ('\0' != *spec && 0 != isspace((unsigned char)*spec)) ++spec;
    result = ('\0' == *spec) ? 1 : 0;
  }
  return result;
}


static int eval_terms_match_text(const char* text, int text_len,
  const char* spec)
{
  int result = 1;
  int count = 0;
  int term_pos = 0;
  if (NULL == text || text_len <= 0) result = 0;
  while (0 != result && NULL != spec && NULL != text && text_len > 0) {
    int term_len = 0;
    const char* token = libxs_strtoken(spec, ",", term_pos, &term_len);
    char term[128];
    if (NULL == token) break;
    if (term_len > 0) {
      if (term_len >= (int)sizeof(term)) result = 0;
      else {
        memcpy(term, token, (size_t)term_len);
        term[term_len] = '\0';
        if (0 == text_contains_ci(text, text_len, term)) result = 0;
      }
      ++count;
    }
    ++term_pos;
  }
  if (0 == count && 0 == eval_terms_empty(spec)) result = 0;
  return result;
}


static int eval_terms_match_answers(const corpus_entry_t* entries[],
  int nanswers, const char* spec, int top_only)
{
  int result = 1;
  int count = 0;
  int term_pos = 0;
  if (NULL == entries || nanswers <= 0) result = 0;
  while (0 != result && NULL != spec && NULL != entries && nanswers > 0) {
    int term_len = 0;
    const char* token = libxs_strtoken(spec, ",", term_pos, &term_len);
    char term[128];
    if (NULL == token) break;
    if (term_len > 0) {
      int found = 0;
      int answer_pos;
      if (term_len >= (int)sizeof(term)) result = 0;
      else {
        memcpy(term, token, (size_t)term_len);
        term[term_len] = '\0';
        for (answer_pos = 0; answer_pos < nanswers && 0 == found;
          ++answer_pos)
        {
          if ((0 == top_only || 0 == answer_pos)
            && NULL != entries[answer_pos]
            && 0 != text_contains_ci(entries[answer_pos]->text,
              entries[answer_pos]->text_len, term))
          {
            found = 1;
          }
        }
        if (0 == found) result = 0;
      }
      ++count;
    }
    ++term_pos;
  }
  if (0 == count && 0 == eval_terms_empty(spec)) result = 0;
  return result;
}


static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_FAILURE;
  int npass = 0, ntop = 0, nany = 0, nreply = 0, nfact = 0;
  int ncases = 0;
  int have_facts = (0 != answer_relation_facts_size
    || 0 != answer_docdef_facts_size) ? 1 : 0;
  FILE* file;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL == corpus || NULL == lexicon || NULL == rules) return EXIT_FAILURE;
  file = fopen(converse_path_eval, "r");
  if (NULL == file) {
    fprintf(stderr, "eval: no %s file found\n", converse_path_eval);
  }
  while (NULL != file) {
    char line[EVAL_LINE_MAX];
    char* fields[4];
    const corpus_entry_t* entries[ANSWER_MAX];
    double scores[ANSWER_MAX];
    int nanswers;
    int top_pass;
    int any_pass;
    int reply_pass;
    int fact_pass;
    int fact_checked;
    int pass;
    char reply[COMPOSE_MAXTEXT];
    char rewritten[COMPOSE_MAXTEXT];
    const char* qtext;
    size_t qlen;
    if (NULL == fgets(line, (int)sizeof(line), file)) break;
    if (EXIT_SUCCESS != eval_parse_line(line, fields)) continue;
    ++ncases;
    if ('>' == fields[0][0]) {
      char* cont = eval_trim(fields[0] + 1);
      if (EXIT_SUCCESS == conv_rewrite(cont, strlen(cont), rewritten,
        sizeof(rewritten)))
      {
        qtext = rewritten;
      }
      else qtext = cont;
    }
    else {
      conv_reset();
      qtext = fields[0];
    }
    qlen = strlen(qtext);
    fact_pass = 1;
    fact_checked = (have_facts && NULL != fields[3]
      && 0 == eval_terms_empty(fields[3])) ? 1 : 0;
    if (0 != fact_checked) {
      if (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
        reply, sizeof(reply)))
      {
        fact_pass = eval_terms_match_text(reply, (int)strlen(reply),
          fields[3]);
      }
      else fact_pass = 0;
    }
    conv_remember(qtext, qlen);
    nanswers = answer_select(corpus, qtext, qlen,
      ANSWER_MAX, lexicon, rules, nrules,
      answer_model, profile, entries, scores);
    top_pass = eval_terms_match_answers(entries, nanswers, fields[1], 1);
    any_pass = eval_terms_match_answers(entries, nanswers, fields[1], 0);
    reply_pass = 1;
    LIBXS_UNUSED(scores);
    if (0 != eval_terms_empty(fields[1])) {
      if (0 != fact_checked) {
        pass = fact_pass;
        fprintf(stdout, "%s fact %s\n", (0 != fact_pass) ? "PASS" : "FAIL",
          fields[0]);
        if (0 != fact_pass) ++nfact;
      }
      else if (NULL != fields[3] && 0 == eval_terms_empty(fields[3])) {
        fprintf(stdout, "SKIP fact %s\n", fields[0]);
        --ncases;
        continue;
      }
      else {
        pass = (0 == nanswers) ? 1 : 0;
        fprintf(stdout, "%s abstain %s\n", (0 != pass) ? "PASS" : "FAIL",
          fields[0]);
      }
      if (0 != pass) ++npass;
      continue;
    }
    if (0 == eval_terms_empty(fields[2])) {
      if (nanswers <= 0 || EXIT_SUCCESS != answer_reply(qtext,
        qlen, entries[0], lexicon, rules, nrules, reply,
        sizeof(reply)))
      {
        reply_pass = 0;
        reply[0] = '\0';
      }
      else {
        reply_pass = eval_terms_match_text(reply, (int)strlen(reply),
          fields[2]);
      }
    }
    pass = (0 != any_pass && 0 != reply_pass && 0 != fact_pass) ? 1 : 0;
    fprintf(stdout, "%s top %s\n", (0 != top_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s any %s\n", (0 != any_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s reply %s\n", (0 != reply_pass) ? "PASS" : "FAIL",
      fields[0]);
    if (0 != fact_checked) {
      fprintf(stdout, "%s fact %s\n", (0 != fact_pass) ? "PASS" : "FAIL",
        fields[0]);
      if (0 != fact_pass) ++nfact;
    }
    if (0 != top_pass) ++ntop;
    if (0 != any_pass) ++nany;
    if (0 != reply_pass) ++nreply;
    if (0 != pass) ++npass;
  }
  fprintf(stdout,
    "eval[%s]: %d/%d passed (top=%d, any=%d, reply=%d, fact=%d)\n",
    profile->name, npass, ncases, ntop, nany, nreply, nfact);
  if (NULL != file) fclose(file);
  if (ncases > 0 && npass == ncases) result = EXIT_SUCCESS;
  return result;
}


/* The variable-order n-gram engine now lives in libxs_ngram; the wrappers
   below adapt converse's call sites (which thread the model's registry and an
   explicit maxorder) to the single shared model. model == converse_ngram.store
   and maxorder == converse_ngram.maxorder by construction. */
static void ngramk_observe(libxs_registry_t* model, const unsigned int hist[],
  int hlen, unsigned int succ_id, int maxorder)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  libxs_ngram_observe(&converse_ngram, hist, hlen, succ_id);
}


static const ngram_entry_t* ngramk_lookup(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int n)
{
  LIBXS_UNUSED(model);
  return libxs_ngram_lookup(&converse_ngram, hist, hlen, n);
}


static double ngramk_prob(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int next)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  return libxs_ngram_prob(&converse_ngram, hist, hlen, next);
}


static int ngramk_predict_order(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int out_ids[],
  int k, int* order)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  return libxs_ngram_predict(&converse_ngram, hist, hlen, out_ids, k, order);
}


static int ngramk_predict(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int out_ids[], int k)
{
  return ngramk_predict_order(model, hist, hlen, maxorder, out_ids, k, NULL);
}


static int ngram_gran_mode(void)
{
  const char* env = getenv("CONVERSE_GRAN");
  if (NULL != env) {
    if (0 == strcmp(env, "native")) return GRAN_NATIVE;
    if (0 == strcmp(env, "syllable")) return GRAN_SYLLABLE;
    if (0 == strcmp(env, "bpe")) return GRAN_BPE;
  }
  return GRAN_WORD;
}


static int ngram_native_mode(void)
{
  return (GRAN_WORD != ngram_gran_mode()) ? 1 : 0;
}


/* Number of prior whole words to carry as sub-word prediction context, or 0
   (off, byte-identical to piece-only context). Ignored at word/native mode. */
static int ngram_wordctx(void)
{
  int result = 0;
  const char* env = getenv("CONVERSE_WORDCTX");
  int mode = ngram_gran_mode();
  if (NULL != env && '\0' != *env
    && (GRAN_SYLLABLE == mode || GRAN_BPE == mode))
  {
    int v = atoi(env);
    if (v < 0) v = 0;
    if (v > NGRAM_ORDER_MAX) v = NGRAM_ORDER_MAX;
    result = v;
  }
  return result;
}


static int ngram_is_vowel(unsigned char c)
{
  c = (unsigned char)tolower(c);
  return ('a' == c || 'e' == c || 'i' == c || 'o' == c || 'u' == c
    || 'y' == c) ? 1 : 0;
}


static int ngram_is_wordchar(unsigned char c)
{
  return (0 != isalnum(c)) ? 1 : 0;
}


static int bpe_add_symbol(const char* bytes, int len)
{
  int result = -1;
  if (len > 0 && len <= BPE_SYMBOL_MAX) {
    if (bpe_nsymbols >= bpe_cap_symbols) {
      int grown = (bpe_cap_symbols > 0) ? bpe_cap_symbols * 2 : 512;
      bpe_symbol_t* next = (bpe_symbol_t*)realloc(bpe_symbols,
        (size_t)grown * sizeof(*next));
      if (NULL != next) {
        bpe_symbols = next;
        bpe_cap_symbols = grown;
      }
    }
    if (bpe_nsymbols < bpe_cap_symbols) {
      bpe_symbols[bpe_nsymbols].len = len;
      memcpy(bpe_symbols[bpe_nsymbols].bytes, bytes, (size_t)len);
      result = bpe_nsymbols;
      ++bpe_nsymbols;
    }
  }
  return result;
}


static void bpe_free(void)
{
  free(bpe_symbols);
  bpe_symbols = NULL;
  bpe_nsymbols = 0;
  bpe_cap_symbols = 0;
  libxs_registry_destroy(bpe_merges);
  bpe_merges = NULL;
}


/* Learn byte-pair merges from the training split only. Words are byte runs
   split on whitespace (a leading-space marker keeps word starts distinct);
   each iteration adds the most frequent adjacent symbol pair as a new symbol
   and records its rank so encoding can replay the merges. */
static void bpe_build(const libxs_registry_t* corpus, int holdout)
{
  int nmerges = BPE_MERGES_DEFAULT;
  const char* env = getenv("CONVERSE_BPE_MERGES");
  bpe_word_t* words = NULL;
  long nwords = 0, cap_words = 0;
  int merge;
  bpe_free();
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 0) nmerges = v;
  }
  bpe_merges = libxs_registry_create();
  if (NULL == corpus || NULL == bpe_merges) return;
  { int b;
    char one[1];
    for (b = 0; b < 256; ++b) {
      one[0] = (char)(unsigned char)b;
      bpe_add_symbol(one, 1);
    }
  }
  { const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && nwords < BPE_WORD_CAP) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      if (0 == predict_is_test(index, holdout)) {
        int pos = 0;
        while (pos < entry->text_len && nwords < BPE_WORD_CAP) {
          int wlen = 0, marker;
          bpe_word_t w;
          while (pos + wlen < entry->text_len
            && 0 == isspace((unsigned char)entry->text[pos + wlen])) ++wlen;
          if (wlen > 0) {
            int i;
            w.count = 1;
            w.nsyms = 0;
            marker = (pos > 0) ? 1 : 0;
            if (0 != marker && w.nsyms < BPE_WORD_MAX) {
              w.syms[w.nsyms++] = (int)(unsigned char)' ';
            }
            for (i = 0; i < wlen && w.nsyms < BPE_WORD_MAX; ++i) {
              w.syms[w.nsyms++] =
                (int)(unsigned char)entry->text[pos + i];
            }
            if (nwords >= cap_words) {
              long grown = (cap_words > 0) ? cap_words * 2 : 4096;
              bpe_word_t* next = (bpe_word_t*)realloc(words,
                (size_t)grown * sizeof(*next));
              if (NULL == next) break;
              words = next;
              cap_words = grown;
            }
            words[nwords++] = w;
            pos += wlen;
          }
          while (pos < entry->text_len
            && 0 != isspace((unsigned char)entry->text[pos])) ++pos;
        }
      }
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  for (merge = 0; merge < nmerges; ++merge) {
    libxs_registry_t* counts = libxs_registry_create();
    bpe_pair_t best;
    long best_count = 0;
    long w;
    int new_sym;
    char merged[BPE_SYMBOL_MAX];
    int merged_len;
    if (NULL == counts) break;
    best.a = best.b = -1;
    for (w = 0; w < nwords; ++w) {
      int i;
      for (i = 0; i + 1 < words[w].nsyms; ++i) {
        bpe_pair_t pair;
        long* slot;
        long acc;
        pair.a = words[w].syms[i];
        pair.b = words[w].syms[i + 1];
        slot = (long*)libxs_registry_get(counts, &pair, sizeof(pair), NULL);
        acc = (NULL != slot) ? *slot + words[w].count : words[w].count;
        if (NULL != slot) *slot = acc;
        else libxs_registry_set(counts, &pair, sizeof(pair), &acc,
          sizeof(acc), NULL);
        if (acc > best_count) {
          best_count = acc;
          best = pair;
        }
      }
    }
    libxs_registry_destroy(counts);
    if (best.a < 0 || best_count < 2) break;
    merged_len = bpe_symbols[best.a].len + bpe_symbols[best.b].len;
    if (merged_len > BPE_SYMBOL_MAX) break;
    memcpy(merged, bpe_symbols[best.a].bytes, (size_t)bpe_symbols[best.a].len);
    memcpy(merged + bpe_symbols[best.a].len, bpe_symbols[best.b].bytes,
      (size_t)bpe_symbols[best.b].len);
    new_sym = bpe_add_symbol(merged, merged_len);
    if (new_sym < 0) break;
    { bpe_rank_t rec;
      rec.rank = merge;
      rec.merged = new_sym;
      libxs_registry_set(bpe_merges, &best, sizeof(best), &rec, sizeof(rec),
        NULL);
    }
    for (w = 0; w < nwords; ++w) {
      int i = 0, out = 0;
      while (i < words[w].nsyms) {
        if (i + 1 < words[w].nsyms && words[w].syms[i] == best.a
          && words[w].syms[i + 1] == best.b)
        {
          words[w].syms[out++] = new_sym;
          i += 2;
        }
        else words[w].syms[out++] = words[w].syms[i++];
      }
      words[w].nsyms = out;
    }
  }
  free(words);
  fprintf(stderr, "  bpe: %d symbols (%d merges) from %ld words\n",
    bpe_nsymbols, bpe_nsymbols - 256, nwords);
}


/* Encode one whitespace-delimited byte run [text,len) into BPE pieces by
   greedily applying the lowest-rank applicable merge, then intern each piece.
   Falls back to single bytes for anything the merges do not cover, so the
   encoder never fails on unseen input. */
static int bpe_encode_run(const char* text, int len, libxs_token_t tokens[],
  int max, int start, libxs_lexicon_t* lexicon, int create)
{
  int result = start;
  int marker = (len > 0 && ' ' == text[0]) ? 1 : 0;
  int syms[BPE_WORD_MAX];
  int nsyms = 0, i;
  for (i = 0; i < len && nsyms < BPE_WORD_MAX; ++i) {
    syms[nsyms++] = (int)(unsigned char)text[i];
  }
  for (;;) {
    int best_rank = -1, best_pos = -1, best_sym = -1;
    for (i = 0; i + 1 < nsyms; ++i) {
      bpe_pair_t pair;
      const bpe_rank_t* rec;
      pair.a = syms[i];
      pair.b = syms[i + 1];
      rec = (const bpe_rank_t*)libxs_registry_get(bpe_merges, &pair,
        sizeof(pair), NULL);
      if (NULL != rec && (best_rank < 0 || rec->rank < best_rank)) {
        best_rank = rec->rank;
        best_pos = i;
        best_sym = rec->merged;
      }
    }
    if (best_pos < 0) break;
    syms[best_pos] = best_sym;
    for (i = best_pos + 1; i + 1 < nsyms; ++i) syms[i] = syms[i + 1];
    --nsyms;
  }
  for (i = 0; i < nsyms && result < max; ++i) {
    const bpe_symbol_t* sym = bpe_symbols + syms[i];
    int nbytes = sym->len;
    unsigned int id = libxs_lexicon_id(lexicon, sym->bytes, sym->len,
      LIBXS_LEXEME_WORD, create);
    if (0 == id) break;
    if (0 == i && 0 != marker && nbytes > 0) --nbytes;
    tokens[result].id = id;
    tokens[result].length = (unsigned short)nbytes;
    tokens[result].flags = (unsigned short)(LIBXS_TOKEN_WORD
      | ((0 == i && 0 != marker) ? LIBXS_TOKEN_BREAK : 0));
    ++result;
  }
  return result;
}


/* Split a word [text,wlen) into syllable pieces by a simple VC|CV heuristic:
   cut before a consonant that is followed by a vowel, once the current piece
   already contains a vowel. Caps piece length; always ends at word end. */
static int ngram_syllable_split(const char* text, int wlen, int piece_begin[],
  int piece_len[], int max)
{
  int result = 0;
  int start = 0, i = 0, seen_vowel = 0;
  while (i < wlen && result < max) {
    int cut = 0;
    if (i > start) {
      unsigned char c = (unsigned char)text[i];
      if (0 != seen_vowel && 0 == ngram_is_vowel(c)
        && i + 1 < wlen && 0 != ngram_is_vowel((unsigned char)text[i + 1]))
      {
        cut = 1;
      }
      else if (i - start >= NGRAM_NATIVE_WIDTH && 0 != seen_vowel) cut = 1;
    }
    if (0 != cut) {
      piece_begin[result] = start;
      piece_len[result] = i - start;
      ++result;
      start = i;
      seen_vowel = 0;
    }
    if (0 != ngram_is_vowel((unsigned char)text[i])) seen_vowel = 1;
    ++i;
  }
  if (start < wlen && result < max) {
    piece_begin[result] = start;
    piece_len[result] = wlen - start;
    ++result;
  }
  return result;
}


/* Emits LIBXS_TOKEN_BREAK on a token preceded by whitespace in the source, so
   libxs_token_word_next groups the pieces of one word. Native granularity cuts
   fixed-width chunks across word boundaries and hence marks none. When word_ids
   is non-NULL, each piece receives the whole-word lexicon id of the word it
   belongs to (its own id for native chunks and standalone punctuation), which
   lets a caller build word-span context over sub-word emission. */
static int ngram_native_tokens(libxs_lexicon_t* lexicon, const char* text,
  int text_len, libxs_token_t tokens[], unsigned int word_ids[], int max,
  int create)
{
  int result = 0;
  int pos = 0;
  int have_break = 0;
  int mode = ngram_gran_mode();
  if (GRAN_BPE == mode) {
    while (pos < text_len && result < max) {
      int wlen = 0;
      int run_start = pos;
      int marker = (pos > 0) ? 1 : 0;
      char run[BPE_WORD_MAX + 1];
      int run_len = 0;
      int prev = result;
      while (pos + wlen < text_len
        && 0 == isspace((unsigned char)text[pos + wlen])) ++wlen;
      if (0 == wlen) {
        while (pos < text_len
          && 0 != isspace((unsigned char)text[pos])) ++pos;
        continue;
      }
      if (0 != marker && run_len < BPE_WORD_MAX) run[run_len++] = ' ';
      { int i;
        for (i = 0; i < wlen && run_len < BPE_WORD_MAX; ++i) {
          run[run_len++] = text[run_start + i];
        }
      }
      result = bpe_encode_run(run, run_len, tokens, max, result,
        lexicon, create);
      if (NULL != word_ids && result > prev) {
        unsigned int wid = libxs_lexicon_id(lexicon, text + run_start,
          wlen, LIBXS_LEXEME_WORD, create);
        int j;
        for (j = prev; j < result; ++j) word_ids[j] = wid;
      }
      pos = run_start + wlen;
      while (pos < text_len && 0 != isspace((unsigned char)text[pos])) ++pos;
    }
    return result;
  }
  if (GRAN_SYLLABLE != mode) {
    while (pos < text_len && result < max) {
      int len = text_len - pos;
      unsigned int id;
      if (len > NGRAM_NATIVE_WIDTH) len = NGRAM_NATIVE_WIDTH;
      id = libxs_lexicon_id(lexicon, text + pos, len,
        LIBXS_LEXEME_WORD, create);
      if (0 == id) break;
      tokens[result].id = id;
      tokens[result].length = (unsigned short)len;
      tokens[result].flags = LIBXS_TOKEN_WORD;
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      pos += len;
    }
    return result;
  }
  while (pos < text_len && result < max) {
    unsigned char c = (unsigned char)text[pos];
    if (0 == ngram_is_wordchar(c)) {
      unsigned int id = libxs_lexicon_id(lexicon, text + pos, 1,
        LIBXS_LEXEME_PUNCT, create);
      if (0 == id) break;
      tokens[result].id = id;
      tokens[result].length = 1;
      tokens[result].flags = (unsigned short)(LIBXS_TOKEN_PUNCT
        | ((0 != have_break) ? LIBXS_TOKEN_BREAK : 0));
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      have_break = (0 != isspace(c)) ? 1 : 0;
      ++pos;
    }
    else {
      int wlen = 0;
      int piece_begin[COMPOSE_MAXTEXT / 2];
      int piece_len[COMPOSE_MAXTEXT / 2];
      int np, pi;
      unsigned int wid = 0;
      while (pos + wlen < text_len
        && 0 != ngram_is_wordchar((unsigned char)text[pos + wlen])) ++wlen;
      if (NULL != word_ids) {
        wid = libxs_lexicon_id(lexicon, text + pos, wlen,
          LIBXS_LEXEME_WORD, create);
      }
      np = ngram_syllable_split(text + pos, wlen, piece_begin, piece_len,
        (int)(sizeof(piece_len) / sizeof(*piece_len)));
      for (pi = 0; pi < np && result < max; ++pi) {
        char buf[LIBXS_LEXEME_MAXBYTES + 1];
        int plen = piece_len[pi];
        int off = 0;
        unsigned int id;
        if (0 == pi) buf[off++] = ' ';
        if (plen > (int)sizeof(buf) - 1 - off) plen = (int)sizeof(buf) - 1 - off;
        memcpy(buf + off, text + pos + piece_begin[pi], (size_t)plen);
        id = libxs_lexicon_id(lexicon, buf, off + plen,
          LIBXS_LEXEME_WORD, create);
        if (0 == id) break;
        tokens[result].id = id;
        tokens[result].length = (unsigned short)piece_len[pi];
        tokens[result].flags = (unsigned short)(LIBXS_TOKEN_WORD
          | ((0 == pi && 0 != have_break) ? LIBXS_TOKEN_BREAK : 0));
        if (NULL != word_ids) word_ids[result] = wid;
        ++result;
      }
      have_break = 0;
      pos += wlen;
    }
  }
  return result;
}


static int ngram_order(void)
{
  int result = (0 != converse_order_max) ? NGRAM_ORDER_MAX : 2;
  const char* env = getenv("CONVERSE_NGRAM_ORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


static void ngram_hist_push(unsigned int hist[], int* hlen, int cap,
  unsigned int id)
{
  if (*hlen < cap) hist[(*hlen)++] = id;
  else {
    int s;
    for (s = 1; s < cap; ++s) hist[s - 1] = hist[s];
    hist[cap - 1] = id;
  }
}


/* Word-span context for predicting sub-word token i: the whole-word ids of the
   preceding wctx words followed by the pieces of the current word emitted so
   far, kept as the most-recent cap entries. Rebuilt per position so that train
   and eval derive identical keys; groups are delimited by LIBXS_TOKEN_BREAK. */
static int ngram_wordctx_hist(const libxs_token_t nat[],
  const unsigned int word_ids[], int i, int wctx, unsigned int hist[], int cap)
{
  int hlen = 0;
  int wstart = i;
  unsigned int words[NGRAM_ORDER_MAX];
  int nw = 0, p, k;
  while (wstart > 0 && 0 == (nat[wstart].flags & LIBXS_TOKEN_BREAK)) --wstart;
  p = wstart;
  while (p > 0 && nw < wctx && nw < (int)(sizeof(words) / sizeof(*words))) {
    int ws = p - 1;
    while (ws > 0 && 0 == (nat[ws].flags & LIBXS_TOKEN_BREAK)) --ws;
    words[nw++] = word_ids[ws];
    p = ws;
  }
  for (k = nw - 1; k >= 0; --k) ngram_hist_push(hist, &hlen, cap, words[k]);
  for (k = wstart; k < i; ++k) ngram_hist_push(hist, &hlen, cap, nat[k].id);
  return hlen;
}


/* Per-order footprint of the n-gram store (gated by CONVERSE_NGRAM_STATS). */
static void ngram_stats(const libxs_registry_t* model)
{
  const char* env = getenv("CONVERSE_NGRAM_STATS");
  libxs_ngram_stats_t st;
  st.entries = 0;
  st.nbytes = 0;
  LIBXS_UNUSED(model);
  if (NULL != env && '\0' != *env && '0' != *env
    && EXIT_SUCCESS == libxs_ngram_stats(&converse_ngram, &st))
  {
    int n;
    for (n = 1; n <= LIBXS_NGRAM_ORDER_MAX; ++n) {
      if (st.keys[n] > 0) {
        fprintf(stdout,
          "ngram-stats[n=%d]: keys=%ld obs=%.0f obs/key=%.2f full=%.1f%%\n",
          n, st.keys[n], st.obs[n], st.obs[n] / (double)st.keys[n],
          100.0 * (double)st.saturated[n] / (double)st.keys[n]);
      }
    }
    if (st.entries > 0) {
      fprintf(stdout,
        "ngram-stats[all]: entries=%lu bytes=%lu bytes/entry=%.1f\n",
        (unsigned long)st.entries, (unsigned long)st.nbytes,
        (double)st.nbytes / (double)st.entries);
    }
  }
}


static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  int maxorder = ngram_order();
  if (0 != ngram_native_mode()) {
    int wctx = ngram_wordctx();
    libxs_token_t nat[COMPOSE_MAXTEXT];
    unsigned int word_ids[COMPOSE_MAXTEXT];
    int ntok = ngram_native_tokens(lexicon, text, text_len, nat,
      (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 1);
    unsigned int hist[NGRAM_ORDER_MAX];
    int hlen = 0, i;
    if (NULL == model) return;
    for (i = 0; i < ntok; ++i) {
      if (0 != wctx) {
        hlen = ngram_wordctx_hist(nat, word_ids, i, wctx, hist,
          NGRAM_ORDER_MAX);
        if (hlen > 0) ngramk_observe(model, hist, hlen, nat[i].id, maxorder);
      }
      else {
        if (hlen > 0) ngramk_observe(model, hist, hlen, nat[i].id, maxorder);
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, nat[i].id);
      }
    }
    return;
  }
  { libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    if (NULL != model && NULL != lexicon && NULL != rules && nrules > 0
      && text_len > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
        &stream, (const unsigned char*)text, (size_t)text_len, rules, nrules,
        NULL, 0, 1))
    {
      size_t pos;
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      for (pos = 0; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          if (hlen > 0) ngramk_observe(model, hist, hlen, lex->id, maxorder);
          if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = lex->id;
          else {
            int s;
            for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
            hist[NGRAM_ORDER_MAX - 1] = lex->id;
          }
        }
        if (0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) hlen = 0;
      }
    }
    libxs_lexeme_stream_release(&stream);
  }
}


static int predict_is_test(long index, int holdout)
{
  const char* tail;
  if (holdout <= 0) return 0;
  tail = getenv("CONVERSE_HOLDOUT_TAIL");
  if (NULL != tail && '0' != tail[0] && predict_ntotal > 0) {
    long split = predict_ntotal - predict_ntotal / (long)holdout;
    return (index >= split) ? 1 : 0;
  }
  return (0 == (index % (long)holdout)) ? 1 : 0;
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


static libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  libxs_ngram_destroy(&converse_ngram);
  if (EXIT_SUCCESS != libxs_ngram_create(&converse_ngram, ngram_order())) {
    return NULL;
  }
  if (NULL != corpus) {
    const void* key = NULL;
    size_t cursor = 0;
    long index = 0;
    void* value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      if (0 == predict_is_test(index, holdout)) {
        ngram_train_text(converse_ngram.store, lexicon, rules, nrules,
          entry->text, entry->text_len);
      }
      ++index;
      value = libxs_registry_next(corpus, &key, &cursor);
    }
  }
  return converse_ngram.store;
}


/* Rank the successors of a single record by count (converse's legacy order-2
   adapters expose a bare entry; the library ranks internally via predict). */
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


static void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  LIBXS_UNUSED(model);
  libxs_ngram_finalize(&converse_ngram, vocab);
}


static double ngram_unigram_prior(unsigned int id)
{
  double result = 0.0;
  if (NULL != converse_ngram.unifreq && 0 != id
    && id <= converse_ngram.unifreq_size)
  {
    result = (double)converse_ngram.unifreq[id] / converse_ngram.unifreq_total;
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


static double* token_emb = NULL;
static unsigned int token_emb_size = 0;


static void token_emb_free(void)
{
  free(token_emb);
  token_emb = NULL;
  token_emb_size = 0;
}


static const double* token_emb_get(unsigned int id)
{
  static const double zero[TOKEN_EMB_DIM] = { 0 };
  if (NULL != token_emb && 0 != id && id <= token_emb_size) {
    return token_emb + (size_t)id * TOKEN_EMB_DIM;
  }
  return zero;
}


static unsigned int token_emb_hash(unsigned int id)
{
  return (id * 2654435761u) % TOKEN_EMB_CTX;
}


static void token_emb_cooc_text(double* cooc, double* rowcnt,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  libxs_lexeme_stream_t stream;
  libxs_lexeme_stream_init(&stream);
  if (NULL != cooc && NULL != lexicon && NULL != rules && nrules > 0
    && text_len > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, (const unsigned char*)text, (size_t)text_len, rules, nrules,
      NULL, 0, 0))
  {
    unsigned int window[2 * TOKEN_EMB_WINDOW + 1];
    int fill = 0;
    size_t pos;
    for (pos = 0; pos <= stream.size; ++pos) {
      const libxs_lexeme_t* lex = (pos < stream.size)
        ? (stream.data + pos) : NULL;
      int boundary = (NULL == lex
        || 0 != (lex->flags & LIBXS_LEXEME_SENTENCE)) ? 1 : 0;
      if (NULL != lex && 0 == boundary
        && 0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        int i;
        for (i = 0; i < fill; ++i) {
          unsigned int other = window[i];
          cooc[(size_t)lex->id * TOKEN_EMB_CTX + token_emb_hash(other)] += 1.0;
          cooc[(size_t)other * TOKEN_EMB_CTX + token_emb_hash(lex->id)] += 1.0;
        }
        if (fill < 2 * TOKEN_EMB_WINDOW) {
          window[fill] = lex->id;
          ++fill;
        }
        else {
          for (i = 1; i < fill; ++i) window[i - 1] = window[i];
          window[fill - 1] = lex->id;
        }
        if (NULL != rowcnt) rowcnt[lex->id] += 1.0;
      }
      if (0 != boundary) fill = 0;
    }
  }
  libxs_lexeme_stream_release(&stream);
}


static void token_emb_reduce(double* cooc, unsigned int vocab)
{
  static double gram[TOKEN_EMB_CTX * TOKEN_EMB_CTX];
  static double basis[TOKEN_EMB_CTX * TOKEN_EMB_DIM];
  static double next[TOKEN_EMB_CTX * TOKEN_EMB_DIM];
  unsigned int id, i, j, d, e, iter;
  for (i = 0; i < TOKEN_EMB_CTX * TOKEN_EMB_CTX; ++i) gram[i] = 0.0;
  for (id = 1; id <= vocab; ++id) {
    const double* row = cooc + (size_t)id * TOKEN_EMB_CTX;
    for (i = 0; i < TOKEN_EMB_CTX; ++i) {
      if (0.0 != row[i]) {
        for (j = 0; j < TOKEN_EMB_CTX; ++j) {
          gram[i * TOKEN_EMB_CTX + j] += row[i] * row[j];
        }
      }
    }
  }
  for (i = 0; i < TOKEN_EMB_CTX; ++i) {
    for (d = 0; d < TOKEN_EMB_DIM; ++d) {
      unsigned int state = (i + 1) * 2654435761u + (d + 1) * 40503u;
      state ^= state >> 13; state *= 1274126177u; state ^= state >> 16;
      basis[i * TOKEN_EMB_DIM + d] = ((double)(state % 2000003u)
        / 1000001.5) - 1.0;
    }
  }
  for (iter = 0; iter < TOKEN_EMB_ITER; ++iter) {
    for (i = 0; i < TOKEN_EMB_CTX; ++i) {
      for (d = 0; d < TOKEN_EMB_DIM; ++d) {
        double acc = 0.0;
        for (j = 0; j < TOKEN_EMB_CTX; ++j) {
          acc += gram[i * TOKEN_EMB_CTX + j] * basis[j * TOKEN_EMB_DIM + d];
        }
        next[i * TOKEN_EMB_DIM + d] = acc;
      }
    }
    for (d = 0; d < TOKEN_EMB_DIM; ++d) {
      double norm = 0.0;
      for (e = 0; e < d; ++e) {
        double dot = 0.0;
        for (i = 0; i < TOKEN_EMB_CTX; ++i) {
          dot += next[i * TOKEN_EMB_DIM + d] * next[i * TOKEN_EMB_DIM + e];
        }
        for (i = 0; i < TOKEN_EMB_CTX; ++i) {
          next[i * TOKEN_EMB_DIM + d] -= dot * next[i * TOKEN_EMB_DIM + e];
        }
      }
      for (i = 0; i < TOKEN_EMB_CTX; ++i) {
        norm += next[i * TOKEN_EMB_DIM + d] * next[i * TOKEN_EMB_DIM + d];
      }
      norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
      for (i = 0; i < TOKEN_EMB_CTX; ++i) {
        next[i * TOKEN_EMB_DIM + d] *= norm;
      }
    }
    for (i = 0; i < TOKEN_EMB_CTX * TOKEN_EMB_DIM; ++i) basis[i] = next[i];
  }
  for (id = 1; id <= vocab; ++id) {
    const double* row = cooc + (size_t)id * TOKEN_EMB_CTX;
    double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
    double norm = 0.0;
    for (d = 0; d < TOKEN_EMB_DIM; ++d) {
      double acc = 0.0;
      for (i = 0; i < TOKEN_EMB_CTX; ++i) {
        if (0.0 != row[i]) acc += row[i] * basis[i * TOKEN_EMB_DIM + d];
      }
      emb[d] = acc;
      norm += acc * acc;
    }
    norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
    for (d = 0; d < TOKEN_EMB_DIM; ++d) emb[d] *= norm;
  }
}


static void token_emb_backfill(libxs_lexicon_t* lexicon,
  const double* rowcnt, unsigned int vocab)
{
  unsigned int id;
  for (id = 1; id <= vocab; ++id) {
    unsigned int flags = 0;
    int len = 0;
    const char* text = libxs_lexicon_text(lexicon, id, &len, &flags);
    if (NULL != text && len > 0 && len <= LIBXS_LEXEME_MAXBYTES
      && 0 != (flags & LIBXS_LEXEME_WORD)
      && rowcnt[id] < (double)TOKEN_EMB_BACKFILL_MIN)
    {
      char word[LIBXS_LEXEME_MAXBYTES + 1];
      char cand[LIBXS_LEXEME_MAXBYTES + 1];
      unsigned int other, best_id = 0;
      int best_dist = 1 + len / 4;
      memcpy(word, text, (size_t)len);
      word[len] = '\0';
      for (other = 1; other <= vocab; ++other) {
        unsigned int cand_flags = 0;
        int cand_len = 0;
        const char* cand_text = libxs_lexicon_text(lexicon, other, &cand_len,
          &cand_flags);
        if (other != id && NULL != cand_text && cand_len > 0
          && cand_len <= LIBXS_LEXEME_MAXBYTES
          && 0 != (cand_flags & LIBXS_LEXEME_WORD)
          && rowcnt[other] >= (double)TOKEN_EMB_BACKFILL_REF
          && (word[0] | 32) == (cand_text[0] | 32)
          && cand_len > len - best_dist && cand_len < len + best_dist)
        {
          int dist;
          memcpy(cand, cand_text, (size_t)cand_len);
          cand[cand_len] = '\0';
          dist = libxs_stridist(word, cand);
          if (dist >= 0 && dist < best_dist) {
            best_dist = dist;
            best_id = other;
          }
        }
      }
      if (0 != best_id) {
        double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
        const double* ref = token_emb + (size_t)best_id * TOKEN_EMB_DIM;
        double w = rowcnt[id] / (rowcnt[id] + 2.0);
        double norm = 0.0;
        int d;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) {
          emb[d] = w * emb[d] + (1.0 - w) * ref[d];
          norm += emb[d] * emb[d];
        }
        norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) emb[d] *= norm;
      }
    }
  }
}


static void token_emb_probe(libxs_lexicon_t* lexicon, unsigned int vocab)
{
  const char* probe = getenv("CONVERSE_EMB_PROBE");
  char word[LIBXS_LEXEME_MAXBYTES + 1];
  int len;
  while (NULL != probe && '\0' != *probe) {
    const char* end = strchr(probe, ',');
    unsigned int id;
    len = (NULL != end) ? (int)(end - probe) : (int)strlen(probe);
    if (len > 0 && len <= LIBXS_LEXEME_MAXBYTES) {
      memcpy(word, probe, (size_t)len);
      word[len] = '\0';
      id = libxs_lexicon_id(lexicon, word, len, 0, 0);
      if (0 != id && id <= vocab) {
        const double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
        unsigned int best[5] = { 0 }, other, slot, nbest = 0;
        double best_sim[5] = { 0 };
        fprintf(stderr, "emb[%s]:", word);
        for (other = 1; other <= vocab; ++other) {
          if (other != id) {
            const double* cand = token_emb + (size_t)other * TOKEN_EMB_DIM;
            double sim = 0.0;
            int d;
            for (d = 0; d < TOKEN_EMB_DIM; ++d) sim += emb[d] * cand[d];
            for (slot = 0; slot < nbest; ++slot) {
              if (sim > best_sim[slot]) break;
            }
            if (slot < 5) {
              unsigned int move;
              if (nbest < 5) ++nbest;
              for (move = nbest - 1; move > slot; --move) {
                best[move] = best[move - 1];
                best_sim[move] = best_sim[move - 1];
              }
              best[slot] = other;
              best_sim[slot] = sim;
            }
          }
        }
        for (slot = 0; slot < nbest; ++slot) {
          int blen = 0;
          const char* btext = libxs_lexicon_text(lexicon, best[slot], &blen,
            NULL);
          if (NULL != btext && blen > 0) {
            fprintf(stderr, " %.*s(%.2f)", blen, btext, best_sim[slot]);
          }
        }
        fprintf(stderr, "\n");
      }
      else fprintf(stderr, "emb[%s]: not in lexicon\n", word);
    }
    probe = (NULL != end) ? (end + 1) : (probe + len);
  }
}


static void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  token_emb_free();
  if (NULL != corpus && vocab > 0) {
    double* cooc = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_CTX,
      sizeof(double));
    double* rowcnt = (double*)calloc((size_t)(vocab + 1), sizeof(double));
    token_emb = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_DIM,
      sizeof(double));
    if (NULL != cooc && NULL != rowcnt && NULL != token_emb) {
      double colsum[TOKEN_EMB_CTX];
      double total = 0.0;
      const void* key = NULL;
      size_t cursor = 0;
      long index = 0;
      unsigned int id, i;
      const char* env;
      void* value = libxs_registry_begin(corpus, &key, &cursor);
      token_emb_size = vocab;
      while (NULL != value) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        if (0 == predict_is_test(index, holdout)) {
          token_emb_cooc_text(cooc, rowcnt, lexicon, rules, nrules,
            entry->text, entry->text_len);
        }
        ++index;
        value = libxs_registry_next(corpus, &key, &cursor);
      }
      for (i = 0; i < TOKEN_EMB_CTX; ++i) colsum[i] = 0.0;
      for (id = 1; id <= vocab; ++id) {
        const double* row = cooc + (size_t)id * TOKEN_EMB_CTX;
        for (i = 0; i < TOKEN_EMB_CTX; ++i) {
          colsum[i] += row[i];
          total += row[i];
        }
      }
      if (total > 0.0) {
        for (id = 1; id <= vocab; ++id) {
          double* row = cooc + (size_t)id * TOKEN_EMB_CTX;
          double rowsum = 0.0;
          for (i = 0; i < TOKEN_EMB_CTX; ++i) rowsum += row[i];
          for (i = 0; i < TOKEN_EMB_CTX; ++i) {
            if (row[i] > 0.0 && rowsum > 0.0 && colsum[i] > 0.0) {
              double pmi = log((row[i] * total) / (rowsum * colsum[i]));
              row[i] = (pmi > 0.0) ? pmi : 0.0;
            }
            else row[i] = 0.0;
          }
        }
        token_emb_reduce(cooc, vocab);
        env = getenv("CONVERSE_EMB_BACKFILL");
        if (NULL == env || '0' != *env) {
          token_emb_backfill(lexicon, rowcnt, vocab);
        }
        token_emb_probe(lexicon, vocab);
      }
    }
    free(cooc);
    free(rowcnt);
  }
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
      for (slot = 0; slot < k && slot < converse_ngram.backoff_count; ++slot) {
        out_ids[slot] = converse_ngram.backoff_ids[slot];
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
      NULL, 0, 0))
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


/* Fill hist[] with the trailing content-token ids of the prompt (at most
   NGRAM_ORDER_MAX), returning the count. Mirrors ngram_last_context but keeps
   the whole window the deep store can use, not just the final two ids. */
static int ngram_history(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, const char* text, int text_len, unsigned int hist[])
{
  libxs_lexeme_stream_t stream;
  int hlen = 0;
  libxs_lexeme_stream_init(&stream);
  if (NULL != lexicon && NULL != rules && nrules > 0 && text_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, (size_t)text_len, rules, nrules,
      NULL, 0, 0))
  {
    size_t pos;
    for (pos = 0; pos < stream.size; ++pos) {
      const libxs_lexeme_t* lex = stream.data + pos;
      if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = lex->id;
        else {
          int s;
          for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
          hist[NGRAM_ORDER_MAX - 1] = lex->id;
        }
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return hlen;
}


static int ngram_gen_minorder(void)
{
  int result = 2;
  const char* env = getenv("CONVERSE_GEN_MINORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}


/* Grounded greedy generation over the deep k-context store. Each step takes
   the top successor from the longest attested context and reports that
   context order; generation stops when the order falls below the grounding
   floor (the generative form of "abstain rather than invent"), at a repeat,
   or at the length budget. Prints the continuation plus the mean/min order
   that quantifies how well grounded it was. */
/* Greedy grounded continuation of the prompt into out[] (space-joined words).
   Returns the number of generated tokens (0 = nothing cleared the grounding
   floor). Optional order_mean/order_min report how attested the run was. */
static int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out)
{
  enum { GEN_MAX = 40 };
  unsigned int hist[NGRAM_ORDER_MAX];
  unsigned int recent[GEN_MAX];
  int hlen, maxorder = ngram_order();
  int minorder = ngram_gen_minorder();
  int step, nrecent = 0;
  long order_sum = 0;
  int order_min = NGRAM_ORDER_MAX + 1;
  size_t pos = 0;
  if (NULL != out && out_size > 0) out[0] = '\0';
  hlen = ngram_history(lexicon, rules, nrules, text, text_len, hist);
  for (step = 0; hlen > 0 && step < GEN_MAX; ++step) {
    unsigned int ids[1];
    int got_order = 0, len = 0, i, repeat = 0;
    const char* word;
    int n = ngramk_predict_order(model, hist, hlen, maxorder, ids, 1,
      &got_order);
    if (n <= 0 || got_order < minorder) break;
    for (i = 0; i < nrecent; ++i) if (recent[i] == ids[0]) ++repeat;
    if (repeat >= 3) break;
    word = libxs_lexicon_text(lexicon, ids[0], &len, NULL);
    if (NULL == word || len <= 0) break;
    if (NULL != out && pos + (size_t)len + 1 < out_size) {
      if (pos > 0) out[pos++] = ' ';
      memcpy(out + pos, word, (size_t)len);
      pos += (size_t)len;
      out[pos] = '\0';
    }
    order_sum += got_order;
    if (got_order < order_min) order_min = got_order;
    if (nrecent < GEN_MAX) recent[nrecent++] = ids[0];
    if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = ids[0];
    else {
      int s;
      for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
      hist[NGRAM_ORDER_MAX - 1] = ids[0];
    }
  }
  if (NULL != order_mean) {
    *order_mean = (step > 0) ? (double)order_sum / (double)step : 0.0;
  }
  if (NULL != order_min_out) *order_min_out = (step > 0) ? order_min : 0;
  return step;
}


static void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, const char* text,
  int text_len)
{
  char out[COMPOSE_MAXTEXT];
  double order_mean = 0.0;
  int order_min = 0;
  int ntok = ngram_generate(model, lexicon, rules, nrules, text, text_len,
    out, sizeof(out), &order_mean, &order_min);
  int minorder = ngram_gen_minorder();
  LIBXS_UNUSED(order);
  if (ntok > 0) {
    printf("generate: %s\n", out);
    printf("grounding: %d tokens, mean order %.1f, min order %d\n",
      ntok, order_mean, order_min);
  }
  else printf("(no grounded continuation at min order %d)\n", minorder);
}


/* Held-out generation quality: seed each test sentence with its first GEN_SEED
   content tokens, greedily extend, and count how many of the sentence's actual
   remaining tokens the generator reproduces before diverging. A gradient-free,
   generation-native counterpart to BPC (which only scores one-step prediction).
   Also reports the mean grounding order over generated tokens. */
static int ngram_gen_eval(libxs_registry_t* model,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int holdout, const char* kind)
{
  enum { GEN_SEED = 3, GEN_LOOK = 20 };
  int result = EXIT_FAILURE;
  long nsent = 0, sum_repro = 0, gen_tokens = 0, order_sum = 0, index = 0;
  int maxorder = ngram_order();
  int minorder = ngram_gen_minorder();
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
        (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, NULL, 0, 0))
    {
      unsigned int truth[GEN_SEED + GEN_LOOK];
      int ntruth = 0;
      size_t pos;
      for (pos = 0; pos < stream.size && ntruth < GEN_SEED + GEN_LOOK; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
          && 0 != lex->id)
        {
          truth[ntruth++] = lex->id;
        }
      }
      if (ntruth > GEN_SEED) {
        unsigned int hist[NGRAM_ORDER_MAX];
        int hlen = 0, t, repro = 0, diverged = 0;
        for (t = 0; t < GEN_SEED; ++t) hist[hlen++] = truth[t];
        for (t = GEN_SEED; t < ntruth && 0 == diverged; ++t) {
          unsigned int ids[1];
          int got_order = 0;
          int n = ngramk_predict_order(model, hist, hlen, maxorder, ids, 1,
            &got_order);
          if (n <= 0 || got_order < minorder) break;
          ++gen_tokens;
          order_sum += got_order;
          if (ids[0] == truth[t]) ++repro;
          else diverged = 1;
          if (hlen < NGRAM_ORDER_MAX) hist[hlen++] = truth[t];
          else {
            int s;
            for (s = 1; s < NGRAM_ORDER_MAX; ++s) hist[s - 1] = hist[s];
            hist[NGRAM_ORDER_MAX - 1] = truth[t];
          }
        }
        sum_repro += repro;
        ++nsent;
      }
    }
    libxs_lexeme_stream_release(&stream);
    ++index;
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (nsent > 0) {
    fprintf(stdout,
      "gen-eval[%s%s]: sentences=%ld mean-reproduced=%.2f "
      "mean-order=%.2f minorder=%d\n",
      kind, (holdout > 0) ? ":heldout" : "", nsent,
      (double)sum_repro / (double)nsent,
      (gen_tokens > 0) ? (double)order_sum / (double)gen_tokens : 0.0,
      minorder);
    result = EXIT_SUCCESS;
  }
  return result;
}


static int ngram_eval(libxs_registry_t* model, const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int order, int holdout, const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, index = 0;
  double sum_bits = 0.0, sum_bytes = 0.0;
  const double inv_log2 = 1.0 / log(2.0);
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  FILE* file;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    int native = ngram_native_mode();
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && (0 != native || EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, NULL, 0, 0)))
    {
      int wctx = ngram_wordctx();
      libxs_token_t nat[COMPOSE_MAXTEXT];
      unsigned int word_ids[COMPOSE_MAXTEXT];
      int ntok = (0 != native)
        ? ngram_native_tokens(lexicon, entry->text, entry->text_len,
          nat, (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 0)
        : (int)stream.size;
      int ti;
      unsigned int hist[NGRAM_ORDER_MAX];
      int hlen = 0;
      int maxorder = ngram_order();
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
            int n = ngramk_predict(model, hist, hlen, maxorder, ids,
              NGRAM_TOPK);
            double p = ngramk_prob(model, hist, hlen, maxorder, cur);
            int rank;
            ++npairs;
            sum_bits += -log(p) * inv_log2;
            sum_bytes += (curlen > 0) ? (double)curlen : 1.0;
            for (rank = 0; rank < n; ++rank) {
              if (ids[rank] == cur) {
                if (0 == rank) ++ntop1;
                ++ntopk;
                break;
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
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-ngram[%s%s]: top1=%.1f%% top%d=%.1f%% n=%ld bpc=%.3f\n",
      kind, (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      (sum_bytes > 0.0) ? sum_bits / sum_bytes : 0.0);
    result = EXIT_SUCCESS;
  }
  file = fopen(converse_path_predict_eval, "r");
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
  return result;
}


static libxs_predict_t* token_predict_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const answer_predict_profile_t* profile, int use_emb, int holdout)
{
  int ninputs = (0 != use_emb) ? 2 * TOKEN_EMB_DIM : 2;
  libxs_predict_t* model = libxs_predict_create(ninputs, 1);
  long pushed = 0;
  if (NULL != model && NULL != corpus && NULL != profile) {
    const void* key = NULL;
    size_t cursor = 0;
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
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, NULL, 0, 0))
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
              double in[2 * TOKEN_EMB_DIM];
              double out[1];
              token_input_vector(prev2, prev1, use_emb, in);
              out[0] = (double)lex->id;
              if (EXIT_SUCCESS == libxs_predict_push(NULL, model, in, out)) {
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
      value = libxs_registry_next(corpus, &key, &cursor);
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
  void* value;
  if (NULL == model || NULL == corpus || NULL == lexicon) return EXIT_FAILURE;
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, NULL, 0, 0))
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
    value = libxs_registry_next(corpus, &key, &cursor);
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
    for (slot = 0; slot < k && slot < converse_ngram.backoff_count; ++slot) {
      ids[slot] = converse_ngram.backoff_ids[slot];
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
    value = libxs_registry_begin(corpus, &key, &cursor);
    while (NULL != value && pushed < TOKEN_PREDICT_TRAIN_MAX) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      libxs_lexeme_stream_t stream;
      int is_train = (0 == predict_is_test(index, holdout));
      libxs_lexeme_stream_init(&stream);
      if (0 != is_train && entry->text_len > 0
        && EXIT_SUCCESS == libxs_lexeme_stream_encode(
        lexicon, &stream, (const unsigned char*)entry->text,
        (size_t)entry->text_len, rules, nrules, NULL, 0, 0))
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
      value = libxs_registry_next(corpus, &key, &cursor);
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
  void* value;
  if (NULL == ngram || NULL == reranker || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, NULL, 0, 0))
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
    value = libxs_registry_next(corpus, &key, &cursor);
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


/* Approximate NN over the static datastore is enabled by CONVERSE_KNNLM_ANN;
   default off keeps the exact brute-force scan (bit-identical results). */
static int knnlm_ann_mode(void)
{
  const char* env = getenv("CONVERSE_KNNLM_ANN");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


/* Quantize a context vector's leading dims to a Hilbert code: locality in the
   embedding space becomes locality along the 1-D code so a window around the
   query code holds its near neighbors. Values are the prev1 half of the input
   (the more predictive token, per the 4x weighting in knnlm_scan). */
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


/* Exact-distance rerank of one candidate against the query into the running
   top-K (same metric and heap discipline as knnlm_scan). */
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


/* Retrieve the static-cache neighbors of the query from a window around its
   Hilbert code, then exact-rerank them. Replaces the O(N) scan of the static
   cache; the dynamic store is still scanned brute-force by the caller. */
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


static void knnlm_dyn_insert(unsigned int prev2, unsigned int prev1,
  unsigned int next)
{
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
    token_input_vector(prev2, prev1, 1,
      knnlm_dyn_in + (size_t)knnlm_dyn_size * 2 * TOKEN_EMB_DIM);
    knnlm_dyn_next[knnlm_dyn_size] = next;
    ++knnlm_dyn_size;
  }
}


static void knnlm_scan(const double* in, const double* cin,
  const unsigned int* cnext, int count, unsigned int near_next[],
  double near_dist[], int* nnear)
{
  int i, dim;
  for (i = 0; i < count; ++i) {
    const double* entry = cin + (size_t)i * 2 * TOKEN_EMB_DIM;
    double dist = 0.0;
    int slot;
    if (0 == cnext[i]) continue;
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
      near_next[slot] = cnext[i];
      near_dist[slot] = dist;
      if (*nnear < KNNLM_K) ++*nnear;
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
    }
    else knnlm_cache_free();
  }
}


static int knnlm_vote(const libxs_predict_t* store, unsigned int prev2,
  unsigned int prev1, unsigned int vote_ids[], double vote_p[], int maxvote)
{
  int result = 0;
  if (NULL != store && 0 != prev1) {
    double in[2 * TOKEN_EMB_DIM];
    unsigned int near_next[KNNLM_K];
    double near_dist[KNNLM_K];
    unsigned int uniq_id[KNNLM_K];
    double uniq_w[KNNLM_K];
    int nnear = 0, nuniq = 0, i, j;
    double wtotal = 0.0;
    int ann = knnlm_ann_mode();
    if (knnlm_cache_model != store) {
      knnlm_cache_build(store);
      if (0 != ann) knnlm_ann_build();
    }
    token_input_vector(prev2, prev1, 1, in);
    if (0 != ann) {
      knnlm_ann_scan(in, near_next, near_dist, &nnear);
    }
    else {
      knnlm_scan(in, knnlm_cache_in, knnlm_cache_next, knnlm_cache_size,
        near_next, near_dist, &nnear);
    }
    knnlm_scan(in, knnlm_dyn_in, knnlm_dyn_next, knnlm_dyn_size,
      near_next, near_dist, &nnear);
    for (i = 0; i < nnear; ++i) {
      unsigned int next = near_next[i];
      double w = 1.0 / (0.05 + near_dist[i]);
      for (j = 0; j < nuniq; ++j) {
        if (uniq_id[j] == next) break;
      }
      if (j == nuniq) {
        uniq_id[j] = next;
        uniq_w[j] = 0.0;
        ++nuniq;
      }
      uniq_w[j] += w;
      wtotal += w;
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


static int knnlm_topk(libxs_registry_t* ngram, const libxs_predict_t* store,
  unsigned int prev2, unsigned int prev1, int order, unsigned int out_ids[],
  int k)
{
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
  int nvote = knnlm_vote(store, prev2, prev1, vote_ids, vote_p,
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
  return result;
}


static int knnlm_eval(libxs_registry_t* ngram, const libxs_predict_t* store,
  const libxs_registry_t* corpus, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int order, int holdout,
  const char* kind)
{
  int result = EXIT_FAILURE;
  long npairs = 0, ntop1 = 0, ntopk = 0, seen = 0, index = 0;
  int stride = predict_eval_stride();
  const char* dyn_env = getenv("CONVERSE_KNNLM_DYN");
  int dynamic = (NULL != dyn_env && '0' != dyn_env[0]) ? 1 : 0;
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  if (NULL == ngram || NULL == store || NULL == corpus || NULL == lexicon) {
    return EXIT_FAILURE;
  }
  if (0 != dynamic) knnlm_dyn_reset();
  value = libxs_registry_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    int is_test = (0 == holdout || 0 != predict_is_test(index, holdout));
    libxs_lexeme_stream_init(&stream);
    if (0 != is_test && entry->text_len > 0
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(
      lexicon, &stream, (const unsigned char*)entry->text,
      (size_t)entry->text_len, rules, nrules, NULL, 0, 0))
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
              int n = knnlm_topk(ngram, store, prev2, prev1, order, ids,
                NGRAM_TOPK);
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
            if (0 != dynamic) knnlm_dyn_insert(prev2, prev1, lex->id);
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
    value = libxs_registry_next(corpus, &key, &cursor);
  }
  if (npairs > 0) {
    fprintf(stdout,
      "predict-%s%s%s: top1=%.1f%% top%d=%.1f%% n=%ld (stride=%d)\n",
      kind, (0 != dynamic) ? ":dyn" : "", (holdout > 0) ? ":heldout" : "",
      100.0 * (double)ntop1 / (double)npairs, NGRAM_TOPK,
      100.0 * (double)ntopk / (double)npairs, npairs,
      stride);
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
  unsigned int ids[NGRAM_TOPK];
  int n, i;
  ngram_last_context(lexicon, rules, nrules, text, text_len, &prev2, &prev1);
  n = knnlm_topk(ngram, store, prev2, prev1, order, ids, NGRAM_TOPK);
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
      if (0 == knnlm_topk(ngram, store, step2, step1, order, step_ids, 1)) {
        break;
      }
      word = libxs_lexicon_text(lexicon, step_ids[0], &len, NULL);
      if (NULL == word || len <= 0) break;
      printf(" %.*s", len, word);
      step2 = step1;
      step1 = step_ids[0];
    }
    printf("\n");
  }
}


static int corpus_profile_for_path(const char* path)
{
  int result = PROFILE_PROSE;
  if (0 <= converse_profile_override) result = converse_profile_override;
  else if (NULL != path) {
    size_t len = strlen(path);
    if (len >= 3 && '.' == path[len - 3]
      && ('m' == path[len - 2] || 'M' == path[len - 2])
      && ('d' == path[len - 1] || 'D' == path[len - 1]))
    {
      result = PROFILE_MARKDOWN;
    }
  }
  return result;
}


static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like)
{
  int result = 0;
  int min_bytes = (0 != code_like) ? 2 : 8;
  int min_words = (0 != code_like) ? 0 : 3;
  while (len > 0 && 0 != isspace((unsigned char)text[len - 1])) --len;
  if (len >= min_bytes && len < COMPOSE_MAXTEXT
    && count_words(text, len) >= min_words)
  {
    corpus_entry_t entry;
    if (EXIT_SUCCESS == corpus_entry_build(&entry, text, len,
      SCALE_PARAGRAPH, lexicon, rules, nrules))
    {
      corpus_entry_set_section(&entry, section, section_len);
      if (0 != corpus_store_entry(corpus, &entry)) result = 1;
    }
  }
  return result;
}


static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, const char* section, int section_len,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int code_like)
{
  int result = 0;
  if (len < COMPOSE_MAXTEXT) {
    result = corpus_md_store(corpus, text, len, section, section_len,
      lexicon, rules, nrules, code_like);
  }
  else {
    int line_start = 0, i;
    for (i = 0; i <= len; ++i) {
      if (i == len || '\n' == text[i]) {
        if (i > line_start) {
          result += corpus_md_store(corpus, text + line_start,
            i - line_start, section, section_len, lexicon, rules, nrules,
            code_like);
        }
        line_start = i + 1;
      }
    }
  }
  return result;
}


static int corpus_ingest_markdown(libxs_registry_t* corpus,
  const unsigned char* text, size_t text_size, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int nblocks = 0;
  char section[ENTRY_SECTION_MAX];
  int section_len = 0;
  size_t line_start = 0, block_start = 0;
  int in_fence = 0, block_code = 0, block_has = 0;
  size_t i;
  section[0] = '\0';
  for (i = 0; i <= text_size; ++i) {
    if (i == text_size || '\n' == text[i]) {
      size_t ls = line_start;
      int line_len = (int)(i - line_start);
      int fence = 0;
      while (ls < i && 0 != isspace((unsigned char)text[ls])) ++ls;
      fence = (i - ls >= 3 && '`' == text[ls] && '`' == text[ls + 1]
        && '`' == text[ls + 2]) ? 1 : 0;
      if (0 != in_fence) {
        block_has = 1;
        if (0 != fence) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(i - block_start), section, section_len, lexicon, rules,
            nrules, 1);
          in_fence = 0;
          block_start = i + 1;
          block_has = 0;
        }
      }
      else if (0 != fence) {
        if (0 != block_has) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
        }
        in_fence = 1;
        block_start = line_start;
        block_code = 1;
        block_has = 0;
      }
      else if ('#' == (ls < i ? text[ls] : 0)) {
        size_t h = ls;
        if (0 != block_has) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
          block_has = 0;
        }
        while (h < i && '#' == text[h]) ++h;
        while (h < i && 0 != isspace((unsigned char)text[h])) ++h;
        section_len = (int)(i - h);
        if (section_len >= ENTRY_SECTION_MAX) section_len = ENTRY_SECTION_MAX - 1;
        if (section_len > 0) {
          memcpy(section, text + h, (size_t)section_len);
          section[section_len] = '\0';
        }
        else section[0] = '\0';
        block_start = i + 1;
        block_code = 0;
      }
      else if (0 == line_len) {
        size_t bs = block_start;
        int header_only = 0;
        while (bs < line_start && 0 != isspace((unsigned char)text[bs])) ++bs;
        if (0 != block_has && line_start - bs > 7
          && 0 == strncmp((const char*)text + bs, "Header:", 7))
        {
          header_only = 1;
        }
        if (0 != block_has && 0 == header_only) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), section, section_len, lexicon,
            rules, nrules, block_code);
          block_has = 0;
        }
        if (0 == header_only) {
          block_start = i + 1;
          block_code = 0;
        }
      }
      else {
        int table = ('|' == text[ls]) ? 1 : 0;
        if (0 == block_has) block_code = table;
        block_has = 1;
      }
      line_start = i + 1;
    }
  }
  if (0 != block_has) {
    nblocks += corpus_md_emit_block(corpus, text + block_start,
      (int)(text_size - block_start), section, section_len, lexicon, rules,
      nrules, block_code);
  }
  fprintf(stderr, "  ingested %s: %d markdown blocks\n", path, nblocks);
  return (nblocks > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}


static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_FAILURE;
  int profile;
  FILE* f;
  unsigned char* text = NULL;
  size_t text_size = 0;
  if (NULL == corpus || NULL == path) return EXIT_FAILURE;
  profile = corpus_profile_for_path(path);
  f = fopen(path, "rb");
  if (NULL != f) {
    fseek(f, 0, SEEK_END);
    text_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    if (text_size > 0) {
      text = (unsigned char*)malloc(text_size + 1);
      if (NULL != text) {
        if (fread(text, 1, text_size, f) == text_size) {
          text[text_size] = 0;
          result = EXIT_SUCCESS;
        }
      }
    }
    fclose(f);
  }
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0
    && PROFILE_MARKDOWN != profile)
  {
    unsigned char* reflowed = NULL;
    size_t reflowed_size = 0;
    if (EXIT_SUCCESS == libxs_text_reflow(text, text_size,
      &reflowed, &reflowed_size))
    {
      free(text);
      text = reflowed;
      text_size = reflowed_size;
    }
  }
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0
    && PROFILE_MARKDOWN == profile)
  {
    result = corpus_ingest_markdown(corpus, text, text_size, path,
      lexicon, rules, nrules);
  }
  else if (EXIT_SUCCESS == result && NULL != text && text_size > 0) {
    size_t sent_start = 0;
    int nsentences = 0, nparagraphs = 0;
    char current_section[ENTRY_SECTION_MAX];
    int current_section_len = 0;
    size_t i;
    current_section[0] = '\0';
    for (i = 0; i <= text_size; ++i) {
      int is_sent_end = (i == text_size)
        ? 1 : is_sentence_end_text(text, text_size, i);
      if (0 != is_sent_end && i > sent_start) {
          size_t sent_end = (i < text_size) ? i + 1 : i;
          int len = (int)(sent_end - sent_start);
          while (sent_end < text_size) {
            size_t close_size = text_closer_size(text, text_size, sent_end);
            if (0 == close_size) break;
            sent_end += close_size;
          }
          len = (int)(sent_end - sent_start);
          while (len > 0 && 0 != isspace(text[sent_start + len - 1])) --len;
          if (len > 8) {
            char title[ENTRY_SECTION_MAX];
            int title_len = corpus_title_prefix(text + sent_start, len,
              title, (int)sizeof(title));
            if (title_len > 0) {
              memcpy(current_section, title, (size_t)title_len + 1);
              current_section_len = title_len;
            }
          }
          if (len >= COMPOSE_MAXTEXT) {
            size_t frag_start = sent_start;
            while (frag_start < sent_end) {
              size_t scan = frag_start;
              size_t frag_end = frag_start;
              while (scan < sent_end && scan - frag_start < 240) {
                if (',' == text[scan] || ';' == text[scan]
                  || ':' == text[scan] || '.' == text[scan]
                  || '?' == text[scan] || '!' == text[scan])
                {
                  frag_end = scan + 1;
                }
                ++scan;
              }
              if (frag_end > frag_start) {
                size_t trim_start = frag_start;
                int frag_len;
                while (trim_start < frag_end
                  && 0 != isspace(text[trim_start])) ++trim_start;
                frag_len = (int)(frag_end - trim_start);
                while (frag_len > 0
                  && 0 != isspace(text[trim_start + frag_len - 1])) --frag_len;
                if (frag_len > 24 && frag_len < COMPOSE_MAXTEXT
                  && count_words(text + trim_start, frag_len) >= 4)
                {
                  corpus_entry_t entry;
                  if (EXIT_SUCCESS == corpus_entry_build(&entry,
                    text + trim_start, frag_len, SCALE_SENTENCE,
                    lexicon, rules, nrules))
                  {
                    corpus_entry_set_section(&entry, current_section,
                      current_section_len);
                    if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
                  }
                }
              }
              while (frag_start < sent_end && ',' != text[frag_start]
                && ';' != text[frag_start] && ':' != text[frag_start]
                && '.' != text[frag_start] && '?' != text[frag_start]
                && '!' != text[frag_start]) ++frag_start;
              if (frag_start < sent_end) ++frag_start;
              while (frag_start < sent_end && 0 != isspace(text[frag_start])) {
                ++frag_start;
              }
            }
          }
          if (len > 8 && len < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + sent_start, len);
            if (nwords >= 3) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + sent_start, len, SCALE_SENTENCE,
                lexicon, rules, nrules))
              {
                corpus_entry_set_section(&entry, current_section,
                  current_section_len);
                if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
              }
            }
          }
          sent_start = sent_end;
          while (sent_start < text_size && 0 != isspace(text[sent_start])) {
            ++sent_start;
          }
          i = (sent_start > 0) ? sent_start - 1 : 0;
      }
    }
    { size_t para_start = 0, p;
      char para_section[ENTRY_SECTION_MAX];
      int para_section_len = 0;
      para_section[0] = '\0';
      for (p = 0; p < text_size; ++p) {
        if ('\n' == text[p] && p + 1 < text_size && '\n' == text[p + 1]) {
          int plen = (int)(p - para_start);
          while (plen > 0 && 0 != isspace(text[para_start + plen - 1]))
            --plen;
          if (plen > 8) {
            char title[ENTRY_SECTION_MAX];
            int title_len = corpus_title_prefix(text + para_start, plen,
              title, (int)sizeof(title));
            if (title_len > 0) {
              memcpy(para_section, title, (size_t)title_len + 1);
              para_section_len = title_len;
            }
          }
          if (plen >= COMPOSE_MAXTEXT) {
            size_t frag_start = para_start;
            size_t para_end = para_start + (size_t)plen;
            while (frag_start < para_end) {
              size_t scan = frag_start;
              size_t frag_end = frag_start;
              while (scan < para_end && scan - frag_start < 240) {
                if (',' == text[scan] || ';' == text[scan]
                  || ':' == text[scan] || '.' == text[scan]
                  || '?' == text[scan] || '!' == text[scan])
                {
                  frag_end = scan + 1;
                }
                ++scan;
              }
              if (frag_end > frag_start) {
                size_t trim_start = frag_start;
                int frag_len;
                while (trim_start < frag_end
                  && 0 != isspace(text[trim_start])) ++trim_start;
                frag_len = (int)(frag_end - trim_start);
                while (frag_len > 0
                  && 0 != isspace(text[trim_start + frag_len - 1])) --frag_len;
                if (frag_len > 24 && frag_len < COMPOSE_MAXTEXT
                  && count_words(text + trim_start, frag_len) >= 4)
                {
                  corpus_entry_t entry;
                  if (EXIT_SUCCESS == corpus_entry_build(&entry,
                    text + trim_start, frag_len, SCALE_SENTENCE,
                    lexicon, rules, nrules))
                  {
                    corpus_entry_set_section(&entry, para_section,
                      para_section_len);
                    if (0 != corpus_store_entry(corpus, &entry)) ++nsentences;
                  }
                }
              }
              while (frag_start < para_end && ',' != text[frag_start]
                && ';' != text[frag_start] && ':' != text[frag_start]
                && '.' != text[frag_start] && '?' != text[frag_start]
                && '!' != text[frag_start]) ++frag_start;
              if (frag_start < para_end) ++frag_start;
              while (frag_start < para_end && 0 != isspace(text[frag_start])) {
                ++frag_start;
              }
            }
          }
          if (plen > 40 && plen < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + para_start, plen);
            if (nwords >= 8) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + para_start, plen, SCALE_PARAGRAPH,
                lexicon, rules, nrules))
              {
                corpus_entry_set_section(&entry, para_section,
                  para_section_len);
                if (0 != corpus_store_entry(corpus, &entry)) ++nparagraphs;
              }
            }
          }
          while (p + 1 < text_size && '\n' == text[p + 1]) ++p;
          para_start = p + 1;
        }
      }
    }
    fprintf(stderr, "  ingested %s: %d sentences, %d paragraphs\n",
      path, nsentences, nparagraphs);
  }
  free(text);
  return result;
}


static int corpus_ingest_basename(libxs_registry_t* corpus,
  const char* basename, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules)
{
  static const char* const separators[] = { "-", "_", "." };
  size_t basename_len;
  size_t ningested = 0;
  unsigned int part;
  int sep_index;
  int result = EXIT_SUCCESS;
  if (NULL == corpus || NULL == basename || '\0' == basename[0]) {
    return EXIT_FAILURE;
  }
  basename_len = strlen(basename);
  { char* path = (char*)malloc(basename_len + 5);
    if (NULL == path) return EXIT_FAILURE;
    sprintf(path, "%s", basename);
    if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
      rules, nrules)) ++ningested;
    sprintf(path, "%s.txt", basename);
    if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
      rules, nrules)) ++ningested;
    free(path);
  }
  for (part = 1; part <= CORPUS_BASENAME_PART_MAX; ++part) {
    for (sep_index = 0; sep_index < 3; ++sep_index) {
      char digits[16];
      char* path;
      size_t path_size;
      sprintf(digits, "%u", part);
      path_size = basename_len + strlen(separators[sep_index])
        + strlen(digits) + 5;
      path = (char*)malloc(path_size);
      if (NULL == path) {
        result = EXIT_FAILURE;
        part = CORPUS_BASENAME_PART_MAX;
        break;
      }
      sprintf(path, "%s%s%s.txt", basename, separators[sep_index], digits);
      if (EXIT_SUCCESS == corpus_ingest_file(corpus, path, lexicon,
        rules, nrules)) ++ningested;
      free(path);
    }
  }
  if (0 == ningested && EXIT_SUCCESS == result) {
    fprintf(stderr, "basename %s: no companion text files (name only)\n",
      basename);
    result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result) {
    fprintf(stderr, "basename %s: %lu files\n", basename,
      (unsigned long)ningested);
  }
  return result;
}


static double converse_score(const libxs_fprint_t* candidate,
  const libxs_fprint_t* query, const libxs_fprint_t* prev)
{
  double low = 0, high = 0, coherence = 0;
  int kmax = candidate->order < query->order
    ? candidate->order : query->order;
  int k;
  for (k = 0; k <= kmax; ++k) {
    double va = (candidate->nk[k] > 0)
      ? candidate->acc_sq[k] / candidate->nk[k] : 0;
    double vb = (query->nk[k] > 0)
      ? query->acc_sq[k] / query->nk[k] : 0;
    double ma = (candidate->nk[k] > 0)
      ? candidate->acc_sum[k] / candidate->nk[k] : 0;
    double mb = (query->nk[k] > 0)
      ? query->acc_sum[k] / query->nk[k] : 0;
    double dv = va - vb, dm = ma - mb;
    double d2 = dv * dv + dm * dm;
    if (k <= 1) low += d2;
    else high += d2;
  }
  if (NULL != prev) {
    for (k = 0; k <= kmax; ++k) {
      double va = (candidate->nk[k] > 0)
        ? candidate->acc_sq[k] / candidate->nk[k] : 0;
      double vb = (prev->nk[k] > 0)
        ? prev->acc_sq[k] / prev->nk[k] : 0;
      double ma = (candidate->nk[k] > 0)
        ? candidate->acc_sum[k] / candidate->nk[k] : 0;
      double mb = (prev->nk[k] > 0)
        ? prev->acc_sum[k] / prev->nk[k] : 0;
      double dv = va - vb, dm = ma - mb;
      coherence += dv * dv + dm * dm;
    }
  }
  return (low + 0.2 * coherence) / (1.0 + 0.1 * high);
}


static int entry_used(const corpus_entry_t* e,
  const corpus_entry_t* const used[], int nused)
{
  int result = 0, u;
  for (u = 0; u < nused && 0 == result; ++u) {
    if (used[u]->text_len == e->text_len
      && 0 == libxs_memcmp(used[u]->text, e->text, (size_t)e->text_len))
    {
      result = 1;
    }
  }
  return result;
}


static const corpus_entry_t* select_best(void* const candidates[],
  int ncandidates, const corpus_entry_t* const used[], int nused,
  const libxs_fprint_t* query, const libxs_fprint_t* prev,
  int require_scale, unsigned char preferred_scale)
{
  const corpus_entry_t* result = NULL;
  double best_score = 1e30;
  int ci;
  for (ci = 0; ci < ncandidates; ++ci) {
    const corpus_entry_t* e = (const corpus_entry_t*)candidates[ci];
    if (e->text_len >= 40
      && (0 == require_scale || e->scale == preferred_scale)
      && 0 == entry_used(e, used, nused))
    {
      double score = converse_score(&e->fprint, query, prev);
      if (score < best_score) {
        best_score = score;
        result = e;
      }
    }
  }
  return result;
}


static uint64_t query_hilbert_code(const libxs_fprint_t* fp)
{
  unsigned int coords[COMPOSE_NDIMS];
  int k;
  for (k = 0; k <= 4 && k <= fp->order; ++k) {
    double v = fp->l2[k], m = fp->mean[k];
    if (v < 0) v = 0;
    if (v > 1.0) v = 1.0;
    if (m < -1.0) m = -1.0;
    if (m > 1.0) m = 1.0;
    coords[k] = (unsigned int)(v * ((1 << COMPOSE_BITS) - 1));
    coords[5 + k] = (unsigned int)((m + 1.0) * 0.5 * ((1 << COMPOSE_BITS) - 1));
  }
  for (k = fp->order + 1; k <= 4; ++k) {
    coords[k] = 0;
    coords[5 + k] = 0;
  }
  return libxs_hilbert_bits(coords, COMPOSE_NDIMS, COMPOSE_BITS);
}


static int respond(const libxs_spatial_t* spatial,
  const libxs_registry_t* corpus, const char* query_text,
  size_t query_len, const libxs_fprint_t* query, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  int result = EXIT_SUCCESS;
  char output[65536];
  size_t out_pos = 0;
  int step;
  libxs_fprint_t prev_fprint;
  int have_prev = 0;
  const corpus_entry_t* used[64];
  int nused = 0;
  void* candidates[256];
  int ncandidates;
  uint64_t qcode = query_hilbert_code(query);
  unsigned char preferred_scale = (query->n > 60)
    ? SCALE_PARAGRAPH : SCALE_SENTENCE;

  if (0 != is_question_query(query_text, query_len, lexicon, rules, nrules)) {
    char rewritten[COMPOSE_MAXTEXT];
    const char* q = query_text;
    size_t qlen = query_len;
    if (EXIT_SUCCESS == conv_rewrite(query_text, query_len, rewritten,
      sizeof(rewritten)))
    {
      q = rewritten;
      qlen = strlen(rewritten);
    }
    if (0 != answer_query(corpus, q, qlen, budget,
      lexicon, rules, nrules, answer_model, profile))
    {
      conv_remember(q, qlen);
      return result;
    }
    { char gen[COMPOSE_MAXTEXT];
      int ntok = ngram_generate(converse_ngram.store, lexicon, rules, nrules,
        q, qlen, gen, sizeof(gen), NULL, NULL);
      if (ntok > 0) {
        printf("%s\n", gen);
        return result;
      }
    }
    printf("I do not know from the corpus.\n");
    return result;
  }

  ncandidates = libxs_spatial_nearest(spatial, qcode, 256, candidates);

  for (step = 0; step < budget; ++step) {
    const libxs_fprint_t* prev = (0 != have_prev) ? &prev_fprint : NULL;
    const corpus_entry_t* best_entry = select_best(candidates, ncandidates,
      used, nused, query, prev, 1, preferred_scale);

    if (NULL == best_entry) {
      best_entry = select_best(candidates, ncandidates,
        used, nused, query, prev, 0, preferred_scale);
    }
    if (NULL == best_entry) break;
    if (nused < 64) used[nused++] = best_entry;

    { const char* txt = best_entry->text;
      int tlen = best_entry->text_len;
      while (tlen > 0 && 0 != isspace((unsigned char)*txt)) {
        ++txt; --tlen;
      }
      while (tlen > 0 && 0 != isspace((unsigned char)txt[tlen - 1])) {
        --tlen;
      }
      if (tlen <= 0) continue;
      if (out_pos > 0 && out_pos + 1 < sizeof(output)) {
        output[out_pos++] = '\n';
      }
      if (out_pos + (size_t)tlen < sizeof(output)) {
        memcpy(output + out_pos, txt, (size_t)tlen);
        out_pos += (size_t)tlen;
      }
    }

    prev_fprint = best_entry->fprint;
    have_prev = 1;
  }

  if (out_pos > 0) {
    printf("%.*s\n", (int)out_pos, output);
  }
  return result;
}
