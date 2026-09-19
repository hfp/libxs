#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mix.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_hash.h>
#include <libxs/libxs_hist.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_timer.h>

#include "converse_core.h"

#include <unistd.h>

#define RESPONSE_BUDGET 1
#define CONVERSE_STAGE_MAX 12
/**
 * The single constant the rule learner's structural tests rest on: how unlikely
 * a member's counts must be, under what the asserted members manage, before the
 * learner is willing to call it a non-member. Every rate those tests compare
 * against is measured from the corpus; this is the only number chosen.
 */
#define ANSWER_RULES_ALPHA 1e-3

#define LEXICON_FILE "converse.lex"
#define PREDICT_FILE "converse.prd"
#define BRIDGE_FILE "converse.bridges"

#define RELATION_FILE "converse.relations"
/**
 * Language rules, shared by every corpus of that language and committed to the
 * repository (unlike the per-corpus .relations file). Kept language-neutral in
 * name so it can be a symlink to the actual language, e.g. english.rules.
 */
#define LANGUAGE_FILE "converse.rules"
/**
 * Normalization rules, kept apart from the interpretive rules because they are
 * a different kind: they rewrite what the tokenizer interns and therefore alter
 * the token stream every model is built from. Separating them keeps prediction
 * results independent of a change motivated by question answering, and keeps
 * the effect measurable by loading or omitting one file.
 */
#define NORM_FILE "converse.norms"
#define RELATION_LINE_MAX 1024
#define EVAL_FILE "converse.eval"
/** Expectations that hold with rule learning ON; the answers differ. */
#define EVAL_LEARN_FILE "converse.learn.eval"

#define NGRAM_FILE "converse.ngram"
/** Derived from the corpus and the rules: rebuilt whenever the stamp differs. */
#define FACTS_FILE "converse.facts"
#define PARENT_FILE "converse.par"
#define SOURCE_FILE "converse.src"

#define NGRAM_NATIVE_WIDTH 4

#define BPE_SYMBOL_MAX 32
#define BPE_WORD_MAX 128
#define BPE_MERGES_DEFAULT 750
#define BPE_WORD_CAP 80000
#define PREDICT_EVAL_FILE "converse.predict"
#define CONVERSE_PATH_MAX 512

#if !defined(TOKEN_EMB_WINDOW)
# define TOKEN_EMB_WINDOW 4
#endif
#if !defined(TOKEN_EMB_ITER)
# define TOKEN_EMB_ITER 24
#endif
#define TOKEN_EMB_BACKFILL_MIN 3
#define TOKEN_EMB_BACKFILL_REF 5

#define CORPUS_BASENAME_PART_MAX 999


enum { PROFILE_PROSE = 0, PROFILE_MARKDOWN = 1 };

enum { GRAN_WORD = 0, GRAN_NATIVE = 1, GRAN_SYLLABLE = 2, GRAN_BPE = 3,
  GRAN_META_NATIVE = 4, GRAN_META_WORD = 5, GRAN_META_SYLLABLE = 6 };

typedef struct ngram_key_t {
  unsigned int a;
  unsigned int b;
} ngram_key_t;

typedef struct token_emb_pair_t {
  unsigned int center;
  unsigned int context;
} token_emb_pair_t;

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

/**
 * A heading and the byte at which the text it names begins. `depth` is how many
 * blank lines precede the heading, which is the document's OWN statement of its
 * hierarchy, and is what lets a subsection heading be skipped so an answer is
 * credited to the source it belongs to rather than to a part of one.
 */
typedef struct corpus_section_t {
  size_t begin;
  int depth;
  int len;
  char text[ENTRY_SECTION_MAX];
} corpus_section_t;


static const answer_predict_profile_t answer_predict_profiles[] = {
  { "raw", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "poly2", LIBXS_PREDICT_INTERPOLATE, LIBXS_PREDICT_RAW, 0, 2, 0.0, 0.0, 0, 0, 0, 0 },
  { "smooth", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 0, 0, 0, 0 },
  { "temporal", LIBXS_PREDICT_TEMPORAL, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 1, ANSWER_PREDICT_INPUTS, 0, 1 },
  { "rf", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_RF, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "fisher", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_FISHER, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "hknn", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_HKNN, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  /**
   * The two cells that complete the mode x decompose square. "raw" is
   * (AUTO,RAW) and "hknn" is (CLASSIFY,HKNN), so those two differ in BOTH
   * variables and a difference between them cannot be attributed. These fill in
   * (CLASSIFY,RAW) and (AUTO,HKNN) so the partition can be varied with the mode
   * held fixed, and the other way round.
   */
  { "classify", LIBXS_PREDICT_CLASSIFY, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "autohknn", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_HKNN, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  /**
   * k-means with the cluster count raised rather than auto-selected, kept as the
   * record of a REFUTED hypothesis: raising it does not buy the scan back for
   * free. Measured at 2 MB, 40000 entries - auto (200 clusters) scans 3819 on
   * average for bpc 2.176; 1000 clusters scans 1594 for 2.184; the kd-tree
   * partition scans 155 for 2.180. So more clusters is dominated on BOTH axes,
   * and across all three the bits track the scan: a smaller cluster is less
   * local evidence, so the cost and the estimate are the same quantity. Only an
   * INDEX over an unchanged cluster can be free, which is why that is the
   * remaining option and a partition tweak is not.
   */
  { "raw1k", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 1000, 1, 0.0, 0.0, 0, 0, 0, 0 }
};

static char converse_path_corpus[CONVERSE_PATH_MAX] = CORPUS_FILE;
static char converse_path_parent[CONVERSE_PATH_MAX] = PARENT_FILE;
static char converse_path_source[CONVERSE_PATH_MAX] = SOURCE_FILE;
static char converse_path_lexicon[CONVERSE_PATH_MAX] = LEXICON_FILE;
static char converse_path_predict[CONVERSE_PATH_MAX] = PREDICT_FILE;
static char converse_path_bridge[CONVERSE_PATH_MAX] = BRIDGE_FILE;
static char converse_path_relation[CONVERSE_PATH_MAX] = RELATION_FILE;
static char converse_path_language[CONVERSE_PATH_MAX] = LANGUAGE_FILE;
/**
 * A corpus in another language needs its own function words and its own class
 * members, NOT the shared ones plus its own: an English `person` rule asserted
 * over a German text matched a German pronoun spelled the same way, and the
 * learner then carried a pronoun as an asserted class member. Empty unless the
 * namespace names one.
 */
static char converse_path_language_own[CONVERSE_PATH_MAX] = "";
static char converse_path_norm[CONVERSE_PATH_MAX] = NORM_FILE;
static char converse_path_facts[CONVERSE_PATH_MAX] = FACTS_FILE;
static char converse_path_eval[CONVERSE_PATH_MAX] = EVAL_FILE;
static char converse_path_eval_learn[CONVERSE_PATH_MAX] = EVAL_LEARN_FILE;
static char converse_path_predict_eval[CONVERSE_PATH_MAX] = PREDICT_EVAL_FILE;

static answer_relation_rule_t* answer_relation_rules = NULL;
static size_t answer_relation_rules_size = 0;
static libxs_lexnorm_t* answer_lexnorms = NULL;
static int answer_lexnorms_size = 0;
/**
 * Source file an entry came from, assigned in ingest order (1-based; 0 means
 * unknown). A section alone cannot identify a source in the documentation
 * corpus: 14 pages carry a "Usage" heading and 12 carry "Example", so two
 * entries with equal section text may well be different modules. Coherence
 * measurements that compare sections therefore need this to tell a same-page
 * join from a cross-page one, and only the latter combines two sources.
 */
static unsigned short corpus_source_id = 0;
/**
 * Paths of the ingested files, indexed by the source id every record carries. The
 * ids were anonymous before, which is why a citation could name a section but never
 * a file; persisted beside the corpus so a warm start can still resolve them.
 */
static char** corpus_source_paths = NULL;
static int corpus_source_npaths = 0;
/** Line map of the file being ingested: see libxs_text_reflow_map. */
static size_t* corpus_ingest_lines = NULL;
static size_t corpus_ingest_nlines = 0;
/**
 * Where each section of the file being ingested begins. Built once from the
 * LINE structure of the document, because a section is a property of the
 * document and not of whichever pass happens to be scanning it: the sentence
 * pass and the paragraph pass used to track a "current section" each and
 * disagreed wherever a subsection heading stood between them. Since the section
 * is part of the entry key, disagreement stored the same sentence TWICE under
 * two headings, and one reply came back twice with two different citations.
 */
static corpus_section_t* corpus_sections = NULL;
/**
 * Monotone cursor for mapping a byte offset to a line of the SOURCE FILE.
 *
 * Ingest visits offsets in increasing order within each pass, so a cursor turns
 * what would be a scan per entry into one scan per pass. `map` is the reflow line
 * map when the text was reflowed, since a line of the reflowed text is not a line
 * of the file the reader will open.
 */
static int corpus_sections_size = 0;
static int corpus_sections_cap = 0;

static long predict_ntotal = 0;

static libxs_ngram_t converse_ngram;
static libxs_ngram_t converse_skip;
static int converse_skip_on = 0;

static int converse_profile_override = -1;
static int converse_order_max = 0;
static const char* converse_stage_name[CONVERSE_STAGE_MAX];
static double converse_stage_time[CONVERSE_STAGE_MAX];
static int converse_stage_count = 0;
static long corpus_chain_dropped = 0;

static libxs_timer_tick_t converse_stage_tick = 0;
static bpe_symbol_t* bpe_symbols = NULL;
static int bpe_nsymbols = 0;
static int bpe_cap_symbols = 0;
static libxs_registry_t* bpe_merges = NULL;

static double* token_emb = NULL;
static double* token_semb = NULL;
static double* token_emb_prior = NULL;
static double* token_emb_work = NULL;
static unsigned int token_emb_size = 0;
/**
 * Memo for the successor normalizer. Z depends on the CONTEXT only, so scoring k
 * candidates at one position recomputed the same vocabulary pass k times - which
 * is invisible in the BPC path (one candidate per position) and made generation,
 * where the bank ranks up to GEN_CAND_MAX candidates, cost k vocabulary scans per
 * token. A pure memo: identical values, not an approximation.
 */
static unsigned int token_emb_zids[TOKEN_CTX_MAX];
static int token_emb_znctx = 0;
static unsigned int token_emb_zvocab = 0;
static double token_emb_ztemp = 0.0;
static double token_emb_zmax = 0.0;
static double token_emb_zsum = 0.0;
static int token_emb_zvalid = 0;
/** Tokenizer rules, held here because converse_run_t only borrows them. */
static libxs_lexrule_t converse_lexrules[96];

/**
 * Parent texts live in their OWN registry, and that is not tidiness: the corpus is
 * an open-addressed table iterated by slot, so records of any kind displace the
 * probe sequence of colliding keys and PERMUTE the iteration. Donor selection is
 * defined over corpus order ("a donor is an entry later than the host"), so 393
 * parents mixed into the entry table were enough to move one join in fifty and with
 * it every seam average - while every count stayed identical, which is what made it
 * look like a materialization bug. One registry, one kind of record.
 *
 * Parents are numbered rather than hashed, so a span costs four bytes to point at
 * one. The counter resumes from the highest id the loaded file holds, recovered
 * while the load is visiting every record anyway - otherwise a warm ingest would
 * hand out ids that are already taken.
 */
static libxs_registry_t* corpus_parents = NULL;
static unsigned int corpus_blob_max = 0;
static long corpus_span_mismatch = 0;
static long corpus_span_stored = 0;
static libxs_registry_t* corpus_views = NULL;
static libxs_lexicon_t* corpus_view_lexicon = NULL;
static const libxs_lexrule_t* corpus_view_rules = NULL;
static int corpus_view_nrules = 0;
static long corpus_view_count = 0;
/**
 * Whether to store clause fragments cut from a long PARAGRAPH. They are 99.7% of
 * all fragments and 71% of all entries, and unlike the fragments of an over-long
 * SENTENCE their bytes are already covered by the paragraph entry - so the
 * question of whether they earn their metadata is answerable by measurement.
 */
/**
 * Verify at ingest that a window REBUILDS to the entry it replaces, byte for byte.
 *
 * The claim a derived pool rests on is that materialization is a pure function of
 * the parent bytes, so this compares the two at the one moment both exist. Off by
 * default because it doubles the tokenizer work of ingest; it is the instrument that
 * settles whether a moved figure is a materialization bug or something else.
 */
static int corpus_span_check(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_SPAN_CHECK");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static int corpus_fragments_para(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_FRAGMENTS");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 1;
  }
  return cached;
}


/**
 * The installed judge and the model it opened. Held here rather than in either
 * half because both may consult it and neither may name its type: the QA half
 * rescores answers with it, generation chooses among successors with it, and the
 * recombination probe fills its bpc columns from it. NULL is a supported state
 * throughout - "no instrument", not "no result".
 */
static const converse_judge_t* converse_judge_vtable = NULL;
static void* converse_judge_opened = NULL;


static void corpus_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata);
static const answer_predict_profile_t* answer_predict_profile_find(
  const char* name);
static void answer_predict_profile_list(FILE* stream);
static void answer_relation_rules_free(void);
static char* answer_relation_rule_trim(char* text);
static int answer_relation_rule_kind(const char* text);
static int answer_relation_rule_append(int kind, const char* relation,
  const char* term, int provenance);
static int answer_relation_rule_parse_line(char* line);
static size_t answer_relation_rules_load_file(const char* path);
static void answer_relation_rules_report(FILE* stream);
static void token_emb_pair_probe(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int answer_rules_learn_count(void);
static double answer_rules_learn_env(const char* name, double fallback);
static void answer_rules_centroid(double* dst, const double* sum);
static double answer_rules_cosine(const double* lhs, const double* rhs);
static double answer_rules_score(const double* cfwd, const double* cbwd,
  unsigned int id);
static void answer_rules_member(double* csum, double* ssum, unsigned int id);
static double answer_rules_tail(long k, long n, double p);
static int answer_rules_implausible(long k, long n, double pmin);
static double answer_rules_ceiling(long k, long n);
static int answer_rules_excluded(int test, unsigned int id, const long* freq,
  const long* nhead, const long* nmod, const long* nintro,
  const double* pmin);
static void answer_rules_count_intro(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const unsigned char* isintro, long* nintro, unsigned int vocab);
static size_t answer_relation_rules_learn(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void answer_lexnorms_build(void);
static void converse_namespace_init(const char* prefix);
static libxs_registry_t* corpus_load(void);
static int corpus_save(const libxs_registry_t* corpus);
static libxs_lexicon_t* converse_lexicon_load(int* stale);
static int converse_lexicon_save(const libxs_lexicon_t* lexicon);
static libxs_predict_t* converse_predict_load(void);
static int converse_predict_save(const libxs_predict_t* model);
static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos);
static int corpus_shuffle_words(unsigned char* text, int len);
static int corpus_shuffle_mode(void);
static void corpus_entry_set_section(corpus_entry_t* entry,
  const char* section, int section_len);
static void corpus_sections_build(const unsigned char* text, size_t size);
static int corpus_fragments_para(void);
static int corpus_store_clauses(libxs_registry_t* corpus,
  const unsigned char* text, size_t begin, size_t end, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int derive);
static int corpus_word_byte(const char* text, int at, int len);
static int corpus_word_upper(const char* text, int at, int len);
static unsigned int corpus_word_id(libxs_lexicon_t* lexicon, const char* word,
  int word_len);
static int corpus_section_at(size_t offset, const char** text);
static unsigned int corpus_chain_max(void);
static void corpus_key_from_text(const corpus_entry_t* entry,
  unsigned char key[], size_t* key_size);
static int corpus_record_text(const void* value, size_t value_size,
  const char** text, int* len, const char** section, int* section_len);
static unsigned int corpus_store_blob(const unsigned char* text, int len,
  const char* section, int section_len);
/** The 1-based source-file line an ingest offset belongs to. */
static unsigned int corpus_line_at(const unsigned char* text, size_t size,
  size_t offset);
static void corpus_lines_index(const unsigned char* text, size_t size);
static int corpus_source_path_set(int id, const char* path);
static void corpus_source_paths_free(void);
static void corpus_source_paths_save(const char* path);
static void corpus_source_paths_load(const char* path);
static int corpus_store_record(libxs_registry_t* corpus,
  const corpus_entry_t* entry, const corpus_span_t* span);
static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry);
static double answer_weak_label(const corpus_entry_t* entry, int query_type);
static void answer_predict_report(const char* label,
  const libxs_predict_t* model, int ntrain,
  const answer_predict_profile_t* profile);
static libxs_predict_t* converse_predict_train(const libxs_registry_t* corpus,
  const answer_predict_profile_t* profile);
static void ngramk_observe(libxs_registry_t* model, const unsigned int hist[],
  int hlen, unsigned int succ_id, int maxorder);
static int ngram_gran_mode(void);
static int ngram_is_vowel(unsigned char c);
static unsigned long ngram_utf8_decode(const char* text, int len, int* width);
static int ngram_is_vowel_cp(unsigned long cp);
static int bpe_add_symbol(const char* bytes, int len);
static void bpe_free(void);
static void bpe_build(const libxs_registry_t* corpus, int holdout);
static int bpe_encode_run(const char* text, int len, libxs_lexeme_t tokens[],
  int max, int start, libxs_lexicon_t* lexicon, int create);
static int ngram_onset_legal(const char* text, int a, int b);
static int ngram_metatoken_granularity(int mode);
static unsigned int ngram_metatoken_flags(int kind, int sentence,
  int have_break);
static int ngram_metatoken_tokens(libxs_lexicon_t* lexicon,
  const char* text, int text_len, libxs_lexeme_t tokens[],
  unsigned int word_ids[], int max, int create, int granularity);
static int converse_predict_on(void);
static int converse_stage_on(void);
static int ngram_skip(void);
static double ngram_skip_mu(void);
static void ngram_skip_observe(const unsigned int hist[], int hlen,
  unsigned int succ_id);
static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len);
static void token_emb_free(void);
static int token_emb_distonly(void);
static int token_emb_ready(void);
static const double* token_semb_get(unsigned int id);
static double ngram_emb_decay(void);
static int token_emb_succ_prepare(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp);
static void token_emb_pair_observe(libxs_registry_t* pairs,
  unsigned int center, unsigned int context);
static void token_emb_cooc_text(libxs_registry_t* pairs, double* rowcnt,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len);
static void token_emb_spmm(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab, const double* in, int transpose,
  double* out);
static void token_emb_orthonormalize(double* block, unsigned int vocab);
static int token_emb_reduce(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab);
static void token_emb_backfill(libxs_lexicon_t* lexicon,
  const double* rowcnt, unsigned int vocab);
static void ngram_syllable_probe(void);
static void token_emb_probe(libxs_lexicon_t* lexicon, unsigned int vocab);
static int ngram_history(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, const char* text, int text_len, unsigned int hist[]);
static int ngram_render_separated(void);
static int corpus_profile_for_path(const char* path);
static int corpus_md_sentences(void);
static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, size_t offset, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int code_like);
static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, size_t offset, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int code_like);
static int corpus_ingest_markdown(libxs_registry_t* corpus,
  const unsigned char* text, size_t text_size, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);


/**
 * Load-time layout check, and the reason the fixup exists at all beyond forcing
 * the registry to COPY its values.
 *
 * A stored record occupies exactly what its KIND says it does, so that identity is
 * a stamp on the layout that wrote it. Any change to a fixed field - the token
 * arrays are 288 of those bytes and now tunable - moves the metadata size, and a
 * corpus written by the other layout would be read as garbage: `text_len` still
 * parses whatever bytes land under it, which is exactly what makes the misreading
 * silent. The kind tag catches what the size alone cannot, namely a layout that
 * merely PERMUTED its fields and so kept every record the same length. Counting
 * mismatches lets the caller discard the file and re-ingest instead.
 */
static void corpus_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  const corpus_entry_t* entry = (const corpus_entry_t*)value;
  if (NULL != entry && NULL != udata) {
    size_t want = 0;
    switch (corpus_value_kind(value)) {
      case ENTRY_KIND_FULL: {
        want = CORPUS_ENTRY_META_SIZE
          + ((0 < entry->text_len) ? (size_t)entry->text_len : 0) + 1;
      } break;
      case ENTRY_KIND_SPAN: {
        want = sizeof(corpus_span_t);
        ++corpus_span_stored;
      } break;
      default: want = 0;
    }
    if (value_size != want) ++(*(size_t*)udata);
  }
}


/**
 * The same check for the parent file, which also recovers the id counter. A parent
 * whose id is unknown would be handed out twice and two spans would then disagree
 * about what their text is.
 */
static void corpus_parent_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  const corpus_blob_t* blob = (const corpus_blob_t*)value;
  if (NULL != blob && NULL != udata) {
    size_t want = 0;
    if (ENTRY_KIND_BLOB == corpus_value_kind(value)) {
      want = CORPUS_BLOB_META_SIZE
        + ((0 < blob->text_len) ? (size_t)blob->text_len : 0) + 1;
      if (value_size == want && 12 == key_size && NULL != key) {
        unsigned int id = 0;
        memcpy(&id, (const char*)key + 4, 4);
        if (corpus_blob_max < id) corpus_blob_max = id;
      }
    }
    if (value_size != want) ++(*(size_t*)udata);
  }
}


const answer_predict_profile_t* answer_predict_profile_default(void)
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


static void answer_relation_rules_free(void)
{
  free(answer_relation_rules);
  answer_relation_rules = NULL;
  answer_relation_rules_size = 0;
  free(answer_lexnorms);
  answer_lexnorms = NULL;
  answer_lexnorms_size = 0;
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
    else if (0 == strcmp(text, "negate")) result = RELATION_RULE_NEGATE;
    else if (0 == strcmp(text, "norm")) result = RELATION_RULE_NORM;
    /**
     * `caps|nouns` ASSERTS that this language capitalizes every noun, which is a
     * fact about the orthography and so cannot be derived from the corpus: a
     * corpus cannot tell you whether it is written in such a language, only what
     * its capitals do. With it, the same case census that identifies NAMES in
     * English identifies NOUNS instead - the signal is unchanged, the question
     * the orthography answers is not.
     */
    else if (0 == strcmp(text, "caps")) result = RELATION_RULE_CAPS;
    /**
     * `where|in`, `why|because`, `how|by` are the MARKERS that make a sentence
     * answer that question, and they were English string literals in this file -
     * the one thing the rule layer exists to prevent. They are function words, so
     * they are language facts and not corpus facts, which is why they belong in the
     * shared language file rather than in a per-corpus one. A corpus whose rules
     * REPLACE that file (a corpus in another language) must state its own, and
     * until it does its entries carry no such flag: absent rather than English.
     *
     * `place|forest` is different in kind - a place NOUN, i.e. a class seed of
     * exactly the shape `person|term` has, which the class learner then grows.
     */
    else if (0 == strcmp(text, "where")) result = RELATION_RULE_WHERE;
    else if (0 == strcmp(text, "why")) result = RELATION_RULE_WHY;
    else if (0 == strcmp(text, "how")) result = RELATION_RULE_HOW;
    else if (0 == strcmp(text, "place")) result = RELATION_RULE_PLACE;
    /**
     * `topic|about` declares what introduces the SUBJECT of a question rather
     * than a fact about it: "what do we know about X" asks for everything the
     * corpus states of X, and the marker is what says which token X is.
     */
    else if (0 == strcmp(text, "topic")) result = RELATION_RULE_TOPIC;
    /**
     * `copula|is` declares the verb that states what something IS, which is the
     * one relation shape encyclopaedic prose uses for nearly every definition
     * ("X is a Y", "X, a Y, ..."). It is also the word an appositive omits, so a
     * reply rendering one takes it from here rather than from a literal in the C:
     * the FIRST declared copula is the one used, so the language file picks it.
     */
    else if (0 == strcmp(text, "copula")) result = RELATION_RULE_COPULA;
    /**
     * `article|a` is a closed class of its own and not a subset of `skip`. The
     * type extractor first used the skip class to recognize "X is A Y", and skip
     * declares every function word - "and", "would", "he" - so it admitted
     * "Hansel, and thrust into his pockets ..." as an apposition. An article is
     * what makes the phrase after a copula or a comma a NOUN phrase, which is
     * exactly the distinction being drawn.
     */
    else if (0 == strcmp(text, "article")) result = RELATION_RULE_ARTICLE;
    /**
     * `prep|of` is the syntactic class, distinct from the `where`/`why`/`how`
     * markers, which say what a sentence ANSWERS rather than how it is built. A
     * name inside a prepositional phrase is not the subject of the clause, and
     * that single fact is what separates "Aristotle is a Greek philosopher" from
     * "a bust OF Aristotle is a nearly ubiquitous ornament".
     */
    else if (0 == strcmp(text, "prep")) result = RELATION_RULE_PREP;
    /**
     * `own|belongs` is both the marker that recognizes a possession question and
     * the verb a reply to one is rendered with, so an enumerated answer states
     * "A, B and C belong to X" in the language's own words rather than in a
     * literal here. The FIRST declared term is the singular and the second the
     * plural, which is the only ordering this needs.
     */
    else if (0 == strcmp(text, "own")) result = RELATION_RULE_OWN;
    /**
     * `poss|apostrophe-s` names the ORTHOGRAPHY of possession, which differs by
     * language and cannot be derived from a corpus - the same kind of assertion
     * `caps|nouns` is, and stated the same way: the term names a shape rather than
     * a word, so no vocabulary enters the C. English marks it with an apostrophe
     * and an s ("Hansel's finger") and with a bare apostrophe after an s-final
     * name ("Jones' car"); German marks it with a BARE s and no apostrophe at all
     * ("Muellers Muehle"), taking the apostrophe alone when the name already ends
     * in an s sound ("Ines' Tasche"). Declaring the wrong shape costs nothing but
     * silence, because the name census still has to recognize what remains once
     * the mark is removed.
     */
    else if (0 == strcmp(text, "poss")) result = RELATION_RULE_POSS;
    /**
     * `aux|had` declares the AUXILIARIES, and they are declared for one reason: the
     * word an auxiliary governs is a VERB. That frame is the cheapest verb detector
     * a corpus offers and it needs no morphology, so the class of verbs can be
     * DERIVED from the corpus instead of listed here. It is deliberately used only
     * to REJECT - see answer_verbs_build - because the frame is incomplete by
     * construction.
     */
    else if (0 == strcmp(text, "aux")) result = RELATION_RULE_AUX;
    /**
     * `agent|by` is the word a PASSIVE names its agent with, and it is one word per
     * language rather than a class: English "by", German "von". Declaring it apart
     * from the prepositions is what lets the passive shape be recognized at all -
     * "was visited by Odysseus" is an edge, "was visited in Athens" is not.
     */
    else if (0 == strcmp(text, "agent")) result = RELATION_RULE_AGENT;
    /**
     * `link|connected` marks a question about the KNOWLEDGE GRAPH - how two entities
     * relate - which is a different question from any single shape's: it is answered
     * by a PATH through facts rather than by one of them.
     */
    else if (0 == strcmp(text, "link")) result = RELATION_RULE_LINK;
    /**
     * `genitive|of` is the word that marks a possessor AFTER the thing possessed, and
     * it is one word per language like the agent marker - English "of", German "von".
     * Declared apart from the prepositions for the same reason `agent|by` is: "the
     * father of Theseus" relates two entities and "the father in Athens" does not, and
     * only this one preposition of the thirty tells them apart. The `poss|` class
     * cannot hold it, since those terms name a SHAPE ("apostrophe-s") rather than a
     * word of the corpus.
     */
    else if (0 == strcmp(text, "genitive")) result = RELATION_RULE_GENITIVE;
    /**
     * `join|ampersand` names an ORTHOGRAPHIC joiner, the way `poss|apostrophe-s` names
     * an orthographic possessive: the term names a SHAPE so no vocabulary enters the C.
     * It is read in ONE place only, inside an article-headed phrase, and the reason is
     * measured rather than chosen - see E17.
     */
    else if (0 == strcmp(text, "join")) result = RELATION_RULE_JOIN;
    /**
     * `ask|who|who` declares a QUESTION WORD: the first field is the question KIND,
     * which is a tag this code knows the way it knows "apostrophe-s", and the second
     * is the word the language uses for it. German writes `ask|who|wer`.
     *
     * It exists because the query classifier was the LAST place English survived in
     * the C - it tested for "who", "what", "where" as literals while `where|`, `why|`
     * and `how|` sat declared in the rule file two screens away. Those declare the
     * markers that make a SENTENCE answer a question; this declares the words that make
     * a QUERY ask one, which is the other half and was never written down.
     */
    else if (0 == strcmp(text, "ask")) result = RELATION_RULE_ASK;
    /**
     * `pron|it` declares the BACK-REFERENCE pronouns, the words a follow-up uses to
     * point at what was just discussed. The multi-turn rewrite substitutes the
     * remembered topic for one, so which words those are is a fact about the language
     * and belongs here - the last eight English literals the C still held.
     *
     * Only the third-person back-references belong: a first- or second-person pronoun
     * refers to a participant in the conversation and never to the topic, so
     * substituting one would answer about the wrong thing.
     */
    else if (0 == strcmp(text, "pron")) result = RELATION_RULE_PRON;
    /**
     * `result|made` is the light verb that introduces a RESULTING STATE - "Hansel is
     * to be made fat" - and the shape reads whatever word follows it as the predicate.
     * One word per language, declared for the same reason `agent|by` is, and it was the
     * last English literal left in a fact layer.
     *
     * Declaring it does not make the shape trustworthy: only 2 of its 16 facts on the
     * tales are true, the rest idioms ("made use of", "made her way"), which is why a
     * reply built on it is LABELLED rather than trusted. What declaring buys is that a
     * corpus in another language can silence it, or name its own word.
     */
    else if (0 == strcmp(text, "result")) result = RELATION_RULE_RESULT;
  }
  return result;
}


static int answer_relation_rule_append(int kind, const char* relation,
  const char* term, int provenance)
{
  int result = EXIT_FAILURE;
  answer_relation_rule_t* rules;
  if (kind > 0 && NULL != term && '\0' != term[0]
    && strlen(term) < sizeof(answer_relation_rules[0].term)
    && ((RELATION_RULE_ALIAS != kind && RELATION_RULE_NORM != kind
        && RELATION_RULE_ASK != kind)
      || (NULL != relation && '\0' != relation[0]
        && strlen(relation) < sizeof(answer_relation_rules[0].relation))))
  {
    rules = (answer_relation_rule_t*)realloc(answer_relation_rules,
      (answer_relation_rules_size + 1) * sizeof(*rules));
    if (NULL != rules) {
      answer_relation_rules = rules;
      memset(answer_relation_rules + answer_relation_rules_size, 0,
        sizeof(*answer_relation_rules));
      answer_relation_rules[answer_relation_rules_size].kind = kind;
      answer_relation_rules[answer_relation_rules_size].provenance =
        provenance;
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
    if ((RELATION_RULE_ALIAS == kind || RELATION_RULE_NORM == kind
      || RELATION_RULE_ASK == kind) && NULL != fields[2])
    {
      result = answer_relation_rule_append(kind, fields[1], fields[2],
        RELATION_RULE_ASSERTED);
    }
    else if (RELATION_RULE_ALIAS != kind && RELATION_RULE_NORM != kind
      && RELATION_RULE_ASK != kind)
    {
      result = answer_relation_rule_append(kind, NULL, fields[1],
        RELATION_RULE_ASSERTED);
    }
  }
  return result;
}


static size_t answer_relation_rules_load_file(const char* path)
{
  size_t result = 0;
  FILE* file;
  if (NULL == path) return 0;
  /* Appends: successive files layer (language rules, then per-corpus rules). */
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
  answer_lexnorms_build();
  return result;
}


static void answer_relation_rules_report(FILE* stream)
{
  if (NULL != stream) {
    fprintf(stream, "relation rules: %lu loaded\n",
      (unsigned long)answer_relation_rules_size);
  }
}

/**
 * Score named successions with the directed embedding: CONVERSE_EMB_PAIRS="a>b,c>d".
 *
 * Aimed at the one failure the paper calls unsolvable with counts. `be made fat`
 * (true) and `be made in` (false) each occur EXACTLY ONCE, so relative
 * attestation has nothing to compare and the seam gate cannot separate them.
 * Sparse PMI was already retired here for a related reason: with zero observed
 * co-occurrence both sides collapse onto the smoothing floor. A LOW-RANK
 * COMPLETION is a different instrument - it interpolates from the whole matrix,
 * so it assigns a graded score to a pair it never saw, which is the only way a
 * continuous signal can exist where the counts are flat. Whether that score
 * ORDERS good before bad is the question; this prints it rather than assuming it.
 */
static void token_emb_pair_probe(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  const char* spec = getenv("CONVERSE_EMB_PAIRS");
  if (NULL != spec && '\0' != *spec && NULL != lexicon) {
    const unsigned int vocab = libxs_lexicon_size(lexicon);
    const char* at = spec;
    if (0 == token_emb_ready()) {
      token_emb_build(corpus, lexicon, rules, nrules, 0);
    }
    fprintf(stderr, "successor pairs (directed=%d temp=%.2f):\n",
      token_emb_directed(), ngram_emb_temp());
    while ('\0' != *at) {
      char lhs[64], rhs[64];
      int ln = 0, rn = 0;
      while ('\0' != *at && '>' != *at && ',' != *at) {
        if (ln + 1 < (int)sizeof(lhs)) lhs[ln++] = *at;
        ++at;
      }
      if ('>' == *at) ++at;
      while ('\0' != *at && ',' != *at) {
        if (rn + 1 < (int)sizeof(rhs)) rhs[rn++] = *at;
        ++at;
      }
      if (',' == *at) ++at;
      lhs[ln] = '\0';
      rhs[rn] = '\0';
      if (0 < ln && 0 < rn) {
        const unsigned int a = libxs_lexicon_id(lexicon, lhs, ln, 0, 0);
        const unsigned int b = libxs_lexicon_id(lexicon, rhs, rn, 0, 0);
        if (0 != a && 0 != b) {
          const double p = token_emb_succ_prob(&a, 1, b, vocab,
            ngram_emb_temp());
          const int rank = token_emb_succ_rank(&a, 1, b, vocab);
          fprintf(stderr, "  %-10s > %-12s p=%.3e rank=%d of %u\n",
            lhs, rhs, p, rank, vocab);
        }
        else {
          fprintf(stderr, "  %-10s > %-12s (not in vocabulary)\n", lhs, rhs);
        }
      }
    }
  }
}


static int answer_rules_learn_count(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RULES_LEARN");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static double answer_rules_learn_env(const char* name, double fallback)
{
  const char* env = getenv(name);
  double result = fallback;
  if (NULL != env && '\0' != *env) result = atof(env);
  return result;
}


static void answer_rules_centroid(double* dst, const double* sum)
{
  double norm = 0.0;
  int d;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) norm += sum[d] * sum[d];
  norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) dst[d] = sum[d] * norm;
}


static double answer_rules_cosine(const double* lhs, const double* rhs)
{
  double result = 0.0;
  int d;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) result += lhs[d] * rhs[d];
  return result;
}


/**
 * How much a word looks like the class described by the two centroids. A member
 * must look like one in BOTH directions: followed by what persons are followed
 * by, AND preceded by what persons are preceded by. Scoring the weaker side is
 * what excludes a function word - one can take person-like continuations while
 * nothing puts a determiner in front of it, and nsrc cannot see that because a
 * function word occurs in EVERY source (it held the highest nsrc of any
 * candidate, 58).
 */
static double answer_rules_score(const double* cfwd, const double* cbwd,
  unsigned int id)
{
  const double* e = token_emb_get(id);
  const double* v = token_semb_get(id);
  double cf = 0.0, cp = 0.0, vnorm = 0.0;
  int d;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) {
    cf += cfwd[d] * e[d];
    cp += cbwd[d] * v[d];
    vnorm += v[d] * v[d];
  }
  vnorm = (vnorm > 0.0) ? (1.0 / sqrt(vnorm)) : 0.0;
  cp *= vnorm;
  return (cf < cp) ? cf : cp;
}


/**
 * Add one member to the two class accumulators. The forward rows are already
 * unit-length (they are the row-normalized projection), but token_semb is raw V,
 * so without normalizing here the member with the largest magnitude would set the
 * successor direction on its own - and magnitude in V is not membership.
 */
static void answer_rules_member(double* csum, double* ssum, unsigned int id)
{
  const double* e = token_emb_get(id);
  const double* v = token_semb_get(id);
  double norm = 0.0;
  int d;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) norm += v[d] * v[d];
  norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) {
    csum[d] += e[d];
    ssum[d] += v[d] * norm;
  }
}


/**
 * Probability of at most k successes in n draws at rate p, by the term ratio so
 * that no factorial is ever formed.
 */
static double answer_rules_tail(long k, long n, double p)
{
  double result = 0.0;
  if (0 < n && 0.0 < p && p < 1.0 && 0 <= k) {
    const double odds = p / (1.0 - p);
    double term = pow(1.0 - p, (double)n);
    long i;
    result = term;
    for (i = 1; i <= k && i <= n; ++i) {
      term *= odds * (double)(n - i + 1) / (double)i;
      result += term;
    }
  }
  return result;
}


/**
 * Non-zero if a candidate shows a behaviour IMPLAUSIBLY far below the weakest
 * rate any ASSERTED member shows - k of n where the seeds manage pmin.
 *
 * This is not a cut on the rate. A rate ignores how much evidence stands behind
 * it, so a hapax with an unlucky context looks exactly like a word that truly
 * never behaves this way; the tail keeps them apart, and it keeps SILENT where
 * there is too little evidence to speak. Everything it compares against is
 * derived from what is asserted, so the only constant here is the one below.
 */
static int answer_rules_implausible(long k, long n, double pmin)
{
  return (0.0 < pmin && pmin < 1.0 && 0 < n
    && answer_rules_tail(k, n, pmin) < ANSWER_RULES_ALPHA) ? 1 : 0;
}


/**
 * The highest rate at which this member's counts would still NOT look
 * implausible - the point where the very test the candidates face would begin
 * to flag the member itself.
 *
 * This is what an asserted member is ENTITLED TO CLAIM about the class, and it
 * is the third reading of the same tail. A member seen ten times is consistent
 * with a far higher rate than it happens to show, so it cannot pull the bar down
 * to what is really sampling noise; a member seen two hundred times pins the bar
 * close to what it actually shows. Taking the minimum of the OBSERVED rates
 * instead lets whichever member has the least evidence speak loudest, which is
 * the same mistake as taking the widest separation for a section boundary.
 */
static double answer_rules_ceiling(long k, long n)
{
  double lo = 0.0, hi = 1.0;
  int i;
  for (i = 0; i < 40 && 0 < n; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (answer_rules_tail(k, n, mid) < ANSWER_RULES_ALPHA) hi = mid;
    else lo = mid;
  }
  return lo;
}


/**
 * The three structural tests, all the same shape and all read against what the
 * ASSERTED members manage. `pmin` holds their weakest rate for each, in the
 * order the test index uses.
 *
 *  0 HEADS   - a word that MODIFIES where a member HEADS is an attribute of one,
 *               not one of them.
 *  1 INTRODUCED - one of the words that actually INTRODUCE a member introduces
 *               it. A verb passes the wider "some function word precedes it"
 *               reading easily, because a pronoun is a function word too.
 */
static int answer_rules_excluded(int test, unsigned int id, const long* freq,
  const long* nhead, const long* nmod, const long* nintro,
  const double* pmin)
{
  int result;
  if (0 == test) {
    result = answer_rules_implausible(nhead[id], nhead[id] + nmod[id], pmin[0]);
  }
  else {
    result = answer_rules_implausible(nintro[id], freq[id], pmin[1]);
  }
  return result;
}


/**
 * Count, for every word, how often one of the LEARNED INTRODUCERS precedes it.
 *
 * The set cannot be known before the corpus has been read once - it is the
 * function words that actually stand in front of an asserted member - so this
 * is a second pass rather than another counter in the first one.
 */
static void answer_rules_count_intro(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const unsigned char* isintro, long* nintro, unsigned int vocab)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    libxs_lexeme_stream_t stream;
    libxs_lexeme_stream_init(&stream);
    if (SCALE_SENTENCE == entry->scale && entry->text_len > 0
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT)
      && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
        (const unsigned char*)entry->text, (size_t)entry->text_len,
        rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
    {
      size_t pos;
      for (pos = 1; pos < stream.size; ++pos) {
        const libxs_lexeme_t* lex = stream.data + pos;
        const libxs_lexeme_t* prev = stream.data + pos - 1;
        if (0 != (lex->flags & LIBXS_LEXEME_WORD) && 0 != lex->id
          && lex->id <= vocab && 0 != prev->id && prev->id <= vocab
          && 0 != isintro[prev->id])
        {
          ++nintro[lex->id];
        }
      }
    }
    libxs_lexeme_stream_release(&stream);
    value = corpus_iter_next(corpus, &key, &cursor);
  }
}


/**
 * Propose person-class terms from the corpus instead of requiring every one to
 * be configured by hand.
 *
 * WHY THIS IS THE PIECE THAT WAS MISSING. The fact layer is the one path in this
 * system that produces unseen, grammatical, attributed sentences without
 * diverging, because what it renders is a PROPOSITION rather than a token path -
 * there is nowhere for it to divert to. But its reach is set by `person|...`
 * rules written by hand, so the capability was a demonstration and not learning.
 * Every consumer already goes through answer_relation_rule_has_term, so widening
 * the class here widens facts, replies and citations at once.
 *
 * The geometry is the directed successor embedding: two words sit together when
 * they are FOLLOWED by similar things, and a class like "person" is exactly a set
 * of words that take the same continuations ("the girl said", "the boy said").
 * That is also why this is not circular with the seed list - the similarity is
 * computed from corpus succession, never from the rule file.
 *
 * SIDE-SIGNALS decide acceptance, not similarity alone:
 *  - nsrc, the number of distinct sections the word occurs in. This separates a
 *    CLASS member from a CHARACTER: a class word recurs across sources, whereas
 *    a character lives in one. A term with a single source is a name wearing a
 *    class's clothes.
 *  - freq, so a hapax cannot enter the class on one lucky neighbourhood.
 *  - two thresholds. At or above accept the term becomes a rule; between
 *    speculate and accept it is REPORTED AND NOT USED, so a reply never rests on
 *    a guess without that having been a decision. Unlocking those is a separate
 *    act, which is what makes "speculation" a mode rather than an accident.
 *
 * REFINEMENT (CONVERSE_RULES_REFINE) folds each ACCEPTED term back into the
 * centroid, so the class is described by its members and not by its seeds alone.
 * Only the accepted band feeds back: a speculative term is reported and never
 * becomes part of what the next round compares against, which is what turns that
 * band from a label into a guard. Mode 1 scores against the grown centroid alone
 * and mode 2 against the weaker of the grown and the seed one; 0 is seeds only.
 * Drift - the cosine between the current centroid and the seed centroid - is
 * REPORTED per round and gates nothing, because no measurement yet says what a
 * poisoned round looks like.
 */
static size_t answer_relation_rules_learn(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  enum { LEARN_SEED_MAX = 64 };
  size_t result = 0;
  size_t nspecul = 0;
  const int want = answer_rules_learn_count();
  const unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  if (0 < want && 0 < vocab && NULL != corpus) {
    const double accept = answer_rules_learn_env("CONVERSE_RULES_ACCEPT", 0.55);
    const double specul = answer_rules_learn_env("CONVERSE_RULES_SPECULATE",
      0.40);
    const int minsrc = (int)answer_rules_learn_env("CONVERSE_RULES_MINSRC", 3.0);
    const int minfreq = (int)answer_rules_learn_env("CONVERSE_RULES_MINFREQ",
      5.0);
    const int refine = (int)answer_rules_learn_env("CONVERSE_RULES_REFINE", 0.0);
    const int ceiling = (int)answer_rules_learn_env("CONVERSE_RULES_CEILING",
      0.0);
    double* csum = (double*)calloc(TOKEN_EMB_DIM, sizeof(double));
    double* ssum = (double*)calloc(TOKEN_EMB_DIM, sizeof(double));
    double* cwork = (double*)calloc(4 * TOKEN_EMB_DIM, sizeof(double));
    long* freq = (long*)calloc((size_t)vocab + 1, sizeof(long));
    /**
     * How often each word is the HEAD of its phrase rather than a MODIFIER in
     * it. The distributional score cannot tell those apart - an adjective is
     * preceded and followed by what the noun it modifies is preceded and
     * followed by - so this is a positional count, not a similarity: a head is
     * followed by a boundary or a function word, a modifier by another content
     * word. Measured here before anything is allowed to depend on it.
     */
    long* nhead = (long*)calloc((size_t)vocab + 1, sizeof(long));
    long* nmod = (long*)calloc((size_t)vocab + 1, sizeof(long));
    /**
     * The mirror count: how often the word is preceded by a FUNCTION word. A
     * noun and the adjective before it are both introduced by a determiner,
     * whereas an interjection stands on its own - so the two positional counts
     * fail on different things, which is the whole reason for measuring both
     * before either is allowed to gate anything.
     */
    /**
     * WHICH sources a word occurs in, as a saturating bitmap of hashed section
     * identity rather than an exact set.
     *
     * The exact set needed a table of section names, and a table has a size: at
     * 64 it silently tracked only the first 64 sections a corpus offered, so on a
     * 473-section corpus `nsrc` was not merely capped, it was BIASED toward words
     * appearing early. It also cost a linear scan of that table per entry.
     *
     * A bitmap answers the only question asked of it - "at least minsrc distinct
     * sources" - in bounded memory and with no positional bias. Two collisions
     * can hide one source, so it UNDERCOUNTS a word occurring in many sections;
     * that direction is safe, because the test admits on a lower bound.
     */
    unsigned int* srcmask = (unsigned int*)calloc(2 * ((size_t)vocab + 1),
      sizeof(unsigned int));
    libxs_hist_t* hist_seed = libxs_hist_create(10, 2, NULL, NULL, NULL);
    libxs_hist_t* hist_cand = libxs_hist_create(10, 2, NULL, NULL, NULL);
    unsigned char* pushed = (unsigned char*)calloc((size_t)vocab + 1, 1);
    /**
     * The three arrays behind the LEARNED INTRODUCER test. `isseed` marks the
     * asserted members so the first pass can record what stands in front of one;
     * those predecessors become `isintro`, and `nintro` then counts how often any
     * word is introduced by one of them. This is narrower than "preceded by a
     * function word", which a verb passes easily because a pronoun is one.
     */
    unsigned char* isseed = (unsigned char*)calloc((size_t)vocab + 1, 1);
    unsigned char* isintro = (unsigned char*)calloc((size_t)vocab + 1, 1);
    /**
     * The NOUN test, live only where the language file asserts `caps|nouns`.
     * Where an orthography capitalizes every noun, "never lower-case AND
     * capitalized where position did not force it" identifies nouns at nearly
     * perfect precision - the same rule that identifies NAMES in English, whose
     * orthography reserves capitals for them instead. It is asserted rather than
     * measured because no corpus can say which language it is written in.
     */
    const int caps = (0 != answer_relation_rule_has_term(RELATION_RULE_CAPS,
      "nouns", 5)) ? 1 : 0;
    long* ncap = (long*)calloc((size_t)vocab + 1, sizeof(long));
    long* nupper = (long*)calloc((size_t)vocab + 1, sizeof(long));
    long* nfree = (long*)calloc((size_t)vocab + 1, sizeof(long));
    long* nprec = (long*)calloc((size_t)vocab + 1, sizeof(long));
    long* nintro = (long*)calloc((size_t)vocab + 1, sizeof(long));
    if (NULL != csum && NULL != ssum && NULL != cwork && NULL != freq
      && NULL != srcmask && NULL != nhead && NULL != nmod
      && NULL != pushed && NULL != isseed && NULL != isintro
      && NULL != nprec && NULL != nintro && NULL != ncap && NULL != nupper
      && NULL != nfree)
    {
      double* centroid = cwork;
      double* scentroid = cwork + TOKEN_EMB_DIM;
      double* cseed = cwork + 2 * TOKEN_EMB_DIM;
      double* sseed = cwork + 3 * TOKEN_EMB_DIM;
      const char* seedterm[LEARN_SEED_MAX];
      long seedk[LEARN_SEED_MAX][2], seedn[LEARN_SEED_MAX][2];
      double pmin[2];
      unsigned int seenmask[2];
      int nseed = 0, shown = 0, exhausted = 0, nbucket = 0;
      const void* key = NULL;
      size_t cursor = 0;
      unsigned int id;
      void* value;
      /* The learner needs the embedding, so it says so rather than depending on
       * a prediction kind having been asked for. */
      if (0 == token_emb_ready()) {
        token_emb_build(corpus, lexicon, rules, nrules, 0);
      }
      seenmask[0] = 0;
      seenmask[1] = 0;
      { size_t rule_pos;
        for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
          const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
          if (RELATION_RULE_PERSON == rule->kind
            && RELATION_RULE_ASSERTED == rule->provenance)
          {
            const unsigned int sid = libxs_lexicon_id(lexicon, rule->term,
              (int)strlen(rule->term), 0, 0);
            if (0 != sid && sid <= vocab) isseed[sid] = 1;
          }
        }
      }
      value = corpus_iter_begin(corpus, &key, &cursor);
      while (NULL != value) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        libxs_lexeme_stream_t stream;
        libxs_lexeme_stream_init(&stream);
        /**
         * Sentence scale only, and no fragments: the corpus holds each text at
         * sentence AND paragraph scale, so counting both would double every
         * frequency, and a long sentence is stored again as OVERLAPPING clause
         * fragments whose bytes the sentence entry already covers. Those weight a
         * word by how many fragments happen to span it, which is heaviest at a
         * clause boundary - exactly where the positional counts below are read.
         */
        if (SCALE_SENTENCE == entry->scale && entry->text_len > 0
          && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT)
          && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
            (const unsigned char*)entry->text, (size_t)entry->text_len,
            rules, nrules, answer_lexnorms, answer_lexnorms_size, 0))
        {
          int at = -1;
          size_t pos;
          if (0 < entry->section_len) {
            const unsigned int h = libxs_hash(entry->section,
              (size_t)entry->section_len, 0);
            at = (int)(h & 63);
          }
          /**
           * The surface pass. It is the only pass that must see the written form
           * rather than the interned one, so it reads the entry text directly.
           */
          if (0 != caps) {
            static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
            const int heading_len = corpus_title_len(entry->text,
              entry->text_len);
            const char* token;
            int token_index = 0, token_len = 0;
            while (NULL != (token = libxs_strtoken(entry->text, delims,
              token_index, &token_len)))
            {
              int trimmed = 0, end = token_len;
              while (end > trimmed
                && 0 == corpus_word_byte(token, trimmed, token_len)) ++trimmed;
              while (end > trimmed
                && 0 == corpus_word_byte(token, end - 1, token_len)) --end;
              if (end > trimmed) {
                const unsigned int wid = corpus_word_id(lexicon,
                  token + trimmed, end - trimmed);
                if (0 != wid && wid <= vocab) {
                  ++ncap[wid];
                  if (0 != corpus_word_upper(token, trimmed, end)) {
                    ++nupper[wid];
                    if (0 == corpus_case_forced(entry->text,
                      (int)(token - entry->text) + trimmed, heading_len))
                    {
                      ++nfree[wid];
                    }
                  }
                }
              }
              ++token_index;
            }
          }
          for (pos = 0; pos < stream.size; ++pos) {
            const libxs_lexeme_t* lex = stream.data + pos;
            if (0 != (lex->flags & LIBXS_LEXEME_WORD) && 0 != lex->id
              && lex->id <= vocab)
            {
              ++freq[lex->id];
              if (0 <= at) {
                srcmask[2 * (size_t)lex->id + (at >> 5)] |= 1u << (at & 31);
                seenmask[at >> 5] |= 1u << (at & 31);
              }
              if (0 < pos) {
                const libxs_lexeme_t* prev = stream.data + pos - 1;
                /**
                 * What stands in front of an ASSERTED member is what an
                 * introducer is - ANY word, not one the library already calls a
                 * function word. That flag is language-specific, and a test built
                 * on it went silently inert on a corpus whose language it does
                 * not cover; the learned set has its own evidence test, so the
                 * flag was never load-bearing. Measured on English, the flag test
                 * then excluded NOTHING the learned one did not already exclude,
                 * so it is gone rather than merely widened.
                 */
                if (0 != isseed[lex->id] && 0 != prev->id && prev->id <= vocab
                  && 0 != (prev->flags & LIBXS_LEXEME_WORD))
                {
                  ++nprec[prev->id];
                }
              }
              /* The last word of a sentence is a head by definition: there is
               * nothing left for it to modify. */
              if (pos + 1 < stream.size) {
                const libxs_lexeme_t* next = stream.data + pos + 1;
                if (0 != (next->flags & LIBXS_LEXEME_WORD)
                  && 0 == (next->flags & LIBXS_LEXEME_STOP))
                {
                  ++nmod[lex->id];
                }
                else ++nhead[lex->id];
              }
              else ++nhead[lex->id];
            }
          }
        }
        libxs_lexeme_stream_release(&stream);
        value = corpus_iter_next(corpus, &key, &cursor);
      }
      { size_t rule_pos;
        for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
          const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
          if (RELATION_RULE_PERSON == rule->kind) {
            const unsigned int sid = libxs_lexicon_id(lexicon, rule->term,
              (int)strlen(rule->term), 0, 0);
            if (0 != sid && sid <= vocab && 0 == token_emb_isnull(sid)) {
              answer_rules_member(csum, ssum, sid);
              ++nseed;
            }
          }
        }
      }
      if (0 < nseed) {
        answer_rules_centroid(centroid, csum);
        answer_rules_centroid(scentroid, ssum);
        memcpy(cseed, centroid, TOKEN_EMB_DIM * sizeof(double));
        memcpy(sseed, scentroid, TOKEN_EMB_DIM * sizeof(double));
        /* Section BUCKETS, not sections: nsrc counts distinct buckets, and
         * saying "sections" would overstate its resolution. corpus_sections_size
         * is per FILE and would report only the last one of a multi-file corpus. */
        { int b, w;
          for (w = 0; w < 2; ++w) {
            unsigned int bits = seenmask[w];
            for (b = 0; b < 32; ++b) nbucket += (int)((bits >> b) & 1u);
          }
        }
        fprintf(stderr, "rule learning: %d seeds, %d source buckets of 64,"
          " accept>=%.2f speculate>=%.2f minsrc=%d minfreq=%d refine=%d\n",
          nseed, nbucket, accept, specul, minsrc, minfreq, refine);
        /* The SEEDS are the known-good group: they are person nouns by
         * assertion, so their head fraction is what a member looks like. */
        /**
         * An introducer is not simply a function word that was ever seen in
         * front of a member. Taken that way the set fills up with conjunctions,
         * which stand in front of everything, and the test it feeds then passes
         * the verbs it exists to catch. A word earns the label by preceding a
         * member MORE OFTEN THAN ITS OWN FREQUENCY EXPLAINS - the same tail, now
         * read from the other end.
         */
        { long ntoken = 0, nslot = 0;
          for (id = 1; id <= vocab; ++id) {
            ntoken += freq[id];
            if (0 != isseed[id]) nslot += freq[id];
          }
          for (id = 1; id <= vocab; ++id) {
            if (0 < nprec[id] && 0 < ntoken && 0 < nslot) {
              const double base = (double)freq[id] / (double)ntoken;
              if (1.0 - answer_rules_tail(nprec[id] - 1, nslot, base) < 1e-3) {
                isintro[id] = 1;
              }
            }
          }
        }
        answer_rules_count_intro(corpus, lexicon, rules, nrules, isintro,
          nintro, vocab);
        /**
         * The null for each test is the weakest rate an asserted member shows,
         * which makes it an EXTREMUM - and an extremum is set by whichever
         * member is most unusual rather than by the population. Two ways that
         * bites were measured here: a member whose spelling collides with an
         * unrelated word in another language, and a member carrying two senses.
         * Either one sits far below the rest and drags the bar down with it, and
         * the test it feeds then excludes nothing.
         *
         * So a member must first look like the OTHERS to speak for them: each is
         * asked, against the pooled rate of the remaining members, whether its own
         * counts are implausible. That is the same tail the candidates face, and
         * it costs no new constant. An outlier is REPORTED rather than dropped
         * quietly, because a member that does not behave like the class it
         * asserts is a fact about the rule file the reader wants.
         */
        { size_t rule_pos;
          long k[2], n[2];
          int nseedstat = 0, s, nintroducer = 0;
          k[0] = 0; k[1] = 0; n[0] = 0; n[1] = 0;
          for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
            const answer_relation_rule_t* rule = answer_relation_rules
              + rule_pos;
            if (RELATION_RULE_PERSON == rule->kind
              && RELATION_RULE_ASSERTED == rule->provenance)
            {
              const unsigned int sid = libxs_lexicon_id(lexicon, rule->term,
                (int)strlen(rule->term), 0, 0);
              if (0 != sid && sid <= vocab && nseedstat < LEARN_SEED_MAX
                && 0 < (nhead[sid] + nmod[sid]) && 0 == pushed[sid])
              {
                pushed[sid] = 1;
                seedterm[nseedstat] = rule->term;
                seedk[nseedstat][0] = nhead[sid];
                seedn[nseedstat][0] = nhead[sid] + nmod[sid];
                seedk[nseedstat][1] = nintro[sid];
                seedn[nseedstat][1] = freq[sid];
                k[0] += seedk[nseedstat][0];
                n[0] += seedn[nseedstat][0];
                k[1] += seedk[nseedstat][1];
                n[1] += seedn[nseedstat][1];
                ++nseedstat;
              }
            }
          }
          pmin[0] = 1.0;
          pmin[1] = 1.0;
          for (s = 0; s < nseedstat; ++s) {
            static const char* const which[] = { "", "  OUTLIER head",
              "  OUTLIER intro", "  OUTLIER head+intro" };
            double sample[2];
            int t, out = 0;
            for (t = 0; t < 2; ++t) {
              const long rest_k = k[t] - seedk[s][t];
              const long rest_n = n[t] - seedn[s][t];
              sample[t] = (0 < seedn[s][t])
                ? ((double)seedk[s][t] / (double)seedn[s][t]) : 0.0;
              /* Per test, not per member: a member may sit low on one dimension
               * and still speak for the class on the other. */
              if (0 < rest_n && 0 != answer_rules_implausible(seedk[s][t],
                seedn[s][t], (double)rest_k / (double)rest_n))
              {
                out |= (1 << t);
              }
              else {
                /**
                 * MEASURED AND NOT ADOPTED (CONVERSE_RULES_CEILING=1 restores it).
                 * The ceiling fixes the stated defect - a member with ten
                 * observations can no longer drag the bar down to what is really
                 * sampling noise - but it OVERSHOOTS for a well-evidenced member
                 * too, raising the bar above rates that members demonstrably do
                 * show. On the two corpora the two effects cancel: German gains
                 * one class noun of 25, English loses one error of 40. A minimum
                 * over observed rates is biased low and a minimum over ceilings
                 * is biased high; neither is the estimator this wants.
                 */
                const double rate = (0 != ceiling)
                  ? answer_rules_ceiling(seedk[s][t], seedn[s][t]) : sample[t];
                if (rate < pmin[t]) pmin[t] = rate;
              }
            }
            libxs_hist_push(NULL, hist_seed, sample);
            fprintf(stderr, "  seed %-12s head=%.3f/%.3f intro=%.3f/%.3f"
              " of %ld%s\n", seedterm[s], sample[0],
              answer_rules_ceiling(seedk[s][0], seedn[s][0]), sample[1],
              answer_rules_ceiling(seedk[s][1], seedn[s][1]), seedn[s][0],
              which[out]);
          }
          for (id = 1; id <= vocab; ++id) nintroducer += isintro[id];
          fprintf(stderr, "  weakest asserted: head=%.3f intro=%.3f"
            " from %d introducers\n", pmin[0], pmin[1], nintroducer);
          if (0 != caps) {
            int nnoun = 0;
            for (id = 1; id <= vocab; ++id) {
              if (freq[id] >= minfreq && nupper[id] == ncap[id]
                && 0 != nfree[id]) ++nnoun;
            }
            fprintf(stderr, "  caps|nouns asserted: %d of the vocabulary are"
              " nouns by orthography\n", nnoun);
          }
        }
        if (0.0 == answer_rules_learn_env("CONVERSE_RULES_INTRO", 1.0)) {
          pmin[1] = 0.0;
        }
        if (0.0 == answer_rules_learn_env("CONVERSE_RULES_HEAD", 1.0)) {
          pmin[0] = 0.0;
        }
        { int test;
          static const char* const label[] = { "modifies rather than heads",
            "not introduced as a member is" };
          for (test = 0; test < 2; ++test) {
            int nout = 0;
            for (id = 1; id <= vocab; ++id) {
              if (freq[id] >= minfreq
                && 0 != answer_rules_excluded(test, id, freq, nhead, nmod,
                  nintro, pmin))
              {
                int textlen = 0;
                const char* text = libxs_lexicon_text(lexicon, id, &textlen,
                  NULL);
                if (NULL != text && 0 < textlen) {
                  if (0 == nout) fprintf(stderr, "  %s:", label[test]);
                  if (nout < 16) {
                    fprintf(stderr, "%s %.*s", (0 != nout) ? "," : "", textlen,
                      text);
                  }
                  ++nout;
                }
              }
            }
            if (0 != nout) fprintf(stderr, " (%d words)\n", nout);
          }
        }
        while (shown < want && 0 == exhausted) {
          unsigned int best = 0;
          double bestcos = 0.0;
          int bestsrc = 0;
          for (id = 1; id <= vocab; ++id) {
            if (freq[id] >= minfreq && 0 == token_emb_isnull(id)
              && (0 == caps || (nupper[id] == ncap[id] && 0 != nfree[id]))
              && 0 == answer_rules_excluded(0, id, freq, nhead, nmod, nintro,
                pmin)
              && 0 == answer_rules_excluded(1, id, freq, nhead, nmod, nintro,
                pmin))
            {
              int textlen = 0;
              const char* text = libxs_lexicon_text(lexicon, id, &textlen, NULL);
              if (NULL != text && 0 < textlen
                && 0 == answer_relation_rule_has_term(RELATION_RULE_PERSON,
                  text, textlen))
              {
                double cos = answer_rules_score(centroid, scentroid, id);
                int nsrc = 0, s;
                /**
                 * Under REFINE=2 the term must resemble the GROWN class and the
                 * SEEDS, which is the same weaker-of-both construction applied to
                 * provenance instead of direction: the seed side is fixed, so a
                 * centroid that walks away from it cannot carry candidates along.
                 */
                if (2 == refine) {
                  const double seedcos = answer_rules_score(cseed, sseed, id);
                  if (seedcos < cos) cos = seedcos;
                }
                for (s = 0; s < 2; ++s) {
                  unsigned int bits = srcmask[2 * (size_t)id + s];
                  while (0 != bits) {
                    nsrc += (int)(bits & 1u);
                    bits >>= 1;
                  }
                }
                if (nsrc >= minsrc && (0 == best || cos > bestcos)) {
                  best = id;
                  bestcos = cos;
                  bestsrc = nsrc;
                }
              }
            }
          }
          if (0 == best || bestcos < specul) exhausted = 1;
          else {
            int textlen = 0;
            const char* text = libxs_lexicon_text(lexicon, best, &textlen, NULL);
            const int ok = (bestcos >= accept) ? 1 : 0;
            { const long total = nhead[best] + nmod[best];
              const double head = (0 < total)
                ? ((double)nhead[best] / (double)total) : 0.0;
              const double intro = (0 < freq[best])
                ? ((double)nintro[best] / (double)freq[best]) : 0.0;
              double sample[2];
              fprintf(stderr, "  %-14s min-cos=%.3f freq=%ld nsrc=%d"
                " head=%.3f intro=%.3f -> %s\n",
                (NULL != text) ? text : "?", bestcos, freq[best], bestsrc,
                head, intro, (0 != ok) ? "ACCEPTED" : "PROPOSED");
              sample[0] = head;
              sample[1] = intro;
              libxs_hist_push(NULL, hist_cand, sample);
            }
            /**
             * The margin is ADMITTED, not discarded. Its terms cannot be
             * promoted by moving the threshold - wrong terms score
             * between right ones - so the choice is to waste
             * the band or to carry the uncertainty forward. Carrying it forward
             * is only honest if it reaches the reader, which is what the
             * speculative flag is for: every reply resting on one says so.
             */
            if (NULL != text && EXIT_SUCCESS == answer_relation_rule_append(
              RELATION_RULE_PERSON, NULL, text,
              (0 != ok) ? RELATION_RULE_LEARNED : RELATION_RULE_PROPOSED))
            {
              if (0 != ok) ++result;
              else ++nspecul;
            }
            if (0 != ok && 0 != refine) {
              answer_rules_member(csum, ssum, best);
              answer_rules_centroid(centroid, csum);
              answer_rules_centroid(scentroid, ssum);
              fprintf(stderr, "    refit %d members, drift fwd=%.3f bwd=%.3f\n",
                nseed + (int)result, answer_rules_cosine(centroid, cseed),
                answer_rules_cosine(scentroid, sseed));
            }
            /* Either way it is now in the class, so stop re-proposing it. */
            freq[best] = 0;
            ++shown;
          }
        }
        /* Both counts are labelled in replies: see the provenance comment in
         * converse.h for why the accepted band is not trusted either. */
        /**
         * A cap that binds is REPORTED. `want` is a cap and never a target: E1
         * measured that precision at a fixed count cannot be improved, because
         * every term an exclusion removes is replaced by the next wrong one just
         * below it. So a run that stops at the cap has not measured the class, it
         * has measured the first N of it, and saying nothing would read as
         * "these are the terms that qualify".
         */
        fprintf(stderr, "rule learning: %lu accepted, %lu proposed"
          " (both labelled in replies)%s\n", (unsigned long)result,
          (unsigned long)nspecul, (0 == exhausted)
            ? " -- CAP REACHED, more terms clear the bar" : "");
        libxs_hist_print(stderr, hist_seed, NULL,
          "head fraction of ASSERTED members");
        libxs_hist_print(stderr, hist_cand, NULL,
          "head fraction of PROPOSED terms");
      }
    }
    libxs_hist_destroy(hist_seed);
    libxs_hist_destroy(hist_cand);
    free(csum);
    free(ssum);
    free(cwork);
    free(freq);
    free(srcmask);
    free(nhead);
    free(nmod);
    free(pushed);
    free(isseed);
    free(isintro);
    free(ncap);
    free(nupper);
    free(nfree);
    free(nprec);
    free(nintro);
  }
  return result;
}

/**
 * Materialize the `norm|from|to` rules as a libxs_lexnorm_t table, which the
 * tokenizer applies during encoding. This is the library's data-only
 * normalization facility: the mechanism is language-agnostic, the vocabulary
 * lives in the caller-owned rule file. Rebuilt whenever rules are (re)loaded;
 * with no norm rules the table is empty and encoding is bit-identical.
 */
static void answer_lexnorms_build(void)
{
  size_t rule_pos;
  free(answer_lexnorms);
  answer_lexnorms = NULL;
  answer_lexnorms_size = 0;
  if (0 == answer_relation_rules_size) return;
  answer_lexnorms = (libxs_lexnorm_t*)calloc(answer_relation_rules_size,
    sizeof(*answer_lexnorms));
  if (NULL == answer_lexnorms) return;
  for (rule_pos = 0; rule_pos < answer_relation_rules_size; ++rule_pos) {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (RELATION_RULE_NORM == rule->kind
      && strlen(rule->relation) <= LIBXS_LEXEME_MAXBYTES
      && strlen(rule->term) <= LIBXS_LEXEME_MAXBYTES)
    {
      strcpy(answer_lexnorms[answer_lexnorms_size].from, rule->relation);
      strcpy(answer_lexnorms[answer_lexnorms_size].to, rule->term);
      ++answer_lexnorms_size;
    }
  }
}


/**
 * Whether text IS a term of this kind, as opposed to CONTAINING one. The
 * containment test is for sentences; a single token has to match whole or the
 * marker "in" would fire on "into" and on every word holding those two letters.
 */
/** The first declared term of a kind, or NULL: vocabulary a renderer may use. */
const char* answer_relation_rule_first_term(int kind, int* term_len)
{
  const char* result = NULL;
  size_t rule_pos;
  if (NULL != term_len) *term_len = 0;
  for (rule_pos = 0; rule_pos < answer_relation_rules_size && NULL == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (rule->kind == kind && '\0' != rule->term[0]) {
      result = rule->term;
      if (NULL != term_len) *term_len = (int)strlen(rule->term);
    }
  }
  return result;
}


/**
 * The nth declared term of a kind, with its first field, or NULL past the end.
 *
 * Needed where a class is a MAP rather than a set: `ask|who|wer` binds a question
 * KIND to a language's word, so a reader has to walk the pairs instead of testing
 * membership.
 */
const char* answer_relation_rule_term_at(int kind, int index,
  const char** relation)
{
  const char* result = NULL;
  size_t rule_pos;
  int seen = 0;
  if (NULL != relation) *relation = NULL;
  for (rule_pos = 0; rule_pos < answer_relation_rules_size && NULL == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
    if (rule->kind == kind && '\0' != rule->term[0]) {
      if (seen == index) {
        result = rule->term;
        if (NULL != relation) *relation = rule->relation;
      }
      else ++seen;
    }
  }
  return result;
}


int answer_relation_rule_is_term(int kind, const char* text, int text_len)
{
  int result = 0;
  size_t rule_pos;
  if (NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (rule->kind == kind && 0 != libxs_striequal(rule->term,
        strlen(rule->term), text, (size_t)text_len))
      {
        result = 1;
      }
    }
  }
  return result;
}


int answer_relation_rule_has_term(int kind, const char* text,
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


/**
 * Where the class term matching this text came from: ASSERTED, LEARNED or
 * PROPOSED. Text matching no term of that kind is reported as ASSERTED, since a
 * caller asks this only about a term it already matched.
 *
 * Separate from answer_relation_rule_has_term rather than folded into it: every
 * consumer needs membership, only the ones that can end up in a reply need the
 * provenance, and returning both from one call would burden all of them. The
 * STRONGEST provenance wins - an asserted term that matches would have admitted
 * the text on its own, so a learned one alongside it changes nothing.
 */
int answer_relation_rule_provenance(int kind, const char* text, int text_len)
{
  int result = RELATION_RULE_PROPOSED + 1;
  size_t rule_pos;
  if (NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < answer_relation_rules_size
      && RELATION_RULE_ASSERTED != result; ++rule_pos)
    {
      const answer_relation_rule_t* rule = answer_relation_rules + rule_pos;
      if (rule->kind == kind && rule->provenance < result
        && 0 != text_contains_word_ci(text, text_len, rule->term))
      {
        result = rule->provenance;
      }
    }
  }
  return (RELATION_RULE_PROPOSED < result) ? RELATION_RULE_ASSERTED : result;
}

/**
 * Resolve the per-corpus namespace from the prefix: state and companion files
 * live in the current directory keyed by the prefix basename, so each corpus
 * owns its own converse.dat/eval/relations/... A prefix of "." (the default)
 * takes the working directory's own name, which reproduces the plain
 * "converse.*" layout when run from the sample directory.
 */
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
    sprintf(converse_path_parent, "%s.par", base);
    sprintf(converse_path_source, "%s.src", base);
    sprintf(converse_path_lexicon, "%s.lex", base);
    sprintf(converse_path_predict, "%s.prd", base);
    sprintf(converse_path_norm, "%s.norms", base);
    sprintf(converse_path_bridge, "%s.bridges", base);
    sprintf(converse_path_relation, "%s.relations", base);
    sprintf(converse_path_language_own, "%s.rules", base);
    sprintf(converse_path_eval, "%s.eval", base);
    sprintf(converse_path_eval_learn, "%s.learn.eval", base);
    sprintf(converse_path_predict_eval, "%s.predict", base);
    sprintf(converse_path_facts, "%s.facts", base);
  }
}


static libxs_registry_t* corpus_parents_get(void)
{
  if (NULL == corpus_parents) corpus_parents = libxs_registry_create();
  return corpus_parents;
}


/**
 * Read the parent texts a stored corpus refers to. Returns how many it holds, so a
 * corpus carrying spans with no parents to resolve can be discarded rather than
 * silently losing its windows.
 */
static size_t corpus_parents_load(void)
{
  size_t result = 0;
  FILE* f = fopen(converse_path_parent, "rb");
  if (NULL != f) {
    long len;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf) {
        if ((long)fread(buf, 1, (size_t)len, f) == len) {
          size_t stale = 0;
          libxs_registry_t* loaded = libxs_registry_load(buf, (size_t)len,
            corpus_parent_fixup, &stale);
          if (NULL != loaded) {
            libxs_registry_info_t info;
            if (0 == stale && EXIT_SUCCESS == libxs_registry_info(loaded,
              &info))
            {
              result = info.size;
            }
            if (0 < result) {
              /* The views were rebuilt FROM the parents being replaced, so they
               * no longer stand for anything this registry holds. */
              corpus_view_free();
              if (NULL != corpus_parents) {
                libxs_registry_destroy(corpus_parents);
              }
              corpus_parents = loaded;
            }
            else libxs_registry_destroy(loaded);
          }
        }
        free(buf);
      }
    }
    fclose(f);
  }
  return result;
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
          size_t stale = 0;
          corpus_span_stored = 0;
          result = libxs_registry_load(buf, (size_t)len, corpus_fixup, &stale);
          if (0 != stale && NULL != result) {
            fprintf(stderr, "corpus %s: %lu entries do not match this build's"
              " layout, discarded\n", converse_path_corpus,
              (unsigned long)stale);
            libxs_registry_destroy(result);
            result = NULL;
          }
          if (NULL != result) corpus_source_paths_load(converse_path_source);
          if (NULL != result && 0 < corpus_span_stored
            && 0 == corpus_parents_load())
          {
            fprintf(stderr, "corpus %s: %ld windows have no parent text in %s,"
              " discarded\n", converse_path_corpus, corpus_span_stored,
              converse_path_parent);
            libxs_registry_destroy(result);
            result = NULL;
          }
        }
        free(buf);
      }
    }
    fclose(f);
  }
  return result;
}


static int corpus_save_file(const libxs_registry_t* registry, const char* path)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (NULL == registry || NULL == path) return EXIT_FAILURE;
  if (EXIT_SUCCESS == libxs_registry_save(registry, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_registry_save(registry, buf, &size)) {
        FILE* f = fopen(path, "wb");
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


static int corpus_save(const libxs_registry_t* corpus)
{
  int result = corpus_save_file(corpus, converse_path_corpus);
  if (NULL != corpus_parents) {
    corpus_save_file(corpus_parents, converse_path_parent);
  }
  /* The ids inside the records are only useful with the names they stand for. */
  corpus_source_paths_save(converse_path_source);
  return result;
}


/**
 * Load the persisted vocabulary. Optional stale separates the two reasons for
 * a NULL result: no file yet, which an empty vocabulary answers, and a file the
 * library refused, which it must not, because the corpus and the ranker saved
 * beside it hold ids into the vocabulary that is gone.
 */
static libxs_lexicon_t* converse_lexicon_load(int* stale)
{
  libxs_lexicon_t* result = NULL;
  FILE* file = fopen(converse_path_lexicon, "rb");
  if (NULL != stale) *stale = 0;
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
          if (NULL == result && NULL != stale) *stale = 1;
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


int count_words(const unsigned char* text, int length)
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


size_t text_closer_size(const unsigned char* text, size_t size,
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


int text_ends_sentence(const char* text, int text_len)
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


int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id)
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


int lexeme_text_is(const libxs_lexicon_t* lexicon,
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

/**
 * Destroy word order inside each sentence while keeping the multiset of words
 * (CONVERSE_SHUFFLE=1). A language model must degrade sharply; a model that is
 * really counting unigrams and local collocations will barely notice. Applied
 * at ingest so training and evaluation see the identical transformed text.
 * The permutation is a coprime affine map of the word index - reproducible,
 * and derived from the sentence length so different sentences permute
 * differently.
 */
static int corpus_shuffle_words(unsigned char* text, int len)
{
  int begin[COMPOSE_MAXTEXT / 2], length[COMPOSE_MAXTEXT / 2];
  unsigned char copy[COMPOSE_MAXTEXT];
  int nwords = 0, pos = 0, out = 0, i;
  size_t stride;
  if (len <= 0 || len > COMPOSE_MAXTEXT) return EXIT_FAILURE;
  while (pos < len && nwords < (int)(sizeof(begin) / sizeof(*begin))) {
    int wlen = 0;
    while (pos + wlen < len && 0 == isspace(text[pos + wlen])) ++wlen;
    if (wlen > 0) {
      begin[nwords] = pos;
      length[nwords] = wlen;
      ++nwords;
    }
    pos += (wlen > 0) ? wlen : 1;
    while (pos < len && 0 != isspace(text[pos])) ++pos;
  }
  if (nwords < 2) return EXIT_FAILURE;
  stride = libxs_coprime_bias((size_t)nwords, -1.0);
  for (i = 0; i < nwords && out < len; ++i) {
    const size_t pick = LIBXS_SHUFFLE_INDEX(i, (size_t)nwords, stride, 1);
    const int w = (int)pick;
    if (out > 0 && out < len) copy[out++] = ' ';
    if (out + length[w] > len) break;
    memcpy(copy + out, text + begin[w], (size_t)length[w]);
    out += length[w];
  }
  if (out > 0) {
    memcpy(text, copy, (size_t)out);
    while (out < len) text[out++] = ' ';
  }
  return (out > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}


static int corpus_shuffle_mode(void)
{
  const char* env = getenv("CONVERSE_SHUFFLE");
  return (NULL != env && '0' != *env) ? 1 : 0;
}


int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int create)
{
  int result = EXIT_FAILURE;
  const size_t shape = (size_t)len;
  libxs_lexeme_stream_t stream;
  libxs_fprint_t full;
  unsigned char shuffled[COMPOSE_MAXTEXT];
  size_t lexeme_pos;
  libxs_lexeme_stream_init(&stream);
  if (NULL == entry || NULL == text || len <= 0) return EXIT_FAILURE;
  if (0 != corpus_shuffle_mode() && len <= COMPOSE_MAXTEXT) {
    memcpy(shuffled, text, (size_t)len);
    if (EXIT_SUCCESS == corpus_shuffle_words(shuffled, len)) text = shuffled;
  }
  memset(entry, 0, sizeof(*entry));
  if (EXIT_SUCCESS == libxs_fprint(&full, LIBXS_DATATYPE_U8,
    text, 1, &shape, NULL, FPRINT_ORDER, 0, 0, 0))
  {
    corpus_fprint_pack(&entry->fprint, &full);
    entry->kind = ENTRY_KIND_FULL;
    entry->text_len = len;
    memcpy(entry->text, text, (size_t)len);
    entry->connector = CONN_NEWLINE;
    entry->scale = scale;
    entry->source = corpus_source_id;
    result = EXIT_SUCCESS;
  }
  if (EXIT_SUCCESS == result && NULL != lexicon && NULL != rules
    && nrules > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, text, (size_t)len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, create))
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
      { int marker_len = 0;
        const char* marker = libxs_lexicon_text(lexicon, lexeme->id,
          &marker_len, NULL);
        if (NULL != marker && 0 < marker_len) {
          if (0 != answer_relation_rule_is_term(RELATION_RULE_WHERE, marker,
            marker_len)) entry->lexical_flags |= ENTRY_LEX_PLACE;
          if (0 != answer_relation_rule_is_term(RELATION_RULE_WHY, marker,
            marker_len)) entry->lexical_flags |= ENTRY_LEX_CAUSE;
          if (0 != answer_relation_rule_is_term(RELATION_RULE_HOW, marker,
            marker_len)) entry->lexical_flags |= ENTRY_LEX_METHOD;
        }
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


/**
 * UTF-8 aware helpers for the SURFACE pass, which is the only pass that reads
 * written form rather than interned form.
 *
 * `isalpha` is false for every byte of a multi-byte letter, so trimming a token's
 * non-alpha margins with it EATS a leading umlaut and leaves a stump that matches
 * nothing: an umlaut-initial noun was invisible to the noun test. Transliterating
 * the corpus (ae for a-umlaut) would also work and the `norm` rules could do it,
 * but that changes what citations and replies PRINT, and the lexeme path already
 * handles these bytes - only this pass did not.
 *
 * The Latin-1 supplement in UTF-8 is 0xC3 followed by 0x80-0xBE, upper case below
 * 0x9F and lower case above, differing by 0x20 exactly as ASCII does. That is all
 * this needs to know; nothing here decodes further.
 */
static int corpus_word_byte(const char* text, int at, int len)
{
  const unsigned char c = (unsigned char)text[at];
  LIBXS_UNUSED(len);
  return (0 != isalpha(c) || 0x80 <= c) ? 1 : 0;
}


static int corpus_word_upper(const char* text, int at, int len)
{
  const unsigned char c = (unsigned char)text[at];
  int result = 0;
  if (0xc3 == c && at + 1 < len) {
    const unsigned char d = (unsigned char)text[at + 1];
    result = (0x80 <= d && d <= 0x9e && 0x97 != d) ? 1 : 0;
  }
  else result = (0 != isupper(c)) ? 1 : 0;
  return result;
}


/**
 * Lexicon id of a SURFACE word. The lexicon interns lower-cased text but looks
 * up raw bytes, so a capitalized word silently misses without this.
 */
static unsigned int corpus_word_id(libxs_lexicon_t* lexicon, const char* word,
  int word_len)
{
  unsigned int result = 0;
  if (NULL != lexicon && NULL != word && 0 < word_len
    && word_len <= LIBXS_LEXEME_MAXBYTES)
  {
    char lower[LIBXS_LEXEME_MAXBYTES + 1];
    int pos;
    for (pos = 0; pos < word_len; ++pos) {
      const unsigned char c = (unsigned char)word[pos];
      /* One case rule, two encodings: the supplement differs by 0x20 too. */
      if (0xc3 == c && pos + 1 < word_len
        && 0 != corpus_word_upper(word, pos, word_len))
      {
        lower[pos] = (char)c;
        lower[pos + 1] = (char)(((unsigned char)word[pos + 1]) + 0x20);
        ++pos;
      }
      else lower[pos] = (char)tolower(c);
    }
    lower[word_len] = '\0';
    result = libxs_lexicon_id(lexicon, lower, word_len, 0, 0);
  }
  return result;
}


/**
 * Is the capital at `at` FORCED by position rather than chosen by the author?
 *
 * Three positions force one, and all three are structural: the start of the
 * text, anything after a sentence or clause terminator, and the first word
 * inside an opening quotation mark - an utterance begins there, which is why
 * interjections look like names. A heading forces every capital in it, and the
 * heading map already says where one is.
 *
 * This lives in core because two consumers read it and must not disagree: the
 * NAME census, which asks whether an author CHOSE a capital, and the NOUN test,
 * which asks the same question of a language that capitalizes every noun. One
 * question, two capabilities, decided by what the orthography encodes.
 */
int corpus_case_forced(const char* text, int at, int heading_len)
{
  int result = (at <= heading_len) ? 1 : 0;
  int scan = at;
  while (0 == result && 0 < scan && ' ' == text[scan - 1]) --scan;
  if (0 == result) {
    if (0 == scan) result = 1;
    else {
      const unsigned char c = (unsigned char)text[scan - 1];
      if ('.' == c || '!' == c || '?' == c || ':' == c || ';' == c
        || '\n' == c || '"' == c || '\'' == c || '`' == c)
      {
        result = 1;
      }
      /* The typeset quotes are three bytes and end with these, which is enough
       * to recognise them without decoding: no other character here does. The
       * last two are the guillemets a German edition opens an utterance with. */
      else if (0x98 == c || 0x9c == c || 0xab == c || 0xbb == c) result = 1;
    }
  }
  return result;
}


/**
 * Bytes of the heading this span opens with, 0 if it does not open with one.
 *
 * A heading is a LINE holding no lower-case letter, not a leading run of
 * capitals. Scanning the run instead cost every one-word title (a lone run has
 * one word, and one word had to be rejected because the capital that opens an
 * ordinary sentence looks identical), truncated any title containing internal
 * punctuation at the first mark, and ran a title into a second heading below it.
 * Prose cannot be a line without a lower-case letter, so the line boundary
 * decides what the word count was standing in for - and a title that stands
 * alone as a paragraph, which the run scan could only recognise when the title
 * happened to contain punctuation, is recognised by the same rule.
 *
 * Ingest asks here to capture a section and the reply path asks here to strip a
 * captured heading back off. Two definitions of "heading" is what let a reply
 * open mid-word: the stripper counted the sentence's own first capital as the
 * title's second word and ate it.
 */
int corpus_title_len(const char* text, int len)
{
  int result = 0;
  /**
   * A heading starts with a letter. Requiring that rejects all-caps text which
   * merely opens with punctuation - a quoted letter signature in dialogue, for
   * instance - that otherwise looks exactly like a title here.
   */
  if (NULL != text && 0 < len && 0 != isalpha((unsigned char)text[0])) {
    int end = 0, nupper = 0, lower = 0;
    while (end < len && '\n' != text[end] && '\r' != text[end]) {
      const unsigned char ch = (unsigned char)text[end];
      if (0 != isalpha(ch)) {
        if (0 != isupper(ch)) ++nupper;
        else lower = 1;
      }
      ++end;
    }
    if (0 == lower && 3 <= nupper) {
      while (0 < end && 0 != isspace((unsigned char)text[end - 1])) --end;
      result = end;
    }
  }
  return result;
}


/**
 * Non-zero if the line is markup rather than content: bracketed, or carrying a
 * field separator no prose uses.
 *
 * The heading rule reads a line's SHAPE and never its letters, which is what makes
 * it work across editions - and a caption is shaped exactly like a heading: short,
 * alone, and set off by blank lines. So captions became sections and answers were
 * credited to them. A vertical bar is the one mark that separates the two without
 * reading any words, since it is field syntax in every markup that has it and
 * punctuation in no prose.
 */
int corpus_line_markup(const char* text, int len)
{
  int result = 0;
  if (NULL != text && 0 < len) {
    if ('[' == text[0]) {
      int end = len;
      while (0 < end && 0 != isspace((unsigned char)text[end - 1])) --end;
      if (0 < end && ']' == text[end - 1]) result = 1;
    }
    if (0 == result && NULL != memchr(text, '|', (size_t)len)) result = 1;
    /* An entity reference is markup by the same argument as the bar: "&nbsp;" and
     * "&ndash;" are syntax, and a line carrying one was never prose a reader wrote.
     * Without this, "Best Picture&nbsp;&ndash; 1928 to present" was a section and
     * answers were credited to it. */
    if (0 == result) {
      int at;
      for (at = 0; at < len && 0 == result; ++at) {
        if ('&' == text[at]) {
          int scan = at + 1;
          while (scan < len && scan < at + 9 && 0 == result) {
            if (';' == text[scan]) result = 1;
            else if (0 == isalnum((unsigned char)text[scan])
              && '#' != text[scan]) break;
            ++scan;
          }
        }
      }
    }
  }
  return result;
}


/**
 * The 1-based line of the source file an offset belongs to.
 *
 * Two things make this exact rather than approximate. The offset is into the text
 * ingest works on, which for prose is the REFLOWED text - a line there is not a
 * line of the file a reader opens - so the reflow map is consulted when there is
 * one. And the map is monotone, so the answer is the last input line whose output
 * offset does not exceed the position asked about.
 */
static unsigned int corpus_line_at(const unsigned char* text, size_t size,
  size_t offset)
{
  unsigned int result = 1;
  if (NULL != corpus_ingest_lines && 0 < corpus_ingest_nlines) {
    size_t lo = 0, hi = corpus_ingest_nlines - 1, found = 0;
    while (lo <= hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (corpus_ingest_lines[mid] <= offset) {
        found = mid;
        lo = mid + 1;
      }
      else if (0 < mid) hi = mid - 1;
      else break;
    }
    result = (unsigned int)(found + 1);
  }
  else if (NULL != text && offset <= size) {
    size_t at;
    for (at = 0; at < offset; ++at) {
      if ('\n' == text[at]) ++result;
    }
  }
  return result;
}


/**
 * Build the same map for text that was NOT reflowed, so both ingest paths ask
 * corpus_line_at the same question and neither needs the text again.
 */
static void corpus_lines_index(const unsigned char* text, size_t size)
{
  free(corpus_ingest_lines);
  corpus_ingest_lines = NULL;
  corpus_ingest_nlines = 0;
  if (NULL != text) {
    size_t* lines = (size_t*)malloc((size + 2) * sizeof(*lines));
    if (NULL != lines) {
      size_t at, n = 0;
      lines[n++] = 0;
      for (at = 0; at < size; ++at) {
        if ('\n' == text[at]) lines[n++] = at + 1;
      }
      corpus_ingest_lines = lines;
      corpus_ingest_nlines = n;
    }
  }
}


/** Remember which file a source id stands for, so a citation can name it. */
static int corpus_source_path_set(int id, const char* path)
{
  int result = EXIT_FAILURE;
  if (0 <= id && NULL != path) {
    if (id >= corpus_source_npaths) {
      const int grown = id + 1;
      char** paths = (char**)realloc(corpus_source_paths,
        (size_t)grown * sizeof(*paths));
      if (NULL != paths) {
        int at;
        for (at = corpus_source_npaths; at < grown; ++at) paths[at] = NULL;
        corpus_source_paths = paths;
        corpus_source_npaths = grown;
      }
    }
    if (id < corpus_source_npaths) {
      const size_t len = strlen(path);
      char* copy = (char*)malloc(len + 1);
      if (NULL != copy) {
        memcpy(copy, path, len + 1);
        free(corpus_source_paths[id]);
        corpus_source_paths[id] = copy;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


const char* corpus_source_path(unsigned int id)
{
  const char* result = NULL;
  if ((int)id < corpus_source_npaths) result = corpus_source_paths[id];
  return result;
}


const char* converse_norms_path(void)
{
  return converse_path_norm;
}


static void corpus_source_paths_free(void)
{
  int at;
  for (at = 0; at < corpus_source_npaths; ++at) free(corpus_source_paths[at]);
  free(corpus_source_paths);
  corpus_source_paths = NULL;
  corpus_source_npaths = 0;
}


/** One path per line, the line number being the source id it resolves. */
static void corpus_source_paths_save(const char* path)
{
  FILE* f = (NULL != path && 0 < corpus_source_npaths) ? fopen(path, "w") : NULL;
  if (NULL != f) {
    int at;
    for (at = 0; at < corpus_source_npaths; ++at) {
      fprintf(f, "%s\n", (NULL != corpus_source_paths[at])
        ? corpus_source_paths[at] : "");
    }
    fclose(f);
  }
}


static void corpus_source_paths_load(const char* path)
{
  FILE* f = (NULL != path) ? fopen(path, "r") : NULL;
  if (NULL != f) {
    char line[CONVERSE_PATH_MAX];
    int at = 0;
    while (NULL != fgets(line, (int)sizeof(line), f)) {
      size_t len = strlen(line);
      while (0 < len && ('\n' == line[len - 1] || '\r' == line[len - 1])) {
        line[--len] = '\0';
      }
      if (0 < len) corpus_source_path_set(at, line);
      ++at;
    }
    fclose(f);
  }
}


static int corpus_sections_append(const char* text, int len, size_t begin,
  int depth)
{
  int result = EXIT_SUCCESS;
  if (corpus_sections_size == corpus_sections_cap) {
    const int cap = (0 < corpus_sections_cap) ? (2 * corpus_sections_cap) : 64;
    corpus_section_t* grown = (corpus_section_t*)realloc(corpus_sections,
      (size_t)cap * sizeof(*grown));
    if (NULL != grown) {
      corpus_sections = grown;
      corpus_sections_cap = cap;
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    corpus_section_t* section = corpus_sections + corpus_sections_size;
    int copy_len = len;
    if (copy_len >= ENTRY_SECTION_MAX) copy_len = ENTRY_SECTION_MAX - 1;
    memcpy(section->text, text, (size_t)copy_len);
    section->text[copy_len] = '\0';
    section->len = copy_len;
    section->begin = begin;
    section->depth = depth;
    ++corpus_sections_size;
  }
  return result;
}


/**
 * Map the headings of one file.
 *
 * A heading is a LINE that STANDS ALONE at the SEPARATION THAT DIVIDES THIS FILE.
 * Nothing here reads the letters, so no edition's typography is assumed: three
 * corpora that mark their titles by case, by centring, and by a trailing period
 * are all read by the same rule, and a fourth that does something else again
 * would be too.
 *
 * Three properties carry it, and each replaced a reading that failed on some
 * edition:
 *  - STANDS ALONE separates a title from VERSE. A spell or a rhyme inside a tale
 *    is set apart and indented exactly as a title is, but it is a BLOCK; only its
 *    last line is followed by blank space, and that line is not preceded by any.
 *  - AT LEAST TWO blank lines, because ONE is how paragraphs are separated. That
 *    is the only depth excluded, and it is excluded because it is the ordinary
 *    case rather than because it measured badly.
 *  - SHORTER than the file's median line. Where a file holds one paragraph per
 *    LINE, the opening paragraph of each part also stands alone at a separation
 *    of its own and outvoted the titles; a title is not a thousand characters of
 *    prose. The median is the file's own, so nothing assumes a line width.
 *  - the MOST COMMON such depth, not the widest. The widest is an extremum, and
 *    an extremum is set by whichever line is most unusual: two pieces of front
 *    and back matter carrying one blank line more than sixty-one titles reduced a
 *    whole book to two sections. The mode answers identically wherever the widest
 *    was already right.
 *
 * A heading below that separation is a SUBSECTION, which names a part of a
 * section rather than a source, so it does not open one: crediting an answer to a
 * part number says nothing about where it came from.
 */
static void corpus_sections_build(const unsigned char* text, size_t size)
{
  enum { SECTION_DEPTH_MAX = 64, SECTION_LINE_MAX = 1024 };
  size_t pos = 0, line_start = 0;
  int depths[SECTION_DEPTH_MAX];
  int lengths[SECTION_LINE_MAX];
  int blanks = 0, level = 0, pass, d, median = 0;
  long nline = 0, seen = 0;
  corpus_sections_size = 0;
  for (d = 0; d < SECTION_DEPTH_MAX; ++d) depths[d] = 0;
  for (d = 0; d < SECTION_LINE_MAX; ++d) lengths[d] = 0;
  for (pos = 0, line_start = 0; pos <= size && NULL != text; ++pos) {
    if (pos == size || '\n' == text[pos]) {
      int len = (int)(pos - line_start), indent = 0;
      while (0 < len && 0 != isspace(text[line_start + len - 1])) --len;
      while (indent < len && 0 != isspace(text[line_start + indent])) ++indent;
      if (indent < len) {
        const int body = len - indent;
        ++lengths[(body < SECTION_LINE_MAX) ? body : (SECTION_LINE_MAX - 1)];
        ++nline;
      }
      line_start = pos + 1;
    }
  }
  for (d = 0; d < SECTION_LINE_MAX && 2 * seen < nline; ++d) {
    seen += lengths[d];
    median = d;
  }
  for (pass = 0; pass < 2 && NULL != text; ++pass) {
    if (1 == pass) {
      /* Deeper wins a tie, which is what the widest-separation reading did. */
      for (d = 2; d < SECTION_DEPTH_MAX; ++d) {
        if (0 < depths[d] && (0 == level || depths[d] >= depths[level])) {
          level = d;
        }
      }
      if (0 == level) level = 2;
    }
    line_start = 0;
    blanks = 0;
    for (pos = 0; pos <= size; ++pos) {
      if (pos == size || '\n' == text[pos]) {
        int len = (int)(pos - line_start), indent = 0;
        while (0 < len && 0 != isspace(text[line_start + len - 1])) --len;
        while (indent < len && 0 != isspace(text[line_start + indent])) ++indent;
        /* Markup neither separates two sections nor is content, and where it
         * stands between a heading and the space above it, counting it as
         * content hides the separation the heading was given. */
        if (indent < len && 0 != corpus_line_markup((const char*)text
          + line_start + indent, len - indent))
        {
          len = 0;
        }
        if (0 == len) ++blanks;
        else {
          const size_t at = line_start + (size_t)indent;
          const int body = len - indent;
          size_t scan = pos + 1;
          int alone, cased = 0;
          while (scan < size && '\n' != text[scan] && 0 != isspace(text[scan])) {
            ++scan;
          }
          alone = (scan >= size || '\n' == text[scan]) ? 1 : 0;
          /**
           * And a candidate that does not FIT the section field is prose, not a
           * title. The median test alone is toothless on a corpus written one
           * PARAGRAPH per line - the median is then a paragraph, so every lead
           * sentence is "shorter than the median" and became a section, which is
           * how answers came to be credited to "Anisotropy (the opposite of
           * isotropy) is the property of being ". A title that cannot be stored
           * whole cannot be a citation either, so the storage bound is the honest
           * one to test here.
           */
          /**
           * And the line must hold NO LOWER-CASE LETTER, which is the definition
           * corpus_title_len already used for a heading inside an entry - so there
           * is one definition of "heading" in this file instead of two that
           * disagree. It is what finally separates a title from prose on a corpus
           * whose titles were dropped: every remaining false section on the wiki
           * extracts was an ordinary sentence ("Rand's papers at The Library of
           * Congress"), and no shape test could refuse them because they are shaped
           * like titles. A prose corpus whose titles are mixed case loses its
           * sections here and keeps its FILE and LINE, which is the trade this
           * project takes every time: lose a truth rather than assert a falsehood.
           */
          for (d = 0; d < len - indent && 0 == cased; ++d) {
            if (0 != islower((unsigned char)text[at + d])) cased = 1;
          }
          if (0 < body && body < median && body < ENTRY_SECTION_MAX
            && 0 == cased && 0 != alone && (1 < blanks || 0 == line_start))
          {
            /* The first heading of a file has no separation above it and is
             * still the outermost one there is, so it never votes and always
             * qualifies. */
            const int depth = (0 == line_start) ? level : blanks;
            if (0 == pass) {
              if (0 != line_start && depth < SECTION_DEPTH_MAX) {
                ++depths[depth];
              }
            }
            else if (depth >= level) {
              /* The section begins AT its heading, not after it: ingest stores
               * the first sentence of a section from the heading onward, so a
               * section that began below its own heading would credit the first
               * sentence of every section to the previous one. */
              corpus_sections_append((const char*)text + at, body, at, depth);
            }
          }
          blanks = 0;
        }
        line_start = pos + 1;
      }
    }
  }
}


/** The section covering this byte: the last heading at or before it. */
static int corpus_section_at(size_t offset, const char** text)
{
  int result = 0;
  int lo = 0, hi = corpus_sections_size - 1, found = -1;
  *text = NULL;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (corpus_sections[mid].begin <= offset) {
      found = mid;
      lo = mid + 1;
    }
    else hi = mid - 1;
  }
  if (0 <= found) {
    *text = corpus_sections[found].text;
    result = corpus_sections[found].len;
  }
  return result;
}


/**
 * Upper bound on the fingerprint-collision probe chain. 65536 was effectively
 * unbounded and made ingest quadratic on collision-heavy corpora.
 */
static unsigned int corpus_chain_max(void)
{
  unsigned int result = 65536;
  const char* env = getenv("CONVERSE_CHAIN_MAX");
  if (NULL != env && '\0' != *env) {
    const long v = atol(env);
    if (v >= 1 && v <= 65536) result = (unsigned int)v;
  }
  return result;
}

/**
 * Registry key = IDENTITY of the entry's content, not its fingerprint. The two
 * jobs were conflated: a Hilbert code of the fingerprint is locality-preserving
 * ON PURPOSE, so similar texts share a cell, and using it as a unique key forced
 * a linear probe over every colliding entry with a full text compare per step.
 * On collision-heavy corpora (wiki markup, code) that made ingest quadratic -
 * 2x the enwik8 text cost 10.8x the ingest CPU.
 *
 * A content hash collides only by accident, so insertion and duplicate detection
 * are one lookup. The similarity index gets its codes from the entries instead
 * (libxs_spatial_build_codes), which is where locality belongs.
 *
 * The 128-bit key (two independent hashes plus the length) makes an accidental
 * collision between DIFFERENT texts negligible. One is still resolved rather
 * than dropped: the walk appends a seq only from the second probe on, and the
 * registry supports mixed key sizes, so the common case stays 16 bytes and no
 * entry is lost to a genuine hash collision.
 */
static void corpus_key_from_text(const corpus_entry_t* entry,
  unsigned char key[], size_t* key_size)
{
  const unsigned int len = (unsigned int)entry->text_len;
  const unsigned int h1 = libxs_hash(entry->text, len, 0x9e3779b9u);
  const unsigned int h2 = libxs_hash(entry->text, len, 0x85ebca6bu);
  const unsigned int hs = (0 < entry->section_len)
    ? libxs_hash(entry->section, (unsigned int)entry->section_len, 0xc2b2ae35u)
    : 0u;
  memcpy(key, &h1, 4);
  memcpy(key + 4, &h2, 4);
  memcpy(key + 8, &hs, 4);
  memcpy(key + 12, &len, 4);
  *key_size = 16;
}


static const corpus_blob_t* corpus_blob_get(unsigned int id)
{
  const corpus_blob_t* result = NULL;
  if (NULL != corpus_parents) {
    unsigned char key[12];
    size_t key_size = 0;
    corpus_blob_key(id, key, &key_size);
    result = (const corpus_blob_t*)libxs_registry_get(corpus_parents, key,
      key_size, NULL);
  }
  return result;
}


/**
 * The text and section a stored record stands for, whichever kind it is. The
 * duplicate walk compares texts, and a span's text lives in its parent, so the
 * comparison needs one definition of "the text of this record" rather than a
 * special case at the one site that happens to hit a span first.
 *
 * The section is reported ABSENT below CORPUS_ENTRY_META_SIZE rather than read: an
 * entry is stored at its actual text length, so a size test against sizeof would
 * be true of essentially every entry and would collapse the comparison to "equal
 * only if the new record has no section". With sections that is never true, so
 * re-ingesting a text stored a SECOND COPY of every sentence in it - the corpus
 * doubled on each warm run, which is the mechanism behind the standing warm-start
 * warning.
 */
static int corpus_record_text(const void* value, size_t value_size,
  const char** text, int* len, const char** section, int* section_len)
{
  int result = EXIT_FAILURE;
  *text = NULL; *len = 0; *section = NULL; *section_len = 0;
  if (ENTRY_KIND_FULL == corpus_value_kind(value)) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    *text = entry->text;
    *len = entry->text_len;
    if (CORPUS_ENTRY_META_SIZE <= value_size) {
      *section = entry->section;
      *section_len = entry->section_len;
    }
    result = EXIT_SUCCESS;
  }
  else if (ENTRY_KIND_SPAN == corpus_value_kind(value)) {
    const corpus_span_t* span = (const corpus_span_t*)value;
    const corpus_blob_t* blob = corpus_blob_get(span->parent);
    if (NULL != blob
      && (size_t)span->offset + (size_t)span->text_len
        <= (size_t)blob->text_len)
    {
      *text = blob->text + span->offset;
      *len = span->text_len;
      *section = blob->section;
      *section_len = blob->section_len;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Store the parent text a set of windows will be cut from, and return its id.
 * Nothing de-duplicates parents: two identical paragraphs are rare, and the windows
 * they yield collapse in the content key space anyway, which leaves at worst an
 * unreferenced parent - and the caller removes that one, because on a warm
 * re-ingest EVERY parent is that one.
 */
static unsigned int corpus_store_blob(const unsigned char* text, int len,
  const char* section, int section_len)
{
  unsigned int result = 0;
  libxs_registry_t* parents = corpus_parents_get();
  if (NULL != parents && NULL != text && 0 < len) {
    const size_t size = CORPUS_BLOB_META_SIZE + (size_t)len + 1;
    corpus_blob_t* blob = (corpus_blob_t*)malloc(size);
    if (NULL != blob) {
      unsigned char key[12];
      size_t key_size = 0;
      int copy_len = (NULL != section) ? section_len : 0;
      if (copy_len >= ENTRY_SECTION_MAX) copy_len = ENTRY_SECTION_MAX - 1;
      memset(blob, 0, CORPUS_BLOB_META_SIZE);
      blob->kind = (unsigned short)ENTRY_KIND_BLOB;
      blob->connector = CONN_NEWLINE;
      blob->scale = SCALE_PARAGRAPH;
      blob->text_len = len;
      blob->source = (unsigned short)corpus_source_id;
      if (0 < copy_len) {
        memcpy(blob->section, section, (size_t)copy_len);
        blob->section_len = (unsigned short)copy_len;
      }
      memcpy(blob->text, text, (size_t)len);
      blob->text[len] = '\0';
      corpus_blob_key(corpus_blob_max + 1, key, &key_size);
      if (NULL != libxs_registry_set(parents, key, key_size, blob, size,
        NULL))
      {
        ++corpus_blob_max;
        result = corpus_blob_max;
      }
      free(blob);
    }
  }
  return result;
}


void corpus_view_bind(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules)
{
  corpus_view_lexicon = lexicon;
  corpus_view_rules = rules;
  corpus_view_nrules = nrules;
}


/**
 * Release every materialized window.
 *
 * A view is OWNED here and its pointer is handed out, so whoever kept one -
 * recombination's pivot index is the only such reader - must release its own
 * structures FIRST. That order is not incidental: the two live in different
 * translation units, and freeing this side first would leave the index holding
 * pointers into freed memory with every count still looking correct.
 */
void corpus_view_free(void)
{
  if (NULL != corpus_views) {
    const void* key = NULL;
    size_t cursor = 0;
    void* value = libxs_registry_begin(corpus_views, &key, &cursor);
    while (NULL != value) {
      free(*(corpus_entry_t**)value);
      value = libxs_registry_next(corpus_views, &key, &cursor);
    }
    libxs_registry_destroy(corpus_views);
    corpus_views = NULL;
  }
  corpus_view_count = 0;
}


/**
 * Rebuild a window into the entry it stood for. Everything the stored entry carried
 * is recovered here: the tokens and the fingerprint from the same bytes through the
 * same builder, the section and the SOURCE from the parent - the source because the
 * builder stamps whichever file ingest is currently reading, which by materialization
 * time has moved on, and recombination reports same-source agreement.
 */
static int corpus_span_build(const corpus_span_t* span, corpus_entry_t* entry)
{
  int result = EXIT_FAILURE;
  const corpus_blob_t* blob = corpus_blob_get(span->parent);
  if (NULL != blob && 0 < span->text_len
    && (size_t)span->offset + (size_t)span->text_len <= (size_t)blob->text_len
    && EXIT_SUCCESS == corpus_entry_build(entry,
      (const unsigned char*)blob->text + span->offset, span->text_len,
      span->scale, corpus_view_lexicon, corpus_view_rules,
      corpus_view_nrules, 0))
  {
    entry->lexical_flags |= ENTRY_LEX_FRAGMENT;
    entry->source = blob->source;
    entry->line = span->line;
    corpus_entry_set_section(entry, blob->section, blob->section_len);
    result = EXIT_SUCCESS;
  }
  return result;
}


int corpus_value_viable(const void* value)
{
  int result = 0;
  const unsigned int kind = corpus_value_kind(value);
  if (ENTRY_KIND_FULL == kind) result = 1;
  else if (ENTRY_KIND_SPAN == kind) {
    const corpus_span_t* span = (const corpus_span_t*)value;
    const corpus_blob_t* blob = corpus_blob_get(span->parent);
    result = (NULL != blob && 0 < span->text_len
      && (size_t)span->offset + (size_t)span->text_len
        <= (size_t)blob->text_len) ? 1 : 0;
  }
  return result;
}


const corpus_entry_t* corpus_entry_scan(const void* value,
  corpus_entry_t* scratch)
{
  const corpus_entry_t* result = NULL;
  const unsigned int kind = corpus_value_kind(value);
  if (ENTRY_KIND_FULL == kind) result = (const corpus_entry_t*)value;
  else if (ENTRY_KIND_SPAN == kind && NULL != scratch
    && EXIT_SUCCESS == corpus_span_build((const corpus_span_t*)value, scratch))
  {
    result = scratch;
  }
  return result;
}


/**
 * Cache key: WHICH WINDOW, not where the window's record happens to sit.
 *
 * Keying on the span record's address would make one registry's storage part of
 * another registry's keys, and then any reallocation, removal or teardown on the
 * corpus side silently redefines what a cached view stands for - a stale entry
 * returned for a different window, with nothing to notice it. The parent id and
 * the offset identify the window itself, so the cache survives the corpus moving
 * its records and cannot outlive its meaning.
 */
static void corpus_view_key(const corpus_span_t* span, unsigned int key[2])
{
  key[0] = span->parent;
  key[1] = span->offset;
}


const corpus_entry_t* corpus_entry_view(const void* value)
{
  const corpus_entry_t* result = NULL;
  const unsigned int kind = corpus_value_kind(value);
  if (ENTRY_KIND_FULL == kind) result = (const corpus_entry_t*)value;
  else if (ENTRY_KIND_SPAN == kind) {
    unsigned int key[2];
    corpus_entry_t** cached = NULL;
    corpus_view_key((const corpus_span_t*)value, key);
    if (NULL == corpus_views) corpus_views = libxs_registry_create();
    if (NULL != corpus_views) {
      cached = (corpus_entry_t**)libxs_registry_get(corpus_views,
        key, sizeof(key), NULL);
    }
    if (NULL != cached) result = *cached;
    else if (NULL != corpus_views) {
      corpus_entry_t entry;
      if (EXIT_SUCCESS == corpus_span_build((const corpus_span_t*)value, &entry))
      {
        const size_t size = corpus_entry_size(&entry);
        corpus_entry_t* made = (corpus_entry_t*)malloc(size);
        if (NULL != made) {
          memcpy(made, &entry, size);
          if (NULL != libxs_registry_set(corpus_views, key, sizeof(key),
            &made, sizeof(made), NULL))
          {
            result = made;
            ++corpus_view_count;
          }
          else free(made);
        }
      }
    }
  }
  return result;
}


/**
 * Store a record under the content key of `entry`, writing either the entry itself
 * or, when `span` is given, the sixteen bytes that locate the same text inside its
 * parent. The duplicate walk is shared deliberately: what collapses two identical
 * windows is the KEY, so a derived pool de-duplicates exactly as a stored one did
 * and no separate rule can drift away from it.
 */
static int corpus_store_record(libxs_registry_t* corpus,
  const corpus_entry_t* entry, const corpus_span_t* span)
{
  int result = 0;
  unsigned char key[20];
  size_t key_size = 0;
  unsigned int seq;
  int matched = 0;
  if (NULL == corpus || NULL == entry) return 0;
  corpus_key_from_text(entry, key, &key_size);
  /**
   * The sequence walk is a linear probe over every entry sharing a quantized
   * fingerprint, and it re-reads each one to compare text. On prose the chains
   * are short, but the key quantizes to COMPOSE_NDIMS x COMPOSE_BITS, so a
   * corpus of many near-identical short texts (wiki markup, code) collides
   * heavily and ingest becomes quadratic: 2x the enwik8 text cost 10.8x the
   * ingest CPU. The cap bounds the walk so ingest stays linear; a text beyond it
   * is dropped rather than stored, which is reported so a silent truncation
   * cannot be mistaken for a rare fingerprint (the same discipline as the
   * recomb postings cap).
   */
  for (seq = 0; seq < corpus_chain_max(); ++seq) {
    void* existing;
    if (0 < seq) {
      const unsigned short seq_key = (unsigned short)seq;
      memcpy(key + 16, &seq_key, 2);
      key_size = 18;
    }
    existing = libxs_registry_get(corpus, key, key_size, NULL);
    if (NULL == existing) {
      if (0 == matched) {
        if (NULL != span) {
          libxs_registry_set(corpus, key, key_size, span, sizeof(*span), NULL);
        }
        else {
          libxs_registry_set(corpus, key, key_size,
            entry, corpus_entry_size(entry), NULL);
        }
        result = 1;
      }
      break;
    }
    else {
      const char* old_text = NULL;
      const char* old_section = NULL;
      int old_len = 0, old_section_len = 0;
      size_t old_size = libxs_registry_value_size(corpus, key,
        key_size, NULL);
      if (EXIT_SUCCESS == corpus_record_text(existing, old_size,
          &old_text, &old_len, &old_section, &old_section_len)
        && old_len == entry->text_len
        && 0 == libxs_memcmp(old_text, entry->text, (size_t)entry->text_len)
        && old_section_len == (int)entry->section_len
        && (0 == old_section_len || 0 == libxs_memcmp(old_section,
          entry->section, (size_t)old_section_len)))
      {
        const corpus_entry_t* old_entry = (ENTRY_KIND_FULL
          == corpus_value_kind(existing))
          ? (const corpus_entry_t*)existing : NULL;
        matched = 1;
        /* Replace only to gain token metadata; sizes now vary by text
         * length, so an unequal size is no longer evidence of anything.
         * A span was built from the same text as this entry, so it has the
         * same tokens and there is nothing to gain from replacing it. */
        if (NULL != old_entry) {
          if (0 == old_entry->ntokens && entry->ntokens > 0) {
            libxs_registry_set(corpus, key, key_size,
              entry, corpus_entry_size(entry), NULL);
          }
          if (old_entry->ntokens > 0) break;
        }
        else break;
      }
    }
  }
  /* Chain exhausted without placing or matching: the text is dropped. */
  if (0 == result && 0 == matched) ++corpus_chain_dropped;
  return result;
}


static int corpus_store_entry(libxs_registry_t* corpus,
  const corpus_entry_t* entry)
{
  return corpus_store_record(corpus, entry, NULL);
}


int answer_relation_reply(const answer_relation_match_t* match,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  size_t pos = 0;
  const char* copula;
  if (NULL == match || NULL == output || 0 == output_size
    || match->answer_len <= 0 || match->relation_len <= 0) return EXIT_FAILURE;
  copula = (0 != match->plural) ? " were " : " was ";
  /**
   * The ACTIVE shape is re-emitted in the voice the corpus used, which is what
   * makes it grammatical without consulting morphology. A passive rendering needs
   * the relation word to be a PARTICIPLE, and the active shape reads the SURFACE
   * form of the clause: "Gretel laid the spit" passivizes, but "Frederick walked
   * the fastest" and "Lily let the nut fall" do not, and nothing in the extractor
   * knows which is which. Stating what the corpus stated cannot be ungrammatical.
   */
  if (0 != match->active) {
    if (match->actor_len > 0 && (size_t)match->actor_len
      + (size_t)match->relation_len + (size_t)match->answer_len + 4
      < output_size)
    {
      memcpy(output + pos, match->actor, (size_t)match->actor_len);
      pos += (size_t)match->actor_len;
      output[pos++] = ' ';
      memcpy(output + pos, match->relation, (size_t)match->relation_len);
      pos += (size_t)match->relation_len;
      output[pos++] = ' ';
      memcpy(output + pos, match->answer, (size_t)match->answer_len);
      pos += (size_t)match->answer_len;
      output[pos++] = '.';
      output[pos] = '\0';
      result = EXIT_SUCCESS;
    }
  }
  else if ((size_t)match->answer_len + strlen(copula)
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


int answer_features_fill(const corpus_entry_t* entry,
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
  if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0) use_sketch = 1;
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


libxs_predict_t* answer_predict_create(
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


int answer_predict_build_model(libxs_predict_t* model,
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
  memset(&info, 0, sizeof(info));
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
  size_t cursor = 0, key_size = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = corpus_iter_begin_length(corpus, &key, &key_size, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    /* Key size from the iterator: see the note in answer_relation_reply. */
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, key_size, NULL)
      : sizeof(*entry);
    if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0
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
    value = corpus_iter_next_length(corpus, &key, &key_size, &cursor);
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


int text_find_ci(const char* text, int text_len, const char* term)
{
  int result = -1;
  if (NULL != text && NULL != term && 0 < text_len) {
    const char* hit = libxs_strimem(text, (size_t)text_len, term,
      strlen(term));
    if (NULL != hit) result = (int)(hit - text);
  }
  return result;
}


int text_contains_word_ci(const char* text, int text_len,
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


char* eval_trim(char* text)
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

/**
 * The variable-order n-gram engine now lives in libxs_ngram; the wrappers
 * below adapt converse's call sites (which thread the model's registry and an
 * explicit maxorder) to the single shared model. model == converse_ngram.store
 * and maxorder == converse_ngram.maxorder by construction.
 */
static void ngramk_observe(libxs_registry_t* model, const unsigned int hist[],
  int hlen, unsigned int succ_id, int maxorder)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  libxs_ngram_observe(&converse_ngram, hist, hlen, succ_id);
}

/**
 * The pure count-based estimate: interpolated backoff with NO skip tier folded
 * in. The bank needs this because it carries skip as its own slot, and an order
 * expert that already contained skip at a fixed weight would double-count it -
 * making the learned skip weight meaningless.
 */
double ngramk_prob_exact(const unsigned int hist[], int hlen,
  unsigned int next)
{
  return libxs_ngram_prob(&converse_ngram, hist, hlen, next);
}


double ngramk_prob(libxs_registry_t* model, const unsigned int hist[],
  int hlen, int maxorder, unsigned int next)
{
  double p = ngramk_prob_exact(hist, hlen, next);
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  if (0 != converse_skip_on) {
    double p_skip = ngram_skip_prob(hist, hlen, next);
    if (p_skip > 0.0) {
      double mu = ngram_skip_mu();
      p = (1.0 - mu) * p + mu * p_skip;
    }
  }
  return p;
}

/**
 * Adapter matching converse_recomb_prob_t. The recombination syntax gate needs a
 * word-scale backoff probability and nothing more, so the model handle, the
 * maximum order and the skip-gram interpolation all stay on this side of the
 * boundary rather than being threaded through it.
 */
double recomb_word_prob(const unsigned int hist[], int hlen,
  unsigned int next)
{
  return ngramk_prob(NULL, hist, hlen, ngram_maxorder(), next);
}


int ngramk_predict_order(libxs_registry_t* model,
  const unsigned int hist[], int hlen, int maxorder, unsigned int out_ids[],
  int k, int* order)
{
  LIBXS_UNUSED(model); LIBXS_UNUSED(maxorder);
  return libxs_ngram_predict(&converse_ngram, hist, hlen, out_ids, k, order);
}


static int ngram_gran_mode(void)
{
  const char* env = getenv("CONVERSE_GRAN");
  if (NULL != env) {
    if (0 == strcmp(env, "native")) return GRAN_NATIVE;
    if (0 == strcmp(env, "syllable")) return GRAN_SYLLABLE;
    if (0 == strcmp(env, "bpe")) return GRAN_BPE;
    if (0 == strcmp(env, "meta-native")) return GRAN_META_NATIVE;
    if (0 == strcmp(env, "meta-word")) return GRAN_META_WORD;
    if (0 == strcmp(env, "meta-syllable")) return GRAN_META_SYLLABLE;
  }
  return GRAN_WORD;
}


int ngram_native_mode(void)
{
  return (GRAN_WORD != ngram_gran_mode()) ? 1 : 0;
}

/**
 * Number of prior whole words to carry as sub-word prediction context, or 0
 * (off, byte-identical to piece-only context). Ignored at word/native mode.
 */
int ngram_wordctx(void)
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

/**
 * Signed-length adapter over libxs_utf8_decode, which is the strict form: an
 * invalid or truncated sequence reports width 1, so scanning always advances and
 * a malformed byte is simply not a vowel.
 */
static unsigned long ngram_utf8_decode(const char* text, int len, int* width)
{
  return libxs_utf8_decode((const unsigned char*)text, (size_t)len, width);
}

/**
 * Vowel test over CODE POINTS rather than bytes.  The byte test cannot see an
 * encoded letter at all, so every accented vowel read as a consonant and the
 * splitter cut German and French words at the wrong places - a defect, not a
 * tuning choice.  Latin-1 Supplement and Latin Extended-A cover the vowels of
 * the languages this corpus set contains; anything outside is not claimed as a
 * vowel rather than guessed at.
 */
static int ngram_is_vowel_cp(unsigned long cp)
{
  int result = 0;
  if (cp < 128) result = ngram_is_vowel((unsigned char)cp);
  else if (0xC0 <= cp && 0x24F >= cp) {
    /* fold to the base letter: the accented ranges run in blocks whose
     * residues mod the block size follow the base vowel order */
    static const char* const vowels = "aeiouy";
    unsigned long base = 0;
    if (0xC0 <= cp && 0xFF >= cp) {
      static const char latin1[64] = {
        'a','a','a','a','a','a','a','c','e','e','e','e','i','i','i','i',
        'd','n','o','o','o','o','o','x','o','u','u','u','u','y','t','s',
        'a','a','a','a','a','a','a','c','e','e','e','e','i','i','i','i',
        'd','n','o','o','o','o','o','/','o','u','u','u','u','y','t','y'
      };
      base = (unsigned long)latin1[cp - 0xC0];
    }
    if (0 != base && NULL != strchr(vowels, (int)base)) result = 1;
  }
  return result;
}

/**
 * A UTF-8 continuation or lead byte counts as a word character so that a pivot
 * span stays whole on a non-ASCII corpus: ctype rejects every byte of an encoded
 * letter, which would cut a German word mid-letter and offer the fragments as
 * separate pivots.
 *
 * Unlike the tokenizer's predicate this does NOT exclude encoded punctuation, and
 * it does not need to: this only extends a span whose FIRST byte the tokenizer
 * already classified as a word, so a leading typographic quote cannot start one.
 * The span is re-tokenized before its id is taken, so the authoritative decision
 * still belongs to the tokenizer.
 */
int ngram_is_wordchar(unsigned char c)
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

/**
 * Learn byte-pair merges from the training split only. Words are byte runs
 * split on whitespace (a leading-space marker keeps word starts distinct);
 * each iteration adds the most frequent adjacent symbol pair as a new symbol
 * and records its rank so encoding can replay the merges.
 */
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
    corpus_entry_t scratch;
    void* value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value && nwords < BPE_WORD_CAP) {
      const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
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
      value = corpus_iterx_next(corpus, &key, &cursor);
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

/**
 * Encode one whitespace-delimited byte run [text,len) into BPE pieces by
 * greedily applying the lowest-rank applicable merge, then intern each piece.
 * Falls back to single bytes for anything the merges do not cover, so the
 * encoder never fails on unseen input.
 */
static int bpe_encode_run(const char* text, int len, libxs_lexeme_t tokens[],
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
    tokens[result].flags = (unsigned short)(LIBXS_LEXEME_WORD
      | ((0 == i && 0 != marker) ? LIBXS_LEXEME_BREAK : 0));
    ++result;
  }
  return result;
}

/**
 * Split a word [text,wlen) into syllable pieces by a simple VC|CV heuristic:
 * cut before a consonant that is followed by a vowel, once the current piece
 * already contains a vowel. Caps piece length; always ends at word end.
 */
/**
 * Whether the consonant run text[a,b) is a legal syllable ONSET, i.e. may begin
 * a syllable in this orthography.
 *
 * This is the notion the previous rule lacked entirely: it cut before the last
 * consonant preceding a vowel, which splits `strength` inside `ngth` and
 * `rhythm` inside `thm` because it cannot tell a cluster from a coda. Maximal
 * onset says the onset takes as many consonants as may legally start a syllable
 * and the rest stay as the coda of the previous one.
 *
 * The digraph list is English-leaning and that is a KNOWN limit, recorded rather
 * than hidden: syllabification is strongly language-dependent (German compounds,
 * sch/tsch; Italian near-perfect CV), which is exactly the per-language cost
 * this project has refused to pay by hand. It is here to make the unit
 * measurable at all - the current output is wrong on words a speaker of any of
 * these languages reads correctly - not as the final rule.
 */
static int ngram_onset_legal(const char* text, int a, int b)
{
  const int n = b - a;
  int result = 0;
  if (0 >= n) result = 1; /* empty onset is always legal */
  else if (1 == n) result = 1;
  else if (2 == n) {
    static const char* const two[] = {
      "bl","br","ch","cl","cr","dr","fl","fr","gl","gn","gr","kl","kn","kr",
      "ph","pl","pr","qu","sc","sh","sk","sl","sm","sn","sp","st","sw","th",
      "tr","tw","wh","wr","ts","pf","sz","gh","dw","vr","zw",
      NULL
    };
    int k;
    char c0 = (char)tolower((unsigned char)text[a]);
    char c1 = (char)tolower((unsigned char)text[a + 1]);
    for (k = 0; NULL != two[k] && 0 == result; ++k) {
      if (two[k][0] == c0 && two[k][1] == c1) result = 1;
    }
  }
  else if (3 == n) {
    static const char* const three[] = {
      "chr","phr","sch","scr","shr","spl","spr","str","thr","tsch",
      NULL
    };
    int k;
    char c0 = (char)tolower((unsigned char)text[a]);
    char c1 = (char)tolower((unsigned char)text[a + 1]);
    char c2 = (char)tolower((unsigned char)text[a + 2]);
    for (k = 0; NULL != three[k] && 0 == result; ++k) {
      if (three[k][0] == c0 && three[k][1] == c1 && three[k][2] == c2) {
        result = 1;
      }
    }
  }
  return result;
}

/**
 * Split a word into syllable-like pieces by maximal onset.
 *
 * Walk the letters (code points, so an encoded vowel is seen); after a vowel has
 * been seen, a run of consonants followed by another vowel is a boundary, and
 * the cut goes as late as legality allows: try the longest legal onset first and
 * shorten it until the remainder before it is non-empty. A word with no vowel at
 * all stays one piece rather than being chopped at a fixed width, which is what
 * produced `stre|ngth`.
 */
int ngram_syllable_split(const char* text, int wlen, int piece_begin[],
  int piece_len[], int max)
{
  int result = 0;
  int start = 0, i = 0, seen_vowel = 0;
  while (i < wlen && result < max) {
    int w = 1;
    const unsigned long cp = ngram_utf8_decode(text + i, wlen - i, &w);
    if (0 == ngram_is_vowel_cp(cp)) {
      if (0 != seen_vowel) {
        /* scan the whole consonant run, then decide where it divides */
        int run = i, j = i, cut;
        while (j < wlen) {
          int wj = 1;
          const unsigned long cj = ngram_utf8_decode(text + j, wlen - j, &wj);
          if (0 != ngram_is_vowel_cp(cj)) break;
          j += wj;
        }
        if (j < wlen) { /* a vowel follows, so this run spans a boundary */
          cut = run;
          while (cut < j && 0 == ngram_onset_legal(text, cut, j)) ++cut;
          /* never cut at the piece start: that would emit an empty piece */
          if (cut <= start) cut = (start < j) ? j : wlen;
          if (cut > start && cut < wlen) {
            piece_begin[result] = start;
            piece_len[result] = cut - start;
            ++result;
            start = cut;
            seen_vowel = 0;
          }
          i = (cut > i) ? cut : j;
          continue;
        }
        i = j; /* trailing consonants: they belong to the final piece */
        continue;
      }
    }
    else seen_vowel = 1;
    i += w;
  }
  if (start < wlen && result < max) {
    piece_begin[result] = start;
    piece_len[result] = wlen - start;
    ++result;
  }
  return result;
}


static int ngram_metatoken_granularity(int mode)
{
  int result = -1;
  if (GRAN_META_NATIVE == mode) result = LIBXS_TOKEN_GRANULARITY_NATIVE;
  else if (GRAN_META_WORD == mode) result = LIBXS_TOKEN_GRANULARITY_WORD;
  else if (GRAN_META_SYLLABLE == mode) {
    result = LIBXS_TOKEN_GRANULARITY_SYLLABLE;
  }
  return result;
}


static unsigned int ngram_metatoken_flags(int kind, int sentence,
  int have_break)
{
  unsigned int result = 0;
  if (LIBXS_TOKEN_TEXT == kind) {
    result = LIBXS_LEXEME_WORD;
  }
  else if (LIBXS_TOKEN_NUMBER == kind) result = LIBXS_LEXEME_NUMBER;
  else result = LIBXS_LEXEME_PUNCT;
  if (LIBXS_TOKEN_MARKUP == kind || LIBXS_TOKEN_SPACE == kind) {
    result |= LIBXS_LEXEME_MARKUP;
  }
  if (0 != sentence) result |= LIBXS_LEXEME_SENTENCE;
  if (0 != have_break) result |= LIBXS_LEXEME_BREAK;
  return result;
}


static int ngram_metatoken_tokens(libxs_lexicon_t* lexicon,
  const char* text, int text_len, libxs_lexeme_t tokens[],
  unsigned int word_ids[], int max, int create, int granularity)
{
  int result = 0;
  libxs_token_stream_t stream;
  libxs_tokenizer_t* tokenizer = libxs_tokenizer_create(granularity);
  size_t token_pos = 0;
  int have_break = 0;
  libxs_token_stream_init(&stream);
  if (NULL != tokenizer && NULL != lexicon && NULL != text && text_len > 0
    && EXIT_SUCCESS == libxs_token_stream_encode(tokenizer, &stream,
      (const unsigned char*)text, (size_t)text_len))
  {
    const char bos = '\1';
    unsigned int bos_id = libxs_lexicon_id(lexicon, &bos, 1,
      LIBXS_LEXEME_MARKUP, create);
    if (0 == bos_id && 0 == create) {
      bos_id = libxs_lexicon_id(lexicon, &bos, 1,
        LIBXS_LEXEME_MARKUP, 1);
    }
    if (0 != bos_id && result < max) {
      tokens[result].id = bos_id;
      tokens[result].length = 0;
      tokens[result].flags = LIBXS_LEXEME_MARKUP;
      if (NULL != word_ids) word_ids[result] = bos_id;
      ++result;
    }
    while (token_pos < stream.size && result < max) {
      size_t payload_size = 0;
      size_t cells = libxs_token_span(stream.data, stream.size, token_pos,
        &payload_size);
      unsigned char payload[LIBXS_LEXEME_MAXBYTES];
      libxs_token_info_t info;
      if (0 == cells || 0 == payload_size
        || payload_size > sizeof(payload)
        || EXIT_SUCCESS != libxs_token_read(stream.data, stream.size,
          token_pos, payload, sizeof(payload), &info))
      {
        break;
      }
      else {
        unsigned int flags = ngram_metatoken_flags(info.kind,
          info.is_sentence, have_break);
        unsigned int id = libxs_lexicon_id(lexicon, (const char*)payload,
          (int)payload_size, flags, create);
        if (0 == id && 0 == create) {
          id = libxs_lexicon_id(lexicon, (const char*)payload,
            (int)payload_size, flags, 1);
        }
        if (0 == id) break;
        tokens[result].id = id;
        tokens[result].length = (unsigned short)payload_size;
        tokens[result].flags = (unsigned short)flags;
        if (NULL != word_ids) word_ids[result] = id;
        ++result;
        have_break = (LIBXS_TOKEN_SPACE == info.kind) ? 1 : 0;
        token_pos += cells;
      }
    }
  }
  libxs_token_stream_release(&stream);
  libxs_tokenizer_destroy(tokenizer);
  return result;
}

/**
 * Emits LIBXS_LEXEME_BREAK on a token preceded by whitespace in the source, so
 * libxs_lexeme_word_next groups the pieces of one word. Native granularity cuts
 * fixed-width chunks across word boundaries and hence marks none. When word_ids
 * is non-NULL, each piece receives the whole-word lexicon id of the word it
 * belongs to (its own id for native chunks and standalone punctuation), which
 * lets a caller build word-span context over sub-word emission.
 */
int ngram_native_tokens(libxs_lexicon_t* lexicon, const char* text,
  int text_len, libxs_lexeme_t tokens[], unsigned int word_ids[], int max,
  int create)
{
  int result = 0;
  int pos = 0;
  int have_break = 0;
  int mode = ngram_gran_mode();
  int meta_granularity = ngram_metatoken_granularity(mode);
  if (meta_granularity >= 0) {
    return ngram_metatoken_tokens(lexicon, text, text_len, tokens, word_ids,
      max, create, meta_granularity);
  }
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
      /**
       * An unknown chunk must be EMITTED with id 0, not abort the entry. When
       * scoring held-out text (create=0) a chunk absent from training is the
       * normal case, and breaking here silently truncated the entry at its first
       * novel chunk - dropping ~95% of the held-out bytes and making the BPC
       * denominator, hence BPC itself, incomparable with the other units. Id 0
       * is the reserved unknown, which the scoring loop skips as non-content,
       * so the bytes are still counted where the caller counts source bytes.
       */
      tokens[result].id = id;
      tokens[result].length = (unsigned short)len;
      tokens[result].flags = LIBXS_LEXEME_WORD;
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      pos += len;
    }
    return result;
  }
  while (pos < text_len && result < max) {
    unsigned char c = (unsigned char)text[pos];
    if (0 == ngram_is_wordchar(c)) {
      char buf[3];
      int off = 0;
      unsigned int id;
      /**
       * Whitespace is NOT emitted as a piece: the boundary it marks is already
       * carried by the leading space baked into the next piece's text (below,
       * and the same convention the byte-pair encoder uses). Emitting it here
       * too would represent one space twice, so concatenating the pieces would
       * render "to  the" and no such text occurs in any corpus.
       */
      if (0 != isspace(c)) {
        have_break = 1;
        ++pos;
        continue;
      }
      if (0 != have_break) buf[off++] = ' ';
      buf[off] = (char)c;
      id = libxs_lexicon_id(lexicon, buf, off + 1, LIBXS_LEXEME_PUNCT, create);
      if (0 == id) break;
      tokens[result].id = id;
      tokens[result].length = 1;
      tokens[result].flags = (unsigned short)(LIBXS_LEXEME_PUNCT
        | ((0 != have_break) ? LIBXS_LEXEME_BREAK : 0));
      if (NULL != word_ids) word_ids[result] = id;
      ++result;
      have_break = 0;
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
        if (0 == pi && 0 != have_break) buf[off++] = ' ';
        if (plen > (int)sizeof(buf) - 1 - off) plen = (int)sizeof(buf) - 1 - off;
        memcpy(buf + off, text + pos + piece_begin[pi], (size_t)plen);
        id = libxs_lexicon_id(lexicon, buf, off + plen,
          LIBXS_LEXEME_WORD, create);
        if (0 == id) break;
        tokens[result].id = id;
        tokens[result].length = (unsigned short)piece_len[pi];
        tokens[result].flags = (unsigned short)(LIBXS_LEXEME_WORD
          | ((0 == pi && 0 != have_break) ? LIBXS_LEXEME_BREAK : 0));
        if (NULL != word_ids) word_ids[result] = wid;
        ++result;
      }
      have_break = 0;
      pos += wlen;
    }
  }
  return result;
}


int ngram_maxorder(void)
{
  int result = (0 != converse_order_max) ? NGRAM_ORDER_MAX : 2;
  const char* env = getenv("CONVERSE_NGRAM_ORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}

/**
 * Score (and train on) each text at ONE scale only. The corpus holds every text
 * at both sentence and paragraph scale, so the default loops see each sentence
 * TWICE: once standalone and once inside its paragraph. That is multiplicity,
 * not two observations - the second copy is the same source bytes. It inflates
 * training counts and, worse, lets a paragraph copy make a sentence's own
 * contexts look attested, which is exactly the confound the slot probe already
 * filters against (it takes SCALE_SENTENCE only).
 *
 * Off by default so published figures stay reproducible; the point of the knob
 * is to measure how much the duplication was worth.
 */
int ngram_dedup_scale(void)
{
  int result = 0;
  const char* env = getenv("CONVERSE_NGRAM_ONESCALE");
  if (NULL != env && '\0' != *env) {
    const int v = atoi(env);
    if (v >= 1 && v <= 2) result = v;
    else if ('0' != env[0]) result = 1;
  }
  return result;
}

/**
 * Wall-clock per startup stage, so a slow run is attributed rather than guessed
 * at. Sec 23 blamed the hierarchy for a stall that was actually
 * converse_predict_train and lost the measurement; the fix is to make the
 * attribution a printed number instead of an inference.
 */
/**
 * The answer ranker trains OFF by default. Measured: it is 40% of wall time on a
 * 2 MB corpus, leaves BPC identical to three decimals, and leaves QA at 9/9 and
 * 14/14 - its only consumer (answer_predict_score) degrades to the unmodified
 * base score when the model is absent. A stage that costs 40% and moves nothing
 * measurable should be opt-in. CONVERSE_NO_PREDICT is still honoured so existing
 * scripts keep working.
 */
static int converse_predict_on(void)
{
  const char* env = getenv("CONVERSE_PREDICT");
  if (NULL != env) return ('0' != env[0] && '\0' != env[0]) ? 1 : 0;
  return 0;
}


static int converse_stage_on(void)
{
  const char* env = getenv("CONVERSE_STAGES");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}


void converse_stage_begin(void)
{
  if (0 != converse_stage_on()) converse_stage_tick = libxs_timer_tick();
}


void converse_stage_end(const char* name)
{
  if (0 != converse_stage_on() && converse_stage_count < CONVERSE_STAGE_MAX) {
    const libxs_timer_tick_t now = libxs_timer_tick();
    converse_stage_name[converse_stage_count] = name;
    converse_stage_time[converse_stage_count]
      = libxs_timer_duration(converse_stage_tick, now);
    ++converse_stage_count;
    converse_stage_tick = now;
  }
}


void converse_stage_report(void)
{
  if (0 != converse_stage_on() && converse_stage_count > 0) {
    double total = 0.0;
    int i;
    for (i = 0; i < converse_stage_count; ++i) {
      total += converse_stage_time[i];
    }
    fprintf(stderr, "stages (s):");
    for (i = 0; i < converse_stage_count; ++i) {
      fprintf(stderr, " %s=%.1f", converse_stage_name[i],
        converse_stage_time[i]);
    }
    fprintf(stderr, " | total=%.1f\n", total);
  }
}

/**
 * Skip-gram tier: an auxiliary store keyed by the pair (w[-3], w[-1]) that
 * abstracts over the varying middle slot, so an unseen exact context can still
 * match a seen "w ___ w" pattern (analogic generalization, no parameters).
 * Off by default -> bit-exact to the exact-only model.
 */
static int ngram_skip(void)
{
  const char* env = getenv("CONVERSE_SKIP");
  return (NULL != env && '0' != env[0] && '\0' != env[0]) ? 1 : 0;
}

/* Interpolation weight for the skip tier (0..1). */
static double ngram_skip_mu(void)
{
  double result = 0.3;
  const char* env = getenv("CONVERSE_SKIP_MU");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}

/**
 * Record a skip-gram observation from a rolling history: keys the pair
 * (hist[hlen-3], hist[hlen-1]) to the successor, requiring hlen >= 3.
 */
static void ngram_skip_observe(const unsigned int hist[], int hlen,
  unsigned int succ_id)
{
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      libxs_ngram_observe(&converse_skip, pair, 2, succ_id);
    }
  }
}

/**
 * Skip-tier probability of succ given the rolling history, or 0 when the tier
 * is off, the context is too short, or the pattern was never seen.
 */
double ngram_skip_prob(const unsigned int hist[], int hlen,
  unsigned int succ_id)
{
  double result = 0.0;
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      const libxs_ngram_entry_t* entry =
        libxs_ngram_lookup(&converse_skip, pair, 2, 2);
      if (NULL != entry && entry->total > 0) {
        unsigned int slot;
        for (slot = 0; slot < entry->nsucc; ++slot) {
          if (entry->succ[slot].id == succ_id) {
            result = (double)entry->succ[slot].count / (double)entry->total;
            break;
          }
        }
      }
    }
  }
  return result;
}

/**
 * Whether the skip tier can speak for this context at all: the pair exists and
 * was observed. Target-independent by construction, so the bank can mark the
 * slot active without reading the answer.
 */
int ngram_skip_ready(const unsigned int hist[], int hlen)
{
  int result = 0;
  if (0 != converse_skip_on && hlen >= 3) {
    unsigned int pair[2];
    pair[0] = hist[hlen - 3];
    pair[1] = hist[hlen - 1];
    if (0 != pair[0] && 0 != pair[1]) {
      const libxs_ngram_entry_t* entry =
        libxs_ngram_lookup(&converse_skip, pair, 2, 2);
      if (NULL != entry && entry->total > 0) result = 1;
    }
  }
  return result;
}


void ngram_hist_push(unsigned int hist[], int* hlen, int cap,
  unsigned int id)
{
  if (*hlen < cap) hist[(*hlen)++] = id;
  else {
    int s;
    for (s = 1; s < cap; ++s) hist[s - 1] = hist[s];
    hist[cap - 1] = id;
  }
}

/**
 * Word-span context for predicting sub-word token i: the whole-word ids of the
 * preceding wctx words followed by the pieces of the current word emitted so
 * far, kept as the most-recent cap entries. Rebuilt per position so that train
 * and eval derive identical keys; groups are delimited by LIBXS_LEXEME_BREAK.
 */
int ngram_wordctx_hist(const libxs_lexeme_t nat[],
  const unsigned int word_ids[], int i, int wctx, unsigned int hist[], int cap)
{
  int hlen = 0;
  int wstart = i;
  unsigned int words[NGRAM_ORDER_MAX];
  int nw = 0, p, k;
  while (wstart > 0 && 0 == (nat[wstart].flags & LIBXS_LEXEME_BREAK)) --wstart;
  p = wstart;
  while (p > 0 && nw < wctx && nw < (int)(sizeof(words) / sizeof(*words))) {
    int ws = p - 1;
    while (ws > 0 && 0 == (nat[ws].flags & LIBXS_LEXEME_BREAK)) --ws;
    words[nw++] = word_ids[ws];
    p = ws;
  }
  for (k = nw - 1; k >= 0; --k) ngram_hist_push(hist, &hlen, cap, words[k]);
  for (k = wstart; k < i; ++k) ngram_hist_push(hist, &hlen, cap, nat[k].id);
  return hlen;
}

/* Per-order footprint of the n-gram store (gated by CONVERSE_NGRAM_STATS). */
void ngram_stats(const libxs_registry_t* model)
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


/**
 * Training reads the vocabulary rather than extending it. The text is corpus
 * text, so ingest has interned every id this needs, and creating here would
 * count each occurrence again on every run: the model is rebuilt per run while
 * the corpus is not, so libxs_lexicon_count would grow without the corpus
 * growing. Every other pass over stored text already looks up.
 */
static void ngram_train_text(libxs_registry_t* model,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  int maxorder = ngram_maxorder();
  if (0 != ngram_native_mode()) {
    int wctx = ngram_wordctx();
    libxs_lexeme_t nat[COMPOSE_MAXTEXT];
    unsigned int word_ids[COMPOSE_MAXTEXT];
    int ntok = ngram_native_tokens(lexicon, text, text_len, nat,
      (0 != wctx) ? word_ids : NULL, COMPOSE_MAXTEXT, 0);
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
        answer_lexnorms, answer_lexnorms_size, 0))
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
          ngram_skip_observe(hist, hlen, lex->id);
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


int predict_is_test(long index, int holdout)
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


libxs_registry_t* ngram_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  libxs_ngram_destroy(&converse_ngram);
  if (0 != converse_skip_on) libxs_ngram_destroy(&converse_skip);
  converse_skip_on = ngram_skip();
  if (EXIT_SUCCESS != libxs_ngram_create(&converse_ngram, ngram_maxorder())) {
    return NULL;
  }
  if (0 != converse_skip_on
    && EXIT_SUCCESS != libxs_ngram_create(&converse_skip, 2))
  {
    converse_skip_on = 0;
  }
  if (NULL != corpus) {
    const void* key = NULL;
    size_t cursor = 0;
    corpus_entry_t scratch;
    long index = 0;
    const int onescale = ngram_dedup_scale();
    void* value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
      if (0 == predict_is_test(index, holdout)
        && (0 == onescale
          || (((2 == onescale) ? SCALE_PARAGRAPH : SCALE_SENTENCE)
              == entry->scale
            && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))))
      {
        ngram_train_text(converse_ngram.store, lexicon, rules, nrules,
          entry->text, entry->text_len);
      }
      ++index;
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
  }
  return converse_ngram.store;
}


void ngram_backoff_build(libxs_registry_t* model,
  const libxs_lexicon_t* lexicon)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  LIBXS_UNUSED(model);
  libxs_ngram_finalize(&converse_ngram, vocab);
  if (0 != converse_skip_on) libxs_ngram_finalize(&converse_skip, vocab);
}


static void token_emb_free(void)
{
  free(token_emb);
  free(token_semb);
  free(token_emb_prior);
  free(token_emb_work);
  token_emb = NULL;
  token_semb = NULL;
  token_emb_prior = NULL;
  token_emb_work = NULL;
  token_emb_size = 0;
  token_emb_zvalid = 0;
}

/**
 * Maximum FORWARD distance at which a co-occurrence is counted, or 0 for the
 * historical symmetric window.
 *
 * The nine flat axes all varied how the SYMMETRIC PPMI factorization is built or
 * searched - radius, rank, iterations, hashing, weighting, temperature, heads,
 * projections - and never which matrix is factorized. So the objective itself
 * was never a variable: the representation is fitted to "what appears near
 * what" and then asked "what comes next here". At 1 the matrix becomes
 * PPMI(next=j | cur=i), whose row space places two tokens together when they are
 * FOLLOWED by similar things, which is the predictive geometry rather than the
 * topical one. It discards distances 2..radius, which the flat radius axis
 * (1..5 all 48.7% top-1) measured at zero.
 */
int token_emb_directed(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_DIRECTED");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
    if (cached > TOKEN_EMB_WINDOW) cached = TOKEN_EMB_WINDOW;
  }
  return cached;
}

/**
 * Count ONLY forward distance exactly CONVERSE_EMB_DIRECTED, instead of pooling
 * 1..N.
 *
 * This exists to answer one question before a much larger build. Extending the
 * slot's context by summing vectors is refuted: the optimal weight on a token
 * further back is zero, because token_emb[p] answers "what follows p" and asking
 * that of a token k positions back is a question about the wrong relation. The
 * principled repair is a SEPARATE factorization per distance, so distance k's
 * evidence is about position k. That costs k SVDs and k vocabulary-sized tables,
 * so it should only be built if a distance-k-only model carries real information
 * about the successor at all - which is what this measures.
 */
static int token_emb_distonly(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_EMB_DISTONLY");
    cached = (NULL != env && '\0' != *env && 0 != atoi(env)) ? 1 : 0;
  }
  return cached;
}


static int token_emb_ready(void)
{
  return (NULL != token_emb && 0 != token_emb_size) ? 1 : 0;
}

/** Successor-side row of V: the geometry of what PRECEDES this token. */
static const double* token_semb_get(unsigned int id)
{
  static const double zero[TOKEN_EMB_DIM] = { 0 };
  if (NULL != token_semb && 0 != id && id <= token_emb_size) {
    return token_semb + (size_t)id * TOKEN_EMB_DIM;
  }
  return zero;
}


const double* token_emb_get(unsigned int id)
{
  static const double zero[TOKEN_EMB_DIM] = { 0 };
  if (NULL != token_emb && 0 != id && id <= token_emb_size) {
    return token_emb + (size_t)id * TOKEN_EMB_DIM;
  }
  return zero;
}

/**
 * Non-zero if the id has no usable embedding: out of range, or an all-zero row
 * (a word that never occurred in the training split has an empty PPMI row, so
 * it contributes nothing to a retrieval query). This is the headroom available
 * to subword composition.
 */
int token_emb_isnull(unsigned int id)
{
  const double* emb = token_emb_get(id);
  int d, result = 1;
  for (d = 0; d < TOKEN_EMB_DIM && 0 != result; ++d) {
    if (0.0 != emb[d]) result = 0;
  }
  return result;
}


double ngram_emb_temp(void)
{
  static int done = 0;
  static double cached = 1.0;
  if (0 == done) {
    const char* env = getenv("CONVERSE_EMB_TEMP");
    const double v = (NULL != env && '\0' != *env) ? atof(env) : 1.0;
    cached = (v > 0.0) ? v : 1.0;
    done = 1;
  }
  return cached;
}

/** Per-position weight decay, oldest token weighted decay^(distance-1). */
static double ngram_emb_decay(void)
{
  static int done = 0;
  static double cached = 1.0;
  if (0 == done) {
    const char* env = getenv("CONVERSE_EMB_DECAY");
    const double v = (NULL != env && '\0' != *env) ? atof(env) : 1.0;
    cached = (v > 0.0 && v <= 1.0) ? v : 1.0;
    done = 1;
  }
  return cached;
}

/**
 * P(cand | ctx) from the successor score, as a proper distribution over the
 * whole vocabulary.
 *
 * The link is not an arbitrary softmax. PMI is a log ratio,
 * PMI(p,c) = log[P(c|p)/P(c)], so exponentiating it RESCALES THE UNIGRAM PRIOR:
 * P(c|p) = P(c) * exp(PMI(p,c)) / Z. Two things follow that a bare softmax over
 * the scores would not give. Where the reconstruction says nothing the expert
 * degrades exactly to the unigram prior rather than to uniform, which is what
 * lets it be TOTAL without being absurd on common words; and the temperature has
 * a defined neutral value of 1 instead of needing to be fitted before the
 * mechanism can be judged at all.
 *
 * Cost is one vocabulary pass with an exp per entry, per position. That is the
 * price of a total scorer and the reason this is behind a slot knob.
 */
static int token_emb_succ_prepare(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp)
{
  int result = 0;
  if (NULL != token_emb && NULL != token_semb && NULL != token_emb_prior
    && NULL != token_emb_work && vocab <= token_emb_size && 0 < nctx
    && nctx <= TOKEN_CTX_MAX)
  {
    int same = (0 != token_emb_zvalid && vocab == token_emb_zvocab
      && temp == token_emb_ztemp && nctx == token_emb_znctx) ? 1 : 0;
    int i;
    for (i = 0; i < nctx && 0 != same; ++i) {
      if (ctx[i] != token_emb_zids[i]) same = 0;
    }
    if (0 == same) {
      const double scale = (temp > 0.0) ? (1.0 / temp) : 1.0;
      const double decay = ngram_emb_decay();
      double u[TOKEN_EMB_DIM];
      double max = 0.0, sum = 0.0, w = 1.0;
      unsigned int id, nvalid = 0;
      int d;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) u[d] = 0.0;
      /* Walk from the immediate predecessor backwards so weight 1 is the nearest
       * token and the decay applies to older testimony. */
      for (i = nctx - 1; i >= 0; --i) {
        const double* e = token_emb_get(ctx[i]);
        for (d = 0; d < TOKEN_EMB_DIM; ++d) u[d] += w * e[d];
        w *= decay;
      }
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0) {
          const double* v = token_semb + (size_t)id * TOKEN_EMB_DIM;
          double x = 0.0;
          for (d = 0; d < TOKEN_EMB_DIM; ++d) x += u[d] * v[d];
          token_emb_work[id] = x * scale + log(token_emb_prior[id]);
          if (0 == nvalid || token_emb_work[id] > max) max = token_emb_work[id];
          ++nvalid;
        }
      }
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0) sum += exp(token_emb_work[id] - max);
      }
      for (i = 0; i < nctx; ++i) token_emb_zids[i] = ctx[i];
      token_emb_znctx = nctx;
      token_emb_zvocab = vocab;
      token_emb_ztemp = temp;
      token_emb_zmax = max;
      token_emb_zsum = sum;
      token_emb_zvalid = 1;
    }
    result = (token_emb_zsum > 0.0) ? 1 : 0;
  }
  return result;
}


double token_emb_succ_prob(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab, double temp)
{
  double result = 0.0;
  if (0 != cand && cand <= vocab
    && 0 != token_emb_succ_prepare(ctx, nctx, vocab, temp)
    && token_emb_prior[cand] > 0.0)
  {
    result = exp(token_emb_work[cand] - token_emb_zmax) / token_emb_zsum;
  }
  return result;
}

/**
 * Append up to want embedding-proposed successors to the candidate list, best
 * first, skipping ids already offered by the counts.
 *
 * This is what makes the successor embedding reach generation at all. The bank
 * can only REORDER what it is given, so with an attested-only candidate list its
 * measured ceiling is the inlist share (22.23% on novel context) no matter how
 * good the expert is. Proposing is a different act from ranking, and only this
 * one can put a successor in front of the caller that the counts never saw here.
 *
 * The selection reads token_emb_work, which the memo has already filled with the
 * whole log-distribution for this context, so no dot product is repeated: it is
 * want passes of a comparison over the vocabulary.
 */
int token_emb_succ_append(const unsigned int ctx[], int nctx,
  unsigned int vocab, double temp, unsigned int ids[], int n, int max, int want)
{
  int result = 0;
  if (0 != token_emb_succ_prepare(ctx, nctx, vocab, temp)) {
    int more = 1;
    while (0 != more && result < want && (n + result) < max) {
      unsigned int best = 0;
      double bestval = 0.0;
      unsigned int id;
      for (id = 1; id <= vocab; ++id) {
        if (token_emb_prior[id] > 0.0
          && (0 == best || token_emb_work[id] > bestval))
        {
          int taken = 0, k;
          for (k = 0; k < n + result && 0 == taken; ++k) {
            if (ids[k] == id) taken = 1;
          }
          if (0 == taken) {
            best = id;
            bestval = token_emb_work[id];
          }
        }
      }
      if (0 != best) {
        ids[n + result] = best;
        ++result;
      }
      else more = 0;
    }
  }
  return result;
}

/**
 * Rank of cand among all vocabulary ids by the successor score, 0 = best. No
 * sort: the rank is the number of ids that strictly outscore it, which is one
 * pass and needs no candidate list at all - the point being that this scorer is
 * TOTAL where the count model is partial.
 */
int token_emb_succ_rank(const unsigned int ctx[], int nctx,
  unsigned int cand, unsigned int vocab)
{
  int result = -1;
  /* Shares the prepared distribution so the probe scores exactly what the slot
   * scores: a probe that built its own context vector could report a mechanism
   * nobody is running. */
  if (0 != cand && cand <= vocab
    && 0 != token_emb_succ_prepare(ctx, nctx, vocab, ngram_emb_temp())
    && token_emb_prior[cand] > 0.0)
  {
    const double best = token_emb_work[cand];
    unsigned int id;
    int rank = 0;
    for (id = 1; id <= vocab; ++id) {
      if (id != cand && token_emb_prior[id] > 0.0
        && token_emb_work[id] > best)
      {
        ++rank;
      }
    }
    result = rank;
  }
  return result;
}


static void token_emb_pair_observe(libxs_registry_t* pairs,
  unsigned int center, unsigned int context)
{
  token_emb_pair_t key;
  double* count;
  key.center = center;
  key.context = context;
  count = (double*)libxs_registry_get(pairs, &key, sizeof(key), NULL);
  if (NULL != count) *count += 1.0;
  else {
    double one = 1.0;
    libxs_registry_set(pairs, &key, sizeof(key), &one, sizeof(one), NULL);
  }
}


static void token_emb_cooc_text(libxs_registry_t* pairs, double* rowcnt,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* text, int text_len)
{
  libxs_lexeme_stream_t stream;
  libxs_lexeme_stream_init(&stream);
  if (NULL != pairs && NULL != lexicon && NULL != rules && nrules > 0
    && text_len > 0 && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
      &stream, (const unsigned char*)text, (size_t)text_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    /* Trailing buffer of the last TOKEN_EMB_WINDOW ids: each token pairs with
     * every buffered predecessor and both directions are counted, so the union
     * over positions is a symmetric window of that same radius (no 2*W+1). */
    unsigned int window[TOKEN_EMB_WINDOW];
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
        const int dir = token_emb_directed();
        const int distonly = token_emb_distonly();
        int i;
        for (i = 0; i < fill; ++i) {
          if (0 == dir) {
            token_emb_pair_observe(pairs, lex->id, window[i]);
            token_emb_pair_observe(pairs, window[i], lex->id);
          }
          else if ((0 == distonly && (fill - i) <= dir)
            || (0 != distonly && (fill - i) == dir))
          {
            /* Row = predecessor, column = successor, so the row of the matrix
             * is the successor profile of that token and nothing symmetrizes
             * it. window[fill-1] is the immediate predecessor. */
            token_emb_pair_observe(pairs, window[i], lex->id);
          }
        }
        if (fill < TOKEN_EMB_WINDOW) {
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

/* Multiply the sparse PPMI matrix (CSR) by a dense (vocab+1) x DIM block:
 * out = A * in when transpose is zero, out = A^T * in otherwise. */
static void token_emb_spmm(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab, const double* in, int transpose,
  double* out)
{
  size_t n = ((size_t)vocab + 1) * TOKEN_EMB_DIM, k;
  unsigned int id;
  for (k = 0; k < n; ++k) out[k] = 0.0;
  for (id = 1; id <= vocab; ++id) {
    size_t nz;
    for (nz = rowptr[id]; nz < rowptr[id + 1]; ++nz) {
      const double v = val[nz];
      if (0.0 != v) {
        const size_t src = (size_t)(0 != transpose ? id : colidx[nz])
          * TOKEN_EMB_DIM;
        const size_t dst = (size_t)(0 != transpose ? colidx[nz] : id)
          * TOKEN_EMB_DIM;
        int d;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) out[dst + d] += v * in[src + d];
      }
    }
  }
}

/* Gram-Schmidt orthonormalization of the DIM columns of a (vocab+1) x DIM
 * block, in place. */
static void token_emb_orthonormalize(double* block, unsigned int vocab)
{
  const size_t rows = (size_t)vocab + 1;
  int d, e;
  for (d = 0; d < TOKEN_EMB_DIM; ++d) {
    double norm = 0.0;
    size_t i;
    for (e = 0; e < d; ++e) {
      double dot = 0.0;
      for (i = 0; i < rows; ++i) {
        dot += block[i * TOKEN_EMB_DIM + d] * block[i * TOKEN_EMB_DIM + e];
      }
      for (i = 0; i < rows; ++i) {
        block[i * TOKEN_EMB_DIM + d] -= dot * block[i * TOKEN_EMB_DIM + e];
      }
    }
    for (i = 0; i < rows; ++i) {
      norm += block[i * TOKEN_EMB_DIM + d] * block[i * TOKEN_EMB_DIM + d];
    }
    norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
    for (i = 0; i < rows; ++i) block[i * TOKEN_EMB_DIM + d] *= norm;
  }
}

/**
 * Truncated SVD of the sparse PPMI matrix by subspace iteration on A^T A,
 * which is never materialized: each iteration applies A and then A^T, so the
 * cost is O(nnz * DIM) and the full context vocabulary is carried exactly.
 * Embeddings are the projected rows A*V (that is U*Sigma), row-normalized.
 */
static int token_emb_reduce(const size_t* rowptr, const unsigned int* colidx,
  const double* val, unsigned int vocab)
{
  int result = EXIT_FAILURE;
  const size_t block = ((size_t)vocab + 1) * TOKEN_EMB_DIM;
  double* basis = (double*)malloc(block * sizeof(double));
  double* work = (double*)malloc(block * sizeof(double));
  if (NULL != basis && NULL != work) {
    /* Symmetry-breaking start: LIBXS_SHUFFLE_INDEX is a coprime affine map,
     * hence a bijection onto [0, block), so the values are an evenly spread
     * (not clustered) permutation and reproducible across runs. */
    const size_t stride = libxs_coprime_bias(block, -1.0);
    unsigned int id, iter;
    size_t i;
    int d;
    for (i = 0; i < block; ++i) {
      const size_t k = LIBXS_SHUFFLE_INDEX(i, block, stride, 1);
      basis[i] = (2.0 * (double)k / (double)block) - 1.0;
    }
    token_emb_orthonormalize(basis, vocab);
    for (iter = 0; iter < TOKEN_EMB_ITER; ++iter) {
      token_emb_spmm(rowptr, colidx, val, vocab, basis, 0, work);
      token_emb_spmm(rowptr, colidx, val, vocab, work, 1, basis);
      token_emb_orthonormalize(basis, vocab);
    }
    token_emb_spmm(rowptr, colidx, val, vocab, basis, 0, work);
    for (id = 1; id <= vocab; ++id) {
      const double* row = work + (size_t)id * TOKEN_EMB_DIM;
      double* emb = token_emb + (size_t)id * TOKEN_EMB_DIM;
      double norm = 0.0;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) norm += row[d] * row[d];
      norm = (norm > 0.0) ? (1.0 / sqrt(norm)) : 0.0;
      for (d = 0; d < TOKEN_EMB_DIM; ++d) emb[d] = row[d] * norm;
    }
    /**
     * basis still holds V after the final projection, so the successor side is
     * free: it was already computed and then thrown away with the scratch. With
     * the directed matrix <token_emb[p], token_semb[c]> is the rank-DIM
     * reconstruction of PPMI(c follows p), and a low-rank completion is nonzero
     * on pairs NEVER OBSERVED - the one property a count model cannot have.
     *
     * Kept UNNORMALIZED while token_emb rows are unit-normalized, so the product
     * is not calibrated as a probability: within one context the row scale is a
     * positive constant and drops out of any ranking, but the magnitude that was
     * normalized away was confidence, and a fixed temperature does not recover
     * it.
     */
    if (NULL != token_semb) {
      for (id = 1; id <= vocab; ++id) {
        const double* row = basis + (size_t)id * TOKEN_EMB_DIM;
        double* semb = token_semb + (size_t)id * TOKEN_EMB_DIM;
        for (d = 0; d < TOKEN_EMB_DIM; ++d) semb[d] = row[d];
      }
    }
    result = EXIT_SUCCESS;
  }
  free(basis);
  free(work);
  return result;
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

/**
 * Print the split of comma-separated words, so the splitter can be inspected on
 * the words it is known to get wrong without running a corpus. The unit was
 * never fairly evaluated because the output was broken; a probe makes the repair
 * checkable in one command instead of inferred from a BPC delta.
 */
static void ngram_syllable_probe(void)
{
  const char* probe = getenv("CONVERSE_SYLLABLE_PROBE");
  while (NULL != probe && '\0' != *probe) {
    const char* end = strchr(probe, ',');
    const int len = (NULL != end) ? (int)(end - probe) : (int)strlen(probe);
    if (0 < len && LIBXS_LEXEME_MAXBYTES >= len) {
      int begin[LIBXS_LEXEME_MAXBYTES], plen[LIBXS_LEXEME_MAXBYTES];
      const int np = ngram_syllable_split(probe, len, begin, plen,
        LIBXS_LEXEME_MAXBYTES);
      int k;
      fprintf(stderr, "syllable[%.*s]:", len, probe);
      for (k = 0; k < np; ++k) {
        fprintf(stderr, "%s%.*s", (0 < k) ? "|" : " ", plen[k],
          probe + begin[k]);
      }
      fprintf(stderr, " (%d pieces)\n", np);
    }
    probe = (NULL != end) ? (end + 1) : NULL;
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


void token_emb_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int holdout)
{
  unsigned int vocab = (NULL != lexicon) ? libxs_lexicon_size(lexicon) : 0;
  token_emb_free();
  if (NULL != corpus && vocab > 0) {
    libxs_registry_t* pairs = libxs_registry_create();
    double* rowcnt = (double*)calloc((size_t)vocab + 1, sizeof(double));
    double* rowsum = (double*)calloc((size_t)vocab + 1, sizeof(double));
    double* colsum = (double*)calloc((size_t)vocab + 1, sizeof(double));
    size_t* rowptr = (size_t*)calloc((size_t)vocab + 2, sizeof(size_t));
    token_emb = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_DIM,
      sizeof(double));
    token_semb = (double*)calloc((size_t)(vocab + 1) * TOKEN_EMB_DIM,
      sizeof(double));
    token_emb_prior = (double*)calloc((size_t)vocab + 1, sizeof(double));
    token_emb_work = (double*)calloc((size_t)vocab + 1, sizeof(double));
    if (NULL != pairs && NULL != rowcnt && NULL != rowsum && NULL != colsum
      && NULL != rowptr && NULL != token_emb && NULL != token_semb
      && NULL != token_emb_prior && NULL != token_emb_work)
    {
      double total = 0.0;
      const void* key = NULL;
      size_t cursor = 0, nnz = 0;
      long index = 0;
      unsigned int id;
      corpus_entry_t scratch;
      void* value = corpus_iterx_begin(corpus, &key, &cursor);
      token_emb_size = vocab;
      while (NULL != value) {
        const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
        if (0 == predict_is_test(index, holdout)) {
          token_emb_cooc_text(pairs, rowcnt, lexicon, rules, nrules,
            entry->text, entry->text_len);
        }
        ++index;
        value = corpus_iterx_next(corpus, &key, &cursor);
      }
      /* rowcnt was already accumulated for the backfill, so the unigram prior
       * the successor distribution rescales costs one normalization. */
      { double ntok = 0.0;
        for (id = 1; id <= vocab; ++id) ntok += rowcnt[id];
        if (ntok > 0.0) {
          for (id = 1; id <= vocab; ++id) {
            token_emb_prior[id] = rowcnt[id] / ntok;
          }
        }
      }
      key = NULL;
      cursor = 0;
      value = libxs_registry_begin(pairs, &key, &cursor);
      while (NULL != value) {
        const token_emb_pair_t* pair = (const token_emb_pair_t*)key;
        const double count = *(const double*)value;
        if (pair->center <= vocab && pair->context <= vocab) {
          rowsum[pair->center] += count;
          colsum[pair->context] += count;
          total += count;
          ++rowptr[pair->center + 1];
          ++nnz;
        }
        value = libxs_registry_next(pairs, &key, &cursor);
      }
      for (id = 1; id <= vocab + 1; ++id) rowptr[id] += rowptr[id - 1];
      if (total > 0.0 && nnz > 0) {
        unsigned int* colidx = (unsigned int*)malloc(nnz * sizeof(*colidx));
        double* val = (double*)malloc(nnz * sizeof(*val));
        size_t* fill = (size_t*)malloc(((size_t)vocab + 1) * sizeof(*fill));
        if (NULL != colidx && NULL != val && NULL != fill) {
          const char* env;
          for (id = 0; id <= vocab; ++id) fill[id] = rowptr[id];
          key = NULL;
          cursor = 0;
          value = libxs_registry_begin(pairs, &key, &cursor);
          while (NULL != value) {
            const token_emb_pair_t* pair = (const token_emb_pair_t*)key;
            const double count = *(const double*)value;
            if (pair->center <= vocab && pair->context <= vocab
              && rowsum[pair->center] > 0.0 && colsum[pair->context] > 0.0)
            {
              const double pmi = log((count * total)
                / (rowsum[pair->center] * colsum[pair->context]));
              const size_t at = fill[pair->center]++;
              colidx[at] = pair->context;
              val[at] = (pmi > 0.0) ? pmi : 0.0;
            }
            value = libxs_registry_next(pairs, &key, &cursor);
          }
          fprintf(stderr, "embedding: vocab=%u nnz=%lu density=%.4f%%"
            " dim=%d window=%d directed=%d%s\n", vocab, (unsigned long)nnz,
            100.0 * (double)nnz / ((double)vocab * (double)vocab),
            TOKEN_EMB_DIM, TOKEN_EMB_WINDOW, token_emb_directed(),
            (0 != token_emb_distonly()) ? " (exact distance)" : "");
          if (EXIT_SUCCESS == token_emb_reduce(rowptr, colidx, val, vocab)) {
            env = getenv("CONVERSE_EMB_BACKFILL");
            if (NULL == env || '0' != *env) {
              token_emb_backfill(lexicon, rowcnt, vocab);
            }
            token_emb_probe(lexicon, vocab);
          }
        }
        free(colidx);
        free(val);
        free(fill);
      }
    }
    libxs_registry_destroy(pairs);
    free(rowcnt);
    free(rowsum);
    free(colsum);
    free(rowptr);
  }
}

/**
 * Fill hist[] with the trailing content-token ids of the prompt (at most
 * NGRAM_ORDER_MAX), returning the count. Mirrors ngram_last_context but keeps
 * the whole window the deep store can use, not just the final two ids.
 */
static int ngram_history(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, const char* text, int text_len, unsigned int hist[])
{
  libxs_lexeme_stream_t stream;
  const int native = ngram_native_mode();
  int hlen = 0;
  libxs_lexeme_stream_init(&stream);
  /**
   * The prompt has to be read in the unit the model was trained on, or the
   * seeded history holds ids from a different vocabulary and nothing matches.
   */
  if (0 != native) {
    libxs_lexeme_t nat[COMPOSE_MAXTEXT];
    const int ntok = (NULL != lexicon && 0 < text_len)
      ? ngram_native_tokens(lexicon, text, text_len, nat, NULL,
        COMPOSE_MAXTEXT, 0) : 0;
    int ti;
    for (ti = 0; ti < ntok; ++ti) {
      if (0 != nat[ti].id) {
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, nat[ti].id);
      }
    }
  }
  else if (NULL != lexicon && NULL != rules && nrules > 0 && text_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &stream,
      (const unsigned char*)text, (size_t)text_len, rules, nrules,
      answer_lexnorms, answer_lexnorms_size, 0))
  {
    size_t pos;
    for (pos = 0; pos < stream.size; ++pos) {
      const libxs_lexeme_t* lex = stream.data + pos;
      if (0 != (lex->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
        && 0 != lex->id)
      {
        ngram_hist_push(hist, &hlen, NGRAM_ORDER_MAX, lex->id);
      }
    }
  }
  libxs_lexeme_stream_release(&stream);
  return hlen;
}

/**
 * Successors offered to the hierarchical model at each generation step (1 = the
 * count-ranked winner, i.e. plain greedy decoding). Widening this is what gives
 * the byte model anything to say about generation: it never proposes a
 * continuation, it only chooses among ones the n-gram already attested, so the
 * grounding gate below still decides what may be shown.
 */
int ngram_gen_ncand(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_GEN_NCAND");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 1;
    if (cached < 1) cached = 1;
    if (cached > GEN_CAND_MAX) cached = GEN_CAND_MAX;
  }
  return cached;
}

/**
 * Whether rendering has to supply the word separator itself.
 *
 * The modeling units (native chunks, syllable pieces, byte-pair pieces) bake the
 * boundary into the piece text, so concatenating them reproduces the source and
 * a separator here would duplicate it. The lexical view cannot: word ids come
 * from the shared normalized vocabulary that grounded QA matches terms against,
 * which must stay bare. So this is not a granularity table but the one-bit
 * question "modeling view or lexical view", and it belongs in exactly one place:
 * every site that re-derived it was a chance to render text no corpus contains.
 */
static int ngram_render_separated(void)
{
  return (0 == ngram_native_mode()) ? 1 : 0;
}

/**
 * Append one piece's text at pos, inserting the separator first when the view
 * needs one and the piece may be preceded by something. Returns the new end, or
 * pos unchanged when the piece does not fit or has no text.
 */
size_t ngram_render_append(char* out, size_t out_size, size_t pos,
  const char* piece, int len, int leading)
{
  size_t result = pos;
  if (NULL != out && NULL != piece && 0 < len
    && pos + (size_t)len + 1 < out_size)
  {
    if (0 != leading && 0 != ngram_render_separated()) out[result++] = ' ';
    memcpy(out + result, piece, (size_t)len);
    result += (size_t)len;
    out[result] = '\0';
  }
  return result;
}

/**
 * Choose among the count-ranked successors with the byte model, returning the
 * index to take. Falls back to the n-gram's own first choice whenever the model
 * is absent or nothing could be scored, so enabling this can only reorder
 * candidates the n-gram already offered.
 */
int ngram_gen_select(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int nids, const char* context, int context_len)
{
  int result = 0;
  if (1 < nids && 0 != converse_judge_active()) {
    const char* candidates[GEN_CAND_MAX];
    int lengths[GEN_CAND_MAX];
    char words[GEN_CAND_MAX][64];
    int slot, chosen;
    for (slot = 0; slot < nids; ++slot) {
      int len = 0;
      const char* word = libxs_lexicon_text(lexicon, ids[slot], &len, NULL);
      /* A candidate always follows the context, so it always takes a leading. */
      const size_t at = ngram_render_append(words[slot], sizeof(words[slot]), 0,
        word, len, 1);
      candidates[slot] = (0 < at) ? words[slot] : NULL;
      lengths[slot] = (int)at;
    }
    chosen = converse_judge_choose(context, context_len, candidates,
      lengths, nids);
    if (0 <= chosen) result = chosen;
  }
  return result;
}


int ngram_gen_minorder(void)
{
  int result = 2;
  const char* env = getenv("CONVERSE_GEN_MINORDER");
  if (NULL != env && '\0' != *env) {
    int v = atoi(env);
    if (v >= 1 && v <= NGRAM_ORDER_MAX) result = v;
  }
  return result;
}

/**
 * Grounded greedy generation over the deep k-context store. Each step takes
 * the top successor from the longest attested context and reports that
 * context order; generation stops when the order falls below the grounding
 * floor (the generative form of "abstain rather than invent"), at a repeat,
 * or at the length budget. Prints the continuation plus the mean/min order
 * that quantifies how well grounded it was.
 */
/**
 * Greedy grounded continuation of the prompt into out[] (space-joined words).
 * Returns the number of generated tokens (0 = nothing cleared the grounding
 * floor). Optional order_mean/order_min report how attested the run was.
 */
int ngram_generate(libxs_registry_t* model, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  char* out, size_t out_size, double* order_mean, int* order_min_out)
{
  enum { GEN_MAX = 40 };
  unsigned int hist[NGRAM_ORDER_MAX];
  unsigned int recent[GEN_MAX];
  int hlen, maxorder = ngram_maxorder();
  int minorder = ngram_gen_minorder();
  int step, nrecent = 0;
  long order_sum = 0;
  int order_min = NGRAM_ORDER_MAX + 1;
  size_t pos = 0;
  if (NULL != out && out_size > 0) out[0] = '\0';
  hlen = ngram_history(lexicon, rules, nrules, text, text_len, hist);
  for (step = 0; hlen > 0 && step < GEN_MAX; ++step) {
    unsigned int ids[GEN_CAND_MAX];
    int got_order = 0, len = 0, i, repeat = 0, pick = 0;
    const char* word;
    int n = ngramk_predict_order(model, hist, hlen, maxorder, ids,
      ngram_gen_ncand(), &got_order);
    if (n <= 0 || got_order < minorder) break;
    if (1 < n && 0 != converse_judge_active()) {
      char context[COMPOSE_MAXTEXT];
      int context_len = text_len;
      if (context_len >= (int)sizeof(context)) {
        context_len = (int)sizeof(context) - 1;
      }
      memcpy(context, text + text_len - context_len, (size_t)context_len);
      if (NULL != out && 0 < pos) {
        context_len = (int)ngram_render_append(context, sizeof(context),
          (size_t)context_len, out, (int)pos, 1);
      }
      pick = ngram_gen_select(lexicon, ids, n, context, context_len);
    }
    if (0 != pick) {
      const unsigned int chosen = ids[pick];
      ids[pick] = ids[0];
      ids[0] = chosen;
    }
    for (i = 0; i < nrecent; ++i) if (recent[i] == ids[0]) ++repeat;
    if (repeat >= 3) break;
    word = libxs_lexicon_text(lexicon, ids[0], &len, NULL);
    if (NULL == word || len <= 0) break;
    pos = ngram_render_append(out, out_size, pos, word, len,
      (0 < pos) ? 1 : 0);
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


void ngram_complete(libxs_registry_t* model, libxs_lexicon_t* lexicon,
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

/**
 * Store the clause windows of a span as sentence-scale fragments; returns how
 * many were stored.
 *
 * This was TWO identical loops, and they cut clauses for two unrelated reasons:
 * the clauses of an over-long SENTENCE, which have no other representation at all
 * because the whole-sentence branch requires a length below COMPOSE_MAXTEXT, and
 * the clauses of a long PARAGRAPH, whose bytes the paragraph entry already covers.
 * They differ only in their bound and their section. Writing the cutting rule
 * twice is how two definitions of one thing begin, which this line has now paid
 * for more than once.
 *
 * The windows OVERLAP by construction: each starts at a clause boundary and runs
 * to the LAST boundary within 240 bytes, so a clause belongs to every window that
 * reaches it. They are recombination's donor pool - measured, not assumed:
 * withholding the paragraph ones costs a third of the splice yield and twenty
 * points of same-section coherence while every eval stays green.
 *
 * `derive` stores them as SPANS of one parent text instead of as entries in their
 * own right, which is what the two reasons above finally buy: the windows of a long
 * paragraph are byte ranges of a text that is worth keeping once, while the windows
 * of an over-long sentence are the only copy of theirs and stay entries. The parent
 * is written on the first window rather than up front, so a paragraph that yields
 * none costs nothing.
 */
static int corpus_store_clauses(libxs_registry_t* corpus,
  const unsigned char* text, size_t begin, size_t end, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int derive)
{
  int result = 0;
  unsigned int parent = 0;
  size_t frag_start = begin;
  while (frag_start < end) {
    size_t scan = frag_start;
    size_t frag_end = frag_start;
    while (scan < end && scan - frag_start < 240) {
      if (',' == text[scan] || ';' == text[scan] || ':' == text[scan]
        || '.' == text[scan] || '?' == text[scan] || '!' == text[scan])
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
        && 0 != isspace(text[trim_start + frag_len - 1]))
      {
        --frag_len;
      }
      if (frag_len > 24 && frag_len < COMPOSE_MAXTEXT
        && count_words(text + trim_start, frag_len) >= 4)
      {
        corpus_entry_t entry;
        if (0 != derive && 0 == parent) {
          parent = corpus_store_blob(text + begin, (int)(end - begin),
            section, section_len);
        }
        if (EXIT_SUCCESS == corpus_entry_build(&entry, text + trim_start,
          frag_len, SCALE_SENTENCE, lexicon, rules, nrules, 1))
        {
          corpus_span_t span;
          entry.lexical_flags |= ENTRY_LEX_FRAGMENT;
          entry.line = corpus_line_at(text, end, trim_start);
          corpus_entry_set_section(&entry, section, section_len);
          memset(&span, 0, sizeof(span));
          span.kind = (unsigned short)ENTRY_KIND_SPAN;
          span.connector = entry.connector;
          span.scale = entry.scale;
          span.text_len = frag_len;
          span.parent = parent;
          span.offset = (unsigned int)(trim_start - begin);
          span.line = entry.line;
          if (0 != parent && 0 != corpus_span_check()) {
            corpus_entry_t rebuilt;
            if (EXIT_SUCCESS != corpus_span_build(&span, &rebuilt)
              || 0 != libxs_memcmp(&rebuilt, &entry, corpus_entry_size(&entry)))
            {
              ++corpus_span_mismatch;
            }
          }
          if (0 != corpus_store_record(corpus, &entry,
            (0 != parent) ? &span : NULL)) ++result;
        }
      }
    }
    while (frag_start < end && ',' != text[frag_start]
      && ';' != text[frag_start] && ':' != text[frag_start]
      && '.' != text[frag_start] && '?' != text[frag_start]
      && '!' != text[frag_start]) ++frag_start;
    if (frag_start < end) ++frag_start;
    while (frag_start < end
      && 0 != isspace(text[frag_start])) ++frag_start;
  }
  /**
   * A parent whose every window turned out to be a duplicate is unreferenced, and
   * a warm re-ingest makes that the NORM: the windows collapse in the key space
   * while the parent, keyed by a fresh number, does not. Left alone it grows the
   * parent file by the whole corpus on every warm run - the same shape as the
   * warm-start doubling the section comparison once caused.
   */
  if (0 != parent && 0 == result && NULL != corpus_parents) {
    unsigned char key[12];
    size_t key_size = 0;
    corpus_blob_key(parent, key, &key_size);
    libxs_registry_remove(corpus_parents, key, key_size, NULL);
    /* The id is handed out again, so any window cached under it would now stand
     * for a different parent's text: one registry's removal invalidating the
     * other's contents, which is the reason the views are dropped here. */
    corpus_view_free();
    if (corpus_blob_max == parent) --corpus_blob_max;
  }
  return result;
}


/**
 * Index prose markdown blocks at sentence scale in addition to paragraph scale.
 *
 * Off by default, and the default is not tidiness: the n-gram trainer scans every
 * corpus entry without filtering on scale, so a sentence indexed alongside its
 * paragraph is trained on twice and every bits-per-character figure for the
 * documentation moves. Enabling this therefore changes a published baseline, and
 * the mechanisms that need it (recombination, which gates on sentence scale and
 * otherwise finds no host at all) are probes rather than defaults.
 *
 * Code blocks and tables are never split: they have no sentences, and splicing
 * them would produce text the word-level seam gates cannot judge.
 */
static int corpus_md_sentences(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_MD_SENTENCES");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static int corpus_md_store(libxs_registry_t* corpus,
  const unsigned char* text, int len, size_t offset, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int code_like)
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
      SCALE_PARAGRAPH, lexicon, rules, nrules, 1))
    {
      entry.line = corpus_line_at(NULL, 0, offset);
      corpus_entry_set_section(&entry, section, section_len);
      if (0 != corpus_store_entry(corpus, &entry)) result = 1;
    }
    if (0 == code_like && 0 != corpus_md_sentences()) {
      int begin = 0, at;
      for (at = 0; at <= len; ++at) {
        const int is_end = (at == len)
          ? 1 : is_sentence_end_text(text, (size_t)len, (size_t)at);
        if (0 != is_end) {
          int end = (at < len) ? at + 1 : at;
          int span;
          size_t close_size = text_closer_size(text, (size_t)len, (size_t)end);
          while (0 != close_size) {
            end += (int)close_size;
            close_size = text_closer_size(text, (size_t)len, (size_t)end);
          }
          span = end - begin;
          while (0 < span && 0 != isspace((unsigned char)text[begin])) {
            ++begin;
            --span;
          }
          while (0 < span
            && 0 != isspace((unsigned char)text[begin + span - 1])) --span;
          /* A block that is one sentence is already stored above. */
          if (8 < span && span < len && 3 <= count_words(text + begin, span)
            && EXIT_SUCCESS == corpus_entry_build(&entry, text + begin, span,
              SCALE_SENTENCE, lexicon, rules, nrules, 1))
          {
            entry.line = corpus_line_at(NULL, 0, offset + (size_t)begin);
            corpus_entry_set_section(&entry, section, section_len);
            corpus_store_entry(corpus, &entry);
          }
          begin = end;
        }
      }
    }
  }
  return result;
}


static int corpus_md_emit_block(libxs_registry_t* corpus,
  const unsigned char* text, int len, size_t offset, const char* section,
  int section_len, libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules, int code_like)
{
  int result = 0;
  if (len < COMPOSE_MAXTEXT) {
    result = corpus_md_store(corpus, text, len, offset, section, section_len,
      lexicon, rules, nrules, code_like);
  }
  else {
    int line_start = 0, i;
    for (i = 0; i <= len; ++i) {
      if (i == len || '\n' == text[i]) {
        if (i > line_start) {
          result += corpus_md_store(corpus, text + line_start,
            i - line_start, offset + (size_t)line_start, section, section_len,
            lexicon, rules, nrules, code_like);
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
            (int)(i - block_start), block_start, section, section_len, lexicon,
            rules, nrules, 1);
          in_fence = 0;
          block_start = i + 1;
          block_has = 0;
        }
      }
      else if (0 != fence) {
        if (0 != block_has) {
          nblocks += corpus_md_emit_block(corpus, text + block_start,
            (int)(line_start - block_start), block_start, section, section_len,
            lexicon, rules, nrules, block_code);
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
            (int)(line_start - block_start), block_start, section, section_len,
            lexicon, rules, nrules, block_code);
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
            (int)(line_start - block_start), block_start, section, section_len,
            lexicon, rules, nrules, block_code);
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
      (int)(text_size - block_start), block_start, section, section_len,
      lexicon, rules, nrules, block_code);
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
          /* The id is spent only on a file that was actually read, so it stands for
           * a name a citation can print: -b probes many candidates that do not
           * exist, and numbering those left the table mostly empty. */
          if (corpus_source_id < 0xffff) ++corpus_source_id;
          corpus_source_path_set((int)corpus_source_id, path);
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
    /* The map comes from the same pass, because a line of the reflowed text is
     * not a line of the file a reader opens. */
    if (EXIT_SUCCESS == libxs_text_reflow_map(text, text_size,
      &reflowed, &reflowed_size, &corpus_ingest_lines, &corpus_ingest_nlines))
    {
      free(text);
      text = reflowed;
      text_size = reflowed_size;
    }
  }
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0
    && PROFILE_MARKDOWN == profile)
  {
    corpus_lines_index(text, text_size);
    result = corpus_ingest_markdown(corpus, text, text_size, path,
      lexicon, rules, nrules);
  }
  else if (EXIT_SUCCESS == result && NULL != text && text_size > 0) {
    size_t sent_start = 0;
    int nsentences = 0, nparagraphs = 0, nfragsent = 0, nfragpara = 0;
    const char* current_section = NULL;
    int current_section_len = 0;
    size_t i;
    corpus_sections_build(text, text_size);
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
          current_section_len = corpus_section_at(sent_start,
            &current_section);
          if (len >= COMPOSE_MAXTEXT) {
            const int made = corpus_store_clauses(corpus, text,
              sent_start, sent_end, current_section, current_section_len,
              lexicon, rules, nrules, 0);
            nfragsent += made;
            nsentences += made;
          }
          if (len > 8 && len < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + sent_start, len);
            if (nwords >= 3) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + sent_start, len, SCALE_SENTENCE,
                lexicon, rules, nrules, 1))
              {
                entry.line = corpus_line_at(text, text_size, sent_start);
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
      const char* para_section = NULL;
      int para_section_len = 0;
      for (p = 0; p < text_size; ++p) {
        if ('\n' == text[p] && p + 1 < text_size && '\n' == text[p + 1]) {
          int plen = (int)(p - para_start);
          while (plen > 0 && 0 != isspace(text[para_start + plen - 1]))
            --plen;
          para_section_len = corpus_section_at(para_start, &para_section);
          if (plen >= COMPOSE_MAXTEXT && 0 != corpus_fragments_para()) {
            const int made = corpus_store_clauses(corpus, text,
              para_start, para_start + (size_t)plen, para_section,
              para_section_len, lexicon, rules, nrules, 1);
            nfragpara += made;
            nsentences += made;
          }
          if (plen > 40 && plen < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + para_start, plen);
            if (nwords >= 8) {
              corpus_entry_t entry;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + para_start, plen, SCALE_PARAGRAPH,
                lexicon, rules, nrules, 1))
              {
                entry.line = corpus_line_at(text, text_size, para_start);
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
    /* Sections are reported because attribution rests on them and a heading the
     * scan misses is invisible in every other figure. */
    fprintf(stderr, "  fragments: %d from over-long sentences (the only copy),"
      " %d from long paragraphs (covered twice)\n", nfragsent, nfragpara);
    if (0 != corpus_span_check()) {
      fprintf(stderr, "  windows rebuilt: %ld of %d differ from the entry"
        " they replace\n", corpus_span_mismatch, nfragpara);
    }
    fprintf(stderr, "  ingested %s: %d sentences, %d paragraphs, %d sections\n",
      path, nsentences, nparagraphs, corpus_sections_size);
  }
  free(text);
  free(corpus_ingest_lines);
  corpus_ingest_lines = NULL;
  corpus_ingest_nlines = 0;
  return result;
}


int corpus_ingest_basename(libxs_registry_t* corpus,
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


void converse_judge_install(const converse_judge_t* judge)
{
  converse_judge_vtable = judge;
}


int converse_judge_verbose(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_HIER_RESCORE");
    cached = (NULL != env && '0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Open the judge when it was both installed and asked for, so the default path
 * pays neither the training time nor the memory. A binary without the judge
 * linked reaches this too, and simply has none.
 */
void converse_judge_open(const libxs_registry_t* corpus)
{
  if (0 != converse_judge_verbose() && NULL != converse_judge_vtable
    && NULL == converse_judge_opened)
  {
    converse_judge_opened = converse_judge_vtable->open(corpus,
      ngram_maxorder());
  }
}


void converse_judge_close(void)
{
  if (NULL != converse_judge_opened && NULL != converse_judge_vtable) {
    converse_judge_vtable->close(converse_judge_opened);
  }
  converse_judge_opened = NULL;
}


int converse_judge_active(void)
{
  return (NULL != converse_judge_opened) ? 1 : 0;
}


int converse_judge_rescore(const char* query, int query_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates, double bits[])
{
  int result = EXIT_FAILURE;
  if (NULL != converse_judge_opened) {
    result = converse_judge_vtable->rescore(converse_judge_opened, query,
      query_length, candidates, candidate_lengths, ncandidates, bits);
  }
  return result;
}


int converse_judge_choose(const char* context, int context_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates)
{
  int result = -1;
  if (NULL != converse_judge_opened) {
    result = converse_judge_vtable->choose(converse_judge_opened, context,
      context_length, candidates, candidate_lengths, ncandidates);
  }
  return result;
}


int converse_judge_seam_bits(const char* prefix, int prefix_length,
  const char* suffix, int suffix_length, int score_length, double* bits)
{
  int result = EXIT_FAILURE;
  if (NULL != converse_judge_opened) {
    result = converse_judge_vtable->seam_bits(converse_judge_opened, prefix,
      prefix_length, suffix, suffix_length, score_length, bits);
  }
  return result;
}


const libxs_lexnorm_t* converse_lexnorms(void)
{
  return answer_lexnorms;
}


int converse_lexnorms_size(void)
{
  return answer_lexnorms_size;
}


const answer_relation_rule_t* converse_rules(void)
{
  return answer_relation_rules;
}


size_t converse_rules_size(void)
{
  return answer_relation_rules_size;
}


libxs_ngram_t* converse_ngram_handle(void)
{
  return &converse_ngram;
}


const char* converse_bridge_path(void)
{
  return converse_path_bridge;
}


const char* converse_predict_eval_path(void)
{
  return converse_path_predict_eval;
}


const char* converse_facts_path(void)
{
  return converse_path_facts;
}


/**
 * The fixture to score against. Rule learning changes what the system asserts,
 * so it needs its OWN expectations - scoring a learned run against the asserted
 * fixture would either fail on answers that are correct for that mode or, worse,
 * pass because the modes happen to agree on the cases nobody re-checked. The
 * asserted fixture is used when no learn-mode file exists, which keeps every
 * historical command behaving as before.
 */
const char* converse_eval_path(void)
{
  const char* result = converse_path_eval;
  if (0 < answer_rules_learn_count()) {
    FILE* probe = fopen(converse_path_eval_learn, "r");
    if (NULL != probe) {
      fclose(probe);
      result = converse_path_eval_learn;
    }
  }
  return result;
}


/** The normalizer shares the setup but none of the modes, so it has its own. */
static void converse_usage_norm(const char* program)
{
  fprintf(stderr,
    "Usage: %s [-e] [-r N] [-m N] [-t FILE] [-b PREFIX] corpus1.txt"
    " [corpus2.txt ...]\n"
    "  Normalize spellings against the corpus vocabulary; without -e or -t,\n"
    "  read one word per line and print what it decodes to.\n"
    "  -e: report vocabulary packing and decode accuracy, and exit.\n"
    "  -t FILE: correct the forms in FILE (one per line) and write the\n"
    "           accepted corrections to the norms table.\n"
    "  -r N: largest edit distance a correction spans (default 1).\n"
    "  -m N: distance the runner-up must add beyond it (default 1).\n"
    "  -L: learn from the corpus, save state, and exit.\n"
    "  -p STRUCTURE: corpus structure (prose|markdown; default by ext).\n"
    "  -b PREFIX: ingest matching files next to PREFIX.\n",
    program);
}


static void converse_usage_all(const char* program)
{
  fprintf(stderr,
    "Usage: %s [-e] [-n N] [-P PROFILE] [-b PREFIX] corpus1.txt [corpus2.txt ...]\n"
    "  Interactive conversational agent.\n"
    "  -e: run converse.eval evaluation and exit.\n"
    "  -E: run next-token prediction evaluation and exit.\n"
    "  -L: learn from the corpus, save state, and exit.\n"
    "  -c: read prompts and print next-token continuations.\n"
    "  -K KIND: next-token model "
    "(bigram|trigram|predict|embed|rerank|knnlm|hier).\n"
    "  -H N: held-out split; train on non-test, eval 1-in-N test.\n"
    "  -T PREFIX: fixed held-out corpus for -E (train all of -b, eval on -T).\n"
    "  -n N: response sentence budget (default %d).\n"
    "  -P PROFILE: answer predictor profile.\n"
    "  -p STRUCTURE: corpus structure (prose|markdown; default by ext).\n"
    "  -x: extreme mode; deepest n-gram context (affects -c and -E).\n"
    "  -b PREFIX: ingest matching files next to PREFIX.\n"
    "Environment variables (override defaults):\n"
    "  CONVERSE_NGRAM_ORDER=N   n-gram context order 1..%d (default 2; -x=%d).\n"
    "  CONVERSE_GRAN=UNIT       token unit: word|native|syllable|bpe|"
    "meta-native|meta-word|meta-syllable.\n"
    "  CONVERSE_GEN_MINORDER=N  generation grounding floor (default 2).\n"
    "  CONVERSE_GEN_CONTORDER=F -c continuation mean-order floor (default 3).\n"
    "  CONVERSE_BPE_MERGES=N    BPE merge budget (default 750).\n"
    "  CONVERSE_HOLDOUT_TAIL=1  held-out split is a contiguous tail.\n"
    "  CONVERSE_SHUFFLE=1       shuffle words in each sentence (control).\n"
    "  CONVERSE_EVAL_STRIDE=N   evaluate 1-in-N test items (default 40).\n"
    "  CONVERSE_GEN_EVAL=1      -E reports generation reproduction.\n"
    "  CONVERSE_GEN_NCAND=N     successors offered per step 1..%d (default 1).\n"
    "  CONVERSE_GEN_FULL=1      gen-eval scores every lookahead position\n"
    "                           (teacher-forced) instead of stopping at the\n"
    "                           first miss; adds per-position top1/mrr.\n"
    "  CONVERSE_GEN_EMBRANK=1   rank the truth by the successor-side embedding\n"
    "                           over the whole vocabulary (needs -K embed).\n"
    "  CONVERSE_EMB_DIRECTED=N  factorize PPMI(next|cur) at forward distance\n"
    "                           1..N instead of the symmetric window (0=off).\n"
    "  CONVERSE_NGRAM_BANK_EMB=1 carry the successor-side embedding as a bank\n"
    "                           slot: total, and scores unattested successors.\n"
    "  CONVERSE_EMB_TEMP=F      temperature of that slot (default 1 = plain\n"
    "                           exp(PMI) rescaling of the unigram prior).\n"
    "  CONVERSE_EMB_CTX=N       history tokens the slot conditions on 1..%d\n"
    "                           (default 1); summing them combines per-token\n"
    "                           PMI evidence multiplicatively.\n"
    "  CONVERSE_EMB_DECAY=F     weight decay per extra history token (0<F<=1).\n"
    "  CONVERSE_GEN_EMBCAND=N   let the embedding PROPOSE N successors per step\n"
    "                           (default 0). Off by default because a proposed\n"
    "                           successor was never attested here, so the path\n"
    "                           synthesizes instead of selecting.\n"
    "  CONVERSE_NGRAM_STATS=1   print per-order n-gram footprint.\n"
    "  CONVERSE_KNNLM_LAMBDA=F  fixed kNN-LM interpolation weight.\n"
    "  CONVERSE_KNNLM_DYN=1     dynamic (test-time) kNN-LM datastore.\n"
    "  CONVERSE_KNNLM_ANN=1     Hilbert-indexed kNN-LM retrieval (faster).\n"
    "  CONVERSE_KNNLM_CTX=N     kNN-LM context tokens 2..8 (default 2).\n"
    "  CONVERSE_KNNLM_DECAY=F   kNN-LM context decay 0..1 (default 0.5).\n"
    "  CONVERSE_KNNLM_WEIGHTS=L per-position context weights (overrides decay).\n"
    "  CONVERSE_KNNLM_ORDERPROBE=1 report context order sensitivity.\n"
    "  CONVERSE_KNNLM_CONTROL=N control vote: 1=unigram, 2=arbitrary nbrs.\n"
    "  CONVERSE_KNNLM_TEMP=F    softmax temperature (0=inverse distance).\n"
    "  CONVERSE_KNNLM_HEADS=N   retrieval heads over embedding subspaces.\n"
    "  CONVERSE_HIER_MINCOUNT=N hierarchy known-unit count (default 2).\n"
    "  CONVERSE_HIER_CLOCK_ORDER=N byte/state context order 1..6 (default 2).\n"
    "  CONVERSE_HIER_STATE_DECAY=F recurrent-state decay 0..<1.\n"
    "  CONVERSE_HIER_TOP_STRIDE=N PPM top-k evaluation stride (default 40).\n"
    "  CONVERSE_HIER_EXPERT_ORDER=N highest mixed PPM order 0..6.\n"
    "  CONVERSE_HIER_EXPERT_RATE=F expert update rate 0..1 (default 0.15).\n"
    "  CONVERSE_HIER_EXPERT_SHARE=F fixed share 0..<1 (default 0.005).\n"
    "  CONVERSE_SKIP=1          add skip-gram (w _ w) generalization tier.\n"
    "  CONVERSE_SKIP_MU=F       skip-gram interpolation weight (default 0.3).\n"
    "  CONVERSE_NGRAM_BANK_SKIP=1 carry the skip tier as its own bank slot\n"
    "                           (learned weight instead of fixed SKIP_MU).\n"
    "  CONVERSE_NGRAM_BANK_PRIOR=1 carry the unigram prior as a bank slot\n"
    "                           (a total expert where counts are partial).\n"
    "  CONVERSE_NGRAM_BANK_PREDICT=1 carry libxs_predict as a bank slot\n"
    "                           (distribution via prob_observe).\n"
    "  CONVERSE_NGRAM_BANK_FROZEN=1 warm up the predict slot on the training\n"
    "                           entries, commit, then score frozen: the\n"
    "                           figure stops depending on iteration order.\n"
    "  CONVERSE_NGRAM_BANK_WARMUP=N observe only the first N entries during\n"
    "                           warm-up (0=all); the bank settles long before\n"
    "                           the store is exhausted.\n"
    "  CONVERSE_GEN_RELEVANCE=F require F content-word overlap between an\n"
    "                           answer and its continuation (default 0=off).\n"
    "  CONVERSE_NGRAM_BANK_GEO=1 geometric (log-linear) pool instead of the\n"
    "                           linear one; exactly normalized.\n"
    "  CONVERSE_SYLLABLE_PROBE=w1,w2 print the syllable split of each word.\n"
    "  CONVERSE_SYLLABLE_ORACLE=1 bound any cut rule: cheapest split per word\n"
    "                           vs the heuristic's, over held-out words.\n"
    "  CONVERSE_EMB_PROBE=w1,w2 print nearest embedding neighbors.\n"
    "  CONVERSE_EMB_BACKFILL=0  disable rare-token vector backfill.\n"
    "  CONVERSE_RECOMB=N        recombination probe over N sentences.\n"
    "  CONVERSE_RECOMB_REFERENT=1 test whether a pivot denotes one thing.\n"
    "  CONVERSE_RECOMB_COMPOSE=1 -c may synthesize when replay fails.\n"
    "  CONVERSE_PREDICT=1       train the answer ranker (off by default:\n"
    "                           40%% of wall, no measured BPC or QA change).\n",
    program, RESPONSE_BUDGET, NGRAM_ORDER_MAX, NGRAM_ORDER_MAX,
    GEN_CAND_MAX, TOKEN_CTX_MAX);
  answer_predict_profile_list(stderr);
}


static void converse_usage(const char* program, int role)
{
  if (CONVERSE_ROLE_NORM == role) converse_usage_norm(program);
  else converse_usage_all(program);
}


/**
 * Parse the command line into `run`. `basenames` may be NULL when the caller
 * only needs the modes, which is how converse_role_of asks the same parser the
 * same question without acquiring anything.
 */
static int converse_parse(int argc, char* argv[], converse_run_t* run,
  const char* basenames[], int* nbasenames, int* first)
{
  int result = EXIT_SUCCESS;
  int i = 1;
  while (i < argc && '-' == argv[i][0] && '\0' != argv[i][1]
    && EXIT_SUCCESS == result)
  {
    if (0 == strcmp(argv[i], "-e")) {
      run->eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-E")) {
      run->predict_eval_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-L")) {
      run->learn_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-c")) {
      run->complete_mode = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-x")) {
      converse_order_max = 1;
      ++i;
    }
    else if (0 == strcmp(argv[i], "-r") && i + 1 < argc) {
      run->norm_radius = atoi(argv[i + 1]);
      if (run->norm_radius < 0) run->norm_radius = 0;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-m") && i + 1 < argc) {
      run->norm_margin = atoi(argv[i + 1]);
      if (run->norm_margin < 0) run->norm_margin = 0;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-t") && i + 1 < argc) {
      run->norm_words = argv[i + 1];
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-H") && i + 1 < argc) {
      run->ngram_holdout = atoi(argv[i + 1]);
      if (run->ngram_holdout < 0) run->ngram_holdout = 0;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-K") && i + 1 < argc) {
      if (0 == strcmp(argv[i + 1], "bigram")) {
        run->ngram_kind = "bigram";
        run->ngram_order = 1;
      }
      else if (0 == strcmp(argv[i + 1], "trigram")) {
        run->ngram_kind = "trigram";
        run->ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "predict")) {
        run->ngram_kind = "predict";
        run->ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "embed")) {
        run->ngram_kind = "embed";
        run->ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "rerank")) {
        run->ngram_kind = "rerank";
        run->ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "knnlm")) {
        run->ngram_kind = "knnlm";
        run->ngram_order = 2;
      }
      else if (0 == strcmp(argv[i + 1], "hier")) {
        run->ngram_kind = "hier";
        run->ngram_order = 2;
      }
      else {
        fprintf(stderr,
          "unknown prediction kind: %s "
          "(use bigram|trigram|predict|embed|rerank|knnlm|hier)\n",
          argv[i + 1]);
        result = EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-n") && i + 1 < argc) {
      run->budget = atoi(argv[i + 1]);
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-P") && i + 1 < argc) {
      run->profile = answer_predict_profile_find(argv[i + 1]);
      if (NULL == run->profile) {
        fprintf(stderr, "unknown predictor profile: %s\n", argv[i + 1]);
        answer_predict_profile_list(stderr);
        result = EXIT_FAILURE;
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
        result = EXIT_FAILURE;
      }
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-b") && i + 1 < argc) {
      if (NULL != basenames) basenames[*nbasenames] = argv[i + 1];
      ++*nbasenames;
      i += 2;
    }
    else if (0 == strcmp(argv[i], "-T") && i + 1 < argc) {
      run->test_prefix = argv[i + 1];
      i += 2;
    }
    else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      answer_predict_profile_list(stderr);
      result = EXIT_FAILURE;
    }
  }
  *first = i;
  return result;
}


/**
 * Which half the parsed modes need. -L is served by whichever binary was run,
 * because it only writes the shared state that setup itself builds.
 *
 * The recombination probe rides -E: it is a grounded-half mechanism and the only
 * reason the QA role admits -E at all. A dedicated flag would be cleaner and
 * would break every documented invocation, so the mode surface carries it here
 * instead.
 */
static int converse_run_role(const converse_run_t* run)
{
  const int lm_kind = (0 != strcmp(run->ngram_kind, "bigram")
    && 0 != strcmp(run->ngram_kind, "trigram")) ? 1 : 0;
  int result = CONVERSE_ROLE_QA;
  if (0 != run->predict_eval_mode) {
    result = (0 == lm_kind && NULL != getenv("CONVERSE_RECOMB"))
      ? CONVERSE_ROLE_QA : CONVERSE_ROLE_LM;
  }
  else if (0 != run->complete_mode && 0 != lm_kind) {
    result = CONVERSE_ROLE_LM;
  }
  return result;
}


int converse_role_of(int argc, char* argv[])
{
  converse_run_t probe;
  int nbasenames = 0, first = 0;
  memset(&probe, 0, sizeof(probe));
  probe.ngram_kind = "trigram";
  converse_parse(argc, argv, &probe, NULL, &nbasenames, &first);
  return converse_run_role(&probe);
}


/** Reject a mode this binary's half does not serve, before any corpus work. */
static int converse_role_gate(const converse_run_t* run)
{
  const int wanted = converse_run_role(run);
  int result = EXIT_SUCCESS;
  if (CONVERSE_ROLE_NORM == run->role) {
    /* -e is the normalizer's own evaluation; the prediction modes are not its */
    if (0 != run->predict_eval_mode || 0 != run->complete_mode) {
      fprintf(stderr, "the normalizer serves neither -E nor -c:"
        " use converse-lm\n");
      result = EXIT_FAILURE;
    }
  }
  else if (NULL != run->norm_words || 0 <= run->norm_radius
    || 0 <= run->norm_margin)
  {
    fprintf(stderr, "-t, -r and -m are normalization: use converse-norm\n");
    result = EXIT_FAILURE;
  }
  else if (CONVERSE_ROLE_ALL != run->role && 0 == run->learn_mode
    && wanted != run->role)
  {
    if (CONVERSE_ROLE_LM == wanted) {
      fprintf(stderr, "%s: use converse-lm\n", (0 != run->predict_eval_mode)
        ? "-E is next-token prediction"
        : "a prediction kind under -c is next-token prediction");
    }
    else {
      fprintf(stderr, "%s: use converse-qa\n", (0 != run->eval_mode)
        ? "-e is grounded QA evaluation"
        : ((0 != run->complete_mode) ? "-c is grounded continuation"
          : "interactive answering is grounded QA"));
    }
    result = EXIT_FAILURE;
  }
  return result;
}


/**
 * Build the BPE merge table when the tokenizer is in BPE mode, and nothing
 * otherwise. A half asks for the tokenizer it was configured with and does not
 * need to know the granularities.
 */
void converse_bpe_prepare(const libxs_registry_t* corpus, int holdout)
{
  if (GRAN_BPE == ngram_gran_mode()) bpe_build(corpus, holdout);
}


int converse_setup(int argc, char* argv[], int role, converse_run_t* run)
{
  const char** basenames = NULL;
  int nbasenames = 0, first = 0, warm_start = 0;
  int result = EXIT_SUCCESS;
  memset(run, 0, sizeof(*run));
  run->role = role;
  run->budget = RESPONSE_BUDGET;
  run->ngram_order = 2;
  run->ngram_kind = "trigram";
  run->profile = answer_predict_profile_default();
  run->rules = converse_lexrules;
  /* negative is unspecified: a radius of zero is a meaningful request */
  run->norm_radius = -1;
  run->norm_margin = -1;
  if (argc < 2) {
    converse_usage(argv[0], role);
    result = EXIT_FAILURE;
  }
  else {
    basenames = (const char**)malloc((size_t)argc * sizeof(*basenames));
    if (NULL == basenames) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = converse_parse(argc, argv, run, basenames, &nbasenames, &first);
  }
  if (EXIT_SUCCESS == result && first == argc && 0 == nbasenames) {
    fprintf(stderr, "no corpus source given\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) result = converse_role_gate(run);
  if (EXIT_SUCCESS == result) {
    converse_namespace_init((nbasenames > 0) ? basenames[0] : NULL);
    if (0 != run->eval_mode && nbasenames + argc - first < 1) {
      fprintf(stderr,
        "eval mode expects at least one corpus source and reads %s\n",
        converse_path_eval);
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    int stale = 0;
    run->corpus = corpus_load();
    if (NULL == run->corpus) run->corpus = libxs_registry_create();
    run->lexicon = converse_lexicon_load(&stale);
    if (0 != stale) {
      fprintf(stderr,
        "%s was written by a different version and the corpus beside it "
        "refers to its ids: remove %s and the companion files to rebuild\n",
        converse_path_lexicon, converse_path_lexicon);
      result = EXIT_FAILURE;
    }
    if (NULL == run->lexicon && 0 == stale) {
      run->lexicon = libxs_lexicon_create();
    }
    run->answer_model = converse_predict_load();
    run->nrules = libxs_lexrule_defaults(converse_lexrules, 96);
    if (NULL == run->corpus || NULL == run->lexicon || run->nrules <= 0) {
      result = EXIT_FAILURE;
    }
    else corpus_view_bind(run->lexicon, converse_lexrules, run->nrules);
  }
  if (EXIT_SUCCESS == result) {
    /**
     * Rules layer: the language file (committed, shared by every corpus of that
     * language) is loaded first, then the optional per-corpus file extends it.
     * Loading appends, so a corpus adds its own vocabulary without restating the
     * language's function words, and neither file contains anything the source
     * needs to know about.
     *
     * `<prefix>.rules` REPLACES the shared file rather than extending it, because
     * a corpus that brings its own language rules is not in the shared file's
     * language, and the shared file's assertions are then not merely unhelpful -
     * they are claims about a vocabulary this corpus does not have. Extending is
     * still what `<prefix>.relations` does, for a corpus in the same language.
     */
    if ('\0' == converse_path_language_own[0]
      || 0 == answer_relation_rules_load_file(converse_path_language_own))
    {
      answer_relation_rules_load_file(converse_path_language);
    }
    answer_relation_rules_load_file(converse_path_norm);
    answer_relation_rules_load_file(converse_path_relation);
    answer_relation_rules_report(stderr);
    /* Needs no corpus and no model, so it runs before any of them is built. */
    ngram_syllable_probe();
    /**
     * A warm start reuses the persisted corpus and lexicon instead of
     * re-ingesting: the state is complete when the corpus loaded non-empty and
     * the lexicon is populated. Learn mode (-L) always rebuilds. Only the cheap
     * fact index is rebuilt each run, by whichever half needs it.
     *
     * The predictor is deliberately not evidence of state. It is skippable and
     * off by default, so requiring it meant no default run ever warmed up: each
     * one re-ingested sources the corpus already held, which the content-keyed
     * corpus absorbs silently and libxs_lexicon_count does not, the occurrence
     * counts doubling per run.
     */
    { libxs_registry_info_t warm;
      warm.size = 0;
      warm_start = (0 == run->learn_mode
        && libxs_lexicon_size(run->lexicon) > 0
        && EXIT_SUCCESS == libxs_registry_info(run->corpus, &warm)
        && warm.size > 0) ? 1 : 0;
    }
  }
  if (EXIT_SUCCESS == result && 0 == warm_start) {
    int basename_index;
    const int have_positional = (first < argc) ? 1 : 0;
    converse_stage_begin();
    for (basename_index = 0; basename_index < nbasenames
      && EXIT_SUCCESS == result; ++basename_index)
    {
      if (EXIT_SUCCESS != corpus_ingest_basename(run->corpus,
        basenames[basename_index], run->lexicon, converse_lexrules,
        run->nrules) && 0 == have_positional)
      {
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      int argi;
      for (argi = first; argi < argc; ++argi) {
        corpus_ingest_file(run->corpus, argv[argi], run->lexicon,
          converse_lexrules, run->nrules);
      }
      converse_stage_end("ingest");
      corpus_save(run->corpus);
      converse_lexicon_save(run->lexicon);
      converse_stage_end("save");
      /**
       * The answer ranker is skippable, and on a large corpus it has to be. It is
       * the only consumer of this model, it degrades to the unmodified base score
       * when the model is absent (see answer_predict_score), and training it is the
       * stage that stops a big corpus being usable at all: pushing every entry x
       * query-type superlinearly outgrows ingest, so 88k Fortran sentences never
       * reach the model that the recombination and byte-model tracks actually need.
       * CONVERSE_NO_PREDICT=1 trades a ranking refinement for reaching those tracks.
       */
      if (0 != converse_predict_on()) {
        libxs_predict_t* trained = converse_predict_train(run->corpus,
          run->profile);
        if (NULL != trained) {
          libxs_predict_destroy(run->answer_model);
          run->answer_model = trained;
          converse_predict_save(run->answer_model);
        }
        converse_stage_end("predict_train");
      }
      else {
        fprintf(stderr, "predict: training skipped"
          " (set CONVERSE_PREDICT=1 to enable)\n");
      }
    }
  }
  else if (EXIT_SUCCESS == result) {
    fprintf(stderr, "warm start: reusing %s\n", converse_path_corpus);
  }
  free(basenames);
  if (EXIT_SUCCESS == result) {
    const void* key = NULL;
    size_t cursor = 0, nfrag = 0, ntext = 0, ntok = 0, nall = 0;
    size_t nspan = 0, nblob = 0, nspantext = 0, nblobtext = 0;
    size_t nbig[4];
    void* value;
    /* Before the facts, because a widened person class is what makes new facts
     * extractable at all: run it after and the learned terms would sit unused. */
    answer_relation_rules_learn(run->corpus, run->lexicon, converse_lexrules,
      run->nrules);
    token_emb_pair_probe(run->corpus, run->lexicon, converse_lexrules,
      run->nrules);
    /**
     * Where the corpus file's bytes actually go, and how many records STAND FOR an
     * entry. Worth reporting because the shape is counter-intuitive: an entry
     * carries FIXED metadata - the token id and flag arrays at ENTRY_TOKEN_MAX,
     * and the section NAME - whether or not it uses them, so a corpus of short
     * entries is mostly metadata. Any work on the corpus size should read this
     * first rather than assume.
     *
     * The count excludes the parent texts, and that is not cosmetic: it is what
     * every train/test SPLIT is taken over, so counting them would shift the
     * holdout boundary and move figures that have nothing to do with parents.
     */
    nbig[0] = 0; nbig[1] = 0; nbig[2] = 0; nbig[3] = 0;
    if (NULL != corpus_parents) {
      value = libxs_registry_begin(corpus_parents, &key, &cursor);
      while (NULL != value) {
        ++nblob;
        nblobtext += (size_t)((const corpus_blob_t*)value)->text_len;
        value = libxs_registry_next(corpus_parents, &key, &cursor);
      }
      key = NULL;
      cursor = 0;
    }
    value = libxs_registry_begin(run->corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      switch (corpus_value_kind(value)) {
        case ENTRY_KIND_SPAN: {
          ++nspan;
          nspantext += (size_t)((const corpus_span_t*)value)->text_len;
        } break;
        default: {
          ++nall;
          if (0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT)) ++nfrag;
          if (0 < entry->text_len) ntext += (size_t)entry->text_len;
          ntok += entry->ntokens;
          if (entry->ntokens > 16) ++nbig[0];
          if (entry->ntokens > 24) ++nbig[1];
          if (entry->ntokens > 32) ++nbig[2];
          if (entry->ntokens >= ENTRY_TOKEN_MAX) ++nbig[3];
        }
      }
      value = libxs_registry_next(run->corpus, &key, &cursor);
    }
    fprintf(stderr, "corpus: %lu sentences\n", (unsigned long)(nall + nspan));
    { if (0 < nall) {
        const size_t meta = nall * CORPUS_ENTRY_META_SIZE
          + nspan * sizeof(corpus_span_t) + nblob * CORPUS_BLOB_META_SIZE;
        fprintf(stderr, "corpus bytes: %lu meta + %lu text = %lu"
          " (%lu fragments, %lu tokens of %lu slots)\n",
          (unsigned long)meta, (unsigned long)(ntext + nblobtext),
          (unsigned long)(meta + ntext + nblobtext), (unsigned long)nfrag,
          (unsigned long)ntok, (unsigned long)(nall * ENTRY_TOKEN_MAX));
        if (0 < nspan) {
          fprintf(stderr, "corpus windows: %lu spans over %lu parents,"
            " %lu located Bytes of %lu kept\n", (unsigned long)nspan,
            (unsigned long)nblob, (unsigned long)nspantext,
            (unsigned long)nblobtext);
        }
        fprintf(stderr, "corpus tokens: >16 %lu, >24 %lu, >32 %lu, at cap %lu"
          " of %lu entries\n", (unsigned long)nbig[0], (unsigned long)nbig[1],
          (unsigned long)nbig[2], (unsigned long)nbig[3], (unsigned long)nall);
      }
    }
    if (corpus_chain_dropped > 0) {
      fprintf(stderr, "  fingerprint chain cap reached: %ld texts dropped"
        " (CONVERSE_CHAIN_MAX=%u)\n", corpus_chain_dropped, corpus_chain_max());
    }
    run->nsentences = (long)(nall + nspan);
    predict_ntotal = run->nsentences;
    if (0 != run->learn_mode) {
      fprintf(stderr, "learned: %s%s, %s, %s\n", converse_path_corpus,
        (NULL != corpus_parents) ? " + parents" : "",
        converse_path_lexicon, converse_path_predict);
    }
    else run->pending = 1;
  }
  return result;
}


void converse_release(converse_run_t* run)
{
  if (NULL != run) {
    converse_judge_close();
    token_emb_free();
    bpe_free();
    libxs_ngram_destroy(&converse_ngram);
    if (0 != converse_skip_on) libxs_ngram_destroy(&converse_skip);
    converse_skip_on = 0;
    if (NULL != run->corpus && NULL != run->lexicon) {
      converse_lexicon_save(run->lexicon);
    }
    libxs_predict_destroy(run->answer_model);
    libxs_lexicon_destroy(run->lexicon);
    corpus_view_free();
    libxs_registry_destroy(run->corpus);
    if (NULL != corpus_parents) {
      libxs_registry_destroy(corpus_parents);
      corpus_parents = NULL;
    }
    answer_relation_rules_free();
    free(corpus_sections);
    corpus_sections = NULL;
    corpus_sections_size = 0;
    corpus_sections_cap = 0;
    corpus_source_paths_free();
    memset(run, 0, sizeof(*run));
  }
}
