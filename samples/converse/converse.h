#ifndef CONVERSE_H
#define CONVERSE_H

#include <libxs/libxs_predict.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_reg.h>

#define FPRINT_ORDER 4
#define CORPUS_FILE "converse.dat"
#define COMPOSE_NDIMS 10
#define COMPOSE_BITS 6
#define COMPOSE_MAXTEXT 512
/**
 * Token slots an entry carries, and the single largest lever on corpus size: the
 * two arrays are 288 of the 560 FIXED bytes an entry spends, and they ran at ~35%
 * occupancy at the old value of 48.
 *
 * 32 was chosen by measurement, not taste. Entries exceeding it are 1.1% on the
 * tales and 2.9% on Wikipedia prose, and only 9 and 181 respectively ever reached
 * 48 at all. Cutting to 32 takes the corpus file down 13% while the QA gates stay
 * at 19/19 and 14/14 and EVERY recombination figure is byte-identical - made,
 * pivot, ceiling, floor, coherence, penalty and the pivot index alike. 24 was also
 * measured (-19%) and declined: it truncates 10% of entries for six more points.
 *
 * Changing this changes the STORED LAYOUT. There is no compatibility burden -
 * corpus_fixup detects a file written by another layout and discards it so the
 * next run re-ingests - but a sweep still costs a full re-ingest each step:
 * make ECFLAGS="-DENTRY_TOKEN_MAX=48".
 */
#ifndef ENTRY_TOKEN_MAX
# define ENTRY_TOKEN_MAX 32
#endif
#define ENTRY_SECTION_MAX 64

#define ENTRY_LEX_ENTITY 0x0001u
#define ENTRY_LEX_NUMBER 0x0002u
#define ENTRY_LEX_QUESTION 0x0004u
#define ENTRY_LEX_PLACE 0x0008u
#define ENTRY_LEX_CAUSE 0x0010u
#define ENTRY_LEX_METHOD 0x0020u
/**
 * The entry is a clause FRAGMENT cut from a larger sentence, not a sentence.
 * Ingest stores both, so a byte of source text belongs to several entries at
 * the same scale; anything that must count each source byte once (above all the
 * BPC denominator) has to exclude these.
 */
#define ENTRY_LEX_FRAGMENT 0x0040u

#define GEN_CAND_MAX 8
#define EVAL_LINE_MAX 2048
#define NGRAM_ORDER_MAX LIBXS_NGRAM_ORDER_MAX
#if !defined(TOKEN_EMB_DIM)
# define TOKEN_EMB_DIM 16
#endif
#define TOKEN_CTX_MAX 8
#define ANSWER_PREDICT_INPUTS 10

/**
 * Bytes an entry occupies through its section field, i.e. every fixed-offset
 * field. A stored value at least this large carries complete metadata; the old
 * test was "entry_size >= sizeof(*entry)", which variable-length text makes
 * false for every entry.
 */
#define CORPUS_ENTRY_META_SIZE \
  (sizeof(corpus_entry_t) - COMPOSE_MAXTEXT)

#define CORPUS_BLOB_META_SIZE offsetof(corpus_blob_t, text)


/**
 * Which half of the system a binary exposes.
 *
 * The two halves answer different questions and are being separated into
 * different papers and different translation units: QA is grounded answering,
 * attribution and grounded recombination, LM is next-token prediction and its
 * quantification. The role gates the MODES an entry point accepts, so each
 * binary has a coherent command surface and nothing links a mode it cannot serve.
 * CONVERSE_ROLE_ALL keeps the historical single-binary behaviour so every
 * documented reproduction command still runs.
 */
enum {
  CONVERSE_ROLE_ALL = 0,
  CONVERSE_ROLE_QA = 1,
  CONVERSE_ROLE_LM = 2
};

enum { CONN_SPACE = 0, CONN_COMMA = 1, CONN_PERIOD = 2, CONN_NEWLINE = 3 };
enum { SCALE_PHRASE = 0, SCALE_SENTENCE = 1, SCALE_PARAGRAPH = 2 };

/**
 * text is LAST so an entry can be stored at its actual length
 * (corpus_entry_size) instead of the full COMPOSE_MAXTEXT. The corpus dominates
 * memory - 1512 B per entry for a mean 34 B of enwik8 sentence text, which is
 * what made 90 MB exhaust RAM - and every field before text keeps a fixed
 * offset, so readers are unaffected. The registry stores variable-size values
 * and readers already consult libxs_registry_value_size, which is why the
 * section helpers take an entry_size.
 */
/**
 * The stored projection of a fingerprint. libxs_fprint_t is 624 B because its
 * eight arrays are sized to LIBXS_FPRINT_MAXORDER (8) and three of them are
 * streaming accumulators used only while building. Converse needs exactly four
 * arrays at FPRINT_ORDER: l2 and mean for the Hilbert key, acc_sq/acc_sum/nk for
 * the similarity score. At 120k entries per 4 MB of text the difference is the
 * single largest term in corpus memory.
 */
typedef struct corpus_fprint_t {
  double l2[FPRINT_ORDER + 1];
  double mean[FPRINT_ORDER + 1];
  double acc_sq[FPRINT_ORDER + 1];
  double acc_sum[FPRINT_ORDER + 1];
  int nk[FPRINT_ORDER + 1];
  int order;
} corpus_fprint_t;

/**
 * What a stored corpus value IS. The tag is FIRST in every record kind and at the
 * same offset in all of them, so a reader can classify a value before it knows
 * how large the value is - which is the whole point: a span is sixteen bytes and
 * a full entry is at least 464, and reading `scale` out of the former would be
 * out of bounds. Zero is deliberately not a valid kind, so a record written by a
 * layout that predates the tag (where these bytes were fprint.l2[0]) or one left
 * uninitialized by memset does not pass for a full entry.
 */
enum {
  ENTRY_KIND_FULL = 0x4655u,
  ENTRY_KIND_SPAN = 0x5350u,
  ENTRY_KIND_BLOB = 0x424cu
};

/**
 * kind, connector, scale and text_len lead so the tag sits at offset zero and the
 * eight bytes that classify a record cost nothing: they fill the slack the
 * fingerprint's alignment used to leave behind it, and an entry is the same 464
 * bytes of metadata it was before the tag existed.
 */
typedef struct corpus_entry_t {
  unsigned short kind;
  unsigned char connector;
  unsigned char scale;
  int text_len;
  corpus_fprint_t fprint;
  unsigned short ntokens;
  unsigned short ncontent;
  unsigned short nentities;
  unsigned short nnumbers;
  unsigned short lexical_flags;
  unsigned short source;
  /** 1-based line of the SOURCE FILE this text begins on; 0 if unknown. */
  unsigned int line;
  unsigned int token_ids[ENTRY_TOKEN_MAX];
  unsigned short token_flags[ENTRY_TOKEN_MAX];
  unsigned short section_len;
  char section[ENTRY_SECTION_MAX];
  char text[COMPOSE_MAXTEXT];
} corpus_entry_t;

/**
 * The common prefix of every record kind: enough to classify a value and, for the
 * kinds that carry text, to know how long that text is. Nothing beyond these eight
 * bytes may be read before the kind has been established.
 */
typedef struct corpus_head_t {
  unsigned short kind;
  unsigned char connector;
  unsigned char scale;
  int text_len;
} corpus_head_t;

/**
 * A clause window that is LOCATED rather than stored: which parent text it was cut
 * from, and where. Sixteen bytes instead of the 464 of metadata plus text an entry
 * spends, which matters because the windows are the bulk of the corpus - 11466 of
 * 16092 entries on the tales - and every one of them is a byte range of a
 * paragraph that is already being kept.
 *
 * A span is keyed exactly as the entry it replaces was, by the content hash of its
 * text, so the SET of keys and therefore the corpus iteration ORDER is the same
 * either way. That is what lets the derived pool reproduce the recombination
 * figures rather than merely resemble them, and it is also why de-duplication
 * needs no second thought: the collapse of overlapping windows already happened in
 * the key space.
 */
typedef struct corpus_span_t {
  unsigned short kind;
  unsigned char connector;
  unsigned char scale;
  int text_len;
  unsigned int parent;
  unsigned int offset;
  /**
   * Line of the source file this window begins on, STORED rather than derived from
   * the parent. Deriving it would mean counting newlines inside the parent text,
   * and prose is ingested REFLOWED: a cosmetic line break is a space there, so the
   * count would be short by every line reflow joined. Four bytes settles it, and
   * the stored value is the one the span-check invariant compares.
   */
  unsigned int line;
} corpus_span_t;

/**
 * The parent text of a set of spans, with the section and source they inherit.
 * Section and source live here rather than in each span because a window cannot
 * span two paragraphs, so all windows of one parent agree on both - and the
 * section alone is 66 of the bytes a span would otherwise have to carry.
 */
typedef struct corpus_blob_t {
  unsigned short kind;
  unsigned char connector;
  unsigned char scale;
  int text_len;
  unsigned short source;
  unsigned short section_len;
  char section[ENTRY_SECTION_MAX];
  char text[1];
} corpus_blob_t;

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

enum { RELATION_RULE_ALIAS = 1, RELATION_RULE_PERSON, RELATION_RULE_SKIP,
  RELATION_RULE_NEGATE, RELATION_RULE_NORM, RELATION_RULE_CAPS,
  RELATION_RULE_WHERE, RELATION_RULE_WHY, RELATION_RULE_HOW,
  RELATION_RULE_PLACE, RELATION_RULE_TOPIC, RELATION_RULE_COPULA,
  RELATION_RULE_ARTICLE, RELATION_RULE_PREP, RELATION_RULE_OWN,
  RELATION_RULE_POSS, RELATION_RULE_AUX, RELATION_RULE_AGENT,
  RELATION_RULE_LINK, RELATION_RULE_GENITIVE, RELATION_RULE_JOIN,
  RELATION_RULE_ASK, RELATION_RULE_PRON, RELATION_RULE_RESULT };

/**
 * Where a rule came from. ASSERTED means someone wrote it in the rule file.
 * The other two come from rule learning, above and below its acceptance bar.
 *
 * Both learned levels are labelled in replies, not just the margin. The margin
 * cannot be promoted by moving the threshold - wrong terms score between right
 * ones - and the ACCEPTED band is not trustworthy either: at the default bar
 * adjectives and interjections enter the person class, and one adjective is
 * enough to turn a correct reply into a confident false assertion. Labelling
 * only the margin would say the accepted band is safe, which it is not.
 */
enum { RELATION_RULE_ASSERTED = 0, RELATION_RULE_LEARNED,
  RELATION_RULE_PROPOSED };

typedef struct answer_relation_rule_t {
  int kind;
  int provenance;
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
  /** Render in the voice the corpus used; see answer_relation_reply. */
  int active;
  double score;
} answer_relation_match_t;


/**
 * What one invocation has to work with once the shared state is loaded: the
 * corpus, the lexicon and the rules every model is built from, plus the modes
 * parsed from the command line. Passed to whichever half serves the invocation,
 * so the setup that both halves need exists once and neither half parses
 * arguments.
 */
typedef struct converse_run_t {
  libxs_registry_t* corpus;
  libxs_lexicon_t* lexicon;
  libxs_predict_t* answer_model;
  libxs_lexrule_t* rules;
  const answer_predict_profile_t* profile;
  const char* ngram_kind;
  const char* test_prefix;
  long nsentences;
  int nrules;
  int budget;
  int ngram_order;
  int ngram_holdout;
  int eval_mode;
  int predict_eval_mode;
  int complete_mode;
  int learn_mode;
  int role;
  /** A half must run: 0 after -L, which is complete once setup returns. */
  int pending;
} converse_run_t;


/**
 * The byte model as an INSTRUMENT the entry point installs, not a dependency the
 * halves name.
 *
 * Every judge in this system is a callback for the same reason converse_recomb's
 * word_prob and seam_bits are: a seam or candidate score has never separated a
 * true continuation from a fluent false one, so all of them are diagnostics.
 * Making them a hook is what lets the grounded half compile and link without the
 * byte model, while the binary that measures with it installs one. Every entry
 * point below degrades to "no instrument" rather than failing: rescore and choose
 * leave the caller's order untouched, seam_bits reports no bits.
 */
typedef struct converse_judge_t {
  void* (*open)(const libxs_registry_t* corpus, int maxorder);
  void (*close)(void* model);
  int (*rescore)(const void* model, const char* query, int query_length,
    const char* const candidates[], const int candidate_lengths[],
    int ncandidates, double bits[]);
  int (*choose)(const void* model, const char* context, int context_length,
    const char* const candidates[], const int candidate_lengths[],
    int ncandidates);
  int (*seam_bits)(const void* model, const char* prefix, int prefix_length,
    const char* suffix, int suffix_length, int score_length, double* bits);
} converse_judge_t;


LIBXS_INLINE void corpus_fprint_pack(corpus_fprint_t* dst,
  const libxs_fprint_t* src)
{
  int k;
  for (k = 0; k <= FPRINT_ORDER; ++k) {
    const int use = (k <= src->order) ? 1 : 0;
    dst->l2[k] = (0 != use) ? src->l2[k] : 0.0;
    dst->mean[k] = (0 != use) ? src->mean[k] : 0.0;
    dst->acc_sq[k] = (0 != use) ? src->acc_sq[k] : 0.0;
    dst->acc_sum[k] = (0 != use) ? src->acc_sum[k] : 0.0;
    dst->nk[k] = (0 != use) ? src->nk[k] : 0;
  }
  dst->order = (src->order < FPRINT_ORDER) ? src->order : FPRINT_ORDER;
}


/** Widen the stored projection back to the library form. */
LIBXS_INLINE void corpus_fprint_unpack(libxs_fprint_t* dst,
  const corpus_fprint_t* src)
{
  int k;
  memset(dst, 0, sizeof(*dst));
  for (k = 0; k <= FPRINT_ORDER; ++k) {
    dst->l2[k] = src->l2[k];
    dst->mean[k] = src->mean[k];
    dst->acc_sq[k] = src->acc_sq[k];
    dst->acc_sum[k] = src->acc_sum[k];
    dst->nk[k] = src->nk[k];
  }
  dst->order = src->order;
}


/** Bytes actually occupied by an entry: everything up to its text length. */
LIBXS_INLINE size_t corpus_entry_size(const corpus_entry_t* entry)
{
  const size_t used = (0 < entry->text_len) ? (size_t)entry->text_len : 0;
  return sizeof(*entry) - COMPOSE_MAXTEXT + used + 1;
}


/** The kind of a stored value, or zero if it carries no valid tag. */
LIBXS_INLINE unsigned int corpus_value_kind(const void* value)
{
  return (NULL != value) ? ((const corpus_head_t*)value)->kind : 0u;
}


/** The scale of a stored value; readable for every record kind. */
LIBXS_INLINE unsigned int corpus_value_scale(const void* value)
{
  return (NULL != value) ? ((const corpus_head_t*)value)->scale : 0u;
}


/**
 * Blob keys are twelve bytes, distinct from the sixteen and eighteen of a content
 * key, so the two key spaces cannot meet in the one registry that holds both.
 */
LIBXS_INLINE void corpus_blob_key(unsigned int id, unsigned char key[],
  size_t* key_size)
{
  const unsigned int magic = 0x424c4f42u;
  memcpy(key, &magic, 4);
  memcpy(key + 4, &id, 4);
  memset(key + 8, 0, 4);
  *key_size = 12;
}

/**
 * The entry a stored value stands for: itself if it is a full record, the
 * materialized window if it is a span, and nothing if it is a parent text.
 *
 * A span has to become a real entry because its readers compare token ids and
 * fingerprints, not text - so "span" means REBUILT ON DEMAND, not referenced in
 * place. This flavour CACHES the rebuild, because the pivot index keeps entry
 * pointers for the length of a run; it therefore costs what the stored windows used
 * to, and only the reader that needs stable pointers should use it.
 */
const corpus_entry_t* corpus_entry_view(const void* value);

/**
 * The same entry, materialized into the caller's scratch and forgotten again.
 *
 * This is the flavour that pays for itself: the passes that TRAIN on the windows -
 * the successor embedding, the byte-pair vocabulary, the byte model - read each one
 * once and never look back, so they need no more memory than a single entry. Which
 * of the two a reader wants is decided by whether it keeps the pointer, and getting
 * that wrong is not silent: a cached walk grows to the size of the pool.
 */
const corpus_entry_t* corpus_entry_scan(const void* value,
  corpus_entry_t* scratch);

/** Whether a stored value stands for an entry that can be materialized. */
int corpus_value_viable(const void* value);

/** Bind the lexicon a materialized window must be rebuilt against. */
void corpus_view_bind(libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules,
  int nrules);

/** Release every materialized window. */
void corpus_view_free(void);


/**
 * Enumerate the corpus over FULL entries, skipping every other record kind.
 *
 * Every reader that walks the corpus wants entries, and all but one of them wants
 * only the entries that carry their own metadata. Filtering in the ITERATOR rather
 * than at the cast is what makes the other kinds safe to introduce: a reader keeps
 * dereferencing what it is handed, and there is one place - not forty - where a
 * value is classified. The one reader that does want the derived records
 * (recombination, whose donor pool they are) asks for them explicitly.
 */
LIBXS_INLINE void* corpus_iter_begin(const libxs_registry_t* corpus,
  const void** key, size_t* cursor)
{
  void* result = libxs_registry_begin(corpus, key, cursor);
  while (NULL != result && ENTRY_KIND_FULL != corpus_value_kind(result)) {
    result = libxs_registry_next(corpus, key, cursor);
  }
  return result;
}


LIBXS_INLINE void* corpus_iter_next(const libxs_registry_t* corpus,
  const void** key, size_t* cursor)
{
  void* result = libxs_registry_next(corpus, key, cursor);
  while (NULL != result && ENTRY_KIND_FULL != corpus_value_kind(result)) {
    result = libxs_registry_next(corpus, key, cursor);
  }
  return result;
}


/**
 * The same FULL-entry enumeration, yielding each record's key size as well. The
 * corpus deliberately holds keys of more than one size, so a reader that needs the
 * stored size of a value has to ask the iterator for the key it belongs to.
 */
LIBXS_INLINE void* corpus_iter_begin_length(const libxs_registry_t* corpus,
  const void** key, size_t* key_size, size_t* cursor)
{
  void* result = libxs_registry_begin_length(corpus, key, key_size, cursor);
  while (NULL != result && ENTRY_KIND_FULL != corpus_value_kind(result)) {
    result = libxs_registry_next_length(corpus, key, key_size, cursor);
  }
  return result;
}


LIBXS_INLINE void* corpus_iter_next_length(const libxs_registry_t* corpus,
  const void** key, size_t* key_size, size_t* cursor)
{
  void* result = libxs_registry_next_length(corpus, key, key_size, cursor);
  while (NULL != result && ENTRY_KIND_FULL != corpus_value_kind(result)) {
    result = libxs_registry_next_length(corpus, key, key_size, cursor);
  }
  return result;
}


/**
 * Enumerate the corpus over every record that STANDS FOR an entry, materializing
 * the spans on the way. Yields the values in the same order and at the same
 * ordinals the full-entry flavour would have had before the windows were derived,
 * which is what donor selection ("a donor is an entry later in corpus order")
 * depends on.
 */
LIBXS_INLINE void* corpus_iterx_begin(const libxs_registry_t* corpus,
  const void** key, size_t* cursor)
{
  void* result = libxs_registry_begin(corpus, key, cursor);
  while (NULL != result && 0 == corpus_value_viable(result)) {
    result = libxs_registry_next(corpus, key, cursor);
  }
  return result;
}


LIBXS_INLINE void* corpus_iterx_next(const libxs_registry_t* corpus,
  const void** key, size_t* cursor)
{
  void* result = libxs_registry_next(corpus, key, cursor);
  while (NULL != result && 0 == corpus_value_viable(result)) {
    result = libxs_registry_next(corpus, key, cursor);
  }
  return result;
}


LIBXS_INLINE void corpus_key_from_fprint(const corpus_fprint_t* fp,
  unsigned char key[], size_t* key_size)
{
  unsigned int coords[COMPOSE_NDIMS];
  uint64_t hcode;
  int k;
  for (k = 0; k <= FPRINT_ORDER && k <= fp->order; ++k) {
    double v = fp->l2[k];
    double m = fp->mean[k];
    unsigned int qv, qm;
    if (v < 0) v = 0;
    if (v > 1.0) v = 1.0;
    if (m < -1.0) m = -1.0;
    if (m > 1.0) m = 1.0;
    qv = (unsigned int)(v * ((1 << COMPOSE_BITS) - 1));
    qm = (unsigned int)((m + 1.0) * 0.5 * ((1 << COMPOSE_BITS) - 1));
    coords[k] = qv;
    coords[FPRINT_ORDER + 1 + k] = qm;
  }
  for (k = fp->order + 1; k <= FPRINT_ORDER; ++k) {
    coords[k] = 0;
    coords[FPRINT_ORDER + 1 + k] = 0;
  }
  hcode = libxs_hilbert_bits(coords, COMPOSE_NDIMS, COMPOSE_BITS);
  memcpy(key, &hcode, 8);
  *key_size = 8;
}


void converse_judge_install(const converse_judge_t* judge);
/** Open the installed judge when CONVERSE_HIER_RESCORE asks for one. */
void converse_judge_open(const libxs_registry_t* corpus);
void converse_judge_close(void);
int converse_judge_active(void);
/** CONVERSE_HIER_RESCORE, so a caller can report what the judge changed. */
int converse_judge_verbose(void);
int converse_judge_rescore(const char* query, int query_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates, double bits[]);
int converse_judge_choose(const char* context, int context_length,
  const char* const candidates[], const int candidate_lengths[],
  int ncandidates);
/** Matches converse_recomb_seam_t, so it binds straight into the recomb host. */
int converse_judge_seam_bits(const char* prefix, int prefix_length,
  const char* suffix, int suffix_length, int score_length, double* bits);


/**
 * Parse the command line, reject a mode this role does not serve, then load or
 * ingest the corpus, the lexicon, the rules and the answer ranker: everything
 * both halves read and everything -L writes. Fills `run` on success.
 */
int converse_setup(int argc, char* argv[], int role, converse_run_t* run);

/** Release what converse_setup acquired, including the core model caches. */
void converse_release(converse_run_t* run);

/**
 * Which half an invocation needs, from its flags alone. The role gate and the
 * ALL binary's dispatch read this same answer, so "which binary serves this
 * command" is decided in one place.
 */
int converse_role_of(int argc, char* argv[]);

const libxs_lexnorm_t* converse_lexnorms(void);
int converse_lexnorms_size(void);
const answer_relation_rule_t* converse_rules(void);
size_t converse_rules_size(void);
libxs_ngram_t* converse_ngram_handle(void);
const char* converse_bridge_path(void);
const char* converse_eval_path(void);
const char* converse_predict_eval_path(void);
/** Where the derived fact layer is cached; see the stamp in converse_qa.c. */
const char* converse_facts_path(void);

#endif /*CONVERSE_H*/
