#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_ngram.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_hash.h>
#include <libxs/libxs_hist.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_malloc.h>

#include "converse_qa.h"
#include "converse_recomb.h"

#define ANSWER_MAX 4

#define ANSWER_MIN_SCORE 0.35

#define BRIDGE_LINE_MAX 2048

/** Fact-field marker: this expectation depends on loaded rules. */
#define EVAL_RULE_GOVERNED "-"

#define CONV_TOPIC_MAX 64


typedef struct answer_bridge_t {
  const char* name;
  const char* query;
  const char* evidence;
  const char* reply;
  double score;
} answer_bridge_t;

/**
 * Facts one word reaches. `at` indexes the fact array rather than pointing into
 * it, and lives on the heap rather than inside the registry value, so the list
 * grows instead of being capped.
 */
typedef struct answer_fact_postings_t {
  unsigned int n;
  unsigned int cap;
  unsigned int* at;
} answer_fact_postings_t;

/** Word id -> the facts that word reaches. */
typedef struct answer_fact_index_t {
  libxs_registry_t* store;
  unsigned int nkeys;
} answer_fact_index_t;

#define ANSWER_FACTS_MAGIC 0x54434643u
#define ANSWER_FACTS_VERSION 11
#define ANSWER_LOCATION_PHRASE_MAX 96
/** Propositions one attribute collection may rest on, and cite. */
#define ANSWER_TOPIC_MAX 4
#define ANSWER_TYPE_PHRASE_MAX 128
enum { ANSWER_TYPE_COPULAR = 0, ANSWER_TYPE_APPOSITIVE, ANSWER_TYPE_KIN };
/** Items one enumerated possession answer may state, and cite. */
#define ANSWER_OWN_MAX 6
#define ANSWER_OWN_ITEM_MAX 64

/** Header of the derived-layer cache; see answer_facts_stamp. */
typedef struct answer_facts_header_t {
  unsigned int magic;
  unsigned int version;
  unsigned int stamp;
  unsigned int ncase;
  unsigned int nrelation;
  unsigned int nidentity;
  unsigned int ndescribe;
  unsigned int ndocdef;
  unsigned int nlocation;
  unsigned int ntype;
  unsigned int nown;
} answer_facts_header_t;

typedef struct answer_relation_fact_t {
  char answer[128];
  char relation[64];
  char actor[64];
  char section[ENTRY_SECTION_MAX];
  int answer_len;
  int relation_len;
  int actor_len;
  int section_len;
  unsigned short source;
  unsigned int line;
  int plural;
  int made;
  int active;
  double score;
} answer_relation_fact_t;

typedef struct answer_identity_fact_t {
  char name[64];
  char role[64];
  char section[ENTRY_SECTION_MAX];
  int name_len;
  int role_len;
  int section_len;
  unsigned short source;
  unsigned int line;
  /** Where the role's class term came from; ASSERTED outranks any score. */
  int provenance;
  double score;
} answer_identity_fact_t;

/**
 * WHERE something happened, as a proposition rather than as a sentence.
 *
 * The phrase is VERBATIM - the bytes from just after the actor through the place
 * noun, exactly as the source has them - so a reply is a name followed by a span
 * of that name's own sentence. It is therefore grammatical whenever the source was
 * and cannot state anything the corpus does not, which is the same reason the
 * relation layer renders a proposition instead of generating one.
 *
 * The phrase bound is a REPRESENTATION, not a threshold: a location phrase that
 * runs the length of a clause is not a location phrase, and one that does not fit
 * is dropped rather than truncated into a claim the corpus never made.
 */
typedef struct answer_location_fact_t {
  char actor[64];
  char phrase[ANSWER_LOCATION_PHRASE_MAX];
  char place[64];
  char section[ENTRY_SECTION_MAX];
  int actor_len;
  int phrase_len;
  int place_len;
  int section_len;
  unsigned short source;
  unsigned int line;
  int provenance;
  double score;
} answer_location_fact_t;

/**
 * What an entity IS, from the two shapes prose states a type in: the copular
 * ("Aristotle was a Greek philosopher") and the appositive ("Aristotle, a Greek
 * philosopher, wrote"). These are the definitional shapes, which is why they are
 * the two E4 starts with - encyclopaedic prose states nearly every type this way,
 * and the entity census already supplies the names to hang them on.
 *
 * The phrase is VERBATIM apart from one word: an appositive omits the copula, and
 * the reply takes that word from the rule file (`copula|is`) rather than from a
 * literal here, so the language declares it. Everything else is the source's own
 * bytes in the source's own order.
 */
typedef struct answer_type_fact_t {
  char name[64];
  char phrase[ANSWER_TYPE_PHRASE_MAX];
  char section[ENTRY_SECTION_MAX];
  int name_len;
  int phrase_len;
  int section_len;
  /** The entity at the other end, when the shape names one (kinship). */
  char partner[64];
  int partner_len;
  unsigned short source;
  unsigned int line;
  int shape;
  double score;
} answer_type_fact_t;

/**
 * WHAT BELONGS TO WHOM, from the one shape English marks possession with
 * unambiguously: the possessive apostrophe. "Hector's father" states a relation
 * between Hector and a father in a way "the father of Hector" does not - "of"
 * carries partition, origin, material and authorship as well, so it is a different
 * shape and not this one.
 *
 * The item is the possessed noun phrase, verbatim, and the owner is the name run
 * before the apostrophe. An enumerated answer is then a set of independently
 * attested items, each with its own citation.
 */
typedef struct answer_own_fact_t {
  char owner[64];
  char item[ANSWER_OWN_ITEM_MAX];
  char section[ENTRY_SECTION_MAX];
  int owner_len;
  int item_len;
  int section_len;
  unsigned short source;
  unsigned int line;
  double score;
} answer_own_fact_t;

/**
 * What the article frame attests about one word: how often it HEADS an article-led
 * phrase, how often that article was an indefinite one, and how often each declared
 * indefinite article stands directly before it. The last is what lets a reply pick
 * between "a" and "an" from the corpus rather than from a rule in the C.
 */
typedef struct answer_noun_t {
  long head;
  long mod;
} answer_noun_t;

typedef struct answer_describe_fact_t {
  char role[64];
  char text[192];
  char section[ENTRY_SECTION_MAX];
  int role_len;
  int text_len;
  int section_len;
  unsigned short source;
  unsigned int line;
  double score;
} answer_describe_fact_t;

typedef struct answer_docdef_fact_t {
  char title[ENTRY_SECTION_MAX];
  char header[64];
  char text[COMPOSE_MAXTEXT];
  int title_len;
  int header_len;
  int text_len;
  unsigned short source;
  unsigned int line;
} answer_docdef_fact_t;


static answer_bridge_t* answer_bridge_loaded = NULL;
static size_t answer_bridge_loaded_size = 0;

/**
 * Sections of the most recent fact reply, so a citation can be printed without
 * threading an out-parameter through five resolver signatures. Cleared by
 * answer_fact_reply before dispatch and set by whichever resolver answered;
 * empty means the fact carried no section and no citation is printed.
 *
 * Several, because a reply can rest on several facts, and the two paths that
 * aggregate over facts cited NOTHING. For a claim about attribution an
 * unattributed answer is worse than a misattributed one, since nothing signals
 * it. Each contributing source is named; the usual case contributes one and
 * reads exactly as a single citation always did.
 */
static char answer_fact_section[4 * ENTRY_SECTION_MAX];
/**
 * Where the reply came from in the FILES, beside the titles it came from.
 *
 * A title is only as good as the corpus's own structure - a corpus with no
 * headings has none to give - while a file and a line always exist, so this is the
 * part of a citation that cannot be wrong. One entry per source, because a reply
 * resting on two files should name a range in each rather than one range spanning
 * both, and the range collapses to a single line when it is one line.
 */
typedef struct answer_origin_t {
  unsigned int source;
  unsigned int first;
  unsigned int last;
} answer_origin_t;

#define ANSWER_ORIGIN_MAX 4
static answer_origin_t answer_origins[ANSWER_ORIGIN_MAX];
static int answer_norigins = 0;
static int answer_fact_section_len = 0;
/**
 * Class term the most recent fact reply rested on, and where that term came
 * from, when it was not asserted in the rule file. Carried the same way the
 * citation is, and for the same reason: five resolvers would otherwise each grow
 * an out-parameter. Empty means the reply rests on asserted rules only.
 */
static char answer_fact_learned[64];
static int answer_fact_learned_len = 0;
static int answer_fact_learned_from = RELATION_RULE_ASSERTED;
static libxs_lexicon_t* answer_query_lexicon = NULL;
static const libxs_lexrule_t* answer_query_rules = NULL;
static int answer_query_nrules = 0;
/**
 * Per-token case census: occurrences that carried an initial upper-case letter,
 * and occurrences in total.
 *
 * LIBXS_LEXEME_ENTITY is exactly that per OCCURRENCE (the default rule set marks
 * WORD_INITIAL_UPPER), and the lexicon interns the LOWER-CASED form, so both
 * surface forms of a word share one id. Aggregating the flag by id is therefore
 * what separates a name from a common word: a name is never written lower-case
 * and a common noun is - and the test needs no threshold, because one
 * lower-case occurrence is enough to settle it.
 */
static unsigned int* answer_case_upper = NULL;
static unsigned int* answer_case_total = NULL;
/** Capitals the author CHOSE: the position did not force them. */
static unsigned int* answer_case_unforced = NULL;
/** Occurrences within two words of a quotation boundary (measurement only). */
static unsigned int* answer_case_attrib = NULL;
static unsigned int answer_case_size = 0;
static answer_relation_fact_t* answer_relation_facts = NULL;
static size_t answer_relation_facts_size = 0;
/**
 * The relation a fact states, as a word id, mapped to the facts stating it -
 * with the ALIAS CLOSURE baked in, so a question naming any alias of a relation
 * is one lookup rather than a scan crossed with the rule file. Built after the
 * facts are, because a fact reached through an alias is stored under the
 * canonical relation and only the finished array knows all of them.
 */
static answer_fact_index_t answer_relation_by_relation = { NULL, 0 };
static answer_location_fact_t* answer_location_facts = NULL;
static size_t answer_location_facts_size = 0;
static answer_type_fact_t* answer_type_facts = NULL;
static size_t answer_type_facts_size = 0;
static answer_own_fact_t* answer_own_facts = NULL;
static size_t answer_own_facts_size = 0;
/**
 * Words the corpus uses as VERBS, derived rather than declared. Build-time only:
 * the layer that reads it runs while the facts are being extracted, so a warm run
 * that reuses the cached facts needs no verb set at all.
 */
static libxs_registry_t* answer_verbs = NULL;
static long answer_verbs_nkeys = 0;
/**
 * Words the corpus uses as NOUNS, derived from the article frame the same way, and
 * build-time only for the same reason. Counted rather than flagged, because the
 * question a caller asks is which of the two frames attests a word more often.
 */
static libxs_registry_t* answer_nouns = NULL;
static long answer_nouns_nkeys = 0;
static answer_identity_fact_t* answer_identity_facts = NULL;
static size_t answer_identity_facts_size = 0;
static answer_describe_fact_t* answer_describe_facts = NULL;
static size_t answer_describe_facts_size = 0;
static answer_docdef_fact_t* answer_docdef_facts = NULL;
static size_t answer_docdef_facts_size = 0;
static char conv_topic[CONV_TOPIC_MAX] = "";
static int conv_topic_len = 0;

static long answer_hier_nreorder = 0;
static long answer_hier_nchanged = 0;


static void answer_bridge_free_const(const char* ptr);
static char* answer_bridge_copy_trim(const char* text);
static void answer_bridge_free_loaded(void);
static int answer_bridge_append_loaded(const char* name, const char* query,
  const char* evidence, const char* score, const char* reply);
static int answer_bridge_parse_line(char* line);
static size_t answer_bridge_load_file(const char* path);
static void answer_bridge_report(FILE* stream);
static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len);
static int text_starts_sentence(const char* text, int text_len);
static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id);
static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text);
static int lexeme_stream_has_similar_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  int tolerance);
static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon);
static int query_type_prefers_sentence(int query_type);
static int corpus_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* corpus);
static int answer_query_section(const char* query_text, size_t query_len,
  char* title, int title_size);
static int corpus_entry_section_copy(const corpus_entry_t* entry,
  size_t entry_size, char* title, int title_size);
static int corpus_entry_section_match(const corpus_entry_t* entry,
  size_t entry_size, const char* title, int title_len);
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size);
static void answer_case_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static void answer_case_free(void);
/** Lexicon id of a word as written; 0 when the corpus never saw it. */
static unsigned int answer_word_id(const char* word, int word_len);
static void answer_case_report(FILE* stream);
static int answer_word_is_name(const char* word, int word_len);
static int answer_query_relation_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size);
static int answer_relation_copy_name(char* output, int output_size,
  const char* text, int text_len, int begin, int end);
static int answer_relation_actor_has_token(const char* actor, int actor_len,
  const char* token, int token_len);
static int answer_relation_copy_section_head(char* output, int output_size,
  const corpus_entry_t* entry);
static int answer_relation_copy_antecedent(char* output, int output_size,
  const char* text, int text_len, int cue_pos);
static int answer_relation_find_person_before(const char* text, int text_len,
  int limit, int* term_pos, int* term_len);
/** Only a person answers a "who" question: a name, or a person-class term. */
static int answer_relation_answer_is_person(const char* text, int text_len);
static int answer_relation_copy_person_before(char* output, int output_size,
  const char* text, int text_len, int limit);
static int answer_relation_match_query(const char* query_text,
  size_t query_len, int query_type, const corpus_entry_t* entry,
  answer_relation_match_t* match);
static int answer_relation_section_title(char* output, int output_size,
  const corpus_entry_t* entry, const answer_relation_match_t* match);
static int answer_relation_same_answer(const char* lhs, int lhs_len,
  const char* rhs, int rhs_len);
static void answer_relation_facts_free(void);
static int answer_relation_fact_append(const corpus_entry_t* entry,
  const answer_relation_match_t* match);
static int answer_relation_fact_extract_actor(const char* text, int text_len,
  int verb_pos, int* scan_pos, char* actor, int actor_size);
static int answer_relation_fact_extract_made(const corpus_entry_t* entry,
  int made_pos);
/** Whether any declared function class holds this word. */
static int answer_word_is_function(const char* word, int word_len);
static int answer_relation_fact_extract_active(const corpus_entry_t* entry);
static size_t answer_relation_facts_build(const libxs_registry_t* corpus);
/** Index the finished facts by relation, alias closure included. */
static void answer_relation_facts_index(void);
static void answer_relation_facts_report(FILE* stream);
static void answer_identity_facts_free(void);
static int answer_identity_word_is_name(const char* word, int word_len);
static int answer_identity_fact_append(const char* name, int name_len,
  const char* role, int role_len, const corpus_entry_t* entry, double score);
static int answer_identity_is_connective(const char* word, int word_len);
static size_t answer_identity_facts_build(const libxs_registry_t* corpus);
static void answer_identity_facts_report(FILE* stream);
static int answer_identity_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_location_facts_free(void);
static int answer_location_fact_append(const char* actor, int actor_len,
  const char* phrase, int phrase_len, const char* place, int place_len,
  const corpus_entry_t* entry, double score);
static size_t answer_location_facts_build(const libxs_registry_t* corpus);
static void answer_location_facts_report(FILE* stream);
static int answer_query_type_text(const char* query_text,
  size_t query_len);
static int answer_location_query_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size);
static int answer_location_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_type_facts_free(void);
static int answer_type_fact_append(const char* name, int name_len,
  const char* phrase, int phrase_len, int shape,
  const corpus_entry_t* entry, double score);
static size_t answer_type_facts_build(const libxs_registry_t* corpus);
static void answer_type_facts_report(FILE* stream);
static int answer_type_render(const answer_type_fact_t* fact, char* output,
  size_t output_size);
static int answer_type_fact_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size);
static int answer_type_reply_shape(const char* query_text, size_t query_len,
  char* output, size_t output_size, int shape_only);
/** Answers only from the kinship shape, which is asked before identity. */
static int answer_type_kin_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size);
static int answer_type_kin_append(const corpus_entry_t* entry, int heading_len,
  const char* role, int role_len);
/** Bytes of the PURE name in a token, with any declared possessive mark off. */
static int answer_name_strip(const char* word, int word_len, int* mark);
static void answer_verbs_free(void);
static size_t answer_verbs_build(const libxs_registry_t* corpus);
static void answer_verbs_report(FILE* stream);
static int answer_word_is_verb(const char* word, int word_len);
static long answer_word_verb_count(const char* word, int word_len);
static void answer_nouns_free(void);
static size_t answer_nouns_build(const libxs_registry_t* corpus);
static void answer_nouns_report(FILE* stream);
static answer_noun_t* answer_noun_record(const char* word, int word_len,
  int create);
/** Whether the corpus uses this word more as a noun than as a verb. */
static int answer_word_is_noun(const char* word, int word_len);
static void answer_own_facts_free(void);
static int answer_own_fact_append(const char* owner, int owner_len,
  const char* item, int item_len, const corpus_entry_t* entry, double score);
static size_t answer_own_facts_build(const libxs_registry_t* corpus);
static void answer_own_facts_report(FILE* stream);
static int answer_own_fact_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size);
static int answer_topic_query_name(const char* query_text, size_t query_len,
  char* name, int name_size);
static int answer_topic_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size);
static void answer_describe_facts_free(void);
static int answer_describe_fact_append(const char* role, int role_len,
  const char* text, int text_len, const corpus_entry_t* entry, double score);
static int answer_describe_word_is_article(const char* word, int word_len);
static size_t answer_describe_facts_build(const libxs_registry_t* corpus);
static void answer_describe_facts_report(FILE* stream);
static int answer_describe_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void answer_docdef_facts_free(void);
static int answer_docdef_header(const char* text, int text_len,
  char* header, int header_size, int* header_len);
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus);
static void answer_docdef_facts_report(FILE* stream);
/** Reuse the derived layer when it was built from exactly this input. */
static int answer_facts_load(const libxs_registry_t* corpus,
  const libxs_lexicon_t* lexicon);
static void answer_facts_save(const libxs_registry_t* corpus,
  const libxs_lexicon_t* lexicon);
static int answer_docdef_term(const char* query_text, size_t query_len,
  char* term, int term_size);
static int answer_docdef_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static void conv_reset(void);
static void conv_remember(const char* query_text, size_t query_len);
static int conv_word_is_pronoun(const char* word, int len);
static int conv_rewrite(const char* query_text, size_t query_len,
  char* out, size_t out_size);
static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact);
static int answer_relation_fact_actor_match(const char* query_actor,
  int query_actor_len, const answer_relation_fact_t* fact);
static int answer_relation_fact_section_match(const char* query_section,
  int query_section_len, const answer_relation_fact_t* fact);
static int answer_relation_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size);
static int answer_relation_aggregate_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output,
  size_t output_size);
static double answer_identity_score(const char* query_text, size_t query_len,
  int query_type, const corpus_entry_t* entry);
static int answer_features(const libxs_lexeme_stream_t* query,
  const corpus_entry_t* entry, size_t entry_size, int query_type,
  double inputs[ANSWER_PREDICT_INPUTS]);
static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile);
static double answer_predict_score(const libxs_predict_t* model,
  const double inputs[ANSWER_PREDICT_INPUTS], double base_score);
static int answer_bridge_query_group_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const char* group, int group_len);
static int answer_bridge_query_match(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* spec);
static int answer_bridge_evidence_group_match(const char* text, int text_len,
  const char* group, int group_len);
static int answer_bridge_evidence_match(const corpus_entry_t* entry,
  const char* spec);
static const answer_bridge_t* answer_bridge_match(
  const libxs_lexeme_stream_t* query, const libxs_lexicon_t* lexicon,
  const corpus_entry_t* entry);
static double answer_semantic_bridge_score(const answer_bridge_t* bridge);
static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type);
/** Whether a DECLARED link term makes this a question about the graph. */
static int answer_graph_asked(const char* query_text, size_t query_len);
static int answer_select(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX]);
static int answer_reply_role(char* output, size_t output_size,
  const char* name, int name_len, const char* role);
static void answer_strip_heading_prefix(const char** text, int* text_len);
static size_t answer_append_clean(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len);
static int answer_frame_after(char* output, size_t output_size,
  size_t* output_pos, const char* text, int text_len, const char* marker);
static int answer_frame_keywords_after(char* output, size_t output_size,
  size_t* output_pos, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* text, int text_len,
  const char* marker);
static int answer_bridge_expand_reply(const answer_bridge_t* bridge,
  const char* text, int text_len, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, char* output, size_t output_size);
static int answer_reply_what_is(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const char* text, int text_len,
  char* output, size_t output_size);
static int answer_reply(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_evidence_sentence(const char* query_text, size_t query_len,
  const corpus_entry_t* entry, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules,
  char* output, size_t output_size);
static int answer_query_is_negated(const char* query_text, size_t query_len);
static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output, size_t output_size);
static int answer_citation_len(const char* section, int section_len);
/** The whole citation a reader sees, so the evaluation can check the same text. */
static int answer_citation_text(const char* section, int section_len,
  char* output, size_t output_size);
static void answer_print_citation(const char* section, int section_len);
static void answer_fact_section_set(const char* section, int section_len);
/** Answer a KNOWLEDGE-GRAPH question by stating the path between two entities. */
static int answer_link_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size);
/** The two entities a graph question names, or zero if it names no pair. */
static int answer_link_query(const char* query_text, size_t query_len,
  char* first, int first_size, char* second, int second_size);
/** Record where one item of a reply came from: source id and line. */
static void answer_origin_add(unsigned int source, unsigned int line);
/** Name one more source, for a reply that rests on several facts. */
static void answer_fact_section_add(const char* section, int section_len);
static void answer_fact_learned_set(const char* term, int term_len,
  int provenance);
static void answer_print_learned(void);
static int answer_hier_reorder(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX],
  int answer_count);
/** What a reader is shown for an answer list, printed unless print is 0. */
static int answer_render(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[], int answer_count,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int print, char* output, size_t output_size);
static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  char* out_reply, size_t out_size);
static int text_contains_ci(const char* text, int text_len, const char* term);
static int text_find_word_ci(const char* text, int text_len, const char* term);
static int eval_parse_line(char* line, char* fields[5]);
static int eval_terms_empty(const char* spec);
static int eval_terms_match_text(const char* text, int text_len,
  const char* spec);
static int eval_terms_match_answers(const corpus_entry_t* entries[],
  int nanswers, const char* spec, int top_only);
static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static int recomb_compose_on(void);
static double ngram_gen_contfloor(void);
static double gen_relevance_min(void);
static double gen_overlap(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* a, int alen,
  const char* b, int blen);
static void complete_respond(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile, int budget,
  const char* text, int text_len);
static double converse_score(const corpus_fprint_t* candidate,
  const libxs_fprint_t* query, const corpus_fprint_t* prev);
static int entry_used(const corpus_entry_t* e,
  const corpus_entry_t* const used[], int nused);
static const corpus_entry_t* select_best(void* const candidates[],
  int ncandidates, const corpus_entry_t* const used[], int nused,
  const libxs_fprint_t* query, const corpus_fprint_t* prev,
  int require_scale, unsigned char preferred_scale);
static uint64_t query_hilbert_code(const libxs_fprint_t* fp);
static int respond(const libxs_spatial_t* spatial,
  const libxs_registry_t* corpus, const char* query_text,
  size_t query_len, const libxs_fprint_t* query, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);


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
  memset(&bridge, 0, sizeof(bridge));
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


static int answer_relation_rule_alias_pos(const char* relation,
  const char* text, int text_len, int* alias_len)
{
  int result = -1;
  size_t rule_pos;
  if (NULL != relation && NULL != text && text_len > 0) {
    for (rule_pos = 0; rule_pos < converse_rules_size() && result < 0;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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
      converse_lexnorms(), converse_lexnorms_size(), 1))
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


/**
 * The query type of raw text, for the resolvers that see the question but not the
 * token stream. It re-encodes rather than threading a type through five
 * signatures, and the query is one sentence, so that costs nothing measurable.
 */
static int answer_query_type_text(const char* query_text, size_t query_len)
{
  int result = QUERY_GENERIC;
  libxs_lexeme_stream_t query;
  libxs_lexeme_stream_init(&query);
  if (NULL != query_text && 0 < query_len && NULL != answer_query_lexicon
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(answer_query_lexicon, &query,
      (const unsigned char*)query_text, query_len, answer_query_rules,
      answer_query_nrules, converse_lexnorms(), converse_lexnorms_size(), 0))
  {
    result = query_type_of(&query, answer_query_lexicon);
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


/**
 * The question KIND a declared `ask|` tag names, or QUERY_GENERIC for an unknown tag.
 *
 * The tags are identifiers rather than English - the same status `poss|apostrophe-s`
 * has - so a rule file names the kind in this fixed vocabulary and supplies its own
 * word for it: `ask|who|wer`.
 */
static int query_type_of_tag(const char* tag)
{
  int result = QUERY_GENERIC;
  if (NULL != tag) {
    if (0 == strcmp(tag, "who")) result = QUERY_WHO;
    else if (0 == strcmp(tag, "what")) result = QUERY_WHAT;
    else if (0 == strcmp(tag, "where")) result = QUERY_WHERE;
    else if (0 == strcmp(tag, "when")) result = QUERY_WHEN;
    else if (0 == strcmp(tag, "why")) result = QUERY_WHY;
    else if (0 == strcmp(tag, "how")) result = QUERY_HOW;
    else if (0 == strcmp(tag, "yesno")) result = QUERY_YESNO;
  }
  return result;
}


/**
 * Which question this query asks, read from the DECLARED question words.
 *
 * This function was the LAST PLACE ENGLISH SURVIVED IN THE C: it tested for "who",
 * "what", "where" and eleven more as literals, while `where|`, `why|` and `how|` sat
 * declared in the rule file. Those declare the markers that make a SENTENCE answer a
 * question; `ask|` declares the words that make a QUERY ask one, which is the other
 * half and had never been written down.
 *
 * QUERY_GENERIC now means the RULE FILE does not recognize the question - a fact
 * about the rules rather than about the corpus, and the signal that lets a reply say
 * it did not understand instead of guessing.
 */
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
        const char* tag;
        const char* term;
        int index = 0;
        while (QUERY_GENERIC == result
          && NULL != (term = answer_relation_rule_term_at(RELATION_RULE_ASK,
            index, &tag)))
        {
          if (0 != lexeme_text_is(lexicon, lexeme, term)) {
            result = query_type_of_tag(tag);
          }
          ++index;
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

/**
 * Build the similarity index from the entries' own fingerprints. The registry key
 * is now a content hash, so libxs_spatial_build - which reads the first 8 Bytes
 * of each key as the code - would index hash values and destroy locality. The
 * code is recomputed here from the fingerprint, which is where it belongs.
 *
 * Iteration uses the _length flavor because this registry deliberately holds keys
 * of two sizes (16 Bytes, or 18 when a hash collision needed resolving); the
 * plain iterator cannot report which, and the Bytes beyond key_size are
 * undefined.
 */
static int corpus_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* corpus)
{
  int result = EXIT_FAILURE;
  libxs_registry_info_t info = { 0 };
  if (NULL == sp || NULL == corpus) return EXIT_FAILURE;
  libxs_registry_info(corpus, &info);
  if (0 < info.size) {
    uint64_t* codes = (uint64_t*)libxs_malloc(NULL,
      info.size * sizeof(uint64_t), LIBXS_MALLOC_AUTO);
    void** values = (void**)libxs_malloc(NULL,
      info.size * sizeof(void*), LIBXS_MALLOC_AUTO);
    if (NULL != codes && NULL != values) {
      const void* key = NULL;
      size_t key_size = 0, cursor = 0;
      int count = 0;
      void* value = corpus_iter_begin_length(corpus, &key, &key_size,
        &cursor);
      while (NULL != value && count < (int)info.size) {
        const corpus_entry_t* entry = (const corpus_entry_t*)value;
        unsigned char code_key[16];
        size_t code_size = 0;
        uint64_t code = 0;
        corpus_key_from_fprint(&entry->fprint, code_key, &code_size);
        memcpy(&code, code_key, 8);
        codes[count] = code;
        values[count] = value;
        ++count;
        value = corpus_iter_next_length(corpus, &key, &key_size, &cursor);
      }
      result = libxs_spatial_build_codes(sp, codes, values, count);
    }
    libxs_free(codes);
    libxs_free(values);
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


/**
 * The entry's section, for a caller that must remember it rather than test it.
 * An entry is stored at its actual text length, so entry_size is what says the
 * fixed-offset fields are present at all.
 */
static int corpus_entry_section_copy(const corpus_entry_t* entry,
  size_t entry_size, char* title, int title_size)
{
  int result = 0;
  if (NULL != title && 0 < title_size) {
    title[0] = '\0';
    if (NULL != entry && entry_size >= CORPUS_ENTRY_META_SIZE) {
      result = entry->section_len;
      if (result >= title_size) result = title_size - 1;
      if (0 < result) {
        memcpy(title, entry->section, (size_t)result);
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
  if (NULL != entry && entry_size >= CORPUS_ENTRY_META_SIZE
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


/**
 * Count, per token id, how many occurrences carried an initial upper-case letter
 * and how many there were. One corpus pass, and the only pass that has to see
 * the SURFACE form rather than the interned one.
 */
static void answer_case_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  LIBXS_UNUSED(rules); LIBXS_UNUSED(nrules);
  answer_case_free();
  if (NULL == corpus || NULL == lexicon) return;
  answer_case_size = libxs_lexicon_size(lexicon) + 1;
  answer_case_upper = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_upper));
  answer_case_total = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_total));
  answer_case_unforced = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_unforced));
  answer_case_attrib = (unsigned int*)calloc(answer_case_size,
    sizeof(*answer_case_attrib));
  if (NULL == answer_case_upper || NULL == answer_case_total
    || NULL == answer_case_unforced || NULL == answer_case_attrib)
  {
    answer_case_free();
    return;
  }
  answer_query_lexicon = lexicon;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale && 0 < entry->text_len
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const int heading_len = corpus_title_len(entry->text, entry->text_len);
      /* Two words on each side of a quotation boundary: the ring holds the two
       * behind, the countdown the two ahead. */
      unsigned int prev_id[2];
      int token_index = 0, after = 0, nprev = 0;
      const char* token;
      int token_len = 0;
      prev_id[0] = 0;
      prev_id[1] = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const int at = (int)(token - entry->text);
        const int raw_len = token_len;
        int trimmed = 0;
        int opens = 0, closes = 0, scan;
        while (token_len > trimmed
          && 0 == isalpha((unsigned char)token[trimmed])) ++trimmed;
        while (token_len > trimmed
          && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
        /* The quotation mark is attached to the word, not standing between two
         * of them, so the mark is looked for in the token's own margins. */
        for (scan = 0; scan < trimmed; ++scan) {
          const unsigned char c = (unsigned char)token[scan];
          if (0x98 == c || 0x9c == c || '\'' == c || '"' == c) opens = 1;
        }
        for (scan = token_len; scan < raw_len; ++scan) {
          const unsigned char c = (unsigned char)token[scan];
          if (0x99 == c || 0x9d == c || '\'' == c || '"' == c) closes = 1;
        }
        if (token_len > trimmed) {
          const unsigned int id = answer_word_id(token + trimmed,
            token_len - trimmed);
          if (0 != id && id < answer_case_size) {
            ++answer_case_total[id];
            if (0 != isupper((unsigned char)token[trimmed])) {
              ++answer_case_upper[id];
              if (0 == corpus_case_forced(entry->text, at + trimmed,
                heading_len))
              {
                ++answer_case_unforced[id];
              }
            }
            if (0 < after) {
              ++answer_case_attrib[id];
              --after;
            }
            if (0 != opens) {
              int back;
              for (back = 0; back < nprev; ++back) {
                if (0 != prev_id[back]) ++answer_case_attrib[prev_id[back]];
              }
            }
            prev_id[1] = prev_id[0];
            prev_id[0] = id;
            if (nprev < 2) ++nprev;
          }
        }
        if (0 != closes) after = 2;
        if (0 != opens) {
          nprev = 0;
          prev_id[0] = 0;
          prev_id[1] = 0;
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
}


static void answer_case_free(void)
{
  free(answer_case_upper);
  free(answer_case_total);
  free(answer_case_unforced);
  free(answer_case_attrib);
  answer_case_upper = NULL;
  answer_case_total = NULL;
  answer_case_unforced = NULL;
  answer_case_attrib = NULL;
  answer_case_size = 0;
}


/**
 * What the census decided, and how much the UNFORCED rule changed it.
 *
 * The attribution column is a MEASUREMENT, not an input: a speaker stands within
 * two words of a quotation boundary, so if that frame carried nameness the
 * accepted names would sit in it far more often than the words the rule turned
 * down. Reported so the question can be answered from counts rather than from
 * intuition.
 */
static void answer_case_report(FILE* stream)
{
  unsigned int id, nname = 0, nforced = 0;
  libxs_hist_t* hist_name = libxs_hist_create(8, 2, NULL, NULL, NULL);
  libxs_hist_t* hist_other = libxs_hist_create(8, 2, NULL, NULL, NULL);
  if (NULL == stream || NULL == answer_case_total) return;
  for (id = 1; id < answer_case_size; ++id) {
    if (0 != answer_case_total[id]
      && answer_case_upper[id] == answer_case_total[id])
    {
      double sample[2];
      sample[0] = (double)answer_case_attrib[id]
        / (double)answer_case_total[id];
      sample[1] = (double)answer_case_total[id];
      if (0 != answer_case_unforced[id]) {
        ++nname;
        libxs_hist_push(NULL, hist_name, sample);
      }
      else {
        ++nforced;
        libxs_hist_push(NULL, hist_other, sample);
      }
    }
  }
  fprintf(stream, "names: %u attested, %u rejected as position-forced\n",
    nname, nforced);
  libxs_hist_print(stream, hist_name, NULL, "attribution rate of NAMES");
  libxs_hist_print(stream, hist_other, NULL, "attribution rate of FORCED");
  libxs_hist_destroy(hist_name);
  libxs_hist_destroy(hist_other);
}


/**
 * Does the corpus use this word as a NAME? True when every occurrence carried an
 * initial upper-case letter, which is a construction rule rather than a
 * threshold: a single lower-case occurrence makes the word a common one.
 *
 * The word arrives as the questioner typed it, so it is lower-cased before the
 * lookup - libxs_lexicon_id matches the stored bytes, and the lexicon stores the
 * normalized (lower-case) form, so passing a capitalized word would simply
 * miss.
 */
/**
 * The lexicon id of a word as written anywhere, 0 if the corpus never saw it.
 * Never creates: an id minted here would enter the vocabulary every model is
 * built from, so a question could change a BPC figure.
 */
static unsigned int answer_word_id(const char* word, int word_len)
{
  unsigned int result = 0;
  if (NULL != word && 0 < word_len && word_len <= LIBXS_LEXEME_MAXBYTES
    && NULL != answer_query_lexicon)
  {
    char lower[LIBXS_LEXEME_MAXBYTES + 1];
    int pos;
    for (pos = 0; pos < word_len; ++pos) {
      lower[pos] = (char)tolower((unsigned char)word[pos]);
    }
    lower[word_len] = '\0';
    result = libxs_lexicon_id(answer_query_lexicon, lower, word_len, 0, 0);
  }
  return result;
}


/**
 * Index key for a word: a hash of its lower-cased bytes, deliberately NOT a
 * lexicon id.
 *
 * An id is 0 for any word the corpus never attested, and a fact's relation is
 * the CANONICAL name from the rule file, which the corpus need not use even
 * where its aliases are everywhere - so keying on ids would drop exactly those
 * facts out of the index, silently. Every word has a hash. A collision costs a
 * candidate that the predicates then reject; it cannot cost an answer.
 */
static unsigned int answer_word_key(const char* word, int word_len)
{
  unsigned int result = 0;
  char lower[64];
  if (NULL != word && 0 < word_len && word_len < (int)sizeof(lower)) {
    int pos;
    for (pos = 0; pos < word_len; ++pos) {
      lower[pos] = (char)tolower((unsigned char)word[pos]);
    }
    result = libxs_hash(lower, (unsigned int)word_len, 0x9e3779b9u);
    if (0 == result) result = 1;
  }
  return result;
}


static int answer_word_is_name(const char* word, int word_len)
{
  int result = 0;
  const unsigned int id = answer_word_id(word, word_len);
  if (0 != id && NULL != answer_case_total && id < answer_case_size
    && 0 != answer_case_total[id])
  {
    /**
     * Two conditions, both counts of positions. Never written lower-case, as
     * before - and capitalized at least once where the POSITION did not force
     * it. The second is what the first was standing in for: a word that only
     * ever opens a sentence, a heading or a quoted utterance is capitalized by
     * typesetting, and reading that as a name is the same mistake as reading a
     * questioner's shift key as one.
     */
    result = (answer_case_upper[id] == answer_case_total[id]
      && 0 != answer_case_unforced[id]) ? 1 : 0;
  }
  return result;
}


/**
 * The word a "who/what is X" question asks about, skipping any leading skip|
 * term. Capitalization is NOT reported: it used to be, and the three resolvers
 * routed on it - identity required an upper-case initial, the relation paths
 * required a lower-case one. That made the user's typing decide which layer
 * answered, and it was wrong in both directions: a name typed lower-case was
 * refused an answer its capitalized spelling gets, while a capitalized role word
 * escaped the relation layer and fell through to raw evidence under a
 * misattributed citation. Whether a word NAMES something is a property of the
 * corpus, which the fact index already established when it built its names, so
 * the resolvers consult that instead.
 */
static int answer_query_be_word(const char* query_text, size_t query_len,
  char* word, int word_size)
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


/**
 * Is this token one of the WORDS of the actor phrase? Word-bounded, because a
 * substring test let a short actor match inside a longer, unrelated word: the
 * sentence was scanned for the letters rather than for the word, so the reply
 * asserted a relation to an actor that is not in it, and cited the source.
 */
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
    result = text_contains_word_ci(actor, actor_len, token_buf);
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
    for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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


/**
 * Can this text answer a "who" question at all? Either it is written as a proper
 * noun HERE, or a rule puts it in the person class. Nothing else is a person,
 * and the relation layer serves QUERY_WHO only.
 *
 * Capitalization decides because this is CORPUS text, where a capital mid-
 * sentence is the author saying "name" - the same signal the entity lexrule
 * reads. That is the opposite of the query-side defect fixed earlier, where
 * capitalization of the QUERY routed an answer: a reader's shift key is not
 * evidence about the world, and an author's is. The object always follows a
 * verb, so it is never capitalized merely by standing first in its sentence.
 */
static int answer_relation_answer_is_person(const char* text, int text_len)
{
  return (NULL != text && 0 < text_len
    && (0 != isupper((unsigned char)text[0])
      || 0 != answer_relation_rule_has_term(RELATION_RULE_PERSON, text,
        text_len))) ? 1 : 0;
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
  int relation_len;
  int actor_len;
  if (NULL == query_text || NULL == entry || NULL == match
    || QUERY_WHO != query_type) return 0;
  memset(match, 0, sizeof(*match));
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  if (relation_len <= 0 || 0 != answer_word_is_name(relation, relation_len)) {
    return 0;
  }
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
    int actor_seen = text_contains_word_ci(entry->text, entry->text_len,
      actor);
    if (rel_pos < 0) alt_pos = answer_relation_rule_alias_pos(relation,
      entry->text, entry->text_len, &alt_len);
    verb_pos = (rel_pos >= 0) ? rel_pos : alt_pos;
    verb_len = (rel_pos >= 0) ? relation_len : alt_len;
    /**
     * The actor must be ATTESTED IN THIS SENTENCE. There used to be an escape
     * hatch here: if the actor was absent but the word "he" stood before the
     * verb, the actor counted as seen - treating a pronoun as an anaphor for
     * whatever the reader happened to ask about. It left the actor
     * unconstrained, so the reply asserted a relation to an actor the corpus
     * never mentions AND attached a citation to it. Resolving the pronoun would
     * need anaphora; refusing to claim an actor the sentence never names needs
     * nothing, and is the truth about the sentence.
     */
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
      /**
       * Only a PERSON can answer a "who" question, so the direct object has to
       * be one. This used to be a blacklist of five pronouns, which is the
       * judge-it-afterwards pattern that has never held: a first-person pronoun
       * object and an adjective left behind after the real object was skipped
       * both became answers, and both were then aggregated into an otherwise
       * correct sentence. A pronoun object still falls through to the search
       * before the verb below; an object that is neither a person nor a pronoun
       * now yields no fact at all, which is the truth about it.
       */
      if (obj_end > obj_begin
        && 0 != answer_relation_answer_is_person(entry->text + obj_begin,
          obj_end - obj_begin))
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
    { int made_len = 0;
      const char* made_term = answer_relation_rule_first_term(
        RELATION_RULE_RESULT, &made_len);
      while (NULL != made_term && 0 < made_len && rel_pos > made_scan) {
        int next_pos = text_find_word_ci(entry->text + made_scan,
          rel_pos - made_scan, made_term);
        if (next_pos < 0) break;
        made_pos = made_scan + next_pos;
        made_scan = made_pos + made_len;
      }
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


static void answer_fact_index_free(answer_fact_index_t* index)
{
  if (NULL != index) {
    if (NULL != index->store) {
      const void* key = NULL;
      size_t cursor = 0;
      void* value = libxs_registry_begin(index->store, &key, &cursor);
      while (NULL != value) {
        free(((answer_fact_postings_t*)value)->at);
        value = libxs_registry_next(index->store, &key, &cursor);
      }
      libxs_registry_destroy(index->store);
    }
    index->store = NULL;
    index->nkeys = 0;
  }
}


/**
 * Record that the word `id` reaches the fact at `at`.
 *
 * The registry copies a VALUE, so the value is a HEADER and the list of facts
 * hangs off it: registering the list itself would copy the whole thing on every
 * append, which is what forces a fixed cap and a truncation count elsewhere in
 * the sample. Here the list simply grows, and the facts are referred to by
 * position rather than by pointer because the fact array is grown by realloc -
 * a stored pointer would dangle at the next append.
 */
static int answer_fact_index_add(answer_fact_index_t* index, unsigned int id,
  unsigned int at)
{
  int result = EXIT_SUCCESS;
  answer_fact_postings_t* postings;
  if (NULL == index || 0 == id) return EXIT_FAILURE;
  if (NULL == index->store) {
    index->store = libxs_registry_create();
    index->nkeys = 0;
    if (NULL == index->store) result = EXIT_FAILURE;
  }
  postings = (EXIT_SUCCESS == result)
    ? (answer_fact_postings_t*)libxs_registry_get(index->store, &id,
        sizeof(id), NULL) : NULL;
  if (EXIT_SUCCESS == result && NULL == postings) {
    answer_fact_postings_t fresh;
    memset(&fresh, 0, sizeof(fresh));
    postings = (NULL != libxs_registry_set(index->store, &id, sizeof(id),
      &fresh, sizeof(fresh), NULL))
      ? (answer_fact_postings_t*)libxs_registry_get(index->store, &id,
          sizeof(id), NULL) : NULL;
    if (NULL != postings) ++index->nkeys;
    else result = EXIT_FAILURE;
  }
  /* Appended in fact order and never out of it, so a lookup yields the facts in
   * the order a scan of the array would have reached them. */
  if (EXIT_SUCCESS == result && (0 == postings->n
    || postings->at[postings->n - 1] != at))
  {
    if (postings->n == postings->cap) {
      const unsigned int cap = (0 < postings->cap) ? (2 * postings->cap) : 8;
      unsigned int* grown = (unsigned int*)realloc(postings->at,
        (size_t)cap * sizeof(*grown));
      if (NULL != grown) {
        postings->at = grown;
        postings->cap = cap;
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) postings->at[postings->n++] = at;
  }
  return result;
}


static const unsigned int* answer_fact_index_get(
  const answer_fact_index_t* index, unsigned int id, unsigned int* n)
{
  const unsigned int* result = NULL;
  *n = 0;
  if (NULL != index && NULL != index->store && 0 != id) {
    const answer_fact_postings_t* postings =
      (const answer_fact_postings_t*)libxs_registry_get(index->store, &id,
        sizeof(id), NULL);
    if (NULL != postings) {
      result = postings->at;
      *n = postings->n;
    }
  }
  return result;
}


static void answer_relation_facts_free(void)
{
  answer_fact_index_free(&answer_relation_by_relation);
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
  memset(&fact, 0, sizeof(fact));
  fact.answer_len = 0;
  /**
   * The section title stands in for the answer of a PASSIVE fact, whose patient
   * names the thing the section is about. An ACTIVE fact's object is a phrase of
   * its own clause, so substituting the title replaces it with something the clause
   * never said - on the wiki fixture that is how a caption read as a heading
   * ("Thumb|300px|A Farmer In ...") became the object of a fact.
   */
  if (match->actor_len > 0 && 0 == match->active) {
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
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
  if (fact.section_len > 0) {
    memcpy(fact.section, entry->section, (size_t)fact.section_len);
    fact.section[fact.section_len] = '\0';
  }
  fact.plural = match->plural;
  fact.made = match->made;
  fact.active = match->active;
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
    /**
     * The actor must be something the corpus uses as a NOUN or attests as a NAME.
     * Neither test alone works here: an alias actor is often a common noun ("the
     * wolf"), which no census knows, and often a name ("Gretel"), which the article
     * frame never sees. Requiring one OR the other is what stops a sentence-initial
     * function word from becoming an entity - "Then eaten Grandmother" and "Do
     * eaten Grandmother" were edges of the fact graph, and a multi-hop walk over
     * that graph would state paths through a node named "Then".
     */
    if (end > begin && end - begin < actor_size
      && 0 == answer_relation_rule_has_term(RELATION_RULE_SKIP,
        text + begin, end - begin)
      && (0 != answer_word_is_noun(text + begin, end - begin)
        || 0 != answer_identity_word_is_name(text + begin, end - begin)))
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
  /**
   * The relation word must be in the SAME clause as "made". Skipping every
   * non-alphanumeric byte walked straight across a sentence end: "your fortune
   * would be made.' 'Very true: but how" yielded the relation "Very", and the
   * matcher then supplied an answer from the entry, asserting "Hans is to be made
   * Very." A count of 84 facts showed nothing; the attribute collection printed it.
   * One bound, no threshold - the same one the location layer needed.
   */
  rel_begin = made_pos + 4;
  while (rel_begin < entry->text_len
    && 0 == isalnum((unsigned char)entry->text[rel_begin]))
  {
    const char ch = entry->text[rel_begin];
    if ('.' == ch || '!' == ch || '?' == ch || ';' == ch || ':' == ch
      || ',' == ch)
    {
      return 0;
    }
    ++rel_begin;
  }
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


/**
 * Extract the PASSIVE shape generally: a patient, a copula, a verb, "by", a name.
 *
 * The alias rules already read passives whose verb the rule file DECLARES
 * ("alias|eaten|devoured"), which is what makes them askable. This reads the shape
 * itself, so any verb the corpus uses gives an entity-to-entity edge - "Achilles
 * was visited by Odysseus" - without a rule per verb. The frame is the evidence:
 * a copula, one word, and a declared "by" followed by a NAME is a passive in
 * English, and no morphology or verb class is consulted to see it.
 *
 * The agent must be a name, because that is what makes the fact an edge from a node
 * rather than a sentence about nobody. The patient may be anything, since a true
 * proposition about a named agent is worth keeping even when what it acted on is a
 * common noun.
 *
 * Facts are appended to the RELATION layer rather than to a new one: that layer is
 * already indexed by relation with the alias closure, already answers "who was V by
 * X", and is already collected by the attribute walk. A new layer would have needed
 * all three again.
 */
static int answer_relation_fact_extract_passive(const corpus_entry_t* entry,
  int by_pos, int by_len)
{
  int result = 0;
  const char* text = entry->text;
  const char* text_end = entry->text + entry->text_len;
  const char* verb_end = text + by_pos;
  const char* verb;
  const char* copula_end;
  const char* copula;
  int verb_len = 0, copula_len = 0;
  /* Backwards: the verb, then the copula that must govern it. */
  while (verb_end > text && 0 != isspace((unsigned char)verb_end[-1])) {
    --verb_end;
  }
  verb = verb_end;
  while (verb > text && 0 != isalpha((unsigned char)verb[-1])) --verb;
  verb_len = (int)(verb_end - verb);
  copula_end = verb;
  while (copula_end > text && 0 != isspace((unsigned char)copula_end[-1])) {
    --copula_end;
  }
  copula = copula_end;
  while (copula > text && 0 != isalpha((unsigned char)copula[-1])) --copula;
  copula_len = (int)(copula_end - copula);
  if (2 < verb_len && verb_len < 64 && 0 < copula_len
    && 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA, copula,
      copula_len)
    && 0 == answer_relation_rule_is_term(RELATION_RULE_SKIP, verb, verb_len)
    && 0 == answer_relation_rule_is_term(RELATION_RULE_COPULA, verb, verb_len))
  {
    const char* agent = text + by_pos + by_len;
    const char* agent_end;
    int agent_len = 0;
    while (agent < text_end && 0 != isspace((unsigned char)*agent)) ++agent;
    agent_end = agent;
    while (agent_end < text_end
      && ('-' == *agent_end || 0 != isalpha((unsigned char)*agent_end)))
    {
      ++agent_end;
    }
    /* The agent is a maximal NAME RUN, so "World War II" is not just "World". */
    while (agent_end < text_end && ' ' == agent_end[0]) {
      const char* more = agent_end + 1;
      int more_len = 0;
      while (more + more_len < text_end
        && ('-' == more[more_len]
          || 0 != isalpha((unsigned char)more[more_len]))) ++more_len;
      if (0 == more_len
        || 0 == answer_identity_word_is_name(more, more_len)) break;
      agent_end = more + more_len;
    }
    agent_len = (int)(agent_end - agent);
    if (2 < agent_len && agent_len < 64
      && 0 != answer_identity_word_is_name(agent, agent_len))
    {
      /**
       * The patient is the NOUN PHRASE before the copula, not the clause.
       *
       * Taking the clause read the conjunction, the subordinator and the auxiliary
       * chain into the patient - "and | used | NASA", "Algeria has | inhabited |
       * Berbers", "tossed away after | invented | Athena" - so the edge was right
       * and the thing it pointed at was a fragment. Walking backwards instead: over
       * the auxiliaries, which belong to the verb rather than to the patient, then
       * word by word while the word can belong to a noun phrase, stopping AT an
       * article since a phrase starts there. An empty result rejects the fact, which
       * is the right answer whenever the patient is a pronoun or implicit.
       */
      const char* patient = NULL;
      const char* patient_end = copula;
      const char* patient_last = NULL;
      int patient_len = 0;
      int taken = 0;
      while (patient_end > text && 0 != isspace((unsigned char)patient_end[-1])) {
        --patient_end;
      }
      while (taken < 5) {
        const char* word_end = patient_end;
        const char* word_begin = word_end;
        int stop = 0;
        while (word_begin > text
          && 0 != isalpha((unsigned char)word_begin[-1])) --word_begin;
        if (word_begin == word_end) break;
        { const int len = (int)(word_end - word_begin);
          if (0 != answer_relation_rule_is_term(RELATION_RULE_AUX, word_begin,
            len))
          {
            if (NULL != patient) break;
          }
          else if (0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
            word_begin, len))
          {
            patient = word_begin;
            stop = 1;
          }
          else if (0 != answer_relation_rule_is_term(RELATION_RULE_SKIP,
              word_begin, len)
            || 0 != answer_relation_rule_is_term(RELATION_RULE_PREP, word_begin,
              len)
            || 0 != answer_relation_rule_is_term(RELATION_RULE_NEGATE,
              word_begin, len)
            || 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA,
              word_begin, len)) break;
          else {
            patient = word_begin;
            /* The END of the phrase is the LAST content word, which walking
             * backwards meets FIRST: taking it from the copula instead put the
             * stepped-over auxiliary back in ("Algeria has | inhabited"). */
            if (NULL == patient_last) patient_last = word_end;
            ++taken;
          }
        }
        if (0 != stop) break;
        patient_end = word_begin;
        while (patient_end > text
          && 0 != isspace((unsigned char)patient_end[-1])) --patient_end;
        if (patient_end > text && 0 == isalpha((unsigned char)patient_end[-1])) {
          break;
        }
      }
      if (NULL != patient && NULL != patient_last && patient_last > patient) {
        patient_len = (int)(patient_last - patient);
      }
      if (NULL != patient && 2 < patient_len && patient_len < 128) {
        answer_relation_match_t match;
        memset(&match, 0, sizeof(match));
        memcpy(match.answer, patient, (size_t)patient_len);
        match.answer_len = patient_len;
        memcpy(match.relation, verb, (size_t)verb_len);
        match.relation_len = verb_len;
        memcpy(match.actor, agent, (size_t)agent_len);
        match.actor_len = agent_len;
        match.plural = (0 != libxs_striequal(copula, (size_t)copula_len,
          "were", 4) || 0 != libxs_striequal(copula, (size_t)copula_len,
            "are", 3)) ? 1 : 0;
        match.score = 1.0;
        if (EXIT_SUCCESS == answer_relation_fact_append(entry, &match)) {
          result = 1;
        }
      }
    }
  }
  return result;
}


static int answer_word_is_function(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word && 0 < word_len) {
    result = (0 != answer_relation_rule_is_term(RELATION_RULE_SKIP, word,
        word_len)
      || 0 != answer_relation_rule_is_term(RELATION_RULE_PREP, word, word_len)
      || 0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE, word,
        word_len)
      || 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA, word, word_len)
      || 0 != answer_relation_rule_is_term(RELATION_RULE_AUX, word, word_len)
      || 0 != answer_relation_rule_is_term(RELATION_RULE_NEGATE, word,
        word_len)) ? 1 : 0;
  }
  return result;
}


/**
 * Extract the ACTIVE transitive shape: a name, a verb, and what it acted on.
 *
 * The frame is a maximal name run, ONE word, and either another name run or an
 * article-headed noun phrase - "Etruscans brought the Greek alphabet", "Achilles
 * defeated Memnon". Only the DERIVED verb class says the middle word is a verb,
 * which is the point: that class comes from the auxiliary frame
 * (answer_verbs_build), so no verb is written in the C and no morphology is read.
 *
 * THE SAME CLASS IS USED IN BOTH POLARITIES HERE, and both are safe for the same
 * reason. As a REQUIREMENT on the middle word, what the class lacks - English
 * irregular past simple, which no auxiliary governs - costs a fact never extracted
 * rather than a fact that is wrong ("took", "gave", "wrote" are lost this way, and
 * "became" belongs to the type shape anyway). As a REJECTION on the word before the
 * subject, a name the corpus puts after a verb is that verb's OBJECT and not the
 * subject of its clause. What made a requirement unsafe in the location layer was
 * that it discarded a truth already found; here it declines to look.
 *
 * The subject must be a name, so the fact hangs off a node; the object may be a
 * common noun, since a true proposition about a named actor is worth keeping. Same
 * asymmetry as the passive shape, and the facts land in the same layer.
 */
static int answer_relation_fact_extract_active(const corpus_entry_t* entry)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const int heading_len = corpus_title_len(entry->text, entry->text_len);
  const char* text_end = entry->text + entry->text_len;
  const char* name = NULL;
  const char* name_end = NULL;
  const char* token;
  int name_len = 0, name_subject = 0;
  int token_index = 0, token_len = 0;
  int result = 0;
  while (NULL != (token = libxs_strtoken(entry->text, delims, token_index,
    &token_len)))
  {
    const char* raw = token;
    while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
      ++token;
      --token_len;
    }
    while (token_len > 0
      && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
    if (token_len > 0 && raw >= entry->text + heading_len) {
      if (0 != answer_identity_word_is_name(token, token_len)) {
        const int extend = (NULL != name && NULL != name_end
          && name_end + 1 == token && ' ' == *name_end) ? 1 : 0;
        if (0 == extend) {
          const char* before = raw;
          while (before > entry->text
            && 0 != isspace((unsigned char)before[-1])) --before;
          name = token;
          name_subject = 1;
          /* A name a COMMA introduces is a list item or an apposition, the same
           * left-edge test the type shape needed. */
          if (before > entry->text && ',' == before[-1]) name_subject = 0;
          if (0 != name_subject && before > entry->text) {
            const char* word_end = before;
            const char* word = before;
            while (word > entry->text
              && 0 != isalpha((unsigned char)word[-1])) --word;
            if (word < word_end) {
              const int len = (int)(word_end - word);
              if (0 != answer_relation_rule_is_term(RELATION_RULE_PREP, word,
                  len)
                /**
                 * A LOWER-CASE NOUN before the name makes the name that noun's
                 * appositive rather than the subject of a clause: "the postal
                 * authority in West Germany turned a large radio dish" is not
                 * Germany turning one. Lower case is required because a capitalized
                 * one belongs to the name run itself ("President-elect Lincoln"),
                 * and answer_word_is_noun rather than mere attestation because the
                 * broader test costs 31 facts including fixture edges ("Berbers
                 * adopted Islam", "French invaded Algiers").
                 */
                || (0 != islower((unsigned char)*word)
                  && 0 != answer_word_is_noun(word, len))
                || 0 != answer_word_is_verb(word, len))
              {
                name_subject = 0;
              }
            }
          }
        }
        name_len = (int)(token + token_len - name);
        name_end = token + token_len;
      }
      else if (NULL != name && NULL != name_end && 0 != name_subject
        && name_end + 1 == token && ' ' == *name_end
        && 2 < token_len && token_len < 64
        && 0 != islower((unsigned char)*token)
        /* A token carrying anything but letters is not a word of the clause:
         * "east of the Indus and/or Ganges" read "and/or" as the verb, since the
         * corpus does put that string where the auxiliary frame looks. */
        && token_len == (int)strspn(token, "abcdefghijklmnopqrstuvwxyz")
        && 0 == answer_word_is_function(token, token_len)
        /* And the corpus must not use it more as a NOUN, because then this is a
         * bare appositive and not a clause: "Egyptian god Horus" and "Sparta
         * defeated Athens" are the same three token kinds in the same order, and
         * only the corpus's own usage of the middle word tells them apart. What
         * the noun class takes here the TYPE layer states properly. */
        && 0 == answer_word_is_noun(token, token_len)
        && 0 != answer_word_is_verb(token, token_len))
      {
        const char* object = token + token_len;
        const char* object_end;
        int object_len = 0;
        while (object < text_end && ' ' == *object) ++object;
        object_end = object;
        while (object_end < text_end
          && ('-' == *object_end || 0 != isalpha((unsigned char)*object_end)))
        {
          ++object_end;
        }
        if (object_end > object) {
          const int head_len = (int)(object_end - object);
          if (0 != answer_identity_word_is_name(object, head_len)) {
            while (object_end < text_end && ' ' == *object_end) {
              const char* more = object_end + 1;
              int more_len = 0;
              while (more + more_len < text_end
                && ('-' == more[more_len]
                  || 0 != isalpha((unsigned char)more[more_len]))) ++more_len;
              if (0 == more_len
                || 0 == answer_identity_word_is_name(more, more_len)) break;
              object_end = more + more_len;
            }
            object_len = (int)(object_end - object);
          }
          /**
           * Otherwise the object must open with an ARTICLE, which is what makes it
           * a noun phrase: the same bound the type shape needed to stop capturing
           * every predicate. The phrase then runs to the first declared function
           * word, so "the son of Philip" ends at the preposition, and a single space
           * is the only separator admitted, so a comma ends the clause and with it
           * the phrase.
           */
          else if (0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
            object, head_len))
          {
            int taken = 0;
            while (object_end - object < 120
              && object_end < text_end && ' ' == *object_end)
            {
              const char* more = object_end + 1;
              int more_len = 0;
              /**
               * AN AMPERSAND JOINS, and only here. "the Alton & Sangamon Railroad" and
               * "the Boys & Girls Club" are each ONE thing, and the phrase used to end
               * at the mark, so "Lincoln represented the Alton" asserted something the
               * corpus does not say. The ARTICLE is what licenses the join: it says the
               * whole is one noun phrase, so the joined parts cannot be two entities.
               * Without that mark the join is REFUTED - 36 `&` pairs on 2 MB and many
               * join two entities ("Milius & Francis Ford Coppola"), with nothing in the
               * text to tell the readings apart. Article-headed: 2 of 2 correct.
               */
              if ('&' == *more && more + 2 < text_end && ' ' == more[1]
                && 0 != answer_relation_rule_is_term(RELATION_RULE_JOIN,
                  "ampersand", 9))
              {
                more += 2;
                while (more + more_len < text_end
                  && 0 != isalpha((unsigned char)more[more_len])) ++more_len;
                if (0 == more_len
                  || 0 == answer_identity_word_is_name(more, more_len)) break;
                object_end = more + more_len;
                ++taken;
                continue;
              }
              while (more + more_len < text_end
                && ('-' == more[more_len] || '\'' == more[more_len]
                  || 0 != isalpha((unsigned char)more[more_len]))) ++more_len;
              if (0 == more_len
                || 0 != answer_word_is_function(more, more_len)) break;
              object_end = more + more_len;
              ++taken;
            }
            /**
             * The phrase runs to where the corpus ends one, not for a fixed number of
             * words. A COUNT was the bound and it cut the longer phrases at a
             * modifier ("the first successful internal-combustion", "the most common
             * character"), which is a false proposition rather than a short one.
             *
             * THE TAIL CANNOT BE GATED BY EITHER DERIVED CLASS, and three variants
             * were REFUTED by reading the facts. Cutting BACK to the last attested
             * phrase head truncated "the history books" to "the history", because head
             * attestation is sparse for plurals - the error being removed,
             * reintroduced. Requiring the tail to appear in the article frame at all
             * cost 26 truths for 6 rejections, since that frame sees a fraction of the
             * nouns a corpus uses. Rejecting a tail in the DERIVED VERB class cost 19
             * truths for 2, because English nouns are verb-homographous exactly where
             * they are frequent ("the Australian Open", "a spectacular run", "the
             * claim", "the use"). What remains - a participle or an undeclared adverb
             * taken into the phrase - is a gap in a DECLARED class, and belongs in
             * the rule file rather than in a test here.
             */
            if (0 < taken) object_len = (int)(object_end - object);
          }
        }
        if (0 < object_len) {
          /**
           * A copula or auxiliary AFTER the phrase means the phrase is the subject
           * of an EMBEDDED CLAUSE rather than what the verb acted on: "Jorindel saw
           * the nightingale was gone" is not Jorindel seeing a nightingale, and
           * "Catherine thought the door was open" is not Catherine thinking a door.
           * Verbs of perception and cognition take a clause, and nothing else about
           * the frame distinguishes one - this mark does, and it is declared.
           */
          const char* after = object_end;
          int after_len = 0;
          while (after < text_end && ' ' == *after) ++after;
          while (after + after_len < text_end
            && 0 != isalpha((unsigned char)after[after_len])) ++after_len;
          if (0 < after_len
            && (0 != answer_relation_rule_is_term(RELATION_RULE_COPULA, after,
                after_len)
              || 0 != answer_relation_rule_is_term(RELATION_RULE_AUX, after,
                after_len)))
          {
            object_len = 0;
          }
        }
        if (2 < object_len && name_len < 64 && object_len < 128) {
          answer_relation_match_t match;
          memset(&match, 0, sizeof(match));
          memcpy(match.answer, object, (size_t)object_len);
          match.answer_len = object_len;
          memcpy(match.relation, token, (size_t)token_len);
          match.relation_len = token_len;
          memcpy(match.actor, name, (size_t)name_len);
          match.actor_len = name_len;
          match.active = 1;
          match.score = 1.0;
          if (EXIT_SUCCESS == answer_relation_fact_append(entry, &match)) {
            ++result;
          }
        }
        name = NULL;
        name_end = NULL;
        name_subject = 0;
      }
      else {
        name = NULL;
        name_end = NULL;
        name_subject = 0;
      }
    }
    ++token_index;
  }
  return result;
}


static size_t answer_relation_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  corpus_entry_t scratch;
  void* value;
  size_t result = 0;
  answer_relation_facts_free();
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = corpus_entry_scan(value, &scratch);
    int made_scan = 0;
    size_t rule_pos;
    /**
     * A CAPTION IS NOT A PROPOSITION. The section table already refuses to read one
     * as a heading, and the same mark answers the question this layer was asking
     * wrongly: "thumb|300px|Alexander the Great fighting Persian king Darius III"
     * yielded "BC | original | Greek" and a photo credit yielded "Hale Bopp | using
     * | a standard". The text STAYS in the corpus - it is words a reader wrote and
     * the language models count it - and only the fact layers decline to state a
     * proposition on its behalf.
     */
    if (0 != corpus_line_markup(entry->text, entry->text_len)) {
      value = corpus_iterx_next(corpus, &key, &cursor);
      continue;
    }
    { int made_len = 0;
      const char* made_term = answer_relation_rule_first_term(
        RELATION_RULE_RESULT, &made_len);
      while (NULL != made_term && 0 < made_len
        && made_scan < entry->text_len)
      {
        int made_pos = text_find_word_ci(entry->text + made_scan,
          entry->text_len - made_scan, made_term);
        if (made_pos < 0) break;
        made_pos += made_scan;
        if (0 != answer_relation_fact_extract_made(entry, made_pos)) ++result;
        made_scan = made_pos + made_len;
      }
    }
    { /* The passive shape, wherever the declared agent marker stands. */
      int by_len = 0;
      const char* by = answer_relation_rule_first_term(RELATION_RULE_AGENT,
        &by_len);
      int by_scan = 0;
      while (NULL != by && 0 < by_len && by_scan < entry->text_len) {
        int by_pos = text_find_word_ci(entry->text + by_scan,
          entry->text_len - by_scan, by);
        if (by_pos < 0) break;
        by_pos += by_scan;
        if (0 != answer_relation_fact_extract_passive(entry, by_pos, by_len)) {
          ++result;
        }
        by_scan = by_pos + by_len;
      }
    }
    result += (size_t)answer_relation_fact_extract_active(entry);
    for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
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
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  answer_relation_facts_index();
  return result;
}


/**
 * Index the finished facts by the relation they state, and by every alias of it.
 *
 * Baking the alias closure in here is what makes the query side ONE lookup: a
 * question naming an aliased verb finds the facts stored under the canonical
 * one, without the scan having to consult the rule file per fact. The index only
 * ever proposes CANDIDATES - answer_relation_fact_relation_match still decides
 * - so the reply cannot depend on the index being exactly right, only on its
 * being complete.
 */
static void answer_relation_facts_index(void)
{
  size_t fact_pos;
  answer_fact_index_free(&answer_relation_by_relation);
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
    size_t rule_pos;
    if (fact->relation_len <= 0) continue;
    answer_fact_index_add(&answer_relation_by_relation,
      answer_word_key(fact->relation, fact->relation_len),
      (unsigned int)fact_pos);
    for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
      if (RELATION_RULE_ALIAS == rule->kind
        && 0 != libxs_striequal(rule->relation, strlen(rule->relation),
          fact->relation, (size_t)fact->relation_len))
      {
        answer_fact_index_add(&answer_relation_by_relation,
          answer_word_key(rule->term, (int)strlen(rule->term)),
          (unsigned int)fact_pos);
      }
    }
  }
}


/**
 * The count, and with CONVERSE_FACTS_LIST=1 the facts. Same reason the location
 * layer lists its own: a count cannot show that a fact is false, and this layer
 * ASSERTS. "Hans is to be made Very." was found this way and not by any metric.
 */
static void answer_relation_facts_report(FILE* stream)
{
  if (NULL != stream) {
    const char* list = getenv("CONVERSE_FACTS_LIST");
    fprintf(stream, "relation facts: %lu learned, %u indexed relations\n",
      (unsigned long)answer_relation_facts_size,
      answer_relation_by_relation.nkeys);
    if (NULL != list && '\0' != *list && '0' != *list) {
      size_t fact_pos;
      for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
        const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
        const char* shape = "was";
        if (0 != fact->made) shape = "made";
        else if (0 != fact->active) shape = "active";
        fprintf(stream, "  relation[%s] %s | %s | %s\n", shape, fact->answer,
          fact->relation, (0 < fact->actor_len) ? fact->actor : "-");
      }
    }
  }
}


static void answer_identity_facts_free(void)
{
  free(answer_identity_facts);
  answer_identity_facts = NULL;
  answer_identity_facts_size = 0;
}


/**
 * Is this token a name the corpus attests, not merely a capitalized word?
 *
 * An initial capital ALONE was the test, which is the same mistake as deciding a
 * query's meaning from its capitalization: the first word of a sentence and a
 * heading in capitals both pass it, so the extractor bound them to roles and the
 * fact table filled with pairs no reader would recognise. The corpus already
 * answers the question - a name is a word never attested in lower case, which
 * the case census counts at ingest - so the census decides and the capital is
 * only a cheap pre-filter for it.
 */
static int answer_identity_word_is_name(const char* word, int word_len)
{
  int result = 0;
  if (NULL != word && word_len > 1
    && 0 != isupper((unsigned char)word[0])
    && 0 != answer_word_is_name(word, word_len)
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
  memset(&fact, 0, sizeof(fact));
  memcpy(fact.name, name, (size_t)name_len);
  fact.name[name_len] = '\0';
  fact.name_len = name_len;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  /* The ORIGIN is not part of the section: a fact from an entry with no heading
   * still came from a file and a line, and keeping this inside the section guard
   * silently dropped every citation on a corpus that has no titles. */
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section[entry->section_len] = '\0';
    fact.section_len = entry->section_len;
  }
  fact.score = score;
  fact.provenance = answer_relation_rule_provenance(RELATION_RULE_PERSON,
    fact.role, fact.role_len);
  for (fact_pos = 0; fact_pos < answer_identity_facts_size; ++fact_pos) {
    answer_identity_fact_t* old_fact = answer_identity_facts + fact_pos;
    if (old_fact->name_len == fact.name_len
      && 0 != text_contains_word_ci(old_fact->name, old_fact->name_len,
        fact.name))
    {
      /**
       * PROVENANCE OUTRANKS SCORE. One name binds to one role here, and the
       * score is a similarity: with rule learning on, a term the learner added
       * to the person class can outscore an asserted one and silently rebind the
       * name to it, turning a correct reply into a confident false assertion.
       * Preferring the asserted role is a total order over three known values -
       * no statistic, no threshold, and nothing fitted to a corpus. Scores still
       * decide WITHIN a provenance, which is the comparison they can support.
       */
      if (fact.provenance < old_fact->provenance
        || (fact.provenance == old_fact->provenance
          && fact.score > old_fact->score))
      {
        memcpy(old_fact->role, fact.role, (size_t)fact.role_len + 1);
        old_fact->role_len = fact.role_len;
        memcpy(old_fact->section, fact.section, (size_t)fact.section_len + 1);
        old_fact->section_len = fact.section_len;
        old_fact->provenance = fact.provenance;
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
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
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
    /**
     * Sentence scale only, fragments excluded. A role->name binding is defined
     * within ONE sentence, so the paragraph copy and every comma fragment of a
     * sentence re-derive the same facts from the same bytes: pure duplicated
     * work, and on a large corpus it dominates (13.4 s of 54.9 s at 16 MB of
     * enwik8). The facts themselves are unchanged because a fragment cannot
     * contain a binding its parent sentence does not.
     */
    if (SCALE_SENTENCE != entry->scale
      || 0 != (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      value = corpus_iter_next(corpus, &key, &cursor);
      continue;
    }
    while (NULL != (token = libxs_strtoken(entry->text, delims,
      token_index, &token_len)))
    {
      const char* scan;
      for (scan = prev_end; scan < token; ++scan) {
        if ('.' == *scan || '!' == *scan || '?' == *scan
          || ';' == *scan || ':' == *scan) have_role = 0;
      }
      prev_end = token + token_len;
      /**
       * A role is a WORD, so the quotation marks around it are not part of it.
       * The delimiter list is ASCII and the corpus is typeset, so a token can
       * arrive as a quote character followed by the word - and the class test
       * matches on word containment, so it accepted the pair and then stored the
       * punctuation as the role.
       */
      while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
        ++token;
        --token_len;
      }
      while (token_len > 0
        && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
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
    value = corpus_iter_next(corpus, &key, &cursor);
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
  int query_section_len;
  const answer_identity_fact_t* best = NULL;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_identity_facts_size) return EXIT_FAILURE;
  name_len = answer_query_be_word(query_text, query_len, name,
    (int)sizeof(name));
  if (name_len <= 0 || 0 == answer_word_is_name(name, name_len)) {
    return EXIT_FAILURE;
  }
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
    answer_fact_section_set(best->section, best->section_len);
    answer_origin_add(best->source, best->line);
    answer_fact_learned_set(best->role, best->role_len,
      answer_relation_rule_provenance(RELATION_RULE_PERSON, best->role,
        best->role_len));
    result = answer_reply_role(output, output_size, best->name,
      best->name_len, best->role);
  }
  return result;
}


static void answer_location_facts_free(void)
{
  free(answer_location_facts);
  answer_location_facts = NULL;
  answer_location_facts_size = 0;
}


static int answer_location_fact_append(const char* actor, int actor_len,
  const char* phrase, int phrase_len, const char* place, int place_len,
  const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_location_fact_t fact;
  answer_location_fact_t* facts;
  size_t fact_pos;
  if (NULL == actor || actor_len <= 0 || actor_len >= (int)sizeof(fact.actor)
    || NULL == phrase || phrase_len <= 0
    || phrase_len >= (int)sizeof(fact.phrase)
    || NULL == place || place_len <= 0 || place_len >= (int)sizeof(fact.place))
  {
    return EXIT_FAILURE;
  }
  memset(&fact, 0, sizeof(fact));
  memcpy(fact.actor, actor, (size_t)actor_len);
  fact.actor_len = actor_len;
  memcpy(fact.phrase, phrase, (size_t)phrase_len);
  fact.phrase_len = phrase_len;
  memcpy(fact.place, place, (size_t)place_len);
  fact.place_len = place_len;
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section_len = entry->section_len;
  }
  fact.score = score;
  fact.provenance = answer_relation_rule_provenance(RELATION_RULE_PLACE,
    fact.place, fact.place_len);
  /**
   * An actor may be in several places - unlike a role, which binds once - so
   * only the SAME pair collapses, and then the tighter binding wins. Provenance
   * outranks the score for the same reason it does for an identity: a place term
   * the learner proposed must not displace one the rule file asserts.
   */
  for (fact_pos = 0; fact_pos < answer_location_facts_size; ++fact_pos) {
    answer_location_fact_t* old_fact = answer_location_facts + fact_pos;
    if (old_fact->actor_len == fact.actor_len
      && 0 != libxs_striequal(old_fact->actor, (size_t)old_fact->actor_len,
        fact.actor, (size_t)fact.actor_len)
      && old_fact->place_len == fact.place_len
      && 0 != libxs_striequal(old_fact->place, (size_t)old_fact->place_len,
        fact.place, (size_t)fact.place_len))
    {
      if (fact.provenance < old_fact->provenance
        || (fact.provenance == old_fact->provenance
          && fact.score > old_fact->score))
      {
        *old_fact = fact;
      }
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_location_fact_t*)realloc(answer_location_facts,
      (answer_location_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_location_facts = facts;
      answer_location_facts[answer_location_facts_size] = fact;
      ++answer_location_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Bind an actor to a place within ONE clause: a name, then a location MARKER
 * declared by the rule file (`where|in`), then a member of the place class
 * (`place|forest`). All three come from data - the name from the case census, the
 * marker and the class from the rules - so no English enters this file.
 *
 * The clause bound is what keeps it honest, and it must include the COMMA. A
 * sentence names several actors and several places, and any pair of them is a
 * plausible-looking fact; only the pair inside one clause, with the actor BEFORE
 * the marker, is one the sentence states. Measured on the tales: with sentence
 * punctuation alone as the boundary, 8 of 25 facts were clearly true and 8 clearly
 * false, and EVERY false one crossed a comma into a coordinated clause with a
 * different subject - "Roland went away, and the girl stood ... in the field"
 * became a claim about Roland. Coordination is exactly where the subject changes,
 * and the comma is punctuation rather than vocabulary, so the fix costs no English
 * in this file.
 *
 * The heading is skipped for the same reason: ingest stores a section's first
 * sentence from its heading onward, so the title word would otherwise be the
 * nearest preceding "name" and every section would state a fact about itself.
 */
static size_t answer_location_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  enum { LOCATION_GAP_MAX = 4 };
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_location_facts_free();
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    /* Only sentences carrying a location marker at all, which the ingest flag
     * already records: the rest cannot state a location and are most of them. */
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT)
      && 0 != (entry->lexical_flags & ENTRY_LEX_PLACE))
    {
      const int heading_len = corpus_title_len(entry->text, entry->text_len);
      const char* actor = NULL;
      const char* actor_end = NULL;
      const char* prev_end = entry->text + heading_len;
      const char* token;
      const char* marker = NULL;
      int actor_len = 0, gap = 0;
      int token_index = 0, token_len = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* scan;
        if (token < entry->text + heading_len) { ++token_index; continue; }
        for (scan = prev_end; scan < token; ++scan) {
          if ('.' == *scan || '!' == *scan || '?' == *scan
            || ';' == *scan || ':' == *scan || ',' == *scan)
          {
            actor = NULL;
            actor_len = 0;
            marker = NULL;
          }
        }
        prev_end = token + token_len;
        while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
          ++token;
          --token_len;
        }
        while (token_len > 0
          && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
        if (token_len > 0) {
          if (NULL != marker && 0 != answer_relation_rule_is_term(
            RELATION_RULE_PLACE, token, token_len))
          {
            /* The actor must precede the MARKER, not merely the place: "at Mr
             * Korbes's house" names an owner, and reading it as an actor stated
             * that Mr was at a house the sentence never puts him in. */
            if (NULL != actor && NULL != actor_end && actor_end < marker) {
              const char* begin = actor_end;
              int phrase_len;
              while (begin < token && 0 != isspace((unsigned char)*begin)) {
                ++begin;
              }
              phrase_len = (int)(token + token_len - begin);
              if (0 != answer_location_fact_append(actor, actor_len, begin,
                phrase_len, token, token_len, entry,
                (double)(ANSWER_LOCATION_PHRASE_MAX - phrase_len)))
              {
                ++result;
              }
            }
            marker = NULL;
          }
          else if (0 != answer_relation_rule_is_term(RELATION_RULE_WHERE,
            token, token_len))
          {
            marker = token;
            gap = 0;
          }
          else if (NULL != marker && ++gap >= LOCATION_GAP_MAX) {
            marker = NULL;
          }
          if (0 != answer_identity_word_is_name(token, token_len)) {
            /**
             * A name the corpus puts AFTER a verb is that verb's OBJECT, not the
             * subject of the clause: "to carry Snowdrop away into the wide wood"
             * stated that Snowdrop went there, and "told Sultan to come into the
             * wood" that Sultan did. The verb class is DERIVED from the corpus
             * (answer_verbs_build), and it is used here only to reject, because it
             * is incomplete: a past-simple verb never appears in the frame that
             * derives it, so what is missing costs a missed rejection rather than a
             * lost fact.
             */
            const char* prev = token;
            int prev_len = 0;
            while (prev > entry->text
              && 0 != isspace((unsigned char)prev[-1])) --prev;
            while (prev - prev_len > entry->text
              && 0 != isalpha((unsigned char)prev[-prev_len - 1])) ++prev_len;
            if (0 == answer_word_is_verb(prev - prev_len, prev_len)) {
              actor = token;
              actor_len = token_len;
              actor_end = token + token_len;
            }
          }
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
  return result;
}


/**
 * Report the count, and with CONVERSE_FACTS_LIST=1 the facts themselves. A count
 * says nothing about whether a fact is true, and this layer ASSERTS rather than
 * ranks, so the only honest way to judge it is to read what it extracted.
 */
static void answer_location_facts_report(FILE* stream)
{
  if (NULL != stream && 0 < answer_location_facts_size) {
    const char* list = getenv("CONVERSE_FACTS_LIST");
    fprintf(stream, "location facts: %lu learned\n",
      (unsigned long)answer_location_facts_size);
    if (NULL != list && '\0' != *list && '0' != *list) {
      size_t fact_pos;
      for (fact_pos = 0; fact_pos < answer_location_facts_size; ++fact_pos) {
        const answer_location_fact_t* fact = answer_location_facts + fact_pos;
        fprintf(stream, "  location[%s] %s %s\n",
          (RELATION_RULE_ASSERTED == fact->provenance) ? "asserted"
            : ((RELATION_RULE_LEARNED == fact->provenance) ? "learned"
              : "proposed"), fact->actor, fact->phrase);
      }
    }
  }
}


/** The name a location question asks about: the first one it names. */
static int answer_location_query_actor(const char* query_text,
  size_t query_len, char* actor, int actor_size)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  int result = 0;
  int token_index = 0, token_len = 0;
  const char* token;
  if (NULL == query_text || 0 == query_len || NULL == actor) return 0;
  while (0 == result && NULL != (token = libxs_strtoken(query_text, delims,
    token_index, &token_len)))
  {
    while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
      ++token;
      --token_len;
    }
    while (token_len > 0
      && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
    if (token_len > 0 && token_len < actor_size
      && 0 != answer_word_is_name(token, token_len))
    {
      memcpy(actor, token, (size_t)token_len);
      actor[token_len] = '\0';
      result = token_len;
    }
    ++token_index;
  }
  return result;
}


static int answer_location_fact_reply(const char* query_text,
  size_t query_len, char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char actor[64];
  char query_section[ENTRY_SECTION_MAX];
  int actor_len, query_section_len;
  const answer_location_fact_t* best = NULL;
  size_t fact_pos, pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_location_facts_size
    || QUERY_WHERE != answer_query_type_text(query_text, query_len))
  {
    return EXIT_FAILURE;
  }
  actor_len = answer_location_query_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  if (actor_len <= 0) return EXIT_FAILURE;
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  for (fact_pos = 0; fact_pos < answer_location_facts_size; ++fact_pos) {
    const answer_location_fact_t* fact = answer_location_facts + fact_pos;
    if (fact->actor_len == actor_len
      && 0 != libxs_striequal(fact->actor, (size_t)fact->actor_len,
        actor, (size_t)actor_len)
      && (query_section_len <= 0 || fact->section_len <= 0
        || 0 != text_contains_ci(fact->section, fact->section_len,
          query_section))
      && (NULL == best || fact->provenance < best->provenance
        || (fact->provenance == best->provenance
          && fact->score > best->score)))
    {
      best = fact;
    }
  }
  if (NULL != best
    && (size_t)(best->actor_len + best->phrase_len + 3) < output_size)
  {
    pos = answer_append_clean(output, output_size, 0, best->actor,
      best->actor_len);
    if (pos + 1 < output_size) output[pos++] = ' ';
    pos = answer_append_clean(output, output_size, pos, best->phrase,
      best->phrase_len);
    if (pos + 1 < output_size) {
      output[pos++] = '.';
      output[pos] = '\0';
      answer_fact_section_set(best->section, best->section_len);
      answer_origin_add(best->source, best->line);
      answer_fact_learned_set(best->place, best->place_len, best->provenance);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static void answer_type_facts_free(void)
{
  free(answer_type_facts);
  answer_type_facts = NULL;
  answer_type_facts_size = 0;
}


/**
 * The OF-GENITIVE possessor in a type phrase, so kinship stated that way is an edge
 * too: "Aegeus is the father of Theseus", "Python was a child of Gaia".
 *
 * The possessive appositive ("Lincoln's father Thomas") already names its possessor in
 * a field, and this is the SAME relation written the other way round - so the field is
 * the same and only the frame differs. Everything the frame rests on is declared: the
 * role is a `person|` term, so "the Royal Governor of Virginia" is not kinship, and the
 * marker is the declared `genitive|` word, so a corpus that marks a possessor
 * differently says so in its rule file rather than here.
 *
 * The possessor may carry ONE article and ONE modifier ("the son of the mortal
 * Peleus"), the same single hop the possessive shape allows, and the run stops at the
 * first word that is not a name - which takes the first conjunct of "of Priam and
 * Hecuba", and that is a true edge because coordination distributes.
 */
static int answer_type_partner_of(const char* phrase, int phrase_len,
  const char* name, int name_len, char* out, int out_size)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  int result = 0;
  int marker_len = 0;
  const char* marker = answer_relation_rule_first_term(RELATION_RULE_GENITIVE,
    &marker_len);
  const char* token;
  int token_index = 0, token_len = 0;
  if (NULL == phrase || 0 >= phrase_len || NULL == marker || 0 >= marker_len
    || 0 == isalpha((unsigned char)*marker))
  {
    return 0;
  }
  while (0 == result && NULL != (token = libxs_strtoken(phrase, delims,
    token_index, &token_len)))
  {
    if (0 != answer_relation_rule_is_term(RELATION_RULE_PERSON, token,
      token_len))
    {
      const char* scan = token + token_len;
      const char* end = phrase + phrase_len;
      int hop;
      while (scan < end && ' ' == *scan) ++scan;
      if (scan + marker_len <= end
        && 0 != libxs_striequal(scan, (size_t)marker_len, marker,
          (size_t)marker_len)
        && (scan + marker_len == end || ' ' == scan[marker_len]))
      {
        int article = 0, modifier = 0;
        scan += marker_len;
        for (hop = 0; hop < 3 && scan < end; ++hop) {
          const char* word;
          int word_len = 0;
          while (scan < end && ' ' == *scan) ++scan;
          word = scan;
          while (word + word_len < end
            && ('-' == word[word_len]
              || 0 != isalpha((unsigned char)word[word_len]))) ++word_len;
          if (0 == word_len) break;
          if (0 != answer_identity_word_is_name(word, word_len)) {
            const char* run_end = word + word_len;
            while (run_end < end && ' ' == *run_end) {
              const char* more = run_end + 1;
              int more_len = 0;
              while (more + more_len < end
                && ('-' == more[more_len]
                  || 0 != isalpha((unsigned char)more[more_len]))) ++more_len;
              if (0 == more_len
                || 0 == answer_identity_word_is_name(more, more_len)) break;
              run_end = more + more_len;
            }
            word_len = (int)(run_end - word);
            /* "the father of Aegeus" under the name Aegeus states nothing. */
            if (1 < word_len && word_len < out_size
              && 0 == libxs_striequal(word, (size_t)word_len, name,
                (size_t)name_len))
            {
              memcpy(out, word, (size_t)word_len);
              out[word_len] = '\0';
              result = word_len;
            }
            break;
          }
          /* At most one article and one modifier may stand before the name, which is
           * the single hop the possessive shape allows: "of the mortal Peleus". */
          if (0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE, word,
            word_len))
          {
            if (0 != article) break;
            article = 1;
          }
          else if (0 == modifier && 0 != islower((unsigned char)*word)) {
            modifier = 1;
          }
          else break;
          scan = word + word_len;
        }
      }
    }
    ++token_index;
  }
  return result;
}


static int answer_type_fact_append(const char* name, int name_len,
  const char* phrase, int phrase_len, int shape,
  const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_type_fact_t fact;
  answer_type_fact_t* facts;
  size_t fact_pos;
  if (NULL == name || name_len <= 0 || name_len >= (int)sizeof(fact.name)
    || NULL == phrase || phrase_len <= 0
    || phrase_len >= (int)sizeof(fact.phrase))
  {
    return EXIT_FAILURE;
  }
  memset(&fact, 0, sizeof(fact));
  memcpy(fact.name, name, (size_t)name_len);
  fact.name_len = name_len;
  memcpy(fact.phrase, phrase, (size_t)phrase_len);
  fact.phrase_len = phrase_len;
  fact.shape = shape;
  fact.score = score;
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
  /* Filled for EVERY shape rather than in one extractor, because the of-genitive
   * reaches this field through the copular shape as often as the appositive one. */
  fact.partner_len = answer_type_partner_of(phrase, phrase_len, name, name_len,
    fact.partner, (int)sizeof(fact.partner));
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section_len = entry->section_len;
  }
  /* One type per name, the TIGHTEST binding: a corpus states what something is
   * more than once, and the shortest statement of it is the definition rather
   * than a sentence that happens to contain one. */
  for (fact_pos = 0; fact_pos < answer_type_facts_size; ++fact_pos) {
    answer_type_fact_t* old_fact = answer_type_facts + fact_pos;
    if (old_fact->name_len == fact.name_len
      && 0 != libxs_striequal(old_fact->name, (size_t)old_fact->name_len,
        fact.name, (size_t)fact.name_len))
    {
      if (fact.score > old_fact->score) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_type_fact_t*)realloc(answer_type_facts,
      (answer_type_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_type_facts = facts;
      answer_type_facts[answer_type_facts_size] = fact;
      ++answer_type_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Extract what each named entity IS, from the copular and appositive shapes.
 *
 * COPULAR: the name must be IMMEDIATELY before the copula, with only whitespace
 * between them. That single bound is what keeps the subject right - "the daughter
 * of the king was a queen" names a queen, and reading the nearest earlier name as
 * the subject would have attributed it to the king.
 * APPOSITIVE: name, comma, gloss, comma - and the gloss must OPEN WITH AN ARTICLE.
 * Prose separates a great many things with commas, and requiring the determiner is
 * what distinguishes a noun phrase in apposition from the next clause. The articles
 * are their own declared class: using the skip class for this admitted "Hansel, and
 * thrust into his pockets ...", because skip declares every function word.
 * Both stop at clause punctuation, and the complement must fit the field: a type
 * that runs the length of a sentence is not a type.
 */
/**
 * The POSSESSIVE KINSHIP appositive: a possessed role word standing directly before a
 * name relates the two - "Lincoln's father Thomas", "Alexander's mother Olympias".
 *
 * Unlike the other shapes here this one is triggered by the ROLE, which is why both
 * sides can be read by pointer and no state has to be carried across tokens: the role
 * class is DECLARED (`person|`), so a match is evidence before either flank is seen.
 * Declared is what makes the shape trustworthy - the same frame with a derived noun
 * class admits "the 4th century Amsterdam" and asserts nonsense.
 *
 * The possessor is the second argument, so this is a relation and not a type, and it
 * is stated with the corpus's own possessive: the phrase runs verbatim from the
 * possessive name through the role, and only the copula is supplied. A pronoun
 * possessor ("his mother Thetis") is skipped, because resolving it is coreference and
 * the fact would rest on a guess.
 */
static int answer_type_kin_append(const corpus_entry_t* entry, int heading_len,
  const char* role, int role_len)
{
  int result = 0;
  const char* limit = entry->text + heading_len;
  const char* text_end = entry->text + entry->text_len;
  const char* owner_end;
  const char* owner;
  int owner_len, pure_len = 0, mark = 0;
  int hop;
  if (role <= limit + 1 || ' ' != role[-1]
    || 0 != isupper((unsigned char)*role)) return 0;
  owner_end = role - 1;
  /* At most one modifier between the two - "Lincoln's oldest son Robert". */
  for (hop = 0; hop < 2; ++hop) {
    owner = owner_end;
    while (owner > limit && 0 == isspace((unsigned char)owner[-1])
      && ',' != owner[-1] && '.' != owner[-1]) --owner;
    owner_len = (int)(owner_end - owner);
    if (0 == owner_len) return 0;
    pure_len = answer_name_strip(owner, owner_len, &mark);
    if (0 != mark && 0 < pure_len
      && 0 != answer_identity_word_is_name(owner, pure_len)) break;
    mark = 0;
    if (0 != hop || owner <= limit + 1 || ' ' != owner[-1]
      || 0 != answer_word_is_function(owner, owner_len)) return 0;
    owner_end = owner - 1;
  }
  if (0 != mark) {
    const char* name = role + role_len;
    const char* name_end;
    int name_len;
    while (name < text_end && ' ' == *name) ++name;
    name_end = name;
    while (name_end < text_end
      && ('-' == *name_end || 0 != isalpha((unsigned char)*name_end)))
    {
      ++name_end;
    }
    /* The name is a maximal RUN, so "Thomas Lincoln" is not just "Thomas". */
    while (name_end < text_end && ' ' == *name_end) {
      const char* more = name_end + 1;
      int more_len = 0;
      while (more + more_len < text_end
        && ('-' == more[more_len]
          || 0 != isalpha((unsigned char)more[more_len]))) ++more_len;
      if (0 == more_len
        || 0 == answer_identity_word_is_name(more, more_len)) break;
      name_end = more + more_len;
    }
    name_len = (int)(name_end - name);
    if (1 < name_len && name_len < 64
      && 0 != answer_identity_word_is_name(name, name_len)
      /* Not the possessor itself: "Lincoln's father Lincoln" states nothing. */
      && 0 == libxs_striequal(name, (size_t)name_len, owner, (size_t)pure_len))
    {
      int copula_len = 0;
      const char* copula = answer_relation_rule_first_term(
        RELATION_RULE_COPULA, &copula_len);
      const int span_len = (int)(role + role_len - owner);
      if (NULL != copula && 0 < copula_len
        && name_len + copula_len + span_len + 3 < ANSWER_TYPE_PHRASE_MAX)
      {
        char phrase[ANSWER_TYPE_PHRASE_MAX];
        size_t at = 0;
        memcpy(phrase + at, name, (size_t)name_len);
        at += (size_t)name_len;
        phrase[at++] = ' ';
        memcpy(phrase + at, copula, (size_t)copula_len);
        at += (size_t)copula_len;
        phrase[at++] = ' ';
        memcpy(phrase + at, owner, (size_t)span_len);
        at += (size_t)span_len;
        phrase[at] = '\0';
        if (EXIT_SUCCESS == answer_type_fact_append(name, name_len, phrase,
          (int)at, ANSWER_TYPE_KIN, entry,
          (double)(ANSWER_TYPE_PHRASE_MAX - (int)at)))
        {
          /* The possessor is the entity at the other end, and naming it in its own
           * field is what lets the graph traverse this fact instead of parsing the
           * sentence it renders as. */
          answer_type_fact_t* stored = answer_type_facts
            + (answer_type_facts_size - 1);
          if (0 < pure_len && pure_len < (int)sizeof(stored->partner)) {
            memcpy(stored->partner, owner, (size_t)pure_len);
            stored->partner[pure_len] = '\0';
            stored->partner_len = pure_len;
          }
          result = 1;
        }
      }
    }
  }
  return result;
}


static size_t answer_type_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_type_facts_free();
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const int heading_len = corpus_title_len(entry->text, entry->text_len);
      const char* text_end = entry->text + entry->text_len;
      const char* name = NULL;
      const char* name_end = NULL;
      const char* token;
      int name_len = 0, token_index = 0, token_len = 0;
      /* Whether the name last seen is an item of a list or sits inside a
       * prepositional phrase, i.e. is NOT the subject of its clause. Recorded with
       * the name because both shapes ask the same question of it. */
      int name_in_phrase = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* raw = token;
        if (token < entry->text + heading_len) { ++token_index; continue; }
        while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
          ++token;
          --token_len;
        }
        while (token_len > 0
          && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
        if (token_len > 0) {
          if (NULL != name && NULL != name_end
            && 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA, token,
              token_len))
          {
            const char* gap;
            int adjacent = 1;
            for (gap = name_end; gap < raw && 0 != adjacent; ++gap) {
              if (0 == isspace((unsigned char)*gap)) adjacent = 0;
            }
            if (0 != adjacent && 0 == name_in_phrase) {
              const char* comp = token + token_len;
              const char* end;
              int art_len = 0;
              while (comp < text_end
                && 0 != isspace((unsigned char)*comp)) ++comp;
              while (comp + art_len < text_end
                && 0 != isalpha((unsigned char)comp[art_len])) ++art_len;
              end = comp + art_len;
              while (end < text_end && ',' != *end && '.' != *end
                && ';' != *end && ':' != *end && '!' != *end && '?' != *end)
              {
                ++end;
              }
              while (end > comp && 0 != isspace((unsigned char)end[-1])) --end;
              /**
               * The complement must open with an ARTICLE, which is what makes it a
               * noun phrase and so a TYPE. Without that test the shape captured
               * every predicate the copula introduces - "Snowdrop was dead", "Tom
               * was calling out" - true spans, but not statements of what
               * something IS, and answering "What is X?" with one is wrong.
               */
              if (art_len > 0 && end > comp + art_len
                && 0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
                  comp, art_len)
                && EXIT_SUCCESS == answer_type_fact_append(name, name_len,
                  name, (int)(end - name), ANSWER_TYPE_COPULAR, entry,
                  (double)(ANSWER_TYPE_PHRASE_MAX - (int)(end - name))))
              {
                ++result;
              }
            }
          }
          /* The kinship shape triggers on the DECLARED role, not on a name, so it
           * reads both of its flanks itself. */
          if (0 != answer_relation_rule_has_term(RELATION_RULE_PERSON, token,
            token_len))
          {
            result += (size_t)answer_type_kin_append(entry, heading_len, token,
              token_len);
          }
          if (0 != answer_identity_word_is_name(token, token_len)) {
            const char* comma = token + token_len;
            const char* before = raw;
            /**
             * A NAME IS A MAXIMAL RUN of name tokens, not its last one. Taking one
             * token made "Atlas Shrugged" into "Shrugged" and "Thomas Lincoln" into
             * "Thomas", so the fact was keyed and rendered under half a name - and
             * the subject test then examined the wrong left edge, since what
             * precedes the RUN is what says whether the run is the subject.
             */
            const int extend = (NULL != name && name_end + 1 == token
              && ' ' == *name_end) ? 1 : 0;
            if (0 == extend) {
              name = token;
              name_in_phrase = 0;
            }
            name_len = (int)(token + token_len - name);
            name_end = token + token_len;
            /**
             * A name that a COMMA already introduces is an item of a list, not the
             * subject of an apposition. This is the single largest error class the
             * shape had on Wikipedia: "Brunei, the Philippines, and Vietnam" and
             * "China, the United States" read exactly like "Priam, the king of
             * Troy" unless the left side is checked, and every one of them became a
             * confident false type.
             */
            while (before > entry->text
              && 0 != isspace((unsigned char)before[-1])) --before;
            if (0 == extend && before > entry->text && ',' == before[-1]) {
              name_in_phrase = 1;
            }
            /**
             * And a name inside a PREPOSITIONAL PHRASE is not the clause subject,
             * which was the other half of the errors: "a bust of Aristotle is a
             * nearly ubiquitous ornament" made Aristotle the ornament, and "the
             * railway in Angola is" made Angola the railway. The preposition class
             * is declared, so this reads the rule file rather than English.
             */
            if (0 == extend && 0 == name_in_phrase && before > entry->text) {
              const char* word_end = before;
              const char* word = before;
              while (word > entry->text
                && 0 != isalpha((unsigned char)word[-1])) --word;
              if (word < word_end && 0 != answer_relation_rule_is_term(
                RELATION_RULE_PREP, word, (int)(word_end - word)))
              {
                name_in_phrase = 1;
              }
            }
            /* Appositive: the gloss is what stands between the two commas. */
            while (comma < text_end
              && 0 != isspace((unsigned char)*comma)) ++comma;
            if (comma < text_end && ',' == *comma) {
              const char* gloss = comma + 1;
              const char* end;
              const char* after;
              int article_len = 0;
              int after_len = 0;
              while (gloss < text_end
                && 0 != isspace((unsigned char)*gloss)) ++gloss;
              while (gloss + article_len < text_end
                && 0 != isalpha((unsigned char)gloss[article_len]))
              {
                ++article_len;
              }
              end = gloss + article_len;
              while (end < text_end && ',' != *end && '.' != *end
                && ';' != *end && ':' != *end && '!' != *end
                && '?' != *end) ++end;
              /**
               * And what FOLLOWS the closing comma settles the rest: a list
               * continues with another article-led item, while an apposition
               * returns to the sentence's predicate. Two lookaheads, no scoring.
               */
              after = end + 1;
              while (after < text_end
                && 0 != isspace((unsigned char)*after)) ++after;
              while (after + after_len < text_end
                && 0 != isalpha((unsigned char)after[after_len])) ++after_len;
              if (0 == name_in_phrase && article_len > 0 && end < text_end
                && ',' == *end
                && 0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
                  gloss, article_len)
                && (0 == after_len
                  || 0 == answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
                    after, after_len))
                && count_words((const unsigned char*)gloss,
                  (int)(end - gloss)) >= 2)
              {
                char phrase[ANSWER_TYPE_PHRASE_MAX];
                int copula_len = 0;
                const char* copula = answer_relation_rule_first_term(
                  RELATION_RULE_COPULA, &copula_len);
                if (NULL != copula && 0 < copula_len
                  && (size_t)(name_len + copula_len + (end - gloss) + 3)
                    < sizeof(phrase))
                {
                  size_t at = 0;
                  memcpy(phrase + at, name, (size_t)name_len);
                  at += (size_t)name_len;
                  phrase[at++] = ' ';
                  memcpy(phrase + at, copula, (size_t)copula_len);
                  at += (size_t)copula_len;
                  phrase[at++] = ' ';
                  memcpy(phrase + at, gloss, (size_t)(end - gloss));
                  at += (size_t)(end - gloss);
                  phrase[at] = '\0';
                  if (EXIT_SUCCESS == answer_type_fact_append(name, name_len,
                    phrase, (int)at, ANSWER_TYPE_APPOSITIVE, entry,
                    (double)(ANSWER_TYPE_PHRASE_MAX - (int)at)))
                  {
                    ++result;
                  }
                }
              }
            }
          }
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_type_facts_report(FILE* stream)
{
  if (NULL != stream && 0 < answer_type_facts_size) {
    const char* list = getenv("CONVERSE_FACTS_LIST");
    fprintf(stream, "type facts: %lu learned\n",
      (unsigned long)answer_type_facts_size);
    if (NULL != list && '\0' != *list && '0' != *list) {
      size_t fact_pos;
      for (fact_pos = 0; fact_pos < answer_type_facts_size; ++fact_pos) {
        const answer_type_fact_t* fact = answer_type_facts + fact_pos;
        const char* shape = "appositive";
        if (ANSWER_TYPE_COPULAR == fact->shape) shape = "copular";
        else if (ANSWER_TYPE_KIN == fact->shape) shape = "kin";
        fprintf(stream, "  type[%s] %s  [%s:%u]\n", shape, fact->phrase,
          (NULL != corpus_source_path(fact->source))
            ? corpus_source_path(fact->source) : "?", fact->line);
      }
    }
  }
}


static int answer_type_render(const answer_type_fact_t* fact, char* output,
  size_t output_size)
{
  int result = EXIT_FAILURE;
  if (NULL != fact && NULL != output && 0 < output_size
    && 0 < fact->phrase_len)
  {
    size_t at = answer_append_clean(output, output_size, 0, fact->phrase,
      fact->phrase_len);
    if (at + 2 < output_size) {
      output[at++] = '.';
      output[at] = '\0';
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Answer from the type layer, optionally from ONE shape only.
 *
 * The kinship shape is asked before the identity layer and the others after it, so
 * the filter is what keeps that from needing a second copy of this function. Asking
 * kinship first is not a preference between layers: an identity fact and a kinship
 * fact about the same name are the SAME evidence, and the kinship one keeps the
 * possessor ("Olympias is Alexander's mother" against "Olympias is the mother").
 */
static int answer_type_reply_shape(const char* query_text, size_t query_len,
  char* output, size_t output_size, int shape_only)
{
  int result = EXIT_FAILURE;
  char name[64];
  const answer_type_fact_t* best = NULL;
  int name_len, query_type;
  size_t fact_pos;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_type_facts_size) return EXIT_FAILURE;
  query_type = answer_query_type_text(query_text, query_len);
  if (QUERY_WHO != query_type && QUERY_WHAT != query_type) {
    return EXIT_FAILURE;
  }
  name_len = answer_query_be_word(query_text, query_len, name,
    (int)sizeof(name));
  if (name_len <= 0 || 0 == answer_word_is_name(name, name_len)) {
    return EXIT_FAILURE;
  }
  for (fact_pos = 0; fact_pos < answer_type_facts_size; ++fact_pos) {
    const answer_type_fact_t* fact = answer_type_facts + fact_pos;
    /* Word containment, not equality: the fact holds the full run ("Thomas
     * Lincoln") and a question names one word of it. */
    if ((shape_only < 0 || shape_only == fact->shape)
      && 0 != text_contains_word_ci(fact->name, fact->name_len, name)
      && (NULL == best || fact->score > best->score))
    {
      best = fact;
    }
  }
  if (NULL != best
    && EXIT_SUCCESS == answer_type_render(best, output, output_size))
  {
    answer_fact_section_set(best->section, best->section_len);
    answer_origin_add(best->source, best->line);
    result = EXIT_SUCCESS;
  }
  return result;
}


static int answer_type_fact_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size)
{
  return answer_type_reply_shape(query_text, query_len, output, output_size, -1);
}


static int answer_type_kin_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size)
{
  return answer_type_reply_shape(query_text, query_len, output, output_size,
    ANSWER_TYPE_KIN);
}


/**
 * The KNOWLEDGE GRAPH: what the fact layers say about how two entities relate.
 *
 * No index and no new storage. The edges are the facts that already exist - a
 * relation fact whose actor and answer are both entities, and a kinship fact, which
 * names its partner in a field for exactly this reason - so an edge VIEW over them
 * is all a walk needs. An index was priced and declined (C2): at these fact counts a
 * scan per query is cheaper than a structure that has to be kept true.
 */
typedef struct answer_edge_t {
  const char* from;
  int from_len;
  const char* to;
  int to_len;
  /* Which layer states it, so the reply can be rendered by that layer rather than
   * reassembled here from the pieces. */
  const answer_relation_fact_t* relation;
  const answer_type_fact_t* type;
} answer_edge_t;


/** Whether a fact field names an ENTITY rather than a phrase about one. */
static int answer_edge_is_entity(const char* text, int text_len)
{
  int result = 0;
  if (NULL != text && 1 < text_len) {
    int at = 0, word_len = 0, clean = 1;
    while (at < text_len && 0 != clean) {
      word_len = 0;
      while (at + word_len < text_len
        && ('-' == text[at + word_len]
          || 0 != isalpha((unsigned char)text[at + word_len]))) ++word_len;
      if (0 == word_len) clean = 0;
      /* The FIRST word carries the census evidence, and the rest only has to be
       * part of the same run: "Ross Perot" is one entity, "the wolf" is not, and
       * testing the whole string as one word rejected every two-word name. */
      else if (0 == at
        && 0 == answer_identity_word_is_name(text, word_len)) clean = 0;
      else {
        at += word_len;
        if (at < text_len) {
          if (' ' == text[at]) ++at;
          else clean = 0;
        }
      }
    }
    result = clean;
  }
  return result;
}


/** Fill edge from a fact if it states one, and say whether it does. */
static int answer_edge_of_relation(const answer_relation_fact_t* fact,
  answer_edge_t* edge)
{
  int result = 0;
  if (NULL != fact && NULL != edge && 0 == fact->made
    && 0 != answer_edge_is_entity(fact->actor, fact->actor_len)
    && 0 != answer_edge_is_entity(fact->answer, fact->answer_len))
  {
    memset(edge, 0, sizeof(*edge));
    edge->from = fact->actor;
    edge->from_len = fact->actor_len;
    edge->to = fact->answer;
    edge->to_len = fact->answer_len;
    edge->relation = fact;
    result = 1;
  }
  return result;
}


static int answer_edge_of_type(const answer_type_fact_t* fact,
  answer_edge_t* edge)
{
  int result = 0;
  if (NULL != fact && NULL != edge && 0 < fact->partner_len
    && 0 != answer_edge_is_entity(fact->name, fact->name_len)
    && 0 != answer_edge_is_entity(fact->partner, fact->partner_len))
  {
    memset(edge, 0, sizeof(*edge));
    edge->from = fact->name;
    edge->from_len = fact->name_len;
    edge->to = fact->partner;
    edge->to_len = fact->partner_len;
    edge->type = fact;
    result = 1;
  }
  return result;
}


/** The next edge touching this entity, from a cursor over both fact layers. */
static int answer_edge_next(const char* name, int name_len, size_t* cursor,
  answer_edge_t* edge)
{
  int result = 0;
  if (NULL != name && 0 < name_len && NULL != cursor && NULL != edge) {
    while (0 == result && *cursor < answer_relation_facts_size) {
      const answer_relation_fact_t* fact = answer_relation_facts + *cursor;
      ++*cursor;
      if (0 != answer_edge_of_relation(fact, edge)
        && (0 != text_contains_word_ci(edge->from, edge->from_len, name)
          || 0 != text_contains_word_ci(edge->to, edge->to_len, name)))
      {
        /* Oriented so the entity asked about is always the near end: an edge reads
         * from either side, and the caller should not have to test which. */
        if (0 == text_contains_word_ci(edge->from, edge->from_len, name)) {
          const char* text = edge->from;
          const int len = edge->from_len;
          edge->from = edge->to;
          edge->from_len = edge->to_len;
          edge->to = text;
          edge->to_len = len;
        }
        result = 1;
      }
    }
    while (0 == result
      && *cursor - answer_relation_facts_size < answer_type_facts_size)
    {
      const answer_type_fact_t* fact = answer_type_facts
        + (*cursor - answer_relation_facts_size);
      ++*cursor;
      if (0 != answer_edge_of_type(fact, edge)
        && (0 != text_contains_word_ci(edge->from, edge->from_len, name)
          || 0 != text_contains_word_ci(edge->to, edge->to_len, name)))
      {
        if (0 == text_contains_word_ci(edge->from, edge->from_len, name)) {
          const char* text = edge->from;
          const int len = edge->from_len;
          edge->from = edge->to;
          edge->from_len = edge->to_len;
          edge->to = text;
          edge->to_len = len;
        }
        result = 1;
      }
    }
  }
  return result;
}


/**
 * GRAPH REACH: the edges, the entities they touch, the pairs one middle joins, and the
 * largest degree - reported so the connectivity table can be RE-RUN instead of
 * quoted from memory. It existed only as a hand-count, which is why it went stale
 * three times while the fact counts beneath it moved (E24, E16 STEP 3, E26).
 *
 * The pair count is EXACTLY what a two-hop reply can answer: two entities with no edge
 * of their own, joined by a common middle, counted once per unordered pair. That is a
 * bound on REACH and never on correctness - a path's precision is the PRODUCT of its
 * hops, so only reading a sample can report it, which is what the wiki track does.
 */
static void answer_graph_report(FILE* stream)
{
  enum { GRAPH_ENTITY_MAX = 4096, GRAPH_NEIGHBOUR_MAX = 64 };
  const char* list = (NULL != stream) ? getenv("CONVERSE_FACTS_LIST") : NULL;
  if (NULL != list && '\0' != *list && '0' != *list) {
    libxs_registry_t* seen = libxs_registry_create();
    libxs_registry_t* joined = libxs_registry_create();
    char (*names)[128] = (char(*)[128])malloc(GRAPH_ENTITY_MAX * sizeof(*names));
    size_t edges = 0, entities = 0, pairs = 0, dropped = 0;
    int degree_max = 0;
    size_t fact_pos, entity_pos;
    for (fact_pos = 0;
      fact_pos < answer_relation_facts_size + answer_type_facts_size; ++fact_pos)
    {
      answer_edge_t edge;
      const int ok = (fact_pos < answer_relation_facts_size)
        ? answer_edge_of_relation(answer_relation_facts + fact_pos, &edge)
        : answer_edge_of_type(answer_type_facts
          + (fact_pos - answer_relation_facts_size), &edge);
      if (0 != ok) {
        int side;
        ++edges;
        for (side = 0; side < 2; ++side) {
          const char* text = (0 == side) ? edge.from : edge.to;
          const int len = (0 == side) ? edge.from_len : edge.to_len;
          char key[128];
          int at;
          if (0 < len && len < (int)sizeof(key) && NULL != seen) {
            for (at = 0; at < len; ++at) {
              key[at] = (char)tolower((unsigned char)text[at]);
            }
            key[len] = '\0';
            if (NULL == libxs_registry_get(seen, key, (size_t)len + 1, NULL)) {
              const long fresh = 1;
              if (NULL != libxs_registry_set(seen, key, (size_t)len + 1, &fresh,
                sizeof(fresh), NULL))
              {
                if (NULL != names && entities < GRAPH_ENTITY_MAX) {
                  memcpy(names[entities], text, (size_t)len);
                  names[entities][len] = '\0';
                }
                else ++dropped;
                ++entities;
              }
            }
          }
        }
      }
    }
    for (entity_pos = 0; NULL != names && NULL != joined
      && entity_pos < entities && entity_pos < GRAPH_ENTITY_MAX; ++entity_pos)
    {
      const char* middle = names[entity_pos];
      const int middle_len = (int)strlen(middle);
      char neighbour[GRAPH_NEIGHBOUR_MAX][128];
      size_t cursor = 0;
      answer_edge_t walk;
      int degree = 0, a, b;
      while (0 != answer_edge_next(middle, middle_len, &cursor, &walk)) {
        if (degree < GRAPH_NEIGHBOUR_MAX && 0 < walk.to_len
          && walk.to_len < (int)sizeof(neighbour[0]))
        {
          memcpy(neighbour[degree], walk.to, (size_t)walk.to_len);
          neighbour[degree][walk.to_len] = '\0';
          ++degree;
        }
      }
      if (degree > degree_max) degree_max = degree;
      for (a = 0; a < degree; ++a) {
        for (b = a + 1; b < degree; ++b) {
          const int lo = (0 < strcmp(neighbour[a], neighbour[b])) ? b : a;
          const int hi = (lo == a) ? b : a;
          char key[264];
          size_t key_len;
          size_t probe = 0;
          answer_edge_t direct;
          int adjacent = 0;
          if (0 == strcmp(neighbour[a], neighbour[b])) continue;
          /* A pair the corpus relates DIRECTLY is one proposition, not a path. */
          while (0 == adjacent && 0 != answer_edge_next(neighbour[lo],
            (int)strlen(neighbour[lo]), &probe, &direct))
          {
            if (0 != libxs_striequal(direct.to, (size_t)direct.to_len,
              neighbour[hi], strlen(neighbour[hi])))
            {
              adjacent = 1;
            }
          }
          if (0 != adjacent) continue;
          key_len = (size_t)sprintf(key, "%s|%s", neighbour[lo], neighbour[hi]);
          if (NULL == libxs_registry_get(joined, key, key_len + 1, NULL)) {
            const long fresh = 1;
            if (NULL != libxs_registry_set(joined, key, key_len + 1, &fresh,
              sizeof(fresh), NULL))
            {
              ++pairs;
            }
          }
        }
      }
    }
    fprintf(stream, "graph reach: %lu edges, %lu entities, %lu pairs joined by one"
      " middle, max degree %d\n", (unsigned long)edges, (unsigned long)entities,
      (unsigned long)pairs, degree_max);
    /* No silent cap: a bounded walk that dropped anything says so. */
    if (0 != dropped) {
      fprintf(stream, "graph reach: %lu entities beyond the %d scanned\n",
        (unsigned long)dropped, (int)GRAPH_ENTITY_MAX);
    }
    free(names);
    if (NULL != joined) libxs_registry_destroy(joined);
    if (NULL != seen) libxs_registry_destroy(seen);
  }
}


/** State one edge as the layer that holds it would state it, and cite it. */
static int answer_edge_render(const answer_edge_t* edge, char* output,
  size_t output_size)
{
  int result = EXIT_FAILURE;
  if (NULL != edge && NULL != output && 0 < output_size) {
    if (NULL != edge->relation) {
      answer_relation_match_t match;
      const answer_relation_fact_t* fact = edge->relation;
      memset(&match, 0, sizeof(match));
      memcpy(match.answer, fact->answer, (size_t)fact->answer_len + 1);
      match.answer_len = fact->answer_len;
      memcpy(match.relation, fact->relation, (size_t)fact->relation_len + 1);
      match.relation_len = fact->relation_len;
      memcpy(match.actor, fact->actor, (size_t)fact->actor_len + 1);
      match.actor_len = fact->actor_len;
      match.plural = fact->plural;
      match.active = fact->active;
      result = answer_relation_reply(&match, output, output_size);
      if (EXIT_SUCCESS == result) {
        answer_fact_section_add(fact->section, fact->section_len);
        answer_origin_add(fact->source, fact->line);
      }
    }
    else if (NULL != edge->type) {
      result = answer_type_render(edge->type, output, output_size);
      if (EXIT_SUCCESS == result) {
        answer_fact_section_add(edge->type->section, edge->type->section_len);
        answer_origin_add(edge->type->source, edge->type->line);
      }
    }
  }
  return result;
}


/**
 * The two entities a graph question asks about.
 *
 * A declared `link|` term is what says the question is about the graph at all, and
 * the two entities are the census names it mentions - exactly two, or the question
 * is not about a pair and this resolver has nothing to say.
 */
static int answer_link_query(const char* query_text, size_t query_len,
  char* first, int first_size, char* second, int second_size)
{
  int result = 0;
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"\'";
  char buffer[512];
  size_t copy = query_len;
  int found = 0, linked = 0;
  int token_index = 0, token_len = 0;
  const char* token;
  if (NULL == query_text || NULL == first || NULL == second) return 0;
  if (copy >= sizeof(buffer)) copy = sizeof(buffer) - 1;
  memcpy(buffer, query_text, copy);
  buffer[copy] = '\0';
  first[0] = '\0';
  second[0] = '\0';
  while (NULL != (token = libxs_strtoken(buffer, delims, token_index,
    &token_len)))
  {
    if (0 != answer_relation_rule_is_term(RELATION_RULE_LINK, token,
      token_len))
    {
      linked = 1;
    }
    else if (0 != answer_identity_word_is_name(token, token_len)) {
      char* slot = (0 == found) ? first : second;
      const int size = (0 == found) ? first_size : second_size;
      /* A name is a RUN, so "Ross Perot" is one entity and not two: counting words
       * made every two-word name look like a pair and the question unanswerable. */
      int run_len = token_len;
      const char* next;
      int next_len = 0;
      while (NULL != (next = libxs_strtoken(buffer, delims, token_index + 1,
        &next_len)))
      {
        if (next != token + run_len + 1 || ' ' != token[run_len]
          || 0 == answer_identity_word_is_name(next, next_len)) break;
        run_len = (int)(next + next_len - token);
        ++token_index;
      }
      if (run_len < size && 2 > found) {
        memcpy(slot, token, (size_t)run_len);
        slot[run_len] = '\0';
        ++found;
      }
      else ++found;
    }
    ++token_index;
  }
  if (0 != linked && 2 == found) result = 1;
  return result;
}


/**
 * Answer a graph question by STATING THE PATH, never by composing it.
 *
 * A direct edge is one proposition; two edges through a middle entity are two, and
 * both are stated with their own citations. What this deliberately does NOT do is
 * assert a relation between the endpoints: no sentence in the corpus says one, so
 * composing it would be inference, and inference here has always had to be judged.
 * The reader is given the two attested facts and draws the connection.
 */
static int answer_link_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char first[64], second[64];
  if (NULL == output || 0 == output_size
    || 0 == answer_link_query(query_text, query_len, first, (int)sizeof(first),
      second, (int)sizeof(second)))
  {
    return EXIT_FAILURE;
  }
  { answer_edge_t edge;
    size_t cursor = 0;
    char text[COMPOSE_MAXTEXT];
    answer_fact_section_set(NULL, 0);
    output[0] = '\0';
    /* A DIRECT edge first: two facts about a pair the corpus states outright would
     * be a longer answer than the one it states. */
    while (EXIT_FAILURE == result
      && 0 != answer_edge_next(first, (int)strlen(first), &cursor, &edge))
    {
      if (0 != text_contains_word_ci(edge.to, edge.to_len, second)
        && EXIT_SUCCESS == answer_edge_render(&edge, text, sizeof(text)))
      {
        size_t len = strlen(text);
        if (len < output_size) {
          memcpy(output, text, len + 1);
          result = EXIT_SUCCESS;
        }
      }
    }
    cursor = 0;
    while (EXIT_FAILURE == result
      && 0 != answer_edge_next(first, (int)strlen(first), &cursor, &edge))
    {
      char middle[64];
      answer_edge_t onward;
      size_t inner = 0;
      int middle_len = edge.to_len;
      if (middle_len >= (int)sizeof(middle)) continue;
      memcpy(middle, edge.to, (size_t)middle_len);
      middle[middle_len] = '\0';
      if (0 != text_contains_word_ci(middle, middle_len, first)) continue;
      while (EXIT_FAILURE == result
        && 0 != answer_edge_next(middle, middle_len, &inner, &onward))
      {
        if (0 == text_contains_word_ci(onward.to, onward.to_len, second)
          || 0 != text_contains_word_ci(onward.to, onward.to_len, first))
        {
          continue;
        }
        if (EXIT_SUCCESS == answer_edge_render(&edge, text, sizeof(text))) {
          size_t pos = strlen(text);
          if (pos + 2 < output_size) {
            memcpy(output, text, pos);
            output[pos++] = ' ';
            output[pos] = '\0';
            if (EXIT_SUCCESS == answer_edge_render(&onward, text,
              sizeof(text)))
            {
              const size_t len = strlen(text);
              if (pos + len < output_size) {
                memcpy(output + pos, text, len + 1);
                result = EXIT_SUCCESS;
              }
            }
          }
        }
      }
    }
  }
  return result;
}


static void answer_verbs_free(void)
{
  if (NULL != answer_verbs) {
    libxs_registry_destroy(answer_verbs);
    answer_verbs = NULL;
  }
  answer_verbs_nkeys = 0;
}


static void answer_nouns_free(void)
{
  if (NULL != answer_nouns) {
    libxs_registry_destroy(answer_nouns);
    answer_nouns = NULL;
  }
  answer_nouns_nkeys = 0;
}


/**
 * Derive the words this corpus uses as VERBS from the one frame that says so
 * without morphology: an AUXILIARY governs a verb.
 *
 * The frame is `aux`, then any number of declared function words, then the word
 * itself - "would not GO", "had never SEEN", "will soon COME" - so the adverbs and
 * negators between the two are stepped over rather than mistaken for the verb. Both
 * the auxiliaries and the function words are declared, so no English is written
 * here, and the class is a fact about the CORPUS rather than about English.
 *
 * IT IS INCOMPLETE BY CONSTRUCTION, and that decides how it may be used. English
 * narrative is written in the past simple, which no auxiliary governs: "went",
 * "stood" and "looked" never appear in this frame and so are never derived. A
 * requirement built on it would therefore reject true facts wholesale - measured on
 * the tales, "Hans went into the stable" would have been the first casualty. It is
 * used only to REJECT: a name the corpus puts AFTER a verb is that verb's object,
 * and a missing verb then costs a missed rejection rather than a lost truth.
 */
static size_t answer_verbs_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_verbs_free();
  answer_verbs = libxs_registry_create();
  if (NULL == corpus || NULL == answer_verbs) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const char* token;
      int token_index = 0, token_len = 0, governed = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* word = token;
        int word_len = token_len;
        while (word_len > 0 && 0 == isalpha((unsigned char)*word)) {
          ++word;
          --word_len;
        }
        while (word_len > 0
          && 0 == isalpha((unsigned char)word[word_len - 1])) --word_len;
        if (word_len > 0) {
          if (0 != answer_relation_rule_is_term(RELATION_RULE_AUX, word,
            word_len))
          {
            governed = 1;
          }
          else if (0 != governed) {
            if (0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE, word,
                word_len)
              || 0 != answer_relation_rule_is_term(RELATION_RULE_PREP, word,
                word_len))
            {
              /* An ARTICLE or a PREPOSITION means a noun phrase follows, not a
               * verb: stepping over them made "had a BIRD" derive "bird". The
               * frame is abandoned rather than continued. */
              governed = 0;
            }
            else if (0 != answer_relation_rule_is_term(RELATION_RULE_SKIP, word,
                word_len)
              || 0 != answer_relation_rule_is_term(RELATION_RULE_NEGATE, word,
                word_len))
            {
              /* Stepped over: a negator or a declared filler is not the verb. */
            }
            else if (0 != isupper((unsigned char)*word)) {
              /**
               * A CAPITAL says this is not the verb, and it is what put nouns in
               * the class: the optative and the month both put a capitalized word
               * straight after an auxiliary ("May God", "May General", "May King"),
               * and those words are attested noun heads 104 to 165 times each. A
               * finite verb governed by an auxiliary is lower case mid-clause, so
               * the capital costs only what a title-cased heading would have given.
               * This matters in BOTH polarities: such a member makes the location
               * layer reject a true actor and the active shape read a noun as a verb
               * ("the Middle Persian work Arda Wiraz"), and it is over-represented in
               * multi-hop paths because a frequent word joins many entities.
               */
              governed = 0;
            }
            else {
              if (2 < word_len && word_len < 32) {
                char lower[32];
                long* count;
                int at;
                for (at = 0; at < word_len; ++at) {
                  lower[at] = (char)tolower((unsigned char)word[at]);
                }
                lower[word_len] = '\0';
                count = (long*)libxs_registry_get(answer_verbs, lower,
                  (size_t)word_len + 1, NULL);
                if (NULL != count) ++*count;
                else {
                  const long fresh = 1;
                  if (NULL != libxs_registry_set(answer_verbs, lower,
                    (size_t)word_len + 1, &fresh, sizeof(fresh), NULL))
                  {
                    ++answer_verbs_nkeys;
                  }
                }
                ++result;
              }
              governed = 0;
            }
          }
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
  return result;
}


static int answer_word_is_verb(const char* word, int word_len)
{
  int result = 0;
  if (NULL != answer_verbs && NULL != word && 2 < word_len && word_len < 32) {
    char lower[32];
    int at;
    for (at = 0; at < word_len; ++at) {
      lower[at] = (char)tolower((unsigned char)word[at]);
    }
    lower[word_len] = '\0';
    result = (NULL != libxs_registry_get(answer_verbs, lower,
      (size_t)word_len + 1, NULL)) ? 1 : 0;
  }
  return result;
}


static long answer_word_verb_count(const char* word, int word_len)
{
  long result = 0;
  if (NULL != answer_verbs && NULL != word && 2 < word_len && word_len < 32) {
    char lower[32];
    const long* count;
    int at;
    for (at = 0; at < word_len; ++at) {
      lower[at] = (char)tolower((unsigned char)word[at]);
    }
    lower[word_len] = '\0';
    count = (const long*)libxs_registry_get(answer_verbs, lower,
      (size_t)word_len + 1, NULL);
    if (NULL != count) result = *count;
  }
  return result;
}


static answer_noun_t* answer_noun_record(const char* word, int word_len,
  int create)
{
  answer_noun_t* result = NULL;
  if (NULL != answer_nouns && NULL != word && 2 < word_len && word_len < 32) {
    char lower[32];
    int at;
    for (at = 0; at < word_len; ++at) {
      lower[at] = (char)tolower((unsigned char)word[at]);
    }
    lower[word_len] = '\0';
    result = (answer_noun_t*)libxs_registry_get(answer_nouns, lower,
      (size_t)word_len + 1, NULL);
    if (NULL == result && 0 != create) {
      answer_noun_t fresh;
      memset(&fresh, 0, sizeof(fresh));
      result = (answer_noun_t*)libxs_registry_set(answer_nouns, lower,
        (size_t)word_len + 1, &fresh, sizeof(fresh), NULL);
      if (NULL != result) ++answer_nouns_nkeys;
    }
  }
  return result;
}


/**
 * Derive the words this corpus uses as NOUNS from the frame that says so without
 * morphology: an ARTICLE heads a noun phrase.
 *
 * The naive reading of that frame - the word after an article is a noun - is
 * REFUTED by measurement, because a participle modifies from exactly that position:
 * "the married couple", "the published work", "the defeated army" would take
 * `married`, `published` and `defeated` out of the verb class, and on Wikipedia that
 * costs the most valuable edges there are (Sparta defeated Athens, Meade defeated
 * Lee). So the frame is read STRICTLY: the word must END the phrase, which is what a
 * HEAD does and what a modifier never does. The phrase ends where the text does, at
 * anything that is not a letter, or at a declared preposition, auxiliary or copula.
 * Measured on wiki8m, that separates cleanly - `god` 65 heads, `king` 109, `name`
 * 294, `work` 89, against 0 for `defeated`, `signed`, `won`, `held`, `killed`.
 *
 * The counts, not a flag, are the point: a caller asks which of the two derived
 * frames attests a word MORE OFTEN, so there is no threshold to pick - the corpus
 * decides, and a word both frames attest equally is read as the noun, since the verb
 * frame is the one known to be incomplete.
 */
static size_t answer_nouns_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_nouns_free();
  answer_nouns = libxs_registry_create();
  if (NULL == corpus || NULL == answer_nouns) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const char* text_end = entry->text + entry->text_len;
      const char* token;
      int token_index = 0, token_len = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        const char* word = token;
        int word_len = token_len;
        while (word_len > 0 && 0 == isalpha((unsigned char)*word)) {
          ++word;
          --word_len;
        }
        while (word_len > 0
          && 0 == isalpha((unsigned char)word[word_len - 1])) --word_len;
        if (word_len > 0 && 0 != answer_relation_rule_is_term(
          RELATION_RULE_ARTICLE, word, word_len))
        {
          const char* head = word + word_len;
          int head_len = 0;
          while (head < text_end && ' ' == *head) ++head;
          while (head + head_len < text_end
            && 0 != isalpha((unsigned char)head[head_len])) ++head_len;
          if (2 < head_len) {
            const char* after = head + head_len;
            int after_len = 0;
            int ends;
            while (after < text_end && ' ' == *after) ++after;
            while (after + after_len < text_end
              && 0 != isalpha((unsigned char)after[after_len])) ++after_len;
            /**
             * A COMMA does not end a noun phrase, and reading it as one is what put
             * adjectives in the class: "a poor, ragged girl" made `poor` a head, and
             * the type shape then asserted "Snowdrop is a poor". Only the marks that
             * end a CLAUSE end the phrase for this purpose.
             */
            /**
             * And the DERIVED VERB CLASS is what says the clause continues, which is
             * the other half of reading this frame: "the girl said" ends the phrase at
             * `girl`, while "the little girl" does not end it at `little`. Without
             * that, every article-led noun followed by an ordinary word counted as
             * modifier evidence and the dominance test rejected `girl` itself. The
             * class is incomplete, so what it misses costs evidence rather than
             * inventing it - the same trade as everywhere else it is read.
             */
            ends = ((0 == after_len
                && (after >= text_end || '.' == *after || '!' == *after
                  || '?' == *after || ';' == *after || ':' == *after))
              || 0 != answer_relation_rule_is_term(RELATION_RULE_PREP, after,
                after_len)
              || 0 != answer_relation_rule_is_term(RELATION_RULE_AUX, after,
                after_len)
              || 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA, after,
                after_len)
              || 0 != answer_word_is_verb(after, after_len)) ? 1 : 0;
            { answer_noun_t* record = answer_noun_record(head, head_len, 1);
              if (NULL != record) {
                if (0 != ends) {
                  ++record->head;
                  ++result;
                }
                else ++record->mod;
              }
            }
          }
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
  return result;
}


/**
 * Is the corpus's own usage of this word more nominal than verbal? Asked where a
 * shape has to tell a clause from an appositive, since "Sparta defeated Athens" and
 * "Egyptian god Horus" are the same three token kinds in the same order.
 */
static int answer_word_is_noun(const char* word, int word_len)
{
  int result = 0;
  const answer_noun_t* record = answer_noun_record(word, word_len, 0);
  if (NULL != record && 0 < record->head) {
    /* Both comparisons are between two attestations of the SAME word, so neither
     * carries a constant to tune: more often a head than a verb, and more often a
     * head than a modifier. The second is what excludes adjectives, which stand
     * where a noun stands but head nothing ("the little girl" against "the girl"). */
    result = (record->head >= answer_word_verb_count(word, word_len)
      && record->head >= record->mod) ? 1 : 0;
  }
  return result;
}


static void answer_nouns_report(FILE* stream)
{
  if (NULL != stream && 0 < answer_nouns_nkeys) {
    fprintf(stream, "nouns derived: %ld from the article frame\n",
      answer_nouns_nkeys);
  }
}


static void answer_verbs_report(FILE* stream)
{
  if (NULL != stream && 0 < answer_verbs_nkeys) {
    const char* list = getenv("CONVERSE_FACTS_LIST");
    fprintf(stream, "verbs derived: %ld from the auxiliary frame\n",
      answer_verbs_nkeys);
    if (NULL != list && '\0' != *list && '0' != *list) {
      const void* key = NULL;
      size_t cursor = 0;
      void* value = libxs_registry_begin(answer_verbs, &key, &cursor);
      int shown = 0;
      while (NULL != value && shown < 40) {
        if (2 <= *(const long*)value) {
          fprintf(stream, "  verb %s (%ld)\n", (const char*)key,
            *(const long*)value);
          ++shown;
        }
        value = libxs_registry_next(answer_verbs, &key, &cursor);
      }
    }
  }
}


/**
 * THE PURE NAME inside a token: what remains once a declared possessive mark is
 * removed, and how many bytes that mark took.
 *
 * This matters far beyond the possession layer. A name is a NODE, and every layer
 * that extracts one keys its facts by it, so "Hansel's" and "Hansel" must resolve
 * to the same bytes or the two facts never join and a question about one misses the
 * other. The mark is orthography, declared per language (`poss|apostrophe-s` for
 * English, `poss|s` for German), so the shapes are tried in the order most specific
 * first: an apostrophe with an s, then a bare apostrophe, then a bare s.
 *
 * The census is the safety net for the bare-s shape, which is otherwise ambiguous:
 * "Muellers" leaves "Mueller" and is accepted because that is a known name, while
 * "Hans" leaves "Han" and is not.
 */
static int answer_name_strip(const char* word, int word_len, int* mark)
{
  int result = word_len;
  int taken = 0;
  if (NULL != word && 2 < word_len) {
    const int last = (int)(unsigned char)word[word_len - 1];
    const int has_s = ('s' == last || 'S' == last) ? 1 : 0;
    int at = word_len - (0 != has_s ? 1 : 0);
    int mark_len = 0;
    if (2 <= at && '\'' == word[at - 1]) mark_len = 1;
    else if (4 <= at && (char)0xe2 == word[at - 3]
      && (char)0x80 == word[at - 2] && (char)0x99 == word[at - 1])
    {
      mark_len = 3;
    }
    if (0 != mark_len && 0 != has_s
      && 0 != answer_relation_rule_is_term(RELATION_RULE_POSS,
        "apostrophe-s", 12))
    {
      taken = mark_len + 1;
    }
    else if (0 != mark_len && 0 == has_s
      && 0 != answer_relation_rule_is_term(RELATION_RULE_POSS, "apostrophe",
        10))
    {
      taken = mark_len;
    }
    else if (0 == mark_len && 0 != has_s
      && 0 != answer_relation_rule_is_term(RELATION_RULE_POSS, "s", 1))
    {
      taken = 1;
    }
    if (0 < taken && word_len - taken > 1) result = word_len - taken;
    else taken = 0;
  }
  if (NULL != mark) *mark = taken;
  return result;
}


/** Whether a token names somebody, once its possessive mark is off. */
static int answer_name_token(const char* word, int word_len, int* pure_len)
{
  int mark = 0;
  const int len = answer_name_strip(word, word_len, &mark);
  const int is_name = (0 < len && 0 != answer_identity_word_is_name(word, len))
    ? 1 : 0;
  if (NULL != pure_len) *pure_len = (0 != is_name) ? len : word_len;
  return is_name;
}


static void answer_own_facts_free(void)
{
  free(answer_own_facts);
  answer_own_facts = NULL;
  answer_own_facts_size = 0;
}


static int answer_own_fact_append(const char* owner, int owner_len,
  const char* item, int item_len, const corpus_entry_t* entry, double score)
{
  int result = EXIT_FAILURE;
  answer_own_fact_t fact;
  answer_own_fact_t* facts;
  size_t fact_pos;
  if (NULL == owner || owner_len <= 0 || owner_len >= (int)sizeof(fact.owner)
    || NULL == item || item_len <= 0 || item_len >= (int)sizeof(fact.item))
  {
    return EXIT_FAILURE;
  }
  memset(&fact, 0, sizeof(fact));
  memcpy(fact.owner, owner, (size_t)owner_len);
  fact.owner_len = owner_len;
  memcpy(fact.item, item, (size_t)item_len);
  fact.item_len = item_len;
  fact.score = score;
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
  if (NULL != entry && entry->section_len > 0
    && entry->section_len < (int)sizeof(fact.section))
  {
    memcpy(fact.section, entry->section, (size_t)entry->section_len);
    fact.section_len = entry->section_len;
  }
  /**
   * An owner may own many things, so only the same item collapses - and one item
   * being a PREFIX of another is the same item read to different lengths. Without
   * that, "Curdken's hat go" stood beside "Curdken's hat" and the enumeration
   * listed one possession twice, once ungrammatically. The shorter survives, since
   * the run stops at the first word that cannot belong to the phrase.
   */
  for (fact_pos = 0; fact_pos < answer_own_facts_size; ++fact_pos) {
    answer_own_fact_t* old_fact = answer_own_facts + fact_pos;
    const int shared = (old_fact->item_len < fact.item_len)
      ? old_fact->item_len : fact.item_len;
    if (old_fact->owner_len == fact.owner_len
      && 0 != libxs_striequal(old_fact->owner, (size_t)old_fact->owner_len,
        fact.owner, (size_t)fact.owner_len)
      && 0 != libxs_striequal(old_fact->item, (size_t)shared, fact.item,
        (size_t)shared))
    {
      if (fact.item_len < old_fact->item_len) *old_fact = fact;
      result = EXIT_SUCCESS;
      break;
    }
  }
  if (EXIT_SUCCESS != result) {
    facts = (answer_own_fact_t*)realloc(answer_own_facts,
      (answer_own_facts_size + 1) * sizeof(*facts));
    if (NULL != facts) {
      answer_own_facts = facts;
      answer_own_facts[answer_own_facts_size] = fact;
      ++answer_own_facts_size;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * Extract possessions from the one shape English marks unambiguously: the
 * possessive apostrophe.
 *
 * THE APOSTROPHE IS INSIDE THE TOKEN. The tokenizer's delimiters are punctuation
 * that separates words, and an apostrophe joins one - so "Curdken's" arrives whole,
 * the name test fails on it, and a first version of this found ZERO possessions in
 * a corpus holding twenty-nine. The mark is therefore looked for within the token:
 * an apostrophe, ASCII or the typographic U+2019 the corpus is typeset with, ending
 * the token before a final s.
 *
 * The possessed phrase is the NEXT WORD only. "the wolf's skin and the cake"
 * possesses a skin, and taking the rest of the clause would have claimed the cake.
 */
static size_t answer_own_facts_build(const libxs_registry_t* corpus)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_own_facts_free();
  if (NULL == corpus) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
      const int heading_len = corpus_title_len(entry->text, entry->text_len);
      const char* token;
      int token_index = 0, token_len = 0;
      while (NULL != (token = libxs_strtoken(entry->text, delims,
        token_index, &token_len)))
      {
        if (token >= entry->text + heading_len && 2 < token_len) {
          const char* word = token;
          int word_len = token_len;
          int mark = 0;
          int owner_len;
          while (word_len > 0 && 0 == isalpha((unsigned char)*word)) {
            ++word;
            --word_len;
          }
          owner_len = answer_name_strip(word, word_len, &mark);
          if (0 != mark) {
            if (0 < owner_len
              && 0 != answer_identity_word_is_name(word, owner_len))
            {
              /**
               * The item is the whole run of CONTENT words after the apostrophe,
               * not the first word: "Ashputtel's two little birds" possesses birds
               * and taking one word claimed a "two", while "Snowdrop's old ..."
               * claimed an "old". The run ends at the first declared function word
               * - article, preposition, copula or skip - which is where an English
               * noun phrase ends, and at any punctuation, which the tokenizer's
               * delimiters already mark.
               */
              const char* item = NULL;
              const char* item_end = NULL;
              int taken = 0;
              int next = token_index + 1;
              int part_len = 0;
              const char* part;
              while (taken < 4 && NULL != (part = libxs_strtoken(entry->text,
                delims, next, &part_len)))
              {
                const char* head = part;
                int head_len = part_len;
                while (head_len > 0 && 0 == isalpha((unsigned char)*head)) {
                  ++head;
                  --head_len;
                }
                while (head_len > 0
                  && 0 == isalpha((unsigned char)head[head_len - 1]))
                {
                  --head_len;
                }
                /* A token holding an apostrophe is another POSSESSIVE, not a
                 * possessed noun: it opens a new relation rather than continuing
                 * this one, and reading it as an item said "Snowdrop's Snowdrop's
                 * old enemy". The trim does not catch it, since the token still
                 * ends in a letter. */
                if (head_len <= 0 || head_len != part_len) break;
                { int mark_pos = 0, has_mark = 0;
                  for (mark_pos = 0; mark_pos + 1 < head_len; ++mark_pos) {
                    if ('\'' == head[mark_pos]
                      || ((char)0xe2 == head[mark_pos]
                        && (char)0x80 == head[mark_pos + 1])) has_mark = 1;
                  }
                  if (0 != has_mark) break;
                }
                if (NULL != item_end && item_end + 1 != head) break;
                if (0 != answer_relation_rule_is_term(RELATION_RULE_SKIP, head,
                    head_len)
                  || 0 != answer_relation_rule_is_term(RELATION_RULE_ARTICLE,
                    head, head_len)
                  || 0 != answer_relation_rule_is_term(RELATION_RULE_PREP,
                    head, head_len)
                  || 0 != answer_relation_rule_is_term(RELATION_RULE_COPULA,
                    head, head_len)) break;
                if (NULL == item) item = head;
                item_end = head + head_len;
                ++taken;
                ++next;
              }
              if (NULL != item && NULL != item_end
                && 2 < (int)(item_end - item)
                && (int)(item_end - item) < ANSWER_OWN_ITEM_MAX
                && EXIT_SUCCESS == answer_own_fact_append(word, owner_len,
                  item, (int)(item_end - item), entry,
                  (double)(ANSWER_OWN_ITEM_MAX - (int)(item_end - item))))
              {
                ++result;
              }
            }
          }
        }
        ++token_index;
      }
    }
    value = corpus_iter_next(corpus, &key, &cursor);
  }
  return result;
}


static void answer_own_facts_report(FILE* stream)
{
  if (NULL != stream && 0 < answer_own_facts_size) {
    const char* list = getenv("CONVERSE_FACTS_LIST");
    fprintf(stream, "possession facts: %lu learned\n",
      (unsigned long)answer_own_facts_size);
    if (NULL != list && '\0' != *list && '0' != *list) {
      size_t fact_pos;
      for (fact_pos = 0; fact_pos < answer_own_facts_size; ++fact_pos) {
        const answer_own_fact_t* fact = answer_own_facts + fact_pos;
        fprintf(stream, "  own %s <- %s\n", fact->owner, fact->item);
      }
    }
  }
}


/**
 * Answer a possession question by ENUMERATING what is attested, not by picking one.
 *
 * A question asking what belongs to somebody has a SET as its answer, and stating
 * one member of a set is a different claim from stating the set. Each item is an
 * independently attested possessive phrase and contributes its own citation, so the
 * enumeration is grounded item by item rather than as a whole. The verb and its
 * plural come from the rule file (`own|belongs`, `own|belong`), so the reply is
 * assembled from declared vocabulary and corpus bytes only.
 */
static int answer_own_fact_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char owner[64];
  const answer_own_fact_t* items[ANSWER_OWN_MAX];
  int owner_len = 0, count = 0, marked = 0, item, token_index = 0;
  int token_len = 0;
  size_t fact_pos, pos = 0;
  const char* token;
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_own_facts_size) return EXIT_FAILURE;
  /* The question must MARK possession, and the owner is the name after it. */
  while (0 == owner_len && NULL != (token = libxs_strtoken(query_text, delims,
    token_index, &token_len)))
  {
    while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
      ++token;
      --token_len;
    }
    while (token_len > 0
      && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
    if (token_len > 0) {
      int pure_len = token_len;
      const int named = answer_name_token(token, token_len, &pure_len);
      if (0 != marked && 0 != named && pure_len < (int)sizeof(owner)) {
        /* The PURE name, so a question written possessively reaches the node the
         * facts are keyed by. */
        memcpy(owner, token, (size_t)pure_len);
        owner[pure_len] = '\0';
        owner_len = pure_len;
      }
      else if (0 != answer_relation_rule_is_term(RELATION_RULE_OWN, token,
        token_len)) marked = 1;
    }
    ++token_index;
  }
  if (owner_len <= 0) return EXIT_FAILURE;
  for (fact_pos = 0; fact_pos < answer_own_facts_size
    && count < ANSWER_OWN_MAX; ++fact_pos)
  {
    const answer_own_fact_t* fact = answer_own_facts + fact_pos;
    if (0 != text_contains_word_ci(fact->owner, fact->owner_len, owner)) {
      items[count++] = fact;
    }
  }
  if (0 < count) {
    int verb_len = 0;
    const char* verb = answer_relation_rule_first_term(RELATION_RULE_OWN,
      &verb_len);
    if (1 < count) {
      /* The plural is the SECOND declared term, so the rule file supplies both. */
      size_t rule_pos;
      int seen = 0;
      for (rule_pos = 0; rule_pos < converse_rules_size(); ++rule_pos) {
        const answer_relation_rule_t* rule = converse_rules() + rule_pos;
        if (RELATION_RULE_OWN == rule->kind && 0 != seen++) {
          verb = rule->term;
          verb_len = (int)strlen(rule->term);
          break;
        }
      }
    }
    for (item = 0; item < count; ++item) {
      const char* joiner = (0 == item) ? "" : ((item + 1 == count)
        ? " and " : ", ");
      const size_t joiner_len = strlen(joiner);
      if (pos + joiner_len + (size_t)items[item]->item_len + 2 >= output_size) {
        break;
      }
      memcpy(output + pos, joiner, joiner_len);
      pos += joiner_len;
      pos = answer_append_clean(output, output_size, pos, items[item]->item,
        items[item]->item_len);
      answer_origin_add(items[item]->source, items[item]->line);
      answer_fact_section_add(items[item]->section,
        items[item]->section_len);
    }
    if (NULL != verb && 0 < verb_len && 0 < pos
      && pos + (size_t)verb_len + (size_t)owner_len + 8 < output_size)
    {
      output[pos++] = ' ';
      memcpy(output + pos, verb, (size_t)verb_len);
      pos += (size_t)verb_len;
      memcpy(output + pos, " to ", 4);
      pos += 4;
      memcpy(output + pos, owner, (size_t)owner_len);
      pos += (size_t)owner_len;
      output[pos++] = '.';
      output[pos] = '\0';
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


/**
 * The name an attribute question is about: the one that FOLLOWS the topic marker.
 *
 * Position, not scoring: "what do we know about Gretel" states which token is the
 * subject by putting it after "about", and a question that names two people asks
 * about the one it marks. Without the marker there is no attribute question here at
 * all, so this resolver never competes for a question that is about something else.
 */
static int answer_topic_query_name(const char* query_text, size_t query_len,
  char* name, int name_size)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"";
  int result = 0;
  int token_index = 0, token_len = 0, marked = 0;
  const char* token;
  if (NULL == query_text || 0 == query_len || NULL == name) return 0;
  while (0 == result && NULL != (token = libxs_strtoken(query_text, delims,
    token_index, &token_len)))
  {
    while (token_len > 0 && 0 == isalpha((unsigned char)*token)) {
      ++token;
      --token_len;
    }
    while (token_len > 0
      && 0 == isalpha((unsigned char)token[token_len - 1])) --token_len;
    if (token_len > 0) {
      int pure_len = token_len;
      const int named = answer_name_token(token, token_len, &pure_len);
      if (0 != marked && 0 != named && pure_len < name_size) {
        memcpy(name, token, (size_t)pure_len);
        name[pure_len] = '\0';
        result = pure_len;
      }
      else if (0 != answer_relation_rule_is_term(RELATION_RULE_TOPIC, token,
        token_len))
      {
        marked = 1;
      }
    }
    ++token_index;
  }
  return result;
}


/**
 * Everything the corpus STATES about one name, as several cited propositions.
 *
 * This is navigation rather than retrieval: the facts are already extracted and
 * already grammatical, so collecting them is a walk over what is attested, and the
 * reply cites every source it rests on. Nothing is inferred and nothing is joined
 * across facts - a proposition that needed two facts to be true would be a hop,
 * and a hop is only sound where every step of it is attested.
 *
 * The reply is labelled with the WEAKEST term it rests on, not the strongest: an
 * attribute collection holding one proposed class member is a collection a reader
 * must be able to discount, and naming the best of four would hide exactly that.
 */
static int answer_topic_reply(const char* query_text, size_t query_len,
  char* output, size_t output_size)
{
  int result = EXIT_FAILURE;
  char name[64];
  char items[ANSWER_TOPIC_MAX][COMPOSE_MAXTEXT];
  const char* sections[ANSWER_TOPIC_MAX];
  int section_lens[ANSWER_TOPIC_MAX];
  unsigned int origin_sources[ANSWER_TOPIC_MAX];
  unsigned int origin_lines[ANSWER_TOPIC_MAX];
  int worst = RELATION_RULE_ASSERTED;
  char worst_term[64];
  int worst_term_len = 0;
  int name_len, count = 0, item, relation_base;
  size_t fact_pos, pos = 0;
  if (NULL == query_text || NULL == output || 0 == output_size) {
    return EXIT_FAILURE;
  }
  name_len = answer_topic_query_name(query_text, query_len, name,
    (int)sizeof(name));
  if (name_len <= 0) return EXIT_FAILURE;
  worst_term[0] = '\0';
  /* What it IS, first: a role binds a name to a class and reads as a definition. */
  for (fact_pos = 0; fact_pos < answer_identity_facts_size
    && count < ANSWER_TOPIC_MAX; ++fact_pos)
  {
    const answer_identity_fact_t* fact = answer_identity_facts + fact_pos;
    if (fact->name_len == name_len
      && 0 != libxs_striequal(fact->name, (size_t)fact->name_len, name,
        (size_t)name_len)
      && EXIT_SUCCESS == answer_reply_role(items[count], sizeof(items[count]),
        fact->name, fact->name_len, fact->role))
    {
      sections[count] = fact->section;
      section_lens[count] = fact->section_len;
      origin_sources[count] = fact->source;
      origin_lines[count] = fact->line;
      if (fact->provenance > worst) {
        worst = fact->provenance;
        memcpy(worst_term, fact->role, (size_t)fact->role_len + 1);
        worst_term_len = fact->role_len;
      }
      ++count;
    }
  }
  /**
   * What happened to or through it: a relation fact reads from either side, since
   * "X was eaten by the wolf" is a fact about X and about the wolf alike.
   *
   * WHICH of them, when a Wikipedia entity has thirty, was the last arbitrary choice
   * in a reply - the first four in fact order, which is registry hash order and so
   * means nothing to a reader. The corpus supplies the order itself: the EARLIEST
   * cited facts are the ones its lead states, and a lead states what defines its
   * subject. So this selects and shows the earliest, which is a relevance order the
   * text asserts rather than one this code scores. Prediction was considered for the
   * same slot and DECLINED: ordering by a model would be a judgement where the corpus
   * already answers, and the field it would rank on is the one being replaced.
   */
  relation_base = count;
  for (fact_pos = 0; fact_pos < answer_relation_facts_size; ++fact_pos) {
    const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
    const int is_answer = (fact->answer_len > 0
      && 0 != text_contains_word_ci(fact->answer, fact->answer_len, name))
      ? 1 : 0;
    const int is_actor = (fact->actor_len > 0
      && 0 != text_contains_word_ci(fact->actor, fact->actor_len, name))
      ? 1 : 0;
    if ((0 != is_answer || 0 != is_actor) && fact->relation_len > 0
      && 0 == fact->made)
    {
      answer_relation_match_t match;
      char rendered[COMPOSE_MAXTEXT];
      memset(&match, 0, sizeof(match));
      memcpy(match.answer, fact->answer, (size_t)fact->answer_len + 1);
      match.answer_len = fact->answer_len;
      memcpy(match.relation, fact->relation, (size_t)fact->relation_len + 1);
      match.relation_len = fact->relation_len;
      memcpy(match.actor, fact->actor, (size_t)fact->actor_len + 1);
      match.actor_len = fact->actor_len;
      match.plural = fact->plural;
      match.made = fact->made;
      match.active = fact->active;
      if (EXIT_SUCCESS == answer_relation_reply(&match, rendered,
        sizeof(rendered)))
      {
        int slot = count;
        while (slot > relation_base
          && (origin_sources[slot - 1] > fact->source
            || (origin_sources[slot - 1] == fact->source
              && origin_lines[slot - 1] > fact->line)))
        {
          --slot;
        }
        if (count >= ANSWER_TOPIC_MAX) {
          if (slot < ANSWER_TOPIC_MAX) --count;
          else slot = -1;
        }
        if (0 <= slot) {
          int at;
          for (at = count; at > slot; --at) {
            memcpy(items[at], items[at - 1], sizeof(items[at]));
            sections[at] = sections[at - 1];
            section_lens[at] = section_lens[at - 1];
            origin_sources[at] = origin_sources[at - 1];
            origin_lines[at] = origin_lines[at - 1];
          }
          memcpy(items[slot], rendered, sizeof(rendered));
          sections[slot] = fact->section;
          section_lens[slot] = fact->section_len;
          origin_sources[slot] = fact->source;
          origin_lines[slot] = fact->line;
          ++count;
        }
      }
    }
  }
  /* Where it was: the location layer's propositions, already verbatim. */
  for (fact_pos = 0; fact_pos < answer_location_facts_size
    && count < ANSWER_TOPIC_MAX; ++fact_pos)
  {
    const answer_location_fact_t* fact = answer_location_facts + fact_pos;
    if (fact->actor_len == name_len
      && 0 != libxs_striequal(fact->actor, (size_t)fact->actor_len, name,
        (size_t)name_len))
    {
      size_t at = answer_append_clean(items[count], sizeof(items[count]), 0,
        fact->actor, fact->actor_len);
      if (at + 2 < sizeof(items[count])) items[count][at++] = ' ';
      at = answer_append_clean(items[count], sizeof(items[count]), at,
        fact->phrase, fact->phrase_len);
      if (at + 2 < sizeof(items[count])) {
        items[count][at++] = '.';
        items[count][at] = '\0';
        sections[count] = fact->section;
        section_lens[count] = fact->section_len;
        origin_sources[count] = fact->source;
        origin_lines[count] = fact->line;
        if (fact->provenance > worst) {
          worst = fact->provenance;
          memcpy(worst_term, fact->place, (size_t)fact->place_len + 1);
          worst_term_len = fact->place_len;
        }
        ++count;
      }
    }
  }
  /**
   * SPECULATION LAST, and only when there is nothing else to say.
   *
   * The `made` template takes whatever word follows "made" as the predicate, and on
   * the tales only 2 of its 16 facts are true - the rest are idioms ("made use
   * of", "made her way", "made room"). No score separates them (E5 measured that),
   * so such a fact is labelled rather than trusted. But labelling the COLLECTION by
   * its weakest member made the whole thing lose to ranked evidence, and a reader
   * asking about Hansel then got an arbitrary sentence instead of the true, cited
   * proposition the collection already held. A set of independent propositions is
   * not as weak as its weakest member: the attested ones are stated plainly, and
   * speculation speaks only when the corpus attests nothing at all.
   */
  for (fact_pos = 0; fact_pos < answer_relation_facts_size && 0 == count;
    ++fact_pos)
  {
    const answer_relation_fact_t* fact = answer_relation_facts + fact_pos;
    if (0 != fact->made && fact->relation_len > 0 && fact->answer_len > 0
      && 0 != text_contains_word_ci(fact->answer, fact->answer_len, name))
    {
      answer_relation_match_t match;
      memset(&match, 0, sizeof(match));
      memcpy(match.answer, fact->answer, (size_t)fact->answer_len + 1);
      match.answer_len = fact->answer_len;
      memcpy(match.relation, fact->relation, (size_t)fact->relation_len + 1);
      match.relation_len = fact->relation_len;
      match.plural = fact->plural;
      match.made = fact->made;
      if (EXIT_SUCCESS == answer_relation_reply(&match, items[count],
        sizeof(items[count])))
      {
        sections[count] = fact->section;
        section_lens[count] = fact->section_len;
        origin_sources[count] = fact->source;
        origin_lines[count] = fact->line;
        worst = RELATION_RULE_PROPOSED;
        memcpy(worst_term, fact->relation, (size_t)fact->relation_len + 1);
        worst_term_len = fact->relation_len;
        ++count;
      }
    }
  }
  output[0] = '\0';
  for (item = 0; item < count; ++item) {
    const size_t len = strlen(items[item]);
    if (pos + len + 2 >= output_size) break;
    if (0 < pos) output[pos++] = ' ';
    memcpy(output + pos, items[item], len);
    pos += len;
    output[pos] = '\0';
    answer_origin_add(origin_sources[item], origin_lines[item]);
    answer_fact_section_add(sections[item], section_lens[item]);
  }
  if (0 < pos) {
    if (RELATION_RULE_ASSERTED < worst) {
      answer_fact_learned_set(worst_term, worst_term_len, worst);
    }
    result = EXIT_SUCCESS;
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
  memset(&fact, 0, sizeof(fact));
  if (text_len >= (int)sizeof(fact.text)) text_len = (int)sizeof(fact.text) - 1;
  memcpy(fact.role, role, (size_t)role_len);
  fact.role[role_len] = '\0';
  fact.role_len = role_len;
  memcpy(fact.text, text, (size_t)text_len);
  fact.text[text_len] = '\0';
  fact.text_len = text_len;
  fact.section_len = entry->section_len;
  fact.source = (NULL != entry) ? entry->source : 0;
  fact.line = (NULL != entry) ? entry->line : 0;
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
  if (NULL == corpus || 0 == converse_rules_size()) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    /* Fragments re-derive their parent sentence's facts; see the identity
     * builder. Sentence scale alone was not enough. */
    if (SCALE_SENTENCE == entry->scale
      && 0 == (entry->lexical_flags & ENTRY_LEX_FRAGMENT))
    {
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
                  && ';' != *end && '!' != *end && '?' != *end
                  && '\n' != *end && '\r' != *end) ++end;
                /**
                 * CUT OFF by a line break rather than closed by punctuation or by
                 * the end of the entry: the clause is incomplete ("A young fox,
                 * who said:" - the speech was on the next line) and describes
                 * nothing, so it is not a description. Rejected the same way a
                 * clause-less role is, by never becoming a fact.
                 */
                if (end < text_end && ('\n' == *end || '\r' == *end)) {
                  end = clause;
                }
                else score = 2.0;
              }
            }
            /**
             * A described role needs a DESCRIPTION. Without the relative clause
             * `end` still points at the role itself, so the fact would be the
             * article plus the queried word and the reply would restate the
             * question: "Who is the wife?" -> "A wife." Such a fact cannot inform
             * any answer, so it is never stored, and the query abstains - which
             * is the truth, since the corpus describes no wife.
             */
            if (end > clause
              && EXIT_SUCCESS == answer_describe_fact_append(token, token_len,
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
    value = corpus_iter_next(corpus, &key, &cursor);
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
    (int)sizeof(role));
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
    answer_fact_section_set(best->section, best->section_len);
    answer_origin_add(best->source, best->line);
    answer_fact_learned_set(best->role, best->role_len,
      answer_relation_rule_provenance(RELATION_RULE_PERSON, best->role,
        best->role_len));
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

/**
 * Parse an optional leading "Header: `name`" line and return the byte offset
 * of the definition prose that follows it (0 when absent). Fills header with
 * the base module name (backticks and trailing extension stripped).
 */
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

/**
 * Learn module definitions from Markdown ingest: the first paragraph under a
 * heading becomes that title's definition, with the "Header:" filename kept as
 * an alias. Structural only (no corpus vocabulary), so prose corpora without
 * headings contribute nothing.
 */
static size_t answer_docdef_facts_build(const libxs_registry_t* corpus)
{
  const void* key = NULL;
  size_t cursor = 0;
  void* value;
  size_t result = 0;
  answer_docdef_facts_free();
  if (NULL == corpus) return 0;
  value = corpus_iter_begin(corpus, &key, &cursor);
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
            memset(fact, 0, sizeof(*fact));
            if (text_len >= (int)sizeof(fact->text)) {
              text_len = (int)sizeof(fact->text) - 1;
            }
            memcpy(fact->title, entry->section, (size_t)entry->section_len);
            fact->title[entry->section_len] = '\0';
            fact->title_len = entry->section_len;
            fact->source = entry->source;
            /* The header prefix is STRIPPED from what this fact states, so its line
             * is the entry's plus whatever the strip skipped over - otherwise a
             * definition is cited to the line its "Header:" line sits on. */
            fact->line = entry->line;
            { int at;
              for (at = 0; at < offset && at < entry->text_len; ++at) {
                if ('\n' == entry->text[at]) ++fact->line;
              }
            }
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
    value = corpus_iter_next(corpus, &key, &cursor);
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


/**
 * What the derived layer was built FROM, in one word.
 *
 * The facts and the name census are a function of the corpus, the rule file
 * (including anything rule learning appended) and the normalization. Reusing
 * them across a change in any of those is the warm-start trap this project has
 * been bitten by repeatedly, and it fails SILENTLY: every reply stays fluent and
 * becomes wrong. So the cache carries a stamp and is discarded unless it
 * matches, which makes staleness unrepresentable rather than unlikely.
 *
 * What the stamp does not catch is a corpus edited to exactly the same entry
 * count under exactly the same rules. Catching that needs a scan of the corpus,
 * which is the work the cache exists to avoid - so it is stated here rather
 * than papered over. Deleting the file is always safe.
 */
static unsigned int answer_facts_stamp(const libxs_registry_t* corpus,
  const libxs_lexicon_t* lexicon)
{
  unsigned int result = ANSWER_FACTS_VERSION;
  size_t pos;
  result = libxs_hash(&result, sizeof(result),
    (unsigned int)libxs_registry_size(corpus));
  result = libxs_hash(&result, sizeof(result),
    (unsigned int)libxs_lexicon_size(lexicon));
  /* Field by field, never the whole struct: a fixed-size text field carries
   * indeterminate bytes past its terminator, so hashing the struct hashes them
   * and the stamp differs from itself between two runs on identical input. */
  for (pos = 0; pos < converse_rules_size(); ++pos) {
    const answer_relation_rule_t* rule = converse_rules() + pos;
    const unsigned int kinds = (unsigned int)(rule->kind * 8 + rule->provenance);
    result = libxs_hash(&kinds, sizeof(kinds), result);
    result = libxs_hash(rule->relation, (unsigned int)strlen(rule->relation),
      result);
    result = libxs_hash(rule->term, (unsigned int)strlen(rule->term), result);
  }
  for (pos = 0; pos < (size_t)converse_lexnorms_size(); ++pos) {
    const libxs_lexnorm_t* norm = converse_lexnorms() + pos;
    result = libxs_hash(norm->from, (unsigned int)strlen(norm->from), result);
    result = libxs_hash(norm->to, (unsigned int)strlen(norm->to), result);
  }
  return result;
}


static int answer_facts_write(FILE* file, const void* data, size_t count,
  size_t size)
{
  return (0 == count
    || (NULL != data && fwrite(data, size, count, file) == count))
    ? EXIT_SUCCESS : EXIT_FAILURE;
}


static void* answer_facts_read(FILE* file, size_t count, size_t size)
{
  void* result = NULL;
  if (0 < count) {
    result = malloc(count * size);
    if (NULL != result && fread(result, size, count, file) != count) {
      free(result);
      result = NULL;
    }
  }
  return result;
}


static void answer_facts_save(const libxs_registry_t* corpus,
  const libxs_lexicon_t* lexicon)
{
  FILE* file = fopen(converse_facts_path(), "wb");
  if (NULL != file) {
    answer_facts_header_t header;
    int ok;
    memset(&header, 0, sizeof(header));
    header.magic = ANSWER_FACTS_MAGIC;
    header.version = ANSWER_FACTS_VERSION;
    header.stamp = answer_facts_stamp(corpus, lexicon);
    header.ncase = answer_case_size;
    header.nrelation = (unsigned int)answer_relation_facts_size;
    header.nidentity = (unsigned int)answer_identity_facts_size;
    header.ndescribe = (unsigned int)answer_describe_facts_size;
    header.ndocdef = (unsigned int)answer_docdef_facts_size;
    header.nlocation = (unsigned int)answer_location_facts_size;
    header.ntype = (unsigned int)answer_type_facts_size;
    header.nown = (unsigned int)answer_own_facts_size;
    ok = (1 == fwrite(&header, sizeof(header), 1, file))
      ? EXIT_SUCCESS : EXIT_FAILURE;
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file, answer_case_upper,
      header.ncase, sizeof(*answer_case_upper));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file, answer_case_total,
      header.ncase, sizeof(*answer_case_total));
    /**
     * All four census arrays, not the two the Hilbert-era cache wrote. A reply
     * asks answer_word_is_name whether a word is a name, and that reads the
     * UNFORCED counts - so a cache holding only upper and total restored a
     * census that every query then dereferenced through a NULL pointer. It was a
     * SEGFAULT on the first question of any run that hit the cache, which is why
     * a third consecutive warm run appeared to "stop evaluating".
     */
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_case_unforced, header.ncase, sizeof(*answer_case_unforced));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file, answer_case_attrib,
      header.ncase, sizeof(*answer_case_attrib));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_relation_facts, header.nrelation, sizeof(*answer_relation_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_identity_facts, header.nidentity, sizeof(*answer_identity_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_describe_facts, header.ndescribe, sizeof(*answer_describe_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_docdef_facts, header.ndocdef, sizeof(*answer_docdef_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file,
      answer_location_facts, header.nlocation,
      sizeof(*answer_location_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file, answer_type_facts,
      header.ntype, sizeof(*answer_type_facts));
    if (EXIT_SUCCESS == ok) ok = answer_facts_write(file, answer_own_facts,
      header.nown, sizeof(*answer_own_facts));
    fclose(file);
    /* A half-written cache would be read back as a valid one, so it is removed
     * rather than left for the next run to trust. */
    if (EXIT_SUCCESS != ok) remove(converse_facts_path());
  }
}


static int answer_facts_load(const libxs_registry_t* corpus,
  const libxs_lexicon_t* lexicon)
{
  int result = EXIT_FAILURE;
  FILE* file = fopen(converse_facts_path(), "rb");
  if (NULL != file) {
    answer_facts_header_t header;
    memset(&header, 0, sizeof(header));
    if (1 == fread(&header, sizeof(header), 1, file)
      && ANSWER_FACTS_MAGIC == header.magic
      && ANSWER_FACTS_VERSION == header.version
      && header.stamp == answer_facts_stamp(corpus, lexicon))
    {
      unsigned int* case_upper = (unsigned int*)answer_facts_read(file,
        header.ncase, sizeof(*answer_case_upper));
      unsigned int* case_total = (unsigned int*)answer_facts_read(file,
        header.ncase, sizeof(*answer_case_total));
      unsigned int* case_unforced = (unsigned int*)answer_facts_read(file,
        header.ncase, sizeof(*answer_case_unforced));
      unsigned int* case_attrib = (unsigned int*)answer_facts_read(file,
        header.ncase, sizeof(*answer_case_attrib));
      answer_relation_fact_t* relation = (answer_relation_fact_t*)
        answer_facts_read(file, header.nrelation,
          sizeof(*answer_relation_facts));
      answer_identity_fact_t* identity = (answer_identity_fact_t*)
        answer_facts_read(file, header.nidentity,
          sizeof(*answer_identity_facts));
      answer_describe_fact_t* describe = (answer_describe_fact_t*)
        answer_facts_read(file, header.ndescribe,
          sizeof(*answer_describe_facts));
      answer_docdef_fact_t* docdef = (answer_docdef_fact_t*)
        answer_facts_read(file, header.ndocdef, sizeof(*answer_docdef_facts));
      answer_location_fact_t* location = (answer_location_fact_t*)
        answer_facts_read(file, header.nlocation,
          sizeof(*answer_location_facts));
      answer_type_fact_t* type = (answer_type_fact_t*)
        answer_facts_read(file, header.ntype, sizeof(*answer_type_facts));
      answer_own_fact_t* own = (answer_own_fact_t*)
        answer_facts_read(file, header.nown, sizeof(*answer_own_facts));
      /* All or nothing: a partially adopted layer answers from one half of a
       * corpus with the census of another. */
      if ((0 == header.ncase || (NULL != case_upper && NULL != case_total
          && NULL != case_unforced && NULL != case_attrib))
        && (0 == header.nrelation || NULL != relation)
        && (0 == header.nidentity || NULL != identity)
        && (0 == header.ndescribe || NULL != describe)
        && (0 == header.ndocdef || NULL != docdef)
        && (0 == header.nlocation || NULL != location)
        && (0 == header.ntype || NULL != type)
        && (0 == header.nown || NULL != own))
      {
        answer_case_free();
        answer_relation_facts_free();
        answer_identity_facts_free();
        answer_describe_facts_free();
        answer_docdef_facts_free();
        answer_location_facts_free();
        answer_type_facts_free();
        answer_own_facts_free();
        answer_case_upper = case_upper;
        answer_case_total = case_total;
        answer_case_unforced = case_unforced;
        answer_case_attrib = case_attrib;
        answer_case_size = header.ncase;
        answer_relation_facts = relation;
        answer_relation_facts_size = header.nrelation;
        answer_identity_facts = identity;
        answer_identity_facts_size = header.nidentity;
        answer_describe_facts = describe;
        answer_describe_facts_size = header.ndescribe;
        answer_docdef_facts = docdef;
        answer_docdef_facts_size = header.ndocdef;
        answer_location_facts = location;
        answer_location_facts_size = header.nlocation;
        answer_type_facts = type;
        answer_type_facts_size = header.ntype;
        answer_own_facts = own;
        answer_own_facts_size = header.nown;
        answer_relation_facts_index();
        result = EXIT_SUCCESS;
      }
      else {
        free(case_upper); free(case_total); free(case_unforced);
        free(case_attrib); free(relation);
        free(identity); free(describe); free(docdef); free(location);
        free(type); free(own);
      }
    }
    fclose(file);
  }
  return result;
}

/**
 * Extract the subject term from "what is [the] X" or "what does X do",
 * skipping a leading article. Language-generic, no corpus vocabulary.
 */
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

/**
 * Answer "What is X?" / "What does X do?" from a learned module definition,
 * matching X against either the heading text or the Header: filename alias.
 */
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
    answer_fact_section_set(best->title, best->title_len);
    answer_origin_add(best->source, best->line);
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

/**
 * Remember the subject of a successfully answered question so that a later
 * follow-up can refer back to it. Uses the same subject extraction as the
 * definition path; only a real subject term updates the topic.
 */
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


/**
 * Is this a BACK-REFERENCE pronoun - a word a follow-up points at the topic with?
 *
 * Declared (`pron|it`), which retires the last eight English literals the C held. The
 * class is deliberately third-person only: "me" and "you" name a participant in the
 * conversation and never the topic, so substituting one would answer about the wrong
 * thing. A language that back-references differently says so in its rule file.
 */
static int conv_word_is_pronoun(const char* word, int len)
{
  return answer_relation_rule_is_term(RELATION_RULE_PRON, word, len);
}

/**
 * Rewrite a follow-up against the remembered topic. Two grounded moves:
 * substitute a back-reference pronoun (it/its/that/...) with the topic, and,
 * when a question carries no subject of its own, append the topic so the
 * answer path is scoped to it. Returns EXIT_SUCCESS only when a rewrite was
 * made, leaving the original query untouched otherwise.
 */
/**
 * Does this question already name what it is ABOUT?
 *
 * The multi-turn rewrite carries a remembered topic in by appending " of the <topic>"
 * to a question with no subject of its own, and that is right for "What does it do?"
 * and wrong for everything else. `answer_docdef_term` was the only test, and it reads
 * a DEFINITION term, so "Tell me about the wolf?" looked subjectless: asked alone it
 * abstained correctly, asked after a question about Gretel it answered ABOUT GRETEL,
 * cited - a confident wrong answer, and the same defect E16 STEP 3 fixed at a
 * different trigger.
 *
 * The test is DECLARATIVE and deliberately not the derived noun class, which was tried
 * first and is too weak on narrative prose: "wolf" is not a noun by that class on the
 * tales, because the phrase-end rule needs a derived VERB after it and the tales are
 * past simple, which no auxiliary governs. So a subject is any word the rule file does
 * NOT account for - not a function word, not a question word, not the topic marker,
 * not a back-reference pronoun. A genuine follow-up ("What does it do?") is made
 * entirely of declared words and still rewrites.
 */
static int conv_query_has_subject(const char* query_text, size_t query_len)
{
  static const char delims[] = " \t\r\n,.;:!?()[]{}\"'";
  int result = 0;
  int token_index = 0, token_len = 0;
  const char* token;
  char buffer[COMPOSE_MAXTEXT];
  size_t copy = query_len;
  if (NULL == query_text || 0 == query_len) return 0;
  if (copy >= sizeof(buffer)) copy = sizeof(buffer) - 1;
  memcpy(buffer, query_text, copy);
  buffer[copy] = '\0';
  while (0 == result && NULL != (token = libxs_strtoken(buffer, delims,
    token_index, &token_len)))
  {
    if (1 < token_len && 0 == answer_word_is_function(token, token_len)
      && 0 == answer_relation_rule_is_term(RELATION_RULE_ASK, token, token_len)
      && 0 == answer_relation_rule_is_term(RELATION_RULE_TOPIC, token,
        token_len)
      && 0 == conv_word_is_pronoun(token, token_len))
    {
      result = 1;
    }
    ++token_index;
  }
  return result;
}


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
    if (own_len <= 0 && pos > 0
      && 0 == conv_query_has_subject(query_text, query_len))
    {
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


/**
 * Does the question ask about the relation this fact states? The relation is one
 * word on both sides, so this is EQUALITY, directly or through an alias rule.
 *
 * It used to be a substring test, which is the same defect the actor matching
 * carried: a longer word containing the relation borrowed the fact, so a
 * question about a different verb - including one built by negating this verb
 * with a prefix - was answered by it, asserted, and cited. A relation that
 * merely CONTAINS another relation is a different relation.
 */
static int answer_relation_fact_relation_match(const char* query_relation,
  const answer_relation_fact_t* fact)
{
  int result = 0;
  size_t rule_pos;
  if (NULL == query_relation || NULL == fact) return 0;
  if (0 != libxs_striequal(query_relation, strlen(query_relation),
    fact->relation, (size_t)fact->relation_len))
  {
    result = 1;
  }
  for (rule_pos = 0; rule_pos < converse_rules_size() && 0 == result;
    ++rule_pos)
  {
    const answer_relation_rule_t* rule = converse_rules() + rule_pos;
    if (RELATION_RULE_ALIAS == rule->kind
      && 0 != libxs_striequal(rule->relation, strlen(rule->relation),
        fact->relation, (size_t)fact->relation_len)
      && 0 != libxs_striequal(query_relation, strlen(query_relation),
        rule->term, strlen(rule->term)))
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
  const answer_relation_fact_t* answer_facts[RELATION_FACT_MAX];
  int answer_lens[RELATION_FACT_MAX];
  int answer_made[RELATION_FACT_MAX];
  double answer_scores[RELATION_FACT_MAX];
  int relation_len;
  int actor_len;
  int query_section_len;
  int count = 0;
  int slot;
  const unsigned int* candidates;
  unsigned int ncandidates = 0;
  unsigned int candidate;
  if (NULL == query_text || NULL == output || 0 == output_size
    || 0 == answer_relation_facts_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (relation_len <= 0 || 0 != answer_word_is_name(relation, relation_len)) {
    return EXIT_FAILURE;
  }
  for (slot = 0; slot < RELATION_FACT_MAX; ++slot) {
    answers[slot][0] = '\0';
    answer_facts[slot] = NULL;
    answer_lens[slot] = 0;
    answer_made[slot] = 0;
    answer_scores[slot] = 0.0;
  }
  /**
   * Candidates come from the index, which holds the facts stating this relation
   * under any of its names, in fact order. The three predicates below still
   * decide, so the index can only cost a scan, never an answer. A relation with
   * no postings is one no fact states under any of its names, which is the same
   * nothing the scan would have found.
   */
  candidates = answer_fact_index_get(&answer_relation_by_relation,
    answer_word_key(relation, relation_len), &ncandidates);
  for (candidate = 0; candidate < ncandidates; ++candidate) {
    const answer_relation_fact_t* fact = answer_relation_facts
      + candidates[candidate];
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
            answer_facts[slot] = fact;
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
          answer_facts[insert] = answer_facts[insert - 1];
          answer_lens[insert] = answer_lens[insert - 1];
          answer_made[insert] = answer_made[insert - 1];
          answer_scores[insert] = answer_scores[insert - 1];
          --insert;
        }
        memcpy(answers[insert], fact->answer,
          (size_t)fact->answer_len + 1);
        answer_facts[insert] = fact;
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
    /* The emitted answers ARE the class terms this reply rests on, so the rule
     * layer is asked about them rather than a flag being cached per fact. */
    for (item = 0; item < count && 0 == answer_fact_learned_len; ++item) {
      answer_fact_learned_set(answers[item], answer_lens[item],
        answer_relation_rule_provenance(RELATION_RULE_PERSON, answers[item],
          answer_lens[item]));
    }
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
      /* Named once the reply exists, so a resolver that assembles nothing
       * leaves no citation behind. Every fact that reached the reply names its
       * source, and a reply resting on two tales is credited to both rather
       * than to neither. */
      for (item = 0; item < count; ++item) {
        if (NULL != answer_facts[item]) {
          answer_origin_add(answer_facts[item]->source,
            answer_facts[item]->line);
          answer_fact_section_add(answer_facts[item]->section,
            answer_facts[item]->section_len);
        }
      }
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
  size_t cursor = 0, key_size = 0;
  void* value;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len;
  char answers[RELATION_AGG_MAX][64];
  char answer_sections[RELATION_AGG_MAX][ENTRY_SECTION_MAX];
  int answer_section_lens[RELATION_AGG_MAX];
  int answer_lens[RELATION_AGG_MAX];
  double answer_scores[RELATION_AGG_MAX];
  int count = 0;
  char relation[64];
  char actor[64];
  int relation_len;
  int actor_len;
  if (NULL == corpus || NULL == query_text || NULL == output
    || 0 == output_size) return EXIT_FAILURE;
  relation_len = answer_query_be_word(query_text, query_len, relation,
    (int)sizeof(relation));
  actor_len = answer_query_relation_actor(query_text, query_len, actor,
    (int)sizeof(actor));
  query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  if (query_section_len > 0 && relation_len > 0
    && 0 == answer_word_is_name(relation, relation_len)
    && actor_len > 0)
  {
    int slot;
    for (slot = 0; slot < RELATION_AGG_MAX; ++slot) {
      answers[slot][0] = '\0';
      answer_sections[slot][0] = '\0';
      answer_section_lens[slot] = 0;
      answer_lens[slot] = 0;
      answer_scores[slot] = 0.0;
    }
    value = corpus_iter_begin_length(corpus, &key, &key_size, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      /* The corpus deliberately holds keys of two sizes, so the size must come
       * from the iterator; a hardcoded one silently misses every entry. */
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, key_size, NULL)
        : sizeof(*entry);
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
              answer_section_lens[slot] = corpus_entry_section_copy(entry,
                entry_size, answer_sections[slot], ENTRY_SECTION_MAX);
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
            memcpy(answer_sections[insert], answer_sections[insert - 1],
              (size_t)answer_section_lens[insert - 1] + 1);
            answer_section_lens[insert] = answer_section_lens[insert - 1];
            answer_lens[insert] = answer_lens[insert - 1];
            answer_scores[insert] = answer_scores[insert - 1];
            --insert;
          }
          memcpy(answers[insert], candidate, (size_t)candidate_len + 1);
          answer_section_lens[insert] = corpus_entry_section_copy(entry,
            entry_size, answer_sections[insert], ENTRY_SECTION_MAX);
          answer_lens[insert] = candidate_len;
          answer_scores[insert] = candidate_score;
          ++count;
        }
      }
      value = corpus_iter_next_length(corpus, &key, &key_size, &cursor);
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
        /* The query named a section and every contributor had to match it, so
         * these agree in the ordinary case; naming them all is still what makes
         * the aggregate say where it came from. */
        for (item = 0; item < count; ++item) {
          answer_fact_section_add(answer_sections[item],
            answer_section_lens[item]);
        }
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
  int word_len;
  answer_relation_match_t relation_match;
  if (QUERY_WHO != query_type || NULL == entry) return 0.0;
  word_len = answer_query_be_word(query_text, query_len, word,
    (int)sizeof(word));
  if (word_len <= 0) return 0.0;
  if (0 == answer_word_is_name(word, word_len)
    && 0 != answer_relation_match_query(query_text,
    query_len, query_type, entry, &relation_match))
  {
    result = relation_match.score;
  }
  if (0 == text_contains_word_ci(entry->text, entry->text_len, word)) {
    return result;
  }
  if (0 != answer_word_is_name(word, word_len)) {
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
    { int made_len = 0;
      const char* made_term = answer_relation_rule_first_term(
        RELATION_RULE_RESULT, &made_len);
      if (NULL != made_term && 0 != text_contains_word_ci(entry->text,
        entry->text_len, made_term))
      {
        result += 0.35;
      }
    }
    if (0 != (entry->lexical_flags & ENTRY_LEX_ENTITY)) result += 0.15;
  }
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


static libxs_predict_t* answer_predict_build(const libxs_registry_t* corpus,
  const libxs_lexeme_stream_t* query, libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, int query_type,
  const answer_predict_profile_t* profile)
{
  libxs_predict_t* result = NULL;
  libxs_predict_t* model = NULL;
  const void* key = NULL;
  size_t cursor = 0, key_size = 0;
  void* value;
  int ntrain = 0;
  if (NULL == corpus || NULL == query || NULL == lexicon) return NULL;
  model = answer_predict_create(profile);
  if (NULL == model) return NULL;
  value = corpus_iter_begin_length(corpus, &key, &key_size, &cursor);
  while (NULL != value) {
    const corpus_entry_t* entry = (const corpus_entry_t*)value;
    size_t entry_size = (NULL != key)
      ? libxs_registry_value_size(corpus, key, key_size, NULL) : sizeof(*entry);
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
    value = corpus_iter_next_length(corpus, &key, &key_size, &cursor);
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
          term_len, 1);
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
  int query_terms = 0, term_matches = 0, exact_matches = 0;
  int entry_words;
  double bridge_score;
  const answer_bridge_t* bridge;
  double overlap;
  libxs_lexeme_stream_t entry_stream;
  size_t query_pos;
  libxs_lexeme_stream_init(&entry_stream);
  if (NULL == query || NULL == lexicon || NULL == entry) return 0.0;
  if (entry_size >= CORPUS_ENTRY_META_SIZE && entry->ntokens > 0) {
    use_sketch = 1;
  }
  else if (EXIT_SUCCESS != libxs_lexeme_stream_encode(lexicon, &entry_stream,
      (const unsigned char*)entry->text, (size_t)entry->text_len,
      rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
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
      int matched, exact;
      total += weight;
      ++query_terms;
      if (0 != long_term) ++long_terms;
      matched = ((0 != use_sketch && 0 != entry_sketch_has_id(entry,
          lexeme->id))
        || (0 == use_sketch && 0 != lexeme_stream_has_id(&entry_stream,
          lexeme->id))) ? 1 : 0;
      exact = matched;
      if (0 == matched && lexeme->length >= 5) {
        int term_len = 0;
        const char* term = libxs_lexicon_text(lexicon, lexeme->id,
          &term_len, NULL);
        if (0 == entry_stream_ready
          && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon,
            &entry_stream, (const unsigned char*)entry->text,
            (size_t)entry->text_len, rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
        {
          entry_stream_ready = 1;
        }
        if (0 != entry_stream_ready
          && 0 != lexeme_stream_has_similar_text(&entry_stream, lexicon,
            term, term_len, 1))
        {
          matched = 1;
        }
      }
      if (0 != matched) {
        score += weight;
        ++matches;
        ++term_matches;
        if (0 != exact) ++exact_matches;
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
  if (QUERY_GENERIC != query_type && term_matches > 0 && 0 == exact_matches) {
    return 0.0;
  }
  if (QUERY_GENERIC != query_type) min_overlap = 0.40;
  overlap = score / total;
  if (bridge_score > 0.0 && overlap < bridge_score) overlap = bridge_score;
  /**
   * A question with a single content term ("what is the X") scores a perfect
   * overlap as soon as that one term matches, so the ratio cannot distinguish
   * a real hit from a passage that merely mentions something spelled like X.
   * Require the lone term to match EXACTLY: an approximate match on the only
   * term the query carries is how an entity absent from the corpus gets a
   * confident reply.
   */
  if (query_terms <= 1 && term_matches > 0 && 0 == exact_matches) {
    return 0.0;
  }
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


/**
 * Which slot already holds this text, or -1.
 *
 * Ingest keys an entry on its text AND its section, and stores the same span at
 * more than one scale, so one sentence can exist as several entries: a stock
 * phrase recurs across sources, and every sentence short enough to be a
 * paragraph of its own is stored twice. The answer list is what the reader sees,
 * so the same sentence must not occupy two of its slots - which is what returned
 * one reply twice at `-n 2`. Rejected here rather than at print time, so the
 * reply path, the evaluation and the recomb host all see one answer list.
 */
/**
 * Has this exact text already been printed in this reply? Records it if not.
 *
 * An answer list free of repeated entries is still not a reply free of repeated
 * sentences: evidence extraction pulls the matching sentence out of a paragraph
 * as readily as out of the sentence stored beside it, so two different entries
 * render one sentence. The reader sees printed text, so printed text is what is
 * compared.
 */
static int answer_shown_repeat(char shown[][COMPOSE_MAXTEXT], int* nshown,
  const char* text, int text_len)
{
  int result = 0;
  int pos;
  for (pos = 0; pos < *nshown && 0 == result; ++pos) {
    if ((int)strlen(shown[pos]) == text_len
      && 0 == libxs_memcmp(shown[pos], text, (size_t)text_len))
    {
      result = 1;
    }
  }
  if (0 == result && *nshown < ANSWER_MAX && text_len < COMPOSE_MAXTEXT) {
    memcpy(shown[*nshown], text, (size_t)text_len);
    shown[*nshown][text_len] = '\0';
    ++(*nshown);
  }
  return result;
}


static int answer_slot_with_text(const corpus_entry_t* const entries[],
  int limit, const corpus_entry_t* entry)
{
  int result = -1;
  int slot;
  for (slot = 0; slot < limit && 0 > result; ++slot) {
    if (NULL != entries[slot] && entries[slot]->text_len == entry->text_len
      && 0 == libxs_memcmp(entries[slot]->text, entry->text,
        (size_t)entry->text_len))
    {
      result = slot;
    }
  }
  return result;
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
  size_t cursor = 0, key_size = 0;
  void* value;
  int answer_count = 0;
  int slot, held;
  int query_type = QUERY_GENERIC;
  int result = 0;
  int limit = budget;
  const int graph_asked = answer_graph_asked(query_text, query_len);
  /**
   * Evidence selection scores term overlap, which is blind to polarity: a
   * negated question overlaps the affirmative sentence on every content word
   * and would rank it first. Abstain instead - the corpus states positives,
   * so a complement is not answerable from it by selection either.
   *
   * A GRAPH QUESTION is refused here for a related reason: a sentence about one
   * entity is not how two relate, and when the graph CAN answer, the fact chain wins
   * and this is never read. Refusing here is also what makes the abstention hold on
   * the eval's own probe path rather than only in the interactive one.
   */
  int negated = answer_query_is_negated(query_text, query_len);
  const libxs_predict_t* predictor = answer_model;
  libxs_predict_t* query_predictor = NULL;
  char query_section[ENTRY_SECTION_MAX];
  int query_section_len = answer_query_section(query_text, query_len,
    query_section, (int)sizeof(query_section));
  const char* rank_query_text = query_text;
  size_t rank_query_len = query_len;
  char query_be_word[64];
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
  if (0 == negated && 0 == graph_asked && NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)rank_query_text, rank_query_len, rules, nrules,
      converse_lexnorms(), converse_lexnorms_size(), 1))
  {
    query_type = query_type_of(&query, lexicon);
    query_be_len = answer_query_be_word(rank_query_text, rank_query_len,
      query_be_word, (int)sizeof(query_be_word));
    if (NULL == predictor) {
      query_predictor = answer_predict_build(corpus, &query, lexicon,
        rules, nrules, query_type, profile);
      predictor = query_predictor;
    }
    value = corpus_iter_begin_length(corpus, &key, &key_size, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = (const corpus_entry_t*)value;
      size_t entry_size = (NULL != key)
        ? libxs_registry_value_size(corpus, key, key_size, NULL) : sizeof(*entry);
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
        value = corpus_iter_next_length(corpus, &key, &key_size, &cursor);
        continue;
      }
      if (0 != query_type_prefers_sentence(query_type)
        && SCALE_SENTENCE == entry->scale
        && (0 == text_starts_sentence(entry->text, entry->text_len)
          || 0 == text_ends_sentence(entry->text, entry->text_len)))
      {
        value = corpus_iter_next(corpus, &key, &cursor);
        continue;
      }
      if (QUERY_WHO == query_type && query_be_len > 0
        && 0 == answer_word_is_name(query_be_word, query_be_len)
        && 0 == answer_relation_match_query(rank_query_text, rank_query_len,
          query_type, entry, &relation_match))
      {
        value = corpus_iter_next(corpus, &key, &cursor);
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
        held = answer_slot_with_text(entries, limit, entry);
        if (0 <= held) {
          /* Keep the better-scoring occurrence: the score carries the scale
           * preference, so this is what picks a sentence over the fragment cut
           * out of it. */
          if (score > scores[held]) {
            entries[held] = entry;
            scores[held] = score;
            while (0 < held && scores[held] > scores[held - 1]) {
              const corpus_entry_t* swap_entry = entries[held - 1];
              const double swap_score = scores[held - 1];
              entries[held - 1] = entries[held];
              scores[held - 1] = scores[held];
              entries[held] = swap_entry;
              scores[held] = swap_score;
              --held;
            }
          }
        }
        else for (slot = 0; slot < limit; ++slot) {
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
      value = corpus_iter_next(corpus, &key, &cursor);
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


/**
 * Drop the heading a stored entry opens with: ingest stores the first sentence of
 * a section from its heading onward, so the heading is part of that entry's text
 * and would otherwise open the reply.
 *
 * corpus_title_len decides what a heading is. The scan that used to stand here
 * had its own answer and got it wrong in both directions: a one-word title was
 * kept, while the initial capital of the sentence beneath it counted as the
 * title's second word and was eaten, so the reply began mid-word.
 */
static void answer_strip_heading_prefix(const char** text, int* text_len)
{
  if (NULL != text && NULL != *text && NULL != text_len && 0 < *text_len) {
    const int title_len = corpus_title_len(*text, *text_len);
    if (0 < title_len) {
      const char* next = *text + title_len;
      int remaining = *text_len - title_len;
      while (0 < remaining && 0 != isspace((unsigned char)*next)) {
        ++next;
        --remaining;
      }
      if (0 < remaining) {
        *text = next;
        *text_len = remaining;
      }
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
      rules, nrules, converse_lexnorms(), converse_lexnorms_size(), 1))
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
      converse_lexnorms(), converse_lexnorms_size(), 1))
  {
    query_type = query_type_of(&query, lexicon);
    bridge = answer_bridge_match(&query, lexicon, entry);
    be_len = answer_query_be_word(query_text, query_len, be_word,
      (int)sizeof(be_word));
  }
  if (NULL != bridge && NULL != bridge->reply) {
    result = answer_bridge_expand_reply(bridge, text, text_len, lexicon,
      rules, nrules, output, output_size);
  }
  else if (QUERY_WHO == query_type && be_len > 0
    && 0 == answer_word_is_name(be_word, be_len)
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
    converse_lexnorms(), converse_lexnorms_size(), 1))
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

/**
 * Non-zero if the query carries a negator, per the caller-owned `negate|`
 * rules. The extractors answer affirmative questions only: they find what the
 * corpus asserts, never what it denies, and a complement ("who is NOT the
 * <role>") is not groundable from evidence that states only positives. The
 * vocabulary stays in the rule file - no negation words in the source - and
 * with no rules loaded the check is inert, exactly like the other rule kinds.
 */
static int answer_query_is_negated(const char* query_text, size_t query_len)
{
  int result = 0;
  size_t rule_pos;
  libxs_lexeme_stream_t query;
  libxs_lexeme_stream_init(&query);
  /**
   * RULE ORDER: normalization first, interpretation second. The negators are
   * matched against the ENCODED query, so any `norm|` rewriting has already
   * been applied and the two rule kinds compose in a defined order instead of
   * inspecting different representations of the same text. A consequence worth
   * stating: a `norm|` rule must not map a negator onto its affirmative form,
   * or it would erase the polarity this check depends on.
   */
  if (NULL != answer_query_lexicon && answer_query_nrules > 0
    && NULL != query_text && query_len > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(answer_query_lexicon,
      &query, (const unsigned char*)query_text, query_len,
      answer_query_rules, answer_query_nrules,
      converse_lexnorms(), converse_lexnorms_size(), 0))
  {
    for (rule_pos = 0; rule_pos < converse_rules_size() && 0 == result;
      ++rule_pos)
    {
      const answer_relation_rule_t* rule = converse_rules() + rule_pos;
      if (RELATION_RULE_NEGATE == rule->kind
        && 0 != lexeme_stream_has_text(&query, answer_query_lexicon,
          rule->term))
      {
        result = 1;
      }
    }
  }
  else { /* no lexicon available yet: fall back to raw-text matching */
    result = answer_relation_rule_has_term(RELATION_RULE_NEGATE, query_text,
      (int)query_len);
  }
  libxs_lexeme_stream_release(&query);
  return result;
}


/**
 * Ask each fact resolver in turn and keep the BEST-GROUNDED answer, not the
 * first one.
 *
 * This used to be a short-circuit chain, so dispatch order alone decided which
 * resolver spoke. That is fine while every resolver rests on asserted knowledge
 * and wrong as soon as one does not: a resolver holding only a LEARNED binding
 * pre-empted a later one that could have answered from an ASSERTED term, and no
 * amount of ordering WITHIN a resolver can reach that - the competition is
 * between resolvers. Provenance orders them for the same reason it orders the
 * facts inside one, and by the same total order over three known values.
 *
 * With no rules learned every answer is ASSERTED, so the first success wins and
 * breaks the loop: the historical behaviour and the historical cost, exactly.
 */
/**
 * Is this a question about the GRAPH? A DECLARED link term says so, and nothing else
 * does - not the endpoints resolving to census names, and not the failure of every
 * other resolver. Consulted by the fact chain and by evidence selection alike, so the
 * invariant "a graph question is answered by a path or not at all" cannot hold on one
 * path and lapse on another. Two definitions of one rule is the defect E16 STEP 3
 * found in the section table; this is the same rule asked once.
 */
static int answer_graph_asked(const char* query_text, size_t query_len)
{
  return (NULL != query_text && 0 < query_len
    && 0 != answer_relation_rule_has_term(RELATION_RULE_LINK, query_text,
      (int)query_len)) ? 1 : 0;
}


/**
 * Does the RULE FILE recognize this as a question it knows how to ask?
 *
 * A declared `ask|` word is the whole test, so this reports a fact about the RULES and
 * never about the corpus - which is exactly the distinction a reply has to draw. Two
 * failures were being rendered identically and only one of them is the project's
 * result:
 *
 *   understood, unsupported  "Who was eaten by the elephant?"  -> abstain, as always;
 *                            the shape is read perfectly and the corpus has no elephant
 *   NOT understood           "Hansel Gretel forest"            -> say so, then offer
 *                            what the corpus does hold
 *
 * Abstaining on the second is silence where something useful exists; answering it
 * unlabelled is the confident non-answer the abstention discipline exists to prevent.
 * Saying which of the two happened is neither, and it costs no new mechanism: the
 * signal is the one the classifier already computes.
 */
static int answer_query_recognized(const char* query_text, size_t query_len)
{
  int result = 0;
  int index = 0;
  const char* term;
  const char* tag;
  if (NULL == query_text || 0 == query_len) return 0;
  while (0 == result
    && NULL != (term = answer_relation_rule_term_at(RELATION_RULE_ASK, index,
      &tag)))
  {
    if (0 != text_contains_word_ci(query_text, (int)query_len, term)) {
      result = 1;
    }
    ++index;
  }
  return result;
}


static int answer_fact_reply(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, char* output, size_t output_size)
{
  const int graph_asked = answer_graph_asked(query_text, query_len);
  int result = EXIT_FAILURE;
  int best = RELATION_RULE_PROPOSED + 1;
  char best_output[COMPOSE_MAXTEXT];
  char best_section[sizeof(answer_fact_section)];
  char best_learned[sizeof(answer_fact_learned)];
  answer_origin_t best_origins[ANSWER_ORIGIN_MAX];
  int best_norigins = 0;
  int best_section_len = 0;
  int best_learned_len = 0;
  int best_learned_from = RELATION_RULE_ASSERTED;
  int step;
  best_output[0] = '\0';
  best_section[0] = '\0';
  best_learned[0] = '\0';
  if (0 == answer_query_is_negated(query_text, query_len)) {
    /* A graph question may be answered by the GRAPH and by nothing else, so the chain
     * stops after step 0. Without this the same question in two phrasings behaved
     * differently: "How are Hansel and Gretel connected?" reached the identity layer,
     * which answered "Hansel is the boy." because that phrasing puts a copula next to
     * a name, while "What connects Hansel and Gretel?" abstained. */
    const int steps = (0 != graph_asked) ? 1 : 11;
    for (step = 0; step < steps && RELATION_RULE_ASSERTED < best; ++step) {
      int ok;
      answer_fact_section_set(NULL, 0);
      answer_fact_learned_set(NULL, 0, RELATION_RULE_ASSERTED);
      output[0] = '\0';
      switch (step) {
        /* The graph question is asked first because it is the only one that reads a
         * PAIR of entities, so no other resolver can be answering it. */
        case 0: ok = answer_link_reply(query_text, query_len, output,
          output_size); break;
        case 1: ok = answer_relation_fact_reply(query_text, query_len,
          output, output_size); break;
        case 2: ok = answer_relation_aggregate_reply(corpus, query_text,
          query_len, output, output_size); break;
        case 3: ok = answer_type_kin_reply(query_text, query_len,
          output, output_size); break;
        case 4: ok = answer_identity_fact_reply(query_text, query_len,
          output, output_size); break;
        case 5: ok = answer_describe_fact_reply(query_text, query_len,
          output, output_size); break;
        case 6: ok = answer_location_fact_reply(query_text, query_len,
          output, output_size); break;
        case 7: ok = answer_type_fact_reply(query_text, query_len,
          output, output_size); break;
        case 8: ok = answer_own_fact_reply(query_text, query_len,
          output, output_size); break;
        case 9: ok = answer_topic_reply(query_text, query_len,
          output, output_size); break;
        default: ok = answer_docdef_fact_reply(query_text, query_len,
          output, output_size); break;
      }
      if (EXIT_SUCCESS == ok) {
        /* A reply names the term it rests on only when that term was not
         * asserted, so an empty label IS the asserted case. */
        const int prov = (0 < answer_fact_learned_len)
          ? answer_fact_learned_from : RELATION_RULE_ASSERTED;
        if (prov < best) {
          size_t len = strlen(output);
          if (len >= sizeof(best_output)) len = sizeof(best_output) - 1;
          memcpy(best_output, output, len);
          best_output[len] = '\0';
          memcpy(best_section, answer_fact_section,
            (size_t)answer_fact_section_len + 1);
          best_section_len = answer_fact_section_len;
          memcpy(best_learned, answer_fact_learned,
            (size_t)answer_fact_learned_len + 1);
          best_learned_len = answer_fact_learned_len;
          best_learned_from = answer_fact_learned_from;
          /* The winning resolver's ORIGINS travel with its section, or the reset
           * at the top of the next step discards what it found. */
          memcpy(best_origins, answer_origins, sizeof(best_origins));
          best_norigins = answer_norigins;
          best = prov;
        }
      }
    }
  }
  if (RELATION_RULE_PROPOSED >= best) {
    size_t len = strlen(best_output);
    if (len >= output_size) len = output_size - 1;
    memcpy(output, best_output, len);
    output[len] = '\0';
    answer_fact_section_set(best_section, best_section_len);
    memcpy(answer_origins, best_origins, sizeof(answer_origins));
    answer_norigins = best_norigins;
    answer_fact_learned_set(best_learned, best_learned_len, best_learned_from);
    result = EXIT_SUCCESS;
  }
  else {
    answer_fact_section_set(NULL, 0);
    answer_fact_learned_set(NULL, 0, RELATION_RULE_ASSERTED);
  }
  return result;
}

/**
 * Print the section an answer came from, when the corpus supplies one. Sections
 * are the story titles or Markdown headings recorded per entry at ingest, so a
 * citation is only emitted when the corpus actually carries that structure -
 * never invented, and silently omitted for flat text.
 */
/**
 * Say when a reply rests on a class term that was not asserted. This is the
 * whole point of admitting the learned bands: the reply is offered, and the
 * reader is told which word it hangs on, so one that should not be in the class
 * reads as a guess rather than as an assertion.
 */
static void answer_print_learned(void)
{
  if (0 < answer_fact_learned_len) {
    printf("%s: rests on the %s class term \"%s\"\n",
      (RELATION_RULE_PROPOSED == answer_fact_learned_from)
        ? "speculative" : "learned",
      (RELATION_RULE_PROPOSED == answer_fact_learned_from)
        ? "proposed" : "learned", answer_fact_learned);
  }
}


/**
 * Bytes of a section that form the citation: ONE line.
 *
 * A section captured before the heading map was built could carry the lines
 * that followed the heading, and printing it whole named two sources at once.
 * Stopping at the first line break recovers the intended title. The map makes
 * that unreachable for a freshly ingested corpus, but a persisted one predating
 * it still has such sections stored.
 *
 * Shared with the evaluation so a checked citation is the SAME text the reader is
 * shown, rather than the raw section a fixture would then have to describe.
 */
/**
 * How much of a section may be printed as a citation, and zero when it may not.
 *
 * A citation names a TITLE, and prose that the heading rule read as one is not a
 * title: a corpus whose extract dropped its titles offers lead sentences instead,
 * and crediting an answer to "For the region in northwest Iran, see ..." says
 * nothing about where it came from. Two marks refuse those without reading words -
 * a comma, and a clause-ending mark at the end - and the cost of refusing is now
 * bearable, because the FILE and LINE are cited whether a title exists or not. A
 * title that ends in a period loses its half of the citation and keeps the half
 * that cannot be wrong.
 */
static int answer_citation_len(const char* section, int section_len)
{
  int result = 0;
  if (NULL != section) {
    int at;
    while (result < section_len && '\n' != section[result]
      && '\r' != section[result] && '\0' != section[result]) ++result;
    for (at = 0; at < result; ++at) {
      if (',' == section[at]) {
        result = 0;
        break;
      }
    }
    if (0 < result) {
      const char last = section[result - 1];
      if ('.' == last || '!' == last || '?' == last || ':' == last) result = 0;
    }
  }
  return result;
}


/**
 * Print the citation: the titles when the corpus has any, and always the files and
 * lines. One range per file, and a single line rather than a range when the reply
 * rests on one line of it - "grimm.txt:2104-2110" against "grimm.txt:2104".
 */
/**
 * Format the citation: the title when the corpus has one, and always the files and
 * lines. One range per file, and a single line rather than a range when the reply
 * rests on one line of it - "grimm.txt:2104-2110" against "grimm.txt:2104".
 *
 * One formatter, because the EVALUATION has to check the same string a reader sees.
 * Checking the title alone left the file and line ungated, which is the half that
 * cannot be wrong and therefore the half worth asserting in a fixture.
 */
static int answer_citation_text(const char* section, int section_len,
  char* output, size_t output_size)
{
  int result = 0;
  const int len = answer_citation_len(section, section_len);
  if (NULL != output && 0 < output_size) {
    size_t pos = 0;
    int at, nfiles = 0;
    output[0] = '\0';
    if (0 < len && (size_t)len + 1 < output_size) {
      memcpy(output, section, (size_t)len);
      pos = (size_t)len;
      output[pos] = '\0';
    }
    for (at = 0; at < answer_norigins; ++at) {
      const char* path = corpus_source_path(answer_origins[at].source);
      char range[ENTRY_SECTION_MAX * 8];
      size_t range_len;
      if (NULL == path) continue;
      if (answer_origins[at].first != answer_origins[at].last) {
        sprintf(range, "%s%s:%u-%u", (0 < nfiles) ? "; " : "", path,
          answer_origins[at].first, answer_origins[at].last);
      }
      else {
        sprintf(range, "%s%s:%u", (0 < nfiles) ? "; " : "", path,
          answer_origins[at].first);
      }
      range_len = strlen(range);
      if (0 < pos && 0 == nfiles) {
        if (pos + 2 >= output_size) break;
        memcpy(output + pos, " (", 2);
        pos += 2;
      }
      if (pos + range_len + 2 >= output_size) break;
      memcpy(output + pos, range, range_len);
      pos += range_len;
      output[pos] = '\0';
      ++nfiles;
    }
    if (0 < len && 0 < nfiles && pos + 2 < output_size) {
      output[pos++] = ')';
      output[pos] = '\0';
    }
    result = (int)pos;
  }
  return result;
}


static void answer_print_citation(const char* section, int section_len)
{
  char text[COMPOSE_MAXTEXT];
  if (0 < answer_citation_text(section, section_len, text, sizeof(text))) {
    printf("citation: %s\n", text);
  }
}


static void answer_fact_learned_set(const char* term, int term_len,
  int provenance)
{
  answer_fact_learned_len = 0;
  answer_fact_learned[0] = '\0';
  answer_fact_learned_from = RELATION_RULE_ASSERTED;
  if (NULL != term && term_len > 0 && RELATION_RULE_ASSERTED != provenance
    && term_len < (int)sizeof(answer_fact_learned))
  {
    memcpy(answer_fact_learned, term, (size_t)term_len);
    answer_fact_learned[term_len] = '\0';
    answer_fact_learned_len = term_len;
    answer_fact_learned_from = provenance;
  }
}


/** Record one item's origin, widening the range of a source already named. */
static void answer_origin_add(unsigned int source, unsigned int line)
{
  if (0 != source && 0 != line) {
    int at;
    for (at = 0; at < answer_norigins; ++at) {
      if (answer_origins[at].source == source) {
        if (line < answer_origins[at].first) answer_origins[at].first = line;
        if (line > answer_origins[at].last) answer_origins[at].last = line;
        return;
      }
    }
    if (answer_norigins < ANSWER_ORIGIN_MAX) {
      answer_origins[answer_norigins].source = source;
      answer_origins[answer_norigins].first = line;
      answer_origins[answer_norigins].last = line;
      ++answer_norigins;
    }
  }
}


static void answer_fact_section_set(const char* section, int section_len)
{
  answer_fact_section_len = 0;
  answer_fact_section[0] = '\0';
  answer_norigins = 0;
  if (NULL != section && section_len > 0
    && section_len < (int)sizeof(answer_fact_section))
  {
    memcpy(answer_fact_section, section, (size_t)section_len);
    answer_fact_section[section_len] = '\0';
    answer_fact_section_len = section_len;
  }
}


/**
 * Name one more source this reply rests on. Already-named and oversized sources
 * are dropped rather than truncated: half a title is a citation to nothing.
 */
static void answer_fact_section_add(const char* section, int section_len)
{
  static const char joiner[] = "; ";
  if (NULL != section && 0 < section_len
    && NULL == libxs_strimem(answer_fact_section,
      (size_t)answer_fact_section_len, section, (size_t)section_len))
  {
    const int join_len = (0 < answer_fact_section_len)
      ? (int)(sizeof(joiner) - 1) : 0;
    if (answer_fact_section_len + join_len + section_len
      < (int)sizeof(answer_fact_section))
    {
      memcpy(answer_fact_section + answer_fact_section_len, joiner,
        (size_t)join_len);
      answer_fact_section_len += join_len;
      memcpy(answer_fact_section + answer_fact_section_len, section,
        (size_t)section_len);
      answer_fact_section_len += section_len;
      answer_fact_section[answer_fact_section_len] = '\0';
    }
  }
}

/**
 * Reorder the selected candidates by the hierarchical model's conditional code
 * length, -log2 P(candidate | query), normalized per byte so a long sentence is
 * not penalized for its length. Selection stays in charge of WHICH sentences are
 * admitted - this only reorders them - so the abstention discipline is
 * untouched: a candidate promoted here was already attested and already cleared
 * the selection threshold. Returns the (unchanged) candidate count.
 *
 * The reorder is opt-in and off by default: it is a measurement of whether a
 * better-calibrated likelihood carries downstream, and the QA harness is the
 * gate. Without a built model, or when a candidate cannot be scored, the
 * selection order is preserved exactly.
 */
static int answer_hier_reorder(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[ANSWER_MAX], double scores[ANSWER_MAX],
  int answer_count)
{
  if (0 != converse_judge_active() && 1 < answer_count) {
    const char* candidates[ANSWER_MAX];
    int lengths[ANSWER_MAX];
    double bits[ANSWER_MAX];
    int slot, nvalid = 0;
    for (slot = 0; slot < answer_count; ++slot) {
      candidates[slot] = (NULL != entries[slot]) ? entries[slot]->text : NULL;
      lengths[slot] = (NULL != entries[slot]) ? entries[slot]->text_len : 0;
    }
    if (EXIT_SUCCESS == converse_judge_rescore(query_text, (int)query_len,
      candidates, lengths, answer_count, bits))
    {
      for (slot = 0; slot < answer_count; ++slot) {
        if (0 < lengths[slot]) ++nvalid;
      }
    }
    /**
     * Only reorder when EVERY candidate scored: a partial ranking would mix two
     * incomparable orderings, and preferring whichever subset the model happened
     * to handle is not a measurement of the model.
     */
    if (nvalid == answer_count) {
      const corpus_entry_t* was_first = entries[0];
      ++answer_hier_nreorder;
      for (slot = 1; slot < answer_count; ++slot) {
        const corpus_entry_t* entry = entries[slot];
        const double bpc = bits[slot];
        const double score = scores[slot];
        int probe = slot;
        while (0 < probe && bpc > bits[probe - 1]) {
          entries[probe] = entries[probe - 1];
          scores[probe] = scores[probe - 1];
          bits[probe] = bits[probe - 1];
          --probe;
        }
        entries[probe] = entry;
        scores[probe] = score;
        bits[probe] = bpc;
      }
      if (was_first != entries[0]) {
        ++answer_hier_nchanged;
        if (1 < converse_judge_verbose()) {
          fprintf(stderr, "  rescore[%.*s]\n    was: [%d B] %.*s\n"
            "    now: [%d B] %.*s (%.3f bpc)\n", (int)query_len, query_text,
            was_first->text_len,
            (was_first->text_len < 90) ? was_first->text_len : 90,
            was_first->text, entries[0]->text_len,
            (entries[0]->text_len < 90) ? entries[0]->text_len : 90,
            entries[0]->text, bits[0]);
        }
      }
    }
  }
  return answer_count;
}


static size_t answer_visible_append(char* output, size_t output_size,
  size_t output_pos, const char* text, int text_len)
{
  size_t result = output_pos;
  if (NULL != output && 0 < text_len
    && output_pos + (size_t)text_len + 2 < output_size)
  {
    if (0 < output_pos) output[result++] = '\n';
    memcpy(output + result, text, (size_t)text_len);
    result += (size_t)text_len;
    output[result] = '\0';
  }
  return result;
}


/**
 * Render an answer list the way a reader sees it; print it unless asked not to.
 *
 * ONE renderer, because the evaluation used to check `answer_reply` - which
 * FAILS for most retrieved answers, those being shown as evidence instead - so
 * everything the reply path does after that was ungated, and a reply that began
 * mid-word passed. What the reader is shown and what the fixture scores are now
 * the same bytes by construction.
 *
 * Returns 2 when a composed reply answered, 1 when evidence was shown, 0 when
 * nothing could be rendered; the caller's out-parameter carries only a composed
 * reply, which is what it has always carried.
 */
static int answer_render(const char* query_text, size_t query_len,
  const corpus_entry_t* entries[], int answer_count,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int print, char* output, size_t output_size)
{
  int result = 0;
  int slot;
  size_t pos = 0;
  char reply[COMPOSE_MAXTEXT];
  char shown[ANSWER_MAX][COMPOSE_MAXTEXT];
  int nshown = 0;
  if (NULL != output && 0 < output_size) output[0] = '\0';
  if (0 < answer_count && NULL != entries[0]
    && EXIT_SUCCESS == answer_reply(query_text, query_len, entries[0],
      lexicon, rules, nrules, reply, sizeof(reply)))
  {
    pos = answer_visible_append(output, output_size, pos, reply,
      (int)strlen(reply));
    /* Ranked evidence cites the entry it came from, so the origin is the entry's
     * own rather than a fact's - registered whether or not this run prints, since
     * the evaluation reads the same citation without printing it. */
    answer_fact_section_set(entries[0]->section, entries[0]->section_len);
    answer_origin_add(entries[0]->source, entries[0]->line);
    if (0 != print) {
      printf("%s\n", reply);
      answer_print_citation(entries[0]->section, entries[0]->section_len);
    }
    result = 2;
  }
  for (slot = 0; 0 == result && slot < answer_count
    && NULL != entries[slot]; ++slot)
  {
    const char* text = entries[slot]->text;
    int text_len = entries[slot]->text_len;
    if (EXIT_SUCCESS == answer_evidence_sentence(query_text, query_len,
      entries[slot], lexicon, rules, nrules, reply, sizeof(reply)))
    {
      if (0 == answer_shown_repeat(shown, &nshown, reply,
        (int)strlen(reply)))
      {
        pos = answer_visible_append(output, output_size, pos, reply,
          (int)strlen(reply));
        if (0 != print) {
          if (nshown > 1) printf("\n");
          printf("%s\n", reply);
          answer_fact_section_set(entries[slot]->section,
            entries[slot]->section_len);
          answer_origin_add(entries[slot]->source, entries[slot]->line);
          answer_print_citation(entries[slot]->section,
            entries[slot]->section_len);
        }
      }
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
          && 0 != text_ends_sentence(text, text_len)))
      && 0 == answer_shown_repeat(shown, &nshown, text, text_len))
    {
      pos = answer_visible_append(output, output_size, pos, text, text_len);
      if (0 != print) {
        if (nshown > 1) printf("\n");
        printf("%.*s\n", text_len, text);
      }
    }
  }
  if (0 == result && 0 < nshown) result = 1;
  return result;
}


static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile,
  char* out_reply, size_t out_size)
{
  const corpus_entry_t* entries[ANSWER_MAX];
  double scores[ANSWER_MAX];
  int answer_count;
  int rendered;
  int fact_ok, fact_prov;
  char reply[COMPOSE_MAXTEXT];
  char fact_section[sizeof(answer_fact_section)];
  int fact_section_len;
  char visible[4 * COMPOSE_MAXTEXT];
  if (NULL != out_reply && out_size > 0) out_reply[0] = '\0';
  fact_ok = (EXIT_SUCCESS == answer_fact_reply(corpus, query_text, query_len,
    reply, sizeof(reply))) ? 1 : 0;
  /**
   * A GRAPH QUESTION IS ANSWERED BY A PATH OR NOT AT ALL.
   *
   * Asking how two entities relate is not answered by a sentence about one of them,
   * and the fall-through answered "How are Achilles and Lincoln connected?" with an
   * unrelated paragraph about Achilles - a confident non-answer, which is the failure
   * the abstention discipline exists to prevent.
   *
   * THE FIRST VERSION OF THIS GUARD WAS TOO WEAK: it asked only when no other layer
   * had spoken, so a graph question that some other layer could answer was never
   * tested, and the same question in two phrasings behaved differently - "What
   * connects Hansel and Gretel?" abstained while "How are Hansel and Gretel
   * connected?" replied "Hansel is the boy." (the identity layer answering a question
   * nobody asked, because that phrasing puts a copula next to a name). It also asked
   * whether BOTH endpoints resolve to census names, which let "What connects Hansel
   * and the witch?" fall through to a raw quotation.
   *
   * A DECLARED link term is what makes it a graph question - not the endpoints
   * resolving, and not the failure of everything else. So the test is the declared
   * term, and the only admissible answer is the path the graph states.
   */
  if (0 == fact_ok && 0 != answer_graph_asked(query_text, query_len)) return 0;
  fact_prov = (0 < answer_fact_learned_len)
    ? answer_fact_learned_from : RELATION_RULE_ASSERTED;
  memcpy(fact_section, answer_fact_section,
    (size_t)answer_fact_section_len + 1);
  fact_section_len = answer_fact_section_len;
  /**
   * A fact answer resting on ASSERTED knowledge is final, exactly as it always
   * was - so with no rules learned nothing below this line ever runs, and the
   * cost and the output are unchanged.
   */
  if (0 != fact_ok && RELATION_RULE_ASSERTED == fact_prov) {
    printf("%s\n", reply);
    answer_print_citation(fact_section, fact_section_len);
    answer_print_learned();
    if (NULL != out_reply && out_size > 0) {
      size_t rn = strlen(reply);
      if (rn >= out_size) rn = out_size - 1;
      memcpy(out_reply, reply, rn);
      out_reply[rn] = '\0';
    }
    return 1;
  }
  answer_count = answer_select(corpus, query_text, query_len, budget,
    lexicon, rules, nrules, answer_model, profile, entries, scores);
  answer_count = answer_hier_reorder(query_text, query_len, entries, scores,
    answer_count);
  /**
   * The FACT layer pre-empted the RANKED layer unconditionally, which is wrong
   * as soon as the fact rests on a learned term and the ranked answer does not:
   * the reader was handed a guess while an asserted answer went unasked for.
   * Rendered quietly first, because deciding needs the text - the class term a
   * reply rests on is read out of the reply, the same way its label is.
   */
  rendered = answer_render(query_text, query_len, entries, answer_count,
    lexicon, rules, nrules, 0, visible, sizeof(visible));
  if (0 != fact_ok && (0 == rendered
    || answer_relation_rule_provenance(RELATION_RULE_PERSON, visible,
        (int)strlen(visible)) >= fact_prov))
  {
    printf("%s\n", reply);
    answer_print_citation(fact_section, fact_section_len);
    answer_print_learned();
    if (NULL != out_reply && out_size > 0) {
      size_t rn = strlen(reply);
      if (rn >= out_size) rn = out_size - 1;
      memcpy(out_reply, reply, rn);
      out_reply[rn] = '\0';
    }
    return 1;
  }
  /* Ranked evidence is reached only when no proposition was found. If the rule file
   * also did not recognize the QUESTION, the reader is told that before being handed
   * a sentence, so a relevant-looking quotation is never mistaken for an answer. */
  if (0 < answer_count && 0 == answer_query_recognized(query_text, query_len)) {
    printf("I did not recognize the question. The closest the corpus comes:\n");
  }
  rendered = answer_render(query_text, query_len, entries, answer_count,
    lexicon, rules, nrules, 1, visible, sizeof(visible));
  if (2 == rendered && NULL != out_reply && out_size > 0) {
    size_t rn = strlen(visible);
    if (rn >= out_size) rn = out_size - 1;
    memcpy(out_reply, visible, rn);
    out_reply[rn] = '\0';
  }
  LIBXS_UNUSED(scores);
  return (answer_count > 0) ? 1 : 0;
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


/**
 * question|evidence-terms|reply-terms|fact-terms|citation-terms
 *
 * The last two are optional, so a three- or four-field line behaves exactly as
 * before. The citation field is what makes ATTRIBUTION testable: the fixture
 * states which source the answer must be credited to, and a reply that is right
 * about the world but wrong about where it came from now fails.
 */
static int eval_parse_line(char* line, char* fields[5])
{
  int result = EXIT_FAILURE;
  char* cursor;
  int field_pos;
  if (NULL != fields) {
    for (field_pos = 0; field_pos < 5; ++field_pos) fields[field_pos] = NULL;
  }
  if (NULL != line && NULL != fields) {
    cursor = eval_trim(line);
    if ('\0' != *cursor && '#' != *cursor) {
      /* The trailing segment belongs to the field it reached, so a line with
       * three, four or five fields fills exactly those and leaves the rest
       * unset. */
      for (field_pos = 0; field_pos < 4 && NULL != cursor; ++field_pos) {
        char* sep = strchr(cursor, '|');
        if (NULL != sep) {
          *sep = '\0';
          fields[field_pos] = eval_trim(cursor);
          cursor = sep + 1;
        }
        else {
          fields[field_pos] = eval_trim(cursor);
          cursor = NULL;
        }
      }
      if (NULL != cursor) fields[4] = eval_trim(cursor);
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
  int npass = 0, ntop = 0, nany = 0, nreply = 0, nfact = 0, ncite = 0;
  int ncases = 0;
  int have_facts = (0 != answer_relation_facts_size
    || 0 != answer_docdef_facts_size) ? 1 : 0;
  FILE* file;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL == corpus || NULL == lexicon || NULL == rules) return EXIT_FAILURE;
  file = fopen(converse_eval_path(), "r");
  if (NULL == file) {
    fprintf(stderr, "eval: no %s file found\n", converse_eval_path());
  }
  while (NULL != file) {
    char line[EVAL_LINE_MAX];
    char* fields[5];
    const corpus_entry_t* entries[ANSWER_MAX];
    double scores[ANSWER_MAX];
    int nanswers;
    int top_pass;
    int any_pass;
    int reply_pass;
    int fact_pass;
    int fact_checked;
    int have_fact;
    int fact_prov;
    char fact_section[sizeof(answer_fact_section)];
    int fact_section_len;
    int cite_pass;
    int cite_len;
    const char* cite = NULL;
    char cite_text[COMPOSE_MAXTEXT];
    int pass;
    char reply[COMPOSE_MAXTEXT];
    char visible[4 * COMPOSE_MAXTEXT];
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
      && 0 == eval_terms_empty(fields[3])
      && 0 != strcmp(fields[3], EVAL_RULE_GOVERNED)) ? 1 : 0;
    cite_len = 0;
    /* Unconditionally, because an interactive query tries the fact resolvers
     * FIRST: what the reader is shown is this reply whenever it succeeds, and
     * whether the fixture states fact-terms has nothing to do with it. */
    have_fact = (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
      reply, sizeof(reply))) ? 1 : 0;
    fact_prov = (0 < answer_fact_learned_len)
      ? answer_fact_learned_from : RELATION_RULE_ASSERTED;
    memcpy(fact_section, answer_fact_section,
      (size_t)answer_fact_section_len + 1);
    fact_section_len = answer_fact_section_len;
    conv_remember(qtext, qlen);
    nanswers = answer_select(corpus, qtext, qlen,
      ANSWER_MAX, lexicon, rules, nrules,
      answer_model, profile, entries, scores);
    nanswers = answer_hier_reorder(qtext, qlen, entries, scores, nanswers);
    top_pass = eval_terms_match_answers(entries, nanswers, fields[1], 1);
    any_pass = eval_terms_match_answers(entries, nanswers, fields[1], 0);
    reply_pass = 1;
    LIBXS_UNUSED(scores);
    /**
     * THE TEXT THE READER IS SHOWN, decided exactly as answer_query decides it:
     * answer_render is the renderer the interactive path prints with, called
     * with printing off, and a fact answer resting on a LEARNED term yields to a
     * better-grounded ranked one. Both the reply-terms and the FACT-terms are
     * scored against this, because a fixture states what the reader must be
     * told; which layer says it is an implementation detail.
     */
    answer_render(qtext, qlen, entries, nanswers, lexicon, rules, nrules, 0,
      visible, sizeof(visible));
    if (0 != have_fact && ('\0' == visible[0]
      || answer_relation_rule_provenance(RELATION_RULE_PERSON, visible,
          (int)strlen(visible)) >= fact_prov))
    {
      size_t rn = strlen(reply);
      if (rn >= sizeof(visible)) rn = sizeof(visible) - 1;
      memcpy(visible, reply, rn);
      visible[rn] = '\0';
      cite = cite_text;
      cite_len = answer_citation_text(fact_section, fact_section_len, cite_text,
        sizeof(cite_text));
    }
    if (0 != fact_checked) {
      fact_pass = (0 != have_fact) ? eval_terms_match_text(visible,
        (int)strlen(visible), fields[3]) : 0;
    }
    /* Before the fact-only and abstention branches, both of which return early:
     * a fact reply is exactly the kind of answer whose attribution matters. */
    if (0 == cite_len && 0 < nanswers && NULL != entries[0]) {
      cite = cite_text;
      cite_len = answer_citation_text(entries[0]->section,
        entries[0]->section_len, cite_text, sizeof(cite_text));
    }
    cite_pass = 1;
    if (NULL != fields[4] && 0 == eval_terms_empty(fields[4])) {
      cite_pass = (0 < cite_len)
        ? eval_terms_match_text(cite, cite_len, fields[4]) : 0;
      fprintf(stdout, "%s cite %s\n", (0 != cite_pass) ? "PASS" : "FAIL",
        fields[0]);
      if (0 != cite_pass) ++ncite;
      else if (0 < cite_len) {
        fprintf(stdout, "     cited \"%.*s\", expected \"%s\"\n", cite_len,
          cite, fields[4]);
      }
      else fprintf(stdout, "     cited nothing, expected \"%s\"\n", fields[4]);
    }
    if (0 != eval_terms_empty(fields[1])) {
      if (0 != fact_checked) {
        pass = (0 != fact_pass && 0 != cite_pass) ? 1 : 0;
        fprintf(stdout, "%s fact %s\n", (0 != fact_pass) ? "PASS" : "FAIL",
          fields[0]);
        if (0 != fact_pass) ++nfact;
      }
      else if (NULL != fields[3] && 0 == eval_terms_empty(fields[3])
        && 0 != strcmp(fields[3], EVAL_RULE_GOVERNED))
      {
        fprintf(stdout, "SKIP fact %s\n", fields[0]);
        --ncases;
        continue;
      }
      /**
       * A lone EVAL_RULE_GOVERNED marker in the fact field means the expected
       * abstention depends on loaded rules (negation, for instance, is a rule
       * kind): check it when rules are present, skip it when they are not.
       * This keeps one fixture valid in both configurations - the property
       * that proves no language vocabulary is compiled into the source -
       * without pretending a rule-driven capability works without rules.
       */
      else if (NULL != fields[3]
        && 0 == strcmp(fields[3], EVAL_RULE_GOVERNED)
        && 0 == converse_rules_size())
      {
        fprintf(stdout, "SKIP rules %s\n", fields[0]);
        --ncases;
        continue;
      }
      else {
        /**
         * An abstention case must abstain on EVERY path an interactive query
         * would take, not merely on evidence selection: the fact and
         * definition resolvers answer first and are the ones that can
         * fabricate, so a check that only looked at answer_select would pass
         * a question the engine actually answers.
         */
        char probe[COMPOSE_MAXTEXT];
        int answered = (0 != nanswers) ? 1 : 0;
        if (EXIT_SUCCESS == answer_fact_reply(corpus, qtext, qlen,
          probe, sizeof(probe)))
        {
          answered = 1;
        }
        pass = (0 == answered) ? 1 : 0;
        fprintf(stdout, "%s abstain %s\n", (0 != pass) ? "PASS" : "FAIL",
          fields[0]);
      }
      if (0 != pass) ++npass;
      continue;
    }
    if (0 == eval_terms_empty(fields[2])) {
      reply_pass = eval_terms_match_text(visible, (int)strlen(visible),
        fields[2]);
    }
    pass = (0 != any_pass && 0 != reply_pass && 0 != fact_pass
      && 0 != cite_pass) ? 1 : 0;
    fprintf(stdout, "%s top %s\n", (0 != top_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s any %s\n", (0 != any_pass) ? "PASS" : "FAIL",
      fields[0]);
    fprintf(stdout, "%s reply %s\n", (0 != reply_pass) ? "PASS" : "FAIL",
      fields[0]);
    /* Say what was replied, the way the citation check says what was cited: a
     * reply expectation is otherwise unwritable without rebuilding to look. */
    if (0 == reply_pass && 0 == eval_terms_empty(fields[2])) {
      fprintf(stdout, "     replied \"%s\", expected \"%s\"\n", visible,
        fields[2]);
    }
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
    "eval[%s]: %d/%d passed (top=%d, any=%d, reply=%d, fact=%d, cite=%d)\n",
    profile->name, npass, ncases, ntop, nany, nreply, nfact, ncite);
  if (0 != converse_judge_active()) {
    fprintf(stderr, "  hier rescore: %ld rankings, top-1 changed on %ld\n",
      answer_hier_nreorder, answer_hier_nchanged);
  }
  if (NULL != file) fclose(file);
  if (ncases > 0 && npass == ncases) result = EXIT_SUCCESS;
  return result;
}

/**
 * Mean-order floor a continuation must clear to be shown as attested. The
 * per-step gate (ngram_gen_minorder) is dominated by the low-order seeding
 * transient, so a whole continuation is judged by its MEAN grounding order:
 * at maxorder 2 every run averages 2.0 (pure bigram drift, suppressed); with
 * deeper context (-x) genuinely attested passages average much higher.
 */
/**
 * Whether the -c cascade may synthesize. Off by default: every other tier emits
 * text that occurs verbatim in the corpus, and this one does not, so it is a
 * different promise to the caller and should be asked for.
 */
static int recomb_compose_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    cached = (NULL != getenv("CONVERSE_RECOMB_COMPOSE")) ? 1 : 0;
  }
  return cached;
}


static double ngram_gen_contfloor(void)
{
  double result = 3.0;
  const char* env = getenv("CONVERSE_GEN_CONTORDER");
  if (NULL != env && '\0' != *env) {
    double v = atof(env);
    if (v >= 1.0 && v <= (double)NGRAM_ORDER_MAX) result = v;
  }
  return result;
}

/**
 * Completion-mode response: for a question, answer from the corpus (facts +
 * retrieval) first, then add a grounded continuation generated over the whole
 * corpus n-gram model and seeded from the ANSWER, so -c is never empty and the
 * continuation extends what was said. Non-questions keep the pure generation
 * probe. Continuation is labeled and its grounding reported.
 */
/**
 * Minimum content-word overlap for a continuation to be shown with an answer.
 * Zero keeps the historical behaviour (attestation only), which is bit-exact.
 */
static double gen_relevance_min(void)
{
  double result = 0.0;
  const char* env = getenv("CONVERSE_GEN_RELEVANCE");
  if (NULL != env && '\0' != *env) {
    const double v = atof(env);
    if (v >= 0.0 && v <= 1.0) result = v;
  }
  return result;
}

/**
 * Content-word overlap between two texts, as a fraction of the smaller content
 * set. The same measure recomb_overlap applies to entry pairs, over lexeme
 * streams instead: the seam here joins an answer to a generated continuation,
 * neither of which is a corpus entry.
 *
 * The grounding gate already in place verifies that a continuation is deeply
 * ATTESTED, which is not the same as being about the same thing - a deep context
 * match on a common word leads wherever that phrasing went in training, so a
 * fluent continuation can drift to an unrelated scene. Shared content beyond stop
 * words is the cheapest evidence of aboutness, and it needs no new machinery.
 */
static double gen_overlap(libxs_lexicon_t* lexicon,
  const libxs_lexrule_t* rules, int nrules, const char* a, int alen,
  const char* b, int blen)
{
  double result = 0.0;
  libxs_lexeme_stream_t sa, sb;
  libxs_lexeme_stream_init(&sa);
  libxs_lexeme_stream_init(&sb);
  if (EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &sa,
      (const unsigned char*)a, (size_t)alen, rules, nrules, converse_lexnorms(),
      converse_lexnorms_size(), 0)
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &sb,
      (const unsigned char*)b, (size_t)blen, rules, nrules, converse_lexnorms(),
      converse_lexnorms_size(), 0))
  {
    size_t ia, ib;
    int na = 0, nb = 0, shared = 0;
    for (ia = 0; ia < sa.size; ++ia) {
      const libxs_lexeme_t* la = sa.data + ia;
      if (0 == la->id || 0 != (la->flags & LIBXS_LEXEME_STOP)) continue;
      if (0 == (la->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))) continue;
      ++na;
      for (ib = 0; ib < sb.size; ++ib) {
        if (sb.data[ib].id == la->id) {
          ++shared;
          break;
        }
      }
    }
    for (ib = 0; ib < sb.size; ++ib) {
      const libxs_lexeme_t* lb = sb.data + ib;
      if (0 != lb->id && 0 == (lb->flags & LIBXS_LEXEME_STOP)
        && 0 != (lb->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER)))
      {
        ++nb;
      }
    }
    { const int smaller = (na < nb) ? na : nb;
      if (0 < smaller) result = (double)shared / (double)smaller;
    }
  }
  libxs_lexeme_stream_release(&sa);
  libxs_lexeme_stream_release(&sb);
  return result;
}


static void complete_respond(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile, int budget,
  const char* text, int text_len)
{
  if (0 != is_question_query(text, (size_t)text_len, lexicon, rules, nrules)) {
    char answer[COMPOSE_MAXTEXT];
    char gen[COMPOSE_MAXTEXT];
    double order_mean = 0.0;
    int order_min = 0;
    int answered = answer_query(corpus, text, (size_t)text_len, budget,
      lexicon, rules, nrules, answer_model, profile, answer, sizeof(answer));
    const char* seed = (0 != answered && '\0' != answer[0]) ? answer : text;
    int seed_len = (0 != answered && '\0' != answer[0])
      ? (int)strlen(answer) : text_len;
    int ntok = ngram_generate(converse_ngram_handle()->store, lexicon, rules, nrules,
      seed, seed_len, gen, sizeof(gen), &order_mean, &order_min);
    int grounded = (ntok > 0 && order_mean >= ngram_gen_contfloor()) ? 1 : 0;
    int composed = 0;
    double relevance = -1.0;
    /**
     * Attested is not the same as relevant. The grounding gate above checks that
     * the continuation is deep replay; this checks that it is about the same
     * thing as the answer it is being appended to. Off by default (threshold 0),
     * so the historical output is unchanged unless the gate is asked for.
     */
    if (0 != grounded) {
      const double need = gen_relevance_min();
      relevance = gen_overlap(lexicon, rules, nrules, seed, seed_len, gen,
        (int)strlen(gen));
      if (relevance < need) grounded = 0;
    }
    if (0 != grounded) {
      printf("continuation: %s\n", gen);
      printf("grounding: %d tokens, mean order %.1f, min order %d,"
        " relevance %.2f\n", ntok, order_mean, order_min, relevance);
    }
    /**
     * Composition is the LAST tier, tried only once replay has failed. A
     * continuation is verbatim attested text, so it is better grounded than a
     * splice: offering a synthesized sentence while an attested one was available
     * would trade grounding for novelty without saying so. Seeded from the answer
     * when there is one, because that is the text retrieval already judged
     * relevant to the query.
     */
    if (0 == grounded && 0 != recomb_compose_on()) {
      char syn[COMPOSE_MAXTEXT];
      int nfront = 0, ncand = 0;
      const int slen = converse_recomb_compose_best(corpus, lexicon, rules,
        nrules, seed, seed_len, syn, sizeof(syn), &nfront, &ncand);
      if (0 < slen) {
        composed = 1;
        printf("composed: %s\n", syn);
        /**
         * The provenance line is not decoration. This sentence is in the corpus
         * NOWHERE, so a reader has to be able to tell it from the attested replies
         * above it, and the front size says whether the objective actually chose
         * it or was indifferent among that many.
         */
        printf("synthesis: spliced at a shared term, %d candidate%s,"
          " %d on the front (not attested; every word is)\n",
          ncand, (1 == ncand) ? "" : "s", nfront);
        /**
         * State what was NOT checked. Five separate gate ideas were measured and
         * none can tell a true join from a fluent false one: the evidence that
         * would decide it lies at clause scale, where only 2% of four-word content
         * spans recur even at 27x this corpus. So the honest output is a caveat
         * rather than a confidence, and it is worded as the specific thing left
         * unverified - "sense of the shared term" - because a generic disclaimer
         * on every line is quickly ignored.
         *
         * The front size carries the rest: when the objective is indifferent among
         * most candidates it did not really choose, and saying so is what keeps
         * this consistent with the abstention discipline the QA side uses.
         */
        printf("unverified: the shared term may carry a different sense in each"
          " half; no gate here checks that%s\n",
          (nfront > 1 && nfront * 2 >= ncand)
            ? ", and the objective was near-indifferent among these" : "");
      }
    }
    if (0 == answered && 0 == grounded && 0 == composed) {
      printf("I do not know from the corpus.\n");
    }
  }
  else {
    ngram_complete(converse_ngram_handle()->store, lexicon, rules, nrules, 0,
      text, text_len);
  }
}


static double converse_score(const corpus_fprint_t* candidate,
  const libxs_fprint_t* query, const corpus_fprint_t* prev)
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
  const libxs_fprint_t* query, const corpus_fprint_t* prev,
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
  corpus_fprint_t prev_fprint;
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
    /**
     * THE QUESTION AS ASKED COMES FIRST, and the multi-turn rewrite is a FALLBACK.
     *
     * The rewrite exists for a follow-up that cannot stand alone ("What does it
     * do?"), and it carries the remembered topic in by appending " of the <topic>"
     * to anything that has no term of its own. Applied first, that broke every
     * self-contained question asked after a successful one: "What connects Admetus
     * and Python?" became a question naming a third entity, and the graph resolver
     * refused a pair it could not identify. A rewrite that is only reached when the
     * question fails on its own cannot damage a question that does not need it.
     */
    if (0 != answer_query(corpus, q, qlen, budget,
      lexicon, rules, nrules, answer_model, profile, NULL, 0))
    {
      conv_remember(q, qlen);
      return result;
    }
    if (EXIT_SUCCESS == conv_rewrite(query_text, query_len, rewritten,
      sizeof(rewritten)))
    {
      q = rewritten;
      qlen = strlen(rewritten);
    }
    if (q != query_text && 0 != answer_query(corpus, q, qlen, budget,
      lexicon, rules, nrules, answer_model, profile, NULL, 0))
    {
      conv_remember(q, qlen);
      return result;
    }
    { char gen[COMPOSE_MAXTEXT];
      double order_mean = 0.0;
      int ntok = ngram_generate(converse_ngram_handle()->store, lexicon, rules, nrules,
        q, qlen, gen, sizeof(gen), &order_mean, NULL);
      if (ntok > 0 && order_mean >= ngram_gen_contfloor()) {
        printf("%s\n", gen);
        return result;
      }
    }
    printf("I do not know from the corpus.\n");
    return result;
  }

  ncandidates = libxs_spatial_nearest(spatial, qcode, 256, candidates);

  for (step = 0; step < budget; ++step) {
    const corpus_fprint_t* prev = (0 != have_prev) ? &prev_fprint : NULL;
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


/**
 * Recombination over the corpus. The byte model and the word probability are
 * DIAGNOSTICS here, not prerequisites: both only produce seam scores, and a seam
 * score has never separated a true join from a fluent false one - the last
 * attempt ranked the false continuation 50x above the true one - so
 * grammaticality is enforced by the clause constraint instead. Running without
 * them drops the bpc and penalty columns and nothing else, which is also what
 * lets this half link without the byte model at all.
 */
static void converse_qa_recomb(converse_run_t* run,
  libxs_registry_t* ngram_model)
{
  const char* env = getenv("CONVERSE_RECOMB");
  const int limit = (NULL != env && '\0' != *env && 0 < atoi(env))
    ? atoi(env) : 50;
  converse_recomb_host_t host;
  host.ends_sentence = text_ends_sentence;
  host.is_wordchar = libxs_lexeme_is_word_char;
  host.entry_build = corpus_entry_build;
  host.word_prob = (NULL != ngram_model) ? recomb_word_prob : NULL;
  host.seam_bits = (0 != converse_judge_active())
    ? converse_judge_seam_bits : NULL;
  host.maxorder = ngram_maxorder();
  if (0 == converse_judge_active()) {
    fprintf(stderr, "recomb: no seam judge"
      " (CONVERSE_HIER_RESCORE=1 adds the bpc columns)\n");
  }
  converse_recomb_probe_run(run->corpus, run->lexicon, run->rules, run->nrules,
    limit, &host);
}


/**
 * Read prompts and print grounded continuations. Questions answer from the
 * corpus first (with a continuation on top); everything else replays the
 * n-gram. A prediction kind under -c is served by the other half, so this path
 * only ever holds the grounded generator.
 */
static void converse_qa_complete(converse_run_t* run,
  libxs_registry_t* ngram_model)
{
  converse_recomb_host_t compose;
  char line[4096];
  int compose_ready = 0;
  if (0 != recomb_compose_on()) {
    compose.ends_sentence = text_ends_sentence;
    compose.is_wordchar = libxs_lexeme_is_word_char;
    compose.entry_build = corpus_entry_build;
    compose.word_prob = (NULL != ngram_model) ? recomb_word_prob : NULL;
    compose.seam_bits = (0 != converse_judge_active())
      ? converse_judge_seam_bits : NULL;
    compose.maxorder = ngram_maxorder();
    if (0 == converse_judge_active()) {
      fprintf(stderr, "compose: no seam judge"
        " (CONVERSE_HIER_RESCORE=1 adds the bpc columns)\n");
    }
    /* One corpus pass, kept for the session: per-query would dominate. */
    if (EXIT_SUCCESS == converse_recomb_open(run->corpus, run->lexicon,
      &compose))
    {
      compose_ready = 1;
    }
    else fprintf(stderr, "compose: pivot index could not be built\n");
  }
  printf("> ");
  fflush(stdout);
  while (NULL != fgets(line, (int)sizeof(line), stdin)) {
    size_t len = strlen(line);
    while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
    if (0 < len) {
      if (0 != is_question_query(line, len, run->lexicon, run->rules,
        run->nrules))
      {
        complete_respond(run->corpus, run->lexicon, run->rules, run->nrules,
          run->answer_model, run->profile, run->budget, line, (int)len);
      }
      else ngram_complete(ngram_model, run->lexicon, run->rules, run->nrules,
        run->ngram_order, line, (int)len);
    }
    printf("> ");
    fflush(stdout);
  }
  if (0 != compose_ready) converse_recomb_close();
}


/** Interactive answering: the corpus is searched, never sampled from. */
static int converse_qa_answer(converse_run_t* run)
{
  libxs_spatial_t spatial;
  int result = corpus_spatial_build(&spatial, run->corpus);
  if (EXIT_SUCCESS == result) {
    libxs_registry_t* ngram_model = ngram_build(run->corpus, run->lexicon,
      run->rules, run->nrules, 0);
    char line[4096];
    ngram_backoff_build(ngram_model, run->lexicon);
    conv_reset();
    printf("> ");
    fflush(stdout);
    while (NULL != fgets(line, (int)sizeof(line), stdin)) {
      size_t len = strlen(line);
      libxs_fprint_t query;
      size_t shape;
      while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
      if (0 < len) {
        shape = len;
        libxs_fprint(&query, LIBXS_DATATYPE_U8, line, 1,
          &shape, NULL, FPRINT_ORDER, 0, 0, 0);
        respond(&spatial, run->corpus, line, len, &query, run->budget,
          run->lexicon, run->rules, run->nrules, run->answer_model,
          run->profile);
      }
      printf("> ");
      fflush(stdout);
    }
    libxs_spatial_destroy(&spatial);
  }
  return result;
}


int converse_qa_run(converse_run_t* run)
{
  int result = EXIT_SUCCESS;
  answer_query_lexicon = run->lexicon;
  answer_query_rules = run->rules;
  answer_query_nrules = run->nrules;
  answer_bridge_load_file(converse_bridge_path());
  answer_bridge_report(stderr);
  /* Before the facts and before any query: the resolvers ask it which words the
   * corpus uses as names, which is what decides who answers. */
  /* The derived layer is most of a warm start - the census and the four fact
   * builds each walk the whole corpus - so it is cached under a stamp of what
   * it was built from, and rebuilt whenever that differs. */
  if (EXIT_SUCCESS != answer_facts_load(run->corpus, run->lexicon)) {
    answer_case_build(run->corpus, run->lexicon, run->rules, run->nrules);
    answer_case_report(stderr);
    converse_stage_end("f_case");
    answer_verbs_build(run->corpus);
    converse_stage_end("f_verbs");
    /* After the verbs, because the noun test compares the two frames' counts. */
    answer_nouns_build(run->corpus);
    converse_stage_end("f_nouns");
    answer_relation_facts_build(run->corpus);
    converse_stage_end("f_relation");
    answer_identity_facts_build(run->corpus);
    converse_stage_end("f_identity");
    answer_describe_facts_build(run->corpus);
    converse_stage_end("f_describe");
    answer_docdef_facts_build(run->corpus);
    converse_stage_end("f_docdef");
    answer_location_facts_build(run->corpus);
    converse_stage_end("f_location");
    answer_type_facts_build(run->corpus);
    converse_stage_end("f_type");
    answer_own_facts_build(run->corpus);
    converse_stage_end("f_own");
    answer_facts_save(run->corpus, run->lexicon);
    converse_stage_end("f_save");
  }
  else converse_stage_end("f_load");
  answer_nouns_report(stderr);
  answer_relation_facts_report(stderr);
  answer_identity_facts_report(stderr);
  answer_describe_facts_report(stderr);
  answer_docdef_facts_report(stderr);
  answer_location_facts_report(stderr);
  answer_type_facts_report(stderr);
  answer_own_facts_report(stderr);
  answer_verbs_report(stderr);
  answer_graph_report(stderr);
  converse_judge_open(run->corpus);
  if (0 != run->eval_mode) {
    result = eval_converse(run->corpus, run->lexicon, run->rules, run->nrules,
      run->answer_model, run->profile);
  }
  else if (0 != run->predict_eval_mode || 0 != run->complete_mode) {
    libxs_registry_t* ngram_model;
    converse_bpe_prepare(run->corpus, run->ngram_holdout);
    converse_stage_end("test_ingest");
    ngram_model = ngram_build(run->corpus, run->lexicon, run->rules,
      run->nrules, run->ngram_holdout);
    ngram_backoff_build(ngram_model, run->lexicon);
    converse_stage_end("ngram_build");
    converse_stage_begin();
    if (0 != run->predict_eval_mode) {
      converse_qa_recomb(run, ngram_model);
      ngram_stats(ngram_model);
      converse_stage_end("eval");
      converse_stage_report();
    }
    else converse_qa_complete(run, ngram_model);
  }
  else result = converse_qa_answer(run);
  /* Reported for every mode, not only the prediction eval: setup dominates an
   * interactive session and a fixture run, and those were the two the timer
   * could not be read in. */
  converse_stage_report();
  converse_judge_close();
  answer_case_free();
  answer_bridge_free_loaded();
  answer_relation_facts_free();
  answer_identity_facts_free();
  answer_describe_facts_free();
  answer_docdef_facts_free();
  answer_location_facts_free();
  answer_type_facts_free();
  answer_own_facts_free();
  answer_verbs_free();
  answer_nouns_free();
  return result;
}
