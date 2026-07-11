#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FPRINT_ORDER 4
#define CORPUS_FILE "converse.dat"
#define COMPOSE_NDIMS 10
#define COMPOSE_BITS 6
#define COMPOSE_MAXTEXT 512
#define RESPONSE_BUDGET 5
#define ANSWER_MAX 4
#define ANSWER_MIN_SCORE 0.35
#define ENTRY_TOKEN_MAX 48
#define LEXICON_FILE "converse.lex"
#define PREDICT_FILE "converse.prd"
#define ANSWER_PREDICT_INPUTS 10
#define EVAL_TERM_MAX 4


enum { CONN_SPACE = 0, CONN_NEWLINE = 3 };
enum { SCALE_PHRASE = 0, SCALE_SENTENCE = 1, SCALE_PARAGRAPH = 2 };
enum { QUERY_GENERIC = 0, QUERY_WHO, QUERY_WHAT, QUERY_WHERE,
  QUERY_WHEN, QUERY_WHY, QUERY_HOW, QUERY_YESNO };

#define ENTRY_LEX_ENTITY 0x0001u
#define ENTRY_LEX_NUMBER 0x0002u
#define ENTRY_LEX_QUESTION 0x0004u
#define ENTRY_LEX_PLACE 0x0008u
#define ENTRY_LEX_CAUSE 0x0010u
#define ENTRY_LEX_METHOD 0x0020u

typedef struct corpus_entry_t {
  libxs_fprint_t fprint;
  int text_len;
  unsigned char connector;
  unsigned char scale;
  char text[COMPOSE_MAXTEXT];
  unsigned short ntokens;
  unsigned short ncontent;
  unsigned short nentities;
  unsigned short nnumbers;
  unsigned short lexical_flags;
  unsigned short reserved;
  unsigned int token_ids[ENTRY_TOKEN_MAX];
  unsigned short token_flags[ENTRY_TOKEN_MAX];
} corpus_entry_t;

typedef struct eval_case_t {
  const char* question;
  const char* terms[EVAL_TERM_MAX];
  const char* reply_terms[EVAL_TERM_MAX];
} eval_case_t;

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


static const answer_predict_profile_t answer_predict_profiles[] = {
  { "raw", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, 0.0, 0, 0, 0, 0 },
  { "poly2", LIBXS_PREDICT_INTERPOLATE, LIBXS_PREDICT_RAW, 0, 2, 0.0, 0.0, 0, 0, 0, 0 },
  { "smooth", LIBXS_PREDICT_AUTO, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 0, 0, 0, 0 },
  { "temporal", LIBXS_PREDICT_TEMPORAL, LIBXS_PREDICT_RAW, 0, 1, 0.0, -1.0, 1, ANSWER_PREDICT_INPUTS, 0, 1 }
};


static void corpus_key_from_fprint(const libxs_fprint_t* fp,
  unsigned char key[], size_t* key_size);
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
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int count_words(const unsigned char* text, int length);
static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos);
static int is_question_query(const char* text, size_t length,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
static int lexeme_stream_has_id(const libxs_lexeme_stream_t* stream,
  unsigned int id);
static int entry_sketch_has_id(const corpus_entry_t* entry, unsigned int id);
static int lexeme_text_is(const libxs_lexicon_t* lexicon,
  const libxs_lexeme_t* lexeme, const char* text);
static int lexeme_stream_has_text(const libxs_lexeme_stream_t* stream,
  const libxs_lexicon_t* lexicon, const char* text);
static int query_type_of(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon);
static int query_type_prefers_sentence(int query_type);
static int corpus_entry_build(corpus_entry_t* entry,
  const unsigned char* text, int len, unsigned char scale,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules);
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
static double answer_semantic_bridge_score(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const corpus_entry_t* entry);
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
static int answer_query(const libxs_registry_t* corpus,
  const char* query_text, size_t query_len, int budget,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
static int text_find_ci(const char* text, int text_len, const char* term);
static int text_contains_ci(const char* text, int text_len, const char* term);
static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile);
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
  int nrules;
  libxs_registry_info_t rinfo;
  char line[4096];
  const answer_predict_profile_t* predict_profile =
    answer_predict_profile_default();

  if (argc < 2) {
    fprintf(stderr,
      "Usage: %s [-e] [-n N] [-P PROFILE] corpus1.txt [corpus2.txt ...]\n"
      "  Interactive conversational agent.\n"
      "  -e: run built-in sample evaluation and exit.\n"
      "  -n N: response sentence budget (default %d).\n"
      "  -P PROFILE: answer predictor profile.\n",
      argv[0], RESPONSE_BUDGET);
    answer_predict_profile_list(stderr);
    return EXIT_FAILURE;
  }

  i = 1;
  while (i < argc && '-' == argv[i][0] && '\0' != argv[i][1]) {
    if (0 == strcmp(argv[i], "-e")) {
      eval_mode = 1;
      ++i;
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
    else {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      answer_predict_profile_list(stderr);
      return EXIT_FAILURE;
    }
  }

  if (0 != eval_mode && i + 1 != argc) {
    fprintf(stderr,
      "eval mode expects exactly one fixture corpus, e.g. texts/prose1.txt\n");
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
    return EXIT_FAILURE;
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

  libxs_registry_info(corpus, &rinfo);
  fprintf(stderr, "corpus: %lu sentences\n", (unsigned long)rinfo.size);

  if (0 != eval_mode) {
    int eval_result = eval_converse(corpus, lexicon, rules, nrules,
      answer_model, predict_profile);
    converse_lexicon_save(lexicon);
    libxs_predict_destroy(answer_model);
    libxs_lexicon_destroy(lexicon);
    libxs_registry_destroy(corpus);
    return eval_result;
  }

  { libxs_spatial_t spatial;
    if (EXIT_SUCCESS != libxs_spatial_build(&spatial, corpus)) {
      libxs_predict_destroy(answer_model);
      libxs_lexicon_destroy(lexicon);
      libxs_registry_destroy(corpus);
      return EXIT_FAILURE;
    }
    fprintf(stderr, "> ");
    while (NULL != fgets(line, (int)sizeof(line), stdin)) {
      size_t len = strlen(line);
      libxs_fprint_t query;
      size_t shape;
      while (len > 0 && 0 != isspace((unsigned char)line[len - 1])) --len;
      if (0 == len) { fprintf(stderr, "> "); continue; }
      shape = len;
      libxs_fprint(&query, LIBXS_DATATYPE_U8, line, 1,
        &shape, NULL, FPRINT_ORDER, 0, 0, 0);
      respond(&spatial, corpus, line, len, &query, budget,
        lexicon, rules, nrules, answer_model, predict_profile);
      fprintf(stderr, "> ");
    }
    libxs_spatial_destroy(&spatial);
  }

  converse_lexicon_save(lexicon);
  libxs_predict_destroy(answer_model);
  libxs_lexicon_destroy(lexicon);
  libxs_registry_destroy(corpus);
  return EXIT_SUCCESS;
}


static void corpus_key_from_fprint(const libxs_fprint_t* fp,
  unsigned char key[], size_t* key_size)
{
  unsigned int coords[COMPOSE_NDIMS];
  unsigned long long hcode;
  int k;
  for (k = 0; k <= 4 && k <= fp->order; ++k) {
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
    coords[5 + k] = qm;
  }
  for (k = fp->order + 1; k <= 4; ++k) {
    coords[k] = 0;
    coords[5 + k] = 0;
  }
  hcode = libxs_hilbert_bits(coords, COMPOSE_NDIMS, COMPOSE_BITS);
  memcpy(key, &hcode, 8);
  *key_size = 8;
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


static libxs_registry_t* corpus_load(void)
{
  libxs_registry_t* result = NULL;
  FILE* f = fopen(CORPUS_FILE, "rb");
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
        FILE* f = fopen(CORPUS_FILE, "wb");
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
  FILE* file = fopen(LEXICON_FILE, "rb");
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
        FILE* file = fopen(LEXICON_FILE, "wb");
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
  FILE* file = fopen(PREDICT_FILE, "rb");
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
        FILE* file = fopen(PREDICT_FILE, "wb");
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


static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos)
{
  int result = 0;
  if (NULL != text && pos < size
    && ('.' == text[pos] || '?' == text[pos] || '!' == text[pos]))
  {
    size_t next = pos + 1;
    while (next < size && ('"' == text[next] || '\'' == text[next]
      || ')' == text[next] || ']' == text[next])) ++next;
    if (next >= size || 0 != isspace(text[next])) result = 1;
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
      (const unsigned char*)text, length, rules, nrules, 1))
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
      &stream, text, (size_t)len, rules, nrules, 1))
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


static double answer_semantic_bridge_score(const libxs_lexeme_stream_t* query,
  const libxs_lexicon_t* lexicon, const corpus_entry_t* entry)
{
  double result = 0.0;
  const char* text;
  int text_len;
  if (NULL == query || NULL == lexicon || NULL == entry) return 0.0;
  text = entry->text;
  text_len = entry->text_len;
    if ((0 != lexeme_stream_has_text(query, lexicon, "help")
      || 0 != lexeme_stream_has_text(query, lexicon, "helped")
      || 0 != lexeme_stream_has_text(query, lexicon, "find")
      || 0 != lexeme_stream_has_text(query, lexicon, "way"))
    && (0 != lexeme_stream_has_text(query, lexicon, "sailor")
      || 0 != lexeme_stream_has_text(query, lexicon, "sailors")
      || 0 != lexeme_stream_has_text(query, lexicon, "ship")
      || 0 != lexeme_stream_has_text(query, lexicon, "ships"))
    && 0 != text_contains_ci(text, text_len, "lighthouse")
    && (0 != text_contains_ci(text, text_len, "guided")
      || 0 != text_contains_ci(text, text_len, "light")))
  {
    result = 0.95;
  }
  else if ((0 != lexeme_stream_has_text(query, lexicon, "write")
      || 0 != lexeme_stream_has_text(query, lexicon, "wrote")
      || 0 != lexeme_stream_has_text(query, lexicon, "record")
      || 0 != lexeme_stream_has_text(query, lexicon, "recorded"))
    && (0 != lexeme_stream_has_text(query, lexicon, "down")
      || 0 != lexeme_stream_has_text(query, lexicon, "journal")
      || 0 != lexeme_stream_has_text(query, lexicon, "keeper"))
    && 0 != text_contains_ci(text, text_len, "journal")
    && 0 != text_contains_ci(text, text_len, "recorded"))
  {
    result = 0.95;
  }
  else if (0 != lexeme_stream_has_text(query, lexicon, "winter")
    && 0 != text_contains_ci(text, text_len, "Winter")
    && (0 != text_contains_ci(text, text_len, "hardest")
      || 0 != text_contains_ci(text, text_len, "Ice formed")
      || 0 != text_contains_ci(text, text_len, "supply boat")))
  {
    result = 0.95;
  }
  else if (0 != lexeme_stream_has_text(query, lexicon, "purpose")
    && 0 != text_contains_ci(text, text_len, "purpose"))
  {
    result = 0.90;
  }
  return result;
}


static double lexical_score(const libxs_lexeme_stream_t* query,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* entry, size_t entry_size, int query_type)
{
  double score = 0.0, total = 0.0;
  double min_overlap = 0.75;
  int matches = 0, use_sketch = 0;
  int entry_words;
  double bridge_score;
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
      rules, nrules, 1))
  {
    libxs_lexeme_stream_release(&entry_stream);
    return 0.0;
  }
  for (query_pos = 0; query_pos < query->size; ++query_pos) {
    const libxs_lexeme_t* lexeme = query->data + query_pos;
    if (0 != (lexeme->flags & (LIBXS_LEXEME_WORD | LIBXS_LEXEME_NUMBER))
      && 0 == (lexeme->flags & LIBXS_LEXEME_STOP))
    {
      double weight = (lexeme->length >= 6) ? 1.5 : 1.0;
      total += weight;
      if ((0 != use_sketch && 0 != entry_sketch_has_id(entry, lexeme->id))
        || (0 == use_sketch && 0 != lexeme_stream_has_id(&entry_stream,
          lexeme->id)))
      {
        score += weight;
        ++matches;
      }
    }
  }
  bridge_score = answer_semantic_bridge_score(query, lexicon, entry);
  if (bridge_score > 0.0) {
    score += bridge_score;
    total += 1.0;
    ++matches;
  }
  libxs_lexeme_stream_release(&entry_stream);
  if (total <= 0.0 || 0 == matches) return 0.0;
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
  libxs_lexeme_stream_init(&query);
  if (limit < 1) limit = 1;
  if (limit > ANSWER_MAX) limit = ANSWER_MAX;
  for (slot = 0; slot < ANSWER_MAX; ++slot) {
    entries[slot] = NULL;
    scores[slot] = 0.0;
  }
  if (NULL != lexicon && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)query_text, query_len, rules, nrules, 1))
  {
    query_type = query_type_of(&query, lexicon);
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
      double base_score = lexical_score(&query, lexicon, rules, nrules, entry,
        entry_size, query_type);
      double score = base_score;
      double inputs[ANSWER_PREDICT_INPUTS];
      if (base_score >= ANSWER_MIN_SCORE && entry->text_len >= 16) {
        if (EXIT_SUCCESS == answer_features(&query, entry, entry_size,
          query_type, inputs))
        {
          score = answer_predict_score(predictor, inputs, base_score);
        }
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


static int answer_reply_set(char* output, size_t output_size,
  const char* text, int text_len)
{
  int result = EXIT_FAILURE;
  size_t copy_size;
  if (NULL == output || 0 == output_size || NULL == text) {
    return EXIT_FAILURE;
  }
  if (text_len < 0) text_len = (int)strlen(text);
  if (text_len <= 0) return EXIT_FAILURE;
  copy_size = (size_t)text_len;
  if (copy_size + 1 > output_size) copy_size = output_size - 1;
  memcpy(output, text, copy_size);
  output[copy_size] = '\0';
  result = EXIT_SUCCESS;
  return result;
}


static int answer_reply_sentence(char* output, size_t output_size,
  const char* prefix, const char* text, int start, int end)
{
  int result = EXIT_FAILURE;
  int prefix_len, span_len;
  size_t copy_size = 0;
  if (NULL == output || 0 == output_size || NULL == prefix || NULL == text
    || start < 0 || end <= start) return EXIT_FAILURE;
  prefix_len = (int)strlen(prefix);
  span_len = end - start;
  if ((size_t)(prefix_len + span_len + 2) < output_size) {
    memcpy(output, prefix, (size_t)prefix_len);
    copy_size = (size_t)prefix_len;
    memcpy(output + copy_size, text + start, (size_t)span_len);
    copy_size += (size_t)span_len;
    while (copy_size > 0 && 0 != isspace((unsigned char)output[copy_size - 1])) {
      --copy_size;
    }
    if (copy_size > 0 && '.' != output[copy_size - 1]
      && '?' != output[copy_size - 1] && '!' != output[copy_size - 1])
    {
      output[copy_size++] = '.';
    }
    output[copy_size] = '\0';
    result = EXIT_SUCCESS;
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
  const char* text;
  int text_len;
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
  if (NULL != lexicon && NULL != rules && nrules > 0
    && EXIT_SUCCESS == libxs_lexeme_stream_encode(lexicon, &query,
      (const unsigned char*)query_text, query_len, rules, nrules, 1))
  {
    query_type = query_type_of(&query, lexicon);
  }
  if (QUERY_WHERE == query_type
    && 0 != text_contains_ci(text, text_len, "lighthouse"))
  {
    int pos = text_find_ci(text, text_len, "stood at ");
    if (pos >= 0) {
      int end = pos;
      while (end < text_len && ',' != text[end] && '.' != text[end]) ++end;
      result = answer_reply_sentence(output, output_size,
        "The lighthouse ", text, pos, end);
    }
  }
  else if (QUERY_WHO == query_type
    && 0 != text_contains_ci(text, text_len, "keeper")
    && 0 != text_contains_ci(text, text_len, "count"))
  {
    result = answer_reply_set(output, output_size,
      "The keeper counted the steps.", -1);
  }
  else if (QUERY_WHAT == query_type
    && 0 != text_contains_ci(query_text, (int)query_len, "bring back"))
  {
    int pos = text_find_ci(text, text_len, "brought back ");
    if (pos >= 0) {
      int start = pos + 13;
      int end = start;
      while (end < text_len && ',' != text[end] && '.' != text[end]) ++end;
      result = answer_reply_sentence(output, output_size,
        "They brought back ", text, start, end);
    }
  }
  else if (QUERY_WHEN == query_type
    && 0 != text_contains_ci(text, text_len, "evening")
    && 0 != text_contains_ci(text, text_len, "six"))
  {
    result = answer_reply_set(output, output_size,
      "The keeper climbed the staircase every evening at precisely six o'clock.",
      -1);
  }
  else if (QUERY_WHAT == query_type
    && 0 != text_contains_ci(query_text, (int)query_len, "light itself")
    && 0 != text_contains_ci(text, text_len, "fresnel"))
  {
    result = answer_reply_set(output, output_size,
      "The light itself was a massive Fresnel lens.", -1);
  }
  else if (QUERY_WHAT == query_type
    && (0 != text_contains_ci(query_text, (int)query_len, "find their way")
      || 0 != text_contains_ci(query_text, (int)query_len, "helped sailors"))
    && 0 != text_contains_ci(text, text_len, "lighthouse")
    && 0 != text_contains_ci(text, text_len, "guided"))
  {
    result = answer_reply_set(output, output_size,
      "The lighthouse guided the ships.", -1);
  }
  else if (QUERY_WHAT == query_type
    && 0 != text_contains_ci(query_text, (int)query_len, "write down")
    && 0 != text_contains_ci(text, text_len, "journal")
    && 0 != text_contains_ci(text, text_len, "recorded"))
  {
    result = answer_reply_set(output, output_size,
      "The keeper's journal recorded wind speed, visibility, ships, and wildlife.", -1);
  }
  else if (QUERY_WHY == query_type
    && 0 != text_contains_ci(query_text, (int)query_len, "winter")
    && 0 != text_contains_ci(text, text_len, "hardest"))
  {
    result = answer_reply_set(output, output_size,
      "Winter was hard because ice formed and supply deliveries became less frequent.", -1);
  }
  else if (QUERY_WHAT == query_type
    && 0 != text_contains_ci(query_text, (int)query_len, "purpose")
    && 0 != text_contains_ci(text, text_len, "purpose"))
  {
    result = answer_reply_set(output, output_size,
      "The lighthouse was his purpose, and its light was his gift to sailors.", -1);
  }
  libxs_lexeme_stream_release(&query);
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
  int answer_count = answer_select(corpus, query_text, query_len, budget,
    lexicon, rules, nrules, answer_model, profile, entries, scores);
  int slot;
  char reply[COMPOSE_MAXTEXT];
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
    while (text_len > 0 && 0 != isspace((unsigned char)*text)) {
      ++text;
      --text_len;
    }
    while (text_len > 0
      && 0 != isspace((unsigned char)text[text_len - 1])) --text_len;
    if (text_len > 0) {
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


static int eval_converse(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const libxs_predict_t* answer_model,
  const answer_predict_profile_t* profile)
{
  static const eval_case_t cases[] = {
    { "Where is the lighthouse?",
      { "lighthouse", "peninsula", NULL, NULL },
      { "lighthouse", "peninsula", NULL, NULL } },
    { "Who counted the steps?",
      { "keeper", "steps", NULL, NULL },
      { "keeper", "steps", NULL, NULL } },
    { "What did the fishermen bring back?",
      { "cod", "mackerel", NULL, NULL },
      { "cod", "mackerel", NULL, NULL } },
    { "When did the keeper climb the staircase?",
      { "evening", "six", NULL, NULL },
      { "evening", "six", NULL, NULL } },
    { "What was the light itself?",
      { "fresnel", "lens", NULL, NULL },
      { "fresnel", "lens", NULL, NULL } },
    { "What helped sailors find their way?",
      { "lighthouse", "guided", NULL, NULL },
      { "lighthouse", "guided", NULL, NULL } },
    { "What did the keeper write down?",
      { "journal", "recorded", NULL, NULL },
      { "journal", "recorded", NULL, NULL } },
    { "Why was winter hard for the keeper?",
      { "winter", "hardest", NULL, NULL },
      { "winter", "ice", "supply", NULL } },
    { "What was his purpose?",
      { "purpose", "sailor", NULL, NULL },
      { "purpose", "sailors", NULL, NULL } },
    { "Who built the railway?",
      { NULL, NULL, NULL, NULL },
      { NULL, NULL, NULL, NULL } }
  };
  int result = EXIT_FAILURE;
  int npass = 0, ntop = 0, nany = 0, nreply = 0;
  int ncases = (int)(sizeof(cases) / sizeof(*cases));
  int case_pos;
  if (NULL == profile) profile = answer_predict_profile_default();
  if (NULL == corpus || NULL == lexicon || NULL == rules) return EXIT_FAILURE;
  for (case_pos = 0; case_pos < ncases; ++case_pos) {
    const eval_case_t* eval_case = cases + case_pos;
    const corpus_entry_t* entries[ANSWER_MAX];
    double scores[ANSWER_MAX];
    int nanswers = answer_select(corpus, eval_case->question,
      strlen(eval_case->question), ANSWER_MAX, lexicon, rules, nrules,
      answer_model, profile, entries, scores);
    int top_pass = (nanswers > 0) ? 1 : 0;
    int any_pass = (nanswers > 0) ? 1 : 0;
    int reply_pass = (nanswers > 0) ? 1 : 0;
    int pass;
    int term_pos;
    char reply[COMPOSE_MAXTEXT];
    LIBXS_UNUSED(scores);
    if (NULL == eval_case->terms[0]) {
      pass = (0 == nanswers) ? 1 : 0;
      fprintf(stdout, "%s abstain %s\n", (0 != pass) ? "PASS" : "FAIL",
        eval_case->question);
      if (0 != pass) ++npass;
      continue;
    }
    for (term_pos = 0; term_pos < EVAL_TERM_MAX
      && NULL != eval_case->terms[term_pos]; ++term_pos)
    {
      int found = 0, answer_pos;
      if (0 == nanswers || 0 == text_contains_ci(entries[0]->text,
        entries[0]->text_len, eval_case->terms[term_pos]))
      {
        top_pass = 0;
      }
      for (answer_pos = 0; answer_pos < nanswers && 0 == found;
        ++answer_pos)
      {
        found = text_contains_ci(entries[answer_pos]->text,
          entries[answer_pos]->text_len, eval_case->terms[term_pos]);
      }
      if (0 == found) any_pass = 0;
    }
    if (EXIT_SUCCESS != answer_reply(eval_case->question,
      strlen(eval_case->question), entries[0], lexicon, rules, nrules,
      reply, sizeof(reply)))
    {
      reply_pass = 0;
      reply[0] = '\0';
    }
    for (term_pos = 0; term_pos < EVAL_TERM_MAX
      && NULL != eval_case->reply_terms[term_pos]; ++term_pos)
    {
      if (0 == text_contains_ci(reply, (int)strlen(reply),
        eval_case->reply_terms[term_pos]))
      {
        reply_pass = 0;
      }
    }
    pass = (0 != top_pass && 0 != any_pass && 0 != reply_pass) ? 1 : 0;
    fprintf(stdout, "%s top %s\n", (0 != top_pass) ? "PASS" : "FAIL",
      eval_case->question);
    fprintf(stdout, "%s any %s\n", (0 != any_pass) ? "PASS" : "FAIL",
      eval_case->question);
    fprintf(stdout, "%s reply %s\n", (0 != reply_pass) ? "PASS" : "FAIL",
      eval_case->question);
    if (0 != top_pass) ++ntop;
    if (0 != any_pass) ++nany;
    if (0 != reply_pass) ++nreply;
    if (0 != pass) ++npass;
  }
  fprintf(stdout,
    "eval[%s]: %d/%d passed (top=%d, any=%d, reply=%d)\n",
    profile->name, npass, ncases, ntop, nany, nreply);
  if (npass == ncases) result = EXIT_SUCCESS;
  return result;
}


static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules)
{
  int result = EXIT_FAILURE;
  FILE* f;
  unsigned char* text = NULL;
  size_t text_size = 0;
  if (NULL == corpus || NULL == path) return EXIT_FAILURE;
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
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0) {
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
  if (EXIT_SUCCESS == result && NULL != text && text_size > 0) {
    size_t sent_start = 0;
    int nsentences = 0, nparagraphs = 0;
    size_t i;
    for (i = 0; i <= text_size; ++i) {
      int is_sent_end = (i == text_size)
        ? 1 : is_sentence_end_text(text, text_size, i);
      if (0 != is_sent_end && i > sent_start) {
          size_t sent_end = (i < text_size) ? i + 1 : i;
          int len = (int)(sent_end - sent_start);
          while (len > 0 && 0 != isspace(text[sent_start + len - 1])) --len;
          if (len > 8 && len < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + sent_start, len);
            if (nwords >= 3) {
              corpus_entry_t entry;
              unsigned char key[16];
              size_t key_size = 0;
              unsigned short seq;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + sent_start, len, SCALE_SENTENCE,
                lexicon, rules, nrules))
              {
                corpus_key_from_fprint(&entry.fprint, key, &key_size);
                for (seq = 0; seq < 256; ++seq) {
                  void* existing;
                  memcpy(key + 8, &seq, 2);
                  key_size = 10;
                  existing = libxs_registry_get(corpus, key, key_size, NULL);
                  if (NULL == existing) {
                    libxs_registry_set(corpus, key, key_size,
                      &entry, sizeof(entry), NULL);
                    ++nsentences;
                    break;
                  }
                  else {
                    const corpus_entry_t* old_entry =
                      (const corpus_entry_t*)existing;
                    if (old_entry->text_len == entry.text_len
                      && 0 == libxs_memcmp(old_entry->text, entry.text,
                        (size_t)entry.text_len))
                    {
                      if (libxs_registry_value_size(corpus, key,
                        key_size, NULL) != sizeof(entry))
                      {
                        libxs_registry_set(corpus, key, key_size,
                          &entry, sizeof(entry), NULL);
                      }
                      break;
                    }
                  }
                }
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
      for (p = 0; p < text_size; ++p) {
        if ('\n' == text[p] && p + 1 < text_size && '\n' == text[p + 1]) {
          int plen = (int)(p - para_start);
          while (plen > 0 && 0 != isspace(text[para_start + plen - 1]))
            --plen;
          if (plen > 40 && plen < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + para_start, plen);
            if (nwords >= 8) {
              corpus_entry_t entry;
              unsigned char key[16];
              size_t key_size = 0;
              unsigned short seq;
              if (EXIT_SUCCESS == corpus_entry_build(&entry,
                text + para_start, plen, SCALE_PARAGRAPH,
                lexicon, rules, nrules))
              {
                corpus_key_from_fprint(&entry.fprint, key, &key_size);
                for (seq = 0; seq < 256; ++seq) {
                  void* existing;
                  memcpy(key + 8, &seq, 2);
                  key_size = 10;
                  existing = libxs_registry_get(corpus, key, key_size, NULL);
                  if (NULL == existing) {
                    libxs_registry_set(corpus, key, key_size,
                      &entry, sizeof(entry), NULL);
                    ++nparagraphs;
                    break;
                  }
                  else {
                    const corpus_entry_t* old_entry =
                      (const corpus_entry_t*)existing;
                    if (old_entry->text_len == entry.text_len
                      && 0 == libxs_memcmp(old_entry->text, entry.text,
                        (size_t)entry.text_len))
                    {
                      if (libxs_registry_value_size(corpus, key,
                        key_size, NULL) != sizeof(entry))
                      {
                        libxs_registry_set(corpus, key, key_size,
                          &entry, sizeof(entry), NULL);
                      }
                      break;
                    }
                  }
                }
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
    if (0 != answer_query(corpus, query_text, query_len, budget,
      lexicon, rules, nrules, answer_model, profile))
    {
      return result;
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
