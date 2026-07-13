#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>

#include "converse.h"

#define MAX_SENTENCES 4096


typedef struct sentence_t {
  int byte_offset;
  int byte_length;
  libxs_fprint_t fprint;
} sentence_t;

typedef struct document_t {
  unsigned char* text;
  size_t text_size;
  sentence_t sentences[MAX_SENTENCES];
  int nsentences;
} document_t;

static int count_words(const unsigned char* text, int length);
static int extract_words(const unsigned char* text, int length,
  int* offsets, int* lengths, int max_words);
static int word_in_sentence(const unsigned char* word, int wlen,
  const unsigned char* sent, int slen);
static int is_sentence_end_text(const unsigned char* text, size_t size,
  size_t pos);
static int token_is_content(const libxs_token_t* token);
static int token_stream_has_id(const libxs_token_stream_t* stream,
  unsigned int id);
static double compose_lexical_score(const libxs_token_stream_t* target,
  const libxs_token_stream_t* candidate,
  const libxs_token_stream_t* generated);
static int split_sentences(document_t* doc);
static int fingerprint_sentences(document_t* doc);
static double sentence_redundancy(const document_t* doc, int a, int b);
static int find_best_fusion_pair(const document_t* doc, int* out_a, int* out_b);
static int fuse_sentences(const document_t* doc, int a, int b,
  char* output, size_t output_size, size_t* output_len);
static int read_input(unsigned char** data, size_t* size, FILE* file);
static libxs_registry_t* corpus_load(void);
static int corpus_save(const libxs_registry_t* corpus);
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  int reflow);
static int compose_document(const libxs_registry_t* corpus,
  const libxs_fprint_t* target, const unsigned char* target_text,
  size_t target_size, int budget);


static int summarize_document(document_t* doc, int target_n);


int main(int argc, char* argv[])
{
  unsigned char* input = NULL;
  size_t input_size = 0;
  document_t doc;
  int target_n = 0, compose_mode = 0, reflow_mode = 0, argi = 1;
  int i, result = EXIT_FAILURE;

  memset(&doc, 0, sizeof(doc));

  if (1 < argc && (0 == strcmp(argv[1], "-h") || 0 == strcmp(argv[1], "--help"))) {
    fprintf(stderr,
      "Usage: %s [-n N] [-g] [-r] [file...]\n"
      "  No arguments: read from stdin.\n"
      "  -n N: target sentence count (fusion) or phrase budget (compose).\n"
      "  -g: compose mode (first file = target, all files ingested).\n"
      "  -r: reflow text (join cosmetic line breaks).\n"
      "  Single file: summarize (iterative fusion).\n"
      "  Compose example: %s -g -r -n 32 target.txt corpus1.txt corpus2.txt\n",
      argv[0], argv[0]);
    return EXIT_FAILURE;
  }

  while (argi < argc && '-' == argv[argi][0] && '\0' != argv[argi][1]) {
    if (0 == strcmp(argv[argi], "-n") && argi + 1 < argc) {
      target_n = atoi(argv[argi + 1]);
      argi += 2;
    }
    else if (0 == strcmp(argv[argi], "-g")) {
      compose_mode = 1;
      ++argi;
    }
    else if (0 == strcmp(argv[argi], "-r")) {
      reflow_mode = 1;
      ++argi;
    }
    else break;
  }

  if (argi >= argc || (argi + 1 == argc && 0 == strcmp(argv[argi], "-"))) {
    result = read_input(&input, &input_size, stdin);
  }
  else if (0 != compose_mode) {
    FILE* f = fopen(argv[argi], "rb");
    if (NULL != f) {
      result = read_input(&input, &input_size, f);
      fclose(f);
    }
  }
  else if (argi + 1 == argc) {
    FILE* f = fopen(argv[argi], "rb");
    if (NULL != f) {
      result = read_input(&input, &input_size, f);
      fclose(f);
    }
    else {
      input = (unsigned char*)malloc(strlen(argv[argi]) + 1);
      if (NULL != input) {
        input_size = strlen(argv[argi]);
        memcpy(input, argv[argi], input_size);
        input[input_size] = 0;
        result = EXIT_SUCCESS;
      }
    }
  }
  else {
    size_t total = 0;
    for (i = argi; i < argc; ++i) total += strlen(argv[i]) + ((argi < i) ? 1 : 0);
    input = (unsigned char*)malloc(total + 1);
    if (NULL != input) {
      size_t offset = 0;
      for (i = argi; i < argc; ++i) {
        size_t length = strlen(argv[i]);
        if (argi < i) { input[offset] = ' '; ++offset; }
        memcpy(input + offset, argv[i], length);
        offset += length;
      }
      input[offset] = 0;
      input_size = total;
      result = EXIT_SUCCESS;
    }
  }

  if (EXIT_SUCCESS == result && 0 != reflow_mode && NULL != input) {
    unsigned char* reflowed = NULL;
    size_t reflowed_size = 0;
    if (EXIT_SUCCESS == libxs_text_reflow(input, input_size,
      &reflowed, &reflowed_size))
    {
      free(input);
      input = reflowed;
      input_size = reflowed_size;
    }
  }

  if (EXIT_SUCCESS == result && 0 != compose_mode) {
    libxs_registry_t* corpus = corpus_load();
    if (NULL == corpus) corpus = libxs_registry_create();
    if (NULL != corpus) {
      libxs_fprint_t target;
      size_t shape;
      int budget = (0 < target_n) ? target_n : 12;
      while (input_size > 0 && 0 != isspace(input[input_size - 1])) {
        --input_size;
      }
      shape = input_size;
      for (i = argi; i < argc; ++i) {
        corpus_ingest_file(corpus, argv[i], reflow_mode);
      }
      corpus_save(corpus);
      libxs_fprint(&target, LIBXS_DATATYPE_U8, input, 1,
        &shape, NULL, FPRINT_ORDER, 0, 0, 0);
      result = compose_document(corpus, &target, input, input_size, budget);
      libxs_registry_destroy(corpus);
    }
    else result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result) {
    while (input_size > 0 && 0 != isspace(input[input_size - 1])) --input_size;
    doc.text = input;
    doc.text_size = input_size;
    result = split_sentences(&doc);
    if (EXIT_SUCCESS == result) {
      result = fingerprint_sentences(&doc);
    }
    if (EXIT_SUCCESS == result) {
      result = summarize_document(&doc, target_n);
    }
  }

  free(doc.text);
  return result;
}


static int summarize_document(document_t* doc, int target_n)
{
  int result = EXIT_SUCCESS;
  int fuse_a = -1, fuse_b = -1, round = 0;
  char fused[16384];
  size_t fused_len = 0;
  int i, orig_words = 0, orig_sentences;

  fprintf(stderr, "sentences: %d\n", doc->nsentences);
  for (i = 0; i < doc->nsentences; ++i) {
    const sentence_t* s = doc->sentences + i;
    orig_words += count_words(doc->text + s->byte_offset, s->byte_length);
    fprintf(stderr, "  S%d: [%d..%d] \"%.*s\" decay=%.3f\n", i,
      s->byte_offset, s->byte_offset + s->byte_length,
      (s->byte_length > 60) ? 60 : s->byte_length,
      (const char*)doc->text + s->byte_offset,
      libxs_fprint_decay(&s->fprint));
  }
  orig_sentences = doc->nsentences;

  while (EXIT_SUCCESS == result && doc->nsentences >= 2
    && (0 == target_n || doc->nsentences > target_n))
  {
    fuse_a = -1;
    fuse_b = -1;
    if (EXIT_SUCCESS != find_best_fusion_pair(doc, &fuse_a, &fuse_b)) break;
    if (EXIT_SUCCESS != fuse_sentences(doc, fuse_a, fuse_b,
      fused, sizeof(fused), &fused_len)) break;

    ++round;
    if (0 != target_n) {
      fprintf(stderr, "  [round %d] S%d + S%d -> %d sentences\n",
        round, fuse_a, fuse_b, doc->nsentences - 1);
    }
    else {
      const sentence_t* sa = doc->sentences + fuse_a;
      const sentence_t* sb = doc->sentences + fuse_b;
      double red = sentence_redundancy(doc, fuse_a, fuse_b);
      fprintf(stderr, "\nfusion candidate: S%d + S%d (redundancy=%.3f)\n",
        fuse_a, fuse_b, red);
      fprintf(stderr, "  A: \"%.*s\"\n", sa->byte_length,
        (const char*)doc->text + sa->byte_offset);
      fprintf(stderr, "  B: \"%.*s\"\n", sb->byte_length,
        (const char*)doc->text + sb->byte_offset);
      fprintf(stderr, "  fused: \"%.*s\"\n", (int)fused_len, fused);
    }

    {
      unsigned char* new_text = NULL;
      size_t new_size = 0, pos = 0;
      int new_nsentences = 0;
      int fused_offset = 0, fused_total = 0;
      int tail_len = 0;
      const sentence_t* sa = doc->sentences + fuse_a;

      while (sa->byte_offset + sa->byte_length - tail_len > sa->byte_offset
        && 0 != isspace(doc->text[sa->byte_offset + sa->byte_length - 1 - tail_len]))
      {
        ++tail_len;
      }
      fused_total = (int)fused_len + tail_len;

      for (i = 0; i < doc->nsentences; ++i) {
        if (i == fuse_a) new_size += (size_t)fused_total;
        else if (i != fuse_b) new_size += (size_t)doc->sentences[i].byte_length;
      }
      new_text = (unsigned char*)malloc(new_size + 1);
      if (NULL == new_text) { result = EXIT_FAILURE; break; }

      for (i = 0; i < doc->nsentences; ++i) {
        if (i == fuse_a) {
          fused_offset = (int)pos;
          memcpy(new_text + pos, fused, fused_len);
          pos += fused_len;
          if (tail_len > 0) {
            memcpy(new_text + pos,
              doc->text + sa->byte_offset + sa->byte_length - tail_len,
              (size_t)tail_len);
            pos += (size_t)tail_len;
          }
        }
        else if (i != fuse_b) {
          const sentence_t* s = doc->sentences + i;
          memcpy(new_text + pos, doc->text + s->byte_offset,
            (size_t)s->byte_length);
          pos += (size_t)s->byte_length;
        }
      }
      new_text[pos] = 0;
      new_size = pos;

      free(doc->text);
      doc->text = new_text;
      doc->text_size = new_size;

      pos = 0;
      for (i = 0; i < doc->nsentences; ++i) {
        if (i == fuse_a) {
          sentence_t* s = doc->sentences + new_nsentences;
          s->byte_offset = fused_offset;
          s->byte_length = fused_total;
          ++new_nsentences;
          pos += (size_t)fused_total;
        }
        else if (i != fuse_b) {
          sentence_t* s = doc->sentences + new_nsentences;
          s->byte_offset = (int)pos;
          s->byte_length = doc->sentences[i].byte_length;
          s->fprint = doc->sentences[i].fprint;
          ++new_nsentences;
          pos += (size_t)doc->sentences[i].byte_length;
        }
      }
      doc->nsentences = new_nsentences;

      {
        const size_t shape = (size_t)doc->sentences[fuse_a > fuse_b
          ? fuse_a - 1 : fuse_a].byte_length;
        const int fi = (fuse_a > fuse_b) ? fuse_a - 1 : fuse_a;
        libxs_fprint(&doc->sentences[fi].fprint, LIBXS_DATATYPE_U8,
          doc->text + doc->sentences[fi].byte_offset, 1, &shape, NULL,
          FPRINT_ORDER, 0, 0, 0);
      }
    }

    if (0 == target_n) break;
  }

  if (EXIT_SUCCESS == result) {
    int si, word_budget;
    word_budget = (0 < target_n && orig_sentences > 0)
      ? (orig_words / orig_sentences) + 1 : 0;
    fprintf(stderr, "\nsummary (%d sentences, %d/%d words):\n",
      doc->nsentences, word_budget * doc->nsentences, orig_words);
    for (si = 0; si < doc->nsentences; ++si) {
      const sentence_t* s = doc->sentences + si;
      const unsigned char* src = doc->text + s->byte_offset;
      int slen = s->byte_length, j, prev_space = 1, wcount = 0;
      int last_punct = -1;
      while (slen > 0 && 0 != isspace(src[slen - 1])) --slen;
      for (j = 0; j < slen; ++j) {
        if (0 != isspace(src[j])) {
          if (0 == prev_space) prev_space = 1;
        }
        else {
          if (0 != prev_space) ++wcount;
          if (word_budget > 0 && wcount > word_budget) {
            if (last_punct > 0) slen = last_punct + 1;
            else slen = j;
            break;
          }
          if ('.' == src[j] || '!' == src[j] || '?' == src[j]) {
            last_punct = j;
          }
          prev_space = 0;
        }
      }
      prev_space = 1;
      while (slen > 0 && 0 != isspace(src[slen - 1])) --slen;
      for (j = 0; j < slen; ++j) {
        if (0 != isspace(src[j])) {
          if (0 == prev_space) { putchar(' '); prev_space = 1; }
        }
        else {
          putchar(src[j]);
          prev_space = 0;
        }
      }
      putchar('\n');
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


static int token_is_content(const libxs_token_t* token)
{
  int result = 0;
  if (NULL != token
    && 0 != (token->flags & (LIBXS_TOKEN_WORD | LIBXS_TOKEN_NUMBER))
    && 0 == (token->flags & LIBXS_TOKEN_STOP))
  {
    result = 1;
  }
  return result;
}


static int token_stream_has_id(const libxs_token_stream_t* stream,
  unsigned int id)
{
  int result = 0;
  size_t token_pos;
  if (NULL != stream && 0 != id) {
    for (token_pos = 0; token_pos < stream->size && 0 == result;
      ++token_pos)
    {
      if (stream->data[token_pos].id == id) result = 1;
    }
  }
  return result;
}


static double compose_lexical_score(const libxs_token_stream_t* target,
  const libxs_token_stream_t* candidate,
  const libxs_token_stream_t* generated)
{
  double covered = 0.0, total = 0.0;
  int repeated = 0, ncontent = 0;
  size_t token_pos;
  if (NULL == target || NULL == candidate) return 0.0;
  for (token_pos = 0; token_pos < target->size; ++token_pos) {
    const libxs_token_t* token = target->data + token_pos;
    if (0 != token_is_content(token)) {
      double weight = (token->length >= 6) ? 1.5 : 1.0;
      total += weight;
      if (0 != token_stream_has_id(candidate, token->id)) {
        covered += weight;
      }
    }
  }
  for (token_pos = 0; token_pos < candidate->size; ++token_pos) {
    const libxs_token_t* token = candidate->data + token_pos;
    if (0 != token_is_content(token)) {
      ++ncontent;
      if (NULL != generated
        && 0 != token_stream_has_id(generated, token->id))
      {
        ++repeated;
      }
    }
  }
  if (total > 0.0) covered /= total;
  if (ncontent > 0) covered -= 0.15 * ((double)repeated / (double)ncontent);
  return covered;
}


static int split_sentences(document_t* doc)
{
  int result = EXIT_FAILURE;
  size_t byte_start = 0;
  size_t i;
  if (NULL == doc || NULL == doc->text) return EXIT_FAILURE;
  doc->nsentences = 0;
  for (i = 0; i <= doc->text_size; ++i) {
    int is_end = (i == doc->text_size)
      ? 1 : is_sentence_end_text(doc->text, doc->text_size, i);
    if (0 != is_end && i > byte_start) {
      int end_pos = (int)((i < doc->text_size) ? i + 1 : i);
      int nwords;
      while (end_pos < (int)doc->text_size
        && 0 != isspace(doc->text[end_pos]))
      {
        ++end_pos;
      }
      nwords = count_words(doc->text + byte_start, end_pos - (int)byte_start);
      if (nwords < 3) continue;
      if (doc->nsentences < MAX_SENTENCES) {
        sentence_t* s = doc->sentences + doc->nsentences;
        s->byte_offset = (int)byte_start;
        s->byte_length = end_pos - (int)byte_start;
        ++doc->nsentences;
      }
      byte_start = end_pos;
      i = (byte_start > 0) ? byte_start - 1 : 0;
    }
  }
  result = (doc->nsentences > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
  return result;
}


static int fingerprint_sentences(document_t* doc)
{
  int result = EXIT_SUCCESS;
  int i;
  if (NULL == doc) return EXIT_FAILURE;
  for (i = 0; EXIT_SUCCESS == result && i < doc->nsentences; ++i) {
    sentence_t* s = doc->sentences + i;
    const size_t shape = (size_t)s->byte_length;
    result = libxs_fprint(&s->fprint, LIBXS_DATATYPE_U8,
      doc->text + s->byte_offset, 1, &shape, NULL,
      FPRINT_ORDER, 0, 0, 0);
  }
  return result;
}


static double sentence_redundancy(const document_t* doc, int a, int b)
{
  double result = 0.0;
  if (NULL != doc && a != b
    && 0 <= a && a < doc->nsentences
    && 0 <= b && b < doc->nsentences)
  {
    const sentence_t* sa = doc->sentences + a;
    const sentence_t* sb = doc->sentences + b;
    char buf_a[8192], buf_b[8192];
    int la = (sa->byte_length < (int)sizeof(buf_a) - 1)
      ? sa->byte_length : (int)sizeof(buf_a) - 1;
    int lb = (sb->byte_length < (int)sizeof(buf_b) - 1)
      ? sb->byte_length : (int)sizeof(buf_b) - 1;
    int word_count = 0, defect;
    memcpy(buf_a, doc->text + sa->byte_offset, (size_t)la);
    buf_a[la] = '\0';
    memcpy(buf_b, doc->text + sb->byte_offset, (size_t)lb);
    buf_b[lb] = '\0';
    defect = libxs_stridiff(buf_a, buf_b, NULL, 1, &word_count);
    if (word_count > 0 && defect >= 0) {
      result = 1.0 - (double)defect / (double)word_count;
    }
  }
  return result;
}


static double novelty_ratio(const document_t* doc, int a, int b)
{
  const sentence_t* sa = doc->sentences + a;
  const sentence_t* sb = doc->sentences + b;
  int words_b_off[128], words_b_len[128], nwords_b;
  int nnovel = 0, i;
  nwords_b = extract_words(doc->text + sb->byte_offset, sb->byte_length,
    words_b_off, words_b_len, 128);
  if (nwords_b <= 0) return 0.0;
  for (i = 0; i < nwords_b; ++i) {
    const unsigned char* w = doc->text + sb->byte_offset + words_b_off[i];
    int k, all_digit = 1;
    for (k = 0; k < words_b_len[i] && 0 != all_digit; ++k) {
      if (0 == isdigit(w[k])) all_digit = 0;
    }
    if (0 != all_digit) continue;
    if (0 == word_in_sentence(w, words_b_len[i],
      doc->text + sa->byte_offset, sa->byte_length))
    {
      ++nnovel;
    }
  }
  return (double)nnovel / (double)nwords_b;
}


static int find_best_fusion_pair(const document_t* doc, int* out_a, int* out_b)
{
  int result = EXIT_FAILURE;
  int i, best_i = -1, best_j = -1;
  double best_score = 1e30;
  if (NULL == doc || NULL == out_a || NULL == out_b) return EXIT_FAILURE;
  if (doc->nsentences < 2) return EXIT_FAILURE;

  for (i = 0; i < doc->nsentences - 1; ++i) {
    const sentence_t* sa = doc->sentences + i;
    const sentence_t* sb = doc->sentences + i + 1;
    double dist = libxs_fprint_diff(&sa->fprint, &sb->fprint, NULL);
    double score;
    if (doc->nsentences <= 256) {
      double nov = novelty_ratio(doc, i, i + 1);
      score = dist / (nov + 0.01);
    }
    else {
      score = dist;
    }
    if (score < best_score) {
      best_score = score;
      best_i = i;
      best_j = i + 1;
    }
  }

  if (best_i >= 0) {
    *out_a = best_i;
    *out_b = best_j;
    result = EXIT_SUCCESS;
  }
  return result;
}


static int extract_words(const unsigned char* text, int length,
  int* offsets, int* lengths, int max_words)
{
  int nwords = 0, i = 0;
  while (i < length && nwords < max_words) {
    while (i < length && (0 != isspace(text[i]) || 0 != ispunct(text[i]))) ++i;
    if (i < length) {
      int start = i;
      while (i < length && 0 == isspace(text[i]) && 0 == ispunct(text[i])) ++i;
      offsets[nwords] = start;
      lengths[nwords] = i - start;
      ++nwords;
    }
  }
  return nwords;
}


static int word_in_sentence(const unsigned char* word, int wlen,
  const unsigned char* sent, int slen)
{
  int off[128], len[128], nw, i;
  char wb[64], sb[64];
  int max_dist;
  if (wlen <= 0 || wlen >= (int)sizeof(wb)) return 0;
  if (wlen <= 3) return 0;
  memcpy(wb, word, (size_t)wlen);
  wb[wlen] = '\0';
  max_dist = (wlen > 5) ? 1 : 0;
  nw = extract_words(sent, slen, off, len, 128);
  for (i = 0; i < nw; ++i) {
    int sn = (len[i] < (int)sizeof(sb) - 1) ? len[i] : (int)sizeof(sb) - 1;
    int dist;
    memcpy(sb, sent + off[i], (size_t)sn);
    sb[sn] = '\0';
    dist = libxs_stridist(wb, sb);
    if (dist <= max_dist) return 1;
  }
  return 0;
}


static int fuse_sentences(const document_t* doc, int a, int b,
  char* output, size_t output_size, size_t* output_len)
{
  int result = EXIT_FAILURE;
  size_t pos = 0;
  const sentence_t* sa;
  const sentence_t* sb;
  int words_b_off[128], words_b_len[128], nwords_b;
  int novel[128], nnovel = 0;
  int i;
  if (NULL == doc || NULL == output || NULL == output_len) return EXIT_FAILURE;
  if (a < 0 || a >= doc->nsentences || b < 0 || b >= doc->nsentences) {
    return EXIT_FAILURE;
  }
  sa = doc->sentences + a;
  sb = doc->sentences + b;

  nwords_b = extract_words(doc->text + sb->byte_offset, sb->byte_length,
    words_b_off, words_b_len, 128);

  for (i = 0; i < nwords_b; ++i) {
    const unsigned char* w = doc->text + sb->byte_offset + words_b_off[i];
    int k, all_digit = 1;
    for (k = 0; k < words_b_len[i] && 0 != all_digit; ++k) {
      if (0 == isdigit(w[k])) all_digit = 0;
    }
    if (0 != all_digit) continue;
    if (words_b_len[i] < 4) continue;
    if (0 == word_in_sentence(w, words_b_len[i],
      doc->text + sa->byte_offset, sa->byte_length))
    {
      novel[nnovel++] = i;
    }
  }

  if (0 == nnovel) {
    if ((size_t)sa->byte_length < output_size) {
      memcpy(output, doc->text + sa->byte_offset, (size_t)sa->byte_length);
      *output_len = (size_t)sa->byte_length;
      result = EXIT_SUCCESS;
    }
  }
  else {
    int trim = sa->byte_length;
    for (;;) {
      int changed = 0;
      while (trim > 0 && 0 != isspace(doc->text[sa->byte_offset + trim - 1])) {
        --trim; changed = 1;
      }
      while (trim > 0 && ('.' == doc->text[sa->byte_offset + trim - 1]
        || '!' == doc->text[sa->byte_offset + trim - 1]
        || '?' == doc->text[sa->byte_offset + trim - 1]))
      {
        --trim; changed = 1;
      }
      if (0 == changed) break;
    }
    if (count_words(doc->text + sa->byte_offset, trim) < 3) {
      if ((size_t)sa->byte_length < output_size) {
        memcpy(output, doc->text + sa->byte_offset, (size_t)sa->byte_length);
        *output_len = (size_t)sa->byte_length;
        result = EXIT_SUCCESS;
      }
    }
    else if ((size_t)trim < output_size) {
      memcpy(output, doc->text + sa->byte_offset, (size_t)trim);
      pos = (size_t)trim;
    }

    if (pos > 0 && pos + 2 < output_size) {
      const unsigned char* sb_text = doc->text + sb->byte_offset;
      int span_start = words_b_off[novel[0]];
      int span_end = words_b_off[novel[nnovel - 1]]
        + words_b_len[novel[nnovel - 1]];
      int bridge_start, bridge_len;
      bridge_start = span_start;
      while (bridge_start > 0 && 0 != isspace(sb_text[bridge_start - 1])) {
        --bridge_start;
      }
      while (bridge_start > 0 && 0 != ispunct(sb_text[bridge_start - 1])) {
        --bridge_start;
      }
      if (bridge_start > 0 && 0 != ispunct(sb_text[bridge_start])) {
        bridge_len = span_end - bridge_start;
        if (pos + (size_t)bridge_len < output_size) {
          memcpy(output + pos, sb_text + bridge_start, (size_t)bridge_len);
          pos += (size_t)bridge_len;
        }
      }
      else {
        int slen = span_end - span_start;
        output[pos++] = ',';
        output[pos++] = ' ';
        if (pos + (size_t)slen < output_size) {
          memcpy(output + pos, sb_text + span_start, (size_t)slen);
          if (0 != isupper((unsigned char)output[pos])) {
            int wend = span_start + 1, has_internal_upper = 0;
            while (wend < span_end && 0 == isspace(sb_text[wend])
              && 0 == ispunct(sb_text[wend]))
            {
              if (0 != isupper(sb_text[wend])) has_internal_upper = 1;
              ++wend;
            }
            if (0 == has_internal_upper) {
              output[pos] = (char)tolower((unsigned char)output[pos]);
            }
          }
          pos += (size_t)slen;
        }
      }
    }

    if (pos > 0) {
      while (pos > 0 && 0 != isspace((unsigned char)output[pos - 1])) --pos;
      if (pos > 0 && '.' != output[pos - 1]
        && '!' != output[pos - 1] && '?' != output[pos - 1])
      {
        if (pos < output_size) output[pos++] = '.';
      }
      *output_len = pos;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int read_input(unsigned char** data, size_t* size, FILE* file)
{
  int result = EXIT_FAILURE;
  unsigned char* buffer = NULL;
  size_t used = 0, capacity = 0;
  int ch;
  if (NULL != data && NULL != size && NULL != file) {
    while (EOF != (ch = fgetc(file))) {
      if (used == capacity) {
        const size_t next_cap = (0 != capacity) ? (2 * capacity) : 256;
        unsigned char* next = (unsigned char*)realloc(buffer, next_cap + 1);
        if (NULL == next) { free(buffer); buffer = NULL; used = 0; break; }
        buffer = next;
        capacity = next_cap;
      }
      buffer[used++] = (unsigned char)ch;
    }
    if (NULL != buffer || 0 == used) {
      if (NULL == buffer) buffer = (unsigned char*)malloc(1);
      if (NULL != buffer) {
        buffer[used] = 0;
        *data = buffer;
        *size = used;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static void corpus_fixup(void* value, const void* key,
  size_t key_size, size_t value_size, void* udata)
{
  LIBXS_UNUSED(value); LIBXS_UNUSED(key);
  LIBXS_UNUSED(key_size); LIBXS_UNUSED(value_size);
  LIBXS_UNUSED(udata);
}


static libxs_registry_t* corpus_load(void)
{
  libxs_registry_t* corpus = NULL;
  FILE* f = fopen(CORPUS_FILE, "rb");
  if (NULL != f) {
    long len;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf && (long)fread(buf, 1, (size_t)len, f) == len) {
        corpus = libxs_registry_load(buf, (size_t)len, corpus_fixup, NULL);
      }
      free(buf);
    }
    fclose(f);
  }
  return corpus;
}


static int corpus_save(const libxs_registry_t* corpus)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
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




static int corpus_ingest_file(libxs_registry_t* corpus, const char* path,
  int reflow)
{
  int result = EXIT_FAILURE;
  FILE* f;
  unsigned char* text = NULL;
  size_t text_size = 0;
  LIBXS_UNUSED(reflow);
  if (NULL == corpus || NULL == path) return EXIT_FAILURE;
  f = fopen(path, "rb");
  if (NULL != f) {
    result = read_input(&text, &text_size, f);
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
    int nregistered = 0;
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
              unsigned int seq;
              const size_t shape = (size_t)len;
              memset(&entry, 0, sizeof(entry));
              if (EXIT_SUCCESS == libxs_fprint(&entry.fprint,
                LIBXS_DATATYPE_U8, text + sent_start, 1,
                &shape, NULL, FPRINT_ORDER, 0, 0, 0))
              {
                entry.text_len = len;
                memcpy(entry.text, text + sent_start, (size_t)len);
                entry.connector = CONN_NEWLINE;
                entry.scale = SCALE_SENTENCE;
                corpus_key_from_fprint(&entry.fprint, key, &key_size);
                for (seq = 0; seq < 65536; ++seq) {
                  unsigned short seq_key = (unsigned short)seq;
                  const corpus_entry_t* old_entry;
                  memcpy(key + 8, &seq_key, 2);
                  key_size = 10;
                  old_entry = (const corpus_entry_t*)libxs_registry_get(
                    corpus, key, key_size, NULL);
                  if (NULL == old_entry) {
                    libxs_registry_set(corpus, key, key_size,
                      &entry, sizeof(entry), NULL);
                    ++nregistered;
                    break;
                  }
                  if (old_entry->text_len == entry.text_len
                    && 0 == libxs_memcmp(old_entry->text, entry.text,
                      (size_t)entry.text_len)) break;
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
    fprintf(stderr, "  ingested %s: %d sentences\n", path, nregistered);
  }
  free(text);
  return result;
}


static void emit_connector(unsigned char connector, char* output,
  size_t output_size, size_t* pos)
{
  switch ((int)connector) {
    case CONN_COMMA:
      if (*pos + 2 < output_size) {
        output[(*pos)++] = ',';
        output[(*pos)++] = ' ';
      }
      break;
    case CONN_PERIOD:
      if (*pos + 2 < output_size) {
        output[(*pos)++] = '.';
        output[(*pos)++] = ' ';
      }
      break;
    case CONN_NEWLINE:
      if (*pos + 1 < output_size) {
        output[(*pos)++] = '\n';
      }
      break;
    default:
      if (*pos + 1 < output_size) {
        output[(*pos)++] = ' ';
      }
      break;
  }
}


static double compose_distance(const libxs_fprint_t* a,
  const libxs_fprint_t* b)
{
  double acc = 0;
  int kmax = a->order < b->order ? a->order : b->order;
  int k;
  for (k = 0; k <= kmax; ++k) {
    double va = (a->nk[k] > 0) ? a->acc_sq[k] / a->nk[k] : 0;
    double vb = (b->nk[k] > 0) ? b->acc_sq[k] / b->nk[k] : 0;
    double ma = (a->nk[k] > 0) ? a->acc_sum[k] / a->nk[k] : 0;
    double mb = (b->nk[k] > 0) ? b->acc_sum[k] / b->nk[k] : 0;
    double wk = 1.0;
    double dv, dm;
    if (k > 0) { int i; for (i = 1; i <= k; ++i) wk /= i; }
    dv = va - vb;
    dm = ma - mb;
    acc += wk * (dv * dv + dm * dm);
  }
  return sqrt(acc);
}


static int compose_document(const libxs_registry_t* corpus,
  const libxs_fprint_t* target, const unsigned char* target_text,
  size_t target_size, int budget)
{
  int result = EXIT_SUCCESS;
  libxs_fprint_t running;
  libxs_lexicon_t* lexicon = NULL;
  libxs_token_stream_t target_tokens;
  libxs_token_stream_t generated_tokens;
  libxs_lexrule_t rules[96];
  char output[65536];
  size_t out_pos = 0;
  double prev_dist = 1e30;
  int nrules = 0;
  int step;
  libxs_registry_info_t rinfo;

  libxs_token_stream_init(&target_tokens);
  libxs_token_stream_init(&generated_tokens);
  if (NULL == corpus || NULL == target) return EXIT_FAILURE;
  memset(&rinfo, 0, sizeof(rinfo));
  libxs_registry_info(corpus, &rinfo);
  if (0 == rinfo.size) {
    fprintf(stderr, "corpus is empty (run without -g first to ingest)\n");
    return EXIT_FAILURE;
  }

  lexicon = libxs_lexicon_create();
  nrules = libxs_lexrule_defaults(rules, 96);
  if (NULL == lexicon || nrules <= 0
    || NULL == target_text || 0 == target_size
    || EXIT_SUCCESS != libxs_token_stream_encode(lexicon, &target_tokens,
      target_text, target_size, rules, nrules, NULL, 0, 1))
  {
    libxs_lexicon_destroy(lexicon);
    libxs_token_stream_release(&target_tokens);
    libxs_token_stream_release(&generated_tokens);
    return EXIT_FAILURE;
  }

  memset(&running, 0, sizeof(running));
  fprintf(stderr, "compose: target decay=%.3f, corpus=%lu entries, budget=%d, tokens=%lu\n",
    libxs_fprint_decay(target), (unsigned long)rinfo.size, budget,
    (unsigned long)target_tokens.size);

  { unsigned char prev_conn = CONN_PERIOD;
    const corpus_entry_t* used[256];
    int nused = 0;
    libxs_fprint_t prev_fprint;
    int have_prev = 0;
    double alpha = 0.3;
    for (step = 0; step < budget; ++step) {
      const corpus_entry_t* best_entry = NULL;
      double best_score = 1e30;
      double best_lex_score = 0.0;
      size_t cursor = 0;
      const void* iter_key = NULL;
      void* iter_val;

      iter_val = libxs_registry_begin(corpus, &iter_key, &cursor);
      while (NULL != iter_val) {
        const corpus_entry_t* e = (const corpus_entry_t*)iter_val;
        if (e->text_len >= 8) {
          int skip = 0, u;
          for (u = 0; u < nused && 0 == skip; ++u) {
            if (used[u]->text_len == e->text_len
              && 0 == libxs_memcmp(used[u]->text, e->text, (size_t)e->text_len))
            {
              skip = 1;
            }
          }
          if (0 == skip) {
            libxs_fprint_t trial;
            libxs_token_stream_t candidate_tokens;
            double d, score, lex_score = 0.0;
            trial = running;
            libxs_token_stream_init(&candidate_tokens);
            libxs_fprint_partial(&trial, LIBXS_DATATYPE_U8,
              e->text, e->text_len, FPRINT_ORDER);
            d = compose_distance(&trial, target);
            score = d;
            if (EXIT_SUCCESS == libxs_token_stream_encode(lexicon,
              &candidate_tokens, (const unsigned char*)e->text,
              (size_t)e->text_len, rules, nrules, NULL, 0, 1))
            {
              lex_score = compose_lexical_score(&target_tokens,
                &candidate_tokens, &generated_tokens);
              score *= (1.0 - 0.25 * lex_score);
            }
            if (0 != have_prev) {
              score += alpha * compose_distance(&e->fprint, &prev_fprint);
            }
            if (score < best_score) {
              best_score = score;
              best_lex_score = lex_score;
              best_entry = e;
            }
            libxs_token_stream_release(&candidate_tokens);
          }
        }
        iter_val = libxs_registry_next(corpus, &iter_key, &cursor);
      }

      if (NULL == best_entry) break;
      if (nused < 256) used[nused++] = best_entry;
      if (best_score < prev_dist) prev_dist = best_score;

      { const char* txt = best_entry->text;
        int tlen = best_entry->text_len;
        while (tlen > 0 && 0 != isspace((unsigned char)*txt)) {
          ++txt; --tlen;
        }
        while (tlen > 0 && 0 != isspace((unsigned char)txt[tlen - 1])) {
          --tlen;
        }
        if (tlen <= 0) continue;
        if (out_pos > 0 && out_pos < sizeof(output)) {
          emit_connector(prev_conn, output, sizeof(output), &out_pos);
        }
        if (out_pos + (size_t)tlen < sizeof(output)) {
          memcpy(output + out_pos, txt, (size_t)tlen);
          out_pos += (size_t)tlen;
        }
      }

      libxs_fprint_partial(&running, LIBXS_DATATYPE_U8,
        best_entry->text, best_entry->text_len, FPRINT_ORDER);
      libxs_token_stream_encode(lexicon, &generated_tokens,
        (const unsigned char*)best_entry->text, (size_t)best_entry->text_len,
        rules, nrules, NULL, 0, 1);
      prev_fprint = best_entry->fprint;
      have_prev = 1;
      prev_conn = best_entry->connector;

      fprintf(stderr, "  [%d] score=%.4f lex=%.3f \"%.*s\"\n",
        step + 1, best_score, best_lex_score,
        (best_entry->text_len > 60) ? 60 : best_entry->text_len,
        best_entry->text);
    }
  }

  if (out_pos > 0) {
    fprintf(stderr, "\ncomposed (%d sentences):\n", step);
    printf("%.*s\n", (int)out_pos, output);
  }
  libxs_token_stream_release(&generated_tokens);
  libxs_token_stream_release(&target_tokens);
  libxs_lexicon_destroy(lexicon);
  return result;
}
