#include <libxs/libxs_predict.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_mem.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SENTENCES 256
#define MAX_PHRASES 2048
#define FPRINT_ORDER 4
#define FUSION_NINPUTS 7
#define MODEL_FILE "summarize.bin"
#define MODEL_QUALITY 0.8


typedef struct sentence_t {
  int token_start;
  int token_count;
  int byte_offset;
  int byte_length;
  libxs_fprint_t fprint;
} sentence_t;

typedef struct phrase_t {
  int token_start;
  int token_count;
  int byte_offset;
  int byte_length;
  int sentence;
} phrase_t;

typedef struct document_t {
  unsigned char* text;
  size_t text_size;
  libxs_token_stream_t stream;
  sentence_t sentences[MAX_SENTENCES];
  int nsentences;
  phrase_t phrases[MAX_PHRASES];
  int nphrases;
  int* byte_offsets;
} document_t;


static int extract_words(const unsigned char* text, int length,
  int* offsets, int* lengths, int max_words);
static int word_in_sentence(const unsigned char* word, int wlen,
  const unsigned char* sent, int slen);
static libxs_predict_t* model_load(void);
static int model_save(const libxs_predict_t* model);
static int build_byte_offsets(const document_t* doc, int** offsets);
static int split_sentences(document_t* doc);
static int split_phrases(document_t* doc);
static int fingerprint_sentences(document_t* doc);
static double sentence_redundancy(const document_t* doc, int a, int b);
static int find_best_fusion_pair(const document_t* doc, int* out_a, int* out_b,
  int use_model_io);
static int fuse_sentences(const document_t* doc, int a, int b,
  char* output, size_t output_size, size_t* output_len);
static int read_input(unsigned char** data, size_t* size, FILE* file);


static int summarize_document(document_t* doc, int target_n);


int main(int argc, char* argv[])
{
  unsigned char* input = NULL;
  size_t input_size = 0;
  document_t doc;
  int target_n = 0, argi = 1;
  int i, result = EXIT_FAILURE;

  memset(&doc, 0, sizeof(doc));

  if (1 < argc && (0 == strcmp(argv[1], "-h") || 0 == strcmp(argv[1], "--help"))) {
    fprintf(stderr,
      "Usage: %s [-n sentences] [file|text...]\n"
      "  No arguments: read from stdin.\n"
      "  -n N: reduce to at most N sentences (iterative fusion).\n"
      "  Single file argument: read file contents.\n"
      "  Otherwise: concatenate arguments as literal text.\n",
      argv[0]);
    return EXIT_FAILURE;
  }

  if (argi < argc && 0 == strcmp(argv[argi], "-n")) {
    if (argi + 1 < argc) {
      target_n = atoi(argv[argi + 1]);
      argi += 2;
    }
  }

  if (argi >= argc || (argi + 1 == argc && 0 == strcmp(argv[argi], "-"))) {
    result = read_input(&input, &input_size, stdin);
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

  if (EXIT_SUCCESS == result) {
    while (input_size > 0 && 0 != isspace(input[input_size - 1])) --input_size;
    doc.text = input;
    doc.text_size = input_size;
    result = libxs_tokenize(input, input_size, &doc.stream);
  }
  if (EXIT_SUCCESS == result) {
    result = build_byte_offsets(&doc, &doc.byte_offsets);
  }
  if (EXIT_SUCCESS == result) {
    result = split_sentences(&doc);
  }
  if (EXIT_SUCCESS == result) {
    result = split_phrases(&doc);
  }
  if (EXIT_SUCCESS == result) {
    result = fingerprint_sentences(&doc);
  }
  if (EXIT_SUCCESS == result) {
    result = summarize_document(&doc, target_n);
  }

  free(doc.byte_offsets);
  libxs_token_stream_destroy(&doc.stream);
  free(doc.text);
  return result;
}


static int summarize_document(document_t* doc, int target_n)
{
  int result = EXIT_SUCCESS;
  int fuse_a = -1, fuse_b = -1, round = 0;
  char fused[16384];
  size_t fused_len = 0;
  int i;

  printf("sentences: %d, phrases: %d, tokens: %lu\n",
    doc->nsentences, doc->nphrases, (unsigned long)doc->stream.size);
  for (i = 0; i < doc->nsentences; ++i) {
    const sentence_t* s = doc->sentences + i;
    printf("  S%d: [%d..%d] \"%.*s\" decay=%.3f\n", i,
      s->byte_offset, s->byte_offset + s->byte_length,
      (s->byte_length > 60) ? 60 : s->byte_length,
      (const char*)doc->text + s->byte_offset,
      libxs_fprint_decay(&s->fprint));
  }

  while (EXIT_SUCCESS == result && doc->nsentences >= 2
    && (0 == target_n || doc->nsentences > target_n))
  {
    fuse_a = -1;
    fuse_b = -1;
    if (EXIT_SUCCESS != find_best_fusion_pair(doc, &fuse_a, &fuse_b,
      (0 == round) ? 1 : 0)) break;
    if (EXIT_SUCCESS != fuse_sentences(doc, fuse_a, fuse_b,
      fused, sizeof(fused), &fused_len)) break;

    ++round;
    if (0 != target_n) {
      printf("  [round %d] S%d + S%d -> %d sentences\n",
        round, fuse_a, fuse_b, doc->nsentences - 1);
    }
    else {
      const sentence_t* sa = doc->sentences + fuse_a;
      const sentence_t* sb = doc->sentences + fuse_b;
      double red = sentence_redundancy(doc, fuse_a, fuse_b);
      printf("\nfusion candidate: S%d + S%d (redundancy=%.3f)\n",
        fuse_a, fuse_b, red);
      printf("  A: \"%.*s\"\n", sa->byte_length,
        (const char*)doc->text + sa->byte_offset);
      printf("  B: \"%.*s\"\n", sb->byte_length,
        (const char*)doc->text + sb->byte_offset);
      printf("  fused: \"%.*s\"\n", (int)fused_len, fused);
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
          s->token_start = 0;
          s->token_count = 0;
          s->byte_offset = fused_offset;
          s->byte_length = fused_total;
          ++new_nsentences;
          pos += (size_t)fused_total;
        }
        else if (i != fuse_b) {
          sentence_t* s = doc->sentences + new_nsentences;
          s->byte_offset = (int)pos;
          s->byte_length = doc->sentences[i].byte_length;
          s->token_start = 0;
          s->token_count = 0;
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
    const char* t = (const char*)doc->text;
    int tl = (int)doc->text_size;
    while (tl > 0 && 0 != isspace((unsigned char)*t)) { ++t; --tl; }
    while (tl > 0 && 0 != isspace((unsigned char)t[tl - 1])) --tl;
    printf("\nsummary (%d sentences):\n%.*s\n", doc->nsentences, tl, t);
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


static int split_sentences(document_t* doc)
{
  int result = EXIT_FAILURE;
  int sent_start = 0, byte_start = 0;
  size_t i;
  int byte_pos = 0;
  if (NULL == doc || NULL == doc->byte_offsets) return EXIT_FAILURE;
  doc->nsentences = 0;
  for (i = 0; i < doc->stream.size; ++i) {
    const libxs_token_t* token = doc->stream.data + i;
    const int tlen = (int)libxs_token_len(token);
    byte_pos = doc->byte_offsets[i];
    if (0 != libxs_token_is_sentence_end(token)) {
      int end_pos = byte_pos + tlen;
      int nwords;
      while ((int)(i + 1) < (int)doc->stream.size) {
        const libxs_token_t* next = doc->stream.data + i + 1;
        const int nlen = (int)libxs_token_len(next);
        const int noff = doc->byte_offsets[i + 1];
        int k, all_space = 1;
        if (0 != libxs_token_is_copy(next)) break;
        for (k = 0; k < nlen && 0 != all_space; ++k) {
          if (0 == isspace(next->raw[1 + k])) all_space = 0;
        }
        if (0 == all_space) break;
        ++i;
        end_pos = noff + nlen;
      }
      nwords = count_words(doc->text + byte_start, end_pos - byte_start);
      if (nwords < 3) continue;
      if (doc->nsentences < MAX_SENTENCES) {
        sentence_t* s = doc->sentences + doc->nsentences;
        s->token_start = sent_start;
        s->token_count = (int)(i + 1) - sent_start;
        s->byte_offset = byte_start;
        s->byte_length = end_pos - byte_start;
        ++doc->nsentences;
      }
      sent_start = (int)(i + 1);
      byte_start = end_pos;
    }
  }
  if (sent_start < (int)doc->stream.size && doc->nsentences < MAX_SENTENCES) {
    sentence_t* s = doc->sentences + doc->nsentences;
    const int last_offset = doc->byte_offsets[doc->stream.size - 1];
    const int last_len = (int)libxs_token_len(doc->stream.data + doc->stream.size - 1);
    int total_len = (last_offset + last_len) - byte_start;
    if (total_len > 0) {
      s->token_start = sent_start;
      s->token_count = (int)doc->stream.size - sent_start;
      s->byte_offset = byte_start;
      s->byte_length = total_len;
      ++doc->nsentences;
    }
  }
  result = (doc->nsentences > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
  return result;
}


static int split_phrases(document_t* doc)
{
  int result = EXIT_FAILURE;
  int si;
  if (NULL == doc) return EXIT_FAILURE;
  doc->nphrases = 0;
  for (si = 0; si < doc->nsentences; ++si) {
    const sentence_t* sent = doc->sentences + si;
    int phrase_start = sent->token_start;
    int phrase_byte = sent->byte_offset;
    int ti;
    for (ti = sent->token_start; ti < sent->token_start + sent->token_count; ++ti) {
      const libxs_token_t* token = doc->stream.data + ti;
      const int tlen = (int)libxs_token_len(token);
      const int toff = doc->byte_offsets[ti];
      if (0 != libxs_token_has_break(token) && ti > phrase_start) {
        if (doc->nphrases < MAX_PHRASES) {
          phrase_t* p = doc->phrases + doc->nphrases;
          p->token_start = phrase_start;
          p->token_count = (ti + 1) - phrase_start;
          p->byte_offset = phrase_byte;
          p->byte_length = (toff + tlen) - phrase_byte;
          p->sentence = si;
          ++doc->nphrases;
        }
        phrase_start = ti + 1;
        phrase_byte = toff + tlen;
      }
    }
    if (phrase_start < sent->token_start + sent->token_count
      && doc->nphrases < MAX_PHRASES)
    {
      phrase_t* p = doc->phrases + doc->nphrases;
      const int last_ti = sent->token_start + sent->token_count - 1;
      const int last_off = doc->byte_offsets[last_ti];
      const int last_len = (int)libxs_token_len(doc->stream.data + last_ti);
      p->token_start = phrase_start;
      p->token_count = (last_ti + 1) - phrase_start;
      p->byte_offset = phrase_byte;
      p->byte_length = (last_off + last_len) - phrase_byte;
      p->sentence = si;
      ++doc->nphrases;
    }
  }
  result = (doc->nphrases > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
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


static void pair_features(const document_t* doc, int a, int b, double feat[])
{
  const sentence_t* sa = doc->sentences + a;
  const sentence_t* sb = doc->sentences + b;
  const double red = sentence_redundancy(doc, a, b);
  const double fp_dist = libxs_fprint_diff(&sa->fprint, &sb->fprint, NULL);
  const double decay_a = libxs_fprint_decay(&sa->fprint);
  const double decay_b = libxs_fprint_decay(&sb->fprint);
  const int la = sa->byte_length > 0 ? sa->byte_length : 1;
  const int lb = sb->byte_length > 0 ? sb->byte_length : 1;
  feat[0] = red;
  feat[1] = fp_dist;
  feat[2] = decay_a;
  feat[3] = decay_b;
  feat[4] = (double)la / (double)lb;
  feat[5] = (la + lb > 0) ? (double)(la < lb ? la : lb) / (double)(la + lb) : 0.5;
  feat[6] = novelty_ratio(doc, a, b);
}


static double auto_label_fusion(const document_t* doc, int a, int b)
{
  double quality = 0.0;
  const sentence_t* sa = doc->sentences + a;
  const sentence_t* sb = doc->sentences + b;
  int words_b_off[128], words_b_len[128], nwords_b;
  int nmatched = 0, nnovel = 0, i;
  nwords_b = extract_words(doc->text + sb->byte_offset, sb->byte_length,
    words_b_off, words_b_len, 128);
  for (i = 0; i < nwords_b; ++i) {
    const unsigned char* w = doc->text + sb->byte_offset + words_b_off[i];
    if (0 != word_in_sentence(w, words_b_len[i],
      doc->text + sa->byte_offset, sa->byte_length))
    {
      ++nmatched;
    }
    else {
      ++nnovel;
    }
  }
  if (nwords_b > 0 && nmatched > 0 && nnovel > 0) {
    const double overlap = (double)nmatched / (double)nwords_b;
    const double novelty = (double)nnovel / (double)nwords_b;
    quality = overlap * novelty;
  }
  return quality;
}


static libxs_predict_t* model_load(void)
{
  libxs_predict_t* model = NULL;
  FILE* f = fopen(MODEL_FILE, "rb");
  if (NULL != f) {
    long len;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len > 0) {
      void* buf = malloc((size_t)len);
      if (NULL != buf && (long)fread(buf, 1, (size_t)len, f) == len) {
        model = libxs_predict_load(buf, (size_t)len);
      }
      free(buf);
    }
    fclose(f);
  }
  return model;
}


static int model_save(const libxs_predict_t* model)
{
  int result = EXIT_FAILURE;
  size_t size = 0;
  if (EXIT_SUCCESS == libxs_predict_save(model, NULL, &size) && size > 0) {
    void* buf = malloc(size);
    if (NULL != buf) {
      if (EXIT_SUCCESS == libxs_predict_save(model, buf, &size)) {
        FILE* f = fopen(MODEL_FILE, "wb");
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


static int find_best_fusion_pair(const document_t* doc, int* out_a, int* out_b,
  int use_model_io)
{
  int result = EXIT_FAILURE;
  libxs_predict_t* prior;
  libxs_predict_t* model;
  int i, j, npairs = 0;
  double best_score = -1.0;
  if (NULL == doc || NULL == out_a || NULL == out_b) return EXIT_FAILURE;
  if (doc->nsentences < 2) return EXIT_FAILURE;

  prior = (0 != use_model_io) ? model_load() : NULL;
  if (NULL != prior) {
    for (i = 0; i < doc->nsentences; ++i) {
      const int wi = count_words(doc->text + doc->sentences[i].byte_offset,
        doc->sentences[i].byte_length);
      if (wi < 6) continue;
      for (j = i + 1; j < doc->nsentences; ++j) {
        double feat[FUSION_NINPUTS], predicted = 0.0;
        const int wj = count_words(doc->text + doc->sentences[j].byte_offset,
          doc->sentences[j].byte_length);
        if (wj < 6) continue;
        pair_features(doc, i, j, feat);
        if (feat[6] < 0.1) continue;
        libxs_predict_eval(NULL, prior, feat, &predicted, NULL, 0);
        if (predicted > best_score && predicted > 0.01) {
          best_score = predicted;
          *out_a = i;
          *out_b = j;
          result = EXIT_SUCCESS;
        }
      }
    }
    libxs_predict_destroy(prior);
  }

  model = libxs_predict_create(FUSION_NINPUTS, 1);
  if (NULL != model) {
    for (i = 0; i < doc->nsentences; ++i) {
      for (j = i + 1; j < doc->nsentences; ++j) {
        double feat[FUSION_NINPUTS], quality;
        pair_features(doc, i, j, feat);
        quality = auto_label_fusion(doc, i, j);
        libxs_predict_push(NULL, model, feat, &quality);
        ++npairs;
      }
    }
    if (npairs >= 3
      && EXIT_SUCCESS == libxs_predict_build(model, 0, 0, MODEL_QUALITY))
    {
      if (EXIT_FAILURE == result) {
        for (i = 0; i < doc->nsentences; ++i) {
          const int wi = count_words(doc->text + doc->sentences[i].byte_offset,
            doc->sentences[i].byte_length);
          if (wi < 6) continue;
          for (j = i + 1; j < doc->nsentences; ++j) {
            double feat[FUSION_NINPUTS], predicted = 0.0;
            const int wj = count_words(
              doc->text + doc->sentences[j].byte_offset,
              doc->sentences[j].byte_length);
            if (wj < 6) continue;
            pair_features(doc, i, j, feat);
            if (feat[6] < 0.1) continue;
            libxs_predict_eval(NULL, model, feat, &predicted, NULL, 0);
            if (predicted > best_score && predicted > 0.01) {
              best_score = predicted;
              *out_a = i;
              *out_b = j;
              result = EXIT_SUCCESS;
            }
          }
        }
      }
      if (0 != use_model_io) model_save(model);
    }
    libxs_predict_destroy(model);
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


static int build_byte_offsets(const document_t* doc, int** offsets)
{
  int result = EXIT_FAILURE;
  if (NULL != doc && NULL != offsets && doc->stream.size > 0) {
    int* table = (int*)malloc(doc->stream.size * sizeof(int));
    if (NULL != table) {
      size_t i, byte_pos = 0;
      for (i = 0; i < doc->stream.size; ++i) {
        table[i] = (int)byte_pos;
        byte_pos += libxs_token_len(doc->stream.data + i);
      }
      *offsets = table;
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


