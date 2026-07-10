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


enum { CONN_SPACE = 0, CONN_NEWLINE = 3 };
enum { SCALE_PHRASE = 0, SCALE_SENTENCE = 1, SCALE_PARAGRAPH = 2 };

typedef struct corpus_entry_t {
  libxs_fprint_t fprint;
  int text_len;
  unsigned char connector;
  unsigned char scale;
  char text[COMPOSE_MAXTEXT];
} corpus_entry_t;


static void corpus_key_from_fprint(const libxs_fprint_t* fp,
  unsigned char key[], size_t* key_size);
static libxs_registry_t* corpus_load(void);
static int corpus_save(const libxs_registry_t* corpus);
static int corpus_ingest_file(libxs_registry_t* corpus, const char* path);
static int count_words(const unsigned char* text, int length);


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


static int respond(const libxs_spatial_t* spatial, const libxs_fprint_t* query,
  int budget)
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

  ncandidates = libxs_spatial_nearest(spatial, qcode, 256, candidates);

  for (step = 0; step < budget; ++step) {
    const corpus_entry_t* best_entry = NULL;
    double best_score = 1e30;
    int ci;

    for (ci = 0; ci < ncandidates; ++ci) {
      const corpus_entry_t* e = (const corpus_entry_t*)candidates[ci];
      if (e->text_len >= 40 && e->scale == preferred_scale) {
        int skip = 0, u;
        for (u = 0; u < nused && 0 == skip; ++u) {
          if (used[u]->text_len == e->text_len
            && 0 == memcmp(used[u]->text, e->text, (size_t)e->text_len))
          {
            skip = 1;
          }
        }
        if (0 == skip) {
          double score = converse_score(&e->fprint, query,
            (0 != have_prev) ? &prev_fprint : NULL);
          if (score < best_score) {
            best_score = score;
            best_entry = e;
          }
        }
      }
    }

    if (NULL == best_entry) {
      for (ci = 0; ci < ncandidates; ++ci) {
        const corpus_entry_t* e = (const corpus_entry_t*)candidates[ci];
        if (e->text_len >= 40) {
          int skip = 0, u;
          for (u = 0; u < nused && 0 == skip; ++u) {
            if (used[u]->text_len == e->text_len
              && 0 == memcmp(used[u]->text, e->text, (size_t)e->text_len))
            {
              skip = 1;
            }
          }
          if (0 == skip) {
            double score = converse_score(&e->fprint, query,
              (0 != have_prev) ? &prev_fprint : NULL);
            if (score < best_score) {
              best_score = score;
              best_entry = e;
            }
          }
        }
      }
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


int main(int argc, char* argv[])
{
  libxs_registry_t* corpus;
  int i, budget = RESPONSE_BUDGET;
  libxs_registry_info_t rinfo;
  char line[4096];

  if (argc < 2) {
    fprintf(stderr,
      "Usage: %s [-n N] corpus1.txt [corpus2.txt ...]\n"
      "  Interactive conversational agent.\n"
      "  -n N: response sentence budget (default %d).\n",
      argv[0], RESPONSE_BUDGET);
    return EXIT_FAILURE;
  }

  i = 1;
  if (i < argc && 0 == strcmp(argv[i], "-n") && i + 1 < argc) {
    budget = atoi(argv[i + 1]);
    i += 2;
  }

  corpus = corpus_load();
  if (NULL == corpus) corpus = libxs_registry_create();
  if (NULL == corpus) return EXIT_FAILURE;

  for (; i < argc; ++i) {
    corpus_ingest_file(corpus, argv[i]);
  }
  corpus_save(corpus);

  libxs_registry_info(corpus, &rinfo);
  fprintf(stderr, "corpus: %lu sentences\n", (unsigned long)rinfo.size);

  { libxs_spatial_t spatial;
    if (EXIT_SUCCESS != libxs_spatial_build(&spatial, corpus)) {
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
      respond(&spatial, &query, budget);
      fprintf(stderr, "> ");
    }
    libxs_spatial_destroy(&spatial);
  }

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


static int corpus_ingest_file(libxs_registry_t* corpus, const char* path)
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
    libxs_token_stream_t stream;
    libxs_textrule_t rules[32];
    int nrules;
    libxs_token_stream_init(&stream);
    nrules = libxs_textrule_defaults(rules, 32);
    result = libxs_token_stream_encode(&stream, text, text_size);
    if (EXIT_SUCCESS == result) {
      size_t byte_pos = 0, sent_start = 0;
      int nsentences = 0, nparagraphs = 0;
      size_t i;
      for (i = 0; i <= stream.size; ++i) {
        int is_sent_end = 0, tlen = 0;
        if (i < stream.size) {
          const libxs_token_t* t = stream.data + i;
          libxs_textrule_ctx_t ctx;
          tlen = (int)libxs_token_len(t);
          ctx.text = text;
          ctx.text_size = text_size;
          ctx.byte_pos = (int)byte_pos + tlen - 1;
          ctx.token = t;
          ctx.prev_token = (i > 0) ? stream.data + i - 1 : NULL;
          is_sent_end = libxs_textrule_eval(&ctx, rules, nrules);
        }
        else {
          is_sent_end = 1;
        }
        if (0 != is_sent_end && i > 0) {
          size_t sent_end = byte_pos + (size_t)tlen;
          int len = (int)(sent_end - sent_start);
          while (len > 0 && 0 != isspace(text[sent_start + len - 1])) --len;
          if (len > 8 && len < COMPOSE_MAXTEXT) {
            int nwords = count_words(text + sent_start, len);
            if (nwords >= 3) {
              corpus_entry_t entry;
              unsigned char key[16];
              size_t key_size = 0;
              unsigned short seq;
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
                for (seq = 0; seq < 256; ++seq) {
                  memcpy(key + 8, &seq, 2);
                  key_size = 10;
                  if (NULL == libxs_registry_get(corpus, key,
                    key_size, NULL))
                  {
                    libxs_registry_set(corpus, key, key_size,
                      &entry, sizeof(entry), NULL);
                    ++nsentences;
                    break;
                  }
                }
              }
            }
          }
          sent_start = sent_end;
        }
        byte_pos += (size_t)tlen;
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
                const size_t shape = (size_t)plen;
                memset(&entry, 0, sizeof(entry));
                if (EXIT_SUCCESS == libxs_fprint(&entry.fprint,
                  LIBXS_DATATYPE_U8, text + para_start, 1,
                  &shape, NULL, FPRINT_ORDER, 0, 0, 0))
                {
                  entry.text_len = plen;
                  memcpy(entry.text, text + para_start, (size_t)plen);
                  entry.connector = CONN_NEWLINE;
                  entry.scale = SCALE_PARAGRAPH;
                  corpus_key_from_fprint(&entry.fprint, key, &key_size);
                  for (seq = 0; seq < 256; ++seq) {
                    memcpy(key + 8, &seq, 2);
                    key_size = 10;
                    if (NULL == libxs_registry_get(corpus, key,
                      key_size, NULL))
                    {
                      libxs_registry_set(corpus, key, key_size,
                        &entry, sizeof(entry), NULL);
                      ++nparagraphs;
                      break;
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
    libxs_token_stream_release(&stream);
  }
  free(text);
  return result;
}
