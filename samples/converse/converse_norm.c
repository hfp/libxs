/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "converse_norm.h"
#include <libxs/libxs_str.h>
#include <libxs/libxs_rng.h>

#define NORM_RADIUS_DEFAULT 1
#define NORM_MARGIN_DEFAULT 1
#define NORM_CAND_MAX 16
#define NORM_PACK_MAX 6
#define NORM_PACK_SAMPLE 256
#define NORM_EVAL_SAMPLE 200
#define NORM_EVAL_EDITS 3
#define NORM_LINE_MAX 512
#define NORM_WORDS_MAX 65536


typedef struct norm_word_t {
  unsigned int id;
  unsigned int count;
  int length;
} norm_word_t;

struct converse_norm_t {
  const libxs_lexicon_t* lexicon;
  norm_word_t* words;
  int nwords;
  int radius;
  int margin;
};

/** One scan: the tie set at the minimum distance, and how far the next one is. */
typedef struct norm_scan_t {
  unsigned int ids[NORM_CAND_MAX];
  unsigned int counts[NORM_CAND_MAX];
  int nids;
  int ntied;
  int distance;
  int next;
} norm_scan_t;


static int norm_lower(char* out, int out_size, const char* text, int length)
{
  int result = 0;
  if (NULL != out && 0 < out_size && NULL != text) {
    while (result < length && result + 1 < out_size) {
      out[result] = (char)tolower((unsigned char)text[result]);
      ++result;
    }
    out[result] = 0;
  }
  return result;
}


static void norm_candidate_insert(norm_scan_t* scan, unsigned int id,
  unsigned int count)
{
  int at = (scan->nids < NORM_CAND_MAX) ? scan->nids : (NORM_CAND_MAX - 1);
  if (scan->nids < NORM_CAND_MAX) ++scan->nids;
  else if (count <= scan->counts[NORM_CAND_MAX - 1]) at = -1;
  if (0 <= at) {
    while (0 < at && scan->counts[at - 1] < count) {
      scan->ids[at] = scan->ids[at - 1];
      scan->counts[at] = scan->counts[at - 1];
      --at;
    }
    scan->ids[at] = id;
    scan->counts[at] = count;
  }
}


/**
 * The length difference bounds the edit distance from below, which is what makes
 * the scan affordable. The bound has to reach past the radius by the margin:
 * prune at the radius and the runner-up that decides the margin is never seen.
 */
static void norm_scan(const converse_norm_t* norm, const char* query,
  int length, norm_scan_t* scan)
{
  const int reach = norm->radius + ((0 < norm->margin) ? norm->margin : 0);
  int at;
  scan->nids = 0;
  scan->ntied = 0;
  scan->distance = -1;
  scan->next = -1;
  for (at = 0; at < norm->nwords; ++at) {
    const int clen = norm->words[at].length;
    const int gap = (clen > length) ? (clen - length) : (length - clen);
    if (gap <= reach) {
      int len = 0;
      const char* text = libxs_lexicon_text(norm->lexicon,
        norm->words[at].id, &len, NULL);
      if (NULL != text) {
        const int distance = libxs_stridist(query, text);
        if (0 > scan->distance || distance < scan->distance) {
          scan->next = scan->distance;
          scan->distance = distance;
          scan->nids = 0;
          scan->ntied = 0;
        }
        if (distance == scan->distance) {
          ++scan->ntied;
          norm_candidate_insert(scan, norm->words[at].id,
            norm->words[at].count);
        }
        else if (0 > scan->next || distance < scan->next) {
          scan->next = distance;
        }
      }
    }
  }
}


static void norm_decide(const converse_norm_t* norm, const norm_scan_t* scan,
  converse_norm_hit_t* hit)
{
  hit->id = 0;
  hit->distance = scan->distance;
  hit->margin = (0 <= scan->next) ? (scan->next - scan->distance) : -1;
  hit->ncandidate = scan->ntied;
  if (0 <= scan->distance && scan->distance <= norm->radius && 0 < scan->nids
    /* an exact match is the word itself, so no isolation is asked of it */
    && (0 == scan->distance || 0 > hit->margin || hit->margin >= norm->margin))
  {
    hit->id = scan->ids[0];
  }
}


converse_norm_t* converse_norm_create(const libxs_lexicon_t* lexicon)
{
  converse_norm_t* result = NULL;
  const unsigned int nentry = (NULL != lexicon)
    ? libxs_lexicon_size(lexicon) : 0;
  if (0 < nentry) {
    result = (converse_norm_t*)calloc(1, sizeof(converse_norm_t));
    if (NULL != result) {
      result->words = (norm_word_t*)malloc((size_t)nentry
        * sizeof(norm_word_t));
      if (NULL != result->words) {
        unsigned int id;
        result->lexicon = lexicon;
        result->radius = NORM_RADIUS_DEFAULT;
        result->margin = NORM_MARGIN_DEFAULT;
        for (id = 1; id <= nentry; ++id) {
          int len = 0;
          unsigned int flags = 0;
          const char* text = libxs_lexicon_text(lexicon, id, &len, &flags);
          if (NULL != text && 0 < len && 0 != (flags & LIBXS_LEXEME_WORD)
            && 0 == (flags & (LIBXS_LEXEME_NUMBER | LIBXS_LEXEME_PUNCT
              | LIBXS_LEXEME_MARKUP)))
          {
            result->words[result->nwords].id = id;
            result->words[result->nwords].count =
              libxs_lexicon_count(lexicon, id);
            result->words[result->nwords].length = len;
            ++result->nwords;
          }
        }
      }
      else {
        free(result);
        result = NULL;
      }
    }
  }
  return result;
}


void converse_norm_destroy(converse_norm_t* norm)
{
  if (NULL != norm) {
    free(norm->words);
    free(norm);
  }
}


int converse_norm_size(const converse_norm_t* norm)
{
  int result = 0;
  if (NULL != norm) result = norm->nwords;
  return result;
}


void converse_norm_set_radius(converse_norm_t* norm, int radius)
{
  if (NULL != norm && 0 <= radius) norm->radius = radius;
}


void converse_norm_set_margin(converse_norm_t* norm, int margin)
{
  if (NULL != norm && 0 <= margin) norm->margin = margin;
}


int converse_norm_decode(const converse_norm_t* norm, const char* word,
  int length, converse_norm_hit_t* hit)
{
  int result = EXIT_FAILURE;
  if (NULL != norm && NULL != word && 0 < length && NULL != hit) {
    char query[LIBXS_LEXEME_MAXBYTES + 1];
    const int qlen = norm_lower(query, (int)sizeof(query), word, length);
    if (0 < qlen) {
      norm_scan_t scan;
      norm_scan(norm, query, qlen, &scan);
      norm_decide(norm, &scan, hit);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


int converse_norm_candidates(const converse_norm_t* norm, const char* word,
  int length, unsigned int ids[], int maxids)
{
  int result = 0;
  if (NULL != norm && NULL != word && 0 < length && NULL != ids && 0 < maxids) {
    char query[LIBXS_LEXEME_MAXBYTES + 1];
    const int qlen = norm_lower(query, (int)sizeof(query), word, length);
    if (0 < qlen) {
      norm_scan_t scan;
      norm_scan(norm, query, qlen, &scan);
      while (result < scan.nids && result < maxids) {
        ids[result] = scan.ids[result];
        ++result;
      }
    }
  }
  return result;
}


int converse_norm_packing(const converse_norm_t* norm, int nsample,
  int hist[], int nhist)
{
  int result = 0;
  if (NULL != norm && NULL != hist && 0 < nhist) {
    int at, stride = 1;
    for (at = 0; at < nhist; ++at) hist[at] = 0;
    if (0 < nsample && nsample < norm->nwords) {
      stride = norm->nwords / nsample;
    }
    if (stride < 1) stride = 1;
    for (at = 0; at < norm->nwords; at += stride) {
      int len = 0;
      const char* text = libxs_lexicon_text(norm->lexicon,
        norm->words[at].id, &len, NULL);
      if (NULL != text) {
        int other, best = -1;
        for (other = 0; other < norm->nwords; ++other) {
          const int olen = norm->words[other].length;
          const int gap = (olen > len) ? (olen - len) : (len - olen);
          if (other != at && (0 > best || gap < best)) {
            int tlen = 0;
            const char* otext = libxs_lexicon_text(norm->lexicon,
              norm->words[other].id, &tlen, NULL);
            if (NULL != otext) {
              const int distance = libxs_stridist(text, otext);
              if (0 > best || distance < best) best = distance;
            }
          }
        }
        if (0 <= best) {
          const int slot = (best < nhist) ? best : (nhist - 1);
          ++hist[slot];
          ++result;
        }
      }
    }
  }
  return result;
}


int converse_norm_corrupt(const char* word, int length, int kind, int nedit,
  char* out, int out_size)
{
  int result = 0;
  if (NULL != word && 0 < length && NULL != out && length < out_size) {
    int edit;
    memcpy(out, word, (size_t)length);
    result = length;
    for (edit = 0; edit < nedit; ++edit) {
      const int pick = (0 <= kind) ? kind
        : (int)libxs_rng_u32(CONVERSE_NORM_NKIND);
      const int at = (int)libxs_rng_u32((unsigned int)result);
      if (CONVERSE_NORM_DELETE == pick && 1 < result) {
        memmove(out + at, out + at + 1, (size_t)(result - at - 1));
        --result;
      }
      else if (CONVERSE_NORM_INSERT == pick && result + 1 < out_size) {
        memmove(out + at + 1, out + at, (size_t)(result - at));
        out[at] = (char)('a' + libxs_rng_u32(26));
        ++result;
      }
      else if (CONVERSE_NORM_TRANSPOSE == pick && 1 < result) {
        const int with = (at + 1 < result) ? (at + 1) : (at - 1);
        const char swap = out[at];
        out[at] = out[with];
        out[with] = swap;
      }
      else {
        /* a substitution that keeps the character would not cost an edit */
        const char to = (char)('a' + libxs_rng_u32(26));
        out[at] = (to != out[at]) ? to
          : (char)('a' + ((to - 'a' + 1) % 26));
      }
    }
    out[result] = 0;
  }
  return result;
}


int converse_norm_table(const converse_norm_t* norm,
  const char* const words[], int nwords, libxs_lexnorm_t table[],
  int maxnorms)
{
  int result = 0;
  if (NULL != norm && NULL != words && NULL != table && 0 < maxnorms) {
    int at;
    for (at = 0; at < nwords && result < maxnorms; ++at) {
      const int length = (NULL != words[at]) ? (int)strlen(words[at]) : 0;
      converse_norm_hit_t hit;
      if (0 < length && length <= LIBXS_LEXEME_MAXBYTES
        && EXIT_SUCCESS == converse_norm_decode(norm, words[at], length, &hit)
        && 0 != hit.id && 0 != hit.distance)
      {
        int tlen = 0;
        const char* text = libxs_lexicon_text(norm->lexicon, hit.id, &tlen,
          NULL);
        if (NULL != text && 0 < tlen && tlen <= LIBXS_LEXEME_MAXBYTES) {
          norm_lower(table[result].from, LIBXS_LEXEME_MAXBYTES + 1, words[at],
            length);
          memcpy(table[result].to, text, (size_t)tlen);
          table[result].to[tlen] = 0;
          ++result;
        }
      }
    }
  }
  return result;
}


static void norm_report_packing(const converse_norm_t* norm)
{
  int hist[NORM_PACK_MAX];
  const int measured = converse_norm_packing(norm, NORM_PACK_SAMPLE, hist,
    NORM_PACK_MAX);
  if (0 < measured) {
    int at;
    printf("packing: %i of %i codewords sampled\n", measured,
      converse_norm_size(norm));
    for (at = 0; at < NORM_PACK_MAX; ++at) {
      if (0 != hist[at]) {
        printf("  nearest at %i%s: %5i (%4.1f%%)\n", at,
          (at + 1 == NORM_PACK_MAX) ? "+" : " ", hist[at],
          100.0 * hist[at] / measured);
      }
    }
  }
}


static void norm_report_decode(const converse_norm_t* norm)
{
  int nedit;
  printf("decode: radius %i, margin %i, %i samples per edit count\n",
    norm->radius, norm->margin, NORM_EVAL_SAMPLE);
  for (nedit = 1; nedit <= NORM_EVAL_EDITS; ++nedit) {
    int at, ntest = 0, nexact = 0, nabstain = 0, nwrong = 0;
    libxs_rng_set_seed(1);
    for (at = 0; at < NORM_EVAL_SAMPLE; ++at) {
      const int pick = (int)libxs_rng_u32((unsigned int)norm->nwords);
      int len = 0;
      const char* text = libxs_lexicon_text(norm->lexicon,
        norm->words[pick].id, &len, NULL);
      if (NULL != text && 1 < len) {
        char damaged[LIBXS_LEXEME_MAXBYTES + 1];
        const int dlen = converse_norm_corrupt(text, len, -1, nedit, damaged,
          (int)sizeof(damaged));
        converse_norm_hit_t hit;
        if (0 < dlen
          && EXIT_SUCCESS == converse_norm_decode(norm, damaged, dlen, &hit))
        {
          ++ntest;
          if (0 == hit.id) ++nabstain;
          else if (hit.id == norm->words[pick].id) ++nexact;
          else ++nwrong;
        }
      }
    }
    if (0 < ntest) {
      printf("  %i edit%s: recovered %4.1f%%, wrong %4.1f%%,"
        " abstained %4.1f%%\n", nedit, (1 == nedit) ? " " : "s",
        100.0 * nexact / ntest, 100.0 * nwrong / ntest,
        100.0 * nabstain / ntest);
    }
  }
}


static void norm_prompt(const converse_norm_t* norm)
{
  char line[NORM_LINE_MAX];
  printf("codebook: %i words; type a word per line\n",
    converse_norm_size(norm));
  printf("> ");
  fflush(stdout);
  while (NULL != fgets(line, (int)sizeof(line), stdin)) {
    int length = (int)strlen(line);
    while (0 < length && 0 != isspace((unsigned char)line[length - 1])) {
      line[--length] = 0;
    }
    if (0 < length && length <= LIBXS_LEXEME_MAXBYTES) {
      unsigned int ids[NORM_CAND_MAX];
      converse_norm_hit_t hit;
      if (EXIT_SUCCESS == converse_norm_decode(norm, line, length, &hit)) {
        const int ncand = converse_norm_candidates(norm, line, length, ids,
          NORM_CAND_MAX);
        int at;
        if (0 != hit.id) {
          int tlen = 0;
          const char* text = libxs_lexicon_text(norm->lexicon, hit.id, &tlen,
            NULL);
          printf("%s -> %.*s", line, tlen, (NULL != text) ? text : "");
          if (0 == hit.distance) printf("  (exact)\n");
          else printf("  (distance %i)\n", hit.distance);
        }
        else if (0 > hit.distance) printf("%s: no codeword in reach\n", line);
        else {
          printf("%s: abstained, nearest at %i with %i tied\n", line,
            hit.distance, hit.ncandidate);
        }
        for (at = 0; at < ncand; ++at) {
          int tlen = 0;
          const char* text = libxs_lexicon_text(norm->lexicon, ids[at], &tlen,
            NULL);
          printf("    %.*s (%u)\n", tlen, (NULL != text) ? text : "",
            libxs_lexicon_count(norm->lexicon, ids[at]));
        }
      }
    }
    printf("> ");
    fflush(stdout);
  }
  printf("\n");
}


static int norm_words_read(const char* path, char** words, int maxwords)
{
  int result = 0;
  FILE* file = (NULL != path) ? fopen(path, "r") : NULL;
  if (NULL != file) {
    char line[NORM_LINE_MAX];
    while (result < maxwords && NULL != fgets(line, (int)sizeof(line), file)) {
      int length = (int)strlen(line);
      while (0 < length && 0 != isspace((unsigned char)line[length - 1])) {
        line[--length] = 0;
      }
      if (0 < length && length <= LIBXS_LEXEME_MAXBYTES) {
        char* copy = (char*)malloc((size_t)length + 1);
        if (NULL != copy) {
          memcpy(copy, line, (size_t)length + 1);
          words[result++] = copy;
        }
      }
    }
    fclose(file);
  }
  else fprintf(stderr, "cannot read %s\n", (NULL != path) ? path : "(none)");
  return result;
}


/**
 * The table is written in the relations line format, because that is what
 * converse_setup already loads from the norms path: a run after this one picks
 * the corrections up with nothing added on the consuming side.
 */
static int norm_table_write(const converse_norm_t* norm,
  const char* const words[], int nwords)
{
  int result = EXIT_FAILURE;
  libxs_lexnorm_t* table = (0 < nwords)
    ? (libxs_lexnorm_t*)calloc((size_t)nwords, sizeof(libxs_lexnorm_t)) : NULL;
  if (NULL != table) {
    const int ntable = converse_norm_table(norm, words, nwords, table, nwords);
    const char* path = converse_norms_path();
    FILE* file = (0 < ntable) ? fopen(path, "w") : NULL;
    if (NULL != file) {
      int at;
      for (at = 0; at < ntable; ++at) {
        fprintf(file, "norm|%s|%s\n", table[at].from, table[at].to);
      }
      fclose(file);
      printf("wrote %i of %i corrections to %s\n", ntable, nwords, path);
      result = EXIT_SUCCESS;
    }
    else if (0 == ntable) {
      printf("no correction accepted of %i surface forms\n", nwords);
      result = EXIT_SUCCESS;
    }
    else fprintf(stderr, "cannot write %s\n", path);
    free(table);
  }
  return result;
}


int converse_norm_run(converse_run_t* run)
{
  int result = EXIT_FAILURE;
  if (NULL != run) {
    converse_norm_t* norm = converse_norm_create(run->lexicon);
    if (NULL != norm) {
      if (0 <= run->norm_radius) {
        converse_norm_set_radius(norm, run->norm_radius);
      }
      if (0 <= run->norm_margin) {
        converse_norm_set_margin(norm, run->norm_margin);
      }
      if (0 == converse_norm_size(norm)) {
        fprintf(stderr, "the lexicon holds no word entries\n");
      }
      else if (NULL != run->norm_words) {
        char** words = (char**)calloc(NORM_WORDS_MAX, sizeof(char*));
        if (NULL != words) {
          const int nwords = norm_words_read(run->norm_words, words,
            NORM_WORDS_MAX);
          int at;
          if (0 < nwords) {
            result = norm_table_write(norm, (const char* const*)words, nwords);
          }
          for (at = 0; at < nwords; ++at) free(words[at]);
          free(words);
        }
      }
      else if (0 != run->eval_mode) {
        norm_report_packing(norm);
        norm_report_decode(norm);
        result = EXIT_SUCCESS;
      }
      else {
        norm_prompt(norm);
        result = EXIT_SUCCESS;
      }
      converse_norm_destroy(norm);
    }
    else fprintf(stderr, "no lexicon to build a codebook from\n");
  }
  return result;
}
