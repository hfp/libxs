#ifndef CONVERSE_H
#define CONVERSE_H

#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>

#include <string.h>

#define FPRINT_ORDER 4
#define CORPUS_FILE "converse.dat"
#define COMPOSE_NDIMS 10
#define COMPOSE_BITS 6
#define COMPOSE_MAXTEXT 512
#define ENTRY_TOKEN_MAX 48
#define ENTRY_SECTION_MAX 64

#define ENTRY_LEX_ENTITY 0x0001u
#define ENTRY_LEX_NUMBER 0x0002u
#define ENTRY_LEX_QUESTION 0x0004u
#define ENTRY_LEX_PLACE 0x0008u
#define ENTRY_LEX_CAUSE 0x0010u
#define ENTRY_LEX_METHOD 0x0020u

enum { CONN_SPACE = 0, CONN_COMMA = 1, CONN_PERIOD = 2, CONN_NEWLINE = 3 };
enum { SCALE_PHRASE = 0, SCALE_SENTENCE = 1, SCALE_PARAGRAPH = 2 };

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
  unsigned short section_len;
  char section[ENTRY_SECTION_MAX];
} corpus_entry_t;


static void corpus_key_from_fprint(const libxs_fprint_t* fp,
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

#endif /*CONVERSE_H*/