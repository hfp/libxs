/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef CONVERSE_NORM_H
#define CONVERSE_NORM_H

#include "converse_core.h"

/** Single-character corruptions converse_norm_corrupt can apply. */
enum {
  CONVERSE_NORM_SUBSTITUTE = 0,
  CONVERSE_NORM_DELETE = 1,
  CONVERSE_NORM_INSERT = 2,
  CONVERSE_NORM_TRANSPOSE = 3,
  CONVERSE_NORM_NKIND = 4
};

/** What a decode resolved to, and what it had to choose between. */
typedef struct converse_norm_hit_t {
  /** Lexicon id of the chosen codeword, or 0 when the rule abstained. */
  unsigned int id;
  /** Edit distance to the nearest codeword, or -1 when none was examined. */
  int distance;
  /** Runner-up distance less distance, or -1 when no runner-up was in reach. */
  int margin;
  /** Codewords tied at distance: one means the choice was unambiguous. */
  int ncandidate;
} converse_norm_hit_t;

/**
 * Opaque codebook: the word entries of one lexicon and their frequencies. A
 * surface form is decoded to the nearest codeword under case-insensitive edit
 * distance, so whatever it answers is a word the corpus actually contains.
 *
 * Two entry points share the machinery and differ in whether truth exists. The
 * closed loop corrupts a word it already knows and scores the decode, which is
 * how a radius and a margin are chosen at all. The open loop decodes what a user
 * typed, where nothing says whether the input was a typo, a name, or simply a
 * word the corpus never saw, and where a wrong correction destroys the word it
 * replaces. That asymmetry is why the decision is kept apart from the scan: the
 * rule has to be able to abstain, and an abstention has to stay visible.
 *
 * What limits it is the codebook, not the metric. A vocabulary of natural
 * language packs badly, so most words have a neighbour one edit away and the
 * scan alone cannot separate them; the frequencies break those ties, and
 * converse_norm_packing is what says how often the tie occurs.
 */
typedef struct converse_norm_t converse_norm_t;


/**
 * Build a codebook over the word entries of a lexicon. Punctuation, numbers and
 * markup are left out, so a decode cannot answer with a comma. The lexicon is
 * borrowed and must outlive the codebook.
 */
converse_norm_t* converse_norm_create(const libxs_lexicon_t* lexicon);

/** Release a codebook (NULL is accepted). */
void converse_norm_destroy(converse_norm_t* norm);

/** Number of codewords the scan considers. */
int converse_norm_size(const converse_norm_t* norm);

/** Largest edit distance a decode may accept (default 1). */
void converse_norm_set_radius(converse_norm_t* norm, int radius);

/**
 * Lead over the runner-up a decode must have to commit (default 1). One is
 * permissive on purpose: a runner-up is by construction at least one edit
 * further out, so the default rejects nothing and the knob starts where today's
 * behaviour is. Two demands that the choice be isolated, which on a natural
 * vocabulary is most of what abstention costs. An exact match is exempt either
 * way: a word the corpus holds is not a correction.
 */
void converse_norm_set_margin(converse_norm_t* norm, int margin);

/**
 * Decode one surface form. hit receives the decision and what it was made from,
 * with hit->id zero when the rule abstained. Returns EXIT_SUCCESS whenever the
 * scan ran, abstention included: declining to correct is an answer, not a
 * failure.
 */
int converse_norm_decode(const converse_norm_t* norm, const char* word,
  int length, converse_norm_hit_t* hit);

/**
 * Codewords at the minimum distance, most frequent first. This is the set the
 * decode chose from, so a tie the rule refused to resolve is visible here.
 * Returns how many ids were written.
 */
int converse_norm_candidates(const converse_norm_t* norm, const char* word,
  int length, unsigned int ids[], int maxids);

/**
 * Distance to the nearest OTHER codeword, over a sample of the codebook, as a
 * histogram indexed by distance: hist[1] counts the words with a neighbour one
 * edit away and the last bucket accumulates. This is the packing radius of the
 * vocabulary, and it is what says whether a radius is safe. Sampling is by
 * stride because the cost is quadratic. Returns how many words were measured.
 */
int converse_norm_packing(const converse_norm_t* norm, int nsample,
  int hist[], int nhist);

/**
 * Apply nedit corruptions to word and write the result to out. kind is one of
 * the CONVERSE_NORM_* constants, or negative to draw one per edit. A
 * substitution always changes the character, so nedit edits cost nedit.
 * Returns the length written, or zero on failure.
 */
int converse_norm_corrupt(const char* word, int length, int kind, int nedit,
  char* out, int out_size);

/**
 * Write the accepted decodes of nwords surface forms as a normalization table,
 * which is what libxs_lexeme_stream_encode consumes. An abstention and a form
 * that decodes to itself are both skipped, so the table holds only the
 * corrections the rule was willing to commit to. Returns how many were written.
 */
int converse_norm_table(const converse_norm_t* norm,
  const char* const words[], int nwords, libxs_lexnorm_t table[],
  int maxnorms);

/** The normalizer half: serves the prompt, -e and -t. */
int converse_norm_run(converse_run_t* run);

#endif /*CONVERSE_NORM_H*/
