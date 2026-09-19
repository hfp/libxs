#ifndef CONVERSE_RECOMB_H
#define CONVERSE_RECOMB_H

#include <libxs/libxs_reg.h>
#include <libxs/libxs_token.h>

#include "converse.h"


/**
 * Word-scale probability of a successor given a context, as the recombination
 * syntax gate needs it. The gate compares the seam-crossing span against the
 * span it displaced, so it needs a backoff model but not any particular one: it
 * is passed as a function rather than linked, which keeps the caller's model
 * bridge (and its skip-gram interpolation) on the caller's side of this header.
 */
typedef double (*converse_recomb_prob_t)(const unsigned int hist[], int hlen,
  unsigned int next);


/**
 * Bits the seam costs under the caller's byte model, as a callback for the same
 * reason word_prob is one: it is a DIAGNOSTIC. Six seam scores have been refuted,
 * the last ranking a false continuation 50x above the true one, so grammaticality
 * is enforced by the clause constraint and this only fills the reported bpc
 * columns. Passing it as a function rather than naming the model's type is what
 * lets recombination compile without the byte model at all - NULL simply drops
 * those columns.
 */
typedef int (*converse_recomb_seam_t)(const char* prefix, int prefix_length,
  const char* suffix, int suffix_length, int score_length, double* bits);


/**
 * Text predicates and the corpus encoder the recombination gates rely on. These
 * are owned by converse.c, which is the corpus and text layer; they are declared
 * here rather than duplicated so both translation units share one definition.
 */
typedef struct converse_recomb_host_t {
  int (*ends_sentence)(const char* text, int text_len);
  /**
   * Word-character test over a byte SEQUENCE, not a single byte: an encoded letter
   * spans several bytes and must be accepted whole, while encoded punctuation must
   * not be accepted at all. `length` receives the span so a scan advances by
   * characters rather than bytes.
   */
  int (*is_wordchar)(const unsigned char* text, size_t size, int* length);
  int (*entry_build)(corpus_entry_t* entry, const unsigned char* text, int len,
    unsigned char scale, libxs_lexicon_t* lexicon,
    const libxs_lexrule_t* rules, int nrules, int create);
  converse_recomb_prob_t word_prob;
  converse_recomb_seam_t seam_bits;
  int maxorder;
} converse_recomb_host_t;


/**
 * Run the recombination probe over the first `limit` sentence-scale entries.
 *
 * `judge` is the byte model that scores a seam, and it is required rather than
 * optional: fluency at the junction is the measurement the whole probe is built
 * around, so a NULL judge is a caller error rather than a degraded mode.
 */
void converse_recomb_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int limit,
  const converse_recomb_host_t* host);


/**
 * Build the pivot index and keep it, for a session that composes repeatedly. The
 * probe builds and frees per run, which is right for one batch measurement and
 * wrong interactively: the index costs a full corpus pass, so rebuilding it per
 * query would dominate the response. Entry pointers are stored, so the corpus
 * registry must outlive this and must not be modified.
 */
int converse_recomb_open(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon,
  const converse_recomb_host_t* host);

/** Release what converse_recomb_open built. Safe when open was never called. */
void converse_recomb_close(void);

/**
 * Compose one sentence from `host_text`, chosen by the selection objective rather
 * than by first acceptance.
 *
 * The probe stops at the first donor that passes the gates, which is right for
 * measuring whether a host admits a join and wrong for answering: the reachable
 * set per host is tens of alternatives, and the gates are thresholds that admit
 * all of them. So candidates are enumerated, screened by prerequisites, and the
 * winner is taken from the Pareto front over the remaining trade-offs.
 *
 * Returns the composed length, or 0 when nothing is admissible - which is a
 * legitimate and frequent outcome, and must stay distinguishable from an error.
 * `out_nfront` receives the front size when non-NULL: a front of one is a real
 * choice, a large front means the objective is indifferent and the pick within it
 * is arbitrary, so a caller that reports composition honestly needs to know which.
 */
int converse_recomb_compose_best(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* host_text, int host_len, char* out, size_t out_size,
  int* out_nfront, int* out_ncand);

#endif /*CONVERSE_RECOMB_H*/
