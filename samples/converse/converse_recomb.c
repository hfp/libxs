#include <libxs/libxs.h>
#include <libxs/libxs_token.h>
#include <libxs/libxs_hash.h>
#include <libxs/libxs_mem.h>
#include <libxs/libxs_str.h>

#include "converse.h"
#include "converse_recomb.h"


/**
 * The seam judge and the host callbacks for the current probe run.
 *
 * These are set once by converse_recomb_probe_run and read by the gates below.
 * They are translation-unit state rather than parameters threaded through every
 * gate because the gates are called from inside three nested loops whose shape is
 * the measurement: adding a context argument to each would obscure that the
 * capacity count, the dedup and the gate order are what the numbers depend on.
 */
static const converse_recomb_host_t* recomb_host = NULL;


/** Whether the caller supplied a seam-bits diagnostic. */
static int recomb_have_judge(void)
{
  return (NULL != recomb_host && NULL != recomb_host->seam_bits) ? 1 : 0;
}


/**
 * Grounded recombination: synthesize a sentence that is in the corpus nowhere by
 * splicing two corpus sentences at a shared content term.
 *
 * The slot probe established that no discrete re-keying of contexts reaches the
 * novel positions, so this does not try to generalize a lookup. It composes
 * instead: given A ending "...the girl called X ..." and B containing "... X went
 * into the forest.", emit prefix(A up to and including X) + suffix(B after X).
 * Both halves are verbatim corpus text and the shared term is the licence for the
 * join, so every word remains attributable - which is the property that
 * distinguishes this from generating text.
 *
 * A pivot must be a CONTENT word (not a stop word) appearing in both sentences,
 * and the result must not itself occur in the corpus, or this measures replay.
 * Fluency of the junction is judged by the byte model over a short window after
 * the seam; two controls bound the scale (see converse_recomb_probe_run).
 */
#define RECOMB_MIN_WORDS 4
#define RECOMB_MAX_CAND 64


typedef struct recomb_word_t {
  int begin;
  int end;
  unsigned int id;
  unsigned short flags;
} recomb_word_t;


/**
 * Bytes scored after the seam. Small on purpose: the byte context reaches back
 * only HIER_PPM_ORDER_MAX, so a wide window is dominated by the suffix's own
 * predictability and stops measuring the junction at all. On the reference
 * corpus the gap between a real junction and an arbitrary splice is 0.709 bits at 4
 * bytes, 0.252 at 8, 0.008 at 24, and inverts at 48 - so a wide window does not
 * merely dilute the signal, it destroys it.
 */
/**
 * Minimum shared content words (beyond the pivot) a donor must have with the
 * host, as a percentage of the smaller content set. 0 accepts any pivot match,
 * which is what exposed that a fluent seam does not imply a coherent sentence.
 */
static int recomb_min_overlap(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_MINOVL");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 25;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Whether the well-formedness gate applies: paired delimiters balance and the
 * result ends a sentence. Off by default because a prose corpus splits dialogue
 * across sentences, so a corpus sentence can legitimately carry one quote of a
 * pair and the gate would reject sound joins. It is the right default on markup,
 * where an unclosed backtick and a fragment ending in a comma are both damage.
 */
static int recomb_balance_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_BALANCE");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Syntactic gate on a seam, expressed as attestation rather than as grammar.
 *
 * The byte-level seam judge cannot see grammar: "the guards would not to go
 * away" costs little per byte because every byte pair is common. The fault is at
 * WORD scale, and the corpus already knows it - the ungrammatical continuation
 * occurs zero times while its grammatical alternatives all occur. So the gate asks the
 * n-gram, not a rule file: no English syntax is declared anywhere, and the same
 * test works unchanged on another language.
 *
 * Two things make this subtler than "reject unattested":
 *  - A bigram is too short. "not to" occurs 19 times, so the bigram at the seam
 *    of the bad example is perfectly attested; only the trigram exposes it.
 *  - A trigram is often legitimately unseen: "guards would not" also occurs zero
 *    times and is fine. So an absolute zero test would reject good text.
 * Hence the comparison is RELATIVE: the seam-crossing trigram's probability under
 * backoff, against the same span's probability in the host sentence it replaced.
 * A seam is rejected when it is worse than what it displaced by more than a
 * tolerance, which asks "did the join make this less like the corpus" rather than
 * "is this exact sequence present".
 */
/**
 * Zero when the host supplies no word probability, which makes the seam penalty
 * zero and so leaves the grammar gate permissive rather than crashing on a NULL
 * callback. That is the intended behaviour for a build without the n-gram: the
 * penalty is a DIAGNOSTIC, since a seam score has never been able to tell a true
 * join from a fluent false one, and grammaticality is now enforced by the clause
 * constraint instead.
 */
static double recomb_span_bits(libxs_lexicon_t* lexicon,
  const unsigned int ids[], int n)
{
  double result = 0.0;
  LIBXS_UNUSED(lexicon);
  if (NULL != recomb_host && NULL != recomb_host->word_prob && 1 < n) {
    const int maxorder = recomb_host->maxorder;
    int at;
    for (at = 1; at < n; ++at) {
      const int hlen = (at < maxorder) ? at : maxorder;
      const double p = recomb_host->word_prob(ids + at - hlen, hlen, ids[at]);
      result += -log((p > 1e-300) ? p : 1e-300) / log(2.0);
    }
    result /= (double)(n - 1);
  }
  return result;
}


static double recomb_grammar_tol(void)
{
  static double cached = -1.0;
  if (cached < 0.0) {
    const char* env = getenv("CONVERSE_RECOMB_GRAMTOL");
    cached = (NULL != env && '\0' != *env) ? atof(env) : 0.5;
    if (cached < 0.0) cached = 0.0;
  }
  return cached;
}


static int recomb_seam_window(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_WINDOW");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 4;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Reject a splice whose paired delimiters do not balance.
 *
 * A cut inside a bracketed or quoted run leaves an opener without its closer, and
 * neither word gate can see it: the seam trigram is over lexemes, which carry no
 * punctuation, and the byte model scores a stray backtick as cheap because
 * backticks are frequent. On the documentation corpus 28% of otherwise accepted
 * joins were malformed this way, so this is structural damage the existing gates
 * are blind to rather than a stylistic preference.
 *
 * The test is on paired delimiters, which is a property of the text's markup and
 * not of any language, so nothing language-specific is added. An apostrophe is
 * deliberately not treated as a quote: it is a letter in many words and pairing it
 * would reject correct prose.
 *
 * DIRECTED quote pairs are counted like brackets rather than toggled. ASCII " and
 * ` are their own closer so parity is all that can be checked, but a corpus using
 * guillemets or typographic quotes distinguishes opener from closer, and a toggle
 * would call the reversed order balanced. This is what makes the gate work on
 * German: 50% of otherwise accepted joins on a German fairy-tale corpus left a
 * guillemet unclosed, and an ASCII-only test saw none of them.
 */
static int recomb_balanced(const char* text, int len)
{
  int result = 1;
  int paren = 0, bracket = 0, brace = 0, tick = 0, quote = 0;
  int guillemet = 0, curly_double = 0;
  int at;
  for (at = 0; at < len; ++at) {
    const unsigned char* u = (const unsigned char*)text + at;
    const int left = len - at;
    switch (text[at]) {
      case '(': ++paren; break;
      case ')': --paren; break;
      case '[': ++bracket; break;
      case ']': --bracket; break;
      case '{': ++brace; break;
      case '}': --brace; break;
      case '`': tick ^= 1; break;
      case '"': quote ^= 1; break;
      default: break;
    }
    /* U+00AB/U+00BB, and the U+201x quotation block. */
    if (1 < left && 0xC2u == u[0]) {
      if (0xABu == u[1]) ++guillemet;
      else if (0xBBu == u[1]) --guillemet;
    }
    else if (2 < left && 0xE2u == u[0] && 0x80u == u[1]) {
      /**
       * Double quotes only. U+2019 is also the typographic APOSTROPHE, so pairing
       * the single quotes would reject correct English prose ("don't") for the
       * same reason the ASCII apostrophe is left out.
       */
      if (0x9Cu == u[2]) ++curly_double;
      else if (0x9Du == u[2]) --curly_double;
    }
    if (paren < 0 || bracket < 0 || brace < 0) {
      result = 0;
      break;
    }
  }
  if (0 != paren || 0 != bracket || 0 != brace || 0 != tick || 0 != quote
    || 0 != guillemet || 0 != curly_double)
  {
    result = 0;
  }
  return result;
}


/**
 * Split text into word spans with their lexicon ids and flags. Offsets are taken
 * from the raw text rather than from the lexeme stream, because a splice needs
 * exact byte positions and the stream carries lengths only.
 */
static int recomb_words(libxs_lexicon_t* lexicon, const char* text,
  int text_len, recomb_word_t words[], int max)
{
  int result = 0;
  int pos = 0;
  while (pos < text_len && result < max) {
    int span = 1;
    if (0 != recomb_host->is_wordchar((const unsigned char*)text + pos,
      (size_t)(text_len - pos), &span))
    {
      int end = pos;
      while (end < text_len
        && 0 != recomb_host->is_wordchar((const unsigned char*)text + end,
             (size_t)(text_len - end), &span)) end += span;
      words[result].begin = pos;
      words[result].end = end;
      words[result].id = libxs_lexicon_id(lexicon, text + pos, end - pos,
        LIBXS_LEXEME_WORD, 0);
      words[result].flags = 0;
      if (0 != words[result].id) {
        unsigned int flags = 0;
        libxs_lexicon_text(lexicon, words[result].id, NULL, &flags);
        words[result].flags = (unsigned short)flags;
      }
      ++result;
      pos = end;
    }
    else ++pos;
  }
  return result;
}


/**
 * Splice at the given pivot positions: everything of A through pivot word a,
 * then everything of B after pivot word b. Returns the composed length, or 0 if
 * it does not fit or either side would be degenerate.
 */
/**
 * Whether the grafted tail may begin anywhere, or only at a clause boundary.
 *
 * Six seam signals have now been refuted, the last of them by 50x in the WRONG
 * direction, and they share a form: each measures PREDICTABILITY. At a seam
 * predictability and correctness are anti-correlated, because the fluent generic
 * continuation is the likely-wrong one - which is why the byte model rates
 * splices as better than real corpus text. So stop scoring the seam and constrain
 * it: if the donor's tail starts at a clause boundary, the graft joins two
 * complete constituents and grammaticality holds BY CONSTRUCTION rather than by a
 * judgement that has never worked.
 *
 * Punctuation only, deliberately. A conjunction list would catch more joins and
 * would be English, and the language-neutrality claim is worth more than the
 * extra candidates; if conjunctions are wanted they belong in the relation rule
 * file as DATA, the way every other language-specific vocabulary here does.
 */
static int recomb_clause_only(void)
{
  static int cached = -1;
  if (cached < 0) {
    /* ON by default: the samples improve visibly (ungrammatical seams 4 -> 1 of
       12, coherent 2 -> 5) and it costs 8 points of yield, 400 joins from 475
       tries instead of 435. CONVERSE_RECOMB_CLAUSE=0 restores the unconstrained
       splice, which every recomb figure published before 2026-08-13 used. */
    const char* env = getenv("CONVERSE_RECOMB_CLAUSE");
    cached = (NULL != env && '0' == *env) ? 0 : 1;
  }
  return cached;
}


static int recomb_clause_start(const char* text, int len, int at)
{
  int result = 0;
  int pos = at;
  while (pos < len && 0 != isspace((unsigned char)text[pos])) ++pos;
  if (pos < len) {
    const unsigned char ch = (unsigned char)text[pos];
    if (',' == ch || ';' == ch || ':' == ch || '.' == ch || '!' == ch
      || '?' == ch || '-' == ch)
    {
      result = 1;
    }
  }
  return result;
}


/**
 * Words of the grafted tail that may not repeat what the host prefix already
 * said. 0 disables the check.
 *
 * The clause constraint cannot see this defect: a loop's seam IS at a clause
 * boundary, so the join is grammatical and still says the same thing twice.
 * Worse, the coherence gate PREFERS it - content overlap is what MINOVL rewards
 * and a duplicated half is overlap at its maximum, which is why the two loops in
 * the reference sample scored ovl=0.89 and ovl=0.74, near the top of the run. So
 * repetition is not a fluency question to be scored; it is decidable, and it is
 * rejected by construction like the rest.
 */
static int recomb_repeat_words(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_REPEAT");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 4;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Does the grafted tail say something the host prefix already said?
 *
 * Every window of `want` consecutive tail words is looked for in the prefix, not
 * just the one at the graft: the second loop in that sample repeats in the
 * MIDDLE of the tail rather than at the junction, so checking the junction alone
 * would pass it. Only
 * prefix-versus-tail is compared, so a source sentence that repeats itself is
 * left alone - that is attested text, and rejecting it would cost candidates
 * without removing a defect the splice introduced.
 */
static int recomb_repeats(const char* text, int seam, int len)
{
  const int want = recomb_repeat_words();
  int result = 0;
  if (0 < want && 0 < seam && seam < len) {
    int begin[COMPOSE_MAXTEXT / 2];
    int end[COMPOSE_MAXTEXT / 2];
    const int max = (int)(sizeof(begin) / sizeof(*begin));
    int nwords = 0;
    int pos = seam;
    while (pos < len && nwords < max) {
      int span = 1;
      if (0 != recomb_host->is_wordchar((const unsigned char*)text + pos,
        (size_t)(len - pos), &span))
      {
        int at = pos;
        while (at < len && 0 != recomb_host->is_wordchar(
          (const unsigned char*)text + at, (size_t)(len - at), &span))
        {
          at += span;
        }
        begin[nwords] = pos;
        end[nwords] = at;
        ++nwords;
        pos = at;
      }
      else ++pos;
    }
    for (pos = 0; pos + want <= nwords && 0 == result; ++pos) {
      /* The haystack is the PREFIX of a buffer whose remainder holds the span
         being looked for, which is why this needs the size-explicit form. */
      result = (NULL != libxs_strimem(text, (size_t)seam, text + begin[pos],
        (size_t)(end[pos + want - 1] - begin[pos]))) ? 1 : 0;
    }
  }
  return result;
}


static int recomb_splice(const char* a, int a_len, int a_end,
  const char* b, int b_len, int b_after, char* out, size_t out_size)
{
  int result = 0;
  if (0 < a_end && a_end <= a_len && 0 <= b_after && b_after <= b_len
    && (size_t)(a_end + b_len - b_after) + 1 < out_size
    && (0 == recomb_clause_only()
      || 0 != recomb_clause_start(b, b_len, b_after)))
  {
    memcpy(out, a, (size_t)a_end);
    memcpy(out + a_end, b + b_after, (size_t)(b_len - b_after));
    result = a_end + (b_len - b_after);
    out[result] = '\0';
    if (0 != recomb_repeats(out, a_end, result)) {
      out[0] = '\0';
      result = 0;
    }
  }
  return result;
}


/**
 * Content-word overlap between two entries, excluding the pivot itself, as a
 * fraction of the smaller content set.
 *
 * A fluent seam says nothing about whether the two halves are ABOUT the same
 * thing: a pivot on a common word can join unrelated scenes. Shared content
 * beyond the pivot is the cheapest available evidence that they are, and it uses
 * the token ids the corpus already stores.
 */
static double recomb_overlap(const corpus_entry_t* a, const corpus_entry_t* b,
  unsigned int pivot)
{
  double result = 0.0;
  int na = 0, nb = 0, shared = 0;
  int ai, bi;
  for (ai = 0; ai < (int)a->ntokens && ai < ENTRY_TOKEN_MAX; ++ai) {
    const unsigned int id = a->token_ids[ai];
    if (0 == id || id == pivot) continue;
    if (0 != (a->token_flags[ai] & LIBXS_LEXEME_STOP)) continue;
    ++na;
    for (bi = 0; bi < (int)b->ntokens && bi < ENTRY_TOKEN_MAX; ++bi) {
      if (b->token_ids[bi] == id) {
        ++shared;
        break;
      }
    }
  }
  for (bi = 0; bi < (int)b->ntokens && bi < ENTRY_TOKEN_MAX; ++bi) {
    const unsigned int id = b->token_ids[bi];
    if (0 != id && id != pivot
      && 0 == (b->token_flags[bi] & LIBXS_LEXEME_STOP)) ++nb;
  }
  { const int smaller = (na < nb) ? na : nb;
    if (0 < smaller) result = (double)shared / (double)smaller;
  }
  return result;
}


/**
 * Bits penalty the join incurs at word scale: the seam-crossing span's cost minus
 * the cost of the same span as it ran in the host sentence. Positive means the
 * splice made the sequence less corpus-like. RECOMB_SPAN words either side of the
 * pivot are compared, so the measured span is exactly the one the join altered.
 */
static double recomb_seam_penalty(libxs_lexicon_t* lexicon,
  const recomb_word_t awords[], int nawords, int ai,
  const recomb_word_t bwords[], int nbwords, int bi)
{
  enum { RECOMB_SPAN = 2 };
  unsigned int spliced[2 * RECOMB_SPAN], original[2 * RECOMB_SPAN];
  int nspliced = 0, noriginal = 0;
  int at;
  for (at = ai - RECOMB_SPAN + 1; at <= ai; ++at) {
    if (0 <= at && at < nawords && 0 != awords[at].id) {
      spliced[nspliced++] = awords[at].id;
      original[noriginal++] = awords[at].id;
    }
  }
  for (at = bi + 1; at <= bi + RECOMB_SPAN; ++at) {
    if (at < nbwords && 0 != bwords[at].id) spliced[nspliced++] = bwords[at].id;
  }
  for (at = ai + 1; at <= ai + RECOMB_SPAN; ++at) {
    if (at < nawords && 0 != awords[at].id) {
      original[noriginal++] = awords[at].id;
    }
  }
  return (1 < nspliced && 1 < noriginal)
    ? recomb_span_bits(lexicon, spliced, nspliced)
      - recomb_span_bits(lexicon, original, noriginal)
    : 0.0;
}


static int recomb_is_verbatim(const libxs_registry_t* corpus,
  const char* text, int text_len)
{
  int result = 0;
  const void* key = NULL;
  size_t cursor = 0;
  void* value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value && 0 == result) {
    const corpus_entry_t* entry = corpus_entry_view(value);
    if (entry->text_len >= text_len) {
      int at, span = entry->text_len - text_len;
      for (at = 0; at <= span && 0 == result; ++at) {
        if (0 == memcmp(entry->text + at, text, (size_t)text_len)) result = 1;
      }
    }
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  return result;
}


/**
 * Inverted index from a content-word id to the sentence entries containing it.
 *
 * Donor selection is a pivot lookup, but it was written as a corpus scan: every
 * host re-read every entry and re-tokenized it to discover whether the two shared
 * a content word. The index answers the same question directly, so the scan
 * disappears and the tokenization is paid once per entry at build time rather than
 * once per (host, entry) pair.
 *
 * Postings used to be CAPPED, because the registry copies a value and the value
 * was the whole list: a fixed array per key, paid in full for every key however
 * few donors it had. The cap bought that back and cost a silent truncation of the
 * frequent pivots - exactly the ones a scan would have reached. Now the value is
 * a HEADER and the list hangs off it, so memory is proportional to the postings
 * that exist and no pivot is cut short.
 */

/**
 * The ordinal is carried beside the pointer because donor selection is defined to
 * take an entry LATER in corpus order than the host. That constraint is what makes
 * the probe's output independent of how a candidate was reached, so dropping it
 * while replacing the scan would silently change which donor wins and move every
 * reported number.
 */
typedef struct recomb_posting_t {
  long ordinal;
  const corpus_entry_t* entry;
} recomb_posting_t;

typedef struct recomb_postings_t {
  int n;
  int cap;
  recomb_posting_t* at;
} recomb_postings_t;


static libxs_registry_t* recomb_index = NULL;
static long recomb_index_nkeys = 0;
static long recomb_index_nposts = 0;


/**
 * Predicate-like pivot suppression: constrain the OPERATOR instead of filtering
 * its output.
 *
 * Every truth failure observed so far splices on a predicate-like pivot ("made"
 * joining "made fat" to "made in his whole kingdom"), while the join that was true
 * and useful pivoted on an argument-like word ("children"). Four separate gate
 * ideas were measured and none separates those two cases: both seam trigrams occur
 * exactly ONCE, 79% of content bigrams occur once, and clause-scale recurrence only
 * reaches 6.6% at 27x the corpus. So there is no attestation to test against, and a
 * gate that cannot be built can still be avoided - by not generating the failure.
 *
 * The proxy is derived from counts, not from a word list, so no language vocabulary
 * enters the sample: a lexeme is predicate-like when it frequently FOLLOWS one of
 * the function words the rules already declare skippable. Measured on the
 * reference corpus: predicate-like pivots dominate, while concrete nouns score 0.
 * It costs 58% of pivot tokens,
 * which is the honest price of removing a failure class by construction.
 *
 * This is a probe. It is off by default because it trades reachable set for
 * precision and that trade has not been measured on more than one corpus.
 */
static libxs_registry_t* recomb_predicate = NULL;
static long recomb_predicate_nkeys = 0;


static int recomb_nopredicate_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_NOPRED");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 0;
    if (cached < 0) cached = 0;
  }
  return cached;
}


static void recomb_predicate_free(void)
{
  if (NULL != recomb_predicate) {
    libxs_registry_destroy(recomb_predicate);
    recomb_predicate = NULL;
  }
  recomb_predicate_nkeys = 0;
}


/**
 * Count, per lexeme, how often it directly follows a STOP word. The tokenizer
 * already flags those from the rules file, so the class comes from data that
 * exists rather than from anything declared here.
 */
static int recomb_predicate_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon)
{
  int result = EXIT_SUCCESS;
  recomb_predicate_free();
  recomb_predicate = libxs_registry_create();
  if (NULL == recomb_predicate) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const void* key = NULL;
    size_t cursor = 0;
    void* value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = corpus_entry_view(value);
      if (SCALE_SENTENCE == entry->scale && 0 < entry->text_len) {
        recomb_word_t words[COMPOSE_MAXTEXT / 2];
        const int nwords = recomb_words(lexicon, entry->text, entry->text_len,
          words, (int)(sizeof(words) / sizeof(*words)));
        int at;
        for (at = 1; at < nwords; ++at) {
          const unsigned int id = words[at].id;
          long* count;
          if (0 == id) continue;
          if (0 == (words[at - 1].flags & LIBXS_LEXEME_STOP)) continue;
          if (0 != (words[at].flags & LIBXS_LEXEME_STOP)) continue;
          count = (long*)libxs_registry_get(recomb_predicate, &id, sizeof(id),
            NULL);
          if (NULL != count) ++*count;
          else {
            const long fresh = 1;
            if (NULL != libxs_registry_set(recomb_predicate, &id, sizeof(id),
              &fresh, sizeof(fresh), NULL)) ++recomb_predicate_nkeys;
          }
        }
      }
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
  }
  return result;
}


static int recomb_is_predicate(unsigned int id)
{
  const long* count;
  const int threshold = recomb_nopredicate_on();
  if (0 >= threshold || NULL == recomb_predicate) return 0;
  count = (const long*)libxs_registry_get(recomb_predicate, &id, sizeof(id),
    NULL);
  return (NULL != count && *count >= (long)threshold) ? 1 : 0;
}


static void recomb_index_free(void)
{
  if (NULL != recomb_index) {
    const void* key = NULL;
    size_t cursor = 0;
    void* value = libxs_registry_begin(recomb_index, &key, &cursor);
    while (NULL != value) {
      free(((recomb_postings_t*)value)->at);
      value = libxs_registry_next(recomb_index, &key, &cursor);
    }
    libxs_registry_destroy(recomb_index);
    recomb_index = NULL;
  }
  recomb_index_nkeys = 0;
  recomb_index_nposts = 0;
}


/**
 * Build the pivot index over sentence-scale entries. Entry pointers are stored, so
 * the index is only valid while the corpus registry is alive and unmodified -
 * which holds for the probe, whose corpus is complete before it runs.
 */
static int recomb_index_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon)
{
  int result = EXIT_SUCCESS;
  recomb_index_free();
  recomb_index = libxs_registry_create();
  if (NULL == recomb_index) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const void* key = NULL;
    size_t cursor = 0;
    long ordinal = 0;
    void* value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = corpus_entry_view(value);
      if (SCALE_SENTENCE == entry->scale && 0 < entry->text_len) {
        recomb_word_t words[COMPOSE_MAXTEXT / 2];
        const int nwords = recomb_words(lexicon, entry->text, entry->text_len,
          words, (int)(sizeof(words) / sizeof(*words)));
        int at;
        for (at = 1; at < nwords - 1; ++at) {
          const unsigned int id = words[at].id;
          recomb_postings_t* postings;
          if (0 == id || 0 != (words[at].flags & LIBXS_LEXEME_STOP)) continue;
          postings = (recomb_postings_t*)libxs_registry_get(recomb_index,
            &id, sizeof(id), NULL);
          if (NULL == postings) {
            recomb_postings_t fresh;
            memset(&fresh, 0, sizeof(fresh));
            if (NULL != libxs_registry_set(recomb_index, &id, sizeof(id),
              &fresh, sizeof(fresh), NULL))
            {
              ++recomb_index_nkeys;
              postings = (recomb_postings_t*)libxs_registry_get(recomb_index,
                &id, sizeof(id), NULL);
            }
          }
          if (NULL != postings
            && (0 == postings->n || postings->at[postings->n - 1].entry != entry))
          {
            if (postings->n == postings->cap) {
              const int cap = (0 < postings->cap) ? (2 * postings->cap) : 8;
              recomb_posting_t* grown = (recomb_posting_t*)realloc(postings->at,
                (size_t)cap * sizeof(*grown));
              if (NULL != grown) {
                postings->at = grown;
                postings->cap = cap;
              }
            }
            if (postings->n < postings->cap) {
              postings->at[postings->n].ordinal = ordinal;
              postings->at[postings->n].entry = entry;
              ++postings->n;
              ++recomb_index_nposts;
            }
          }
        }
      }
      ++ordinal;
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
  }
  return result;
}


/**
 * Signals emitted AT the join, rather than reconstructed from the result.
 *
 * The three numbers the gates already compute are near-orthogonal on this data
 * (seam bits against content overlap correlate 0.006), so combining them is worth
 * doing rather than redundant. But they are only the signals that happened to be
 * printed; several more are free at the moment of splicing and expensive or
 * impossible afterwards - above all WHICH occurrence of the pivot was used, which
 * is the raw material for the referent question and is lost once the sentence is a
 * flat string.
 *
 * Signals are kept as a vector rather than folded here. Weights are one way to
 * reach a single number, but they are not the only one and they are the most
 * assumption-laden: a perfect overlap can mask a badly broken clause, and a sweep
 * of six weightings picked the same damaged candidate five times. The vector keeps
 * that decision at the selection site, where a Pareto front, a fingerprint
 * distance, or a hierarchy of prerequisites can be applied instead.
 */
typedef struct recomb_signal_t {
  double seam_bits;      /* byte-model fluency at the junction (lower better) */
  double overlap;        /* content-word agreement beyond the pivot (higher) */
  double gram;           /* relative attestation of the seam span (lower) */
  double balance;        /* prefix/suffix word-count symmetry, 0..1 (higher) */
  double flag_agree;     /* lexical-class agreement of the two halves (higher) */
  double fpjoin;         /* junction anomaly vs predicted merge (lower) */
  int pivot_host_pos;    /* word index of the pivot in the host */
  int pivot_donor_pos;   /* word index of the pivot in the donor */
  int nwords;            /* words in the composed sentence */
  int truncated;         /* donor suffix was cut at a clause boundary */
  int cross_source;      /* halves come from different source files */
  /**
   * Whether the composite ends a sentence. This belongs with the prerequisites and
   * not among the trade-offs: left to compete, the balance objective rewards
   * prefix/suffix symmetry and so favours mid-sentence truncations, which measured
   * WORSE than no selection at all (54% of survivors ended properly against 66% of
   * the rejected). A property that selection can trade away is not a property that
   * survives selection.
   */
  int ends_sentence;
  /**
   * Referent probe. `pmi` is the candidate SIGNAL (minimum association across the
   * seam, in bits) and `crossing` is the manufactured LABEL (-1 when neither class
   * applies). They are deliberately adjacent and deliberately not combined: the
   * label exists to test the signal, so reading either as the other would make the
   * measurement circular. See recomb_referent_build.
   */
  double pmi;
  int crossing;
} recomb_signal_t;


/**
 * Referent probe: does a shared pivot DENOTE the same thing in both halves?
 *
 * This is the one question none of the four gates can ask. The pivot is a word
 * type, so two documents using one term for different things splice as readily as
 * two describing the same one, and the specimen that motivates this probe passes
 * every gate and the whole selection objective. Every other signal here is a
 * property of surfaces; a search driven by them would produce more fluent,
 * coherent, attributable falsehoods, which is why this is measured before any
 * search is built.
 *
 * TWO CLASSES, MANUFACTURED RATHER THAN DETECTED. There is no labelled set of
 * crossings: the proper-noun detector found four positives in 137 joins, which
 * cannot separate anything, and this file's own warning is not to tune a threshold
 * against an unmeasured rate. So the classes are built the way the fluency ceiling
 * and floor already are - by construction:
 *
 *  - CROSSING (y=1): host and donor come from different sources AND each carries a
 *    proper noun exclusive to its own source. Different casts, so the referents
 *    differ by construction rather than by judgement.
 *  - SAME (y=0): host and donor come from one source, so a shared term has one
 *    cast behind it.
 *
 * Neither class is a quality label and neither is the signal being tested. That
 * matters because the oracle is available as a signal already - cross_source is
 * in the vector - and it is a poor test on its own: 40-42% of accepted joins are
 * cross-source while crossings are a few percent, so rejecting on it alone would
 * discard two fifths of the output at very low precision. The probe therefore
 * prints that baseline next to the measured signal; a signal that cannot beat it
 * has bought nothing.
 */
typedef struct recomb_referent_t {
  /** Sources in which this lexeme occurs, and how many distinct ones. */
  unsigned short source[4];
  int nsources;
  /** Sentence-scale occurrences, for the marginal in the association score. */
  long count;
} recomb_referent_t;


static libxs_registry_t* recomb_cast = NULL;
static libxs_registry_t* recomb_pair = NULL;
static long recomb_cast_nkeys = 0;
static long recomb_pair_nkeys = 0;
static long recomb_pair_total = 0;
/**
 * Sentences scanned. The marginals count sentences containing a term while the
 * joint counts sentences containing both, so both probabilities must be taken over
 * this same base. Normalizing the marginals by the number of PAIRS instead is off
 * by the mean pairs-per-sentence - a constant, but a large one, and it shifts
 * every PMI positive by log2 of it.
 */
static long recomb_nsentences = 0;


/**
 * Referent-probe accumulators, at translation-unit scope for the same reason the
 * judge and the host callbacks are: the joins they summarize are enumerated deep
 * inside three nested loops whose signature already carries thirteen arguments,
 * and eight more would bury the loop structure that the measurement depends on.
 * Reset by recomb_referent_build so a second run cannot inherit the first's counts.
 */
#define RECOMB_REF_MAX 16384

static long ref_nsame = 0;
static long ref_ncross = 0;
static long ref_nunlabelled = 0;
static double ref_pmi_same = 0.0;
static double ref_pmi_cross = 0.0;
static long ref_xsrc = 0;
static long ref_xsrc_cross = 0;
/**
 * Retained per-class values, so the two classes can be compared by their overlap
 * rather than only by their means. Capped, and the cap is REPORTED when reached:
 * a truncated sample looks exactly like a small one, which is the same failure the
 * postings cap produced before it was printed.
 */
static double ref_same[RECOMB_REF_MAX];
static double ref_cross[RECOMB_REF_MAX];
static int ref_nsame_kept = 0;
static int ref_ncross_kept = 0;
static int ref_capped = 0;
/**
 * The contrast class that makes the test non-trivial: joins that ARE cross-source
 * but carry no exclusive cast on both sides, so they are not crossings.
 *
 * Comparing crossings against SAME-source joins would be circular - the y=0 class
 * is same-source by construction, so cross_source separates it perfectly for free
 * and any signal correlated with source difference inherits that. Within
 * cross-source joins, cross_source is constant and therefore worthless, so a
 * signal that separates here has genuinely added something the vector did not
 * already contain.
 */
static double ref_xnon[RECOMB_REF_MAX];
static int ref_nxnon_kept = 0;


typedef struct recomb_pair_key_t {
  unsigned int lo;
  unsigned int hi;
} recomb_pair_key_t;


static int recomb_referent_on(void)
{
  static int cached = -1;
  if (cached < 0) {
    cached = (NULL != getenv("CONVERSE_RECOMB_REFERENT")) ? 1 : 0;
  }
  return cached;
}


static void recomb_referent_free(void)
{
  if (NULL != recomb_cast) {
    libxs_registry_destroy(recomb_cast);
    recomb_cast = NULL;
  }
  if (NULL != recomb_pair) {
    libxs_registry_destroy(recomb_pair);
    recomb_pair = NULL;
  }
  recomb_cast_nkeys = 0;
  recomb_pair_nkeys = 0;
  recomb_pair_total = 0;
  recomb_nsentences = 0;
  ref_nsame = 0;
  ref_ncross = 0;
  ref_nunlabelled = 0;
  ref_pmi_same = 0.0;
  ref_pmi_cross = 0.0;
  ref_xsrc = 0;
  ref_xsrc_cross = 0;
  ref_nsame_kept = 0;
  ref_ncross_kept = 0;
  ref_nxnon_kept = 0;
  ref_capped = 0;
}


/**
 * Record one labelled join. Called for every accepted candidate rather than only
 * the composer's first acceptance, because the classes are rare and the composer
 * stops early: sampling one join per host would give a handful of crossings, which
 * is the small-sample problem this probe exists to escape.
 */
/**
 * Best threshold on `pos` against `neg`, by Youden's J. Rates are within-class, so
 * they stay readable under the 50:1 imbalance between the two classes here; raw
 * precision would report a couple of percent for a signal that ranks perfectly.
 */
static double recomb_referent_sweep(const double pos[], int npos,
  const double neg[], int nneg, double* out_cut, double* out_tpr,
  double* out_fpr)
{
  double best = -2.0;
  int ci;
  *out_cut = 0.0;
  *out_tpr = 0.0;
  *out_fpr = 0.0;
  for (ci = 0; ci < npos; ++ci) {
    const double cut = pos[ci];
    long tp = 0, fp = 0;
    int k;
    for (k = 0; k < npos; ++k) {
      if (pos[k] <= cut) ++tp;
    }
    for (k = 0; k < nneg; ++k) {
      if (neg[k] <= cut) ++fp;
    }
    { const double tpr = (double)tp / (double)npos;
      const double fpr = (double)fp / (double)nneg;
      if (tpr - fpr > best) {
        best = tpr - fpr;
        *out_cut = cut;
        *out_tpr = tpr;
        *out_fpr = fpr;
      }
    }
  }
  return best;
}


static void recomb_referent_record(const recomb_signal_t* sig)
{
  if (0 != sig->cross_source) ++ref_xsrc;
  if (1 == sig->crossing) {
    ++ref_ncross;
    ref_pmi_cross += sig->pmi;
    if (0 != sig->cross_source) ++ref_xsrc_cross;
    if (ref_ncross_kept < RECOMB_REF_MAX) ref_cross[ref_ncross_kept++] = sig->pmi;
    else ref_capped = 1;
  }
  else if (0 == sig->crossing) {
    ++ref_nsame;
    ref_pmi_same += sig->pmi;
    if (ref_nsame_kept < RECOMB_REF_MAX) ref_same[ref_nsame_kept++] = sig->pmi;
    else ref_capped = 1;
  }
  else {
    ++ref_nunlabelled;
    if (0 != sig->cross_source) {
      if (ref_nxnon_kept < RECOMB_REF_MAX) ref_xnon[ref_nxnon_kept++] = sig->pmi;
      else ref_capped = 1;
    }
  }
}


/**
 * Co-occurrence of two content lexemes in one sentence, order-independent, so the
 * key is the ordered pair. Counting sentences rather than a sliding window keeps
 * this the same unit the joins are made from.
 */
static void recomb_pair_observe(unsigned int a, unsigned int b)
{
  recomb_pair_key_t key;
  long* count;
  if (a == b || 0 == a || 0 == b) return;
  key.lo = (a < b) ? a : b;
  key.hi = (a < b) ? b : a;
  count = (long*)libxs_registry_get(recomb_pair, &key, sizeof(key), NULL);
  if (NULL != count) ++*count;
  else {
    const long fresh = 1;
    if (NULL != libxs_registry_set(recomb_pair, &key, sizeof(key), &fresh,
      sizeof(fresh), NULL)) ++recomb_pair_nkeys;
  }
  ++recomb_pair_total;
}


static long recomb_pair_count(unsigned int a, unsigned int b)
{
  recomb_pair_key_t key;
  const long* count;
  if (a == b || 0 == a || 0 == b) return 0;
  key.lo = (a < b) ? a : b;
  key.hi = (a < b) ? b : a;
  count = (const long*)libxs_registry_get(recomb_pair, &key, sizeof(key), NULL);
  return (NULL != count) ? *count : 0;
}


/**
 * One pass over sentence-scale entries filling both stores: which sources each
 * content lexeme occurs in (the cast index, used only to LABEL) and how often
 * pairs of content lexemes share a sentence (the association statistic, used only
 * to SIGNAL). They are built together because both need the same walk, but they
 * are read for opposite purposes and must not be confused.
 */
static int recomb_referent_build(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon)
{
  int result = EXIT_SUCCESS;
  recomb_referent_free();
  recomb_cast = libxs_registry_create();
  recomb_pair = libxs_registry_create();
  if (NULL == recomb_cast || NULL == recomb_pair) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const void* key = NULL;
    size_t cursor = 0;
    void* value = corpus_iterx_begin(corpus, &key, &cursor);
    while (NULL != value) {
      const corpus_entry_t* entry = corpus_entry_view(value);
      if (SCALE_SENTENCE == entry->scale && 0 < entry->text_len) {
        recomb_word_t words[COMPOSE_MAXTEXT / 2];
        const int nwords = recomb_words(lexicon, entry->text, entry->text_len,
          words, (int)(sizeof(words) / sizeof(*words)));
        int at, other, prior;
        for (at = 0; at < nwords; ++at) {
          const unsigned int id = words[at].id;
          recomb_referent_t* ref;
          int repeat = 0;
          if (0 == id || 0 != (words[at].flags & LIBXS_LEXEME_STOP)) continue;
          /**
           * Count SENTENCES containing the term, not occurrences of it, so the
           * marginal shares a base with the joint and with the sentence total. A
           * term repeated in one sentence would otherwise push its marginal above
           * the number of sentences and make its PMI negative for no reason.
           */
          for (prior = 0; prior < at; ++prior) {
            if (words[prior].id == id) repeat = 1;
          }
          if (0 != repeat) continue;
          ref = (recomb_referent_t*)libxs_registry_get(recomb_cast, &id,
            sizeof(id), NULL);
          if (NULL == ref) {
            recomb_referent_t fresh;
            memset(&fresh, 0, sizeof(fresh));
            fresh.nsources = 1;
            fresh.source[0] = entry->source;
            fresh.count = 1;
            if (NULL != libxs_registry_set(recomb_cast, &id, sizeof(id),
              &fresh, sizeof(fresh), NULL)) ++recomb_cast_nkeys;
          }
          else {
            int seen = 0, s;
            ++ref->count;
            for (s = 0; s < ref->nsources; ++s) {
              if (ref->source[s] == entry->source) seen = 1;
            }
            if (0 == seen
              && ref->nsources < (int)(sizeof(ref->source)
                / sizeof(*ref->source)))
            {
              ref->source[ref->nsources++] = entry->source;
            }
          }
          for (other = at + 1; other < nwords; ++other) {
            int dup = 0;
            if (0 != (words[other].flags & LIBXS_LEXEME_STOP)) continue;
            /* Once per sentence per pair, for the same reason as the marginal. */
            for (prior = at + 1; prior < other; ++prior) {
              if (words[prior].id == words[other].id) dup = 1;
            }
            if (0 == dup) recomb_pair_observe(id, words[other].id);
          }
        }
        ++recomb_nsentences;
      }
      value = corpus_iterx_next(corpus, &key, &cursor);
    }
  }
  return result;
}


/**
 * Is this lexeme a proper noun proper to exactly one source? That is the tell the
 * label needs: a name occurring in one book only belongs to one cast, so a join
 * pairing two such names from different books crosses referents by construction.
 * ENTITY comes from a capitalization rule rather than a word list, so no corpus
 * vocabulary enters the sample. A minimum count keeps incidental capitalization
 * (a sentence-initial word, a one-off) from being read as a cast member.
 */
static int recomb_is_cast(unsigned int id, unsigned short source)
{
  const recomb_referent_t* ref = (NULL != recomb_cast)
    ? (const recomb_referent_t*)libxs_registry_get(recomb_cast, &id,
        sizeof(id), NULL) : NULL;
  return (NULL != ref && 1 == ref->nsources && ref->source[0] == source
    && 4 <= ref->count) ? 1 : 0;
}


/**
 * The manufactured label: 1 when the two halves provably carry different casts, 0
 * when they provably share one, -1 when neither holds and the join is unlabelled.
 *
 * Note what is NOT claimed. This is not a crossing detector and its rate is not
 * the crossing rate: it sees only crossings marked by a source-exclusive proper
 * noun on BOTH sides of the seam, so pronouns and shared common nouns ("the
 * inspector") are invisible to it. That is acceptable here precisely because it is
 * used as a label rather than as a gate - a class that is small but certain is
 * what a separation test needs, whereas a gate would need the true rate.
 */
static int recomb_referent_label(const corpus_entry_t* a,
  const recomb_word_t awords[], int nawords, int ai,
  const corpus_entry_t* b, const recomb_word_t bwords[], int nbwords, int bi)
{
  int result = -1;
  if (NULL == recomb_cast) return -1;
  if (0 != a->source && a->source == b->source) result = 0;
  else if (0 != a->source && 0 != b->source) {
    int host_cast = 0, donor_cast = 0;
    int at;
    for (at = 0; at <= ai && at < nawords; ++at) {
      if (0 != recomb_is_cast(awords[at].id, a->source)) host_cast = 1;
    }
    for (at = bi + 1; at < nbwords; ++at) {
      if (0 != recomb_is_cast(bwords[at].id, b->source)) donor_cast = 1;
    }
    if (0 != host_cast && 0 != donor_cast) result = 1;
  }
  return result;
}


/**
 * The candidate signal: pointwise mutual information between the host's and the
 * donor's most distinctive content terms, minimized over the pair.
 *
 * This generalizes the syntax gate rather than inventing a test. That gate does
 * not ask whether a seam trigram is attested (an absolute zero test would reject
 * "guards would not", which is fine and occurs zero times); it asks whether the
 * join made the span LESS corpus-like than what it displaced. The same reasoning
 * applies here: most content pairs never co-occur, so a zero-co-occurrence test
 * would reject almost everything. What distinguishes a crossing is co-occurrence
 * far BELOW what the two terms' own frequencies predict - which is negative PMI.
 *
 * The statistic is available because the exact sparse co-occurrence pass already
 * exists for the embedding; the difference is that PPMI clamps at max(pmi,0) and
 * so discards exactly the negative tail this needs.
 *
 * Returns the minimum PMI over cross-half content pairs, in bits. Large negative
 * means the halves talk about things the corpus never puts together.
 */
static double recomb_referent_pmi(const recomb_word_t awords[], int nawords,
  int ai, const recomb_word_t bwords[], int nbwords, int bi,
  unsigned int pivot)
{
  double result = 0.0;
  int found = 0;
  int ah, dh;
  if (NULL == recomb_pair || 0 >= recomb_nsentences) return 0.0;
  for (ah = 0; ah <= ai && ah < nawords; ++ah) {
    const unsigned int ida = awords[ah].id;
    const recomb_referent_t* ra;
    if (0 == ida || ida == pivot) continue;
    if (0 != (awords[ah].flags & LIBXS_LEXEME_STOP)) continue;
    ra = (const recomb_referent_t*)libxs_registry_get(recomb_cast, &ida,
      sizeof(ida), NULL);
    if (NULL == ra || 0 >= ra->count) continue;
    for (dh = bi + 1; dh < nbwords; ++dh) {
      const unsigned int idb = bwords[dh].id;
      const recomb_referent_t* rb;
      if (0 == idb || idb == pivot || idb == ida) continue;
      if (0 != (bwords[dh].flags & LIBXS_LEXEME_STOP)) continue;
      rb = (const recomb_referent_t*)libxs_registry_get(recomb_cast, &idb,
        sizeof(idb), NULL);
      if (NULL == rb || 0 >= rb->count) continue;
      { /**
         * Unseen pairs are floored at half an observation rather than dropped.
         * Dropping them would silently exclude the strongest evidence a crossing
         * can offer (two terms the corpus never once pairs), and an exact zero
         * has no logarithm.
         */
        const double joint = (double)recomb_pair_count(ida, idb);
        const double base = (double)recomb_nsentences;
        const double pj = ((0.0 < joint) ? joint : 0.5) / base;
        const double pa = (double)ra->count / base;
        const double pb = (double)rb->count / base;
        if (0.0 < pa && 0.0 < pb && 0.0 < pj) {
          const double pmi = log(pj / (pa * pb)) / log(2.0);
          if (0 == found || pmi < result) result = pmi;
          found = 1;
        }
      }
    }
  }
  return (0 != found) ? result : 0.0;
}


/**
 * Capacity: how many DISTINCT accepted joins a host admits, not whether it admits
 * one.
 *
 * The composer stops at its first acceptance, so its yield answers "does some
 * acceptable donor exist" and is already 98% on one book - ceiling-limited by
 * construction, and therefore blind to what a larger corpus actually supplies. A
 * corpus with more sources does not make a host more likely to have a donor; it
 * gives that host more donors, each producing a different sentence. Those donors
 * are alternatives rather than evidence: nothing is aggregated and there is no
 * confidence to sharpen, unlike the predictor's kNN vote, where k neighbours
 * combine into one better-supported output. So the quantity that can grow is the
 * size of the reachable set, and it needs its own instrument.
 *
 * Distinct pivots are counted beside the total to separate two ways of growing:
 * more donors for a pivot the host already had, versus genuinely new join points.
 */
static int recomb_capacity_max(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("CONVERSE_RECOMB_CAPMAX");
    cached = (NULL != env && '\0' != *env) ? atoi(env) : 512;
    if (cached < 0) cached = 0;
  }
  return cached;
}


/**
 * Distinct-text set for one host, because a raw count of accepted joins overstates
 * the reachable set: different donors frequently share the suffix after the pivot
 * and so produce the identical sentence. Measured on one folk-tale host, 512
 * accepted joins were only 287 distinct texts, so 44% of the count was duplication.
 * A population of candidates that are the same string is not a population, hence
 * the distinct count is what any search over this space actually has to work with.
 *
 * A registry keyed by the join's hash serves as the set; collisions would
 * undercount, which is the safe direction for a claim about how much is reachable.
 */
static int recomb_seen_reset(libxs_registry_t** seen)
{
  int result = EXIT_SUCCESS;
  if (NULL != *seen) libxs_registry_destroy(*seen);
  *seen = libxs_registry_create();
  if (NULL == *seen) result = EXIT_FAILURE;
  return result;
}


static int recomb_seen_add(libxs_registry_t* seen, const char* text, int len)
{
  int result = 0;
  if (NULL != seen && 0 < len) {
    const unsigned int key = libxs_hash(text, (unsigned int)len, 0);
    if (NULL == libxs_registry_get(seen, &key, sizeof(key), NULL)) {
      const unsigned char one = 1;
      libxs_registry_set(seen, &key, sizeof(key), &one, sizeof(one), NULL);
      result = 1;
    }
  }
  return result;
}


static int recomb_signals(recomb_signal_t* sig, libxs_lexicon_t* lexicon,
  const corpus_entry_t* a, const recomb_word_t awords[], int nawords, int ai,
  const corpus_entry_t* b, const recomb_word_t bwords[], int nbwords, int bi,
  const char* text, int len, int seam, int suffix_end, int window)
{
  int result = EXIT_FAILURE;
  double bits = 0.0;
  if (NULL == sig) return EXIT_FAILURE;
  memset(sig, 0, sizeof(*sig));
  if (EXIT_SUCCESS == recomb_host->seam_bits(text, seam,
    text + seam, len - seam, window, &bits))
  {
    const int nprefix = ai + 1;
    const int nsuffix = nbwords - bi - 1;
    const unsigned int shared = (unsigned int)(a->lexical_flags
      & b->lexical_flags);
    const unsigned int either = (unsigned int)(a->lexical_flags
      | b->lexical_flags);
    int nshared = 0, neither = 0, bit;
    for (bit = 0; bit < 16; ++bit) {
      if (0 != (shared & (1u << bit))) ++nshared;
      if (0 != (either & (1u << bit))) ++neither;
    }
    sig->seam_bits = bits;
    sig->overlap = recomb_overlap(a, b, awords[ai].id);
    sig->gram = recomb_seam_penalty(lexicon, awords, nawords, ai, bwords,
      nbwords, bi);
    sig->balance = (0 < nprefix + nsuffix)
      ? 1.0 - fabs((double)(nprefix - nsuffix)) / (double)(nprefix + nsuffix)
      : 0.0;
    sig->flag_agree = (0 < neither) ? (double)nshared / (double)neither : 1.0;
    /**
     * Junction anomaly, and the one signal here that needs no labels.
     *
     * libxs_fprint_join merges two finalized fingerprints from their accumulators
     * and explicitly does NOT recompute derivatives at the junction, so it predicts
     * what the composite would look like if the seam were unremarkable. Comparing
     * that prediction against the composite's actual fingerprint therefore isolates
     * exactly what the merge omits: the discontinuity the splice introduced. Unlike
     * the seam byte score it is computed over the whole sentence rather than a
     * four-byte window, so it does not inherit that window's tie problem.
     */
    { libxs_fprint_t fa, fb, predicted, actual;
      const size_t na = (size_t)seam;
      const size_t nb = (size_t)(len - seam);
      const size_t nall = (size_t)len;
      if (0 < na && 0 < nb
        && EXIT_SUCCESS == libxs_fprint(&fa, LIBXS_DATATYPE_U8, text, 1, &na,
          NULL, FPRINT_ORDER, 0, 0, 0)
        && EXIT_SUCCESS == libxs_fprint(&fb, LIBXS_DATATYPE_U8, text + seam, 1,
          &nb, NULL, FPRINT_ORDER, 0, 0, 0)
        && EXIT_SUCCESS == libxs_fprint(&actual, LIBXS_DATATYPE_U8, text, 1,
          &nall, NULL, FPRINT_ORDER, 0, 0, 0)
        && EXIT_SUCCESS == libxs_fprint_join(&predicted, &fa, &fb))
      {
        /**
         * Normalized by the actual fingerprint's own magnitude. The raw distance
         * scales with the data, so it correlated 0.799 with sentence length and was
         * measuring size rather than the junction. The ratio asks what FRACTION of
         * the composite's own scale the merge failed to predict, which is the
         * length-free question.
         */
        const double scale = libxs_fprint_diff(&actual, &fa, NULL)
          + libxs_fprint_diff(&actual, &fb, NULL);
        const double raw = libxs_fprint_diff(&predicted, &actual, NULL);
        sig->fpjoin = (1e-30 < scale) ? raw / scale : 0.0;
      }
    }
    sig->pivot_host_pos = ai;
    sig->pivot_donor_pos = bi;
    sig->nwords = nprefix + nsuffix;
    sig->truncated = (suffix_end < b->text_len) ? 1 : 0;
    sig->cross_source = (0 != a->source && a->source != b->source) ? 1 : 0;
    sig->ends_sentence = recomb_host->ends_sentence(text, len);
    sig->pmi = recomb_referent_pmi(awords, nawords, ai, bwords, nbwords, bi,
      awords[ai].id);
    sig->crossing = recomb_referent_label(a, awords, nawords, ai, b, bwords,
      nbwords, bi);
    result = EXIT_SUCCESS;
  }
  return result;
}


static void recomb_signal_print(const recomb_signal_t* sig, const char* text,
  int len, int label)
{
  /**
   * The label is provenance, not a quality judgement: 1 marks a pivot join and 0 a
   * FLOOR join - the same host prefix spliced to an arbitrary entry sharing no
   * content word, which is the control this paper already uses to establish that a
   * seam score means anything. Feature selection needs classes and no ground-truth
   * ranking of joins exists, so the honest label is the one whose two sides are
   * known to differ by construction. A signal that cannot separate a pivot join
   * from an arbitrary splice cannot be expected to order pivot joins among
   * themselves.
   */
  fprintf(stderr, "  sig[y=%d bpc=%.3f ovl=%.2f gram%+.2f bal=%.2f flag=%.2f"
    " fpj=%.4f hpos=%d dpos=%d nw=%d trunc=%d xsrc=%d eos=%d pmi%+.2f"
    " ref=%d] %.*s\n",
    label, sig->seam_bits, sig->overlap, sig->gram, sig->balance,
    sig->flag_agree, sig->fpjoin, sig->pivot_host_pos, sig->pivot_donor_pos,
    sig->nwords, sig->truncated, sig->cross_source, sig->ends_sentence,
    sig->pmi, sig->crossing, len, text);
}


/**
 * Floor counterpart of an accepted join: same host prefix, a suffix taken from an
 * entry that shares no content word with the host, cut at an arbitrary position.
 * Returns the composed length or 0. This is deliberately NOT gated - its purpose
 * is to be a negative example, so passing it through the gates would defeat it.
 */
static int recomb_floor_join(const libxs_registry_t* corpus,
  const corpus_entry_t* a, int prefix_end, long skip,
  const corpus_entry_t** out_donor, char* out, size_t out_size)
{
  int result = 0;
  const void* key = NULL;
  size_t cursor = 0;
  long index = 0;
  void* value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value && 0 == result) {
    const corpus_entry_t* b = corpus_entry_view(value);
    if (index++ >= skip && b != a && SCALE_SENTENCE == b->scale
      && b->text_len > 16)
    {
      int shared = 0;
      int ia, ib;
      for (ia = 0; ia < (int)a->ntokens && ia < ENTRY_TOKEN_MAX && 0 == shared;
        ++ia)
      {
        const unsigned int id = a->token_ids[ia];
        if (0 == id || 0 != (a->token_flags[ia] & LIBXS_LEXEME_STOP)) continue;
        for (ib = 0; ib < (int)b->ntokens && ib < ENTRY_TOKEN_MAX; ++ib) {
          if (b->token_ids[ib] == id) { shared = 1; break; }
        }
      }
      if (0 == shared) {
        result = recomb_splice(a->text, a->text_len, prefix_end,
          b->text, b->text_len, b->text_len / 2, out, out_size);
        if (0 != result && NULL != out_donor) *out_donor = b;
      }
    }
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  return result;
}


/**
 * Second operator: end the donor suffix at an earlier sentence-internal boundary
 * instead of always taking it whole.
 *
 * With a single splice operator, composing twice mostly re-cuts the same suffix, so
 * a second hop yields overwhelmingly duplicates and the reachable set looks much
 * smaller than the candidate count. That is a property of having one operator, not
 * of composition, and it is what makes a one-operator depth measurement the wrong
 * evidence about whether the space is worth searching. Truncating the donor at a
 * clause boundary keeps every word attested and reaches sentences repeated splicing
 * cannot, which is the kind of move a genome with mutation would supply.
 *
 * Returns the number of truncation points written to ends[], including the full
 * suffix as the first entry so the caller's behaviour is unchanged at nmax=1.
 */
static int recomb_suffix_ends(const char* text, int text_len, int from,
  int ends[], int nmax)
{
  int result = 0;
  int at;
  if (0 < nmax) ends[result++] = text_len;
  for (at = from; at < text_len && result < nmax; ++at) {
    if (',' == text[at] || ';' == text[at] || ':' == text[at]) {
      int end = at;
      while (0 < end && 0 != isspace((unsigned char)text[end - 1])) --end;
      if (from + 8 < end) ends[result++] = end;
    }
  }
  return result;
}


static long recomb_capacity(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const corpus_entry_t* a,
  const recomb_word_t awords[], int nawords, int* out_npivots,
  int* out_capped, long* out_ndistinct, libxs_registry_t* seen,
  int depth, long* out_hop2, long* out_hop2_distinct,
  libxs_registry_t* seen2)
{
  long result = 0;
  long ndistinct = 0;
  const int capmax = recomb_capacity_max();
  /**
   * Probe: print every accepted join with the scores a selection rule would rank
   * it by. Selection needs the ranking to be monotone in quality, which is a
   * stronger property than the gates were validated for - they are thresholds,
   * and the same byte model used as a ranker elsewhere in this sample picked the
   * most ordinary candidate rather than the best one. Ranking is therefore
   * measured before anything is built on top of it.
   */
  const int rank_probe = (NULL != getenv("CONVERSE_RECOMB_RANK")) ? 1 : 0;
  const int floor_probe = (NULL != getenv("CONVERSE_RECOMB_FLOOR")) ? 1 : 0;
  const int window = recomb_seam_window();
  /**
   * Number of donor truncation points tried per pivot (1 = full suffix only, the
   * single-operator behaviour). Off by default so the reported capacity keeps
   * measuring one mechanism unless a run asks for more.
   */
  const char* trunc_env = getenv("CONVERSE_RECOMB_TRUNC");
  const int ntrunc = (NULL != trunc_env && '\0' != *trunc_env)
    ? atoi(trunc_env) : 1;
  int npivots = 0;
  int at;
  if (NULL != out_capped) *out_capped = 0;
  for (at = 1; at < nawords - 1; ++at) {
    const unsigned int pivot = awords[at].id;
    const recomb_postings_t* postings;
    int pivot_used = 0;
    int pi;
    if (0 == pivot || 0 != (awords[at].flags & LIBXS_LEXEME_STOP)) continue;
    if (at + 1 < RECOMB_MIN_WORDS) continue;
    if (0 != recomb_is_predicate(pivot)) continue;
    postings = (NULL != recomb_index)
      ? (const recomb_postings_t*)libxs_registry_get(recomb_index,
          &pivot, sizeof(pivot), NULL) : NULL;
    if (NULL == postings) continue;
    for (pi = 0; pi < postings->n; ++pi) {
      const corpus_entry_t* b = postings->at[pi].entry;
      recomb_word_t bwords[COMPOSE_MAXTEXT / 2];
      char candidate[COMPOSE_MAXTEXT];
      int nbwords, bi;
      if (b == a) continue;
      if (0 < capmax && result >= capmax) {
        if (NULL != out_capped) *out_capped = 1;
        break;
      }
      nbwords = recomb_words(lexicon, b->text, b->text_len, bwords,
        (int)(sizeof(bwords) / sizeof(*bwords)));
      for (bi = 1; bi < nbwords - 1; ++bi) {
        int ends[8];
        const int nmax = (int)(sizeof(ends) / sizeof(*ends));
        int nends, ei;
        if (pivot != bwords[bi].id) continue;
        if (nbwords - bi - 1 < RECOMB_MIN_WORDS) continue;
        nends = recomb_suffix_ends(b->text, b->text_len, bwords[bi].end, ends,
          (ntrunc < nmax) ? ntrunc : nmax);
        for (ei = 0; ei < nends; ++ei) {
        const int suffix_end = ends[ei];
        int len;
        len = recomb_splice(a->text, a->text_len, awords[at].end,
          b->text, suffix_end, bwords[bi].end, candidate, sizeof(candidate));
        if (0 != len) {
          const int minovl = recomb_min_overlap();
          const double gramtol = recomb_grammar_tol();
          int ok = 1;
          /**
           * Duplicate candidates are discarded FIRST, because the output is
           * determined by (prefix, suffix) and many donors share a suffix: on the
           * folk tales one text was reached 26 times, and half of all candidates
           * were repeats of another. Testing identity before the gates means a
           * repeat costs a hash lookup rather than a corpus scan, and it makes the
           * counted total the number of distinct sentences rather than the number
           * of ways to reach one. The duplicate share is corpus-dependent (53% on
           * folk tales against 37% on multi-author prose), so it measures how much
           * the corpus repeats itself, not a property of composition.
           */
          if (0 == recomb_seen_add(seen, candidate, len)) ok = -1;
          else if (0 != recomb_balance_on()
            && (0 == recomb_balanced(candidate, len)
              || 0 == recomb_host->ends_sentence(candidate, len))) ok = 0;
          else if (0 < minovl && 100.0 * recomb_overlap(a, b,
            pivot) < (double)minovl) ok = 0;
          else if (0.0 < gramtol && recomb_seam_penalty(lexicon, awords,
            nawords, at, bwords, nbwords, bi) > gramtol) ok = 0;
          else if (0 != recomb_is_verbatim(corpus, candidate, len)) ok = 0;
          if (0 < ok) {
            ++result;
            pivot_used = 1;
            ++ndistinct;
            /**
             * Depth 2: treat the accepted join as a host and compose again. The
             * question is not only how large the reachable set becomes but whether
             * a second hop is coherent at all - the syntax gate compares the seam
             * against the span it displaced in the host, and once the host is
             * itself composite that baseline is no longer corpus text.
             */
            if (1 < depth && NULL != out_hop2) {
              /**
               * The composite is the host of the second hop, so it needs its own
               * entry: overlap and the seam penalty are both computed against the
               * host, and passing the original sentence would score the wrong text.
               * Only the fields those two gates read are needed.
               */
              corpus_entry_t host2;
              recomb_word_t cwords[COMPOSE_MAXTEXT / 2];
              int ncwords, nsub_pivots = 0, sub_capped = 0;
              if (EXIT_SUCCESS == recomb_host->entry_build(&host2,
                (const unsigned char*)candidate, len, SCALE_SENTENCE, lexicon,
                rules, nrules))
              {
                ncwords = recomb_words(lexicon, host2.text, host2.text_len,
                  cwords, (int)(sizeof(cwords) / sizeof(*cwords)));
                *out_hop2 += recomb_capacity(corpus, lexicon, rules, nrules,
                  &host2, cwords, ncwords, &nsub_pivots, &sub_capped,
                  out_hop2_distinct, seen2, 1, NULL, NULL, NULL);
              }
            }
            if (0 != rank_probe || 0 != recomb_referent_on()) {
              recomb_signal_t sig;
              const int seam = awords[at].end;
              if (EXIT_SUCCESS == recomb_signals(&sig, lexicon, a, awords,
                nawords, at, b, bwords, nbwords, bi, candidate, len, seam,
                suffix_end, window))
              {
                if (0 != rank_probe) recomb_signal_print(&sig, candidate, len, 1);
                if (0 != recomb_referent_on()) recomb_referent_record(&sig);
              }
              /* One floor example per accepted join keeps the classes balanced. */
              if (0 != floor_probe) {
                char fl[COMPOSE_MAXTEXT];
                const corpus_entry_t* fdonor = NULL;
                const int flen = recomb_floor_join(corpus, a, awords[at].end,
                  (long)at, &fdonor, fl, sizeof(fl));
                if (0 != flen && NULL != fdonor) {
                  recomb_word_t fwords[COMPOSE_MAXTEXT / 2];
                  const int nfwords = recomb_words(lexicon, fdonor->text,
                    fdonor->text_len, fwords,
                    (int)(sizeof(fwords) / sizeof(*fwords)));
                  recomb_signal_t fsig;
                  int fbi = 0, fw;
                  for (fw = 1; fw < nfwords - 1; ++fw) {
                    if (fwords[fw].end >= fdonor->text_len / 2) { fbi = fw; break; }
                  }
                  if (0 < fbi && EXIT_SUCCESS == recomb_signals(&fsig, lexicon,
                    a, awords, nawords, at, fdonor, fwords, nfwords, fbi, fl,
                    flen, seam, fdonor->text_len, window))
                  {
                    recomb_signal_print(&fsig, fl, flen, 0);
                  }
                }
              }
            }
          }
        }
        }
      }
    }
    if (0 != pivot_used) ++npivots;
  }
  if (NULL != out_npivots) *out_npivots = npivots;
  if (NULL != out_ndistinct) *out_ndistinct += ndistinct;
  return result;
}


/**
 * Try to build one novel sentence by splicing entry a with some later entry that
 * shares a content word. Returns the composed length, 0 if none worked.
 */
static int recomb_compose(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const corpus_entry_t* a,
  const recomb_word_t awords[], int nawords, long skip,
  char* out, size_t out_size, int* pivot_id, const corpus_entry_t** donor,
  double* out_penalty)
{
  int result = 0;
  long best_ordinal = -1;
  int at;
  /**
   * The candidate set comes from the pivot index rather than from a corpus scan,
   * but the WINNER must not change: donor selection is defined as the earliest
   * entry in corpus order that yields an accepted join, so the search is still
   * ordered by entry and the first acceptance for a given entry wins. Ordering by
   * pivot instead - the natural loop shape once an index exists - silently picks
   * a different donor and moves every reported number.
   *
   * Gate order, by contrast, is free to change: the gates are a conjunction, so
   * any order accepts exactly the same joins. Verbatim rejection is the only
   * O(corpus) test and it ran first, so every candidate the cheap gates would have
   * discarded still paid for a full corpus scan; it now runs last.
   */
  for (at = 0; at < nawords; ++at) {
    const unsigned int pivot = awords[at].id;
    const recomb_postings_t* postings;
    int pi;
    if (at < 1 || at >= nawords - 1) continue;
    if (0 == pivot || 0 != (awords[at].flags & LIBXS_LEXEME_STOP)) continue;
    if (at + 1 < RECOMB_MIN_WORDS) continue;
    if (0 != recomb_is_predicate(pivot)) continue;
    postings = (NULL != recomb_index)
      ? (const recomb_postings_t*)libxs_registry_get(recomb_index,
          &pivot, sizeof(pivot), NULL) : NULL;
    if (NULL == postings) continue;
    for (pi = 0; pi < postings->n; ++pi) {
      const corpus_entry_t* b = postings->at[pi].entry;
      const long ordinal = postings->at[pi].ordinal;
      recomb_word_t bwords[COMPOSE_MAXTEXT / 2];
      char candidate[COMPOSE_MAXTEXT];
      int nbwords, bi;
      if (b == a || ordinal < skip) continue;
      if (0 <= best_ordinal && ordinal >= best_ordinal) continue;
      nbwords = recomb_words(lexicon, b->text, b->text_len, bwords,
        (int)(sizeof(bwords) / sizeof(*bwords)));
      for (bi = 1; bi < nbwords - 1; ++bi) {
        int len;
        if (pivot != bwords[bi].id) continue;
        if (nbwords - bi - 1 < RECOMB_MIN_WORDS) continue;
        len = recomb_splice(a->text, a->text_len, awords[at].end,
          b->text, b->text_len, bwords[bi].end, candidate, sizeof(candidate));
        if (0 != len) {
          const int minovl = recomb_min_overlap();
          const double gramtol = recomb_grammar_tol();
          int ok = 1;
          if (0 != recomb_balance_on()
            && (0 == recomb_balanced(candidate, len)
              || 0 == recomb_host->ends_sentence(candidate, len))) ok = 0;
          else if (0 < minovl && 100.0 * recomb_overlap(a, b,
            pivot) < (double)minovl) ok = 0;
          else if (0.0 < gramtol && recomb_seam_penalty(lexicon, awords,
            nawords, at, bwords, nbwords, bi) > gramtol) ok = 0;
          else if (0 != recomb_is_verbatim(corpus, candidate, len)) ok = 0;
          if (0 != ok && (long)len <= (long)out_size) {
            best_ordinal = ordinal;
            result = len;
            memcpy(out, candidate, (size_t)len);
            out[len] = '\0';
            if (NULL != pivot_id) *pivot_id = (int)pivot;
            if (NULL != donor) *donor = b;
            if (NULL != out_penalty) {
              *out_penalty = recomb_seam_penalty(lexicon, awords, nawords,
                at, bwords, nbwords, bi);
            }
            break;
          }
        }
      }
    }
  }
  return result;
}


/**
 * Measure grounded recombination against two controls that bound the scale.
 *
 * The seam score alone means nothing - it has to be read against what fluent and
 * broken junctions cost on the same model. CEILING: verbatim corpus sentences,
 * scored at the same offset (a junction the corpus itself made). FLOOR: splices
 * at an arbitrary word position with no shared pivot. If pivot seams do not land
 * near the ceiling and clearly below the floor, the shared term is not buying
 * fluency and the mechanism is not doing what it claims.
 */
void converse_recomb_probe_run(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  int limit,
  const converse_recomb_host_t* host)
{
  const void* key = NULL;
  size_t cursor = 0;
  long index = 0, nmade = 0, ntried = 0;
  double sum_pivot = 0.0, sum_ceiling = 0.0, sum_floor = 0.0;
  long nceiling = 0, nfloor = 0;
  const int window = recomb_seam_window();
  const int verbose = (NULL != getenv("CONVERSE_RECOMB_SHOW")) ? 1 : 0;
  double sum_overlap = 0.0, sum_penalty = 0.0;
  long nbits = 0;
  long nsection = 0, noverlap0 = 0, nsource = 0;
  long cap_hosts = 0, cap_total = 0, cap_pivots = 0, cap_capped = 0;
  long cap_distinct = 0, cap_hop2 = 0, cap_hop2_distinct = 0;
  /**
   * The referent probe reads every accepted candidate, which only the capacity walk
   * enumerates - the composer itself stops at its first acceptance per host. So it
   * turns that walk on rather than reporting empty classes: a probe whose classes
   * are silently empty is indistinguishable from a probe that measured no effect,
   * which is the same trap the paragraph-scale ingest produced ("made=0 of 0").
   */
  const int capacity_on = (NULL != getenv("CONVERSE_RECOMB_CAPACITY")
    || 0 != recomb_referent_on()) ? 1 : 0;
  const char* depth_env = getenv("CONVERSE_RECOMB_DEPTH");
  const int cap_depth = (NULL != depth_env && '\0' != *depth_env)
    ? atoi(depth_env) : 1;
  libxs_registry_t* cap_seen = NULL;
  libxs_registry_t* cap_seen2 = NULL;
  void* value;
  recomb_host = host;
  if (EXIT_SUCCESS != recomb_index_build(corpus, lexicon)) {
    fprintf(stderr, "recomb: pivot index could not be built\n");
    return;
  }
  if (0 != recomb_nopredicate_on()
    && EXIT_SUCCESS != recomb_predicate_build(corpus, lexicon))
  {
    fprintf(stderr, "recomb: predicate store could not be built\n");
    recomb_index_free();
    return;
  }
  if (0 != recomb_referent_on()
    && EXIT_SUCCESS != recomb_referent_build(corpus, lexicon))
  {
    fprintf(stderr, "recomb: referent stores could not be built\n");
    recomb_index_free();
    return;
  }
  value = corpus_iterx_begin(corpus, &key, &cursor);
  while (NULL != value && nmade < limit) {
    const corpus_entry_t* a = corpus_entry_view(value);
    if (SCALE_SENTENCE == a->scale && a->text_len > 0) {
      recomb_word_t awords[COMPOSE_MAXTEXT / 2];
      const int nawords = recomb_words(lexicon, a->text, a->text_len, awords,
        (int)(sizeof(awords) / sizeof(*awords)));
      char out[COMPOSE_MAXTEXT];
      int pivot_id = 0;
      const corpus_entry_t* donor = NULL;
      double penalty = 0.0;
      int len;
      ++ntried;
      if (0 != capacity_on) {
        int npivots = 0, capped = 0;
        long ncap;
        /**
         * The distinct-text set is per host: two hosts producing the same sentence
         * are two reachable ways to say it, whereas one host reaching it twice is
         * one candidate counted twice.
         */
        if (EXIT_SUCCESS == recomb_seen_reset(&cap_seen)
          && (1 >= cap_depth || EXIT_SUCCESS == recomb_seen_reset(&cap_seen2)))
        {
          ncap = recomb_capacity(corpus, lexicon, rules, nrules, a, awords,
            nawords, &npivots, &capped, &cap_distinct, cap_seen, cap_depth,
            (1 < cap_depth) ? &cap_hop2 : NULL,
            (1 < cap_depth) ? &cap_hop2_distinct : NULL, cap_seen2);
          ++cap_hosts;
          cap_total += ncap;
          cap_pivots += npivots;
          if (0 != capped) ++cap_capped;
        }
      }
      len = recomb_compose(corpus, lexicon, a, awords, nawords, index + 1,
        out, sizeof(out), &pivot_id, &donor, &penalty);
      if (0 < len) {
        double bits = 0.0;
        int seam = 0, have_bits = 0;
        int wi;
        for (wi = 1; wi < nawords; ++wi) {
          if ((int)awords[wi].id == pivot_id) seam = awords[wi].end;
        }
        /**
         * With a judge present this is exactly the historical condition, so the
         * figures stay bit-identical. Without one the candidate still COUNTS -
         * folding the accounting into the diagnostic's success is how a run
         * reports made=0 and reads as "no result" when it means "no instrument".
         */
        have_bits = (0 < seam && seam < len && recomb_have_judge()
          && EXIT_SUCCESS == recomb_host->seam_bits(out,
            seam, out + seam, len - seam, window, &bits)) ? 1 : 0;
        if (0 < seam && seam < len
          && (0 != have_bits || 0 == recomb_have_judge()))
        {
          const double overlap = (NULL != donor)
            ? recomb_overlap(a, donor, (unsigned int)pivot_id) : 0.0;
          /**
           * Same source is a stronger claim than same section, and on the
           * documentation the two disagree: "Usage" and "Example" head many
           * different pages, so equal section text does not mean one document.
           * A cross-source join is the one that combines two sources about one
           * subject - the interesting case - so it is reported separately
           * rather than folded into the section count.
           */
          const int same_src = (NULL != donor && 0 != a->source
            && a->source == donor->source) ? 1 : 0;
          const int same = (NULL != donor && 0 < a->section_len
            && a->section_len == donor->section_len
            && 0 == memcmp(a->section, donor->section,
              (size_t)a->section_len)) ? 1 : 0;
          ++nmade;
          if (0 != have_bits) {
            ++nbits;
            sum_pivot += bits;
          }
          sum_overlap += overlap;
          sum_penalty += penalty;
          if (0 != same) ++nsection;
          if (0 != same_src) ++nsource;
          if (overlap <= 0.0) ++noverlap0;
          if (0 != verbose) {
            fprintf(stderr, "  recomb[%s%.3f bpc ovl=%.2f gram%+.2f %s%s] %.*s\n",
              (0 != have_bits) ? "" : "no-judge ", bits, overlap, penalty,
              (0 != same) ? "same-section" : "CROSS-SECTION",
              (NULL != donor && 0 != a->source)
                ? ((0 != same_src) ? " same-source" : " CROSS-SOURCE") : "",
              len, out);
          }
        }
        /* Ceiling: the same offset inside a real corpus sentence. */
        if (0 < seam && seam < a->text_len && recomb_have_judge()
          && EXIT_SUCCESS == recomb_host->seam_bits(
            a->text, seam, a->text + seam, a->text_len - seam, window, &bits))
        {
          ++nceiling;
          sum_ceiling += bits;
        }
        /* Floor: same prefix, but a suffix taken with no shared pivot. */
        if (0 < seam && nawords > 2 && recomb_have_judge()) {
          const corpus_entry_t* b = NULL;
          const void* fkey = NULL;
          size_t fcursor = 0;
          long fi = 0;
          void* fval = corpus_iterx_begin(corpus, &fkey, &fcursor);
          while (NULL != fval && NULL == b) {
            const corpus_entry_t* cand = corpus_entry_view(fval);
            if (fi++ > index + 7 && SCALE_SENTENCE == cand->scale
              && cand->text_len > seam + 8) b = cand;
            fval = corpus_iterx_next(corpus, &fkey, &fcursor);
          }
          if (NULL != b && EXIT_SUCCESS == recomb_host->seam_bits(
            out, seam, b->text + seam / 2,
            b->text_len - seam / 2, window, &bits))
          {
            ++nfloor;
            sum_floor += bits;
          }
        }
      }
    }
    ++index;
    value = corpus_iterx_next(corpus, &key, &cursor);
  }
  fprintf(stdout, "recomb[window=%d]: made=%ld of %ld tried"
    " | seam bpc: pivot=%.3f ceiling=%.3f floor=%.3f\n",
    window, nmade, ntried,
    (0 < nbits) ? sum_pivot / (double)nbits : 0.0,
    (0 < nceiling) ? sum_ceiling / (double)nceiling : 0.0,
    (0 < nfloor) ? sum_floor / (double)nfloor : 0.0);
  /**
   * Coherence, reported separately because it is a different claim from fluency
   * and the two can disagree: same-section says the halves come from one tale,
   * overlap says they share content words beyond the pivot. A high fluency score
   * with low coherence is the failure mode this measures - a well-formed
   * sentence about two unrelated things.
   */
  fprintf(stdout, "  coherence: same-section=%.1f%% same-source=%.1f%%"
    " mean-overlap=%.2f pivot-only=%.1f%%\n",
    (0 < nmade) ? 100.0 * (double)nsection / (double)nmade : 0.0,
    (0 < nmade) ? 100.0 * (double)nsource / (double)nmade : 0.0,
    (0 < nmade) ? sum_overlap / (double)nmade : 0.0,
    (0 < nmade) ? 100.0 * (double)noverlap0 / (double)nmade : 0.0);
  fprintf(stdout, "  seam grammar: mean word-bits penalty=%+.3f (tol=%.2f)\n",
    (0 < nmade) ? sum_penalty / (double)nmade : 0.0, recomb_grammar_tol());
  /**
   * Truncated postings are reported rather than silently capped: a pivot whose
   * donor list was cut looks exactly like a pivot with few donors, and a bound on
   * coverage that is not printed reads as full coverage.
   */
  fprintf(stdout, "  pivot index: keys=%ld postings=%ld\n",
    recomb_index_nkeys, recomb_index_nposts);
  if (0 != capacity_on) {
    fprintf(stdout, "  capacity: hosts=%ld joins/host=%.1f distinct/host=%.1f"
      " pivots/host=%.2f capped-hosts=%ld (capmax=%d)\n", cap_hosts,
      (0 < cap_hosts) ? (double)cap_total / (double)cap_hosts : 0.0,
      (0 < cap_hosts) ? (double)cap_distinct / (double)cap_hosts : 0.0,
      (0 < cap_hosts) ? (double)cap_pivots / (double)cap_hosts : 0.0,
      cap_capped, recomb_capacity_max());
    if (1 < cap_depth) {
      fprintf(stdout, "  capacity depth2: joins/host=%.1f distinct/host=%.1f"
        " (depth1 distinct=%.1f)\n",
        (0 < cap_hosts) ? (double)cap_hop2 / (double)cap_hosts : 0.0,
        (0 < cap_hosts) ? (double)cap_hop2_distinct / (double)cap_hosts : 0.0,
        (0 < cap_hosts) ? (double)cap_distinct / (double)cap_hosts : 0.0);
    }
  }
  if (0 != recomb_referent_on()) {
    /**
     * The separation, and the baseline it has to beat. Class means alone would
     * overstate the case, so the overlap is reported too: a signal whose classes
     * differ in mean but interleave cannot gate anything. `xsrc` is printed
     * because it is the oracle already in the vector - if PMI does not separate
     * better than "reject every cross-source join", it has bought nothing.
     */
    fprintf(stdout, "  referent: cast-keys=%ld pair-keys=%ld pairs=%ld"
      " | labelled same=%ld crossing=%ld unlabelled=%ld\n",
      recomb_cast_nkeys, recomb_pair_nkeys, recomb_pair_total,
      ref_nsame, ref_ncross, ref_nunlabelled);
    if (0 < ref_nsame && 0 < ref_ncross) {
      const double same_mean = ref_pmi_same / (double)ref_nsame;
      const double cross_mean = ref_pmi_cross / (double)ref_ncross;
      /**
       * SEPARATION, not a distance between the classes.
       *
       * libxs_setdiff_min was the obvious instrument and is the wrong one here: it
       * counts elements that cannot be paired, so with 4617 same against 88
       * crossings it reports the class-size gap (4529) no matter how the values
       * fall. That number looks like a strong negative result and is arithmetic.
       *
       * What a gate would actually do is threshold, so that is what is measured:
       * sweep the crossing values as candidate cutoffs and keep the best AUC-style
       * trade-off. Both rates are expressed WITHIN their own class, which is what
       * makes them readable under a 50:1 imbalance - raw precision cannot be, and
       * would report 2% for a signal that ranks perfectly.
       */
      double cut = 0.0, tpr = 0.0, fpr = 0.0;
      const double youden = recomb_referent_sweep(ref_cross, ref_ncross_kept,
        ref_same, ref_nsame_kept, &cut, &tpr, &fpr);
      fprintf(stdout, "  referent pmi: same=%+.3f crossing=%+.3f"
        " (crossing-same=%+.3f) | n=%d/%d%s\n",
        same_mean, cross_mean, cross_mean - same_mean,
        ref_nsame_kept, ref_ncross_kept,
        (0 != ref_capped) ? " CAPPED" : "");
      fprintf(stdout, "  referent vs same-source: cut=%+.2f bits catches"
        " %.1f%% of crossings at %.1f%% cost (youden=%+.3f) CIRCULAR\n",
        cut, 100.0 * tpr, 100.0 * fpr, youden);
      if (0 < ref_nxnon_kept) {
        /**
         * The test that counts. Both classes are cross-source, so cross_source is
         * constant and cannot contribute; whatever separation remains is the
         * referent signal's own.
         */
        const double yj = recomb_referent_sweep(ref_cross, ref_ncross_kept,
          ref_xnon, ref_nxnon_kept, &cut, &tpr, &fpr);
        fprintf(stdout, "  referent within cross-source: cut=%+.2f bits catches"
          " %.1f%% of crossings at %.1f%% cost (youden=%+.3f) n=%d/%d\n",
          cut, 100.0 * tpr, 100.0 * fpr, yj, ref_nxnon_kept, ref_ncross_kept);
      }
      fprintf(stdout, "  referent baseline xsrc: crossings %ld of %ld"
        " cross-source joins (precision %.1f%%), recall %ld of %ld\n",
        ref_xsrc_cross, ref_xsrc, (0 < ref_xsrc)
          ? 100.0 * (double)ref_xsrc_cross / (double)ref_xsrc : 0.0,
        ref_xsrc_cross, ref_ncross);
    }
    else {
      fprintf(stdout, "  referent: one class empty, no separation measurable"
        " (needs a multi-source corpus with disjoint casts)\n");
    }
  }
  if (NULL != cap_seen) libxs_registry_destroy(cap_seen);
  if (NULL != cap_seen2) libxs_registry_destroy(cap_seen2);
  recomb_referent_free();
  recomb_predicate_free();
  recomb_index_free();
}


int converse_recomb_open(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon,
  const converse_recomb_host_t* host)
{
  int result;
  recomb_host = host;
  result = recomb_index_build(corpus, lexicon);
  if (EXIT_SUCCESS == result && 0 != recomb_nopredicate_on()) {
    result = recomb_predicate_build(corpus, lexicon);
  }
  return result;
}


void converse_recomb_close(void)
{
  recomb_index_free();
  recomb_predicate_free();
  recomb_referent_free();
  recomb_host = NULL;
}


/**
 * Does `a` dominate `b`? Every axis at least as good and one strictly better.
 *
 * Axes are ordered so that LOWER is better, which is why the coherence overlap is
 * negated by the caller: mixing directions here is the mistake a weighted sum
 * hides and a Pareto test exposes as a silent inversion.
 */
/**
 * Does the donor's word sequence CONTAIN the host's, as lexeme ids?
 *
 * Compared over ids rather than bytes because the host is the retrieved answer,
 * which for a fact reply is a normalized quotation of its source: capitalized,
 * terminated with a period, and with the source's line breaks removed. A literal
 * substring test therefore cannot fire - it was tried and did not. Ids are already
 * case-folded and carry no punctuation, so they compare the words themselves.
 */
static int recomb_contains_words(const recomb_word_t hay[], int nhay,
  const recomb_word_t needle[], int nneedle)
{
  int result = 0;
  int at, from = 0;
  /**
   * Leading unknown words are skipped rather than compared. libxs_lexicon_id does
   * NOT case-fold, so a sentence-initial capitalized word yields id 0 - and the
   * host here is a normalized quotation, which capitalizes its first word. Matching
   * on that 0 is what made an earlier version of this guard never fire, and 81% of
   * corpus sentences start with a capital, so this is the common case.
   */
  while (from < nneedle && 0 == needle[from].id) ++from;
  if (from >= nneedle || (nneedle - from) > nhay) return 0;
  for (at = 0; at <= nhay - (nneedle - from) && 0 == result; ++at) {
    int k, same = 1;
    for (k = 0; k < nneedle - from && 0 != same; ++k) {
      if (0 == needle[from + k].id) continue;
      if (hay[at + k].id != needle[from + k].id) same = 0;
    }
    if (0 != same) result = 1;
  }
  return result;
}


static int recomb_dominates(const double a[], const double b[], int n)
{
  int result = 0;
  int strictly_better = 0;
  int at;
  for (at = 0; at < n; ++at) {
    if (a[at] > b[at]) return 0;
    if (a[at] < b[at]) strictly_better = 1;
  }
  if (0 != strictly_better) result = 1;
  return result;
}


int converse_recomb_compose_best(const libxs_registry_t* corpus,
  libxs_lexicon_t* lexicon, const libxs_lexrule_t* rules, int nrules,
  const char* host_text, int host_len, char* out, size_t out_size,
  int* out_nfront, int* out_ncand)
{
  enum { RECOMB_AXES = 3, RECOMB_KEEP = 64 };
  int result = 0;
  corpus_entry_t a;
  recomb_word_t awords[COMPOSE_MAXTEXT / 2];
  int nawords, at;
  int ncand = 0, nfront = 0;
  double cand_axes[RECOMB_KEEP][RECOMB_AXES];
  char cand_text[RECOMB_KEEP][COMPOSE_MAXTEXT];
  int cand_len[RECOMB_KEEP];
  if (NULL != out_nfront) *out_nfront = 0;
  if (NULL != out_ncand) *out_ncand = 0;
  if (NULL == recomb_index || NULL == recomb_host || NULL == host_text
    || 0 >= host_len)
  {
    return 0;
  }
  /**
   * The host is re-encoded rather than looked up: it is the retrieved answer text,
   * which may not be a corpus entry at all (a fact reply is synthesized from the
   * rules). Overlap and the seam penalty are both computed against the host entry,
   * so scoring the wrong text here would silently misrank every candidate.
   */
  if (EXIT_SUCCESS != recomb_host->entry_build(&a,
    (const unsigned char*)host_text, host_len, SCALE_SENTENCE, lexicon, rules,
    nrules))
  {
    return 0;
  }
  nawords = recomb_words(lexicon, a.text, a.text_len, awords,
    (int)(sizeof(awords) / sizeof(*awords)));
  for (at = 1; at < nawords - 1 && ncand < RECOMB_KEEP; ++at) {
    const unsigned int pivot = awords[at].id;
    const recomb_postings_t* postings;
    int pi;
    if (0 == pivot || 0 != (awords[at].flags & LIBXS_LEXEME_STOP)) continue;
    if (at + 1 < RECOMB_MIN_WORDS) continue;
    if (0 != recomb_is_predicate(pivot)) continue;
    postings = (const recomb_postings_t*)libxs_registry_get(recomb_index,
      &pivot, sizeof(pivot), NULL);
    if (NULL == postings) continue;
    for (pi = 0; pi < postings->n && ncand < RECOMB_KEEP; ++pi) {
      const corpus_entry_t* b = postings->at[pi].entry;
      recomb_word_t bwords[COMPOSE_MAXTEXT / 2];
      char candidate[COMPOSE_MAXTEXT];
      int nbwords, bi;
      /**
       * The host cannot donate to itself, and equality is not a strong enough
       * test. The host here is the retrieved ANSWER, which for a fact reply is a
       * rewritten prefix of the sentence it came from - so a donor that CONTAINS
       * the host produces a splice that merely restores the original text. That
       * happened and was reported as a successful composition: the two halves
       * rejoined one corpus sentence which the source file had split across two
       * lines, and `recomb_is_verbatim` could not catch it because the rejoined
       * text (without the line break) occurs nowhere.
       */
      if (NULL == b) continue;
      nbwords = recomb_words(lexicon, b->text, b->text_len, bwords,
        (int)(sizeof(bwords) / sizeof(*bwords)));
      if (0 != recomb_contains_words(bwords, nbwords, awords, nawords)) continue;
      for (bi = 1; bi < nbwords - 1 && ncand < RECOMB_KEEP; ++bi) {
        int len;
        if (pivot != bwords[bi].id) continue;
        if (nbwords - bi - 1 < RECOMB_MIN_WORDS) continue;
        len = recomb_splice(a.text, a.text_len, awords[at].end,
          b->text, b->text_len, bwords[bi].end, candidate, sizeof(candidate));
        if (0 != len && (size_t)len < out_size) {
          recomb_signal_t sig;
          const int seam = awords[at].end;
          if (EXIT_SUCCESS != recomb_signals(&sig, lexicon, &a, awords,
            nawords, at, b, bwords, nbwords, bi, candidate, len, seam,
            bwords[bi].end, recomb_seam_window()))
          {
            continue;
          }
          /**
           * PREREQUISITES, not trade-offs. Measured: leaving ends-a-sentence to
           * compete rewards prefix/suffix symmetry, so selection sought
           * mid-sentence cuts and produced MORE dangling fragments than no
           * selection at all. A property selection can trade away does not survive
           * selection.
           */
          if (0 == sig.ends_sentence) continue;
          if (sig.gram > recomb_grammar_tol()) continue;
          if (0 == recomb_balanced(candidate, len)) continue;
          if (0 != recomb_is_verbatim(corpus, candidate, len)) continue;
          /* Trade-offs, all as lower-is-better; overlap is negated. */
          cand_axes[ncand][0] = sig.seam_bits;
          cand_axes[ncand][1] = -sig.overlap;
          cand_axes[ncand][2] = sig.fpjoin;
          memcpy(cand_text[ncand], candidate, (size_t)len);
          cand_text[ncand][len] = '\0';
          cand_len[ncand] = len;
          ++ncand;
        }
      }
    }
  }
  if (0 < ncand) {
    int best = -1;
    int i, j;
    for (i = 0; i < ncand; ++i) {
      int dominated = 0;
      for (j = 0; j < ncand && 0 == dominated; ++j) {
        if (i != j && 0 != recomb_dominates(cand_axes[j], cand_axes[i],
          RECOMB_AXES))
        {
          dominated = 1;
        }
      }
      if (0 == dominated) {
        ++nfront;
        /**
         * Within the front the objective is indifferent by construction, so the
         * pick cannot be justified by the objective. Taking the first keeps it
         * deterministic and corpus-ordered; the front size is reported so the
         * caller can say that a choice was arbitrary rather than imply it was not.
         */
        if (best < 0) best = i;
      }
    }
    if (0 <= best) {
      memcpy(out, cand_text[best], (size_t)cand_len[best]);
      out[cand_len[best]] = '\0';
      result = cand_len[best];
    }
  }
  if (NULL != out_nfront) *out_nfront = nfront;
  if (NULL != out_ncand) *out_ncand = ncand;
  return result;
}
