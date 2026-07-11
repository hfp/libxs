# Tokenizer

Header: `libxs_token.h`

Fixed-width lexical tokenizer with stable vocabulary IDs, lightweight
classification flags, optional caller-owned normalization tables, and loadable
text/lexical rules.

## Token Format

Every token is exactly 8 bytes:

    Bytes 0-3: vocabulary ID
    Bytes 4-5: source token byte length
    Bytes 6-7: class flags

The vocabulary ID indexes normalized token text in a `libxs_lexicon_t`.
The original source bytes are not stored in the token; callers that need text
use the lexicon for normalized ID-to-text mapping.

Flags mark word, number, punctuation, markup, sentence, question, stopword,
entity, and break classes.

## Stream Encoding

    void libxs_token_stream_init(libxs_token_stream_t *stream);

    int libxs_token_stream_encode(libxs_lexicon_t *lexicon,
      libxs_token_stream_t *stream, const unsigned char *text, size_t size,
      const libxs_lexrule_t *rules, int nrules,
      const libxs_lexnorm_t *norms, int nnorms, int create);

Initialize a stream with `libxs_token_stream_init`, create or load a lexicon,
then encode text into fixed-width lexical tokens. Words are lowercased first.
If `norms` is non-NULL, each normalized word can be rewritten by the caller's
normalization table before lexicon lookup. This keeps language-specific
inflection, stemming, or spelling policy outside the tokenizer core. Pass
`NULL, 0` for no language normalization.

Lexical rules (`rules`, `nrules`) classify normalized tokens after optional
normalization. The `create` flag controls whether missing normalized text is
interned into the lexicon or yields ID 0.

Flags are set automatically during tokenization:

- **Word/number/punctuation flags**: set by basic token shape.
- **Break flag**: set when whitespace preceded the token.
- **Sentence/question flags**: set for `.`, `?`, and `!` punctuation tokens.
- **Markup flag**: set when punctuation content is structural markup.
- **Stop/entity and other semantic flags**: assigned by lexical rules.

Returns `EXIT_SUCCESS` or `EXIT_FAILURE`.

## Token Info

    typedef struct libxs_token_info_t {
      unsigned int id;
      size_t length;
      unsigned int flags;
      int is_word;
      int is_number;
      int is_punct;
      int has_break;
      int is_sentence;
      int is_question;
      int is_stop;
      int is_entity;
      int is_markup;
    } libxs_token_info_t;

    void libxs_token_info(const libxs_token_t *token,
      libxs_token_info_t *info);

Decode all token properties in one call. Fields:

- `id`: normalized vocabulary ID, or 0 for unknown.
- `length`: source byte length of the token.
- `flags`: raw class flag bitset.
- `is_word`, `is_number`, `is_punct`: basic token class.
- `has_break`: non-zero if whitespace preceded the token.
- `is_sentence`: non-zero if the token is a sentence terminator.
- `is_question`: non-zero if the token is `?`.
- `is_stop`: non-zero if lexical rules classify it as a stopword.
- `is_entity`: non-zero if lexical rules classify it as an entity marker.
- `is_markup`: non-zero if the token is structural markup.

## Stream Management

    void libxs_token_stream_init(libxs_token_stream_t *stream);
    int  libxs_token_stream_reserve(libxs_token_stream_t *stream, size_t cap);
    int  libxs_token_stream_push(libxs_token_stream_t *stream, const libxs_token_t *t);
    void libxs_token_stream_release(libxs_token_stream_t *stream);

The stream grows geometrically (doubling). `reserve` pre-allocates
capacity. `release` frees all memory but does not free the stream
struct itself (it may be stack-allocated), then resets the fields.

## Text Rules

Text rules refine raw token signals using loadable knowledge. The
primary use case is sentence-boundary detection: the raw
sentence-end flag fires on every `.` `?` `!`, but rules can
suppress false positives (e.g., "Mr." is not a sentence end).

### Rule Structure

    typedef struct libxs_textrule_t {
      unsigned char tmpl;      /* template ID */
      unsigned char action;    /* SUPPRESS or CONFIRM */
      unsigned char reserved[2];
      unsigned int argument;   /* template-specific parameter */
      unsigned char pad[8];
    } libxs_textrule_t;

Each rule is 16 bytes, directly storable in a registry. The `tmpl`
field selects a compiled match pattern (template); `argument`
parameterizes it; `action` determines the outcome when matched.

### Templates

| ID | Name                        | Matches when...                                |
|----|-----------------------------|-------------------------------------------------|
| 1  | `PREV_WORD_SHORT_UPPER`     | Word before position is 1-4 chars, starts upper |
| 2  | `NEXT_CHAR_UPPER`           | First non-space char after position is uppercase|
| 3  | `NEXT_CHAR_LOWER`           | First non-space char after position is lowercase|
| 4  | `PREV_TOKEN_BREAK`          | Previous token has the BREAK flag               |
| 5  | `INSIDE_DELIMITERS`         | Position is inside paired delimiters            |

For template 1, if `argument` is 0 it matches any qualifying word;
if non-zero it must equal `libxs_textrule_wordhash(word, len)`.

For template 5, `argument & 0xFF` is the opening delimiter character.

### Evaluation

    int libxs_textrule_eval(const libxs_textrule_ctx_t *ctx,
      const libxs_textrule_t *rules, int nrules);

Returns 1 if the position is a sentence boundary, 0 if suppressed.
If the token lacks the sentence-end flag, returns 0 without
consulting rules. Rules are evaluated in order; last match wins.

The context struct provides the evaluation window:

    typedef struct libxs_textrule_ctx_t {
      const unsigned char *text;
      size_t text_size;
      int byte_pos;              /* position of the sentence-end character */
      const libxs_token_t *token;
      const libxs_token_t *prev_token;
    } libxs_textrule_ctx_t;

### Persistence

    int libxs_textrule_load(const libxs_registry_t *reg,
      libxs_textrule_t *rules, int max_rules);

    int libxs_textrule_save(libxs_registry_t *reg,
      const libxs_textrule_t *rules, int nrules);

Rules are stored in a registry under key prefix `TRULE:` followed
by a 2-byte sequence number. Load returns the number of rules
read; save writes them sequentially.

### Defaults

    int libxs_textrule_defaults(libxs_textrule_t *rules, int max_rules);

Populates `rules[]` with built-in knowledge (abbreviation
suppression). Returns the number of rules written. Use this as a
starting point; additional rules can be appended or loaded from a
registry.

### Word Hash

    unsigned int libxs_textrule_wordhash(const unsigned char *word, int len);

Produces a 32-bit hash for use as rule argument. Pass this to
`argument` in a `PREV_WORD_SHORT_UPPER` rule to match a specific
word rather than any short uppercase-initial word.

## Properties

- **Fixed-width**: every token is 8 bytes regardless of content,
  enabling direct indexing and compact storage.
- **UTF-8 safe**: multi-byte sequences are never split mid-codepoint.
- **Deterministic**: identical input, lexicon state, lexical rules, and
  normalization table produce identical IDs.
- **Vocabulary-backed**: normalized token text is stored once in a lexicon and
  referenced by stable token IDs.
- **Language-swappable**: caller-owned normalization tables decide whether and
  how inflections or spelling variants collapse to a common vocabulary entry.
- **Rule-extensible**: lexical classes and sentence boundary refinements are
  data rules, not hard-coded language policy.

## Break Detection

The break flag is set when whitespace preceded the token. Downstream code can
combine this with punctuation and lexical class flags to recover token spacing
or detect phrase boundaries.

## Sentence Detection

The sentence flag fires on punctuation tokens for `.`, `?`, and `!`; the
question flag is also set for `?`. This is the raw signal; use
`libxs_textrule_eval` with a ruleset to suppress false positives
(abbreviations, initials, delimiters, etc).

## Constants

| Name                        | Value       | Meaning                       |
|-----------------------------|-------------|-------------------------------|
| `LIBXS_TOKEN_BYTES`         | 8           | Fixed token size in bytes     |
| `LIBXS_TOKEN_MAXBYTES`      | 63          | Maximum normalized text bytes |
| `LIBXS_TOKEN_WORD`          | 0x0001      | Word token                    |
| `LIBXS_TOKEN_NUMBER`        | 0x0002      | Number token                  |
| `LIBXS_TOKEN_PUNCT`         | 0x0004      | Punctuation token             |
| `LIBXS_TOKEN_MARKUP`        | 0x0008      | Markup-like punctuation       |
| `LIBXS_TOKEN_SENTENCE`      | 0x0010      | Sentence terminator           |
| `LIBXS_TOKEN_QUESTION`      | 0x0020      | Question mark                 |
| `LIBXS_TOKEN_STOP`          | 0x0040      | Stopword class                |
| `LIBXS_TOKEN_ENTITY`        | 0x0080      | Entity class                  |
| `LIBXS_TOKEN_BREAK`         | 0x0100      | Whitespace before token       |

## Example

    libxs_token_stream_t stream;
    libxs_lexicon_t *lexicon;
    libxs_lexrule_t rules[96];
    libxs_lexnorm_t norms[1] = { { "arrived", "arrive" } };
    int nrules;
    size_t i;

    libxs_token_stream_init(&stream);
    lexicon = libxs_lexicon_create();
    nrules = libxs_lexrule_defaults(rules, 96);
    libxs_token_stream_encode(lexicon, &stream,
      (const unsigned char*)"Dr. Smith arrived.", 18,
      rules, nrules, norms, 1, 1);

    /* iterate normalized lexical tokens */
    for (i = 0; i < stream.size; ++i) {
      int len = 0;
      const char *txt = libxs_lexicon_text(lexicon, stream.data[i].id,
        &len, NULL);
      /* "arrived" is stored as "arrive" because norms was supplied. */
    }

    libxs_token_stream_release(&stream);
    libxs_lexicon_destroy(lexicon);
