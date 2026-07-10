# Tokenizer

Header: `libxs_token.h`

Fixed-width byte-level tokenizer with LZ77-style copy references,
boundary detection, sentence segmentation, and loadable text rules.
Produces a reversible token stream from arbitrary byte input.

## Token Format

Every token is exactly 16 bytes (128 bits):

    Byte 0: flags
      Bit 7:   copy flag (1 = copy, 0 = literal)
      Bit 5:   markup flag
      Bit 4:   sentence-end flag
      Bit 3:   break flag (word/punctuation boundary)

    Byte 1: payload length (1..14)

    Bytes 2-15: payload
      Literal: up to 14 bytes of content
      Copy:    bytes 2-5 = 32-bit backward distance (little-endian)
               bytes 6-15 = reserved

Copy tokens reference content that appeared earlier in the same
stream. The backward distance gives the byte offset (1 = previous
byte). Minimum copy length is 3 bytes, maximum is 14.

## Stream Encoding

    void libxs_token_stream_init(libxs_token_stream_t *stream);

    int libxs_token_stream_encode(libxs_token_stream_t *stream,
      const unsigned char *text, size_t size);

Initialize a stream with `libxs_token_stream_init`, then encode a
byte sequence into fixed-width tokens. The encoder is greedy: at each
position it first attempts a copy match (3-14 bytes referencing
earlier content within a 4GB window), and falls back to a literal
token if no match is found. Literal length is determined by natural
break points (whitespace, punctuation, case transitions). Existing
stream contents are kept, so callers can append multiple inputs.

Flags are set automatically during tokenization:

- **Break flag**: set when a word boundary, punctuation mark, or
  character-class transition follows the token.
- **Sentence-end flag**: set when the token ends at a sentence
  terminator (`.` `?` `!`) followed by whitespace or end-of-input.
- **Markup flag**: set when the token content is structural markup
  (LaTeX commands, markdown decoration, bracket sequences).

Returns `EXIT_SUCCESS` or `EXIT_FAILURE`.

## Stream Decoding

    int libxs_token_stream_decode(const libxs_token_stream_t *stream,
      unsigned char **text, size_t *size);

Reconstruct the original byte sequence from a token stream.
Allocates `*text` (caller must free). The roundtrip property
holds: decoding a stream produced by encoding recovers the input.

## Token Info

    typedef struct libxs_token_info_t {
      size_t length;
      unsigned int distance;
      int is_copy;
      int has_break;
      int is_sentence;
      int is_markup;
    } libxs_token_info_t;

    void libxs_token_info(const libxs_token_t *token,
      libxs_token_info_t *info);

Decode all token properties in one call. Fields:

- `length`: number of bytes this token represents (1..14).
- `distance`: backward byte offset for copy tokens (0 if literal).
- `is_copy`: non-zero if the token references earlier content.
- `has_break`: non-zero if a preferred split point follows.
- `is_sentence`: non-zero if the token terminates a sentence.
- `is_markup`: non-zero if the token is structural markup.

## Inline Token Queries

    int          libxs_token_is_copy(const libxs_token_t *token);
    size_t       libxs_token_len(const libxs_token_t *token);
    int          libxs_token_has_break(const libxs_token_t *token);
    int          libxs_token_is_sentence_end(const libxs_token_t *token);
    int          libxs_token_is_markup(const libxs_token_t *token);
    unsigned int libxs_token_distance(const libxs_token_t *token);

Header-only convenience functions. NULL-safe (return 0/false for
NULL input). These are equivalent to reading individual fields from
`libxs_token_info_t` but avoid the struct overhead for
single-property checks.

Note: `libxs_token_is_sentence_end` returns the raw token signal.
Use `libxs_textrule_eval` for rule-refined sentence boundary
detection (handles abbreviations, delimiters, etc).

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

- **Reversible**: tokenization is lossless -- decoding recovers
  the exact input byte-for-byte.
- **Fixed-width**: every token is 16 bytes regardless of content,
  enabling direct indexing and SIMD-friendly layout.
- **UTF-8 safe**: multi-byte sequences are never split mid-codepoint.
- **Deterministic**: identical input always produces identical output.
- **No vocabulary**: the tokenizer operates on raw bytes without
  learned parameters or lookup tables.
- **Rule-extensible**: sentence boundary detection is refined by
  loadable data rules, not hard-coded logic.

## Break Detection Heuristics

A break flag is set at boundaries between:

- Whitespace and non-whitespace
- Punctuation and non-punctuation
- Alphabetic and numeric characters
- Lowercase and uppercase (camelCase boundaries)

These boundaries serve as natural edit points for downstream
text processing (summarization, retrieval, composition).

## Sentence Detection

The sentence-end flag fires when a token's content ends with a
sentence terminator (`.` `?` `!`) and the next character is
whitespace, a quote, or end-of-input. This is the raw signal;
use `libxs_textrule_eval` with a ruleset to suppress false
positives (abbreviations, initials, etc).

## Constants

| Name                        | Value       | Meaning                       |
|-----------------------------|-------------|-------------------------------|
| `LIBXS_TOKEN_BYTES`         | 16          | Fixed token size in bytes     |
| `LIBXS_TOKEN_PAYLOAD`       | 14          | Usable payload bytes          |
| `LIBXS_TOKEN_MIN_COPY`      | 3           | Minimum copy match length     |
| `LIBXS_TOKEN_MAX_COPY`      | 14          | Maximum copy match length     |
| `LIBXS_TOKEN_MAX_DISTANCE`  | 4294967295  | Maximum backward distance     |
| `LIBXS_TOKEN_FLAG_COPY`     | 0x80        | Copy token indicator          |
| `LIBXS_TOKEN_FLAG_MARKUP`   | 0x20        | Markup indicator              |
| `LIBXS_TOKEN_FLAG_SENTENCE` | 0x10        | Sentence-end indicator        |
| `LIBXS_TOKEN_FLAG_BREAK`    | 0x08        | Word boundary indicator       |
| `LIBXS_TOKEN_LEN_MASK`      | 0x0F        | Payload length field (byte 1) |

## Example

    libxs_token_stream_t stream;
    libxs_textrule_t rules[32];
    libxs_textrule_ctx_t ctx;
    int nrules;
    unsigned char *decoded = NULL;
    size_t decoded_size = 0, i, byte_pos = 0;

    libxs_token_stream_init(&stream);
    libxs_token_stream_encode(&stream,
      (const unsigned char*)"Dr. Smith arrived.", 18);
    nrules = libxs_textrule_defaults(rules, 32);

    /* iterate tokens with rule-based sentence detection */
    for (i = 0; i < stream.size; ++i) {
      const libxs_token_t *t = stream.data + i;
      size_t tlen = libxs_token_len(t);
      ctx.text = (const unsigned char*)"Dr. Smith arrived.";
      ctx.text_size = 18;
      ctx.byte_pos = (int)(byte_pos + tlen - 1);
      ctx.token = t;
      ctx.prev_token = (i > 0) ? stream.data + i - 1 : NULL;
      if (libxs_textrule_eval(&ctx, rules, nrules)) {
        /* true sentence boundary (fires at "arrived.", not "Dr.") */
      }
      byte_pos += tlen;
    }

    /* roundtrip decode */
    libxs_token_stream_decode(&stream, &decoded, &decoded_size);
    /* decoded == "Dr. Smith arrived." */

    free(decoded);
    libxs_token_stream_release(&stream);
