# Tokenizer

Header: `libxs_token.h`

Fixed-width byte-level tokenizer with LZ77-style copy references,
boundary detection, and sentence segmentation. Produces a
reversible token stream from arbitrary UTF-8 input.

## Token Format

Every token is exactly 8 bytes (64 bits):

    Byte 0: control byte
      Bit 7:   copy flag (1 = copy, 0 = literal)
      Bit 4:   sentence-end flag
      Bit 3:   break flag (word/punctuation boundary)
      Bits 2-0: payload length (1..7 bytes)

    Bytes 1-7: payload
      Literal: up to 7 bytes of UTF-8 content
      Copy:    bytes 1-2 = 16-bit backward distance (little-endian)
               bytes 3-7 = reserved

Copy tokens reference content that appeared earlier in the same
stream. The backward distance gives the byte offset (1 = previous
byte). Minimum copy length is 3 bytes, maximum is 7.

## Tokenization

```C
int libxs_tokenize(const unsigned char* text, size_t size,
  libxs_token_stream_t* stream);
```

Tokenize a UTF-8 byte sequence into a stream of fixed-width
tokens. The tokenizer is greedy: at each position it first
attempts a copy match (3-7 bytes referencing earlier content
within 64KB), and falls back to a literal token if no match
is found. Literal length is determined by natural break points
(whitespace, punctuation, case transitions, script changes).

Flags are set automatically during tokenization:

- **Break flag**: set when a word boundary, punctuation mark, or
  character-class transition follows the token.
- **Sentence-end flag**: set when the token ends at a sentence
  terminator (`.` `?` `!`) followed by whitespace or end-of-input.

The stream must be zero-initialized before the first call.
Returns `EXIT_SUCCESS` or `EXIT_FAILURE`.

## Decoding

```C
int libxs_token_decode(const libxs_token_stream_t* stream,
  unsigned char** text, size_t* size);
```

Reconstruct the original byte sequence from a token stream.
Allocates `*text` (caller must free). The roundtrip property
holds: `decode(tokenize(input)) == input`.

## Token Info

```C
typedef struct libxs_token_info_t {
  int length;
  int distance;
  int is_copy;
  int has_break;
  int is_sentence;
  int is_markup;
} libxs_token_info_t;

void libxs_token_info(libxs_token_info_t* info,
  const libxs_token_t* token);
```

Decode all token properties in one call. Fields:

- `length`: number of bytes this token represents (1..7).
- `distance`: backward byte offset for copy tokens (0 if literal).
- `is_copy`: non-zero if the token references earlier content.
- `has_break`: non-zero if a preferred split point follows.
- `is_sentence`: non-zero if the token terminates a sentence.
- `is_markup`: non-zero if the token is structural markup.

This is the primary ABI-stable entry point, suitable for
BIND(C)/Fortran interoperability. All fields are plain `int`.

## Inline Token Queries

```C
int          libxs_token_is_copy(const libxs_token_t* token);
size_t       libxs_token_len(const libxs_token_t* token);
int          libxs_token_has_break(const libxs_token_t* token);
int          libxs_token_is_sentence_end(const libxs_token_t* token);
int          libxs_token_is_markup(const libxs_token_t* token);
unsigned int libxs_token_distance(const libxs_token_t* token);
```

Header-only convenience functions for C callers. NULL-safe
(return 0/false for NULL input). These are equivalent to reading
individual fields from `libxs_token_info_t` but avoid the struct
overhead for single-property checks in conditionals.

## Stream Management

```C
int  libxs_token_stream_reserve(libxs_token_stream_t* stream, size_t cap);
int  libxs_token_stream_push(libxs_token_stream_t* stream, const libxs_token_t* t);
void libxs_token_stream_destroy(libxs_token_stream_t* stream);
```

The stream grows geometrically (doubling). `reserve` pre-allocates
capacity. `destroy` frees all memory but does not free the stream
struct itself (it may be stack-allocated).

## Properties

- **Reversible**: tokenization is lossless -- decoding recovers
  the exact input byte-for-byte.
- **Fixed-width**: every token is 8 bytes regardless of content,
  enabling direct indexing and SIMD-friendly layout.
- **UTF-8 safe**: multi-byte sequences are never split mid-codepoint.
- **Deterministic**: identical input always produces identical output.
- **No vocabulary**: the tokenizer operates on raw bytes without
  learned parameters or lookup tables.

## Break Detection Heuristics

A break flag is set at boundaries between:

- Whitespace and non-whitespace
- Punctuation and non-punctuation
- Alphabetic and numeric characters
- Lowercase and uppercase (camelCase boundaries)

These boundaries serve as natural edit points for downstream
text processing (summarization, retrieval, fusion).

## Sentence Detection

The sentence-end flag fires when a token's content ends with a
sentence terminator (`.` `?` `!`) and the next character is
whitespace, a quote, or end-of-input. This enables sentence
segmentation without a separate parsing pass -- iterate the
token stream and split at sentence-end flags.

## Constants

| Name                        | Value  | Meaning                      |
|-----------------------------|--------|------------------------------|
| `LIBXS_TOKEN_BYTES`         | 8      | Fixed token size in bytes    |
| `LIBXS_TOKEN_PAYLOAD`       | 7      | Usable payload bytes         |
| `LIBXS_TOKEN_MIN_COPY`      | 3      | Minimum copy match length    |
| `LIBXS_TOKEN_MAX_COPY`      | 7      | Maximum copy match length    |
| `LIBXS_TOKEN_MAX_DISTANCE`  | 65535  | Maximum backward distance    |
| `LIBXS_TOKEN_FLAG_COPY`     | 0x80   | Copy token indicator         |
| `LIBXS_TOKEN_FLAG_SENTENCE` | 0x10   | Sentence-end indicator       |
| `LIBXS_TOKEN_FLAG_BREAK`    | 0x08   | Word boundary indicator      |
| `LIBXS_TOKEN_LEN_MASK`      | 0x07   | Payload length field         |

## Example

```C
libxs_token_stream_t stream = {0};
unsigned char* decoded = NULL;
size_t decoded_size = 0;

libxs_tokenize((const unsigned char*)"Hello world.", 12, &stream);

/* iterate tokens */
for (size_t i = 0; i < stream.size; ++i) {
  const libxs_token_t* t = stream.data + i;
  if (libxs_token_is_sentence_end(t)) { /* end of sentence */ }
  if (libxs_token_has_break(t)) { /* word boundary follows */ }
}

/* roundtrip decode */
libxs_token_decode(&stream, &decoded, &decoded_size);
/* decoded == "Hello world." */

free(decoded);
libxs_token_stream_destroy(&stream);
```
