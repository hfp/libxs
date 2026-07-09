# Tokenizer Sample

This sample demonstrates the reversible byte-level tokenizer API in
`libxs_token.h`:

- fixed-width 64-bit tokens stored as 8 bytes,
- 1 control byte plus 7 payload bytes,
- literal-run tokens that carry up to 7 UTF-8 bytes directly,
- copy tokens that reference earlier spans for simple compression,
- a deterministic byte-level round trip.

The canonical text stream is UTF-8 bytes. The sample does not depend on a
vocabulary table or a learned merge list. It only uses a few byte-level split
heuristics so the literal runs tend to stop at natural boundaries such as
whitespace, punctuation, digit-to-letter transitions, and lower-to-upper case
changes.

The sample is intentionally small. It gives us a place to inspect three API
properties cheaply:

1. Is the fixed 1-byte control plus 7-byte payload layout expressive enough?
2. Do the simple boundary rules produce useful chunks on mixed-language UTF-8?
3. Is the token stream a sensible feature source for later `libxs_predict`
   experiments, for example kNN or random-forest reply retrieval?

## Build

Build the library first so that the sample can link against it:

```bash
cd ../..
make GNU=1 PEDANTIC=2
cd samples/tokenizer
make GNU=1 PEDANTIC=2
```

Or from the `libxs` root:

```bash
make GNU=1 PEDANTIC=2 samples/tokenizer
```

## Usage

```bash
./tokenizer.x
./tokenizer.x "token tokenization tokenizer token"
./tokenizer.x "MixedCase42 token token MixedCase42"
printf '%s\n' 'UTF-8 text here' | ./tokenizer.x -
```

If no argument is given, the sample runs on a built-in ASCII string. Pass `-`
to read UTF-8 input from standard input.

The program prints each token, decodes the stream back to bytes, and checks
that the round trip matches the original input.

## Layout

The control byte uses the following layout:

```text
bit 7     family   0 = literal, 1 = copy
bit 6..4  flags    reserved for future subtype or metadata bits
bit 3     break    token ended at a preferred split boundary
bit 2..0  len      payload length in bytes, 1..7
```

Literal tokens store the bytes directly in payload positions 1..7. Copy tokens
store a 16-bit backward distance in payload bytes 1 and 2; the remaining bytes
are reserved for future metadata experiments without changing the fixed token
width.

## Next Step

If this direction holds up, the next sample should stay separate from the
tokenizer and live under `samples/predict`, for example `predict_reply.c`. That
sample could turn token streams into fixed-size classic-ML features such as
hashed n-grams, token histograms, or nearest-neighbor windows, and then use
`libxs_predict` for reply retrieval or ranking.
