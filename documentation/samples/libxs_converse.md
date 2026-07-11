# Converse Sample

This sample explores token-ID fingerprint summarization and composition. It
uses the 8-byte `libxs_token_stream_encode` API for lexical IDs and classes,
`libxs_fprint` to compare sentence fingerprints, and the registry plus Hilbert
keys for a small compose-mode corpus.

## Build

Build the library first so that the sample can link against it:

```bash
cd ../..
make GNU=1 PEDANTIC=2
cd samples/converse
make GNU=1 PEDANTIC=2
```

Or from the `libxs` root:

```bash
make GNU=1 PEDANTIC=2 samples/converse
```

## Usage

Summarize a file by repeatedly fusing adjacent sentences:

```bash
./summarize.x texts/prose1.txt
./summarize.x -n 3 texts/prose1.txt
printf '%s\n' 'First sentence. Second related sentence.' | ./summarize.x -
```

The `-n` option sets the target sentence count in summarization mode.

Compose mode ingests one or more files into `compose.dat`, fingerprints and
tokenizes the first input as the target, and emits a short text assembled from
corpus entries:

```bash
./summarize.x -g -n 8 texts/prose1.txt texts/prose2.txt
```

The `-n` option sets the phrase budget in compose mode. The `-g` scorer still
follows the fingerprint trajectory, but it now also rewards non-stopword target
token ID coverage and penalizes repeated generated token IDs. Remove
`compose.dat` to rebuild the local corpus from scratch.

Run the interactive converse sample with one or more corpus files:

```bash
./converse.x -n 3 texts/prejudice.txt texts/prose1.txt
```

Run the built-in sample evaluation over the prose fixture:

```bash
./converse.x -e texts/prose1.txt
```

The evaluator uses the same ingest, lexicon, stored `libxs_predict` model, and
answer-ranking path as interactive mode. It asks a small set of canned questions
and checks that selected evidence contains expected terms, printing a pass/fail
summary.

Question-shaped prompts use a conservative extractive path. The sample encodes
query and corpus chunks with the lexical token layer, stores compact lexical
sketches in the corpus entries, scores non-stopword token ID overlap, and emits
the best matching evidence. Question words select shallow answer types such as
who, where, when, why, how, and yes/no; these types bias ranking toward entity,
place, number, causal, or method markers from the stored sketches. The answer
path also trains and saves a `libxs_predict` model in `converse.prd` from weak
query-type labels derived from the corpus sketches. At chat time the saved model
reranks candidate evidence over numeric sketch features; if no saved model is
available, a per-query model is built as a fallback. If the corpus does not
cover enough of the question terms, it answers `I do not know from the corpus.`
Non-question prompts keep using the fingerprint/Hilbert composition path.

The lexical layer uses compact 8-byte tokens whose first field is a stable
vocabulary ID; the lexicon provides both text-to-ID and ID-to-text mapping.
Loadable lexical rules assign coordinated classes such as stopword, question,
entity, number, sentence, and markup in one place instead of scattering text
edge cases through samples.

This is still not an LLM. The current question path adds lexical semantics and
answer abstention, but it does not learn paraphrase or derive facts. The current
`libxs_predict` use is a weakly supervised reranker over token-sketch features.
Later steps can train on token IDs and class flags more directly, for example
previous-token windows to next-token ID or sentence-feature to
following-sentence transitions.

## Notes

The sample is intentionally experimental. It keeps all state in the sample
directory, writes generated corpus data to `compose.dat` and `converse.dat`,
and saves the converse vocabulary in `converse.lex` so persisted token IDs can
be compared across runs. The answer reranker is saved in `converse.prd`. Reusing
the same corpus files refreshes existing entries instead of duplicating them. It
does not add a stable public summarization API.
