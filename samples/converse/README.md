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

Compose mode ingests one or more files into `converse.dat`, fingerprints and
tokenizes the first input as the target, and emits a short text assembled from
corpus entries:

```bash
./summarize.x -g -n 8 texts/prose1.txt texts/prose2.txt
```

The `-n` option sets the phrase budget in compose mode. The `-g` scorer still
follows the fingerprint trajectory, but it now also rewards non-stopword target
token ID coverage and penalizes repeated generated token IDs. Remove
`converse.dat` to rebuild the local corpus from scratch.

Run the interactive converse sample with one or more corpus files:

```bash
./converse.x -n 3 texts/prejudice.txt texts/prose1.txt
```

Run the built-in sample evaluation over the prose fixture:

```bash
./converse.x -e texts/prose1.txt
./converse.x -P temporal -e texts/prose1.txt
```

The evaluator uses the same ingest, lexicon, stored `libxs_predict` model, and
answer-ranking path as interactive mode. It asks a small set of canned questions
and checks that selected evidence contains expected terms, printing a pass/fail
summary for the top answer, for any selected answer, and for the concise reply.
The cases include direct factual questions, paraphrases, simple grounded
conclusions, and one abstention case. They are tied to `texts/prose1.txt`, so
eval mode expects exactly one fixture corpus rather than a broad `texts/*.txt`
corpus.

Question-shaped prompts use a conservative extractive path. The sample encodes
query and corpus chunks with the lexical token layer, stores compact lexical
sketches in the corpus entries, scores non-stopword token ID overlap, and emits
the best matching evidence. A small bridge table maps a few grounded paraphrase
relations with query-token specs, evidence-text specs, and a concise reply, such
as `find/help/helped way sailor/sailors/ship/ships` to `lighthouse guided/light`.
The sample also loads optional local bridge rules from `converse.bridges`; loaded
rules are tried before the compiled defaults. Each non-comment line has five
pipe-separated fields:

```text
name|query-groups|evidence-groups|score|reply
```

Within query and evidence groups, whitespace separates required groups and `/`
separates alternatives inside a group. Evidence terms can use `_` for a literal
space, for example `Ice_formed/supply_boat`. The reply field may be literal text
or a small answer frame. Built-in frames can fill evidence-backed slots such as
`{after:lighthouse had}`, `{keywords-after:recorded everything:}`, or
`{winter-hardships}`. The `keywords-after` frame tokenizes the selected evidence
span, drops stopwords, de-duplicates token IDs, and composes a compact list.
This lets the sample compose short grounded answers from selected evidence
instead of storing every short answer verbatim. Generic `what is X` questions can
also extract a short phrase around `X`, and unmatched question fallbacks print
the best matching evidence sentence rather than an entire selected paragraph.
For collection-style corpora with uppercase story headings, entries remember the
current heading and questions of the form `In Title, ...` are ranked only against
that story. Overlong sentences and paragraphs are also indexed through bounded
clause fragments so evidence inside long quoted passages remains searchable
without forcing paragraph-length answers. Question words select shallow answer
types such as who, where, when, why, how, and yes/no; these types bias ranking
toward entity, place, number, causal, or method markers from the stored sketches.
Short factual questions prefer sentence-level evidence over
longer paragraph evidence. When a selected top answer matches a grounded
pattern, the chat prints a concise reply; otherwise it falls back to the
selected evidence text. The answer path also trains and saves a `libxs_predict`
model in `converse.prd` from weak query-type labels derived from the corpus
sketches. At chat time the saved model reranks candidate evidence over numeric
sketch features; if no saved model is available, a per-query model is built as a
fallback. If the corpus does not cover enough of the question terms, it answers
`I do not know from the corpus.` Non-question prompts keep using the
fingerprint/Hilbert composition path.

The answer reranker can be run with `-P PROFILE` to compare predictor setups
against the same eval harness. The default `raw` profile uses weighted
`LIBXS_PREDICT_RAW` inputs with `LIBXS_PREDICT_AUTO` output handling,
polynomial order 1, automatic cluster count, and quality 0.0. The `poly2`
profile forces interpolation with order 2, `smooth` enables automatic
multi-cluster smoothing, and `temporal` exercises the standalone timeseries
configuration with `libxs_predict_set_series`, `libxs_predict_set_target`,
`libxs_predict_set_smooth`, and `libxs_predict_set_diff`. Training prints a
compact model summary, for example `predict[persistent:raw]: ...`, and eval
prints `eval[PROFILE]: ...`, so future profile experiments can be compared
directly.

The lexical layer uses compact 8-byte tokens whose first field is a stable
vocabulary ID; the lexicon provides both text-to-ID and ID-to-text mapping.
The tokenizer core only lowercases words; language-specific rewrites such as
`brought -> bring` are passed as caller-owned normalization tables, so a sample
can swap English toy rules for another language profile without changing token
IDs or tokenizer code.
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
directory, writes generated corpus data to `converse.dat`, and saves the
converse vocabulary in `converse.lex` so persisted token IDs can be compared
across runs. The answer reranker is saved in `converse.prd`. Reusing
the same corpus files refreshes existing entries instead of duplicating them. It
does not add a stable public summarization API.
