# Converse Sample

Converse is a small, self-contained language sample built on the libxs
primitives: the 8-byte `libxs_token_stream_encode` lexical layer (stable
vocabulary IDs and class flags), `libxs_fprint` sentence fingerprints, the
registry, and the `libxs_predict` predictor. It climbs a deliberately
conservative ladder -- grounded extractive question answering, learned relation
and identity facts, and an experimental next-token predictor -- with every step
exercised by a small evaluation harness.

The purpose is a fully inspectable place to build and measure a language
capability from the ground up, keeping all corpus knowledge in caller-owned data
rather than in source. It is not an LLM. For the design rationale and the
measured comparisons behind the different modes, see the companion paper under
`~/papers/converse` (`paper.tex`).

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

## Summarize and compose

Summarize a file by repeatedly fusing adjacent sentences (`-n` sets the target
sentence count):

```bash
./summarize.x texts/prose1.txt
./summarize.x -n 3 texts/prose1.txt
printf '%s\n' 'First sentence. Second related sentence.' | ./summarize.x -
```

Compose mode (`-g`) ingests one or more files into `converse.dat`, fingerprints
and tokenizes the first input as the target, and emits a short assembled text
(`-n` sets the phrase budget). Remove `converse.dat` to rebuild from scratch:

```bash
./summarize.x -g -n 8 texts/prose1.txt texts/prose2.txt
```

## Interactive question answering

Run the interactive sample with one or more corpus files:

```bash
./converse.x -n 3 texts/prejudice.txt texts/prose1.txt
./converse.x -b texts/grimm
```

The `-b` option treats its argument as a file prefix and probes known filenames
without scanning the directory: `name`, `name.txt`, and numbered `.txt` parts
using `name-N.txt`, `name_N.txt`, or `name.N.txt`. Missing siblings are fine as
long as at least one candidate file is usable.

Question-shaped prompts are answered extractively and abstain
(`I do not know from the corpus.`) when coverage is too low. Questions of the
form `In Title, ...` are ranked only against a matching uppercase story heading.
Non-question prompts use the fingerprint/Hilbert composition path.

## Answer evaluation

Run a data-driven evaluation over a local fixture. The sample reads
`converse.eval` from the current directory; keep it next to the fixture text it
describes:

```bash
./converse.x -e -b texts/grimm
./converse.x -P temporal -e texts/prose1.txt
```

Each non-comment `converse.eval` line has three required pipe-separated fields
and an optional fourth:

```text
question|expected-evidence-terms|expected-reply-terms|expected-fact-terms
```

Expected terms are comma-separated. An empty evidence field marks an abstention
case; an empty reply field skips the concise-reply check. The optional fourth
field checks the learned-fact reply path and is evaluated only when relation
facts were learned this run (that is, when a `converse.relations` file is
present), so the same fixture passes with or without rules. Three-field lines
behave exactly as before.

## Local rule files

All corpus-specific vocabulary stays in local, ignored rule files rather than in
`converse.c`.

Relation rules (`converse.relations`) keep aliases, person-like terms, and
filler words out of source. Each non-comment line is one of:

```text
alias|query-relation|evidence-verb
person|term
skip|term
```

For example `alias|eaten|devoured`, `person|grandmother`, `skip|the`. After
ingestion the sample rebuilds an in-memory fact index and reports
`relation facts: N learned` and `identity facts: N learned`. Relation questions
consult this index before falling back to raw evidence. Identity facts of the
form `name is the role` draw their role words from the `person|term` rules and
bind a role to a name only within a single sentence; a `Who is X?` question then
answers from the highest-scoring identity fact for X, or abstains.

Bridge rules (`converse.bridges`) provide optional evidence-backed answer frames.
Each non-comment line has five pipe-separated fields:

```text
name|query-groups|evidence-groups|score|reply
```

Within query and evidence groups, whitespace separates required groups and `/`
separates alternatives; evidence terms can use `_` for a literal space. The
reply may be literal text or a small frame such as `{after:lighthouse had}` or
`{keywords-after:recorded everything:}`.

## Next-token prediction

A separate, experimental next-token predictor is trained from the ordered token
stream of the ingested corpus and kept out of the grounded QA path. It is
exercised through its own flags:

```bash
./converse.x -E -b texts/grimm            # next-token accuracy (default trigram)
./converse.x -E -K bigram -b texts/grimm  # choose the model
printf 'the little\n' | ./converse.x -c -b texts/grimm   # suggestions + greedy
```

`-E` reports next-token accuracy; `-c` reads prompts and prints the top few
next-token suggestions plus a short greedy continuation. `-H N` evaluates on a
held-out split (train on the other sentences, test on 1-in-N) for an honest,
non-memorized accuracy. The model is chosen with `-K`:

- `bigram` -- previous token to next token.
- `trigram` (default) -- previous two tokens, backing off to bigram then to a
  global unigram distribution.
- `predict` -- `libxs_predict` over the previous token IDs, `-P PROFILE`.
- `embed` -- `predict` over distributional token embeddings instead of raw IDs.
- `rerank` -- `libxs_predict` reranks the trigram's candidate successors,
  `-P PROFILE`.

An optional `converse.predict` fixture of `context|expected-next` lines adds a
curated check to `-E`. The predictor profiles for `-K predict|embed|rerank` are
selected with `-P` (`raw`, `poly2`, `smooth`, `temporal`, `rf`, `fisher`,
`hknn`); see the paper for which profiles suit prediction and why.

## Local state files

The sample keeps all state in the sample directory and reuses corpus files by
refreshing existing entries instead of duplicating them. The following are local
and ignored by version control:

- `texts/` -- corpus files.
- `converse.dat` -- persisted corpus registry.
- `converse.lex` -- persisted lexicon and token IDs.
- `converse.prd` -- persisted answer reranker.
- `converse.eval` -- evaluation fixture.
- `converse.relations`, `converse.bridges`, `converse.predict` -- optional rule
  and fixture files.

The sample is intentionally experimental and does not add a stable public
summarization API.
