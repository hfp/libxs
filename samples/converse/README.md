# Converse Sample

Converse reads plain text or Markdown files and then answers questions about them,
summarizes them, assembles new text from them, and trains small next-token models on
them. Answers are extracted from the ingested text and carry a citation, or the sample
abstains; nothing is generated from outside the corpus unless you ask for it.

It is built only on libxs primitives (lexemes, metatokens, fingerprints, registries,
n-grams, parameter prediction) and is self-contained: nothing is downloaded, no
external service is contacted, and no data is used beyond the files you give it. It is experimental and adds no stable
public API.

For the design and the measurements, see `~/papers/converse/paper.tex` and
`~/papers/converse/insights.md`.

## Build

Build the library first, then the sample:

```bash
cd ../..
make PEDANTIC=2
cd samples/converse
make PEDANTIC=2
```

Or, from the `libxs` root, `make PEDANTIC=2 samples/converse`.

## Which binary

| binary | serves | use it for |
| :--- | :--- | :--- |
| `converse-qa.x` | interactive, `-e`, `-c`, `-L` | questions, evaluation, recombination |
| `converse-lm.x` | `-E`, `-c -K KIND`, `-L` | next-token models, byte model |
| `converse-norm.x` | prompt, `-e`, `-t`, `-r`, `-m` | spelling normalization |
| `converse.x` | everything above but normalization | anything, and all documented examples |

`converse.x` accepts every command below. The split binaries link only what they
serve, and a mode a binary does not serve is rejected up front, naming the one that
does. One exception worth knowing: the byte model lives in the LM half, so under
`converse-qa.x` the recombination probe prints `no seam judge` and leaves its bpc
columns at zero. Use `converse.x` if you want those columns.

Run any binary without arguments to print its full option and environment list.

## Getting text in

Put files anywhere; `texts/` is the conventional place and is ignored by git. Two
structures are understood:

- **prose** — paragraphs of running text, blank-line separated. Uppercase lines are
  read as section headings and become the citation for everything under them.
- **Markdown** — headings, paragraphs and code blocks; `#` headings become the
  citation.

The structure is chosen by file extension (`.md` is Markdown, everything else is
prose) and can be forced with `-p prose|markdown`.

Pass files directly, or use `-b PREFIX` to pick up a numbered set without listing it:

```bash
./converse.x texts/prose1.txt texts/prose2.txt
./converse.x -b texts/grimm                  # texts/grimm.txt, grimm-1.txt, ...
./converse.x -b docs ~/libxs/documentation/*.md
```

`-b PREFIX` probes `PREFIX`, `PREFIX.txt` and numbered parts (`PREFIX-N.txt`,
`PREFIX_N.txt`, `PREFIX.N.txt`); missing siblings are fine as long as one file is
usable. Markdown files are not probed, so pass them explicitly (a glob is fine, as
above).

`-b` also names the **state files**: the basename of the prefix is the prefix of
everything the run persists, so `-b docs` keeps `docs.dat`, `docs.lex` and so on, and
several corpora can live side by side in one directory. Without `-b`, state is named
`converse.*`.

Any UTF-8 text works. Suitable sources are Project Gutenberg books, a Wikipedia
extract, your own notes, or the documentation of this repository. Larger corpora are
supported; memory is the practical limit.

### Warm and cold runs

State is reused: a second run over the same files loads the persisted corpus, lexicon
and reranker instead of re-ingesting, which is much faster. To force a full rebuild,
delete the state files for that prefix:

```bash
rm -f grimm.dat grimm.par grimm.src grimm.lex grimm.prd grimm.facts
```

Do this before comparing two runs — a warm run answers from state built by the
previous one.

To build the state without asking anything — useful before a batch of runs, or to
time ingestion on its own — use `-L`:

```bash
./converse.x -L -b texts/grimm
```

## Asking questions

```bash
./converse.x -b texts/grimm
```

The sample then reads one question per line:

```text
> Who is Gretel?
Gretel is the girl.
citation: HANSEL AND GRETEL (texts/grimm.txt:2544)
> Where did Hans go?
Hans went into the stable.
citation: CLEVER HANS (texts/grimm.txt:6031)
> Who was eaten by the wolf?
Grandmother and Little Red-Cap were eaten by the wolf.
citation: LITTLE RED-CAP [LITTLE RED RIDING HOOD] (texts/grimm.txt:3104-3112)
> What belongs to Curdken?
hat belongs to Curdken.
citation: THE GOOSE-GIRL (texts/grimm.txt:1771)
> What do we know about Hansel?
Hansel is the boy. Hansel stood still and peeped back at the house.
citation: HANSEL AND GRETEL (texts/grimm.txt:2544-2579)
> Who is Sherlock Holmes?
I do not know from the corpus.
```

What is answerable depends on the rule files (next section). With the shipped rules
the sample answers:

- `Who is X?` — an identity or a definition ("X is a Y", "X, a Y, ...").
- `Where did X go?` / `Where is X?` — a place, as a proposition from X's own sentence.
- `Who was V by X?` — a relation, including passives it was never told about.
- `What belongs to X?` — an enumeration, each item cited.
- `What do we know about X?` — several cited propositions collected about one name.
- `How are X and Y connected?` / `What connects X and Y?` — the path between two
  entities, stated as the propositions it is made of.
- anything else question-shaped — the best matching sentence, ranked and cited.

Every answer is followed by a citation naming where it came from: the section title
when the corpus has headings, and always the file and the line. An answer assembled from several propositions cites the range of lines it rests on
(`texts/grimm.txt:2544-2579`), one range per file, and a single line when it is one
line. Titles are only printed when the text really carries them — a corpus of flat
prose is cited by file and line alone rather than by a sentence that happened to look
like a heading.

A connection question is answered by stating the facts that join the two entities,
each with its own citation, and never by inventing a relation between them:

```text
> How are McClellan and Douglas connected?
Lincoln restored McClellan. Lincoln forced Douglas.
citation: texts/wiki2m.txt:813-895
```

If nothing in the corpus joins them, the sample says so rather than answering with a
sentence about one of them.

Three behaviours to expect:

- The sample **abstains** (`I do not know from the corpus.`) rather than guessing when
  the corpus does not support an answer.
- A question whose KIND it does not recognize — one using none of the declared `ask|`
  words — still gets an answer, introduced by `I did not recognize the question. The
  closest the corpus comes:` so a relevant sentence is never mistaken for a direct
  answer. That is different from abstaining: the question was not read, rather than
  read and unsupported.
- A question of the form `In TITLE, ...` is ranked only against the matching heading,
  which is how to ask about one document in a corpus of many.

Questions are case-insensitive.

Non-question input is treated as a composition request and answered from the
fingerprint path instead. `-n N` sets how many sentences a response may use.

## Checking answers against a fixture

`-e` runs a fixture of questions and expected terms and prints a pass count, which is
how to tell whether a rule-file or corpus change helped:

```bash
./converse.x -e -b texts/grimm
```

The fixture is `<prefix>.eval` (so `grimm.eval` for `-b grimm`, `converse.eval`
otherwise). Each non-comment line has three required fields and two optional ones:

```text
question|evidence-terms|reply-terms|fact-terms|citation-terms
```

The fifth field checks the citation, and it is matched against the whole citation —
title, file and line — so a fixture can assert `texts/grimm.txt:2544`.

Terms are comma-separated and matched case-insensitively. An empty evidence field
marks an abstention case (the question SHOULD go unanswered); an empty reply field
skips the concise-reply check. The fact field is checked only when relation facts were
learned, so one fixture works with and without rule files.

`-P PROFILE` selects the profile used to rank candidate answers (`raw`, `poly2`,
`smooth`, `temporal`, `rf`, `fisher`, `hknn`), for both interactive use and `-e`:

```bash
./converse.x -P temporal -e texts/prose1.txt
```

## Normalizing spellings

`converse-norm.x` reads the corpus vocabulary as a codebook and maps a surface
form to the nearest word in it, so whatever it answers is a word the corpus
contains. With no mode it prompts, one word per line, and prints the decision
together with the words it chose between:

```bash
./converse-norm.x -b texts/grimm
```

```
> rabbet
rabbet -> rabbit  (distance 1)
    rabbit (261)
> quene
quene: abstained, nearest at 2 with 4 tied
    queen (373)
    quite (242)
```

`-r` sets the largest edit distance a correction may span (default 1) and `-m`
how far the runner-up must be beyond the chosen word (default 1, which accepts
everything). `-e` reports how densely the vocabulary packs and how often a
corrupted word is recovered, which is how to pick the two:

```bash
./converse-norm.x -e -r 1 -m 2 -b texts/grimm
```

**The two knobs trade recall for safety, and the wrong direction is expensive:** a
wrong correction silently replaces the word that was typed. On a book-sized corpus
`-r 2` recovers more but is wrong about a quarter of the time, while `-m 2` cuts
that to a few percent and abstains far more often. Run `-e` on your own corpus
before trusting a setting.

`-t` reads surface forms from a file, one per line, and writes the accepted
corrections to `<prefix>.norms`:

```bash
printf 'rabbet\nqueeen\n' > forms.txt
./converse-norm.x -t forms.txt -b texts/grimm
```

Every binary loads that table on its next run and applies it while encoding, so a
question asked with a corrected spelling answers as the corrected one would.
Forms that decode to themselves, and those with no word in reach, are left out.

## Summarizing and composing

`summarize.x` is a separate, smaller entry point:

```bash
./summarize.x texts/prose1.txt              # fuse adjacent sentences
./summarize.x -n 3 texts/prose1.txt         # to a target sentence count
printf 'First sentence. Second one.' | ./summarize.x -
./summarize.x -g -r -n 8 texts/prose1.txt texts/prose2.txt   # compose
```

- default: summarize one file by repeatedly fusing adjacent sentences.
- `-g`: compose mode — ingest all files, take the first as the target, and emit a
  short assembled text (`-n` is the phrase budget).
- `-r`: reflow text first, joining cosmetic line breaks (useful for Gutenberg files).

## Recombining text from the corpus

An experimental mode splices clauses from different sentences into new ones, keeping
each half verbatim and cutting only at clause boundaries:

```bash
CONVERSE_RECOMB=50 CONVERSE_HIER_RESCORE=1 ./converse.x -E -x -b texts/grimm
```

`CONVERSE_RECOMB=N` attempts N joins and reports how many were made, with coherence
and seam statistics. `CONVERSE_RECOMB_COMPOSE=1` additionally lets the interactive
`-c` mode answer with a spliced sentence; it is off by default because every other
mode emits text that occurs verbatim in the corpus and this one does not.

## Next-token prediction

An experimental predictor is trained from the token stream and kept out of the
question-answering path:

```bash
./converse.x -E -b texts/grimm              # accuracy, default trigram
./converse.x -E -K bigram -b texts/grimm    # pick the model
./converse.x -E -H 10 -b texts/grimm        # held-out: train on 9/10, test on 1/10
printf 'the little\n' | ./converse.x -c -b texts/grimm   # suggestions + continuation
CONVERSE_GRAN=meta-word ./converse.x -E -b texts/grimm   # token unit
```

- `-E` reports accuracy, `-c` prints the top few successors plus a short continuation.
- `-H N` evaluates on a held-out split, `-T PREFIX` on a separate corpus, which is
  what to use for an honest, non-memorized figure.
- `-K` selects the model: `bigram`, `trigram` (default), `predict`, `embed`,
  `rerank`, `knnlm`, or `hier` (evaluation-only: metatoken, byte and PPM bits per
  character).
- `-P PROFILE` selects the predictor profile for `predict|embed|rerank` (`raw`,
  `poly2`, `smooth`, `temporal`, `rf`, `fisher`, `hknn`).
- `-x` uses the deepest n-gram context.
- `CONVERSE_GRAN` sets the token unit: `word`, `native`, `syllable`, `bpe`,
  `meta-native`, `meta-word`, `meta-syllable`.

An optional `<prefix>.predict` fixture of `context|expected-next` lines adds a curated
check to `-E`. Many further knobs exist; run the binary with no arguments to list them.

## Teaching it a language and a corpus

Everything corpus- and language-specific lives in local rule files rather than in the
source, and they are what decide which questions can be answered. Two files, both
optional and both ignored by git:

- `converse.rules` — the LANGUAGE: function words and syntactic classes. A
  `<prefix>.rules` file REPLACES it (for a corpus in another language).
- `<prefix>.relations` — this CORPUS: aliases, role words, place names. It EXTENDS
  the language file, so it need not restate function words.

Each non-comment line is `kind|term`:

| kind | declares | example |
| :--- | :--- | :--- |
| `alias` | a query verb and the verb the text uses | `alias\|eaten\|devoured` |
| `person` | a role word that can be bound to a name | `person\|grandmother` |
| `place` | a place noun | `place\|forest` |
| `where`, `why`, `how` | the markers those questions turn on | `where\|in` |
| `topic` | the marker of an attribute question | `topic\|about` |
| `own` | the verb a possession question uses | `own\|belongs` |
| `poss` | how the language WRITES possession | `poss\|apostrophe-s` |
| `copula`, `article`, `prep` | the syntactic classes definitions need | `copula\|is` |
| `aux` | the auxiliaries | `aux\|had` |
| `agent` | the word a passive names its agent with | `agent\|by` |
| `genitive` | the word that marks a possessor after the thing | `genitive\|of` |
| `join` | how the language writes a joiner inside a name | `join\|ampersand` |
| `link` | the words that ask how two entities relate | `link\|connected` |
| `ask` | a question KIND and this language's word for it | `ask\|who\|who` |
| `pron` | the back-reference pronouns a follow-up points with | `pron\|it` |
| `result` | the light verb introducing a resulting state | `result\|made` |
| `skip`, `negate` | filler words and negators | `skip\|the` |

A few notes that matter in use:

- `where|in` plus `place|forest` is what makes `Where ...?` answerable; declaring only
  one of the two answers nothing.
- `aux|had` and `agent|by` are enough to read passives and active clauses generally —
  the sample derives which words the corpus uses as verbs and as nouns from those
  frames and reports `verbs derived: N` and `nouns derived: N`. No verb list is needed.
- `poss|apostrophe-s` (English) against `poss|apostrophe` (German): declare the wrong
  one and possession answers go silent rather than wrong.
- `ask|` is how the sample knows a question's KIND. The first field is the kind (`who`,
  `what`, `where`, `when`, `why`, `how`, `yesno`), the second is the word — so a German
  rule file writes `ask|who|wer`. A question using none of the declared words still gets
  an answer, prefixed with a note that the question was not recognized.
- `person|father` plus `genitive|of` is what makes kinship answerable in BOTH forms —
  "Lincoln's father Thomas" and "Aegeus, the father of Theseus". A role word you do not
  declare is simply not read, so extend the `person|` class for the kinds of relation
  your text states.

The sample can also PROPOSE rules instead of only reading them:
`CONVERSE_RULES_LEARN=N` offers up to N `person|` class members found in the corpus,
which is a way to bootstrap a rule file for new text. What it proposes is speculative,
so it is reported separately and, if a `<prefix>.learn.eval` fixture exists, that
fixture is used for `-e` while learning is on.

With `CONVERSE_FACTS_LIST=1` the sample also prints the FACTS themselves, plus a
`graph reach:` line (edges, entities, pairs joined by one middle, largest degree) —
useful for judging whether a corpus is big enough to ask connection questions of. Prose
of a few MB gives tens of edges; the tales give none, because a story names few entities
that act on each other.

After ingestion the sample reports what it learned (`relation facts: N learned`,
`identity facts`, `location facts`, `type facts`). Set `CONVERSE_FACTS_LIST=1` to
print the facts themselves — reading them is the only way to tell whether they are
true.

Optional `converse.bridges` adds evidence-backed answer frames, five fields per line:

```text
name|query-groups|evidence-groups|score|reply
```

Whitespace separates required groups and `/` separates alternatives; `_` stands for a
literal space in evidence terms. The reply is literal text or a small frame such as
`{after:lighthouse had}`.

## Files the sample writes

All state is kept in the working directory, named after the `-b` prefix
(`converse.*` without one), and all of it is ignored by git:

| file | holds |
| :--- | :--- |
| `<prefix>.dat` | the ingested corpus |
| `<prefix>.par` | the source texts the corpus refers to; required with `.dat` |
| `<prefix>.lex` | lexicon and token IDs |
| `<prefix>.prd` | the answer reranker |
| `<prefix>.src` | the file names the corpus refers to, for citations |
| `<prefix>.facts` | the derived fact layers, rebuilt when the corpus changes |
| `<prefix>.norms` | spelling corrections, written by `converse-norm.x -t` |

Delete one of `.dat` / `.par` and both are rebuilt. The fixtures and rule files
(`<prefix>.eval`, `<prefix>.learn.eval`, `<prefix>.predict`, `converse.rules`,
`<prefix>.relations`, `converse.bridges`) are inputs you write; the sample never
modifies them.
