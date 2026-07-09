# Converse Sample

This sample explores tokenized fingerprint summarization and composition. It
uses `libxs_token` to split input text, `libxs_fprint` to compare sentence
fingerprints, and the registry plus Hilbert keys for a small compose-mode
corpus.

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

Compose mode ingests one or more files into `compose.dat`, fingerprints the
first input as the target, and emits a short text assembled from the nearest
corpus entries:

```bash
./summarize.x -g -n 8 texts/prose1.txt texts/prose2.txt
```

The `-n` option sets the phrase budget in compose mode. Remove `compose.dat`
to rebuild the local corpus from scratch.

## Notes

The sample is intentionally experimental. It keeps all state in the sample
directory, writes generated corpus data to `compose.dat`, and does not add a
stable public summarization API.