# Rosetta

Demonstrates hierarchical type discovery on opaque binary data.
A table of harmonic-series values is encoded as a flat byte blob
with no metadata. The analysis rediscovers the element type,
record stride, and per-field structure -- recovering mathematical
content from anonymous bytes.

## What It Shows

The sample proceeds through the levels described in the algebra
paper (Section "Hierarchical Type Discovery"):

    Step              Tool used                   What it finds
    ----------------------------------------------------------------
    Flat probe        libxs_fprint (UNKNOWN)      inconclusive (expected)
    Stride sweep      libxs_fprint per-column     f64, stride=6
    Shuffle test      libxs_shuffle + fprint      order matters (3-4x)
    Field analysis    libxs_fprint per-field      decay rates per column
    Sort test         libxs_sort_smooth (GREEDY)  already optimally ordered
    Verification      libxs_setdiff_min           0 unmatched, tol=0

Key observations from the output:

  - The flat 1-D probe fails because interleaved fields of different
    scales (1/n mixed with n mixed with ln(n)) look like noise when
    read sequentially. This motivates the stride sweep.

  - The stride sweep requires ALL columns at a candidate width to
    have decay < 1 before accepting it, which eliminates false
    positives from partial correlations at wrong strides.

  - Shuffle stability confirms that the record order carries genuine
    structure (decay increases ~4x after coprime permutation).

  - Per-field analysis reveals:
      [0] decay=0     perfect ramp (sequential index)
      [1] decay~0.63  1/n (decaying but not ultra-smooth)
      [2] decay~0.20  H_n (partial sums, very smooth)
      [5] decay~0.29  converges to Euler-Mascheroni gamma

  - GREEDY sort confirms the data is already optimally ordered
    (the natural 1..64 sequence is the smoothest permutation).

## The Data

For n = 1, 2, ..., 64 the table stores six f64 fields per record:

    [0] n            integer index
    [1] 1/n          harmonic term
    [2] H_n          partial sum of the harmonic series
    [3] ln(n)        natural logarithm
    [4] H_n - ln(n)  difference (converges to gamma from above)
    [5] H_n - ln(n) - 1/(2n)   better gamma estimate

Total: 64 records x 6 fields = 384 doubles = 3072 bytes.

## Build

```bash
cd samples/rosetta
make GNU=1
```

## Usage

```bash
./rosetta.x
```

No arguments. The sample is self-contained: it generates the data,
encodes it as a flat blob, runs the full analysis, and prints the
results.

## Example Output

```
ROSETTA: Recovering structure from opaque bytes
--------------------------------------------------
Ground truth (hidden from the analysis):
  64 records x 6 fields = 384 doubles (3072 bytes)
  Data: harmonic series H_n and derived quantities.
  Field [5] converges to Euler-Mascheroni gamma.
--------------------------------------------------
Level 0: Flat 1-D probe (all bytes as one sequence)
  Input: 3072 opaque bytes
  No type has decay < 1 -- flat stream is not smooth.
  This is expected: interleaved fields of different
  scales look like noise when read sequentially.
--------------------------------------------------
Stride sweep: discovering record layout
  Best type:   f64
  Best stride: 6 elements (48 bytes per record)
  Records:     64
  Avg decay:   0.288244
--------------------------------------------------
Shuffle stability (record-level)
  Avg decay (original): 0.288244
  Avg decay (shuffled): 1.103839
  Ratio:                3.8x
  -> Record order carries structure.
--------------------------------------------------
Per-field decay analysis
  [0] n           decay=0.000000
  [1] 1/n         decay=0.631512
  [2] H_n         decay=0.203226
  [3] ln(n)       decay=0.259621
  [4] H_n-ln(n)   decay=0.342853
  [5] gamma_est   decay=0.292251  (last=0.5771953203, gamma=0.5772156649)
  Smoothest: [0] n (perfect ramp, decay=0)
  -> This reveals a sequential index column.
  Most interesting: [5] gamma_est -- converges to a constant.
--------------------------------------------------
GREEDY sort test (64 rows x 6 cols)
  Data is already optimally ordered for row smoothness.
--------------------------------------------------
Verification: setdiff(original, blob)
  Unmatched: 0, tolerance: 0.00e+00
  -> Byte-perfect recovery.
--------------------------------------------------
```

## Why This Is Interesting

From 3072 anonymous bytes the framework discovers:

  1. The element type (f64) -- not i32, not f32, not i64.
  2. The record structure (6-field records of 48 bytes each).
  3. A sequential index column (field [0], decay = 0).
  4. A column converging to Euler-Mascheroni gamma (field [5]).
  5. That the natural ordering is already optimal (no resorting).

No metadata, no format knowledge, no human guidance. The hierarchical
composition -- stride sweep with all-columns-must-pass filtering,
shuffle stability, per-field decay ranking -- is what makes this
possible. Any single tool alone would either fail (flat probe) or
produce ambiguous results (stride sweep without the all-columns
requirement accepts wrong strides).
