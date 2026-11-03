# Setdiff

Runs small deterministic experiments for `libxs_setdiff` and
`libxs_setdiff_min` and writes the results as CSV files.

## Build

```bash
cd samples/setdiff
make
```

## Usage

```bash
./setdiff.x [outdir [sizes [reps [land_n [land_steps]]]]]
```

| Positional | Default                 | Description |
|------------|-------------------------|-------------|
| outdir     | `-`                     | Existing directory for CSV files, `-` for stdout |
| sizes      | `128 1024 8192 65536`   | Vector sizes for tolerance and scaling runs, as one quoted argument separated by spaces |
| reps       | `10`                    | Repetitions per scaling size |
| land_n     | `32`                    | Tolerance-landscape vector length |
| land_steps | `40`                    | Number of tolerance grid steps |

## Output

- `summary.csv` — order-independence and duplicate-consumption cases
- `landscape.csv` — sampled tolerance landscape
- `tolerance.csv` — `libxs_setdiff_min` compared with bisection
- `scaling.csv` — fixed-tolerance and automatic-tolerance timings
- `complex.csv` — complex-valued validation case

The output directory must exist; the sample does not create it and fails
if it cannot write there. The sample uses a fixed pseudo-random seed, and
timings in `scaling.csv` are CPU time (`clock`).
