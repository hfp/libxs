# Predict Samples

Twelve executables demonstrating fingerprint-guided prediction:

- **predict_params** — Parameter prediction from structured CSV
  (GPU kernel tuning, configuration databases).
- **predict_sunspots** — Timeseries forecasting via sliding-window kNN
  (monthly sunspot numbers, 1749–present).
- **predict_earthquakes** — Spatial prediction of earthquake magnitude
  from location and depth (USGS catalog).
- **predict_discharge** — River discharge forecasting via sliding-window
  kNN with day-of-year seasonality (USGS NWIS daily streamflow).
- **predict_soi** — Southern Oscillation Index prediction from
  anti-correlated Tahiti/Darwin sea level pressure using SPREAD
  decomposition (sum/diff modes).
- **predict_stock** — Paired-stock timeseries prediction from CSV
  using SPREAD decomposition on two correlated price series.
- **predict_crystal** — Crystal system classification from composition
  features (AFLOW ICSD, 7 classes, 60K entries).
- **predict_bits** — Held-out code length under the model against the
  training distribution: whether the inputs inform the output at all.
- **predict_ett** — ETT (Electricity Transformer Temperature) hourly
  forecasting with univariate, multivariate, PCA, and local-attention
  modes (ETTh1).
- **predict_prob** — Probability harness: scores P(y|x) for a candidate
  over the distinct-value support of one output (`libxs_predict_prob`).
- **predict_dist** — Validates `libxs_predict_prob`: normalization,
  agreement of its entry points, and sampling from the distribution.
- **predict_higgs** — Binary classification on the HIGGS dataset
  (28 features); `--help` prints the options.


## Build

    make

Or from the LIBXS root:

    make samples/predict/


## XGBoost Comparison

`predict_params`, `predict_crystal`, `predict_earthquakes`, `predict_ett` and
`predict_higgs` accept an `xgb` keyword that trains XGBoost on exactly the entries
LIBXS was built from and reports both models side by side. It is compiled in when XGBoost is found,
which needs the development headers (a pip-installed `xgboost` ships the shared
library only):

    XGBOOST_ROOT=/path/to/xgboost make    # explicit location
    make                                  # or found via pkg-config
    XGB=0 make                            # never compile it in

Requesting `xgb` from a binary built without it is an error, not a silent
LIBXS-only run.

Keywords may be written anywhere on the command line and numbers fill the
positional slots in order, so `predict_crystal.x data.csv xgb` and
`predict_crystal.x data.csv 0.8 2 0 xgb` are the same run. A token that is
neither a number nor a known keyword is refused. `predict_params` is the
exception in that such a token is taken as the next file name.

Per-output, XGBoost is posed the same task LIBXS chose for itself: a
classification over the attested value set where LIBXS classifies, a regression
where it interpolates. An output with a single attested value is reported as
constant and not trained.

Hyperparameters are fixed so the numbers are reproducible, and overridable:

| Variable      | Default        | Meaning                                   |
| ------------- | -------------- | ----------------------------------------- |
| `XGB_ROUNDS`  | 200            | Boosting rounds                           |
| `XGB_DEPTH`   | 6              | `max_depth`                               |
| `XGB_ETA`     | 0.1            | Learning rate                             |
| `XGB_NTHREAD` | 0              | Threads (0 = XGBoost default)             |
| `XGB_REGOBJ`  | *(per sample)* | Regression objective                      |
| `XGB_LATENCY` | 4096           | Rows the per-query timing uses (0 = skip) |

Timings report XGBoost's phases apart (marshalling, boosting rounds, batched
prediction); only the rounds compare to a LIBXS build. Per-query timings are
reported both one query at a time (`libxs_predict_eval`, and `XGB_LATENCY` rows
through XGBoost's inplace prediction) and batched (`libxs_predict_eval_batch_task`,
and XGBoost's batched call); compare like with like.

`predict_earthquakes` proposes `reg:absoluteerror` for `XGB_REGOBJ`, since it
reports mean absolute error; `predict_params` and `predict_crystal` propose
`reg:squarederror`.

`predict_crystal`, `predict_higgs` and `predict_params` additionally read `GATE`:
a comma-separated list of confidence thresholds (default `0.9`, and `0.5` for
`predict_params`). The first entry drives the
confidence-gated line; supplying more than one also prints a precision/coverage
sweep for both models, which is what compares two confidence signals:

    GATE=0.5,0.6,0.7,0.8,0.9,0.95,0.99 ./predict_crystal.x predict_crystal.csv 0.8 2 0 xgb

`predict_params` also reads `DUMP`: a file that receives one line per validation
query (whether the model was built without it, the inputs, and per varying
parameter the tuned value, the prediction, and the score the gate reads).

### Examples

    ./predict_params.x 0.8 mix xgb /path/to/libxstream/samples/smm/params/tune_multiply_PVC.csv
    ./predict_crystal.x predict_crystal.csv 0.8 2 0 xgb

The parameter CSVs are not part of LIBXS: they ship with LIBXSTREAM in
`samples/smm/params/`.

The parameter CSVs are sorted by problem size, so the default prefix split holds
out the largest shapes and measures extrapolation. Use `mix` to measure
interpolation instead.

`quantile` reports interval coverage split into attested and novel entries; only
the novel column is a calibration.

`GFLOPS` is loaded as an output like every other column. A zero column means it
is withheld and is hidden from the report. Where it is published, the report
additionally weighs every shape by the throughput of its tuned parameters.


## predict_params

Train a prediction model from a CSV file and save it for later use.
Reports validation quality on a held-out subset.

### Usage

    ./predict_params.x [fraction] [auto|cat|compress[Q]|consist[C]|interp|mix|quantile[Q]|rf|hknn|knn|smooth[A]|xgb] [-N] <csvfile> [modelfile [confidence-prefix]]

    fraction   Validation split 0..1 for quality report (default: 0.8).
    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    compress   Drop redundant entries (Q: threshold, default 0.9).
    consist    Round-trip consistency penalty (C: 0..1, default 0.9).
    interp     Force interpolation for all outputs.
    mix        Draw the validation split by co-prime stride rather than as a prefix.
    quantile   Prediction intervals (Q: level 0..0.5, default 0.1).
    rf         Random Forest classification.
    hknn       Hierarchical kNN (Fisher-guided partition).
    knn        Plain kNN.
    smooth     Multi-cluster blending (A: radius, or -1 = auto; default -1).
    xgb        Also train XGBoost on the same split and compare.
    -N         Max polynomial order (default: 0 = auto).
    csvfile    Delimited text file.
    modelfile  Output path for the binary model.
    confidence-prefix  Optional prefix for confidence map files: MHD, and PNG for the slices.

Without `rf`, `hknn` or `knn`, the build chooses the method by trial on entries it holds
back, as the other samples do. Naming a mode fixes it: the saved model keeps only
what that mode reads and cannot be evaluated by another one. A chosen mode leaves
the model open to all of them.

### Example

    ./predict_params.x /path/to/libxstream/samples/smm/params/tune_multiply_PVC.csv
    ./predict_params.x 0.8 hknn tune_multiply_PVC.csv model.bin


## predict_sunspots

Timeseries forecasting using sliding-window nearest-neighbor prediction.

### Usage

    ./predict_sunspots.x <csvfile> [train_fraction] [compress[Q]] [consist[C]] [hknn|rf|none]

    NOPHASE=1  Drop the solar-cycle phase input.
    NOBANK=1   Use a single window view instead of the bank.
    WINDOW=n   Fix the window (default: auto). WINDOW=-1 selects it by
               trial instead, at a multiple of the build time. WINDOW=-w
               also offers w as a candidate.

### Example

    ./predict_sunspots.x predict_sunspots.csv 0.8

The sample carries one auxiliary input alongside the window
(libxs_predict_set_series_aux): the phase of the solar cycle, measured from
the training slice. NOPHASE=1 drops it.

The forecast is averaged over two views of the window
(libxs_predict_set_series_bank), the second reading half the lags of the
first. NOBANK=1 drops back to one. Views are unavailable under a
decomposition.

### Data Source

Monthly mean total sunspot number from
SILSO (World Data Center, Royal Observatory of Belgium).
Semicolon-delimited: year, month, decimal_year, sunspot_number.


## predict_earthquakes

Predict earthquake magnitude from geographic location and depth.

### Usage

    ./predict_earthquakes.x <usgs_csv> [train_fraction] [compress[Q]] [consist[C]] [hknn|rf|none] [xgb]

    consist    Round-trip consistency penalty (C: 0..1, default 0.9).
    none       Plain kNN (otherwise the mode is selected at build by trial).
    xgb        Also train XGBoost on the same split and compare.

The kNN vote reports median or mean as libxs_predict_set_central picks per
output at build time (passing 1 or 2 to it forces median or mean).

### Example

    ./predict_earthquakes.x predict_earthquakes.csv

### Data Source

USGS Earthquake Hazards Program (public domain).
Comma-delimited: time, latitude, longitude, depth, mag, ...


## predict_discharge

River discharge forecasting with day-of-year seasonality and
log-transform on outputs for heavy-tailed data.

### Usage

    ./predict_discharge.x <discharge_tsv> [train_fraction] [compress[Q]] [consist[C]] [hknn|rf|none]

    WINDOW=n   Fix the window (default: auto). A negative value selects it by
               trial; WINDOW=-w also offers w as a candidate.
    BANK=n     Average the forecast over n views of the window (default: 1).

### Example

    ./predict_discharge.x predict_discharge.tsv

### Data Source

USGS National Water Information System (public domain).
Colorado River at Lees Ferry, site 09380000.
Tab-delimited RDB format.


## predict_soi

Southern Oscillation Index prediction from anti-correlated sea level
pressure at Tahiti and Darwin using SPREAD decomposition.

### Usage

    ./predict_soi.x <tahiti_file> <darwin_file> [train_fraction] [compress[Q]] [consist[C]] [hknn|rf]

    WINDOW=n   Fix the window (default: auto). A negative value selects it by
               trial; WINDOW=-w also offers w as a candidate.
    BANK=n     Average the forecast over n views of the window (default: 1).

### Example

    ./predict_soi.x predict_soi_tahiti.dat predict_soi_darwin.dat

### Data Source

NOAA Climate Prediction Center (public domain).
Fixed-width monthly sea level pressure (mb above 1000 mb).


## predict_stock

Multi-stock timeseries prediction with auto-differencing and
PCA/SPREAD decomposition.

### Usage

    ./predict_stock.x <csv_file> [columns] [train_fraction] [compress[Q]] [consist[C]] [hknn|rf]

    columns    Comma-separated 0-based column indices (default: 1,2).

### Example

    ./predict_stock.x stocks.csv 1,2,3


## predict_crystal

Crystal system prediction (7-class classification) from chemical
composition features.

### Usage

    ./predict_crystal.x <crystal_csv> [train_fraction] [order] [nclusters] [compress[Q]] [consist[C]] [fisher|hknn|setdiff|rf|pca|none] [rank|score] [gaps[F]] [xgb]

    consist    Round-trip consistency penalty (C: 0..1, default 0.9).
    fisher     Fisher discriminant feature weighting.
    hknn       Hierarchical kNN (Gini-guided partition).
    setdiff    Setdiff feature selection.
    rf         Random Forest classification.
    pca        Principal component rotation of the inputs.
    none       Raw kNN without feature processing (otherwise the mode is
               selected at build by trial).
    rank       Neighbour modes place inputs by rank.
    score      Neighbour modes place inputs by what they say about the class
               (default: chosen by trial between the two from 8 inputs on).
    gaps       Blank that fraction of input values (F: default 0.1) to exercise
               the absent-value path (libxs_predict_set_missing).
    xgb        Also train XGBoost on the same split and compare.

### Example

    ./predict_crystal.x predict_crystal.csv

### Data Source

AFLOW ICSD catalog (free for academic use).
60,386 entries with Magpie-style composition features
(37 features). Crystal systems: triclinic(1), monoclinic(2),
orthorhombic(3), tetragonal(4), trigonal(5), hexagonal(6),
cubic(7).


## predict_bits

Code length of a held-out stream in bits per event, under the model and
under the training distribution of the same output. Use it to ask whether
the inputs inform the output at all, which an error figure does not answer.

### Usage

    ./predict_bits.x <csvfile> <innames> <outname> [fraction] [vocabulary] [hknn]

    vocabulary  Distinct values the caller considers possible. 0 uses the
                attested support plus one aggregate novel atom, so an
                unattested outcome costs infinite bits; pass a count above
                the support size to keep the figure finite.
    hknn        Non-zero selects the hierarchical partition.

### Example

Run a control alongside any such measurement: a null result is also what a
broken setup reports, so pair the question with one whose answer is known.

    ./predict_bits.x predict_earthquakes.csv latitude,longitude,depth mag 0.8 68
    ./predict_bits.x predict_earthquakes.csv latitude,longitude,mag depth 0.8 9000


## predict_ett

ETT (Electricity Transformer Temperature) hourly forecasting.
Supports univariate and multivariate modes with PCA decomposition
and per-query local-correlation attention.

### Usage

    ./predict_ett.x <ett_csv> [nseries=1..7] [attend|spread|pca|hknn|rf|none|compress[Q]|xgb]

    nseries    Number of input channels (1=OT only, 7=all).
    attend     Per-query local-correlation channel weighting.
    spread     Sum/diff decomposition across channels.
    pca        PCA rotation of multi-channel input space.
    none       No decomposition (otherwise the mode is selected at build by
               trial, which costs one full build per candidate per fold).
    compress   Drop redundant entries (Q: threshold, default 0.9); off unless asked.
    xgb        Also train XGBoost on the same windows and compare.
    WINDOW=n   Fix the window (default: auto). A negative value selects it by
               trial; WINDOW=-w also offers w as a candidate.
    BANK=n     Average the forecast over n views of the window (default: 1).

`xgb` overrides `compress` and is refused alongside `attend`, a decomposition,
or `BANK`. The effective window applies to both models; `WINDOW=96` compares
them at the conventional input length rather than at the auto-selected one.

### Examples

    ./predict_ett.x predict_ett.csv              # univariate OT
    ./predict_ett.x predict_ett.csv 2 pca        # 2ch with PCA
    ./predict_ett.x predict_ett.csv 7 attend     # 7ch with local-attention
    ./predict_ett.x predict_ett.csv 7            # 7ch raw (baseline)

### Data Source

ETTh1 (Electricity Transformer Temperature, hourly) from
Zhou et al. 2021 (Informer). 17,420 hourly readings of an
oil-filled electrical transformer (July 2016 – June 2018).
Seven channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
Target: OT (oil temperature). The sample trains on the first
66.1% (months 1–16, the standard training and validation
months) and tests on the rest, horizon H=96 steps (4 days).

`predict_ett.csv` is ETTh1 as published at
github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv
(a fresh download with CRLF line endings is accepted as well).
