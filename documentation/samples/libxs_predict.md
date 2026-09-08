# Predict Samples

Nine executables demonstrating fingerprint-guided prediction:

- **predict_params** — Parameter prediction from structured CSV
  (GPU kernel tuning, configuration databases).
- **predict_sunspots** — Timeseries forecasting via sliding-window kNN
  (monthly sunspot numbers, 1749-present).
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
  modes (ETTh1, standard benchmark for timeseries LLMs).
## Build

    make

Or from the LIBXS root:

    make samples/predict
## XGBoost Comparison

`predict_params`, `predict_crystal`, `predict_earthquakes` and `predict_ett`
accept an `xgb` keyword that trains XGBoost on **exactly** the entries LIBXS was
built from and reports both models side by side.  It is compiled in when XGBoost is found,
which needs the development headers (a pip-installed `xgboost` ships the shared
library only):

    XGBOOST_ROOT=/path/to/xgboost make    # explicit location
    make                                  # or found via pkg-config
    XGB=0 make                            # never compile it in

Requesting `xgb` from a binary built without it is an error, not a silent
LIBXS-only run.

Keywords may be written anywhere on the command line and numbers fill the
positional slots in order, so `predict_crystal.x data.csv xgb` and
`predict_crystal.x data.csv 0.8 2 0 xgb` are the same run.  A token that is
neither a number nor a known keyword is refused: it used to land in the next
numeric slot as `atof("xgb") == 0`, which trained on two rows and reported an
accuracy for them.  `predict_params` is the exception in that its keywords must
still precede the file names, which cannot be told from keywords by shape.

Per-output, XGBoost is posed the same task LIBXS chose for itself: a
classification over the attested value set where LIBXS classifies, a regression
where it interpolates.  An output with a single attested value is reported as
constant and not trained (five of the sixteen kernel parameters are constant in
`tune_multiply_PVC.csv`, so an average over all outputs is dominated by columns
both models get for free).

Hyperparameters are fixed so the numbers are reproducible, and overridable:

| Variable      | Default        | Meaning                       |
| ------------- | -------------- | ----------------------------- |
| `XGB_ROUNDS`  | 200            | Boosting rounds               |
| `XGB_DEPTH`   | 6              | `max_depth`                   |
| `XGB_ETA`     | 0.1            | Learning rate                 |
| `XGB_NTHREAD` | 0              | Threads (0 = XGBoost default) |
| `XGB_REGOBJ`  | *(per sample)* | Regression objective          |

`XGB_REGOBJ` matters where the reported metric is mean absolute error:
`predict_earthquakes` therefore proposes `reg:absoluteerror`, and forcing
`reg:squarederror` there moves the magnitude error 0.237 to 0.254, which is
enough to invert the comparison.  `predict_params` and `predict_crystal` propose
`reg:squarederror`.

`predict_crystal` additionally reads `GATE`: a comma-separated list of
confidence thresholds (default `0.9`).  The first entry drives the
confidence-gated line; supplying more than one also prints a precision/coverage
sweep for both models.  Comparing two confidence signals at one threshold
compares two different operating points, so a sweep is the only way to tell a
better-calibrated signal from a differently-scaled one:

    GATE=0.5,0.6,0.7,0.8,0.9,0.95,0.99 ./predict_crystal.x predict_crystal.csv 0.8 2 0 xgb

### Examples

    ./predict_params.x 0.8 mix xgb ../smm/params/tune_multiply_PVC.csv
    ./predict_crystal.x predict_crystal.csv 0.8 2 0 xgb

The `mix` keyword matters for the parameter CSVs: they are sorted by problem
size, so the default prefix split holds out the *largest* shapes and measures
extrapolation, which changes which model looks better.  Use `mix` for
interpolation, omit it for extrapolation, and say which one a number came from.

Exact-match on kernel parameters is a proxy for kernel quality: the shipped
CSVs carry no `GFLOPS`, so a differently-parameterized kernel that performs
identically counts as a miss for both models.
## predict_params

Train a prediction model from a CSV file and save it for later use.
Reports validation quality on a held-out subset.

### Usage

    ./predict_params.x [fraction] [auto|cat|compress[Q]|interp|rf|hknn|xgb] [-N] <csvfile> [modelfile [confidence-prefix]]

    fraction   Validation split 0..1 for quality report (default: 0.8).
    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    compress   Drop redundant entries (Q: threshold, default 0.9).
    interp     Force interpolation for all outputs.
    rf         Random Forest classification.
    hknn       Hierarchical kNN (Fisher-guided partition).
    xgb        Also train XGBoost on the same split and compare.
    -N         Max polynomial order (default: 0 = auto).
    csvfile    Delimited text file.
    modelfile  Output path for the binary model.
    confidence-prefix  Optional prefix for confidence map MHD files.

Naming a mode fixes it: the saved model keeps only what that mode reads and
cannot be evaluated by another one. Under `auto` the model stays open to all of
them. The difference shows on `rf`, which answers from its trees and therefore
stores no partition.

### Example

    ./predict_params.x ../../samples/smm/params/tune_multiply_PVC.csv
    ./predict_params.x 0.8 hknn tune_multiply_PVC.csv model.bin
## predict_sunspots

Timeseries forecasting using sliding-window nearest-neighbor prediction.

### Usage

    ./predict_sunspots.x <csvfile> [train_fraction] [compress[Q]] [hknn|rf]

    NOPHASE=1  Drop the solar-cycle phase input.
    NOBANK=1   Use a single window view instead of the bank.
    WINDOW=n   Fix the window (default: auto).  WINDOW=-1 selects it by
               trial instead, which costs about 7x the build time and picks
               9 rather than 14 here (mean error over six months 18.71 ->
               18.30).  WINDOW=-w also offers w as a candidate.

### Example

    ./predict_sunspots.x predict_sunspots.csv 0.8

The sample carries one auxiliary input alongside the window
(libxs_predict_set_series_aux): the phase of the solar cycle, measured from
the training slice so nothing is tuned by hand.  Longer horizons gain the
most (t+6 23.70 -> 21.77, t+3 20.72 -> 18.94); t+1 is unchanged.  NOPHASE=1
drops it.

The forecast is averaged over two views of the window
(libxs_predict_set_series_bank), the second reading half the lags of the
first: t+1 improves 17.52 -> 16.79 and t+6 20.30, at no extra corpus.  Pass a
larger count for more views (a third reaches 20.14 at t+6); NOBANK=1 drops
back to one.  Views are unavailable under a decomposition, where the inputs
are no longer lags.

### Data Source

Monthly mean total sunspot number from
SILSO (World Data Center, Royal Observatory of Belgium).
Semicolon-delimited: year, month, decimal_year, sunspot_number.
## predict_earthquakes

Predict earthquake magnitude from geographic location and depth.

### Usage

    ./predict_earthquakes.x <usgs_csv> [train_fraction] [compress[Q]] [hknn|rf] [xgb]

    xgb        Also train XGBoost on the same split and compare.

The kNN vote reports the median rather than the mean here.
libxs_predict_set_central picks that per output at build time and needs no
argument (0.265 -> 0.249 for kNN, 0.263 -> 0.241 for hknn); pass 1 or 2 to
force median or mean.

### Example

    ./predict_earthquakes.x predict_earthquakes.csv

### Data Source

USGS Earthquake Hazards Program (public domain).
Comma-delimited: time, latitude, longitude, depth, mag, ...
## predict_discharge

River discharge forecasting with day-of-year seasonality and
log-transform on outputs for heavy-tailed data.

### Usage

    ./predict_discharge.x <discharge_tsv> [train_fraction] [compress[Q]] [hknn|rf]

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

    ./predict_soi.x <tahiti_file> <darwin_file> [train_fraction] [compress[Q]] [hknn|rf]

### Example

    ./predict_soi.x predict_soi_tahiti.dat predict_soi_darwin.dat

### Data Source

NOAA Climate Prediction Center (public domain).
Fixed-width monthly sea level pressure (mb above 1000 mb).
## predict_stock

Multi-stock timeseries prediction with auto-differencing and
PCA/SPREAD decomposition.

### Usage

    ./predict_stock.x <csv_file> [columns] [train_fraction] [compress[Q]] [hknn|rf]

    columns    Comma-separated 0-based column indices (default: 1,2).

### Example

    ./predict_stock.x stocks.csv 1,2,3
## predict_crystal

Crystal system prediction (7-class classification) from chemical
composition features.

### Usage

    ./predict_crystal.x <crystal_csv> [train_fraction] [order] [nclusters] [compress[Q]] [fisher|hknn|setdiff|rf|none] [xgb]

    fisher     Fisher discriminant feature weighting.
    hknn       Hierarchical kNN (Gini-guided partition).
    setdiff    Setdiff feature selection.
    rf         Random Forest classification (default).
    none       Raw kNN without feature processing.
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
under the training distribution of the same output.  Use it to ask whether
the inputs inform the output at all, which an error figure does not answer.

### Usage

    ./predict_bits.x <csvfile> <innames> <outname> [fraction] [vocabulary] [hknn]

    vocabulary  Distinct values the caller considers possible.  0 uses the
                attested support plus one aggregate novel atom, so an
                unattested outcome costs infinite bits; pass a count above
                the support size to keep the figure finite.

### Example

Run a control alongside any such measurement: a null result is also what a
broken setup reports, so pair the question with one whose answer is known.

    ./predict_bits.x predict_earthquakes.csv latitude,longitude,depth mag 0.8 68
    ./predict_bits.x predict_earthquakes.csv latitude,longitude,mag depth 0.8 9000

Magnitude is not determined by location (3.265 bits against the training
distribution's 3.248) and depth is (1.478 against 7.615).

## predict_ett

ETT (Electricity Transformer Temperature) hourly forecasting.
Supports univariate and multivariate modes with PCA decomposition
and per-query local-correlation attention.  Standard benchmark
for comparison against transformer-based timeseries models.

### Usage

    ./predict_ett.x <ett_csv> [nseries=1..7] [attend|spread|pca|hknn|rf|nocompress|xgb]

    nseries    Number of input channels (1=OT only, 7=all).
    attend     Per-query local-correlation channel weighting.
    spread     Sum/diff decomposition across channels.
    pca        PCA rotation of multi-channel input space.
    xgb        Also train XGBoost on the same windows and compare.

`xgb` reads the built sliding windows back out of the model, so it implies
`nocompress` (compression prunes entries before they can be read) and is refused
alongside `attend`, a decomposition, or `BANK`, each of which zeroes the weight
of lags the distance does not read.  The effective window applies to both models,
so `WINDOW=96` compares them at the conventional input length rather than at the
auto-selected one — the two rank differently at the two lengths.

### Examples

    ./predict_ett.x predict_ett.csv              # univariate OT
    ./predict_ett.x predict_ett.csv 2 pca        # 2ch with PCA (best MSE)
    ./predict_ett.x predict_ett.csv 7 attend     # 7ch with local-attention
    ./predict_ett.x predict_ett.csv 7            # 7ch raw (baseline)

### Data Source

ETTh1 (Electricity Transformer Temperature, hourly) from
Zhou et al. 2021 (Informer).  17,420 hourly readings of an
oil-filled electrical transformer (July 2016 - June 2018).
Seven channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
Target: OT (oil temperature).  Standard split: train months
1-12, test months 17-24, horizon H=96 steps (4 days).

Download: github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv
Rename to predict_ett.csv (CRLF line endings accepted).
