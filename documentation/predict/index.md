# Self-Diagnosing Parameter Prediction

## Confidence-Gated Models for Sparse Tuning Data

LIBXS Predict

Note: Open with the practical deployment problem: a predictor is only useful
if it knows when a safe rule is better than its guess.

---

## The Problem

CP2K and DBCSR tune GPU kernel parameters for known matrix shapes.

New shapes appear between tuned points.

| Choice | Risk |
| --- | --- |
| Use only fixed rules | Miss tuning opportunities |
| Predict every parameter | Silent slowdowns |
| Predict with confidence | Improve only when evidence is strong |

---

## Method in One Slide

Combine distance-weighted kNN voting with polynomial fingerprint diagnostics.

The model returns:

- A predicted value.
- A per-output confidence score.
- A signal that the safe rule should stay in control.

Note: This is the talk's main phrase: not just prediction, but deployment
decision support.

---

## GPU Kernel Dispatch

GPU small-matrix kernel dispatch.

Inputs: `M`, `N`, `K`.

Outputs: batch size, block sizes, workgroup shape, loop unroll, access mode,
and algorithm selectors.

Training data: tabulated tuned kernels across GPU architectures.

---

## Why Ordinary Accuracy Is Not Enough

Some parameters encode hidden hardware constraints.

Nearby matrix shapes can agree on a value that is wrong for the query.

Example from deployment:

| Shape | Predicted BK | Rule BK | Result |
| --- | ---: | ---: | ---: |
| 21 x 22 x 23 | 4 | 21 | 487 vs. 991 GFLOP/s |

The average model error is not the operational risk.

---

## Deployment Policy

Structural parameters remain rule controlled:

`BS`, `BM`, `BN`, `BK`, `WS`

Preference parameters are confidence gated:

`WG`, `LU`, `NZ`, `AL`, `TB`, `TC`, `AP`, `AA`, `AB`, `AC`

Only near-unanimous neighbor agreement overrides the default.

---

## Confidence Signals

| Signal | When | Meaning |
| --- | --- | --- |
| Fingerprint decay | Build time | Output is smooth or categorical |
| kNN vote fraction | Query time | Neighbors agree for this query |

Fingerprint behavior chooses the per-output mode.

| Output behavior | Chosen path |
| --- | --- |
| Smooth trend | Polynomial interpolation |
| Discrete class | Distance-weighted kNN vote |
| Constant | Store the constant |
| Erratic | Low confidence or rule deferral |

---

## Override Rule

```text
if output is structural:
    use safe rule
else if confidence >= threshold:
    use prediction
else:
    use safe rule
```

This makes abstention part of LIBXS behavior.

---

## What Confidence Gating Buys

It changes the failure mode.

| Without gating | With gating |
| --- | --- |
| Wrong values silently deploy | Low evidence defers |
| Average error hides risk | Per-output confidence is visible |
| Outliers look like model bugs | Outliers identify missing data |

---

## Beyond Kernel Dispatch

The same LIBXS machinery now handles:

- Timeseries forecasting
- Spatial prediction
- Cross-series decomposition
- Non-stationary series with auto-differencing
- Materials classification

The common interface is still prediction plus confidence.

---

## How to Read Literature Comparisons

The external numbers are orientation, not a strict leaderboard.

Published studies often differ in:

- Feature sets and preprocessing.
- Train/test split and forecast horizon.
- Whether temporal or spatial context is included.
- Error metric: MAE, RMSE, NSE, accuracy, or speedup.

The useful question is whether our errors land in the quality range of
specialized trained methods.

---

## Sunspots vs. Literature

Monthly mean sunspot number, 1749-2026.

1-step monthly forecast quality.

| Method | Error measure | Reported quality |
| --- | --- | ---: |
| Ours, kNN W=12 | MAE | 17.6 |
| XGBoost + DL ensemble | MAE | 19.8 |
| Informer transformer | MAE | 22.4 |
| LSTM | MAE | about 25 |
| SARIMA | MAE | 45.5 |
| NASA operational | MAE | 38.5 |

Dense recurring cycles give dense historical support.

---

## River Discharge vs. Literature

Colorado River at Lees Ferry, daily values from 2000-2025.

Window: 14 days plus differences and day of year.

| Method | Error measure | Reported quality |
| --- | --- | ---: |
| Ours, W=14 + day of year | NSE, t+1 | about 0.90 |
| kNN + data assimilation | NSE, 1d / 7d | 0.99 / 0.79 |
| Spatio-temporal GNN | NSE, Lees Ferry | 0.78 |
| LSTM, CAMELS basins | NSE, 1d | 0.86-0.88 |

Our 1-day error is about 6%; 7-day error is about 11%.

Confidence remains 1.000 for the seasonal regime.

---

## Earthquakes vs. Literature

USGS M4.5+ catalog, 2022-2025.

Spatial inputs: latitude, longitude, depth. Output: magnitude.

| Method | Error measure | Reported quality |
| --- | --- | ---: |
| Ours, adaptive blending | MAE | about 0.261 |
| Review consensus, spatial | MAE | about 0.26 |
| Tuned Random Forest | MAE | 0.283 |
| Supervised ML ensemble | MAE | 0.184 |
| Hybrid GRU | MAE | 0.397 |

Lower MAE methods generally use richer temporal or sequential features.

Average confidence is 0.694: location alone does not determine magnitude.

---

## SOI vs. Literature

Southern Oscillation prediction, normalized RMSE.

| Method | 1-month | 6-month |
| --- | ---: | ---: |
| Ours, SPREAD W=12 | about 0.11 | about 0.12 |
| ANN | 0.23 | n/a |
| EEMD-TCN | 0.30 | 0.50 |
| LSTM | 0.35 | 0.55 |
| Persistence | about 0.3 | about 0.9 |

This uses a conversion from MAE to normalized RMSE.

---

## Crystals vs. Literature

Crystal system classification from composition.

AFLOW ICSD composition features to 7 crystal classes.

| Method | Accuracy | Notes |
| --- | ---: | --- |
| Ours, RF, 37 features | 79.6% | all answers |
| Ours, RF, conf. >= 0.9 | 95.0% | 53.7% coverage |
| Ours, kNN + Fisher | 70.7% | 37 features |
| Ours, kNN, compact subset | 75.1% | 17 features |
| RF + Magpie | about 75% | 30-40 features |
| MLP + Magpie | about 72% | 30-40 features |
| CRYSPNet | about 80% | 271 features |

The confidence-gated rows trade coverage for reliability.

---

## What the Literature Context Says

Across domains, the method is usually in the range of specialized trained
models on the reported metric.

The distinctive result is not only lower error.

It is the extra deployment signal:

- Dense recurring domains: confidence near 1.0.
- Ambiguous domains: lower confidence.
- Classification: high-confidence subset reaches much higher accuracy.

---

## Why This Matters for Atomistic Codes

Simulation setup often needs plausible structure or kernel choices before
expensive computation begins.

A confidence-gated predictor can say:

- This guess is supported enough to use.
- This case is ambiguous; keep the conservative path.
- This regime deserves new measurements or a missing feature.

---

## Comparison to Heavier Models

Gaussian processes, neural networks, tree ensembles, and conformal methods can
all expose uncertainty.

The practical difference here is packaging:

- No iterative gradient training.
- O(1) append per entry before rebuild.
- Deterministic construction from tables.
- Per-output confidence in LIBXS.
- Complete Fortran interface for Fortran-heavy target codes.

---

## Design Pattern

Separate what the system predicts from what it is allowed to control.

```text
prediction: value + confidence
policy: threshold + rule ownership
action: override or defer
```

This makes learned tuning compatible with hard-won domain rules.

---

## Takeaways

- Sparse tuning spaces reward abstention.
- Confidence must be per output, not just per model.
- Fingerprints diagnose smoothness and mode choice.
- kNN votes expose local deployment evidence.
- Rule deferral turns uncertainty into safe behavior.

---

## Closing Thought

The useful model is not the one that always has an answer.

It is the one that knows when its answer should not be in charge.