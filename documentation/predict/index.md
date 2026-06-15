# Self-Diagnosing Parameter Prediction

## Confidence-Gated Models for Sparse Tuning Data

LIBXS Predict

Note: Open with the deployment problem: a predictor is useful only if it
knows when a safe rule should stay in charge.

---

## The Problem

CP2K and DBCSR tune GPU kernels for known matrix shapes.

Deployment sees new shapes between tuned points.

| Choice | Risk |
| --- | --- |
| Fixed rules only | Miss local tuning opportunities |
| Predict everything | Silent slowdowns |
| Confidence-gated | Override only with evidence |

---

## Method in One Slide

Distance-weighted *k*NN voting plus polynomial fingerprint diagnostics.

The model returns:

- Predicted value.
- Per-output confidence.
- Override/defer signal.

Note: The main phrase is not just prediction, but deployment decision
support.

---

## GPU Kernel Dispatch

Small-matrix GPU kernel dispatch.

Inputs: `M`, `N`, `K`.

Outputs: batch size, block sizes, workgroup shape, loop unroll, layout,
and access selectors.

Training data: tuned kernels across GPU architectures (Intel PVC
shown here; method is device-agnostic).

---

## Why Ordinary Accuracy Is Not Enough

Some parameters encode hidden hardware constraints.

Nearby shapes can agree on a value that is wrong for the query.

| Shape | Predicted BK | Rule BK | Result |
| --- | ---: | ---: | ---: |
| 21 × 22 × 23 | 4 | 21 | 487 vs. 991 GF/s |

Average error is not the operational risk.

Note: This example motivates policy separation. The current full-rerun
evidence is summarized later.

---

## Deployment Policy

Separate ownership from prediction.

| Rule controlled | Confidence gated |
| --- | --- |
| `BS`, `BM`, `BN`, `BK`, `WS` | `WG`, `LU`, `AL`, `AA`, `AB` |
| structural safety | preference/access choices |
| source rules stay authoritative | override near-unanimously |

SMM kernel parameters: BS batch-size, BM/BN/BK block extents,
WS work-sharing, WG workgroup shape, LU unroll, AL/AA/AB access modes.

---

## Confidence Signals

| Signal | Time | Used for |
| --- | --- | --- |
| Fingerprint decay | Build | constant, smooth, categorical, erratic |
| *k*NN vote fraction | Query | per-output deployment confidence |

Fingerprint behavior chooses the output mode.

Neighbor agreement decides whether a prediction may act.

---

## Override Rule

```text
if output is rule-owned:
    use safe rule
else if confidence ≥ threshold:
    use prediction
else:
    use safe rule
```

Abstention is part of LIBXS behavior.  Learned tuning becomes
compatible with hard-won domain rules.

---

## PVC Tuning Structure

![PVC tuning impact by arithmetic-intensity bin](assets/pvc_ai_performance_slide.png)

1339 PVC kernels, three reruns per mode.  Tuning gives +1.3% over
handwritten rules; LOO prediction reaches +1.1%.  The gain
concentrates in compute-heavy shapes (AI 2–4: +6.8%, 41 distinct BK
values).  Other bins are near neutral — the rules are already strong.

---

## PVC Confidence Projection

![Saved PVC predictor confidence over the M×N×K cube](assets/pvc_confidence_projection.png)

Over the M × N × K cube (739k queries), 39% fall below the 0.9
threshold (defer to rules).  The distribution is bimodal: confident
or clearly insufficient — the gate fires decisively.

---

## What Confidence Gating Buys

It changes the failure mode.

| Without gating | With gating |
| --- | --- |
| Wrong values silently deploy | Low evidence defers |
| Average error hides risk | Per-output confidence is visible |
| Outliers look like bugs | Outliers identify missing data |

---

## Beyond Kernel Dispatch

The same LIBXS machinery handles:

- Timeseries forecasting.
- Spatial prediction.
- Cross-series decomposition.
- Non-stationary series with auto-differencing.
- Materials classification.

The interface is still prediction plus confidence.

---

## Crystal System Prediction

<!-- .slide: data-background-image="assets/crystal_system_wheel_slide.png" data-background-size="contain" data-background-position="right center" style="text-align: left" -->

- 60 386 compositions
- 37 features
- 7 crystal systems

The sample is a mixed classification problem  
where confidence decides whether to act.

Note: This is the key slide for computational chemistry audience.
Structure initialization in CP2K/FHI-aims requires symmetry information;
a confidence-gated predictor can provide it or abstain.

---

## Secondary Evidence

| Domain | Ours | Literature | Confidence |
| --- | ---: | ---: | --- |
| Sunspots | MAE 17.6 | MAE 19.8–45.5 | 1.0 (dense cycles) |
| Discharge | 0.23 err/σ | 0.10–0.47 | 1.0 (seasonal) |
| SOI | nRMSE 0.11 | 0.23–0.55 | 1.0 (spread modes) |
| Earthquakes | MAE 0.265 | 0.184–0.283 | 0.694 (ambiguous) |
| Crystals | 79.6% → 95.0% (conf ≥ 0.9) | ≈75–80% | 54% gated coverage |

Confidence separates dense-coverage domains from genuinely ambiguous
ones.  Literature comparisons are orienting — different features, splits,
metrics.

---

## Why This Matters for Atomistic Codes

Simulation setup often needs plausible structure or kernel choices before
expensive computation begins.

A confidence-gated predictor can say:

- This guess is supported enough to use.
- This case is ambiguous; keep the conservative path.
- This regime deserves new measurements or another feature.

---

## Fortran-First Feedback Loop

No Python, no framework dependency — links into your Fortran binary.

| Running application moment | LIBXS call | Effect |
| --- | --- | --- |
| Load existing knowledge | `libxs_predict_load_csv` | seed model from file |
| New measured case | `libxs_predict_push` | append evidence 𝒪(1) |
| Checkpoint or idle point | `libxs_predict_build` | rebuild model cheaply |
| Next query | `libxs_predict_eval` | value + confidence |

Start from a CSV of prior runs or start empty — learn from completed
work, and let later decisions use the stronger local evidence.

---

## Takeaways

- Sparse tuning spaces reward abstention.
- Confidence must be per output.
- Running jobs can add evidence and rebuild at checkpoints.
- Fingerprints diagnose mode choice.
- *k*NN votes expose local evidence.
- Rule deferral turns uncertainty into safe behavior.

---

## Closing Thought

The useful model is not the one that always has an answer.

It is the one that knows when its answer should not be in charge.

---

## If It Is Hardcoded, It Is a Candidate

Any magic constant or fixed heuristic that was fitted once and
never revisited is a prediction opportunity.

| Pattern | Example |
| --- | --- |
| Fitted polynomial | HFX cost model (12 coefficients) |
| User-chosen integer | k-point grid, parallel group size |
| Compile-time constant | `max_elements_per_block = 32` |
| Static threshold | `eps_filter` in LS-SCF |

If the code ships a number that someone tuned by hand — confidence-gated
prediction can learn a better one at runtime or abstain safely.

---

## Ideas: Work Packages

Five places where confidence-gated prediction plugs into CP2K.
Each is self-contained — pick the one that excites you.

---

## WP 1 — HFX Cost Model

Shell-quartet cost drives load balancing.  Today: a fixed polynomial
fitted once.  Tomorrow: learn from measured timings, improve every SCF.

```text
SCF step 1‥3:  time quartets → push(9 inputs, wall-time)
checkpoint:     build(auto, 0, 0.0)
step 4+:        eval(inputs) → cost estimate + confidence
                low confidence? keep polynomial
```

| Where | Scope |
| --- | --- |
| `hfx_load_balance_methods.F` | rank-local, no MPI change |
| Training data from the run itself | no external CSV needed |

---

## WP 2 — K-Point Mesh and Grouping

Users guess `nkp_grid` and `parallel_group_size`.  A database of
converged runs can recommend both — or abstain and keep the default.

| Input | Output | Mode |
| --- | --- | --- |
| cell shape, basis size, target accuracy | `nkp_grid(3)` | classify |
| nkp, natom, nprocs | `parallel_group_size` | classify |

```text
load_csv("kpoints.csv", …)      — community-contributed
build(0, 0, 0.0)
eval(cell_params) → grid + confidence
confidence < 0.9? keep user input
```

One-shot at setup.  No MPI-redistribution.

---

## WP 3 — SCF Convergence

MD / geometry-opt repeats SCF hundreds of times on similar densities.
Predict iteration count to adapt `max_scf`, DIIS depth, preconditioner.

```text
after converged SCF:
  push([δ₀, displacement, gap, α], [niter])
  build(…)

before next SCF:
  eval([…]) → predicted niter + confidence
  confidence high? set max_scf = predicted + margin
  confidence low?  new basin — keep conservative default
```

Incremental on rank 0.  Acts on local control integers, no
communication.

---

## WP 4 — Filter Threshold (LS-SCF)

`eps_filter` balances sparsity against accuracy.  Static today;
optimal value drifts as the density matrix evolves.

```text
        ε tight           ε loose
accuracy  |████████████████······|  sparsity
          ↑ cubic cost     ↑ lost digits

push(iteration, sparsity_frac, timing)   — temporal mode
eval() → next eps_filter + confidence
```

Adjusts a single scalar between iterations.  No matrix layout
change.  Too aggressive triggers confidence drop — automatic
fallback to the previous value.

---

## WP 5 — DBCSR Block Size

`max_elements_per_block = 32` — hardcoded, never tuned per system.

| Input | Output | Mode |
| --- | --- | --- |
| matrix dims, avg block, sparsity, nprocs | block grouping | classify |

Closest analogue to PVC kernel dispatch.  Load a platform CSV at
init, predict once per unique matrix structure.

```text
load_csv("dbcsr_blocks_pvc.csv", …)
build(0, 0, 0.0)
eval([N, M, nnz_ratio, nprocs]) → block_size + confidence
confidence < 0.9? keep 32
```

One-shot at matrix creation.  Changing block size means
MPI-redistribution — cheap at setup, prohibitive mid-run.

---

## WP 5 — When to Predict

CP2K creates few distinct matrix structures per run.  Matrices that
share row/column block-size arrays are structurally identical (S, H, P,
K for the same basis).

```text
dbcsr_create(matrix, row_blk_sizes, col_blk_sizes, dist, …)
  │
  key = hash(row_blk_sizes, col_blk_sizes)
  │
  seen before?  → reuse cached block_size
  new?          → eval(key_features) → block_size + confidence
                  confidence < 0.9? keep 32
                  cache result
```

Hook: `cp_dbcsr_operations.F` where `max_elements_per_block` is
consumed.  Typically 1–2 unique structures per run — the cache
is trivial.
