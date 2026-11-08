# Bracket Units

This sample fits piecewise-constant targets with networks of truncated-power
units ⟨w·x − b⟩ᵐ, and measures how sharp the learned edges are. The 1-D targets
are a step (`step`), a square wave (`square`), and the step with Gaussian noise
of σ = 0.1 on the training samples (`noisy`). The 2-D targets are a checkerboard
(`checker`), the same checkerboard turned by 30° (`turned`, see `BRACKET_ANGLE`),
and a disc (`disc`).
Training uses Adam with full batches; the two-dimensional models use LIBXS GEMM.
By default the training data are pixels, i.e., averages over cells, as an image
sensor records them.

| Variant             | Model |
|---------------------|-------|
| `direct`            | One hidden layer of ⟨w·x − b⟩¹ (ReLU) units fitted to the samples. |
| `integrated`        | The same model fitted to the running integral, then differentiated. |
| `*_wd`              | As above, with decoupled weight decay. |
| `integrated_merge`  | Integrated, then jumps within one sample spacing merged and refined. |
| `integrated_greedy` | As merged, then jumps inserted where the residual calls for one. |
| `integrated_prune`  | As greedy, then jumps removed while the fit stays within the estimated noise. |
| `mlp`               | Two hidden ReLU layers fitted to the pixels. |
| `mlp_int`           | The same network fitted to the integral along (1,1), then ∂/∂x + ∂/∂y. |
| `tensor`            | Products ⟨x − a⟩¹⟨y − b⟩¹ fitted to the pixels. |
| `tensor_int`        | The same products fitted to the summed-area table, then ∂²/∂x∂y. |
| `sep`               | Sums of products A(x)·B(y) of 1-D bracket fits, fitted to the pixels. |
| `sep_int`           | The same fitted to the summed-area table (merged, greedy), then ∂²/∂x∂y. |
| `rot`               | One product A(u)·B(v) in a turned frame (u, v), fitted to the pixels. |
| `rot_int`           | The same fitted to running sums in that frame, then ∂²/∂u∂v; the angle is searched. |
| `box`               | Products of clamped ramps (⟨x − a⟩¹ − ⟨x − b⟩¹)(⟨y − e⟩¹ − ⟨y − g⟩¹), fitted to the pixels. |
| `box_int`           | The same fitted to the summed-area table, then ∂²/∂x∂y: a sum of rectangles. |
| `box_greedy`        | Rectangles added one at a time where the residual gains most, over a background; refined on the pixels. |
| `knn`               | libxs_predict on the class (sign) of each pixel: nearest neighbours. |
| `knn_score`         | As `knn`, in the label-aware score coordinate. |
| `fisher`            | As `knn`, with Fisher-weighted inputs. |
| `hknn`              | As `knn`, over a kd-partition. |
| `rf`                | A random forest on the class of each pixel. |
| `knn_avg`           | Nearest neighbours regressing the pixel values (pixel data only). |
| `xgb`               | XGBoost on the class of each pixel, if built with it. |

The variants up to `integrated_prune` run on each 1-D target, the others on
each 2-D target.

## Build and Run

```bash
cd samples/bracket
make
./bracket.x
```

The binary takes no arguments; the environment variables below configure it.
The two-dimensional networks dominate the runtime. The sample links BLAS by
default; `make BLAS=0` falls back to the built-in GEMM, which is slower.
XGBoost is found as in the predict sample (`XGBOOST_ROOT`), and `make XGB=0`
builds without it. With XGBoost enabled (the default), the build overrides
`PEDANTIC` to 0.

Each row of the printed table reports:

| Column   | Meaning                                                                                  |
|----------|------------------------------------------------------------------------------------------|
| `err`    | Error against the noise-free target on the fine evaluation grid.                         |
| `pos/h`  | Mean distance between true and learned edges, in training-grid spacings.                 |
| `slope`  | Growth of the order-1 fingerprint norm under refinement: 0.5 is a jump, 0 a smooth edge. |
| `eps/h`  | Effective edge width implied by that norm, in training-grid spacings.                    |
| `fpdist` | Fingerprint distance (`libxs_fprint_diff`, orders 0 and 1) to the noise-free target.     |
| `time`   | Time to fit the variant, in seconds.                                                     |

For the 2-D targets, the fingerprint norms are the root mean square over image rows.

Each classifier from `knn` to `xgb` also prints the share of the evaluation grid
it answers at confidence 0.9 and the precision there, the precision of its most
confident 50% and 80%, and, for libxs_predict, its number of coordinates (`ncoord`).

## Environment Variables

| Variable          | Default | Meaning                                                    |
|-------------------|---------|------------------------------------------------------------|
| `BRACKET_OUTDIR`  | none    | Existing directory for MHD and PNG output.                 |
| `BRACKET_NSTEPS`  | 20000   | Adam steps in 1-D; the 2-D models run a quarter of them.   |
| `BRACKET_WD`      | 0.01    | Weight decay of the `*_wd` variants.                       |
| `BRACKET_SEED`    | 1       | Seed for initialization and noise.                         |
| `BRACKET_UNITS`   | 32      | Hidden units of the 1-D models (at most 64).               |
| `BRACKET_SAMPLES` | 64      | Training samples of the 1-D targets (4 to 256).            |
| `BRACKET_WIDTH`   | 64      | Hidden width of the 2-D networks (at most 1024).           |
| `BRACKET_2D`      | 1       | Set to 0 to skip the 2-D targets.                          |
| `BRACKET_PREDICT` | 1       | Set to 0 to skip the classifiers (`knn` to `xgb`).         |
| `BRACKET_TARGET`  | none    | Name of the only target to run, e.g., `turned`.            |
| `BRACKET_PIXEL`   | 1       | Set to 0 for point samples instead of cell averages.       |
| `BRACKET_RANK`    | 4       | Largest number of products in `sep` and `sep_int`.         |
| `BRACKET_ANGLE`   | 30      | Angle of the turned checkerboard in degrees.               |

`bracket_1d_<target>.mhd` stacks the target and each variant's output as rows.
`bracket_2d_<target>.mhd` holds a target (`checker`, `turned`, or `disc`), and
`bracket_2d_<target>_<variant>.mhd` each model's image, clamped to the target's
range [−1, 1] after the metrics are taken. The variants from `knn` on also write
their confidence as `bracket_2d_<target>_<variant>_conf.mhd`. The directory must
exist. Each MHD file has a PNG beside it: the two-dimensional images on [−1, 1]
and the confidence on [0.5, 1], so that images compare side by side, and a 1-D
stack over its own range.
