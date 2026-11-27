# Grid (CP2K)

Builds CP2K's grid miniapp and unit test with LIBXS, and checks the GEMMs
that CP2K's DGEMM backend dispatches through LIBXS. The grid library
collocates Gaussian products onto real-space grids and integrates potentials
over them. The miniapp and the sample tasks come from a CP2K source tree.

## Building

```bash
cd samples/grid
make
```

Produces `grid_miniapp.x` and `grid_unittest.x`. The CP2K source tree is a
sibling of LIBXS by default (`../cp2k`). LIBXS is linked as a library if
built, and compiled header-only otherwise (`XLIB=` forces it). LIBXS
generates no GEMM kernel itself: without LIBXSMM (or MKL), it hands every
GEMM to BLAS.

| Variable    | Default                    | Description                                       |
|-------------|----------------------------|---------------------------------------------------|
| CP2K_ROOT   | `../cp2k` (sibling)        | CP2K source tree                                  |
| LIBXSMMROOT | `../libxsmm` (sibling)     | LIBXSMM, the kernel provider of LIBXS             |
| LIBXSMM     | 1 if `lib/libxsmm.a` found | 0: without LIBXSMM, 1: required                   |
| BLAS_LIBS   | `-lblas`                   | BLAS for the DGEMM backend and the comparison     |
| CFLAGS      | `-O3 -march=native`        | Compiler flags                                    |

## Testing

```bash
make test
```

Runs CP2K's unit test, then replays every sample task of CP2K
(`src/grid/sample_tasks`) with the DGEMM backend twice: with LIBXSMM's
kernels through LIBXS (`LIBXS_GEMM_BACKEND=2`), which records the task list
with its results, and with BLAS (`LIBXS_GEMM_BACKEND=3`), which compares
against them (relative tolerance 1e-12). The comparison needs LIBXSMM
(`LIBXSMM=1`). It fails if any result differs, if a replay fails, or if
LIBXSMM generated no kernel at all. `general_skewed` is skipped: the DGEMM
backend overflows on it with any GEMM. The DGEMM backend's deviation from a
sample's own reference does not count.

## Benchmark

```bash
make bench BENCH="grid_collocate_0_3.tasklist grid_integrate_0_3.tasklist"
```

Replays each task list (written by CP2K with `&GRID DUMP_TASK_LISTS <n>` in
`&GLOBAL`) with the CPU and reference backends, and with the DGEMM backend
using BLAS and, with LIBXSMM, LIBXS's kernels. `BENCHARG` gives the number of
cycles (default 3); `OMP_NUM_THREADS` the number of threads.
