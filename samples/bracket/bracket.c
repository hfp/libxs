/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs.h>
#include <libxs/libxs_gemm.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_rng.h>
#include <libxs/libxs_timer.h>
#include "bracket.h"

#define BRACKET_PATH_MAX 1024
#define BRACKET_PI 3.14159265358979323846
#define BRACKET_PHASE 0.37
#define BRACKET_NOISE 0.1
#define BRACKET_CELLS 4
#define BRACKET_N1 64
#define BRACKET_N1MAX 256
#define BRACKET_H1 32
#define BRACKET_H1MAX 64
#define BRACKET_NPARAM1 (4 * BRACKET_H1MAX + 2)
#define BRACKET_IMAGE1 1024
#define BRACKET_LO1 512
#define BRACKET_HI1 4096
#define BRACKET_LR1 1e-2
#define BRACKET_N2 48
#define BRACKET_U2 64
#define BRACKET_H2 64
#define BRACKET_IMAGE2 256
#define BRACKET_LO2 64
#define BRACKET_HI2 512
#define BRACKET_LR2 1e-2
#define BRACKET_LRMLP 1e-2
#define BRACKET_CHUNK 8192
#define BRACKET_ANGLE 30.0
#define BRACKET_SUPER 16
#define BRACKET_RANKMAX 4
#define BRACKET_ALS 2
#define BRACKET_CELLMAX (2 * BRACKET_N2)
#define BRACKET_STEPDEG 3.0
#define BRACKET_DISC_R 0.3
#define BRACKET_DISC_X 0.53
#define BRACKET_DISC_Y 0.48


typedef struct train_t {
  int nsteps;
  double lr, wd;
} train_t;

typedef struct setup_t {
  const char* target;
  double angle;
  int units, samples, width, run2d, predict, pixel, rank;
} setup_t;

typedef struct model1_t {
  double param[BRACKET_NPARAM1];
  int nh, integrated;
} model1_t;

typedef struct tensor_t {
  double* param;
  int nu, integrated;
} tensor_t;

typedef struct mlp_t {
  double* param;
  int nh, integrated;
} mlp_t;

typedef struct sep_t {
  model1_t ax[BRACKET_RANKMAX], ay[BRACKET_RANKMAX];
  int rank, integrated;
} sep_t;

typedef struct rot_t {
  model1_t a, b;
  double angle, lo, len;
} rot_t;

typedef struct rotdata_t {
  const double* data;
  double *cells, *mask;
  int n;
} rotdata_t;


static char output_dir[BRACKET_PATH_MAX] = { 0 };

static double gaussian(void);
static double cosine_lr(double lr, int step, int nsteps);
static void adam_update(double param[], const double grad[], double m[], double v[],
  int nparam, int ndecay, int step, double lr, double wd);
static int gemm_f64(char transa, char transb, int m, int n, int k,
  const double a[], int lda, const double b[], int ldb, double c[], int ldc);
static double target1_value(const target1_t* target, double x);
static double target1_integral(const target1_t* target, double x);
static void integrate1(const double f[], size_t n, double out[]);
static void integrate_diag(const double f[], size_t n, double out[]);
static void model1_init(model1_t* model, int nh, int integrated);
static double model1_value(const model1_t* model, double x);
static void model1_train(model1_t* model, const double x[], const double y[], int n,
  const train_t* train);
static void model1_merge(model1_t* model, double hcell);
static void model1_insert(model1_t* model, const double x[], const double y[], int n,
  int nmax, const train_t* refine);
static int solve_spd(double a[], double b[], int n);
static double model1_refit(model1_t* model, const double x[], const double avg[], int n,
  int skip);
static void model1_prune(model1_t* model, const double x[], const double y[], int n,
  double sigma);
static double noise_sigma(const double avg[], int n);
static int model1_eval(const void* model, size_t n, double out[]);
static double checker_edge(int i);
static double checker_value(double x, double y, double angle);
static double disc_value(double x, double y, double angle);
static void checker_row(value_fn_t value, double y, double angle, target1_t* row);
static int tensor_init(tensor_t* model, int nu, int integrated);
static void tensor_factor(const double param[], int nu, int axis, size_t n, double lo,
  double h, double relu[], double step[]);
static int tensor_train(tensor_t* model, const double y[], int n, double lo, double h,
  const train_t* train);
static int tensor_eval(const void* model, size_t n, double out[]);
static int box_init(tensor_t* model, int nu, int integrated);
static void box_factor(const double param[], int nu, int axis, size_t n, double lo, double h,
  double ramp[], double below[], double above[]);
static int box_train(tensor_t* model, const double y[], int n, double lo, double h,
  const train_t* train);
static int box_eval(const void* model, size_t n, double out[]);
static int box_cells(tensor_t* model, const double data[], int n, double h,
  const train_t* train);
static int box_greedy(tensor_t* model, const double data[], const double area[], int n,
  double h, int nmax, const train_t* train, const train_t* refine);
static int mlp_init(mlp_t* model, int nh, int integrated);
static int mlp_forward(const double param[], int nh, const double xs[], const double ys[],
  int npts, double a1[], double a2[], double s1[], double out[]);
static int mlp_train(mlp_t* model, const double y[], int n, double lo, double dx,
  const train_t* train);
static int mlp_eval(const void* model, size_t n, double out[]);
static void model1_sample(const model1_t* model, int integrated, const double x[], int n,
  double out[]);
static void factor_fit(model1_t* model, const double x[], const double y[], int n,
  int units, int integrated, double span, const train_t* train, const train_t* refine);
static int sep_train(sep_t* model, const double y[], int n, double lo, double h, int units,
  const train_t* train, const train_t* refine);
static int sep_eval(const void* model, size_t n, double out[]);
static int clip_half(const double xin[], const double yin[], int n, double xout[],
  double yout[], int axis, double bound, int greater);
static int rot_cells(const double data[], int n, double angle, double cells[], double mask[],
  double* lo);
static double rot_rank1(const double cells[], const double mask[], int nc, double a[],
  double b[], int niter);
static double rot_objective(double degrees, const void* data);
static int rot_train(rot_t* model, const double data[], int n, int units, int integrated,
  const train_t* train, const train_t* refine);
static int rot_eval(const void* model, size_t n, double out[]);
static double edge_error(const double y[], size_t n, const target1_t* target);
static int fprint_field(libxs_fprint_t* info, const double data[], int ndims, size_t n,
  int order);
static int env_int(const char name[], int value, int lo, int hi);
static int selected(const setup_t* setup, const char target[]);
static int run_1d(const train_t* train, const setup_t* setup, unsigned int seed);
static int run_2d(const train_t* train, const setup_t* setup, unsigned int seed);
static int run_checker(const train_t* train, const setup_t* setup, unsigned int seed,
  const char target[], value_fn_t value, double angle, double work[]);


int main(void)
{
  const char *const env_dir = getenv("BRACKET_OUTDIR");
  const char *const env_nsteps = getenv("BRACKET_NSTEPS");
  const char *const env_wd = getenv("BRACKET_WD");
  const char *const env_seed = getenv("BRACKET_SEED");
  const char *const env_angle = getenv("BRACKET_ANGLE");
  const char *const env_target = getenv("BRACKET_TARGET");
  int result = EXIT_SUCCESS;
  train_t train;
  setup_t setup;
  unsigned int seed;
  setup.units = env_int("BRACKET_UNITS", BRACKET_H1, 1, BRACKET_H1MAX);
  setup.samples = env_int("BRACKET_SAMPLES", BRACKET_N1, 4, BRACKET_N1MAX);
  setup.width = env_int("BRACKET_WIDTH", BRACKET_H2, 1, 1024);
  setup.run2d = env_int("BRACKET_2D", 1, 0, 1);
  setup.predict = env_int("BRACKET_PREDICT", 1, 0, 1);
  setup.pixel = env_int("BRACKET_PIXEL", 1, 0, 1);
  setup.rank = env_int("BRACKET_RANK", BRACKET_RANKMAX, 1, BRACKET_RANKMAX);
  setup.angle = (NULL != env_angle && 0 != *env_angle ? atof(env_angle) : BRACKET_ANGLE);
  setup.target = env_target;
  train.nsteps = (NULL != env_nsteps ? atoi(env_nsteps) : 0);
  if (0 >= train.nsteps) train.nsteps = 20000;
  train.wd = (NULL != env_wd ? atof(env_wd) : 1e-2);
  train.lr = BRACKET_LR1;
  seed = (unsigned int)(NULL != env_seed ? atoi(env_seed) : 1);
  if (NULL != env_dir && strlen(env_dir) < sizeof(output_dir)) {
    strcpy(output_dir, env_dir);
  }
  libxs_init();
  printf("BRACKET: sharp boundaries from bracket units (steps=%i wd=%g seed=%u)\n",
    train.nsteps, train.wd, seed);
  printf("  units=%i samples=%i width=%i pixel=%i angle=%g\n",
    setup.units, setup.samples, setup.width, setup.pixel, setup.angle);
  printf("  output: %s\n", 0 != output_dir[0] ? output_dir : "stdout only");
  printf("%-8s %-17s %10s %8s %7s %9s %10s %8s\n",
    "target", "variant", "err", "pos/h", "slope", "eps/h", "fpdist", "time");
  if (EXIT_SUCCESS == result) result = run_1d(&train, &setup, seed);
  if (EXIT_SUCCESS == result && 0 != setup.run2d) result = run_2d(&train, &setup, seed);
  if (EXIT_SUCCESS == result) printf("BRACKET: done\n");
  libxs_finalize();
  return result;
}


static int env_int(const char name[], int value, int lo, int hi)
{
  const char *const env = getenv(name);
  const int result = (NULL != env && 0 != *env ? atoi(env) : value);
  return LIBXS_CLMP(result, lo, hi);
}


static int selected(const setup_t* setup, const char target[])
{
  return (NULL == setup->target || 0 == *setup->target || 0 == strcmp(setup->target, target));
}


int write_image(const char name[], const double data[], size_t ncols, size_t nrows,
  const double range[2])
{
  int result = EXIT_SUCCESS;
  if (0 != output_dir[0]) {
    char path[BRACKET_PATH_MAX];
    const size_t dir_len = strlen(output_dir);
    if (dir_len + strlen(name) + 6 < sizeof(path)) {
      libxs_mhd_write_info_t write_info;
      libxs_mhd_info_t info;
      size_t size[2], len;
      strcpy(path, output_dir);
      if ('/' != path[dir_len - 1]) strcat(path, "/");
      strcat(path, name);
      len = strlen(path);
      strcat(path, ".mhd");
      size[0] = ncols;
      size[1] = nrows;
      info.ndims = 2;
      info.ncomponents = 1;
      info.type = LIBXS_DATATYPE_F64;
      info.header_size = 0;
      result = libxs_mhd_write(path, NULL, size, NULL, &info, data, NULL);
      memset(&write_info, 0, sizeof(write_info));
      if (NULL != range) {
        write_info.range[0] = range[0];
        write_info.range[1] = range[1];
      }
      strcpy(path + len, ".png");
      if (EXIT_SUCCESS == result) {
        result = libxs_png_write(path, size, NULL, 1, LIBXS_DATATYPE_F64, data, &write_info);
      }
    }
    else result = EXIT_FAILURE;
    if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: cannot write %s to %s\n", name, output_dir);
  }
  return result;
}


static double gaussian(void)
{
  const double u1 = 1.0 - libxs_rng_f64();
  const double u2 = libxs_rng_f64();
  return sqrt(-2.0 * log(u1)) * cos(2.0 * BRACKET_PI * u2);
}


static double cosine_lr(double lr, int step, int nsteps)
{
  return 0.5 * lr * (1.0 + cos(BRACKET_PI * (double)step / (double)nsteps));
}


/* decoupled weight decay (AdamW) applies to the leading ndecay parameters only */
static void adam_update(double param[], const double grad[], double m[], double v[],
  int nparam, int ndecay, int step, double lr, double wd)
{
  const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  const double scale1 = 1.0 / (1.0 - pow(beta1, (double)step));
  const double scale2 = 1.0 / (1.0 - pow(beta2, (double)step));
  int i;
  for (i = 0; i < nparam; ++i) {
    m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
    if (i < ndecay) param[i] -= lr * wd * param[i];
    param[i] -= lr * (m[i] * scale1) / (sqrt(v[i] * scale2) + epsilon);
  }
}


static int gemm_f64(char transa, char transb, int m, int n, int k,
  const double a[], int lda, const double b[], int ldb, double c[], int ldc)
{
  const double alpha = 1.0, beta = 0.0;
  const libxs_gemm_config_t *const config = libxs_gemm_dispatch(LIBXS_DATATYPE_F64,
    transa, transb, m, n, k, lda, ldb, ldc, &alpha, &beta, NULL);
  int result = EXIT_SUCCESS;
  if (NULL != config) libxs_gemm_call(config, a, b, c);
  else result = EXIT_FAILURE;
  return result;
}


static double target1_value(const target1_t* target, double x)
{
  int i = 0;
  while (i < target->nedges && target->edge[i] <= x) ++i;
  return target->level[i];
}


/* exact integral over [0, x] of the piecewise-constant profile */
static double target1_integral(const target1_t* target, double x)
{
  double result = 0, lo = 0;
  int i;
  for (i = 0; i <= target->nedges; ++i) {
    const double hi = (i < target->nedges ? LIBXS_MIN(target->edge[i], x) : x);
    if (lo < hi) result += target->level[i] * (hi - lo);
    lo = LIBXS_MAX(lo, hi);
  }
  return result;
}


/* trapezoid: a step between two samples integrates to a ramp through their midpoint */
static void integrate1(const double f[], size_t n, double out[])
{
  const double h = 1.0 / (double)(n - 1);
  size_t i;
  out[0] = 0;
  for (i = 1; i < n; ++i) out[i] = out[i - 1] + 0.5 * h * (f[i - 1] + f[i]);
}


/* integral along d = (1,1) from the lower or left boundary, i.e., (dx + dy) F = f */
static void integrate_diag(const double f[], size_t n, double out[])
{
  const double h = 1.0 / (double)(n - 1);
  size_t i, j;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      const size_t k = i + n * j, km = (i - 1) + n * (j - 1);
      out[k] = (0 == i || 0 == j ? 0 : out[km] + 0.5 * h * (f[km] + f[k]));
    }
  }
}


static void model1_init(model1_t* model, int nh, int integrated)
{
  double *const w = model->param, *const c = w + nh;
  double *const b = c + nh, *const d = b + nh;
  int i;
  for (i = 0; i < nh; ++i) {
    w[i] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
    b[i] = w[i] * libxs_rng_f64();
    c[i] = 0.1 * (2.0 * libxs_rng_f64() - 1.0);
  }
  d[0] = d[1] = 0;
  model->nh = nh;
  model->integrated = integrated;
}


/* direct: d0 + d1 x + sum c<wx-b>^1; integrated: its derivative d1 + sum cw<wx-b>^0 */
static double model1_value(const model1_t* model, double x)
{
  const int nh = model->nh;
  const double *const w = model->param, *const c = w + nh;
  const double *const b = c + nh, *const d = b + nh;
  double result = (0 == model->integrated ? d[0] + d[1] * x : d[1]);
  int i;
  for (i = 0; i < nh; ++i) {
    const double u = w[i] * x - b[i];
    if (0 < u) result += (0 == model->integrated ? c[i] * u : c[i] * w[i]);
  }
  return result;
}


/* both variants fit the same function form; only the target (f or its integral) differs */
static void model1_train(model1_t* model, const double x[], const double y[], int n,
  const train_t* train)
{
  const int nh = model->nh;
  double grad[BRACKET_NPARAM1], m[BRACKET_NPARAM1], v[BRACKET_NPARAM1];
  double *const w = model->param, *const c = w + nh;
  double *const b = c + nh, *const d = b + nh;
  double *const gw = grad, *const gc = gw + nh;
  double *const gb = gc + nh, *const gd = gb + nh;
  int step, i, j;
  memset(m, 0, sizeof(m));
  memset(v, 0, sizeof(v));
  for (step = 1; step <= train->nsteps; ++step) {
    memset(grad, 0, sizeof(grad));
    for (j = 0; j < n; ++j) {
      double r = d[0] + d[1] * x[j] - y[j];
      for (i = 0; i < nh; ++i) {
        const double u = w[i] * x[j] - b[i];
        if (0 < u) r += c[i] * u;
      }
      r /= (double)n;
      for (i = 0; i < nh; ++i) {
        const double u = w[i] * x[j] - b[i];
        if (0 < u) {
          gc[i] += r * u;
          gw[i] += r * c[i] * x[j];
          gb[i] -= r * c[i];
        }
      }
      gd[0] += r;
      gd[1] += r * x[j];
    }
    adam_update(model->param, grad, m, v, 4 * nh + 2, 2 * nh, step,
      cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
  }
}


/**
 * Rewrites each unit with w > 0 (exact: <z> = z + <-z>), then replaces jumps within one
 * sample spacing by one jump at their jump-weighted mean: beyond them both give the same
 * ramp, so only knots straddling a sample lose fit. Near-cancelling clusters are dropped;
 * the refinement after the merge absorbs both.
 */
static void model1_merge(model1_t* model, double hcell)
{
  const int nh = model->nh;
  double *const w = model->param, *const c = w + nh, *const b = c + nh, *const d = b + nh;
  double jump[BRACKET_H1MAX], knot[BRACKET_H1MAX], d0 = d[0], d1 = d[1], total = 0;
  int count = 0, nout = 0, changed = 1, i, j;
  for (i = 0; i < nh; ++i) {
    double ww = w[i], bb = b[i];
    if (0 > ww) {
      d0 -= c[i] * bb;
      d1 += c[i] * ww;
      ww = -ww;
      bb = -bb;
    }
    if (0 == ww) {
      if (0 > bb) d0 -= c[i] * bb;
    }
    else if (0 >= bb) {
      d0 -= c[i] * bb;
      d1 += c[i] * ww;
    }
    else if (bb < ww) {
      const double a = bb / ww, jj = c[i] * ww;
      for (j = count; 0 < j && a < knot[j - 1]; --j) {
        knot[j] = knot[j - 1];
        jump[j] = jump[j - 1];
      }
      knot[j] = a;
      jump[j] = jj;
      total += fabs(jj);
      ++count;
    }
  }
  while (0 != changed) { /* merged clusters stay sorted and may merge again */
    nout = 0;
    for (i = 0; i < count; i = j) {
      double sj = jump[i], sa = jump[i] * knot[i], sabs = fabs(jump[i]);
      int k;
      for (j = i + 1; j < count && knot[j] - knot[i] <= 1.05 * hcell; ++j) {
        sj += jump[j];
        sa += jump[j] * knot[j];
        sabs += fabs(jump[j]);
      }
      if (1E-6 * total < fabs(sj) && 0.5 * sabs <= fabs(sj)) {
        knot[nout] = LIBXS_CLMP(sa / sj, knot[i], knot[j - 1]);
        jump[nout] = sj;
        ++nout;
      }
      else if (1E-6 * total < fabs(sj) && 0.1 * sabs < fabs(sj)) {
        for (k = i; k < j; ++k) {
          knot[nout] = knot[k];
          jump[nout] = jump[k];
          ++nout;
        }
      }
    }
    changed = (nout < count ? 1 : 0);
    count = nout;
  }
  model->nh = nout;
  for (i = 0; i < nout; ++i) {
    model->param[i] = 1;
    model->param[nout + i] = jump[i];
    model->param[2 * nout + i] = knot[i];
  }
  model->param[3 * nout] = d0;
  model->param[3 * nout + 1] = d1;
}


/**
 * Against a missed edge: the per-cell residual of the model's increments is split as by a
 * decision stump (largest difference of the means left and right), and a jump of that
 * difference is placed such that the cell's own residual vanishes; then refine and merge.
 */
static void model1_insert(model1_t* model, const double x[], const double y[], int n,
  int nmax, const train_t* refine)
{
  const double h = x[1] - x[0];
  double prev = -1;
  int done = 0, round;
  for (round = 0; 0 == done && model->nh < nmax && round < nmax; ++round) {
    model1_t value = *model;
    double e[BRACKET_N1MAX + 1], g0, sum = 0, total = 0, rms = 0, scale = 0, gain = 0;
    double jump = 0, offset = 0;
    int k, m = -1;
    value.integrated = 0;
    g0 = model1_value(&value, x[0]);
    for (k = 0; k + 1 < n; ++k) {
      const double g1 = model1_value(&value, x[k + 1]);
      e[k] = ((g1 - g0) - (y[k + 1] - y[k])) / h;
      scale = LIBXS_MAX(scale, fabs(y[k + 1] - y[k]) / h);
      rms += e[k] * e[k];
      total += e[k];
      g0 = g1;
    }
    rms = sqrt(rms / (double)(n - 1));
    for (k = 0; k + 2 < n; ++k) {
      const double nl = (double)(k + 1), nr = (double)(n - 2 - k);
      double ml, mr;
      sum += e[k];
      ml = sum / nl;
      mr = (total - sum) / nr;
      if (gain < nl * nr * (ml - mr) * (ml - mr) / (nl + nr)) {
        gain = nl * nr * (ml - mr) * (ml - mr) / (nl + nr);
        jump = ml - mr;
        offset = -ml;
        m = k;
      }
    }
    if (rms <= 1E-9 * scale || (0 <= prev && prev <= rms) || 0 > m || 0 == jump) done = 1;
    else {
      const int nh = model->nh;
      const double a = LIBXS_CLMP(x[m + 1] + (e[m] + offset) * h / jump, x[m], x[m + 1]);
      double tmp[BRACKET_NPARAM1];
      int i;
      memcpy(tmp, model->param, sizeof(double) * (size_t)(3 * nh + 2));
      for (i = 0; i < nh; ++i) {
        model->param[i] = tmp[i];
        model->param[nh + 1 + i] = tmp[nh + i];
        model->param[2 * (nh + 1) + i] = tmp[2 * nh + i];
      }
      model->param[nh] = 1;
      model->param[2 * nh + 1] = jump;
      model->param[3 * nh + 2] = a;
      model->param[3 * (nh + 1)] = tmp[3 * nh];
      model->param[3 * (nh + 1) + 1] = tmp[3 * nh + 1] + offset;
      model->nh = nh + 1;
      model1_train(model, x, y, n, refine);
      model1_merge(model, h);
      prev = rms;
    }
  }
}


/* Cholesky solve of the symmetric positive definite a x = b in place (b receives x) */
static int solve_spd(double a[], double b[], int n)
{
  int result = EXIT_SUCCESS, i, j, k;
  for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
    double d = a[j + n * j];
    for (k = 0; k < j; ++k) d -= a[j + n * k] * a[j + n * k];
    if (0 < d) {
      a[j + n * j] = sqrt(d);
      for (i = j + 1; i < n; ++i) {
        double t = a[i + n * j];
        for (k = 0; k < j; ++k) t -= a[i + n * k] * a[j + n * k];
        a[i + n * j] = t / a[j + n * j];
      }
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < n; ++i) {
      for (k = 0; k < i; ++k) b[i] -= a[i + n * k] * b[k];
      b[i] /= a[i + n * i];
    }
    for (i = n - 1; 0 <= i; --i) {
      for (k = i + 1; k < n; ++k) b[i] -= a[k + n * i] * b[k];
      b[i] /= a[i + n * i];
    }
  }
  return result;
}


/**
 * With the knots of a merged model fixed, its cell averages are linear in the slope and
 * the jumps (a cell holding a knot sees the fraction it covers): least squares on the
 * cell averages, leaving out unit skip (-1: none). Returns the squared residual.
 */
static double model1_refit(model1_t* model, const double x[], const double avg[], int n,
  int skip)
{
  const int nh = model->nh, nc = n - 1;
  double ata[(BRACKET_H1MAX + 1) * (BRACKET_H1MAX + 1)], atb[BRACKET_H1MAX + 1];
  double phi[BRACKET_H1MAX + 1], rss = 0;
  int i, j, k, cols[BRACKET_H1MAX + 1], m = 0;
  for (j = 0; j <= nh; ++j) { /* column nh is the slope */
    if (j != skip) cols[m++] = j;
  }
  memset(ata, 0, sizeof(double) * (size_t)(m * m));
  memset(atb, 0, sizeof(double) * (size_t)m);
  for (k = 0; k < nc; ++k) {
    for (i = 0; i < m; ++i) {
      const int u = cols[i];
      const double a = (nh == u ? 0 : model->param[2 * nh + u]);
      phi[i] = (nh == u ? 1.0 : LIBXS_CLMP((x[k + 1] - a) / (x[k + 1] - x[k]), 0.0, 1.0));
    }
    for (i = 0; i < m; ++i) {
      for (j = 0; j < m; ++j) ata[i + m * j] += phi[i] * phi[j];
      atb[i] += phi[i] * avg[k];
    }
  }
  { /* nearly collinear columns (knots at a bound or in one cell) need the ridge */
    double dmax = 0;
    for (i = 0; i < m; ++i) dmax = LIBXS_MAX(dmax, ata[i + m * i]);
    for (i = 0; i < m; ++i) ata[i + m * i] += 1E-10 * dmax;
  }
  if (EXIT_SUCCESS == solve_spd(ata, atb, m)) {
    for (k = 0; k < nc; ++k) {
      double e = -avg[k];
      for (i = 0; i < m; ++i) {
        const int u = cols[i];
        const double a = (nh == u ? 0 : model->param[2 * nh + u]);
        e += atb[i] * (nh == u ? 1.0
          : LIBXS_CLMP((x[k + 1] - a) / (x[k + 1] - x[k]), 0.0, 1.0));
      }
      rss += e * e;
    }
    if (0 > skip) {
      for (i = 0; i < m; ++i) {
        if (nh == cols[i]) model->param[3 * nh + 1] = atb[i];
        else {
          model->param[cols[i]] = 1;
          model->param[nh + cols[i]] = atb[i];
        }
      }
    }
  }
  else rss = 1E300;
  return rss;
}


/**
 * Robust noise level of cell averages from the absolute deviations of their differences:
 * the median (MAD), unless edges make up so many differences that the median measures
 * them; the lower quartile (0.3186 for a unit normal) then stands in, being far smaller.
 */
static double noise_sigma(const double avg[], int n)
{
  double d[BRACKET_N1MAX + 1], med = 0, mad, quartile;
  int i, j, k;
  for (i = 0; i + 1 < n; ++i) d[i] = avg[i + 1] - avg[i];
  for (k = 0; k < 2; ++k) {
    for (i = 1; i + 1 < n; ++i) { /* insertion sort */
      const double v = d[i];
      for (j = i; 0 < j && v < d[j - 1]; --j) d[j] = d[j - 1];
      d[j] = v;
    }
    med = 0.5 * (d[(n - 2) / 2] + d[(n - 1) / 2]);
    if (0 == k) {
      for (i = 0; i + 1 < n; ++i) d[i] = fabs(d[i] - med);
    }
  }
  mad = med / 0.6745 / sqrt(2.0);
  quartile = d[(n - 2) / 4] / 0.3186 / sqrt(2.0);
  return (mad <= 2 * quartile ? mad : quartile);
}


/**
 * Backward elimination of jumps: the jump whose removal (with all heights refitted) costs
 * least is dropped while the squared residual grows by at most 2 sigma^2 ln(cells).
 */
static void model1_prune(model1_t* model, const double x[], const double y[], int n,
  double sigma)
{
  const double penalty = 2 * sigma * sigma * log((double)(n - 1));
  double avg[BRACKET_N1MAX + 1], rss;
  int done = 0, i, j;
  for (i = 0; i + 1 < n; ++i) avg[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
  model1_merge(model, x[1] - x[0]); /* canonical units (w = 1, knot in b), one per cell */
  rss = model1_refit(model, x, avg, n, -1);
  if (1E299 < rss) done = 1; /* a failed refit prices every removal alike: keep all */
  while (0 == done && 0 < model->nh) {
    double best = 1E300;
    int jbest = -1;
    for (j = 0; j < model->nh; ++j) {
      const double value = model1_refit(model, x, avg, n, j);
      if (value < best) {
        best = value;
        jbest = j;
      }
    }
    if (0 <= jbest && 1E299 > best && best - rss <= penalty) {
      const int nh = model->nh;
      double tmp[BRACKET_NPARAM1];
      memcpy(tmp, model->param, sizeof(double) * (size_t)(3 * nh + 2));
      for (i = j = 0; i < nh; ++i) {
        if (i != jbest) {
          model->param[j] = tmp[i];
          model->param[nh - 1 + j] = tmp[nh + i];
          model->param[2 * (nh - 1) + j] = tmp[2 * nh + i];
          ++j;
        }
      }
      model->param[3 * (nh - 1)] = tmp[3 * nh];
      model->param[3 * (nh - 1) + 1] = tmp[3 * nh + 1];
      model->nh = nh - 1;
      rss = model1_refit(model, x, avg, n, -1);
    }
    else done = 1;
  }
}


static int model1_eval(const void* model, size_t n, double out[])
{
  const model1_t *const model1 = (const model1_t*)model;
  size_t j;
  for (j = 0; j < n; ++j) out[j] = model1_value(model1, (double)j / (double)(n - 1));
  return EXIT_SUCCESS;
}


static double checker_edge(int i)
{
  return ((double)i + BRACKET_PHASE) / BRACKET_CELLS;
}


/* periodic checkerboard turned about the center; angle 0 has its edges at checker_edge */
static double checker_value(double x, double y, double angle)
{
  const double c = cos(angle), s = sin(angle);
  const double u = c * (x - 0.5) + s * (y - 0.5) + 0.5, v = c * (y - 0.5) - s * (x - 0.5) + 0.5;
  const int iu = (int)floor(BRACKET_CELLS * u - BRACKET_PHASE);
  const int iv = (int)floor(BRACKET_CELLS * v - BRACKET_PHASE);
  return (0 == ((iu + iv) & 1) ? 1.0 : -1.0);
}


static double disc_value(double x, double y, double angle)
{
  const double dx = x - BRACKET_DISC_X, dy = y - BRACKET_DISC_Y;
  LIBXS_UNUSED(angle);
  return (dx * dx + dy * dy < BRACKET_DISC_R * BRACKET_DISC_R ? 1.0 : -1.0);
}


/* edges along the row y, each located by bisection */
static void checker_row(value_fn_t value, double y, double angle, target1_t* row)
{
  const int n = BRACKET_HI1;
  double prev = value(0, y, angle);
  int j, k;
  row->name = "row";
  row->nedges = 0;
  row->noise = 0;
  row->level[0] = prev;
  for (j = 1; j < n && row->nedges < BRACKET_MAX_EDGES; ++j) {
    const double next = value((double)j / (double)(n - 1), y, angle);
    if (next != prev) {
      double lo = (double)(j - 1) / (double)(n - 1), hi = (double)j / (double)(n - 1);
      for (k = 0; k < 50; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (value(mid, y, angle) == prev) lo = mid;
        else hi = mid;
      }
      row->edge[row->nedges] = 0.5 * (lo + hi);
      ++row->nedges;
      row->level[row->nedges] = next;
      prev = next;
    }
  }
}


/* parameters: w, v, c (decayed), then b, e */
static int tensor_init(tensor_t* model, int nu, int integrated)
{
  int result = EXIT_SUCCESS, u;
  model->param = (double*)malloc(sizeof(double) * 5 * (size_t)nu);
  model->nu = nu;
  model->integrated = integrated;
  if (NULL != model->param) {
    double *const w = model->param, *const v = w + nu, *const c = v + nu;
    double *const b = c + nu, *const e = b + nu;
    for (u = 0; u < nu; ++u) {
      w[u] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
      v[u] = (0.5 > libxs_rng_f64() ? -1.0 : 1.0) * (0.5 + libxs_rng_f64());
      b[u] = w[u] * libxs_rng_f64();
      e[u] = v[u] * libxs_rng_f64();
      c[u] = 0.1 * (2.0 * libxs_rng_f64() - 1.0);
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


/* column u of relu/step holds <s x - t>^1 and <s x - t>^0 on the grid (axis 0: w,b; 1: v,e) */
static void tensor_factor(const double param[], int nu, int axis, size_t n, double lo,
  double h, double relu[], double step[])
{
  const double *const s = param + (0 == axis ? 0 : nu);
  const double *const t = param + 3 * nu + (0 == axis ? 0 : nu);
  size_t j;
  int u;
  for (u = 0; u < nu; ++u) {
    for (j = 0; j < n; ++j) {
      const double z = s[u] * (lo + h * (double)j) - t[u];
      if (NULL != relu) relu[j + n * (size_t)u] = (0 < z ? z : 0);
      if (NULL != step) step[j + n * (size_t)u] = (0 < z ? 1 : 0);
    }
  }
}


/* G = sum_u c_u p_u q_u^T = (P diag(c)) Q^T, and the gradient needs R Q and R^T P */
static int tensor_train(tensor_t* model, const double y[], int n, double lo, double h,
  const train_t* train)
{
  const int nu = model->nu, nparam = 5 * nu;
  const size_t nn = (size_t)n * (size_t)n, nf = (size_t)n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * (nn + 5 * nf + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, u, j;
  if (EXIT_SUCCESS == result) {
    double *const r = work, *const p = r + nn, *const q = p + nf, *const pc = q + nf;
    double *const rq = pc + nf, *const rp = rq + nf, *const grad = rp + nf;
    double *const m = grad + nparam, *const v = m + nparam;
    const double *const w = model->param, *const vy = w + nu, *const c = vy + nu;
    double *const gw = grad, *const gv = gw + nu, *const gc = gv + nu;
    double *const gb = gc + nu, *const ge = gb + nu;
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      size_t i;
      tensor_factor(model->param, nu, 0, (size_t)n, lo, h, p, NULL);
      tensor_factor(model->param, nu, 1, (size_t)n, lo, h, q, NULL);
      for (u = 0; u < nu; ++u) {
        for (j = 0; j < n; ++j) pc[j + (size_t)n * u] = c[u] * p[j + (size_t)n * u];
      }
      result = gemm_f64('N', 'T', n, n, nu, pc, n, q, n, r, n);
      for (i = 0; i < nn; ++i) r[i] = (r[i] - y[i]) / (double)nn;
      if (EXIT_SUCCESS == result) result = gemm_f64('N', 'N', n, nu, n, r, n, q, n, rq, n);
      if (EXIT_SUCCESS == result) result = gemm_f64('T', 'N', n, nu, n, r, n, p, n, rp, n);
      if (EXIT_SUCCESS == result) {
        for (u = 0; u < nu; ++u) {
          double sc = 0, sw = 0, sb = 0, sv = 0, se = 0;
          for (j = 0; j < n; ++j) {
            const size_t k = j + (size_t)n * u;
            const double x = lo + h * (double)j;
            sc += p[k] * rq[k];
            if (0 < p[k]) {
              sw += x * rq[k];
              sb -= rq[k];
            }
            if (0 < q[k]) {
              sv += x * rp[k];
              se -= rp[k];
            }
          }
          gc[u] = sc;
          gw[u] = c[u] * sw;
          gb[u] = c[u] * sb;
          gv[u] = c[u] * sv;
          ge[u] = c[u] * se;
        }
        adam_update(model->param, grad, m, v, nparam, 3 * nu, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


/* integrated: d^2/dxdy turns each (1,1) product into the (0,0) product of steps */
static int tensor_eval(const void* model, size_t n, double out[])
{
  const tensor_t *const tensor = (const tensor_t*)model;
  const int nu = tensor->nu;
  const size_t nf = n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * 2 * nf);
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), u;
  if (EXIT_SUCCESS == result) {
    double *const p = work, *const q = p + nf;
    const double *const w = tensor->param, *const v = w + nu, *const c = v + nu;
    size_t j;
    if (0 == tensor->integrated) {
      tensor_factor(tensor->param, nu, 0, n, 0, 1.0 / (double)(n - 1), p, NULL);
      tensor_factor(tensor->param, nu, 1, n, 0, 1.0 / (double)(n - 1), q, NULL);
    }
    else {
      tensor_factor(tensor->param, nu, 0, n, 0, 1.0 / (double)(n - 1), NULL, p);
      tensor_factor(tensor->param, nu, 1, n, 0, 1.0 / (double)(n - 1), NULL, q);
    }
    for (u = 0; u < nu; ++u) {
      const double s = (0 == tensor->integrated ? c[u] : c[u] * w[u] * v[u]);
      for (j = 0; j < n; ++j) p[j + n * (size_t)u] *= s;
    }
    result = gemm_f64('N', 'T', (int)n, (int)n, nu, p, (int)n, q, (int)n, out, (int)n);
  }
  free(work);
  return result;
}


/* parameters: ax, bx, ay, by, c; each unit is c (<x-ax>^1 - <x-bx>^1) (<y-ay>^1 - <y-by>^1) */
static int box_init(tensor_t* model, int nu, int integrated)
{
  int result = EXIT_SUCCESS, u;
  model->param = (double*)malloc(sizeof(double) * 5 * (size_t)nu);
  model->nu = nu;
  model->integrated = integrated;
  if (NULL != model->param) {
    for (u = 0; u < nu; ++u) {
      model->param[u] = libxs_rng_f64();
      model->param[nu + u] = model->param[u] + 0.1 + 0.3 * libxs_rng_f64();
      model->param[2 * nu + u] = libxs_rng_f64();
      model->param[3 * nu + u] = model->param[2 * nu + u] + 0.1 + 0.3 * libxs_rng_f64();
      model->param[4 * nu + u] = 0.1 * (2.0 * libxs_rng_f64() - 1.0);
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


/* column u holds the clamped ramp and the indicators beyond its two knots on the grid */
static void box_factor(const double param[], int nu, int axis, size_t n, double lo, double h,
  double ramp[], double below[], double above[])
{
  const double *const a = param + (0 == axis ? 0 : 2 * nu), *const b = a + nu;
  size_t j;
  int u;
  for (u = 0; u < nu; ++u) {
    for (j = 0; j < n; ++j) {
      const double z = lo + h * (double)j, za = z - a[u], zb = z - b[u];
      const size_t k = j + n * (size_t)u;
      if (NULL != ramp) ramp[k] = (0 < za ? za : 0) - (0 < zb ? zb : 0);
      if (NULL != below) below[k] = (0 < za ? 1 : 0);
      if (NULL != above) above[k] = (0 < zb ? 1 : 0);
    }
  }
}


/* as tensor_train, but every term is bounded by the width of its box */
static int box_train(tensor_t* model, const double y[], int n, double lo, double h,
  const train_t* train)
{
  const int nu = model->nu, nparam = 5 * nu;
  const size_t nn = (size_t)n * (size_t)n, nf = (size_t)n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * (nn + 9 * nf + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, u, j;
  if (EXIT_SUCCESS == result) {
    double *const r = work, *const p = r + nn, *const q = p + nf, *const pc = q + nf;
    double *const rq = pc + nf, *const rp = rq + nf, *const pa = rp + nf, *const pb = pa + nf;
    double *const qa = pb + nf, *const qb = qa + nf, *const grad = qb + nf;
    double *const m = grad + nparam, *const v = m + nparam;
    const double *const c = model->param + 4 * nu;
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      size_t i;
      box_factor(model->param, nu, 0, (size_t)n, lo, h, p, pa, pb);
      box_factor(model->param, nu, 1, (size_t)n, lo, h, q, qa, qb);
      for (u = 0; u < nu; ++u) {
        for (j = 0; j < n; ++j) pc[j + (size_t)n * u] = c[u] * p[j + (size_t)n * u];
      }
      result = gemm_f64('N', 'T', n, n, nu, pc, n, q, n, r, n);
      for (i = 0; i < nn; ++i) r[i] = (r[i] - y[i]) / (double)nn;
      if (EXIT_SUCCESS == result) result = gemm_f64('N', 'N', n, nu, n, r, n, q, n, rq, n);
      if (EXIT_SUCCESS == result) result = gemm_f64('T', 'N', n, nu, n, r, n, p, n, rp, n);
      if (EXIT_SUCCESS == result) {
        for (u = 0; u < nu; ++u) {
          double sc = 0, sax = 0, sbx = 0, say = 0, sby = 0;
          for (j = 0; j < n; ++j) {
            const size_t k = j + (size_t)n * u;
            sc += p[k] * rq[k];
            sax -= pa[k] * rq[k];
            sbx += pb[k] * rq[k];
            say -= qa[k] * rp[k];
            sby += qb[k] * rp[k];
          }
          grad[u] = c[u] * sax;
          grad[nu + u] = c[u] * sbx;
          grad[2 * nu + u] = c[u] * say;
          grad[3 * nu + u] = c[u] * sby;
          grad[4 * nu + u] = sc;
        }
        adam_update(model->param, grad, m, v, nparam, 0, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


/* integrated: the mixed derivative of each term is c times the indicator of its rectangle */
static int box_eval(const void* model, size_t n, double out[])
{
  const tensor_t *const box = (const tensor_t*)model;
  const int nu = box->nu;
  const size_t nf = n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * 4 * nf);
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), u;
  if (EXIT_SUCCESS == result) {
    double *const p = work, *const q = p + nf, *const pb = q + nf, *const qb = pb + nf;
    const double *const c = box->param + 4 * nu;
    const double h = 1.0 / (double)(n - 1);
    size_t j;
    if (0 == box->integrated) {
      box_factor(box->param, nu, 0, n, 0, h, p, NULL, NULL);
      box_factor(box->param, nu, 1, n, 0, h, q, NULL, NULL);
    }
    else {
      box_factor(box->param, nu, 0, n, 0, h, NULL, p, pb);
      box_factor(box->param, nu, 1, n, 0, h, NULL, q, qb);
      for (j = 0; j < nf; ++j) {
        p[j] -= pb[j];
        q[j] -= qb[j];
      }
    }
    for (u = 0; u < nu; ++u) {
      for (j = 0; j < n; ++j) p[j + n * (size_t)u] *= c[u];
    }
    result = gemm_f64('N', 'T', (int)n, (int)n, nu, p, (int)n, q, (int)n, out, (int)n);
  }
  free(work);
  return result;
}


/**
 * Refines rectangles on the cell averages: the overlap of a box with a cell is a clamped
 * linear function of its edges, so an edge is moved by c/h times the residual of the cells
 * it lies in, summed along the edge (a difference loss, as for mlp_int).
 */
static int box_cells(tensor_t* model, const double data[], int n, double h,
  const train_t* train)
{
  const int nu = model->nu, nparam = 5 * nu;
  const size_t nn = (size_t)n * (size_t)n, nf = (size_t)n * (size_t)nu;
  double *const work = (double*)malloc(sizeof(double) * (nn + 5 * nf + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, u, i;
  if (EXIT_SUCCESS == result) {
    double *const e = work, *const fx = e + nn, *const fy = fx + nf, *const pc = fy + nf;
    double *const eq = pc + nf, *const ep = eq + nf, *const grad = ep + nf;
    double *const m = grad + nparam, *const v = m + nparam, *const prm = model->param;
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      size_t k;
      for (u = 0; u < nu; ++u) {
        for (i = 0; i < n; ++i) {
          const double x0 = h * i, x1 = x0 + h;
          fx[i + (size_t)n * u] = (LIBXS_CLMP(prm[nu + u], x0, x1)
            - LIBXS_CLMP(prm[u], x0, x1)) / h;
          fy[i + (size_t)n * u] = (LIBXS_CLMP(prm[3 * nu + u], x0, x1)
            - LIBXS_CLMP(prm[2 * nu + u], x0, x1)) / h;
          pc[i + (size_t)n * u] = prm[4 * nu + u] * fx[i + (size_t)n * u];
        }
      }
      result = gemm_f64('N', 'T', n, n, nu, pc, n, fy, n, e, n);
      for (k = 0; k < nn; ++k) e[k] = (e[k] - data[k]) / (double)nn;
      if (EXIT_SUCCESS == result) result = gemm_f64('N', 'N', n, nu, n, e, n, fy, n, eq, n);
      if (EXIT_SUCCESS == result) result = gemm_f64('T', 'N', n, nu, n, e, n, fx, n, ep, n);
      if (EXIT_SUCCESS == result) {
        for (u = 0; u < nu; ++u) {
          const double c = prm[4 * nu + u];
          double gc = 0;
          int q;
          for (i = 0; i < n; ++i) gc += fx[i + (size_t)n * u] * eq[i + (size_t)n * u];
          grad[4 * nu + u] = gc;
          for (q = 0; q < 4; ++q) { /* ax, bx, ay, by: only the cell holding the edge */
            const double pos = prm[q * nu + u], *const side = (2 > q ? eq : ep);
            const int cell = (int)floor(pos / h);
            grad[q * nu + u] = ((0 <= cell && cell < n)
              ? (0 == (q & 1) ? -1.0 : 1.0) * c / h * side[cell + (size_t)n * u] : 0);
          }
        }
        adam_update(prm, grad, m, v, nparam, 0, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


/**
 * Stratification by rectangles: the residual of the cell averages gets a summed-area table,
 * which prices every axis-aligned rectangle in O(1) by its gain S^2/area (a 2-D stump); the
 * best one is added with the mean residual as its height, and all rectangles are refined.
 */
static int box_greedy(tensor_t* model, const double data[], const double area[], int n,
  double h, int nmax, const train_t* train, const train_t* refine)
{
  const size_t nn = (size_t)n * (size_t)n, nc = (size_t)n + 1;
  double *const work = (double*)malloc(sizeof(double) * (nn + nc * nc + 2 * (size_t)n));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), done = 0;
  model->param = (double*)malloc(sizeof(double) * 5 * (size_t)nmax);
  model->nu = 0;
  model->integrated = 1;
  if (NULL == model->param) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    double *const e = work, *const sat = e + nn, *const fx = sat + nc * nc, *const fy = fx + n;
    double total = 0;
    size_t i, j;
    for (i = 0; i < nn; ++i) total += data[i] * data[i];
    { /**
       * Two rectangles meeting inside one pixel leave its average to two edges, which it
       * cannot tell apart (a gap fits as well); a background at the lowest level makes each
       * edge separate a rectangle from what lies beneath it.
       */
      double base = data[0];
      for (i = 1; i < nn; ++i) base = LIBXS_MIN(base, data[i]);
      model->param[0] = model->param[2] = 0;
      model->param[1] = model->param[3] = 1;
      model->param[4] = base;
      model->nu = 1;
    }
    while (0 == done && model->nu < nmax && EXIT_SUCCESS == result) {
      const int nu = model->nu;
      double best = 0, height = 0;
      size_t i0, i1, j0, j1, bi0 = 0, bi1 = 0, bj0 = 0, bj1 = 0;
      int u;
      memcpy(e, data, sizeof(double) * nn);
      for (u = 0; u < nu; ++u) { /* signed overlap of each box with the cells, per axis */
        const double *const prm = model->param, c = prm[4 * nu + u];
        for (i = 0; i < (size_t)n; ++i) {
          const double x0 = h * (double)i, x1 = x0 + h;
          fx[i] = (LIBXS_CLMP(prm[nu + u], x0, x1) - LIBXS_CLMP(prm[u], x0, x1)) / h;
          fy[i] = (LIBXS_CLMP(prm[3 * nu + u], x0, x1) - LIBXS_CLMP(prm[2 * nu + u], x0, x1))
            / h;
        }
        for (j = 0; j < (size_t)n; ++j) {
          for (i = 0; i < (size_t)n; ++i) e[i + n * j] -= c * fx[i] * fy[j];
        }
      }
      for (j = 0; j < nc; ++j) {
        for (i = 0; i < nc; ++i) {
          sat[i + nc * j] = (0 == i || 0 == j ? 0 : sat[i - 1 + nc * j]
            + sat[i + nc * (j - 1)] - sat[i - 1 + nc * (j - 1)] + e[i - 1 + n * (j - 1)]);
        }
      }
      for (j0 = 0; j0 < nc; ++j0) {
        for (j1 = j0 + 1; j1 < nc; ++j1) {
          for (i0 = 0; i0 < nc; ++i0) {
            for (i1 = i0 + 1; i1 < nc; ++i1) {
              const double sum = sat[i1 + nc * j1] - sat[i0 + nc * j1] - sat[i1 + nc * j0]
                + sat[i0 + nc * j0];
              const double cnt = (double)((i1 - i0) * (j1 - j0));
              if (best < sum * sum / cnt) {
                best = sum * sum / cnt;
                height = sum / cnt;
                bi0 = i0;
                bi1 = i1;
                bj0 = j0;
                bj1 = j1;
              }
            }
          }
        }
      }
      if (best <= 1E-12 * total) done = 1;
      else {
        double tmp[5 * BRACKET_U2];
        int q;
        memcpy(tmp, model->param, sizeof(double) * 5 * (size_t)nu);
        for (q = 0; q < 5; ++q) {
          for (u = 0; u < nu; ++u) model->param[q * (nu + 1) + u] = tmp[q * nu + u];
        }
        { /* each edge extends into its neighboring strip by the fraction its residual asks */
          double fl = 0, fr = 0, fb = 0, ft = 0;
          for (j = bj0; j < bj1; ++j) {
            if (0 < bi0) fl += e[bi0 - 1 + n * j];
            if (bi1 < (size_t)n) fr += e[bi1 + n * j];
          }
          for (i = bi0; i < bi1; ++i) {
            if (0 < bj0) fb += e[i + n * (bj0 - 1)];
            if (bj1 < (size_t)n) ft += e[i + n * bj1];
          }
          fl = LIBXS_CLMP(fl / (double)(bj1 - bj0) / height, 0.0, 1.0);
          fr = LIBXS_CLMP(fr / (double)(bj1 - bj0) / height, 0.0, 1.0);
          fb = LIBXS_CLMP(fb / (double)(bi1 - bi0) / height, 0.0, 1.0);
          ft = LIBXS_CLMP(ft / (double)(bi1 - bi0) / height, 0.0, 1.0);
          model->param[nu] = h * ((double)bi0 - fl);
          model->param[(nu + 1) + nu] = h * ((double)bi1 + fr);
          model->param[2 * (nu + 1) + nu] = h * ((double)bj0 - fb);
          model->param[3 * (nu + 1) + nu] = h * ((double)bj1 + ft);
        }
        model->param[4 * (nu + 1) + nu] = height;
        model->nu = nu + 1;
        result = box_cells(model, data, n, h, refine);
      }
    }
    if (EXIT_SUCCESS == result && 0 < model->nu) {
      result = box_cells(model, data, n, h, train);
    }
    LIBXS_UNUSED(area);
  }
  free(work);
  return result;
}


/* parameters: W1 (x then y column), W2, w3 (decayed), then b1, b2, b3 */
static int mlp_init(mlp_t* model, int nh, int integrated)
{
  const size_t nparam = (size_t)(nh * nh + 6 * nh + 1);
  int result = EXIT_SUCCESS, h;
  model->param = (double*)malloc(sizeof(double) * nparam);
  model->nh = nh;
  model->integrated = integrated;
  if (NULL != model->param) {
    double *const w1 = model->param, *const w2 = w1 + 2 * nh, *const w3 = w2 + nh * nh;
    double *const b1 = w3 + nh, *const b2 = b1 + nh, *const b3 = b2 + nh;
    const double s1 = sqrt(3.0), s2 = sqrt(6.0 / nh);
    for (h = 0; h < nh; ++h) {
      w1[h] = s1 * (2.0 * libxs_rng_f64() - 1.0);
      w1[nh + h] = s1 * (2.0 * libxs_rng_f64() - 1.0);
      b1[h] = w1[h] * libxs_rng_f64() + w1[nh + h] * libxs_rng_f64();
      b2[h] = 0;
      w3[h] = 0.1 * s2 * (2.0 * libxs_rng_f64() - 1.0);
    }
    for (h = 0; h < nh * nh; ++h) w2[h] = s2 * (2.0 * libxs_rng_f64() - 1.0);
    b3[0] = 0;
  }
  else result = EXIT_FAILURE;
  return result;
}


/* with s1 given, out is the derivative along (1,1): a sum of products of zero-order brackets */
static int mlp_forward(const double param[], int nh, const double xs[], const double ys[],
  int npts, double a1[], double a2[], double s1[], double out[])
{
  const double *const w1 = param, *const w2 = w1 + 2 * nh, *const w3 = w2 + nh * nh;
  const double *const b1 = w3 + nh, *const b2 = b1 + nh, *const b3 = b2 + nh;
  int result, i, h;
  for (i = 0; i < npts; ++i) {
    for (h = 0; h < nh; ++h) {
      const size_t k = h + (size_t)nh * i;
      const double z = w1[h] * xs[i] + w1[nh + h] * ys[i] - b1[h];
      a1[k] = (0 < z ? z : 0);
      if (NULL != s1) s1[k] = (0 < z ? w1[h] + w1[nh + h] : 0);
    }
  }
  result = gemm_f64('N', 'N', nh, npts, nh, w2, nh, a1, nh, a2, nh);
  if (EXIT_SUCCESS == result && NULL != s1) {
    result = gemm_f64('N', 'N', nh, npts, nh, w2, nh, s1, nh, a1, nh);
  }
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < npts; ++i) {
      double s = (NULL == s1 ? -b3[0] : 0);
      for (h = 0; h < nh; ++h) {
        const size_t k = h + (size_t)nh * i;
        const double z = a2[k] - b2[h];
        a2[k] = (0 < z ? z : 0);
        if (0 < z) s += w3[h] * (NULL == s1 ? z : a1[k]);
      }
      out[i] = s;
    }
  }
  return result;
}


static int mlp_train(mlp_t* model, const double y[], int n, double lo, double dx,
  const train_t* train)
{
  const int nh = model->nh, npts = n * n, nparam = nh * nh + 6 * nh + 1;
  const size_t na = (size_t)nh * (size_t)npts;
  double *const work = (double*)malloc(sizeof(double)
    * (4 * na + 4 * (size_t)npts + 3 * (size_t)nparam));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), step, i, h;
  if (EXIT_SUCCESS == result) {
    double *const a1 = work, *const a2 = a1 + na, *const d1 = a2 + na, *const d2 = d1 + na;
    double *const xs = d2 + na, *const ys = xs + npts, *const out = ys + npts;
    double *const g = out + npts, *const grad = g + npts;
    double *const m = grad + nparam, *const v = m + nparam;
    const double *const w2 = model->param + 2 * nh, *const w3 = w2 + nh * nh;
    double *const gw1 = grad, *const gw2 = gw1 + 2 * nh, *const gw3 = gw2 + nh * nh;
    double *const gb1 = gw3 + nh, *const gb2 = gb1 + nh, *const gb3 = gb2 + nh;
    for (i = 0; i < npts; ++i) {
      xs[i] = lo + dx * (double)(i % n);
      ys[i] = lo + dx * (double)(i / n);
    }
    memset(m, 0, sizeof(double) * 2 * (size_t)nparam);
    for (step = 1; step <= train->nsteps && EXIT_SUCCESS == result; ++step) {
      result = mlp_forward(model->param, nh, xs, ys, npts, a1, a2, NULL, out);
      if (EXIT_SUCCESS == result && 0 == model->integrated) {
        for (i = 0; i < npts; ++i) g[i] = (out[i] - y[i]) / (double)npts;
      }
      else if (EXIT_SUCCESS == result) {
        /* only (1,1)-differences reach the derivative, and they keep order-1 gradients */
        const double scale = 1.0 / ((double)(n - 1) * (double)(n - 1));
        int ix, iy;
        memset(g, 0, sizeof(double) * (size_t)npts);
        for (iy = 1; iy < n; ++iy) {
          for (ix = 1; ix < n; ++ix) {
            const int k = ix + n * iy, km = k - n - 1;
            const double rk = scale * ((out[k] - out[km]) - (y[k] - y[km]));
            g[k] += rk;
            g[km] -= rk;
          }
        }
      }
      if (EXIT_SUCCESS == result) {
        gb3[0] = 0;
        for (h = 0; h < nh; ++h) gw3[h] = 0;
        for (i = 0; i < npts; ++i) {
          gb3[0] -= g[i];
          for (h = 0; h < nh; ++h) {
            const size_t k = h + (size_t)nh * i;
            gw3[h] += g[i] * a2[k];
            d2[k] = (0 < a2[k] ? w3[h] * g[i] : 0);
          }
        }
        result = gemm_f64('N', 'T', nh, nh, npts, d2, nh, a1, nh, gw2, nh);
      }
      if (EXIT_SUCCESS == result) {
        result = gemm_f64('T', 'N', nh, npts, nh, w2, nh, d2, nh, d1, nh);
      }
      if (EXIT_SUCCESS == result) {
        for (h = 0; h < nh; ++h) gw1[h] = gw1[nh + h] = gb1[h] = gb2[h] = 0;
        for (i = 0; i < npts; ++i) {
          for (h = 0; h < nh; ++h) {
            const size_t k = h + (size_t)nh * i;
            gb2[h] -= d2[k];
            if (0 < a1[k]) {
              gw1[h] += d1[k] * xs[i];
              gw1[nh + h] += d1[k] * ys[i];
              gb1[h] -= d1[k];
            }
          }
        }
        adam_update(model->param, grad, m, v, nparam, nh * nh + 3 * nh, step,
          cosine_lr(train->lr, step - 1, train->nsteps), train->wd);
      }
    }
  }
  free(work);
  return result;
}


static int mlp_eval(const void* model, size_t n, double out[])
{
  const mlp_t *const mlp = (const mlp_t*)model;
  const int nh = mlp->nh;
  const size_t npts = n * n, na = (size_t)nh * BRACKET_CHUNK;
  double *const work = (double*)malloc(sizeof(double) * (3 * na + 2 * BRACKET_CHUNK));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    double *const a1 = work, *const a2 = a1 + na, *const s1 = a2 + na;
    double *const xs = s1 + na, *const ys = xs + BRACKET_CHUNK;
    size_t base, i;
    for (base = 0; base < npts && EXIT_SUCCESS == result; base += BRACKET_CHUNK) {
      const size_t size = LIBXS_MIN(npts - base, (size_t)BRACKET_CHUNK);
      for (i = 0; i < size; ++i) {
        xs[i] = (double)((base + i) % n) / (double)(n - 1);
        ys[i] = (double)((base + i) / n) / (double)(n - 1);
      }
      result = mlp_forward(mlp->param, nh, xs, ys, (int)size, a1, a2,
        0 != mlp->integrated ? s1 : NULL, out + base);
    }
  }
  free(work);
  return result;
}


static void model1_sample(const model1_t* model, int integrated, const double x[], int n,
  double out[])
{
  model1_t value = *model;
  int i;
  value.integrated = integrated;
  for (i = 0; i < n; ++i) out[i] = model1_value(&value, x[i]);
}


/* one factor is a 1-D fit; with integration it is merged within span and grown greedily */
static void factor_fit(model1_t* model, const double x[], const double y[], int n,
  int units, int integrated, double span, const train_t* train, const train_t* refine)
{
  model1_init(model, units, integrated);
  model1_train(model, x, y, n, train);
  if (0 != integrated) {
    model1_merge(model, span);
    model1_train(model, x, y, n, refine);
    model1_insert(model, x, y, n, units, refine);
  }
}


/**
 * Alternating least squares on sum_r A_r(x) B_r(y): with B fixed, the best A on the grid
 * is the projection R b / |b|^2, a 1-D target for the factor fit, and vice versa; each
 * term is fitted to the residual left by the ones before (deflation).
 */
static int sep_train(sep_t* model, const double y[], int n, double lo, double h, int units,
  const train_t* train, const train_t* refine)
{
  const size_t nn = (size_t)n * (size_t)n;
  double *const work = (double*)malloc(sizeof(double) * (nn + 4 * (size_t)n));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), rank = model->rank, q, it, i, j;
  model->rank = 0;
  if (EXIT_SUCCESS == result) {
    double *const r = work, *const x = r + nn, *const a = x + n, *const b = a + n;
    double *const t = b + n, norm0 = 0, norm = 0, best = -1;
    int done = 0, istar = 0;
    memcpy(r, y, sizeof(double) * nn);
    for (i = 0; i < n; ++i) x[i] = lo + h * (double)i;
    for (i = 0; i < (int)nn; ++i) norm0 += r[i] * r[i];
    for (q = 0; q < rank && 0 == done; ++q) {
      for (i = 0; i < n; ++i) {
        double rowsq = 0;
        for (j = 0; j < n; ++j) rowsq += r[i + n * j] * r[i + n * j];
        if (best < rowsq) {
          best = rowsq;
          istar = i;
        }
      }
      for (j = 0; j < n; ++j) b[j] = r[istar + n * j];
      for (it = 0; it < BRACKET_ALS && 0 == done; ++it) {
        double bb = 0, aa = 0;
        for (j = 0; j < n; ++j) bb += b[j] * b[j];
        if (0 < bb) {
          for (i = 0; i < n; ++i) {
            double s = 0;
            for (j = 0; j < n; ++j) s += r[i + n * j] * b[j];
            t[i] = s / bb;
          }
          factor_fit(model->ax + q, x, t, n, units, model->integrated, h, train, refine);
          model1_sample(model->ax + q, 0, x, n, a);
          for (i = 0; i < n; ++i) aa += a[i] * a[i];
        }
        if (0 < aa) {
          for (j = 0; j < n; ++j) {
            double s = 0;
            for (i = 0; i < n; ++i) s += r[i + n * j] * a[i];
            t[j] = s / aa;
          }
          factor_fit(model->ay + q, x, t, n, units, model->integrated, h, train, refine);
          model1_sample(model->ay + q, 0, x, n, b);
        }
        else done = 1;
      }
      if (0 == done) {
        norm = 0;
        for (j = 0; j < n; ++j) {
          for (i = 0; i < n; ++i) {
            r[i + n * j] -= a[i] * b[j];
            norm += r[i + n * j] * r[i + n * j];
          }
        }
        model->rank = q + 1;
        best = -1;
        if (norm <= 1E-18 * norm0) done = 1;
      }
    }
  }
  free(work);
  return result;
}


/* integrated: the mixed derivative of sum_r A_r(x) B_r(y) is sum_r A_r'(x) B_r'(y) */
static int sep_eval(const void* model, size_t n, double out[])
{
  const sep_t *const sep = (const sep_t*)model;
  const int rank = sep->rank;
  double *const work = (double*)malloc(sizeof(double) * (2 * (size_t)rank + 1) * n);
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE), q;
  if (EXIT_SUCCESS == result && 0 < rank) {
    double *const xs = work, *const a = xs + n, *const b = a + n * (size_t)rank;
    size_t i;
    for (i = 0; i < n; ++i) xs[i] = (double)i / (double)(n - 1);
    for (q = 0; q < rank; ++q) {
      model1_sample(sep->ax + q, sep->integrated, xs, (int)n, a + n * (size_t)q);
      model1_sample(sep->ay + q, sep->integrated, xs, (int)n, b + n * (size_t)q);
    }
    result = gemm_f64('N', 'T', (int)n, (int)n, rank, a, (int)n, b, (int)n, out, (int)n);
  }
  else if (EXIT_SUCCESS == result) memset(out, 0, sizeof(double) * n * n);
  free(work);
  return result;
}


/* Sutherland-Hodgman step: keeps the part of the polygon on one side of an axis line */
static int clip_half(const double xin[], const double yin[], int n, double xout[],
  double yout[], int axis, double bound, int greater)
{
  int i, m = 0;
  for (i = 0; i < n; ++i) {
    const int j = (i + 1) % n;
    const double ci = (0 == axis ? xin[i] : yin[i]) - bound;
    const double cj = (0 == axis ? xin[j] : yin[j]) - bound;
    const int ini = (0 != greater ? 0 <= ci : 0 >= ci);
    const int inj = (0 != greater ? 0 <= cj : 0 >= cj);
    if (0 != ini) {
      xout[m] = xin[i];
      yout[m] = yin[i];
      ++m;
    }
    if (ini != inj) {
      const double t = ci / (ci - cj);
      xout[m] = xin[i] + t * (xin[j] - xin[i]);
      yout[m] = yin[i] + t * (yin[j] - yin[i]);
      ++m;
    }
  }
  return m;
}


/**
 * Averages the pixel data over cells of the pixel size in a frame turned by angle about the
 * center, weighting each pixel by its exact overlap; a cell is inside the image (mask 1)
 * when pixels cover its whole area. Returns the cells per axis.
 */
static int rot_cells(const double data[], int n, double angle, double cells[], double mask[],
  double* lo)
{
  const double c = cos(angle), s = sin(angle), h = 1.0 / n;
  const int nc = LIBXS_MIN((int)ceil((fabs(c) + fabs(s)) * n - 1E-9), BRACKET_CELLMAX);
  const double ulo = -0.5 * h * nc;
  int i, j, k, l, q;
  memset(cells, 0, sizeof(double) * (size_t)nc * (size_t)nc);
  memset(mask, 0, sizeof(double) * (size_t)nc * (size_t)nc);
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      double pu[4], pv[4], umin = 1E9, umax = -1E9, vmin = 1E9, vmax = -1E9;
      for (q = 0; q < 4; ++q) {
        const double x = h * (i + ((q + 1) >> 1 & 1)) - 0.5, y = h * (j + (q >> 1)) - 0.5;
        pu[q] = c * x + s * y;
        pv[q] = c * y - s * x;
        umin = LIBXS_MIN(umin, pu[q]);
        umax = LIBXS_MAX(umax, pu[q]);
        vmin = LIBXS_MIN(vmin, pv[q]);
        vmax = LIBXS_MAX(vmax, pv[q]);
      }
      for (l = LIBXS_MAX((int)floor((vmin - ulo) / h), 0);
        l <= LIBXS_MIN((int)floor((vmax - ulo) / h), nc - 1); ++l)
      {
        for (k = LIBXS_MAX((int)floor((umin - ulo) / h), 0);
          k <= LIBXS_MIN((int)floor((umax - ulo) / h), nc - 1); ++k)
        {
          double xa[16], ya[16], xb[16], yb[16], area = 0;
          int m = clip_half(pu, pv, 4, xa, ya, 0, ulo + h * k, 1);
          m = clip_half(xa, ya, m, xb, yb, 0, ulo + h * (k + 1), 0);
          m = clip_half(xb, yb, m, xa, ya, 1, ulo + h * l, 1);
          m = clip_half(xa, ya, m, xb, yb, 1, ulo + h * (l + 1), 0);
          for (q = 0; q < m; ++q) {
            area += xb[q] * yb[(q + 1) % m] - xb[(q + 1) % m] * yb[q];
          }
          area = 0.5 * fabs(area);
          cells[k + nc * l] += area * data[i + n * j];
          mask[k + nc * l] += area;
        }
      }
    }
  }
  for (k = 0; k < nc * nc; ++k) {
    const int inside = ((1 - 1E-9) * h * h <= mask[k] ? 1 : 0);
    cells[k] = (0 != inside ? cells[k] / mask[k] : 0);
    mask[k] = inside;
  }
  *lo = ulo;
  return nc;
}


/* masked rank-1 fit a b^T of the cells; returns the relative residual */
static double rot_rank1(const double cells[], const double mask[], int nc, double a[],
  double b[], int niter)
{
  double best = -1, num = 0, den = 0;
  int i, j, it, kstar = 0;
  for (i = 0; i < nc; ++i) {
    double sq = 0;
    for (j = 0; j < nc; ++j) sq += mask[i + nc * j] * cells[i + nc * j] * cells[i + nc * j];
    if (best < sq) {
      best = sq;
      kstar = i;
    }
  }
  for (j = 0; j < nc; ++j) b[j] = cells[kstar + nc * j];
  for (it = 0; it < niter; ++it) {
    for (i = 0; i < nc; ++i) {
      double sa = 0, sb = 0;
      for (j = 0; j < nc; ++j) {
        sa += mask[i + nc * j] * cells[i + nc * j] * b[j];
        sb += mask[i + nc * j] * b[j] * b[j];
      }
      a[i] = (0 < sb ? sa / sb : 0);
    }
    for (j = 0; j < nc; ++j) {
      double sa = 0, sb = 0;
      for (i = 0; i < nc; ++i) {
        sa += mask[i + nc * j] * cells[i + nc * j] * a[i];
        sb += mask[i + nc * j] * a[i] * a[i];
      }
      b[j] = (0 < sb ? sa / sb : 0);
    }
  }
  for (j = 0; j < nc; ++j) {
    for (i = 0; i < nc; ++i) {
      const double e = cells[i + nc * j] - a[i] * b[j];
      num += mask[i + nc * j] * e * e;
      den += mask[i + nc * j] * cells[i + nc * j] * cells[i + nc * j];
    }
  }
  return (0 < den ? num / den : 1);
}


static double rot_objective(double degrees, const void* data)
{
  const rotdata_t *const ctx = (const rotdata_t*)data;
  double ab[2 * BRACKET_CELLMAX], lo;
  const int nc = rot_cells(ctx->data, ctx->n, degrees * BRACKET_PI / 180, ctx->cells,
    ctx->mask, &lo);
  return rot_rank1(ctx->cells, ctx->mask, nc, ab, ab + BRACKET_CELLMAX, 20);
}


/**
 * The frame is found as the angle with the smallest rank-1 residual of the turned cells
 * (coarse scan, then golden-section search); in that frame the factors are 1-D fits to the
 * running sums of the cell averages, alternated as for sep_train. A pixel spans
 * |cos| + |sin| cells along each axis, which is the width an edge is blurred over.
 */
static int rot_train(rot_t* model, const double data[], int n, int units, int integrated,
  const train_t* train, const train_t* refine)
{
  const size_t ncmax = BRACKET_CELLMAX;
  double *const work = (double*)malloc(sizeof(double) * (2 * ncmax * ncmax + 6 * (ncmax + 1)));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    double *const cells = work, *const mask = cells + ncmax * ncmax;
    double *const a = mask + ncmax * ncmax, *const b = a + ncmax + 1, *const f = b + ncmax + 1;
    double *const t = f + ncmax + 1, *const tc = t + ncmax + 1, *const g = tc + ncmax + 1;
    double degrees = 0, fbest = 2, lo = 0, ht, footprint;
    rotdata_t ctx;
    int nc, it, i, j, side;
    ctx.data = data;
    ctx.cells = cells;
    ctx.mask = mask;
    ctx.n = n;
    for (i = 0; i * BRACKET_STEPDEG < 90; ++i) {
      const double value = rot_objective(i * BRACKET_STEPDEG, &ctx);
      if (value < fbest) {
        fbest = value;
        degrees = i * BRACKET_STEPDEG;
      }
    }
    { /* keep the scan's angle unless the search improves on it */
      double refined = degrees;
      if (libxs_gss_min(rot_objective, &ctx, degrees - BRACKET_STEPDEG,
        degrees + BRACKET_STEPDEG, &refined, 40, LIBXS_GSS_EVAL_ENDPOINTS, 0, NULL) < fbest)
      {
        degrees = refined;
      }
    }
    model->angle = degrees * BRACKET_PI / 180;
    nc = rot_cells(data, n, model->angle, cells, mask, &lo);
    ht = 1.0 / nc;
    footprint = fabs(cos(model->angle)) + fabs(sin(model->angle));
    model->lo = lo;
    model->len = (double)nc / n;
    rot_rank1(cells, mask, nc, a, b, 1);
    for (i = 0; i <= nc; ++i) t[i] = ht * i;
    for (i = 0; i < nc; ++i) tc[i] = ht * (i + 0.5);
    for (it = 0; it < BRACKET_ALS; ++it) {
      for (side = 0; side < 2; ++side) {
        double *const p = (0 == side ? a : b), *const q = (0 == side ? b : a);
        model1_t *const factor = (0 == side ? &model->a : &model->b);
        int last = -1;
        for (i = 0; i < nc; ++i) { /* projection, rows without a cell inside take a neighbor */
          double sa = 0, sb = 0;
          for (j = 0; j < nc; ++j) {
            const int k = (0 == side ? i + nc * j : j + nc * i);
            sa += mask[k] * cells[k] * q[j];
            sb += mask[k] * q[j] * q[j];
          }
          if (0 < sb) {
            p[i] = sa / sb;
            for (j = last + 1; j < i; ++j) p[j] = p[i];
            last = i;
          }
        }
        for (j = last + 1; j < nc; ++j) p[j] = (0 <= last ? p[last] : 0);
        if (0 != integrated) {
          f[0] = 0;
          for (i = 0; i < nc; ++i) f[i + 1] = f[i] + ht * p[i];
          factor_fit(factor, t, f, nc + 1, units, 1, ht * footprint, train, refine);
          model1_sample(factor, 0, t, nc + 1, g);
          for (i = 0; i < nc; ++i) p[i] = (g[i + 1] - g[i]) / ht;
        }
        else {
          factor_fit(factor, tc, p, nc, units, 0, ht, train, refine);
          model1_sample(factor, 0, tc, nc, p);
        }
      }
    }
  }
  free(work);
  return result;
}


/* integrated factors evaluate their derivative, so the product is A'(u) B'(v) */
static int rot_eval(const void* model, size_t n, double out[])
{
  const rot_t *const rot = (const rot_t*)model;
  const double c = cos(rot->angle), s = sin(rot->angle);
  size_t i, j;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      const double x = (double)i / (double)(n - 1) - 0.5, y = (double)j / (double)(n - 1) - 0.5;
      out[i + n * j] = model1_value(&rot->a, (c * x + s * y - rot->lo) / rot->len)
        * model1_value(&rot->b, (c * y - s * x - rot->lo) / rot->len);
    }
  }
  return EXIT_SUCCESS;
}


/* mean distance between each edge and the nearest mid-level crossing of y */
static double edge_error(const double y[], size_t n, const target1_t* target)
{
  const double h = 1.0 / (double)(n - 1);
  double result = 0, window = 0.5;
  size_t j;
  int i;
  for (i = 0; i <= target->nedges; ++i) {
    const double lo = (0 == i ? 0.0 : target->edge[i - 1]);
    const double hi = (target->nedges == i ? 1.0 : target->edge[i]);
    window = LIBXS_MIN(window, 0.5 * (hi - lo));
  }
  for (i = 0; i < target->nedges; ++i) {
    const double edge = target->edge[i];
    const double mid = 0.5 * (target->level[i] + target->level[i + 1]);
    double best = window;
    for (j = 0; j + 1 < n; ++j) {
      const double x = (double)j * h;
      if (edge - window <= x && x + h <= edge + window) {
        const double ylo = y[j] - mid, yhi = y[j + 1] - mid;
        if (0 >= ylo * yhi && ylo != yhi) {
          const double cross = x - h * ylo / (yhi - ylo);
          if (fabs(cross - edge) < best) best = fabs(cross - edge);
        }
      }
    }
    result += best;
  }
  return (0 < target->nedges ? result / target->nedges : 0);
}


/* 2-D: root mean square over rows, since the per-axis maximum picks up narrow bands */
static int fprint_field(libxs_fprint_t* info, const double data[], int ndims, size_t n,
  int order)
{
  int result = libxs_fprint(info, LIBXS_DATATYPE_F64, data, 1, &n, NULL, order, 0, 0, 0);
  if (2 == ndims && EXIT_SUCCESS == result) {
    double sq[LIBXS_FPRINT_MAXORDER + 1], mean[LIBXS_FPRINT_MAXORDER + 1];
    libxs_fprint_t row;
    size_t j;
    int k;
    for (k = 0; k <= info->order; ++k) {
      sq[k] = info->l2[k] * info->l2[k];
      mean[k] = info->mean[k];
    }
    for (j = 1; j < n && EXIT_SUCCESS == result; ++j) {
      result = libxs_fprint(&row, LIBXS_DATATYPE_F64, data + n * j, 1, &n, NULL,
        order, 0, 0, 0);
      for (k = 0; k <= info->order && EXIT_SUCCESS == result; ++k) {
        sq[k] += row.l2[k] * row.l2[k];
        mean[k] += row.mean[k];
      }
    }
    for (k = 0; k <= info->order; ++k) {
      info->l2[k] = sqrt(sq[k] / (double)n);
      info->mean[k] = mean[k] / (double)n;
    }
  }
  return result;
}


/**
 * Order-1 norm of a jump J grows as J/sqrt(h) but saturates at J/sqrt(eps) for an edge
 * of width eps: slope 0.5 is a true jump, slope 0 a smooth edge of width sum(J^2)/norm^2.
 */
int report(const study_t* study, const char target[], const char variant[],
  const void* model, eval_fn_t eval, double seconds, double image[])
{
  const size_t nimage = (2 == study->ndims ? study->nimage * study->nimage : study->nimage);
  libxs_fprint_t sweep, fmodel, fclean;
  double err = 0, pos = 0, slope = 0, eps = 0, dist = 0;
  double sx = 0, sy = 0, sxx = 0, sxy = 0, l2 = 0;
  size_t i, n;
  int result = eval(model, study->nimage, image), npoints = 0;
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < nimage; ++i) {
      const double diff = image[i] - study->image[i];
      err += diff * diff;
    }
    err = sqrt(err / (double)nimage);
    pos = edge_error(image + study->nimage * study->rowindex, study->nimage, study->row);
    result = fprint_field(&fmodel, image, study->ndims, study->nimage, 1);
  }
  if (EXIT_SUCCESS == result) {
    result = fprint_field(&fclean, study->image, study->ndims, study->nimage, 1);
  }
  if (EXIT_SUCCESS == result) dist = libxs_fprint_diff(&fmodel, &fclean, NULL);
  for (n = study->nlo; n <= study->nhi && EXIT_SUCCESS == result; n *= 2) {
    result = eval(model, n, study->buffer);
    if (EXIT_SUCCESS == result) {
      result = fprint_field(&sweep, study->buffer, study->ndims, n, 1);
    }
    if (EXIT_SUCCESS == result && 0 < sweep.l2[1]) {
      const double lx = log((double)(n - 1)), ly = log(sweep.l2[1]);
      sx += lx;
      sy += ly;
      sxx += lx * lx;
      sxy += lx * ly;
      l2 = sweep.l2[1];
      ++npoints;
    }
  }
  if (EXIT_SUCCESS == result && 1 < npoints) {
    slope = (npoints * sxy - sx * sy) / (npoints * sxx - sx * sx);
    eps = study->sumjsq / (l2 * l2);
  }
  if (EXIT_SUCCESS == result) {
    printf("%-8s %-17s %10.3e %8.3f %7.3f %9.3e %10.3e %7.2fs\n", target, variant,
      err, pos / study->htrain, slope, eps / study->htrain, dist, seconds);
    fflush(stdout);
  }
  return result;
}


static int run_1d(const train_t* train, const setup_t* setup, unsigned int seed)
{
  static const char *const variants[] = {
    "direct", "direct_wd", "integrated", "integrated_wd", "integrated_merge",
    "integrated_greedy", "integrated_prune"
  };
  const int nvariants = (int)(sizeof(variants) / sizeof(*variants));
  const int ns = setup->samples, nint = (0 != setup->pixel ? ns + 1 : ns);
  const double h = 1.0 / (double)(0 != setup->pixel ? ns : (ns - 1));
  double xd[BRACKET_N1MAX + 1], fd[BRACKET_N1MAX + 1];
  double xi[BRACKET_N1MAX + 1], fi[BRACKET_N1MAX + 1];
  double stack[8 * BRACKET_IMAGE1], buffer[BRACKET_HI1];
  target1_t targets[3];
  study_t study;
  train_t train1 = *train;
  int result = EXIT_SUCCESS, t, k, i;
  targets[0].name = "step";
  targets[0].nedges = 1;
  targets[0].edge[0] = BRACKET_PHASE;
  targets[0].level[0] = 0;
  targets[0].level[1] = 1;
  targets[0].noise = 0;
  targets[1].name = "square";
  targets[1].nedges = BRACKET_CELLS;
  targets[1].noise = 0;
  for (i = 0; i <= BRACKET_CELLS; ++i) {
    if (i < BRACKET_CELLS) targets[1].edge[i] = checker_edge(i);
    targets[1].level[i] = (double)(i & 1);
  }
  targets[2] = targets[0];
  targets[2].name = "noisy";
  targets[2].noise = BRACKET_NOISE;
  train1.lr = BRACKET_LR1;
  memset(&study, 0, sizeof(study));
  study.ndims = 1;
  study.htrain = h;
  study.nimage = BRACKET_IMAGE1;
  study.nlo = BRACKET_LO1;
  study.nhi = BRACKET_HI1;
  study.image = stack;
  study.buffer = buffer;
  for (t = 0; t < 3 && EXIT_SUCCESS == result; ++t) {
    const target1_t *const target = targets + t;
    char name[64];
    if (0 == selected(setup, target->name)) continue;
    study.row = target;
    study.sumjsq = 0;
    for (i = 0; i < target->nedges; ++i) {
      const double jump = target->level[i + 1] - target->level[i];
      study.sumjsq += jump * jump;
    }
    libxs_rng_set_seed(seed + 1000);
    if (0 != setup->pixel) { /* cell averages, and their exact integral at the cell bounds */
      xi[0] = fi[0] = 0;
      for (i = 0; i < ns; ++i) {
        xd[i] = h * ((double)i + 0.5);
        fd[i] = (target1_integral(target, h * (i + 1)) - target1_integral(target, h * i)) / h
          + target->noise * gaussian();
        xi[i + 1] = h * (i + 1);
        fi[i + 1] = fi[i] + h * fd[i];
      }
    }
    else {
      for (i = 0; i < ns; ++i) {
        xd[i] = xi[i] = h * i;
        fd[i] = target1_value(target, xd[i]) + target->noise * gaussian();
      }
      integrate1(fd, (size_t)ns, fi);
    }
    for (i = 0; i < BRACKET_IMAGE1; ++i) {
      stack[i] = target1_value(target, (double)i / (double)(BRACKET_IMAGE1 - 1));
    }
    for (k = 0; k < nvariants && EXIT_SUCCESS == result; ++k) {
      const int integrated = (2 <= k ? 1 : 0);
      libxs_timer_tick_t start;
      model1_t model;
      libxs_rng_set_seed(seed + (unsigned int)t);
      model1_init(&model, setup->units, integrated);
      train1.wd = (1 == k || 3 == k ? train->wd : 0);
      start = libxs_timer_tick();
      if (0 != integrated) model1_train(&model, xi, fi, nint, &train1);
      else model1_train(&model, xd, fd, ns, &train1);
      if (4 <= k) {
        train_t refine = train1;
        refine.nsteps = LIBXS_MAX(train1.nsteps / 10, 1);
        refine.lr = 0.1 * train1.lr;
        model1_merge(&model, h);
        model1_train(&model, xi, fi, nint, &refine);
        if (5 <= k) model1_insert(&model, xi, fi, nint, setup->units, &refine);
        if (6 == k) {
          double avg[BRACKET_N1MAX + 1];
          for (i = 0; i + 1 < nint; ++i) avg[i] = (fi[i + 1] - fi[i]) / (xi[i + 1] - xi[i]);
          model1_prune(&model, xi, fi, nint, noise_sigma(avg, nint - 1));
        }
      }
      result = report(&study, target->name, variants[k], &model, model1_eval,
        libxs_timer_duration(start, libxs_timer_tick()),
        stack + (size_t)(k + 1) * BRACKET_IMAGE1);
    }
    if (EXIT_SUCCESS == result) {
      LIBXS_SNPRINTF(name, sizeof(name), "bracket_1d_%s", target->name);
      result = write_image(name, stack, BRACKET_IMAGE1, (size_t)nvariants + 1, NULL);
    }
  }
  return result;
}


static int run_2d(const train_t* train, const setup_t* setup, unsigned int seed)
{
  const size_t n = BRACKET_N2, nc = n + 1, ni = BRACKET_IMAGE2;
  const size_t nbuffer = (size_t)BRACKET_HI2 * BRACKET_HI2;
  double *const work = (double*)malloc(sizeof(double)
    * (n * n + 3 * nc * nc + 2 * ni * ni + nbuffer));
  int result = (NULL != work ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result && 0 != selected(setup, "checker")) {
    result = run_checker(train, setup, seed, "checker", checker_value, 0, work);
  }
  if (EXIT_SUCCESS == result && 0 != selected(setup, "turned")) {
    result = run_checker(train, setup, seed, "turned", checker_value,
      setup->angle * BRACKET_PI / 180, work);
  }
  if (EXIT_SUCCESS == result && 0 != selected(setup, "disc")) {
    result = run_checker(train, setup, seed, "disc", disc_value, 0, work);
  }
  free(work);
  return result;
}


/**
 * Pixel data: cell averages (exact for the axis-aligned board, supersampled otherwise), their
 * summed-area table at the cell corners, and diagonal steps of h times the average.
 */
static int run_checker(const train_t* train, const setup_t* setup, unsigned int seed,
  const char target[], value_fn_t value, double angle, double work[])
{
  static const char *const variants[] = {
    "mlp", "mlp_int", "tensor", "tensor_int", "sep", "sep_int", "rot", "rot_int",
    "box", "box_int", "box_greedy"
  };
  /* the targets' levels, so every image shares one scale */
  static const double image_range[2] = { -1.0, 1.0 };
  const size_t n = BRACKET_N2, nc = n + 1, ni = BRACKET_IMAGE2;
  double *const data = work, *const area = data + n * n, *const tmp = area + nc * nc;
  double *const diag = tmp + nc * nc, *const image = diag + nc * nc;
  double *const model_image = image + ni * ni;
  double *const buffer = model_image + ni * ni;
  const double yrow = 0.5 * (checker_edge(1) + checker_edge(2));
  const int pixel = setup->pixel;
  const double h = 1.0 / (double)(0 != pixel ? n : (n - 1));
  const double lo = (0 != pixel ? 0.5 * h : 0);
  const int nint = (int)(0 != pixel ? nc : n);
  target1_t row, axis;
  study_t study;
  train_t train2 = *train, train1 = *train, refine;
  int result = EXIT_SUCCESS, k;
  size_t i, j;
  char name[64];
  train2.nsteps = LIBXS_MAX(train->nsteps / 4, 1);
  train2.wd = train1.wd = 0;
  train1.lr = BRACKET_LR1;
  refine = train1;
  refine.nsteps = LIBXS_MAX(train1.nsteps / 10, 1);
  refine.lr = 0.1 * train1.lr;
  axis.name = "axis";
  axis.nedges = BRACKET_CELLS;
  axis.noise = 0;
  for (k = 0; k <= BRACKET_CELLS; ++k) {
    if (k < BRACKET_CELLS) axis.edge[k] = checker_edge(k);
    axis.level[k] = (0 == (k & 1) ? 1.0 : -1.0);
  }
  if (0 != pixel) {
    for (i = 0; i < nc; ++i) tmp[i] = target1_integral(&axis, h * (double)i);
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        double avg = 0;
        if (0 == angle && checker_value == value) {
          avg = (tmp[i + 1] - tmp[i]) * (tmp[j + 1] - tmp[j]) / (h * h);
        }
        else {
          int si, sj;
          for (sj = 0; sj < BRACKET_SUPER; ++sj) {
            for (si = 0; si < BRACKET_SUPER; ++si) {
              avg += value(h * ((double)i + (si + 0.5) / BRACKET_SUPER),
                h * ((double)j + (sj + 0.5) / BRACKET_SUPER), angle);
            }
          }
          avg /= (double)(BRACKET_SUPER * BRACKET_SUPER);
        }
        data[i + n * j] = avg;
      }
    }
    for (j = 0; j < nc; ++j) {
      for (i = 0; i < nc; ++i) {
        const size_t k1 = (i - 1) + nc * (j - 1), d1 = (i - 1) + n * (j - 1);
        area[i + nc * j] = (0 == i || 0 == j ? 0 : area[i - 1 + nc * j]
          + area[i + nc * (j - 1)] - area[k1] + h * h * data[d1]);
        diag[i + nc * j] = (0 == i || 0 == j ? 0 : diag[k1] + h * data[d1]);
      }
    }
  }
  else {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        data[i + n * j] = value(h * (double)i, h * (double)j, angle);
      }
    }
    for (j = 0; j < n; ++j) integrate1(data + n * j, n, tmp + n * j);
    for (i = 0; i < n; ++i) {
      double s = 0;
      area[i] = 0;
      for (j = 1; j < n; ++j) {
        s += 0.5 * h * (tmp[i + n * (j - 1)] + tmp[i + n * j]);
        area[i + n * j] = s;
      }
    }
    integrate_diag(data, n, diag);
  }
  for (j = 0; j < ni; ++j) {
    for (i = 0; i < ni; ++i) {
      image[i + ni * j] = value((double)i / (double)(ni - 1),
        (double)j / (double)(ni - 1), angle);
    }
  }
  checker_row(value, yrow, angle, &row);
  memset(&study, 0, sizeof(study));
  study.ndims = 2;
  study.htrain = h;
  study.nimage = ni;
  study.nlo = BRACKET_LO2;
  study.nhi = BRACKET_HI2;
  study.rowindex = (size_t)(yrow * (double)(ni - 1) + 0.5);
  study.image = image;
  study.buffer = buffer;
  study.row = &row;
  study.sumjsq = 0;
  for (k = 0; k < row.nedges; ++k) {
    const double jump = row.level[k + 1] - row.level[k];
    study.sumjsq += jump * jump;
  }
  LIBXS_SNPRINTF(name, sizeof(name), "bracket_2d_%s", target);
  result = write_image(name, image, ni, ni, image_range);
  for (k = 0; k < 11 && EXIT_SUCCESS == result; ++k) {
    const int integrated = (10 == k ? 1 : (k & 1));
    const libxs_timer_tick_t start = libxs_timer_tick();
    const void* model = NULL;
    eval_fn_t eval = NULL;
    mlp_t mlp;
    tensor_t tensor;
    sep_t sep;
    rot_t rot;
    mlp.param = tensor.param = NULL;
    libxs_rng_set_seed(seed);
    if (2 > k) {
      train2.lr = BRACKET_LRMLP;
      result = mlp_init(&mlp, setup->width, integrated);
      if (EXIT_SUCCESS == result) {
        result = (0 != integrated ? mlp_train(&mlp, diag, nint, 0, h, &train2)
          : mlp_train(&mlp, data, (int)n, lo, h, &train2));
      }
      model = &mlp;
      eval = mlp_eval;
    }
    else if (4 > k) {
      train2.lr = BRACKET_LR2;
      result = tensor_init(&tensor, BRACKET_U2, integrated);
      if (EXIT_SUCCESS == result) {
        result = (0 != integrated ? tensor_train(&tensor, area, nint, 0, h, &train2)
          : tensor_train(&tensor, data, (int)n, lo, h, &train2));
      }
      model = &tensor;
      eval = tensor_eval;
    }
    else if (10 == k) { /* Adam moves each parameter by about lr: refining needs sub-pixel */
      train_t step = train2;
      step.lr = train2.lr = 0.1 * BRACKET_LR2;
      step.nsteps = LIBXS_MAX(train2.nsteps / 10, 1);
      result = box_greedy(&tensor, data, area, (int)n, h, BRACKET_U2, &train2, &step);
      model = &tensor;
      eval = box_eval;
    }
    else if (8 <= k) {
      train2.lr = BRACKET_LR2;
      result = box_init(&tensor, BRACKET_U2, integrated);
      if (EXIT_SUCCESS == result) {
        result = (0 != integrated ? box_train(&tensor, area, nint, 0, h, &train2)
          : box_train(&tensor, data, (int)n, lo, h, &train2));
      }
      model = &tensor;
      eval = box_eval;
    }
    else if (6 <= k) {
      result = rot_train(&rot, data, (int)n, setup->units, integrated, &train1, &refine);
      model = &rot;
      eval = rot_eval;
    }
    else {
      sep.rank = setup->rank;
      sep.integrated = integrated;
      result = (0 != integrated
        ? sep_train(&sep, area, nint, 0, h, setup->units, &train1, &refine)
        : sep_train(&sep, data, (int)n, lo, h, setup->units, &train1, &refine));
      model = &sep;
      eval = sep_eval;
    }
    if (EXIT_SUCCESS == result) {
      result = report(&study, target, variants[k], model, eval,
        libxs_timer_duration(start, libxs_timer_tick()), model_image);
      if (6 <= k && 8 > k) {
        printf("  %s %s angle=%.3f\n", target, variants[k], rot.angle * 180 / BRACKET_PI);
      }
    }
    if (EXIT_SUCCESS == result) { /* metrics are taken; the image shows the target's range */
      for (i = 0; i < ni * ni; ++i) model_image[i] = LIBXS_CLMP(model_image[i], -1.0, 1.0);
      LIBXS_SNPRINTF(name, sizeof(name), "bracket_2d_%s_%s", target, variants[k]);
      result = write_image(name, model_image, ni, ni, image_range);
    }
    free(mlp.param);
    free(tensor.param);
  }
  if (EXIT_SUCCESS == result && 0 != setup->predict) {
    result = bracket_predict(&study, target, data, n, lo, h, pixel, model_image);
  }
  return result;
}
