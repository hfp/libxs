#include <libxs/libxs_math.h>

#define SETDIFF_PATH_MAX 4096


static void* checked_malloc(size_t size);
static double seconds_now(void);
static void swap_double(double* lhs, double* rhs);
static void shuffle_values(double* values, int n);
static void fill_real(double* avals, double* bvals, int n, double perturb);
static int elementwise_mismatch(const double* avals, const double* bvals, int n, double tol);
static double real_range(const double* avals, int na, const double* bvals, int nb);
static double bisection_tol(const double* avals, int na, const double* bvals, int nb, int target, double hi);
static int write_summary(const char outdir[]);
static int write_landscape(const char outdir[], int n, int steps);
static int write_tolerance(const char outdir[], const char sizes[]);
static int write_scaling(const char outdir[], const char sizes[], int reps);
static int write_complex(const char outdir[]);


int main(int argc, char* argv[])
{
  const char* const outdir = (1 < argc ? argv[1] : "results/setdiff");
  const char* const sizes = (2 < argc ? argv[2] : "128 1024 8192 65536");
  const int reps = (3 < argc ? atoi(argv[3]) : 10);
  const int land_n = (4 < argc ? atoi(argv[4]) : 32);
  const int land_steps = (5 < argc ? atoi(argv[5]) : 40);
  int result = EXIT_SUCCESS;
  srand(1u);
  if (EXIT_SUCCESS == result) result = write_summary(outdir);
  if (EXIT_SUCCESS == result) result = write_landscape(outdir, land_n, land_steps);
  if (EXIT_SUCCESS == result) result = write_tolerance(outdir, sizes);
  if (EXIT_SUCCESS == result) result = write_scaling(outdir, sizes, reps);
  if (EXIT_SUCCESS == result) result = write_complex(outdir);
  return result;
}


static void* checked_malloc(size_t size)
{
  void* result = malloc(size);
  if (NULL == result) {
    fprintf(stderr, "allocation failed\n");
    exit(EXIT_FAILURE);
  }
  return result;
}


static double seconds_now(void)
{
  return (double)clock() / (double)CLOCKS_PER_SEC;
}


static void swap_double(double* lhs, double* rhs)
{
  const double tmp = *lhs;
  *lhs = *rhs;
  *rhs = tmp;
}


static void shuffle_values(double* values, int n)
{
  int i;
  for (i = n - 1; 0 < i; --i) {
    const int j = rand() % (i + 1);
    swap_double(values + i, values + j);
  }
}


static void fill_real(double* avals, double* bvals, int n, double perturb)
{
  int i;
  for (i = 0; i < n; ++i) {
    const double delta = perturb * (double)((i % 7) - 3) / 3.0;
    avals[i] = (double)i;
    bvals[i] = (double)i + delta;
  }
  shuffle_values(bvals, n);
}


static int elementwise_mismatch(const double* avals, const double* bvals, int n, double tol)
{
  int i;
  int result = 0;
  for (i = 0; i < n; ++i) {
    if (tol < fabs(avals[i] - bvals[i])) ++result;
  }
  return result;
}


static double real_range(const double* avals, int na, const double* bvals, int nb)
{
  int i;
  double amin = avals[0];
  double amax = avals[0];
  double bmin = bvals[0];
  double bmax = bvals[0];
  double left;
  double right;
  for (i = 1; i < na; ++i) {
    if (avals[i] < amin) amin = avals[i];
    if (amax < avals[i]) amax = avals[i];
  }
  for (i = 1; i < nb; ++i) {
    if (bvals[i] < bmin) bmin = bvals[i];
    if (bmax < bvals[i]) bmax = bvals[i];
  }
  left = fabs(amin - bmax);
  right = fabs(bmin - amax);
  return left < right ? right : left;
}


static double bisection_tol(const double* avals, int na, const double* bvals, int nb, int target, double hi)
{
  int i;
  double lo = 0.0;
  double result = hi;
  for (i = 0; i < 80; ++i) {
    const double mid = 0.5 * (lo + result);
    const int defect = libxs_setdiff(LIBXS_DATATYPE_F64, avals, na, bvals, nb, mid);
    if (defect <= target) result = mid;
    else lo = mid;
  }
  return result;
}


static int write_summary(const char outdir[])
{
  char path[SETDIFF_PATH_MAX];
  FILE* file;
  double avals[8];
  double bvals[8];
  double dup_a[4] = {1.0, 1.0, 1.0, 2.0};
  double dup_b[4] = {1.0, 2.0, 2.0, 2.0};
  int i;
  int result = EXIT_FAILURE;
  LIBXS_SNPRINTF(path, sizeof(path), "%s/summary.csv", outdir);
  file = fopen(path, "w");
  if (NULL != file) {
    fprintf(file, "case,n,elementwise_mismatch,setdiff,tol\n");
    for (i = 0; i < 8; ++i) {
      avals[i] = (double)i;
      bvals[i] = (double)((i + 3) % 8);
    }
    fprintf(file, "permutation,8,%d,%d,0\n",
      elementwise_mismatch(avals, bvals, 8, 0.0),
      libxs_setdiff(LIBXS_DATATYPE_F64, avals, 8, bvals, 8, 0.0));
    fprintf(file, "duplicates,4,NA,%d,0\n",
      libxs_setdiff(LIBXS_DATATYPE_F64, dup_a, 4, dup_b, 4, 0.0));
    result = ferror(file) ? EXIT_FAILURE : EXIT_SUCCESS;
    fclose(file);
  }
  return result;
}


static int write_landscape(const char outdir[], int n, int steps)
{
  char path[SETDIFF_PATH_MAX];
  FILE* file;
  double* avals = (double*)checked_malloc((size_t)n * sizeof(double));
  double* bvals = (double*)checked_malloc((size_t)n * sizeof(double));
  int k;
  int result = EXIT_FAILURE;
  fill_real(avals, bvals, n, 0.45);
  bvals[n / 2] += 2.0;
  LIBXS_SNPRINTF(path, sizeof(path), "%s/landscape.csv", outdir);
  file = fopen(path, "w");
  if (NULL != file) {
    fprintf(file, "tau,defect\n");
    for (k = 0; k <= steps; ++k) {
      const double tau = 2.5 * (double)k / (double)steps;
      fprintf(file, "%.17g,%d\n", tau,
        libxs_setdiff(LIBXS_DATATYPE_F64, avals, n, bvals, n, tau));
    }
    result = ferror(file) ? EXIT_FAILURE : EXIT_SUCCESS;
    fclose(file);
  }
  free(avals);
  free(bvals);
  return result;
}


static int write_tolerance(const char outdir[], const char sizes[])
{
  char path[SETDIFF_PATH_MAX];
  char* copy = (char*)checked_malloc(strlen(sizes) + 1);
  char* token;
  FILE* file;
  int result = EXIT_SUCCESS;
  strcpy(copy, sizes);
  LIBXS_SNPRINTF(path, sizeof(path), "%s/tolerance.csv", outdir);
  file = fopen(path, "w");
  if (NULL == file) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    fprintf(file, "n,gss_defect,gss_tol,bisect_defect,bisect_tol,abs_delta\n");
    token = strtok(copy, " ");
    while (NULL != token && EXIT_SUCCESS == result) {
      const int n = atoi(token);
      double* avals = (double*)checked_malloc((size_t)n * sizeof(double));
      double* bvals = (double*)checked_malloc((size_t)n * sizeof(double));
      double gss_tol = -1.0;
      double span;
      double bisect;
      int final_defect;
      int gss_defect;
      fill_real(avals, bvals, n, 0.5);
      span = real_range(avals, n, bvals, n);
      final_defect = libxs_setdiff(LIBXS_DATATYPE_F64, avals, n, bvals, n, span);
      gss_defect = libxs_setdiff_min(LIBXS_DATATYPE_F64, avals, n, bvals, n, &gss_tol);
      bisect = bisection_tol(avals, n, bvals, n, final_defect, span);
      fprintf(file, "%d,%d,%.17g,%d,%.17g,%.17g\n",
        n, gss_defect, gss_tol, final_defect, bisect, fabs(gss_tol - bisect));
      free(avals);
      free(bvals);
      token = strtok(NULL, " ");
    }
    if (ferror(file)) result = EXIT_FAILURE;
    fclose(file);
  }
  free(copy);
  return result;
}


static int write_scaling(const char outdir[], const char sizes[], int reps)
{
  char path[SETDIFF_PATH_MAX];
  char* copy = (char*)checked_malloc(strlen(sizes) + 1);
  char* token;
  FILE* file;
  int result = EXIT_SUCCESS;
  strcpy(copy, sizes);
  LIBXS_SNPRINTF(path, sizeof(path), "%s/scaling.csv", outdir);
  file = fopen(path, "w");
  if (NULL == file) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    fprintf(file, "n,reps,fixed_seconds,min_seconds,fixed_defect,min_defect,min_tol\n");
    token = strtok(copy, " ");
    while (NULL != token && EXIT_SUCCESS == result) {
      const int n = atoi(token);
      double* avals = (double*)checked_malloc((size_t)n * sizeof(double));
      double* bvals = (double*)checked_malloc((size_t)n * sizeof(double));
      double tol = -1.0;
      double t0;
      double t1;
      double fixed_seconds;
      int fixed_defect = 0;
      int min_defect = 0;
      int r;
      fill_real(avals, bvals, n, 0.25);
      t0 = seconds_now();
      for (r = 0; r < reps; ++r) {
        fixed_defect += libxs_setdiff(LIBXS_DATATYPE_F64, avals, n, bvals, n, 0.25);
      }
      t1 = seconds_now();
      fixed_seconds = t1 - t0;
      t0 = seconds_now();
      for (r = 0; r < reps; ++r) {
        min_defect += libxs_setdiff_min(LIBXS_DATATYPE_F64, avals, n, bvals, n, &tol);
      }
      t1 = seconds_now();
      fprintf(file, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.17g\n",
        n, reps, fixed_seconds, t1 - t0,
        (double)fixed_defect / (double)reps,
        (double)min_defect / (double)reps, tol);
      free(avals);
      free(bvals);
      token = strtok(NULL, " ");
    }
    if (ferror(file)) result = EXIT_FAILURE;
    fclose(file);
  }
  free(copy);
  return result;
}


static int write_complex(const char outdir[])
{
  char path[SETDIFF_PATH_MAX];
  FILE* file;
  const int n = 64;
  double* avals = (double*)checked_malloc((size_t)(2 * n) * sizeof(double));
  double* bvals = (double*)checked_malloc((size_t)(2 * n) * sizeof(double));
  double tol = -1.0;
  int i;
  int defect;
  int result = EXIT_FAILURE;
  for (i = 0; i < n; ++i) {
    avals[2 * i + 0] = (double)(i % 8);
    avals[2 * i + 1] = (double)(i / 8);
    bvals[2 * i + 0] = avals[2 * i + 0] + 0.01 * (double)((i % 3) - 1);
    bvals[2 * i + 1] = avals[2 * i + 1] + 0.01 * (double)(((i + 1) % 3) - 1);
  }
  defect = libxs_setdiff_min(LIBXS_DATATYPE_C64, avals, n, bvals, n, &tol);
  LIBXS_SNPRINTF(path, sizeof(path), "%s/complex.csv", outdir);
  file = fopen(path, "w");
  if (NULL != file) {
    fprintf(file, "n,defect,tol,fixed_defect_at_tol\n");
    fprintf(file, "%d,%d,%.17g,%d\n", n, defect, tol,
      libxs_setdiff(LIBXS_DATATYPE_C64, avals, n, bvals, n, tol));
    result = ferror(file) ? EXIT_FAILURE : EXIT_SUCCESS;
    fclose(file);
  }
  free(avals);
  free(bvals);
  return result;
}
