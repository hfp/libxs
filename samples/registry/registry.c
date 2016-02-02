#include <libxs.h>
#include <libxs_timer.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This (micro-)benchmark optionally takes a number of dispatches to be performed.
 * The program measures the duration needed to figure out whether a requested matrix
 * multiplication is available or not. The measured duration excludes the time taken
 * to actually generate the code during the first dispatch.
 */
int main(int argc, char* argv[])
{
  const int size = LIBXS_DEFAULT(1 << 27, 1 < argc ? atoi(argv[1]) : 0);
  unsigned long long start;
  double dcall, ddisp;
  int i;

  fprintf(stdout, "Dispatching %i calls %s internal synchronization...\n", size,
#if 0 != LIBXS_SYNC
    "with");
#else
    "without");
#endif
#if 0 != LIBXS_JIT
  { const char *const jit = getenv("LIBXS_JIT");
    if (0 != jit && '0' == *jit) {
      fprintf(stderr, "\tWarning: JIT support has been disabled at runtime!\n");
    }
  }
#else
  fprintf(stderr, "\tWarning: JIT support has been disabled at build time!\n");
#endif

  /* first invocation may initialize some internals (libxs_init),
   * or actually generate code (code gen. time is out of scope)
   */
  libxs_dmmdispatch(LIBXS_AVG_M, LIBXS_AVG_N, LIBXS_AVG_K,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);

  /* run non-inline function to measure call overhead of an "empty" function */
  start = libxs_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < size; ++i) {
    libxs_init(); /* subsequent calls are not doing any work */
  }
  dcall = libxs_timer_duration(start, libxs_timer_tick());

  start = libxs_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < size; ++i) {
    libxs_dmmdispatch(LIBXS_AVG_M, LIBXS_AVG_N, LIBXS_AVG_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
  }
  ddisp = libxs_timer_duration(start, libxs_timer_tick());

  if (0 < dcall && 0 < ddisp) {
    fprintf(stdout, "\tdispatch calls/s: %.0f Hz\n", size / ddisp);
    fprintf(stdout, "\tempty calls/s: %.0f Hz\n", size / dcall);
    fprintf(stdout, "\toverhead: %.0fx\n", ddisp / dcall);
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}

