/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_mem.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE double
#endif


static int check_itrans(ELEM_TYPE* inout, const ELEM_TYPE* ref,
  int m, int n, int ldi, int ldo);


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 64);
  const int n = (2 < argc ? atoi(argv[2]) : m);
  const int ldi = (3 < argc ? LIBXS_MAX(atoi(argv[3]), m) : m);
  const int ldo = (4 < argc ? LIBXS_MAX(atoi(argv[4]), n) : n);
  const size_t size_in = (size_t)ldi * n;
  const size_t size_out = (size_t)ldo * m;
  const size_t size_max = LIBXS_MAX(size_in, size_out);
  const size_t scratch_size = (size_t)m * n;
  ELEM_TYPE *mat = NULL, *ref = NULL, *scratch = NULL;
  int result = EXIT_SUCCESS;
  int i, j;

  libxs_init();

  if (0 >= m || 0 >= n) {
    libxs_itrans(NULL, sizeof(ELEM_TYPE), m, n, ldi, ldo, NULL);
  }
  else {
    mat = (ELEM_TYPE*)malloc(size_max * sizeof(ELEM_TYPE));
    ref = (ELEM_TYPE*)malloc(size_in * sizeof(ELEM_TYPE));
    scratch = (m != n || ldi != ldo)
      ? (ELEM_TYPE*)malloc(scratch_size * sizeof(ELEM_TYPE)) : NULL;

    if (NULL != mat && NULL != ref) {
      /* initialize: A(i,j) stored column-major with leading dim ldi */
      for (j = 0; j < n; ++j) {
        for (i = 0; i < ldi; ++i) {
          const ELEM_TYPE value = (ELEM_TYPE)((size_t)ldi * j + i + 1);
          mat[(size_t)ldi * j + i] = value;
          if (i < m) ref[(size_t)ldi * j + i] = value;
        }
      }

      /* serial itrans */
      libxs_itrans(mat, sizeof(ELEM_TYPE), m, n, ldi, ldo, scratch);
      result = check_itrans(mat, ref, m, n, ldi, ldo);
      if (EXIT_SUCCESS != result) {
        fprintf(stderr, "  (serial itrans, m=%i n=%i ldi=%i ldo=%i)\n",
          m, n, ldi, ldo);
      }

      if (EXIT_SUCCESS == result) {
        /* re-initialize for task variant */
        for (j = 0; j < n; ++j) {
          for (i = 0; i < ldi; ++i) {
            mat[(size_t)ldi * j + i] =
              (ELEM_TYPE)((size_t)ldi * j + i + 1);
          }
        }

        /* parallel itrans_task */
#if defined(_OPENMP)
#       pragma omp parallel default(none) shared(mat, scratch, m, n, ldi, ldo)
        {
          libxs_itrans_task(mat, sizeof(ELEM_TYPE), m, n, ldi, ldo,
            scratch, omp_get_thread_num(), omp_get_num_threads());
        }
#else
        libxs_itrans_task(mat, sizeof(ELEM_TYPE), m, n, ldi, ldo,
          scratch, 0, 1);
#endif
        result = check_itrans(mat, ref, m, n, ldi, ldo);
        if (EXIT_SUCCESS != result) {
          fprintf(stderr, "  (task itrans, m=%i n=%i ldi=%i ldo=%i)\n",
            m, n, ldi, ldo);
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }

    free(scratch);
    free(ref);
    free(mat);
  }

  libxs_finalize();
  return result;
}


static int check_itrans(ELEM_TYPE* inout, const ELEM_TYPE* ref,
  int m, int n, int ldi, int ldo)
{
  int r, c;
  for (c = 0; c < m; ++c) {
    for (r = 0; r < n; ++r) {
      const ELEM_TYPE expected = ref[(size_t)ldi * r + c];
      const ELEM_TYPE actual = inout[(size_t)ldo * c + r];
      if (expected != actual) {
        fprintf(stderr, "ERROR: mismatch at (r=%i,c=%i): expected=%g actual=%g\n",
          r, c, (double)expected, (double)actual);
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}
