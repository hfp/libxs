/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PREDICT_ARGS_H
#define PREDICT_ARGS_H

/**
 * Command line convention shared by the prediction samples.  A token is either
 * a bare number, which fills the next positional slot, or a keyword, which may
 * be written anywhere.  A token that is neither is refused.
 *
 * The samples used to look for keywords only past a fixed position, so a
 * keyword written where a number was expected became atof("xgb") == 0: the run
 * trained on two rows, reported an accuracy for them, and exited successfully.
 * Refusing the token is the point of this file, and letting keywords float is
 * what makes refusing it safe rather than merely stricter.
 */

#include <libxs/libxs_predict.h>


/** Non-zero if the whole token is a number, and therefore a positional. */
LIBXS_INLINE int predict_isnum(const char* arg)
{
  int result = 0;
  if (NULL != arg && '\0' != *arg) {
    char* end = NULL;
    strtod(arg, &end);
    result = (NULL != end && '\0' == *end) ? 1 : 0;
  }
  return result;
}


/**
 * Non-zero if the token is exactly this keyword.  Comparing the whole word
 * rather than its first letter is what keeps a file named `results.csv` from
 * being read as a request for a random forest.
 */
LIBXS_INLINE int predict_iskey(const char* arg, const char* name)
{
  return (NULL != arg && NULL != name && 0 == strcmp(arg, name)) ? 1 : 0;
}


/**
 * Keyword carrying an optional value, written as `name` or `name<number>`.
 * Non-zero when the token is this keyword, yielding either the value written
 * or the supplied default.
 */
LIBXS_INLINE int predict_keyval(const char* arg, const char* name,
  double dflt, double* value)
{
  const size_t n = (NULL != name) ? strlen(name) : 0;
  int result = 0;
  if (NULL != arg && 0 < n && NULL != value && 0 == strncmp(arg, name, n)) {
    const char* rest = arg + n;
    if ('\0' == *rest) {
      *value = dflt;
      result = 1;
    }
    else if (0 != predict_isnum(rest)) {
      *value = atof(rest);
      result = 1;
    }
  }
  return result;
}

#endif /*PREDICT_ARGS_H*/
