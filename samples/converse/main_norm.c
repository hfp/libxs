/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "converse_norm.h"


/** Entry point: the lexicon as a codebook, and what decodes to what. */
int main(int argc, char* argv[])
{
  converse_run_t run;
  int result = converse_setup(argc, argv, CONVERSE_ROLE_NORM, &run);
  if (EXIT_SUCCESS == result && 0 != run.pending) result = converse_norm_run(&run);
  converse_release(&run);
  return result;
}
