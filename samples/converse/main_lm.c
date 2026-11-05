/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "converse_lm.h"


/** Entry point: next-token prediction and its quantification. */
int main(int argc, char* argv[])
{
  converse_run_t run;
  int result;
  libxs_init();
  converse_judge_install(converse_lm_judge());
  result = converse_setup(argc, argv, CONVERSE_ROLE_LM, &run);
  if (EXIT_SUCCESS == result && 0 != run.pending) result = converse_lm_run(&run);
  converse_release(&run);
  libxs_finalize();
  return result;
}
