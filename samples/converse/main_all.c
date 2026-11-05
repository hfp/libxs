/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "converse_qa.h"
#include "converse_lm.h"


/**
 * Entry point: both halves, for reproduction commands predating the split.
 *
 * Linking both is also what makes the byte model available as a judge to the
 * grounded half, which is the one capability the split takes away from
 * converse-qa: with this binary CONVERSE_HIER_RESCORE still rescores answers and
 * the recombination probe still reports its bpc columns.
 */
int main(int argc, char* argv[])
{
  converse_run_t run;
  const int role = converse_role_of(argc, argv);
  int result;
  libxs_init();
  converse_judge_install(converse_lm_judge());
  result = converse_setup(argc, argv, CONVERSE_ROLE_ALL, &run);
  if (EXIT_SUCCESS == result && 0 != run.pending) {
    result = (CONVERSE_ROLE_LM == role) ? converse_lm_run(&run)
      : converse_qa_run(&run);
  }
  converse_release(&run);
  libxs_finalize();
  return result;
}
