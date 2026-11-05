/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef CONVERSE_LM_H
#define CONVERSE_LM_H

#include "converse_core.h"

/**
 * The prediction half: next-token models and their quantification. Serves -E,
 * every -K kind, the expert bank, the retrieval stores and the probes.
 */
int converse_lm_run(converse_run_t* run);

/**
 * The byte model, as the instrument an entry point installs. This half is the
 * one that may name converse_hier_t, so the vtable is built here; a binary that
 * links this half can offer answer rescoring and seam bits, one that does not
 * simply has no judge.
 */
const converse_judge_t* converse_lm_judge(void);

#endif /*CONVERSE_LM_H*/
