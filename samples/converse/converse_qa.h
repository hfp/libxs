/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef CONVERSE_QA_H
#define CONVERSE_QA_H

#include "converse_core.h"

/**
 * The grounded half: answering from the corpus, attribution, calibrated refusal
 * and grounded recombination. Serves interactive answering, -e and -c.
 *
 * Nothing here names a prediction model. The byte model reaches this half only
 * as an installed converse_judge_t, which is why the answer path links without
 * it: rescoring a candidate list is a measurement, and with no judge installed
 * the selection order stands unchanged.
 */
int converse_qa_run(converse_run_t* run);

#endif /*CONVERSE_QA_H*/
