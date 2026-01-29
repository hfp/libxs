/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MALLOC_H
#define LIBXS_MALLOC_H

#include "libxs_macros.h"


/**
 * Initialize the pool by drawing from the given storage a number of chunks of the given size.
 * If the capacity of the pool is num, the storage must be at least num x size.
 * The same num-counter must be used for pmalloc/pfree when referring to the same pool.
 */
LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Allocate from the given pool by using the original num-counter (libxs_pmalloc_init). */
LIBXS_API void* libxs_pmalloc(void* pool[], size_t* i);
/** Bring pointer back into the pool by using original num-counter (libxs_pmalloc_init). */
LIBXS_API void libxs_pfree(const void* pointer, void* pool[], size_t* i);

#endif /*LIBXS_MALLOC_H*/
