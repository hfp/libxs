/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#ifndef LIBXS_PREFETCH_H
#define LIBXS_PREFETCH_H

#include "libxs.h"

#if (0 != LIBXS_PREFETCH)
# define LIBXS_PREFETCH_DECL(DECL) DECL;
# if 0 != ((LIBXS_PREFETCH) & 2) || 0 != ((LIBXS_PREFETCH) & 4)
#   define LIBXS_PREFETCH_A(EXPR) (EXPR)
# else
#   define LIBXS_PREFETCH_A(EXPR) 0
# endif
# if 0 != ((LIBXS_PREFETCH) & 8)
#   define LIBXS_PREFETCH_B(EXPR) (EXPR)
# else
#   define LIBXS_PREFETCH_B(EXPR) 0
# endif
# if 0
#   define LIBXS_PREFETCH_C(EXPR) (EXPR)
# else
#   define LIBXS_PREFETCH_C(EXPR) 0
# endif
# define LIBXS_PREFETCH_ANEXT(XARGS, NEXT) , LIBXS_PREFETCH_A(0 == (XARGS) ? (NEXT) : (0 != (XARGS)->pa ? (XARGS)->pa : (NEXT)))
# define LIBXS_PREFETCH_BNEXT(XARGS, NEXT) , LIBXS_PREFETCH_B(0 == (XARGS) ? (NEXT) : (0 != (XARGS)->pb ? (XARGS)->pb : (NEXT)))
# define LIBXS_PREFETCH_CNEXT(XARGS, NEXT) , LIBXS_PREFETCH_C(0 == (XARGS) ? (NEXT) : (0 != (XARGS)->pc ? (XARGS)->pc : (NEXT)))
#else
# define LIBXS_PREFETCH_DECL(DECL)
# define LIBXS_PREFETCH_A(EXPR) 0
# define LIBXS_PREFETCH_B(EXPR) 0
# define LIBXS_PREFETCH_C(EXPR) 0
# define LIBXS_PREFETCH_ANEXT(XARGS, NEXT)
# define LIBXS_PREFETCH_BNEXT(XARGS, NEXT)
# define LIBXS_PREFETCH_CNEXT(XARGS, NEXT)
#endif

#endif /*LIBXS_PREFETCH_H*/
