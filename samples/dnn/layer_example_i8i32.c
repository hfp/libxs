/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#include <libxs.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

#define CHKERR_LIBXS_DNN(A) if ( A != LIBXS_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxs_dnn_get_error(A) );

#define USE_OVERWRITE

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

LIBXS_INLINE void zero_buf_int8(char* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0;
  }
}

LIBXS_INLINE void zero_buf_uint8(unsigned char* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0;
  }
}

LIBXS_INLINE void zero_buf_int32(int* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0;
  }
}

LIBXS_INLINE void copy_buf_int8(char* src, char* dst, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}


LIBXS_INLINE void copy_buf_uint8(unsigned char* src, unsigned char* dst, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXS_INLINE void init_buf_int8(char* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_int8(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (char)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%3) : (rand()%3)-1));
  }
}

LIBXS_INLINE void init_buf_uint8(unsigned char* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_uint8(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (unsigned char)((initOne != 0) ? 1 : (rand()%3));
  }
}

LIBXS_INLINE void init_buf_int32(int* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_int32(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (int)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7)-3));
  }
}

LIBXS_INLINE void set_zeropad_nchw_uint8(unsigned char* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, unsigned char, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXS_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXS_INLINE void set_zeropad_nchw_int32(int* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, int, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXS_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXS_INLINE void copy_internal_nchw(unsigned char* dst , unsigned char* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, unsigned char, input, src, C, H, W);
  LIBXS_VLA_DECL(4, unsigned char, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXS_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXS_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}


LIBXS_INLINE void naive_conv_fp_int8(naive_conv_t* param, const unsigned char* input, int* output, const char* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXS_VLA_DECL(4,         int,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, const unsigned char,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const char,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                 LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}

LIBXS_INLINE void naive_conv_bp_int8(naive_conv_t* param, int* input, const unsigned char* output, const char* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXS_VLA_DECL(4, const unsigned char,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4,         int,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const char,     filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                  LIBXS_VLA_ACCESS(4,   input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) += (int)
                  (LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                    * LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXS_INLINE void naive_conv_wu_int8(naive_conv_t* param, const unsigned char *input, const unsigned char *output, int *filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXS_VLA_DECL(4, unsigned char, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, unsigned char,  input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4,       int, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (ofm = 0; ofm < nOfm; ++ofm) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (img = 0; img < nImg; ++img) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) += (int)
                 (( LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * ( LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)));
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  unsigned char *naive_input;
  char *naive_filter;
  unsigned char *naive_output_bp, *naive_input_save, *naive_output_save;
  int *naive_output_fp, *naive_input_bp;
  int *naive_libxs_input;
  int *naive_libxs_output;
  int *naive_libxs_filter;
  unsigned char *input_libxs;
  char *filter_libxs;
  int *output_libxs;
  int *dinput_libxs;
  int *naive_filter_wu;
  int *dfilter_libxs;
  unsigned char *doutput_libxs;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  void* scratch;
  size_t scratch_size;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 18;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 1;           /* padding in input, height */
  int padw = 1;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'L';

  const char *const env_check = getenv("CHECK")/*, *const env_winograd = getenv("WINOGRAD")*/;
  const double check = LIBXS_ABS(0 == env_check ? 1 : atof(env_check));
  /*const int algo_winograd = (0 == env_winograd ? 0 : atoi(env_winograd));*/
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();       /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif
  int padding_mode = 0;   /* padding mode */
  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double lpOps = 0.0; /* number of low precision operations */
  int i;

  libxs_dnn_conv_desc conv_desc;
  libxs_dnn_layer* libxs_handle;
  libxs_dnn_tensor* libxs_input;
  libxs_dnn_tensor* libxs_output;
  libxs_dnn_tensor* libxs_filter;
  libxs_dnn_tensor* libxs_dinput;
  libxs_dnn_tensor* libxs_doutput;
  libxs_dnn_tensor* libxs_dfilter;
  libxs_dnn_tensor_datalayout* libxs_layout;
  libxs_dnn_err_t status;

  libxs_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd, 0, sizeof(norms_upd));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type padding_mode\n", argv[0]);
    return 0;
  }
  srand(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIfm       = atoi(argv[i++]);
  if (argc > i) nOfm       = atoi(argv[i++]);
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) padw       = atoi(argv[i++]);
  if (argc > i) padh       = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) format     = *(argv[i++]);
  if (argc > i) padding_mode = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;
  pad_w = padw;
  pad_h = padh;

  if (0 == padding_mode) {
    pad_h_in = 0;
    pad_w_in = 0;
    pad_h_out = 0;
    pad_w_out = 0;
  }
  else {
    /* TODO: change "1" to "0" if "padding_mode = -1" is acknowledged */
    if (1 < padding_mode) pad_w = padding_mode;
    pad_h_in = pad_h;
    pad_w_in = pad_w;
    pad_h_out = pad_h;
    pad_w_out = pad_w;
  }

  /* deriving some values for naive code */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

  /* print some summary */
  printf("##########################################\n");
  printf("#    Setting Up Common    #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXS_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(unsigned char))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(int))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(unsigned char))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(int))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(char))/(1024.0*1024.0) );

  /* allocate data */
  naive_input           = (unsigned char*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  naive_input_save      = (unsigned char*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  naive_output_fp       = (int*  )libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(int),   2097152);
  naive_input_bp        = (int*  )libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(int),   2097152);
  naive_filter_wu       = (int*  )libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(int), 2097152);
  naive_output_save     = (unsigned char*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(unsigned char), 2097152);
  naive_output_bp       = (unsigned char*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(unsigned char), 2097152);
  naive_libxs_input   = (int*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(int),   2097152);
  naive_libxs_output  = (int*  )libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(int),   2097152);
  naive_libxs_filter  = (int*  )libxs_aligned_malloc( nOfm*nIfm*kh*kw*sizeof(int),   2097152);
  naive_filter          = (char*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(char), 2097152);
  input_libxs         = (unsigned char*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  filter_libxs        = (char*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(char), 2097152);
  output_libxs        = (int*) libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(int), 2097152);
  dinput_libxs         = (int*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(int), 2097152);
  dfilter_libxs        = (int*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(int), 2097152);
  doutput_libxs        = (unsigned char*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(unsigned char), 2097152);

  /* initialize data */
  unsigned char  *naive_input_tmp  = (unsigned char*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(unsigned char), 2097152);
  unsigned char  *naive_output_bp_tmp  = (unsigned char*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(unsigned char), 2097152);
  zero_buf_uint8(naive_input, nImg*nIfm*ifhp*ifwp);
  if (padding_mode == 0 ) {
    init_buf_uint8(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
    init_buf_uint8(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    init_buf_uint8(naive_input_tmp,      nImg*nIfm*ifh*ifw, 0, 0);
    init_buf_uint8(naive_output_bp_tmp,  nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
  }
  copy_buf_uint8(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  copy_buf_uint8(naive_output_bp, naive_output_save, nImg*nOfm*ofhp*ofwp);
  init_buf_int8(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  zero_buf_int32(naive_output_fp,      nImg*nOfm*ofhp*ofwp);
  zero_buf_int32(naive_input_bp,      nImg*nIfm*ifhp*ifwp);
  zero_buf_int32(naive_filter_wu,     nOfm*nIfm*kh*kw);
  zero_buf_int32(output_libxs,      nImg*nOfm*ofhp*ofwp);
  zero_buf_int32(dinput_libxs,      nImg*nIfm*ifhp*ifwp);
  zero_buf_int32(naive_libxs_output, nImg*nOfm*ofhp*ofwp);
  zero_buf_int32(naive_libxs_input,  nImg*nIfm*ifhp*ifwp);
  zero_buf_int32(naive_libxs_filter, nOfm*nIfm*kh*kw);

  if (LIBXS_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* run naive convolutions */
    if (type == 'A' || type == 'F') {
      naive_conv_fp_int8(&naive_param, naive_input, naive_output_fp, naive_filter);
    }

    if (type == 'A' || type == 'B') {
      naive_conv_bp_int8(&naive_param, naive_input_bp, naive_output_bp, naive_filter);
    }

    if (type == 'A' || type == 'U') {
      naive_conv_wu_int8(&naive_param, naive_input_save, naive_output_save, naive_filter_wu);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  printf("\n");
  printf("##########################################\n");
  printf("#     Setting Up    (custom-Storage)     #\n");
  printf("##########################################\n");

  /* setup LIBXS handle */
  conv_desc.N = nImg;
  conv_desc.C = nIfm;
  conv_desc.H = ifh;
  conv_desc.W = ifw;
  conv_desc.K = nOfm;
  conv_desc.R = kh;
  conv_desc.S = kw;
  conv_desc.u = stride_h;
  conv_desc.v = stride_w;
  conv_desc.pad_h = pad_h;
  conv_desc.pad_w = pad_w;
  conv_desc.pad_h_in = pad_h_in;
  conv_desc.pad_w_in = pad_w_in;
  conv_desc.pad_h_out = pad_h_out;
  conv_desc.pad_w_out = pad_w_out;
  conv_desc.threads = nThreads;
  conv_desc.algo = LIBXS_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
  conv_desc.filter_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
  conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
#if defined(USE_OVERWRITE)
  conv_desc.options = LIBXS_DNN_CONV_OPTION_OVERWRITE | LIBXS_DNN_CONV_OPTION_ACTIVATION_UNSIGNED;
#else
  conv_desc.options = LIBXS_DNN_CONV_OPTION_ACTIVATION_UNSIGNED;
#endif
  conv_desc.datatype_in = LIBXS_DNN_DATATYPE_I8;
  conv_desc.datatype_out = LIBXS_DNN_DATATYPE_I32;

  libxs_handle = libxs_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXS_DNN( status );

  /* setup LIBXS buffers and filter */
  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_INPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_input  = libxs_dnn_link_tensor( libxs_layout,  input_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_output  = libxs_dnn_link_tensor( libxs_layout,  output_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_FILTER, &status ); CHKERR_LIBXS_DNN( status );
  libxs_filter  = libxs_dnn_link_tensor( libxs_layout,  filter_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_doutput  = libxs_dnn_link_tensor( libxs_layout,  doutput_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dinput = libxs_dnn_link_tensor( libxs_layout,  dinput_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dfilter  = libxs_dnn_link_tensor( libxs_layout,  dfilter_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  /* copy in data to LIBXS format */
  /* we can also use the layout functions and set the data on our
     own external to the library, @TODO, we plan to add an example here */
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_input, (void*)naive_input_save, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_doutput, (void*)naive_output_save, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_output ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_dinput ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_dfilter ) );
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_filter, (void*)naive_filter, LIBXS_DNN_TENSOR_FORMAT_KCRS ) );

  /* bind buffers and filter to handle */
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_input, LIBXS_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dinput, LIBXS_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_output, LIBXS_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_filter, LIBXS_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_doutput, LIBXS_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dfilter, LIBXS_DNN_GRADIENT_FILTER ) );

  /* let's allocate and bind scratch */
  scratch_size = libxs_dnn_get_scratch_size( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, &status );
  CHKERR_LIBXS_DNN( status );
  scratch = (void*)libxs_aligned_malloc( scratch_size, 2097152 );
  CHKERR_LIBXS_DNN( status );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_scratch( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, scratch ) );
  /* set scratch to bogus to make sure that libxs takes care of zeroing internally */
  if ((type == 'A' || type == 'F') && LIBXS_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - FWD (custom-Storage)  #\n");
    printf("##############################################\n");
    /* run LIBXS convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXS_DNN( libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXS_DNN( libxs_dnn_copyout_tensor( libxs_output, (void*)naive_libxs_output, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );

    /* compare */
    libxs_matdiff(LIBXS_DATATYPE_I32, nImg*nOfm*ofhp*ofwp, 1, naive_output_fp, naive_libxs_output, 0, 0, &norms_fwd);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_fwd);
  }

  if ((type == 'A' || type == 'B') && (nIfm > 3) && LIBXS_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - BWD (custom-Storage)  #\n");
    printf("##############################################\n");
    /* run LIBXS convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXS_DNN( libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXS_DNN( libxs_dnn_copyout_tensor( libxs_dinput, (void*)naive_libxs_input, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );

    /* compare */
    libxs_matdiff(LIBXS_DATATYPE_I32, nImg*nIfm*ifhp*ifwp, 1, naive_input_bp, naive_libxs_input, 0, 0, &norms_bwd);
    printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_bwd);
  }

  if ((type == 'A' || type == 'U') && (nIfm > 3) && LIBXS_NEQ(0, check)) {
    printf("##############################################\n");
    printf("#  Check Correctness - UPD (custom-Storage)  #\n");
    printf("##############################################\n");
    /* run LIBXS convolutions */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXS_DNN( libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_UPD, 0, tid ) );
    }
    /* copy out data */
    CHKERR_LIBXS_DNN( libxs_dnn_copyout_tensor( libxs_dfilter, (void*)naive_libxs_filter, LIBXS_DNN_TENSOR_FORMAT_KCRS ) );

    /* compare */
    libxs_matdiff(LIBXS_DATATYPE_I32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxs_filter, 0, 0, &norms_upd);
    printf("L1 reference  : %.25f\n", norms_upd.l1_ref);
    printf("L1 test       : %.25f\n", norms_upd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_upd);
  }

  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXS convolution for performance */
    l_start = libxs_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_FWD, 0, tid );
      }
    }
    l_end = libxs_timer_tick();
    l_total = libxs_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
      ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
      norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
  }

  if ((type == 'A' || type == 'B') && (nIfm > 3)) {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXS convolution for performance */
    l_start = libxs_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_BWD, 0, tid );
      }
    }
    l_end = libxs_timer_tick();
    l_total = libxs_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("bp time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
  }

   if ((type == 'A' || type == 'U') && (nIfm > 3)) {
    printf("##########################################\n");
    printf("#   Performance - UPD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXS convolution for performance */
    l_start = libxs_timer_tick();
    for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_UPD, 0, tid );
      }
    }
    l_end = libxs_timer_tick();
    l_total = libxs_timer_duration(l_start, l_end);
    lpOps = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

    printf("GOP  = %.5g\n", lpOps*1e-9/(double)iters);
    printf("wu time = %.5g\n", ((double)(l_total/iters)));
    printf("GOPS  = %.5g\n", (lpOps*1e-9)/l_total);

    printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (lpOps*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
  }


  /* clean-up */
  CHKERR_LIBXS_DNN( libxs_dnn_release_scratch( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL ) );
  libxs_free(scratch);
  CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_input ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_output ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_filter ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_conv_layer( libxs_handle ) );

  /* deallocate data */
  libxs_free(naive_input);
  libxs_free(naive_input_bp);
  libxs_free(naive_output_fp);
  libxs_free(naive_output_bp);
  libxs_free(naive_libxs_output);
  libxs_free(naive_libxs_input);
  libxs_free(naive_filter);
  libxs_free(input_libxs);
  libxs_free(output_libxs);
  libxs_free(filter_libxs);
  libxs_free(dinput_libxs);
  libxs_free(doutput_libxs);
  libxs_free(dfilter_libxs);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXS_ABS(0 == env_check_scale ? 100.0 : atof(env_check_scale));
    if (LIBXS_NEQ(0, check) && check < 100.0 * check_scale * diff.normf_rel) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

