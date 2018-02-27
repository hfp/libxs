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
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar,
 * Ankush Mandal (Intel Corp.)
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
/*#define USE_FUSED_BATCH_STATS*/
/*#define USE_FUSED_MAX_STATS */
#define FP64_BN_STATS
/*#define USE_FUSED_RELU_BWD*/
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

LIBXS_INLINE void zero_buf_int16(short* buf, long size) {
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

LIBXS_INLINE void copy_buf_int16(short* src, short* dst, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXS_INLINE void zero_buf_f32(float* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0;
  }
}

LIBXS_INLINE void init_buf_int16(short* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_int16(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (short)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXS_INLINE void init_buf_int32(int* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf_int32(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (int)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXS_INLINE void set_zeropad_nchw_int16(short* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, short, input, nchw, C, H, W);
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

LIBXS_INLINE void copy_internal_nchw(short* dst , short* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, short, input, src, C, H, W);
  LIBXS_VLA_DECL(4, short, new_input, dst, C, H+2*pad_h, W+2*pad_w);
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


LIBXS_INLINE void naive_conv_fp_int16(naive_conv_t* param, const short* input, float* output, const short* filter)
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

  LIBXS_VLA_DECL(4,       float,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);


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
                LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                   (1.0 *  LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                *  (1.0 *  LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXS_INLINE void naive_conv_bp_int16(naive_conv_t* param, float* input, const short* output, const short* filter)
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

  LIBXS_VLA_DECL(4, const short,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4,       float,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);

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
                  LIBXS_VLA_ACCESS(4,   input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) += (float)
                  ((1.0*LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp))
                  *(1.0*LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw)));
              }
            }
          }
        }
      }
    }
  }
}

LIBXS_INLINE void naive_conv_wu_int16(naive_conv_t* param, const short* input, const short* output, float* filter)
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

  LIBXS_VLA_DECL(4, const short, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, const short,  input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4,       float, filter_t, filter, nIfm, kh, kw);

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
                LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) += (float)
                 ((1.0*LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * (1.0*LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)));
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
  short *naive_input, *naive_filter;
  short *naive_output_bp, *naive_input_save, *naive_output_save;
  float *naive_output_fp, *naive_input_bp, *naive_filter_wu;
  float *naive_libxs_input, *naive_libxs_output, *naive_libxs_filter;
  short *input_libxs, *filter_libxs;
  float *output_libxs, *dinput_libxs;
  float *dfilter_libxs;
  short *doutput_libxs;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  void* scratch;
  size_t scratch_size;
#ifdef FP32_BN_STATS
  float *batchstats_libxs;
#endif
#ifdef FP64_BN_STATS
  double *batchstats_libxs;
#endif
#ifdef USE_FUSED_MAX_STATS
  float *maxstats_libxs_fwd;
  float *maxstats_libxs_bwd;
  float *maxstats_libxs_upd;
#endif

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
  libxs_dnn_tensor* libxs_batchstats;
  libxs_dnn_tensor* libxs_maxstats_fwd;
  libxs_dnn_tensor* libxs_maxstats_bwd;
  libxs_dnn_tensor* libxs_maxstats_upd;
  libxs_dnn_tensor_datalayout* libxs_layout;
  libxs_dnn_err_t status;

  libxs_matdiff_info norms_fwd, norms_bwd, norms_upd, diff, norms_batchstats;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd, 0, sizeof(norms_upd));
  memset(&norms_batchstats, 0, sizeof(norms_batchstats));
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

  if (type != 'A' && type != 'F' && type != 'B'&& type != 'U') {
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
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(short))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(int))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  naive_input           = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  naive_input_save      = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  naive_output_fp       = (float*  )libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float),   2097152);
  naive_input_bp        = (float*  )libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float),   2097152);
  naive_filter_wu       = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_output_bp       = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  naive_output_save     = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  naive_libxs_input   = (float*  )libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float),   2097152);
  naive_libxs_output  = (float*  )libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float),   2097152);
  naive_libxs_filter  = (float*  )libxs_aligned_malloc( nOfm*nIfm*kh*kw*sizeof(float),   2097152);
  naive_filter          = (short*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  input_libxs         = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  filter_libxs        = (short*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  output_libxs        = (float*) libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_libxs        = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dfilter_libxs       = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  doutput_libxs       = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
#ifdef FP32_BN_STATS
  batchstats_libxs    = (float*)libxs_aligned_malloc( 2*nImg*nOfm*        sizeof(float), 2097152);
#endif
#ifdef FP64_BN_STATS
  batchstats_libxs    = (double*)libxs_aligned_malloc( 2*nImg*nOfm*        sizeof(double), 2097152);
#endif
#ifdef USE_FUSED_MAX_STATS
  maxstats_libxs_fwd    = (float*)libxs_aligned_malloc(nImg*16*sizeof(float), 2097152);
  maxstats_libxs_bwd    = (float*)libxs_aligned_malloc(nImg*16*sizeof(float), 2097152);
  maxstats_libxs_upd    = (float*)libxs_aligned_malloc(nImg*16*sizeof(float), 2097152);
#endif

  /* initialize data */
  short  *naive_input_tmp  = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  short  *naive_output_bp_tmp  = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  zero_buf_int16(naive_input, nImg*nIfm*ifhp*ifwp);
  if (padding_mode == 0 ) {
    init_buf_int16(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
    init_buf_int16(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    init_buf_int16(naive_input_tmp,      nImg*nIfm*ifh*ifw, 0, 0);
    init_buf_int16(naive_output_bp_tmp,  nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
  }
  copy_buf_int16(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  copy_buf_int16(naive_output_bp, naive_output_save, nImg*nOfm*ofhp*ofwp);
  init_buf_int16(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  zero_buf_f32(naive_output_fp,      nImg*nOfm*ofhp*ofwp);
  zero_buf_f32(naive_input_bp,      nImg*nIfm*ifhp*ifwp);
  zero_buf_f32(naive_filter_wu,     nOfm*nIfm*kh*kw);
  zero_buf_f32(output_libxs,      nImg*nOfm*ofhp*ofwp);
  zero_buf_f32(dinput_libxs,      nImg*nIfm*ifhp*ifwp);
  zero_buf_f32(naive_libxs_output, nImg*nOfm*ofhp*ofwp);
  zero_buf_f32(naive_libxs_input,  nImg*nIfm*ifhp*ifwp);
  zero_buf_f32(naive_libxs_filter, nOfm*nIfm*kh*kw);

  if (LIBXS_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* run naive convolutions */
    if (type == 'A' || type == 'F') {
      naive_conv_fp_int16(&naive_param, naive_input, naive_output_fp, naive_filter);
    }

    if (type == 'A' || type == 'B') {
      naive_conv_bp_int16(&naive_param, naive_input_bp, naive_output_bp, naive_filter);
    }

    if (type == 'A' || type == 'U') {
      naive_conv_wu_int16(&naive_param, naive_input_save, naive_output_save, naive_filter_wu);
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
#if defined(USE_FUSED_BIAS)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BIAS;
#elif defined(USE_FUSED_RELU)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_RELU;
#elif defined(USE_FUSED_BIAS_RELU)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BIAS_RELU;
#elif (defined(USE_FUSED_BATCH_STATS) && defined(USE_FUSED_MAX_STATS))
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS_AND_MAX;
#elif (defined(USE_FUSED_RELU_BWD) && defined(USE_FUSED_MAX_STATS))
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_RELU_BWD_AND_MAX;
#elif defined(USE_FUSED_BATCH_STATCH_RELU_BWD)
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD;
#elif defined(USE_FUSED_BATCH_STATCH_RELU_BWD_AND_MAX)
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD_AND_MAX;
#elif defined(USE_FUSED_BATCH_STATS)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS;
#elif defined(USE_FUSED_MAX_STATS)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_MAX_STATS;
#elif defined(USE_FUSED_RELU_BWD)
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_RELU_BWD;
#else
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
#endif
#if defined(USE_OVERWRITE)
  conv_desc.options = LIBXS_DNN_CONV_OPTION_OVERWRITE;
#else
  conv_desc.options = LIBXS_DNN_CONV_OPTION_NONE;
#endif
  conv_desc.datatype_in = LIBXS_DNN_DATATYPE_I16;
  conv_desc.datatype_out = LIBXS_DNN_DATATYPE_F32;

  libxs_handle = libxs_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXS_DNN( status );

  /* setup LIBXS buffers and filter */
  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_INPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_input  = libxs_dnn_link_tensor( libxs_layout,  input_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_output  = libxs_dnn_link_tensor( libxs_layout,  output_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dinput = libxs_dnn_link_tensor( libxs_layout,  dinput_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
  libxs_doutput  = libxs_dnn_link_tensor( libxs_layout,  doutput_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXS_DNN( status );
  libxs_filter  = libxs_dnn_link_tensor( libxs_layout,  filter_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dfilter  = libxs_dnn_link_tensor( libxs_layout,  dfilter_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_BATCH_STATS, &status ); CHKERR_LIBXS_DNN( status );
  libxs_batchstats  = libxs_dnn_link_tensor( libxs_layout, batchstats_libxs, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

#ifdef USE_FUSED_MAX_STATS
  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_MAX_STATS_FWD, &status ); CHKERR_LIBXS_DNN( status );
  libxs_maxstats_fwd  = libxs_dnn_link_tensor( libxs_layout, maxstats_libxs_fwd, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_MAX_STATS_BWD, &status ); CHKERR_LIBXS_DNN( status );
  libxs_maxstats_bwd  = libxs_dnn_link_tensor( libxs_layout, maxstats_libxs_bwd, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );

  libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_MAX_STATS_UPD, &status ); CHKERR_LIBXS_DNN( status );
  libxs_maxstats_upd  = libxs_dnn_link_tensor( libxs_layout, maxstats_libxs_upd, &status ); CHKERR_LIBXS_DNN( status );
  libxs_dnn_destroy_tensor_datalayout( libxs_layout );
#endif

  /* copy in data to LIBXS format */
  /* we can also use the layout functions and set the data on our
     own external to the library, @TODO, we plan to add an example here */
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_input, (void*)naive_input_save, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_doutput, (void*)naive_output_save, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_output ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_dinput ) );
  CHKERR_LIBXS_DNN( libxs_dnn_zero_tensor( libxs_dfilter ) );
  CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_filter, (void*)naive_filter, LIBXS_DNN_TENSOR_FORMAT_KCRS ) );
#ifdef FP32_BN_STATS
    zero_buf_f32(batchstats_libxs, 2*nImg*nOfm);
#endif
#ifdef FP64_BN_STATS
    zero_buf_f32((float *) batchstats_libxs, 4*nImg*nOfm);
#endif

  /* bind buffers and filter to handle */
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_input, LIBXS_DNN_REGULAR_INPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dinput, LIBXS_DNN_GRADIENT_INPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_output, LIBXS_DNN_REGULAR_OUTPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_doutput, LIBXS_DNN_GRADIENT_OUTPUT ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_filter, LIBXS_DNN_REGULAR_FILTER ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dfilter, LIBXS_DNN_GRADIENT_FILTER ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_batchstats, LIBXS_DNN_BATCH_STATS ) );

#ifdef USE_FUSED_MAX_STATS
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_maxstats_fwd, LIBXS_DNN_MAX_STATS_FWD ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_maxstats_bwd, LIBXS_DNN_MAX_STATS_BWD ) );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_maxstats_upd, LIBXS_DNN_MAX_STATS_UPD ) );
#endif

  /* let's allocate and bind scratch */
  scratch_size = libxs_dnn_get_scratch_size( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, &status );
  CHKERR_LIBXS_DNN( status );
  scratch = (void*)libxs_aligned_malloc( scratch_size, 2097152 );
  CHKERR_LIBXS_DNN( status );
  CHKERR_LIBXS_DNN( libxs_dnn_bind_scratch( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, scratch ) );
  /* set scratch to bogus to make sure that libxs takes care of zeroing internally */
  //init_buf_int16( (short*)scratch, scratch_size/2, 0, 0 );

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
    libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_fp, naive_libxs_output, 0, 0, &norms_fwd);
    printf("L1 reference  : %.25f\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25f\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_fwd);

#ifdef USE_FUSED_MAX_STATS
    {
      int img_i = 0;
      int ch_i = 0;
      int pxl_i = 0;
      float max_naive = 0.0;
      float max_libxs = 0.0;
      LIBXS_VLA_DECL(3, float, val_naive, naive_output_fp, nOfm, ofhp*ofwp);
      for ( img_i = 0; img_i < nImg; ++img_i ) {
        for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
          for ( pxl_i = 0; pxl_i < ofhp*ofwp; ++pxl_i ) {
            max_naive = LIBXS_MAX( max_naive , fabs(val_naive[img_i][ch_i][pxl_i]) );
          }
        }
      }
      for ( img_i = 0; img_i < nImg; ++img_i ) {
        for ( ch_i = 0; ch_i < 16; ++ch_i ) {
          max_libxs = LIBXS_MAX( max_libxs, maxstats_libxs_fwd[img_i*16+ch_i]);
        }
      }
      printf("\nABSOLUTE MAX VALUES FWD:\n");
      printf("Referen. max abs FWD value: %.25f\n", max_naive);
      printf("LIBXS  max abs FWD value: %.25f\n", max_libxs);
      printf("L2 abs.error  : %.24f\n\n", max_naive-max_libxs);
    }
#endif

#if defined(USE_FUSED_BATCH_STATS)
    {
      float *ch_sum, *ch_sum_fuse;
      float *ch_sum2, *ch_sum2_fuse;
      int img_i = 0;
      int ch_i = 0;
      int ch_j = 0;
      int pxl_i = 0;
#ifdef FP32_BN_STATS
      LIBXS_VLA_DECL(4, float, sum_fuse,  batchstats_libxs, nOfm/16, nImg, 16);
#endif
#ifdef FP64_BN_STATS
      LIBXS_VLA_DECL(4, double, sum_fuse,  batchstats_libxs, nOfm/16, nImg, 16);
#endif
      LIBXS_VLA_DECL(3, float, sum_naive, naive_output_fp,       nOfm, ofhp*ofwp);

      ch_sum       = (float*) malloc(nOfm*sizeof(float));
      ch_sum_fuse  = (float*) malloc(nOfm*sizeof(float));
      ch_sum2      = (float*) malloc(nOfm*sizeof(float));
      ch_sum2_fuse = (float*) malloc(nOfm*sizeof(float));

      for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
        ch_sum_fuse[ch_i] = 0.0f;
        ch_sum2_fuse[ch_i] = 0.0f;
        ch_sum[ch_i] = 0.0f;
        ch_sum2[ch_i] = 0.0f;
      }
      for ( ch_i = 0; ch_i < nOfm/16; ++ch_i ) {
        for ( ch_j = 0; ch_j < 16; ++ch_j ) {
          for ( img_i = 0; img_i < nImg; ++img_i ) {
#ifdef FP32_BN_STATS
            ch_sum_fuse[(ch_i*16) + ch_j]  += sum_fuse[0][ch_i][img_i][ch_j];
            ch_sum2_fuse[(ch_i*16) + ch_j] += sum_fuse[1][ch_i][img_i][ch_j];
#endif
#ifdef FP64_BN_STATS
            double acc1, acc2;
            acc1 = (double) ch_sum_fuse[(ch_i*16) + ch_j];
            acc1 += (double) sum_fuse[0][ch_i][img_i][ch_j];
            acc2 = (double) ch_sum2_fuse[(ch_i*16) + ch_j];
            acc2 += (double) sum_fuse[1][ch_i][img_i][ch_j];
            ch_sum_fuse[(ch_i*16) + ch_j] = (float) acc1;
            ch_sum2_fuse[(ch_i*16) + ch_j]= (float) acc2;
#endif
          }
        }
      }

      for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
        double dsum = 0.0;
        double dsum2 = 0.0;
        for ( pxl_i = 0; pxl_i < ofhp*ofwp; ++pxl_i ) {
          for ( img_i = 0; img_i < nImg; ++img_i ) {
            dsum  +=  sum_naive[img_i][ch_i][pxl_i];
            dsum2 +=  (sum_naive[img_i][ch_i][pxl_i]*sum_naive[img_i][ch_i][pxl_i]);
          }
        }
        ch_sum[ch_i]  = (float) dsum;
        ch_sum2[ch_i] = (float) dsum2;
      }

      libxs_matdiff(LIBXS_DATATYPE_F32, nOfm, 1, ch_sum, ch_sum_fuse, 0, 0, &norms_batchstats);
      printf("Channel Sum:\n");
      printf("L1 reference  : %.25f\n", norms_batchstats.l1_ref);
      printf("L1 test       : %.25f\n", norms_batchstats.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_batchstats.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_batchstats.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_batchstats.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_batchstats.linf_rel);
      printf("Check-norm    : %.24f\n", norms_batchstats.normf_rel);

      libxs_matdiff(LIBXS_DATATYPE_F32, nOfm, 1, ch_sum2, ch_sum2_fuse, 0, 0, &norms_batchstats);
      printf("\nChannel Sum2:\n");
      printf("L1 reference  : %.25f\n", norms_batchstats.l1_ref);
      printf("L1 test       : %.25f\n", norms_batchstats.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_batchstats.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_batchstats.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_batchstats.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_batchstats.linf_rel);
      printf("Check-norm    : %.24f\n", norms_batchstats.normf_rel);

      free(ch_sum);
      free(ch_sum2);
      free(ch_sum_fuse);
      free(ch_sum2_fuse);
    }
#endif
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
    libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_bp, naive_libxs_input, 0, 0, &norms_bwd);
    printf("L1 reference  : %.25f\n", norms_bwd.l1_ref);
    printf("L1 test       : %.25f\n", norms_bwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_bwd);

#ifdef USE_FUSED_MAX_STATS
    {
      int img_i = 0;
      int ch_i = 0;
      int pxl_i = 0;
      float max_naive = 0.0;
      float max_libxs = 0.0;
      LIBXS_VLA_DECL(3, float, val_naive, naive_input_bp, nIfm, ifhp*ifwp);
      for ( img_i = 0; img_i < nImg; ++img_i ) {
        for ( ch_i = 0; ch_i < nIfm; ++ch_i ) {
          for ( pxl_i = 0; pxl_i < ifhp*ifwp; ++pxl_i ) {
            max_naive = LIBXS_MAX( max_naive , fabs(val_naive[img_i][ch_i][pxl_i]) );
          }
        }
      }
      for ( img_i = 0; img_i < nImg; ++img_i ) {
        for ( ch_i = 0; ch_i < 16; ++ch_i ) {
          max_libxs = LIBXS_MAX( max_libxs, maxstats_libxs_bwd[img_i*16+ch_i]);
        }
      }
      printf("\nABSOLUTE MAX VALUES BWD:\n");
      printf("Referen. max abs BWD value: %.25f\n", max_naive);
      printf("LIBXS  max abs BWD value: %.25f\n", max_libxs);
      printf("L2 abs.error  : %.24f\n\n", max_naive-max_libxs);
    }
#endif
  }

  if ((type == 'A' || type == 'U') && LIBXS_NEQ(0, check)) {
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
    libxs_matdiff(LIBXS_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxs_filter, 0, 0, &norms_upd);
    printf("L1 reference  : %.25f\n", norms_upd.l1_ref);
    printf("L1 test       : %.25f\n", norms_upd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
    printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
    libxs_matdiff_reduce(&diff, &norms_upd);

#ifdef USE_FUSED_MAX_STATS
    {
      int thread_i = 0;
      int entry_i = 0;
      int c,k,r,s;
      float max_naive = 0.0;
      float max_libxs = 0.0;

      for ( entry_i = 0; entry_i < nOfm*nIfm*kh*kw; ++entry_i) {
        max_naive = LIBXS_MAX( max_naive , fabs(naive_filter_wu[entry_i]));
      }

      for ( thread_i = 0; thread_i < nImg; ++thread_i) {
        for ( entry_i = 0; entry_i < 16; ++entry_i ) {
          max_libxs = LIBXS_MAX( max_libxs, maxstats_libxs_upd[thread_i*16+entry_i]);
        }
      }

      printf("\nABSOLUTE MAX VALUES UPD:\n");
      printf("Referen. max abs UPD value: %.25f\n", max_naive);
      printf("LIBXS  max abs UPD value: %.25f\n", max_libxs);
      printf("L2 abs.error  : %.24f\n\n", max_naive-max_libxs);
    }
#endif

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

  if (type == 'A' || type == 'U') {
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
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_batchstats ) );
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

