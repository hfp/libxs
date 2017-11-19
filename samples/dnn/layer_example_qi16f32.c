/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar,
   Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxs.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

# define USE_OVERWRITE
/*# define USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE*/
# define USE_FUSED_BATCH_STATS

#define FP64_BN_STATS
/*#define USE_FUSED_RELU_BWD*/
#if !defined(USE_FUSED_BIAS) && 0
# define USE_FUSED_BIAS
#endif
#if !defined(USE_FUSED_RELU) && 0
# define USE_FUSED_RELU
#endif

#if !defined(USE_FUSED) && 0
# define USE_FUSED_BIAS_RELU
#endif

#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

#define CHKERR_LIBXS_DNN(A) if ( A != LIBXS_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxs_dnn_get_error(A) );

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

LIBXS_INLINE void zero_buf(float* buf, long size) {
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXS_INLINE void zero_buf_i16(short* buf, long size) {
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}


LIBXS_INLINE void copy_buf(float* src, float* dst, long size) {
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXS_INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
  }
}

LIBXS_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, float, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXS_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0.0;
        }
      }
    }
  }
}

LIBXS_INLINE void copy_internal_nchw(float* dst , float* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXS_VLA_DECL(4, float, input, src, C, H, W);
  LIBXS_VLA_DECL(4, float, new_input, dst, C, H+2*pad_h, W+2*pad_w);
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

LIBXS_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C)
{
  LIBXS_VLA_DECL(4,       float, output, nhwc, H, W, C);
  LIBXS_VLA_DECL(4, const float,  input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXS_VLA_ACCESS(4, output, n, h, w, c, H, W, C) =
          LIBXS_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}


LIBXS_INLINE void naive_copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C)
{
  LIBXS_VLA_DECL(4,       float, output, nchw, C, H, W);
  LIBXS_VLA_DECL(4, const float,  input, nhwc, H, W, C);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXS_VLA_ACCESS(4, output, n, c, h, w, C, H, W) =
          LIBXS_VLA_ACCESS(4,  input, n, h, w, c, H, W, C);
        }
      }
    }
  }
}


LIBXS_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, float* rsck, int R, int S, int C, int K)
{
  LIBXS_VLA_DECL(4,       float, output, rsck, S, C, K);
  LIBXS_VLA_DECL(4, const float,  input, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXS_VLA_ACCESS(4, output, r, s, c, k, S, C, K) =
          LIBXS_VLA_ACCESS(4,  input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}


LIBXS_INLINE void naive_copy_RSCK_to_KCRS(const float* rsck, float* kcrs, int R, int S, int C, int K)
{
  LIBXS_VLA_DECL(4, const float,  input, rsck, S, C, K);
  LIBXS_VLA_DECL(4,       float, output, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXS_VLA_ACCESS(4, output, k, c, r, s, C, R, S) =
            LIBXS_VLA_ACCESS(4,  input, r, s, c, k, S, C, K);
        }
      }
    }
  }
}


LIBXS_INLINE void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias)
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

  LIBXS_VLA_DECL(4,       float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, const float,  input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#if defined(USE_FUSED_BIAS) || defined(USE_FUSED_BIAS_RELU)
#if defined(_OPENMP)
# pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) = bias[ofm];
        }
      }
    }
  }
#endif

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
                  LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
           (LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) < 0.0f) ? 0.0f : LIBXS_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
#endif
    }
  }
}

LIBXS_INLINE void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save)
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

  LIBXS_VLA_DECL(4, const float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4,       float,  input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4,       float,  naive_input_t,  naive_input_save + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXS_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

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
                LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) +=
                  LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                * LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if defined(USE_FUSED_RELU_BWD) 
      for (ij = 0; ij < ifh; ij++) {
        for (ii = 0; ii < ifw; ii++) {
          if ( LIBXS_VLA_ACCESS(4,  naive_input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) == 0.0 ) {
            LIBXS_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) = 0.0;
          }
        }
      }
#endif
    }
  }
}

LIBXS_INLINE void naive_conv_wu(naive_conv_t* param, const float* input, const float* output, float* filter)
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

  LIBXS_VLA_DECL(4, const float, output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXS_VLA_DECL(4, const float,  input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
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
                LIBXS_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) +=
                  LIBXS_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXS_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
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
  float *naive_input, *naive_output, *naive_output_save, *naive_filter, *naive_filter_wu, *naive_output_bp, *naive_output_wu, *naive_libxs_output;
  float *naive_libxs_input, *naive_libxs_filter, *naive_input_save, *naive_filter_save;
  float *naive_bias, *naive_dbias, *dbias_libxs;
  float *output_libxs, *dinput_libxs, *dfilter_libxs, *bias_libxs;
  short *input_libxs, *filter_libxs, *doutput_libxs, *filtertr_libxs;
  short *i16_naive_input, *i16_naive_filter, *i16_naive_doutput;
  float *dq_naive_input, *dq_naive_filter, *dq_naive_doutput;
  unsigned char scf_input, scf_filter, scf_doutput, scf_filtertr;
  
#ifdef FP32_BN_STATS
  float *batchstats_libxs;
#endif
#ifdef FP64_BN_STATS
  double *batchstats_libxs;
#endif

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 0;           /* padding in input, height */
  int padw = 0;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  int padding_mode = 0;   /* padding mode */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'L';

  const char *const env_check = getenv("CHECK"), *const env_winograd = getenv("WINOGRAD");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
  const int algo_winograd = (0 == env_winograd ? 0 : atoi(env_winograd));

  #if defined(_OPENMP)
  int nThreads = omp_get_max_threads();      /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;

  libxs_dnn_conv_desc conv_desc;
  libxs_dnn_layer* libxs_handle;
  libxs_dnn_tensor* libxs_input;
  libxs_dnn_tensor* libxs_output;
  libxs_dnn_tensor* libxs_filter;
  libxs_dnn_tensor* libxs_dinput;
  libxs_dnn_tensor* libxs_doutput;
  libxs_dnn_tensor* libxs_dfilter;
  libxs_dnn_tensor* libxs_filter_tr;
  libxs_dnn_tensor* libxs_bias;
  libxs_dnn_tensor* libxs_dbias;
  libxs_dnn_tensor* libxs_batchstats;
  libxs_dnn_tensor_datalayout* libxs_layout;
  libxs_dnn_err_t status;

  libxs_matdiff_info norms_fwd, norms_bwd, norms_upd, diff, norms_batchstats, norms_quant;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd, 0, sizeof(norms_upd));
  memset(&norms_batchstats, 0, sizeof(norms_batchstats));
  memset(&diff, 0, sizeof(diff));
  memset(&norms_quant, 0, sizeof(norms_quant));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type format padding_mode\n", argv[0]);
    return 0;
  }
  srand48(1);

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

    pad_h_in = 0;
    pad_w_in = 0;
    pad_h_out = 0;
    pad_w_out = 0;

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
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
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
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXS_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );
#if defined(USE_OVERWRITE)
  printf("Using Overwrite Option\n");
#endif

  /* allocate data */
  naive_input           = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_save      = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output          = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_save     = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_bp       = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_wu       = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxs_output  = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxs_input   = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_filter          = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_save     = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_wu       = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_libxs_filter  = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  input_libxs         = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  filter_libxs        = (short*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  output_libxs        = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_libxs        = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dfilter_libxs       = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  doutput_libxs       = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  filtertr_libxs      = (short*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  i16_naive_input       = (short*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(short), 2097152);
  i16_naive_filter      = (short*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(short), 2097152);
  i16_naive_doutput     = (short*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(short), 2097152);
  dq_naive_input        = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dq_naive_filter       = (float*)libxs_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  dq_naive_doutput      = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
#ifdef FP32_BN_STATS
  batchstats_libxs    = (float*)libxs_aligned_malloc( 2*nImg*nOfm*        sizeof(float), 2097152);
#endif
#ifdef FP64_BN_STATS
  batchstats_libxs    = (double*)libxs_aligned_malloc( 2*nImg*nOfm*       sizeof(double), 2097152);
#endif
  naive_bias            = (float*)libxs_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_dbias           = (float*)libxs_aligned_malloc( nOfm*               sizeof(float), 2097152);
  bias_libxs          = (float*)libxs_aligned_malloc( nOfm*               sizeof(float), 2097152);
  dbias_libxs         = (float*)libxs_aligned_malloc( nOfm*               sizeof(float), 2097152);

  /* initialize data */
  float *naive_input_tmp           = (float*)libxs_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  if (padding_mode == 0 ) {
    init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  } else {
    init_buf(naive_input_tmp,          nImg*nIfm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
  }
#if defined(USE_FUSED_RELU_BWD)
  /* Initialize some entries with zeros  */
  {
    int i;
    for (i = 0; i < nImg*nIfm*ifhp*ifwp; i++ ) {
      if ( ((i%16) == 2) || ((i%16) == 3) || ((i%16) == 7) || ((i%16) == 14) ) {
        naive_input[i] = 0.0;
      }
    }
  }
#endif

  float *naive_output_bp_tmp       = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  float *naive_output_wu_tmp       = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  if (padding_mode == 0 ) {
    init_buf(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
    init_buf(naive_output_wu,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    init_buf(naive_output_bp_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    init_buf(naive_output_wu_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_wu , naive_output_wu_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w); 
  }
  set_zeropad_nchw(naive_input, nImg, nIfm, ifhp, ifwp, pad_h_in, pad_w_in);
  set_zeropad_nchw(naive_output_bp, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_output_wu, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);

  copy_buf(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_output_save,    nImg*nOfm*ofhp*ofwp);

  float *naive_output_tmp          = (float*)libxs_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  if (padding_mode == 0 ) {
    init_buf(naive_output,       nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    init_buf(naive_output_tmp,       nImg*nOfm*ofh*ofw, 0, 0);
  }
  set_zeropad_nchw(naive_output, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);

  copy_buf(naive_output, naive_output_save, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxs_output, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxs_input,  nImg*nIfm*ifhp*ifwp);
  init_buf(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  copy_buf(naive_filter, naive_filter_wu, nOfm*nIfm*kh*kw);
  zero_buf(naive_libxs_filter, nOfm*nIfm*kh*kw);
  init_buf(naive_bias,           nOfm, 0, 0);
  init_buf(naive_dbias,          nOfm, 0, 0);

  /* first touch LIBXS */
  zero_buf_i16( input_libxs    , nImg*nIfm*ifhp*ifwp );
  zero_buf_i16( filter_libxs   , nOfm*nIfm*kh*kw );
  zero_buf( output_libxs   , nImg*nOfm*ofhp*ofwp );
  zero_buf( dinput_libxs   , nImg*nIfm*ifhp*ifwp );
  zero_buf( dfilter_libxs  , nOfm*nIfm*kh*kw );
  zero_buf_i16( doutput_libxs  , nImg*nOfm*ofhp*ofwp );
  zero_buf_i16( filtertr_libxs , nOfm*nIfm*kh*kw );

  printf("##########################################\n");
  printf("#         Computing Reference ...        #\n");
  printf("##########################################\n");
  if (type == 'A' || type == 'F') {
#ifdef USE_OVERWRITE
    zero_buf(naive_output,    nImg*nOfm*ofhp*ofwp);
#endif
    naive_conv_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias);
  }
  if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
#ifdef USE_OVERWRITE
    zero_buf(naive_input,         nImg*nIfm*ifhp*ifwp);
#endif
    naive_conv_bp(&naive_param, naive_input, naive_output_bp, naive_filter, naive_input_save);
  }
  if (type == 'A' || type == 'U') {
    /* NB: We reuse naive_input_save for weight update because the input should not
     * have been modified between forward propagation and weight update; it further
     * helps in exploiting reuse to converted data. */
#ifdef USE_OVERWRITE
    zero_buf(naive_filter_wu,          nOfm*nIfm*kh*kw);
#endif
    naive_conv_wu(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);
  }
  printf("##########################################\n");
  printf("#      Computing Reference ... done      #\n");
  printf("##########################################\n");

  if (format == 'A' || format == 'L') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
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
    conv_desc.algo = (0 == algo_winograd ? LIBXS_DNN_CONV_ALGO_DIRECT : LIBXS_DNN_CONV_ALGO_AUTO);
    conv_desc.buffer_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
    conv_desc.filter_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
    conv_desc.options = LIBXS_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE_OVERWRITE;
#elif defined(USE_OVERWRITE)
    conv_desc.options = LIBXS_DNN_CONV_OPTION_OVERWRITE;
#else
    conv_desc.options = LIBXS_DNN_CONV_OPTION_NONE;
#endif
#if defined(USE_FUSED_BIAS)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BIAS;
#elif defined(USE_FUSED_RELU)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_RELU;
#elif defined(USE_FUSED_BIAS_RELU)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BIAS_RELU;
#elif defined(USE_FUSED_BATCH_STATS)
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS;
#elif defined(USE_FUSED_RELU_BWD)
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_RELU_BWD;
#elif defined(USE_FUSED_BATCH_STATCH_RELU_BWD)
   conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD;
#else
    conv_desc.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
#endif
    /*conv_desc.options = LIBXS_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE;*/
    conv_desc.datatype_in = LIBXS_DNN_DATATYPE_I16;
    conv_desc.datatype_out = LIBXS_DNN_DATATYPE_F32;

    libxs_handle = libxs_dnn_create_conv_layer( conv_desc, &status );
    CHKERR_LIBXS_DNN( status );

    /* setup LIBXS buffers and filter */
    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_INPUT, &status ); CHKERR_LIBXS_DNN( status );
    libxs_input  = libxs_dnn_link_tensor( libxs_layout,  input_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_INPUT, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dinput = libxs_dnn_link_tensor( libxs_layout, dinput_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
    libxs_output  = libxs_dnn_link_tensor( libxs_layout,  output_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_OUTPUT, &status ); CHKERR_LIBXS_DNN( status );
    libxs_doutput = libxs_dnn_link_tensor( libxs_layout, doutput_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_FILTER, &status ); CHKERR_LIBXS_DNN( status );
    libxs_filter  = libxs_dnn_link_tensor( libxs_layout,  filter_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_FILTER, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dfilter = libxs_dnn_link_tensor( libxs_layout, dfilter_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_BIAS, &status ); CHKERR_LIBXS_DNN( status );
    libxs_bias  = libxs_dnn_link_tensor( libxs_layout,  bias_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_GRADIENT_BIAS, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dbias = libxs_dnn_link_tensor( libxs_layout, dbias_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_REGULAR_FILTER_TRANS, &status ); CHKERR_LIBXS_DNN( status );
    libxs_filter_tr  = libxs_dnn_link_tensor( libxs_layout, filtertr_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    libxs_layout = libxs_dnn_create_tensor_datalayout( libxs_handle, LIBXS_DNN_BATCH_STATS, &status ); CHKERR_LIBXS_DNN( status );
    libxs_batchstats  = libxs_dnn_link_tensor( libxs_layout, batchstats_libxs, &status ); CHKERR_LIBXS_DNN( status );
    libxs_dnn_destroy_tensor_datalayout( libxs_layout );

    /* quantize input, filter, and Bias */
    libxs_dnn_quantize( naive_input_save, i16_naive_input,  nImg*nIfm*ifhp*ifwp, 2, &scf_input,  LIBXS_DNN_QUANT_BIAS_ROUND );
    libxs_dnn_quantize( naive_filter,     i16_naive_filter, nIfm*nOfm*kw*kh    , 2, &scf_filter, LIBXS_DNN_QUANT_BIAS_ROUND );

    /* set scaling factors into tensors */
    libxs_dnn_set_qtensor_scf( libxs_input,  scf_input );
    libxs_dnn_set_qtensor_scf( libxs_filter, scf_filter );

    /* dequantize to check quantization error */
    libxs_dnn_dequantize( i16_naive_input,  dq_naive_input,  nImg*nIfm*ifhp*ifwp, scf_input );
    libxs_dnn_dequantize( i16_naive_filter, dq_naive_filter, nIfm*nOfm*kw*kh,     scf_filter ); 

    /* copy in data to LIBXS format */
    /* we can also use the layout functions and set the data on our
       own external to the library, @TODO, we plan to add an example here */
    CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_input,  (void*)i16_naive_input,   LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_output, (void*)naive_output_save, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_filter, (void*)i16_naive_filter,  LIBXS_DNN_TENSOR_FORMAT_KCRS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_bias,   (void*)naive_bias,        LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
    zero_buf_i16(filtertr_libxs, nOfm*nIfm*kh*kw);
#ifdef FP32_BN_STATS 
    zero_buf(batchstats_libxs, 2*nImg*nOfm);
#endif
#ifdef FP64_BN_STATS 
    zero_buf((float *) batchstats_libxs, 4*nImg*nOfm);
#endif

    /* bind buffers and filter to handle */
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_input,      LIBXS_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dinput,     LIBXS_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_output,     LIBXS_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_doutput,    LIBXS_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_filter,     LIBXS_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dfilter,    LIBXS_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_bias,       LIBXS_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_dbias,      LIBXS_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_filter_tr,  LIBXS_DNN_REGULAR_FILTER_TRANS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_tensor( libxs_handle, libxs_batchstats, LIBXS_DNN_BATCH_STATS ) );

    /* let's allocate and bind scratch */
    scratch_size = libxs_dnn_get_scratch_size( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, &status );
    CHKERR_LIBXS_DNN( status );
    scratch = (void*)libxs_aligned_malloc( scratch_size, 2097152 );
    CHKERR_LIBXS_DNN( status );
    CHKERR_LIBXS_DNN( libxs_dnn_bind_scratch( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL, scratch ) );
    /* set scratch to bogus to make sure that libxs takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXS convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
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

      /* norms quantization */
      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_save, dq_naive_input, 0, 0, &norms_quant);
      printf("Input Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      libxs_matdiff(LIBXS_DATATYPE_F32, nIfm*nOfm*kw*kh, 1, naive_filter, dq_naive_filter, 0, 0, &norms_quant);
      printf("Filter Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      /* compare */
      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output, naive_libxs_output, 0, 0, &norms_fwd);
      printf("Output:\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxs_matdiff_reduce(&diff, &norms_fwd);

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
        LIBXS_VLA_DECL(3, float, sum_naive, naive_output,       nOfm, ofhp*ofwp);

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
          for ( img_i = 0; img_i < nImg; ++img_i ) {
            for ( ch_j = 0; ch_j < 16; ++ch_j ) {
#ifdef FP32_BN_STATS    
              ch_sum_fuse[(ch_i*16) + ch_j]  += sum_fuse[0][ch_i][img_i][ch_j];           
              ch_sum2_fuse[(ch_i*16) + ch_j] += sum_fuse[1][ch_i][img_i][ch_j];
#endif
#ifdef FP64_BN_STATS 
              ch_sum_fuse[(ch_i*16) + ch_j]  += (float) sum_fuse[0][ch_i][img_i][ch_j];           
              ch_sum2_fuse[(ch_i*16) + ch_j] += (float) sum_fuse[1][ch_i][img_i][ch_j];
#endif
            }
          }
        }
        for ( img_i = 0; img_i < nImg; ++img_i ) {
          for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
            for ( pxl_i = 0; pxl_i < ofhp*ofwp; ++pxl_i ) {
              ch_sum[ch_i]  += sum_naive[img_i][ch_i][pxl_i];
              ch_sum2[ch_i] += (sum_naive[img_i][ch_i][pxl_i]*sum_naive[img_i][ch_i][pxl_i]);
            }
          }
        }

        libxs_matdiff(LIBXS_DATATYPE_F32, nOfm, 1, ch_sum, ch_sum_fuse, 0, 0, &norms_batchstats);
        printf("Channel Sum:\n");
        printf("L1 reference  : %.25g\n", norms_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_batchstats.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_batchstats.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_batchstats.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_batchstats.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_batchstats.linf_rel);
        printf("Check-norm    : %.24f\n", norms_batchstats.normf_rel);

        libxs_matdiff(LIBXS_DATATYPE_F32, nOfm, 1, ch_sum2, ch_sum2_fuse, 0, 0, &norms_batchstats);
        printf("Channel Sum2:\n");
        printf("L1 reference  : %.25g\n", norms_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_batchstats.l1_tst);
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

    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");

      /* quantize input, filter, and Bias */
      libxs_dnn_quantize( naive_output_bp, i16_naive_doutput, nImg*nOfm*ofhp*ofwp, 2, &scf_doutput,  LIBXS_DNN_QUANT_BIAS_ROUND );
  
      /* set scaling factors into tensors */
      libxs_dnn_set_qtensor_scf( libxs_doutput,  scf_doutput );

      /* dequantize to check quantization error */
      libxs_dnn_dequantize( i16_naive_doutput,  dq_naive_doutput, nImg*nOfm*ofhp*ofwp, scf_doutput );

      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor(    libxs_doutput, (void*)i16_naive_doutput, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor(    libxs_dinput,  (void*)naive_input_save,  LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
      CHKERR_LIBXS_DNN( libxs_dnn_trans_reg_filter( libxs_handle ) );
#endif

      /* run LIBXS convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
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

      /* norms quantization */
      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_bp, dq_naive_doutput, 0, 0, &norms_quant);
      printf("del-Output Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      libxs_matdiff(LIBXS_DATATYPE_F32, nIfm*nOfm*kw*kh, 1, naive_filter, dq_naive_filter, 0, 0, &norms_quant);
      printf("Filter Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      /* compare */
      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_libxs_input, 0, 0, &norms_bwd);
      printf("del-Input:\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxs_matdiff_reduce(&diff, &norms_bwd);
    }

    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* quantize input, filter, and Bias */
      libxs_dnn_quantize( naive_input_save, i16_naive_input,   nImg*nIfm*ifhp*ifwp, 2, &scf_input,    LIBXS_DNN_QUANT_BIAS_ROUND );
      libxs_dnn_quantize( naive_output_wu,  i16_naive_doutput, nImg*nOfm*ofhp*ofwp, 2, &scf_doutput,  LIBXS_DNN_QUANT_BIAS_ROUND );
  
      /* set scaling factors into tensors */
      libxs_dnn_set_qtensor_scf( libxs_input,  scf_input );
      libxs_dnn_set_qtensor_scf( libxs_doutput,  scf_doutput );

      /* dequantize to check quantization error */
      libxs_dnn_dequantize( i16_naive_input,    dq_naive_input,   nImg*nIfm*ifhp*ifwp, scf_input );
      libxs_dnn_dequantize( i16_naive_doutput,  dq_naive_doutput, nImg*nOfm*ofhp*ofwp, scf_doutput );

      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_input,   (void*)i16_naive_input,   LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_doutput, (void*)i16_naive_doutput, LIBXS_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXS_DNN( libxs_dnn_copyin_tensor( libxs_dfilter, (void*)naive_filter,      LIBXS_DNN_TENSOR_FORMAT_KCRS ) );

      /* run LIBXS convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXS_DNN( libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }
      if (conv_desc.options == LIBXS_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
        CHKERR_LIBXS_DNN( libxs_dnn_reduce_wu_filters( libxs_handle, LIBXS_DNN_GRADIENT_FILTER ) );
      }

      /* copy out data */
      CHKERR_LIBXS_DNN( libxs_dnn_copyout_tensor( libxs_dfilter, (void*)naive_libxs_filter, LIBXS_DNN_TENSOR_FORMAT_KCRS ) );

      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_save, dq_naive_input, 0, 0, &norms_quant);
      printf("Input Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      /* norms quantization */
      libxs_matdiff(LIBXS_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_bp, dq_naive_doutput, 0, 0, &norms_quant);
      printf("del-Output Quantization:\n");
      printf("L1 reference  : %.25g\n", norms_quant.l1_ref);
      printf("L1 test       : %.25g\n", norms_quant.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_quant.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_quant.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_quant.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_quant.linf_rel);
      printf("Check-norm    : %.24f\n", norms_quant.normf_rel);

      /* compare */
      libxs_matdiff(LIBXS_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxs_filter, 0, 0, &norms_upd);
      printf("del-Filter:\n");
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxs_matdiff_reduce(&diff, &norms_upd);
    }

    if ((type == 'A' || type == 'F') && LIBXS_FEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXS convolution for performance */
      l_start = libxs_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxs_timer_tick();
      l_total = libxs_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) && LIBXS_FEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXS convolution for performance */
      l_start = libxs_timer_tick();

#if defined(_OPENMP)
#   pragma omp parallel  private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxs_timer_tick();
      l_total = libxs_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    if ((type == 'A' || type == 'U') && LIBXS_FEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXS convolution for performance */
      l_start = libxs_timer_tick();

#if defined(_OPENMP)
#   pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxs_dnn_execute_st( libxs_handle, LIBXS_DNN_COMPUTE_KIND_UPD, 0, tid );
          if (conv_desc.options == LIBXS_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
            CHKERR_LIBXS_DNN( libxs_dnn_reduce_wu_filters( libxs_handle, LIBXS_DNN_GRADIENT_FILTER ) );
          }
        }
      }
      l_end = libxs_timer_tick();
      l_total = libxs_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXS_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXS_DNN( libxs_dnn_release_scratch( libxs_handle, LIBXS_DNN_COMPUTE_KIND_ALL ) );
    libxs_free(scratch);
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_REGULAR_FILTER_TRANS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_release_tensor( libxs_handle, LIBXS_DNN_BATCH_STATS ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_input ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_output ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_filter ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_dinput ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_doutput ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_dfilter ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_bias ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_dbias ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_filter_tr ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( libxs_batchstats ) );
    CHKERR_LIBXS_DNN( libxs_dnn_destroy_conv_layer( libxs_handle ) );
  }

  /* deallocate data */
  libxs_free(naive_input);
  libxs_free(naive_input_save);
  libxs_free(naive_output);
  libxs_free(naive_output_save);
  libxs_free(naive_output_bp);
  libxs_free(naive_output_wu);
  libxs_free(naive_libxs_output);
  libxs_free(naive_libxs_input);
  libxs_free(naive_filter);
  libxs_free(naive_filter_save);
  libxs_free(naive_filter_wu);
  libxs_free(naive_libxs_filter);
  libxs_free(input_libxs);
  libxs_free(filter_libxs);
  libxs_free(output_libxs);
  libxs_free(dinput_libxs);
  libxs_free(dfilter_libxs);
  libxs_free(doutput_libxs);
  libxs_free(filtertr_libxs);
  libxs_free(batchstats_libxs);
  libxs_free(naive_bias);
  libxs_free(naive_dbias);
  libxs_free(bias_libxs);
  libxs_free(dbias_libxs);
  libxs_free(i16_naive_input);
  libxs_free(i16_naive_filter);
  libxs_free(i16_naive_doutput);
  libxs_free(dq_naive_input);
  libxs_free(dq_naive_filter);
  libxs_free(dq_naive_doutput);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXS_ABS(0 == env_check_scale ? 100.0 : atof(env_check_scale));
    if (0 == LIBXS_FEQ(0, check) && check < 100.0 * check_scale * diff.normf_rel) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

