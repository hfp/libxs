# Interface for Convolutions
To achieve best performance with small convolutions for CNN on SIMD architectures, a specific data layout must be used. As this layout depends on several architectural parameters, the goal of the LIBXS's interface is to hide this complexity from the user by providing copy-in and copy-out routines. This happens using opaque data types which themselves are later bound to a convolution operation.

The interface is available for C. There is a collection ([samples/dnn](https://github.com/hfp/libxs/tree/master/samples/dnn)) of benchmarks-style code samples with focus on Convolutional Neural Networks (CNNs). Further, an [example](https://github.com/hfp/libxs/tree/master/samples/iconv#image-convolution) performing an image convolution is provided as well. The general concept of the interface is circled around a few handle types: `libxs_dnn_layer`, `libxs_dnn_buffer`, `libxs_dnn_bias`, and `libxs_dnn_filter`. A handle is setup by calling a create-function:

```C
/** Simplified LIBXS types which are needed to create a handle. */
/** Structure which describes the input and output of data (DNN). */
typedef struct libxs_dnn_conv_desc {
  int N;                                    /* number of images in mini-batch */
  int C;                                    /* number of input feature maps */
  int H;                                    /* height of input image */
  int W;                                    /* width of input image */
  int K;                                    /* number of output feature maps */
  int R;                                    /* height of filter kernel */
  int S;                                    /* width of filter kernel */
  int u;                                    /* vertical stride */
  int v;                                    /* horizontal stride */
  int pad_h;                                /* height of logical rim padding to input
                                               for adjusting output height */
  int pad_w;                                /* width of logical rim padding to input
                                               for adjusting output width */
  int pad_h_in;                             /* height of zero-padding in input buffer,
                                               must equal to pad_h for direct conv */
  int pad_w_in;                             /* width of zero-padding in input buffer,
                                               must equal to pad_w for direct conv */
  int pad_h_out;                            /* height of zero-padding in output buffer */
  int pad_w_out;                            /* width of zero-padding in output buffer */
  int threads;                              /* number of threads to use when running
                                               convolution */
  libxs_dnn_datatype datatype;            /* datatypes use for all input and outputs */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxs_dnn_tensor_format filter_format;  /* format which is for filter buffers */
  libxs_dnn_conv_algo algo;               /* convolution algorithm used */
  libxs_dnn_conv_option options;          /* additional options */
  libxs_dnn_conv_fuse_op fuse_ops;        /* used ops into convolutions */
} libxs_dnn_conv_desc;

/** Type of algorithm used for convolutions. */
typedef enum libxs_dnn_conv_algo {
  /** let the library decide */
  LIBXS_DNN_CONV_ALGO_AUTO,   /* ignored for now */
  /** direct convolution. */
  LIBXS_DNN_CONV_ALGO_DIRECT
} libxs_dnn_conv_algo;

/** Denotes the element/pixel type of an image/channel. */
typedef enum libxs_dnn_datatype {
  LIBXS_DNN_DATATYPE_F32,
  LIBXS_DNN_DATATYPE_I32,
  LIBXS_DNN_DATATYPE_I16,
  LIBXS_DNN_DATATYPE_I8
} libxs_dnn_datatype;

libxs_dnn_layer* libxs_dnn_create_conv_layer(
  libxs_dnn_conv_desc conv_desc, libxs_dnn_err_t* status);
libxs_dnn_err_t libxs_dnn_destroy_conv_layer(
  const libxs_dnn_layer* handle);
```

A sample call looks like (without error checks):
```C
/* declare LIBXS variables */
libxs_dnn_conv_desc conv_desc;
libxs_dnn_err_t status;
libxs_dnn_layer* handle;
/* setting conv_desc values.... */
conv_desc.N = ...
/* create handle */
handle = libxs_dnn_create_conv_layer(conv_desc, &status);
```

Next activation and filter buffers need to be linked, initialized and bound to the handle. Afterwards the convolution can be executed in a threading environment of choice (error checks are omitted for brevity):

```C
float *input, *output, *filter;
libxs_dnn_buffer* libxs_reg_input;
libxs_dnn_buffer* libxs_reg_output;
libxs_dnn_filter* libxs_reg_filter;

/* allocate data */
input = (float*)libxs_aligned_malloc(...);
output = ...;

/* link data to buffers */
libxs_reg_input = libxs_dnn_link_buffer(  libxs_handle, LIBXS_DNN_INPUT, input,
                                              LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);
libxs_reg_output = libxs_dnn_link_buffer( libxs_handle, LIBXS_DNN_OUTPUT, output,
                                              LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);
libxs_reg_filter = libxs_dnn_link_filter( libxs_handle, LIBXS_DNN_FILTER, filter,
                                              LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);

/* copy in data to LIBXS format: naive format is: */
/* (mini-batch)(number-featuremaps)(featuremap-height)(featuremap-width) for layers, */
/* and the naive format for filters is: */
/* (number-output-featuremaps)(number-input-featuremaps)(kernel-height)(kernel-width) */
libxs_dnn_copyin_buffer(libxs_reg_input, (void*)naive_input, LIBXS_DNN_TENSOR_FORMAT_NCHW);
libxs_dnn_zero_buffer(libxs_reg_output);
libxs_dnn_copyin_filter(libxs_reg_filter, (void*)naive_filter, LIBXS_DNN_TENSOR_FORMAT_KCRS);

/* bind layer to handle */
libxs_dnn_bind_input_buffer(libxs_handle, libxs_reg_input, LIBXS_DNN_REGULAR_INPUT);
libxs_dnn_bind_output_buffer(libxs_handle, libxs_reg_output, LIBXS_DNN_REGULAR_OUTPUT);
libxs_dnn_bind_filter(libxs_handle, libxs_reg_filter, LIBXS_DNN_REGULAR_FILTER);

/* allocate and bind scratch */
scratch = libxs_aligned_scratch(libxs_dnn_get_scratch_size(
  libxs_handle, LIBXS_DNN_COMPUTE_KIND_FWD, &status), 2097152);
libxs_dnn_bind_scratch(libxs_handle, LIBXS_DNN_COMPUTE_KIND_FWD, scratch);

/* run the convolution */
#pragma omp parallel
{
  libxs_dnn_convolve_st(libxs_handle, LIBXS_DNN_CONV_KIND_FWD, 0,
    omp_get_thread_num(), omp_get_num_threads());
}

/* copy out data */
libxs_dnn_copyout_buffer(libxs_output, (void*)naive_libxs_output,
  LIBXS_DNN_TENSOR_FORMAT_NCHW);

/* clean up */
libxs_dnn_release_scratch(...);
libxs_dnn_release_buffer(...);
...
libxs_dnn_destroy_buffer(...);
...
libxs_dnn_destroy_conv_layer(...);
```

