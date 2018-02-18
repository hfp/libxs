# LIBXS

[![License](https://img.shields.io/badge/license-BSD3-blue.svg)](LICENSE.md) [![Travis CI](https://travis-ci.org/hfp/libxs.svg?branch=master "Master branch build status")](https://github.com/hfp/libxs/wiki/Status) [![ReadtheDocs](https://readthedocs.org/projects/libxs/badge/?version=latest "Read the Docs")](http://libxs.readthedocs.io/)

LIBXS is a library for small dense and small sparse matrix-matrix multiplications as well as for deep learning primitives such as small convolutions targeting Intel Architecture. Small matrix multiplication kernels are generated for the following instruction set extensions: Intel&#160;SSE, Intel&#160;AVX, Intel&#160;AVX2, IMCI (KNCni) for Intel&#160;Xeon&#160;Phi coprocessors ("KNC"), and Intel&#160;AVX&#8209;512 as found in the Intel&#160;Xeon&#160;Phi processor family&#160;(Knights Landing "KNL", Knights Mill "KNM") and Intel&#160;Xeon processors (Skylake-SP "SKX"). Historically small matrix multiplications were only optimized for the Intel&#160;Many Integrated Core Architecture "MIC") using intrinsic functions, meanwhile optimized assembly code is targeting all afore mentioned instruction set extensions (static code generation), and Just&#8209;In&#8209;Time (JIT) code generation is targeting Intel&#160;AVX and beyond. Optimized code for small convolutions is JIT-generated for Intel&#160;AVX2 and Intel&#160;AVX&#8209;512.

**Where to go for documentation?**

* **ReadtheDocs**: [main](http://libxs.readthedocs.io/) and [sample](http://libxs.readthedocs.io/libxs_samples/) documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxs/raw/master/documentation/libxs.pdf) documentation file, and separate [sample](https://github.com/hfp/libxs/raw/master/documentation/libxs_samples.pdf) documentation.

**<a name="what-is-a-small-matrix-multiplication"></a>What is a small matrix multiplication?** When characterizing the problem-size using the M, N, and K parameters, a problem-size suitable for LIBXS falls approximately within *(M&#160;N&#160;K)<sup>1/3</sup>&#160;&lt;=&#160;128* (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). The library is typically used to generate code up to the specified [threshold](documentation/libxs_tune.md#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M or K dimension), but also miss to implement a tiling scheme to effectively utilize the cache hierarchy. For auto-dispatched problem-sizes above the configurable threshold (explicitly JIT'ted code is **not** subject to the threshold), LIBXS is falling back to BLAS. In terms of GEMM, the supported kernels are limited to *Alpha := 1*, *Beta := \{ 1, 0 \}*, *TransA := 'N'*, and *TransB = 'N'*.

**<a name="what-is-a-small-convolution"></a>What is a small convolution?** In the last years, new workloads such as deep learning and more specifically convolutional neural networks (CNN) emerged, and are pushing the limits of today's hardware. One of the expensive kernels is a small convolution with certain kernel sizes such that calculations in the frequency space is not the most efficient method when compared with direct convolutions. LIBXS's current support for convolutions aims for an easy to use invocation of small (direct) convolutions, which are intended for CNN training and classification.

For more questions and answers, please have a look at [https://github.com/hfp/libxs/wiki/Q&A](https://github.com/hfp/libxs/wiki/Q&A).

Documented functionality and available domains:

* MM: [Matrix Multiplication](#matrix-multiplication)
* DNN: [Deep Neural Networks](#deep-neural-networks)
* AUX: [Service Functions](#service-functions)
* PERF: [Performance](#performance)
* BE: [Backend](#jit-backend)

For additional functionality, please have a look at [https://github.com/hfp/libxs/tree/master/include](https://github.com/hfp/libxs/tree/master/include).

# Overview

The main interface file is *generated*, and it is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxs/blob/master/src/template/libxs.h#L36) and [FORTRAN](https://github.com/hfp/libxs/blob/master/src/template/libxs.f#L32). The main interface consists of the [general interface](#general-interface) as well as the [matrix multiplication](#matrix-multiplication) interface.

There are two ways to incorporate LIBXS into an application:

* [Classic Library (ABI)](#classic-library-abi) and [Link Instructions](#link-instructions)
* [Header-Only](#header-only)

### Classic Library (ABI)

The build system relies on GNU&#160;Make (typically associated with the `make` command, but e.g. FreeBSD is calling it `gmake`). The build can be customized by using key&#8209;value pairs. Key&#8209;value pairs can be supplied in two ways: (1)&#160;after the "make" command, or (2)&#160;prior to the "make" command (`env`) which is effectively the same as exporting the key&#8209;value pair as an environment variable (`export`, or `setenv`). Both methods can be mixed, however the second method may require the `-e` flag. Please note that the CXX, CC, and FC keys are considered in any case.

The build system considers a set of given key-value pairs as a single unique build, and triggers a rebuild for a distinct set of flags. For more advanced builds or additional background, please consult the section about [Customization](documentation/libxs_tune.md). To generate the interface of the library inside of the 'include' directory and to build the static library (by default, STATIC=1 is activated). Run any or both of the following commands:

```bash
make STATIC=0
make
```

On CRAY systems, the CRAY Compiling Environment (CCE) should be used regardless of using the CRAY compiler, the Intel Compiler, or the GNU&#160;Compiler Collection (GCC). The CCE is eventually suppressing to build shared libraries (STATIC=0). In any case, (1)&#160;switch to the desired compiler (module load/switch), and (2)&#160;rely on:

```bash
make CXX=CC CC=cc FC=ftn
```

A variety of build environments is out-of-the-box compatible, see [https://github.com/hfp/libxs/wiki/Compatibility](https://github.com/hfp/libxs/wiki/Compatibility). If the build process is not successful, it may help to avoid advanced GCC flags. This is useful with a tool chain, which pretends to be GCC-compatible (or is treated as such) but fails to consume the afore mentioned flags:

```bash
make COMPATIBLE=1
```

To test and validate a build, please consult [https://github.com/hfp/libxs/wiki/Validation](https://github.com/hfp/libxs/wiki/Validation). To run some basic sanity checks, remember that each set of given key-value pairs represents a different build (and test):

```bash
make STATIC=0 tests
```

To remove intermediate files, or to remove all generated files and folders (including the interface and the library archives), run one of the following commands:

```bash
make clean
make realclean
```

**NOTE**: By default, C/C++ and FORTRAN compilers are needed (some sample code is written in C++). Beside of specifying the compilers (`make CXX=g++ CC=gcc FC=gfortran` and maybe `AR=ar`), the need for a FORTRAN compiler can be relaxed (`make FC=` or `make FORTRAN=0`). The latter affects the availability of the MODule file and the corresponding 'libxsf' library (the interface 'libxs.f' is still generated). FORTRAN code can make use of LIBXS in three ways:

* By relying on the module file, and by linking against 'libxsf', 'libxs', and (optionally) 'libxsext',
* By including the interface 'libxs.f' and linking against 'libxs', and (optionally) 'libxsext', or
* By optionally declaring a SUBROUTINE, or by simply calling a SUBROUTINE (FORTRAN&#160;77).

A FORTRAN&#160;77 interface is implicitly available and documented in 'libxs.f' (comments). It can be useful to have a look at the [implementation of the FORTRAN&#160;77 interface](https://github.com/hfp/libxs/search?q=implementation+provided+for+Fortran+77+compatibility).

### Link Instructions

The library is agnostic with respect to the threading-runtime, and therefore an application is free to use any threading runtime (e.g., OpenMP). The library is also thread-safe, and multiple application threads can call LIBXS's routines concurrently. Enabling OpenMP for LIBXS's main library is supported as well (OMP=1), and mostly affects the synchronization primitives used inside of the library. All of the "omp" functionality (function postfix) is served by the 'libxsext' library, which is automatically built with OpenMP enabled. When using this "omp" functionality, 'libxsext' needs to be present at the link line.

Similarly, an application is free to choose any BLAS or LAPACK library (if the link model available on the OS supports this), and therefore linking GEMM routines when linking LIBXS itself (by supplying BLAS=1&#124;2) may prevent a user from making this decision at the time of linking the actual application. To use LIBXS without GEMM-related functionality, any BLAS-dependency can be removed in two ways: (1)&#160;building a special library with `make BLAS=0`, or (2)&#160;linking the application against the 'libxsnoblas' library. If an application however uses BLAS already, the [Call Wrapper](documentation/libxs_mm.md#call-wrapper) can be used to intercept existing BLAS calls (and to rely on LIBXS instead).

**NOTE**: LIBXS does not support to dynamically link 'libxs' or 'libxsext' ("so"), when BLAS is linked statically ("a"). If BLAS is linked statically, the static version of LIBXS must be used!

### Header-Only

Version&#160;1.4.4 introduced support for "header-only" usage in C and C++. By only including 'libxs_source.h' allows to get around building the library. However, this gives up on a clearly defined application binary interface (ABI). An ABI may allow for hot-fixes after deploying an application (when relying on the shared library form), and it may also ensure to only rely on the public interface of LIBXS. In contrast, the header-only form not only exposes the internal implementation of LIBXS but it can also reduce the turnaround time during development of an application (due to longer compilation times). The header file is intentionally named "libxs_**source**.h" since this header file relies on the [src](https://github.com/hfp/libxs/tree/master/src) directory (with the implications as noted earlier).

To use the header-only form, 'libxs_source.h' needs to be *generated*. The build target shown below ('header-only') has been introduced in LIBXS&#160;1.6.2, but `make cheader` can be used alternatively (or must be used instead in case of earlier versions). Generating the C interface is necessary since the library must be configured (see [configuration](https://github.com/hfp/libxs/blob/master/src/template/libxs_config.h) template).

```bash
make header-only
```

**NOTE**: building an application now applies the same build settings to LIBXS! For instance, to omit debug code inside of LIBXS `NDEBUG` must be defined (`-DNDEBUG`).

### Installation

Installing LIBXS makes possibly the most sense when combining the JIT backend ([enabled by default](documentation/libxs_be.md)) with a collection of statically generated SSE kernels (by specifying M, N, K, or MNK). If the JIT backend is not disabled, statically generated kernels are only registered for dispatch if the CPUID flags at runtime are not supporting a more specific instruction set extension (code path). Since the JIT backend does not support or generate SSE code by itself, the library is compiled by selecting SSE code generation if not specified otherwise (AVX=1&#124;2&#124;3, or with SSE=0 falling back to an "arch-native" approach). Limiting the static code path to SSE4.2 allows to practically target any deployed system, however using SSE=0 and AVX=0 together is falling back to generic code, and any static kernels are not specialized using the assembly code generator.

There are two main mechanisms to install LIBXS (both mechanisms can be combined): (1)&#160;building the library in an out&#8209;of&#8209;tree fashion, and (2)&#160;installing into a certain location. Building in an out&#8209;of&#8209;tree fashion looks like:

```bash
cd libxs-install
make -f /path/to/libxs/Makefile
```

For example, installing into a specific location (incl. a selection of statically generated Intel&#160;SSE kernels) looks like:

```bash
make MNK="1 2 3 4 5" PREFIX=/path/to/libxs-install install
```

Performing `make install-minimal` omits the documentation (default: 'PREFIX/share/libxs'). Moreover, PINCDIR, POUTDIR, PBINDIR, and PDOCDIR allow to customize the locations underneath of the PREFIX location. To build a general package for an unpredictable audience (Linux distribution, or similar), it is advised to not over-specify or customize the build step i.e., JIT, SSE, AVX, OMP, BLAS, etc. should not be used. The following is building and installing a complete set of libraries where the generated interface matches both the static and the shared libraries:

```bash
make PREFIX=/path/to/libxs-install STATIC=0 install
make PREFIX=/path/to/libxs-install install
```

# General Interface

To initialize the dispatch-table or other internal resources, an explicit initialization routine helps to avoid lazy initialization overhead when calling LIBXS for the first time. The library deallocates internal resources at program exit, but also provides a companion to the afore mentioned initialization (finalize).

```C
/** Initialize the library; pay for setup cost at a specific point. */
void libxs_init(void);
/** De-initialize the library and free internal memory (optional). */
void libxs_finalize(void);
```

## Deep Neural Networks<a name="interface-for-convolutions"></a>

This domain (DNN) is detailed by a separate [document](documentation/libxs_dnn.md). Please also note on how to [Get Started with TensorFlow&trade; using LIBXS](documentation/tensorflow.md).

### Service Functions

For convenient operation of the library and to ease integration, some service routines are available. These routines may not belong to the core functionality of LIBXS (SMM or DNN domain), but users are encouraged to use this domain (AUX). There are two categories: (1)&#160;routines which are available for C and Fortran, and (2)&#160;routines that are only available per C interface.

The [service function domain (AUX)](documentation/libxs_aux.md) contains routines for:

* [Getting and setting the target architecture](documentation/libxs_aux.md#getting-and-setting-the-target-architecture)
* [Getting and setting the verbosity](documentation/libxs_aux.md#getting-and-setting-the-verbosity)
* [Measuring time durations (timer)](documentation/libxs_aux.md#timer-facility)
* [Loading and storing data (I/O)](documentation/libxs_aux.md#meta-image-file-io)
* [Allocating memory](documentation/libxs_aux.md#memory-allocation)

### Backend<a name="jit-backend"></a>

More information about the JIT-backend and the code generator can be found in a separate [document](documentation/libxs_be.md), which also includes information about LIBXS's stand-alone <a name="generator-driver"></a>[generator-driver](documentation/libxs_be.md#generator-driver) programs.

# Verbose Mode

The [verbose mode](documentation/libxs_aux.md#getting-and-setting-the-verbosity) (level of verbosity) allows for an insight into the code dispatch mechanism by receiving a small tabulated statistic as soon as the library terminates. The design point for this functionality is to not impact the performance of any critical code path i.e., verbose mode is always enabled and does not require symbols (SYM=1) or debug code (DBG=1). The statistics appears (`stderr`) when the environment variable LIBXS_VERBOSE is set to a non-zero value. For example:

```bash
LIBXS_VERBOSE=1 ./myapplication
[... application output]

HSW/SP      TRY    JIT    STA    COL
   0..13      0      0      0      0
  14..23      0      0      0      0
 24..128      3      3      0      0
```

The tables are distinct between single-precision and double-precision, but either table is pruned if all counters are zero. If both tables are pruned, the library shows the code path which would have been used for JIT'ting the code: `LIBXS_TARGET=hsw` (otherwise the code path is shown in the table's header). The actual counters are collected for three buckets: small kernels (MNK<sup>1/3</sup>&#160;&lt;=&#160;13), medium-sized kernels (13&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;23), and larger kernels (23&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;128; the actual upper bound depends on LIBXS_MAX_MNK as selected at compile-time). Keep in mind, that "larger" is supposedly still small in terms of arithmetic intensity (which grows linearly with the kernel size). Unfortunately, the arithmetic intensity depends on the way a kernel is used (which operands are loaded/stored into main memory) and it is not performance-neutral to collect this information.

The TRY counter represents all attempts to register statically generated kernels, and all attempts to dynamically generate and register kernels. The TRY counter includes rejected JIT requests due to unsupported GEMM arguments. The JIT and STA counters distinct the successful cases of the afore mentioned event (TRY) into dynamically (JIT) and statically (STA) generated code. In case the capacity (O(*n*)&#160;=&#160;10<sup>5</sup>) of the code registry is exhausted, no more kernels can be registered although further attempts are not prevented. Registering many kernels (O(*n*)&#160;=&#160;10<sup>3</sup>) may ramp the number of hash key collisions (COL), which can degrade performance. The latter is prevented if the small thread-local cache is utilized effectively.

Since explicitly JIT-generated code (`libxs_?mmdispatch`) does not fall under the THRESHOLD criterion, the above table is extended by one line if large kernels have been requested. This indicates a missing threshold-criterion (customized dispatch), or asks for cache-blocking the matrix multiplication. The latter is already implemented by LIBXS's "medium-sized" GEMM routines (`libxs_?gemm_omp`), which perform a tiled multiplication. Setting a verbosity level of at least two summarizes the number of registered JIT-generated kernels, which includes the total size and counters for GEMM, MCOPY (matrix copy), and TCOPY (matrix transpose) kernels.

```bash
Registry: 20 MB (gemm=0 mcopy=14 tcopy=0)
```

If the call-wrapper is used, an additional runtime statistic becomes available (see [Call Wrapper](documentation/libxs_mm.md#call-wrapper)).

**NOTE**: Setting LIBXS_VERBOSE to a negative value will binary-dump each generated JIT kernel to a file with each file being named like the function name shown in [Intel&#160;VTune](documentation/libxs_prof.md#intelvtuneamplifier). Disassembly of the raw binary files can be accomplished by:

```bash
objdump -D -b binary -m i386 -M x86-64 [JIT-dump-file]
```

## Performance

<a name="profiling"></a>Profiling an application, which uses LIBXS's JIT-code is well-supported. The library supports Intel&#160;VTune&#160;Amplifier and Linux&#160;perf. Details are given on how to include profiler support, and how to run the application.

* [Profiling using Intel&#160;VTune&#160;Amplifier](documentation/libxs_prof.md#intelvtuneamplifier)
* [Profiling using Linux&#160;perf](documentation/libxs_prof.md#linuxperf)

<a name="tuning"></a>At build time, a variety of options exist to customize LIBXS. The library is setup for a broad range of use cases, which include sophisticated defaults for general use.

* [Customizing performance](documentation/libxs_tune.md#tuning)
* <a name="auto-dispatch"></a>[Tuning auto-dispatch](documentation/libxs_tune.md#auto-dispatch)

<a name="results"></a>To find performance results of applications or performance reproducers, the repository provides an orphaned branch called "results" which collects collateral material such as measured performance results along with explanatory figures. The results can be found at [https://github.com/hfp/libxs/tree/results#libxs-results](https://github.com/hfp/libxs/tree/results#libxs-results), or the results can be cloned as shown below.

```bash
git clone --branch results \
  https://github.com/hfp/libxs.git \
  libxs-results
```

Please note that comparing performance results depends on whether the operands of the matrix multiplication are streamed or not. For example, multiplying with all matrices covered by the L1 cache may have an emphasis towards an implementation which perhaps performs worse for the real workload (if this real workload needs to stream some or all matrices from the main memory). Most of the [code samples](https://github.com/hfp/libxs/tree/master/samples) are aimed to reproduce performance results, and it is encouraged to model the exact case or to look at real [applications](#applications).

# High Performance Computing (HPC)

**\[1]&#160;[https://cp2k.org/](https://cp2k.org/)**: Open Source Molecular Dynamics with its DBCSR component processing batches of small matrix multiplications ("matrix stacks") out of a problem-specific distributed block-sparse matrix. Starting with [CP2K 3.0](https://www.cp2k.org/version_history), LIBXS can be used to substitute CP2K's 'libsmm' library. Prior to CP2K 3.0, only the [Intel-branch of CP2K](https://github.com/cp2k/cp2k/tree/intel) integrated LIBXS (see [https://github.com/hfp/libxs/raw/master/documentation/cp2k.pdf](https://github.com/hfp/libxs/raw/master/documentation/cp2k.pdf)).

**\[2]&#160;[https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, for simulating dynamic rupture processes. LIBXS provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see [https://github.com/TUM-I5/seissol_kernels/](https://github.com/TUM-I5/seissol_kernels/).

**\[3]&#160;[https://github.com/NekBox/NekBox](https://github.com/NekBox/NekBox)**: NekBox is a highly scalable and portable spectral element code, which is inspired by the [Nek5000](https://nek5000.mcs.anl.gov/) code. NekBox is specialized for box geometries, and intended to prototype new methods as well as to leverage FORTRAN beyond the FORTRAN&#160;77 standard. LIBXS can be used to substitute the [MXM_STD](https://github.com/Nek5000/NekBox/blob/box/mxm_std.F90) code. Please also note LIBXS's [NekBox reproducer](https://github.com/hfp/libxs/tree/master/samples/nek#nek-sample-collection).

**\[4]&#160;[https://github.com/Nek5000/Nek5000](https://github.com/Nek5000/Nek5000)**: Nek5000 is the open-source, highly-scalable, always-portable spectral element code from [https://nek5000.mcs.anl.gov/](https://nek5000.mcs.anl.gov/). The development branch of the Nek5000 code [incorporates](https://github.com/Nek5000/Nek5000/blob/develop/core/mxm_wrapper.f) LIBXS.

**\[5]&#160;[http://pyfr.org/](http://pyfr.org/)**: PyFR is an open-source Python based framework for solving advection-diffusion type problems on streaming architectures using the flux reconstruction approach. PyFR&#160;1.6.0 optionally [incorporates LIBXS](http://pyfr.org/user_guide.php) as a matrix multiplication provider for the OpenMP backend. Please also note LIBXS's [PyFR-related code sample](https://github.com/hfp/libxs/tree/master/samples/pyfr).

**\[6]&#160;[http://dial3343.org/about/](http://dial3343.org/about/)**: The Extreme-scale Discontinuous Galerkin Environment (EDGE) is a solver for hyperbolic partial differential equations with emphasis on seismic simulations. EDGE optionally uses LIBXS, but highly recommends the library due to severe performance-limitations of the vanilla kernels.

### Machine Learning (ML)

**\[7]&#160;[https://github.com/baidu-research/DeepBench](https://github.com/baidu-research/DeepBench#deepbench)**: The primary purpose of DeepBench is to benchmark operations that are important to deep learning on different hardware platforms. LIBXS's DNN primitives have been [incorporated into DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxs_conv) to demonstrate an increased performance of deep learning on Intel hardware. In addition, LIBXS's [DNN sample folder](https://github.com/hfp/libxs/tree/master/samples/dnn) contains scripts to run convolutions extracted from popular benchmarks in a stand-alone fashion.

**\[8]&#160;[https://www.tensorflow.org/](https://tensorflow.org/)**: TensorFlow&trade; is an open source software library for numerical computation using data flow graphs. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team for the purposes of conducting machine learning and deep neural networks research. LIBXS can be [used](documentation/tensorflow.md#tensorflow-with-libxs) to increase the performance of TensorFlow on Intel hardware.

**\[9]&#160;[https://github.com/IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe#skimcaffe-specific-description)**: SkimCaffe from Intel Labs is a Caffe branch for training of sparse CNNs, which provide 80-95% sparsity in convolutions and fully-connected layers. LIBXS's SPMDM domain (SParseMatrix-DenseMatrix multiplication) evolved from SkimCaffe, and since then LIBXS implements the sparse operations in SkimCaffe.

### Automated Driving (AD)

**\[10]&#160;[https://software.seek.intel.com/accelerating-eigen-math-library](https://software.seek.intel.com/accelerating-eigen-math-library)**: Accelerating The Eigen Math Library for Automated Driving Workloads: The Need for Speed in Kalman Filtering. An article in [Issue&#160;31](https://software.intel.com/sites/default/files/managed/4f/73/parallel-universe-issue-31.pdf) of The Parallel Universe magazine.

## References

**\[1]&#160;[http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153](http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153)**: LIBXS: Accelerating Small Matrix Multiplications by Runtime Code Generation ([paper](http://www.computer.org/csdl/proceedings/sc/2016/8815/00/8815a981.pdf)). SC'16: The International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City (Utah).

**\[2]&#160;[http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html)**: LIBXS: A High Performance Library for Small Matrix Multiplications ([poster](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf) and [abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file3.pdf)). SC'15: The International Conference for High Performance Computing, Networking, Storage and Analysis, Austin (Texas).

**\[3]&#160;[https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast](https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast)**: Intel Xeon&#160;Phi Delivers Competitive Performance For Deep Learning - And Getting Better Fast. Article mentioning LIBXS's performance of convolution kernels with [DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxs_conv). Intel Corporation, 2016.

