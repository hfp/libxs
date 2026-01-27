 Microbenchmark

This code sample benchmarks the performance of (1)&#160;the dispatch mechanism, and (2)&#160;the time needed to JIT-generate code for the first time. Both mechanisms are relevant when replacing GEMM calls (see [Call Wrapper](https://libxs.readthedocs.io/libxs_mm/#call-wrapper) section of the reference documentation), or in any case of calling LIBXS's native [GEMM functionality](https://libxs.readthedocs.io/libxs_mm/).

**Command Line Interface (CLI)**

* Optionally takes the number of dispatches/code-generations (default:&#160;10000).
* Optionally takes the number of threads (default:&#160;1).

**Measurements (Benchmark)**

* Duration of an empty function call (serves as a reference timing).
* Duration to find an already generated kernel (cached/non-cached).
* Duration to JIT-generate a GEMM kernel.

In case of a multi-threaded benchmark, the timings represent a highly contended request (worst case). For thread-scaling, it can be observed that read-only accesses (code dispatch) stay roughly with a constant duration whereas write-accesses (code generation) are serialized and hence the duration scales linearly with the number of threads.

The [Fortran example](https://github.com/hfp/libxs/blob/main/samples/utilities/dispatch/dispatch.f) (`dispatch.f`) could use `libxs_dmmdispatch` (or similar) like the C code (`dispatch.c`) but intentionally shows the lower-level dispatch interface `libxs_xmmdispatch` and also omits using the LIBXS module. Not using the module confirms: the same task can be achieved by relying only on FORTRAN&#160;77 language level.

## User-Data Dispatch

Further, another [Fortran example](https://github.com/hfp/libxs/blob/main/samples/utilities/dispatch/dispatch_udt.f) about [user-data dispatch](https://libxs.readthedocs.io/libxs_aux/#user-data-dispatch) is not exactly a benchmark. Dispatching user-data containing multiple kernels can obviously save multiple singular dispatches. The C interface for dispatching user-data is designed to follow the same flow as the Fortran interface.

