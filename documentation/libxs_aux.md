# Target Architecture<a name="getting-and-setting-the-target-architecture"></a>

This functionality is available for the C and Fortran interface. There are [ID based](https://github.com/hfp/libxs/blob/main/include/libxs_cpuid.h#L47) (same for C and Fortran) and string based functions to query the code path (as determined by the CPUID), or to set the code path regardless of the presented CPUID features. The latter may degrade performance if a lower set of instruction set extensions is requested, which can be still useful for studying the performance impact of different instruction set extensions.  
**Note**: There is no additional check performed if an unsupported instruction set extension is requested, and incompatible JIT-generated code may be executed (unknown instruction signaled).

```C
int libxs_get_target_archid(void);
void libxs_set_target_archid(int id);

const char* libxs_get_target_arch(void);
void libxs_set_target_arch(const char* arch);
```

Available code paths (IDs and corresponding strings):

* LIBXS_TARGET_ARCH_GENERIC: "**generic**", "none", "0"
* LIBXS_X86_GENERIC: "**x86**", "x64", "sse2"
* LIBXS_X86_SSE3: "**sse3**"
* LIBXS_X86_SSE42: "**wsm**", "nhm", "sse4", "sse4_2", "sse4.2"
* LIBXS_X86_AVX: "**snb**", "avx"
* LIBXS_X86_AVX2: "**hsw**", "avx2"
* LIBXS_X86_AVX512_MIC: "**knl**", "mic"
* LIBXS_X86_AVX512_KNM: "**knm**"
* LIBXS_X86_AVX512_CORE: "**skx**", "skl", "avx3", "avx512"
* LIBXS_X86_AVX512_CLX: "**clx**"
* LIBXS_X86_AVX512_CPX: "**cpx**"
* LIBXS_X86_AVX512_SPR: "**spr**"

The **bold** names are returned by `libxs_get_target_arch` whereas `libxs_set_target_arch` accepts all of the above strings (similar to the environment variable LIBXS_TARGET).

### Verbosity Level<a name="getting-and-setting-the-verbosity"></a>

The [verbose mode](index.md#verbose-mode) (level of verbosity) can be controlled using the C or Fortran API, and there is an environment variable which corresponds to `libxs_set_verbosity` (LIBXS_VERBOSE).

```C
int libxs_get_verbosity(void);
void libxs_set_verbosity(int level);
```

### Timer Facility

Due to the performance oriented nature of LIBXS, timer-related functionality is available for the C and Fortran interface ([libxs_timer.h](https://github.com/hfp/libxs/blob/main/include/libxs_timer.h#L37) and [libxs.f](https://github.com/hfp/libxs/blob/main/include/libxs.f#L32)). The timer is used in many of the [code samples](https://github.com/hfp/libxs/tree/main/samples) to measure the duration of executing a region of the code. The timer is based on a monotonic clock tick, which uses a platform-specific resolution. The counter may rely on the time stamp counter instruction (RDTSC), which is not necessarily counting CPU cycles (reasons are out of scope in this context). However, `libxs_timer_ncycles` delivers raw clock ticks (RDTSC).

```C
typedef unsigned long long libxs_timer_tickint;
libxs_timer_tickint libxs_timer_tick(void);
double libxs_timer_duration(
  libxs_timer_tickint tick0,
  libxs_timer_tickint tick1);
libxs_timer_tickint libxs_timer_ncycles(
  libxs_timer_tickint tick0,
  libxs_timer_tickint tick1);
```

### User-Data Dispatch

To register a user-defined key-value pair with LIBXS's fast key-value store, the key must be binary reproducible. Structured key-data (`struct` or `class` type which can be padded in a compiler-specific fashion) must be completely cleared, i.e., all gaps may be zero-filled before initializing data members (`memset(&mykey, 0, sizeof(mykey))`). This is because some compilers can leave padded data uninitialized, which breaks binary reproducible keys, hence the flow is: clear heterogeneous keys (struct), initialize data-members, and register. The size of the key is arbitrary but limited to LIBXS_DESCRIPTOR_MAXSIZE (96 Byte), and the size of the value can be of an arbitrary size. The given value is copied and may be initialized at registration-time or when dispatched. Registered data is released at program termination but can be manually unregistered and released (`libxs_xrelease`), e.g., to register a larger value for an existing key.

```C
void* libxs_xregister(const void* key, size_t key_size, size_t value_size, const void* value_init);
void* libxs_xdispatch(const void* key, size_t key_size);
```

The Fortran interface is designed to follow the same flow as the <span>C&#160;language</span>: <span>(1)&#160;</span>`libxs_xdispatch` is used to query the value, and <span>(2)&#160;if</span> the value is a NULL-pointer, it is registered per `libxs_xregister`. Similar to C (`memset`), structured key-data must be zero-filled (`libxs_xclear`) even when followed by an element-wise initialization. A key based on a contiguous array has no gaps by definition and it is enough to initialize the array elements. A [Fortran example](https://github.com/hfp/libxs/blob/main/samples/utilities/dispatch/dispatch_udt.f) is given as part of the [Dispatch Microbenchmark](https://github.com/hfp/libxs/tree/main/samples/utilities/dispatch).

```Fortran
FUNCTION libxs_xregister(key, keysize, valsize, valinit)
  TYPE(C_PTR), INTENT(IN), VALUE :: key
  TYPE(C_PTR), INTENT(IN), VALUE, OPTIONAL :: valinit
  INTEGER(C_INT), INTENT(IN) :: keysize, valsize
  TYPE(C_PTR) :: libxs_xregister
END FUNCTION

FUNCTION libxs_xdispatch(key, keysize)
  TYPE(C_PTR), INTENT(IN), VALUE :: key
  INTEGER(C_INT), INTENT(IN) :: keysize
  TYPE(C_PTR) :: libxs_xdispatch
END FUNCTION
```

**Note**: This functionality can be used to, e.g., dispatch multiple kernels in one step if a code location relies on multiple kernels. This way, one can pay the cost of dispatch one time per task rather than according to the number of JIT-kernels used by this task. However, the functionality is not limited to multiple kernels, but any data can be registered and queried. User-data dispatch uses the same implementation as regular code-dispatch.

### Memory Allocation

The C interface ([libxs_malloc.h](https://github.com/hfp/libxs/blob/main/include/libxs_malloc.h)) provides functions for aligned memory one of which allows to specify the alignment (or to request an automatically selected alignment). The automatic alignment is also available with a `malloc` compatible signature. The size of the automatic alignment depends on a heuristic, which uses the size of the requested buffer.  
**Note**: The function `libxs_free` must be used to deallocate buffers allocated by LIBXS's allocation functions.

```C
void* libxs_malloc(size_t size);
void* libxs_aligned_malloc(size_t size, size_t alignment);
void* libxs_aligned_scratch(size_t size, size_t alignment);
void libxs_free(const volatile void* memory);
int libxs_get_malloc_info(const void* m, libxs_malloc_info* i);
int libxs_get_scratch_info(libxs_scratch_info* info);
```

The library exposes two memory allocation domains: <span>(1)&#160;default</span> memory allocation, and <span>(2)&#160;scratch</span> memory allocation. There are similar service functions for both domains that allow to customize the allocation and deallocation function. The "context form" even supports a user-defined "object", which may represent an allocator or any other external facility. To set the allocator of the default domain is analogous to setting the allocator of the scratch memory domain (shown below).

```C
int libxs_set_scratch_allocator(void* context,
  libxs_malloc_function malloc_fn, libxs_free_function free_fn);
int libxs_get_scratch_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);
```

The scratch memory allocation is very effective and delivers a decent speedup over subsequent regular memory allocations. In contrast to the default allocator, a watermark for repeatedly allocated and deallocated buffers is established. The scratch memory domain is (arbitrarily) limited to <span>4&#160;GB</span> of memory which can be adjusted to a different number of Bytes (available per [libxs_malloc.h](https://github.com/hfp/libxs/blob/main/include/libxs_malloc.h), and also per environment variable LIBXS_SCRATCH_LIMIT with optional "k|K", "m|M", "g|G" units, unlimited per "-1").

```C
void libxs_set_scratch_limit(size_t nbytes);
size_t libxs_get_scratch_limit(void);
```

By establishing a pool of "temporary" memory, the cost of repeated allocation and deallocation cycles is avoided when the watermark is reached. The scratch memory is scope-oriented with a limited number of pools for buffers of different lifetime or held for different threads. The [verbose mode](index.md#verbose-mode) with a verbosity level of at least two (LIBXS_VERBOSE=2) shows some statistics about the populated scratch memory.

```bash
Scratch: 173 MB (mallocs=5, pools=1)
```

To improve thread-scalability and to avoid frequent memory allocation/deallocation, the scratch memory allocator can be leveraged by [intercepting existing malloc/free calls](libxs_tune.md#intercepted-allocations).

**Note**: be careful with scratch memory as it only grows during execution (in between `libxs_init` and `libxs_finalize` unless `libxs_release_scratch` is called). This is true even when `libxs_free` is (and should be) used!

### Meta Image File I/O

Loading and storing data (I/O) is normally out of LIBXS's scope. However, comparing results (correctness) or writing files for visual inspection is clearly desired. This is particularly useful for the DNN domain. The MHD library domain provides support for the Meta Image File format (MHD). Tools such as [ITK-SNAP](http://itksnap.org/) or [ParaView](https://www.paraview.org/) can be used to inspect, compare, and modify images (even beyond two-dimensional images).

Writing an image is per `libxs_mhd_write`, and loading an image is split in two stages: <span>(1)&#160;</span>`libxs_mhd_read_header`, and <span>(2)&#160;</span>`libxs_mhd_read`. The first step allows to allocate a properly sized buffer, which is then used to obtain the data per `libxs_mhd_read`. When reading data, an on-the-fly type conversion is supported. Further, data that is already in memory can be compared against file-data without allocating memory or reading this file into memory.

To load an image from a familiar format (JPG, PNG, etc.), one may save the raw data using for instance [IrfanView](http://www.irfanview.com/) and rely on a "header-only" MHD-file (plain text). This may look like:

```ini
NDims = 2
DimSize = 202 134
ElementType = MET_UCHAR
ElementNumberOfChannels = 1
ElementDataFile = mhd_image.raw
```

In the above case, a single channel (gray-scale) 202x134-image is described with pixel data stored separately (`mhd_image.raw`). Multi-channel images are expected to interleave the pixel data. The pixel type is per `libxs_mhd_elemtype` ([libxs_mhd.h](https://github.com/hfp/libxs/blob/main/include/libxs_mhd.h#L38)).

### Thread Synchronization

LIBXS comes with a number of light-weight abstraction layers (macro and API-based), which are distinct from the internal API (include files in [src](https://github.com/hfp/libxs/tree/main/src) directory) and that are exposed for general use (and hence part of the [include](https://github.com/hfp/libxs/tree/main/include) directory).

The synchronization layer is mainly based on macros: LIBXS_LOCK_\* provide spin-locks, mutexes, and reader-writer locks (LIBXS_LOCK_SPINLOCK, LIBXS_LOCK_MUTEX, and LIBXS_LOCK_RWLOCK respectively). Usually the spin-lock is also named LIBXS_LOCK_DEFAULT. The implementation is intentionally based on OS-native primitives unless LIBXS is reconfigured (per LIBXS_LOCK_SYSTEM) or built using `make OMP=1` (using OpenMP inside of the library is not recommended). The life cycle of a lock looks like:

```C
/* attribute variable and lock variable */
LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK_DEFAULT) attr;
LIBXS_LOCK_TYPE(LIBXS_LOCK_DEFAULT) lock;
/* attribute initialization */
LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK_DEFAULT, &attr);
/* lock initialization per initialized attribute */
LIBXS_LOCK_INIT(LIBXS_LOCK_DEFAULT, &lock, &attr);
/* the attribute can be destroyed */
LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK_DEFAULT, &attr);
/* lock destruction (usage: see below/next code block) */
LIBXS_LOCK_DESTROY(LIBXS_LOCK_DEFAULT, &lock);
```

Once the lock is initialized (or an array of locks), it can be exclusively locked or try-locked, and released at the end of the locked section (LIBXS_LOCK_ACQUIRE, LIBXS_LOCK_TRYLOCK, and LIBXS_LOCK_RELEASE respectively):

```C
LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, &lock);
/* locked code section */
LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, &lock);
```

If the lock-kind is LIBXS_LOCK_RWLOCK, non-exclusive a.k.a. shared locking allows to permit multiple readers (LIBXS_LOCK_ACQREAD, LIBXS_LOCK_TRYREAD, and LIBXS_LOCK_RELREAD) if the lock is not acquired exclusively (see above). An attempt to only read-lock anything else but an RW-lock is an exclusive lock (see above).

```C
if (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK) ==
    LIBXS_LOCK_TRYREAD(LIBXS_LOCK_RWLOCK, &rwlock))
{ /* locked code section */
  LIBXS_LOCK_RELREAD(LIBXS_LOCK_RWLOCK, &rwlock);
}
```

Locking different sections for read (LIBXS_LOCK_ACQREAD, LIBXS_LOCK_RELREAD) and write (LIBXS_LOCK_ACQUIRE, LIBXS_LOCK_RELEASE) may look like:

```C
LIBXS_LOCK_ACQREAD(LIBXS_LOCK_RWLOCK, &rwlock);
/* locked code section: only reads are performed */
LIBXS_LOCK_RELREAD(LIBXS_LOCK_RWLOCK, &rwlock);

LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_RWLOCK, &rwlock);
/* locked code section: exclusive write (no R/W) */
LIBXS_LOCK_RELEASE(LIBXS_LOCK_RWLOCK, &rwlock);
```

For a lock not backed by an OS level primitive (fully featured lock), the synchronization layer also a simple lock based on atomic operations:

```C
static union { char pad[LIBXS_CACHELINE]; volatile LIBXS_ATOMIC_LOCKTYPE state; } lock;
LIBXS_ATOMIC_ACQUIRE(&lock.state, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_RELAXED);
/* locked code section */
LIBXS_ATOMIC_RELEASE(&lock.state, LIBXS_ATOMIC_RELAXED);
```
