/**
 * Execute a generated function, inlined code, or fall back to the linked LAPACK implementation.
 * If M, N, and K does not change for multiple calls, it is more efficient to query and reuse
 * the function pointer (libxs_?mm_dispatch).
 */
#define LIBXS_MM(REAL, M, N, K, A, B, C) \
  if ((LIBXS_MAX_MNK) >= ((M) * (N) * (K))) { \
    const LIBXS_BLASPREC(libxs_, REAL, mm_function) libxs_mm_function_ = \
      LIBXS_BLASPREC(libxs_, REAL, mm_dispatch)((M), (N), (K)); \
    if (libxs_mm_function_) { \
      libxs_mm_function_((A), (B), (C)); \
    } \
    else { \
      LIBXS_IMM(REAL, int, M, N, K, A, B, C); \
    } \
  } \
  else { \
    LIBXS_BLASMM(REAL, int, M, N, K, A, B, C); \
  }

/** Type of a function generated for a specific M, N, and K. */
typedef LIBXS_TARGET(mic) void (*libxs_smm_function)(const float*, const float*, float*);
typedef LIBXS_TARGET(mic) void (*libxs_dmm_function)(const double*, const double*, double*);

/** Query the pointer of a generated function; zero if it does not exist. */
LIBXS_EXTERN_C LIBXS_TARGET(mic) libxs_smm_function libxs_smm_dispatch(int m, int n, int k);
LIBXS_EXTERN_C LIBXS_TARGET(mic) libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k);

/** Dispatched matrix-matrix multiplication; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_smm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_MM(float, m, n, k, a, b, c);
}

/** Dispatched matrix-matrix multiplication; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_MM(double, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_simm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_IMM(float, int, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dimm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_IMM(double, int, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; single-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_sblasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(float, int, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; double-precision. */
LIBXS_INLINE LIBXS_TARGET(mic) void libxs_dblasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(double, int, m, n, k, a, b, c);
}

#if defined(__cplusplus)

/** Dispatched matrix-matrix multiplication. */
LIBXS_TARGET(mic) inline void libxs_mm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)        { libxs_smm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_mm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)     { libxs_dmm(m, n, k, a, b, c); }

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_TARGET(mic) inline void libxs_imm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)       { libxs_simm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_imm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)    { libxs_dimm(m, n, k, a, b, c); }

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_TARGET(mic) inline void libxs_blasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)    { libxs_sblasmm(m, n, k, a, b, c); }
LIBXS_TARGET(mic) inline void libxs_blasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) { libxs_dblasmm(m, n, k, a, b, c); }

/** Call libxs_smm_dispatch, or libxs_dmm_dispatch depending on T. */
template<typename T> class LIBXS_TARGET(mic) libxs_mm_dispatch { typedef void function_type; };

template<> class LIBXS_TARGET(mic) libxs_mm_dispatch<float> {
  typedef libxs_smm_function function_type;
  mutable/*target:mic*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_smm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

template<> class LIBXS_TARGET(mic) libxs_mm_dispatch<double> {
  typedef libxs_dmm_function function_type;
  mutable/*target:mic*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_dmm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

#endif /*__cplusplus*/
