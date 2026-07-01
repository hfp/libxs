/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_timer.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

static const char input_names[] = "M,N,K";
static const char output_names[] =
  "BS,BM,BN,BK,WS,WG,LU,NZ,AL,TB,TC,AP,AA,AB,AC,XF";

enum { NINPUTS = 3, NOUTPUTS = 16 };

static const int confidence_outputs[] = { 5, 6, 8, 12, 13 };

static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal);
static int write_confidence_maps(const char* prefix, const void* buffer,
  size_t size, const libxs_predict_t* reference, int ntotal);
static double deployment_confidence(const libxs_predict_info_t* info);


int main(int argc, char* argv[])
{
  int argi = 1, mode = LIBXS_PREDICT_AUTO, use_rf = 0, use_hknn = 0;
  int order_arg = 0;
  double quality = 0, smooth = 0, consistency = 0;
  double eval_fraction = 0.8;
  const char *filename, *modelfile, *confidence_prefix;
  int result = EXIT_FAILURE;
  if (argi < argc && '0' <= argv[argi][0] && '9' >= argv[argi][0]) {
    eval_fraction = atof(argv[argi]);
    ++argi;
  }
  while (argi < argc && 'a' <= argv[argi][0] && argv[argi][0] <= 'z') {
    if ('a' == argv[argi][0]) mode = LIBXS_PREDICT_AUTO;
    else if ('c' == argv[argi][0] && 'a' == argv[argi][1]) {
      mode = LIBXS_PREDICT_CLASSIFY;
    }
    else if ('c' == argv[argi][0] && 'o' == argv[argi][1]
      && 'n' == argv[argi][2])
    {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p) ++p;
      consistency = ('\0' != *p) ? atof(p) : 0.9;
    }
    else if ('c' == argv[argi][0] && 'o' == argv[argi][1]) {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p) ++p;
      quality = ('\0' != *p) ? atof(p) : 0.9;
    }
    else if ('i' == argv[argi][0]) mode = LIBXS_PREDICT_INTERPOLATE;
    else if ('r' == argv[argi][0]) use_rf = 1;
    else if ('h' == argv[argi][0]) use_hknn = 1;
    else if ('s' == argv[argi][0]) {
      const char* p = argv[argi];
      while ('\0' != *p && (*p < '0' || *p > '9') && '.' != *p
        && '-' != *p) ++p;
      smooth = ('\0' != *p) ? atof(p) : -1.0;
    }
    else break;
    ++argi;
  }
  if (argi < argc && '-' == argv[argi][0] && '\0' != argv[argi][1]) {
    order_arg = atoi(argv[argi]);
    ++argi;
  }
  filename = (argi < argc) ? argv[argi] : NULL;
  modelfile = (argi + 1 < argc) ? argv[argi + 1] : NULL;
  confidence_prefix = (argi + 2 < argc) ? argv[argi + 2] : NULL;
  { static char modelpath[512];
    if (NULL == modelfile && NULL != filename) {
      const char* sep = strrchr(filename, '/');
      const char* base = (NULL != sep) ? (sep + 1) : filename;
      const char* dot = strrchr(base, '.');
      size_t len = (NULL != dot) ? (size_t)(dot - base) : strlen(base);
      if (len >= sizeof(modelpath) - 4) len = sizeof(modelpath) - 5;
      memcpy(modelpath, base, len);
      memcpy(modelpath + len, ".bin", 5);
      modelfile = modelpath;
    }
  }
  if (NULL == filename) {
    fprintf(stdout,
      "Usage: %s [fraction] [auto|cat|compress[Q]|consist[C]|interp|rf|hknn|smooth[A]]"
      " [-N] <csvfile> [modelfile [confidence-prefix]]\n"
      "  fraction: validation split 0..1 for quality report (default: 0.8)\n"
      "  auto:     auto-detect mode per output (default)\n"
      "  cat:      force categorical (kNN) for all outputs\n"
      "  compress: drop predictable entries (Q: threshold, default 0.9)\n"
      "  consist:  round-trip consistency penalty (C: 0..1, default 0.9)\n"
      "  interp:   force interpolation for all outputs\n"
      "  rf:       Random Forest classification\n"
      "  hknn:     hierarchical kNN (Fisher-guided partition)\n"
      "  smooth:   multi-cluster blending (A: radius or -1=auto, default: auto)\n"
      "  -N: max polynomial order (default: 0 = auto)\n"
      "  confidence-prefix: optional prefix for saved-model confidence maps\n"
      "  Trains on all entries, saves the model, and reports\n"
      "  quality on a held-out validation set.\n", argv[0]);
  }
  else {
    libxs_predict_t* source = libxs_predict_create(NINPUTS, NOUTPUTS);
    if (NULL != source) {
      const int ntotal = libxs_predict_load_csv(source, filename, NULL,
        input_names, output_names, NULL, 0, NULL);
      if (0 < ntotal) {
        libxs_predict_t* model = libxs_predict_create(NINPUTS, NOUTPUTS);
        fprintf(stdout, "Loaded %d entries from %s\n", ntotal, filename);
        if (NULL != model) {
          libxs_timer_tick_t tick;
          int i, build_ok = EXIT_FAILURE;
          double inputs[NINPUTS], outputs[NOUTPUTS], dt_build;
          libxs_predict_set_mode(model, mode);
          if (0 != use_rf) libxs_predict_set_decompose(model, LIBXS_PREDICT_RF);
          else if (0 != use_hknn) libxs_predict_set_decompose(model, LIBXS_PREDICT_HKNN);
          if (0.0 != smooth) libxs_predict_set_smooth(model, smooth);
          if (0.0 != consistency) libxs_predict_set_consistency(model, consistency);
          for (i = 0; i < ntotal; ++i) {
            libxs_predict_get(source, i, inputs, outputs);
            libxs_predict_push(NULL, model, inputs, outputs);
          }
          tick = libxs_timer_tick();
#if defined(_OPENMP)
#         pragma omp parallel
          { const int br = libxs_predict_build_task(NULL, model, 0, order_arg,
              quality, omp_get_thread_num(), omp_get_num_threads());
            if (0 == omp_get_thread_num()) build_ok = br;
          }
#else
          build_ok = libxs_predict_build_task(NULL, model, 0, order_arg,
            quality, 0, 1);
#endif
          dt_build = libxs_timer_duration(tick, libxs_timer_tick());
          if (EXIT_SUCCESS == build_ok) {
            { libxs_predict_query_t qi;
              memset(&qi, 0, sizeof(qi));
              libxs_predict_query(model, &qi);
              fprintf(stdout, "Built: %d clusters, %.1fx compression, order=%d"
                " (%d entries, %.2f s)\n", qi.nclusters, qi.compression,
                qi.order, qi.nentries, dt_build);
            }
            { const int nval = LIBXS_MAX(
                (int)(ntotal * eval_fraction + 0.5), 1);
              libxs_predict_t* val_model =
                libxs_predict_create(NINPUTS, NOUTPUTS);
              if (NULL != val_model) {
                double vi[NINPUTS], vo[NOUTPUTS];
                libxs_predict_set_mode(val_model, mode);
                if (0 != use_rf) {
                  libxs_predict_set_decompose(val_model, LIBXS_PREDICT_RF);
                }
                else if (0 != use_hknn) {
                  libxs_predict_set_decompose(val_model, LIBXS_PREDICT_HKNN);
                }
                if (0.0 != smooth) libxs_predict_set_smooth(val_model, smooth);
                if (0.0 != consistency) {
                  libxs_predict_set_consistency(val_model, consistency);
                }
                for (i = 0; i < nval; ++i) {
                  libxs_predict_get(source, i, vi, vo);
                  libxs_predict_push(NULL, val_model, vi, vo);
                }
                if (EXIT_SUCCESS == libxs_predict_build(
                  val_model, 0, order_arg, quality))
                {
                  evaluate(val_model, source, ntotal);
                }
                libxs_predict_destroy(val_model);
              }
            }
            { size_t size = 0;
              void* buffer;
              libxs_predict_save(model, NULL, &size);
              buffer = malloc(size);
              if (NULL != buffer) {
                if (EXIT_SUCCESS == libxs_predict_save(model, buffer, &size)) {
                  FILE* out = fopen(modelfile, "wb");
                  if (NULL != out) {
                    fwrite(buffer, 1, size, out);
                    fclose(out);
                    fprintf(stdout, "Saved model to %s (%lu bytes)\n",
                      modelfile, (unsigned long)size);
                    if (NULL == confidence_prefix || EXIT_SUCCESS ==
                      write_confidence_maps(confidence_prefix, buffer, size,
                        source, ntotal))
                    {
                      result = EXIT_SUCCESS;
                    }
                  }
                }
                free(buffer);
              }
            }
          }
          libxs_predict_destroy(model);
        }
      }
      else {
        fprintf(stderr, "Failed to load entries from %s\n", filename);
      }
      libxs_predict_destroy(source);
    }
  }
  return result;
}


static void evaluate(const libxs_predict_t* model,
  const libxs_predict_t* reference, int ntotal)
{
  double* all_inputs = (double*)malloc((size_t)ntotal * NINPUTS * sizeof(double));
  double* all_predicted = (double*)malloc((size_t)ntotal * NOUTPUTS * sizeof(double));
  if (NULL != all_inputs && NULL != all_predicted) {
    double maxerr[NOUTPUTS] = { 0 }, sumerr[NOUTPUTS] = { 0 };
    libxs_timer_tick_t tick;
    double dt_eval;
    int i, j;
    for (i = 0; i < ntotal; ++i) {
      libxs_predict_get(reference, i, all_inputs + (size_t)i * NINPUTS, NULL);
    }
    tick = libxs_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel
    { const int tid = omp_get_thread_num(), ntasks = omp_get_num_threads();
      libxs_predict_eval_batch_task(model, all_inputs, all_predicted,
        ntotal, 1, tid, ntasks);
    }
#else
    libxs_predict_eval_batch(model, all_inputs, all_predicted, ntotal, 1);
#endif
    dt_eval = libxs_timer_duration(tick, libxs_timer_tick());
    for (i = 0; i < ntotal; ++i) {
      double expected[NOUTPUTS];
      libxs_predict_get(reference, i, NULL, expected);
      for (j = 0; j < NOUTPUTS; ++j) {
        const double err = LIBXS_DELTA(
          all_predicted[(size_t)i * NOUTPUTS + j], expected[j]);
        sumerr[j] += err;
        if (err > maxerr[j]) maxerr[j] = err;
      }
    }
    fprintf(stdout, "Validation (%d samples):\n", ntotal);
    fprintf(stdout, "  param   avg-err   max-err\n");
    for (j = 0; j < NOUTPUTS; ++j) {
      int len = 0;
      const char* name = libxs_strtoken(output_names, ",", j, &len);
      fprintf(stdout, "  %-4.*s  %9.2e %9.2e\n",
        len, name,
        (0 < ntotal) ? (sumerr[j] / ntotal) : 0.0, maxerr[j]);
    }
    fprintf(stdout, "Eval: %d queries (%.2f s)\n", ntotal, dt_eval);
    { const int nconf = (int)(sizeof(confidence_outputs)
        / sizeof(confidence_outputs[0]));
      const double threshold = 0.9;
      int gated_correct[5] = {0}, gated_wrong[5] = {0}, deferred[5] = {0};
      int ci;
      for (i = 0; i < ntotal; ++i) {
        double expected[NOUTPUTS];
        libxs_predict_info_t info;
        libxs_predict_eval(NULL, model,
          all_inputs + (size_t)i * NINPUTS, NULL, &info, 1);
        libxs_predict_get(reference, i, NULL, expected);
        for (ci = 0; ci < nconf; ++ci) {
          const int oi = confidence_outputs[ci];
          if (NULL != info.confidence && info.confidence[oi] >= threshold) {
            if (NULL != info.values
              && LIBXS_ROUNDX(int, info.values[oi]) == (int)expected[oi])
            {
              ++gated_correct[ci];
            }
            else {
              ++gated_wrong[ci];
            }
          }
          else {
            ++deferred[ci];
          }
        }
      }
      fprintf(stdout, "Gated deployment (threshold=%.1f):\n", threshold);
      fprintf(stdout,
        "  param  correct  wrong  deferred  coverage  precision\n");
      for (ci = 0; ci < nconf; ++ci) {
        const int oi = confidence_outputs[ci];
        const int acted = gated_correct[ci] + gated_wrong[ci];
        int len = 0;
        const char* name = libxs_strtoken(output_names, ",", oi, &len);
        fprintf(stdout, "  %-4.*s  %5d  %5d  %5d     %5.1f%%    %5.1f%%\n",
          len, name, gated_correct[ci], gated_wrong[ci], deferred[ci],
          (ntotal > 0) ? 100.0 * acted / ntotal : 0.0,
          (acted > 0) ? 100.0 * gated_correct[ci] / acted : 100.0);
      }
    }
  }
  free(all_inputs);
  free(all_predicted);
}


static int write_confidence_maps(const char* prefix, const void* buffer,
  size_t size, const libxs_predict_t* reference, int ntotal)
{
  libxs_predict_t* model = libxs_predict_load(buffer, size);
  libxs_timer_tick_t tick;
  double inputs[NINPUTS];
  float *cube = NULL, *min_k = NULL, *mean_k = NULL;
  size_t ext3[3], ext2[2], nelements, mi, ni, ki, nm, nn, nk;
  int mmin, nmin, kmin, mmax, nmax, kmax, i;
  int result = EXIT_FAILURE;
  if (NULL != model && 0 < ntotal) {
    libxs_predict_get(reference, 0, inputs, NULL);
    mmin = mmax = (int)inputs[0];
    nmin = nmax = (int)inputs[1];
    kmin = kmax = (int)inputs[2];
    for (i = 1; i < ntotal; ++i) {
      int m_value, n_value, k_value;
      libxs_predict_get(reference, i, inputs, NULL);
      m_value = (int)inputs[0];
      n_value = (int)inputs[1];
      k_value = (int)inputs[2];
      if (m_value < mmin) mmin = m_value;
      if (mmax < m_value) mmax = m_value;
      if (n_value < nmin) nmin = n_value;
      if (nmax < n_value) nmax = n_value;
      if (k_value < kmin) kmin = k_value;
      if (kmax < k_value) kmax = k_value;
    }
    nm = (size_t)(mmax - mmin + 1);
    nn = (size_t)(nmax - nmin + 1);
    nk = (size_t)(kmax - kmin + 1);
    nelements = nm * nn * nk;
    cube = (float*)malloc(nelements * sizeof(float));
    min_k = (float*)malloc(nm * nn * sizeof(float));
    mean_k = (float*)malloc(nm * nn * sizeof(float));
    if (NULL != cube && NULL != min_k && NULL != mean_k) {
      tick = libxs_timer_tick();
      for (ni = 0; ni < nn; ++ni) {
        for (mi = 0; mi < nm; ++mi) {
          const size_t index2 = ni * nm + mi;
          float min_value = 1.0f, sum_value = 0.0f;
          for (ki = 0; ki < nk; ++ki) {
            libxs_predict_info_t info;
            double confidence;
            const size_t index3 = (ki * nn + ni) * nm + mi;
            inputs[0] = (double)(mmin + (int)mi);
            inputs[1] = (double)(nmin + (int)ni);
            inputs[2] = (double)(kmin + (int)ki);
            libxs_predict_eval(NULL, model, inputs, NULL, &info, 1);
            confidence = deployment_confidence(&info);
            cube[index3] = (float)confidence;
            if (confidence < min_value) min_value = (float)confidence;
            sum_value += (float)confidence;
          }
          min_k[index2] = min_value;
          mean_k[index2] = sum_value / (float)nk;
        }
      }
      { char filename[1024];
        libxs_mhd_info_t info3 = { 3, 1, LIBXS_DATATYPE_F32, 0 };
        libxs_mhd_info_t info2 = { 2, 1, LIBXS_DATATYPE_F32, 0 };
        ext3[0] = nm;
        ext3[1] = nn;
        ext3[2] = nk;
        ext2[0] = nm;
        ext2[1] = nn;
        LIBXS_SNPRINTF(filename, sizeof(filename), "%s_cube.mhd", prefix);
        result = libxs_mhd_write(filename, NULL, ext3, ext3, &info3, cube, NULL);
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence cube to %s\n", filename);
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s_minK.mhd", prefix);
          result = libxs_mhd_write(filename, NULL, ext2, ext2, &info2,
            min_k, NULL);
        }
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence minK projection to %s\n", filename);
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s_meanK.mhd", prefix);
          info2.header_size = 0;
          result = libxs_mhd_write(filename, NULL, ext2, ext2, &info2,
            mean_k, NULL);
        }
        if (EXIT_SUCCESS == result) {
          fprintf(stdout, "Saved confidence meanK projection to %s\n", filename);
          fprintf(stdout, "Confidence maps: M=%d..%d N=%d..%d K=%d..%d"
            " (%lu queries, %.2f s)\n", mmin, mmax, nmin, nmax, kmin, kmax,
            (unsigned long)nelements,
            libxs_timer_duration(tick, libxs_timer_tick()));
        }
      }
    }
  }
  free(mean_k);
  free(min_k);
  free(cube);
  libxs_predict_destroy(model);
  return result;
}


static double deployment_confidence(const libxs_predict_info_t* info)
{
  double result = 0.0;
  int i;
  if (NULL != info && NULL != info->confidence) {
    result = 1.0;
    for (i = 0; i < (int)(sizeof(confidence_outputs) /
      sizeof(confidence_outputs[0])); ++i)
    {
      const int output = confidence_outputs[i];
      if (output < info->noutputs && info->confidence[output] < result) {
        result = info->confidence[output];
      }
    }
  }
  return result;
}
