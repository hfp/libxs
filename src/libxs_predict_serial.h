/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * How a block of doubles is stored. The corpus is most of a model file and its
 * columns carry few values each - a tuned parameter takes one of a handful, and
 * several take exactly one - so storing every entry as eight bytes stores the
 * same number over and over. The tag says which form follows, out of band rather
 * than as a value inside the data: an in-band marker cannot be told from a datum
 * that happens to equal it, and NaN is already meaningful here.
 */
#define INTERNAL_LIBXS_PREDICT_DENC_RAW 0
#define INTERNAL_LIBXS_PREDICT_DENC_CONST 1
#define INTERNAL_LIBXS_PREDICT_DENC_DICT 2
/* a one-byte index addresses this many, which also bounds the distinct scan */
#define INTERNAL_LIBXS_PREDICT_DENC_DICTMAX 256


/**
 * Writes count values read with the given stride, in the smallest of the three
 * forms, and returns the bytes written. NEVER more than the verbatim form takes:
 * the dictionary is chosen only where it is actually smaller, which is what lets
 * the size query remain a plain upper bound instead of a second copy of this
 * decision that has to agree with it. With dst NULL nothing is written and the
 * size is returned, for a size query that has to be exact.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_denc_write(unsigned char* dst,
  const double* values, size_t count, size_t stride)
{
  double dict[INTERNAL_LIBXS_PREDICT_DENC_DICTMAX];
  unsigned char* const start = dst;
  size_t i, ndict = 0, result;
  int tag = INTERNAL_LIBXS_PREDICT_DENC_RAW;
  if (0 != count) {
    for (i = 0; i < count && 0 != (ndict + 1); ++i) {
      const double v = values[i * stride];
      size_t d = 0;
      while (d < ndict && dict[d] != v) ++d;
      if (d == ndict) {
        if (INTERNAL_LIBXS_PREDICT_DENC_DICTMAX == ndict) {
          ndict = (size_t)-1; /* too many to index in a byte: verbatim */
        }
        else dict[ndict++] = v;
      }
    }
    if ((size_t)-1 != ndict) {
      if (1 == ndict) tag = INTERNAL_LIBXS_PREDICT_DENC_CONST;
      /* the dictionary pays for itself only past its own header */
      else if ((3 + ndict * sizeof(double) + count)
        < (1 + count * sizeof(double)))
      {
        tag = INTERNAL_LIBXS_PREDICT_DENC_DICT;
      }
    }
  }
  if (NULL == dst) {
    result = 1 + ((INTERNAL_LIBXS_PREDICT_DENC_CONST == tag) ? sizeof(double)
      : ((INTERNAL_LIBXS_PREDICT_DENC_DICT == tag)
        ? (2 + ndict * sizeof(double) + count) : (count * sizeof(double))));
  }
  else {
    *dst++ = (unsigned char)tag;
    if (INTERNAL_LIBXS_PREDICT_DENC_CONST == tag) {
      memcpy(dst, dict, sizeof(double));
      dst += sizeof(double);
    }
    else if (INTERNAL_LIBXS_PREDICT_DENC_DICT == tag) {
      const uint16_t n16 = (uint16_t)ndict;
      memcpy(dst, &n16, 2);
      dst += 2;
      memcpy(dst, dict, ndict * sizeof(double));
      dst += ndict * sizeof(double);
      for (i = 0; i < count; ++i) {
        const double v = values[i * stride];
        size_t d = 0;
        while (dict[d] != v) ++d;
        *dst++ = (unsigned char)d;
      }
    }
    else for (i = 0; i < count; ++i) {
      const double v = values[i * stride];
      memcpy(dst, &v, sizeof(double));
      dst += sizeof(double);
    }
    result = (size_t)(dst - start);
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_crc(const void* buffer,
  size_t size, uint32_t* crc)
{
  int result = EXIT_SUCCESS;
  if (size > (size_t)0xFFFFFFFFU) result = EXIT_FAILURE;
  else *crc = (uint32_t)libxs_hash_iso3309(buffer, size, 0);
  return result;
}


/**
 * The writers' primitives, local to the writers below (#undef after
 * libxs_predict_save): each appends at dst + result, and with dst NULL nothing
 * is written and only the size (result) grows, so one pass both sizes and
 * writes a record.
 */
#define WRITE_BLK(PTR, SZ) do { \
  if (NULL != dst) memcpy(dst + result, (PTR), (SZ)); \
  result += (SZ); \
} while (0)
#define WRITE_U32(V) do { const uint32_t v_ = (uint32_t)(V); WRITE_BLK(&v_, 4); } while (0)
#define WRITE_U16(V) do { const uint16_t v_ = (uint16_t)(V); WRITE_BLK(&v_, 2); } while (0)
#define WRITE_U8(V) do { const uint8_t v_ = (uint8_t)(V); WRITE_BLK(&v_, 1); } while (0)
#define WRITE_F64(V) do { const double v_ = (V); WRITE_BLK(&v_, 8); } while (0)
/* an encoded column (internal_libxs_predict_denc_write) */
#define WRITE_DENC(VALUES, COUNT, STRIDE) (result += internal_libxs_predict_denc_write( \
  (NULL != dst) ? (dst + result) : NULL, (VALUES), (COUNT), (STRIDE)))


/**
 * One cluster of an hKNN file, a flat cluster (nout outputs) and a per-output one
 * (the outputs of its group) alike, so a loaded per-output cluster carries what
 * the shared assembly built: centroid, radius, the fit of every column, the
 * points, the encoded outputs, weights, input hashes and member indices. With
 * dst NULL nothing is written and the size is returned, which keeps the size
 * query exact. A column of an empty cluster is written as zeros, and a cluster
 * without a fit (a version-1 per-output one) as order UINT8_MAX with no terms.
 * A compact record (a per-output cluster) holds the fit and its members alone,
 * two bytes each below 65536 entries: everything else an entry carries is in
 * the flat cluster holding it, and the reader takes it from there.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_hknn_cluster_write(unsigned char* dst,
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  int nout, int has_ih, int compact)
{
  const int mc = model->ncoord, ne = cl->nentries;
  const int fit = (NULL != cl->coeffs && NULL != cl->order) ? 1 : 0;
  const uint32_t n32 = (uint32_t)ne;
  const uint8_t mo = (uint8_t)(0 != fit ? cl->maxorder : UINT8_MAX);
  const uint8_t ke = (uint8_t)cl->k_eff;
  const double zero = 0;
  size_t result = 0;
  int j, k;
  if (NULL != cl->centroid) WRITE_BLK(cl->centroid, (size_t)mc * sizeof(double));
  else for (k = 0; k < mc; ++k) WRITE_BLK(&zero, sizeof(double));
  WRITE_BLK(&cl->dmax, sizeof(double));
  WRITE_BLK(&n32, sizeof(n32));
  WRITE_BLK(&mo, 1);
  WRITE_BLK(&ke, 1);
  for (j = 0; j < nout; ++j) {
    const uint8_t order = (uint8_t)(0 != fit ? cl->order[j] : 0);
    const uint8_t interp = (uint8_t)((NULL != cl->interpolated) ? cl->interpolated[j] : 0);
    const uint8_t mode = (uint8_t)((NULL != cl->mode) ? cl->mode[j] : 0);
    const uint32_t nd = (uint32_t)((NULL != cl->ndistinct) ? cl->ndistinct[j] : 0);
    const double err = (NULL != cl->errors) ? cl->errors[j] : 0;
    const double rms = (NULL != cl->out_rms) ? cl->out_rms[j] : 0;
    WRITE_BLK(&order, 1);
    WRITE_BLK(&interp, 1);
    WRITE_BLK(&mode, 1);
    WRITE_BLK(&nd, sizeof(nd));
    WRITE_BLK(&err, sizeof(double));
    WRITE_BLK(&rms, sizeof(double));
    if (0 != fit) {
      WRITE_BLK(cl->coeffs + (size_t)j * (cl->maxorder + 1),
        ((size_t)order + 1) * sizeof(double));
    }
  }
  if (0 < ne && 0 != compact) {
    for (k = 0; k < ne; ++k) {
      const int idx = (NULL != cl->sorted_idx) ? cl->sorted_idx[k] : -1;
      if (65536 > model->nentries) {
        const uint16_t i16 = (uint16_t)idx;
        WRITE_BLK(&i16, sizeof(i16));
      }
      else {
        const uint32_t i32 = (uint32_t)idx;
        WRITE_BLK(&i32, sizeof(i32));
      }
    }
  }
  else if (0 < ne) {
    WRITE_BLK(cl->kd_pts, (size_t)ne * (size_t)mc * sizeof(double));
    for (j = 0; j < nout; ++j) { /* per column, where the cardinality is */
      WRITE_DENC(cl->raw_outputs + j, (size_t)ne, (size_t)nout);
    }
    if (0 != model->has_eweight) {
      for (k = 0; k < ne; ++k) {
        const double w = (NULL != cl->eweight) ? cl->eweight[k] : 1.0;
        WRITE_BLK(&w, sizeof(double));
      }
    }
    if (0 != has_ih) WRITE_BLK(cl->ihash, (size_t)ne * sizeof(uint64_t));
    for (k = 0; k < ne; ++k) {
      const uint32_t idx = (uint32_t)((NULL != cl->sorted_idx) ? cl->sorted_idx[k] : -1);
      WRITE_BLK(&idx, sizeof(idx));
    }
  }
  return result;
}


/**
 * A per-output tree (see internal_libxs_predict_hknn_tree_t): its node count, then
 * per node the input it splits on (UINT16_MAX for a leaf), the threshold of a
 * split, and the right child of a split or the cluster of a leaf. With dst NULL
 * nothing is written and the size is returned.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_hknn_tree_write(unsigned char* dst,
  const internal_libxs_predict_hknn_tree_t* tree)
{
  const uint32_t nn = (uint32_t)((NULL != tree) ? tree->nnodes : 0);
  size_t result = 0;
  uint32_t k;
  WRITE_U32(nn);
  for (k = 0; k < nn; ++k) {
    const uint16_t f = (uint16_t)((0 <= tree->nodes[k].dim) ? tree->nodes[k].dim : UINT16_MAX);
    WRITE_U16(f);
    if (UINT16_MAX != f) WRITE_F64(tree->nodes[k].threshold);
    WRITE_U32(tree->nodes[k].next);
  }
  return result;
}


/**
 * One tree's nodes as columns: the feature and label of every node and the jump
 * of every split, then the values and the leaves' confidences, each in the form
 * internal_libxs_predict_denc_write finds smallest. Values repeat within a tree:
 * a threshold sits between two of the values an input takes, and a leaf of a
 * tuned parameter holds one of a few dozen. With dst NULL the size is returned.
 * scratch holds as many doubles as the tree has nodes.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_rf_tree_write(unsigned char* dst,
  const internal_libxs_predict_rf_tree_t* tree, double* scratch)
{
  size_t result = 0;
  int k, nleaf = 0;
  for (k = 0; k < tree->nnodes; ++k) {
    const internal_libxs_predict_rf_node_t* const nd = tree->nodes + k;
    WRITE_U16(nd->feature);
    WRITE_U8(nd->label);
    /* a split's jump, which a leaf has not (int32 as stored) */
    if (UINT16_MAX != nd->feature) WRITE_U32((int32_t)nd->data.right);
    scratch[k] = nd->value;
  }
  WRITE_DENC(scratch, (size_t)tree->nnodes, 1);
  for (k = 0; k < tree->nnodes; ++k) {
    if (UINT16_MAX == tree->nodes[k].feature) {
      scratch[nleaf++] = tree->nodes[k].data.leafp;
    }
  }
  WRITE_DENC(scratch, (size_t)nleaf, 1);
  return result;
}


/**
 * Derive per-output variance from the serialized raw outputs. Loaded models
 * carry raw_outputs but not out_var, and eval reads out_var to score the
 * confidence of many-valued outputs; without this a loaded model would report
 * a constant confidence where a built one reports neighbor concentration.
 */
LIBXS_API_INLINE int internal_libxs_predict_load_var(
  internal_libxs_predict_cluster_t* cl, int nout)
{
  int result = EXIT_SUCCESS;
  cl->out_mean = (double*)calloc((size_t)nout, sizeof(double));
  cl->out_var = (double*)calloc((size_t)nout, sizeof(double));
  if (NULL == cl->out_mean || NULL == cl->out_var) result = EXIT_FAILURE;
  else if (0 < cl->nentries && NULL != cl->raw_outputs) {
    const int nc = cl->nentries;
    int j, k;
    for (k = 0; k < nc; ++k) {
      for (j = 0; j < nout; ++j) {
        cl->out_mean[j] += cl->raw_outputs[(size_t)k * nout + j];
      }
    }
    for (j = 0; j < nout; ++j) cl->out_mean[j] /= nc;
    for (k = 0; k < nc; ++k) {
      for (j = 0; j < nout; ++j) {
        const double d = cl->raw_outputs[(size_t)k * nout + j] - cl->out_mean[j];
        cl->out_var[j] += d * d;
      }
    }
    for (j = 0; j < nout; ++j) cl->out_var[j] /= nc;
  }
  return result;
}


/**
 * Rebuild the global entry set from the per-cluster data. A loaded model
 * otherwise has no entries, which silently disables libxs_predict_inverse (it
 * abstains), the refinement loop, and the local-error diagnostic - so a saved
 * model answered differently from the model it was saved from. kd_pts holds
 * normalized inputs, so they are mapped back through the model's coordinate;
 * sorted_idx supplies the global position of each cluster-local entry.
 */
LIBXS_API_INLINE int internal_libxs_predict_load_entries(libxs_predict_t* model)
{
  const int m = model->ninputs, n = model->noutputs;
  const int p = model->nentries;
  int result = EXIT_SUCCESS;
  /**
   * The cluster count is tested, not just the pointer: calloc of zero elements
   * returns a non-NULL block, so a model saved without a partition passed here
   * and had its whole entry set and arena allocated from nothing to fill them
   * with. Every entry then carried NULL inputs, which libxs_predict_inverse
   * dereferences instead of abstaining.
   */
  int recoverable = (0 < p && 0 < model->nclusters
    && NULL != model->clusters && NULL != model->input_rng);
  if (0 != recoverable) {
    /**
     * sorted_idx supplies the global position of each cluster-local entry, so
     * without it no entry can be placed. A version-1 flat file carries none;
     * allocating the entry set anyway would leave every entry with NULL inputs,
     * which libxs_predict_inverse would dereference instead of abstaining.
     */
    int c;
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (0 < cl->nentries && NULL == cl->sorted_idx) recoverable = 0;
    }
  }
  /* a score maps a whole bin to one point, so its input cannot come back */
  if (NULL != model->input_score) recoverable = 0;
  if (0 != recoverable && NULL != model->weights) {
    /**
     * Feature selection (Fisher, setdiff, PCA) zeroes the weight of a dropped
     * input, and kd_pts stores the weighted value, so that coordinate cannot be
     * recovered. Reconstructing it as garbage would be worse than abstaining:
     * inverse would return plausible-looking wrong inputs instead of signalling
     * no result.
     */
    int j;
    for (j = 0; j < m; ++j) {
      if (0 == model->weights[j]) recoverable = 0;
    }
  }
  if (0 != recoverable) {
    model->entries = (internal_libxs_predict_entry_t*)calloc(
      (size_t)p, sizeof(internal_libxs_predict_entry_t));
    model->assignments = (int*)calloc((size_t)p, sizeof(int));
    if (NULL == model->entries || NULL == model->assignments) {
      result = EXIT_FAILURE;
    }
    else {
      int c, k;
      model->capacity = p;
      /* sized once: the entries are filled in cluster order, and every growth
       * would otherwise re-seat all of them again */
      if (NULL == internal_libxs_predict_slot(model, p - 1)) {
        result = EXIT_FAILURE;
      }
      for (c = 0; c < model->nclusters && EXIT_SUCCESS == result; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        if (NULL == cl->sorted_idx || NULL == cl->kd_pts
          || NULL == cl->raw_outputs)
        {
          continue;
        }
        for (k = 0; k < cl->nentries && EXIT_SUCCESS == result; ++k) {
          const int gi = cl->sorted_idx[k];
          if (0 > gi || p <= gi) {
            result = EXIT_FAILURE;
          }
          else {
            internal_libxs_predict_entry_t* e = &model->entries[gi];
            double* slot = internal_libxs_predict_slot(model, gi);
            e->inputs = slot;
            e->outputs = (NULL != slot) ? (slot + m) : NULL;
            if (NULL == e->inputs || NULL == e->outputs) {
              result = EXIT_FAILURE;
            }
            else {
              internal_libxs_predict_denormalize(model,
                cl->kd_pts + (size_t)k * model->ncoord, e->inputs);
              memcpy(e->outputs, cl->raw_outputs + (size_t)k * n,
                (size_t)n * sizeof(double));
              model->assignments[gi] = c;
            }
          }
        }
      }
    }
  }
  return result;
}


/**
 * Outputs served by one per-output partition. A per-output cluster stores its
 * outputs strided by this, so the writer, the reader, and eval must agree on it
 * or the payload is truncated and then read past its end - the clamp mirrors
 * what eval applies (see libxs_predict_eval).
 */
LIBXS_API_INLINE int internal_libxs_predict_gsize(const int* po_groups,
  int noutputs, int group)
{
  int result = 0, j;
  for (j = 0; j < noutputs; ++j) {
    if ((NULL != po_groups && po_groups[j] == group)
      || (NULL == po_groups && j == group))
    {
      ++result;
    }
  }
  return (0 < result) ? result : 1;
}


/**
 * Non-zero if the input hashes are written: where a distance of zero is not the
 * same inputs, and every cluster with entries has them. A model loaded from a
 * file without them has none to write, and its clusters keep the distance test.
 */
LIBXS_API_INLINE int internal_libxs_predict_has_ihash(const libxs_predict_t* model)
{
  int result = (0 == internal_libxs_predict_coincide(model)) ? 1 : 0, c, j;
  for (c = 0; c < model->nclusters && 0 != result; ++c) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    if (0 < cl->nentries && NULL == cl->ihash) result = 0;
  }
  if (NULL != model->hknn_po_clusters && NULL != model->hknn_po_nclusters) {
    const int ng = (model->hknn_ngroups > 0) ? model->hknn_ngroups : model->noutputs;
    for (j = 0; j < ng && 0 != result; ++j) {
      for (c = 0; NULL != model->hknn_po_clusters[j]
        && c < model->hknn_po_nclusters[j] && 0 != result; ++c)
      {
        const internal_libxs_predict_cluster_t* pcl = &model->hknn_po_clusters[j][c];
        if (0 < pcl->nentries && NULL == pcl->ihash) result = 0;
      }
    }
  }
  return result;
}


/* the resolved neighbour counts, one byte each behind a flag, and the reach */
LIBXS_API_INLINE size_t internal_libxs_predict_ksel_write(unsigned char* dst,
  const libxs_predict_t* model)
{
  size_t result = 0;
  int j;
  /* version 2: an older file derives the count; one without bit 2 has no reach */
  WRITE_U8((NULL != model->k_sel ? 1 : 0) | (0 < model->reach_eff ? 2 : 0));
  if (NULL != model->k_sel) {
    for (j = 0; j < model->noutputs; ++j) WRITE_U8(model->k_sel[j]);
  }
  if (0 < model->reach_eff) WRITE_U8(LIBXS_MIN(model->reach_eff, 255));
  return result;
}


/** The block both file kinds share, from the coordinate flag to the input weights. */
LIBXS_API_INLINE size_t internal_libxs_predict_common_write(unsigned char* dst,
  const libxs_predict_t* model)
{
  const int m = model->ninputs, n = model->noutputs;
  size_t result = 0;
  int j;
  /* introduced with version 2; an older file has no rank coordinate */
  WRITE_U16((NULL != model->input_knot ? LIBXS_PREDICT_KNOTS : 0)
    | (NULL != model->input_score ? LIBXS_PREDICT_SCOREFLAG : 0));
  WRITE_F64(model->quality);
  /**
   * Introduced with version 2. These change how a model ANSWERS, not what it
   * holds, and a file that lost them answered differently from the model that
   * was saved: a confidence map rendered from a saved buffer could not show a
   * consistency penalty at all. A version-1 file carries none and keeps the
   * zero that create() gives, which is "off" for each of them.
   */
  WRITE_F64(model->consistency);
  WRITE_F64(model->smooth);
  WRITE_F64(model->quantile);
  WRITE_F64(model->floor);
  /* The conformal correction, behind a flag: a model that was never calibrated
   * carries none, and a model that was answers with the interval it was
   * measured to give rather than losing the fit across a save. */
  WRITE_U8(0 != model->qconform_ok && NULL != model->qconform ? 1 : 0);
  if (0 != model->qconform_ok && NULL != model->qconform) {
    for (j = 0; j < n; ++j) WRITE_F64(model->qconform[j]);
  }
  WRITE_BLK(model->input_min, (size_t)m * sizeof(double));
  WRITE_BLK(model->input_rng, (size_t)m * sizeof(double));
  if (NULL != model->input_knot) {
    WRITE_BLK(model->input_knot, (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double));
  }
  if (NULL != model->input_score) {
    for (j = 0; j < m; ++j) WRITE_U8(model->input_sdim[j]);
    WRITE_BLK(model->input_score,
      (size_t)m * LIBXS_PREDICT_SCORESIZE * sizeof(double));
  }
  if (NULL != model->weights) {
    WRITE_BLK(model->weights, (size_t)m * sizeof(double));
  }
  return result;
}


/**
 * Converged escape weights, when the probability API has been used. These
 * are an optimization, not model state: a fresh bank re-learns them within a
 * few hundred queries, at a measured cost of 0.004-0.03 bits. Carrying them
 * means a rebuilt or reloaded model does not re-pay that transient. Written
 * last and guarded by a presence flag so a reader that predates them, or a
 * model that never scored a probability, is unaffected.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_escape_write(unsigned char* dst,
  const libxs_predict_t* model)
{
  size_t result = 0;
  if (NULL != model->escape_w) {
    WRITE_U8(1);
    WRITE_U8(LIBXS_PREDICT_NESCAPE);
    WRITE_BLK(model->escape_w,
      (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE * sizeof(double));
  }
  else WRITE_U8(0);
  return result;
}


/** An hKNN file without its trailing CRC; with dst NULL the size is returned. */
LIBXS_API_INLINE size_t internal_libxs_predict_hknn_write(unsigned char* dst,
  const libxs_predict_t* model, int has_ih)
{
  const int n = model->noutputs;
  size_t result = 0;
  int c, j;
  WRITE_U32(LIBXS_PREDICT_MAGIC_HKNN);
  WRITE_U16(LIBXS_PREDICT_VERSION);
  WRITE_U16(model->ninputs);
  WRITE_U16(n);
  WRITE_U16(model->nclusters);
  WRITE_U16(NULL != model->weights ? 1 : 0);
  /* introduced with version 2; an older file simply has no weights */
  WRITE_U8(0 != model->has_eweight ? 1 : 0);
  WRITE_U8(has_ih);
  result += internal_libxs_predict_ksel_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  result += internal_libxs_predict_common_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  for (c = 0; c < model->nclusters; ++c) {
    result += internal_libxs_predict_hknn_cluster_write(
      (NULL != dst) ? (dst + result) : NULL, model, &model->clusters[c], n,
      has_ih, 0);
  }
  if (NULL != model->hknn_po_assignments && NULL != model->hknn_po_clusters) {
    const int ng = (model->hknn_ngroups > 0) ? model->hknn_ngroups : n;
    /**
     * The group count is emitted because outputs may share a partition, in
     * which case ng < n and a reader that assumed one group per output would
     * consume the wrong number of assignment blocks and desynchronize for the
     * remainder of the payload. The per-output group map follows for the
     * same reason: eval indexes it to find an output's partition.
     */
    WRITE_U16(ng);
    for (j = 0; j < n; ++j) {
      WRITE_U16((NULL != model->hknn_po_groups) ? model->hknn_po_groups[j] : j);
    }
    /**
     * No assignments since version 2: the members each per-output cluster
     * lists are its assignment, and an entry's points, outputs, weight and
     * hash are those of the flat cluster holding it. Storing them again per
     * output made a file of n outputs carry its corpus n + 1 times.
     */
    WRITE_U8(1);
    for (j = 0; j < ng; ++j) {
      const int po_nc = (NULL != model->hknn_po_clusters[j])
        ? model->hknn_po_nclusters[j] : 0;
      const int gsz = internal_libxs_predict_gsize(model->hknn_po_groups, n, j);
      WRITE_U16(po_nc);
      for (c = 0; c < po_nc; ++c) {
        result += internal_libxs_predict_hknn_cluster_write(
          (NULL != dst) ? (dst + result) : NULL, model,
          &model->hknn_po_clusters[j][c], gsz, has_ih, 1);
      }
      /* since version 2, the tree a query descends to its cluster by */
      result += internal_libxs_predict_hknn_tree_write(
        (NULL != dst) ? (dst + result) : NULL,
        (NULL != model->hknn_po_trees) ? (model->hknn_po_trees + j) : NULL);
    }
  }
  else WRITE_U8(0);
  result += internal_libxs_predict_escape_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  return result;
}


/**
 * A flat file without its trailing CRC; with dst NULL the size is returned.
 * rf_scratch holds as many doubles as the largest tree has nodes.
 */
LIBXS_API_INLINE size_t internal_libxs_predict_flat_write(unsigned char* dst,
  const libxs_predict_t* model, int has_sidx, int has_ih, double* rf_scratch)
{
  const int m = model->ninputs, n = model->noutputs;
  const int has_ew = (0 != model->has_eweight) ? 1 : 0;
  size_t result = 0;
  int c, j;
  WRITE_U32(LIBXS_PREDICT_MAGIC);
  WRITE_U16(LIBXS_PREDICT_VERSION);
  WRITE_U16(m);
  WRITE_U16(n);
  WRITE_U16(model->nclusters);
  WRITE_U8(NULL != model->weights ? 1 : 0);
  WRITE_U8(NULL != model->transforms ? 1 : 0);
  WRITE_U16(model->nseries);
  WRITE_U16(model->window);
  WRITE_U16(model->target);
  WRITE_U16(model->decompose);
  WRITE_U16(model->naux);
  WRITE_U16(model->nderiv);
  WRITE_U8(model->eval_mode);
  WRITE_U8(model->diff_order);
  /**
   * Biased by one so the unsigned field can carry -1 (confidence-gated).
   * That also keeps a model saved before 0 meant "off" behaving as it did:
   * every such file stored 0, which was the confidence-gated default at
   * the time, and decodes to -1 rather than to the off that 0 means now.
   */
  WRITE_U8(LIBXS_CLMP(model->refine + 1, 0, 255));
  WRITE_U8(NULL != model->decompose_mat ? 1 : 0);
  WRITE_U8(has_sidx);
  /* introduced with version 2; an older file simply has no weights */
  WRITE_U8(has_ew);
  WRITE_U8(has_ih);
  result += internal_libxs_predict_ksel_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  WRITE_U16(model->order);
  WRITE_U8(model->iterations);
  WRITE_U32(model->nentries);
  result += internal_libxs_predict_common_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  if (NULL != model->transforms) {
    for (j = 0; j < n; ++j) WRITE_U8(model->transforms[j]);
  }
  if (NULL != model->decompose_mat) {
    WRITE_BLK(model->decompose_mat, (size_t)m * (size_t)m * sizeof(double));
  }
  for (c = 0; c < model->nclusters; ++c) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    const size_t ne = (size_t)cl->nentries;
    WRITE_BLK(cl->centroid, (size_t)model->ncoord * sizeof(double));
    WRITE_F64(cl->dmax);
    WRITE_U16(cl->nentries);
    WRITE_U8(cl->maxorder);
    WRITE_U8(cl->k_eff);
    for (j = 0; j < n; ++j) WRITE_U8(cl->order[j]);
    for (j = 0; j < n; ++j) WRITE_U8(cl->interpolated[j]);
    for (j = 0; j < n; ++j) WRITE_U8(cl->mode[j]);
    for (j = 0; j < n; ++j) WRITE_U16(cl->ndistinct[j]);
    WRITE_BLK(cl->errors, (size_t)n * sizeof(double));
    WRITE_BLK(cl->out_rms, (size_t)n * sizeof(double));
    WRITE_BLK(cl->kd_pts, ne * (size_t)model->ncoord * sizeof(double));
    /* per column and not per entry: the cardinality is a property of the
     * parameter, and an entry mixes all of them together */
    for (j = 0; j < n; ++j) WRITE_DENC(cl->raw_outputs + j, ne, (size_t)n);
    if (0 != has_ew) WRITE_BLK(cl->eweight, ne * sizeof(double));
    if (0 != has_ih && 0 < ne) WRITE_BLK(cl->ihash, ne * sizeof(uint64_t));
    /**
     * sorted_idx is what recency weighting and the local-error diagnostic
     * read; without it a loaded model silently takes a different path from
     * the model it was saved from. U32 because it indexes the global entry
     * set, which is not bounded by 64K like a per-cluster count.
     */
    if (0 != has_sidx) {
      int si;
      for (si = 0; si < cl->nentries; ++si) WRITE_U32(cl->sorted_idx[si]);
    }
    for (j = 0; j < n; ++j) {
      WRITE_BLK(cl->coeffs + (size_t)j * (cl->maxorder + 1),
        (size_t)(cl->order[j] + 1) * sizeof(double));
    }
  }
  if (NULL != model->rf) {
    const internal_libxs_predict_rf_t* const rf = model->rf;
    const int total_trees = rf->ntrees * rf->noutputs;
    WRITE_U16(rf->ntrees);
    WRITE_U16(rf->noutputs);
    for (j = 0; j < rf->noutputs; ++j) WRITE_U16((int16_t)rf->label_offset[j]);
    for (j = 0; j < rf->noutputs; ++j) WRITE_U8(rf->regress[j]);
    for (j = 0; j < rf->noutputs; ++j) WRITE_U8(rf->nclass[j]);
    /**
     * The bin count leads the curve rather than being implied by the build
     * that reads it: a model outlives the value the library was compiled
     * with, and a curve read at the wrong width is a confidence that is
     * wrong rather than absent. Zero says the measurement was not made, and
     * the share is then reported unchanged.
     */
    WRITE_U8(NULL != rf->calib ? LIBXS_PREDICT_RF_CALIB : 0);
    if (NULL != rf->calib) {
      const int ncal = rf->noutputs * LIBXS_PREDICT_RF_CALIB;
      for (j = 0; j < ncal; ++j) WRITE_F64(rf->calib[j]);
    }
    for (c = 0; c < total_trees; ++c) {
      const internal_libxs_predict_rf_tree_t* tree = &rf->trees[c];
      WRITE_U32(tree->nnodes);
      WRITE_U8(NULL != tree->incr ? 1 : 0);
      result += internal_libxs_predict_rf_tree_write(
        (NULL != dst) ? (dst + result) : NULL, tree, rf_scratch);
      if (NULL != tree->incr) {
        WRITE_BLK(tree->incr, (size_t)tree->nnodes
          * (size_t)rf->nclass[c / rf->ntrees] * sizeof(double));
      }
    }
  }
  result += internal_libxs_predict_escape_write(
    (NULL != dst) ? (dst + result) : NULL, model);
  return result;
}


LIBXS_API int libxs_predict_save(const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 == model->built || NULL == size) {
    result = EXIT_FAILURE;
  }
  else {
    const int hknn = (LIBXS_PREDICT_HKNN == model->decompose
      && NULL != model->hknn_po_assignments) ? 1 : 0;
    const int has_ih = internal_libxs_predict_has_ihash(model);
    double* rf_scratch = NULL;
    int rf_scratch_pool = 0, has_sidx = 1, c;
    /**
     * A model loaded from a version-1 flat file has no sorted_idx to write, so
     * its presence is a flag rather than an invariant. Fabricating indices
     * instead would silently enable recency weighting on an invented order.
     */
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (0 < cl->nentries && NULL == cl->sorted_idx) has_sidx = 0;
    }
    if (0 == hknn && NULL != model->rf) { /* sizes the forest's columns */
      const int total_trees = model->rf->ntrees * model->rf->noutputs;
      int maxnodes = 1;
      for (c = 0; c < total_trees; ++c) {
        maxnodes = LIBXS_MAX(maxnodes, model->rf->trees[c].nnodes);
      }
      rf_scratch = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)maxnodes * sizeof(double), rf_scratch_pool);
      if (NULL == rf_scratch) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      /* exact, as the records encode their columns by content */
      const size_t body = (0 != hknn)
        ? internal_libxs_predict_hknn_write(NULL, model, has_ih)
        : internal_libxs_predict_flat_write(NULL, model, has_sidx, has_ih,
          rf_scratch);
      const size_t required = body + sizeof(uint32_t); /* trailing CRC32 */
      if (NULL == buffer) *size = required;
      else if (*size < required) {
        *size = required;
        result = EXIT_FAILURE;
      }
      else {
        unsigned char* const dst = (unsigned char*)buffer;
        uint32_t crc = 0;
        if (0 != hknn) internal_libxs_predict_hknn_write(dst, model, has_ih);
        else {
          internal_libxs_predict_flat_write(dst, model, has_sidx, has_ih,
            rf_scratch);
        }
        result = internal_libxs_predict_crc(buffer, body, &crc);
        if (EXIT_SUCCESS == result) {
          memcpy(dst + body, &crc, sizeof(crc));
          *size = required;
        }
      }
    }
    LIBXS_PREDICT_FREE(rf_scratch, rf_scratch_pool);
  }
  return result;
}
#undef WRITE_DENC
#undef WRITE_F64
#undef WRITE_U8
#undef WRITE_U16
#undef WRITE_U32
#undef WRITE_BLK


LIBXS_API_INLINE int internal_libxs_predict_read(
  const unsigned char** src, const unsigned char* end, void* dst, size_t sz)
{
  int result = EXIT_SUCCESS;
  if (*src + sz > end) {
    result = EXIT_FAILURE;
  }
  else {
    memcpy(dst, *src, sz);
    *src += sz;
  }
  return result;
}


/** Reads what denc_write produced into a strided destination. */
LIBXS_API_INLINE int internal_libxs_predict_denc_read(
  const unsigned char** src, const unsigned char* end,
  double* values, size_t count, size_t stride)
{
  int result = EXIT_SUCCESS;
  uint8_t tag = 0;
  result = internal_libxs_predict_read(src, end, &tag, 1);
  if (EXIT_SUCCESS == result) {
    size_t i;
    if (INTERNAL_LIBXS_PREDICT_DENC_CONST == tag) {
      double v = 0;
      result = internal_libxs_predict_read(src, end, &v, sizeof(double));
      if (EXIT_SUCCESS == result) {
        for (i = 0; i < count; ++i) values[i * stride] = v;
      }
    }
    else if (INTERNAL_LIBXS_PREDICT_DENC_DICT == tag) {
      double dict[INTERNAL_LIBXS_PREDICT_DENC_DICTMAX];
      uint16_t ndict = 0;
      result = internal_libxs_predict_read(src, end, &ndict, 2);
      if (EXIT_SUCCESS == result
        && INTERNAL_LIBXS_PREDICT_DENC_DICTMAX < ndict)
      {
        result = EXIT_FAILURE; /* a file claiming more than a byte can index */
      }
      if (EXIT_SUCCESS == result) {
        result = internal_libxs_predict_read(src, end, dict,
          (size_t)ndict * sizeof(double));
      }
      for (i = 0; i < count && EXIT_SUCCESS == result; ++i) {
        uint8_t d = 0;
        result = internal_libxs_predict_read(src, end, &d, 1);
        if (EXIT_SUCCESS == result && d >= ndict) result = EXIT_FAILURE;
        if (EXIT_SUCCESS == result) values[i * stride] = dict[d];
      }
    }
    else if (INTERNAL_LIBXS_PREDICT_DENC_RAW == tag) {
      for (i = 0; i < count && EXIT_SUCCESS == result; ++i) {
        result = internal_libxs_predict_read(src, end,
          values + i * stride, sizeof(double));
      }
    }
    else result = EXIT_FAILURE; /* a tag this build does not know */
  }
  return result;
}


/**
 * The per-entry outputs, whichever way the file stores them. Version 1 wrote the
 * block verbatim and must keep loading: the encoding is a version-2 layout, and
 * decoding a verbatim block consumes the first value as a tag and everything
 * after it is nonsense. The rule lives here so the three places that read these
 * blocks cannot disagree about which files are encoded.
 */
LIBXS_API_INLINE int internal_libxs_predict_read_outputs(
  const unsigned char** src, const unsigned char* end,
  double* values, size_t count, size_t ncol, int version)
{
  int result = EXIT_SUCCESS;
  if (1 < version) {
    size_t j;
    for (j = 0; j < ncol && EXIT_SUCCESS == result; ++j) {
      result = internal_libxs_predict_denc_read(src, end,
        values + j, count, ncol);
    }
  }
  else {
    result = internal_libxs_predict_read(src, end, values,
      count * ncol * sizeof(double));
  }
  return result;
}


/**
 * Reject a file-supplied element count that cannot be covered by the remaining
 * payload before it is used to size an allocation: the division avoids the
 * overflow that a plain nelem*esz product would incur for hostile counts.
 */
LIBXS_API_INLINE int internal_libxs_predict_avail(const unsigned char* src,
  const unsigned char* end, size_t nelem, size_t esz)
{
  int result = EXIT_SUCCESS;
  if (src > end || nelem > (size_t)(end - src) / esz) result = EXIT_FAILURE;
  return result;
}


/**
 * Optional trailing escape-weight block, shared by both container formats.
 * Absence is normal - an older file, or a model that never scored a
 * probability - so a missing or mismatched block leaves the bank at its
 * uniform prior rather than failing the load. A different expert count means
 * the grid changed, in which case the stored weights describe experts that no
 * longer exist and must be ignored.
 */
LIBXS_API_INLINE void internal_libxs_predict_read_escape(
  libxs_predict_t* model, const unsigned char** src, const unsigned char* end)
{
  uint8_t present = 0;
  if (*src < end
    && EXIT_SUCCESS == internal_libxs_predict_read(src, end, &present, 1)
    && 0 != present)
  {
    uint8_t nesc = 0;
    const size_t nw = (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE;
    if (EXIT_SUCCESS == internal_libxs_predict_read(src, end, &nesc, 1)
      && LIBXS_PREDICT_NESCAPE == nesc
      && EXIT_SUCCESS == internal_libxs_predict_avail(*src, end,
        nw, sizeof(double)))
    {
      double* w = (double*)malloc(nw * sizeof(double));
      if (NULL != w) {
        if (EXIT_SUCCESS == internal_libxs_predict_read(src, end, w,
          nw * sizeof(double)))
        {
          free(model->escape_w);
          model->escape_w = w;
        }
        else free(w);
      }
    }
  }
}


/* ne input hashes into an array of their own (see internal_libxs_predict_has_ihash) */
LIBXS_API_INLINE int internal_libxs_predict_read_ihash(const unsigned char** src,
  const unsigned char* end, int ne, uint64_t** ihash)
{
  int result = internal_libxs_predict_avail(*src, end, (size_t)ne, sizeof(uint64_t));
  if (EXIT_SUCCESS == result) {
    *ihash = (uint64_t*)malloc((size_t)ne * sizeof(uint64_t));
    result = (NULL != *ihash) ? internal_libxs_predict_read(src, end, *ihash,
      (size_t)ne * sizeof(uint64_t)) : EXIT_FAILURE;
  }
  return result;
}


/** n unsigned values of width 1 or 2 bytes into an int array. */
LIBXS_API_INLINE int internal_libxs_predict_read_ints(const unsigned char** src,
  const unsigned char* end, int dst[], int n, int width)
{
  int result = EXIT_SUCCESS, i;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    if (1 == width) {
      uint8_t v = 0;
      result = internal_libxs_predict_read(src, end, &v, 1);
      dst[i] = (int)v;
    }
    else {
      uint16_t v = 0;
      result = internal_libxs_predict_read(src, end, &v, 2);
      dst[i] = (int)v;
    }
  }
  return result;
}


/**
 * A cluster's points, its outputs (ncol per entry, as internal_libxs_predict_read_outputs
 * reads them), its entry weights (has_ew) and its input hashes (has_ih, ne > 0).
 */
LIBXS_API_INLINE int internal_libxs_predict_points_read(const unsigned char** src,
  const unsigned char* end, libxs_predict_t* model, internal_libxs_predict_cluster_t* cl,
  int ne, int ncol, int version, int has_ew, int has_ih)
{
  const size_t npt = (size_t)ne * (size_t)model->ncoord;
  int result = internal_libxs_predict_avail(*src, end, npt, sizeof(double));
  if (EXIT_SUCCESS == result) {
    cl->kd_pts = (double*)malloc(npt * sizeof(double));
    result = (NULL != cl->kd_pts) ? internal_libxs_predict_read(src, end,
      cl->kd_pts, npt * sizeof(double)) : EXIT_FAILURE;
  }
  /* no size check ahead of the read: an encoded block is smaller than the
   * verbatim one by an amount only its tags reveal, and read_outputs bounds
   * every value it takes either way */
  if (EXIT_SUCCESS == result) {
    cl->raw_outputs = (double*)malloc((size_t)ne * (size_t)ncol * sizeof(double));
    result = (NULL != cl->raw_outputs) ? internal_libxs_predict_read_outputs(
      src, end, cl->raw_outputs, (size_t)ne, (size_t)ncol, version) : EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && 0 != has_ew) {
    result = internal_libxs_predict_avail(*src, end, (size_t)ne, sizeof(double));
    if (EXIT_SUCCESS == result) {
      cl->eweight = (double*)malloc((size_t)ne * sizeof(double));
      result = (NULL != cl->eweight) ? internal_libxs_predict_read(src, end,
        cl->eweight, (size_t)ne * sizeof(double)) : EXIT_FAILURE;
      if (EXIT_SUCCESS == result) model->has_eweight = 1;
    }
  }
  if (EXIT_SUCCESS == result && 0 != has_ih && 0 < ne) {
    result = internal_libxs_predict_read_ihash(src, end, ne, &cl->ihash);
  }
  return result;
}


/**
 * The tree of group g as internal_libxs_predict_hknn_tree_write stored it. Every
 * jump must lead forward, so a descent ends whatever the file holds, a split
 * must name an input the model has, and a leaf a cluster the group has.
 */
LIBXS_API_INLINE int internal_libxs_predict_hknn_tree_read(const unsigned char** src,
  const unsigned char* end, libxs_predict_t* model, int g, int po_nc)
{
  internal_libxs_predict_hknn_tree_t* const tree = model->hknn_po_trees + g;
  uint32_t nn = 0, k;
  int result = internal_libxs_predict_read(src, end, &nn, 4);
  if (EXIT_SUCCESS == result && 0 < nn) {
    result = (INT_MAX > nn) ? internal_libxs_predict_avail(*src, end, (size_t)nn,
      sizeof(uint16_t) + sizeof(uint32_t)) : EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      tree->nodes = (libxs_kdtree_node_t*)malloc((size_t)nn * sizeof(libxs_kdtree_node_t));
      if (NULL == tree->nodes) result = EXIT_FAILURE;
      else tree->nnodes = (int)nn;
    }
    for (k = 0; k < nn && EXIT_SUCCESS == result; ++k) {
      uint16_t f = 0;
      uint32_t next = 0;
      double thr = 0;
      result = internal_libxs_predict_read(src, end, &f, 2);
      if (EXIT_SUCCESS == result && UINT16_MAX != f) {
        result = internal_libxs_predict_read(src, end, &thr, 8);
      }
      if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &next, 4);
      if (EXIT_SUCCESS == result) {
        if (UINT16_MAX != f) {
          if ((int)f >= model->ninputs || k + 1 >= nn || next <= k + 1 || next >= nn) {
            result = EXIT_FAILURE;
          }
        }
        else if ((uint32_t)po_nc <= next) result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        tree->nodes[k].dim = (UINT16_MAX != f) ? (int)f : -1;
        tree->nodes[k].threshold = thr;
        tree->nodes[k].next = (int)next;
      }
    }
  }
  return result;
}


/**
 * A cluster written by internal_libxs_predict_hknn_cluster_write, into cl (zeroed)
 * for nout outputs. Bounds as everywhere: an order within the stored maximum, a
 * count no file can claim past its own size, and the derived spread restored.
 */
LIBXS_API_INLINE int internal_libxs_predict_hknn_cluster_read(const unsigned char** src,
  const unsigned char* end, libxs_predict_t* model, internal_libxs_predict_cluster_t* cl,
  int nout, int has_ew, int has_ih, int compact)
{
  const int mc = model->ncoord;
  uint32_t n32 = 0;
  uint8_t mo = 0, ke = 0;
  int result = EXIT_SUCCESS, j, k, ne = 0;
  cl->centroid = (double*)malloc((size_t)mc * sizeof(double));
  cl->order = (int*)malloc((size_t)nout * sizeof(int));
  cl->interpolated = (int*)malloc((size_t)nout * sizeof(int));
  cl->mode = (int*)malloc((size_t)nout * sizeof(int));
  cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
  cl->errors = (double*)malloc((size_t)nout * sizeof(double));
  cl->out_rms = (double*)malloc((size_t)nout * sizeof(double));
  if (NULL == cl->centroid || NULL == cl->order || NULL == cl->interpolated
    || NULL == cl->mode || NULL == cl->ndistinct || NULL == cl->errors
    || NULL == cl->out_rms)
  {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = internal_libxs_predict_read(src, end, cl->centroid, (size_t)mc * sizeof(double));
  }
  if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &cl->dmax, 8);
  if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &n32, 4);
  if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &mo, 1);
  if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &ke, 1);
  if (EXIT_SUCCESS == result && (INT_MAX < n32 || LIBXS_PREDICT_KNN < ke
    || (LIBXS_FPRINT_MAXORDER < mo && UINT8_MAX != mo)))
  {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    ne = cl->nentries = (int)n32;
    cl->k_eff = (int)ke;
    /* no fit leaves coeffs NULL, which is how eval tells it */
    if (UINT8_MAX != mo) {
      cl->maxorder = (int)mo;
      cl->coeffs = (double*)calloc((size_t)nout * (size_t)(cl->maxorder + 1), sizeof(double));
      if (NULL == cl->coeffs) result = EXIT_FAILURE;
    }
  }
  for (j = 0; j < nout && EXIT_SUCCESS == result; ++j) {
    uint8_t order = 0, interp = 0, mode = 0;
    uint32_t nd = 0;
    result = internal_libxs_predict_read(src, end, &order, 1);
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &interp, 1);
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &mode, 1);
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &nd, 4);
    if (EXIT_SUCCESS == result) {
      result = internal_libxs_predict_read(src, end, cl->errors + j, sizeof(double));
    }
    if (EXIT_SUCCESS == result) {
      result = internal_libxs_predict_read(src, end, cl->out_rms + j, sizeof(double));
    }
    /* coeffs is sized by maxorder: an order beyond it would overflow it */
    if (EXIT_SUCCESS == result && (cl->maxorder < (int)order || INT_MAX < nd)) {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      cl->order[j] = (int)order;
      cl->interpolated[j] = (int)interp;
      cl->mode[j] = (int)mode;
      cl->ndistinct[j] = (int)nd;
      if (NULL != cl->coeffs) {
        result = internal_libxs_predict_read(src, end,
          cl->coeffs + (size_t)j * (cl->maxorder + 1), ((size_t)order + 1) * sizeof(double));
      }
    }
  }
  /* members alone, the rest is derived (see internal_libxs_predict_hknn_po_derive) */
  if (EXIT_SUCCESS == result && 0 < ne && 0 != compact) {
    const size_t width = (65536 > model->nentries) ? sizeof(uint16_t) : sizeof(uint32_t);
    result = internal_libxs_predict_avail(*src, end, (size_t)ne, width);
    if (EXIT_SUCCESS == result) {
      cl->sorted_idx = (int*)malloc((size_t)ne * sizeof(int));
      if (NULL == cl->sorted_idx) result = EXIT_FAILURE;
    }
    for (k = 0; k < ne && EXIT_SUCCESS == result; ++k) {
      uint32_t idx = 0;
      if (sizeof(uint16_t) == width) {
        uint16_t i16 = 0;
        result = internal_libxs_predict_read(src, end, &i16, 2);
        idx = i16;
      }
      else result = internal_libxs_predict_read(src, end, &idx, 4);
      cl->sorted_idx[k] = (INT_MAX >= idx) ? (int)idx : -1;
    }
  }
  else if (EXIT_SUCCESS == result && 0 < ne) {
    result = internal_libxs_predict_points_read(src, end, model, cl, ne, nout,
      LIBXS_PREDICT_VERSION, has_ew, has_ih);
    if (EXIT_SUCCESS == result) {
      cl->sorted_idx = (int*)malloc((size_t)ne * sizeof(int));
      if (NULL == cl->sorted_idx) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = internal_libxs_predict_avail(*src, end, (size_t)ne, sizeof(uint32_t));
    }
    for (k = 0; k < ne && EXIT_SUCCESS == result; ++k) {
      uint32_t idx = 0;
      result = internal_libxs_predict_read(src, end, &idx, 4);
      cl->sorted_idx[k] = (INT_MAX >= idx) ? (int)idx : -1;
    }
  }
  if (EXIT_SUCCESS == result && 0 < ne && 0 == compact) {
    result = internal_libxs_predict_load_var(cl, nout);
  }
  return result;
}


/**
 * A tree written by internal_libxs_predict_rf_tree_write, nn nodes as the count
 * before it said. A split must name an input the model has and jump forward
 * within the tree, since the tree is packed preorder: a split's left child is
 * the next node and its right child the jump away.
 */
LIBXS_API_INLINE int internal_libxs_predict_rf_tree_read(const unsigned char** src,
  const unsigned char* end, int ninp, int nn, internal_libxs_predict_rf_tree_t* tree)
{
  int result = internal_libxs_predict_avail(*src, end, (size_t)nn, 3);
  int scratch_pool = 0, nleaf = 0, k;
  double* scratch = NULL;
  if (EXIT_SUCCESS == result) {
    tree->nodes = (internal_libxs_predict_rf_node_t*)
      internal_libxs_predict_rf_nodes_malloc(
        (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
    scratch = (double*)LIBXS_PREDICT_MALLOC((size_t)nn * sizeof(double),
      scratch_pool);
    tree->nnodes = nn;
    if (NULL == tree->nodes || NULL == scratch) result = EXIT_FAILURE;
  }
  for (k = 0; k < nn && EXIT_SUCCESS == result; ++k) {
    internal_libxs_predict_rf_node_t* const nd = tree->nodes + k;
    uint16_t f = 0;
    uint8_t lab = 0;
    result = internal_libxs_predict_read(src, end, &f, 2);
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &lab, 1);
    nd->feature = f;
    nd->label = lab;
    nd->value = 0;
    if (EXIT_SUCCESS == result && UINT16_MAX != f) {
      int32_t r = 0;
      result = internal_libxs_predict_read(src, end, &r, 4);
      if (EXIT_SUCCESS == result && (ninp <= f || 1 >= r || nn - k <= r)) {
        result = EXIT_FAILURE;
      }
      nd->data.right = r;
    }
    else if (EXIT_SUCCESS == result) {
      nd->data.leafp = 0;
      ++nleaf;
    }
  }
  if (EXIT_SUCCESS == result) {
    result = internal_libxs_predict_denc_read(src, end, scratch, (size_t)nn, 1);
  }
  for (k = 0; k < nn && EXIT_SUCCESS == result; ++k) tree->nodes[k].value = scratch[k];
  if (EXIT_SUCCESS == result) {
    result = internal_libxs_predict_denc_read(src, end, scratch, (size_t)nleaf, 1);
  }
  if (EXIT_SUCCESS == result) {
    int leaf = 0;
    for (k = 0; k < nn; ++k) {
      if (UINT16_MAX == tree->nodes[k].feature) {
        tree->nodes[k].data.leafp = (float)scratch[leaf++];
      }
    }
  }
  LIBXS_PREDICT_FREE(scratch, scratch_pool);
  return result;
}


/**
 * The per-output clusters of group g, completed from the flat clusters: each
 * member's points, outputs, weight and input hash are copied from where its
 * entry sits in the flat partition (flat_c, flat_k), which is what the build
 * stored for it, so the model answers as it did before it was saved. Their
 * members are the group's assignment, and every entry must be one exactly once.
 */
LIBXS_API_INLINE int internal_libxs_predict_hknn_po_derive(libxs_predict_t* model,
  int g, const int flat_c[], const int flat_k[], int has_ih)
{
  const int p = model->nentries, n = model->noutputs, mc = model->ncoord;
  const int po_nc = model->hknn_po_nclusters[g];
  const int gsz = internal_libxs_predict_gsize(model->hknn_po_groups, n, g);
  internal_libxs_predict_cluster_t* const cls = model->hknn_po_clusters[g];
  int result = EXIT_SUCCESS, outs_pool = 0, ci, i, k, li = 0;
  int* const outs = (int*)LIBXS_PREDICT_MALLOC((size_t)n * sizeof(int), outs_pool);
  int* const assign = (int*)malloc((size_t)LIBXS_MAX(p, 1) * sizeof(int));
  model->hknn_po_assignments[g] = assign;
  if (NULL == outs || NULL == assign || 0 >= gsz) result = EXIT_FAILURE;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    if (model->hknn_po_groups[i] == g) outs[li++] = i;
  }
  for (i = 0; i < p && EXIT_SUCCESS == result; ++i) assign[i] = -1;
  for (ci = 0; ci < po_nc && EXIT_SUCCESS == result; ++ci) {
    internal_libxs_predict_cluster_t* const cl = cls + ci;
    const int ne = cl->nentries;
    if (0 >= ne) continue;
    cl->kd_pts = (double*)malloc((size_t)ne * (size_t)mc * sizeof(double));
    cl->raw_outputs = (double*)malloc((size_t)ne * (size_t)gsz * sizeof(double));
    if (0 != model->has_eweight) cl->eweight = (double*)malloc((size_t)ne * sizeof(double));
    if (0 != has_ih) cl->ihash = (uint64_t*)malloc((size_t)ne * sizeof(uint64_t));
    if (NULL == cl->kd_pts || NULL == cl->raw_outputs || NULL == cl->sorted_idx
      || (0 != model->has_eweight && NULL == cl->eweight)
      || (0 != has_ih && NULL == cl->ihash))
    {
      result = EXIT_FAILURE;
    }
    for (k = 0; k < ne && EXIT_SUCCESS == result; ++k) {
      const int e = cl->sorted_idx[k];
      if (0 > e || p <= e || 0 <= assign[e]) result = EXIT_FAILURE;
      else {
        const internal_libxs_predict_cluster_t* const fc = model->clusters + flat_c[e];
        const int fk = flat_k[e];
        assign[e] = ci;
        memcpy(cl->kd_pts + (size_t)k * mc, fc->kd_pts + (size_t)fk * mc,
          (size_t)mc * sizeof(double));
        for (li = 0; li < gsz; ++li) {
          cl->raw_outputs[(size_t)k * gsz + li] = fc->raw_outputs[(size_t)fk * n + outs[li]];
        }
        if (NULL != cl->eweight) {
          cl->eweight[k] = (NULL != fc->eweight) ? fc->eweight[fk] : 1.0;
        }
        if (NULL != cl->ihash) {
          if (NULL != fc->ihash) cl->ihash[k] = fc->ihash[fk];
          else result = EXIT_FAILURE;
        }
      }
    }
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_load_var(cl, gsz);
  }
  for (i = 0; i < p && EXIT_SUCCESS == result; ++i) {
    if (0 > assign[i]) result = EXIT_FAILURE;
  }
  LIBXS_PREDICT_FREE(outs, outs_pool);
  return result;
}


/** Flag bytes a version-2 header carries, each read as zero or one (none before). */
LIBXS_API_INLINE int internal_libxs_predict_flags_read(const unsigned char** src,
  const unsigned char* end, int version, int flags[], int count)
{
  int result = EXIT_SUCCESS, i;
  for (i = 0; i < count; ++i) flags[i] = 0;
  for (i = 0; i < count && 1 < version && EXIT_SUCCESS == result; ++i) {
    uint8_t v = 0;
    result = internal_libxs_predict_read(src, end, &v, 1);
    flags[i] = (0 != v) ? 1 : 0;
  }
  return result;
}


/** What internal_libxs_predict_ksel_write wrote (version 2): neighbour counts and reach. */
LIBXS_API_INLINE int internal_libxs_predict_ksel_read(const unsigned char** src,
  const unsigned char* end, libxs_predict_t* model)
{
  uint8_t v = 0, r = 0;
  int result = internal_libxs_predict_read(src, end, &v, 1);
  if (EXIT_SUCCESS == result && 0 != (v & 1)) {
    model->k_sel = (int*)malloc((size_t)model->noutputs * sizeof(int));
    if (NULL == model->k_sel) result = EXIT_FAILURE;
    else {
      int j;
      for (j = 0; j < model->noutputs && EXIT_SUCCESS == result; ++j) {
        uint8_t kv = 0;
        result = internal_libxs_predict_read(src, end, &kv, 1);
        model->k_sel[j] = (int)kv;
      }
    }
  }
  if (EXIT_SUCCESS == result && 0 != (v & 2)) {
    result = internal_libxs_predict_read(src, end, &r, 1);
  }
  /* what the file holds is the reach in effect, kept without a trial to rerun */
  if (EXIT_SUCCESS == result) {
    libxs_predict_set_reach(model, (0 < r) ? (int)r : LIBXS_PREDICT_REACH_NONE);
  }
  return result;
}


/**
 * What internal_libxs_predict_common_write wrote: settings, conformal correction,
 * input extent, rank knots and score table, and input weights (has_weights).
 */
LIBXS_API_INLINE int internal_libxs_predict_common_read(const unsigned char** src,
  const unsigned char* end, libxs_predict_t* model, int version, int has_weights)
{
  const int m = model->ninputs, n = model->noutputs;
  uint16_t nknots = 0;
  double v = 0;
  int result = EXIT_SUCCESS;
  if (1 < version) result = internal_libxs_predict_read(src, end, &nknots, 2);
  if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &v, 8);
  if (EXIT_SUCCESS == result) model->quality = v;
  /* a version-1 file carries none of these and keeps create()'s zero, which is
   * "off" for each, so it answers as such a file always did */
  if (EXIT_SUCCESS == result && 1 < version) {
    uint8_t has_q = 0;
    result = internal_libxs_predict_read(src, end, &v, 8);
    if (EXIT_SUCCESS == result) model->consistency = v;
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &v, 8);
    if (EXIT_SUCCESS == result) model->smooth = v;
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &v, 8);
    if (EXIT_SUCCESS == result) model->quantile = v;
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &v, 8);
    if (EXIT_SUCCESS == result) model->floor = v;
    /* absent in a version-2 file that was never calibrated: the model then
     * answers with its raw interval */
    if (EXIT_SUCCESS == result) result = internal_libxs_predict_read(src, end, &has_q, 1);
    if (EXIT_SUCCESS == result && 0 != has_q) {
      model->qconform = (double*)malloc((size_t)n * sizeof(double));
      if (NULL == model->qconform) result = EXIT_FAILURE;
      else {
        result = internal_libxs_predict_read(src, end, model->qconform,
          (size_t)n * sizeof(double));
        if (EXIT_SUCCESS == result) model->qconform_ok = 1;
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    model->input_min = (double*)malloc((size_t)m * sizeof(double));
    model->input_rng = (double*)malloc((size_t)m * sizeof(double));
    if (NULL == model->input_min || NULL == model->input_rng) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = internal_libxs_predict_read(src, end,
      model->input_min, (size_t)m * sizeof(double));
  }
  if (EXIT_SUCCESS == result) {
    result = internal_libxs_predict_read(src, end,
      model->input_rng, (size_t)m * sizeof(double));
  }
  /* a score table rides on the knots, flagged in their count */
  if (EXIT_SUCCESS == result && 0 != nknots
    && LIBXS_PREDICT_KNOTS != (nknots & ~LIBXS_PREDICT_SCOREFLAG))
  {
    result = EXIT_FAILURE;
  }
  else if (EXIT_SUCCESS == result && 0 != nknots) {
    model->input_knot = (double*)malloc(
      (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double));
    if (NULL == model->input_knot) result = EXIT_FAILURE;
    else {
      result = internal_libxs_predict_read(src, end, model->input_knot,
        (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double));
    }
    if (EXIT_SUCCESS == result && 0 != (nknots & LIBXS_PREDICT_SCOREFLAG)) {
      model->input_score = (double*)malloc(
        (size_t)m * LIBXS_PREDICT_SCORESIZE * sizeof(double));
      model->input_sdim = (int*)malloc((size_t)m * sizeof(int));
      if (NULL == model->input_score || NULL == model->input_sdim) {
        result = EXIT_FAILURE;
      }
      else {
        int j;
        for (j = 0; j < m && EXIT_SUCCESS == result; ++j) {
          uint8_t width = 0;
          result = internal_libxs_predict_read(src, end, &width, 1);
          model->input_sdim[j] = LIBXS_MIN((int)width, LIBXS_PREDICT_SCOREDIMS);
        }
        if (EXIT_SUCCESS == result) {
          result = internal_libxs_predict_read(src, end, model->input_score,
            (size_t)m * LIBXS_PREDICT_SCORESIZE * sizeof(double));
        }
        /* the clusters are read at the width this settles */
        if (EXIT_SUCCESS == result) internal_libxs_predict_coords(model);
      }
    }
  }
  if (EXIT_SUCCESS == result && 0 != has_weights) {
    model->weights = (double*)malloc((size_t)m * sizeof(double));
    if (NULL != model->weights) {
      result = internal_libxs_predict_read(src, end,
        model->weights, (size_t)m * sizeof(double));
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE libxs_predict_t* internal_libxs_predict_load_hknn(
  const unsigned char* src, const unsigned char* end, int version)
{
  libxs_predict_t* model = NULL;
  uint16_t ninp = 0, nout = 0, nclust = 0, has_weights = 0;
  int ok = EXIT_SUCCESS, c, j, has_ew = 0, has_ih = 0;
  ok = internal_libxs_predict_read(&src, end, &ninp, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nout, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nclust, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_weights, 2);
  if (EXIT_SUCCESS == ok) {
    model = libxs_predict_create((int)ninp, (int)nout);
    if (NULL == model) ok = EXIT_FAILURE;
    /* create() stamps the current version; this one came from a file */
    else model->version = (int)version;
  }
  if (EXIT_SUCCESS == ok) {
    int flags[2];
    ok = internal_libxs_predict_flags_read(&src, end, version, flags, 2);
    has_ew = flags[0];
    has_ih = flags[1];
  }
  if (EXIT_SUCCESS == ok) {
    if (1 < version) ok = internal_libxs_predict_ksel_read(&src, end, model);
    else libxs_predict_set_reach(model, LIBXS_PREDICT_REACH_NONE);
  }
  if (EXIT_SUCCESS == ok) {
    model->decompose = LIBXS_PREDICT_HKNN;
    ok = internal_libxs_predict_common_read(&src, end, model, version,
      (int)has_weights);
  }
  if (EXIT_SUCCESS == ok) {
    model->nclusters = (int)nclust;
    model->clusters = (internal_libxs_predict_cluster_t*)calloc(
      (size_t)nclust, sizeof(internal_libxs_predict_cluster_t));
    model->eval_buf = (double*)malloc(
      (size_t)nout * 6 * sizeof(double) + (size_t)nout * sizeof(int));
    if (NULL == model->clusters || NULL == model->eval_buf) ok = EXIT_FAILURE;
  }
  for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
    internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    /* since version 2 a flat cluster is the record a per-output one is */
    if (1 < version) {
      ok = internal_libxs_predict_hknn_cluster_read(&src, end, model, cl,
        (int)nout, has_ew, has_ih, 0);
    }
    else {
      uint16_t ne = 0;
      uint8_t ke = 0;
      cl->centroid = (double*)malloc((size_t)model->ncoord * sizeof(double));
      cl->mode = (int*)malloc((size_t)nout * sizeof(int));
      cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
      if (NULL == cl->centroid || NULL == cl->mode || NULL == cl->ndistinct) {
        ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end,
          cl->centroid, (size_t)model->ncoord * sizeof(double));
      }
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cl->dmax, 8);
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end, &ne, 2);
        cl->nentries = (int)ne;
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end, &ke, 1);
        cl->k_eff = (int)ke;
        if (EXIT_SUCCESS == ok && cl->k_eff > LIBXS_PREDICT_KNN) ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read_ints(&src, end, cl->mode, (int)nout, 1);
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read_ints(&src, end, cl->ndistinct, (int)nout, 2);
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_points_read(&src, end, model, cl, ne,
          (int)nout, version, has_ew, has_ih);
      }
      /**
       * A version-1 file carries no fit residual, and every consumer reads a zero
       * out_rms as "no calibration" and falls back to out_var, derived below.
       */
      if (EXIT_SUCCESS == ok) {
        cl->out_rms = (double*)calloc((size_t)nout, sizeof(double));
        if (NULL == cl->out_rms) ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_load_var(cl, nout);
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_avail(src, end, (size_t)ne, sizeof(uint16_t));
      }
      if (EXIT_SUCCESS == ok) {
        cl->sorted_idx = (int*)malloc((size_t)ne * sizeof(int));
        if (NULL == cl->sorted_idx) ok = EXIT_FAILURE;
        else {
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read_ints(&src, end, cl->sorted_idx, (int)ne, 2);
          }
        }
      }
    }
  }
  /**
   * sorted_idx indexes the global entry set and the per-output assignments,
   * so it can only be bounded once every cluster contributed its count.
   */
  if (EXIT_SUCCESS == ok) {
    int total = 0;
    for (c = 0; c < (int)nclust; ++c) total += model->clusters[c].nentries;
    for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      int k;
      for (k = 0; k < cl->nentries && EXIT_SUCCESS == ok; ++k) {
        if (0 > cl->sorted_idx[k] || cl->sorted_idx[k] >= total) ok = EXIT_FAILURE;
      }
    }
    /* set for every output count: only the multi-output branch below did */
    model->nentries = total;
  }
  if (EXIT_SUCCESS == ok && src < end && (int)nout > 1) {
    int p = 0, i, ng = (int)nout, flat_pool = 0, flatk_pool = 0;
    int *flat_c = NULL, *flat_k = NULL;
    uint8_t has_po_clusters = 0;
    uint16_t ngroups = 0;
    for (c = 0; c < (int)nclust; ++c) p += model->clusters[c].nentries;
    model->nentries = p;
    /**
     * Version 1 has no group count and no group map: outputs never share a
     * partition, so the assignment blocks that follow are per-output. The
     * identity map is materialized rather than left NULL so that eval's group
     * filtering stays on one path for both versions.
     */
    if (1 < version) {
      ok = internal_libxs_predict_read(&src, end, &ngroups, 2);
      if (EXIT_SUCCESS == ok) {
        if (0 < ngroups && (int)ngroups <= (int)nout) ng = (int)ngroups;
        else ok = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == ok) {
      model->hknn_ngroups = ng;
      model->hknn_po_groups = (int*)calloc((size_t)nout, sizeof(int));
      if (NULL == model->hknn_po_groups) ok = EXIT_FAILURE;
    }
    for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
      if (1 < version) {
        uint16_t g = 0;
        ok = internal_libxs_predict_read(&src, end, &g, 2);
        if (EXIT_SUCCESS == ok && (int)g < ng) model->hknn_po_groups[j] = (int)g;
        else ok = EXIT_FAILURE;
      }
      else model->hknn_po_groups[j] = j;
    }
    model->hknn_po_assignments = (int**)calloc((size_t)nout, sizeof(int*));
    model->hknn_po_nclusters = (int*)calloc((size_t)nout, sizeof(int));
    if (NULL == model->hknn_po_assignments || NULL == model->hknn_po_nclusters) {
      ok = EXIT_FAILURE;
    }
    /* since version 2 the members of the per-output clusters are the assignment */
    if (EXIT_SUCCESS == ok && 1 == version) {
      for (j = 0; j < ng && EXIT_SUCCESS == ok; ++j) {
        ok = internal_libxs_predict_avail(src, end, (size_t)p, sizeof(uint16_t));
        if (EXIT_SUCCESS == ok) {
          model->hknn_po_assignments[j] = (int*)malloc(
            (size_t)p * sizeof(int));
          if (NULL == model->hknn_po_assignments[j]) ok = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == ok) {
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read_ints(&src, end,
              model->hknn_po_assignments[j], p, 2);
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok) {
      ok = internal_libxs_predict_read(&src, end, &has_po_clusters, 1);
    }
    if (EXIT_SUCCESS == ok && 0 != has_po_clusters && 1 < version) {
      /* where each entry sits in the flat partition, which the derivation reads */
      flat_c = (int*)LIBXS_PREDICT_MALLOC((size_t)LIBXS_MAX(p, 1) * sizeof(int), flat_pool);
      flat_k = (int*)LIBXS_PREDICT_MALLOC((size_t)LIBXS_MAX(p, 1) * sizeof(int), flatk_pool);
      if (NULL == flat_c || NULL == flat_k) ok = EXIT_FAILURE;
      for (i = 0; i < p && EXIT_SUCCESS == ok; ++i) flat_c[i] = -1;
      for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
        const internal_libxs_predict_cluster_t* const fc = model->clusters + c;
        int k;
        for (k = 0; k < fc->nentries && EXIT_SUCCESS == ok; ++k) {
          const int e = (NULL != fc->sorted_idx) ? fc->sorted_idx[k] : -1;
          if (0 > e || p <= e || 0 <= flat_c[e]) ok = EXIT_FAILURE;
          else {
            flat_c[e] = c;
            flat_k[e] = k;
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok && 0 != has_po_clusters) {
      model->hknn_po_clusters = (internal_libxs_predict_cluster_t**)calloc(
        (size_t)nout, sizeof(internal_libxs_predict_cluster_t*));
      model->hknn_po_trees = (internal_libxs_predict_hknn_tree_t*)calloc(
        (size_t)nout, sizeof(internal_libxs_predict_hknn_tree_t));
      if (NULL == model->hknn_po_clusters || NULL == model->hknn_po_trees) {
        ok = EXIT_FAILURE;
      }
      /* the writer emits one block per GROUP, not per output */
      for (j = 0; j < ng && EXIT_SUCCESS == ok; ++j) {
        const int gsz = internal_libxs_predict_gsize(
          model->hknn_po_groups, (int)nout, j);
        uint16_t po_nc = 0;
        int ci;
        ok = internal_libxs_predict_read(&src, end, &po_nc, 2);
        if (EXIT_SUCCESS == ok && po_nc > 0) {
          internal_libxs_predict_cluster_t* cls =
            (internal_libxs_predict_cluster_t*)calloc(
              (size_t)po_nc, sizeof(internal_libxs_predict_cluster_t));
          if (NULL == cls) { ok = EXIT_FAILURE; break; }
          model->hknn_po_clusters[j] = cls;
          model->hknn_po_nclusters[j] = (int)po_nc;
          for (ci = 0; ci < (int)po_nc && EXIT_SUCCESS == ok; ++ci) {
            if (1 < version) {
              ok = internal_libxs_predict_hknn_cluster_read(&src, end, model,
                cls + ci, gsz, has_ew, has_ih, 1);
            }
            else {
              uint16_t ne = 0;
              uint8_t ke = 0;
              ok = internal_libxs_predict_read(&src, end, &ne, 2);
              if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ke, 1);
              if (EXIT_SUCCESS == ok && ke > LIBXS_PREDICT_KNN) ok = EXIT_FAILURE;
              if (EXIT_SUCCESS == ok) {
                cls[ci].nentries = (int)ne;
                cls[ci].k_eff = (int)ke;
                cls[ci].mode = (int*)calloc((size_t)gsz, sizeof(int));
                cls[ci].ndistinct = (int*)calloc((size_t)gsz, sizeof(int));
                cls[ci].centroid = (double*)malloc((size_t)model->ncoord * sizeof(double));
                if (NULL == cls[ci].mode || NULL == cls[ci].ndistinct || NULL == cls[ci].centroid) {
                  ok = EXIT_FAILURE;
                }
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read_ints(&src, end, cls[ci].mode, gsz, 1);
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read_ints(&src, end, cls[ci].ndistinct, gsz, 2);
              }
              if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cls[ci].dmax, 8);
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end,
                  cls[ci].centroid, (size_t)model->ncoord * sizeof(double));
              }
              if (EXIT_SUCCESS == ok && ne > 0) {
                /* version 1 stores the outputs raw, so their size is known */
                ok = internal_libxs_predict_avail(src, end, (size_t)ne
                  * ((size_t)model->ncoord + (size_t)gsz), sizeof(double));
                if (EXIT_SUCCESS == ok) {
                  cls[ci].kd_pts = (double*)malloc(
                    (size_t)ne * (size_t)model->ncoord * sizeof(double));
                  cls[ci].raw_outputs = (double*)malloc(
                    (size_t)ne * (size_t)gsz * sizeof(double));
                  if (NULL == cls[ci].kd_pts || NULL == cls[ci].raw_outputs) {
                    ok = EXIT_FAILURE;
                  }
                }
                if (EXIT_SUCCESS == ok) {
                  ok = internal_libxs_predict_read(&src, end,
                    cls[ci].kd_pts, (size_t)ne * (size_t)model->ncoord * sizeof(double));
                }
                if (EXIT_SUCCESS == ok) {
                  ok = internal_libxs_predict_read_outputs(&src, end,
                    cls[ci].raw_outputs, (size_t)ne, (size_t)gsz, version);
                }
                if (EXIT_SUCCESS == ok && 0 != has_ih) {
                  ok = internal_libxs_predict_read_ihash(&src, end, ne, &cls[ci].ihash);
                }
              }
            }
          }
        }
        if (EXIT_SUCCESS == ok && 1 < version && 0 < po_nc) {
          ok = internal_libxs_predict_hknn_po_derive(model, j, flat_c, flat_k, has_ih);
        }
        if (EXIT_SUCCESS == ok && 1 < version) {
          ok = internal_libxs_predict_hknn_tree_read(&src, end, model, j, (int)po_nc);
        }
      }
    }
    /* an assignment selects a per-output cluster, so it must index one */
    if (EXIT_SUCCESS == ok && NULL != model->hknn_po_clusters) {
      for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
        const int po_nc = model->hknn_po_nclusters[j];
        if (NULL == model->hknn_po_assignments[j] || NULL == model->hknn_po_clusters[j]) {
          continue;
        }
        for (i = 0; i < p && EXIT_SUCCESS == ok; ++i) {
          if (model->hknn_po_assignments[j][i] >= po_nc) ok = EXIT_FAILURE;
        }
        for (i = 0; i < po_nc && EXIT_SUCCESS == ok; ++i) {
          internal_libxs_predict_cluster_t* const pcl = model->hknn_po_clusters[j] + i;
          int k, ki = 0;
          /* version 1 stores no members: they are the assigned entries, ascending */
          if (NULL == pcl->sorted_idx && 0 < pcl->nentries) {
            pcl->sorted_idx = (int*)malloc((size_t)pcl->nentries * sizeof(int));
            if (NULL == pcl->sorted_idx) ok = EXIT_FAILURE;
            for (k = 0; k < p && EXIT_SUCCESS == ok; ++k) {
              if (model->hknn_po_assignments[j][k] != i) continue;
              if (ki < pcl->nentries) pcl->sorted_idx[ki++] = k;
              else ok = EXIT_FAILURE;
            }
            if (EXIT_SUCCESS == ok && ki != pcl->nentries) ok = EXIT_FAILURE;
          }
          for (k = 0; k < pcl->nentries && EXIT_SUCCESS == ok; ++k) {
            if (0 > pcl->sorted_idx[k] || pcl->sorted_idx[k] >= p) ok = EXIT_FAILURE;
          }
        }
      }
    }
    LIBXS_PREDICT_FREE(flat_k, flatk_pool);
    LIBXS_PREDICT_FREE(flat_c, flat_pool);
  }
  if (EXIT_SUCCESS == ok) {
    /**
     * The escape block is fixed-length and written last, so it is located from
     * the end rather than by having consumed every preceding field. The hknn
     * reader tolerates a short read of optional sections, which would otherwise
     * leave src mid-payload and skip the block silently. Version 1 has no such
     * block, and probing from the end would read model payload as weights.
     */
    const size_t esclen = 2 * sizeof(uint8_t)
      + (size_t)nout * LIBXS_PREDICT_NESCAPE * sizeof(double);
    if (1 < version && src <= end && (size_t)(end - src) >= esclen) {
      const unsigned char* esc = end - esclen;
      internal_libxs_predict_read_escape(model, &esc, end);
    }
    /**
     * The corpus, as the flat reader recovers it. Without it a loaded hKNN
     * model skipped the round-trip pass silently, so a consistency penalty
     * or a refinement it was saved with did nothing after loading.
     */
    ok = internal_libxs_predict_load_entries(model);
  }
  if (EXIT_SUCCESS == ok && NULL != model) {
    model->built = 1;
    ++model->nbuild;
    internal_libxs_predict_missing_all(model);
    internal_libxs_predict_support_all(model);
    internal_libxs_predict_keff_all(model);
    internal_libxs_predict_kapply(model);
    if (0 >= model->central) internal_libxs_predict_central_all(model);
  }
  else if (NULL != model) {
    libxs_predict_destroy(model);
    model = NULL;
  }
  return model;
}


LIBXS_API_INLINE int internal_libxs_predict_load_rf_tree_v1(
  const unsigned char** src, const unsigned char* end, int ninputs,
  internal_libxs_predict_rf_tree_t* tree)
{
  uint16_t nn = 0;
  int result = internal_libxs_predict_read(src, end, &nn, 2);
  internal_libxs_predict_rf_build_node_t* nodes = NULL;
  int nodes_pool = 0;
  if (EXIT_SUCCESS == result && INT16_MAX < nn) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 0 < nn) {
    result = internal_libxs_predict_avail(*src, end, (size_t)nn,
      2 + 8 + 2 + 2 + 1);
  }
  if (EXIT_SUCCESS == result && 0 < nn) {
    nodes = (internal_libxs_predict_rf_build_node_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nn * sizeof(internal_libxs_predict_rf_build_node_t), nodes_pool);
    if (NULL == nodes) result = EXIT_FAILURE;
  }
  if (NULL != nodes) {
    int k;
    for (k = 0; k < (int)nn && EXIT_SUCCESS == result; ++k) {
      int16_t feature = 0, left = 0, right = 0;
      uint8_t label = 0;
      result = internal_libxs_predict_read(src, end, &feature, 2);
      if (EXIT_SUCCESS == result) {
        result = internal_libxs_predict_read(
          src, end, &nodes[k].threshold, 8);
      }
      if (EXIT_SUCCESS == result) {
        result = internal_libxs_predict_read(src, end, &left, 2);
      }
      if (EXIT_SUCCESS == result) {
        result = internal_libxs_predict_read(src, end, &right, 2);
      }
      if (EXIT_SUCCESS == result) {
        result = internal_libxs_predict_read(src, end, &label, 1);
      }
      if (EXIT_SUCCESS == result && (feature >= ninputs || feature < -1
        || left >= (int16_t)nn || left < -1
        || right >= (int16_t)nn || right < -1))
      {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        nodes[k].feature = (int)feature;
        nodes[k].left = (int)left;
        nodes[k].right = (int)right;
        nodes[k].label = (int)label;
        nodes[k].value = (double)label;
        nodes[k].leafp = 0.f;
      }
    }
    if (EXIT_SUCCESS == result) {
      tree->nnodes = internal_libxs_predict_rf_pack_tree(
        nodes, (int)nn, &tree->nodes);
      if (0 >= tree->nnodes) result = EXIT_FAILURE;
    }
  }
  LIBXS_PREDICT_FREE(nodes, nodes_pool);
  return result;
}


LIBXS_API libxs_predict_t* libxs_predict_load(const void* buffer, size_t size)
{
  libxs_predict_t* model = NULL;
  int hknn = 0;
  if (NULL != buffer
    && size >= sizeof(uint32_t) + 4 * sizeof(uint16_t))
  {
    const unsigned char* src = (const unsigned char*)buffer;
    const unsigned char* end = src + size;
    uint32_t magic = 0;
    uint16_t version = 0, ninp = 0, nout = 0, nclust = 0;
    int has_sidx = 0, has_ew = 0, has_ih = 0;
    int ok = internal_libxs_predict_read(&src, end, &magic, 4);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &version, 2);
    if (EXIT_SUCCESS == ok && (0 == version || LIBXS_PREDICT_VERSION < version)) {
      ok = EXIT_FAILURE;
    }
    /**
     * The CRC trailer arrived with version 2, so the payload end can only be
     * fixed once the version is known: a version-1 payload runs to the end of
     * the buffer, and treating its last four bytes as a checksum would both
     * fail the comparison and truncate the model.
     */
    if (EXIT_SUCCESS == ok && 1 < version) {
      if (size >= 2 * sizeof(uint32_t) + 4 * sizeof(uint16_t)) {
        uint32_t crc = 0, expected = 0;
        end -= sizeof(uint32_t);
        ok = internal_libxs_predict_crc(buffer, size - sizeof(uint32_t), &crc);
        if (EXIT_SUCCESS == ok) memcpy(&expected, end, sizeof(uint32_t));
        if (EXIT_SUCCESS == ok && crc != expected) ok = EXIT_FAILURE;
      }
      else ok = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == ok && LIBXS_PREDICT_MAGIC_HKNN == magic) {
      hknn = 1;
      model = internal_libxs_predict_load_hknn(src, end, (int)version);
      ok = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == ok && magic != LIBXS_PREDICT_MAGIC) ok = EXIT_FAILURE;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ninp, 2);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nout, 2);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nclust, 2);
    if (EXIT_SUCCESS == ok) {
      model = libxs_predict_create((int)ninp, (int)nout);
      if (NULL == model) ok = EXIT_FAILURE;
      /* create() stamps the current version; this one came from a file */
      else model->version = (int)version;
    }
    if (EXIT_SUCCESS == ok) {
      uint8_t has_weights = 0, has_transforms = 0, has_dmat = 0;
      uint8_t eval_mode = 0, diff_order = 0, refine = 0, iterations = 0;
      uint16_t ts_nseries = 0, ts_window = 0, ts_target = 0, ts_decompose = 0;
      uint16_t ts_naux = 0, ts_nderiv = 0;
      uint16_t order = 0;
      ok = internal_libxs_predict_read(&src, end, &has_weights, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_transforms, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_nseries, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_window, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_target, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_decompose, 2);
      /* naux and nderiv arrived with version 2 and default to zero without them */
      if (EXIT_SUCCESS == ok && 1 < version) {
        ok = internal_libxs_predict_read(&src, end, &ts_naux, 2);
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_nderiv, 2);
      }
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &eval_mode, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &diff_order, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &refine, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_dmat, 1);
      if (EXIT_SUCCESS == ok) {
        int flags[3];
        ok = internal_libxs_predict_flags_read(&src, end, version, flags, 3);
        has_sidx = flags[0];
        has_ew = flags[1];
        has_ih = flags[2];
      }
      if (EXIT_SUCCESS == ok && 1 < version) {
        ok = internal_libxs_predict_ksel_read(&src, end, model);
      }
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &order, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &iterations, 1);
      { uint32_t nentries = 0;
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nentries, 4);
        if (EXIT_SUCCESS == ok) model->nentries = (int)nentries;
      }
      /* ahead of the shared block: its score table settles the coordinate width */
      if (EXIT_SUCCESS == ok) {
        model->nseries = (int)ts_nseries;
        model->window = (int)ts_window;
        model->target = (int)ts_target;
        model->decompose = (int)ts_decompose;
        model->naux = (int)ts_naux;
        model->nderiv = (int)ts_nderiv;
        model->eval_mode = (int)eval_mode;
        model->diff_order = (int)diff_order;
        model->refine = (int)refine - 1; /* biased on write, see the writer */
        model->order = (int)order;
        model->iterations = (int)iterations;
        if (0 != diff_order) model->diff_mode = (int)diff_order;
        ok = internal_libxs_predict_common_read(&src, end, model, version,
          (int)has_weights);
      }
      if (EXIT_SUCCESS == ok && 0 != has_transforms) {
        model->transforms = (int*)calloc((size_t)nout, sizeof(int));
        if (NULL != model->transforms) {
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read_ints(&src, end,
              model->transforms, (int)nout, 1);
          }
        }
        else ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok && 0 != has_dmat) {
        const size_t msz = (size_t)ninp * (size_t)ninp;
        model->decompose_mat = (double*)malloc(msz * sizeof(double));
        if (NULL != model->decompose_mat) {
          ok = internal_libxs_predict_read(&src, end,
            model->decompose_mat, msz * sizeof(double));
        }
        else ok = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == ok) {
      model->nclusters = (int)nclust;
      model->clusters = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nclust, sizeof(internal_libxs_predict_cluster_t));
      model->eval_buf = (double*)malloc(
        (size_t)nout * 6 * sizeof(double) + (size_t)nout * sizeof(int));
      if (NULL == model->clusters || NULL == model->eval_buf) ok = EXIT_FAILURE;
    }
    { int c;
      for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
        internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        uint16_t ne = 0;
        uint8_t mo = 0;
        int j;
        cl->centroid = (double*)malloc((size_t)model->ncoord * sizeof(double));
        cl->order = (int*)malloc((size_t)nout * sizeof(int));
        cl->interpolated = (int*)malloc((size_t)nout * sizeof(int));
        cl->mode = (int*)malloc((size_t)nout * sizeof(int));
        cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
        cl->errors = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->centroid || NULL == cl->order || NULL == cl->interpolated
          || NULL == cl->mode || NULL == cl->ndistinct || NULL == cl->errors)
        {
          ok = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->centroid, (size_t)model->ncoord * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cl->dmax, 8);
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end, &ne, 2);
          if (EXIT_SUCCESS == ok) cl->nentries = (int)ne;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end, &mo, 1);
          if (EXIT_SUCCESS == ok) cl->maxorder = (int)mo;
          if (EXIT_SUCCESS == ok && cl->maxorder > LIBXS_FPRINT_MAXORDER) {
            ok = EXIT_FAILURE;
          }
        }
        if (EXIT_SUCCESS == ok) {
          uint8_t ke = 0;
          ok = internal_libxs_predict_read(&src, end, &ke, 1);
          if (EXIT_SUCCESS == ok) cl->k_eff = (int)ke;
          if (EXIT_SUCCESS == ok && cl->k_eff > LIBXS_PREDICT_KNN) {
            ok = EXIT_FAILURE;
          }
        }
        /* coeffs is sized by maxorder: order[j] beyond it would overflow it */
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          if (EXIT_SUCCESS == ok && (int)v > cl->maxorder) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) cl->order[j] = (int)v;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_ints(&src, end, cl->interpolated, (int)nout, 1);
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_ints(&src, end, cl->mode, (int)nout, 1);
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_ints(&src, end, cl->ndistinct, (int)nout, 2);
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->errors, (size_t)nout * sizeof(double));
        }
        /* out_rms arrived with version 2; zero reads as "no calibration" */
        if (EXIT_SUCCESS == ok) {
          if (1 < version) {
            cl->out_rms = (double*)malloc((size_t)nout * sizeof(double));
            if (NULL == cl->out_rms) ok = EXIT_FAILURE;
            else {
              ok = internal_libxs_predict_read(&src, end,
                cl->out_rms, (size_t)nout * sizeof(double));
            }
          }
          else {
            cl->out_rms = (double*)calloc((size_t)nout, sizeof(double));
            if (NULL == cl->out_rms) ok = EXIT_FAILURE;
          }
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_points_read(&src, end, model, cl,
            cl->nentries, (int)nout, (int)version, has_ew, has_ih);
        }
        /**
         * A version-1 flat file has no sorted_idx, and so has no global order to
         * recover. It is left NULL: eval already treats that as "no recency
         * weighting" and the entry set stays unrecoverable, which is what such a
         * model did before - an invented order would be worse than none.
         */
        if (EXIT_SUCCESS == ok && 0 != has_sidx) {
          ok = internal_libxs_predict_avail(src, end,
            (size_t)cl->nentries, sizeof(uint32_t));
        }
        if (EXIT_SUCCESS == ok && 0 != has_sidx) {
          cl->sorted_idx = (int*)malloc((size_t)cl->nentries * sizeof(int));
          if (NULL == cl->sorted_idx) ok = EXIT_FAILURE;
          else {
            int kk;
            for (kk = 0; kk < cl->nentries && EXIT_SUCCESS == ok; ++kk) {
              uint32_t v = 0;
              ok = internal_libxs_predict_read(&src, end, &v, 4);
              if (EXIT_SUCCESS == ok && (uint32_t)model->nentries <= v) {
                ok = EXIT_FAILURE;
              }
              else cl->sorted_idx[kk] = (int)v;
            }
          }
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_load_var(cl, (int)nout);
        }
        if (EXIT_SUCCESS == ok) {
          cl->coeffs = (double*)calloc(
            (size_t)nout * (size_t)(cl->maxorder + 1), sizeof(double));
          if (NULL == cl->coeffs) ok = EXIT_FAILURE;
          for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
            ok = internal_libxs_predict_read(&src, end,
              cl->coeffs + (size_t)j * (cl->maxorder + 1),
              (size_t)(cl->order[j] + 1) * sizeof(double));
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok && src < end && model->decompose == LIBXS_PREDICT_RF) {
      uint16_t rf_ntrees = 0, rf_nouts = 0;
      int j;
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end, &rf_ntrees, 2);
      }
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &rf_nouts, 2);
      /* eval indexes trees[output_idx * ntrees + t] for output_idx < noutputs */
      if (EXIT_SUCCESS == ok && rf_nouts != nout) ok = EXIT_FAILURE;
      if (EXIT_SUCCESS == ok && rf_ntrees > 0 && rf_nouts > 0) {
        internal_libxs_predict_rf_t* rf = (internal_libxs_predict_rf_t*)calloc(
          1, sizeof(internal_libxs_predict_rf_t));
        if (NULL != rf) {
          const int total_trees = (int)rf_ntrees * (int)rf_nouts;
          rf->ntrees = (int)rf_ntrees;
          rf->noutputs = (int)rf_nouts;
          rf->label_offset = (int*)malloc((size_t)rf_nouts * sizeof(int));
          rf->regress = (int*)calloc((size_t)rf_nouts, sizeof(int));
          rf->nclass = (int*)malloc((size_t)rf_nouts * sizeof(int));
          rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
            (size_t)total_trees, sizeof(internal_libxs_predict_rf_tree_t));
          if (NULL != rf->label_offset && NULL != rf->trees
            && NULL != rf->regress && NULL != rf->nclass)
          {
            int ti;
            for (j = 0; j < (int)rf_nouts && EXIT_SUCCESS == ok; ++j) {
              int16_t off = 0;
              ok = internal_libxs_predict_read(&src, end, &off, 2);
              rf->label_offset[j] = (int)off;
            }
            /** Before version 2 every output was folded to a class, and the
             *  leaf carried no value of its own; the label stands in for it. */
            for (j = 0; j < (int)rf_nouts && EXIT_SUCCESS == ok
              && 1 < version; ++j)
            {
              uint8_t reg = 0;
              ok = internal_libxs_predict_read(&src, end, &reg, 1);
              rf->regress[j] = (0 != reg) ? 1 : 0;
            }
            for (j = 0; j < (int)rf_nouts; ++j) rf->nclass[j] = 1;
            for (j = 0; j < (int)rf_nouts && EXIT_SUCCESS == ok
              && 1 < version; ++j)
            {
              uint8_t ncl = 0;
              ok = internal_libxs_predict_read(&src, end, &ncl, 1);
              /* the correction is indexed by node and class, so a width of
               * zero or beyond the fold would leave that indexing unbounded */
              if (EXIT_SUCCESS == ok && (0 == ncl || 128 < ncl)) {
                ok = EXIT_FAILURE;
              }
              else rf->nclass[j] = (int)ncl;
            }
            /**
             * The curve is read at the width the file states, and kept only at
             * the width this build indexes it by: a model written by a library
             * binned differently carries a curve this one cannot address, and
             * reporting the share unchanged is the honest reading of that.
             * Version 1 has no curve at all and takes the same path.
             */
            if (EXIT_SUCCESS == ok && 1 < version) {
              uint8_t ncal = 0;
              ok = internal_libxs_predict_read(&src, end, &ncal, 1);
              if (EXIT_SUCCESS == ok && 0 != ncal) {
                const int ncurve = (int)rf_nouts * (int)ncal;
                ok = internal_libxs_predict_avail(src, end,
                  (size_t)ncurve, sizeof(double));
                if (EXIT_SUCCESS == ok
                  && LIBXS_PREDICT_RF_CALIB == (int)ncal)
                {
                  rf->calib = (double*)malloc(
                    (size_t)ncurve * sizeof(double));
                  if (NULL == rf->calib) ok = EXIT_FAILURE;
                }
                for (j = 0; j < ncurve && EXIT_SUCCESS == ok; ++j) {
                  double v = 0;
                  ok = internal_libxs_predict_read(&src, end, &v, 8);
                  if (NULL != rf->calib) rf->calib[j] = v;
                }
              }
            }
            for (ti = 0; ti < total_trees && EXIT_SUCCESS == ok; ++ti) {
              uint32_t nn = 0;
              uint8_t hasincr = 0;
              int k;
              if (1 >= version) {
                ok = internal_libxs_predict_load_rf_tree_v1(
                  &src, end, (int)ninp, &rf->trees[ti]);
                continue;
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end, &nn, 4);
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end, &hasincr, 1);
              }
              if (EXIT_SUCCESS == ok && nn > 0) {
                if (INT_MAX < nn) ok = EXIT_FAILURE;
                if (EXIT_SUCCESS == ok) {
                  ok = internal_libxs_predict_rf_tree_read(&src, end, (int)ninp,
                    (int)nn, &rf->trees[ti]);
                }
                if (EXIT_SUCCESS == ok && 0 != hasincr) {
                  const int nk = (int)nn * rf->nclass[ti / (int)rf_ntrees];
                  ok = internal_libxs_predict_avail(src, end, (size_t)nk, 8);
                  if (EXIT_SUCCESS == ok) {
                    rf->trees[ti].incr = (double*)malloc(
                      (size_t)nk * sizeof(double));
                    if (NULL == rf->trees[ti].incr) ok = EXIT_FAILURE;
                  }
                  for (k = 0; k < nk && EXIT_SUCCESS == ok; ++k) {
                    ok = internal_libxs_predict_read(&src, end,
                      &rf->trees[ti].incr[k], 8);
                  }
                }
              }
            }
          }
          else ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) model->rf = rf;
          else internal_libxs_predict_rf_free(rf);
        }
      }
    }
    if (EXIT_SUCCESS == ok && 0 == hknn && 1 < version) {
      internal_libxs_predict_read_escape(model, &src, end);
    }
    /* the writer sizes the payload exactly: a remainder signals layout drift */
    if (EXIT_SUCCESS == ok && src != end) ok = EXIT_FAILURE;
    if (EXIT_SUCCESS == ok && 0 == hknn) {
      ok = internal_libxs_predict_load_entries(model);
    }
    if (EXIT_SUCCESS == ok) {
      model->built = 1;
      ++model->nbuild;
      internal_libxs_predict_missing_all(model);
      internal_libxs_predict_support_all(model);
      internal_libxs_predict_keff_all(model);
      internal_libxs_predict_kapply(model);
      if (0 >= model->central) internal_libxs_predict_central_all(model);
    }
    else if (0 == hknn && NULL != model) {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  return model;
}
