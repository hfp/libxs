LIBXS_API_INLINE int internal_libxs_predict_crc(const void* buffer,
  size_t size, uint32_t* crc)
{
  int result = EXIT_SUCCESS;
  if (size > (size_t)0xFFFFFFFFU) result = EXIT_FAILURE;
  else *crc = (uint32_t)libxs_hash_iso3309(buffer, (unsigned int)size, 0);
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
  int recoverable = (0 < p && NULL != model->clusters && NULL != model->input_rng);
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
         would otherwise re-seat all of them again */
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
                cl->kd_pts + (size_t)k * m, e->inputs);
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


LIBXS_API_INLINE int internal_libxs_predict_save_hknn(
  const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  const int m = model->ninputs, n = model->noutputs;
  const int p = model->nentries;
  size_t required = 0;
  int c, j;
  required += sizeof(uint32_t) + sizeof(uint16_t);
  required += 5 * sizeof(uint16_t) + sizeof(double);
  required += (size_t)m * 2 * sizeof(double);
  if (NULL != model->input_knot) required += (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double);
  if (NULL != model->weights) required += (size_t)m * sizeof(double);
  for (c = 0; c < model->nclusters; ++c) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    required += (size_t)m * sizeof(double);
    required += sizeof(double) + sizeof(uint16_t) + sizeof(uint8_t);
    required += (size_t)n;
    required += (size_t)n * sizeof(uint16_t);
    required += (size_t)cl->nentries * (size_t)m * sizeof(double);
    required += (size_t)cl->nentries * (size_t)n * sizeof(double);
    required += (size_t)n * sizeof(double);
    required += (size_t)cl->nentries * sizeof(uint16_t);
  }
  required += 1;
  /* the entry-weight flag, introduced with version 2 */
  required += sizeof(uint8_t);
  /* the resolved neighbour counts, one byte each behind a flag */
  required += sizeof(uint8_t);
  if (NULL != model->k_sel) required += (size_t)n;
  if (NULL != model->hknn_po_assignments && NULL != model->hknn_po_clusters) {
    const int ng = (model->hknn_ngroups > 0) ? model->hknn_ngroups : n;
    required += sizeof(uint16_t) + (size_t)n * sizeof(uint16_t);
    required += (size_t)p * (size_t)ng * sizeof(uint16_t);
    for (j = 0; j < ng; ++j) {
      const int po_nc = model->hknn_po_nclusters[j];
      const int gsz = internal_libxs_predict_gsize(model->hknn_po_groups, n, j);
      required += sizeof(uint16_t);
      if (NULL != model->hknn_po_clusters[j]) {
        for (c = 0; c < po_nc; ++c) {
          const internal_libxs_predict_cluster_t* pcl =
            &model->hknn_po_clusters[j][c];
          required += sizeof(uint16_t) + sizeof(uint8_t) + sizeof(double);
          required += (size_t)gsz * (sizeof(uint8_t) + sizeof(uint16_t));
          required += (size_t)m * sizeof(double);
          if (pcl->nentries > 0) {
            required += (size_t)pcl->nentries * (size_t)m * sizeof(double);
            required += (size_t)pcl->nentries * (size_t)gsz * sizeof(double);
          }
        }
      }
    }
  }
  /* converged escape weights, as in the flat path (see libxs_predict_save) */
  if (NULL != model->escape_w) {
    required += 2 * sizeof(uint8_t)
      + (size_t)n * LIBXS_PREDICT_NESCAPE * sizeof(double);
  }
  else required += sizeof(uint8_t);
  required += sizeof(uint32_t); /* trailing CRC32 */
  if (NULL == buffer) {
    *size = required;
  }
  else if (*size < required) {
    *size = required; result = EXIT_FAILURE;
  }
  else {
    unsigned char* dst = (unsigned char*)buffer;
#define WRITE_U32(V) do { const uint32_t v_=(uint32_t)(V); memcpy(dst,&v_,4); dst+=4; } while(0)
#define WRITE_U16(V) do { const uint16_t v_=(uint16_t)(V); memcpy(dst,&v_,2); dst+=2; } while(0)
#define WRITE_U8(V)  do { *dst++ = (unsigned char)(V); } while(0)
#define WRITE_F64(V) do { const double v_=(V); memcpy(dst,&v_,8); dst+=8; } while(0)
#define WRITE_BLK(PTR,SZ) do { memcpy(dst,(PTR),(SZ)); dst+=(SZ); } while(0)
    WRITE_U32(LIBXS_PREDICT_MAGIC_HKNN);
    WRITE_U16(LIBXS_PREDICT_VERSION);
    WRITE_U16(m);
    WRITE_U16(n);
    WRITE_U16(model->nclusters);
    WRITE_U16(NULL != model->weights ? 1 : 0);
    /* introduced with version 2; an older file simply has no weights */
    WRITE_U8(0 != model->has_eweight ? 1 : 0);
    WRITE_U8(NULL != model->k_sel ? 1 : 0);
    if (NULL != model->k_sel) {
      int ks;
      for (ks = 0; ks < model->noutputs; ++ks) WRITE_U8(model->k_sel[ks]);
    }
    /* introduced with version 2; an older file has no rank coordinate */
    WRITE_U16(NULL != model->input_knot ? LIBXS_PREDICT_KNOTS : 0);
    WRITE_F64(model->quality);
    WRITE_BLK(model->input_min, (size_t)m * sizeof(double));
    WRITE_BLK(model->input_rng, (size_t)m * sizeof(double));
    if (NULL != model->input_knot) {
      WRITE_BLK(model->input_knot, (size_t)m * LIBXS_PREDICT_KNOTS * sizeof(double));
    }
    if (NULL != model->weights) {
      WRITE_BLK(model->weights, (size_t)m * sizeof(double));
    }
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      int k;
      WRITE_BLK(cl->centroid, (size_t)m * sizeof(double));
      WRITE_F64(cl->dmax);
      WRITE_U16(cl->nentries);
      WRITE_U8(cl->k_eff);
      for (j = 0; j < n; ++j) WRITE_U8(cl->mode[j]);
      for (j = 0; j < n; ++j) WRITE_U16(cl->ndistinct[j]);
      WRITE_BLK(cl->kd_pts,
        (size_t)cl->nentries * (size_t)m * sizeof(double));
      WRITE_BLK(cl->raw_outputs,
        (size_t)cl->nentries * (size_t)n * sizeof(double));
      if (0 != model->has_eweight) {
        WRITE_BLK(cl->eweight, (size_t)cl->nentries * sizeof(double));
      }
      WRITE_BLK(cl->out_rms, (size_t)n * sizeof(double));
      for (k = 0; k < cl->nentries; ++k) WRITE_U16(cl->sorted_idx[k]);
    }
    if (NULL != model->hknn_po_assignments
      && NULL != model->hknn_po_clusters)
    {
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
      for (j = 0; j < ng; ++j) {
        int i;
        for (i = 0; i < p; ++i) {
          WRITE_U16(model->hknn_po_assignments[j][i]);
        }
      }
      WRITE_U8(1);
      for (j = 0; j < ng; ++j) {
        const int po_nc = model->hknn_po_nclusters[j];
        const int gsz = internal_libxs_predict_gsize(model->hknn_po_groups, n, j);
        int ci;
        WRITE_U16(po_nc);
        if (NULL != model->hknn_po_clusters[j]) {
          for (ci = 0; ci < po_nc; ++ci) {
            const internal_libxs_predict_cluster_t* pcl =
              &model->hknn_po_clusters[j][ci];
            int gi;
            WRITE_U16(pcl->nentries);
            WRITE_U8(pcl->k_eff);
            for (gi = 0; gi < gsz; ++gi) {
              WRITE_U8(NULL != pcl->mode ? pcl->mode[gi] : 0);
            }
            for (gi = 0; gi < gsz; ++gi) {
              WRITE_U16(NULL != pcl->ndistinct ? pcl->ndistinct[gi] : 0);
            }
            WRITE_F64(pcl->dmax);
            WRITE_BLK(pcl->centroid, (size_t)m * sizeof(double));
            if (pcl->nentries > 0) {
              WRITE_BLK(pcl->kd_pts,
                (size_t)pcl->nentries * (size_t)m * sizeof(double));
              WRITE_BLK(pcl->raw_outputs,
                (size_t)pcl->nentries * (size_t)gsz * sizeof(double));
            }
          }
        }
      }
    }
    else {
      WRITE_U8(0);
    }
    if (NULL != model->escape_w) {
      WRITE_U8(1);
      WRITE_U8(LIBXS_PREDICT_NESCAPE);
      WRITE_BLK(model->escape_w,
        (size_t)n * LIBXS_PREDICT_NESCAPE * sizeof(double));
    }
    else WRITE_U8(0);
    { uint32_t crc = 0;
      result = internal_libxs_predict_crc(buffer,
        (size_t)(dst - (unsigned char*)buffer), &crc);
      if (EXIT_SUCCESS == result) WRITE_U32(crc);
    }
#undef WRITE_U32
#undef WRITE_U16
#undef WRITE_U8
#undef WRITE_F64
#undef WRITE_BLK
    if (EXIT_SUCCESS == result) *size = (size_t)(dst - (unsigned char*)buffer);
  }
  return result;
}


LIBXS_API int libxs_predict_save(const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 == model->built || NULL == size) {
    result = EXIT_FAILURE;
  }
  else if (LIBXS_PREDICT_HKNN == model->decompose
    && NULL != model->hknn_po_assignments)
  {
    result = internal_libxs_predict_save_hknn(model, buffer, size);
  }
  else {
    size_t required = 0;
    int c, j, has_sidx = 1;
    const int has_ew = (0 != model->has_eweight) ? 1 : 0;
    /**
     * A model loaded from a version-1 flat file has no sorted_idx to write, so
     * its presence is a flag rather than an invariant. Fabricating indices
     * instead would silently enable recency weighting on an invented order.
     */
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (0 < cl->nentries && NULL == cl->sorted_idx) has_sidx = 0;
    }
    required += sizeof(uint32_t) + 4 * sizeof(uint16_t) + 2 * sizeof(uint8_t);
    required += 8 * sizeof(uint16_t) + 8 * sizeof(uint8_t) + sizeof(uint32_t)
      + sizeof(double);
    /* the resolved neighbour counts, one byte each behind a flag */
    if (NULL != model->k_sel) required += (size_t)model->noutputs;
    required += (size_t)model->ninputs * 2 * sizeof(double);
    if (NULL != model->input_knot) {
      required += (size_t)model->ninputs * LIBXS_PREDICT_KNOTS * sizeof(double);
    }
    if (NULL != model->weights) required += (size_t)model->ninputs * sizeof(double);
    if (NULL != model->transforms) required += (size_t)model->noutputs * sizeof(uint8_t);
    if (NULL != model->decompose_mat) {
      required += (size_t)model->ninputs * (size_t)model->ninputs * sizeof(double);
    }
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      required += (size_t)model->ninputs * sizeof(double);
      required += sizeof(double);
      required += sizeof(uint16_t) + 2 * sizeof(uint8_t);
      required += (size_t)model->noutputs * 3;
      required += (size_t)model->noutputs * sizeof(uint16_t);
      required += (size_t)model->noutputs * sizeof(double);
      required += (size_t)model->noutputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double);
      if (0 != has_sidx) required += (size_t)cl->nentries * sizeof(uint32_t);
      if (0 != has_ew) required += (size_t)cl->nentries * sizeof(double);
      for (j = 0; j < model->noutputs; ++j) {
        required += (size_t)(cl->order[j] + 1) * sizeof(double);
      }
    }
    if (NULL != model->rf) {
      const int n = model->rf->noutputs;
      const int total_trees = model->rf->ntrees * n;
      required += sizeof(uint16_t) + sizeof(uint16_t);
      required += (size_t)n * sizeof(int16_t);
      /** One read-out kind and one score width per output, one leaf value per
       *  node, and a correction per node and class where a stage was kept. */
      required += (size_t)n * (sizeof(uint8_t) + sizeof(uint8_t));
      for (c = 0; c < total_trees; ++c) {
        required += sizeof(uint16_t) + sizeof(uint8_t);
        required += (size_t)model->rf->trees[c].nnodes * (2 + 8 + 8 + 2 + 2 + 1);
        if (NULL != model->rf->trees[c].incr) {
          required += (size_t)model->rf->trees[c].nnodes
            * (size_t)model->rf->nclass[c / model->rf->ntrees]
            * sizeof(double);
        }
      }
    }
    /**
     * Converged escape weights, when the probability API has been used. These
     * are an optimization, not model state: a fresh bank re-learns them within a
     * few hundred queries, at a measured cost of 0.004-0.03 bits. Carrying them
     * means a rebuilt or reloaded model does not re-pay that transient. Written
     * last and guarded by a presence flag so a reader that predates them, or a
     * model that never scored a probability, is unaffected.
     */
    if (NULL != model->escape_w) {
      required += 2 * sizeof(uint8_t);
      required += (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE
        * sizeof(double);
    }
    else required += sizeof(uint8_t);
    required += sizeof(uint32_t); /* trailing CRC32 */
    if (NULL == buffer) {
      *size = required;
    }
    else if (*size < required) {
      *size = required;
      result = EXIT_FAILURE;
    }
    else {
      unsigned char* dst = (unsigned char*)buffer;
#define WRITE_U32(V) do { const uint32_t v_=(uint32_t)(V); memcpy(dst,&v_,4); dst+=4; } while(0)
#define WRITE_U16(V) do { const uint16_t v_=(uint16_t)(V); memcpy(dst,&v_,2); dst+=2; } while(0)
#define WRITE_U8(V)  do { *dst++ = (unsigned char)(V); } while(0)
#define WRITE_F64(V) do { const double v_=(V); memcpy(dst,&v_,8); dst+=8; } while(0)
#define WRITE_BLK(PTR,SZ) do { memcpy(dst,(PTR),(SZ)); dst+=(SZ); } while(0)
      WRITE_U32(LIBXS_PREDICT_MAGIC);
      WRITE_U16(LIBXS_PREDICT_VERSION);
      WRITE_U16(model->ninputs);
      WRITE_U16(model->noutputs);
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
       * Biased by one so the unsigned field can carry the confidence-gated
       * default (-1). It also makes a file written before 0 meant "off" decode
       * to that default rather than to off: every such file stored 0, because
       * the setting had no other value and the default was 0 at the time.
       */
      WRITE_U8(LIBXS_CLMP(model->refine + 1, 0, 255));
      WRITE_U8(NULL != model->decompose_mat ? 1 : 0);
      WRITE_U8(has_sidx);
      /* introduced with version 2; an older file simply has no weights */
      WRITE_U8(has_ew);
      /* introduced with version 2; an older file derives the count instead */
      WRITE_U8(NULL != model->k_sel ? 1 : 0);
      if (NULL != model->k_sel) {
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(model->k_sel[j]);
      }
      WRITE_U16(model->order);
      WRITE_U8(model->iterations);
      WRITE_U32(model->nentries);
      /* introduced with version 2; an older file has no rank coordinate */
      WRITE_U16(NULL != model->input_knot ? LIBXS_PREDICT_KNOTS : 0);
      WRITE_F64(model->quality);
      WRITE_BLK(model->input_min, (size_t)model->ninputs * sizeof(double));
      WRITE_BLK(model->input_rng, (size_t)model->ninputs * sizeof(double));
      if (NULL != model->input_knot) {
        WRITE_BLK(model->input_knot,
          (size_t)model->ninputs * LIBXS_PREDICT_KNOTS * sizeof(double));
      }
      if (NULL != model->weights) {
        WRITE_BLK(model->weights, (size_t)model->ninputs * sizeof(double));
      }
      if (NULL != model->transforms) {
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(model->transforms[j]);
      }
      if (NULL != model->decompose_mat) {
        const size_t msz = (size_t)model->ninputs * (size_t)model->ninputs;
        WRITE_BLK(model->decompose_mat, msz * sizeof(double));
      }
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        WRITE_BLK(cl->centroid, (size_t)model->ninputs * sizeof(double));
        WRITE_F64(cl->dmax);
        WRITE_U16(cl->nentries);
        WRITE_U8(cl->maxorder);
        WRITE_U8(cl->k_eff);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->order[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->interpolated[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->mode[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U16(cl->ndistinct[j]);
        WRITE_BLK(cl->errors, (size_t)model->noutputs * sizeof(double));
        WRITE_BLK(cl->out_rms, (size_t)model->noutputs * sizeof(double));
        WRITE_BLK(cl->kd_pts, (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double));
        WRITE_BLK(cl->raw_outputs, (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double));
        if (0 != has_ew) {
          WRITE_BLK(cl->eweight, (size_t)cl->nentries * sizeof(double));
        }
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
        for (j = 0; j < model->noutputs; ++j) {
          WRITE_BLK(cl->coeffs + (size_t)j * (cl->maxorder + 1),
            (size_t)(cl->order[j] + 1) * sizeof(double));
        }
      }
      if (NULL != model->rf) {
        const int total_trees = model->rf->ntrees * model->rf->noutputs;
        WRITE_U16(model->rf->ntrees);
        WRITE_U16(model->rf->noutputs);
        for (j = 0; j < model->rf->noutputs; ++j) {
          const int16_t off = (int16_t)model->rf->label_offset[j];
          memcpy(dst, &off, 2); dst += 2;
        }
        for (j = 0; j < model->rf->noutputs; ++j) {
          WRITE_U8(model->rf->regress[j]);
        }
        for (j = 0; j < model->rf->noutputs; ++j) {
          WRITE_U8(model->rf->nclass[j]);
        }
        for (c = 0; c < total_trees; ++c) {
          const internal_libxs_predict_rf_tree_t* tree = &model->rf->trees[c];
          int k;
          WRITE_U16(tree->nnodes);
          WRITE_U8(NULL != tree->incr ? 1 : 0);
          for (k = 0; k < tree->nnodes; ++k) {
            const internal_libxs_predict_rf_node_t* nd = &tree->nodes[k];
            { const int16_t f = (int16_t)nd->feature;
              memcpy(dst, &f, 2); dst += 2;
            }
            WRITE_F64(nd->threshold);
            WRITE_F64(nd->value);
            { const int16_t l = (int16_t)nd->left;
              const int16_t r = (int16_t)nd->right;
              memcpy(dst, &l, 2); dst += 2;
              memcpy(dst, &r, 2); dst += 2;
            }
            WRITE_U8(nd->label);
          }
          if (NULL != tree->incr) {
            const int nk = tree->nnodes * model->rf->nclass[c / model->rf->ntrees];
            for (k = 0; k < nk; ++k) WRITE_F64(tree->incr[k]);
          }
        }
      }
      if (NULL != model->escape_w) {
        WRITE_U8(1);
        WRITE_U8(LIBXS_PREDICT_NESCAPE);
        WRITE_BLK(model->escape_w,
          (size_t)model->noutputs * LIBXS_PREDICT_NESCAPE * sizeof(double));
      }
      else WRITE_U8(0);
      { uint32_t crc = 0;
        result = internal_libxs_predict_crc(buffer,
          (size_t)(dst - (unsigned char*)buffer), &crc);
        if (EXIT_SUCCESS == result) WRITE_U32(crc);
      }
#undef WRITE_U32
#undef WRITE_U16
#undef WRITE_U8
#undef WRITE_F64
#undef WRITE_BLK
      if (EXIT_SUCCESS == result) *size = (size_t)(dst - (unsigned char*)buffer);
    }
  }
  return result;
}


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


LIBXS_API_INLINE libxs_predict_t* internal_libxs_predict_load_hknn(
  const unsigned char* src, const unsigned char* end, int version)
{
  libxs_predict_t* model = NULL;
  uint16_t ninp = 0, nout = 0, nclust = 0, has_weights = 0, nknots = 0;
  int ok = EXIT_SUCCESS, c, j, has_ew = 0;
  uint8_t* ksel = NULL;
  ok = internal_libxs_predict_read(&src, end, &ninp, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nout, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nclust, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_weights, 2);
  if (EXIT_SUCCESS == ok && 1 < version) {
    uint8_t v = 0;
    ok = internal_libxs_predict_read(&src, end, &v, 1);
    if (EXIT_SUCCESS == ok) has_ew = (0 != v);
  }
  if (EXIT_SUCCESS == ok && 1 < version) {
    uint8_t v = 0;
    ok = internal_libxs_predict_read(&src, end, &v, 1);
    if (EXIT_SUCCESS == ok && 0 != v) {
      ksel = (uint8_t*)malloc((size_t)nout);
      if (NULL == ksel) {
        ok = EXIT_FAILURE;
      }
      else {
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          ok = internal_libxs_predict_read(&src, end, ksel + j, 1);
        }
      }
    }
  }
  if (EXIT_SUCCESS == ok) {
    model = libxs_predict_create((int)ninp, (int)nout);
    if (NULL == model) ok = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == ok && NULL != ksel) {
    model->k_sel = (int*)malloc((size_t)nout * sizeof(int));
    if (NULL == model->k_sel) {
      ok = EXIT_FAILURE;
    }
    else {
      for (j = 0; j < (int)nout; ++j) model->k_sel[j] = (int)ksel[j];
    }
  }
  free(ksel);
  if (EXIT_SUCCESS == ok) {
    model->decompose = LIBXS_PREDICT_HKNN;
    model->input_min = (double*)malloc((size_t)ninp * sizeof(double));
    model->input_rng = (double*)malloc((size_t)ninp * sizeof(double));
    if (NULL == model->input_min || NULL == model->input_rng) ok = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == ok && 1 < version) {
    ok = internal_libxs_predict_read(&src, end, &nknots, 2);
  }
  { double quality = 0;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &quality, 8);
    if (EXIT_SUCCESS == ok) model->quality = quality;
  }
  if (EXIT_SUCCESS == ok) {
    ok = internal_libxs_predict_read(&src, end,
      model->input_min, (size_t)ninp * sizeof(double));
  }
  if (EXIT_SUCCESS == ok) {
    ok = internal_libxs_predict_read(&src, end,
      model->input_rng, (size_t)ninp * sizeof(double));
  }
  if (EXIT_SUCCESS == ok && 0 != nknots) {
    if (LIBXS_PREDICT_KNOTS == nknots) {
      model->input_knot = (double*)malloc(
        (size_t)ninp * LIBXS_PREDICT_KNOTS * sizeof(double));
      if (NULL == model->input_knot) ok = EXIT_FAILURE;
      else {
        ok = internal_libxs_predict_read(&src, end, model->input_knot,
          (size_t)ninp * LIBXS_PREDICT_KNOTS * sizeof(double));
      }
    }
    else ok = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == ok && 0 != has_weights) {
    model->weights = (double*)malloc((size_t)ninp * sizeof(double));
    if (NULL != model->weights) {
      ok = internal_libxs_predict_read(&src, end,
        model->weights, (size_t)ninp * sizeof(double));
    }
    else ok = EXIT_FAILURE;
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
    uint16_t ne = 0;
    uint8_t ke = 0;
    int k;
    cl->centroid = (double*)malloc((size_t)ninp * sizeof(double));
    cl->mode = (int*)malloc((size_t)nout * sizeof(int));
    cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
    if (NULL == cl->centroid || NULL == cl->mode || NULL == cl->ndistinct) {
      ok = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == ok) {
      ok = internal_libxs_predict_read(&src, end,
        cl->centroid, (size_t)ninp * sizeof(double));
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
    for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
      uint8_t v = 0;
      ok = internal_libxs_predict_read(&src, end, &v, 1);
      cl->mode[j] = (int)v;
    }
    for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
      uint16_t v = 0;
      ok = internal_libxs_predict_read(&src, end, &v, 2);
      cl->ndistinct[j] = (int)v;
    }
    if (EXIT_SUCCESS == ok) {
      ok = internal_libxs_predict_avail(src, end,
        (size_t)ne * (size_t)ninp, sizeof(double));
    }
    if (EXIT_SUCCESS == ok) {
      cl->kd_pts = (double*)malloc(
        (size_t)ne * (size_t)ninp * sizeof(double));
      if (NULL == cl->kd_pts) ok = EXIT_FAILURE;
      else ok = internal_libxs_predict_read(&src, end,
        cl->kd_pts, (size_t)ne * (size_t)ninp * sizeof(double));
    }
    if (EXIT_SUCCESS == ok) {
      ok = internal_libxs_predict_avail(src, end,
        (size_t)ne * (size_t)nout, sizeof(double));
    }
    if (EXIT_SUCCESS == ok) {
      cl->raw_outputs = (double*)malloc(
        (size_t)ne * (size_t)nout * sizeof(double));
      if (NULL == cl->raw_outputs) ok = EXIT_FAILURE;
      else ok = internal_libxs_predict_read(&src, end,
        cl->raw_outputs, (size_t)ne * (size_t)nout * sizeof(double));
    }
    if (EXIT_SUCCESS == ok && 0 != has_ew) {
      ok = internal_libxs_predict_avail(src, end, (size_t)ne, sizeof(double));
    }
    if (EXIT_SUCCESS == ok && 0 != has_ew) {
      cl->eweight = (double*)malloc((size_t)ne * sizeof(double));
      if (NULL == cl->eweight) ok = EXIT_FAILURE;
      else {
        ok = internal_libxs_predict_read(&src, end, cl->eweight,
          (size_t)ne * sizeof(double));
        if (EXIT_SUCCESS == ok) model->has_eweight = 1;
      }
    }
    /**
     * out_rms arrived with version 2. A version-1 file carries no fit residual,
     * and every consumer reads it as "no calibration" when it is zero and falls
     * back to out_var, which is derived below - what version 1 itself did.
     */
    if (EXIT_SUCCESS == ok) {
      if (1 < version) {
        cl->out_rms = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->out_rms) ok = EXIT_FAILURE;
        else ok = internal_libxs_predict_read(&src, end,
          cl->out_rms, (size_t)nout * sizeof(double));
      }
      else {
        cl->out_rms = (double*)calloc((size_t)nout, sizeof(double));
        if (NULL == cl->out_rms) ok = EXIT_FAILURE;
      }
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
        for (k = 0; k < (int)ne && EXIT_SUCCESS == ok; ++k) {
          uint16_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 2);
          cl->sorted_idx[k] = (int)v;
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
        if (cl->sorted_idx[k] >= total) ok = EXIT_FAILURE;
      }
    }
  }
  if (EXIT_SUCCESS == ok && src < end && (int)nout > 1) {
    int p = 0, i, ng = (int)nout;
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
    if (EXIT_SUCCESS == ok && NULL != model->hknn_po_assignments) {
      for (j = 0; j < ng && EXIT_SUCCESS == ok; ++j) {
        ok = internal_libxs_predict_avail(src, end, (size_t)p, sizeof(uint16_t));
        if (EXIT_SUCCESS == ok) {
          model->hknn_po_assignments[j] = (int*)malloc(
            (size_t)p * sizeof(int));
          if (NULL == model->hknn_po_assignments[j]) ok = EXIT_FAILURE;
        }
        if (EXIT_SUCCESS == ok) {
          for (i = 0; i < p && EXIT_SUCCESS == ok; ++i) {
            uint16_t v = 0;
            ok = internal_libxs_predict_read(&src, end, &v, 2);
            model->hknn_po_assignments[j][i] = (int)v;
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok) {
      ok = internal_libxs_predict_read(&src, end, &has_po_clusters, 1);
    }
    if (EXIT_SUCCESS == ok && 0 != has_po_clusters) {
      model->hknn_po_clusters = (internal_libxs_predict_cluster_t**)calloc(
        (size_t)nout, sizeof(internal_libxs_predict_cluster_t*));
      if (NULL == model->hknn_po_clusters) ok = EXIT_FAILURE;
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
            uint16_t ne = 0;
            uint8_t ke = 0;
            int gi;
            ok = internal_libxs_predict_read(&src, end, &ne, 2);
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ke, 1);
            if (EXIT_SUCCESS == ok && ke > LIBXS_PREDICT_KNN) ok = EXIT_FAILURE;
            if (EXIT_SUCCESS == ok) {
              cls[ci].nentries = (int)ne;
              cls[ci].k_eff = (int)ke;
              cls[ci].mode = (int*)calloc((size_t)gsz, sizeof(int));
              cls[ci].ndistinct = (int*)calloc((size_t)gsz, sizeof(int));
              cls[ci].centroid = (double*)malloc((size_t)ninp * sizeof(double));
              if (NULL == cls[ci].mode || NULL == cls[ci].ndistinct
                || NULL == cls[ci].centroid) ok = EXIT_FAILURE;
            }
            for (gi = 0; gi < gsz && EXIT_SUCCESS == ok; ++gi) {
              uint8_t v = 0;
              ok = internal_libxs_predict_read(&src, end, &v, 1);
              if (EXIT_SUCCESS == ok) cls[ci].mode[gi] = (int)v;
            }
            for (gi = 0; gi < gsz && EXIT_SUCCESS == ok; ++gi) {
              uint16_t v = 0;
              ok = internal_libxs_predict_read(&src, end, &v, 2);
              if (EXIT_SUCCESS == ok) cls[ci].ndistinct[gi] = (int)v;
            }
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cls[ci].dmax, 8);
            if (EXIT_SUCCESS == ok) {
              ok = internal_libxs_predict_read(&src, end,
                cls[ci].centroid, (size_t)ninp * sizeof(double));
            }
            if (EXIT_SUCCESS == ok && ne > 0) {
              ok = internal_libxs_predict_avail(src, end,
                (size_t)ne * ((size_t)ninp + (size_t)gsz), sizeof(double));
              if (EXIT_SUCCESS == ok) {
                cls[ci].kd_pts = (double*)malloc(
                  (size_t)ne * (size_t)ninp * sizeof(double));
                cls[ci].raw_outputs = (double*)malloc(
                  (size_t)ne * (size_t)gsz * sizeof(double));
                if (NULL == cls[ci].kd_pts || NULL == cls[ci].raw_outputs) {
                  ok = EXIT_FAILURE;
                }
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end,
                  cls[ci].kd_pts, (size_t)ne * (size_t)ninp * sizeof(double));
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end, cls[ci].raw_outputs,
                  (size_t)ne * (size_t)gsz * sizeof(double));
              }
            }
          }
        }
      }
    }
    /* an assignment selects a per-output cluster, so it must index one */
    if (EXIT_SUCCESS == ok && NULL != model->hknn_po_clusters) {
      for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
        const int po_nc = model->hknn_po_nclusters[j];
        if (NULL == model->hknn_po_assignments[j]
          || NULL == model->hknn_po_clusters[j]) continue;
        for (i = 0; i < p && EXIT_SUCCESS == ok; ++i) {
          if (model->hknn_po_assignments[j][i] >= po_nc) ok = EXIT_FAILURE;
        }
      }
    }
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
    int has_sidx = 0, has_ew = 0;
    int ok = internal_libxs_predict_read(&src, end, &magic, 4);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &version, 2);
    if (EXIT_SUCCESS == ok
      && (0 == version || LIBXS_PREDICT_VERSION < version)) ok = EXIT_FAILURE;
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
    }
    if (EXIT_SUCCESS == ok) {
      uint8_t has_weights = 0, has_transforms = 0, has_dmat = 0;
      uint8_t eval_mode = 0, diff_order = 0, refine = 0, iterations = 0;
      uint16_t ts_nseries = 0, ts_window = 0, ts_target = 0, ts_decompose = 0;
      uint16_t ts_naux = 0, ts_nderiv = 0;
      uint16_t order = 0, nknots = 0;
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
      if (EXIT_SUCCESS == ok && 1 < version) {
        uint8_t v = 0;
        ok = internal_libxs_predict_read(&src, end, &v, 1);
        if (EXIT_SUCCESS == ok) has_sidx = (0 != v);
      }
      if (EXIT_SUCCESS == ok && 1 < version) {
        uint8_t v = 0;
        ok = internal_libxs_predict_read(&src, end, &v, 1);
        if (EXIT_SUCCESS == ok) has_ew = (0 != v);
      }
      if (EXIT_SUCCESS == ok && 1 < version) {
        uint8_t v = 0;
        ok = internal_libxs_predict_read(&src, end, &v, 1);
        if (EXIT_SUCCESS == ok && 0 != v) {
          model->k_sel = (int*)malloc((size_t)model->noutputs * sizeof(int));
          if (NULL == model->k_sel) ok = EXIT_FAILURE;
          else {
            int ks;
            for (ks = 0; ks < model->noutputs && EXIT_SUCCESS == ok; ++ks) {
              uint8_t kv = 0;
              ok = internal_libxs_predict_read(&src, end, &kv, 1);
              model->k_sel[ks] = (int)kv;
            }
          }
        }
      }
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &order, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &iterations, 1);
      { uint32_t nentries = 0;
        double quality = 0;
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nentries, 4);
        if (EXIT_SUCCESS == ok && 1 < version) {
          ok = internal_libxs_predict_read(&src, end, &nknots, 2);
        }
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &quality, 8);
        if (EXIT_SUCCESS == ok) model->nentries = (int)nentries;
        if (EXIT_SUCCESS == ok) model->quality = quality;
      }
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
      }
      model->input_min = (double*)malloc((size_t)ninp * sizeof(double));
      model->input_rng = (double*)malloc((size_t)ninp * sizeof(double));
      if (NULL == model->input_min || NULL == model->input_rng) ok = EXIT_FAILURE;
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end,
          model->input_min, (size_t)ninp * sizeof(double));
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end,
          model->input_rng, (size_t)ninp * sizeof(double));
      }
      if (EXIT_SUCCESS == ok && 0 != nknots) {
        if (LIBXS_PREDICT_KNOTS == nknots) {
          model->input_knot = (double*)malloc(
            (size_t)ninp * LIBXS_PREDICT_KNOTS * sizeof(double));
          if (NULL == model->input_knot) ok = EXIT_FAILURE;
          else {
            ok = internal_libxs_predict_read(&src, end, model->input_knot,
              (size_t)ninp * LIBXS_PREDICT_KNOTS * sizeof(double));
          }
        }
        else ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok && 0 != has_weights) {
        model->weights = (double*)malloc((size_t)ninp * sizeof(double));
        if (NULL != model->weights) {
          ok = internal_libxs_predict_read(&src, end,
            model->weights, (size_t)ninp * sizeof(double));
        }
        else ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok && 0 != has_transforms) {
        int j;
        model->transforms = (int*)calloc((size_t)nout, sizeof(int));
        if (NULL != model->transforms) {
          for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
            uint8_t v = 0;
            ok = internal_libxs_predict_read(&src, end, &v, 1);
            model->transforms[j] = (int)v;
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
        cl->centroid = (double*)malloc((size_t)ninp * sizeof(double));
        cl->order = (int*)malloc((size_t)nout * sizeof(int));
        cl->interpolated = (int*)malloc((size_t)nout * sizeof(int));
        cl->mode = (int*)malloc((size_t)nout * sizeof(int));
        cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
        cl->errors = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->centroid || NULL == cl->order || NULL == cl->interpolated
          || NULL == cl->mode || NULL == cl->ndistinct || NULL == cl->errors) ok = EXIT_FAILURE;
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->centroid, (size_t)ninp * sizeof(double));
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
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->interpolated[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->mode[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint16_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 2);
          cl->ndistinct[j] = (int)v;
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
            else ok = internal_libxs_predict_read(&src, end,
              cl->out_rms, (size_t)nout * sizeof(double));
          }
          else {
            cl->out_rms = (double*)calloc((size_t)nout, sizeof(double));
            if (NULL == cl->out_rms) ok = EXIT_FAILURE;
          }
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_avail(src, end,
            (size_t)cl->nentries * (size_t)ninp, sizeof(double));
        }
        if (EXIT_SUCCESS == ok) {
          cl->kd_pts = (double*)malloc(
            (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          if (NULL == cl->kd_pts) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->kd_pts, (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          }
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_avail(src, end,
            (size_t)cl->nentries * (size_t)nout, sizeof(double));
        }
        if (EXIT_SUCCESS == ok) {
          cl->raw_outputs = (double*)malloc(
            (size_t)cl->nentries * (size_t)nout * sizeof(double));
          if (NULL == cl->raw_outputs) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->raw_outputs, (size_t)cl->nentries * (size_t)nout * sizeof(double));
          }
        }
        if (EXIT_SUCCESS == ok && 0 != has_ew) {
          ok = internal_libxs_predict_avail(src, end,
            (size_t)cl->nentries, sizeof(double));
        }
        if (EXIT_SUCCESS == ok && 0 != has_ew) {
          cl->eweight = (double*)malloc((size_t)cl->nentries * sizeof(double));
          if (NULL == cl->eweight) ok = EXIT_FAILURE;
          else {
            ok = internal_libxs_predict_read(&src, end, cl->eweight,
              (size_t)cl->nentries * sizeof(double));
            if (EXIT_SUCCESS == ok) model->has_eweight = 1;
          }
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
      ok = internal_libxs_predict_read(&src, end, &rf_ntrees, 2);
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
                 zero or beyond the fold would leave that indexing unbounded */
              if (EXIT_SUCCESS == ok && (0 == ncl || 128 < ncl)) {
                ok = EXIT_FAILURE;
              }
              else rf->nclass[j] = (int)ncl;
            }
            for (ti = 0; ti < total_trees && EXIT_SUCCESS == ok; ++ti) {
              uint16_t nn = 0;
              uint8_t hasincr = 0;
              int k;
              ok = internal_libxs_predict_read(&src, end, &nn, 2);
              if (EXIT_SUCCESS == ok && 1 < version) {
                ok = internal_libxs_predict_read(&src, end, &hasincr, 1);
              }
              if (EXIT_SUCCESS == ok && nn > 0) {
                ok = internal_libxs_predict_avail(src, end, (size_t)nn,
                  (1 < version) ? (2 + 8 + 8 + 2 + 2 + 1) : (2 + 8 + 2 + 2 + 1));
                if (EXIT_SUCCESS == ok) {
                  rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
                    (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
                  rf->trees[ti].nnodes = (int)nn;
                  if (NULL == rf->trees[ti].nodes) ok = EXIT_FAILURE;
                }
                for (k = 0; k < (int)nn && EXIT_SUCCESS == ok; ++k) {
                  int16_t f = 0, l = 0, r = 0;
                  uint8_t lab = 0;
                  ok = internal_libxs_predict_read(&src, end, &f, 2);
                  if (EXIT_SUCCESS == ok) {
                    ok = internal_libxs_predict_read(&src, end,
                      &rf->trees[ti].nodes[k].threshold, 8);
                  }
                  if (EXIT_SUCCESS == ok && 1 < version) {
                    ok = internal_libxs_predict_read(&src, end,
                      &rf->trees[ti].nodes[k].value, 8);
                  }
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &l, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &r, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &lab, 1);
                  /* traversal dereferences inputs[feature] and descends into
                     left/right, hence both must stay in range (-1 is a leaf) */
                  if (EXIT_SUCCESS == ok && (f >= (int16_t)ninp || f < -1
                    || l >= (int16_t)nn || l < -1
                    || r >= (int16_t)nn || r < -1)) ok = EXIT_FAILURE;
                  if (EXIT_SUCCESS == ok) {
                    rf->trees[ti].nodes[k].feature = (int)f;
                    rf->trees[ti].nodes[k].left = (int)l;
                    rf->trees[ti].nodes[k].right = (int)r;
                    rf->trees[ti].nodes[k].label = (int)lab;
                    if (1 >= version) rf->trees[ti].nodes[k].value = (double)lab;
                  }
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
          else {
            if (NULL != rf->trees) {
              int ti;
              for (ti = 0; ti < total_trees; ++ti) {
                free(rf->trees[ti].nodes);
                free(rf->trees[ti].incr);
              }
              free(rf->trees);
            }
            free(rf->label_offset);
            free(rf->regress);
            free(rf->nclass);
            free(rf);
          }
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
      const char* tenv = getenv("LIBXS_PREDICT_TANGENT");
      if (NULL != tenv) model->tangent = atoi(tenv);
      if (0 != model->tangent && NULL != model->clusters) {
        int c;
        for (c = 0; c < model->nclusters; ++c) {
          internal_libxs_predict_cluster_tangent(
            &model->clusters[c], model->ninputs, model->tangent);
        }
      }
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
