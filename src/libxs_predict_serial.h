LIBXS_API_INLINE int internal_libxs_predict_save_hknn(
  const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  const int m = model->ninputs, n = model->noutputs;
  const int p = model->nentries;
  size_t required = 0;
  int c, j;
  required += sizeof(uint32_t) + sizeof(uint16_t);
  required += 4 * sizeof(uint16_t) + sizeof(double);
  required += (size_t)m * 2 * sizeof(double);
  if (NULL != model->weights) required += (size_t)m * sizeof(double);
  for (c = 0; c < model->nclusters; ++c) {
    const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
    required += (size_t)m * sizeof(double);
    required += sizeof(double) + sizeof(uint16_t) + sizeof(uint8_t);
    required += (size_t)n;
    required += (size_t)n * sizeof(uint16_t);
    required += (size_t)cl->nentries * (size_t)m * sizeof(double);
    required += (size_t)cl->nentries * (size_t)n * sizeof(double);
    required += (size_t)cl->nentries * sizeof(uint16_t);
  }
  required += 1;
  if (NULL != model->hknn_po_assignments && NULL != model->hknn_po_clusters) {
    required += (size_t)p * (size_t)n * sizeof(uint16_t);
    for (j = 0; j < n; ++j) {
      const int po_nc = model->hknn_po_nclusters[j];
      required += sizeof(uint16_t);
      for (c = 0; c < po_nc; ++c) {
        const internal_libxs_predict_cluster_t* pcl =
          &model->hknn_po_clusters[j][c];
        required += sizeof(uint16_t) + 2 + sizeof(uint16_t) + sizeof(double);
        required += (size_t)m * sizeof(double);
        if (pcl->nentries > 0) {
          required += (size_t)pcl->nentries * (size_t)m * sizeof(double);
          required += (size_t)pcl->nentries * sizeof(double);
        }
      }
    }
  }
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
    WRITE_F64(model->quality);
    WRITE_BLK(model->input_min, (size_t)m * sizeof(double));
    WRITE_BLK(model->input_rng, (size_t)m * sizeof(double));
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
      for (k = 0; k < cl->nentries; ++k) WRITE_U16(cl->sorted_idx[k]);
    }
    if (NULL != model->hknn_po_assignments
      && NULL != model->hknn_po_clusters)
    {
      for (j = 0; j < n; ++j) {
        int i;
        for (i = 0; i < p; ++i) {
          WRITE_U16(model->hknn_po_assignments[j][i]);
        }
      }
      WRITE_U8(1);
      for (j = 0; j < n; ++j) {
        const int po_nc = model->hknn_po_nclusters[j];
        int ci;
        WRITE_U16(po_nc);
        for (ci = 0; ci < po_nc; ++ci) {
          const internal_libxs_predict_cluster_t* pcl =
            &model->hknn_po_clusters[j][ci];
          WRITE_U16(pcl->nentries);
          WRITE_U8(pcl->k_eff);
          WRITE_U8(pcl->mode ? pcl->mode[0] : 0);
          WRITE_U16(pcl->ndistinct ? pcl->ndistinct[0] : 0);
          WRITE_F64(pcl->dmax);
          WRITE_BLK(pcl->centroid, (size_t)m * sizeof(double));
          if (pcl->nentries > 0) {
            WRITE_BLK(pcl->kd_pts,
              (size_t)pcl->nentries * (size_t)m * sizeof(double));
            WRITE_BLK(pcl->raw_outputs,
              (size_t)pcl->nentries * sizeof(double));
          }
        }
      }
    }
    else {
      WRITE_U8(0);
    }
#undef WRITE_U32
#undef WRITE_U16
#undef WRITE_U8
#undef WRITE_F64
#undef WRITE_BLK
    *size = (size_t)(dst - (unsigned char*)buffer);
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
    int c, j;
    required += sizeof(uint32_t) + 4 * sizeof(uint16_t) + 2 * sizeof(uint8_t);
    required += 5 * sizeof(uint16_t) + 5 * sizeof(uint8_t) + sizeof(uint32_t)
      + sizeof(double);
    required += (size_t)model->ninputs * 2 * sizeof(double);
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
      required += (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double);
      for (j = 0; j < model->noutputs; ++j) {
        required += (size_t)(cl->order[j] + 1) * sizeof(double);
      }
    }
    if (NULL != model->rf) {
      const int n = model->rf->noutputs;
      const int total_trees = model->rf->ntrees * n;
      required += sizeof(uint16_t) + sizeof(uint16_t);
      required += (size_t)n * sizeof(int16_t);
      for (c = 0; c < total_trees; ++c) {
        required += sizeof(uint16_t);
        required += (size_t)model->rf->trees[c].nnodes * (2 + 8 + 2 + 2 + 1);
      }
    }
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
      WRITE_U8(model->eval_mode);
      WRITE_U8(model->diff_order);
      WRITE_U8(model->refine);
      WRITE_U8(NULL != model->decompose_mat ? 1 : 0);
      WRITE_U16(model->order);
      WRITE_U8(model->iterations);
      WRITE_U32(model->nentries);
      WRITE_F64(model->quality);
      WRITE_BLK(model->input_min, (size_t)model->ninputs * sizeof(double));
      WRITE_BLK(model->input_rng, (size_t)model->ninputs * sizeof(double));
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
        WRITE_BLK(cl->kd_pts, (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double));
        WRITE_BLK(cl->raw_outputs, (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double));
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
        for (c = 0; c < total_trees; ++c) {
          const internal_libxs_predict_rf_tree_t* tree = &model->rf->trees[c];
          int k;
          WRITE_U16(tree->nnodes);
          for (k = 0; k < tree->nnodes; ++k) {
            const internal_libxs_predict_rf_node_t* nd = &tree->nodes[k];
            { const int16_t f = (int16_t)nd->feature;
              memcpy(dst, &f, 2); dst += 2;
            }
            WRITE_F64(nd->threshold);
            { const int16_t l = (int16_t)nd->left;
              const int16_t r = (int16_t)nd->right;
              memcpy(dst, &l, 2); dst += 2;
              memcpy(dst, &r, 2); dst += 2;
            }
            WRITE_U8(nd->label);
          }
        }
      }
#undef WRITE_U32
#undef WRITE_U16
#undef WRITE_U8
#undef WRITE_F64
#undef WRITE_BLK
      *size = (size_t)(dst - (unsigned char*)buffer);
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


LIBXS_API_INLINE libxs_predict_t* internal_libxs_predict_load_hknn(
  const unsigned char* src, const unsigned char* end)
{
  libxs_predict_t* model = NULL;
  uint16_t ninp = 0, nout = 0, nclust = 0, has_weights = 0;
  int ok = EXIT_SUCCESS, c, j;
  ok = internal_libxs_predict_read(&src, end, &ninp, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nout, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nclust, 2);
  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_weights, 2);
  if (EXIT_SUCCESS == ok) {
    model = libxs_predict_create((int)ninp, (int)nout);
    if (NULL == model) ok = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == ok) {
    model->decompose = LIBXS_PREDICT_HKNN;
    model->input_min = (double*)malloc((size_t)ninp * sizeof(double));
    model->input_rng = (double*)malloc((size_t)ninp * sizeof(double));
    if (NULL == model->input_min || NULL == model->input_rng) ok = EXIT_FAILURE;
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
      (size_t)nout * 4 * sizeof(double) + (size_t)nout * sizeof(int));
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
      cl->kd_pts = (double*)malloc(
        (size_t)ne * (size_t)ninp * sizeof(double));
      if (NULL == cl->kd_pts) ok = EXIT_FAILURE;
      else ok = internal_libxs_predict_read(&src, end,
        cl->kd_pts, (size_t)ne * (size_t)ninp * sizeof(double));
    }
    if (EXIT_SUCCESS == ok) {
      cl->raw_outputs = (double*)malloc(
        (size_t)ne * (size_t)nout * sizeof(double));
      if (NULL == cl->raw_outputs) ok = EXIT_FAILURE;
      else ok = internal_libxs_predict_read(&src, end,
        cl->raw_outputs, (size_t)ne * (size_t)nout * sizeof(double));
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
  if (EXIT_SUCCESS == ok && src < end && (int)nout > 1) {
    int p = 0, i;
    uint8_t has_po_clusters = 0;
    for (c = 0; c < (int)nclust; ++c) p += model->clusters[c].nentries;
    model->nentries = p;
    model->hknn_po_assignments = (int**)calloc((size_t)nout, sizeof(int*));
    model->hknn_po_nclusters = (int*)calloc((size_t)nout, sizeof(int));
    if (NULL != model->hknn_po_assignments) {
      for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
        model->hknn_po_assignments[j] = (int*)malloc(
          (size_t)p * sizeof(int));
        if (NULL == model->hknn_po_assignments[j]) ok = EXIT_FAILURE;
        else {
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
      for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
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
            uint8_t ke = 0, md = 0;
            uint16_t nd = 0;
            ok = internal_libxs_predict_read(&src, end, &ne, 2);
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ke, 1);
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &md, 1);
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nd, 2);
            if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cls[ci].dmax, 8);
            if (EXIT_SUCCESS == ok) {
              cls[ci].nentries = (int)ne;
              cls[ci].k_eff = (int)ke;
              cls[ci].mode = (int*)malloc(sizeof(int));
              cls[ci].ndistinct = (int*)malloc(sizeof(int));
              cls[ci].centroid = (double*)malloc((size_t)ninp * sizeof(double));
              if (NULL == cls[ci].mode || NULL == cls[ci].ndistinct
                || NULL == cls[ci].centroid) ok = EXIT_FAILURE;
              else {
                cls[ci].mode[0] = (int)md;
                cls[ci].ndistinct[0] = (int)nd;
              }
            }
            if (EXIT_SUCCESS == ok) {
              ok = internal_libxs_predict_read(&src, end,
                cls[ci].centroid, (size_t)ninp * sizeof(double));
            }
            if (EXIT_SUCCESS == ok && ne > 0) {
              cls[ci].kd_pts = (double*)malloc(
                (size_t)ne * (size_t)ninp * sizeof(double));
              cls[ci].raw_outputs = (double*)malloc(
                (size_t)ne * sizeof(double));
              if (NULL == cls[ci].kd_pts || NULL == cls[ci].raw_outputs) {
                ok = EXIT_FAILURE;
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end,
                  cls[ci].kd_pts, (size_t)ne * (size_t)ninp * sizeof(double));
              }
              if (EXIT_SUCCESS == ok) {
                ok = internal_libxs_predict_read(&src, end,
                  cls[ci].raw_outputs, (size_t)ne * sizeof(double));
              }
            }
          }
        }
      }
    }
  }
  if (EXIT_SUCCESS == ok) {
    model->built = 1;
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
  if (NULL != buffer && size >= sizeof(uint32_t) + 4 * sizeof(uint16_t)) {
    const unsigned char* src = (const unsigned char*)buffer;
    const unsigned char* end = src + size;
    uint32_t magic = 0;
    uint16_t version = 0, ninp = 0, nout = 0, nclust = 0;
    int ok = EXIT_SUCCESS;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &magic, 4);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &version, 2);
    if (EXIT_SUCCESS == ok && LIBXS_PREDICT_MAGIC_HKNN == magic
      && LIBXS_PREDICT_VERSION == version)
    {
      return internal_libxs_predict_load_hknn(src, end);
    }
    if (EXIT_SUCCESS == ok && (magic != LIBXS_PREDICT_MAGIC
      || version != LIBXS_PREDICT_VERSION)) ok = EXIT_FAILURE;
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
      uint16_t order = 0;
      ok = internal_libxs_predict_read(&src, end, &has_weights, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_transforms, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_nseries, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_window, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_target, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_decompose, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &eval_mode, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &diff_order, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &refine, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_dmat, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &order, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &iterations, 1);
      { uint32_t nentries = 0;
        double quality = 0;
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nentries, 4);
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &quality, 8);
        if (EXIT_SUCCESS == ok) model->nentries = (int)nentries;
        if (EXIT_SUCCESS == ok) model->quality = quality;
      }
      if (EXIT_SUCCESS == ok) {
        model->nseries = (int)ts_nseries;
        model->window = (int)ts_window;
        model->target = (int)ts_target;
        model->decompose = (int)ts_decompose;
        model->eval_mode = (int)eval_mode;
        model->diff_order = (int)diff_order;
        model->refine = (int)refine;
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
        (size_t)nout * 4 * sizeof(double) + (size_t)nout * sizeof(int));
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
        }
        if (EXIT_SUCCESS == ok) {
          uint8_t ke = 0;
          ok = internal_libxs_predict_read(&src, end, &ke, 1);
          if (EXIT_SUCCESS == ok) cl->k_eff = (int)ke;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->order[j] = (int)v;
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
          cl->raw_outputs = (double*)malloc(
            (size_t)cl->nentries * (size_t)nout * sizeof(double));
          if (NULL == cl->raw_outputs) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->raw_outputs, (size_t)cl->nentries * (size_t)nout * sizeof(double));
          }
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
      if (EXIT_SUCCESS == ok && rf_ntrees > 0 && rf_nouts > 0) {
        internal_libxs_predict_rf_t* rf = (internal_libxs_predict_rf_t*)calloc(
          1, sizeof(internal_libxs_predict_rf_t));
        if (NULL != rf) {
          const int total_trees = (int)rf_ntrees * (int)rf_nouts;
          rf->ntrees = (int)rf_ntrees;
          rf->noutputs = (int)rf_nouts;
          rf->label_offset = (int*)malloc((size_t)rf_nouts * sizeof(int));
          rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
            (size_t)total_trees, sizeof(internal_libxs_predict_rf_tree_t));
          if (NULL != rf->label_offset && NULL != rf->trees) {
            int ti;
            for (j = 0; j < (int)rf_nouts && EXIT_SUCCESS == ok; ++j) {
              int16_t off = 0;
              ok = internal_libxs_predict_read(&src, end, &off, 2);
              rf->label_offset[j] = (int)off;
            }
            for (ti = 0; ti < total_trees && EXIT_SUCCESS == ok; ++ti) {
              uint16_t nn = 0;
              int k;
              ok = internal_libxs_predict_read(&src, end, &nn, 2);
              if (EXIT_SUCCESS == ok && nn > 0) {
                rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
                  (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
                rf->trees[ti].nnodes = (int)nn;
                if (NULL == rf->trees[ti].nodes) ok = EXIT_FAILURE;
                for (k = 0; k < (int)nn && EXIT_SUCCESS == ok; ++k) {
                  int16_t f = 0, l = 0, r = 0;
                  uint8_t lab = 0;
                  ok = internal_libxs_predict_read(&src, end, &f, 2);
                  if (EXIT_SUCCESS == ok) {
                    ok = internal_libxs_predict_read(&src, end,
                      &rf->trees[ti].nodes[k].threshold, 8);
                  }
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &l, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &r, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &lab, 1);
                  rf->trees[ti].nodes[k].feature = (int)f;
                  rf->trees[ti].nodes[k].left = (int)l;
                  rf->trees[ti].nodes[k].right = (int)r;
                  rf->trees[ti].nodes[k].label = (int)lab;
                }
              }
            }
          }
          else ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) model->rf = rf;
          else {
            if (NULL != rf->trees) {
              int ti;
              for (ti = 0; ti < total_trees; ++ti) free(rf->trees[ti].nodes);
              free(rf->trees);
            }
            free(rf->label_offset);
            free(rf);
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok) {
      model->built = 1;
    }
    else if (NULL != model) {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  return model;
}
