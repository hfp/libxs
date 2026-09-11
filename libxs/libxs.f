!=======================================================================!
! Copyright (c) 2009-2026 Hans Pabst                                    !
! Copyright (c) 2009-2026 Intel Corporation                             !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      MODULE LIBXS
        INCLUDE 'libxs_spec.fi'
        PUBLIC :: libxs_gemm_dispatch, libxs_gemm_dispatch_ptr
        PUBLIC :: libxs_syrk_dispatch, libxs_syrk_dispatch_cpy
        PUBLIC :: libxs_syr2k_dispatch, libxs_syr2k_dispatch_cpy
        INTERFACE libxs_gemm_dispatch
          MODULE PROCEDURE libxs_gemm_dispatch_base
          MODULE PROCEDURE libxs_gemm_dispatch_ptr_base
        END INTERFACE
        INTERFACE libxs_gemm_dispatch_ptr
          MODULE PROCEDURE libxs_gemm_dispatch_ptr_base
        END INTERFACE
        INTERFACE libxs_syrk_dispatch
          MODULE PROCEDURE libxs_syrk_dispatch_base
          MODULE PROCEDURE libxs_syrk_dispatch_cpy_base
        END INTERFACE
        INTERFACE libxs_syrk_dispatch_cpy
          MODULE PROCEDURE libxs_syrk_dispatch_cpy_base
        END INTERFACE
        INTERFACE libxs_syr2k_dispatch
          MODULE PROCEDURE libxs_syr2k_dispatch_base
          MODULE PROCEDURE libxs_syr2k_dispatch_cpy_base
        END INTERFACE
        INTERFACE libxs_syr2k_dispatch_cpy
          MODULE PROCEDURE libxs_syr2k_dispatch_cpy_base
        END INTERFACE
      CONTAINS
        INCLUDE 'libxs_procs.fi'
      END MODULE
