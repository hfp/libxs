!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      MODULE LIBXS
        INCLUDE 'libxs_spec.fi'
        PUBLIC :: libxs_gemm_dispatch
        PUBLIC :: libxs_syrk_dispatch
        PUBLIC :: libxs_syr2k_dispatch
        INTERFACE libxs_gemm_dispatch
          MODULE PROCEDURE libxs_gemm_dispatch_base
        END INTERFACE
        INTERFACE libxs_syrk_dispatch
          MODULE PROCEDURE libxs_syrk_dispatch_base
        END INTERFACE
        INTERFACE libxs_syr2k_dispatch
          MODULE PROCEDURE libxs_syr2k_dispatch_base
        END INTERFACE
      CONTAINS
        INCLUDE 'libxs_procs.fi'
      END MODULE
