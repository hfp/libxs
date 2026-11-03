!=======================================================================!
! Copyright (c) 2009-2026 Hans Pabst                                    !
! Copyright (c) 2009-2026 Intel Corporation                             !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      ! Packs n signed values of up to maxbits bits as libxs_vpack does
      ! for a group of 8 with LIBXS_PACK=1: header bytes (the group,
      ! then a width per group), then 8 interleaved streams (value 8*c+l
      ! in stream l, all at the same bit position), so every shift is
      ! uniform over the lanes. Returns the number of words written.
      FUNCTION vpack_f(packed, values, n, maxbits)                      &
     &  BIND(C, NAME="vpack_f")
        USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T, C_SIZE_T,     &
     &                                         C_INT
        IMPLICIT NONE
        INTEGER(C_SIZE_T) :: vpack_f
        INTEGER(C_SIZE_T), INTENT(IN), VALUE :: n
        INTEGER(C_INT), INTENT(IN), VALUE :: maxbits
        INTEGER(C_INT64_T), INTENT(INOUT) :: packed(*)
        INTEGER(C_INT64_T), INTENT(IN) :: values(n)
        INTEGER(C_INT64_T) :: z(8)
        INTEGER(C_SIZE_T) :: ngroups, nhead, i, j, lo, b, m
        INTEGER :: w, off, s

        vpack_f = 0
        IF (0 < n .AND. 0 < maxbits .AND. 64 >= maxbits) THEN
          ngroups = (n + 7) / 8
          nhead = (1 + ngroups + 7) / 8
          packed(1:nhead) = 0
          packed(1) = 8 ! group size
          s = 64 - maxbits
          j = 0
          off = 0
          DO i = 0, ngroups - 1
            m = MIN(8_C_SIZE_T, n - 8 * i)
            z = 0
            ! sign-extended from maxbits, then the sign folded (zig-zag)
            z(1:m) = SHIFTA(ISHFT(values(8*i+1:8*i+m), s), s)
            z = IEOR(ISHFT(z, 1), SHIFTA(z, 63))
            w = 64 - LEADZ(IANY(z))
            b = i + 1 ! header byte of the width
            packed(b/8+1) = IOR(packed(b/8+1),                          &
     &        ISHFT(INT(w, C_INT64_T), INT(8 * MOD(b, 8_C_SIZE_T))))
            IF (0 < w) THEN
              lo = nhead + 8 * j
              IF (0 /= off) THEN
                packed(lo+1:lo+8) = IOR(packed(lo+1:lo+8),              &
     &            ISHFT(z, off))
              ELSE
                packed(lo+1:lo+8) = z
              END IF
              IF (64 < off + w) THEN ! spills into the next word
                packed(lo+9:lo+16) = ISHFT(z, off - 64)
              END IF
              off = off + w
              IF (64 <= off) THEN
                off = off - 64
                j = j + 1
              END IF
            END IF
          END DO
          vpack_f = nhead + 8 * j
          IF (0 /= off) vpack_f = vpack_f + 8 ! partial last word
        END IF
      END FUNCTION

      ! Unpacks n values written by vpack_f (or by libxs_vpack for a
      ! group of 8 with LIBXS_PACK=1); returns the words read.
      FUNCTION vunpack_f(values, packed, n) BIND(C, NAME="vunpack_f")
        USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT64_T, C_SIZE_T
        IMPLICIT NONE
        INTEGER(C_SIZE_T) :: vunpack_f
        INTEGER(C_SIZE_T), INTENT(IN), VALUE :: n
        INTEGER(C_INT64_T), INTENT(OUT) :: values(n)
        INTEGER(C_INT64_T), INTENT(IN) :: packed(*)
        INTEGER(C_INT64_T) :: z(8), mask
        INTEGER(C_SIZE_T) :: ngroups, nhead, i, j, lo, b, m
        INTEGER :: w, off

        vunpack_f = 0
        IF (0 < n) THEN
        IF (8 == IAND(packed(1), 255_C_INT64_T)) THEN
          ngroups = (n + 7) / 8
          nhead = (1 + ngroups + 7) / 8
          j = 0
          off = 0
          DO i = 0, ngroups - 1
            m = MIN(8_C_SIZE_T, n - 8 * i)
            b = i + 1 ! header byte of the width
            w = INT(IAND(ISHFT(packed(b/8+1),                           &
     &        -INT(8 * MOD(b, 8_C_SIZE_T))), 255_C_INT64_T))
            IF (0 < w) THEN
              mask = ISHFT(-1_C_INT64_T, w - 64)
              lo = nhead + 8 * j
              IF (64 < off + w) THEN ! spills into the next word
                z = IAND(IOR(ISHFT(packed(lo+1:lo+8), -off),            &
     &                       ISHFT(packed(lo+9:lo+16), 64 - off)), mask)
              ELSE
                z = IAND(ISHFT(packed(lo+1:lo+8), -off), mask)
              END IF
              off = off + w
              IF (64 <= off) THEN
                off = off - 64
                j = j + 1
              END IF
            ELSE
              z = 0
            END IF
            values(8*i+1:8*i+m) = IEOR(ISHFT(z(1:m), -1),               &
     &        -IAND(z(1:m), 1_C_INT64_T)) ! zig-zag
          END DO
          vunpack_f = nhead + 8 * j
          IF (0 /= off) vunpack_f = vunpack_f + 8 ! partial last word
        END IF
        END IF
      END FUNCTION
