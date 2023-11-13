!-------------------------------------------------------------------------------
!
! NAME: ortlrini.f90
!
! DESC: Initialize orthogonalization of a low rank representation 
!
! AUTH: Yang Xinshuo
!
! DATE: 20141003
!
!-------------------------------------------------------------------------------
subroutine ortlrini(lrank,pivots)
    implicit none
    integer*8, intent(in)    :: lrank
    integer*8, intent(inout) :: pivots(lrank)
    integer*8                :: k
    do k=1,lrank
        pivots(k) = k
    enddo
end subroutine ortlrini
!-------------------------------------------------------------------------------
!
! END: ortlrini.f90
!
!-------------------------------------------------------------------------------