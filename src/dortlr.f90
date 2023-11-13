!-------------------------------------------------------------------------------
!
! NAME: dortlr.f90
!
! DESC: Reorthogonalize a low rank representation 
!
! AUTH: Yang Xinshuo
!
! DATE: 20141003
!
!-------------------------------------------------------------------------------
subroutine dortlr(nnu,nnv,u,v,lrankini,lrank,sval,eps,niter)
    implicit none
    character, parameter :: name*8 = 'dortlr: '
    integer*4,intent(in) :: nnu,nnv,lrankini,niter
    real*8,intent(in) :: eps
    integer*4,intent(out) :: lrank
    real*8,intent(inout) :: u(nnu,lrankini),v(nnv,lrankini),sval(lrankini)

    integer*4 :: iter,j,k,kmax,l,ll,itemp,mm,icount,info
    integer*4,allocatable :: ipivot(:),iempty(:)
    real*8 :: dalpha,fact1,fact2,eps2,ddot,fff,smax
    real*8,parameter :: done=1.0d0

    lrank = lrankini
    eps2 = eps*eps

    allocate(ipivot(lrankini),stat=info)
    if (info .ne. 0) then
       print '(A, "memory allocation for ipivot failed")', name
       stop
    endif

    allocate(iempty(lrankini),stat=info)
    if (info .ne. 0) then
       print '(A, "memory allocation for iempty failed")', name
       stop
    endif

    do iter = 1,niter
        ! Orthogonalize u-vectors

        do j = 1, lrank

            ! Pivoting: find index for the largest sval
            smax = -111
            do k = j, lrank
                if (smax.lt.sval(ipivot(k))) then
                    kmax = k
                    smax = sval(ipivot(k))
                endif
            enddo

            ! Swap into the leading position
            itemp = ipivot(j)
            ipivot(j) = ipivot(kmax)
            ipivot(kmax) = itemp

            mm = ipivot(j)

            ! Check if sval's are large enough:
            ! if they are smaller than a threshold, then reset the rank
            ! (run out of pivots)
            if (sval(mm).le.eps) then
                lrank = j-1
                goto 999
            endif


            do 555 l = j+1,lrank
                ll = ipivot(l)

                ! if sval is smaller than a threshold, set sval to zero
                ! to be ignored from now on

                if (sval(ll).le.eps) then
                    sval(ll) = 0
                    goto 555
                endif

                ! Modify u
                dalpha = ddot(nnu, u(1,mm),1,u(1,ll),1)/nnu
                call daxpy(nnu,-dalpha,u(1,mm),1,u(1,ll),1)
                fact1=sqrt(ddot(nnu, u(1,ll),1,u(1,ll),1)/nnu)
                if (fact1*sval(ll).gt.eps2) then
                    fff=done/fact1
                    call dscal(nnu,fff,u(1,ll),1)
                endif

                ! Modify v
                dalpha = dalpha*sval(ll)/sval(mm)
                call daxpy(nnv,dalpha,v(1,ll),1,v(1,mm),1)

                ! Adjust s-values
                sval(ll) = sval(ll)*fact1

        555 continue ! l loop

            ! Adjust the s-value of the pivot vector
            fact2 = sqrt(ddot(nnv, v(1,mm),1,v(1,mm),1)/nnv)
            if (fact2*sval(mm).gt.eps2) then
                fff=done/fact2
                call dscal(nnv,fff,v(1,mm),1)
            endif
            sval(mm) = sval(mm)*fact2
        enddo ! j loop

    999 continue
            
        do j = 1,lrank  

            ! Pivoting:find index for the largest sval
            smax = -111
            do k = j,lrank
                if ( smax.lt.sval(ipivot(k)) ) then
                    kmax = k
                    smax = sval(ipivot(k))
                endif
            enddo

            ! Swap into the leading position
            itemp        = ipivot(j)
            ipivot(j)    = ipivot(kmax)
            ipivot(kmax) = itemp

            mm=ipivot(j)

            ! Check if sval's are large enough
            ! if they are smaller than a threshold, then reset the rank 
            ! (run out of pivots)
            if (sval(mm).le.eps) then
                lrank = j-1
                goto 1000
            endif

            do 666 l=j+1,lrank
                ll=ipivot(l)

                ! If sval is smaller than a threshold, set sval to zero
                ! to be ignored from now on
                if (sval(ll).le.eps) then
                    sval(ll)=0
                    goto 666
                endif

                ! modify v
                dalpha = ddot(nnv, v(1,mm),1,v(1,ll),1)/nnv
                call daxpy(nnv,-dalpha,v(1,mm),1,v(1,ll),1)
                fact1=sqrt(ddot(nnv, v(1,ll),1,v(1,ll),1)/nnv)
                if (fact1*sval(ll).gt.eps2) then
                    fff=done/fact1
                    call dscal(nnv,fff,v(1,ll),1)
                endif

                ! modify u
                dalpha = dalpha*sval(ll)/sval(mm)
                call daxpy(nnu,dalpha,u(1,ll),1,u(1,mm),1)

                ! adjust s-values
                sval(ll) = sval(ll)*fact1
        666 continue

            fact2 = sqrt(ddot(nnu, u(1,mm),1,u(1,mm),1)/nnu)
            if (fact2*sval(mm).gt.eps2) then
                fff=done/fact2
                call dscal(nnu,fff,u(1,mm),1)
            endif
            sval(mm) = sval(mm)*fact2

        enddo ! j loop

   1000 continue
    
    enddo ! iter loop


    ! Fill in gaps, if any
    ! Find all pivots less than the new rank (lrank) with actual location beyond lrank
    ! These vectors and s values need to be swaped to locations within the range 1<= .. <=lrank  
    icount = 1         
    do l = lrank+1,lrankini
        if (ipivot(l).le.lrank) then
            iempty(icount)=ipivot(l)
            icount = icount + 1
        endif
    enddo

    ! Find all pivots larger than the new rank (lrank). These locations are overwritten
    ! by vectors and s values within the range 1<= .. <=lrank  found above

    icount = 1
    do l=1,lrank
        if (ipivot(l).gt.lrank) then
            call dcopy(nnu,u(1,ipivot(l)),1,u(1,iempty(icount)),1)
            call dcopy(nnv,v(1,ipivot(l)),1,v(1,iempty(icount)),1)
            sval(iempty(icount)) = sval(ipivot(l)) 
            ipivot(l) = iempty(icount)
            icount = icount + 1
        endif
    enddo

    deallocate(ipivot)
    deallocate(iempty)

    return

end subroutine dortlr
!-------------------------------------------------------------------------------
!
! END: dortlr.f90
!
!-------------------------------------------------------------------------------
