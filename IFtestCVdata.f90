      double precision,dimension(:,:),ALLOCATABLE :: std0, del, std1, stdt0
      double precision,dimension(:),ALLOCATABLE :: tar, ddel, deuc, d2euc, euc, addel
      integer ,dimension(:),ALLOCATABLE :: turn, iddel
      integer di,ti,tsam
      character filename*128

      open(10, file='Train_ICA100.dat', status='old')
      open(11, file='std1.dat', status='old')
      open(12, file='Test_ICA100.dat', status='old')
      
      !nsam = XNSAM
      !ndis = XNDIS
      nsam = 23814
      ndis = 203
      ntar = 206
      tsam = 3982
      ALLOCATE( std0(1:nsam,1:ndis) )
      ALLOCATE( std1(1:nsam,1:ntar) )
      ALLOCATE( stdt0(1:tsam,1:ndis) )
      
      ALLOCATE( tar(1:nsam) )
      ALLOCATE( euc(1:ndis) )
      ALLOCATE( deuc(1:nsam) )
      ALLOCATE( d2euc(1:nsam) )
      deuc(:) = 1000d0
      d2euc(:) = 1000d0

      do i=1,nsam
      read(10,*) (std0(i,j), j=1,ndis)
      read(11,*) (std1(i,j), j=1,ntar)
      enddo
      do i=1,tsam
      read(12,*) (stdt0(i,j), j=1,ndis)
      enddo

      !do ti=1,ntar
      filename="iFCV-M-pp1.dat"
      open(21, file=filename, status='replace')
      filename="iFCV-M-pp2.dat"
      open(22, file=filename, status='replace')
      filename="iFCV-M-pp3.dat"
      open(23, file=filename, status='replace')
      filename="iFCV-M-pp4.dat"
      open(24, file=filename, status='replace')
      filename="iFCV-M-pp5.dat"
      open(25, file=filename, status='replace')
      filename="iFCV-M-pp6.dat"
      open(26, file=filename, status='replace')
      do di=1,tsam
        write(*,"(i5)") di
        do i=1,nsam
          seuc = 0d0
          do j=1,ndis
            tmp = std0(i,j) - stdt0(di,j)
            seuc = seuc + tmp * tmp
          enddo
          deuc(i) = SQRT(seuc)
        enddo
        write(21,"(i5)") MINLOC(deuc)
        deuc(MINLOC(deuc)) = 1000d0
        write(22,"(i5)") MINLOC(deuc)
        deuc(MINLOC(deuc)) = 1000d0
        write(23,"(i5)") MINLOC(deuc)
        deuc(MINLOC(deuc)) = 1000d0
        write(24,"(i5)") MINLOC(deuc)
        deuc(MINLOC(deuc)) = 1000d0
        write(25,"(i5)") MINLOC(deuc)
        deuc(MINLOC(deuc)) = 1000d0
        write(26,"(i5)") MINLOC(deuc)
        deuc(:) = 1000d0
      enddo

      !open(30, file='fedis.dat', status='replace')
      !read(40,*) (addel(i), i=1,nsam)
      !!read(41,*) (iddel(i), i=1,nsam)
      !!write(21,"(f25.16)") (addel(i), i=1,nsam)
      !call SORT(nsam, addel, turn)
      !!write(22,"(f25.16)") (addel(i), i=1,nsam)
      !!write(24,"(i5)") (iddel(i), i=1,nsam)
      !write(30,"(i5)") (turn(i)-1, i=1,40)

      contains
      subroutine SORT(N,data,turn)
        integer,intent(in)::N
        integer,intent(out)::turn(1:N)
        double precision,intent(inout)::data(1:N)
        integer::i,j,ti
        double precision::tmp
      
        do i=1,N
           turn(i)=i
        enddo
      
        do i=1,N-1
           do j=i+1,N
              if(data(i) .lt. data(j))then
              !if(data(i) .gt. data(j))then
                 tmp=data(i)
                 data(i)=data(j)
                 data(j)=tmp
               
                 ti=turn(i)
                 turn(i)=turn(j)
                 turn(j)=ti
              end if
           end do
        end do
      
        return
      end subroutine SORT

      !FUNCTION gaussian( x, z ) RESULT ( res )
      !  
      !  IMPLICIT NONE
      !  DOUBLE PRECISION, INTENT ( IN ) :: x, z
      !  DOUBLE PRECISION :: res 
      !  DOUBLE PRECISION, PARAMETER :: pi=atan(1d0)*4d0

      !  res=1d0/(SQRT(2d0*pi)*z)*exp(-(x)**2d0/(2d0*z**2d0))
      !  
      !END FUNCTION gaussian

      end
