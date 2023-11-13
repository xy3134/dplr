CC = gcc
FORTRAN = gfortran

CFLAG = -Wall

CSOURCE := dplrini.c dplrd2p.c dplrresizelr.c dplrp2d.c dplrscal.c \
			dplrcpy.c dplraxpy.c dplrrnk.c dplrreduce.c dplrpmv.c \
			dplrpdmm.c dplrprk.c dplrppmm.c dplrinv.c dplrnz.c \
			dplrempty.c
FSOURCE := ortlrini.f90 dortlr.f90 

COBJECTS := $(CSOURCE:.c=.o)
FOBJECTS := $(FSOURCE:.f90=.o)

all: $(COBJECTS) $(FOBJECTS)

$(COBJECTS): $(CSOURCE)
	$(CC) $(CFLAG) -c $(CSOURCE) 

$(FOBJECTS): $(FSOURCE)
	$(FORTRAN) -c $(FSOURCE)

clean:
	rm -rf *.o
