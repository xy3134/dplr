CC = gcc
FORTRAN = gfortran

CCFLAG = -Wall

# Path for BLAS and LAPACK libraries
LIB_PATH = ~/lib

LIBS = $(LIB_PATH)/liblapack.a $(LIB_PATH)/libblas.a -lgfortran -lm

SRC_PATH = ../src

DPLR_OBJS := $(SRC_PATH)/*.o

all: main

main: testdplr.c $(DPLR_OBJS)
	$(CC) $(CCFLAG) -o testdplr testdplr.c $(DPLR_OBJS) $(LIBS) 

$(DPLR_OBJS):
	cd $(SRC_PATH); make

clean:
	cd $(SRC_PATH); rm -rf *.o; rm -rf *.o
