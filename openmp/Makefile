# Makefile for COMP4300/8300 Assignment 2 (OpenMP)
# Peter Strazdins, SOCO ANU, Apr 21     
.SUFFIXES:
.PRECIOUS: %.o

HDRS=serAdvect.h parAdvect.h
OBJS=serAdvect.o parAdvect.o
PROG=testAdvect
FFT_CONV_KERNEL=0
CCFLAGS=-O3 -fopenmp -Wattributes -Wall -DFFT_CONV_KERNEL=$(FFT_CONV_KERNEL)
LINKERFLAGS=

ifeq ($(FFT_CONV_KERNEL), 1)
	LINKERFLAGS=$(shell pkg-config --cflags --libs fftw3)
endif

all: $(PROG) 

%: %.o $(OBJS)
	gcc-13 -o $* $*.o $(OBJS) $(LINKERFLAGS) -lgomp -lm
%.o: %.c $(HDRS)
	gcc-13 -Wall $(CCFLAGS) -c $*.c
clean:
	rm -f *.o $(PROG)
