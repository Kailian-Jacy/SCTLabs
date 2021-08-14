CC=g++
CFLAGS=-mcmodel=medium
# compiler may crash when static array too large,
# add `-mcmodel=medium` in this case.

all:
	$(CC) -o gemm hw.cpp $(CFLAGS) -O3 -fopenmp -g -march=skylake-avx512

.PHONY: run
run: all
	./gemm

.PHONY: clean
clean:
	rm -rf *.o gemm