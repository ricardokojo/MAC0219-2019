CC=nvcc
MPICC=mpic++
CCFLAGS=-g -c
LDFLAGS=-Xcompiler -fopenmp -lpng -lmpi
BRUCUTUFLAGS=-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent \
-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include \
-I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr//lib \
-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi

dmbrot: cpu.o gpu.o img_util.o main.o
	$(CC) -o $@ $(LDFLAGS) $<

cpu.o: cpu.cpp
	$(CC) $(CCFLAGS) $<

gpu.o: gpu.cu
	$(CC) $(CCFLAGS) $<

img_util.o: img_util.cpp
	$(CC) $(CCFLAGS) $<

main.o: main.cpp
	$(MPICC) $(CCFLAGS) $<

.PHONY: clean
clean:
	rm -f dmbrot cpu.o gpu.o img_util.o main.o

# nvcc -o dmbrot -Xcompiler -fopenmp -lpng dmbrot.cu 
