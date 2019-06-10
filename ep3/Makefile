CC=nvcc
CCFLAGS=
LDFLAGS=-Xcompiler -fopenmp -lpng -lmpi

dmbrot: dmbrot.cu
	$(CC) $(CCFLAGS) -o $@ $(LDFLAGS) $<

.PHONY: clean
clean:
	rm -f mbrot

# nvcc -o dmbrot -Xcompiler -fopenmp -lpng dmbrot.cu 
