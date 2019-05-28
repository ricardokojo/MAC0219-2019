CC=nvcc
CCFLAGS=
LDFLAGS=-Xcompiler -fopenmp -lpng

mbrot: mbrot.cu
	$(CC) $(CCFLAGS) -o $@ $(LDFLAGS) $<

.PHONY: clean
clean:
	rm -f mbrot

# nvcc -o mbrot -Xcompiler -fopenmp -lpng mbrot.cu 
