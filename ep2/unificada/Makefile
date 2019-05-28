CC=nvcc
CCFLAGS=
LDFLAGS= -lpng -fopenmp

mbrot: mbrot.cu
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f mbrot
