NVCC=nvcc
CC=gcc
REAL ?= float
CCFLAGS_CUDA=-g -G -DREAL=$(REAL)
CCFLAGS= -DREAL=$(REAL)
LDFLAGS=
EXE=add


.PHONY: gpu
gpu:
	$(NVCC) $(CCFLAGS_CUDA) $(EXE).cu -o $(EXE) $(LDFLAGS)

.PHONY: cpu
cpu:
	$(CC) $(CCFLAGS) $(EXE).c -o $(EXE) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE)
