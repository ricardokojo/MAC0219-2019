NVCC=nvcc
CCFLAGS= -Xcompiler -Wall -Xcompiler -O0 -Xptxas --opt-level=0 -rdc=true
OBJS=divergence.o gpu_work_v2.o gpu_work_v1.o

%.o: %.cu utils.h Makefile
	$(NVCC) $(CCFLAGS) -c $< -o $@

divergence: $(OBJS) utils.h Makefile
	$(NVCC) $(CCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f divergence *.o
