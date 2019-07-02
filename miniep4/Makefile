NVCC=nvcc
CCFLAGS= -Xcompiler -Wall -Xcompiler -O0 -Xptxas --opt-level=0 -rdc=true
OBJS=divergence.o gpu_work_v2.o gpu_work_v1.o bootstrap.o
HDRS=utils.h bootstrap.h

%.o: %.cu $(HDRS) Makefile
	$(NVCC) $(CCFLAGS) -c $< -o $@

divergence: $(OBJS) $(HDRS) Makefile
	$(NVCC) $(CCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f divergence *.o
