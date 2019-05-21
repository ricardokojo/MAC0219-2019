CC=g++
CCFLAGS=
LDFLAGS= -lpng -fopenmp 

pragma_mbrot: pragma_mbrot.cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f pragma_mbrot
