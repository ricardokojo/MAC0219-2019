CC=g++
CCFLAGS=
LDFLAGS= -lpng -fopenmp 

mbrot: mbrot.cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f mbrot
