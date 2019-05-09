CC=g++
CCFLAGS=
LDFLAGS=-lpthread

mbrot: mbrot.cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f mbrot
