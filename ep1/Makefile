CC=g++
CCFLAGS=
LDFLAGS=-lpthread

pgrep: pgrep.cpp
	$(CC) $(CCFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f pgrep
