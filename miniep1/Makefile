CC=gcc
CCFLAGS=-Wall -O0
LDFLAGS=-lm -lpthread
IF ?= 0
IF_STR = $(shell for i in `seq 1 $(IF)`; do echo 'IF_GT_MAX '; done)

.PHONY: all
all:
	sed "s!//@#@IF!$(IF_STR)!" contention.c > .contention.c
	$(CC) $(CCFLAGS) .contention.c -o contention $(LDFLAGS)
	rm -f .contention.c

.PHONY: clean
clean:
	rm -f contention
