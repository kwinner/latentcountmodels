CFLAGS=-O3

ifeq ($(shell uname -n),”shannon.cs.umass.edu”)
CFLAGS += -std=c99 -lm
endif 

all: test_gdual libgdual.so

test_gdual: test_gdual.c gdual.c
	gcc -g test_gdual.c gdual.c -o test_gdual $(CFLAGS)

libgdual.so: gdual.o
	gcc -shared -o libgdual.so gdual.o $(CFLAGS) -lprofiler

gdual.o: gdual.c
	gcc -c -Wall -Werror -fpic gdual.c $(CFLAGS)
clean:
	-rm test_gdual gdual.o libgdual.so
