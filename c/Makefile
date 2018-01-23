all: test_gdual libgdual.so

test_gdual: test_gdual.c gdual.c
	gcc -g test_gdual.c gdual.c -o test_gdual -O3

libgdual.so: gdual.o
	gcc -shared -o libgdual.so gdual.o -O3

gdual.o: gdual.c
	gcc -c -Wall -Werror -fpic gdual.c -O3
clean:
	-rm test_gdual gdual.o libgdual.so
