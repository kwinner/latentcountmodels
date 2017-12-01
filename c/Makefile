all: test_gdual libgdual.so

test_gdual: test_gdual.c gdual.c
	gcc test_gdual.c gdual.c -o test_gdual

libgdual.so: gdual.o
	gcc -shared -o libgdual.so gdual.o

gdual.o: gdual.c
	gcc -c -Wall -Werror -fpic gdual.c
clean:
	-rm test_gdual gdual.o libgdual.so
