EXT=$(shell python3-config --extension-suffix)
CFLAGS=-std=c99 -lm

default: opt

debug: CFLAGS += -g -O0
debug: cygdual$(EXT)

opt: CFLAGS += -O3
opt: cygdual$(EXT)

cygdual$(EXT): ../c/gdual.h ../c/gdual.c cygdual.pyx setup.py
	echo $(EXT)
	CFLAGS="$(CFLAGS)" python setup.py build_ext --inplace

clean:
	rm cygdual.c
	rm cygdual$(EXT)
