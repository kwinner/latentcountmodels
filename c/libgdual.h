#include <stddef.h>
#include <stdint.h>

typedef struct ls {
    double l;
    int8_t s;
} ls;

typedef ls* gdual_t;

double ls_to_double( ls x );
ls double_to_ls(double x);
ls ls_zero();
ls ls_add( ls x, ls y );
ls ls_mult( ls x, ls y );
ls ls_exp( ls x );

void gdual_print( gdual_t a, size_t n );
void gdual_print_as_double( gdual_t a, size_t n );

void gdual_exp( gdual_t u, gdual_t v, size_t n);
