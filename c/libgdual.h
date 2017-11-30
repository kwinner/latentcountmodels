#include <stddef.h>
#include <stdint.h>

typedef struct ls {
    double l;
    int8_t s;
} ls;

double ls_to_double( ls x );
ls double_to_ls(double x);
ls ls_zero();
ls ls_add( ls x, ls y );
ls ls_mult( ls x, ls y );
ls ls_exp( ls x );
void ls_print( ls *a, size_t n );
void ls_print_as_double( ls *a, size_t n );

void gdual_exp( ls* u, ls *v, size_t n);
void gdual_exp_tmp( double* u, double* v, size_t n);
