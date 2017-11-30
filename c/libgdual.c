#include <stdio.h>
#include <math.h>
#include "libgdual.h"

#define NEG_INF (-1.0/0)
#define SIGN(x) ((x) >= 0 ? 1 : -1)

double ls_to_double( ls x) {
    return exp(x.l) * x.s;
}

ls double_to_ls(double x) {
    ls result;
    result.l = log(fabs(x));
    result.s = SIGN(x);
    return result;
}

ls ls_zero( ) {
    ls result;
    result.l = NEG_INF;
    result.s = 0;
    return result;
}

ls ls_add( ls x, ls y ) {

    // If x has a larger magnitude, then the sign of the output will equal the sign of x, and:
    //   -- If sign(y) = sign(x), then
    //
    //             |x + y| = |x| + |y|
    //        log(|x + y|) = log( |x| + |y| )
    //                     = log |x| + log( 1 + |y|/|x| )
    //                     = log |x| + log( 1 + exp( log|y| - log|x| ) )
    // 
    //   -- If sign(y) != sign(x), then
    //
    //             |x + y| = |x| - |y|
    //        log(|x + y|) = log( |x| - |y| )
    //                     = log |x| + log( 1 - |y|/|x| )
    //                     = log |x| + log( 1 - exp( log|y| - log|x| ) )
    //
    // The case when y has a larger magnitude is symmetric.
    
    ls result;
    double sign = (x.s == y.s) ? 1.0 : -1.0;
    if ( x.l > y.l ) {
        result.s = x.s;
        result.l = x.l + log1p( sign * exp(y.l - x.l) );
    }
    else {
        result.s = y.s;
        result.l = y.l + log1p( sign * exp(x.l - y.l) );
    }
    return result;
}

ls ls_mult( ls x, ls y ) {
    ls result;
    result.l = x.l + y.l;
    result.s = x.s * y.s;
    return result;
}

ls ls_exp( ls x ) {
    ls result;
    result.l = exp(x.l) * x.s;
    result.s = 1;
    return result;
}

void ls_print( ls *a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%hhd * exp(%.8e)", a[i].s, a[i].l);
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}

void ls_print_as_double( ls *a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%.8e", ls_to_double(a[i]));
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}


void gdual_exp( ls* u, ls *v, size_t n) { 
    
    ls u_tilde[n];
    
    for (int i = 0; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls(i) );
    }

    v[0] = ls_exp(u[0]);
    
    for (int k = 1; k < n; k++) { 
        v[k] = ls_zero();
        for (int j = 1; j <= k; j++) {
            v[k] = ls_add( v[k], ls_mult( v[k-j], u_tilde[j] ) );
        }
        v[k] = ls_mult( v[k], double_to_ls(1.0 / k) );
    }
} 

void gdual_exp_tmp( double* u, double *v, size_t n) {

    ls u_ls[n];
    ls v_ls[n];

    for (int i = 0; i < n; i++) {
        u_ls[i] = double_to_ls(u[i]);
    }

    gdual_exp(u_ls, v_ls, n);

    for (int i = 0; i < n; i++) {
        v[i] = ls_to_double(v_ls[i]);
    }
    
}
