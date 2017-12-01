#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "gdual.h"

#define NEG_INF (-1.0/0)
#define SIGN(x) ((x) >= 0 ? 1 : -1)

#define LS(s, m) ((struct ls) {s, m})

double ls_to_double( ls x) {
    return exp(x.mag) * x.sign;
}

ls double_to_ls(double x) {
    ls result;
    result.mag = log(fabs(x));
    result.sign = SIGN(x);
    return result;
}

ls ls_zero( ) {
    ls result;
    result.mag = NEG_INF;
    result.sign = 0;
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
    double sign = (x.sign == y.sign) ? 1.0 : -1.0;
    if ( x.mag > y.mag ) {
        result.sign = x.sign;
        result.mag = x.mag + log1p( sign * exp(y.mag - x.mag) );
    }
    else {
        result.sign = y.sign;
        result.mag = y.mag + log1p( sign * exp(x.mag - y.mag) );
    }
    return result;
}

ls ls_mult( ls x, ls y ) {
    ls result;
    result.mag = x.mag + y.mag;
    result.sign = x.sign * y.sign;
    return result;
}

ls ls_exp( ls x ) {
    ls result;
    result.mag = exp(x.mag) * x.sign;
    result.sign = 1;
    return result;
}

void gdual_print( gdual_t a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%d * exp(%.8e)", (int) a[i].sign, a[i].mag);
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}

void gdual_print_as_double( gdual_t a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%.8e", ls_to_double(a[i]));
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}


void gdual_exp( gdual_t u, gdual_t v, size_t n) { 
    
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

void magsign2ls( mag_t* mag, sign_t* sign, ls* x, size_t n)
{
    for (int i = 0; i < n; i++) {
        x[i].mag = mag[i];
        x[i].sign = sign[i];
    }
}

void ls2magsign(ls *x, mag_t* mag, sign_t* sign, size_t n)
{
    for (int i = 0; i < n; i++) {
        mag[i] = x[i].mag;
        sign[i] = x[i].sign;
    }
}

void _gdual_exp( mag_t*  u_mag,
                 sign_t* u_sign,
                 mag_t*  v_mag,
                 sign_t* v_sign,
                 int *nin)
{

    int n = *nin;
    
    ls u[n];
    ls v[n];

    magsign2ls(u_mag, u_sign, u, n);

    gdual_exp(u, v, n);
    
    ls2magsign(v, v_mag, v_sign, n);
}
