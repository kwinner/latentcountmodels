#include <stdio.h>
#include <math.h>

#define NEG_INF (-1.0/0)
#define SIGN(x) ((x) >= 0 ? 1 : -1)

typedef struct ls {
    double l;
    int s;
} ls;


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

void ls_print_array( ls *a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%d * exp(%.8e)", a[i].s, a[i].l);
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}

void ls_print_array_as_double( ls *a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%.8e", ls_to_double(a[i]));
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}


void gdual_exp( ls* u, ls *v, size_t n);

int main() {
    
    double t[] = {-1.38629436e+00,
                  -2.50000000e-01,
                  3.12500000e-02,
                  -5.20833333e-03,
                  9.76562500e-04,
                  -1.95312500e-04,
                  4.06901042e-05};
    
    int n = 7;
    
    ls u[n];
    ls v[n];
    
    for (int i = 0; i < n; i++) {
        u[i] = double_to_ls(t[i]);
    }

    gdual_exp( u, v, n );

    ls_print_array(u, n);
    ls_print_array_as_double(u, n);
    
    ls_print_array(u, n);
    ls_print_array_as_double(v, n); // Should print [2.50000000e-01, -6.25000001e-02, 1.56250000e-02, -3.90625000e-03, 9.76562501e-04, -2.44140625e-04, 6.10351563e-05]
    
    return(0);
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
