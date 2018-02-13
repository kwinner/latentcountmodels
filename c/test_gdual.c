#include "gdual.h"
#include <stdio.h>

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

    printf("**** EXP ****\n");
    gdual_exp( v, u, n );
    gdual_print(v, n);
    gdual_print_as_double(v, n); // Should print [2.50000000e-01, -6.25000001e-02, 1.56250000e-02, -3.90625000e-03, 9.76562501e-04, -2.44140625e-04, 6.10351563e-05]
    printf("\n");
    
    printf("**** INV ****\n");
    gdual_inv(v, u, n);
    gdual_print(v, n);
    gdual_print_as_double(v, n);
    printf("\n");
    
    printf("**** MUL ****\n");
    gdual_mul_same(v, u, u, n);
    gdual_print(v, n);
    gdual_print_as_double(v, n);
    printf("\n");

    printf("**** POW ****\n");
    gdual_pow(v, u, 2.0, n);
    gdual_print(v, n);
    gdual_print_as_double(v, n);
    printf("\n");
    
    u[0] = double_to_ls(2.5); // log requires u[0] to be positive

    printf("**** LOG ****\n");
    gdual_log( v, u, n );
    gdual_print(v, n);
    gdual_print_as_double(v, n);
    printf("\n");

    return(0);
}
