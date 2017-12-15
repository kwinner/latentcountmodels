#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "gdual.h"
#include <strings.h>

/*****************************************/
/* Log-sign number system                */
/*****************************************/
#define NEG_INF -INFINITY
#define POS_INF INFINITY
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
    result.sign = 1;
    return result;
}

ls ls_neg( ls x ) {
    ls result;
    result.mag = x.mag;
    result.sign = -x.sign;
    return result;
}

ls ls_subtract( ls x, ls y ) {
    return ls_add(x, ls_neg(y));
}

ls ls_add( ls x, ls y ) {

    
    /***********************************************************************/
    /* If x has a larger magnitude, then the sign of the output will equal */
    /* the sign of x, and:                                                 */
    /*                                                                     */
    /*   -- If sign(y) = sign(x), then                                     */
    /*                                                                     */
    /*             |x + y| = |x| + |y|                                     */
    /*        log(|x + y|) = log( |x| + |y| )                              */
    /*                     = log |x| + log( 1 + |y|/|x| )                  */
    /*                     = log |x| + log( 1 + exp( log|y| - log|x| ) )   */
    /*                                                                     */
    /*   -- If sign(y) != sign(x), then                                    */
    /*                                                                     */
    /*             |x + y| = |x| - |y|                                     */
    /*        log(|x + y|) = log( |x| - |y| )                              */
    /*                     = log |x| + log( 1 - |y|/|x| )                  */
    /*                     = log |x| + log( 1 - exp( log|y| - log|x| ) )   */
    /*                                                                     */
    /* The case when y has a larger magnitude is symmetric.                */
    /***********************************************************************/
    
    ls result;
    if ( x.mag == NEG_INF ) {
        result.sign = y.sign;
        result.mag  = y.mag;
    } else {
        sign_t sign = (x.sign == y.sign) ? 1 : -1;
        if ( x.mag > y.mag ) {
            result.sign = x.sign;
            double arg = sign * exp(y.mag - x.mag);
            assert(arg >= -1.0);
            result.mag = x.mag + log1p( arg );
        }
        else {
            result.sign = y.sign;
            double arg = sign * exp(x.mag - y.mag);
            assert(arg >= -1.0);
            result.mag = y.mag + log1p( arg );
        }
    }
    return result;
}

ls ls_mult( ls x, ls y ) {
    ls result;
    result.mag  = x.mag + y.mag;
    result.sign = x.sign * y.sign;
    return result;
}

ls ls_inv( ls x ) {
    ls result;
    result.mag  = -x.mag;        // log (1/z) = -log(z)
    result.sign = x.sign;
    return result;
}

ls ls_div( ls x, ls y ) {
    return ls_mult(x, ls_inv(y));
}

ls ls_exp( ls x ) {
    ls result;
    result.mag = exp(x.mag) * x.sign;
    result.sign = 1;
    return result;
}

ls ls_log( ls x ) {
    // If
    //        z = exp(x.mag) * x.sign,
    // then
    //    log(z) = x.mag + log(x.sign)

    // This should correclty handle the case where x is zero or negative
    return double_to_ls( x.mag + log(x.sign) );
}

ls ls_pow( ls x, double r ) {
    
    ls result;
    result.sign = SIGN(pow(x.sign, r)); 
    result.mag  = r * x.mag;    // log(x^r) = r log(x)
    return result;
}


/**************************************************************************/
/* Gdual implementation                                                   */
/**************************************************************************/

void gdual_print( ls* a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%d * exp(%.8e)", a[i].sign, a[i].mag);
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}

void gdual_print_as_double( ls* a, size_t n ) {
    printf("%s", "[");
    for (int i = 0; i < n; i++ ) {
        printf("%.8e", ls_to_double(a[i]));
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}


// Compute v = u + c*w
void gdual_u_plus_cw( ls* v, ls* u, ls* w, double c, size_t n) {

    ls c_ls = double_to_ls(c);
    
    for (int k = 0; k < n; k++) {
        v[k] = ls_add(u[k], ls_mult(c_ls, w[k]));
    }
    
}

// Compute v = u + w
void gdual_add( ls* v, ls* u, ls* w, size_t n) {
    for (int k = 0; k < n; k++) {
        v[k] = ls_add(u[k], w[k]);
    }
}

// Compute v = u * c
void gdual_scalar_mul( ls* v, ls* u, double c, size_t n) {
    ls c_ls = double_to_ls(c);
    for (int k = 0; k < n; k++) {
        v[k] = ls_mult(c_ls, u[k]);
    }
}

void gdual_exp( ls* v,      /* The result */
                ls* u,
                size_t n) { 

    /* Reference: Chapter 13 p. 308 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */
    
    ls u_tilde[n];
    
    for (int i = 1; i < n; i++) {
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


void gdual_log( ls* v,      /* The result */
                ls* u,
                size_t n)
{

    /* Reference: Chapter 13 p. 308 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    ls u_tilde[n];
    ls v_tilde[n];

    for (int i = 1; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls(i) );
    }

    v[0] = ls_log(u[0]);
    
    for (int k = 1; k < n; k++)
    {
        v_tilde[k] = u_tilde[k];
        for (int j = 1; j < k; j++) {
            v_tilde[k] = ls_subtract( v_tilde[k], ls_mult(u[k-j], v_tilde[j]) );
        }
        v_tilde[k] = ls_div( v_tilde[k], u[0] );
    }

    for (int i = 1; i < n; i++) {
        v[i] = ls_mult( v_tilde[i], double_to_ls(1.0/i) );
    }
}

// Compute v = u^r
void gdual_pow( ls* v, ls* u, double r, size_t n )
{
    printf("n = %d\n", (int) n);
    
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    ls u_tilde[n];
    ls v_tilde[n];

    for (int i = 1; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls((double) i) );
    }

    // The scalar function
    v[0] = ls_pow(u[0], r);

    // The recurrence
    for (int k = 1; k < n; k++) {

        v_tilde[k] = ls_zero();

        for (int j = 1; j <=k; j++) { // Upper bound <= k
            v_tilde[k] = ls_add( v_tilde[k],
                                 ls_mult(v[k-j], u_tilde[j]) );
        }

        v_tilde[k] = ls_mult( v_tilde[k], double_to_ls(r) );

        for (int j = 1; j < k; j++) { // Upper bound < k
            v_tilde[k] = ls_subtract( v_tilde[k],
                                      ls_mult( u[k-j], v_tilde[j]) );
        }

        v_tilde[k] = ls_div( v_tilde[k], u[0] );

        v[k] = ls_mult( v_tilde[k], double_to_ls(1.0/k) );
    }

}


/***************************/
/* Binary gdual operations */
/***************************/

// compute v = u * w
void gdual_mul ( ls* v,
                 ls* u,
                 ls* w,
                 size_t n)
{
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    for (int k = 0; k < n; k++) {

        v[k] = ls_zero();

        // Note that j <= k is the correct termination condition, and is
        // different from gdual_div below
        for (int j = 0; j <= k; j++) { 
            v[k] = ls_add( v[k], ls_mult(u[j], w[k-j]) );
        }
    }
}

// compute v = u / w
void gdual_div ( ls* v,
                 ls* u,
                 ls* w,
                 size_t n)
{
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    for (int k = 0; k < n; k++) {

        v[k] = u[k];

        // Note that j < k is the correct termination condition, and is
        // different from gdual_mul below
        for (int j = 0; j < k; j++) {
            v[k] = ls_subtract( v[k], ls_mult(v[j], w[k-j]) );
        }

        v[k] = ls_div( v[k], w[0]);
    }
}


// Compute v = 1/w
void gdual_inv( ls* v, /* the result */
                ls* w,
                size_t n)
{
    
    // Create a gdual equal to the scalar one
    ls one[n];
    one[0] = double_to_ls( 1.0 );
    for (int i = 1; i < n; i++) {
        one[i] = ls_zero();
    }
    gdual_div(v, one, w, n);
}

void gdual_compose( ls* res,
                    ls* u,
                    ls* v,
                    size_t n)
{
    assert(0);    
}

void gdual_compose_affine( ls* res,
                           ls* u,
                           ls* v,
                           size_t n)
{
    assert(0);    
}



/*******************************************************************/
/* Convert between log-sign arrays and separate arrays             */
/* for log-magnitude and sign                                      */
/*******************************************************************/
void magsign2ls( ls* x, mag_t* mag, sign_t* sign, size_t n)
{
    for (int i = 0; i < n; i++) {
        x[i].mag = mag[i];
        x[i].sign = sign[i];
    }
}

void ls2magsign( mag_t* mag, sign_t* sign, ls *x, size_t n)
{
    for (int i = 0; i < n; i++) {
        mag[i] = x[i].mag;
        sign[i] = x[i].sign;
    }
}

/*************************************************************/
/* Wrappers for gdual operations that accept separate arrays */
/* for log-magnitude and sign                                */
/*************************************************************/


void _gdual_exp( mag_t*  u_mag,
                 sign_t* u_sign,
                 mag_t*  v_mag,
                 sign_t* v_sign,
                 int *nin)
{

    int n = *nin;
    
    ls u[n];
    ls v[n];

    magsign2ls(u, u_mag, u_sign, n);

    gdual_exp(u, v, n);
    
    ls2magsign(v_mag, v_sign, v, n);
}
