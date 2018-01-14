#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "gdual.h"
#include <strings.h>
#include <stdlib.h>

#ifdef WITH_FFT
#include <fftw3.h>
#endif

#define ERR(s) { fprintf(stderr, "%s\n", s); return; }
//#define ERR(s) { PyErr_SetString(PyExc_RuntimeError, s); return; }

/*****************************************/
/* Log-sign number system                */
/*****************************************/
#define NEG_INF -INFINITY
#define POS_INF INFINITY
#define SIGN(x) ((x) >= 0 ? 1 : -1)
#define LS(s, m) ((struct ls) {s, m})
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

double ls_to_double(ls x) {
    return exp(x.mag) * x.sign;
}

long double ls_to_long_double(ls x) {
    return expl(x.mag) * x.sign;
}

ls double_to_ls(double x) {
    ls result;
    result.mag = log(fabs(x));
    result.sign = SIGN(x);
    return result;
}

ls long_double_to_ls(long double x) {
    ls result;
    result.mag = (double) logl(fabsl(x));
    result.sign = SIGN(x);
    return result;
}

ls ls_zero( ) {
    ls result;
    result.mag = NEG_INF;
    result.sign = 0;
    return result;
}

int ls_is_zero(ls x) {
    return x.mag==NEG_INF || x.sign == 0;
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

    if ( ls_is_zero(x) )
        return y;

    if ( ls_is_zero(y) )
        return x;

    sign_t sign = (x.sign == y.sign) ? 1 : -1;
    if ( x.mag > y.mag ) {
        return (struct ls) {
            (double) (x.mag + log1pl( sign * expl(y.mag - x.mag))),
                x.sign
                };
    }
    else {
        return (struct ls) {
            (double) (y.mag + log1pl( sign * expl(x.mag - y.mag))),
                y.sign
                };
    }
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


void ls_mat_mul(ls *Z, ls *X, ls *Y, size_t m, size_t n, size_t p) {
    // Compute Z = X * Y
    //    X: m x n
    //    Y: n x p
    //    Z: m x p
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            Z[i*p + j] = ls_zero();
            for (size_t k = 0; k < n; k++) {
                // Z[i,j] += X[i,k]*Y[k,j]
                Z[i*p + j] = ls_add( Z[i*p + j],
                                     ls_mult( X[i*n + k],
                                              Y[k*p + j] ));
            }
            
        }
    }
}

/**************************************************************************/
/* Gdual implementation                                                   */
/**************************************************************************/

void gdual_print( ls* a, size_t n ) {
    printf("%s", "[");
    for (size_t i = 0; i < n; i++ ) {
        printf("%d * exp(%.8e)", a[i].sign, (double) a[i].mag);
        if (i < n-1) {
            printf("%s", ", ");
        }
    }
    printf("%s", "]\n");
}

void gdual_print_as_double( ls* a, size_t n ) {
    printf("%s", "[");
    for (size_t i = 0; i < n; i++ ) {
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
    
    for (size_t k = 0; k < n; k++) {
        v[k] = ls_add(u[k], ls_mult(c_ls, w[k]));
    }
    
}

// Compute v = u + w (same sizes)
void gdual_add( ls* v, ls* u, ls* w, size_t n) {
    for (size_t k = 0; k < n; k++) {
        v[k] = ls_add(u[k], w[k]);
    }
}

// Compute v = u + w (different sizes)
void gdual_add_different( ls* v, size_t n, ls* u, size_t u_len, ls* w, size_t w_len) {
    
    size_t k = 0;

    // Both u[k] and w[k] exist
    while(k < n && k < u_len && k < w_len) {
        v[k] = ls_add(u[k], w[k]);
        k++;
    }

    // Only u[k] exists. Assume w[k] = 0
    while(k < n && k < u_len) {
        v[k] = u[k];
        k++;
    }

    // Only w[k] exists. Assume u[k] = 0
    while(k < n && k < w_len) {
        v[k] = w[k];
        k++;
    }

    // Neither exist
    while(k < n) {
        v[k] = ls_zero();
        k++;
    }
}


// Compute v = u * c
void gdual_scalar_mul( ls* v, ls* u, double c, size_t n) {
    ls c_ls = double_to_ls(c);
    for (size_t k = 0; k < n; k++) {
        v[k] = ls_mult(c_ls, u[k]);
    }
}

// Compute v = -u
void gdual_neg( ls* v, ls* u, size_t n) {
    gdual_scalar_mul(v, u, -1.0, n);
}


void gdual_exp( ls* v,      /* The result */
                ls* u,
                size_t n) { 

    /* Reference: Chapter 13 p. 308 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */
    
    ls u_tilde[n];
    
    for (size_t i = 0; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls(i) );
    }

    v[0] = ls_exp(u[0]);
    
    for (size_t k = 1; k < n; k++) { 
        v[k] = ls_zero();
        for (size_t j = 1; j <= k; j++) {
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

    for (size_t i = 0; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls(i) );
    }

    v[0] = ls_log(u[0]);
    
    for (size_t k = 1; k < n; k++)
    {
        v_tilde[k] = u_tilde[k];
        for (size_t j = 1; j < k; j++) {
            v_tilde[k] = ls_subtract( v_tilde[k], ls_mult(u[k-j], v_tilde[j]) );
        }
        v_tilde[k] = ls_div( v_tilde[k], u[0] );
    }

    for (size_t i = 1; i < n; i++) {
        v[i] = ls_mult( v_tilde[i], double_to_ls(1.0/i) );
    }
}

int is_int(double z) {
    return (fabs(round(z) - z) <= 1e-8);
}

// Compute v = u^r
void gdual_pow( ls* v, ls* u, double r, size_t n )
{

    // Special case for r = 0
    if (r == 0.0) {
        ls ZERO = ls_zero();
        for (size_t i = 0; i < n; i++) {
            v[i] = ZERO;
        }
        v[0] = double_to_ls(1.0);
        return;
    }

    // Special case for r = 1
    if (r == 1.0) {
        for (size_t i = 0; i < n; i++) {
            v[i] = u[i];
        }
        return;
    }

    if (is_int(r)) {
        gdual_pow_int( v, u, (int) round(r), n );
    }
    else
    {
        gdual_pow_fractional(v, u, r, n);
    }
}

void gdual_pow_int( ls* v, ls* u, int r, size_t n ) {

    // This method first normalizes the gdual to have a positive leading coefficient
    // then calls the fractional version

    // Find index k of first nonzero coefficient
    size_t k = 0;
    while (ls_is_zero(u[k]) && k < n) {
        k++;
    }

    // Truncate to trailing n-k coefficients and multiply by sign
    // so the leading coefficient is positive. This effectively divides by (sign * x^k)
    ls u_norm[n-k];
    int sign = u[k].sign;
    gdual_scalar_mul( u_norm, u+k, sign, n-k);
    
    // Now raise u_norm to the rth power
    ls v_norm[n-k];
    gdual_pow_fractional(v_norm, u_norm, (double) r, n-k);

    // Multiply by sign^r, which is equal to sign if r is odd and 1 if r is even
    sign = r % 2 == 0 ? 1 : sign;
    gdual_scalar_mul(v_norm, v_norm, sign, n-k);
    
    // Now multiply by x^kr by first placing k*r leading zeros, and then placing
    // the coefficients of v_norm
    ls ZERO = ls_zero();
    for (size_t i = 0; i < (size_t) k*r; i++) {
        v[i] = ZERO;
    }
    for (size_t i = (size_t) k*r; i < n; i++) {
        v[i] = v_norm[i-k*r];
    }
    
    return;
}

void gdual_pow_fractional( ls* v, ls* u, double r, size_t n ) {

    if ( ls_is_zero(u[0]) ) {
        ERR("Leading coefficient is zero. Cannot raise to a fractional power using this method");
    }
    if ( u[0].sign < 0 ) {
        ERR("Cannot raise negative number to fractional power");
    }
    
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    ls u_tilde[n];
    ls v_tilde[n];

    for (size_t i = 1; i < n; i++) {
        u_tilde[i] = ls_mult( u[i], double_to_ls((double) i) );
    }

    // The scalar function
    v[0] = ls_pow(u[0], r);

    // The recurrence
    for (size_t k = 1; k < n; k++) {

        v_tilde[k] = ls_zero();

        for (size_t j = 1; j <=k; j++) { // Upper bound <= k
            v_tilde[k] = ls_add( v_tilde[k],
                                 ls_mult(v[k-j], u_tilde[j]) );
        }

        v_tilde[k] = ls_mult( v_tilde[k], double_to_ls(r) );

        for (size_t j = 1; j < k; j++) { // Upper bound < k
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
void gdual_mul ( ls* v,         /* output */
                 size_t n,      /* truncation order */
                 ls* u,         /* first input */
                 size_t u_len,  /* length of u */
                 ls* w,         /* second input */
                 size_t w_len)  /* length of w */
{
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */
    
    for (size_t k = 0; k < n; k++) {

        v[k] = ls_zero();

        // If u and w are long enough, we will iterate over 0 <= j <= k
        size_t lower = 0, upper = k;
        if (u_len - 1 < k)
        {
            upper = u_len - 1;     // u too short --> set upper = last entry of u
        }
        if (w_len - 1 < k)
        {
            lower = k - w_len + 1; // w too short --> set lower so k - lower = last entry of w
        }
        for (size_t j = lower; j <= upper; j++) { 
            v[k] = ls_add( v[k], ls_mult(u[j], w[k-j]) );
        }
    }
}


#if 1
// compute v = u * w
void gdual_mul_same ( ls* v,         /* output */
                      ls* u,
                      ls* w,
                      size_t n)
{
    gdual_mul(v, n, u, n, w, n);
}
#else
// compute v = u * w 
void gdual_mul_same ( ls* v, 
                      ls* u, 
                      ls* w, 
                      size_t n) 
{ 
    /* Reference: Chapter 13 p. 305 from  
       Griewank, A. and Walther, A. Evaluating derivatives: principles and  
       techniques of algorithmic differentiation. SIAM, 2008. */ 

    for (size_t k = 0; k < n; k++) { 

        v[k] = ls_zero(); 

        // Note that j <= k is the correct termination condition, and is 
        // different from gdual_div below 
        for (size_t j = 0; j <= k; j++) {  
            v[k] = ls_add( v[k], ls_mult(u[j], w[k-j]) ); 
        } 
    } 
} 
#endif

#ifdef WITH_FFT

#if FFTW_USE_LONGDOUBLE
/* Types/functions for long double FFT */
typedef long double FFTW_REAL;
typedef fftwl_complex FFTW_COMPLEX;
typedef fftwl_plan FFTW_PLAN;
#define LS2REAL ls_to_long_double
#define REAL2LS long_double_to_ls
#define FFTW_MALLOC fftwl_malloc
#define FFTW_FREE fftwl_free
#define FFTW_DESTROY_PLAN fftwl_destroy_plan
#define FFTW_PLAN_DFT_R2C_1D fftwl_plan_dft_r2c_1d
#define FFTW_PLAN_DFT_C2R_1D fftwl_plan_dft_c2r_1d
#define FFTW_EXECUTE fftwl_execute
#define LOG_MAXVAL 11356.52340629414395
#else 
/* Types/functions for double precision FFT */
typedef double FFTW_REAL;
typedef fftw_complex FFTW_COMPLEX;
typedef fftw_plan FFTW_PLAN;
#define LS2REAL ls_to_double
#define REAL2LS double_to_ls
#define FFTW_MALLOC fftw_malloc
#define FFTW_FREE fftw_free
#define FFTW_DESTROY_PLAN fftw_destroy_plan
#define FFTW_PLAN_DFT_R2C_1D fftw_plan_dft_r2c_1d
#define FFTW_PLAN_DFT_C2R_1D fftw_plan_dft_c2r_1d
#define FFTW_EXECUTE fftw_execute
#define LOG_MAXVAL 709.78271289338397
#endif // FFTW_USE_LONGDOUBLE

// compute v = u * w
void gdual_mul_fft ( ls* v,
                     ls* u,
                     ls* w,
                     size_t n)
{
    size_t N = 2*n - 1; // padded size (size of non-truncated convolution)
    size_t i;

    FFTW_REAL
        *u_real,   // u coefs as real numbers
        *w_real,   // w coefs as real numbers
        *u_conv_w; // inverse FFT of U*W
    
    FFTW_COMPLEX
        *U,        // FFT of u
        *W,        // FFT of W
        *UW;       // U*W

    FFTW_PLAN
        plan_fwd_u,
        plan_fwd_w,
        plan_rev_UW;

    // Allocate arrays
    u_real   = (FFTW_REAL*) FFTW_MALLOC(N * sizeof(FFTW_REAL));
    w_real   = (FFTW_REAL*) FFTW_MALLOC(N * sizeof(FFTW_REAL));
    u_conv_w = (FFTW_REAL*) FFTW_MALLOC(N * sizeof(FFTW_REAL));
    
    U  = (FFTW_COMPLEX*) FFTW_MALLOC(n * sizeof(FFTW_COMPLEX));
    W  = (FFTW_COMPLEX*) FFTW_MALLOC(n * sizeof(FFTW_COMPLEX));
    UW = (FFTW_COMPLEX*) FFTW_MALLOC(n * sizeof(FFTW_COMPLEX));

    // Get maximum of u, w
    double u_mag_min = POS_INF;
    double u_mag_max = NEG_INF;
    double w_mag_min = POS_INF;
    double w_mag_max = NEG_INF;
    for(i = 0; i < n; i++) {
        u_mag_min = u[i].mag < u_mag_min ? u[i].mag : u_mag_min;
        u_mag_max = u[i].mag > u_mag_max ? u[i].mag : u_mag_max;
        w_mag_min = w[i].mag < w_mag_min ? w[i].mag : w_mag_min;
        w_mag_max = w[i].mag > w_mag_max ? w[i].mag : w_mag_max;
    }
    if (u_mag_max == NEG_INF) u_mag_max = 0;
    if (w_mag_max == NEG_INF) w_mag_max = 0;

    printf("u min/max=%.4f/%.4f, w min/max=%.4f/%.4f\n", u_mag_min, u_mag_max, w_mag_min, w_mag_max);

    // Set shift amounts to avoid overflow, but retain as
    // much precision as possible and minimize overflow given
    // that we won't overflow
    double u_shift = (-u_mag_max); // + 0.25*LOG_MAXVAL;
    double w_shift = (-w_mag_max); // + 0.25*LOG_MAXVAL;
    // printf("u_shift=%.5f, w_shift=%.4f\n", u_shift, w_shift);

    // Populate first n entries of u_real, w_real with input data,
    // but scale magnitude of each array by its maximum
    for (i = 0; i < n; i++) {
        u_real[i] = LS2REAL( (struct ls) {u[i].mag + u_shift, u[i].sign} );
        w_real[i] = LS2REAL( (struct ls) {w[i].mag + w_shift, w[i].sign} );
    }
    // Zero-pad: append N-n zeros to u_real, w_real
    bzero(u_real + n, (N-n)*sizeof(FFTW_REAL));
    bzero(w_real + n, (N-n)*sizeof(FFTW_REAL));

    // Plan FFT 
    plan_fwd_u = FFTW_PLAN_DFT_R2C_1D(N, u_real, U, FFTW_ESTIMATE);
    plan_fwd_w = FFTW_PLAN_DFT_R2C_1D(N, w_real, W, FFTW_ESTIMATE);

    // Execute FFT
    FFTW_EXECUTE(plan_fwd_u);
    FFTW_EXECUTE(plan_fwd_w);

    // Multiply in frequency domain (complex numbers)
    for(i = 0; i < n; i++) {
        UW[i][0] = U[i][0] * W[i][0] - U[i][1] * W[i][1]; // real part
        UW[i][1] = U[i][0] * W[i][1] + U[i][1] * W[i][0]; // imaginary part
    }

    // Plan inverse FFT
    plan_rev_UW = FFTW_PLAN_DFT_C2R_1D(N, UW, u_conv_w, FFTW_ESTIMATE);

    // Execute inverse FFT
    FFTW_EXECUTE(plan_rev_UW);

    // Extract first n coefficients of u_conv_w,
    // convert to logsign, and store in v. Also
    // restore original scaling once back in log-space
    for(i = 0; i < n; i++) {
        v[i] = REAL2LS( u_conv_w[i] / N );
        v[i].mag -= (u_shift + w_shift); // Restore original scaling
    }

    FFTW_DESTROY_PLAN(plan_fwd_u);
    FFTW_DESTROY_PLAN(plan_fwd_w);
    FFTW_DESTROY_PLAN(plan_rev_UW);

    FFTW_FREE(u_real);
    FFTW_FREE(w_real);
    FFTW_FREE(u_conv_w);
    FFTW_FREE(U);
    FFTW_FREE(W);
    FFTW_FREE(UW);
}

#endif // #if WITH_FFTW

// compute v = u / w
void gdual_div ( ls* v,
                 ls* u,
                 ls* w,
                 size_t n)
{
    /* Reference: Chapter 13 p. 305 from 
       Griewank, A. and Walther, A. Evaluating derivatives: principles and 
       techniques of algorithmic differentiation. SIAM, 2008. */

    for (size_t k = 0; k < n; k++) {

        v[k] = u[k];

        // Note that j < k is the correct termination condition, and is
        // different from gdual_mul below
        for (size_t j = 0; j < k; j++) {
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
    for (size_t i = 1; i < n; i++) {
        one[i] = ls_zero();
    }
    gdual_div(v, one, w, n);
}

void gdual_compose_same( ls* res,
                         ls* u,
                         ls* w,
                         size_t n )
{
    gdual_compose( res, n, u, n, w, n);
}

void gdual_compose( ls* res,
                    size_t n,
                    ls* u,
                    size_t u_len,
                    ls* w,
                    size_t w_len)
{
    ls *A, *B, *C;
    size_t i, j;

    // Store and zero constant entry of w
    ls w0 = w[0];
    w[0] = ls_zero();

    // Determine size and number of chunks of u
    size_t k        = (int) ceil(sqrt(u_len));
    size_t n_chunks = (int) ceil(u_len / k);

    // Create arrays
    B = (ls*) malloc( n_chunks *     k * sizeof(ls) );
    A = (ls*) malloc(        k * w_len * sizeof(ls) );
    C = (ls*) malloc( n_chunks * w_len * sizeof(ls) ); // C = B*A

    ls ZERO = ls_zero();
    
    // Fill B with chunks of u
    for (i = 0; i < n_chunks; i++) {

        // Fill row as long as there is data in u
        for (j = 0; j < k && i*k + j < u_len; j++) {
            B[i*k + j] = u[i*k + j];
        }
        
        // Fill remainder of row with zeros. Will only
        // apply to last row
        for ( ; j < k; j++)
        {
            B[i*k + j] = ZERO;
        }
    }
    
    // Fill rows of A with powers of w
    ls *row, *prev;

    // First row: set to 1
    row  = A;
    prev = NULL;
    row[0] = double_to_ls(1.0);
    for (j = 1; j < w_len; j++) {
        A[j] = ZERO;
    }

    // Remaining rows: multiply previous row times w
    for (row = A + w_len, prev = A; row < A + k * w_len; row += w_len, prev += w_len) {
        gdual_mul_same( row, prev, w, w_len);        
    }

    // Multiply B and C
    ls_mat_mul(C, B, A, n_chunks, k, w_len);

    // Now we need to compute
    //
    //   sum_i u_i(w) w^{ki}
    //
    // where u_i(w) is the composition of the ith chunk of u with
    // w, which is now stored in the ith row of C
    //
    // We can treat this as a block composition of the polynomial
    // with "coefficents" u_i(w) evaluated at the input value w^k.
    //
    // We will use Horner's method for the block composition
    
    ls *val = (ls *) malloc(n * sizeof(ls));
    ls *tmp = (ls *) malloc(n * sizeof(ls));
    
    // Compute val = w^k, the "input value" for the Horner's method
    // by multiplying the final row of A, which is equal to w^{k-1},
    // by w. Truncate to order n, the requested ouptut truncation order.
    gdual_mul(val, n, prev, w_len, w, w_len);
    
    // Begin Horner's method

    // Set res to last row of C
    ls *C_row = C + (n_chunks-1) * w_len;
    memcpy(res, C_row, n * sizeof(ls));
    
    for (C_row -= w_len; C_row >= C; C_row -= w_len ) {
        gdual_mul_same(tmp, res, val, n);
        gdual_add_different(res, n, tmp, n, C_row, w_len);
    }
    
    free(val);
    free(tmp);
    free(A);
    free(B);
    free(C);

    // restore constant entry of w
    w[0] = w0;
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
    for (size_t i = 0; i < n; i++) {
        x[i].mag = mag[i];
        x[i].sign = sign[i];
    }
}

void ls2magsign( mag_t* mag, sign_t* sign, ls *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
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
