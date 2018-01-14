#include <stddef.h>
#include <stdint.h>

// Log-sign number system types
typedef double mag_t;
typedef int32_t sign_t;

typedef struct ls {
    mag_t  mag;
    sign_t sign;
} ls;

// Log-sign number system functions
double ls_to_double( ls x );
ls double_to_ls(double x);
ls ls_zero();
ls ls_add( ls x, ls y );
ls ls_neg( ls x );
ls ls_subtract( ls x, ls y );
ls ls_mult( ls x, ls y );
ls ls_inv( ls x );
ls ls_div( ls x, ls y );
ls ls_log( ls x );
ls ls_exp( ls x );

// gduals
void gdual_print( ls* a, size_t n );
void gdual_print_as_double( ls* a, size_t n );

// Unary operations
void            gdual_exp( ls* res, ls* u, size_t n);
void            gdual_log( ls* res, ls* u, size_t n);
void            gdual_inv( ls* res, ls* u, size_t n);
void            gdual_neg( ls* res, ls* u, size_t n);

// Binary-ish operations
void     gdual_scalar_mul( ls* res, ls* u, double c, size_t n);
void            gdual_pow( ls* res, ls* u, double r, size_t n);
void      gdual_u_plus_cw( ls* res, ls* u, ls* w, double c, size_t n);

void gdual_pow_int( ls* v, ls* u, int r, size_t n );
void gdual_pow_fractional( ls* v, ls* u, double r, size_t n );


// Binary operations: different-sized operands
void            gdual_mul( ls* res, size_t n, ls* u, size_t u_len, ls* w, size_t w_len);
void        gdual_compose( ls* res, size_t n, ls* u, size_t u_len, ls* w, size_t w_len);

// Binary operations: same-sized operands
void       gdual_mul_same( ls* res, ls* u, ls* w, size_t n);
void            gdual_add( ls* res, ls* u, ls* w, size_t n);
void            gdual_div( ls* res, ls* u, ls* w, size_t n);
void   gdual_compose_same( ls* res, ls* u, ls* w, size_t n);
void gdual_compose_affine( ls* res, ls* u, ls* w, size_t n);

#ifdef WITH_FFT
void        gdual_mul_fft( ls* res, ls* u, ls* w, size_t n);
#endif

// Wrappers that accept separate log-magnitude and sign arrays
void magsign2ls( ls* x, mag_t* mag, sign_t* sign, size_t n);
void ls2magsign( mag_t* mag, sign_t* sign, ls *x, size_t n);

void _gdual_exp( mag_t*  u_mag,
                 sign_t* u_sign,
                 mag_t*  v_mag,
                 sign_t* v_sign,
                 int* nin);

