#include "../c/gdual.h"
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>

// create an R ls object (to be returned to R, presumably)
#define INITIALIZE_R_LS(n)                                     \
  SEXP mag  = PROTECT(allocVector(REALSXP, n));                \
  SEXP sign = PROTECT(allocVector(INTSXP,  n));                \
  SEXP res  = PROTECT(allocVector(VECSXP,  2));                \
  SET_VECTOR_ELT(res, 0, mag);                                 \
  SET_VECTOR_ELT(res, 1, sign);                                \
  SEXP ls_class = PROTECT(allocVector(STRSXP, 2));             \
  SET_STRING_ELT(ls_class, 0, PROTECT(mkChar("ls")));          \
  SET_STRING_ELT(ls_class, 1, PROTECT(mkChar("data.frame")));  \
  classgets(res, ls_class);                                    \
  SEXP ls_names = PROTECT(allocVector(STRSXP, 2));             \
  SET_STRING_ELT(ls_names, 0, PROTECT(mkChar("mag")));         \
  SET_STRING_ELT(ls_names, 1, PROTECT(mkChar("sign")));        \
  setAttrib(res, R_NamesSymbol, ls_names);                     \
  SEXP ls_rows  = PROTECT(allocVector(INTSXP, n));             \
  int* ls_rows_access = INTEGER(ls_rows);                      \
  for(int i = 0; i < n; i++) {                                 \
    ls_rows_access[i] = i;                                     \
  }                                                            \
  setAttrib(res, R_RowNamesSymbol, ls_rows);                   \

// and a matching function to call before returning to R
// unprotects a number of R objects used in a single LS obj
#define RELEASE_R_LS(n) UNPROTECT(10);

// create an R lsgd object (to be returned to R, presumably)
// the only difference between an LS and an LSGD is the class vec
#define INITIALIZE_R_LSGD(n)                                   \
  SEXP mag  = PROTECT(allocVector(REALSXP, n));                \
  SEXP sign = PROTECT(allocVector(INTSXP,  n));                \
  SEXP res  = PROTECT(allocVector(VECSXP,  2));                \
  SET_VECTOR_ELT(res, 0, mag);                                 \
  SET_VECTOR_ELT(res, 1, sign);                                \
  SEXP ls_class = PROTECT(allocVector(STRSXP, 3));             \
  SET_STRING_ELT(ls_class, 0, PROTECT(mkChar("lsgd")));          \
  SET_STRING_ELT(ls_class, 1, PROTECT(mkChar("ls")));          \
  SET_STRING_ELT(ls_class, 2, PROTECT(mkChar("data.frame")));  \
  classgets(res, ls_class);                                    \
  SEXP ls_names = PROTECT(allocVector(STRSXP, 2));             \
  SET_STRING_ELT(ls_names, 0, PROTECT(mkChar("mag")));         \
  SET_STRING_ELT(ls_names, 1, PROTECT(mkChar("sign")));        \
  setAttrib(res, R_NamesSymbol, ls_names);                     \
  SEXP ls_rows  = PROTECT(allocVector(INTSXP, n));             \
  int* ls_rows_access = INTEGER(ls_rows);                      \
  for(int i = 0; i < n; i++) {                                 \
    ls_rows_access[i] = i;                                     \
  }                                                            \
  setAttrib(res, R_RowNamesSymbol, ls_rows);

// and a matching function to call before returning to R
// unprotects a number of R objects used in a single LSGD obj
#define RELEASE_R_LSGD(n) UNPROTECT(11);

void sexp_to_ls(ls* c_obj, SEXP* r_obj, size_t n) {
  mag_t*  mag  = REAL(   VECTOR_ELT(*r_obj, 0));
  sign_t* sign = INTEGER(VECTOR_ELT(*r_obj, 1));
  
  magsign2ls(c_obj, mag, sign, n);
}

void ls_to_sexp(SEXP* r_obj, ls* c_obj, size_t n) {
  //extract the mag and sign vectors from the r object
  mag_t*  mag  = REAL(   VECTOR_ELT(*r_obj, 0));
  sign_t* sign = INTEGER(VECTOR_ELT(*r_obj, 1));
  
  //split up the c object and put the components in mag, sign
  ls2magsign(mag, sign, c_obj, n);
}

size_t r_ls_length(SEXP r_obj) {
  return length(VECTOR_ELT(r_obj, 0));
}

/*****************
 * LS OPERATIONS *
 *****************/
// note in the following methods there are generally 2 versions:
// one prefixed by an _ and one without
// the version without prefix instantiates a new SEXP object for the result
// the version with _ prefix has an extra parameter to hold the result
//   note: even in the _ version, the operation is not done in place
//         but in a new vector of ls structs. nevertheless, the _ version
//         can avoid unnecessary memory allocation in R for "in-place" operations

typedef ls (ls_unary_op_t)     (ls);
typedef ls (ls_binary_op_t)    (ls, ls);
typedef ls (ls_scalar_op_t)    (ls, double);

/*** LS UNARY OPS ***/

// perform a unary operation on all elements in a vector of LNS numbers
void _ls_elementwise_unary_op(ls_unary_op_t func, SEXP res, SEXP x) {
  size_t n = r_ls_length(x);
  ls x_c[n], res_c[n];
  
  sexp_to_ls(x_c, &x, n);
  
  for(int i = 0; i < n; i++) {
    res_c[i] = func(x_c[i]);
  }
  
  ls_to_sexp(&res, res_c, n);
}

// negate a vector of LNS numbers, result stored in first parameter
SEXP _ls_neg_R(SEXP res, SEXP x) {
  _ls_elementwise_unary_op(&ls_neg, res, x);
  
  return res;
}

// negate a vector of LNS numbers
SEXP ls_neg_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_unary_op(&ls_neg, res, x);
  
  RELEASE_R_LS(n)
  return res;
}

// invert all elements of a vector of LNS numbers, result stored in first parameter
SEXP _ls_inv_R(SEXP res, SEXP x) {
  _ls_elementwise_unary_op(&ls_inv, res, x);
  
  return res;
}

// invert all elements of a vector of LNS numbers
SEXP ls_inv_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_unary_op(&ls_inv, res, x);
  
  RELEASE_R_LS(n)
  return res;
}

// exp all entries of a vector of LNS numbers, result stored in first parameter
SEXP _ls_exp_R(SEXP res, SEXP x) {
  _ls_elementwise_unary_op(&ls_exp, res, x);
  
  return res;
}

// exp all entries of a vector of LNS numbers
SEXP ls_exp_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_unary_op(&ls_exp, res, x);
  
  RELEASE_R_LS(n)
  return res;
}

// take the log of all entries of a vector of LNS numbers, result stored in first parameter
SEXP _ls_log_R(SEXP res, SEXP x) {
  _ls_elementwise_unary_op(&ls_log, res, x);
  
  return res;
}

// take the log of all entries of a vector of LNS numbers
SEXP ls_log_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_unary_op(&ls_log, res, x);
  
  RELEASE_R_LS(n)
  return res;
}

/*** LS BINARY OPS ***/

// perform an elementwise operation on two vectors of LNS numbers
void _ls_elementwise_binary_op(ls_binary_op_t func, SEXP res, SEXP x, SEXP y) {
  size_t n_x = r_ls_length(x);
  size_t n_y = r_ls_length(y);
  
  ls x_c[n_x], y_c[n_y]; 
  
  sexp_to_ls(x_c, &x, n_x);
  sexp_to_ls(y_c, &y, n_y);
  
  if(n_x >= n_y) {
    ls res_c[n_x];
    
    if(n_y == 1 || n_y != n_x) {
      for(int i = 0; i < n_x; i++) {
        res_c[i] = func(x_c[i], y_c[0]);
      }
    } else {
      for(int i = 0; i < n_x; i++) {
        res_c[i] = func(x_c[i], y_c[i]);
      }
    }
    
    ls_to_sexp(&res, res_c, n_x);
  } else {
    ls res_c[n_y];
    
    if(n_x == 1 || n_x != n_y) {
      for(int i = 0; i < n_y; i++) {
        res_c[i] = func(x_c[0], y_c[i]);
      }
    } else {
      for(int i = 0; i < n_y; i++) {
        res_c[i] = func(x_c[i], y_c[i]);
      }
    }
    
    ls_to_sexp(&res, res_c, n_y);
  }
}

// elementwise add vectors (or scalars) in LNS, result stored in first parameter
SEXP _ls_add_R(SEXP res, SEXP x, SEXP y) {
  _ls_elementwise_binary_op(&ls_add, res, x, y);
  
  return res;
}

// elementwise add vectors (or scalars) in LNS
SEXP ls_add_R(SEXP x, SEXP y) {
  size_t n_x = r_ls_length(x);
  size_t n_y = r_ls_length(y);
  size_t n   = (n_x >= n_y) ? n_x : n_y;
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_binary_op(&ls_add, res, x, y);
  
  RELEASE_R_LS(n)
  return res;
}

// elementwise subtract y from x, result stored in first parameter
SEXP _ls_sub_R(SEXP res, SEXP x, SEXP y) {
  _ls_elementwise_binary_op(&ls_subtract, res, x, y);
  
  return res;
}

// elementwise subtract y from x
SEXP ls_sub_R(SEXP x, SEXP y) {
  size_t n_x = r_ls_length(x);
  size_t n_y = r_ls_length(y);
  size_t n   = (n_x >= n_y) ? n_x : n_y;
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_binary_op(&ls_subtract, res, x, y);
  
  RELEASE_R_LS(n)
  return res;
}

// elementwise multiply vectors in LNS, result stored in first parameter
SEXP _ls_mul_R(SEXP res, SEXP x, SEXP y) {
  _ls_elementwise_binary_op(&ls_mult, res, x, y);
  
  return res;
}

// elementwise multiply vectors in LNS
SEXP ls_mul_R(SEXP x, SEXP y) {
  size_t n_x = r_ls_length(x);
  size_t n_y = r_ls_length(y);
  size_t n   = (n_x >= n_y) ? n_x : n_y;
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_binary_op(&ls_mult, res, x, y);
  
  RELEASE_R_LS(n)
  return res;
}

// elementwise divide the entries of x by the entries of y, result stored in first parameter
SEXP _ls_div_R(SEXP res, SEXP x, SEXP y) {
  _ls_elementwise_binary_op(&ls_div, res, x, y);
  
  return res;
}

// elementwise divide the entries of x by the entries of y
SEXP ls_div_R(SEXP x, SEXP y) {
  size_t n_x = r_ls_length(x);
  size_t n_y = r_ls_length(y);
  size_t n   = (n_x >= n_y) ? n_x : n_y;
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_binary_op(&ls_div, res, x, y);
  
  RELEASE_R_LS(n)
  return res;
}

/*** LS SCALAR OPS ***/

void _ls_elementwise_scalar_op(ls_scalar_op_t func, SEXP res, SEXP x, SEXP r) {
  size_t n = r_ls_length(x);
  ls x_c[n], res_c[n];
  
  sexp_to_ls(x_c, &x, n);
  
  size_t r_length = length(r);
  if(r_length == 1 || r_length != n) {
    // r is a scalar or has a different length than x
    // use the same value for all terms of x
    double r_c = REAL(r)[0];
    
    for(int i = 0; i < n; i++) {
      res_c[i] = func(x_c[i], r_c);
    }
  } else {
    // r is a scalar or has a different length than x
    // use the same value for all terms of x
    double* r_c = REAL(r);
    
    for(int i = 0; i < n; i++) {
      res_c[i] = func(x_c[i], r_c[i]);
    }
  }
  
  ls_to_sexp(&res, res_c, n);
}

// raise all terms in x to the rth power, result stored in first parameter
SEXP _ls_pow_R(SEXP res, SEXP x, SEXP r) {
  _ls_elementwise_scalar_op(&ls_pow, res, x, r);
  
  return res;
}

// raise all terms in x to the rth power
SEXP ls_pow_R(SEXP x, SEXP r) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LS(n)
    
  _ls_elementwise_scalar_op(&ls_pow, res, x, r);
  
  RELEASE_R_LS(n)
  return res;
}

/*******************
 * LSGD OPERATIONS *
 *******************/

typedef void (lsgd_unary_op_t) (ls*, ls*, size_t);
typedef void (lsgd_binary_op_t)(ls*, ls*, ls*,    size_t);
typedef void (lsgd_scalar_op_t)(ls*, ls*, double, size_t);

/*** LSGD UNARY OPS ***/

// perform a unary operation on a lsgd
void _lsgd_unary_op(lsgd_unary_op_t func, SEXP res, SEXP x) {
  size_t n = r_ls_length(x);
  ls x_c[n], res_c[n];
  
  sexp_to_ls(x_c, &x, n);
  
  func(res_c, x_c, n);
  
  ls_to_sexp(&res, res_c, n);
}

// <-x, dz>_q, result stored in res
SEXP _lsgd_neg_R(SEXP res, SEXP x) {
  _lsgd_unary_op(&gdual_neg, res, x);
  
  return res;
}

// <-x, dz>_q
SEXP lsgd_neg_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_unary_op(&gdual_neg, res, x);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <1/x, dz>_q, result stored in res
SEXP _lsgd_inv_R(SEXP res, SEXP x) {
  _lsgd_unary_op(&gdual_inv, res, x);
  
  return res;
}

// <1/x, dz>_q
SEXP lsgd_inv_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_unary_op(&gdual_inv, res, x);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <exp(x), dz>_q, result stored in res
SEXP _lsgd_exp_R(SEXP res, SEXP x) {
  _lsgd_unary_op(&gdual_exp, res, x);
  
  return res;
}

// <exp(x), dz>_q
SEXP lsgd_exp_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_unary_op(&gdual_exp, res, x);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <log(x), dz>_q, result stored in res
SEXP _lsgd_log_R(SEXP res, SEXP x) {
  _lsgd_unary_op(&gdual_log, res, x);
  
  return res;
}

// <log(x), dz>_q
SEXP lsgd_log_R(SEXP x) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_unary_op(&gdual_log, res, x);
  
  RELEASE_R_LSGD(n)
  return res;
}

/*** LSGD BINARY OPS ***/

// perform a binary operation on two (equal length) lsgds
// todo: support for different length lsgds
void _lsgd_binary_op(lsgd_binary_op_t func, SEXP res, SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  ls x_c[n], y_c[n], res_c[n];
  
  sexp_to_ls(x_c, &x, n);
  sexp_to_ls(y_c, &y, n);
  
  func(res_c, x_c, y_c, n);
  
  ls_to_sexp(&res, res_c, n);
}

// <x+y,dz>_q, result stored in first parameter
SEXP _lsgd_add_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_add, res, x, y);
  
  return res;
}

// <x+y,dz>_q
SEXP lsgd_add_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_add, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <x-y,dz>_q, result stored in first parameter
SEXP _lsgd_sub_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_sub, res, x, y);
  
  return res;
}

// <x-y,dz>_q
SEXP lsgd_sub_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_sub, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <x*y,dz>_q, result stored in first parameter
SEXP _lsgd_mul_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_mul_same, res, x, y);
  
  return res;
}

// <x*y,dz>_q
SEXP lsgd_mul_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_mul_same, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <x/y,dz>_q, result stored in first parameter
SEXP _lsgd_div_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_div, res, x, y);
  
  return res;
}

// <x/y,dz>_q
SEXP lsgd_div_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_div, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <x(y),dz>_q, result stored in first parameter
SEXP _lsgd_compose_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_compose_same, res, x, y);
  
  return res;
}

// <x(y),dz>_q
SEXP lsgd_compose_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_compose_same, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

// <x(y),dz>_q, for affine y result stored in first parameter
SEXP _lsgd_compose_affine_R(SEXP res, SEXP x, SEXP y) {
  _lsgd_binary_op(&gdual_compose_affine, res, x, y);
  
  return res;
}

// <x(y),dz>_q, for affine y
SEXP lsgd_compose_affine_R(SEXP x, SEXP y) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_binary_op(&gdual_compose_affine, res, x, y);
  
  RELEASE_R_LSGD(n)
  return res;
}

/*** LSGD SCALAR OPS ***/

// perform a scalar operation (i.e. pow) on an lsgd
void _lsgd_scalar_op(lsgd_scalar_op_t func, SEXP res, SEXP x, SEXP r) {
  size_t n = r_ls_length(x);
  ls x_c[n], res_c[n];
  
  sexp_to_ls(x_c, &x, n);
  double r_c = REAL(r)[0];
  
  func(res_c, x_c, r_c, n);
  
  ls_to_sexp(&res, res_c, n);
}

// <x^r,dz>_q, result stored in first parameter
SEXP _lsgd_pow_R(SEXP res, SEXP x, SEXP r) {
  _lsgd_scalar_op(&gdual_pow, res, x, r);
  
  return res;
}

// <x^r,dz>_q
SEXP lsgd_pow_R(SEXP x, SEXP r) {
  size_t n = r_ls_length(x);
  INITIALIZE_R_LSGD(n)
    
  _lsgd_scalar_op(&gdual_pow, res, x, r);
  
  RELEASE_R_LSGD(n)
  return res;
}