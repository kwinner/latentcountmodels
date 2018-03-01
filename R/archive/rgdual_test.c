#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>

#include "../c/gdual.h"

ls lsR_to_lsc(SEXP ls_s4) {
  ls result;
  result.mag  = *REAL(GET_SLOT(ls_s4, Rf_mkString("mag")));
  result.sign = *INTEGER(GET_SLOT(ls_s4, Rf_mkString("sign")));
  return result;
}

void hello(int *n) {
  int i;
  
  for(i=0; i < *n; i++) {
    Rprintf("Hello\n");
  }
}