
dyn.load("../c/libgdual.so")

logsign = function(u) {
    u_ls = list(
        mag  = log(abs(u)),
        sign = sign(u)
    )
    return(u_ls)
}

gdual_exp = function(u) {
    
    n = length(u$mag)
    
    result = .C("_gdual_exp", u_mag  = as.double(u$mag),
                u_sign = as.integer(u$sign),
                v_mag  = double(n),
                v_sign = integer(n),
                nin    = as.integer(n))

    v = list(mag=result$v_mag, sign=result$v_sign)
    
    return(v)
}

n = 6
u = rnorm(n)

u_ls = logsign(u)

print(gdual_exp(u_ls))
