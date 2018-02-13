
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
    
    result = .C("_gdual_exp", 
                v_mag  = double(n),
                v_sign = integer(n),
                u_mag  = as.double(u$mag),
                u_sign = as.integer(u$sign),
                nin    = as.integer(n))

    v = list(mag=result$v_mag, sign=result$v_sign)
    
    return(v)
}

# n = 6
# u = rnorm(n)
u = c(1,2,3,4,5,6)

u_ls = logsign(u)

print(u_ls)

print(gdual_exp(u_ls))
