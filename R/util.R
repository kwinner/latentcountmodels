# function for formating a number of seconds as HMS
s2hms <- function(s) {
  h <- floor(s / 3600)
  m <- floor((s - 3600*h) / 60)
  s <- floor(s - 3600*h - 60*m)
  
  if (h > 0)
    sprintf('%dh%02dm%02ds', h, m, s)
  else if (m > 0)
    sprintf('%dm%02ds', m, s)
  else
    sprintf('%ds', s)
}

# function to parse a simple numeric matrix to a simple string form
mat2str <- function(x) {
  if (is.vector(x) || is.list(x) || nrow(x) == 1)
    return(paste('[', toString(x), ']', sep=''))
  else
    return(paste('[[', paste(apply(x, 1, toString), collapse = '];['), ']]', sep=''))
}

# function to parse a standard numeric matrix in string form
str2mat <- function(str) {
  str <- trimws(str)
  
  # remove leading/trailing brackets
  str <- regmatches(str, regexpr('(?=\\[*)[^\\[].*[^\\]](?=\\]*)', str, perl=TRUE))
  
  # split by matrix rows
  if (grepl('\\];\\[', str))
    str <- regmatches(str, gregexpr('\\];\\[', str), invert=TRUE)[[1]]
  
  if (length(str) == 1) {
    # build as a row vector
    mat <- as.numeric(strsplit(str, ',')[[1]])
    
    # convert to a 2D array
    dim(mat) <- c(1, length(mat))
  } else {
    # convert each row to numeric matrix
    mat <- lapply(str, function(row) {as.numeric(strsplit(row, ',')[[1]])})
    
    # append to matrix
    mat <- do.call(rbind, mat)
  }
  
  return(mat)
}