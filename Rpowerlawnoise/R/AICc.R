## Copyright (C) 2020 by Landmark Acoustics LLC

#' Corrected Akaike Information Criterion
#'
#' I think this is Williamson's correction, but of course I will look up the
#' proper attribution before I publish this too widely. The idea is that extra
#' parameters are penalized more harshly when there is a small sample size.
#' @param model Some object that defines the [logLik()] function
#' @importFrom stats coef logLik nobs
#' @export
AICc <- function(model){
    k <- length(coef(model))
    n <- nobs(model)
    correction <- k*(k + 1) / (n - k - 1)
    L <- as.numeric(logLik(model))
    return( 2 * (k - L + correction) )
}

