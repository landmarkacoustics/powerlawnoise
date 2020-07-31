## Copyright (C) 2020 by Landmark Acoustics LLC

#' Spectral slopes from simulated power law noise.
#'
#' This data set comes from running the default simulation in pypowerlawnoise.
#' There are spectral slopes from many different time series generated with
#' Kasdin's autoregressive (AR) model.
#'
#' @format A data frame with 1440768 rows and 4 variables:
#' \describe{
#'     \item{Power:}{The expected power law, \eqn{\alpha}, for the simulation.}
#'     \item{Degree:}{The number of terms in the AR model, \eqn{K}.}
#'     \item{Size:}{The number of samples in the time series, \eqn{N}.}
#'     \item{Slope:}{The observed slope of time series' log-log power spectrum.}
#' }
"results"
