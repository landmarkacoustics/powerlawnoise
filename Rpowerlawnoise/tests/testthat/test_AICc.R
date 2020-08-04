## Copyright (C) 2020 by Landmark Acoustics LLC

library(Rpowerlawnoise)


test_that("AICc penalizes extra parameters", {
    N <- 10
    x <- rnorm(N)
    y <- 10*x + rnorm(N)
    m <- lm(y ~ x)
    expect_lt(AICc(m), AIC(m))
})
