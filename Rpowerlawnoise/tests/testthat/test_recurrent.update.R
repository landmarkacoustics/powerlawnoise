## Copyright (C) 2020 by Landmark Acoustics LLC

library(Rpowerlawnoise)


test_that("recursion creates sub-models", {
    fo <- A ~ 1
    xtras <- LETTERS[2:3]
    should.be <- list(A ~ B,
                      A ~ B + C,
                      A ~ C)
    expect_equivalent(recurrent.update(fo, xtras), should.be)
})
