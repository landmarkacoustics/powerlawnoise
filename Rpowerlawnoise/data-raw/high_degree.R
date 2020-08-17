## Copyright (C) 2020 by Landmark Acoustics LLC
#' Code for generating really big degrees to better understand violet noise.

size.options <- expand.grid(Two=0:12, Three=0:7)
size.options$N <- 2^size.options$Two * 3^size.options$Three
size.options <- subset(size.options, N > 400 & N < 5000)
size.options <- size.options[order(size.options$N),]

command <- "python3"
arguments <- paste("-m pypowerlawnoise",
                 "data/high_degree.csv",
                 paste("-N", paste(size.options$N, collapse=" ")),
                 "-n 201",
                 "-r 20",
                 "--seed 42",
                 "-d 200",
                 "-m 20")
