## Copyright (C) 2020 by Landmark Acoustics LLC
#' I'm trying to learn to use tidyverse techniques in an attempt to handle
#' the ginormous data frame that I generated in "high_degree.csv"

library(dplyr, warn.conflicts=FALSE)
library(nlme, warn.conflicts=FALSE)

cat("Opening Database\n")

con <- DBI::dbConnect(RSQLite::SQLite(),
                      "data-raw/power_law_results.sqlite3")

results <- tbl(con, "HighDegree")

alphas <- (results %>%
           select(Power) %>%
           distinct() %>%
           arrange(Power) %>%
           collect())$Power

extract.lambda <- function(model) {
    B <- coef(model)
    c(Sample.Size=nobs(model),
      Intercept=B[1],
      Size.Term=B[2],
      Degree.Ratio.Term=B[3],
      AIC=AIC(model))
}

fh <- file("data-raw/high_degree_model_info.csv",
           "w",
           encoding="UTF-8")

cat(paste("Power",
          "Sample.Size",
          "Intercept",
          "Size.Term",
          "Degree.Ratio",
          "AIC",
          sep=","),
    "\n",
    file=fh,
    sep="")

cat("Fitting Models\n")
cat("......Progress......\n")

for(i in seq_along(alphas[1:10])) {

    if(i %% 10 == 0) {
        cat(".", append=TRUE)
        flush(stdout())
    }

    alpha=alphas[i]

    moo <- gls(Slope.Error ~ Inverse.Sqrt.Size + Inverse.Degree.Ratio,
        results %>% filter(Power==alpha),
        weights=varPower(form=~Size))

    cat(paste(c(alpha,
                extract.lambda(moo)),
              collapse=","),
        "\n",
        file=fh,
        sep="",
        append=TRUE)

    rm(moo)

    gc()
}

close(fh)

close(con)
