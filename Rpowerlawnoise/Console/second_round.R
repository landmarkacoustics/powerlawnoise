## Copyright (C) 2020 by Landmark Acoustics LLC

adams.results <- read.csv("../pypowerlawnoise/data/power_law_output.csv")

sizes <- sort(unique(adams.results$Size))
size.colors <- rainbow(length(sizes), alpha=0.75)
degrees <- sort(unique(adams.results$Degree))
slope.info <- expand.grid(Size=sizes, Degree=degrees)
slope.models <- lapply(seq_len(nrow(slope.info)),
                       function(i) {
                           lm(Slope ~ Power - 1, adams.results,
                              subset=Size == slope.info$Size[i] &
                                  Degree == slope.info$Degree[i])
                       })
slope.info$Metaslope <- sapply(slope.models, coef)
slope.info$SS.Residuals <- sapply(slope.models, function(m)sum(residuals(m)^2))
slope.info$Degree.Ratio <- with(slope.info, Degree^2 / Size)
plot(Metaslope ~ Degree, slope.info,
	 type="n", las=1,
     xlab="Degree of Model", ylab="Metaslope between Slope and Power")
invisible(lapply(seq_along(sizes), function(i){
    points(Metaslope ~ Degree, slope.info, subset=Size==sizes[i],
	       type="b", lty=1, col=size.colors[i], pch=16)
}))
legend("bottomright", legend=sizes, title="Length of Time Series",
       pch=16, col=size.colors[seq_along(sizes)], lty=1, bty="n")

xl <- with(subset(slope.info, Degree.Ratio>=0.01 & Degree.Ratio <= 100), range(Degree.Ratio))
drs <- 10^seq(xl[1], xl[2], length.out=101)
plot(Metaslope ~ I(Degree.Ratio^2), slope.info, col=size.colors[factor(Size)], las=1, xlab=expression(K^2/N), ylab="Metaslope", log="x", xlim=10^c(-2,2), ylim=c(0.85, 0.9))
grid()
abline(v=1, lty=2)
