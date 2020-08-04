## Copyright (C) 2020 by Landmark Acoustics LLC

plot.slopes <- function(file.name="Slope versus Exponent.png",
                        side.length=8,
                        point.size=14) {
    dev.new(height=side.length,
            width=side.length,
            pointsize=point.size)

    plot(0, 0,
         type="n", las=1,
         xlab=expression(paste("Model Exponent (", alpha, ")", sep="")),
         ylab=paste("Actual Spectral Slope (a)"),
         xlim=c(-2, 2), ylim=c(-2, 2))

    grid()

    points(Slope ~ Power, results, pch=".", col=size.colors[factor(Size)])

    abline(0, 1, lty=2)

    legend("topleft", legend=sizes, pch=16, col=size.colors, title="N", bty="n")


    dev.print(png, filename=file.name,
              res=90,
              width=side.length,
              height=side.length,
              units="in")

    NULL
}
