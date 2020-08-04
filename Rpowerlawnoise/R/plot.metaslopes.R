## Copyright (C) 2020 by Landmark Acoustics LLC


plot.metaslopes <- function(file.name="Metaslope versus Degree Ratio.png",
                            side.length=7,
                            point.size=14) {

    dev.new(width=side.length,
            height=side.length,
            pointsize=point.size)

    xl <- with(subset(slope.info, Degree.Ratio > 0.1 & Degree.Ratio < 10),
               range(Degree.Ratio))

    plot(0, 0, type="n",
         las=1, log="x",
         xlab=expression(paste("Degree Ratio (", K/sqrt(N), ")", sep="")),
         ylab="Metaslope (M)",
         xlim=xl, ylim=c(0.85, 0.9)
         )

    grid()
    
    points(Metaslope ~ Degree.Ratio, slope.info,
           pch=16,
           col=size.colors[factor(Size)])

    abline(v=1.0, lty=2)

    legend("topleft", legend=sizes,
           title="N",
           pch=16, col=size.colors,
           bty="n")

    dev.print(png, filename=file.name,
              width=side.length, height=side.length, units="in",
              res=90)
}
