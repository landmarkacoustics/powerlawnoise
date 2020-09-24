## Copyright (C) 2020 by Landmark Acoustics LLC

con <- DBI::dbConnect(RSQLite::SQLite(), "data-raw/power_law_results.sqlite3")
high.degree <- tbl(con, "HighDegree")
model.frame <- high.degree %>% select(Power) %>% distinct %>% collect
fo <- Slope.Error^2 ~ Inverse.Sqrt.Size + Inverse.Degree.Ratio
fam = Gamma(link="log")
model.frame <- model.frame %>% add_column(Model = model.frame$Power %>% map(~glm(fo, fam, high.degree %>% filter(Power>.x-0.01 & Power < .x+0.01), model=FALSE)))
model.frame$AIC <- model.frame$Model %>% map_dbl(AIC)
fo <- as.formula(paste("Inverse.Degree.Ratio ~", paste("I(Power^", 1:20, ")", sep="", collapse=" + ")))
pink.model <- step(lm(fo, model.frame %>% filter(Power <= 0)))
write.csv(enframe(coef(pink.model)), file="../manuscript/pink_coefs.csv")
model.frame$Pink.Prediction <- predict(pink.model, model.frame)
blue.model <- step(lm(update(fo, . ~ . - 1), model.frame %>% filter(Power > 0)))
write.csv(enframe(coef(blue.model)), file="../manuscript/blue_coefs.csv")
model.frame$Blue.Prediction <- predict(pink.model, model.frame)
model.frame$Pink.Prediction[model.frame$Power > 0] <- NA
model.frame$Blue.Prediction[model.frame$Power < 0] <- NA
save(model.frame, pink.model, blue.model, file="data-raw/model_frame.Rdata")
ggplot(model.frame, aes(x=Power)) +
    geom_line(aes(y=Pink.Prediction), lwd=2, col="red") +
    geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) +
    geom_point(aes(y=Inverse.Degree.Ratio), col="black", cex=3, pch=1) +
    theme_bw() +
    labs(x=expression(alpha), y=expression(B[2](alpha)))
ggsave("../manuscript/degree_ratio_parameter.eps")
