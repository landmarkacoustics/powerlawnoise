
R version 3.6.3 (2020-02-29) -- "Holding the Windsock"
Copyright (C) 2020 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

[Previously saved workspace restored]

> setwd('/home/ben/Source/Open/powerlawnoise/Rpowerlawnoise/')
> library(tidyverse)
── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──
✔ ggplot2 3.3.2     ✔ purrr   0.3.4
✔ tibble  3.0.3     ✔ dplyr   1.0.2
✔ tidyr   1.1.2     ✔ stringr 1.4.0
✔ readr   1.3.1     ✔ forcats 0.5.0
── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()
> load("data-raw/model_frame.Rdata")
> fo <- as.formula(paste("Slope.Error^2 ~", paste("I(Power^", 1:20, ")", sep="", collapse=" + ")))
> fo
Slope.Error^2 ~ I(Power^1) + I(Power^2) + I(Power^3) + I(Power^4) + 
    I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + I(Power^9) + 
    I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + I(Power^14) + 
    I(Power^15) + I(Power^16) + I(Power^17) + I(Power^18) + I(Power^19) + 
    I(Power^20)
> ?drop1
> better.pink <- step(lm(fo, model.frame %>% filter(Power <= 0)))
Error in eval(predvars, data, env) : object 'Slope.Error' not found
> fo <- as.formula(paste("Inverse.Degree.Ratio ~", paste("I(Power^", 1:20, ")", sep="", collapse=" + ")))
> better.pink <- step(lm(fo, model.frame %>% filter(Power <= 0)))
Start:  AIC=-1384.27
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^18) + 
    I(Power^19) + I(Power^20)


Step:  AIC=-1384.27
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^18) + 
    I(Power^20)


Step:  AIC=-1384.27
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^20)


Step:  AIC=-1384.27
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^17) + I(Power^20)


Step:  AIC=-1384.27
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^15) + I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^5)   1 8.7228e-07 8.1377e-05 -1385.2
- I(Power^6)   1 8.7413e-07 8.1379e-05 -1385.2
- I(Power^7)   1 8.8454e-07 8.1389e-05 -1385.2
- I(Power^8)   1 8.9593e-07 8.1400e-05 -1385.2
- I(Power^9)   1 9.0871e-07 8.1413e-05 -1385.1
- I(Power^4)   1 9.2154e-07 8.1426e-05 -1385.1
- I(Power^10)  1 9.2425e-07 8.1429e-05 -1385.1
- I(Power^11)  1 9.4344e-07 8.1448e-05 -1385.1
- I(Power^12)  1 9.6655e-07 8.1471e-05 -1385.1
- I(Power^13)  1 9.9348e-07 8.1498e-05 -1385.0
- I(Power^1)   1 1.0265e-06 8.1531e-05 -1385.0
- I(Power^15)  1 1.0574e-06 8.1562e-05 -1385.0
- I(Power^17)  1 1.1323e-06 8.1637e-05 -1384.9
- I(Power^3)   1 1.2149e-06 8.1719e-05 -1384.8
- I(Power^20)  1 1.2596e-06 8.1764e-05 -1384.7
<none>                      8.0504e-05 -1384.3
- I(Power^2)   1 2.5948e-06 8.3099e-05 -1383.1

Step:  AIC=-1385.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^6) + I(Power^7) + I(Power^8) + I(Power^9) + 
    I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + I(Power^15) + 
    I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^6)   1 2.6000e-09 8.1379e-05 -1387.2
- I(Power^7)   1 1.3300e-08 8.1390e-05 -1387.2
- I(Power^8)   1 2.5800e-08 8.1402e-05 -1387.2
- I(Power^9)   1 4.0000e-08 8.1417e-05 -1387.1
- I(Power^10)  1 5.6700e-08 8.1433e-05 -1387.1
- I(Power^11)  1 7.6600e-08 8.1453e-05 -1387.1
- I(Power^4)   1 9.0700e-08 8.1467e-05 -1387.1
- I(Power^12)  1 9.9900e-08 8.1477e-05 -1387.1
- I(Power^13)  1 1.2660e-07 8.1503e-05 -1387.0
- I(Power^15)  1 1.8940e-07 8.1566e-05 -1387.0
- I(Power^1)   1 2.0610e-07 8.1583e-05 -1386.9
- I(Power^17)  1 2.6280e-07 8.1639e-05 -1386.9
- I(Power^20)  1 3.8840e-07 8.1765e-05 -1386.7
- I(Power^3)   1 7.0390e-07 8.2081e-05 -1386.3
<none>                      8.1377e-05 -1385.2
- I(Power^2)   1 3.6263e-06 8.5003e-05 -1382.8

Step:  AIC=-1387.18
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^7) + I(Power^8) + I(Power^9) + I(Power^10) + 
    I(Power^11) + I(Power^12) + I(Power^13) + I(Power^15) + I(Power^17) + 
    I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^1)   1 5.4420e-07 8.1924e-05 -1388.5
- I(Power^9)   1 6.2130e-07 8.2001e-05 -1388.4
- I(Power^10)  1 6.2740e-07 8.2007e-05 -1388.4
- I(Power^11)  1 6.5890e-07 8.2038e-05 -1388.4
- I(Power^8)   1 6.5920e-07 8.2039e-05 -1388.4
- I(Power^12)  1 7.0580e-07 8.2085e-05 -1388.3
- I(Power^13)  1 7.6240e-07 8.2142e-05 -1388.2
- I(Power^7)   1 7.8020e-07 8.2160e-05 -1388.2
- I(Power^15)  1 8.9090e-07 8.2270e-05 -1388.1
- I(Power^17)  1 1.0286e-06 8.2408e-05 -1387.9
- I(Power^20)  1 1.2399e-06 8.2619e-05 -1387.7
<none>                      8.1379e-05 -1387.2
- I(Power^4)   1 3.5585e-06 8.4938e-05 -1384.9
- I(Power^3)   1 8.2187e-06 8.9598e-05 -1379.5
- I(Power^2)   1 1.7648e-05 9.9027e-05 -1369.4

Step:  AIC=-1388.51
Inverse.Degree.Ratio ~ I(Power^2) + I(Power^3) + I(Power^4) + 
    I(Power^7) + I(Power^8) + I(Power^9) + I(Power^10) + I(Power^11) + 
    I(Power^12) + I(Power^13) + I(Power^15) + I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^9)   1 1.6300e-07 8.2086e-05 -1390.3
- I(Power^10)  1 1.7700e-07 8.2101e-05 -1390.3
- I(Power^8)   1 1.7800e-07 8.2101e-05 -1390.3
- I(Power^11)  1 2.0900e-07 8.2133e-05 -1390.2
- I(Power^12)  1 2.5200e-07 8.2176e-05 -1390.2
- I(Power^7)   1 2.5900e-07 8.2182e-05 -1390.2
- I(Power^13)  1 3.0400e-07 8.2227e-05 -1390.1
- I(Power^15)  1 4.2100e-07 8.2345e-05 -1390.0
- I(Power^17)  1 5.5000e-07 8.2473e-05 -1389.8
- I(Power^20)  1 7.5100e-07 8.2675e-05 -1389.6
<none>                      8.1924e-05 -1388.5
- I(Power^4)   1 5.5140e-06 8.7438e-05 -1383.9
- I(Power^3)   1 2.5255e-05 1.0718e-04 -1363.4
- I(Power^2)   1 1.5364e-04 2.3556e-04 -1283.8

Step:  AIC=-1390.31
Inverse.Degree.Ratio ~ I(Power^2) + I(Power^3) + I(Power^4) + 
    I(Power^7) + I(Power^8) + I(Power^10) + I(Power^11) + I(Power^12) + 
    I(Power^13) + I(Power^15) + I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^8)   1 0.00000011 0.00008220 -1392.2
- I(Power^10)  1 0.00000013 0.00008222 -1392.1
- I(Power^11)  1 0.00000034 0.00008243 -1391.9
- I(Power^12)  1 0.00000057 0.00008265 -1391.6
- I(Power^13)  1 0.00000078 0.00008287 -1391.3
- I(Power^7)   1 0.00000084 0.00008293 -1391.3
- I(Power^15)  1 0.00000115 0.00008324 -1390.9
- I(Power^17)  1 0.00000145 0.00008354 -1390.5
<none>                      0.00008209 -1390.3
- I(Power^20)  1 0.00000180 0.00008389 -1390.1
- I(Power^4)   1 0.00003023 0.00011232 -1360.6
- I(Power^3)   1 0.00010123 0.00018332 -1311.2
- I(Power^2)   1 0.00042201 0.00050410 -1209.0

Step:  AIC=-1392.17
Inverse.Degree.Ratio ~ I(Power^2) + I(Power^3) + I(Power^4) + 
    I(Power^7) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^15) + I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
<none>                      0.00008220 -1392.2
- I(Power^20)  1 0.00001595 0.00009815 -1376.3
- I(Power^17)  1 0.00001933 0.00010153 -1372.8
- I(Power^15)  1 0.00002294 0.00010514 -1369.3
- I(Power^13)  1 0.00002849 0.00011069 -1364.1
- I(Power^12)  1 0.00003242 0.00011462 -1360.6
- I(Power^11)  1 0.00003752 0.00011972 -1356.2
- I(Power^10)  1 0.00004429 0.00012649 -1350.6
- I(Power^7)   1 0.00008634 0.00016854 -1321.7
- I(Power^4)   1 0.00028104 0.00036324 -1244.1
- I(Power^3)   1 0.00054276 0.00062496 -1189.3
- I(Power^2)   1 0.00136165 0.00144385 -1104.7
> names(model.frame)
[1] "Power"                "Model"                "(Intercept)"         
[4] "Inverse.Sqrt.Size"    "Inverse.Degree.Ratio" "AIC"                 
> model.frame <- model.frame %>% mutate(Pink.Prediction = predict(better.pink, .data))
Error: Problem with `mutate()` input `Pink.Prediction`.
✖ cannot coerce class ‘"rlang_data_pronoun"’ to a data.frame
ℹ Input `Pink.Prediction` is `predict(better.pink, .data)`.
Run `rlang::last_error()` to see where the error occurred.
> model.frame <- model.frame %>% mutate(Pink.Prediction = predict(better.pink, .x))
Error: Problem with `mutate()` input `Pink.Prediction`.
✖ object '.x' not found
ℹ Input `Pink.Prediction` is `predict(better.pink, .x)`.
Run `rlang::last_error()` to see where the error occurred.
> model.frame$Pink.Prediction <- predict(better.pink, model.frame)
> better.blue <- step(lm(fo, model.frame %>% filter(Power >= 0)))
Start:  AIC=-1207.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^18) + 
    I(Power^19) + I(Power^20)


Step:  AIC=-1207.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^18) + 
    I(Power^20)


Step:  AIC=-1207.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^16) + I(Power^17) + I(Power^20)


Step:  AIC=-1207.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^14) + I(Power^15) + I(Power^17) + I(Power^20)


Step:  AIC=-1207.19
Inverse.Degree.Ratio ~ I(Power^1) + I(Power^2) + I(Power^3) + 
    I(Power^4) + I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + 
    I(Power^9) + I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + 
    I(Power^15) + I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
- I(Power^1)   1 2.4010e-06 0.00046721 -1208.7
<none>                      0.00046481 -1207.2
- I(Power^2)   1 9.9040e-06 0.00047472 -1207.1
- I(Power^3)   1 1.3240e-05 0.00047805 -1206.3
- I(Power^4)   1 1.9724e-05 0.00048454 -1205.0
- I(Power^5)   1 2.7742e-05 0.00049255 -1203.3
- I(Power^6)   1 3.6503e-05 0.00050131 -1201.5
- I(Power^7)   1 4.5670e-05 0.00051048 -1199.7
- I(Power^8)   1 5.5111e-05 0.00051992 -1197.9
- I(Power^9)   1 6.4777e-05 0.00052959 -1196.0
- I(Power^10)  1 7.4655e-05 0.00053947 -1194.2
- I(Power^11)  1 8.4745e-05 0.00054956 -1192.3
- I(Power^12)  1 9.5058e-05 0.00055987 -1190.4
- I(Power^13)  1 1.0560e-04 0.00057041 -1188.5
- I(Power^15)  1 1.2743e-04 0.00059224 -1184.7
- I(Power^17)  1 1.5032e-04 0.00061513 -1180.9
- I(Power^20)  1 1.8683e-04 0.00065165 -1175.1

Step:  AIC=-1208.67
Inverse.Degree.Ratio ~ I(Power^2) + I(Power^3) + I(Power^4) + 
    I(Power^5) + I(Power^6) + I(Power^7) + I(Power^8) + I(Power^9) + 
    I(Power^10) + I(Power^11) + I(Power^12) + I(Power^13) + I(Power^15) + 
    I(Power^17) + I(Power^20)

              Df  Sum of Sq        RSS     AIC
<none>                      0.00046721 -1208.7
- I(Power^3)   1 2.4391e-05 0.00049160 -1205.5
- I(Power^2)   1 3.0921e-05 0.00049813 -1204.2
- I(Power^4)   1 3.2142e-05 0.00049935 -1204.0
- I(Power^5)   1 4.2270e-05 0.00050948 -1201.9
- I(Power^6)   1 5.2869e-05 0.00052008 -1199.8
- I(Power^7)   1 6.3524e-05 0.00053074 -1197.8
- I(Power^8)   1 7.4179e-05 0.00054139 -1195.8
- I(Power^9)   1 8.4864e-05 0.00055208 -1193.8
- I(Power^10)  1 9.5627e-05 0.00056284 -1191.9
- I(Power^11)  1 1.0651e-04 0.00057372 -1189.9
- I(Power^12)  1 1.1755e-04 0.00058477 -1188.0
- I(Power^13)  1 1.2878e-04 0.00059599 -1186.1
- I(Power^15)  1 1.5190e-04 0.00061911 -1182.2
- I(Power^17)  1 1.7603e-04 0.00064324 -1178.4
- I(Power^20)  1 2.1439e-04 0.00068161 -1172.5
>  model.frame$Blue.Prediction <- predict(better.blue, model.frame)
> model.frame$Pink.Prediction[model.frame$Power > 0] <- NA
> model.frame$Blue.Prediction[model.frame$Power < 0] <- NA
> ggplot(model.frame, aes(x=Power)) + geom_point(aes(y=Inverse.Degree.Ratio), col=black, pt.cex=2) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(y=Blue.Prediction, col="blue", lwd=2) + theme_minimal()
Error in layer(data = data, mapping = mapping, stat = stat, geom = GeomPoint,  : 
  object 'black' not found
> ggplot(model.frame, aes(x=Power)) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", pt.cex=3) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(y=Blue.Prediction, col="blue", lwd=2) + theme_minimal()
Error in layer(data = data, mapping = mapping, stat = stat, geom = GeomLine,  : 
  object 'Blue.Prediction' not found
In addition: Warning message:
Ignoring unknown parameters: pt.cex 
> names(model.frame)
[1] "Power"                "Model"                "(Intercept)"         
[4] "Inverse.Sqrt.Size"    "Inverse.Degree.Ratio" "AIC"                 
[7] "Pink.Prediction"      "Blue.Prediction"     
> ggplot(model.frame, aes(x=Power)) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", pt.cex=3) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) + theme_minimal()
Warning messages:
1: Ignoring unknown parameters: pt.cex 
2: Removed 100 row(s) containing missing values (geom_path). 
3: Removed 100 row(s) containing missing values (geom_path). 
> ggplot(model.frame, aes(x=Power)) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", cex=3) + theme_minimal()
Warning messages:
1: Removed 100 row(s) containing missing values (geom_path). 
2: Removed 100 row(s) containing missing values (geom_path). 
> ggplot(model.frame, aes(x=Power)) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", cex=3, pch=1) + theme_minimal()
Warning messages:
1: Removed 100 row(s) containing missing values (geom_path). 
2: Removed 100 row(s) containing missing values (geom_path). 
> enframe(coef(better.pink))
# A tibble: 12 x 2
   name            value
   <chr>           <dbl>
 1 (Intercept) -0.00227 
 2 I(Power^2)   1.63    
 3 I(Power^3)   4.98    
 4 I(Power^4)   4.98    
 5 I(Power^7)   4.22    
 6 I(Power^10) 20.0     
 7 I(Power^11) 36.3     
 8 I(Power^12) 27.0     
 9 I(Power^13)  8.53    
10 I(Power^15) -0.532   
11 I(Power^17)  0.0314  
12 I(Power^20)  0.000292
> write.csv(enframe(coef(better.pink)), file="../manuscript/pink_coefs.csv")
> write.csv(enframe(coef(better.blue)), file="../manuscript/blue_coefs.csv")
> enframe(coef(better.blue))
# A tibble: 16 x 2
   name           value
   <chr>          <dbl>
 1 (Intercept)  9.42e-5
 2 I(Power^2)   4.42e+0
 3 I(Power^3)  -7.95e+1
 4 I(Power^4)   8.19e+2
 5 I(Power^5)  -4.84e+3
 6 I(Power^6)   1.79e+4
 7 I(Power^7)  -4.41e+4
 8 I(Power^8)   7.49e+4
 9 I(Power^9)  -8.96e+4
10 I(Power^10)  7.53e+4
11 I(Power^11) -4.34e+4
12 I(Power^12)  1.60e+4
13 I(Power^13) -3.10e+3
14 I(Power^15)  1.00e+2
15 I(Power^17) -3.87e+0
16 I(Power^20)  2.36e-2
> ggplot(model.frame, aes(x=Power)) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", cex=3, pch=1) + theme_bw()
Warning messages:
1: Removed 100 row(s) containing missing values (geom_path). 
2: Removed 100 row(s) containing missing values (geom_path). 
> ggplot(model.frame, aes(x=Power)) + geom_line(aes(y=Pink.Prediction), lwd=2, col="red") + geom_line(aes(y=Blue.Prediction), col="blue", lwd=2) + geom_point(aes(y=Inverse.Degree.Ratio), col="black", cex=3, pch=1) + theme_bw() + labs(x=expression(alpha), y=expression(B[2](alpha)))
Warning messages:
1: Removed 100 row(s) containing missing values (geom_path). 
2: Removed 100 row(s) containing missing values (geom_path). 
> ggsave("../manuscript/degree_ratio_parameter.png")
Saving 6.99 x 6.99 in image
Warning messages:
1: Removed 100 row(s) containing missing values (geom_path). 
2: Removed 100 row(s) containing missing values (geom_path). 
> q("no")

Process R finished at Sat Sep 19 11:25:23 2020
