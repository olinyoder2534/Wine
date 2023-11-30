library(caret)
library(e1071)
library(MLmetrics)

reg_rf_wine <- read.csv('/Users/olinyoder/Desktop/Fall 2023/BSAN 480/Data/wine+quality/WineQuality2RFRegressionEval2.csv')

head(reg_rf_wine, 1)

conf_matrix <- confusionMatrix(factor(reg_rf_wine$PREDICTION2), factor(reg_rf_wine$Quality))
conf_matrix
#~72% accuracy for the RF in regression

