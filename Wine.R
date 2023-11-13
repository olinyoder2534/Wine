
library(MASS)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(brant)
library(rms)
library(xgboost)

#LOADING DATA
#----
wineQuality <- read.csv('FilePath.csv')
wineLookup <- read.csv('FilePath.csv')

dim(wineQuality)

head(wineLookup)
head(wineQuality, 1)

#Join data
#joining data is only necessary for xgboost model (or you could one-hot encode "Color" from the wineQuality dataset instead of joining)
wine <- merge(wineQuality, wineLookup, by = "ColorID")
dim(wine)
head(wine)
summary(wine)

dim(wine)
sum(is.na(wine))
t(t(sapply(wine, class)))
#----

#PREPROCESSING
#----
#most visualizations and basic data exploration have been done elsewhere
#no major changes are blatantly necessary

par(mfrow = c(1,2))
#boxplot hist -- to change between hist & boxplot
hist(wine$FixedAcidity, main = "FixedAcidity")
hist(wine$VolatileAcidity, main = "VolatileAcidity")
hist(wine$CitricAcid, main = "CitricAcid")
hist(wine$ResidualSugar, main = "ResidualSugar")
hist(wine$Chlorides, main = "Chlorides")
hist(wine$FreeSulfurDioxide, main = "FreeSulfurDioxide")
hist(wine$TotalSulfurDioxide, main = "TotalSulfurDioxide")
hist(wine$Density, main = "Density")
hist(wine$pH, main = "pH")
hist(wine$Sulphates, main = "Sulphates")
hist(wine$Alcohol, main = "Alcohol")
hist(wine$Quality, main = "Quality")
#most variables are slightly skewed right, not too major of a concern
#transformations could be made, but I am going to hold off for now
#can check by color, too

#check for multicollinearity
#GGally::ggpairs(wine)
#no obvious multicollinearity issues

#recode variables
wine$ColorID <- as.factor(wine$ColorID)
wine$Quality <- as.factor(wine$Quality)

#train-test split
training.samples1 <- wine$Quality %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data1  <- wine[training.samples1, ]
test.data1 <- wine[-training.samples1, ]
#----

#ORDINAL LOGISTIC REGRESSION
#----
#saturated model
model1 <- polr(Quality ~ ., data = train.data1, Hess=TRUE)
summary(model1)
summary_table <- coef(summary(model1))
pval <- pnorm(abs(summary_table[, "t value"]),lower.tail = FALSE)* 2
summary_table <- cbind(summary_table, "p value" = round(pval,3))
summary_table

#predictions
predictions <- round(predict(model1,test.data1,type = "p"), 3)

#test for proportional odds assumption
#brant test (check to see if the change in the independent variables is constant between steps in the dependent variables)
brant(model1)
#multiple issues, could it be bc of the high number of levels (3-9)? Would it be an issue if levels were just low-mid-high?

#cross validation
ctrl <- trainControl(method = "cv", number = 5)

cv_model <- train(
  Quality ~ . - ColorID, 
  data = wine, 
  method = "polr",
  trControl = ctrl
)

cv_model
#using method = logistic, ~ 54% accuracy

#Feature selection
#using stepwise AIC
step_model <- stepAIC(model1, direction = "both")
summary(step_model)

brant(step_model)

#cross validation
num_folds <- 5
folds <- createFolds(wine$Quality, k = num_folds)

cv_results <- lapply(folds, function(fold_indices) {
  train_fold <- wine[-fold_indices, ]
  valid_fold <- wine[fold_indices, ]
  cv_model1 <- polr(Quality ~ . - ColorID, data = train_fold, Hess = TRUE)
  step_model1 <- stepAIC(cv_model1, direction = "both", trace = 0)
  predictions1 <- predict(step_model1, newdata = valid_fold, type = "class")
  accuracy <- sum(predictions1 == valid_fold$Quality) / length(valid_fold$Quality)
  return(accuracy)
})

average_accuracy <- mean(unlist(cv_results))
average_accuracy
#~54% total accuracy
#stepwise did not improve accuracy
#----

#RANDOM FOREST
#----
winerf <- randomForest(Quality ~ .- ColorID, data=train.data1, proximity=TRUE)
#winerf

winep2 <- predict(winerf, test.data1)
confusionMatrix(winep2, test.data1$Quality)

#Optimal RF
mtry_values1 <- c(seq(2,5))
ntree_values1 <- c(seq(100, 1000, by = 100))

#initialize variables
best_mtry1 <- NULL
best_ntree1 <- NULL
best_oob_error1 <- Inf

#optimize mtry and ntree
results_df1 <- data.frame(mtry1 = integer(0), ntree1 = integer(0), oob_error1 = double(0))

for (mtry1 in mtry_values1) {
  for (ntree1 in ntree_values1) {
    # Train a random forest model with the current mtry and ntree values
    rf_model1 <- randomForest(Quality ~ .-ColorID, data = train.data1, mtry = mtry1, ntree = ntree1)
    
    # Calculate out-of-bag error
    oob_error1 <- rf_model1$err.rate[nrow(rf_model1$err.rate), 1]
    
    # Check if this combination results in a lower OOB error
    if (oob_error1 < best_oob_error1) {
      best_mtry1 <- mtry1
      best_ntree1 <- ntree1
      best_oob_error1 <- oob_error1
    }
    
    # Add results to the data frame
    results_df1 <- rbind(results_df1, data.frame(mtry1 = mtry1, ntree1 = ntree1, oob_error1 = oob_error1))
  }
}

best_mtry1
best_ntree1

#scatter plot of ntree vs. mtry vs. OOB error rate
ggplot(results_df1, aes(x = mtry1, y = ntree1, color = oob_error1)) +
  geom_point() +
  scale_color_gradient(low = "black", high = "white") +
  labs(x = "mtry", y = "ntree", color = "OOB Error") +
  theme_minimal()
#can use the simplest amount of mtry + ntree combo -- simplest combo that is dark on the chart

#train model
final_rf1 <- randomForest(Quality ~ ., data = train.data1, mtry = best_mtry1, ntree = best_ntree1, proximity = TRUE)

predictionsrf <- predict(final_rf1, newdata = test.data1)

#confusion matrix
conf_matrix <- confusionMatrix(predictionsrf, test.data1$Quality)
accuracyrf <- conf_matrix$overall["Accuracy"]
accuracyrf
#the random forests do a better job at predicting, but at the cost of interpretability
#----

#XGBOOST
#----
wine3 <- wine
wine3$Quality <- ordered(wine3$Quality)
wine3$ColorID <- as.integer(wine3$ColorID)

set.seed(123)
training_samples <- createDataPartition(wine3$Quality, p = 0.7, list = FALSE)
train_data <- wine3[training_samples, ]
test_data <- wine3[-training_samples, ]

wine3$Color <- NULL
train_data$Quality <- as.numeric(train_data$Quality)
test_data$Quality <- as.numeric(test_data$Quality)

dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_data$Quality)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

xgb_model <- xgboost(
  data = dtrain,
  params = params,
  nrounds = 100,
  print_every_n = 10,
  verbose = 1
)

#predictions (note: not using cross validation like other models)
xgpredictions <- predict(xgb_model, newdata = xgb.DMatrix(data = as.matrix(test_data)))
xgpredictions <- xgpredictions + 2
test_data$Quality <- test_data$Quality + 2

predicted_qualities <- round(xgpredictions)
xgaccuracy <- sum(predicted_qualities == test_data$Quality) / length(test_data$Quality)
xgaccuracy
#most likely overfit 
#or I did something wrong, which is also pretty likely
#

#----

#GROUP BY LOW-MED-HIGH
#----
#rather than trying to predict each specific quality on a numeric scale, what if we grouped the wine qualities into distinct buckets?
#Low = Quality 1 - 5
#Mid = Quality 6, 7
#High = Quality 8, 9, 10

#preprocessing
wine2 <- wine
wine2$Quality <- as.numeric(wine2$Quality)
cwine2$Quality <- wine2$Quality + 2
as.data.frame(table(wine2$Quality))
wine2$NewQuality <- with(wine2, ifelse(Quality >= 8, 'High',
                                       ifelse(Quality >= 6 , 'Mid', 'Low')))
wine2$NewQuality <- factor(wine2$NewQuality, order = TRUE, 
                                    levels = c("Low", "Mid", "High"))
wine2$Quality <- NULL
wine2$NewQuality <- as.factor(wine2$NewQuality)

training.samples2 <- wine2$NewQuality %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data2  <- wine2[training.samples2, ]
test.data2 <- wine2[-training.samples2, ]

#ordinal model
model2 <- polr(NewQuality ~ .- ColorID, data = train.data2, Hess=TRUE)
summary(model2)
summary_table2 <- coef(summary(model2))
pval2 <- pnorm(abs(summary_table2[, "t value"]),lower.tail = FALSE)* 2
summary_table2 <- cbind(summary_table2, "p value" = round(pval2,3))
summary_table2

#predictions
predictions2 <- round(predict(model2,test.data2,type = "p"), 3)
predictions2[1,]

#interpretations for the first person in the test set
test.data1[1,]
#low
-128.7379 -((1.393e-01*7.8)+(-4.491e+00*0.88)+(-5.361e-01*0)+(1.181e-01*2.6)+(-7.033e-01*.098)+(1.540e-02*25)+(-6.314e-03*67)+(-1.407e+02*.9968)+(9.928e-01*3.2)+(1.876e+00*.68)+(7.998e-01*9.8)+(-3.403e-01*0))
exp(1.886621)/(1+exp(1.886621))
#p(low) = .868

#mid
-123.7421 -((1.393e-01*7.8)+(-4.491e+00*0.88)+(-5.361e-01*0)+(1.181e-01*2.6)+(-7.033e-01*.098)+(1.540e-02*25)+(-6.314e-03*67)+(-1.407e+02*.9968)+(9.928e-01*3.2)+(1.876e+00*.68)+(7.998e-01*9.8)+(-3.403e-01*0))
exp(6.882421)/(1+exp(6.882421)) - exp(1.886621)/(1+exp(1.886621))
#p(mid) = .131

#high
1 - exp(1.886621)/(1+exp(1.886621)) - (exp(6.882421)/(1+exp(6.882421)) - exp(1.886621)/(1+exp(1.886621)))
#p(high) = .001

#model diagnostics
brant(model2)
#less issues 
#maybe issues stemmed from breaking down the levels too far

#cross validation
ctrl2 <- trainControl(method = "cv", number = 5)

cv_model2 <- train(
  NewQuality ~ . - ColorID, 
  data = wine2, 
  method = "polr",
  trControl = ctrl2
)

cv_model2
#using method = logistic, ~ 71% accuracy

#feature selection
#---
step_model2 <- stepAIC(model2, direction = "both")
summary(step_model2)

#cross validation
folds2 <- createFolds(wine2$NewQuality, k = 5)

cv_results2 <- lapply(folds2, function(fold_indices2) {
  train_fold2 <- wine2[-fold_indices2, ]
  valid_fold2 <- wine2[fold_indices2, ]
  
  cv_model2 <- polr(NewQuality ~ ., data = train_fold2, Hess = TRUE)
  step_model2 <- stepAIC(cv_model2, direction = "both", trace = 0)
  
  predictions3 <- predict(step_model2, newdata = valid_fold2, type = "class")
  
  # Convert predictions to ordered factor with correct levels
  predictions3 <- factor(predictions3, ordered = TRUE, levels = levels(wine2$NewQuality))
  
  accuracy2 <- sum(predictions3 == valid_fold2$NewQuality) / length(valid_fold2$NewQuality)
  return(accuracy2)
})

average_accuracy2 <- mean(unlist(cv_results2))
average_accuracy2
#~71% accuracy
#---

#random forest
winerf2 <- randomForest(NewQuality ~ .- ColorID, data=train.data2, proximity=TRUE)
#winerf2

winep22 <- predict(winerf2, test.data2)
confusionMatrix(winep22, test.data2$NewQuality)
#~81 accuracy
#----

#RED VS WHITE

#WHITE WINE
#----
#data prep
WhiteWine <- subset(wine, Color == "White")
head(WhiteWine,1)
dim(WhiteWine)
t(t(sapply(WhiteWine, class)))
WhiteWine$ColorID <- NULL
WhiteWine$Color <- NULL
dim(WhiteWine)

#ordinal regression
#saturated model
whitemodel1 <- polr(Quality ~ ., data = WhiteWine, Hess=TRUE)
summary(whitemodel1)
whitesummary_table <- coef(summary(whitemodel1))
whitepval <- pnorm(abs(whitesummary_table[, "t value"]),lower.tail = FALSE)* 2
whitesummary_table <- cbind(whitesummary_table, "p value" = round(whitepval,3))
whitesummary_table

#brant test (check to see if the change in the independent variables is constant between steps in the dependent variables)
brant(whitemodel1)
#multiple issues

#cross validation
whitectrl <- trainControl(method = "cv", number = 5)

white_cv_model <- train(
  Quality ~ ., 
  data = WhiteWine, 
  method = "polr",
  trControl = whitectrl
)

white_cv_model
#----

#RED WINE
#----
RedWine <- subset(wine, Color == "Red")
head(RedWine,1)
dim(RedWine)
t(t(sapply(RedWine, class)))
RedWine$ColorID <- NULL
RedWine$Color <- NULL
dim(RedWine)
as.data.frame(table(RedWine$Quality))
#remove the unused factor level
RedWine$Quality <- droplevels(RedWine$Quality)
as.data.frame(table(RedWine$Quality))

#ordinal regression
#saturated model
Redmodel1 <- polr(Quality ~ ., data = RedWine, Hess=TRUE)
summary(Redmodel1)
Redsummary_table <- coef(summary(Redmodel1))
Redpval <- pnorm(abs(Redsummary_table[, "t value"]),lower.tail = FALSE)* 2
Redsummary_table <- cbind(Redsummary_table, "p value" = round(Redpval,3))
Redsummary_table

#brant test (check to see if the change in the independent variables is constant between steps in the dependent variables)
brant(Redmodel1)
#multiple issues

#cross validation
Redctrl <- trainControl(method = "cv", number = 5)

Red_cv_model <- train(
  Quality ~ ., 
  data = RedWine, 
  method = "polr",
  trControl = Redctrl
)

Red_cv_model
#the saturated model for red wine predicts at a higher accuracy than white wine
#----
