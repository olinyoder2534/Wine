
library(MASS)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)
library(brant)

#LOADING DATA
#----
wineQuality <- read.csv('FilePath.csv')
wineLookup <- read.csv('FilePath.csv')

dim(wineQuality)

head(wineLookup)
head(wineQuality, 1)

#Join data
#not necessary/redundant
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
#visualizations and basic data exploration have been done elsewhere
#no major transformations are needed

#check for multicollinearity
#GGally::ggpairs(wine)
#nothing obvious signs

#recode variables
wine$ColorID <- as.factor(wine$ColorID)
wine$Quality <- as.factor(wine$Quality)

#train-test split
training.samples1 <- wine$Quality %>% 
  createDataPartition(p = 0.7, list = FALSE)
train.data1  <- wine[training.samples1, ]
test.data1 <- wine[-training.samples1, ]
#----

#ORDINAL REGRESSION
#----
#Simple model
model1 <- polr(Quality ~ .- ColorID, data = train.data1, Hess=TRUE)
summary(model1)
summary_table <- coef(summary(model1))
pval <- pnorm(abs(summary_table[, "t value"]),lower.tail = FALSE)* 2
summary_table <- cbind(summary_table, "p value" = round(pval,3))
summary_table

#predictions
predictions <- round(predict(model1,test.data1,type = "p"), 3)

#brant test (check to see if the change in the independent variables is constant between steps in the dependent variables)
brant(model1)
#multiple issues

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

#cross validation
num_folds <- 5
folds <- createFolds(wine$Quality, k = num_folds)

cv_results <- lapply(folds, function(fold_indices) {
  train_fold <- wine[-fold_indices, ]
  valid_fold <- wine[fold_indices, ]
  cv_model <- polr(Quality ~ . - ColorID, data = train_fold, Hess = TRUE)
  predictions <- predict(cv_model, newdata = valid_fold, type = "class")
  accuracy <- sum(predictions == valid_fold$Quality) / length(valid_fold$Quality)
  return(accuracy)
})

average_accuracy <- mean(unlist(cv_results))
average_accuracy
#~54% total accuracy
#stepwise did not improve accuracy nor take care of any issues with the proportional odds assumption
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

predictions <- predict(final_rf1, newdata = test.data1)

#confusion matrix
conf_matrix <- confusionMatrix(predictions, test.data1$Quality)
accuracyrf <- conf_matrix$overall["Accuracy"]
accuracyrf
#the random forests do a better job at predicting, but at the cost of interpretability
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
as.data.frame(table(wine2$Quality))
wine2$Quality <- wine2$Quality + 2
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

#predcitions
predictions2 <- round(predict(model2,test.data2,type = "p"), 3)
predictions2[1,]

test.data1[1,]
#interpretations for the first person in the test set
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

#random forest
winerf2 <- randomForest(NewQuality ~ .- ColorID, data=train.data2, proximity=TRUE)
#winerf

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
#simple model
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
#simple model
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
#the saturated model is better at predicting red wine than white wine

