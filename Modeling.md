## Modeling

### Question
Can we predict the quality of a wine based on its attributes?

### Preprocessing
Since the response variable (_Quality_ ) is ordinal, we took two approaches:​
 1) Regression-based models​
 2) Classification based models with _Quality_ bucketed into three groups
   * Low (<= 5)
   * Mid (6,7)
   * High (>= 8)
      * Buckets were formed using a net-promoter-score based approach. Since there are no observations of where  _Quality_  = 10, ranges for buckets were shifted down one.
     
![CountByQualityCategory](https://github.com/olinyoder2534/Wine/blob/main/CountByQualityCategory.png)
           
### Regression
Before training a model, additional preprocessing was needed. First, we removed the unnecessary, duplicate variables, which included  _ColorID_ and  _QualityCategory_. We elected to not perform any pre-modeling feature selection, but would later incorporate models using feature-reduction techniques. Lastly, rather than using a simple train-test split, we elected to use 5-fold cross-validation and RMSE as a comparison metric. 

---

 **1. OLS** 
 
RMSE: 0.735 (± 0.021)

Formula (with scales variables): 
```math
Quality = 6.0908 - 0.2457(VolatileAcidity) - 0.0793(TotalSulfurDioxide) + 0.1074(Sulphates) + 0.2971(ResidualSugar) + 0.0802(pH) + 0.0876(FreeSulfurDioxide) + 0.1103(FixedAcidity) - 0.3116(Density) - 0.3613(ColorWhite) - 0.0091(CitricAcid) - 0.0265(Chlorides) + 0.2656(Alcohol)
```

![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_OLS_R.png)

Error Distribution
![Error Distribution](https://github.com/olinyoder2534/Wine/blob/main/ED_OLS_R.png)

 _Density_ ,  _ResidualSugar_ , and  _Alcohol_  account for roughly half of the feature importance in the model. Additionally, the residuals may be slightly skewed left, but generally follow an approximately normal distribution with the median at -0.091456. The skewness could be due to the chart showing  the residuals for only the first cross-validation fold. 

Can we improve prediction accuracy using L1 regularization?

---
 
**2. Lasso Regression**

RMSE: 0.762 (± 0.021)

Formula (with scales variables): 
```math
Quality = 5.8184 - 0.1313(VolatileAcidity) + 0.2934(Alcohol)
```
![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_L1_R.png)

Error Distribution
![Error Distribution](https://github.com/olinyoder2534/Wine/blob/main/ED_L1_R.png)

L1 regularization removed all variables except  _Alcohol_  and  _VolitileAcidity_ , which were the third and fourth most important features, respectively, in predicting  _Quality_  using OLS. While the OLS model had a lower RMSE, it was also much more complex. Which model to use depends on the value of simplicity.

Can we improve prediction accuracy using a tree-based approach?

---


 **3. Random Forest**
 RMSE: 0.652
 
|Hyperparamter    | Value |
| -------- | ------- |
| Number of trees | 100 |
| Max trees depth | 13 |
| Min samples per leaf | 8 |
| Min samples to split | 24 |

![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_RF_R.png)

Error Distribution
![Error Distribution](https://github.com/olinyoder2534/Wine/blob/main/ED_RF_R.png)

The random forest performs better (lower RMSE) than the linear regression based approaches, but at the cost of interpretability. Similar to the lasso regression approach, though, the random forest values  _Alcohol_  and  _VolitileAcidity_ most in predicting  _Quality_ . 

 ---

**Model Comparison**

| Model    | RMSE |
| -------- | ------- |
| OLS | 0.735 (± 0.021) |
| Lasso | 0.762 (± 0.021) |
| Random Forest | 0.652 (± 0.025) |

Of the three models, the random forest performed the best. However, for two of the models, we used a linear regression based approach on an ordinal response variable. As such, we also trained an ordinal regression model ([link](https://github.com/olinyoder2534/Wine/blob/main/Wine.R)).

Ordinal regression and classical regression often use different comparison metrics. While we could have used a ten class classification random forest, we elected to compare the models by computing RMSE for the ordinal regression model and use accuracy for the random forest. To calculate accuracy from the regression-based random forest, we randomly selected half of the original data and made predictions on it. While this does mean testing the model on data it was also trained on, the alternative was to use two train-test splits. However, this would involve significantly diminishing the size of the already small data set.

We rounded the random forest's predictions to the nearest whole number and calculated the accuracy, which was roughly 72% ([link](https://github.com/olinyoder2534/Wine/blob/main/ConvertToAccuracy.R)). Using 5-fold cross-validation, the ordinal regression model with method = logistic performed at 54% accuracy. Likewise, the RMSE for the random forest was 0.652 while RMSE for the ordinal regression model was 1.018, surprisingly worse than the OLS and lasso models. 

### Classification
Using the three classes (low, mid, high) we previously created, we were also able to perform classification. 
  
Like the regression models, before training, we first removed the unnecessary, duplicate variables, which included  _ColorID_ and  _Quality_. We also had to account for the differing number of observations in each class, so we rebalanced _QualityCategory_ to have approximately even ratios. Lastly, rather than using a simple train-test split, we elected to use 5-fold cross-validation and, to keep consistency, accuracy as a comparison metric. 

 ---
 
**1. Logistic Regression**
Accuracy: 0.568

MAUC: 0.767

![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_LR_C.png)

Confusion Matrix (first CV fold):

| Actual ↓ Predicted → | Low |Mid |High | |
| ------- | ------- |------- | ------- | ------- |
| Low |  **27**  | 8 | 1 | 36 |
| Mid | 19 |  **21**  | 13 | 53 |
| High | 2 | 3 |  **30**  | 35 |
|  | 48 | 32 |  44  | 124 |

AUC:
| Class    | AUC |
| -------- | ------- |
| Low | 0.851 |
| Mid | 0.691 |
| High | 0.885 |

The logistic regression model has an accuracy of 0.568, better than .333, which represents random guessing. Similar to the  regression models,  _Alcohol_  is among the most important variables in predicting  _Quality_ . Additionally, the model has a MAUC of 0.767, which is acceptable. In general, the model tends to be able to predict wines of _QualityCategory_ low and high with greater accuracy than mid. 

 ---

**2. Random Forest**

Accuracy: 0.682

MAUC: 0.856
 
| Hyperparamter    | Value |
| -------- | ------- |
| Number of trees | 100 |
| Max trees depth | 13 |
| Min samples per leaf | 1 |
| Min samples to split | 3 |

![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_RF_C.png)

Confusion Matrix (first CV fold):

| Actual ↓ Predicted → | Low |Mid |High | |
| ------- | ------- |------- | ------- | ------- |
| Low |  **30**  | 6 | 0 | 36 |
| Mid | 18 |  **23**  | 12 | 53  |
| High | 0 | 4 |  **31**  | 35 |
|  | 48 | 33 |  43  | 124 |

AUC:
| Class    | AUC |
| -------- | ------- |
| Low | 0.882 |
| Mid | 0.723 |
| High | 0.928 |


The random forest performs better in all metrics compared to the logistic regression model. Like all previous models, the classification random forest values  _Alcohol_  highly in predicting  _Quality_ . 

 ---
 
**3. XGBoost**
 
Accuracy: 0.623

MAUC: 0.784
 
| Hyperparamter    | Value |
| -------- | ------- |
| Actual number of trees | 5 |
| Max trees depth | 3 |
| Eta (learning rate) | 0.2 |
| Alpha (L1 regularization) | 0 |
| Lambda (L2 regularization) | 1 |
| Gamma (Min loss reduction to split a leaf) | 0 |
| Min sum of instance weight in a child | 1 |

![Feature Importance](https://github.com/olinyoder2534/Wine/blob/main/FI_XG_C.png)

Confusion Matrix (first CV fold):

| Actual ↓ Predicted → | Low |Mid |High | |
| ------- | ------- |------- | ------- | ------- |
| Low |  **28**  | 8 | 0 | 36 |
| Mid | 16 |  **18**  | 19 | 53  |
| High | 2 | 5 |  **28**  | 35 |
|  | 46 | 31 |  47  | 124 |

AUC:
| Class    | AUC |
| -------- | ------- |
| Low | 0.862 |
| Mid | 0.587 |
| High | 0.873 |


The XGBoost model values _Alcohol_ in creating its predictions more so than other models. In terms of performance, the XGBoost model performs better overall compared to the logistic regression model, but is worse at predicting mid rated wines. Compared to the random forest, the XGBoost model performs worse across the board. 

  ---
  
  **Model Comparison**

| Model    | Accuracy | MAUC |
| -------- | ------- |------- |
| Logistic Regression | 0.568 | 0.767 |
| Random Forest | 0.682 | 0.856 |
| XGBoost | 0.623 | 0.784 |

Of the three models, the random forest performed the best. Similar to our regression-based approach, we also trained an ordinal regression model using low-mid-high as the eligible responses ([link](https://github.com/olinyoder2534/Wine/blob/main/Wine.R)).

Using 5-fold cross-validation, the ordinal regression model with method = logistic performed at 74% accuracy, slightly higher than the regression random forest and all classification models. 

Surprisingly, the regression random forest model had a higher converted accuracy than the classification random forest. However, this is likely because the data used for predictions for the regression random forest was also responsible for training the model. 

### Shortcomings
* While there is ample data to train a solid model, having more observations would have open doors for more exploration
* There are limited features on Dataiku for ordinal regression

### Conclusion
Regardless of approach, for this data set, random forests were generally the best machine learning model for predicting a quality of wine. Additionally, the alcohol concentration of a wine is the most important attribute in predicting its quality.


  
