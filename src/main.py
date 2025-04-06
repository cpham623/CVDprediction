from logistic_regression import train_logistic
from svm import train_svm
from random_forest import train_random_forest
from gradient_boosting import train_gradient_boosting

train_logistic("../data/heart_2022_no_nans.csv")
train_svm("../data/heart_2022_no_nans.csv")
train_random_forest("../data/heart_2022_no_nans.csv")
train_gradient_boosting("../data/heart_2022_no_nans.csv")