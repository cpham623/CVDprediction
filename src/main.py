import pandas as pd
import numpy as np
from logistic_regression import train_logistic
from svm import train_svm
from random_forest import train_random_forest
from gradient_boosting import train_gradient_boosting

train_logistic("../data/heart_2022_no_nans.csv")
# train_svm("../data/heart_2022_no_nans.csv")
# train_random_forest("../data/heart_2022_no_nans.csv")
# train_gradient_boosting("../data/heart_2022_no_nans.csv")

# checking for NaNs
# df = pd.read_csv("../data/heart_2022_no_nans.csv")
# print(df.isna().sum())
#
# columns_with_nans = df.columns[df.isna().any()].tolist()
# print(columns_with_nans)
