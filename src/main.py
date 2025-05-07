import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocess_new import prepare_data
from logistic_regression import train_logistic
from svm import train_svm
from random_forest import train_random_forest
from gradient_boosting import train_gradient_boosting
from kfoldtuning import prepare_kfold_data, create_model_pipeline
from evaluate_model import evaluate_model
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import tune_and_evaluate

# Path and config
DATA_PATH = "../data/heart_2022_no_nans.csv"
SELECTOR_TYPE = 'filter'  # can be 'filter', 'wrapper', or 'embedded'

def plot_evaluation(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - {title}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {title}")
    plt.grid(False)
    plt.show()

# Preprocess and split data for evaluation
X_train, X_test, y_train, y_test, _ = prepare_data(path=DATA_PATH)


# === RUN ALL FOUR MODELS ===
print("\n================ Logistic Regression ================")
logistic_model = train_logistic(X_train, X_test, y_train, y_test)
evaluate_model(logistic_model, X_test, y_test)
# plot_evaluation(logistic_model, X_test, y_test, "Logistic Regression")

print("\n================ SVM ================")
svm_model = train_svm(X_train, X_test, y_train, y_test)
evaluate_model(svm_model, X_test, y_test)
# plot_evaluation(svm_model, X_test, y_test, "SVM")

print("\n================ Random Forest ================")
rf_model = train_random_forest(X_train, X_test, y_train, y_test)
evaluate_model(rf_model, X_test, y_test)
# plot_evaluation(rf_model, X_test, y_test, "Random Forest")

print("\n================ Gradient Boosting ================")
gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)
evaluate_model(gb_model, X_test, y_test)
# plot_evaluation(gb_model, X_test, y_test, "Gradient Boosting")

# === RUN TUNED MODEL PIPELINES WITH FEATURE IMPORTANCE ===
import tune_and_evaluate
print("\n================ Running tune_and_evaluate pipelines ================")
tune_and_evaluate.run_all()
