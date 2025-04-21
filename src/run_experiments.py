import numpy as np
from preprocess_new import prepare_data
from logistic_regression import train_logistic
from random_forest import train_random_forest
from svm import train_svm
from gradient_boosting import train_gradient_boosting

def evaluate_model(clf, X_test, y_test):
    from sklearn.metrics import roc_auc_score, f1_score
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    print(f"    AUROC: {roc_auc_score(y_test, y_score):.3f}   F1: {f1_score(y_test, y_pred):.3f}")

def run_pipeline(name, X_tr, X_te, y_tr, y_te):
    print(f"\n===== {name} =====")
    print("• LogisticRegression")
    clf = train_logistic(X_tr, X_te, y_tr, y_te)
    evaluate_model(clf, X_te, y_te)

    print("• RandomForest")
    clf = train_random_forest(X_tr, X_te, y_tr, y_te)
    evaluate_model(clf, X_te, y_te)

    print("• SVM")
    clf = train_svm(X_tr, X_te, y_tr, y_te)
    evaluate_model(clf, X_te, y_te)

    print("• GradientBoosting")
    clf = train_gradient_boosting(X_tr, X_te, y_tr, y_te)
    evaluate_model(clf, X_te, y_te)

if __name__ == "__main__":
    # run each feature-selection strategy
    for sel in ['filter','wrapper','embedded']:
        X_tr, X_te, y_tr, y_te, preproc, selector = prepare_data(
            path="data/heart_2022_no_nans.csv",
            k_filter=50,
            k_wrapper=30,
            selector_type=sel
        )
        run_pipeline(f"Selector: {sel}", X_tr, X_te, y_tr, y_te)

