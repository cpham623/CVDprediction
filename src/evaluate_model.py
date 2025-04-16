from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # print evaluation metrics
    print(confusion_matrix(y_test, y_pred))
    print("y_pred distribution:\n", pd.Series(y_pred).value_counts())
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))