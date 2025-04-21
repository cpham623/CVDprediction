from sklearn.svm import SVC
from sklearn.metrics import classification_report


def train_svm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate an SVM model on preprocessed arrays.
    """
    clf = SVC(
        kernel='rbf',           # radial basis function kernel
        class_weight='balanced',# address class imbalance
        probability=True,       # enable predict_proba
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("=== SVM Results ===")
    print(classification_report(y_test, preds, zero_division=0))
    return clf


if __name__ == "__main__":
    # Example standalone usage; uses same prepare_data pipeline
    from preprocess_new import prepare_data
    X_tr, X_te, y_tr, y_te, _, _ = prepare_data(
        '../data/heart_2022_no_nans.csv',
        selector_type='filter'
    )
    train_svm(X_tr, X_te, y_tr, y_te)

