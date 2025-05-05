from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a gradient boosting model on preprocessed arrays.
    """
    clf = GradientBoostingClassifier(
        n_estimators=100,       # number of boosting stages
        learning_rate=0.1,      # learning rate shrinks contribution of each tree
        max_depth=3,            # maximum tree depth
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("=== Gradient Boosting Results ===")
    print(classification_report(y_test, preds, zero_division=0))
    return clf


if __name__ == "__main__":
    # Example standalone usage; uses same prepare_data pipeline
    from preprocess_new import prepare_data
    X_tr, X_te, y_tr, y_te, _, _ = prepare_data(
        '../data/heart_2022_no_nans.csv',
        selector_type='filter'
    )
    train_gradient_boosting(X_tr, X_te, y_tr, y_te)

