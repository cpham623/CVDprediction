from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a random forest model on preprocessed arrays.
    """
    clf = RandomForestClassifier(
        n_estimators=100,       # number of trees in the forest
        max_depth=None,         # no maximum depth
        class_weight='balanced',# handle class imbalance
        random_state=42,
        n_jobs=-1               # use all cores
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("=== Random Forest Results ===")
    print(classification_report(y_test, preds, zero_division=0))
    return clf


if __name__ == "__main__":
    # Example standalone usage; uses same prepare_data pipeline
    from preprocess_new import prepare_data
    X_tr, X_te, y_tr, y_te, _, _ = prepare_data(
        '../data/heart_2022_no_nans.csv',
        selector_type='filter'
    )
    train_random_forest(X_tr, X_te, y_tr, y_te)
