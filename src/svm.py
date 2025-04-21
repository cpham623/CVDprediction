from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

def train_svm(X_train, X_test, y_train, y_test):
    base = LinearSVC(
        class_weight='balanced',
        max_iter=5000,
        random_state=42
    )
    # 3‑fold calibration for probabilities
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    # for AUROC, use clf.predict_proba(X_test)[:,1]
    print("=== SVM (linear + calibrated) Results ===")
    print(classification_report(y_test, preds, zero_division=0))
    return clf


