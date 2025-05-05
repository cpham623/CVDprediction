import pandas as pd
import matplotlib.pyplot as plt

from preprocess_new import (
    load_data, construct_target, drop_leaky_features,
    encode_categoricals, build_preprocessor,
    build_filter_selector, build_wrapper_selector,
    build_embedded_selector
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import (
    CalibratedClassifierCV, calibration_curve
)
from sklearn.metrics import roc_auc_score, f1_score, classification_report


def prepare_raw(path):
    df = load_data(path)
    df = construct_target(df)
    df = drop_leaky_features(df)
    df = encode_categoricals(df)
    X = df.drop('CVD', axis=1)
    y = df['CVD']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_tr, X_te, y_tr, y_te, df.loc[y_te.index]


def run_grid_search(pipe, param_grid, X_tr, y_tr, model_name):
    print(f"Starting grid search for {model_name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, param_grid,
        scoring='roc_auc', cv=cv,
        n_jobs=-1, refit=True, verbose=2
    )
    gs.fit(X_tr, y_tr)
    print(f"Completed grid search for {model_name}. Best params: {gs.best_params_}\n")
    return gs


def report_performance(name, clf, X_te, y_te, df_te):
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_te)[:,1]
    else:
        y_score = clf.decision_function(X_te)
    y_pred = clf.predict(X_te)
    print(f"--- {name} ---")
    print(classification_report(y_te, y_pred, zero_division=0))
    print(f"Overall AUROC: {roc_auc_score(y_te, y_score):.3f}\n")
    for grp, sub in df_te.groupby('RaceEthnicityCategory'):
        idx = sub.index
        if len(idx) < 50: continue
        y_t = y_te.loc[idx]
        X_t = X_te.loc[idx]
        y_pred_t = clf.predict(X_t)
        y_score_t = (clf.predict_proba(X_t)[:,1]
                     if hasattr(clf, 'predict_proba')
                     else clf.decision_function(X_t))
        print(f"{grp}: AUROC={roc_auc_score(y_t, y_score_t):.3f}  F1={f1_score(y_t, y_pred_t):.3f}")
    print()


def plot_calibration(name, clf, X_te, y_te):
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_te)[:,1]
    else:
        probs = clf.decision_function(X_te)
    frac_pos, mean_pred = calibration_curve(y_te, probs, n_bins=10)
    plt.figure()
    plt.plot(mean_pred, frac_pos, 'o-', label=name)
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed frequency')
    plt.title(f'Calibration curve: {name}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X_tr, X_te, y_tr, y_te, df_te = prepare_raw('../data/heart_2022_no_nans.csv')
    preprocessor = build_preprocessor()

    selectors = {
        'filter': build_filter_selector(k=50),
        'wrapper': build_wrapper_selector(n_features=50),
        'embedded': build_embedded_selector()
    }

    model_defs = {
        'LogisticRegression': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            {
                'sel__mi__k': [10,20,30,40],
                'clf__C': [0.01,0.1,1,10]
            }
        ),
        'RandomForest': (
            RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            {
                'sel__mi__k': [10,20,30,40],
                'clf__n_estimators': [100,200,500],
                'clf__max_depth': [None,5,10,20]
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'sel__mi__k': [10,20,30,40],
                'clf__n_estimators': [100,200,500],
                'clf__learning_rate': [0.05,0.1],
                'clf__max_depth': [3,5,10]
            }
        ),
        'SVM': (
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
                cv=5
            ),
            {
                'sel__mi__k': [10,20,30,40],
                'clf__estimator__C': [0.01,0.1,1,10]
            }
        )
    }

    for sel_name, selector in selectors.items():
        for model_name, (estimator, param_grid) in model_defs.items():
            full_name = f"{model_name} + {sel_name}"
            print(f"\n=== Pipeline: {full_name} ===")
            pipe = Pipeline([
                ('prep', preprocessor),
                ('sel', selector),
                ('clf', estimator)
            ])
            gs = run_grid_search(pipe, param_grid, X_tr, y_tr, full_name)
            best = gs.best_estimator_
            report_performance(full_name, best, X_te, y_te, df_te)
            plot_calibration(full_name, best, X_te, y_te)




