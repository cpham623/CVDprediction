import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
from sklearn.metrics import accuracy_score, precision_score, recall_score

from preprocess_new import (
    load_data, construct_target, drop_leaky_features,
    encode_categoricals, build_preprocessor
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import numpy as np

def prepare_raw(path):
    df = load_data(path)
    df = construct_target(df)
    y = df["CVD"]  # Construct target early
    df = drop_leaky_features(df)
    df = encode_categoricals(df, y=y, is_training=True)  # Pass y
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
        y_score = clf.predict_proba(X_te)[:, 1]
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
        y_score_t = (clf.predict_proba(X_t)[:, 1]
                     if hasattr(clf, 'predict_proba')
                     else clf.decision_function(X_t))
        print(f"{grp}: AUROC={roc_auc_score(y_t, y_score_t):.3f}  F1={f1_score(y_t, y_pred_t):.3f}")
    print()

def report_top_features(pipe, original_X):
    try:
        clf = pipe.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            scores = clf.feature_importances_
            names = original_X.columns
        elif hasattr(clf, 'coef_'):
            scores = np.abs(clf.coef_[0])
            names = original_X.columns
        else:
            print("No importances available for this model.")
            return

        sorted_feats = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        print("Top Predictive Features:")
        for name, score in sorted_feats[:20]:
            print(f"{name:40s}: {score:.4f}")

        top_feats = sorted_feats[:20][::-1]
        names, values = zip(*top_feats)
        plt.figure(figsize=(10, 6))
        plt.barh(names, values)
        plt.xlabel("Importance Score")
        plt.title("Top 20 Predictive Features for CVD")
        plt.tight_layout()
        plt.savefig("top_predictive_features.png", dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Error extracting features:", e)

def save_performance_table(model_scores, filename="model_performance.png"):
    df = pd.DataFrame(model_scores).set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 2 + len(df) * 0.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc='center'
    )
    tbl.scale(1, 1.5)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_all():
    X_tr, X_te, y_tr, y_te, df_te = prepare_raw('../data/heart_2022_no_nans.csv')
    preprocessor = build_preprocessor()

    model_defs = {
        'LogisticRegression': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            {
                'clf__C': [0.01, 0.1, 1, 10]
            }
        )
    }

    roc_curves = []

    for model_name, (estimator, param_grid) in model_defs.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', estimator)
        ])

        gs = run_grid_search(pipe, param_grid, X_tr, y_tr, model_name)
        best = gs.best_estimator_
        report_performance(model_name, best, X_te, y_te, df_te)
        if model_name == 'LogisticRegression':
            report_top_features(best, X_tr)

        if hasattr(best, 'predict_proba'):
            y_score = best.predict_proba(X_te)[:, 1]
        else:
            y_score = best.decision_function(X_te)
        fpr, tpr, _ = roc_curve(y_te, y_score)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((model_name, fpr, tpr, roc_auc, y_score))

    print("\nFitting untuned baseline models (no cross-validation)...")
    baseline_models = {
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': CalibratedClassifierCV(
            estimator=LinearSVC(class_weight='balanced', max_iter=5000),
            cv=5
        )
    }

    for model_name, model in baseline_models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', model)
        ])
        pipe.fit(X_tr, y_tr)
        if hasattr(pipe, 'predict_proba'):
            y_score = pipe.predict_proba(X_te)[:, 1]
        else:
            y_score = pipe.decision_function(X_te)
        fpr, tpr, _ = roc_curve(y_te, y_score)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((model_name, fpr, tpr, roc_auc, y_score))

    # === Model Scores ===
    model_scores = []

    for name, fpr, tpr, roc_auc, y_score in roc_curves:
        y_pred = (y_score > 0.5).astype(int)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        acc = accuracy_score(y_te, y_pred)
        ap = average_precision_score(y_te, y_score)
        model_scores.append({
            "Model": name,
            "Accuracy": f"{acc:.2f}",
            "Precision": f"{prec:.2f}",
            "Recall": f"{rec:.2f}",
            "F1 Score": f"{f1:.2f}",
            "AUROC": f"{roc_auc:.2f}",
            "AUPRC": f"{ap:.2f}"
        })

    save_performance_table(model_scores)

    # === Final ROC plot ===
    plt.figure(figsize=(10, 7))
    for name, fpr, tpr, roc_auc, _ in roc_curves:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_all_models.png", dpi=300, bbox_inches='tight')
    plt.show()

    # === Final AUPRC plot ===
    plt.figure(figsize=(10, 7))
    for name, _, _, _, y_score in roc_curves:
        precision, recall, _ = precision_recall_curve(y_te, y_score)
        ap = average_precision_score(y_te, y_score)
        plt.plot(recall, precision, label=f"{name} (AUPRC = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for All Models")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("auprc_curve_all_models.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_all()
