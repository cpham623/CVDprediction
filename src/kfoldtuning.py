from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from preprocess_new import build_preprocessor, build_filter_selector, build_wrapper_selector, build_embedded_selector, load_data, construct_target, drop_leaky_features, encode_categoricals
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import matplotlib.pyplot as plt


def prepare_kfold_data(path):
    df = load_data(path)
    df = construct_target(df)
    df = drop_leaky_features(df)
    df = encode_categoricals(df)
    X = df.drop('CVD', axis=1)
    y = df['CVD']
    return X, y, df


def create_model_pipeline(selector_type, model):
    preprocessor = build_preprocessor()
    if selector_type == 'filter':
        selector = build_filter_selector(k=50)
    elif selector_type == 'wrapper':
        selector = build_wrapper_selector(n_features=50)
    else:
        selector = build_embedded_selector()

    return Pipeline([
        ('prep', preprocessor),
        ('sel', selector),
        ('clf', model)
    ])


def kfold_tune_and_evaluate(X, y, model_name, selector_type, param_grid):
    if model_name == 'LogisticRegression':
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'SVM':
        base_svm = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
        model = CalibratedClassifierCV(base_svm, cv=3)
    else:
        raise ValueError("Unknown model name")

    pipe = create_model_pipeline(selector_type, model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)
    gs.fit(X, y)

    print("Best Parameters:", gs.best_params_)
    return gs.best_estimator_


if __name__ == '__main__':
    X, y, df = prepare_kfold_data('../data/heart_2022_no_nans.csv')

    param_grid_logistic = {
        'sel__mi__k': [10, 20, 30, 40],
        'clf__C': [0.01, 0.1, 1, 10]
    }

    best_model = kfold_tune_and_evaluate(
        X, y,
        model_name='LogisticRegression',
        selector_type='filter',
        param_grid=param_grid_logistic
    )
