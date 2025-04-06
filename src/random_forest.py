from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_pipeline
from evaluate_model import evaluate_model

def train_random_forest(data_path):
    # preprocess, get data split
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    # train model
    model = RandomForestClassifier(
        n_estimators=100,       # num trees (TO ADJUST?)
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # evaluate
    print("=== Random Forest Results ===")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_random_forest("../data/heart_2022_no_nans.csv")
