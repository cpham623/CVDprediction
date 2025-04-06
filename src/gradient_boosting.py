from sklearn.ensemble import GradientBoostingClassifier
from preprocess import preprocess_pipeline
from evaluate_model import evaluate_model

def train_gradient_boosting(data_path):
    # preprocess, get data split
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    # train model
    model = GradientBoostingClassifier(
        n_estimators=100,       # num boosting stages (all values: TO ADJUST?)
        learning_rate=0.1,      # each tree's contribution
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # evaluate
    print("=== Gradient Boosting Results ===")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_gradient_boosting("../data/heart_2022_no_nans.csv")
