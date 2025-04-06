from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_pipeline
from evaluate_model import evaluate_model

def train_logistic(data_path):
    # preprocess, get data split
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # evaluate
    print("=== Logistic Regression Results ===")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_logistic("../data/heart_2022_no_nans.csv")

