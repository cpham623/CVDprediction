from sklearn.svm import SVC
from preprocess import preprocess_pipeline
from evaluate_model import evaluate_model

def train_svm(data_path):
    # preprocess, get data split
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    # train model
    model = SVC(kernel='rbf', probability=True, random_state=42)    # TO ADJUST?
    model.fit(X_train, y_train)

    # evaluate
    print("=== Support Vector Machine (SVM) Results ===")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train_svm("../data/heart_2022_no_nans.csv.csv")
