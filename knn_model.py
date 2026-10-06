import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": "KNN",
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    return model, metrics

def save_knn(model, scaler, path="knn_bundle.pkl"):
    joblib.dump({"model": model, "scaler": scaler}, path)
