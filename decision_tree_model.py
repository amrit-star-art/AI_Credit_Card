import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_decision_tree(X_train, X_test, y_train, y_test, random_state=42):
    model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": "Decision Tree",
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    return model, metrics

def save_tree(model, path="decision_tree.pkl"):
    joblib.dump(model, path)
