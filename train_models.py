import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn_model import train_knn, save_knn
from decision_tree_model import train_decision_tree, save_tree

FEATURES = [
    "annual_fee", "minimum_annual_income", "credit_score_requirement",
    "cashback_rate", "reward_rate", "travel_reward", "dining_reward",
    "online_reward", "fuel_reward", "lounge_access", "lounge_visits"
]
TARGET = "card_category"

def main():
    df = pd.read_csv("verified_cards.csv")
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(str)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.25, random_state=42, stratify=y
    )

    knn, km = train_knn(X_train, X_test, y_train, y_test, n_neighbors=3)
    tree, tm = train_decision_tree(X_train, X_test, y_train, y_test)

    save_knn(knn, scaler)
    save_tree(tree)

    pd.DataFrame([km, tm]).to_csv("model_metrics.csv", index=False)
    print(pd.DataFrame([km, tm]).round(4))
    print("Saved knn_bundle.pkl, decision_tree.pkl, model_metrics.csv")

if __name__ == "__main__":
    main()
