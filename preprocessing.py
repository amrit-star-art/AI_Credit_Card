import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    "annual_fee", "minimum_annual_income", "credit_score_requirement",
    "cashback_rate", "reward_rate", "travel_reward", "dining_reward",
    "online_reward", "fuel_reward", "lounge_access", "lounge_visits"
]

USER_FEATURES = [
    "annual_income", "credit_score", "annual_spending",
    "travel_spend", "dining_spend", "fuel_spend", "online_spend",
    "max_annual_fee", "lounge_importance"
]

def load_cards(path="verified_cards.csv"):
    df = pd.read_csv(path)
    return df

def clean_cards(df):
    df = df.copy()
    for c in BASE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def card_feature_matrix(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[BASE_FEATURES])
    return X, scaler
