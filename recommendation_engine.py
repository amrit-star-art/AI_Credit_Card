import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reward_calculator import estimate_annual_rewards, net_benefit

CARD_FEATURES = [
    "annual_fee", "minimum_annual_income", "credit_score_requirement",
    "cashback_rate", "reward_rate", "travel_reward", "dining_reward",
    "online_reward", "fuel_reward", "lounge_access", "lounge_visits"
]

def user_vector(user, fee_scale=1.0):
    # Maps the user into the same semantic feature space as the card vectors.
    return np.array([[
        user["max_annual_fee"] * fee_scale,
        user["annual_income"],
        user["credit_score"],
        user["preferred_cashback"],
        user["preferred_reward"],
        user["travel_spend"] / 12,
        user["dining_spend"] / 12,
        user["online_spend"] / 12,
        user["fuel_spend"] / 12,
        user["lounge_importance"],
        user["lounge_importance"] * 4
    ]], dtype=float)

def profile_similarity(user, cards, scaler):
    card_X = scaler.transform(cards[CARD_FEATURES])
    # Scale the user using the same scaler fitted on card features.
    uv = user_vector(user)
    uv_scaled = scaler.transform(uv)
    return cosine_similarity(uv_scaled, card_X).flatten()

def build_recommendations(user, cards, predicted_category, scaler, top_n=3):
    cards = cards.copy()
    cards["profile_similarity"] = profile_similarity(user, cards, scaler)

    cards["category_match"] = (cards["card_category"].eq(predicted_category)).astype(float)

    cards["fee_match"] = np.where(
        cards["annual_fee"] <= user["max_annual_fee"], 1.0, 0.0
    )

    # Suitability from user's actual spending profile.
    spend = user["online_spend"] + user["dining_spend"] + user["travel_spend"] + user["fuel_spend"]
    if spend == 0:
        cards["spending_match"] = 0.0
    else:
        weighted = (
            user["online_spend"] * cards["online_reward"] +
            user["dining_spend"] * cards["dining_reward"] +
            user["travel_spend"] * cards["travel_reward"] +
            user["fuel_spend"] * cards["fuel_reward"]
        )
        cards["spending_match"] = (weighted / max(weighted.max(), 1)).fillna(0)

    cards["reward_suitability"] = (
        0.5 * (cards["cashback_rate"] / max(cards["cashback_rate"].max(), 1)) +
        0.5 * (cards["reward_rate"] / max(cards["reward_rate"].max(), 1))
    )

    cards["final_score"] = (
        0.35 * cards["profile_similarity"] +
        0.25 * cards["category_match"] +
        0.20 * cards["reward_suitability"] +
        0.10 * cards["fee_match"] +
        0.10 * cards["spending_match"]
    ) * 100

    cards["estimated_annual_rewards"] = cards.apply(
        lambda r: estimate_annual_rewards(r, user), axis=1
    )
    cards["estimated_net_benefit"] = cards.apply(
        lambda r: net_benefit(r, r["estimated_annual_rewards"]), axis=1
    )

    def why(row):
        reasons = []
        if row["category_match"] == 1:
            reasons.append(f"matches predicted category: {predicted_category}")
        if row["fee_match"] == 1:
            reasons.append("within your annual-fee budget")
        if row["online_reward"] >= 5 and user["online_spend"] > 0:
            reasons.append("strong online-spend benefit")
        if row["dining_reward"] >= 4 and user["dining_spend"] > 0:
            reasons.append("strong dining benefit")
        if row["travel_reward"] >= 4 and user["travel_spend"] > 0:
            reasons.append("strong travel benefit")
        if row["fuel_reward"] >= 4 and user["fuel_spend"] > 0:
            reasons.append("strong fuel benefit")
        if row["lounge_access"] and user["lounge_importance"] >= 1:
            reasons.append("airport lounge access")
        return "; ".join(reasons[:4]) or "good overall profile similarity"

    cards["why_it_matches"] = cards.apply(why, axis=1)
    return cards.sort_values("final_score", ascending=False).head(top_n)
