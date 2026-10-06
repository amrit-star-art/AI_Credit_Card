import os
import joblib
import pandas as pd
import streamlit as st

from live_card_data import load_verified_cards
from recommendation_engine import build_recommendations

st.set_page_config(page_title="SmartCard AI", page_icon="💳", layout="wide")
st.title("💳 SmartCard AI — Real Indian Credit Card Recommendation")
st.caption("Official-issuer-source-derived card snapshot + live source verification + KNN + Decision Tree + Cosine Similarity")

required = ["verified_cards.csv", "knn_bundle.pkl", "decision_tree.pkl"]
missing = [x for x in required if not os.path.exists(x)]
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Run train_models.py first.")
    st.stop()

cards, verification = load_verified_cards()
verified_count = int((verification["status"] == "verified").sum())

with st.sidebar:
    st.header("Your Profile")
    annual_income = st.number_input("Annual income (₹)", min_value=100000, value=600000, step=50000)
    credit_score = st.slider("Credit score", 550, 900, 750)
    annual_spending = st.number_input("Annual credit-card spending (₹)", min_value=0, value=300000, step=10000)
    online_spend = st.number_input("Online spend / month (₹)", min_value=0, value=12000, step=1000)
    dining_spend = st.number_input("Dining / month (₹)", min_value=0, value=5000, step=500)
    travel_spend = st.number_input("Travel / month (₹)", min_value=0, value=5000, step=500)
    fuel_spend = st.number_input("Fuel / month (₹)", min_value=0, value=3000, step=500)
    max_fee = st.number_input("Maximum annual fee (₹)", min_value=0, value=1000, step=100)
    lounge = st.slider("Importance of airport lounge", 0, 5, 2)
    preferred = st.selectbox(
        "Preferred benefit",
        ["Auto", "Cashback", "Shopping", "Dining", "Rewards", "Bills/Shopping"]
    )

st.info(
    f"Source verification: **{verified_count}/{len(verification)}** official sources verified at app startup. "
    "A source check verifies availability/content presence; it does not guarantee future issuer terms."
)

user = {
    "annual_income": annual_income,
    "credit_score": credit_score,
    "annual_spending": annual_spending,
    "travel_spend": travel_spend,
    "dining_spend": dining_spend,
    "fuel_spend": fuel_spend,
    "online_spend": online_spend,
    "max_annual_fee": max_fee,
    "lounge_importance": lounge,
    "preferred_cashback": 1 if preferred in ["Auto", "Cashback"] else 0,
    "preferred_reward": 1 if preferred in ["Rewards", "Shopping"] else 0,
}

# The training bundle was fit on the same verified-card feature schema.
bundle = joblib.load("knn_bundle.pkl")
tree = joblib.load("decision_tree.pkl")

FEATURES = [
    "annual_fee", "minimum_annual_income", "credit_score_requirement",
    "cashback_rate", "reward_rate", "travel_reward", "dining_reward",
    "online_reward", "fuel_reward", "lounge_access", "lounge_visits"
]

X_user = [[
    max_fee, annual_income, credit_score,
    user["preferred_cashback"], user["preferred_reward"],
    travel_spend / 12, dining_spend / 12, online_spend / 12,
    fuel_spend / 12, lounge, lounge * 4
]]
Xs = bundle["scaler"].transform(X_user)
knn_pred = bundle["model"].predict(Xs)[0]
tree_pred = tree.predict(Xs)[0]

predicted_category = preferred if preferred != "Auto" and preferred in cards["card_category"].unique() else knn_pred

# Eligibility filter: only enforce fields that are explicitly known.
eligible = cards.copy()
eligible = eligible[
    (eligible["annual_fee"] <= max_fee) &
    (eligible["credit_score_requirement"] <= credit_score) &
    (
        (eligible["minimum_annual_income"] == 0) |
        (eligible["minimum_annual_income"] <= annual_income)
    )
].copy()

if eligible.empty:
    st.warning("No cards matched the fee/credit-score filters. Increase the fee budget or review your profile.")
    st.stop()

results = build_recommendations(user, eligible, predicted_category, bundle["scaler"], top_n=3)

c1, c2, c3 = st.columns(3)
c1.metric("KNN category", str(knn_pred))
c2.metric("Decision Tree category", str(tree_pred))
c3.metric("Cards considered", len(eligible))

st.subheader("🏆 Top 3 Recommendations")
for _, r in results.iterrows():
    with st.container(border=True):
        st.markdown(f"### {r['card_name']}")
        a,b,c,d = st.columns(4)
        a.metric("Match score", f"{r['final_score']:.1f}/100")
        b.metric("Annual fee", f"₹{r['annual_fee']:,.0f}")
        c.metric("Est. annual rewards", f"₹{r['estimated_annual_rewards']:,.0f}")
        d.metric("Est. net benefit", f"₹{r['estimated_net_benefit']:,.0f}")
        st.write("**Why it matches:**", r["why_it_matches"])
        st.caption(f"Source: {r['source_url']}")

with st.expander("How the recommendation works"):
    st.write("""
    1. **Eligibility filter** removes cards outside the user's fee/credit-score constraints.
    2. **KNN** predicts the closest card category from the user's profile.
    3. **Decision Tree** independently predicts the category.
    4. **Cosine Similarity** measures the user's feature-vector similarity to each eligible card.
    5. A weighted recommendation score combines profile similarity, category match,
       reward suitability, fee match and spending-category match.
    6. Estimated rewards and net benefit are calculated from the stored issuer-source-derived terms.
    """)

st.warning(
    "This is a recommendation/demo system, not a bank approval engine or financial advice. "
    "Issuer approval, eligibility, fees, reward exclusions and offers can change."
)
