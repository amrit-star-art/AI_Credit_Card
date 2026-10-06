def estimate_annual_rewards(card, user):
    """
    Conservative estimate for demonstration.
    It uses category cashback/reward-rate fields as a proxy, not a bank guarantee.
    Reward-point cards use reward_rate as a relative rate and convert it using a
    documented 4 points = ₹1 rule where applicable; the dataset keeps this simplified.
    """
    annual_online = user["online_spend"] * 12
    annual_travel = user["travel_spend"] * 12
    annual_dining = user["dining_spend"] * 12
    annual_fuel = user["fuel_spend"] * 12
    total_annual = user["annual_spending"]

    cashback = 0.0
    cashback += annual_online * (float(card["online_reward"]) / 100.0)
    cashback += annual_dining * (float(card["dining_reward"]) / 100.0)
    cashback += annual_travel * (float(card["travel_reward"]) / 100.0)
    cashback += annual_fuel * (float(card["fuel_reward"]) / 100.0)

    # Base rate applies to remaining/general spend.
    allocated = annual_online + annual_dining + annual_travel + annual_fuel
    other = max(total_annual - allocated, 0)
    cashback += other * (float(card["cashback_rate"]) / 100.0)

    return round(max(cashback, 0), 2)

def net_benefit(card, rewards):
    return round(rewards - float(card["annual_fee"]), 2)
