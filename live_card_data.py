"""
Live / verified source module.

Important:
- It DOES NOT scrape and silently overwrite numeric card terms.
- The verified_cards.csv is a curated snapshot derived from official issuer sources.
- At runtime this module checks that each official source URL is reachable and that
  the expected card name appears in the returned page/PDF text when accessible.
- This avoids pretending that a brittle scraper is an authoritative data API.
- If an issuer changes a fee/reward, the snapshot should be re-checked and updated.
"""

import json
from datetime import datetime
import pandas as pd
import requests

TIMEOUT = 12

def verify_one(card_name, source_url):
    result = {
        "card_name": card_name,
        "source_url": source_url,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "reachable": False,
        "card_name_found": False,
        "status": "unverified"
    }
    try:
        r = requests.get(
            source_url,
            timeout=TIMEOUT,
            headers={"User-Agent": "SmartCardAI-Verification/1.0"}
        )
        result["http_status"] = r.status_code
        result["reachable"] = r.ok
        text = r.text.lower()
        result["card_name_found"] = card_name.lower() in text
        result["status"] = "verified" if (r.ok and result["card_name_found"]) else "needs_review"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "needs_review"
    return result

def verify_all(csv_path="verified_cards.csv"):
    cards = pd.read_csv(csv_path)
    results = [
        verify_one(row.card_name, row.source_url)
        for row in cards.itertuples()
    ]
    return pd.DataFrame(results)

def load_verified_cards(csv_path="verified_cards.csv"):
    cards = pd.read_csv(csv_path)
    verification = verify_all(csv_path)
    cards = cards.merge(
        verification[["card_name", "reachable", "card_name_found", "status", "checked_at"]],
        on="card_name", how="left"
    )
    return cards, verification

if __name__ == "__main__":
    report = verify_all()
    report.to_csv("live_verification_report.csv", index=False)
    print(report[["card_name", "reachable", "card_name_found", "status"]].to_string(index=False))
