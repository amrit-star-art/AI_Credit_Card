# SmartCard AI — Real Indian Credit Card Recommendation

## Data strategy
The project uses a verified snapshot of Indian credit-card terms derived from official issuer sources.
Current sources include HDFC Bank, ICICI Bank, SBI Card and Axis Bank.

`live_card_data.py` checks the official source URLs when the app starts. It verifies reachability and
that the expected card name appears in the returned content. It does not silently scrape and overwrite
financial terms because issuer pages and terms can change format and automated scraping can be brittle.

## Run
```bash
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

## Refresh verification
```bash
python live_card_data.py
```
This creates `live_verification_report.csv`.

## Important
- `minimum_annual_income = 0` means the card-specific source used for this snapshot did not state a
  numeric minimum income; the app does not invent one.
- Reward estimates are illustrative, not guaranteed redemption values.
- Current offers, fees, exclusions and eligibility can change.
- The system never guarantees approval.
