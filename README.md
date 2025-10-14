
# Sovereign Stash — Crypto Risk Radar (Public App)

No login. No data retention. Paste or import your crypto holdings and get a live risk profile.

## Run locally
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

## Deploy (Streamlit Community Cloud)
Push to GitHub → share.streamlit.io → select repo/branch → app.py.
(Optional) set OPENAI_API_KEY in app Secrets to enable AI summary.

## Disclaimer
Educational purposes only — not financial or investment advice.
