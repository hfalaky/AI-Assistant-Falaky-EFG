# chat_interface.py
import os
import json
import pandas as pd
import streamlit as st

from preprocessing import preprocess_portfolio
from recommendation_engine import generate_recommendations
from prompts import generate_advice

st.set_page_config(page_title="EFG Investor Chatbot", layout="wide")

st.title("EFG Investor Chatbot 🧠📈")

# --- Sidebar controls ---
st.sidebar.header("Settings")
model_name = st.sidebar.text_input("Groq Model", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
freshness_minutes = st.sidebar.number_input("Market data stale after (min)", min_value=15, max_value=720, value=120, step=15)
max_items = st.sidebar.number_input("Max recommendations to show (0 = all)", min_value=0, max_value=20, value=0, step=1)

# --- Market section ---
st.subheader("Market Data")
market_file = st.text_input("Market CSV path", "Egypt_Equities.csv")
mkt_status = st.empty()

@st.cache_data(ttl=60*60*2)  # 2 hours cache to align with freshness
def load_market(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing market file: {path}. Run market_scraper.py.")
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError("Market CSV missing 'name' column.")
    return df

try:
    market_df = load_market(market_file)
    mkt_status.success(f"Loaded market file: {market_file} (rows: {len(market_df)})")
except Exception as e:
    mkt_status.error(str(e))
    st.stop()

# --- Portfolio upload ---
st.subheader("Upload Portfolio CSV")
uploaded = st.file_uploader("Drop your merged portfolio CSV", type=["csv"])

@st.cache_data
def preprocess_buffer(file_bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    tmp_path = "_tmp_uploaded_portfolio.csv"
    df.to_csv(tmp_path, index=False)
    # Use your preprocessing function for consistency
    cleaned = preprocess_portfolio(tmp_path)
    return cleaned

if uploaded:
    cleaned_df = preprocess_buffer(uploaded.read())
    st.success(f"Uploaded & cleaned. Rows: {len(cleaned_df)}")
else:
    # fallback: use your default merged active file if present
    default_path = "Active_Clients_Portfolio.csv"
    if os.path.exists(default_path):
        st.info("No upload provided — using Active_Clients_Portfolio.csv")
        cleaned_df = preprocess_portfolio(default_path)
    else:
        st.warning("Please upload a portfolio CSV or place Active_Clients_Portfolio.csv in the project folder.")
        st.stop()

# --- Client picker ---
st.subheader("Select Client")
client_ids = cleaned_df["clientid"].astype(str).tolist()
sel_client = st.selectbox("ClientID", client_ids)
portfolio = cleaned_df[cleaned_df["clientid"] == sel_client].iloc[0].to_dict()

# --- Generate recommendations ---
if st.button("Generate Advice"):
    with st.spinner("Analyzing portfolio and market..."):
        engine_output = generate_recommendations(
            portfolio=portfolio,
            market=market_df,
            max_items=None if max_items == 0 else max_items,
            freshness_policy="degrade",
            stale_after_minutes=int(freshness_minutes),
            market_asof=None
        )
        advice = generate_advice(portfolio, engine_output)

    st.markdown("## Advice")
    st.write(advice["message_text"])

    with st.expander("Structured Recommendations (JSON)"):
        st.json(engine_output)

    with st.expander("Portfolio Snapshot (row)"):
        st.json(portfolio)

st.caption("Note: Informational only, not investment advice. Market data source: Investing.com via your scraper.")
