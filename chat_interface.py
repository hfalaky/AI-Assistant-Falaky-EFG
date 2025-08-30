# chat_interface.py
import os
import json
import pandas as pd
import streamlit as st

from preprocessing import preprocess_portfolio
from recommendation_engine import generate_recommendations
from prompts import generate_advice

st.set_page_config(page_title="EFG Investor Chatbot", layout="wide")

st.title("EFG Investor Chatbot")

# -----------------------------
# Expected portfolio columns
# -----------------------------
EXPECTED_COLUMNS = [
    "ClientID","ClientNameE","ClientAccProfileID","ClientSinceDate_x","DaysAsClient","Age","Category",
    "Source","SourceID","HasTrades2024","Group_x","Group_y","ClientSinceDate_y","CheckPoint1","CheckPoint2",
    "interval_start","interval_end","NumProfiles","ClientName","NetROI","MostProfitableSecurityID",
    "MostProfitableSecurityName","MostProfitableSecurityROI","MostProfitableSector","MostProfitableSectorROI",
    "MostActiveMonth","TradesInTheMostActiveMonth","TotalTradesIn24","TotalTradesVolumeIn24","MostTradedSecurityID",
    "MostTradedSecurity","NumberOfTradesOnMostTradedSecurity","TradesVolumeOfMostTradedSecurity","MostTradedSector",
    "TradesVolumeOfMostTradedSector","NumberOfTradesInMostTradedSector","longest_held_StockID","longest_held_Stock",
    "DurationHeld"
]

# --- Sidebar controls ---
st.sidebar.header("Settings")
model_name = st.sidebar.text_input("Groq Model", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

# --- Market section ---
st.subheader("Market Data")
market_file = st.text_input("Market CSV path", "Egypt_Equities.csv")
mkt_status = st.empty()

@st.cache_data(ttl=60*60*2)  # cache for 2 hours
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

st.markdown(
    "**Upload your portfolio in the following format (column order can vary, but these headers must exist):**"
)
st.code(
    "ClientID\tClientNameE\tClientAccProfileID\tClientSinceDate_x\tDaysAsClient\tAge\tCategory\t"
    "Source\tSourceID\tHasTrades2024\tGroup_x\tGroup_y\tClientSinceDate_y\tCheckPoint1\tCheckPoint2\t"
    "interval_start\tinterval_end\tNumProfiles\tClientName\tNetROI\tMostProfitableSecurityID\t"
    "MostProfitableSecurityName\tMostProfitableSecurityROI\tMostProfitableSector\tMostProfitableSectorROI\t"
    "MostActiveMonth\tTradesInTheMostActiveMonth\tTotalTradesIn24\tTotalTradesVolumeIn24\tMostTradedSecurityID\t"
    "MostTradedSecurity\tNumberOfTradesOnMostTradedSecurity\tTradesVolumeOfMostTradedSecurity\tMostTradedSector\t"
    "TradesVolumeOfMostTradedSector\tNumberOfTradesInMostTradedSector\tlongest_held_StockID\tlongest_held_Stock\tDurationHeld",
    language="text",
)

uploaded = st.file_uploader("Choose your merged portfolio CSV", type=["csv"])

@st.cache_data
def preprocess_buffer(file_bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    tmp_path = "_tmp_uploaded_portfolio.csv"
    df.to_csv(tmp_path, index=False)
    cleaned = preprocess_portfolio(tmp_path)
    return cleaned

def validate_columns(df: pd.DataFrame) -> list:
    found = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in EXPECTED_COLUMNS if c.lower() not in found]
    return missing

if uploaded:
    cleaned_df = preprocess_buffer(uploaded.read())
    missing = validate_columns(cleaned_df)
    if missing:
        st.warning(
            "Some expected columns are missing after preprocessing. "
            "Please confirm your CSV headers match the required schema.\n\n"
            f"Missing: {', '.join(missing)}"
        )
    st.success(f"Uploaded & cleaned. Rows: {len(cleaned_df)}")
else:
    default_path = "Active_Clients_Portfolio.csv"
    if os.path.exists(default_path):
        st.info("No upload provided — using Active_Clients_Portfolio.csv")
        cleaned_df = preprocess_portfolio(default_path)
        missing = validate_columns(cleaned_df)
        if missing:
            st.warning(
                "Some expected columns are missing after preprocessing in the default file.\n\n"
                f"Missing: {', '.join(missing)}"
            )
    else:
        st.warning("Please upload a portfolio CSV or place Active_Clients_Portfolio.csv in the project folder.")
        st.stop()

# --- Client picker ---
st.subheader("Select Client")
client_ids = cleaned_df["clientid"].astype(str).tolist()
sel_client = st.selectbox("ClientID", client_ids)
portfolio = cleaned_df[cleaned_df["clientid"] == sel_client].iloc[0].to_dict()

# --- Generate recommendations ---
st.subheader("Generate Advice")
if st.button("Run"):
    with st.spinner("Analyzing portfolio and market..."):
        engine_output = generate_recommendations(
            portfolio=portfolio,
            market=market_df,
            max_items=None,              # no cap
            freshness_policy="degrade",  # default freshness handling
            stale_after_minutes=120,
            market_asof=None
        )
        advice = generate_advice(portfolio, engine_output)

    st.markdown("### Advice")
    st.write(advice["message_text"])

    st.markdown("### Structured Recommendations (JSON)")
    st.json(engine_output)

    st.markdown("### Portfolio Snapshot (row)")
    st.json(portfolio)
