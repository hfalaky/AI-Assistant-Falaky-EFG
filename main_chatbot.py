# main_chatbot.py
import os
import json
import pandas as pd

from preprocessing import preprocess_portfolio
from recommendation_engine import generate_recommendations
from prompts import generate_advice

MARKET_CSV = "Egypt_Equities.csv"            # produced by market_scraper.py
DEFAULT_PORTFOLIO_CSV = "Active_Clients_Portfolio.csv"  # your merged active file

def load_market(csv_path: str = MARKET_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Market file not found: {csv_path}. Run market_scraper.py first.")
    df = pd.read_csv(csv_path)
    # safety: ensure expected columns exist minimally
    if "name" not in df.columns:
        raise ValueError("Market CSV missing 'name' column.")
    return df

def load_portfolio_csv(csv_path: str = DEFAULT_PORTFOLIO_CSV) -> pd.DataFrame:
    # Use your preprocessing module so types and filters are consistent
    return preprocess_portfolio(csv_path)

def pick_client(df: pd.DataFrame, client_id: str | None = None) -> dict:
    if client_id:
        sub = df[df["clientid"] == client_id]
        if sub.empty:
            raise ValueError(f"ClientID {client_id} not found in the uploaded/cleaned portfolio.")
        row = sub.iloc[0].to_dict()
        return row
    # default: pick first row
    return df.iloc[0].to_dict()

def run_once(client_id: str | None = None,
             portfolio_csv: str = DEFAULT_PORTFOLIO_CSV,
             market_csv: str = MARKET_CSV) -> dict:
    market_df = load_market(market_csv)
    cleaned_df = load_portfolio_csv(portfolio_csv)
    portfolio = pick_client(cleaned_df, client_id=client_id)

    engine_output = generate_recommendations(
        portfolio=portfolio,
        market=market_df,
        max_items=None,
        freshness_policy="degrade",
        stale_after_minutes=120,
        market_asof=None
    )
    advice = generate_advice(portfolio, engine_output)
    return {
        "client_id": portfolio.get("clientid"),
        "advice_text": advice["message_text"],
        "engine_output": engine_output
    }

if __name__ == "__main__":
    result = run_once(client_id=None)  # or pass a specific ID
    print("\n=== FINAL ADVICE ===\n")
    print(result["advice_text"])
    print("\n=== STRUCTURED JSON ===\n")
    print(json.dumps(result["engine_output"], indent=2, ensure_ascii=False))
