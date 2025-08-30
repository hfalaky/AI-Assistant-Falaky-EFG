# market_scraper.py
import asyncio #Python’s library for asynchronous programming (needed because Crawl4AI is async)
import pandas as pd
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
#AsyncWebCrawler: The headless browser controller
#            from Crawl4AI. It lets us open web pages, fetch their HTML, and simulate a real browser.
#CrawlerRunConfig: Defines how the crawler should run (timeout, retries, etc.).

try:
    from crawl4ai import BrowserConfig
except ImportError:
    from crawl4ai.async_configs import BrowserConfig  # fallback for old versions
#If BrowserConfig is available directly, import it. Otherwise, import it from async_configs.

from io import StringIO
#StringIO is a utility that makes a string behave like a file object.
#Since we have raw HTML text from the scraper, we wrap it in StringIO
#                so Pandas can parse it as if it was reading an HTML file.

import json

INV_URL = "https://www.investing.com/equities/egypt"

# === NEW: sector mapping for EGX30 companies ===
SECTOR_MAP = {
    "Qalaa Holdings": "Industrials",
    "Commercial Int Bank": "Financials",
    "Egyptian Kuwaiti Hld": "Conglomerates",
    "Telecom Egypt": "Telecommunications",
    "EFG Hermes Holdings": "Financials",
    "Palm Hills Develop": "Real Estate",
    "Sidi Kerir": "Energy",
    "T M G Holding": "Real Estate",
    "GB AUTO": "Consumer Discretionary",
    "Madinet Nasr for Housing and Development": "Real Estate",
    "Oriental Weavers": "Consumer Discretionary",
    "Raya Holding": "Technology",
    "Abu Qir Fertilizers and Chemical Industries": "Materials",
    "Misr Cement": "Materials",
    "Alexandria Mineral Oils": "Energy",
    "Credit Agricole Egypt": "Financials",
    "Eastern Tobacco": "Consumer Staples",
    "Beltone Financial Hld": "Financials",
    "Egypt Aluminum": "Materials",
    "Juhayna Food": "Consumer Staples",
    "Orascom Hotels": "Real Estate",
    "Abu Dhabi Islamic Bank": "Financials",
    "Arabian Cement Co SAE": "Materials",
    "Orascom Construction Ltd": "Industrials",
    "Emaar Misr for Development SAE": "Real Estate",
    "Misr Fertilizers": "Materials",
    "Ibnsina Pharma": "Healthcare",
    "Fawry Banking and Payment": "Technology",
    "Tenth of Ramadan": "Industrials",
    "Egypt Kuwait Holding": "Conglomerates",
    "E-finance": "Technology",
}

#basic eda for the df
def _normalize_equities(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "Name": "name",
        "Last": "last",
        "High": "high",
        "Low": "low",
        "Chg. %": "change_pct",
        "Change %": "change_pct",
        "Vol.": "volume",
        "Volume": "volume",
        "Time": "time",
    }
    df = df.rename(columns={c: col_map.get(c.strip(), c.strip().lower()) for c in df.columns})
    #apply column name mapping

    # --- numeric cleaning ---
    for c in ["last", "high", "low"]:
        if c in df.columns: #useless (ensures we only clean them if they actually exist in the scraped df)
            df[c] = (
                df[c].astype(str) #makes values string before clening
                .str.replace(",", "", regex=False) #for millieme (ex: 32k == 32,000)
                .str.replace("\u00a0", "", regex=False)
                .str.replace("—", "", regex=False) #removes dashes (used for missing values on Investing.com)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce") #converts back to numbers
            #errors=coerce means If conversion fails, it replaces the value with NaN instead of crashing.

    if "change_pct" in df.columns: #eda specifically for this cloumn
        s = (
            df["change_pct"].astype(str)
            .str.replace("%", "", regex=False) #will divide by 100...
            .str.replace("+", "", regex=False) #no need
            .str.replace(",", "", regex=False)
            .str.replace("\u00a0", "", regex=False)
            .str.replace("—", "", regex=False)
        )
        df["change_pct"] = pd.to_numeric(s, errors="coerce") / 100.0 #instead of %

    # --- NEW: drop any 'unnamed' columns that may appear from parsing ---
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]

    # --- NEW: round all float columns to 4 dp to avoid long floats in Excel ---
    float_cols = df.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].round(4)

    # === NEW: add sector column ===
    if "name" in df.columns:
        df["sector"] = df["name"].map(SECTOR_MAP).fillna("Unknown")

    return df


def _pick_equities_table(html: str) -> pd.DataFrame:
    #table extractor — it decides which HTML table on the page actually contains the Egyptian stock data

    # Wrap HTML string in StringIO to avoid FutureWarning
    tables = pd.read_html(StringIO(html), flavor="bs4") #looks for html <tables> <==> equivalent to CTRL F table
    #
    #✅ At this point, tables is a list of DataFrames, one for each <table> found on the page.
    for t in tables:
        cols = [str(c).lower() for c in t.columns] #cleans column names for each table
        if "last" in cols and any("chg" in c or "change" in c for c in cols):
            return t #if the table contain a cloumn named last and Change-> its this one
    raise RuntimeError("No equities table found.") #if no table with these columns foun --> error


async def _async_get_market_data(max_rows: int = 200) -> pd.DataFrame:

    #async → This runs asynchronously (non-blocking, perfect for web scraping where network calls take time)

    browser_cfg = BrowserConfig(headless=True) #sets up a browser in headless mode (runs in the background)
    run_cfg = CrawlerRunConfig()  # default runtime settings for Crawl4AI

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=INV_URL, config=run_cfg) #sets it up with the prepared settings
        #result contains the scrapped content (HTML)

    html = getattr(result, "html", "") #extracts html from result
    if not html:
        raise RuntimeError("Empty HTML returned.")

    raw_df = _pick_equities_table(html) #calls function we defined to output the table we want
    equities_df = _normalize_equities(raw_df) #eda on our table

    if max_rows:
        equities_df = equities_df.head(max_rows)
    return equities_df


def get_market_data(max_rows: int = 200) -> pd.DataFrame:
    return asyncio.run(_async_get_market_data(max_rows=max_rows))

def market_df_to_json(df, output_file="Egypt_Equities.json"):
    """
    Convert the equities DataFrame into JSON format for LLM usage.
    """
    equities = df.to_dict(orient="records")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(equities, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(equities)} equities to {output_file}")
    return equities



if __name__ == "__main__":
    df = get_market_data()
    print("✅ Scraped market data")
    print(df.head())

    # Safety: ensure no 'unnamed' and floats rounded before saving
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]
    float_cols = df.select_dtypes(include="float").columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].round(4)

    df.to_csv("Egypt_Equities.csv", index=False)
    print("📂 Saved to Egypt_Equities.csv")

    # Also save JSON for LLM
    market_df_to_json(df)

