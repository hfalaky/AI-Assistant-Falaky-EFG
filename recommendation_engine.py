# recommendation_engine.py
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import pandas as pd


# =========================
# Sector normalization (11 buckets)
# =========================
# NOTE: We display the user's original (mentioned) sector text in messages,
# but we *implicitly* normalize to these 11 buckets for all comparisons.
SECTOR_MAP_11 = {
    # Financials & variants
    "banking": "Financials",
    "financials": "Financials",
    "investment": "Financials",
    "jordan securities": "Financials",
    "pakistani securities": "Financials",
    "kenian securities": "Financials",

    # Materials & variants
    "basic materials": "Materials",
    "materials": "Materials",

    # Industrials & variants
    "industrial": "Industrials",
    "industrials": "Industrials",
    "industries": "Industrials",
    "services": "Industrials",   # adjust to 'Consumer Discretionary' if your data is mostly consumer-facing
    "others": "Industrials",     # safe fallback

    # Consumer Discretionary
    "consumer discretionary": "Consumer Discretionary",
    "consumer services": "Consumer Discretionary",
    "tourism": "Consumer Discretionary",
    "trade": "Consumer Discretionary",

    # Consumer Staples & variants
    "consumer staples": "Consumer Staples",
    "food": "Consumer Staples",

    # Energy
    "energy": "Energy",

    # Healthcare & variants
    "health care": "Healthcare",
    "healthcare": "Healthcare",

    # Technology & variants
    "technology": "Technology",
    "information technology": "Technology",

    # Real Estate & variants
    "real estate": "Real Estate",
    "realestate": "Real Estate",

    # Telecommunications & variants
    "telecommunication services": "Telecommunications",
    "telecommunications": "Telecommunications",
}

def _std_sector_11(label: str) -> str:
    """Normalize arbitrary sector text to one of the 11 buckets."""
    if pd.isna(label) or label is None:
        return "Unknown"
    s = str(label).strip().lower()
    s = s.replace("&", "and")
    s = " ".join(s.split())
    return SECTOR_MAP_11.get(s, "Unknown")


# =========================
# Small utils
# =========================
def _is_unknown(val) -> bool:
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in {"", "na", "n/a", "none", "null", "unknown"}

def _safe_percent(x: Optional[float], dp: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{round(x*100, dp):.{dp}f}%"

def _get_user_sector_labels(portfolio: Dict) -> Tuple[str, str]:
    """
    Return (mentioned_sector_for_display, normalized_sector_for_logic).
    Prefer most-traded; fallback to most-profitable.
    """
    mentioned = (
        portfolio.get("mosttradedsector")
        or portfolio.get("mostprofitablesector")
        or ""
    )
    # If standardized columns exist from preprocessing, use them; else normalize here.
    std = (
        portfolio.get("mosttradedsector_std")
        or portfolio.get("mostprofitablesector_std")
        or _std_sector_11(mentioned)
    )
    return str(mentioned), str(std)

def _ensure_market_sector_std(market: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the market DataFrame has a 'sector_std' column (normalized to 11 buckets).
    If 'sector' doesn't exist, we return the DF unchanged.
    """
    if market is None or market.empty:
        return market
    df = market.copy()
    if "sector" in df.columns:
        df["sector_std"] = df["sector"].apply(_std_sector_11)
    return df


# =========================
# Persona
# =========================
def _infer_risk_persona(portfolio: Dict) -> Tuple[str, float]:
    """
    Infer persona using tenure (daysasclient), activity (trades/month), and holding style (durationheld).
    Returns (label, confidence).
    """
    tenure_days = int(portfolio.get("daysasclient") or 0)
    trades_24 = int(portfolio.get("totaltradesin24") or 0)
    avg_hold = int(portfolio.get("durationheld") or 0)

    months = max(1.0, tenure_days / 30.0)
    tpm = trades_24 / months  # trades per month

    score = 0
    # Higher activity => more aggressive
    if tpm >= 4:
        score += 2
    elif tpm >= 2:
        score += 1

    # Shorter holds => more aggressive
    if avg_hold <= 45:
        score += 2
    elif avg_hold <= 90:
        score += 1

    # Longer tenure => more conservative
    if tenure_days >= 1000:
        score -= 2
    elif tenure_days >= 365:
        score -= 1

    if score >= 2:
        return "Aggressive", 0.75
    elif score <= -1:
        return "Conservative", 0.70
    else:
        return "Balanced", 0.65


# =========================
# Data quality checks
# =========================
def _data_quality_checks(p: Dict) -> List[Dict]:
    issues: List[str] = []

    # Interval dates sanity
    try:
        start = pd.to_datetime(p.get("interval_start"), errors="coerce")
        end = pd.to_datetime(p.get("interval_end"), errors="coerce")
        if pd.notna(start) and pd.notna(end) and start > end:
            issues.append("interval_start occurs after interval_end")
    except Exception:
        pass

    # Unknown sector but sector metrics present
    most_traded_sector = p.get("mosttradedsector")
    sector_vol = float(p.get("tradesvolumeofmosttradedsector") or 0.0)
    sector_trades = int(p.get("numberoftradesinmosttradedsector") or 0)
    if _is_unknown(most_traded_sector) and (sector_vol > 0 or sector_trades > 0):
        issues.append("mosttradedsector is 'Unknown' while sector volume/trades exist")

    recs: List[Dict] = []
    if issues:
        # Suppress trivial interval_start issue if it's the only one (optional UX)
        if not (len(issues) == 1 and issues[0].startswith("interval_start")):
            recs.append({
                "type": "data_quality",
                "priority": 0.30,
                "message": "Some portfolio fields look inconsistent; verify sector/date mapping before acting.",
                "evidence": {"issues": issues},
                "stale": False
            })
    return recs


# =========================
# Sector concentration
# =========================
def _check_sector_concentration(portfolio: Dict, threshold: float = 0.60) -> Optional[Dict]:
    """
    If a single sector dominates trading volume >= threshold, warn for concentration.
    Uses: mosttradedsector, tradesvolumeofmosttradedsector / totaltradesvolumein24
    """
    mentioned_sector = portfolio.get("mosttradedsector")
    if _is_unknown(mentioned_sector):
        return None

    sector_vol = portfolio.get("tradesvolumeofmosttradedsector")
    total_vol = portfolio.get("totaltradesvolumein24")
    try:
        sector_vol = float(sector_vol or 0.0)
        total_vol = float(total_vol or 0.0)
    except Exception:
        return None

    if total_vol <= 0:
        return None

    share = sector_vol / total_vol
    if share >= threshold:
        return {
            "type": "sector_concentration",
            "priority": 0.90,
            "message": f"{mentioned_sector} accounts for {share:.0%} of your 2024 trading volume — consider diversifying.",
            "evidence": {"sector_mentioned": mentioned_sector, "sector_share": round(share, 4)},
            "stale": False
        }
    return None


# =========================
# Diversification suggestion (outside user's normalized sector)
# =========================
def _suggest_diversification(portfolio: Dict, market: pd.DataFrame) -> Optional[Dict]:
    """
    Suggest diversification into an alternative area showing positive momentum.
    - Display uses user's *mentioned* sector label.
    - Comparison excludes user's sector using *normalized* 11-bucket mapping.
    """
    if market is None or market.empty or "name" not in market.columns:
        return None

    # Ensure normalized market sector
    market = _ensure_market_sector_std(market)

    mentioned_sector, normalized_sector = _get_user_sector_labels(portfolio)

    # Candidates: positive movers only
    candidates = market.copy()
    if "change_pct" in candidates.columns:
        candidates["change_pct"] = pd.to_numeric(candidates["change_pct"], errors="coerce")
        candidates = candidates[candidates["change_pct"].fillna(0) > 0]

    if candidates.empty:
        return None

    # Exclude user's normalized main sector if available
    if "sector_std" in candidates.columns and not _is_unknown(normalized_sector):
        candidates = candidates[candidates["sector_std"] != normalized_sector]

    if candidates.empty:
        return None

    # Pick strongest positive mover outside user's normalized sector
    candidates = candidates.sort_values("change_pct", ascending=False)
    alt = candidates.iloc[0]

    alt_name = str(alt.get("name", ""))
    if _is_unknown(alt_name):
        return None

    prefix = f" outside {mentioned_sector}" if not _is_unknown(mentioned_sector) else ""
    msg = f"Consider diversifying{prefix}. Example: {alt_name} (+{_safe_percent(alt.get('change_pct'))} today)."

    ev = {
        "alt_stock": alt_name,
        "alt_sector_std": str(alt.get("sector_std", "Unknown")),
        "user_sector_mentioned": mentioned_sector,
        "user_sector_std": normalized_sector,
        "change_pct": float(alt.get("change_pct") or 0),
    }

    return {
        "type": "diversification",
        "priority": 0.75,
        "message": msg,
        "evidence": ev,
        "stale": False
    }


# =========================
# Within-profitable-sector idea (if known)
# =========================
def _suggest_within_profitable_sector(portfolio: Dict, market: pd.DataFrame) -> Optional[Dict]:
    prof_std = portfolio.get("mostprofitablesector_std")
    if _is_unknown(prof_std):
        return None
    m = _ensure_market_sector_std(market)
    df = m[(m.get("sector_std") == prof_std)]
    if "change_pct" not in df.columns or df.empty:
        return None
    df = df.copy()
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df = df.dropna(subset=["change_pct"]).sort_values("change_pct", ascending=False)
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "type": "within_profitable_sector",
        "priority": 0.70,
        "message": f"In your profitable sector ({prof_std}), {r['name']} is up {_safe_percent(r['change_pct'])} today.",
        "evidence": {"stock": r["name"], "sector_std": prof_std, "change_pct": float(r["change_pct"])},
        "stale": False
    }


# =========================
# NEW: Within-primary-sector idea (same sector as user's main exposure)
# =========================
def _suggest_within_primary_sector(portfolio: Dict, market: pd.DataFrame) -> Optional[Dict]:
    """
    Suggest an idea *within* the client's primary sector (same sector idea).
    Primary sector = mosttradedsector_std, fallback to mostprofitablesector_std.
    Picks the strongest positive mover today in that sector.
    """
    primary_std = (
        portfolio.get("mosttradedsector_std")
        or portfolio.get("mostprofitablesector_std")
        or "Unknown"
    )
    if _is_unknown(primary_std):
        return None

    m = _ensure_market_sector_std(market)
    if m is None or m.empty or "sector_std" not in m.columns:
        return None

    df = m[(m["sector_std"] == primary_std)].copy()
    if df.empty or "change_pct" not in df.columns or "name" not in df.columns:
        return None

    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df = df.dropna(subset=["change_pct", "name"])
    df = df[df["change_pct"] > 0]  # only positive momentum
    if df.empty:
        return None

    df = df.sort_values("change_pct", ascending=False)
    r = df.iloc[0]

    return {
        "type": "within_primary_sector",
        "priority": 0.70,  # below concentration (0.90) and diversification (0.75)
        "message": (
            f"Within your main sector ({primary_std}), {r['name']} is up "
            f"{_safe_percent(r['change_pct'])} today."
        ),
        "evidence": {
            "stock": r["name"],
            "sector_std": primary_std,
            "change_pct": float(r["change_pct"]),
        },
        "stale": False,
    }


# =========================
# Movers split: opportunities vs alerts
# =========================
def _highlight_movers_split(
    market: pd.DataFrame,
    up_thr: float = 0.02,
    down_thr: float = 0.03,
    exclude_names: Optional[List[str]] = None,
    up_limit: Optional[int] = 5,
    down_limit: Optional[int] = 5
) -> List[Dict]:
    """
    Create two kinds of items:
      - top_mover_up: strong positive momentum (>= up_thr)
      - watchlist_drop: sharp losses (<= -down_thr)
    """
    if market is None or market.empty or "change_pct" not in market.columns or "name" not in market.columns:
        return []

    df = market.copy()
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df = df.dropna(subset=["change_pct", "name"])

    if exclude_names:
        ex = set([str(n).strip().lower() for n in exclude_names if not _is_unknown(n)])
        df = df[~df["name"].astype(str).str.strip().str.lower().isin(ex)]

    recs: List[Dict] = []

    ups = df[df["change_pct"] >= up_thr].sort_values("change_pct", ascending=False)
    if up_limit is not None:
        ups = ups.head(up_limit)

    for _, r in ups.iterrows():
        recs.append({
            "type": "top_mover_up",
            "priority": 0.65,
            "message": f"{r['name']} up {_safe_percent(r['change_pct'])} today.",
            "evidence": {"stock": r["name"], "change_pct": float(r["change_pct"])},
            "stale": False
        })

    downs = df[df["change_pct"] <= -down_thr].sort_values("change_pct", ascending=True)
    if down_limit is not None:
        downs = downs.head(down_limit)

    for _, r in downs.iterrows():
        recs.append({
            "type": "watchlist_drop",
            "priority": 0.50,
            "message": f"{r['name']} down {_safe_percent(r['change_pct'])} today — monitor risk.",
            "evidence": {"stock": r["name"], "change_pct": float(r["change_pct"])},
            "stale": False
        })

    return recs


# =========================
# Main API
# =========================
def generate_recommendations(
    portfolio: Dict,
    market: pd.DataFrame,
    max_items: Optional[int] = None,              # None => no hard cap
    freshness_policy: str = "degrade",            # "degrade" | "warn" | "off"
    stale_after_minutes: int = 120,
    market_asof: Optional[datetime] = None        # pass a timestamp if you have it
) -> Dict:
    """
    Build structured recommendations from a single client portfolio snapshot + market DataFrame.
    Display uses user's mentioned sector; logic compares normalized (11 buckets).
    """
    # Persona
    persona, persona_conf = _infer_risk_persona(portfolio)

    # Market freshness
    now = datetime.utcnow()
    asof = market_asof or now  # if none provided, assume "now"
    is_stale = (now - asof) > timedelta(minutes=stale_after_minutes)

    # Build rec candidates
    recs: List[Dict] = []
    recs.extend(_data_quality_checks(portfolio))

    r1 = _check_sector_concentration(portfolio)
    if r1:
        recs.append(r1)

    r2 = _suggest_diversification(portfolio, market)
    if r2:
        recs.append(r2)

    r3 = _suggest_within_profitable_sector(portfolio, market)
    if r3:
        recs.append(r3)

    # NEW: same-sector idea (primary sector)
    r4 = _suggest_within_primary_sector(portfolio, market)
    if r4:
        recs.append(r4)

    # Exclude heavy/known names from movers if available
    exclude = [
        portfolio.get("mosttradedsecurity"),
        portfolio.get("mostprofitablesecurityname")
    ]
    recs.extend(_highlight_movers_split(market, exclude_names=exclude))

    # Apply freshness policy: never hide; optionally degrade/label
    if is_stale:
        for r in recs:
            if freshness_policy == "degrade":
                r["priority"] *= 0.7
                r["stale"] = True
            elif freshness_policy == "warn":
                r["stale"] = True
            # "off" => no change

    # Sort by priority (desc)
    recs = sorted(recs, key=lambda x: x.get("priority", 0), reverse=True)

    # --- de-duplicate by stock name across all recs ---
    seen = set()
    deduped = []
    for r in recs:
        key = str(
            (r.get("evidence", {}).get("alt_stock")
             or r.get("evidence", {}).get("stock")
             or "")
        ).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(r)
    recs = deduped

    # Optional top-K
    if isinstance(max_items, int) and max_items > 0:
        recs = recs[:max_items]

    return {
        "client_id": portfolio.get("clientid"),
        "risk_persona": persona,
        "persona_confidence": round(float(persona_conf), 2),
        "meta": {
            "freshness_policy": freshness_policy,
            "stale_after_minutes": stale_after_minutes,
            "data_timestamp": asof.isoformat(),
            "max_items": max_items
        },
        "recommendations": recs
    }


# =========================
# CLI test (optional)
# =========================
if __name__ == "__main__":
    import json, random
    import pandas as pd

    # Load all client snapshots (list[dict])
    with open("Clients_Portfolios.json", "r", encoding="utf-8") as f:
        portfolios = json.load(f)

    # Pick one at random for testing
    portfolio = random.choice(portfolios)
    print("🔎 Testing with client:", portfolio.get("clientid"))

    # Load market data
    market_df = pd.read_csv("Egypt_Equities.csv")

    # Generate recs (no hard cap; freshness degraded when stale)
    result = generate_recommendations(
        portfolio=portfolio,
        market=market_df,
        max_items=None,
        freshness_policy="degrade",
        stale_after_minutes=120,
        market_asof=None
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
