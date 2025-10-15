# Sovereign Stash – Crypto Portfolio Navigator (public, no-auth)
import os, json, re, time
from datetime import datetime, timedelta, date
from io import BytesIO

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

from utils import ingest

st.set_page_config(page_title="Sovereign Stash – Crypto Portfolio Navigator", page_icon="🛰️", layout="wide")

# ——— Optional OpenAI client (kept for fallback notes) ———
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ——— Constants ———
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
STABLECOIN_IDS = {"tether","usd-coin","binance-usd","dai","true-usd","usdd","frax"}
FOLD_TO_BASE = {"wbtc":"bitcoin","btc.b":"bitcoin","weth":"ethereum","steth":"ethereum","reth":"ethereum","cbeth":"ethereum","frxeth":"ethereum"}
PREFERRED_COINS = {
    "btc":"bitcoin","xbt":"bitcoin","bitcoin":"bitcoin",
    "eth":"ethereum","ethereum":"ethereum",
    "usdt":"tether","usdc":"usd-coin","dai":"dai","busd":"binance-usd",
    "ada":"cardano","sol":"solana","xrp":"ripple","doge":"dogecoin",
    "dot":"polkadot","link":"chainlink","bch":"bitcoin-cash","ltc":"litecoin",
    "trx":"tron","matic":"polygon-pos","avax":"avalanche-2","atom":"cosmos",
    "near":"near","uni":"uniswap","etc":"ethereum-classic",
    "arb":"arbitrum","op":"optimism","apt":"aptos","inj":"injective",
    "ton":"the-open-network","toncoin":"the-open-network",
}

DISCLAIMER = "Educational purposes only — not financial or investment advice. Crypto assets are volatile and you can lose money."

# ——— Theme & CSS ———
def set_plotly_theme(theme: str):
    pio.templates.default = "plotly_white" if theme == "Light" else "plotly_dark"
    px.defaults.color_discrete_sequence = ["#10b981","#2563eb","#f59e0b","#22a2ee","#e11d48"]

def inject_css(theme: str):
    if theme=="Dark":
        bg="#0B0B0C"; text="#E6E6E6"; card="rgba(23,23,24,0.55)"; border="rgba(255,255,255,0.12)"
        btn="#10b981"; btnh="#059669"; btxt="#08100c"
    else:
        bg="#F5F7FB"; text="#0f172a"; card="rgba(255,255,255,0.70)"; border="rgba(15,23,42,0.10)"
        btn="#10b981"; btnh="#059669"; btxt="#ffffff"
    st.markdown(f"""
    <style>
      body, [data-testid="stAppViewContainer"] {{
        background:{bg}; color:{text}; font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
      }}
      .glass-card {{
        background:{card}; border:1px solid {border}; border-radius:18px; padding:18px; margin:14px 0;
        box-shadow:0 10px 30px rgba(0,0,0,0.2); backdrop-filter: blur(10px);
      }}
      .stButton>button, .stDownloadButton>button {{
        background:{btn} !important; color:{btxt} !important; border:0 !important;
        border-radius:14px !important; padding:10px 14px !important; font-weight:700 !important;
        box-shadow:0 6px 18px rgba(16,185,129,0.35) !important;
      }}
      .stButton>button:hover, .stDownloadButton>button:hover {{ background:{btnh} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ——— Data helpers ———
@st.cache_data(ttl=3600)
def fetch_coin_list():
    r = requests.get(f"{COINGECKO_BASE}/coins/list", timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def build_symbol_to_id_map():
    m={}
    for c in fetch_coin_list():
        s=(c.get("symbol") or "").lower(); cid=c.get("id")
        if s and s not in m: m[s]=cid
        if cid and cid not in m: m[cid]=cid
    return m

def symbol_to_id(symbol, s2id):
    if not symbol: return None
    s = symbol.strip().lower()
    if s in PREFERRED_COINS: return PREFERRED_COINS[s]
    # try raw match and sanitized
    if s in s2id: return s2id[s]
    s_clean = re.sub(r"[^a-z0-9-]", "", s)
    return s2id.get(s_clean, s)

@st.cache_data(ttl=3600)
def fetch_market_data(ids):
    if not ids: return []
    out=[]
    for i in range(0, len(ids), 50):
        chunk=",".join(ids[i:i+50])
        r=requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency":"usd","ids":chunk,"price_change_percentage":"24h"},
            timeout=25
        )
        if r.ok: out += r.json()
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history_usd_range(coin_id: str, start_ts: int, end_ts: int, retries: int = 3) -> pd.DataFrame:
    """Fetch daily USD prices; robust to gaps. Returns date + price_usd with NaNs for missing days."""
    if not coin_id or end_ts <= start_ts:
        return pd.DataFrame(columns=["date","price_usd"])
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range"
    params = {"vs_currency":"usd","from":start_ts,"to":end_ts}
    last_err = None
    for a in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1.1*(a+1)); continue
            r.raise_for_status()
            prices = r.json().get("prices", [])
            if not prices:
                return pd.DataFrame(columns=["date","price_usd"])
            df = pd.DataFrame(prices, columns=["ts","price_usd"])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
            df = df[["date","price_usd"]].drop_duplicates("date").sort_values("date")
            # daily frequency with gaps as NaN, plus light ffill (limit 2 days) to avoid over-fragmentation
            df = df.set_index("date").resample("1D").asfreq().ffill(limit=2).reset_index()
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.8*(a+1))
    # graceful failure
    return pd.DataFrame(columns=["date","price_usd"])

# ——— Analytics ———
def calculate_profile_score(df: pd.DataFrame):
    tot = df["value_usd"].sum()
    if tot <= 0: return 0.0, {}
    key = "folded_id" if "folded_id" in df.columns else "id"
    btc_eth = df[key].isin(["bitcoin","ethereum"])
    stable  = df[key].isin(STABLECOIN_IDS)
    btc_eth_pct = df.loc[btc_eth, "value_usd"].sum()/tot*100
    stable_pct  = df.loc[stable,  "value_usd"].sum()/tot*100
    alt_pct     = 100 - btc_eth_pct - stable_pct
    df["abs_24h"] = df["price_change_percentage_24h"].abs().fillna(0.0)
    wv = (df["abs_24h"]*df["value_usd"]).sum()/tot
    div = min(df[key].nunique(), 12)/12
    w = [.30,.30,.20,.10,.10]
    s = [(100-btc_eth_pct)/100, alt_pct/100, (100-stable_pct)/100, 1-div, min(1,wv/50)]
    score = max(0, min(100, sum(a*b for a,b in zip(w,s))*100))
    return score, dict(
        btc_eth_pct=btc_eth_pct, stable_pct=stable_pct, alt_pct=alt_pct,
        weighted_vol_24h_pct=wv, num_assets=df[key].nunique()
    )

def profile_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix":" / 100"},
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":"#10b981"},
               "steps":[{"range":[0,34],"color":"#d1fae5"},
                        {"range":[34,66],"color":"#fde68a"},
                        {"range":[66,100],"color":"#fecaca"}]}
    ))
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0)); return fig

def insights_text(df, comps):
    hhi = ((df["pct_portfolio"]/100.0)**2).sum()
    conc = "well-diversified" if hhi < 0.12 else "moderately concentrated" if hhi < 0.18 else "highly concentrated"
    top3 = df.head(3)["pct_portfolio"].sum()
    return (
        f"• Top 3 = **{top3:.1f}%**, {conc}.\n"
        f"• BTC+ETH **{comps['btc_eth_pct']:.1f}%**, Stable **{comps['stable_pct']:.1f}%**, Alts **{comps['alt_pct']:.1f}%**."
    )

# ——— PDF (images optional) ———
def build_pdf(score, pf_type, insights, alloc_img_path=None, strength_img_path=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Sovereign Stash — Portfolio Report</b>", styles["Title"]), Spacer(1,6),
        Paragraph(f"Profile Score: {round(score,1)} / 100 ({pf_type})", styles["Normal"]), Spacer(1,8),
        Paragraph(insights.replace("\n","<br/>"), styles["Normal"]), Spacer(1,10)
    ]
    if alloc_img_path and os.path.exists(alloc_img_path):
        story += [Image(alloc_img_path, width=480, height=320), Spacer(1,8)]
    if strength_img_path and os.path.exists(strength_img_path):
        story += [Image(strength_img_path, width=480, height=320), Spacer(1,8)]
    story += [Paragraph("<i>"+DISCLAIMER+"</i>", styles["Italic"])]
    doc.build(story); buf.seek(0); return buf.getvalue()

# ——— Main ———
def main():
    # Theme
    with st.sidebar:
        theme = st.selectbox("Theme", ["Dark","Light"], index=0)
    set_plotly_theme(theme); inject_css(theme)

    # Centered hero
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    try: st.image("assets/logo_ss.png", width=130)
    except: pass
    st.markdown("<h1>Sovereign Stash</h1><h3>Crypto Portfolio Navigator</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    # Input
    st.markdown('<div class="glass-card"><b>1) Paste or Import your portfolio</b>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    default_text = """BTC: 0.5
ETH: 2
ADA: 10000
USDT: 1500"""
    text = c1.text_area("Manual entry (one per line like `BTC: 0.5`)", default_text, height=260)
    uploaded = c2.file_uploader("Upload CSV/XLSX (columns: Token, Amount)", type=["csv","xlsx"])
    imported = []
    if uploaded:
        raw = uploaded.read()
        if uploaded.name.lower().endswith(".csv"): imported = ingest.parse_csv_bytes(raw)
        else: imported = ingest.parse_xlsx_bytes(raw)
    fold = c2.checkbox("Fold derivatives (e.g., wBTC→BTC, stETH→ETH)", True)
    analyze = st.button("Analyze Portfolio", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    def parse_lines(txt: str):
        out=[]
        for line in txt.splitlines():
            line=line.strip()
            if not line: continue
            # split by ':' or whitespace
            parts=re.split(r"[:,\s]+", line)
            if len(parts)>=2:
                tok=parts[0]
                try:
                    qty=float(parts[-1].replace(",",""))
                    out.append((tok, qty))
                except: pass
        return out

    if not analyze:
        st.stop()

    rows = parse_lines(text) + imported
    if not rows:
        st.error("No valid holdings found."); st.stop()

    # Enrich with market data
    s2id = build_symbol_to_id_map()
    ids = [symbol_to_id(t, s2id) for t,_ in rows]
    market = fetch_market_data(list(set(ids)))
    m = {x["id"]: x for x in market}
    df = pd.DataFrame([{
        "token": t,
        "id": symbol_to_id(t, s2id),
        "quantity": q,
        "price_usd": m.get(symbol_to_id(t, s2id), {}).get("current_price", 0.0),
        "price_change_percentage_24h": m.get(symbol_to_id(t, s2id), {}).get("price_change_percentage_24h", 0.0),
        "name": m.get(symbol_to_id(t, s2id), {}).get("name", t),
        "symbol": m.get(symbol_to_id(t, s2id), {}).get("symbol", t).upper()
    } for t,q in rows])

    df["value_usd"] = df["quantity"] * df["price_usd"]
    df["folded_id"] = df["id"].apply(lambda x: FOLD_TO_BASE.get(x, x) if fold else x)
    total = float(df["value_usd"].sum())
    df["pct_portfolio"] = df["value_usd"]/total*100 if total>0 else 0.0

    score, comps = calculate_profile_score(df)
    pf_type = "Conservative" if score < 34 else "Balanced" if score < 66 else "Growth-oriented"

    # Current Profile
    st.markdown('<div class="glass-card"><b>2) Current Profile</b>', unsafe_allow_html=True)
    st.dataframe(
        df[["symbol","name","quantity","price_usd","value_usd","pct_portfolio","price_change_percentage_24h"]]
        .style.format({"price_usd":"${:,.4f}","value_usd":"${:,.2f}","pct_portfolio":"{:.2f}%","price_change_percentage_24h":"{:+.2f}%"}),
        use_container_width=True
    )
    st.plotly_chart(profile_gauge(score), use_container_width=True)
    insights = insights_text(df, comps)
    st.markdown(insights)
    st.markdown("</div>", unsafe_allow_html=True)

    # Allocation pie (also try to export, but don't fail if kaleido missing)
    fig_alloc = px.pie(df, names="symbol", values="pct_portfolio", hole=0.35)
    fig_alloc.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_alloc, use_container_width=True)
    alloc_img = None
    try:
        alloc_img = "alloc_chart.png"
        fig_alloc.write_image(alloc_img, width=800, height=500)
    except Exception:
        alloc_img = None  # skip embedding if kaleido not available

    # Relative to BTC (Strength)
    st.markdown('<div class="glass-card"><b>Relative to BTC (Strength)</b>', unsafe_allow_html=True)
    syms = [s for s in df["symbol"].unique() if s != "BTC"]
    sel = st.multiselect("Select tokens", options=syms, default=syms[:5])

    # date_input may return a single date or a tuple; normalize to (start, end)
    dr_val = st.date_input("Date range", (date.today()-timedelta(days=365), date.today()),
                           min_value=date.today()-timedelta(days=5*365))
    if isinstance(dr_val, tuple):
        start_date, end_date = dr_val
    else:
        end_date = dr_val
        start_date = end_date - timedelta(days=365)

    # ensure start < end; if equal, expand 14d window
    if start_date >= end_date:
        start_date = end_date - timedelta(days=14)

    smooth = st.checkbox("Smooth 7d", True)
    logy   = st.checkbox("Log scale Y", False)

    strength_img = None
    coverage_rows = []

    if sel:
        s2id = build_symbol_to_id_map()
        btc_id = "bitcoin"
        start_ts = int(time.mktime(datetime.combine(start_date, datetime.min.time()).timetuple()))
        end_ts   = int(time.mktime(datetime.combine(end_date,   datetime.min.time()).timetuple()))
        now_ts = int(time.time())
        start_ts = max(0, min(start_ts, now_ts-60))
        end_ts   = max(0, min(end_ts,   now_ts))

        btc = fetch_history_usd_range(btc_id, start_ts, end_ts)
        lines = []
        if btc.empty:
            st.info("BTC data unavailable for the chosen range.")
        else:
            btc = btc.sort_values("date")
            for s in sel:
                cid = PREFERRED_COINS.get(s.lower()) or symbol_to_id(s, s2id)
                hist = fetch_history_usd_range(cid, start_ts, end_ts).sort_values("date")
                if hist.empty:
                    st.info(f"Skipping {s}: no data in range.")
                    continue
                # robust nearest-date merge (3d tolerance)
                dfj = pd.merge_asof(
                    hist, btc, on="date", tolerance=pd.Timedelta("3D"),
                    direction="nearest", suffixes=("_tok","_btc")
                )
                # if BTC still NaN at some rows, drop those rows for ratio
                dfj = dfj[dfj["price_usd_btc"].notna()]
                if dfj.empty:
                    st.info(f"Skipping {s}: couldn't align with BTC data.")
                    continue
                dfj["strength"] = dfj["price_usd_tok"] / dfj["price_usd_btc"]
                base = dfj["strength"].dropna()
                base = base.iloc[0] if not base.empty else None
                dfj["strength_idx"] = (dfj["strength"]/base)*100 if base and base>0 else None
                if smooth:
                    dfj["strength_idx"] = dfj["strength_idx"].rolling(7, min_periods=1).mean()
                dfj["symbol"] = s
                # coverage
                coverage_rows.append((s, str(dfj["date"].min().date()), str(dfj["date"].max().date())))
                lines.append(dfj[["date","symbol","strength_idx"]])

            if lines:
                rel = pd.concat(lines, ignore_index=True).sort_values("date")
                fig = px.line(rel, x="date", y="strength_idx", color="symbol",
                              labels={"strength_idx":"Strength vs BTC (start=100)","date":"Date"})
                if logy:
                    fig.update_yaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)
                try:
                    strength_img = "strength_chart.png"
                    fig.write_image(strength_img, width=800, height=500)
                except Exception:
                    strength_img = None
            else:
                st.info("No usable data for selected tokens/date window.")
    if coverage_rows:
        st.dataframe(pd.DataFrame(coverage_rows, columns=["Token","Start","End"]),
                     use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Export
    st.markdown('<div class="glass-card"><b>Export Report</b>', unsafe_allow_html=True)
    pdf_bytes = build_pdf(score, pf_type, insights, alloc_img, strength_img)
    st.download_button("Download PDF Report", data=pdf_bytes,
                       file_name="sovereign_stash_report.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
