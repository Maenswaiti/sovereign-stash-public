# --- app.py (Sovereign Stash — Crypto Portfolio Navigator, public no-auth) ---
import os, json, re, hashlib
from datetime import datetime
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from utils import ingest

st.set_page_config(page_title="Sovereign Stash – Crypto Portfolio Navigator", page_icon="🛰️", layout="wide")

# ===== OpenAI (v1 client) =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ===== Data constants =====
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
STABLECOIN_IDS = {"tether","usd-coin","binance-usd","dai","true-usd","usdd","frax"}
STABLECOIN_SYMBOL_HINTS = {"USDT":"tether","USDC":"usd-coin","BUSD":"binance-usd","DAI":"dai","TUSD":"true-usd"}
FOLD_TO_BASE = {
    "wbtc":"bitcoin","btc.b":"bitcoin",
    "weth":"ethereum","steth":"ethereum","reth":"ethereum","cbeth":"ethereum","frxeth":"ethereum"
}
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

DISCLAIMER_SHORT = (
    "Educational purposes only — not financial or investment advice. "
    "Do your own research and consult a licensed professional. "
    "Crypto assets are volatile and you can lose money."
)

# ===== Theme system (Dark / Light with green accents) =====
def set_plotly_theme(theme: str):
    if theme == "Dark":
        pio.templates["ss_dark"] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111827",  # dark gray (Tailwind gray-900)
                font=dict(color="#F3F4F6"),
                colorway=["#10b981", "#eab308", "#22a2ee", "#9333ea", "#64748b"],
                xaxis=dict(gridcolor="#1f2937"),  # gray-800
                yaxis=dict(gridcolor="#1f2937"),
            )
        )
        pio.templates.default = "ss_dark"
        px.defaults.template = "plotly_dark"
    else:
        pio.templates["ss_light"] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a"),
                colorway=["#10b981", "#0ea5e9", "#eab308", "#9333ea", "#64748b"],
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb"),
            )
        )
        pio.templates.default = "ss_light+plotly_white"
        px.defaults.template = "plotly_white"

def inject_css(theme: str):
    # Dark now uses deep gray (not black) for better contrast; text is light
    if theme == "Dark":
        bg = "#111827"      # gray-900
        text = "#F3F4F6"    # gray-100
        muted = "#9CA3AF"   # gray-400
        card_bg = "linear-gradient(180deg, rgba(31,41,55,0.92), rgba(31,41,55,0.72))"  # gray-800
        border = "rgba(255,255,255,0.10)"
        btn_bg = "#10b981"; btn_bg_hover = "#059669"; btn_text = "#0b1220"
        accent = "#10b981"
        zero_line = "#9CA3AF"
    else:
        bg = "#f7fafc"
        text = "#0f172a"
        muted = "#475569"
        card_bg = "#ffffff"
        border = "rgba(15,23,42,0.10)"
        btn_bg = "#10b981"; btn_bg_hover = "#059669"; btn_text = "#ffffff"
        accent = "#10b981"
        zero_line = "#94a3b8"

    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
      html, body, [data-testid="stAppViewContainer"] {{
        background: {bg};
        color: {text};
        font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }}
      .hero {{ text-align:center; padding: 22px 10px 10px; }}
      .hero h1 {{ font-size: 38px; font-weight: 800; margin: 8px 0 0; }}
      .hero h2 {{ font-size: 22px; font-weight: 700; margin: 0; opacity: .95; }}
      .hero p {{ color: {muted}; }}

      .card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px; padding: 18px; margin: 14px 0;
      }}

      h1, h2, .stMarkdown h2, .stMarkdown h3 {{ color: {text}; }}
      .stMarkdown h2 {{ font-size: 22px; font-weight: 700; }}
      .stMarkdown h3 {{ font-size: 18px; font-weight: 600; }}

      .accent-left {{ border-left:4px solid {accent}; }}
      .pill {{ display:inline-block; padding:3px 10px; border-radius:999px; background:{accent}; color:#0b1220; font-weight:700; }}

      /* Buttons (action + download) */
      .stButton>button, .stDownloadButton>button {{
        background: {btn_bg} !important;
        color: {btn_text} !important;
        border: 0 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(16,185,129,0.30) !important;
      }}
      .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {btn_bg_hover} !important;
      }}
      label[for^="stFileUpload"] {{ font-weight: 600; }}
      [data-testid="stDataFrame"] {{
        border-radius: 12px; overflow: hidden; border:1px solid {border};
      }}
      .zero-line {{ color: {zero_line}; }}
    </style>
    """, unsafe_allow_html=True)

# ===== Market data helpers =====
@st.cache_data(ttl=3600)
def fetch_coin_list():
    r = requests.get(f"{COINGECKO_BASE}/coins/list", timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def build_symbol_to_id_map():
    m = {}
    for c in fetch_coin_list():
        sym = (c.get("symbol") or "").lower()
        cid = c.get("id")
        if sym and sym not in m:
            m[sym] = cid
        name = (c.get("id") or "").lower()
        if name and name not in m:
            m[name] = cid
    return m

def symbol_to_id(symbol: str, s2id_map: Dict[str,str]) -> str:
    if not symbol: return None
    s = symbol.strip().lower()
    if s.upper() in STABLECOIN_SYMBOL_HINTS:
        return STABLECOIN_SYMBOL_HINTS[s.upper()]
    if s in PREFERRED_COINS:
        return PREFERRED_COINS[s]
    if s in s2id_map:
        return s2id_map[s]
    s_clean = re.sub(r"[^a-z0-9\\-]", "", s)
    if s_clean in s2id_map:
        return s2id_map[s_clean]
    if s in ("btc","xbt"):
        return "bitcoin"
    return s

def fetch_market_data(ids: List[str]):
    if not ids: return []
    CHUNK=50; out=[]
    for i in range(0,len(ids),CHUNK):
        chunk=",".join(ids[i:i+CHUNK])
        url=f"{COINGECKO_BASE}/coins/markets"
        params={"vs_currency":"usd","ids":chunk,"order":"market_cap_desc","per_page":250,"page":1,"price_change_percentage":"24h"}
        r=requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        out.extend(r.json())
    return out

# ===== Profile scoring & insights =====
def calculate_profile_score(df: pd.DataFrame):
    total = df["value_usd"].sum()
    if total <= 0: return 0.0, {}
    key_col = "folded_id" if "folded_id" in df.columns else "id"
    btc_eth_mask = df[key_col].isin(["bitcoin","ethereum"])
    stable_mask  = df[key_col].isin(STABLECOIN_IDS)
    btc_eth_pct = df.loc[btc_eth_mask, "value_usd"].sum()/total*100
    stable_pct  = df.loc[stable_mask, "value_usd"].sum()/total*100
    alt_pct     = 100.0 - btc_eth_pct - stable_pct
    if "price_change_percentage_24h" in df.columns:
        df["abs_24h"] = df["price_change_percentage_24h"].abs().fillna(0)
        weighted_vol = (df["abs_24h"]*df["value_usd"]).sum()/total
    else:
        weighted_vol=0.0
    diversification_score = min(df[key_col].nunique(), 12)/12.0
    w_btc_eth=0.30; w_alt=0.30; w_stable=0.20; w_div=0.10; w_vol=0.10
    s_btc_eth = max(0,min(1,(100.0-btc_eth_pct)/100.0))
    s_alt     = max(0,min(1, alt_pct/100.0))
    s_stable  = max(0,min(1,(100.0-stable_pct)/100.0))
    s_div     = 1.0 - diversification_score
    s_vol     = max(0,min(1, weighted_vol/50.0))
    raw = w_btc_eth*s_btc_eth + w_alt*s_alt + w_stable*s_stable + w_div*s_div + w_vol*s_vol
    score=float(max(0,min(100, raw*100.0)))
    comps={"btc_eth_pct": round(btc_eth_pct,2),"stable_pct": round(stable_pct,2),"alt_pct": round(alt_pct,2),
           "weighted_vol_24h_pct": round(weighted_vol,2),"num_assets": int(df[key_col].nunique()),"raw_score": raw}
    return score, comps

def portfolio_insights(df_display: pd.DataFrame, comps: Dict) -> str:
    hhi = float(((df_display["pct_portfolio"]/100.0)**2).sum())
    if   hhi < 0.12: conc = "well-diversified"
    elif hhi < 0.18: conc = "moderately concentrated"
    else: conc = "highly concentrated"
    top3 = df_display.head(3)["pct_portfolio"].sum()
    lines = [
        f"Top 3 positions = **{top3:.1f}%** → {conc}.",
        f"BTC+ETH **{comps['btc_eth_pct']:.1f}%**, Stablecoins **{comps['stable_pct']:.1f}%**, Alts **{comps['alt_pct']:.1f}%**.",
        f"Weighted 24h move **{comps['weighted_vol_24h_pct']:.2f}%** (volatility proxy).",
        f"Unique assets tracked: **{comps['num_assets']}**."
    ]
    return "• " + "\n• ".join(lines)

def generate_ai_summary(metrics: Dict, top_assets: List[Tuple[str,float]]) -> str:
    # robust fallback (shown if no key or API unavailable)
    fallback = [
        f"Profile Score: {metrics.get('risk_score','N/A')} / 100",
        f"BTC+ETH: {metrics.get('btc_eth_pct','N/A')}%",
        f"Stablecoins: {metrics.get('stable_pct','N/A')}%",
        f"Alts: {metrics.get('alt_pct','N/A')}%",
        f"Weighted 24h move (volatility proxy): {metrics.get('weighted_vol_24h_pct','N/A')}%",
        "", "Top holdings:"
    ]
    for t,p in top_assets[:5]:
        fallback.append(f"- {t}: {round(p,2)}%")
    fallback_text = "\n".join(fallback)

    if not client:
        return fallback_text + "\n\n(Add OPENAI_API_KEY in Streamlit Secrets to enable natural-language notes.)"

    prompt = (
        "You are a concise crypto portfolio explainer for a general audience. "
        "Summarize the current portfolio profile and provide two practical, non-financial-advice observations.\n\n"
        "Metrics:\n" + json.dumps(metrics, indent=2) + "\nTop holdings:\n" +
        "\n".join([f"- {t}: {round(p,2)}%" for t,p in top_assets[:8]]) +
        "\n\nOutput: short summary + a one-word Profile label + two bullet observations."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if ("insufficient_quota" in msg) or ("You exceeded your current quota" in msg) or ("code: 429" in msg):
            return "AI temporarily unavailable (quota exceeded).\n\n" + fallback_text
        return f"AI notes unavailable: {e}\n\n" + fallback_text

def profile_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix":" / 100"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#10b981"},
            "steps":[
                {"range":[0,34],"color":"#d1fae5"},
                {"range":[34,66],"color":"#fde68a"},
                {"range":[66,100],"color":"#fecaca"}
            ],
            "threshold":{"line":{"color":"#eab308","width":4},"thickness":0.75,"value":score}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ===== Reporting (PDF) =====
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
def build_pdf_report(total_value, profile_score, pf_type, comps_out, ai_summary, top_table):
    buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=LETTER, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet(); story = []
    story += [Paragraph("<b>Sovereign Stash — Crypto Portfolio Navigator</b>", styles['Title']), Spacer(1,6)]
    story += [Paragraph(f"Generated: {datetime.utcnow().isoformat()}Z", styles['Normal']), Spacer(1,8)]
    story += [Paragraph(f"<b>Total Value:</b> ${total_value:,.2f}", styles['Normal'])]
    story += [Paragraph(f"<b>Profile Score:</b> {profile_score:.2f} / 100 ({pf_type})", styles['Normal']), Spacer(1,8)]
    km = [["BTC+ETH", f"{comps_out['btc_eth_pct']}%"],["Stablecoins", f"{comps_out['stable_pct']}%"],["Alts", f"{comps_out['alt_pct']}%"],
          ["Weighted 24h move (proxy)", f"{comps_out['weighted_vol_24h_pct']}%"],["Unique assets", f"{comps_out['num_assets']}"]]
    tbl = Table([["Metric","Value"]]+km, hAlign='LEFT', colWidths=[180,220])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0), colors.HexColor('#0A6D62')),
        ('TEXTCOLOR',(0,0),(1,0), colors.whitesmoke),
        ('FONTNAME',(0,0),(1,0),'Helvetica-Bold'),
        ('INNERGRID',(0,0),(-1,-1),0.25, colors.grey),
        ('BOX',(0,0),(-1,-1),0.5, colors.grey)
    ]))
    story += [tbl, Spacer(1,10)]
    if top_table:
        t = Table([["Symbol","% Portfolio"]] + top_table[:10], hAlign='LEFT', colWidths=[180,220])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(1,0), colors.HexColor('#E7B622')),
            ('TEXTCOLOR',(0,0),(1,0), colors.black),
            ('FONTNAME',(0,0),(1,0),'Helvetica-Bold'),
            ('INNERGRID',(0,0),(-1,-1),0.25, colors.grey),
            ('BOX',(0,0),(-1,-1),0.5, colors.grey)
        ]))
        story += [Paragraph("<b>Top Holdings</b>", styles['Heading3']), t, Spacer(1,10)]
    story += [Paragraph("<b>AI Portfolio Notes</b>", styles['Heading3']), Paragraph(ai_summary.replace("\n","<br/>"), styles['Normal']), Spacer(1,10)]
    story += [Paragraph("<font size=9>"+DISCLAIMER_SHORT+"</font>", styles['Normal'])]
    doc.build(story); pdf = buf.getvalue(); buf.close(); return pdf

# ===== Session utilities =====
def compute_inputs_hash(text: str, imported_rows: List[Tuple[str,float]], fold: bool) -> str:
    payload = json.dumps({"text": text.strip(), "imported": imported_rows, "fold": fold}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def recompute_analysis(holdings: List[Tuple[str,float]], fold_derivs: bool):
    s2id = build_symbol_to_id_map()
    ids=set(); rows=[]
    for token, qty in holdings:
        cid = symbol_to_id(token, s2id)
        ids.add(cid)
        rows.append({"token_input":token.strip(), "id":cid, "quantity":qty})
    market = fetch_market_data(list(ids))
    m_by_id = {m["id"]:m for m in market}
    enriched=[]
    for r in rows:
        cid = r["id"]; m = m_by_id.get(cid, {})
        price = m.get("current_price", 0.0); change24 = m.get("price_change_percentage_24h", 0.0)
        name = m.get("name", cid or r["token_input"]); symbol = (m.get("symbol") or r["token_input"]).upper()
        val = price * r["quantity"]
        enriched.append({"token_input":r["token_input"],"id":cid,"name":name,"symbol":symbol,
                         "quantity":r["quantity"],"price_usd":price,"value_usd":val,"price_change_percentage_24h":change24})
    df = pd.DataFrame(enriched)
    df["value_usd"]=df["value_usd"].astype(float)
    if fold_derivs: df['folded_id'] = df['id'].apply(lambda x: FOLD_TO_BASE.get((x or '').lower(), x))
    else:           df['folded_id'] = df['id']
    total_value = float(df["value_usd"].sum())
    if total_value <= 0: return None
    df["pct_portfolio"] = df["value_usd"]/total_value*100.0
    df_display = df[["symbol","name","quantity","price_usd","value_usd","pct_portfolio","price_change_percentage_24h"]].sort_values("value_usd", ascending=False)
    profile_score, comps = calculate_profile_score(df)
    comps_out = {
        "risk_score": round(profile_score,2),  # key name reused internally
        "btc_eth_pct": comps.get("btc_eth_pct"),
        "stable_pct": comps.get("stable_pct"),
        "alt_pct": comps.get("alt_pct"),
        "weighted_vol_24h_pct": comps.get("weighted_vol_24h_pct"),
        "num_assets": comps.get("num_assets")
    }
    metrics_for_ai = dict(comps_out); metrics_for_ai["total_value_usd"]=round(total_value,2)
    top_assets = list(zip(df_display["symbol"], df_display["pct_portfolio"]))
    pf_type = "Conservative"
    if profile_score >= 66: pf_type = "Growth-oriented"
    elif profile_score >= 34: pf_type = "Balanced"
    return {
        "df_display": df_display,
        "total_value": total_value,
        "risk_score": profile_score,
        "comps_out": comps_out,
        "metrics_for_ai": metrics_for_ai,
        "top_assets": top_assets,
        "pf_type": pf_type
    }

# ===== App =====
def main():
    # Theme picker (sidebar)
    with st.sidebar:
        st.markdown("### Appearance")
        theme = st.selectbox("Theme", ["Dark","Light"], index=0, key="theme_select")
        st.caption("Tip: switch themes any time. Your choice persists for this session.")

    set_plotly_theme(theme)
    inject_css(theme)

    # Hero with logo
    with st.container():
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            try:
                st.image("assets/logo_ss.png", width=120)
            except Exception:
                st.write(" ")
            st.markdown(
                """
                <div class="hero">
                  <h1>Sovereign Stash</h1>
                  <h2>Crypto Portfolio Navigator</h2>
                  <p>No login. No data saved. Paste or import your holdings for a live portfolio profile view.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div class='card accent-left'><b>Disclaimer:</b> {DISCLAIMER_SHORT}</div>",
        unsafe_allow_html=True
    )

    # --- Input / Analyze (writes frozen state) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Paste or Import your portfolio")

    colA, colB = st.columns([1,1])
    with colA:
        default_text = "BTC: 0.5\nETH: 2.0\nADA: 10000\nUSDT: 1500"
        portfolio_text = st.text_area(
            "Manual entry (one per line, e.g. `BTC: 0.5`):",
            value=st.session_state.get("last_text", default_text),
            height=280,
            key="text_input_area"
        )
    imported_rows = []
    with colB:
        st.write("Or upload **CSV / XLSX / JSON** with columns **Token** and **Amount**.")
        f = st.file_uploader("Upload holdings file", type=["csv","xlsx","xls","json"], key="file_upl")
        if f is not None:
            raw = f.read()
            if f.name.lower().endswith(".csv"): imported_rows = ingest.parse_csv_bytes(raw)
            elif f.name.lower().endswith((".xlsx",".xls")): imported_rows = ingest.parse_xlsx_bytes(raw)
            elif f.name.lower().endswith(".json"): imported_rows = ingest.parse_json_bytes(raw)
            else: imported_rows = ingest.parse_csv_bytes(raw)
            if imported_rows: st.success(f"Imported {len(imported_rows)} rows.")
            else: st.warning("No rows detected. Use columns Token / Amount or a known exchange export.")

    with st.expander("Advanced normalization & mapping"):
        st.caption("Optionally fold well-known derivatives into base assets for profile grouping (e.g., wBTC→BTC, stETH→ETH).")
        fold_derivs = st.checkbox("Fold derivatives into base assets for grouping", value=st.session_state.get("fold_derivs", True), key="fold_checkbox")

    c1, c2 = st.columns([1,1])
    analyze_clicked = c1.button("Analyze Portfolio", use_container_width=True, type="primary")
    reset_clicked   = c2.button("Reset Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # parse helper
    def parse_portfolio(text: str):
        items = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line: continue
            sep = None
            for s in [":", ","]:
                if s in line: sep=s; break
            parts = [p.strip() for p in line.split(sep)] if sep else line.split()
            if len(parts) < 2: continue
            token = parts[0]
            try: qty = float(parts[-1].replace(",",""))
            except Exception: continue
            items.append((token, qty))
        return items

    if reset_clicked:
        for k in ["frozen_hash","frozen_holdings","frozen_result","last_text","fold_derivs","hypo_targets","hypo_asset_weights"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

    if analyze_clicked:
        manual_rows = parse_portfolio(portfolio_text)
        all_rows = manual_rows + imported_rows
        if not all_rows:
            st.error("No valid holdings parsed.")
            st.stop()
        h = compute_inputs_hash(portfolio_text, imported_rows, fold_derivs)
        res = recompute_analysis(all_rows, fold_derivs)
        if res is None:
            st.error("Total value is zero.")
            st.stop()
        st.session_state["frozen_hash"] = h
        st.session_state["frozen_holdings"] = all_rows
        st.session_state["frozen_result"] = res
        st.session_state["last_text"] = portfolio_text
        st.session_state["fold_derivs"] = fold_derivs
        # initialize hypothetical per-asset weights from current pct_portfolio
        cur_df = res["df_display"]
        st.session_state["hypo_asset_weights"] = {sym: float(pct) for sym, pct in zip(cur_df["symbol"], cur_df["pct_portfolio"])}

    if "frozen_result" not in st.session_state:
        st.info("Enter tokens and click **Analyze Portfolio** to see your portfolio profile.")
        st.stop()

    # --- Tabs: Current vs What-If vs Methodology ---
    tab_current, tab_hypo, tab_method = st.tabs(["📊 Current Profile", "🧪 What-If (Hypothesis)", "ℹ️ Methodology"])

    # ===== CURRENT PROFILE (frozen snapshot) =====
    with tab_current:
        res = st.session_state["frozen_result"]
        df_display = res["df_display"]
        total_value = res["total_value"]
        profile_score = res["risk_score"]
        comps_out = res["comps_out"]
        metrics_for_ai = res["metrics_for_ai"]
        top_assets = res["top_assets"]
        pf_type = res["pf_type"]

        ai_notes = generate_ai_summary(metrics_for_ai, top_assets)

        left, right = st.columns([2,1])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("2) Portfolio Overview")
            st.dataframe(
                df_display.style.format({
                    "price_usd":"${:,.4f}","value_usd":"${:,.2f}",
                    "pct_portfolio":"{:.2f}%","price_change_percentage_24h":"{:+.2f}%"
                }),
                height=360, use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Allocation by Asset")
            fig_alloc = px.pie(df_display, names="symbol", values="pct_portfolio", hole=0.35)
            fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
            fig_alloc.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_alloc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("24h % Change by Asset")
            tmp = df_display.copy()
            tmp["change"] = tmp["price_change_percentage_24h"].fillna(0.0)
            tmp = tmp.sort_values("change", ascending=False)
            tmp["direction"] = tmp["change"].apply(lambda v: "Up" if v >= 0 else "Down")
            fig_bar = px.bar(
                tmp, x="symbol", y="change", color="direction",
                color_discrete_map={"Up":"#10b981","Down":"#ef4444"},
                labels={"change":"24h %","symbol":"Asset","direction":""},
                hover_data={"change":":.2f","symbol":True,"direction":False}
            )
            fig_bar.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            fig_bar.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#9CA3AF")  # dark gray zero line
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Green = positive 24h, red = negative. Axis centered at 0% with a bold zero line.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Key Insights")
            st.markdown(portfolio_insights(df_display, comps_out))
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card center">', unsafe_allow_html=True)
            st.subheader("Profile Gauge")
            st.plotly_chart(profile_gauge(profile_score), use_container_width=True)
            st.markdown(f"**Profile Type:** {pf_type}")
            st.markdown('</div>', unsafe_allow_html=True)

            if client:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("AI Portfolio Notes")
                st.text_area("AI-generated notes", value=ai_notes, height=210)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Download Report")
            html = f"""<html><head><meta charset='utf-8'><title>Sovereign Stash Report</title></head>
            <body style='font-family:Arial;background:#ffffff;color:#0f172a;'>
            <h2>Sovereign Stash — Crypto Portfolio Navigator</h2>
            <p>Generated: {datetime.utcnow().isoformat()}Z</p>
            <p>Total Value: ${total_value:,.2f}</p>
            <p>Profile Score: {profile_score:.2f} / 100 ({pf_type})</p>
            <h3>Key Metrics</h3>
            <ul>
              <li>BTC+ETH: {comps_out['btc_eth_pct']}%</li>
              <li>Stablecoins: {comps_out['stable_pct']}%</li>
              <li>Alts: {comps_out['alt_pct']}%</li>
              <li>Weighted 24h move (proxy): {comps_out['weighted_vol_24h_pct']}%</li>
              <li>Unique assets: {comps_out['num_assets']}</li>
            </ul>
            <h3>AI Portfolio Notes</h3>
            <pre style="white-space:pre-wrap;">{ai_notes}</pre>
            <hr><p style='font-size:12px;opacity:0.8'>{DISCLAIMER_SHORT}</p>
            </body></html>"""
            pdf = build_pdf_report(
                total_value, profile_score, pf_type, comps_out,
                ai_notes,
                list(zip(df_display['symbol'].tolist(), [f"{v:.2f}%" for v in df_display['pct_portfolio'].tolist()]))
            )
            st.download_button("Download PDF (1 page)", data=pdf, file_name="sovereign_stash_report.pdf", mime="application/pdf")
            st.download_button("Download HTML (print to PDF)", data=html, file_name="sovereign_stash_report.html", mime="text/html")
            st.markdown('</div>', unsafe_allow_html=True)

    # ===== WHAT-IF / HYPOTHESIS (isolated; no recompute of current) =====
    with tab_hypo:
        res = st.session_state["frozen_result"]
        comps_out = res["comps_out"]
        cur_df = res["df_display"].copy()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What-If (Hypothesis) — Group Targets")
        st.caption("Set BTC/ETH and Stablecoins; Alts is auto-computed so the three always sum to 100%.")

        # Two sliders; third is computed to enforce 100%
        col_g1, col_g2, col_g3 = st.columns(3)
        defaults = st.session_state.get("hypo_targets", {
            "btc_eth": int(comps_out['btc_eth_pct']),
            "stable": int(comps_out['stable_pct']),
        })
        # keep previous choices if present
        btc_eth_val = col_g1.slider("BTC+ETH %", 0, 100, defaults["btc_eth"], key="hypo_btc_eth_enforced")
        stable_max = max(0, 100 - btc_eth_val)
        stable_val = col_g2.slider("Stablecoins %", 0, stable_max, min(defaults["stable"], stable_max), key="hypo_stable_enforced")
        alt_val = 100 - btc_eth_val - stable_val
        col_g3.metric("Alts % (auto)", f"{alt_val}%")

        st.session_state["hypo_targets"] = {"btc_eth": btc_eth_val, "stable": stable_val, "alt": alt_val}

        # Render hypothetical group pie
        hypo_group = pd.DataFrame({
            "group": ["BTC/ETH","Stablecoins","Alts"],
            "target": [btc_eth_val, stable_val, alt_val]
        })
        fig_hypo = px.pie(hypo_group, names="group", values="target", hole=0.35, title="Hypothetical Allocation (by group)")
        fig_hypo.update_layout(margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_hypo, use_container_width=True)

        # compute hypothetical profile score using frozen diversification & vol components
        w_btc_eth=0.30; w_alt=0.30; w_stable=0.20; w_div=0.10; w_vol=0.10
        s_btc_eth = max(0,min(1,(100.0 - btc_eth_val)/100.0))
        s_alt     = max(0,min(1, alt_val/100.0))
        s_stable  = max(0,min(1,(100.0 - stable_val)/100.0))
        s_div     = 1.0 - min(comps_out['num_assets'], 12)/12.0
        s_vol     = max(0,min(1, comps_out['weighted_vol_24h_pct']/50.0))
        hypo_score_group = float(max(0,min(100, (w_btc_eth*s_btc_eth + w_alt*s_alt + w_stable*s_stable + w_div*s_div + w_vol*s_vol)*100.0 )))

        st.markdown("**Hypothetical Profile Score (group targets):**")
        st.plotly_chart(profile_gauge(hypo_score_group), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Per-asset hypothetical editor ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What-If (Per Asset)")
        st.caption("Adjust each token’s target %; on submit, totals normalize to 100% and update the charts below.")

        # initialize state if missing
        if "hypo_asset_weights" not in st.session_state:
            st.session_state["hypo_asset_weights"] = {sym: float(pct) for sym, pct in zip(cur_df["symbol"], cur_df["pct_portfolio"])}

        # Build a form to avoid global reruns while editing
        with st.form("asset_hypo_form", clear_on_submit=False):
            cols = st.columns(3)
            symbols = cur_df["symbol"].tolist()
            # divide into thirds for nicer layout
            third = (len(symbols) + 2) // 3
            edited_weights = {}

            def render_column(col, items):
                for sym in items:
                    default = float(st.session_state["hypo_asset_weights"].get(sym, 0.0))
                    edited_weights[sym] = col.number_input(
                        f"{sym} target %", min_value=0.0, max_value=100.0, value=float(round(default,2)), step=0.5, key=f"hypo_{sym}"
                    )

            render_column(cols[0], symbols[:third])
            render_column(cols[1], symbols[third:2*third])
            render_column(cols[2], symbols[2*third:])

            submitted_assets = st.form_submit_button("Update Per-Asset What-If", use_container_width=True)

        # When submitted: normalize to 100
        if submitted_assets:
            total_raw = sum(edited_weights.values())
            if total_raw <= 0:
                # keep old if user zeroed out everything
                st.warning("Total was 0%. Keeping previous targets.")
            else:
                norm = {k: (v/total_raw)*100.0 for k,v in edited_weights.items()}
                st.session_state["hypo_asset_weights"] = norm

        # Display current (possibly normalized) weights and charts
        weights = st.session_state["hypo_asset_weights"]
        # Ensure we include only current symbols (in case user changed inputs earlier)
        weights = {sym: weights.get(sym, 0.0) for sym in symbols}
        # Make a dataframe for the pie chart
        df_asset_hypo = pd.DataFrame({
            "symbol": list(weights.keys()),
            "target_pct": [float(v) for v in weights.values()]
        }).sort_values("target_pct", ascending=False)

        # Show sum & gentle hint
        total_pct = df_asset_hypo["target_pct"].sum()
        col_sum1, col_sum2 = st.columns([3,1])
        col_sum1.caption("Tip: Enter rough numbers; we normalize to 100% on submit.")
        col_sum2.metric("Sum of targets", f"{total_pct:.2f}%")

        # Pie by asset
        fig_asset = px.pie(df_asset_hypo, names="symbol", values="target_pct", hole=0.35, title="Hypothetical Allocation (by asset)")
        fig_asset.update_layout(margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_asset, use_container_width=True)

        # Compute hypothetical group score from per-asset targets for consistency
        # Map each asset to BTC/ETH / Stable / Alt
        key_col = "folded_id" if "folded_id" in cur_df.columns else "name"
        # We'll derive groups from symbol->id via current df
        symbol_to_id_map = dict(zip(cur_df["symbol"], cur_df.get("name", cur_df["symbol"])))
        # For grouping, better to use 'name'->id where possible; but we only need group totals:
        btc_eth_symbols = set(cur_df.loc[cur_df["name"].isin(["Bitcoin","Ethereum"]), "symbol"].tolist())
        stable_symbols = set(cur_df.loc[cur_df["name"].str.lower().str.replace(" ","-").isin(STABLECOIN_IDS), "symbol"].tolist())

        btc_eth_target = sum(df_asset_hypo.loc[df_asset_hypo["symbol"].isin(btc_eth_symbols), "target_pct"])
        stable_target  = sum(df_asset_hypo.loc[df_asset_hypo["symbol"].isin(stable_symbols), "target_pct"])
        alt_target     = max(0.0, 100.0 - btc_eth_target - stable_target)

        # Score using per-asset-derived groups
        s_btc_eth2 = max(0,min(1,(100.0 - btc_eth_target)/100.0))
        s_alt2     = max(0,min(1, alt_target/100.0))
        s_stable2  = max(0,min(1,(100.0 - stable_target)/100.0))
        w_btc_eth=0.30; w_alt=0.30; w_stable=0.20; w_div=0.10; w_vol=0.10
        s_div2    = 1.0 - min(comps_out['num_assets'], 12)/12.0
        s_vol2    = max(0,min(1, comps_out['weighted_vol_24h_pct']/50.0))
        hypo_score_assets = float(max(0,min(100, (w_btc_eth*s_btc_eth2 + w_alt*s_alt2 + w_stable*s_stable2 + w_div*s_div2 + w_vol*s_vol2)*100.0 )))

        st.markdown("**Hypothetical Profile Score (per-asset targets):**")
        st.plotly_chart(profile_gauge(hypo_score_assets), use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== METHODOLOGY (reference) =====
    with tab_method:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Methodology: How the Profile Score is computed")
        st.markdown(
            """
<div style="line-height:1.6">
<span class="pill">Educational Heuristic</span>
<p style="margin-top:8px;">This is a simplified <b>Portfolio Profile</b> score from 0–100 that reflects how growth-oriented vs. conservative a snapshot may look, using observable traits. It is not financial advice.</p>

### Components & Weights
Weights:
<ul>
<li><b>BTC/ETH</b> — 0.30</li>
<li><b>Alts</b> — 0.30</li>
<li><b>Stablecoins</b> — 0.20</li>
<li><b>Diversification</b> — 0.10</li>
<li><b>Volatility proxy (24h)</b> — 0.10</li>
</ul>

We compute shares by USD value:
<ul>
<li><b>BTC+ETH %</b>: value in Bitcoin or Ethereum (after optional folding like wBTC→BTC, stETH→ETH)</li>
<li><b>Stablecoins %</b>: value in major USD-pegged tokens (e.g., USDT, USDC, DAI)</li>
<li><b>Alts %</b> = 100 − (BTC+ETH %) − (Stablecoins %)</li>
</ul>

Component scores (0–1):
<pre style="white-space:pre-wrap;">
s_btc_eth = (100 − BTC_ETH_pct) / 100    # more BTC/ETH ⇒ more conservative
s_alt     = (Alts_pct) / 100             # more alts ⇒ more growth tilt
s_stable  = (100 − Stable_pct) / 100     # more stables ⇒ more conservative
diversification_score = min(N, 12) / 12  # N = unique assets (after folding)
s_div     = 1 − diversification_score
weighted_vol = Σ(|24h_change_i| * value_i) / total_value
s_vol     = min(1, weighted_vol / 50)    # cap at 50%
</pre>

Final score:
<pre style="white-space:pre-wrap;">
raw = 0.30*s_btc_eth + 0.30*s_alt + 0.20*s_stable + 0.10*s_div + 0.10*s_vol
Profile Score = clamp(raw * 100, 0, 100)
</pre>

Profile type labels:
<ul>
<li><b>0–33</b> → Conservative</li>
<li><b>34–65</b> → Balanced</li>
<li><b>66–100</b> → Growth-oriented</li>
</ul>

### Folding derivatives (optional)
To make BTC/ETH share more meaningful, we group certain derivatives under their base asset (e.g., <b>wBTC → BTC</b>, <b>stETH/wETH/reth/cbeth → ETH</b>). This affects grouping only, not raw holdings.

### What-If section
You can adjust **group targets** (BTC/ETH & Stablecoins; Alts auto-fills to 100%) or edit **per-asset targets**. Per-asset values are normalized to 100% on submit. These simulations keep your current diversification count and 24h proxy.
</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# --- end app.py ---
