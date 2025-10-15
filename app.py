# --- app.py (Sovereign Stash — Crypto Risk Radar, public no-auth) ---
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

st.set_page_config(page_title="Sovereign Stash – Crypto Risk Radar", page_icon="🛰️", layout="wide")

# -------- OpenAI (v1 client) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# -------- Data & risk constants --------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
STABLECOIN_IDS = {"tether","usd-coin","binance-usd","dai","true-usd","usdd","frax"}
STABLECOIN_SYMBOL_HINTS = {"USDT":"tether","USDC":"usd-coin","BUSD":"binance-usd","DAI":"dai","TUSD":"true-usd"}
FOLD_TO_BASE = {
    "wbtc":"bitcoin","btc.b":"bitcoin",
    "weth":"ethereum","steth":"ethereum","reth":"ethereum","cbeth":"ethereum","frxeth":"ethereum"
}
# Prefer canonical CoinGecko IDs for common symbols (fixes BTC/ETH/etc.)
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

# -------- Styling (lighter theme) --------
def inject_styles():
    # load external css if present
    try:
        with open("assets/style.css","r",encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass
    # lighten background & surfaces
    st.markdown("""
    <style>
      :root{ --bg:#0e1420; --glass:#121a28cc; --text:#F2F6FF; --muted:#B8C6D9; }
      .card{ background:linear-gradient(180deg, rgba(18,26,40,0.88), rgba(18,26,40,0.65));
             border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:18px; margin:14px 0; }
    </style>
    """, unsafe_allow_html=True)

def set_brand_plotly_lighter():
    pio.templates["ss_light"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
            colorway=["#10b981", "#eab308", "#2563eb", "#9333ea", "#64748b"],
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb"),
        )
    )
    px.defaults.template = "plotly_white"
    pio.templates.default = "ss_light+plotly_white"

# -------- Market data helpers --------
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
    if s.upper() in STABLECOIN_SYMBOL_HINTS:  # stable hints (legacy)
        return STABLECOIN_SYMBOL_HINTS[s.upper()]
    if s in PREFERRED_COINS:                  # canonical majors
        return PREFERRED_COINS[s]
    if s in s2id_map:                          # exact
        return s2id_map[s]
    s_clean = re.sub(r"[^a-z0-9\\-]", "", s)   # remove separators
    if s_clean in s2id_map:
        return s2id_map[s_clean]
    if s in ("btc","xbt"):                     # final guard
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

# -------- Scoring & insights --------
def calculate_risk_score(df: pd.DataFrame):
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
    fallback = [
        f"Risk Score: {metrics.get('risk_score','N/A')} / 100",
        f"BTC+ETH: {metrics.get('btc_eth_pct','N/A')}%",
        f"Stablecoins: {metrics.get('stable_pct','N/A')}%",
        f"Alts: {metrics.get('alt_pct','N/A')}%",
        f"Weighted 24h vol: {metrics.get('weighted_vol_24h_pct','N/A')}%",
        "", "Top holdings:"
    ]
    for t,p in top_assets[:5]: fallback.append(f"- {t}: {round(p,2)}%")
    fallback.append("\nAdd OPENAI_API_KEY in Streamlit Secrets to enable natural language insights.")
    fallback_text = "\n".join(fallback)
    if not client: return fallback_text
    prompt = (
        "You are a concise crypto investment explainer for a general audience. "
        "Summarize the current risk and provide two practical, non-financial-advice suggestions.\n\n"
        "Metrics:\n" + json.dumps(metrics, indent=2) + "\nTop holdings:\n" +
        "\n".join([f"- {t}: {round(p,2)}%" for t,p in top_assets[:8]]) +
        "\n\nOutput: short summary + a one-word Risk label + two bullet suggestions."
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
        return f"AI summary failed: {e}"

def risk_gauge(score: float):
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

# -------- Reporting (PDF) --------
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
def build_pdf_report(total_value, risk_score, pf_type, comps_out, ai_summary, top_table):
    buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=LETTER, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet(); story = []
    story += [Paragraph("<b>Sovereign Stash — Crypto Risk Radar</b>", styles['Title']), Spacer(1,6)]
    story += [Paragraph(f"Generated: {datetime.utcnow().isoformat()}Z", styles['Normal']), Spacer(1,8)]
    story += [Paragraph(f"<b>Total Value:</b> ${total_value:,.2f}", styles['Normal'])]
    story += [Paragraph(f"<b>Risk Score:</b> {risk_score:.2f} / 100 ({pf_type})", styles['Normal']), Spacer(1,8)]
    km = [["BTC+ETH", f"{comps_out['btc_eth_pct']}%"],["Stablecoins", f"{comps_out['stable_pct']}%"],["Altcoins", f"{comps_out['alt_pct']}%"],
          ["Weighted 24h vol", f"{comps_out['weighted_vol_24h_pct']}%"],["Unique assets", f"{comps_out['num_assets']}"]]
    tbl = Table([["Metric","Value"]]+km, hAlign='LEFT', colWidths=[150,200])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0), colors.HexColor('#0A6D62')),
        ('TEXTCOLOR',(0,0),(1,0), colors.whitesmoke),
        ('FONTNAME',(0,0),(1,0),'Helvetica-Bold'),
        ('INNERGRID',(0,0),(-1,-1),0.25, colors.grey),
        ('BOX',(0,0),(-1,-1),0.5, colors.grey)
    ]))
    story += [tbl, Spacer(1,10)]
    if top_table:
        t = Table([["Symbol","% Portfolio"]] + top_table[:10], hAlign='LEFT', colWidths=[150,200])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(1,0), colors.HexColor('#E7B622')),
            ('TEXTCOLOR',(0,0),(1,0), colors.black),
            ('FONTNAME',(0,0),(1,0),'Helvetica-Bold'),
            ('INNERGRID',(0,0),(-1,-1),0.25, colors.grey),
            ('BOX',(0,0),(-1,-1),0.5, colors.grey)
        ]))
        story += [Paragraph("<b>Top Holdings</b>", styles['Heading3']), t, Spacer(1,10)]
    story += [Paragraph("<b>AI Summary</b>", styles['Heading3']), Paragraph(ai_summary.replace("\n","<br/>"), styles['Normal']), Spacer(1,10)]
    story += [Paragraph("<font size=9>"+DISCLAIMER_SHORT+"</font>", styles['Normal'])]
    doc.build(story); pdf = buf.getvalue(); buf.close(); return pdf

# ========= SESSION UTILITIES =========
def compute_inputs_hash(text: str, imported_rows: List[Tuple[str,float]], fold: bool) -> str:
    payload = json.dumps({
        "text": text.strip(),
        "imported": imported_rows,
        "fold": fold
    }, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def recompute_analysis(holdings: List[Tuple[str,float]], fold_derivs: bool):
    # holdings = list of (token, qty)
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
    if fold_derivs:
        df['folded_id'] = df['id'].apply(lambda x: FOLD_TO_BASE.get((x or '').lower(), x))
    else:
        df['folded_id'] = df['id']
    total_value = float(df["value_usd"].sum())
    if total_value <= 0: return None
    df["pct_portfolio"] = df["value_usd"]/total_value*100.0
    df_display = df[["symbol","name","quantity","price_usd","value_usd","pct_portfolio","price_change_percentage_24h"]].sort_values("value_usd", ascending=False)
    risk_score, comps = calculate_risk_score(df)
    comps_out = {
        "risk_score": round(risk_score,2),
        "btc_eth_pct": comps.get("btc_eth_pct"),
        "stable_pct": comps.get("stable_pct"),
        "alt_pct": comps.get("alt_pct"),
        "weighted_vol_24h_pct": comps.get("weighted_vol_24h_pct"),
        "num_assets": comps.get("num_assets")
    }
    metrics_for_ai = dict(comps_out); metrics_for_ai["total_value_usd"]=round(total_value,2)
    top_assets = list(zip(df_display["symbol"], df_display["pct_portfolio"]))
    pf_type = "Conservative"
    if risk_score >= 66: pf_type = "Aggressive"
    elif risk_score >= 34: pf_type = "Balanced"
    return {
        "df_display": df_display,
        "total_value": total_value,
        "risk_score": risk_score,
        "comps_out": comps_out,
        "metrics_for_ai": metrics_for_ai,
        "top_assets": top_assets,
        "pf_type": pf_type
    }

# -------- App --------
def main():
    inject_styles()
    set_brand_plotly_lighter()

    st.markdown("""
    <div class="hero">
      <img src="assets/logo_ss.png" class="logo" />
      <h1>Sovereign Stash</h1>
      <h2>Crypto Risk Radar</h2>
      <p>No login. No data saved. Paste or import your holdings and get a live risk view.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div class='card' style='border-left:4px solid #E7B622;'><b>Disclaimer:</b> {DISCLAIMER_SHORT}</div>",
        unsafe_allow_html=True
    )

    # -------- Input / Analyze (writes frozen state) --------
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
        st.caption("Optionally fold well-known derivatives into base assets for risk grouping (e.g., wBTC→BTC, stETH→ETH).")
        fold_derivs = st.checkbox("Fold derivatives into base assets for risk grouping", value=st.session_state.get("fold_derivs", True), key="fold_checkbox")

    col_btn1, col_btn2 = st.columns([1,1])
    analyze_clicked = col_btn1.button("Analyze Portfolio", use_container_width=True, type="primary")
    reset_clicked   = col_btn2.button("Reset Analysis", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # helper: parse manual text into (token, qty)
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

    # Reset clears frozen state
    if reset_clicked:
        for k in ["frozen_hash","frozen_holdings","frozen_result","last_text","fold_derivs"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()

    # If Analyze clicked -> recompute and freeze
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

    # If we have no frozen analysis yet, stop here
    if "frozen_result" not in st.session_state:
        st.info("Enter tokens and click **Analyze Portfolio** to see results.")
        st.stop()

    # ========= TABS: Current vs Hypothesis =========
    tab_current, tab_hypo = st.tabs(["📊 Current Analysis", "🧪 Hypothesis (What-If)"])

    # -------- CURRENT ANALYSIS (reads frozen data; never recomputes on UI tweaks) --------
    with tab_current:
        res = st.session_state["frozen_result"]
        df_display = res["df_display"]
        total_value = res["total_value"]
        risk_score = res["risk_score"]
        comps_out = res["comps_out"]
        metrics_for_ai = res["metrics_for_ai"]
        top_assets = res["top_assets"]
        pf_type = res["pf_type"]

        ai_summary = generate_ai_summary(metrics_for_ai, top_assets)

        left, right = st.columns([2,1])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("2) Live Portfolio Overview")
            st.dataframe(
                df_display.style.format({
                    "price_usd":"${:,.4f}","value_usd":"${:,.2f}",
                    "pct_portfolio":"{:.2f}%","price_change_percentage_24h":"{:+.2f}%"
                }),
                height=360, use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Allocation")
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
                color_discrete_map={"Up":"#16a34a","Down":"#dc2626"},
                labels={"change":"24h %","symbol":"Asset","direction":""},
                hover_data={"change":":.2f","symbol":True,"direction":False}
            )
            fig_bar.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            fig_bar.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#64748b")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Green = positive 24h, red = negative. Axis centered at 0% with a bold zero line.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Key Insights")
            st.markdown(portfolio_insights(df_display, comps_out))
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card center">', unsafe_allow_html=True)
            st.subheader("Risk Gauge")
            st.plotly_chart(risk_gauge(risk_score), use_container_width=True)
            st.markdown(f"**Profile:** {pf_type}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("AI Summary")
            st.text_area("AI-generated risk insights", value=ai_summary, height=210)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Download Report")
            html = f"""<html><head><meta charset='utf-8'><title>Sovereign Stash Report</title></head>
            <body style='font-family:Arial;background:#ffffff;color:#0f172a;'>
            <h2>Sovereign Stash — Crypto Risk Radar</h2>
            <p>Generated: {datetime.utcnow().isoformat()}Z</p>
            <p>Total Value: ${total_value:,.2f}</p>
            <p>Risk Score: {risk_score:.2f} / 100 ({pf_type})</p>
            <h3>Key Metrics</h3>
            <ul>
              <li>BTC+ETH: {comps_out['btc_eth_pct']}%</li>
              <li>Stablecoins: {comps_out['stable_pct']}%</li>
              <li>Altcoins: {comps_out['alt_pct']}%</li>
              <li>Weighted 24h volatility: {comps_out['weighted_vol_24h_pct']}%</li>
              <li>Unique assets: {comps_out['num_assets']}</li>
            </ul>
            <h3>AI Summary</h3>
            <pre style="white-space:pre-wrap;">{generate_ai_summary(metrics_for_ai, top_assets)}</pre>
            <hr><p style='font-size:12px;opacity:0.8'>{DISCLAIMER_SHORT}</p>
            </body></html>"""
            pdf = build_pdf_report(
                total_value, risk_score, pf_type, comps_out,
                generate_ai_summary(metrics_for_ai, top_assets),
                list(zip(df_display['symbol'].tolist(), [f"{v:.2f}%" for v in df_display['pct_portfolio'].tolist()]))
            )
            st.download_button("Download PDF (1 page)", data=pdf, file_name="sovereign_stash_report.pdf", mime="application/pdf")
            st.download_button("Download HTML (print to PDF)", data=html, file_name="sovereign_stash_report.html", mime="text/html")
            st.markdown('</div>', unsafe_allow_html=True)

    # -------- HYPOTHESIS (isolated; no recompute of current) --------
    with tab_hypo:
        res = st.session_state["frozen_result"]  # read-only snapshot
        comps_out = res["comps_out"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Rebalancing Hypothesis (What-If) — by Group")
        st.caption("This section uses the initial snapshot only. Changing these controls will not recompute the Current Analysis.")

        # Use a form so slider changes don't trigger live reruns; update only when "Update" is clicked
        with st.form(key="hypo_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            default_btc_eth = int(comps_out['btc_eth_pct'])
            default_stable  = int(comps_out['stable_pct'])
            # read previous selections from session_state to preserve between submits
            prev = st.session_state.get("hypo_targets", {"btc_eth":default_btc_eth, "stable":default_stable})
            tgt_btc_eth = c1.slider("BTC+ETH %", 0, 100, prev.get("btc_eth", default_btc_eth), key="hypo_btc_eth")
            tgt_stable  = c2.slider("Stablecoins %", 0, 100, prev.get("stable",  default_stable),  key="hypo_stable")
            max_alt = max(0, 100 - tgt_btc_eth - tgt_stable)
            tgt_alt = c3.slider("Altcoins %", 0, 100, int(prev.get("alt", max_alt)), key="hypo_alt")

            submitted = st.form_submit_button("Update hypothesis", use_container_width=True)

        # Normalize to 100 and persist the last chosen targets
        total = tgt_btc_eth + tgt_stable + tgt_alt
        if total != 100:
            # adjust alt to make sum 100
            tgt_alt = max(0, 100 - tgt_btc_eth - tgt_stable)
        st.session_state["hypo_targets"] = {"btc_eth":tgt_btc_eth, "stable":tgt_stable, "alt":tgt_alt}

        # Render hypothesis visuals (independent of current)
        hypo = pd.DataFrame({
            "group": ["BTC/ETH","Stablecoins","Altcoins"],
            "target": [tgt_btc_eth, tgt_stable, tgt_alt]
        })
        fig_hypo = px.pie(hypo, names="group", values="target", hole=0.35, title="Hypothetical Allocation (by group)")
        fig_hypo.update_layout(margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_hypo, use_container_width=True)

        # compute hypothetical risk using frozen diversification & vol only
        w_btc_eth=0.30; w_alt=0.30; w_stable=0.20; w_div=0.10; w_vol=0.10
        s_btc_eth = max(0,min(1,(100.0 - tgt_btc_eth)/100.0))
        s_alt     = max(0,min(1, tgt_alt/100.0))
        s_stable  = max(0,min(1,(100.0 - tgt_stable)/100.0))
        s_div     = 1.0 - min(comps_out['num_assets'], 12)/12.0
        s_vol     = max(0,min(1, comps_out['weighted_vol_24h_pct']/50.0))
        hypo_score = float(max(0,min(100, (w_btc_eth*s_btc_eth + w_alt*s_alt + w_stable*s_stable + w_div*s_div + w_vol*s_vol)*100.0 )))

        st.markdown("**Hypothetical Risk Score:**")
        st.plotly_chart(risk_gauge(hypo_score), use_container_width=True)

        delta = hypo_score - comps_out['risk_score']
        delta_txt = "lower" if delta < 0 else "higher"
        st.caption(f"Result: hypothetical risk is **{abs(delta):.2f}** points {delta_txt} than your current score ({comps_out['risk_score']:.2f}).")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# --- end app.py ---
