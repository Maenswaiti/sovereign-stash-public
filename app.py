# Sovereign Stash – Crypto Portfolio Navigator (public, no-auth)
import os, json, re, hashlib, time
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

# --- Streamlit config ---
st.set_page_config(page_title="Sovereign Stash – Crypto Portfolio Navigator", page_icon="🛰️", layout="wide")

# --- OpenAI optional (for AI summaries) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# --- Constants ---
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
STABLECOIN_IDS = {"tether","usd-coin","binance-usd","dai","true-usd","usdd","frax"}
FOLD_TO_BASE = {"wbtc":"bitcoin","btc.b":"bitcoin","weth":"ethereum","steth":"ethereum","reth":"ethereum"}
PREFERRED_COINS = {"btc":"bitcoin","eth":"ethereum","ada":"cardano","usdt":"tether","usdc":"usd-coin","sol":"solana"}
DISCLAIMER = "Educational purposes only — not financial or investment advice. Crypto assets are volatile and you can lose money."

# === Theme ===
def set_plotly_theme(theme):
    pio.templates.default = "plotly_white" if theme == "Light" else "plotly_dark"
    px.defaults.color_discrete_sequence = ["#10b981","#2563eb","#f59e0b","#22a2ee","#e11d48"]

def inject_css(theme):
    if theme=="Dark":
        bg="#0B0B0C"; text="#E6E6E6"; card="rgba(23,23,24,0.55)"; border="rgba(255,255,255,0.12)"
        btn="#10b981"; btnh="#059669"; btxt="#08100c"
    else:
        bg="#F5F7FB"; text="#0f172a"; card="rgba(255,255,255,0.7)"; border="rgba(15,23,42,0.1)"
        btn="#10b981"; btnh="#059669"; btxt="#ffffff"
    st.markdown(f"""
    <style>
    body,[data-testid="stAppViewContainer"]{{background:{bg};color:{text};
    font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}}
    .glass-card{{background:{card};border:1px solid {border};border-radius:18px;padding:18px;
    margin:14px 0;box-shadow:0 10px 30px rgba(0,0,0,0.2);backdrop-filter:blur(10px);}}
    .stButton>button,.stDownloadButton>button{{
      background:{btn}!important;color:{btxt}!important;border:0!important;
      border-radius:14px!important;padding:10px 14px!important;font-weight:700!important;
      box-shadow:0 6px 18px rgba(16,185,129,0.35)!important;}}
    .stButton>button:hover,.stDownloadButton>button:hover{{background:{btnh}!important;}}
    </style>""",unsafe_allow_html=True)

# === Data helpers ===
@st.cache_data(ttl=3600)
def fetch_coin_list():
    return requests.get(f"{COINGECKO_BASE}/coins/list",timeout=25).json()

@st.cache_data(ttl=3600)
def build_symbol_to_id_map():
    m={}
    for c in fetch_coin_list():
        s=(c.get("symbol") or "").lower(); cid=c.get("id")
        if s and s not in m: m[s]=cid
        if cid and cid not in m: m[cid]=cid
    return m

def symbol_to_id(symbol,s2id):
    if not symbol:return None
    s=symbol.strip().lower()
    if s in PREFERRED_COINS:return PREFERRED_COINS[s]
    return s2id.get(s,s)

@st.cache_data(ttl=3600)
def fetch_market_data(ids):
    if not ids:return []
    out=[]
    for i in range(0,len(ids),50):
        chunk=",".join(ids[i:i+50])
        r=requests.get(f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency":"usd","ids":chunk,"price_change_percentage":"24h"},timeout=25)
        if r.ok: out+=r.json()
    return out

@st.cache_data(ttl=3600)
def fetch_history_usd_range(coin_id,start_ts,end_ts,retries=3):
    """Fetch daily USD prices; fills gaps with NaN."""
    if not coin_id:return pd.DataFrame(columns=["date","price_usd"])
    url=f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range"
    params={"vs_currency":"usd","from":start_ts,"to":end_ts}
    for a in range(retries+1):
        try:
            r=requests.get(url,params=params,timeout=30)
            if r.status_code==429: time.sleep(1.2*(a+1)); continue
            r.raise_for_status(); data=r.json().get("prices",[])
            if not data:return pd.DataFrame(columns=["date","price_usd"])
            df=pd.DataFrame(data,columns=["ts","price_usd"])
            df["date"]=pd.to_datetime(df["ts"],unit="ms",utc=True).dt.tz_convert(None)
            df=df[["date","price_usd"]].drop_duplicates("date").sort_values("date")
            df=df.set_index("date").resample("1D").asfreq().ffill(limit=2).reset_index()
            return df
        except: time.sleep(0.8*(a+1))
    return pd.DataFrame(columns=["date","price_usd"])

# === Core analytics ===
def calculate_profile_score(df):
    tot=df["value_usd"].sum()
    if tot<=0:return 0,{}
    key="folded_id" if "folded_id" in df else "id"
    btc_eth=df[key].isin(["bitcoin","ethereum"]); stable=df[key].isin(STABLECOIN_IDS)
    btc_eth_pct=df.loc[btc_eth,"value_usd"].sum()/tot*100
    stable_pct=df.loc[stable,"value_usd"].sum()/tot*100
    alt_pct=100-btc_eth_pct-stable_pct
    df["abs_24h"]=df["price_change_percentage_24h"].abs().fillna(0)
    wv=(df["abs_24h"]*df["value_usd"]).sum()/tot
    div=min(df[key].nunique(),12)/12
    w=[.3,.3,.2,.1,.1]
    s=[(100-btc_eth_pct)/100,alt_pct/100,(100-stable_pct)/100,1-div,min(1,wv/50)]
    sc=max(0,min(100,sum(a*b for a,b in zip(w,s))*100))
    return sc,dict(btc_eth_pct=btc_eth_pct,stable_pct=stable_pct,alt_pct=alt_pct,weighted_vol_24h_pct=wv,num_assets=df[key].nunique())

def portfolio_insights(df,c):
    hhi=((df["pct_portfolio"]/100)**2).sum()
    conc="well-diversified" if hhi<.12 else "moderately concentrated" if hhi<.18 else "highly concentrated"
    top=df.head(3)["pct_portfolio"].sum()
    return f"• Top 3 = **{top:.1f}%**, {conc}.\\n• BTC+ETH **{c['btc_eth_pct']:.1f}%**, Stable **{c['stable_pct']:.1f}%**, Alts **{c['alt_pct']:.1f}%**."

def profile_gauge(score):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=score,number={"suffix":" / 100"},
        gauge={"axis":{"range":[0,100]},"bar":{"color":"#10b981"},
               "steps":[{"range":[0,34],"color":"#d1fae5"},{"range":[34,66],"color":"#fde68a"},{"range":[66,100],"color":"#fecaca"}]}))
    fig.update_layout(height=250,margin=dict(l=0,r=0,t=10,b=0)); return fig

# === PDF export ===
def build_pdf(total,score,pf_type,insights,alloc_chart_path,btc_chart_path):
    buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=LETTER)
    styles=getSampleStyleSheet(); story=[]
    story+=[Paragraph("<b>Sovereign Stash Report</b>",styles["Title"]),Spacer(1,6)]
    story+=[Paragraph(f"Profile Score: {score:.1f} / 100 ({pf_type})",styles["Normal"]),Spacer(1,8)]
    story+=[Paragraph(insights.replace("\n","<br/>"),styles["Normal"]),Spacer(1,8)]
    if os.path.exists(alloc_chart_path):
        story+=[Image(alloc_chart_path,width=480,height=320)]
    if os.path.exists(btc_chart_path):
        story+=[Spacer(1,10),Image(btc_chart_path,width=480,height=320)]
    story+=[Spacer(1,12),Paragraph(DISCLAIMER,styles["Italic"])]
    doc.build(story); buf.seek(0); return buf.getvalue()

# === Main ===
def main():
    with st.sidebar:
        theme=st.selectbox("Theme",["Dark","Light"],0)
    set_plotly_theme(theme); inject_css(theme)

    # Hero centered
    st.markdown("<div style='text-align:center'>",unsafe_allow_html=True)
    st.image("assets/logo_ss.png",width=130)
    st.markdown("<h1>Sovereign Stash</h1><h3>Crypto Portfolio Navigator</h3>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    # Input
    st.markdown('<div class="glass-card"><b>1) Paste or Import your portfolio</b>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    text=c1.text_area("Manual entry","BTC: 0.5\\nETH: 2\\nADA: 10000\\nUSDT: 1500",height=260)
    uploaded=c2.file_uploader("Upload CSV/XLSX (Token,Amount)",type=["csv","xlsx"])
    imported=[]
    if uploaded:
        raw=uploaded.read()
        if uploaded.name.endswith("csv"): imported=ingest.parse_csv_bytes(raw)
        else: imported=ingest.parse_xlsx_bytes(raw)
    fold=c2.checkbox("Fold derivatives (e.g. wBTC→BTC, stETH→ETH)",True)
    analyze=st.button("Analyze Portfolio",type="primary")
    st.markdown("</div>",unsafe_allow_html=True)

    def parse_lines(txt):
        out=[]
        for line in txt.splitlines():
            parts=re.split("[:, ]+",line.strip())
            if len(parts)>=2:
                try: out.append((parts[0],float(parts[-1])))
                except: pass
        return out

    if not analyze: st.stop()
    rows=parse_lines(text)+imported
    if not rows: st.error("No valid holdings."); st.stop()

    # compute
    s2id=build_symbol_to_id_map()
    ids=[symbol_to_id(t,s2id) for t,_ in rows]
    market=fetch_market_data(list(set(ids)))
    m={x["id"]:x for x in market}
    df=pd.DataFrame([{
        "token":t,"id":symbol_to_id(t,s2id),"quantity":q,
        "price_usd":m.get(symbol_to_id(t,s2id),{}).get("current_price",0),
        "price_change_percentage_24h":m.get(symbol_to_id(t,s2id),{}).get("price_change_percentage_24h",0),
        "name":m.get(symbol_to_id(t,s2id),{}).get("name",t),
        "symbol":m.get(symbol_to_id(t,s2id),{}).get("symbol",t).upper()
    } for t,q in rows])
    df["value_usd"]=df["quantity"]*df["price_usd"]
    df["folded_id"]=df["id"].apply(lambda x:FOLD_TO_BASE.get(x,x) if fold else x)
    total=df["value_usd"].sum()
    df["pct_portfolio"]=df["value_usd"]/total*100
    score,c=calculate_profile_score(df)
    pf="Conservative" if score<34 else "Balanced" if score<66 else "Growth-oriented"

    # === Current ===
    st.markdown('<div class="glass-card"><b>2) Current Profile</b>',unsafe_allow_html=True)
    st.dataframe(df[["symbol","name","quantity","price_usd","value_usd","pct_portfolio","price_change_percentage_24h"]]
                 .style.format({"price_usd":"${:,.4f}","value_usd":"${:,.2f}","pct_portfolio":"{:.2f}%","price_change_percentage_24h":"{:+.2f}%"}))
    st.plotly_chart(profile_gauge(score),use_container_width=True)
    insights=portfolio_insights(df,c)
    st.markdown(insights)
    st.markdown("</div>",unsafe_allow_html=True)

    # === Allocation Chart ===
    fig_alloc=px.pie(df,names="symbol",values="pct_portfolio",hole=0.35)
    fig_alloc.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_alloc,use_container_width=True)
    alloc_chart_path="alloc_chart.png"
    fig_alloc.write_image(alloc_chart_path,width=800,height=500)

    # === BTC-relative ===
    st.markdown('<div class="glass-card"><b>Relative to BTC (Strength)</b>',unsafe_allow_html=True)
    syms=[s for s in df["symbol"].unique() if s!="BTC"]
    sel=st.multiselect("Select tokens",options=syms,default=syms[:5])
    dr=st.date_input("Date range",(date.today()-timedelta(days=365),date.today()),min_value=date.today()-timedelta(days=5*365))
    smooth=st.checkbox("Smooth 7d",True); logy=st.checkbox("Log scale Y",False)
    coverage_rows=[]
    btc_chart_path="btc_chart.png"
    if sel:
        s2id=build_symbol_to_id_map(); btc_id="bitcoin"
        start_ts=int(time.mktime(datetime.combine(dr[0],datetime.min.time()).timetuple()))
        end_ts=int(time.mktime(datetime.combine(dr[1],datetime.min.time()).timetuple()))
        btc=fetch_history_usd_range(btc_id,start_ts,end_ts)
        lines=[]
        for s in sel:
            cid=PREFERRED_COINS.get(s.lower()) or symbol_to_id(s,s2id)
            hist=fetch_history_usd_range(cid,start_ts,end_ts)
            if hist.empty or btc.empty: continue
            dfj=pd.merge_asof(hist.sort_values("date"),btc.sort_values("date"),on="date",
                              tolerance=pd.Timedelta("3D"),direction="nearest",suffixes=("_tok","_btc"))
            dfj["strength"]=dfj["price_usd_tok"]/dfj["price_usd_btc"]
            base=dfj["strength"].dropna().iloc[0] if not dfj["strength"].dropna().empty else None
            if base: dfj["strength_idx"]=(dfj["strength"]/base)*100
            if smooth: dfj["strength_idx"]=dfj["strength_idx"].rolling(7,min_periods=1).mean()
            dfj["symbol"]=s
            coverage_rows.append((s,str(dfj["date"].min().date()),str(dfj["date"].max().date())))
            lines.append(dfj[["date","symbol","strength_idx"]])
        if lines:
            rel=pd.concat(lines,ignore_index=True)
            fig=px.line(rel,x="date",y="strength_idx",color="symbol",
                        labels={"strength_idx":"Strength vs BTC (start=100)","date":"Date"})
            if logy: fig.update_yaxes(type="log")
            st.plotly_chart(fig,use_container_width=True)
            fig.write_image(btc_chart_path,width=800,height=500)
        else: st.info("No data for selected tokens/date range.")
    if coverage_rows:
        st.dataframe(pd.DataFrame(coverage_rows,columns=["Token","Start","End"]))
    st.markdown("</div>",unsafe_allow_html=True)

    # === Export ===
    st.markdown('<div class="glass-card"><b>Export Report</b>',unsafe_allow_html=True)
    pdf=build_pdf(total,score,pf,insights,alloc_chart_path,btc_chart_path)
    st.download_button("Download PDF Report",data=pdf,file_name="sovereign_stash_report.pdf",mime="application/pdf")
    st.markdown("</div>",unsafe_allow_html=True)

if __name__=="__main__":
    main()
