
import io, csv, json
from typing import List, Tuple, Dict, Optional
try: import openpyxl
except Exception: openpyxl=None
COMMON_SYMBOL_KEYS = ["symbol","sym","ticker","asset","currency","coin","token","product","ccy","currency code"]
COMMON_QTY_KEYS    = ["quantity","qty","amount","size","balance","total","total balance","free","available","hold","locked","in order","equity","total equity","wallet balance"]
EXCHANGE_HINTS={"binance":{"symbol":["coin","asset"],"qty":["amount","free","locked"]},"coinbase":{"symbol":["currency","asset"],"qty":["balance","total"]},"coinbase pro":{"symbol":["product"],"qty":["size"]},"kraken":{"symbol":["asset"],"qty":["balance"]},"okx":{"symbol":["ccy"],"qty":["bal","eq","cashBal"]},"kucoin":{"symbol":["coin"],"qty":["available","hold","balance"]},"bybit":{"symbol":["coin"],"qty":["total equity","equity","wallet balance","available balance"]},"crypto.com":{"symbol":["currency","coin"],"qty":["balance","total"]},"gemini":{"symbol":["currency"],"qty":["balance","amount","available"]},"bitfinex":{"symbol":["currency"],"qty":["balance"]},"gate.io":{"symbol":["currency","coin"],"qty":["available","locked","total"]},"poloniex":{"symbol":["currency"],"qty":["amount","available"]}}
def _lower_keys(d: Dict[str,str]): return {(k or '').strip().lower():(v or '') for k,v in d.items()}
def _first_present(d: Dict[str,str], keys: List[str]):
    for k in keys:
        if k in d and d[k] not in ('',None): return d[k]
    return None
def _sum_present(d: Dict[str,str], keys: List[str]):
    val=0.0; hit=False
    for k in keys:
        if k in d and d[k] not in ('',None):
            try: val+=float(str(d[k]).replace(',','')); hit=True
            except: pass
    return val if hit else None
def _clean_sym(s:str):
    s=(s or '').strip()
    if '-' in s: s=s.split('-')[0]
    if '/' in s: s=s.split('/')[0]
    return s
def parse_csv_bytes(file_bytes:bytes)->List[Tuple[str,float]]:
    text=file_bytes.decode('utf-8',errors='ignore'); reader=csv.DictReader(io.StringIO(text)); rows=list(reader)
    if not rows: return []
    out=[]
    for name,hint in EXCHANGE_HINTS.items():
        sym_keys=hint['symbol']; qty_keys=hint['qty']; tmp=[]
        for r in rows:
            d=_lower_keys(r); sym=_first_present(d,sym_keys) or _first_present(d,COMMON_SYMBOL_KEYS)
            qty=_sum_present(d,qty_keys) or _first_present(d,COMMON_QTY_KEYS)
            if sym and qty:
                try:
                    q=float(str(qty).replace(',',''))
                    if q!=0: tmp.append((_clean_sym(sym),q))
                except: pass
        if tmp: out.extend(tmp); return out
    for r in rows:
        d=_lower_keys(r); sym=_first_present(d,COMMON_SYMBOL_KEYS); qty=_first_present(d,COMMON_QTY_KEYS)
        if sym and qty:
            try:
                q=float(str(qty).replace(',','')); 
                if q!=0: out.append((_clean_sym(sym),q))
            except: pass
    return out
def parse_xlsx_bytes(file_bytes:bytes)->List[Tuple[str,float]]:
    if openpyxl is None: return []
    f=io.BytesIO(file_bytes); wb=openpyxl.load_workbook(f,data_only=True); ws=wb.active
    headers=[str(c.value).strip().lower() if c.value is not None else '' for c in ws[1]]
    def col_idx(cands):
        for c in cands:
            if c in headers: return headers.index(c)
        return None
    out=[]
    for name,hint in EXCHANGE_HINTS.items():
        si=col_idx(hint['symbol']) or col_idx(COMMON_SYMBOL_KEYS)
        qty_idxs=[headers.index(k) for k in hint['qty'] if k in headers]
        tmp=[]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if si is None: continue
            sym=row[si] if si<len(row) else None
            if sym is None: continue
            qty_val=0.0; hit=False
            for qi in qty_idxs or []:
                try: qty_val+=float(str(row[qi]).replace(',','')); hit=True
                except: pass
            if not hit:
                qidx=col_idx(COMMON_QTY_KEYS)
                if qidx is not None:
                    try: qty_val=float(str(row[qidx]).replace(',','')); hit=True
                    except: pass
            if hit and qty_val!=0: tmp.append((_clean_sym(str(sym)), qty_val))
        if tmp: out.extend(tmp); return out
    si=col_idx(COMMON_SYMBOL_KEYS); qi=col_idx(COMMON_QTY_KEYS)
    if si is not None and qi is not None:
        for row in ws.iter_rows(min_row=2, values_only=True):
            sym=row[si]; qty=row[qi]
            if sym is None or qty in (None,''): continue
            try:
                q=float(str(qty).replace(',','')); 
                if q!=0: out.append((_clean_sym(str(sym)),q))
            except: pass
    return out
def parse_json_bytes(file_bytes:bytes)->List[Tuple[str,float]]:
    try: data=json.loads(file_bytes.decode('utf-8',errors='ignore'))
    except: return []
    out=[]
    if isinstance(data,list):
        for r in data:
            if not isinstance(r,dict): continue
            d={ (k or '').lower():v for k,v in r.items() }
            sym=None; qty=None
            for k in COMMON_SYMBOL_KEYS:
                if k in d: sym=d[k]; break
            for k in COMMON_QTY_KEYS:
                if k in d: qty=d[k]; break
            if sym is not None and qty is not None:
                try:
                    q=float(str(qty).replace(',','')); 
                    if q!=0: out.append((_clean_sym(str(sym)),q))
                except: pass
    elif isinstance(data,dict) and 'holdings' in data:
        for r in data['holdings']:
            if not isinstance(r,dict): continue
            d={ (k or '').lower():v for k,v in r.items() }
            sym=d.get('symbol') or d.get('asset') or d.get('currency')
            qty=d.get('quantity') or d.get('amount') or d.get('balance') or d.get('size')
            if sym and qty:
                try:
                    q=float(str(qty).replace(',','')); 
                    if q!=0: out.append((_clean_sym(str(sym)),q))
                except: pass
    return out
