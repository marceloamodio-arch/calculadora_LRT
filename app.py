# app_st.py
# Calculadora LRT (Streamlit) – versión ST
# Conserva la lógica esencial y criterios de cálculo del código base (Barrios, art. 3 Ley 26.773, RIPTE+3% vs Tasa Activa):contentReference[oaicite:1]{index=1}

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# ---- Config básica ----
st.set_page_config(page_title="Calculadora Indemnizaciones LRT (ST)", layout="wide")

DATASET_RIPTE = "dataset_ripte.csv"
DATASET_TASA  = "dataset_tasa.csv"
DATASET_IPC   = "dataset_ipc.csv"
DATASET_PISOS = "dataset_pisos.csv"

# ---- Utilidades de fecha y formato ----
def safe_parse_date(s) -> Optional[date]:
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    if isinstance(s, (datetime, date)): return s.date() if isinstance(s, datetime) else s
    s = str(s).strip()
    if not s: return None
    fmts = [
        "%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%m/%Y","%Y/%m/%d","%Y-%m",
        "%Y-%m-%d %H:%M:%S","%d/%m/%Y %H:%M:%S","%B %Y","%b %Y","%Y/%m","%m-%Y"
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if f in ("%m/%Y","%Y-%m","%Y/%m","%m-%Y","%B %Y","%b %Y"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            continue
    # heurística año-mes
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt): return None
        return dt.date()
    except Exception:
        return None

def days_in_month(d: date) -> int:
    if d.month == 12:
        nxt = date(d.year+1,1,1)
    else:
        nxt = date(d.year, d.month+1,1)
    return (nxt - date(d.year,d.month,1)).days

def fmt_money(x: float) -> str:
    return f"$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%".replace(".", ",")

# ---- Carga normalizada de datasets (respetando enfoque del código base):contentReference[oaicite:2]{index=2} ----
@st.cache_data
def load_csv_flexible(path: str) -> pd.DataFrame:
    for sep in [",",";","\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path)  # último intento

@st.cache_data
def load_ripte(path: str) -> pd.DataFrame:
    df = load_csv_flexible(path).copy()
    if df.empty: return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    # detectar fecha
    fecha_col = None
    if "año" in df.columns and "mes" in df.columns:
        meses = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
                 "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12,
                 "ene":1,"abr":4,"ago":8,"set":9,"dic":12,
                 "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        def conv_mes(v):
            s = str(v).strip().lower()
            try: return int(float(s))
            except: return meses.get(s, None)
        def mkfecha(row):
            try:
                a = int(row["año"]); m = conv_mes(row["mes"])
                return date(a, m, 1) if m else None
            except: return None
        df["fecha"] = df.apply(mkfecha, axis=1)
    else:
        for c in df.columns:
            if any(k in c for k in ["fecha","periodo","mes"]):
                fecha_col = c; break
        if fecha_col is None: fecha_col = df.columns[0]
        df["fecha"] = df[fecha_col].apply(safe_parse_date)

    # valor ripte
    val_col = None
    for c in df.columns:
        if c in ("indice_ripte","ripte","valor","indice"): val_col = c; break
    if val_col is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        val_col = num_cols[0] if num_cols else df.columns[1]
    df["ripte"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["fecha","ripte"]).sort_values("fecha").reset_index(drop=True)
    return df

@st.cache_data
def load_tasa(path: str) -> pd.DataFrame:
    df = load_csv_flexible(path).copy()
    if df.empty: return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "desde" in df.columns:
        df["desde"] = df["desde"].apply(safe_parse_date)
    if "hasta" in df.columns:
        df["hasta"] = df["hasta"].apply(safe_parse_date)
    else:
        if "desde" in df.columns:
            df["hasta"] = df["desde"]
    # columna tasa
    base_col = None
    for c in ("tasa","valor","porcentaje"):
        if c in df.columns: base_col = c; break
    if base_col:
        df["tasa"] = pd.to_numeric(df[base_col], errors="coerce")
    # fecha auxiliar
    if "desde" in df.columns:
        df["fecha"] = df["desde"]
    keep = [c for c in ["fecha","tasa","desde","hasta"] if c in df.columns]
    df = df.dropna(subset=["fecha","tasa"]).sort_values("fecha").reset_index(drop=True)
    return df[keep]

@st.cache_data
def load_ipc(path: str) -> pd.DataFrame:
    df = load_csv_flexible(path).copy()
    if df.empty: return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    fecha_col = "periodo" if "periodo" in df.columns else None
    if not fecha_col:
        for c in df.columns:
            if any(k in c for k in ["fecha","periodo","mes"]): fecha_col = c; break
        if not fecha_col: fecha_col = df.columns[0]
    val_col = "variacion_mensual" if "variacion_mensual" in df.columns else None
    if not val_col:
        for c in df.columns:
            if any(k in c for k in ["variacion","inflacion","ipc","porcentaje","mensual","indice"]):
                val_col = c; break
        if not val_col:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            val_col = num_cols[0] if num_cols else df.columns[1]
    df["fecha"] = df[fecha_col].apply(safe_parse_date)
    df["ipc"] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["fecha","ipc"]).sort_values("fecha").reset_index(drop=True)
    return df

@st.cache_data
def load_pisos(path: str) -> pd.DataFrame:
    df = load_csv_flexible(path).copy()
    if df.empty: return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    # map rápido
    def pick(cols, keys):
        for k in keys:
            if k in cols: return k
        for c in cols:
            if any(k in c for k in keys): return c
        return None
    c_desde = pick(df.columns, ["fecha_inicio","desde","inicio"])
    c_hasta = pick(df.columns, ["fecha_fin","hasta","fin"])
    c_monto = pick(df.columns, ["monto_minimo","piso","monto","minimo","base"])
    c_norma = pick(df.columns, ["norma","res","resol","nota","exp","srt"])
    df["desde"] = df[c_desde].apply(safe_parse_date) if c_desde else df.iloc[:,0].apply(safe_parse_date)
    df["hasta"] = df[c_hasta].apply(safe_parse_date) if c_hasta else None
    if c_monto:
        df["piso"] = pd.to_numeric(df[c_monto], errors="coerce")
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        df["piso"] = pd.to_numeric(df[num_cols[0]] if num_cols else df.iloc[:, -1], errors="coerce")
    df["resol"] = df[c_norma].astype(str) if c_norma else ""
    df = df.dropna(subset=["desde","piso"]).sort_values("desde").reset_index(drop=True)
    return df[["desde","hasta","piso","resol"]]

# ---- Lógica de negocio (conforme base):contentReference[oaicite:3]{index=3} ----
@dataclass
class InputData:
    pmi_date: date
    final_date: date
    ibm: float
    edad: int
    incapacidad_pct: float
    incluir_20_pct: bool

@dataclass
class Results:
    capital_formula: float
    capital_base: float
    piso_aplicado: bool
    piso_info: str
    adicional_20_pct: float
    ripte_coef: float
    ripte_actualizado: float
    interes_puro_3_pct: float
    total_ripte_3: float
    tasa_activa_pct: float
    total_tasa_activa: float
    inflacion_acum_pct: float

def get_piso_minimo(pisos_df: pd.DataFrame, fecha_pmi: date) -> Tuple[Optional[float], str]:
    if pisos_df.empty: return (None, "")
    cand = None
    for _, r in pisos_df.iterrows():
        d0 = r["desde"]; d1 = r["hasta"] if pd.notna(r["hasta"]) else None
        if d1 is None:
            if fecha_pmi >= d0: cand = (float(r["piso"]), r.get("resol",""))
        else:
            if d0 <= fecha_pmi <= d1: return (float(r["piso"]), r.get("resol",""))
    return cand if cand else (None, "")

def ripte_coef(ripte_df: pd.DataFrame, pmi: date) -> Tuple[float, float, float]:
    if ripte_df.empty: return 1.0, 0.0, 0.0
    ripte_pmi_df = ripte_df[ripte_df["fecha"] <= pmi]
    ripte_pmi = float(ripte_pmi_df.iloc[-1]["ripte"]) if not ripte_pmi_df.empty else float(ripte_df.iloc[0]["ripte"])
    ripte_final = float(ripte_df.iloc[-1]["ripte"])
    coef = ripte_final / ripte_pmi if ripte_pmi > 0 else 1.0
    return coef, ripte_pmi, ripte_final

def calcular_tasa_activa(tasa_df: pd.DataFrame, pmi: date, fin: date, capital_base: float) -> Tuple[float, float]:
    if tasa_df.empty: return 0.0, capital_base
    total_pct = 0.0
    for _, row in tasa_df.iterrows():
        f0 = row["desde"] if "desde" in tasa_df.columns else row["fecha"]
        f1 = row["hasta"] if "hasta" in tasa_df.columns and pd.notna(row.get("hasta")) else date(f0.year, f0.month, days_in_month(f0))
        if isinstance(f0, pd.Timestamp): f0 = f0.date()
        if isinstance(f1, pd.Timestamp): f1 = f1.date()
        ini = max(pmi, f0); fin_i = min(fin, f1)
        if ini <= fin_i:
            dias = (fin_i - ini).days + 1
            val_mensual = float(row["tasa"])
            total_pct += val_mensual * (dias / 30.0)
    total_actual = capital_base * (1.0 + total_pct / 100.0)
    return total_pct, total_actual

def calcular_inflacion(ipc_df: pd.DataFrame, pmi: date, fin: date) -> float:
    if ipc_df.empty: return 0.0
    i0 = pd.Timestamp(pmi.replace(day=1))
    i1 = pd.Timestamp(fin.replace(day=1))
    seg = ipc_df[(pd.to_datetime(ipc_df["fecha"]) >= i0) & (pd.to_datetime(ipc_df["fecha"]) <= i1)]
    if seg.empty: return 0.0
    factor = 1.0
    for _, r in seg.iterrows():
        if pd.notna(r["ipc"]): factor *= (1 + r["ipc"]/100)
    return (factor - 1) * 100

def calcular_capital_formula(ibm: float, edad: int, incapacidad_pct: float) -> float:
    return ibm * 53 * (65 / edad) * (incapacidad_pct / 100.0)

def aplicar_piso(capital_formula: float, piso_min: Optional[float], norma: str, incap_pct: float) -> Tuple[float, bool, str]:
    if piso_min is None: return capital_formula, False, "Sin piso mínimo aplicable"
    piso_prop = piso_min * (incap_pct / 100.0)
    if capital_formula >= piso_prop:
        return capital_formula, False, f"Supera piso mínimo ({norma})"
    else:
        return piso_prop, True, f"Aplica piso mínimo ({norma})"

def motor_calculo(inp: InputData, ripte_df, tasa_df, ipc_df, pisos_df) -> Results:
    base_cap_formula = calcular_capital_formula(inp.ibm, inp.edad, inp.incapacidad_pct)
    piso_min, piso_norma = get_piso_minimo(pisos_df, inp.pmi_date)
    capital_aplicado, piso_apl, piso_info = aplicar_piso(base_cap_formula, piso_min, piso_norma, inp.incapacidad_pct)
    adicional_20 = capital_aplicado * 0.20 if inp.incluir_20_pct else 0.0
    capital_base = capital_aplicado + adicional_20

    coef, ripte_pmi, ripte_fin = ripte_coef(ripte_df, inp.pmi_date)
    ripte_act = capital_base * coef
    dias = (inp.final_date - inp.pmi_date).days
    interes_3 = ripte_act * 0.03 * (dias / 365.0)  # interés puro anual 3% sobre RIPTE actualizado:contentReference[oaicite:4]{index=4}
    total_ripte3 = ripte_act + interes_3

    tasa_pct, total_tasa = calcular_tasa_activa(tasa_df, inp.pmi_date, inp.final_date, capital_base)
    inflacion_pct = calcular_inflacion(ipc_df, inp.pmi_date, inp.final_date)

    return Results(
        capital_formula=base_cap_formula,
        capital_base=capital_base,
        piso_aplicado=piso_apl,
        piso_info=piso_info,
        adicional_20_pct=adicional_20,
        ripte_coef=coef,
        ripte_actualizado=ripte_act,
        interes_puro_3_pct=interes_3,
        total_ripte_3=total_ripte3,
        tasa_activa_pct=tasa_pct,
        total_tasa_activa=total_tasa,
        inflacion_acum_pct=inflacion_pct
    )

# ---- UI ----
st.title("Calculadora de Indemnizaciones LRT – Versión ST (Streamlit)")
st.caption("Mantiene lógica y criterios del sistema original. Carga 4 datasets CSV en la misma carpeta del app.")

colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    pmi = st.date_input("Fecha del siniestro (PMI)", value=date(2019,1,15))
with colB:
    f_final = st.date_input("Fecha de cálculo", value=date.today())
with colC:
    ibm = st.number_input("IBM (Ingreso Base Mensual)", min_value=0.0, step=1000.0, value=300000.0, format="%.2f")
with colD:
    edad = st.number_input("Edad del trabajador/a", min_value=18, max_value=80, value=40, step=1)

colE, colF = st.columns([1,1])
with colE:
    incap = st.number_input("Porcentaje de incapacidad (%)", min_value=1.0, max_value=100.0, value=20.0, step=0.5, format="%.2f")
with colF:
    inc20 = st.checkbox("Incluir 20% (art. 3 Ley 26.773)", value=True)

st.divider()
st.subheader("Datasets requeridos (en la misma carpeta)")
ds1, ds2, ds3, ds4 = st.columns(4)
with ds1: st.code(DATASET_RIPTE, language="text")
with ds2: st.code(DATASET_TASA, language="text")
with ds3: st.code(DATASET_IPC, language="text")
with ds4: st.code(DATASET_PISOS, language="text")

# Carga
try:
    ripte_df = load_ripte(DATASET_RIPTE)
    tasa_df  = load_tasa(DATASET_TASA)
    ipc_df   = load_ipc(DATASET_IPC)
    pisos_df = load_pisos(DATASET_PISOS)
except Exception as e:
    st.error(f"Error cargando datasets: {e}")
    st.stop()

faltan = []
if ripte_df.empty: faltan.append("RIPTE")
if tasa_df.empty:  faltan.append("TASA ACTIVA")
if ipc_df.empty:   faltan.append("IPC")
if pisos_df.empty: faltan.append("PISOS")
if faltan:
    st.warning("Faltan datasets o están vacíos: " + ", ".join(faltan))
    st.stop()

# Calcular
inp = InputData(pmi_date=pmi, final_date=f_final, ibm=ibm, edad=int(edad),
                incapacidad_pct=float(incap), incluir_20_pct=inc20)

if st.button("Calcular", type="primary"):
    res = motor_calculo(inp, ripte_df, tasa_df, ipc_df, pisos_df)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Capital fórmula", fmt_money(res.capital_formula))
        st.metric("Capital base (con 20% si aplica)", fmt_money(res.capital_base))
        st.write(("🟢 " if res.piso_aplicado else "⚪ ") + res.piso_info)
    with c2:
        st.metric("RIPTE actualizado", fmt_money(res.ripte_actualizado))
        st.metric("Interés puro 3% (anual, pro rata)", fmt_money(res.interes_puro_3_pct))
        st.metric("Total RIPTE + 3%", fmt_money(res.total_ripte_3))
    with c3:
        st.metric("Tasa Activa (acum. período)", fmt_pct(res.tasa_activa_pct))
        st.metric("Total con Tasa Activa", fmt_money(res.total_tasa_activa))
        st.metric("Inflación acumulada (referencia)", fmt_pct(res.inflacion_acum_pct))

    st.divider()
    metodo = "RIPTE + 3%" if res.total_ripte_3 >= res.total_tasa_activa else "Tasa Activa BNA"
    total_favorable = res.total_ripte_3 if metodo == "RIPTE + 3%" else res.total_tasa_activa
    st.success(f"Método más favorable: **{metodo}** → {fmt_money(total_favorable)}")

    # Generar HTML imprimible simple
    html = f"""
    <html><head><meta charset="utf-8"><title>Liquidación LRT</title></head>
    <body style="font-family:Times New Roman,serif;">
    <h3 style="border-bottom:1px solid #333;">Cálculo de Indemnización (LRT)</h3>
    <p><b>PMI:</b> {inp.pmi_date.strftime('%d/%m/%Y')} — <b>Fecha cálculo:</b> {inp.final_date.strftime('%d/%m/%Y')}</p>
    <p><b>IBM:</b> {fmt_money(inp.ibm)} — <b>Edad:</b> {inp.edad} — <b>Incapacidad:</b> {inp.incapacidad_pct:.2f}% — <b>20%:</b> {"Incluido" if inp.incluir_20_pct else "No"}</p>
    <hr>
    <p><b>Capital fórmula:</b> {fmt_money(res.capital_formula)}</p>
    <p><b>Capital base:</b> {fmt_money(res.capital_base)} | {res.piso_info}</p>
    <p><b>RIPTE actualizado:</b> {fmt_money(res.ripte_actualizado)} + <b>3% puro:</b> {fmt_money(res.interes_puro_3_pct)} = <b>Total:</b> {fmt_money(res.total_ripte_3)}</p>
    <p><b>Tasa Activa BNA (acum):</b> {fmt_pct(res.tasa_activa_pct)} → <b>Total:</b> {fmt_money(res.total_tasa_activa)}</p>
    <p><b>Método más favorable:</b> {metodo} → <b>{fmt_money(total_favorable)}</b></p>
    </body></html>
    """.strip()

    st.download_button(
        "Descargar HTML para impresión (A4)",
        data=html.encode("utf-8"),
        file_name=f"liquidacion_lrt_{inp.final_date.isoformat()}.html",
        mime="text/html"
    )
