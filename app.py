# app.py - Versión Streamlit de manito7.py (GUI Tkinter reemplazada por Streamlit)
# Dr. Marcelo S. Amodio - Tribunal de Trabajo N° 2 de Quilmes

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import types

st.set_page_config(page_title="Calculadora LRT", layout="wide")

# ---- Stubs para reemplazar messagebox/simpledialog de Tkinter ----
class _DummyMessageBox:
    @staticmethod
    def showerror(title, msg):
        raise RuntimeError(f"{{title}}: {{msg}}")
    @staticmethod
    def showinfo(title, msg):
        st.info(f"**{{title}}** — {{msg}}")

class _DummySimpleDialog:
    @staticmethod
    def askstring(title, prompt, **kwargs):
        return None
    @staticmethod
    def askinteger(title, prompt, **kwargs):
        return None

# ---- Inyección del núcleo lógico del archivo original (sin imports Tkinter) ----
messagebox = _DummyMessageBox
simpledialog = _DummySimpleDialog

# Código original (con GUI removida) embebido literalmente
_CORE_CODE = r"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULADORA INDEMNIZACIONES LEY 24.557
TRIBUNAL DE TRABAJO NRO. 2 QUILMES

Sistema de cálculo de indemnizaciones laborales conforme jurisprudencia "Barrios"
VERSIÓN CORREGIDA - FIX: Fórmula 3% anual
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import webbrowser
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys

# Configuración de colores y estilo
COLORS = {
    'bg': '#FFFFFF',
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#F18F01',
    'info': '#C73E1D',
    'light': '#F8F9FA',
    'dark': '#343A40',
    'highlight_ripte': '#E8F5E8',
    'highlight_tasa': '#E8F0FF'
}

# Configuración de fuentes
FONTS = {
    'title': ('Arial', 14, 'bold'),
    'heading': ('Arial', 12, 'bold'),
    'normal': ('Arial', 10),
    'small': ('Arial', 9),
    'large_money': ('Arial', 16, 'bold')
}

# Password por defecto
DEFAULT_PASSWORD = "todosjuntos"

# Paths de datasets
DATASET_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_RIPTE = os.path.join(DATASET_DIR, "dataset_ripte.csv")
PATH_TASA = os.path.join(DATASET_DIR, "dataset_tasa.csv")
PATH_IPC = os.path.join(DATASET_DIR, "dataset_ipc.csv")
PATH_PISOS = os.path.join(DATASET_DIR, "dataset_pisos.csv")

def safe_parse_date(s) -> Optional[date]:
    """Función corregida de parseo de fechas del BACK6"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    if isinstance(s, (datetime, date)):
        return s.date() if isinstance(s, datetime) else s
    s = str(s).strip()
    if not s:
        return None
    
    # Formatos de fecha ampliados del BACK6
    fmts = [
        "%Y-%m-%d", 
        "%d/%m/%Y", 
        "%d-%m-%Y", 
        "%m/%Y", 
        "%Y/%m/%d", 
        "%Y-%m",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%B %Y",              
        "%b %Y",              
        "%Y/%m",              
        "%m-%Y",              
    ]
    
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if f in ("%m/%Y", "%Y-%m", "%Y/%m", "%m-%Y", "%B %Y", "%b %Y"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            continue
    
    # Manejo especial para formatos de período
    if "/" in s or "-" in s:
        parts = s.replace("/", "-").split("-")
        if len(parts) == 2:
            try:
                year, month = int(parts[0]), int(parts[1])
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return date(year, month, 1)
            except ValueError:
                pass
    
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        if isinstance(dt, pd.Timestamp):
            return dt.date()
        return None
    except Exception:
        return None

def days_in_month(d: date) -> int:
    """Días en el mes"""
    if d.month == 12:
        nxt = date(d.year + 1, 1, 1)
    else:
        nxt = date(d.year, d.month + 1, 1)
    return (nxt - date(d.year, d.month, 1)).days

@dataclass
class InputData:
    """Estructura para los datos de entrada"""
    pmi_date: date
    final_date: date
    ibm: float
    edad: int
    incapacidad_pct: float
    incluir_20_pct: bool

@dataclass
class Results:
    """Estructura para los resultados de cálculo"""
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

class DataManager:
    """Gestor de datasets CSV - Versión corregida basada en BACK6"""
    
    def __init__(self):
        self.ipc_data = None
        self.pisos_data = None
        self.ripte_data = None
        self.tasa_data = None
        self.load_all_datasets()
    
    def _load_csv(self, path):
        """Carga CSV con múltiples separadores"""
        if not os.path.exists(path):
            messagebox.showerror("Error", f"No se encontró el dataset: {path}")
            return pd.DataFrame()
        
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                continue
        
        try:
            return pd.read_csv(path, sep=",", encoding="latin-1")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el dataset {path}.\n{e}")
            return pd.DataFrame()
    
    def load_all_datasets(self):
        """Carga todos los datasets usando métodos del BACK6"""
        try:
            # Cargar datasets
            self.ripte_data = self._load_csv(PATH_RIPTE)
            self.tasa_data = self._load_csv(PATH_TASA)  
            self.ipc_data = self._load_csv(PATH_IPC)
            self.pisos_data = self._load_csv(PATH_PISOS)
            
            # Normalizar usando métodos del BACK6
            self._norm_ripte()
            self._norm_tasa()
            self._norm_ipc()
            self._norm_pisos()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datasets: {str(e)}")
    
    def _norm_ripte(self):
        """Normalización RIPTE del BACK6"""
        if self.ripte_data.empty: 
            return
        cols = [c.lower() for c in self.ripte_data.columns]
        self.ripte_data.columns = cols
        
        # Manejar fecha compuesta (año + mes)
        if 'año' in cols and 'mes' in cols:
            # Diccionario para convertir nombres de meses a números
            meses_dict = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                'ene': 1, 'abr': 4, 'ago': 8, 'set': 9, 'dic': 12
            }
            
            def convertir_mes(valor):
                if pd.isna(valor):
                    return None
                valor_str = str(valor).strip().lower()
                
                # Si ya es un número, devolverlo como entero
                try:
                    return int(float(valor_str))
                except ValueError:
                    pass
                
                # Si es texto, buscar en el diccionario
                if valor_str in meses_dict:
                    return meses_dict[valor_str]
                
                # Intentar buscar por coincidencia parcial
                for mes_nombre, mes_num in meses_dict.items():
                    if mes_nombre.startswith(valor_str[:3]) or valor_str.startswith(mes_nombre[:3]):
                        return mes_num
                
                return None
            
            # Crear fecha combinando año y mes
            def crear_fecha_combined(row):
                try:
                    año = int(row['año'])
                    mes_num = convertir_mes(row['mes'])
                    if mes_num is None:
                        return None
                    return f"{año}-{mes_num:02d}-01"
                except (ValueError, TypeError):
                    return None
            
            self.ripte_data['fecha_combined'] = self.ripte_data.apply(crear_fecha_combined, axis=1)
            fecha_col = 'fecha_combined'
        else:
            fecha_col = None
            for c in cols:
                if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                    fecha_col = c
                    break
            if fecha_col is None:
                fecha_col = cols[0]
        
        # Buscar columna de valor RIPTE
        val_col = None
        if 'indice_ripte' in cols:
            val_col = 'indice_ripte'
        else:
            for c in cols:
                if ("ripte" in c) or ("valor" in c) or ("indice" in c):
                    val_col = c
                    break
            if val_col is None:
                num_cols = self.ripte_data.select_dtypes(include="number").columns.tolist()
                val_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        self.ripte_data["fecha"] = self.ripte_data[fecha_col].apply(safe_parse_date)
        self.ripte_data["ripte"] = pd.to_numeric(self.ripte_data[val_col], errors="coerce")
        self.ripte_data = self.ripte_data.dropna(subset=["fecha", "ripte"]).sort_values("fecha").reset_index(drop=True)

    def _norm_tasa(self):
        """Normalización TASA del BACK6"""
        if self.tasa_data.empty:
            return

        # Encabezados estandarizados
        cols = [str(c).strip().lower() for c in self.tasa_data.columns]
        self.tasa_data.columns = cols

        # Parseo de fechas
        if "desde" in self.tasa_data.columns:
            self.tasa_data["desde"] = self.tasa_data["desde"].apply(safe_parse_date)
        if "hasta" in self.tasa_data.columns:
            self.tasa_data["hasta"] = self.tasa_data["hasta"].apply(safe_parse_date)
        else:
            if "desde" in self.tasa_data.columns:
                self.tasa_data["hasta"] = self.tasa_data["desde"]

        # Columna fecha para el editor
        if "desde" in self.tasa_data.columns:
            self.tasa_data["fecha"] = self.tasa_data["desde"]

        # Determinar columna de valor mensual
        base_col = None
        for cand in ("valor", "porcentaje", "tasa"):
            if cand in self.tasa_data.columns:
                base_col = cand
                break
        if base_col is not None:
            self.tasa_data["tasa"] = pd.to_numeric(self.tasa_data[base_col], errors="coerce")

        # Limpiar y ordenar
        keep_cols = [c for c in ["fecha", "tasa", "desde", "hasta"] if c in self.tasa_data.columns]
        if "fecha" in self.tasa_data.columns and "tasa" in self.tasa_data.columns:
            self.tasa_data = (
                self.tasa_data.dropna(subset=["fecha", "tasa"])
                         .sort_values("fecha")
                         .reset_index(drop=True)
            )[keep_cols]

    def _norm_ipc(self):
        """Normalización IPC del BACK6"""
        if self.ipc_data.empty: 
            return
        cols = [c.lower() for c in self.ipc_data.columns]
        self.ipc_data.columns = cols
        
        # Usar 'periodo' como fecha
        fecha_col = None
        if 'periodo' in cols:
            fecha_col = 'periodo'
        else:
            for c in cols:
                if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                    fecha_col = c
                    break
            if fecha_col is None:
                fecha_col = cols[0]
        
        # Usar 'variacion_mensual' como valor IPC
        val_col = None
        if 'variacion_mensual' in cols:
            val_col = 'variacion_mensual'
        else:
            for c in cols:
                if ("variacion" in c) or ("inflacion" in c) or ("ipc" in c) or ("porcentaje" in c) or ("mensual" in c) or ("indice" in c):
                    val_col = c
                    break
            if val_col is None:
                num_cols = self.ipc_data.select_dtypes(include="number").columns.tolist()
                val_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        self.ipc_data["fecha"] = self.ipc_data[fecha_col].apply(safe_parse_date)
        self.ipc_data["ipc"] = pd.to_numeric(self.ipc_data[val_col], errors="coerce")
        self.ipc_data = self.ipc_data.dropna(subset=["fecha", "ipc"]).sort_values("fecha").reset_index(drop=True)

    def _norm_pisos(self):
        """Normalización PISOS del BACK6"""
        if self.pisos_data.empty: 
            return
        cols = [c.lower() for c in self.pisos_data.columns]
        self.pisos_data.columns = cols
        
        # Mapear nombres de columnas correctos
        desde = None; hasta = None; monto = None; res = None
        
        if 'fecha_inicio' in cols:
            desde = 'fecha_inicio'
        else:
            for c in cols:
                if ("desde" in c) or ("inicio" in c):
                    desde = c
                    break
        
        if 'fecha_fin' in cols:
            hasta = 'fecha_fin'
        else:
            for c in cols:
                if ("hasta" in c) or ("fin" in c):
                    hasta = c
                    break
        
        if 'monto_minimo' in cols:
            monto = 'monto_minimo'
        else:
            for c in cols:
                if ("piso" in c) or ("monto" in c) or ("minimo" in c) or ("base" in c):
                    monto = c
                    break
        
        if 'norma' in cols:
            res = 'norma'
        else:
            for c in cols:
                if ("res" in c) or ("resol" in c) or ("nota" in c) or ("exp" in c) or ("srt" in c) or ("norma" in c):
                    res = c
                    break
        
        if desde:
            self.pisos_data["desde"] = self.pisos_data[desde].apply(safe_parse_date)
        else:
            pcol = cols[0]
            self.pisos_data["desde"] = self.pisos_data[pcol].apply(safe_parse_date)
        
        if hasta:
            self.pisos_data["hasta"] = self.pisos_data[hasta].apply(safe_parse_date)
        else:
            self.pisos_data["hasta"] = None
        
        if monto:
            self.pisos_data["piso"] = pd.to_numeric(self.pisos_data[monto], errors="coerce")
        else:
            num_cols = self.pisos_data.select_dtypes(include="number").columns.tolist()
            self.pisos_data["piso"] = pd.to_numeric(self.pisos_data[num_cols[0]] if num_cols else self.pisos_data[cols[-1]], errors="coerce")
        
        self.pisos_data["resol"] = self.pisos_data[res].astype(str) if res else ""
        self.pisos_data = self.pisos_data.dropna(subset=["desde", "piso"]).sort_values("desde").reset_index(drop=True)
    
    def get_piso_minimo(self, fecha_pmi: date) -> Tuple[Optional[float], str]:
        """Método corregido del BACK6"""
        if self.pisos_data.empty:
            return (None, "")
            
        candidate = None
        for _, r in self.pisos_data.iterrows():
            d0 = r["desde"]
            d1 = r["hasta"] if not pd.isna(r["hasta"]) else None
            if d1 is None:
                if fecha_pmi >= d0:
                    candidate = (float(r["piso"]), r.get("resol", ""))
            else:
                if d0 <= fecha_pmi <= d1:
                    return (float(r["piso"]), r.get("resol", ""))
        return candidate if candidate else (None, "")
    
    def get_ripte_coeficiente(self, fecha_pmi: date, fecha_final: date) -> Tuple[float, float, float]:
        """Cálculo RIPTE corregido"""
        if self.ripte_data.empty:
            return 1.0, 0.0, 0.0
        
        # RIPTE en fecha PMI - buscar el último RIPTE <= fecha_pmi
        ripte_pmi_data = self.ripte_data[self.ripte_data['fecha'] <= fecha_pmi]
        if ripte_pmi_data.empty:
            ripte_pmi = float(self.ripte_data.iloc[0]['ripte'])
        else:
            ripte_pmi = float(ripte_pmi_data.iloc[-1]['ripte'])
        
        # RIPTE más reciente (último disponible)
        ripte_final = float(self.ripte_data.iloc[-1]['ripte'])
        
        # Coeficiente
        coeficiente = ripte_final / ripte_pmi if ripte_pmi > 0 else 1.0
        
        return coeficiente, ripte_pmi, ripte_final
    
    def calcular_tasa_activa(self, fecha_pmi: date, fecha_final: date, capital_base: float) -> Tuple[float, float]:
        """Cálculo corregido de tasa activa del BACK6 - FIJO de comparación de fechas"""
        if self.tasa_data.empty:
            return 0.0, capital_base
            
        total_aporte_pct = 0.0
        
        # Iterar sobre todas las filas de tasa para encontrar intersecciones
        for _, row in self.tasa_data.iterrows():
            # Obtener fechas del período de vigencia de la tasa
            if "desde" in self.tasa_data.columns and not pd.isna(row.get("desde")):
                fecha_desde = row["desde"]
            else:
                fecha_desde = row["fecha"]
                
            if "hasta" in self.tasa_data.columns and not pd.isna(row.get("hasta")):
                fecha_hasta = row["hasta"]
            else:
                # Si no hay fecha hasta, asumir que rige para todo el mes
                fecha_hasta = date(fecha_desde.year, fecha_desde.month, days_in_month(fecha_desde))
            
            # CORRECCIÓN: Asegurar que todas las fechas sean objetos date para comparación
            if isinstance(fecha_desde, pd.Timestamp):
                fecha_desde = fecha_desde.date()
            if isinstance(fecha_hasta, pd.Timestamp):
                fecha_hasta = fecha_hasta.date()
            
            # Calcular intersección del período de la tasa con el período de cálculo
            inicio_interseccion = max(fecha_pmi, fecha_desde)
            fin_interseccion = min(fecha_final, fecha_hasta)
            
            # Si hay intersección válida
            if inicio_interseccion <= fin_interseccion:
                dias_interseccion = (fin_interseccion - inicio_interseccion).days + 1
                
                # Obtener el valor mensual de la tasa
                if "tasa" in self.tasa_data.columns and not pd.isna(row.get("tasa")):
                    valor_mensual_pct = float(row["tasa"])
                elif "valor" in self.tasa_data.columns and not pd.isna(row.get("valor")):
                    valor_mensual_pct = float(row["valor"])
                else:
                    continue  # Si no hay valor de tasa, saltar esta fila
                
                # Calcular aporte prorrateado: Valor mensual (%) × días de intersección / 30
                aporte_pct = valor_mensual_pct * (dias_interseccion / 30.0)
                total_aporte_pct += aporte_pct
        
        # Aplicar fórmula final: total = base × (1 + Total % / 100)
        total_actualizado = capital_base * (1.0 + total_aporte_pct / 100.0)
        
        return total_aporte_pct, total_actualizado
    
    def calcular_inflacion(self, fecha_pmi: date, fecha_final: date) -> float:
        """Cálculo de inflación del BACK6 - FIJO de comparación de fechas"""
        if self.ipc_data.empty:
            return 0.0
            
        # CORRECCIÓN: Convertir a timestamp para comparaciones consistentes
        fecha_inicio_mes = pd.Timestamp(fecha_pmi.replace(day=1))
        fecha_final_mes = pd.Timestamp(fecha_final.replace(day=1))
        
        # Filtrar datos entre fechas
        ipc_periodo = self.ipc_data[
            (pd.to_datetime(self.ipc_data['fecha']) >= fecha_inicio_mes) &
            (pd.to_datetime(self.ipc_data['fecha']) <= fecha_final_mes)
        ]
        
        if ipc_periodo.empty:
            return 0.0
        
        # Fórmula: [(1 + r₁/100) × (1 + r₂/100) × ... × (1 + rₙ/100) - 1] × 100
        factor_acumulado = 1.0
        for _, row in ipc_periodo.iterrows():
            variacion = row['ipc']
            if not pd.isna(variacion):
                factor_acumulado *= (1 + variacion / 100)
        
        inflacion_acumulada = (factor_acumulado - 1) * 100
        return inflacion_acumulada

class Calculator:
    """Motor de cálculos corregido basado en BACK6"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def calcular_indemnizacion(self, input_data: InputData) -> Results:
        """Realiza todos los cálculos - Lógica corregida del BACK6"""
        
        # 1. Cálculo capital fórmula
        capital_formula = self._calcular_capital_formula(input_data)
        
        # 2. Verificar piso mínimo
        piso_minimo, piso_norma = self.data_manager.get_piso_minimo(input_data.pmi_date)
        capital_aplicado, piso_aplicado, piso_info = self._aplicar_piso_minimo(
            capital_formula, piso_minimo, piso_norma, input_data.incapacidad_pct
        )
        
        # 3. CORRECCIÓN CRÍTICA: Adicional 20% (del BACK6)
        adicional_20_pct = capital_aplicado * 0.20 if input_data.incluir_20_pct else 0.0
        capital_base = capital_aplicado + adicional_20_pct
        
        # 4. Actualización RIPTE + 3%
        ripte_coef, ripte_pmi, ripte_final = self.data_manager.get_ripte_coeficiente(
            input_data.pmi_date, input_data.final_date
        )
        ripte_actualizado = capital_base * ripte_coef
        
        # CORRECCIÓN CRÍTICA: Interés puro 3% anual se aplica sobre RIPTE ACTUALIZADO
        dias_transcurridos = (input_data.final_date - input_data.pmi_date).days
        interes_puro_3_pct = ripte_actualizado * 0.03 * (dias_transcurridos / 365.0)
        total_ripte_3 = ripte_actualizado + interes_puro_3_pct
        
        # 5. Tasa activa
        tasa_activa_pct, total_tasa_activa = self.data_manager.calcular_tasa_activa(
            input_data.pmi_date, input_data.final_date, capital_base
        )
        
        # 6. Inflación
        inflacion_acum_pct = self.data_manager.calcular_inflacion(
            input_data.pmi_date, input_data.final_date
        )
        
        return Results(
            capital_formula=capital_formula,
            capital_base=capital_base,
            piso_aplicado=piso_aplicado,
            piso_info=piso_info,
            adicional_20_pct=adicional_20_pct,
            ripte_coef=ripte_coef,
            ripte_actualizado=ripte_actualizado,
            interes_puro_3_pct=interes_puro_3_pct,
            total_ripte_3=total_ripte_3,
            tasa_activa_pct=tasa_activa_pct,
            total_tasa_activa=total_tasa_activa,
            inflacion_acum_pct=inflacion_acum_pct
        )
    
    def _calcular_capital_formula(self, input_data: InputData) -> float:
        """Calcula capital según fórmula CORREGIDA: IBM x 53 x (65/edad) x % incapacidad"""
        return input_data.ibm * 53 * (65 / input_data.edad) * (input_data.incapacidad_pct / 100)
    
    def _aplicar_piso_minimo(self, capital_formula: float, piso_minimo: Optional[float], 
                           piso_norma: str, incapacidad_pct: float) -> Tuple[float, bool, str]:
        """Aplica piso mínimo si corresponde"""
        if piso_minimo is None:
            return capital_formula, False, "No se encontró piso mínimo para la fecha"
        
        # El piso se aplica proporcionalmente a la incapacidad
        piso_proporcional = piso_minimo * (incapacidad_pct / 100)
        
        if capital_formula >= piso_proporcional:
            return capital_formula, False, f"Supera piso mínimo {piso_norma}"
        else:
            return piso_proporcional, True, f"Se aplica piso mínimo {piso_norma}"

class NumberUtils:
    """Utilidades para formateo de números"""
    
    @staticmethod
    def format_money(amount: float) -> str:
        """Formatea cantidad como dinero argentino"""
        return f"$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    
    @staticmethod
    def format_percentage(percentage: float) -> str:
        """Formatea porcentaje"""
        return f"{percentage:.2f}%".replace('.', ',')

class PrintToolsWindow:
    """Ventana de herramientas de impresión"""
    
    def __init__(self, parent, results: Results, input_data: InputData):
        self.results = results
        self.input_data = input_data
        
        self.window = tk.Toplevel(parent)
        self.window.title("Herramientas Print")
        self.window.geometry("800x600")
        self.window.configure(bg=COLORS['bg'])
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Tab 1: Sentencia
        self.create_sentencia_tab(notebook)
        
        # Tab 2: Liquidación
        self.create_liquidacion_tab(notebook)
        
        # Tab 3: Resultado
        self.create_resultado_tab(notebook)
    
    def create_sentencia_tab(self, notebook):
        """Crea tab de sentencia COMPLETAMENTE REESCRITO para funcionar"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Sentencia")
        
        # Frame para texto con scrollbar
        text_frame = tk.Frame(frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text widget CORREGIDO - permite selección y copia
        self.sentencia_text_widget = tk.Text(text_frame, wrap='word', font=FONTS['normal'], 
                                           state='normal', selectbackground='lightblue')
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame, command=self.sentencia_text_widget.yview)
        self.sentencia_text_widget.config(yscrollcommand=scrollbar.set)
        
        # Pack texto y scrollbar
        self.sentencia_text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Configurar tags REALES para formato
        self.sentencia_text_widget.tag_configure("bold", font=(FONTS['normal'][0], FONTS['normal'][1], 'bold'))
        self.sentencia_text_widget.tag_configure("normal", font=FONTS['normal'])
        
        # Generar y insertar texto USANDO LA FUNCIÓN CORRECTA
        self.generate_formatted_sentencia()
        
        # CRÍTICO: Hacer texto seleccionable pero no editable
        self.sentencia_text_widget.config(state='disabled')
        # Permitir selección incluso en estado disabled
        self.sentencia_text_widget.bind("<Button-1>", lambda e: self.sentencia_text_widget.config(state='normal'))
        self.sentencia_text_widget.bind("<ButtonRelease-1>", lambda e: self.sentencia_text_widget.after(10, lambda: self.sentencia_text_widget.config(state='disabled')))
        
        # Botón copiar
        copy_btn = tk.Button(frame, text="Copiar Todo", command=self.copy_sentencia_text,
                           bg=COLORS['secondary'], fg='white')
        copy_btn.pack(pady=10)
    
    def generate_formatted_sentencia(self):
        """FUNCIÓN PRINCIPAL que genera el texto con formato correcto"""
        self.sentencia_text_widget.delete('1.0', 'end')
        
        # a) Fórmula (negrita)
        self.sentencia_text_widget.insert('end', "a) Fórmula:\n", "bold")
        
        # Contenido de la fórmula
        formula_text = f"Valor de IBM ({NumberUtils.format_money(self.input_data.ibm)}) x 53 x "
        formula_text += f"65/edad({self.input_data.edad}) x Incapacidad ({self.input_data.incapacidad_pct}%)\n"
        formula_text += f"Capital calculado: {NumberUtils.format_money(self.results.capital_formula)}\n\n"
        self.sentencia_text_widget.insert('end', formula_text, "normal")
        
        # Texto del piso mínimo CORREGIDO
        piso_text = self.get_piso_text_corrected()
        self.sentencia_text_widget.insert('end', piso_text + "\n\n", "normal")
        
        # b) 20% (negrita)
        self.sentencia_text_widget.insert('end', "b) ", "bold")
        if self.input_data.incluir_20_pct:
            self.sentencia_text_widget.insert('end', f"20% Art. 3 Ley 26.773: {NumberUtils.format_money(self.results.adicional_20_pct)}\n\n", "normal")
        else:
            self.sentencia_text_widget.insert('end', "20% Art. 3 Ley 26.773: no se aplica\n\n", "normal")
        
        # Total (negrita) - PRIMERO
        self.sentencia_text_widget.insert('end', "Total: ", "bold")
        self.sentencia_text_widget.insert('end', f"{NumberUtils.format_money(self.results.capital_base)}\n", "normal")
        
        # Pesos en letras (negrita) - SEGUNDO
        letras = self.convert_to_words_fixed(self.results.capital_base)
        self.sentencia_text_widget.insert('end', letras + "\n\n", "bold")
        
        # c) Comparación (negrita) - TERCERO, DESPUÉS DEL TOTAL
        self.sentencia_text_widget.insert('end', "c) ", "bold")
        comparison_text = self.get_comparison_text_fixed()
        self.sentencia_text_widget.insert('end', comparison_text, "normal")
    
    def get_piso_text_corrected(self) -> str:
        """Genera texto CORREGIDO del piso mínimo"""
        # Obtener información real del piso (simulado por ahora)
        piso_minimo = 750000.0  # Valor de ejemplo
        piso_norma = "Resolución SRT 51/2022"
        piso_proporcional = piso_minimo * (self.input_data.incapacidad_pct / 100)
        
        if self.results.piso_aplicado:
            return (f"El monto es inferior al piso mínimo determinado por la {piso_norma}, "
                   f"que multiplicado por el porcentaje de incapacidad ({self.input_data.incapacidad_pct}%) "
                   f"alcanza la suma de {NumberUtils.format_money(piso_proporcional)}, por lo que se aplica este último.")
        else:
            return (f"Dicho monto supera el piso mínimo determinado por la {piso_norma}, "
                   f"que multiplicado por el porcentaje de incapacidad ({self.input_data.incapacidad_pct}%) "
                   f"alcanza la suma de {NumberUtils.format_money(piso_proporcional)}.")
    
    def convert_to_words_fixed(self, amount: float) -> str:
        """Conversión CORREGIDA a palabras en español"""
        try:
            from num2words import num2words
            entero = int(amount)
            centavos = int(round((amount - entero) * 100))
            palabras_entero = num2words(entero, lang='es').upper()
            return f"SON PESOS {palabras_entero} CON {centavos:02d}/100"
        except ImportError:
            return self.fallback_number_conversion(amount)
    
    def fallback_number_conversion(self, amount: float) -> str:
        """Conversión manual MEJORADA como fallback"""
        entero = int(amount)
        centavos = int(round((amount - entero) * 100))
        
        if entero == 0:
            return f"SON PESOS CERO CON {centavos:02d}/100"
        
        def convertir_centenas(n):
            if n == 0:
                return ""
            elif n < 20:
                unidades = ["", "UNO", "DOS", "TRES", "CUATRO", "CINCO", "SEIS", "SIETE", "OCHO", "NUEVE",
                           "DIEZ", "ONCE", "DOCE", "TRECE", "CATORCE", "QUINCE", "DIECISÉIS", 
                           "DIECISIETE", "DIECIOCHO", "DIECINUEVE"]
                return unidades[n]
            elif n < 30:
                if n == 20:
                    return "VEINTE"
                else:
                    return f"VEINTI{convertir_centenas(n-20)}"
            elif n < 100:
                decenas = ["", "", "VEINTE", "TREINTA", "CUARENTA", "CINCUENTA", 
                          "SESENTA", "SETENTA", "OCHENTA", "NOVENTA"]
                dec = n // 10
                uni = n % 10
                if uni == 0:
                    return decenas[dec]
                else:
                    return f"{decenas[dec]} Y {convertir_centenas(uni)}"
            else:  # 100-999
                centenas = ["", "CIENTO", "DOSCIENTOS", "TRESCIENTOS", "CUATROCIENTOS", 
                           "QUINIENTOS", "SEISCIENTOS", "SETECIENTOS", "OCHOCIENTOS", "NOVECIENTOS"]
                cen = n // 100
                resto = n % 100
                if n == 100:
                    return "CIEN"
                elif resto == 0:
                    return centenas[cen]
                else:
                    return f"{centenas[cen]} {convertir_centenas(resto)}"
        
        # Procesar número completo
        if entero < 1000:
            resultado = convertir_centenas(entero)
        elif entero < 1000000:
            miles = entero // 1000
            resto = entero % 1000
            if miles == 1:
                if resto == 0:
                    resultado = "MIL"
                else:
                    resultado = f"MIL {convertir_centenas(resto)}"
            else:
                if resto == 0:
                    resultado = f"{convertir_centenas(miles)} MIL"
                else:
                    resultado = f"{convertir_centenas(miles)} MIL {convertir_centenas(resto)}"
        else:  # Millones
            millones = entero // 1000000
            resto = entero % 1000000
            if millones == 1:
                if resto == 0:
                    resultado = "UN MILLÓN"
                else:
                    if resto < 1000:
                        resultado = f"UN MILLÓN {convertir_centenas(resto)}"
                    else:
                        miles = resto // 1000
                        unidades = resto % 1000
                        if unidades == 0:
                            if miles == 1:
                                resultado = "UN MILLÓN MIL"
                            else:
                                resultado = f"UN MILLÓN {convertir_centenas(miles)} MIL"
                        else:
                            if miles == 1:
                                resultado = f"UN MILLÓN MIL {convertir_centenas(unidades)}"
                            else:
                                resultado = f"UN MILLÓN {convertir_centenas(miles)} MIL {convertir_centenas(unidades)}"
            else:
                if resto == 0:
                    resultado = f"{convertir_centenas(millones)} MILLONES"
                else:
                    if resto < 1000:
                        resultado = f"{convertir_centenas(millones)} MILLONES {convertir_centenas(resto)}"
                    else:
                        miles = resto // 1000
                        unidades = resto % 1000
                        if unidades == 0:
                            if miles == 1:
                                resultado = f"{convertir_centenas(millones)} MILLONES MIL"
                            else:
                                resultado = f"{convertir_centenas(millones)} MILLONES {convertir_centenas(miles)} MIL"
                        else:
                            if miles == 1:
                                resultado = f"{convertir_centenas(millones)} MILLONES MIL {convertir_centenas(unidades)}"
                            else:
                                resultado = f"{convertir_centenas(millones)} MILLONES {convertir_centenas(miles)} MIL {convertir_centenas(unidades)}"
        
        return f"SON PESOS {resultado} CON {centavos:02d}/100"
    
    def get_comparison_text_fixed(self) -> str:
        """Genera texto de comparación CORREGIDO"""
        fecha_desde_str = self.input_data.pmi_date.strftime('%B %Y')
        meses_esp = {
            'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril',
            'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto',
            'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
        }
        for eng, esp in meses_esp.items():
            fecha_desde_str = fecha_desde_str.replace(eng, esp)
        
        return (f"Mientras la tasa legal aplicable (Tasa Activa Banco Nación) alcanzó para el período "
               f"comprometido ({fecha_desde_str} a la fecha) un total del {NumberUtils.format_percentage(self.results.tasa_activa_pct)}, "
               f"la inflación del mismo período alcanzó la suma de {NumberUtils.format_percentage(self.results.inflacion_acum_pct)}.")
    
    def copy_sentencia_text(self):
        """Copia todo el texto FUNCIONAL"""
        # Habilitar temporalmente para leer
        self.sentencia_text_widget.config(state='normal')
        texto_completo = self.sentencia_text_widget.get('1.0', 'end-1c')
        self.sentencia_text_widget.config(state='disabled')
        
        self.window.clipboard_clear()
        self.window.clipboard_append(texto_completo)
        
        # Efecto visual
        original_bg = self.sentencia_text_widget.cget('bg')
        self.sentencia_text_widget.configure(bg='lightgreen')
        self.window.after(500, lambda: self.sentencia_text_widget.configure(bg=original_bg))
    
    def create_liquidacion_tab(self, notebook):
        """Crea tab de liquidación"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Liquidación")
        
        text_widget = tk.Text(frame, wrap='word', font=FONTS['normal'])
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generar texto de liquidación
        liquidacion_text = self.generate_liquidacion_text()
        text_widget.insert('1.0', liquidacion_text)
        text_widget.config(state='disabled')
        
        # Botón copiar
        copy_btn = tk.Button(frame, text="Copiar", command=lambda: self.copy_text(text_widget),
                           bg=COLORS['secondary'], fg='white')
        copy_btn.pack(pady=10)
    
    def create_resultado_tab(self, notebook):
        """Crea tab de resultado"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Resultado")
        
        text_widget = tk.Text(frame, wrap='word', font=FONTS['normal'])
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generar texto de resultado
        resultado_text = self.generate_resultado_text()
        text_widget.insert('1.0', resultado_text)
        text_widget.config(state='disabled')
        
        # Botones
        btn_frame = tk.Frame(frame, bg=COLORS['bg'])
        btn_frame.pack(pady=10)
        
        copy_btn = tk.Button(btn_frame, text="Copiar", command=lambda: self.copy_text(text_widget),
                           bg=COLORS['secondary'], fg='white')
        copy_btn.pack(side='left', padx=5)
        
        print_btn = tk.Button(btn_frame, text="Imprimir", command=lambda: self.print_text(text_widget),
                            bg=COLORS['primary'], fg='white')
        print_btn.pack(side='left', padx=5)
    
    def generate_liquidacion_text(self) -> str:
        """Genera texto de liquidación con formato EXACTO especificado"""
        # Determinar qué actualización aplicar (la mayor)
        if self.results.total_ripte_3 >= self.results.total_tasa_activa:
            total_actualizacion = self.results.total_ripte_3
            metodo_usado = "RIPTE"
            capital_actualizado = self.results.ripte_actualizado
            interes_3_pct = self.results.interes_puro_3_pct
        else:
            total_actualizacion = self.results.total_tasa_activa
            metodo_usado = "TASA_ACTIVA"
            capital_actualizado = self.results.total_tasa_activa
            interes_3_pct = 0.0  # La tasa activa ya incluye todo
        
        # Calcular tasas judiciales
        tasa_justicia = total_actualizacion * 0.025  # 2.5%
        sobretasa_caja = tasa_justicia * 0.10  # 10% de la tasa
        total_final = total_actualizacion + tasa_justicia + sobretasa_caja
        
        # Formatear fechas
        fecha_pmi_str = self.input_data.pmi_date.strftime('%d/%m/%Y')
        fecha_final_str = self.input_data.final_date.strftime('%d/%m/%Y')
        
        # FORMATO EXACTO como se especificó
        texto = "Quilmes, en la fecha en que se suscribe con firma digital (Ac. SCBA. 3975/20). "
        texto += "**LIQUIDACION** que practica la Actuaria en el presente expediente. "
        texto += "** **\n"
        
        # Capital base
        texto += f"--Capital {NumberUtils.format_money(self.results.capital_base)} \n"
        
        if metodo_usado == "RIPTE":
            # Obtener fechas formateadas para RIPTE
            ripte_final_mes = self.input_data.final_date.strftime('%B/%Y').lower()
            ripte_pmi_mes = self.input_data.pmi_date.strftime('%B %Y').lower()
            
            # Traducir meses a español
            meses_esp = {
                'january': 'enero', 'february': 'febrero', 'march': 'marzo', 'april': 'abril',
                'may': 'mayo', 'june': 'junio', 'july': 'julio', 'august': 'agosto',
                'september': 'septiembre', 'october': 'octubre', 'november': 'noviembre', 'december': 'diciembre'
            }
            for eng, esp in meses_esp.items():
                ripte_final_mes = ripte_final_mes.replace(eng, esp)
                ripte_pmi_mes = ripte_pmi_mes.replace(eng, esp)
            
            # Simular valores RIPTE reales (en implementación real usar data_manager)
            ripte_final_valor = 172674.89
            ripte_pmi_valor = 6058.23
            coef_ripte = ripte_final_valor / ripte_pmi_valor
            coef_pct = (coef_ripte - 1) * 100
            
            texto += f"--Actualización mediante tasa de variación RIPTE, ({ripte_final_mes} {ripte_final_valor:,.2f}/ "
            texto += f"{ripte_pmi_mes} {ripte_pmi_valor:,.2f} = coef {coef_ripte:.2f} = {coef_pct:.0f}%) {NumberUtils.format_money(capital_actualizado)} \n"
            
            # Interés puro 3% si corresponde
            if interes_3_pct > 0:
                texto += f"--Interés puro del 3% anual desde {fecha_pmi_str} hasta {fecha_final_str} {NumberUtils.format_money(interes_3_pct)} \n"
        else:
            # Actualización por tasa activa
            texto += f"--Actualización mediante tasa activa BNA del {NumberUtils.format_percentage(self.results.tasa_activa_pct)} "
            texto += f"desde {fecha_pmi_str} hasta {fecha_final_str} {NumberUtils.format_money(capital_actualizado)} \n"
        
        # SUBTOTAL con formato exacto
        texto += f"--SUBTOTAL {NumberUtils.format_money(total_actualizacion)} \n\n"
        
        # Tasas judiciales con formato exacto (asterisco al final)
        texto += f"*Tasa de Justicia {NumberUtils.format_money(tasa_justicia)} *\n"
        texto += f"Sobretasa Contribución Caja de Abogados {NumberUtils.format_money(sobretasa_caja)} \n\n"
        
        # Total final en negrita
        texto += f"**TOTAL** **{NumberUtils.format_money(total_final)}** \n"
        
        # Conversión a palabras CORREGIDA
        palabras_monto = self.number_to_words_correct(total_final)
        texto += f"Importa la presente liquidación la suma de {palabras_monto} \n\n"
        
        # Texto legal final
        texto += "De la liquidación practicada, traslado a las partes por el plazo de cinco (5) días, "
        texto += "bajo apercibimiento de tenerla por consentida (art 59 de la Ley 15.057 - RC 1840/24 SCBA ) Notifíquese.-"
        
        return texto
    
    def number_to_words_correct(self, amount: float) -> str:
        """Conversión CORREGIDA a palabras en español"""
        try:
            from num2words import num2words
            entero = int(amount)
            centavos = int(round((amount - entero) * 100))
            palabras_entero = num2words(entero, lang='es').upper()
            return f"PESOS {palabras_entero} CON {centavos:02d}/100-"
        except ImportError:
            # Implementación manual CORRECTA
            return self.manual_words_conversion(amount)
    
    def manual_words_conversion(self, amount: float) -> str:
        """Conversión manual CORRECTA para números en español"""
        entero = int(amount)
        centavos = int(round((amount - entero) * 100))
        
        def unidades_a_palabras(n):
            if n == 0: return ""
            elif n == 1: return "UNO"
            elif n == 2: return "DOS"
            elif n == 3: return "TRES"
            elif n == 4: return "CUATRO"
            elif n == 5: return "CINCO"
            elif n == 6: return "SEIS"
            elif n == 7: return "SIETE"
            elif n == 8: return "OCHO"
            elif n == 9: return "NUEVE"
            elif n == 10: return "DIEZ"
            elif n == 11: return "ONCE"
            elif n == 12: return "DOCE"
            elif n == 13: return "TRECE"
            elif n == 14: return "CATORCE"
            elif n == 15: return "QUINCE"
            elif n == 16: return "DIECISÉIS"
            elif n == 17: return "DIECISIETE"
            elif n == 18: return "DIECIOCHO"
            elif n == 19: return "DIECINUEVE"
            elif n == 20: return "VEINTE"
            elif n < 30: return f"VEINTI{unidades_a_palabras(n-20)}"
            elif n < 100:
                decenas = ["", "", "VEINTE", "TREINTA", "CUARENTA", "CINCUENTA", 
                          "SESENTA", "SETENTA", "OCHENTA", "NOVENTA"]
                dec = n // 10
                uni = n % 10
                if uni == 0:
                    return decenas[dec]
                else:
                    return f"{decenas[dec]} Y {unidades_a_palabras(uni)}"
            elif n == 100: return "CIEN"
            elif n < 1000:
                centenas = ["", "CIENTO", "DOSCIENTOS", "TRESCIENTOS", "CUATROCIENTOS",
                           "QUINIENTOS", "SEISCIENTOS", "SETECIENTOS", "OCHOCIENTOS", "NOVECIENTOS"]
                cen = n // 100
                resto = n % 100
                if resto == 0:
                    return centenas[cen]
                else:
                    return f"{centenas[cen]} {unidades_a_palabras(resto)}"
            else:
                return str(n)
        
        def miles_a_palabras(n):
            if n == 0: return ""
            elif n == 1: return "MIL"
            elif n < 1000: return f"{unidades_a_palabras(n)} MIL"
            else: return f"{n} MIL"
        
        def millones_a_palabras(n):
            if n == 0: return ""
            elif n == 1: return "UN MILLÓN"
            else: return f"{unidades_a_palabras(n)} MILLONES"
        
        # Procesar el número completo
        if entero == 0:
            resultado = "CERO"
        elif entero < 1000:
            resultado = unidades_a_palabras(entero)
        elif entero < 1000000:
            miles = entero // 1000
            resto = entero % 1000
            if resto == 0:
                resultado = miles_a_palabras(miles)
            else:
                resultado = f"{miles_a_palabras(miles)} {unidades_a_palabras(resto)}"
        else:  # Millones
            millones = entero // 1000000
            resto = entero % 1000000
            
            if resto == 0:
                resultado = millones_a_palabras(millones)
            elif resto < 1000:
                resultado = f"{millones_a_palabras(millones)} {unidades_a_palabras(resto)}"
            else:
                miles = resto // 1000
                unidades_finales = resto % 1000
                if unidades_finales == 0:
                    resultado = f"{millones_a_palabras(millones)} {miles_a_palabras(miles)}"
                else:
                    resultado = f"{millones_a_palabras(millones)} {miles_a_palabras(miles)} {unidades_a_palabras(unidades_finales)}"
        
        return f"PESOS {resultado} CON {centavos:02d}/100-"
    
    def generate_resultado_text(self) -> str:
        """Genera texto de resultado completo"""
        texto = "CALCULADORA INDEMNIZACIONES LEY 24.557\n"
        texto += "TRIBUNAL DE TRABAJO NRO. 2 QUILMES\n"
        texto += "=" * 50 + "\n\n"
        
        texto += "DATOS DEL CASO:\n"
        texto += f"PMI: {self.input_data.pmi_date.strftime('%d/%m/%Y')}\n"
        texto += f"Fecha final: {self.input_data.final_date.strftime('%d/%m/%Y')}\n"
        texto += f"IBM: {NumberUtils.format_money(self.input_data.ibm)}\n"
        texto += f"Edad: {self.input_data.edad} años\n"
        texto += f"Incapacidad: {self.input_data.incapacidad_pct}%\n"
        texto += f"20% incluido: {'Sí' if self.input_data.incluir_20_pct else 'No'}\n\n"
        
        texto += "RESULTADOS:\n"
        texto += f"1. Capital Base: {NumberUtils.format_money(self.results.capital_base)}\n"
        texto += f"2. RIPTE + 3%: {NumberUtils.format_money(self.results.total_ripte_3)}"
        if self.results.total_ripte_3 >= self.results.total_tasa_activa:
            texto += " *** MÁS FAVORABLE ***"
        texto += "\n"
        texto += f"3. Tasa Activa: {NumberUtils.format_money(self.results.total_tasa_activa)}"
        if self.results.total_tasa_activa > self.results.total_ripte_3:
            texto += " *** MÁS FAVORABLE ***"
        texto += "\n"
        texto += f"4. Inflación Acumulada: {NumberUtils.format_percentage(self.results.inflacion_acum_pct)}\n"
        
        return texto
    
    def copy_text(self, text_widget):
        """Copia texto al portapapeles"""
        self.window.clipboard_clear()
        self.window.clipboard_append(text_widget.get('1.0', 'end-1c'))
        
        # Efecto visual
        original_bg = text_widget.cget('bg')
        text_widget.configure(bg='lightgreen')
        self.window.after(500, lambda: text_widget.configure(bg=original_bg))
    
    def print_text(self, text_widget):
        """Función de impresión MEJORADA - Genera documento A4 profesional"""
        # Solicitar carátula del expediente
        caratula = simpledialog.askstring(
            "Carátula del Expediente", 
            "Ingrese la carátula del expediente:",
            initialvalue="Expediente Nro. XXXXX - Apellido, Nombre c/ XXXX s/ Accidente de Trabajo"
        )
        
        if not caratula:
            return  # Usuario canceló
        
        # Generar documento para impresión
        self.generate_print_document(caratula)
    
    def generate_print_document(self, caratula: str):
        """Genera documento HTML A4 para impresión"""
        import tempfile
        import webbrowser
        
        # Crear HTML con diseño A4
        html_content = self.create_print_html(caratula)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        # Abrir en navegador para imprimir
        webbrowser.open(f'file://{temp_file}')
        
        # Mensaje informativo
        messagebox.showinfo(
            "Documento Generado", 
            "Se ha abierto el documento en su navegador.\n\n"
            "Para imprimir:\n"
            "• Presione Ctrl+P\n"
            "• Configure: Tamaño A4, Orientación Vertical\n"
            "• Para PDF: Seleccione 'Guardar como PDF' como destino"
        )
    
    def create_print_html(self, caratula: str) -> str:
        """Crea HTML con diseño profesional A4"""
        
        # Determinar método más favorable
        if self.results.total_ripte_3 >= self.results.total_tasa_activa:
            metodo_favorable = "RIPTE + 3%"
            total_favorable = self.results.total_ripte_3
        else:
            metodo_favorable = "Tasa Activa BNA"
            total_favorable = self.results.total_tasa_activa
        
        # Calcular tasas judiciales
        tasa_justicia = total_favorable * 0.025
        sobretasa_caja = tasa_justicia * 0.10
        total_final = total_favorable + tasa_justicia + sobretasa_caja
        
        # Formatear fechas
        fecha_pmi = self.input_data.pmi_date.strftime('%d/%m/%Y')
        fecha_final = self.input_data.final_date.strftime('%d/%m/%Y')
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cálculo de Indemnización - {caratula}</title>
    <style>
        @page {{
            size: A4 portrait;
            margin: 2cm 1.5cm;
        }}
        
        @media print {{
            body {{ 
                -webkit-print-color-adjust: exact;
                color-adjust: exact;
            }}
        }}
        
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 12pt;
            line-height: 1.4;
            color: #000;
            margin: 0;
            padding: 0;
        }}
        
        .header {{
            text-align: center;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        
        .header h1 {{
            font-size: 16pt;
            font-weight: bold;
            color: #2E86AB;
            margin: 0 0 10px 0;
        }}
        
        .header h2 {{
            font-size: 14pt;
            color: #666;
            margin: 0;
        }}
        
        .caratula {{
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 25px;
            text-align: center;
            border-radius: 5px;
        }}
        
        .caratula h3 {{
            font-size: 14pt;
            font-weight: bold;
            margin: 0;
            color: #2E86AB;
        }}
        
        .datos-caso {{
            background-color: #fff;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        
        .datos-caso h4 {{
            background-color: #2E86AB;
            color: white;
            margin: 0;
            padding: 10px 15px;
            font-size: 12pt;
            border-radius: 5px 5px 0 0;
        }}
        
        .datos-content {{
            padding: 15px;
        }}
        
        .datos-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px dotted #ccc;
        }}
        
        .datos-row:last-child {{
            border-bottom: none;
        }}
        
        .datos-label {{
            font-weight: bold;
            color: #333;
        }}
        
        .calculos {{
            margin-bottom: 20px;
        }}
        
        .calculo-item {{
            background-color: #f8f9fa;
            border-left: 4px solid #2E86AB;
            padding: 12px;
            margin-bottom: 10px;
        }}
        
        .calculo-titulo {{
            font-weight: bold;
            color: #2E86AB;
            font-size: 11pt;
            margin-bottom: 5px;
        }}
        
        .calculo-valor {{
            font-size: 14pt;
            font-weight: bold;
            color: #000;
        }}
        
        .calculo-detalle {{
            font-size: 10pt;
            color: #666;
            margin-top: 5px;
        }}
        
        .resultado-final {{
            background-color: #e8f5e8;
            border: 2px solid #28a745;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin: 25px 0;
        }}
        
        .resultado-final h3 {{
            color: #28a745;
            margin: 0 0 10px 0;
            font-size: 14pt;
        }}
        
        .resultado-final .monto {{
            font-size: 18pt;
            font-weight: bold;
            color: #000;
            margin: 10px 0;
        }}
        
        .liquidacion {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
        }}
        
        .liquidacion h4 {{
            color: #856404;
            margin-top: 0;
        }}
        
        .footer {{
            text-align: center;
            font-size: 10pt;
            color: #666;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }}
        
        .favorable {{
            background-color: #d4edda !important;
            border-left-color: #28a745 !important;
        }}
        
        .formula {{
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            padding: 12px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 11pt;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CÁLCULO DE INDEMNIZACIÓN LEY 24.557</h1>
        <h2>Tribunal de Trabajo Nro. 2 Quilmes</h2>
    </div>
    
    <div class="caratula">
        <h3>{caratula}</h3>
    </div>
    
    <div class="datos-caso">
        <h4>DATOS DEL CASO</h4>
        <div class="datos-content">
            <div class="datos-row">
                <span class="datos-label">Fecha del siniestro (PMI):</span>
                <span>{fecha_pmi}</span>
            </div>
            <div class="datos-row">
                <span class="datos-label">Fecha de cálculo:</span>
                <span>{fecha_final}</span>
            </div>
            <div class="datos-row">
                <span class="datos-label">Ingreso Base Mensual (IBM):</span>
                <span>{NumberUtils.format_money(self.input_data.ibm)}</span>
            </div>
            <div class="datos-row">
                <span class="datos-label">Edad del trabajador:</span>
                <span>{self.input_data.edad} años</span>
            </div>
            <div class="datos-row">
                <span class="datos-label">Porcentaje de incapacidad:</span>
                <span>{self.input_data.incapacidad_pct}%</span>
            </div>
            <div class="datos-row">
                <span class="datos-label">20% Art. 3 Ley 26.773:</span>
                <span>{'Incluido' if self.input_data.incluir_20_pct else 'No aplicado'}</span>
            </div>
        </div>
    </div>
    
    <div class="formula">
        <strong>Fórmula aplicada:</strong><br>
        IBM ({NumberUtils.format_money(self.input_data.ibm)}) × 53 × 65/edad({self.input_data.edad}) × Incapacidad ({self.input_data.incapacidad_pct}%)<br>
        <strong>Capital calculado:</strong> {NumberUtils.format_money(self.results.capital_formula)}
    </div>
    
    <div class="calculos">
        <div class="calculo-item">
            <div class="calculo-titulo">CAPITAL BASE (Ley 24.557)</div>
            <div class="calculo-valor">{NumberUtils.format_money(self.results.capital_base)}</div>
            <div class="calculo-detalle">
                Capital fórmula: {NumberUtils.format_money(self.results.capital_formula)} | 
                20%: {NumberUtils.format_money(self.results.adicional_20_pct) if self.input_data.incluir_20_pct else 'No aplica'}
            </div>
        </div>
        
        <div class="calculo-item {'favorable' if self.results.total_ripte_3 >= self.results.total_tasa_activa else ''}">
            <div class="calculo-titulo">ACTUALIZACIÓN RIPTE + 3%</div>
            <div class="calculo-valor">{NumberUtils.format_money(self.results.total_ripte_3)}</div>
            <div class="calculo-detalle">
                Coef. RIPTE: {self.results.ripte_coef:.6f} | 
                Actualizado: {NumberUtils.format_money(self.results.ripte_actualizado)} | 
                3% puro: {NumberUtils.format_money(self.results.interes_puro_3_pct)}
            </div>
        </div>
        
        <div class="calculo-item {'favorable' if self.results.total_tasa_activa > self.results.total_ripte_3 else ''}">
            <div class="calculo-titulo">ACTUALIZACIÓN TASA ACTIVA BNA</div>
            <div class="calculo-valor">{NumberUtils.format_money(self.results.total_tasa_activa)}</div>
            <div class="calculo-detalle">
                Porcentual del período: {NumberUtils.format_percentage(self.results.tasa_activa_pct)}
            </div>
        </div>
        
        <div class="calculo-item">
            <div class="calculo-titulo">INFLACIÓN ACUMULADA (Referencia)</div>
            <div class="calculo-valor">{NumberUtils.format_percentage(self.results.inflacion_acum_pct)}</div>
            <div class="calculo-detalle">
                Variación IPC del período {fecha_pmi} - {fecha_final}
            </div>
        </div>
    </div>
    
    <div class="resultado-final">
        <h3>MÉTODO MÁS FAVORABLE: {metodo_favorable}</h3>
        <div class="monto">{NumberUtils.format_money(total_favorable)}</div>
    </div>
    
    <div class="liquidacion">
        <h4>LIQUIDACIÓN JUDICIAL (con tasas)</h4>
        <div class="datos-row">
            <span class="datos-label">Subtotal:</span>
            <span>{NumberUtils.format_money(total_favorable)}</span>
        </div>
        <div class="datos-row">
            <span class="datos-label">Tasa de Justicia (2.5%):</span>
            <span>{NumberUtils.format_money(tasa_justicia)}</span>
        </div>
        <div class="datos-row">
            <span class="datos-label">Sobretasa Caja Abogados (10%):</span>
            <span>{NumberUtils.format_money(sobretasa_caja)}</span>
        </div>
        <div class="datos-row" style="border-top: 2px solid #333; padding-top: 8px; margin-top: 8px;">
            <span class="datos-label"><strong>TOTAL FINAL:</strong></span>
            <span><strong>{NumberUtils.format_money(total_final)}</strong></span>
        </div>
    </div>
    
    <div class="footer">
        <p>Documento generado por Sistema de Cálculo de Indemnizaciones LRT<br>
        Tribunal de Trabajo Nro. 2 Quilmes<br>
        Fecha de emisión: {fecha_final}</p>
    </div>
</body>
</html>
"""
        return html

class MainApplication:
    """Aplicación principal con correcciones aplicadas"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.data_manager = DataManager()
        self.calculator = Calculator(self.data_manager)
        self.current_results = None
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        self.root.title("Calculadora Indemnizaciones LRT")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['bg'])
        
        # Header
        self.create_header()
        
        # Main content
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Form
        self.create_form_panel(main_frame)
        
        # Right panel - Results
        self.create_results_panel(main_frame)
        
        # Footer
        self.create_footer()
        
        # Bind Enter key
        self.root.bind('<Return>', lambda e: self.calculate())
    
    def create_header(self):
        """Crea el encabezado"""
        header_frame = tk.Frame(self.root, bg=COLORS['primary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="CALCULADORA INDEMNIZACIONES LEY 24.557\nTRIBUNAL DE TRABAJO NRO. 2 QUILMES",
            font=FONTS['title'],
            bg=COLORS['primary'],
            fg='white',
            justify='center'
        )
        title_label.pack(expand=True)
    
    def create_form_panel(self, parent):
        """Crea panel del formulario"""
        left_frame = tk.Frame(parent, bg=COLORS['bg'])
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Form container
        form_frame = tk.LabelFrame(left_frame, text="Datos del Caso", font=FONTS['heading'], 
                                  bg=COLORS['light'], fg=COLORS['dark'])
        form_frame.pack(fill='x', pady=(0, 20))
        
        # Variables
        self.var_pmi = tk.StringVar()
        self.var_final = tk.StringVar(value=date.today().strftime("%d/%m/%Y"))
        self.var_ibm = tk.StringVar()
        self.var_edad = tk.StringVar()
        self.var_incapacidad = tk.StringVar()
        self.var_incluir_20 = tk.BooleanVar(value=True)
        
        # Form fields
        self.create_form_field(form_frame, "Fecha del siniestro (PMI):", self.var_pmi)
        self.create_form_field(form_frame, "Fecha final:", self.var_final)
        self.create_form_field(form_frame, "Ingreso Base Mensual (IBM):", self.var_ibm)
        self.create_form_field(form_frame, "Edad del trabajador:", self.var_edad)
        self.create_form_field(form_frame, "Porcentaje de incapacidad (%):", self.var_incapacidad)
        
        # Checkbox - CORRECCIÓN: añadir callback para recalcular automáticamente
        check_frame = tk.Frame(form_frame, bg=COLORS['light'])
        check_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Checkbutton(
            check_frame,
            text="Incluir 20% (art. 3, Ley 26.773)",
            variable=self.var_incluir_20,
            bg=COLORS['light'],
            font=FONTS['normal'],
            command=self._on_checkbox_change  # AÑADIR CALLBACK
        ).pack(anchor='w')
        
        # Calculate button
        calc_button = tk.Button(
            form_frame,
            text="CALCULAR",
            command=self.calculate,
            bg=COLORS['primary'],
            fg='white',
            font=FONTS['heading'],
            pady=10,
            cursor='hand2'
        )
        calc_button.pack(pady=20)
        
        # Formula display
        self.create_formula_display(left_frame)
        
        # Alerts
        self.create_alerts_panel(left_frame)
    
    def _on_checkbox_change(self):
        """CORRECCIÓN: Recalcular cuando cambia el checkbox del 20%"""
        # Solo recalcular si ya hay datos válidos
        if all([self.var_pmi.get(), self.var_final.get(), self.var_ibm.get(), 
                self.var_edad.get(), self.var_incapacidad.get()]):
            try:
                self.calculate()
            except:
                pass  # Ignore si hay errores de validación
    
    def create_form_field(self, parent, label_text, variable):
        """Crea un campo del formulario"""
        field_frame = tk.Frame(parent, bg=COLORS['light'])
        field_frame.pack(fill='x', padx=10, pady=5)
        
        label = tk.Label(field_frame, text=label_text, bg=COLORS['light'], 
                        font=FONTS['normal'], width=25, anchor='w')
        label.pack(side='left')
        
        entry = tk.Entry(field_frame, textvariable=variable, font=FONTS['normal'], width=20)
        entry.pack(side='left', padx=(10, 0))
    
    def create_formula_display(self, parent):
        """Crea display de la fórmula"""
        formula_frame = tk.LabelFrame(parent, text="Fórmula Aplicada", font=FONTS['heading'],
                                     bg=COLORS['light'], fg=COLORS['dark'])
        formula_frame.pack(fill='x', pady=(0, 10))
        
        self.formula_label = tk.Label(
            formula_frame,
            text="Valor de IBM x 53 x Incapacidad x 65/edad",
            bg=COLORS['light'],
            font=FONTS['normal'],
            justify='left'
        )
        self.formula_label.pack(anchor='w', padx=10, pady=10)
    
    def create_alerts_panel(self, parent):
        """Crea panel de alertas"""
        alerts_frame = tk.LabelFrame(parent, text="Últimos Datos Disponibles", 
                                   font=FONTS['heading'], bg=COLORS['info'], fg='white')
        alerts_frame.pack(fill='both', expand=True)
        
        self.alerts_label = tk.Label(
            alerts_frame,
            text=self.get_alerts_text(),
            bg=COLORS['info'],
            fg='white',
            font=FONTS['small'],
            justify='left',
            anchor='nw'
        )
        self.alerts_label.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_results_panel(self, parent):
        """Crea panel de resultados"""
        right_frame = tk.Frame(parent, bg=COLORS['bg'])
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Result cards
        self.capital_card = self.create_result_card(right_frame, "CAPITAL BASE (INDEMNIZACIÓN LEY 24.557)")
        self.ripte_card = self.create_result_card(right_frame, "ACTUALIZACIÓN RIPTE + 3%")
        self.tasa_card = self.create_result_card(right_frame, "ACTUALIZACIÓN TASA ACTIVA (Art. 12 inc. 2 Ley 24.557)")
        self.inflacion_card = self.create_result_card(right_frame, "INFLACIÓN ACUMULADA")
        
        for card in [self.capital_card, self.ripte_card, self.tasa_card, self.inflacion_card]:
            card.pack(fill='x', pady=10)
    
    def create_result_card(self, parent, title):
        """Crea una tarjeta de resultado"""
        card = tk.LabelFrame(parent, text=title, font=FONTS['heading'], 
                           bg=COLORS['light'], fg=COLORS['dark'])
        
        amount_label = tk.Label(card, text="$ 0,00", font=FONTS['large_money'], 
                              bg=COLORS['light'], fg=COLORS['dark'])
        amount_label.pack(anchor='w', padx=10, pady=(10, 5))
        
        desc_label = tk.Label(card, text="", font=FONTS['small'], 
                            bg=COLORS['light'], fg=COLORS['dark'], 
                            justify='left', wraplength=400)
        desc_label.pack(fill='x', padx=10, pady=(0, 10))
        
        card.amount_label = amount_label
        card.desc_label = desc_label
        return card
    
    def create_footer(self):
        """Crea pie de página con botones"""
        footer_frame = tk.Frame(self.root, bg=COLORS['bg'], height=60)
        footer_frame.pack(fill='x', side='bottom', padx=20, pady=10)
        footer_frame.pack_propagate(False)
        
        button_frame = tk.Frame(footer_frame, bg=COLORS['bg'])
        button_frame.pack(anchor='center', expand=True)
        
        buttons = [
            ("Herramientas Print", self.show_print_tools),
            ("Ver Mínimos SRT", self.show_minimos_srt),
            ("Editar Tablas", self.edit_tables),
            ("Información", self.show_info)
        ]
        
        for text, command in buttons:
            btn = tk.Button(
                button_frame,
                text=text,
                command=command,
                bg=COLORS['secondary'],
                fg='white',
                font=FONTS['normal'],
                padx=15,
                cursor='hand2'
            )
            btn.pack(side='left', padx=5)
    
    def get_alerts_text(self):
        """Obtiene texto de alertas mejorado con información completa"""
        alerts = []
        
        # RIPTE con monto y mes
        if not self.data_manager.ripte_data.empty:
            ultimo_ripte = self.data_manager.ripte_data.iloc[-1]
            fecha_ripte = ultimo_ripte['fecha']
            valor_ripte = ultimo_ripte['ripte']
            
            # Formatear mes/año
            meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
            mes_nombre = meses[fecha_ripte.month - 1] if isinstance(fecha_ripte, date) else "N/D"
            año = fecha_ripte.year if isinstance(fecha_ripte, date) else "N/D"
            
            # Calcular salario equivalente aproximado
            salario_equiv = valor_ripte * 1000  # Factor aproximado
            
            alerts.append(f"RIPTE {mes_nombre}/{año}: Índice {valor_ripte:.0f} - Salario: {NumberUtils.format_money(salario_equiv)}")
        
        # IPC con variación y mes
        if not self.data_manager.ipc_data.empty:
            ultimo_ipc = self.data_manager.ipc_data.iloc[-1]
            fecha_ipc = ultimo_ipc['fecha']
            variacion_ipc = ultimo_ipc['ipc']
            
            # Formatear mes/año para IPC
            if isinstance(fecha_ipc, date):
                mes_ipc = f"{fecha_ipc.month:02d}/{fecha_ipc.year}"
            else:
                mes_ipc = "N/D"
            
            alerts.append(f"IPC {mes_ipc}: Variación {NumberUtils.format_percentage(variacion_ipc)}")
        
        # Tasa activa con valor y mes
        if not self.data_manager.tasa_data.empty:
            ultima_tasa = self.data_manager.tasa_data.iloc[-1]
            fecha_tasa = ultima_tasa.get('fecha', None)
            valor_tasa = ultima_tasa['tasa']
            
            # Formatear mes/año para tasa
            if isinstance(fecha_tasa, date):
                mes_tasa = f"{fecha_tasa.month:02d}/{fecha_tasa.year}"
            else:
                mes_tasa = "N/D"
            
            alerts.append(f"TASA ACTIVA {mes_tasa}: {NumberUtils.format_percentage(valor_tasa)}")
        
        # Pisos SRT - norma y monto en la misma línea
        if not self.data_manager.pisos_data.empty:
            ultimo_piso = self.data_manager.pisos_data.iloc[-1]
            norma = ultimo_piso['resol']
            monto_piso = ultimo_piso['piso']
            
            alerts.append(f"PISO SRT {norma}: {NumberUtils.format_money(monto_piso)}")
        
        return '\n'.join(alerts) if alerts else "Verificar datasets CSV"
    
    def calculate(self):
        """Realiza los cálculos"""
        try:
            # Validar y obtener datos
            input_data = self.get_input_data()
            
            # Realizar cálculos
            self.current_results = self.calculator.calcular_indemnizacion(input_data)
            
            # Mostrar resultados
            self.display_results(input_data, self.current_results)
            
            # Actualizar fórmula
            self.update_formula_display(input_data, self.current_results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo: {str(e)}")
    
    def get_input_data(self) -> InputData:
        """Obtiene y valida datos del formulario"""
        try:
            pmi_str = self.var_pmi.get().strip()
            final_str = self.var_final.get().strip()
            
            # Parsear fechas usando la función corregida
            pmi_date = safe_parse_date(pmi_str)
            final_date = safe_parse_date(final_str)
            
            if not pmi_date or not final_date:
                raise ValueError("Fechas inválidas")
            
            # Validar fechas
            if pmi_date > final_date:
                raise ValueError("La fecha PMI no puede ser posterior a la fecha final")
            
            # Parsear números
            ibm = float(self.var_ibm.get().replace("$", "").replace(".", "").replace(",", "."))
            edad = int(self.var_edad.get())
            incapacidad_pct = float(self.var_incapacidad.get().replace("%", "").replace(",", "."))
            
            # Validaciones
            if ibm <= 0:
                raise ValueError("El IBM debe ser mayor a cero")
            if edad <= 0 or edad > 100:
                raise ValueError("La edad debe estar entre 1 y 100 años")
            if incapacidad_pct <= 0 or incapacidad_pct > 100:
                raise ValueError("El porcentaje de incapacidad debe estar entre 0.01 y 100")
            
            return InputData(
                pmi_date=pmi_date,
                final_date=final_date,
                ibm=ibm,
                edad=edad,
                incapacidad_pct=incapacidad_pct,
                incluir_20_pct=self.var_incluir_20.get()
            )
            
        except ValueError as e:
            raise ValueError(f"Error en los datos ingresados: {str(e)}")
    
    def display_results(self, input_data: InputData, results: Results):
        """Muestra los resultados en las tarjetas"""
        
        # Capital Base
        capital_desc = f"Capital fórmula: {NumberUtils.format_money(results.capital_formula)}"
        if results.adicional_20_pct > 0:
            capital_desc += f" + 20%: {NumberUtils.format_money(results.adicional_20_pct)}"
        else:
            capital_desc += " / 20%: no aplica"
        capital_desc += f"\n{results.piso_info}"
        
        self.update_card(self.capital_card, results.capital_base, capital_desc)
        
        # RIPTE + 3%
        ripte_desc = f"Coef. RIPTE: {results.ripte_coef:.6f}\n"
        ripte_desc += f"Total actualizado: {NumberUtils.format_money(results.ripte_actualizado)}\n"
        ripte_desc += f"3% puro: {NumberUtils.format_money(results.interes_puro_3_pct)}"
        
        self.update_card(self.ripte_card, results.total_ripte_3, ripte_desc, 
                        highlight=(results.total_ripte_3 >= results.total_tasa_activa))
        
        # Tasa Activa
        tasa_desc = f"Porcentual total del período: {NumberUtils.format_percentage(results.tasa_activa_pct)}"
        
        self.update_card(self.tasa_card, results.total_tasa_activa, tasa_desc,
                        highlight=(results.total_tasa_activa > results.total_ripte_3))
        
        # Inflación
        inflacion_desc = f"Inflación acumulada del período"
        self.inflacion_card.amount_label.config(text=NumberUtils.format_percentage(results.inflacion_acum_pct))
        self.inflacion_card.desc_label.config(text=inflacion_desc)
    
    def update_card(self, card, amount, description, highlight=False):
        """Actualiza una tarjeta de resultado"""
        if isinstance(amount, (int, float)):
            card.amount_label.config(text=NumberUtils.format_money(amount))
        else:
            card.amount_label.config(text=str(amount))
        
        card.desc_label.config(text=description)
        
        if highlight:
            # CORRECCIÓN: Usar siempre verde para la opción más favorable
            bg_color = COLORS['highlight_ripte']  # Verde para cualquier opción favorable
            card.config(bg=bg_color)
            card.amount_label.config(bg=bg_color)
            card.desc_label.config(bg=bg_color)
        else:
            card.config(bg=COLORS['light'])
            card.amount_label.config(bg=COLORS['light'])
            card.desc_label.config(bg=COLORS['light'])
    
    def update_formula_display(self, input_data: InputData, results: Results):
        """Actualiza display de fórmula con detalles del piso mínimo aplicado"""
        # Fórmula en el orden correcto
        formula_text = f"IBM ({NumberUtils.format_money(input_data.ibm)}) x 53 x "
        formula_text += f"65/edad({input_data.edad}) x Incapacidad ({input_data.incapacidad_pct}%)\n"
        
        # Capital calculado con detalle de piso mínimo si aplica
        if results.piso_aplicado:
            # Cuando se aplica piso mínimo, mostrar ambos valores
            piso_minimo_valor = self._get_piso_minimo_aplicado(input_data.pmi_date, input_data.incapacidad_pct)
            formula_text += f"Capital calculado: {NumberUtils.format_money(results.capital_formula)} "
            formula_text += f"(Se aplica mínimo: {NumberUtils.format_money(piso_minimo_valor)})\n"
        else:
            # Cuando supera el piso, solo mostrar el capital calculado
            formula_text += f"Capital calculado: {NumberUtils.format_money(results.capital_formula)}\n"
        
        # Agregar información del período del piso mínimo
        piso_info_completa = self._add_period_to_piso_info(input_data.pmi_date, results.piso_info)
        formula_text += piso_info_completa
        
        self.formula_label.config(text=formula_text)
    
    def _get_piso_minimo_aplicado(self, pmi_date: date, incapacidad_pct: float) -> float:
        """Obtiene el valor del piso mínimo aplicado (proporcional a la incapacidad)"""
        piso_minimo, _ = self.data_manager.get_piso_minimo(pmi_date)
        if piso_minimo:
            return piso_minimo * (incapacidad_pct / 100)
        return 0.0
    
    def _add_period_to_piso_info(self, pmi_date: date, piso_info: str) -> str:
        """Agrega información del período a la información del piso ya calculada correctamente"""
        if not self.data_manager.pisos_data.empty and ("Supera piso mínimo" in piso_info or "Se aplica piso mínimo" in piso_info):
            # Buscar el piso que se aplicó usando la misma lógica que DataManager.get_piso_minimo()
            candidate_row = None
            for _, row in self.data_manager.pisos_data.iterrows():
                d0 = row["desde"]
                d1 = row["hasta"] if not pd.isna(row["hasta"]) else None
                
                if d1 is None:
                    if pmi_date >= d0:
                        candidate_row = row
                else:
                    if d0 <= pmi_date <= d1:
                        candidate_row = row
                        break
            
            if candidate_row is not None:
                # Formatear el período
                desde_str = candidate_row['desde'].strftime('%d/%m/%Y') if isinstance(candidate_row['desde'], date) else str(candidate_row['desde'])
                
                if not pd.isna(candidate_row['hasta']) and isinstance(candidate_row['hasta'], date):
                    hasta_str = candidate_row['hasta'].strftime('%d/%m/%Y')
                    periodo_texto = f" correspondiente al período {desde_str} - {hasta_str}"
                else:
                    periodo_texto = f" correspondiente al período desde {desde_str}"
                
                return piso_info + periodo_texto
        
        # Si no se puede determinar el período, devolver el texto original
        return piso_info
    
    def show_print_tools(self):
        """Muestra ventana de herramientas de impresión"""
        if self.current_results is None:
            messagebox.showwarning("Aviso", "Primero debe realizar un cálculo")
            return
        
        PrintToolsWindow(self.root, self.current_results, self.get_input_data())
    
    def show_minimos_srt(self):
        """Muestra tabla de mínimos SRT"""
        MinimosWindow(self.root, self.data_manager)
    
    def edit_tables(self):
        """Permite editar las tablas"""
        password = simpledialog.askstring("Autorización", "Ingrese contraseña:", show='*')
        if password != DEFAULT_PASSWORD:
            messagebox.showerror("Acceso Denegado", "Contraseña incorrecta")
            return
        
        EditTablesWindow(self.root, self.data_manager)
    
    def show_info(self):
        """Muestra información del sistema"""
        InfoWindow(self.root)
    
    def run(self):
        """Ejecuta la aplicación"""
        self.root.mainloop()

class MinimosWindow:
    """Ventana para mostrar mínimos SRT"""
    
    def __init__(self, parent, data_manager: DataManager):
        self.data_manager = data_manager
        
        self.window = tk.Toplevel(parent)
        self.window.title("Mínimos SRT")
        self.window.geometry("900x600")
        self.window.configure(bg=COLORS['bg'])
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz"""
        # Header
        header = tk.Label(self.window, text="MÍNIMOS DE LA SRT", font=FONTS['title'],
                         bg=COLORS['primary'], fg='white', pady=15)
        header.pack(fill='x')
        
        # Table frame
        table_frame = tk.Frame(self.window, bg=COLORS['bg'])
        table_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Treeview
        columns = ('resol', 'desde', 'hasta', 'piso')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Headers
        self.tree.heading('resol', text='Norma')
        self.tree.heading('desde', text='Vigencia Desde')
        self.tree.heading('hasta', text='Vigencia Hasta')
        self.tree.heading('piso', text='Monto Mínimo')
        
        # Column widths
        self.tree.column('resol', width=250)
        self.tree.column('desde', width=150)
        self.tree.column('hasta', width=150)
        self.tree.column('piso', width=200)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Load data
        self.load_data()
        
        # Close button
        close_btn = tk.Button(self.window, text="Cerrar", command=self.window.destroy,
                             bg=COLORS['secondary'], fg='white', font=FONTS['normal'])
        close_btn.pack(pady=10)
    
    def load_data(self):
        """Carga datos en la tabla"""
        if not self.data_manager.pisos_data.empty:
            for i, (_, row) in enumerate(self.data_manager.pisos_data.iterrows()):
                # Alternar colores de fila
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                
                # Formatear fechas
                desde_str = row['desde'].strftime('%d/%m/%Y') if isinstance(row['desde'], date) else str(row['desde'])
                hasta_str = row['hasta'].strftime('%d/%m/%Y') if isinstance(row['hasta'], date) and not pd.isna(row['hasta']) else 'Vigente'
                piso_str = NumberUtils.format_money(row['piso'])
                
                self.tree.insert('', 'end', values=(
                    row['resol'],
                    desde_str,
                    hasta_str,
                    piso_str
                ), tags=(tag,))
            
            # Configurar estilos alternados
            self.tree.tag_configure('evenrow', background='#f0f0f0')
            self.tree.tag_configure('oddrow', background='white')

class EditTablesWindow:
    """Ventana para editar tablas"""
    
    def __init__(self, parent, data_manager: DataManager):
        self.data_manager = data_manager
        
        self.window = tk.Toplevel(parent)
        self.window.title("Editar Tablas")
        self.window.geometry("1000x700")
        self.window.configure(bg=COLORS['bg'])
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz"""
        # Warning
        warning = tk.Label(
            self.window,
            text="⚠️ ATENCIÓN: La modificación de tablas podría traer consecuencias irreversibles para los resultados finales",
            font=FONTS['normal'],
            bg=COLORS['info'],
            fg='white',
            pady=10
        )
        warning.pack(fill='x')
        
        # Notebook
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Tabs para cada dataset
        datasets = [
            ('PISOS', self.data_manager.pisos_data, PATH_PISOS),
            ('RIPTE', self.data_manager.ripte_data, PATH_RIPTE),
            ('TASA', self.data_manager.tasa_data, PATH_TASA),
            ('IPC', self.data_manager.ipc_data, PATH_IPC)
        ]
        
        for name, data, filename in datasets:
            if not data.empty:
                frame = tk.Frame(notebook, bg=COLORS['bg'])
                notebook.add(frame, text=name)
                TableEditor(frame, data, filename, self.data_manager, self.on_datasets_saved)
    
    def on_datasets_saved(self):
        """Callback cuando se guardan los datasets"""
        # Recargar los datos
        self.data_manager.load_all_datasets()
        messagebox.showinfo("Datos", "Datasets recargados.")

class TableEditor:
    """Editor de tabla individual"""
    
    def __init__(self, parent, data, filename, data_manager, on_save=None):
        self.parent = parent
        self.data = data.copy()
        self.filename = filename
        self.data_manager = data_manager
        self.on_save = on_save
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz del editor"""
        # Toolbar
        toolbar = tk.Frame(self.parent, bg=COLORS['bg'])
        toolbar.pack(fill='x', pady=10)
        
        tk.Button(toolbar, text="Agregar", command=self.add_row,
                 bg=COLORS['success'], fg='white').pack(side='left', padx=5)
        tk.Button(toolbar, text="Editar", command=self.edit_row,
                 bg=COLORS['primary'], fg='white').pack(side='left', padx=5)
        tk.Button(toolbar, text="Borrar", command=self.delete_row,
                 bg=COLORS['info'], fg='white').pack(side='left', padx=5)
        tk.Button(toolbar, text="Guardar", command=self.save_data,
                 bg=COLORS['secondary'], fg='white').pack(side='right', padx=5)
        
        # Table frame
        table_frame = tk.Frame(self.parent, bg=COLORS['bg'])
        table_frame.pack(fill='both', expand=True)
        
        # Table
        self.tree = ttk.Treeview(table_frame, show='headings')
        
        if not self.data.empty:
            columns = list(self.data.columns)
            self.tree['columns'] = columns
            
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.refresh_table()
    
    def refresh_table(self):
        """Refresca la tabla"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add data
        if not self.data.empty:
            for i, (_, row) in enumerate(self.data.iterrows()):
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                vals = []
                for col in self.data.columns:
                    val = row[col]
                    if isinstance(val, (datetime, date)):
                        val = val.strftime("%d/%m/%Y")
                    elif pd.isna(val):
                        val = ''
                    vals.append(str(val))
                self.tree.insert('', 'end', values=vals, tags=(tag,))
            
            # Configure alternating colors
            self.tree.tag_configure('evenrow', background='#f0f0f0')
            self.tree.tag_configure('oddrow', background='white')
    
    def add_row(self):
        """Agrega nueva fila"""
        self._open_row_editor(new=True)
    
    def edit_row(self):
        """Edita fila seleccionada"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Editar", "Seleccione una fila para editar")
            return
        index = self.tree.index(selection[0])
        self._open_row_editor(new=False, row_index=index)
    
    def delete_row(self):
        """Borra fila seleccionada"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Borrar", "Seleccione una fila para borrar")
            return
        
        if messagebox.askyesno("Confirmar", "¿Está seguro de borrar la fila seleccionada?"):
            # Get index
            index = self.tree.index(selection[0])
            # Drop from dataframe
            self.data = self.data.drop(self.data.index[index]).reset_index(drop=True)
            # Refresh table
            self.refresh_table()
    
    def save_data(self):
        """Guarda los datos"""
        if messagebox.askyesno("Guardar", "ATENCIÓN: la modificación de tablas podría traer consecuencias irreversibles. ¿Continuar?"):
            try:
                self.data.to_csv(self.filename, index=False)
                messagebox.showinfo("Guardar", "Datos guardados exitosamente")
                if callable(self.on_save):
                    self.on_save()
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar: {str(e)}")
    
    def _open_row_editor(self, new: bool, row_index: Optional[int] = None):
        """Abre editor de fila"""
        win = tk.Toplevel(self.parent)
        win.title("Agregar fila" if new else "Editar fila")
        win.geometry("500x400")
        
        cols = list(self.data.columns) if not self.data.empty else []
        if new and not cols:
            messagebox.showinfo("Estructura", "Dataset sin columnas.")
            win.destroy()
            return
        
        entries = {}
        for c in cols:
            fr = tk.Frame(win)
            fr.pack(fill='x', pady=4, padx=8)
            tk.Label(fr, text=c, width=20, anchor="w").pack(side='left')
            ent = tk.Entry(fr, width=30)
            ent.pack(side='left', fill='x', expand=True)
            entries[c] = ent
        
        # Fill existing data
        if not new and row_index is not None:
            row = self.data.iloc[row_index]
            for c in cols:
                val = row[c]
                if isinstance(val, (datetime, date)):
                    val = val.strftime("%d/%m/%Y")
                elif pd.isna(val):
                    val = ''
                entries[c].insert(0, str(val))
        
        def save_row():
            vals = {c: entries[c].get() for c in cols}
            # Parse dates
            for c in cols:
                if ("fecha" in c.lower()) or ("desde" in c.lower()) or ("hasta" in c.lower()) or ("periodo" in c.lower()):
                    dt = safe_parse_date(vals[c])
                    vals[c] = dt if dt else vals[c]
            
            if new:
                new_row = pd.DataFrame([vals])
                self.data = pd.concat([self.data, new_row], ignore_index=True)
            else:
                for c in cols:
                    self.data.at[row_index, c] = vals[c]
            
            self.refresh_table()
            win.destroy()
        
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Guardar", command=save_row, bg=COLORS['primary'], fg='white').pack(side='left', padx=5)
        tk.Button(btn_frame, text="Cancelar", command=win.destroy, bg=COLORS['secondary'], fg='white').pack(side='left', padx=5)

class InfoWindow:
    """Ventana de información"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Información del Sistema")
        self.window.geometry("900x700")
        self.window.configure(bg=COLORS['bg'])
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz"""
        # Header
        header = tk.Label(
            self.window,
            text="INFORMACIÓN DEL SISTEMA",
            font=FONTS['title'],
            bg=COLORS['primary'],
            fg='white',
            pady=20
        )
        header.pack(fill='x')
        
        # Notebook
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Tab 1: Fórmulas
        self.create_formulas_tab(notebook)
        
        # Tab 2: Fuentes
        self.create_sources_tab(notebook)
        
        # Tab 3: Marco Legal
        self.create_legal_tab(notebook)
        
        # Close button
        close_btn = tk.Button(self.window, text="Cerrar", command=self.window.destroy,
                             bg=COLORS['secondary'], fg='white', font=FONTS['heading'])
        close_btn.pack(pady=20)
    
    def create_formulas_tab(self, notebook):
        """Crea tab de fórmulas"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Fórmulas")
        
        text = tk.Text(frame, wrap='word', font=FONTS['normal'], bg='white')
        text.pack(fill='both', expand=True, padx=20, pady=20)
        
        content = """
FÓRMULAS APLICADAS:

1. CAPITAL BASE (Ley 24.557):
   Capital = IBM × 53 × (% Incapacidad / 100) × (65 / Edad)
   
   - Se compara con piso mínimo vigente a la fecha PMI
   - Si el piso es mayor, se aplica el piso proporcional a la incapacidad
   - Se agrega 20% adicional según Art. 3 Ley 26.773 (excepto in itinere)

2. ACTUALIZACIÓN RIPTE + 3% (Doctrina "Barrios"):
   - Coeficiente RIPTE = RIPTE Final / RIPTE PMI
   - Capital actualizado = Capital Base × Coeficiente RIPTE
   - Interés puro 3% = Capital Actualizado RIPTE × 0.03 × (días / 365.25)
   - Total = Capital actualizado + Interés puro 3%

3. TASA ACTIVA BNA (Art. 12 inc. 2 Ley 24.557):
   - Se aplica la tasa activa promedio del Banco Nación
   - Cálculo mensual prorrateado por días
   - Suma acumulativa sin capitalización

4. INFLACIÓN ACUMULADA:
   Inflación = [(1 + r₁/100) × (1 + r₂/100) × ... × (1 + rₙ/100) - 1] × 100
   
   Donde rₙ es la variación mensual del IPC

CRITERIO DE APLICACIÓN:
Se aplica la actualización más favorable entre RIPTE+3% y Tasa Activa.
La inflación se muestra como referencia comparativa.
        """
        
        text.insert('1.0', content)
        text.config(state='disabled')
    
    def create_sources_tab(self, notebook):
        """Crea tab de fuentes"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Fuentes de Datos")
        
        # Grid para las fuentes
        sources_frame = tk.Frame(frame, bg=COLORS['bg'])
        sources_frame.pack(expand=True, padx=20, pady=20)
        
        sources = [
            ("INDEC", "Índice de Precios al Consumidor (IPC)", "https://www.indec.gob.ar/"),
            ("BCRA", "Banco Central - Tasas de referencia", "https://www.bcra.gob.ar/"),
            ("MTySS", "Ministerio de Trabajo - RIPTE", "https://www.argentina.gob.ar/trabajo"),
            ("BNA", "Banco Nación - Tasas activas", "https://www.bna.com.ar/"),
            ("SRT", "Superintendencia de Riesgos del Trabajo", "https://www.srt.gob.ar/")
        ]
        
        for i, (name, desc, url) in enumerate(sources):
            row = i // 2
            col = i % 2
            
            card = tk.LabelFrame(sources_frame, text=name, font=FONTS['heading'],
                               bg=COLORS['light'], fg=COLORS['primary'])
            card.grid(row=row, column=col, padx=10, pady=10, sticky='ew')
            
            tk.Label(card, text=desc, bg=COLORS['light'], 
                    font=FONTS['normal'], wraplength=200).pack(padx=10, pady=5)
            
            tk.Button(card, text="Visitar sitio", 
                     command=lambda u=url: webbrowser.open(u),
                     bg=COLORS['secondary'], fg='white',
                     font=FONTS['small']).pack(pady=5)
        
        sources_frame.grid_columnconfigure(0, weight=1)
        sources_frame.grid_columnconfigure(1, weight=1)
    
    def create_legal_tab(self, notebook):
        """Crea tab de marco legal"""
        frame = tk.Frame(notebook, bg=COLORS['bg'])
        notebook.add(frame, text="Marco Legal")
        
        text = tk.Text(frame, wrap='word', font=FONTS['normal'], bg='white')
        text.pack(fill='both', expand=True, padx=20, pady=20)
        
        content = """
MARCO NORMATIVO:

LEY 24.557 - RIESGOS DEL TRABAJO:
- Art. 14: Fórmula de cálculo de incapacidad permanente parcial
- Art. 12 inc. 2: Actualización por tasa activa BNA

LEY 26.773 - RÉGIMEN DE ORDENAMIENTO LABORAL:
- Art. 3: Incremento del 20% sobre prestaciones dinerarias
- Excepción: No aplica para accidentes in itinere

DECRETO 1694/2009:
- Actualización de prestaciones según RIPTE
- Metodología de aplicación del coeficiente

JURISPRUDENCIA RELEVANTE:

SCBA "Barrios" (C. 124.096, 17/04/2024):
- Inconstitucionalidad sobrevenida del Art. 7 Ley 23.928
- Actualización por RIPTE + 3% anual de interés puro
- Doctrina aplicable ante deterioro inflacionario

SCBA "Muzychuk" (L. 120.800, 14/07/2025):
- Inconstitucionalidad del Decreto 669/19
- IBM histórico actualizado exclusivamente por Art. 12 Ley 24.557

PRINCIPIOS APLICABLES:
- Razonabilidad (Art. 28 CN)
- Derecho de propiedad (Art. 17 CN)
- Reparación integral (Art. 19 CN)
- Tutela judicial efectiva
- Principio protectorio del derecho laboral
        """
        
        text.insert('1.0', content)
        text.config(state='disabled')

def main():
    """Función principal"""
    try:
        # Verificar que existan los datasets
        required_files = ['dataset_ipc.csv', 'dataset_pisos.csv', 'dataset_ripte.csv', 'dataset_tasa.csv']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Archivos Faltantes", 
                f"No se encontraron los siguientes archivos CSV:\n\n" + 
                "\n".join(missing_files) + 
                "\n\nPor favor, coloque estos archivos en el mismo directorio que la aplicación."
            )
            return
        
        # Inicializar aplicación
        app = MainApplication()
        app.run()
        
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error Fatal", f"Error al inicializar la aplicación:\n\n{str(e)}")

# GUI launch removed for Streamlit.

"""

# Ejecutamos el núcleo para definir: InputData, Results, DataManager, Calculator, NumberUtils, etc.
_globals = {}
exec(_CORE_CODE, _globals)

InputData = _globals['InputData']
Results = _globals['Results']
DataManager = _globals['DataManager']
Calculator = _globals['Calculator']
NumberUtils = _globals.get('NumberUtils', None)

# ---- Interfaz Streamlit ----
st.title("⚖️ Calculadora de Indemnizaciones (Ley 24.557)")
st.caption("Versión web basada en el código original. Interfaz Tkinter sustituida por Streamlit, lógica intacta.")

with st.sidebar:
    st.header("Parámetros de entrada")
    pmi = st.date_input("PMI (fecha del siniestro)", value=date(2020,1,1))
    final = st.date_input("Fecha final (hasta)", value=date.today())
    ibm = st.number_input("Ingreso Base Mensual (IBM)", min_value=0.0, step=1000.0, format="%.2f")
    edad = st.number_input("Edad del trabajador", min_value=18, max_value=100, value=40)
    incap = st.number_input("Porcentaje de incapacidad (%)", min_value=0.01, max_value=100.0, value=10.0, step=0.5, format="%.2f")
    inc20 = st.checkbox("Aplicar 20% (art. 3 Ley 26.773)", value=True)
    do_calc = st.button("Calcular")

# Cargar datasets automáticamente al iniciar
if 'dm' not in st.session_state:
    st.session_state.dm = DataManager()
dm = st.session_state.dm

calc = Calculator(dm)

# Panel principal
col1, col2 = st.columns([1.2, 1])

if do_calc:
    try:
        # Construir InputData (misma estructura que el original)
        inp = InputData(
            pmi_date=pmi,
            final_date=final,
            ibm=float(ibm),
            edad=int(edad),
            incapacidad_pct=float(incap),
            incluir_20_pct=bool(inc20),
        )
        res = calc.calcular_indemnizacion(inp)

        with col1:
            st.subheader("Resultados")
            def fmt_money(x):
                if NumberUtils and hasattr(NumberUtils, 'format_money'):
                    return NumberUtils.format_money(x)
                return f"${{x:,.2f}}"
            def fmt_pct(x):
                if NumberUtils and hasattr(NumberUtils, 'format_percent'):
                    return NumberUtils.format_percent(x)
                return f"{{x:.2f}} %"

            st.metric("Capital fórmula", fmt_money(res.capital_formula))
            st.metric("Capital base", fmt_money(res.capital_base))

            if getattr(res, 'piso_aplicado', False):
                st.warning(f"Piso mínimo aplicado: **{{res.piso_info}}**")
            else:
                st.info("No corresponde piso mínimo.")

            st.write("---")
            st.write("### Actualizaciones y accesorios")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**RIPTE**")
                st.write(f"Coeficiente: {{res.ripte_coef:.4f}}")
                st.write(f"Actualización: {{fmt_money(res.ripte_actualizado)}}")
            with c2:
                st.write("**Interés puro 3% anual**")
                st.write(f"{{fmt_money(res.interes_puro_3_pct)}}")
                st.write(f"Total RIPTE + 3%: {{fmt_money(res.total_ripte_3)}}")
            with c3:
                st.write("**Tasa activa**")
                st.write(f"Tasa: {{fmt_pct(res.tasa_activa_pct)}}")
                st.write(f"Total con tasa activa: {{fmt_money(res.total_tasa_activa)}}")

            st.write("---")
            st.write("**Inflación acumulada (IPC):** ", fmt_pct(getattr(res, 'inflacion_acum_pct', 0.0)))

        with col2:
            st.subheader("Detalles")
            st.json({
                "pmi_date": str(inp.pmi_date),
                "final_date": str(inp.final_date),
                "ibm": inp.ibm,
                "edad": inp.edad,
                "incapacidad_pct": inp.incapacidad_pct,
                "incluir_20_pct": inp.incluir_20_pct,
                "piso_info": getattr(res, 'piso_info', ""),
            })

    except Exception as e:
        st.error(f"Error en el cálculo: {{e}}")

st.write("---")
with st.expander("Ver tablas cargadas (RIPTE / Tasa / IPC / Pisos mínimos)"):
    tabs = st.tabs(["RIPTE", "Tasa activa", "IPC", "Pisos mínimos"])
    with tabs[0]:
        st.dataframe(dm.ripte_data if dm.ripte_data is not None else pd.DataFrame())
    with tabs[1]:
        st.dataframe(dm.tasa_data if dm.tasa_data is not None else pd.DataFrame())
    with tabs[2]:
        st.dataframe(dm.ipc_data if dm.ipc_data is not None else pd.DataFrame())
    with tabs[3]:
        st.dataframe(dm.pisos_data if dm.pisos_data is not None else pd.DataFrame())
