# app.py － 百大建商｜關係鏈分析（單頁搜尋 v5）
# 重點：保留「上游 / 直接合作對象」區塊 + 強化「經銷商競爭者」指標 + 圖表可切換圓餅/長條 + 百分比兩位小數

import io
import re
import math
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="百大建商｜關係鏈分析（單頁搜尋 v5）", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜關係鏈分析（單頁搜尋 v5）")
st.caption("D=建設、E=營造、F=水電、G=年使用量(萬元；僅水電視圖顯示備註)、H/J/L=經銷商、I/K/M=配比。選任一公司，查看上游/直接合作對象與競爭者。")

# ====================== Helpers ======================
@st.cache_data
def read_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

def clean_name(x):
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if s == "" or s.lower() in {"nan", "none"} or s == "0":
        return np.nan
    return s

def coerce_num(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).replace(",", "").replace("%", "").strip()
    if s in ("", "nan", "None"):
        return np.nan
    try:

