# app.py － 百大建商｜關係鏈分析（單頁搜尋 v10）
# 更新要點：
# - 建設公司：移除競爭者功能；概覽「次數」欄名改「合作次數」；經銷商區塊改名「終端經銷商」
# - 水電公司：合作對象顯示預估年使用量；圖表以金額(配比×年用量)作為圓餅值
# - 經銷商：競爭者區塊新增「去重後整體被競爭覆蓋率」(Union overlap≤100%)

import io
import re
import math
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ====================== 基本設定與樣式 ======================
st.set_page_config(page_title="百大建商｜關係鏈分析（單頁搜尋 v10）", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜關係鏈分析（單頁搜尋 v10）")
try:
    p = Path(__file__)
    st.caption(
        f"🔖 版本：v10 | 檔案：{p.name} | 修改時間：{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
    )
except Exception:
    st.caption("🔖 版本：v10")

st.markdown(
    """
    <style>
    .chip {display:inline-block; padding:4px 10px; border-radius:999px; background:#F1F5F9; border:1px solid #E2E8F0; font-size:12px; margin-right:8px;}
    .card {padding:16px 16px; border-radius:16px; border:1px solid #E2E8F0; background:#FFFFFF; box-shadow:0 1px 2px rgba(0,0,0,0.04); margin-bottom:12px;}
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { border: 1px solid #e5e7eb; padding: 6px 12px; border-radius: 10px; background: #fafafa; }
    .stTabs [aria-selected="true"] { background: #eef2ff !important; border-color:#c7d2fe !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        return float(s)
    except Exception:
        return np.nan

def normalize_ratio(series):
    s = series.apply(coerce_num)
    if s.max(skipna=True) is not None and s.max(skipna=True) > 1.000001:
        return s / 100.0
    return s

def pct_str(x):
    """輸入 0~1 或 0~100，輸出 'xx.xx%'（四捨五入兩位）"""
    if pd.isna(x):
        return "-"
    v = float(x)
    if v <= 1.0:
        v = v * 100.0
    d = Decimal(str(v)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
    return f"{d}%"

def get_col_by_pos_or_name(df, pos, name_candidates):
    cols = df.columns.tolist()
    try:
        col_by_pos = df.columns[pos]
    except Exception:
        col_by_pos = None
    if col_by_pos is not None:
        return col_by_pos
    for n in name_candidates:
        if n in cols:
            return n
    return None

# 平均配比（按水電等權）
def avg_dealer_ratio_across_unique_mep(rel_subset: pd.DataFrame) -> pd.DataFrame:
    meps = [m for m in rel_subset["水電公司"].dropna().unique().tolist() if isinstance(m, str) and m != ""]
    n = len(meps)
    if n == 0:
        return pd.DataFrame(columns=["經銷商","平均配比"])
    sums = defaultdict(float)
    for mep in meps:
        g = rel_subset[rel_subset["水電公司"] == mep]
        rmap = g.groupby("經銷商")["配比"].mean().to_dict()
        for d, r in rmap.items():
            if pd.isna(d):
                continue
            sums[str(d)] += float(r or 0.0)
    rows = [(dealer, s / n) for dealer, s in sums.items()]
    out = pd.DataFrame(rows, columns=["經銷商","平均配比"]).sort_values("平均配比", ascending=False)
    return out

# ====================== 上傳 ======================
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔（固定欄位：D=建設、E=營造、F=水電、G=年用量、H/J/L=經銷、I/K/M=配比）",
    type=["xlsx", "xls", "csv"],
    help="Excel 需使用 openpyxl 解析",
)
if not file:
    st.info("請先上傳檔案。")
    st.stop()

df_raw = read_any(file)

# 欄位定位（0-based）：D=3 E=4 F=5 G=6 H=7 I=8 J=9 K=10 L=11 M=12
col_dev = get_col_by_pos_or_name(df_raw, 3, ["建商","建設公司","建設公司(業主)"])
col_con = get_col_by_pos_or_name(df_raw, 4, ["營造公司","營造商"])
col_mep = get_col_by_pos_or_name(df_raw, 5, ["水電全名","水電公司","機電公司","機電廠商"])
col_vol = get_col_by_pos_or_name(df_raw, 6, ["年使用量/萬","年使用量(萬)","用量_萬"])
col_dA = get_col_by_pos_or_name(df_raw, 7, ["經銷商A","經銷A","經銷商1"])
col_rA = get_col_by_pos_or_name(df_raw, 8, ["經銷A佔比(%)","經銷商A配比","A配比"])
col_dB = get_col_by_pos_or_name(df_raw, 9, ["經銷商B","經銷B","經銷商2"])
col_rB = get_col_by_pos_or_name(df_raw, 10, ["經銷B佔比(%)","經銷商B配比","B配比"])
col_dC = get_col_by_pos_or_name(df_raw, 11, ["經銷商C","經銷Ｃ","經銷商3"])
col_rC = get_col_by_pos_or_name(df_raw, 12, ["經銷Ｃ佔比(%)","經銷C佔比(%)","經銷商C配比","C配比"])

required = [col_dev, col_con, col_mep]
if any(c is None for c in required):
    st.error("找不到必要欄位（依欄位位置 D/E/F 取得失敗）。請確認資料欄序。")
    st.stop()

# ====================== 轉換（不以G加權；% 兩位小數） ======================
df = df_raw.rename(columns={
    col_dev:"建設公司", col_con:"營造公司", col_mep:"水電公司",
    (col_vol or "G"): "年使用量_萬",
    col_dA:"經銷商A", col_rA:"經銷A比",
    col_dB:"經銷商B", col_rB:"經銷B比",
    col_dC:"經銷商C", col_rC:"經銷C比",
}).copy()

for c in ["建設公司","營造公司","水電公司","經銷商A","經銷商B","經銷商C"]:
    if c in df.columns:
        df[c] = df[c].apply(clean_name)

if "年使用量_萬" in df.columns:
    df["年使用量_萬"] = df["年使用量_萬"].apply(coerce_num)

for c in ["經銷A比","經銷B比","經銷C比"]:
    if c in df.columns:
        df[c] = normalize_ratio(df[c])

# 經銷展開（用配比，不乘以G）
dealer_blocks = []
if "經銷商A" in df.columns and "經銷A比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商A","經銷A比"]]
                         .rename(columns={"經銷商A":"經銷商","經銷A比":"配比"}))
if "經銷商B" in df.columns and "經銷B比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商B","經銷B比"]]
                         .rename(columns={"經銷商B":"經銷商","經銷B比":"配比"}))
if "經銷商C" in df.columns and "經銷C比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商C","經銷C比"]]
                         .rename(columns={"經銷商C":"經銷商","經銷C比":"配比"}))
rel = (pd.concat(dealer_blocks, ignore_index=True)
       if dealer_blocks else pd.DataFrame(columns=["建設公司","營造公司","水電公司","經銷商","配比"]))
rel["經銷商"] = rel["經銷商"].apply(clean_name)
rel = rel.dropna(subset=["經銷商","水電公司"]).copy()

# 水電年用量（每家水電一個值；若重複取首個非空）
mep_vol_map = df.groupby("水電公司")["年使用量_萬"].apply(
    lambda s: s.dropna().iloc[0] if s.dropna().size>0 else np.nan
).to_dict()

# ====================== 角色選擇 ======================
role = st.radio("角色", ["建設公司", "營造公司", "水電公司", "經銷商"], horizontal=True)
chart_type = st.radio("圖表", ["圓餅圖", "長條圖"], index=0, horizontal=True)

def options_for(role):
    if role == "建設公司":
        return sorted(df["建設公司"].dropna().unique().tolist())
    if role == "營造公司":
        return sorted(df["營造公司"].dropna().unique().tolist())
    if role == "水電公司":
        return sorted(df["水電公司"].dropna().unique().tolist())
    if role == "經銷商":
        return sorted(rel["經銷商"].dropna().unique().tolist())
    return []

all_opts = options_for(role)
kw = st.text_input("搜尋公司（模糊）", "")
filtered_opts = [o for o in all_opts if isinstance(o, str) and (kw.lower() in o.lower() if kw else True)]
target = st.selectbox("選擇公司", filtered_opts)
if not target:
    st.stop()

st.markdown(f'<span class="chip">{role}</span><span class="chip">{target}</span>', unsafe_allow_html=True)

# ====================== 共用小工具 ======================
def share_table(df_in, group_cols, name_col):
    cnt = df_in.groupby(group_cols).size().reset_index(name="次數")
    tot = cnt["次數"].sum()
    if tot == 0:
        return pd.DataFrame(columns=[name_col,"次數","占比"])
    cnt["占比"] = cnt["次數"] / tot
    cnt["占比"] = cnt["占比"].apply(pct_str)
    return cnt.sort_values("次數", ascending=False)

def draw_chart(df_plot, name_col, value_col, title):
    if df_plot is None or df_plot.empty:
        st.info("沒有資料可視覺化。")
        return
    pastel = px.colors.qualitative.Pastel
    if chart_type == "長條圖":
        fig = px.bar(df_plot, x=name_col, y=value_col, title=title,
                     color=name_col, color_discrete_sequence=pastel, template="simple_white")
        fig.update_layout(showlegend=False)
    else:
        fig = px.pie(df_plot, names=name_col, values=value_col, title=title,
                     color=name_col, color_discrete_sequence=pastel, template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

# ====================== 資料切片 ======================
down_dealer_raw = None   # 視覺用數值（建設/營造：平均配比；水電：額度_萬）
down_dealer_tbl = None   # 表格用（％字串或含額度）
down_mep = None          # 水電公司列表（含次數/占比；經銷商視圖會加配比）
up_tbl = None            # 上游或合作營造列表

if role == "建設公司":
    df_sel = df[df["建設公司"] == target]
    up_tbl = share_table(df_sel, ["營造公司"], "營造公司")      # for 概覽：營造公司
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")     # for 概覽：水電公司
    rel_sel = rel[rel["建設公司"] == target]
    down_dealer_raw = avg_dealer_ratio_across_unique_mep(rel_sel)
    down_dealer_tbl = down_dealer_raw.copy()
    if not down_dealer_tbl.empty:
        down_dealer_tbl["平均配比"] = down_dealer_tbl["平均配比"].apply(pct_str)

elif role == "營造公司":
    df_sel = df[df["營造公司"] == target]
    up_tbl = share_table(df_sel, ["建設公司"], "建設公司")
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")
    rel_sel = rel[rel["營造公司"] == target]
    down_dealer_raw = avg_dealer_ratio_across_unique_mep(rel_sel)
    down_dealer_tbl = down_dealer_raw.copy()
    if not down_dealer_tbl.empty:
        down_dealer_tbl["平均配比"] = down_dealer_tbl["平均配比"].apply(pct_str)

elif role == "水電公司":
    df_sel = df[df["水電公司"] == target]
    # 經銷商在該水電的配比 + 額度(萬)（配比×年用量）
    rel_sel = rel[rel["水電公司"] == target]
    mep_vol = df_sel["年使用量_萬"].dropna().unique()
    vol_val = float(mep_vol[0]) if len(mep_vol) > 0 and not pd.isna(mep_vol[0]) else 0.0

    dealer_ratio = (rel_sel.groupby("經銷商")["配比"].mean()
                    .reset_index().sort_values("配比", ascending=False))
    dealer_ratio["額度_萬"] = dealer_ratio["配比"].astype(float) * vol_val
    down_dealer_raw = dealer_ratio.rename(columns={"配比":"配比"})  # 視覺用：我們會用 額度_萬
    down_dealer_tbl = down_dealer_raw.copy()
    if not down_dealer_tbl.empty:
        down_dealer_tbl["配比"] = down_dealer_tbl["配比"].apply(pct_str)
        down_dealer_tbl["額度_萬"] = down_dealer_tbl["額度_萬"].round(2)

    # 上游：建設×營造（顯示在概覽）
    up_tbl = share_table(
        df_sel.assign(_公司=df_sel["建設公司"].fillna("")+" × "+df_sel["營造公司"].fillna("")),
        ["_公司"], "公司"
    )
    down_mep = None  # 水電視圖不需要列出其他水電

elif role == "經銷商":
    # 經銷商視圖：合作水電（並加上此經銷在各水電的平均配比）
    df_sel = rel[rel["經銷商"] == target].merge(
        df, on=["建設公司","營造公司","水電公司"], how="left", suffixes=("","_df")
    )
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")
    r_df = (rel[rel["經銷商"] == target]
            .groupby("水電公司")["配比"].mean().reset_index()
            .rename(columns={"配比":"該經銷商配比"}))
    if not r_df.empty:
        r_df["該經銷商配比"] = r_df["該經銷商配比"].apply(pct_str)
        down_mep = down_mep.merge(r_df, on="水電公司", how="left")
    up_tbl = None

# ====================== KPI（橫排） ======================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    cnt_rows = len(df_sel) if isinstance(df_sel, pd.DataFrame) else 0
    n_dev = df_sel["建設公司"].nunique() if "建設公司" in df_sel.columns else 0
    n_con = df_sel["營造公司"].nunique() if "營造公司" in df_sel.columns else 0
    n_mep = df_sel["水電公司"].nunique() if "水電公司" in df_sel.columns else 0
    if role in ["建設公司","營造公司","水電公司"]:
        n_dealer = (rel[rel[role]==target]["經銷商"].nunique()
                    if role in ["建設公司","營造公司"]
                    else (rel[rel["水電公司"]==target]["經銷商"].nunique()))
    else:
        n_dealer = df_sel["經銷商"].nunique() if "經銷商" in df_sel.columns else 0
    c1.metric("資料筆數", f"{cnt_rows:,}")
    c2.metric("建設家數", f"{n_dev:,}")
    c3.metric("營造家數", f"{n_con:,}")
    c4.metric("水電家數", f"{n_mep:,}")
    c5.metric("經銷家數", f"{n_dealer:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# ====================== Tabs：概覽 / 合作對象 / 競爭者 / 匯出 ======================
tab_overview, tab_partners, tab_comp, tab_export = st.tabs(["概覽", "合作對象", "競爭者", "匯出"])

with tab_overview:
    # 依角色的快速總攬
    if role == "建設公司":
        st.markdown("#### 🤝 合作對象")
        # 顯示時把「次數」改名「合作次數」
        if up_tbl is not None and not up_tbl.empty:
            st.write("・營造公司")
            st.dataframe(up_tbl.rename(columns={"次數":"合作次數"}), use_container_width=True)
        if down_mep is not None and not down_mep.empty:
            st.write("・水電公司")
            st.dataframe(down_mep.rename(columns={"次數":"合作次數"}), use_container_width=True)
        if down_dealer_tbl is not None and not down_dealer_tbl.empty:
            st.write("・終端經銷商")
            st.dataframe(down_dealer_tbl, use_container_width=True)

    elif role == "經銷商":
        st.markdown("#### 🤝 合作水電")
        st.dataframe(down_mep if down_mep is not None and not down_mep.empty else pd.DataFrame(),
                     use_container_width=True)

    elif role == "水電公司":
        st.markdown("#### 🤝 合作對象")
        # 經銷商（配比＋額度）＋ 年用量備註
        if down_dealer_tbl is not None and not down_dealer_tbl.empty:
            st.write("・經銷商（配比與額度）")
            st.dataframe(down_dealer_tbl.rename(columns={"額度_萬":"額度(萬)"}),
                         use_container_width=True)
        mep_vol = df_sel["年使用量_萬"].dropna().unique()
        memo = f"{mep_vol[0]} 萬" if len(mep_vol)>0 else "—"
        st.info(f"📌 預估年使用量：{memo}（已用於圖表的金額換算）")

    else:  # 營造公司維持原本呈現
        st.markdown("#### 📌 快速總攬")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**上游**")
            st.dataframe(up_tbl if up_tbl is not None and not up_tbl.empty else pd.DataFrame(),
                         use_container_width=True)
        with c2:
            st.markdown("**直接合作對象**")
            st.write("・水電公司")
            st.dataframe(down_mep if down_mep is not None and not down_mep.empty else pd.DataFrame(),
                         use_container_width=True)
            st.write("・終端經銷商（平均配比｜按水電等權）")
            st.dataframe(down_dealer_tbl if down_dealer_tbl is not None and not down_dealer_tbl.empty else pd.DataFrame(),
                         use_container_width=True)

with tab_partners:
    st.markdown("#### 📈 視覺化")
    # 建設/營造：水電出現次數；建設/營造：經銷商平均配比；水電：經銷商金額；經銷商：水電出現次數
    if role in ["建設公司","營造公司"] and down_mep is not None and not down_mep.empty:
        draw_chart(down_mep, down_mep.columns[0], "次數", f"{role} → 水電公司 合作次數")
    if role == "水電公司" and down_dealer_raw is not None and not down_dealer_raw.empty:
        # 改用金額
        draw_chart(down_dealer_raw.rename(columns={"額度_萬":"金額(萬)"}), "經銷商", "金額(萬)", "水電公司 → 終端經銷商 金額(萬)")
    if role == "營造公司" and down_dealer_raw is not None and not down_dealer_raw.empty:
        draw_chart(down_dealer_raw, "經銷商", "平均配比", "營造公司 → 終端經銷商 平均配比（按水電等權）")
    if role == "建設公司" and down_dealer_raw is not None and not down_dealer_raw.empty:
        draw_chart(down_dealer_raw, "經銷商", "平均配比", "建設公司 → 終端經銷商 平均配比（按水電等權）")
    if role == "經銷商" and down_mep is not None and not down_mep.empty:
        draw_chart(down_mep, "水電公司", "次數", "經銷商 → 水電公司 合作次數")

with tab_comp:
    st.markdown("#### ⚔️ 競爭者")
    # 建設公司：不提供競爭者分析
    if role == "建設公司":
        st.info("此角色不提供競爭者分析。")
    else:
        # === 競爭者函式 ===
        def competitor_table_water(df_base, target_mep):
            g = df_base[df_base["水電公司"].notna()]
            cons = g[g["水電公司"] == target_mep]["營造公司"].dropna().unique().tolist()
            if not cons:
                return pd.DataFrame(columns=["競爭對手","共同出現次數"])
            cand = g[g["營造公司"].isin(cons)]
            co = cand[cand["水電公司"] != target_mep].groupby("水電公司").size().reset_index(name="共同出現次數")
            return co.sort_values("共同出現次數", ascending=False)

        def competitor_table_dealer(rel_base, target_dealer):
            target_clients = rel_base[rel_base["經銷商"] == target_dealer]["水電公司"].dropna().unique().tolist()
            target_client_set = set(target_clients)
            target_total_clients = len(target_client_set)
            tgt_ratio_map = (rel_base[rel_base["經銷商"] == target_dealer]
                             .groupby("水電公司")["配比"].mean().to_dict())
            target_total_market = 0.0
            for mep in target_client_set:
                vol = float(mep_vol_map.get(mep, 0.0) or 0.0)
                r_t = float(tgt_ratio_map.get(mep, 0.0) or 0.0)
                target_total_market += vol * r_t
            stats = {}
            for mep, grp in rel_base.groupby("水電公司"):
                if mep not in target_client_set:
                    continue
                vol = float(mep_vol_map.get(mep, 0.0) or 0.0)
                ratios = grp.groupby("經銷商")["配比"].mean().to_dict()
                if target_dealer not in ratios:
                    continue
                r_t = float(ratios[target_dealer] or 0.0)
                for dealer, r_c in ratios.items():
                    if dealer == target_dealer or pd.isna(dealer):
                        continue
                    d = stats.setdefault(dealer, {"共同客戶數":0,"overlap_ratio_sum":0.0,"共同市場額度":0.0,"重疊市場額度":0.0})
                    d["共同客戶數"] += 1
                    r_min = min(float(r_c or 0.0), r_t)
                    d["overlap_ratio_sum"] += r_min
                    d["共同市場額度"] += vol
                    d["重疊市場額度"] += vol * r_min
            rows = []
            for dealer, d in stats.items():
                shared = d["共同客戶數"]
                if shared <= 0:
                    continue
                comp_index = d["overlap_ratio_sum"] / shared
                shared_pct = (shared / target_total_clients) if target_total_clients > 0 else 0.0
                overlap_market_share = (d["重疊市場額度"] / target_total_market) if target_total_market > 0 else 0.0
                threat = "高" if overlap_market_share > 0.30 else ("中" if overlap_market_share >= 0.15 else "低")
                rows.append({
                    "競爭對手": dealer,
                    "共同客戶數": shared,
                    "共同客戶數占比": pct_str(shared_pct),
                    "競爭指數": pct_str(comp_index),
                    "共同市場額度(萬)": round(d["共同市場額度"], 2),
                    "重疊市場額度(萬)": round(d["重疊市場額度"], 2),
                    "重疊市場占比": pct_str(overlap_market_share),
                    "威脅程度": threat,
                })
            out = pd.DataFrame(rows)
            if out.empty:
                return out
            cat = pd.Categorical(out["威脅程度"], categories=["高","中","低"], ordered=True)
            out = out.assign(_order=cat).sort_values(["_order","重疊市場占比","共同客戶數"], ascending=[True, False, False]).drop(columns="_order")
            return out

        def union_overlap_share(rel_base, target_dealer):
            """去重後的整體被競爭覆蓋率（≤100%）"""
            target_clients = rel_base[rel_base["經銷商"] == target_dealer]["水電公司"].dropna().unique().tolist()
            tgt_ratio_map = (rel_base[rel_base["經銷商"] == target_dealer]
                             .groupby("水電公司")["配比"].mean().to_dict())
            total_target = 0.0
            union_overlap = 0.0
            for mep in target_clients:
                vol = float(mep_vol_map.get(mep, 0.0) or 0.0)
                r_t = float(tgt_ratio_map.get(mep, 0.0) or 0.0)
                comp_sum = float((rel_base[rel_base["水電公司"]==mep]
                                  .groupby("經銷商")["配比"].mean()
                                  .drop(labels=[target_dealer], errors="ignore")
                                  .sum()) or 0.0)
                union_overlap += vol * min(r_t, comp_sum)
                total_target  += vol * r_t
            return (union_overlap / total_target) if total_target > 0 else 0.0

        # === 顯示 ===
        if role == "水電公司":
            st.dataframe(competitor_table_water(df, target), use_container_width=True)
        elif role == "經銷商":
            comp_tbl = competitor_table_dealer(rel, target)
            st.dataframe(comp_tbl, use_container_width=True)
            # 去重後的整體覆蓋率
            union_share = union_overlap_share(rel, target)
            st.caption(f"📌 去重後的『整體被競爭覆蓋率』：{pct_str(union_share)}")
            st.info("註：表格中的『重疊市場占比』為與單一對手的配對式重疊，對手多家相加可能 > 100%；上述去重指標則不會超過 100%。")
        elif role == "營造公司":
            devs = df[df["營造公司"] == target]["建設公司"].dropna().unique().tolist()
            cand = df[df["建設公司"].isin(devs)]
            co = (cand[cand["營造公司"] != target].groupby("營造公司")
                  .size().reset_index(name="共同出現次數").sort_values("共同出現次數", ascending=False))
            st.dataframe(co, use_container_width=True)

with tab_export:
    st.markdown("#### ⬇️ 匯出關係明細（經銷配比展開，不含年用量加權）")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="原始資料")
        df.to_excel(writer, index=False, sheet_name="主檔(固定欄位命名)")
        rel.to_excel(writer, index=False, sheet_name="關係明細(配比)")
    st.download_button(
        "下載 Excel",
        data=output.getvalue(),
        file_name="relations_search_dashboard_v10.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 清除快取
with st.expander("🧹 清除快取"):
    if st.button("清除 @st.cache_data 並重載"):
        st.cache_data.clear()
        st.rerun()
