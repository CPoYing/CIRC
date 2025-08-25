# app.py － 百大建商｜關係鏈分析（單頁搜尋 v4：經銷競爭升級）
# 更新點（僅經銷商的「競爭者」區塊）：
# - 隱藏「同場次數」
# - 新增：共同客戶數、共同客戶數占比、共同市場額度、重疊市場額度、重疊市場占比、威脅程度
# - 「平均重疊配比」→ 改名為「競爭指數」
# - 百分比一律四捨五入到兩位小數

import io
import re
import math
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="百大建商｜關係鏈分析（單頁搜尋 v4）", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜關係鏈分析（單頁搜尋 v4）")
st.caption("固定欄位：D=建設、E=營造、F=水電、G=年使用量(萬元，僅於水電視圖顯示備註)、H/J/L=經銷商、I/K/M=配比。支援任意公司搜尋/選擇，顯示合作對象與競爭者。")

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
    """輸入可為 0~1 或 0~100，輸出 'xx.xx%'（四捨五入兩位）"""
    if pd.isna(x):
        return "-"
    v = float(x)
    if v <= 1.0:
        v = v * 100.0
    d = Decimal(str(v)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
    return f"{d}%"

def p2f(x):
    """保留數值（不轉文字），若輸入 <=1 視為比例，>1 視為百分比（轉回 0~1）"""
    if pd.isna(x):
        return np.nan
    v = float(x)
    return v if v <= 1.0 else v / 100.0

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

# ====================== Upload ======================
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔（固定欄位；不需要側欄操作）",
    type=["xlsx", "xls", "csv"],
    help="Excel 需使用 openpyxl 解析",
)

if not file:
    st.info("請先上傳檔案。")
    st.stop()

df_raw = read_any(file)

# 固定欄位位置（0-based）：D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11, M=12
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
    st.error("找不到必要欄位（依欄位位置 D/E/F 取得失敗）。請確認資料的欄序是否正確。")
    st.stop()

# ====================== Transform（不以G加權，一律以出現次數/配比為主） ======================
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

# 經銷展開（僅用配比，不乘以G）
dealer_blocks = []
if "經銷商A" in df.columns and "經銷A比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商A","經銷A比"]].rename(columns={"經銷商A":"經銷商","經銷A比":"配比"}))
if "經銷商B" in df.columns and "經銷B比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商B","經銷B比"]].rename(columns={"經銷商B":"經銷商","經銷B比":"配比"}))
if "經銷商C" in df.columns and "經銷C比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","經銷商C","經銷C比"]].rename(columns={"經銷商C":"經銷商","經銷C比":"配比"}))

rel = pd.concat(dealer_blocks, ignore_index=True) if dealer_blocks else pd.DataFrame(columns=["建設公司","營造公司","水電公司","經銷商","配比"])
rel["經銷商"] = rel["經銷商"].apply(clean_name)
rel = rel.dropna(subset=["經銷商","水電公司"]).copy()

# 水電年用量（每家水電僅一個預估值；若重複取首個非空）
mep_vol_map = df.groupby("水電公司")["年使用量_萬"].apply(lambda s: s.dropna().iloc[0] if s.dropna().size>0 else np.nan).to_dict()

# ====================== 搜尋 / 選擇 ======================
role = st.radio("選擇角色", ["建設公司", "營造公司", "水電公司", "經銷商"], horizontal=True)
chart_type = st.radio("圖表類型", ["長條圖", "圓餅圖"], horizontal=True)

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
kw = st.text_input("搜尋關鍵字（支援模糊比對）", "")
filtered_opts = [o for o in all_opts if isinstance(o, str) and (kw.lower() in o.lower() if kw else True)]
target = st.selectbox("選擇公司", filtered_opts)
if not target:
    st.stop()

st.markdown("---")
st.subheader(f"🎯 目前選擇：{role}｜{target}")

# ====================== 合作對象（上下游） ======================
def share_table(df_in, group_cols, name_col):
    cnt = df_in.groupby(group_cols).size().reset_index(name="次數")
    tot = cnt["次數"].sum()
    if tot == 0:
        return pd.DataFrame(columns=[name_col,"次數","占比"])
    cnt["占比"] = cnt["次數"] / tot
    cnt["占比"] = cnt["占比"].apply(pct_str)
    return cnt.sort_values("次數", ascending=False)

down_dealer_raw = None
down_dealer = None
down_mep = None
up = None

if role == "建設公司":
    df_sel = df[df["建設公司"] == target]
    up = None
    mid = share_table(df_sel, ["營造公司"], "營造公司")
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")
    rel_sel = rel[rel["建設公司"] == target]
    down_dealer_raw = rel_sel.groupby("經銷商")["配比"].mean().reset_index().rename(columns={"配比":"平均配比"}).sort_values("平均配比", ascending=False)
    down_dealer = down_dealer_raw.copy()
    if not down_dealer.empty:
        down_dealer["平均配比"] = down_dealer["平均配比"].apply(pct_str)

elif role == "營造公司":
    df_sel = df[df["營造公司"] == target]
    up = share_table(df_sel, ["建設公司"], "建設公司")
    mid = None
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")
    rel_sel = rel[rel["營造公司"] == target]
    down_dealer_raw = rel_sel.groupby("經銷商")["配比"].mean().reset_index().rename(columns={"配比":"平均配比"}).sort_values("平均配比", ascending=False)
    down_dealer = down_dealer_raw.copy()
    if not down_dealer.empty:
        down_dealer["平均配比"] = down_dealer["平均配比"].apply(pct_str)

elif role == "水電公司":
    df_sel = df[df["水電公司"] == target]
    up = share_table(df_sel, ["建設公司","營造公司"], "公司")
    mid = None
    down_mep = None
    rel_sel = rel[rel["水電公司"] == target]
    down_dealer_raw = rel_sel.groupby("經銷商")["配比"].mean().reset_index().sort_values("配比", ascending=False).rename(columns={"配比":"配比"})
    down_dealer = down_dealer_raw.copy()
    if not down_dealer.empty:
        down_dealer["配比"] = down_dealer["配比"].apply(pct_str)

    # 水電年用量備註
    mep_vol = df_sel["年使用量_萬"].dropna().unique()
    memo = f"{mep_vol[0]} 萬" if len(mep_vol)>0 else "—"
    st.info(f"📌 預估年使用量（僅備註，不參與上表計算）：{memo}")

elif role == "經銷商":
    df_sel = rel[rel["經銷商"] == target].merge(df, on=["建設公司","營造公司","水電公司"], how="left", suffixes=("","_df"))
    up = share_table(df_sel, ["建設公司","營造公司"], "公司")
    mid = None
    down_mep = share_table(df_sel, ["水電公司"], "水電公司")

# ====================== 競爭者 ======================
st.markdown("---")
st.subheader("⚔️ 競爭者")

def competitor_table_water(df_base, target_mep):
    g = df_base[df_base["水電公司"].notna()]
    cons = g[g["水電公司"] == target_mep]["營造公司"].dropna().unique().tolist()
    if not cons:
        return pd.DataFrame(columns=["競爭對手","共同出現次數"])
    cand = g[g["營造公司"].isin(cons)]
    co = cand[cand["水電公司"] != target_mep].groupby("水電公司").size().reset_index(name="共同出現次數")
    return co.sort_values("共同出現次數", ascending=False)

def competitor_table_dealer(rel_base, df_base, target_dealer):
    """經銷商競爭者：依共同客戶/重疊市場計算，回傳彙整表"""
    # 目標經銷商的客戶（唯一水電）
    target_clients = rel_base[rel_base["經銷商"] == target_dealer]["水電公司"].dropna().unique().tolist()
    target_client_set = set(target_clients)
    target_total_clients = len(target_client_set)

    # 目標經銷商的「總市場額度」= ∑(該水電年用量 × 目標配比)
    # 先計算每個水電上的目標配比（平均）
    tgt_ratio_map = (rel_base[rel_base["經銷商"] == target_dealer]
                     .groupby("水電公司")["配比"].mean().to_dict())
    target_total_market = 0.0
    for mep in target_client_set:
        vol = float(mep_vol_map.get(mep, 0.0) or 0.0)
        r_t = float(tgt_ratio_map.get(mep, 0.0) or 0.0)
        target_total_market += vol * r_t

    # 逐一在共同客戶上，計算各競爭者的統計
    stats = {}  # dealer -> dict
    for mep, grp in rel_base.groupby("水電公司"):
        if mep not in target_client_set:
            continue
        vol = float(mep_vol_map.get(mep, 0.0) or 0.0)
        # 這個水電上，目標與其他經銷的配比
        ratios = grp.groupby("經銷商")["配比"].mean().to_dict()
        if target_dealer not in ratios:
            continue
        r_t = float(ratios[target_dealer] or 0.0)
        for dealer, r_c in ratios.items():
            if dealer == target_dealer or pd.isna(dealer):
                continue
            d = stats.setdefault(dealer, {"共同客戶數":0, "overlap_ratio_sum":0.0,
                                          "共同市場額度":0.0, "重疊市場額度":0.0})
            d["共同客戶數"] += 1
            # 競爭重疊配比（min）
            r_min = min(float(r_c or 0.0), r_t)
            d["overlap_ratio_sum"] += r_min
            # 市場額度
            d["共同市場額度"] += vol
            d["重疊市場額度"] += vol * r_min

    # 彙整成表格
    rows = []
    for dealer, d in stats.items():
        shared = d["共同客戶數"]
        if shared <= 0:
            continue
        comp_index = d["overlap_ratio_sum"] / shared  # 平均 min 配比
        # 共同客戶占比
        shared_pct = (shared / target_total_clients) if target_total_clients > 0 else 0.0
        # 重疊市場占比：以目標經銷商的總市場額度為母數
        overlap_market_share = (d["重疊市場額度"] / target_total_market) if target_total_market > 0 else 0.0
        # 威脅程度
        if overlap_market_share > 0.30:
            threat = "高"
        elif overlap_market_share >= 0.15:
            threat = "中"
        else:
            threat = "低"
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

    out = pd.DataFrame(rows).sort_values(
        ["威脅程度","重疊市場占比","共同客戶數"],
        ascending=[True, False, False]  # 依威脅→占比→客戶數排序
    )
    # 自訂威脅程度排序（高>中>低）
    if not out.empty:
        cat = pd.Categorical(out["威脅程度"], categories=["高","中","低"], ordered=True)
        out = out.assign(_order=cat).sort_values(["_order","重疊市場占比"], ascending=[True, False]).drop(columns="_order")
    return out

# 顯示競爭者表
if role == "水電公司":
    comp_tbl = competitor_table_water(df, target)
    st.dataframe(comp_tbl, use_container_width=True)
elif role == "經銷商":
    comp_tbl = competitor_table_dealer(rel, df, target)
    st.dataframe(comp_tbl, use_container_width=True)
elif role == "建設公司":
    cons = df[df["建設公司"] == target]["營造公司"].dropna().unique().tolist()
    cand = df[df["營造公司"].isin(cons)]
    co = cand[cand["建設公司"] != target].groupby("建設公司").size().reset_index(name="共同出現次數").sort_values("共同出現次數", ascending=False)
    st.dataframe(co, use_container_width=True)
elif role == "營造公司":
    devs = df[df["營造公司"] == target]["建設公司"].dropna().unique().tolist()
    cand = df[df["建設公司"].isin(devs)]
    co = cand[cand["營造公司"] != target].groupby("營造公司").size().reset_index(name="共同出現次數").sort_values("共同出現次數", ascending=False)
    st.dataframe(co, use_container_width=True)

# ====================== 視覺（全域切換：長條圖/圓餅圖） ======================
st.markdown("---")
st.subheader("📈 精簡視覺")

def draw_chart(df_plot, name_col, value_col, title):
    if df_plot is None or df_plot.empty:
        st.info("沒有資料可視覺化。")
        return
    if chart_type == "長條圖":
        fig = px.bar(df_plot.head(15), x=name_col, y=value_col, title=title)
    else:
        fig = px.pie(df_plot, names=name_col, values=value_col, title=title)
    st.plotly_chart(fig, use_container_width=True)

if role in ["建設公司","營造公司"] and down_mep is not None and not down_mep.empty:
    draw_chart(down_mep, down_mep.columns[0], "次數", f"{role} → 水電公司 出現次數")

if role == "水電公司" and down_dealer is not None and not down_dealer.empty:
    # 表格顯示為百分比；圖表以數值欄位（0~1）繪製
    draw_chart(down_dealer_raw, "經銷商", "配比" if "配比" in down_dealer_raw.columns else "平均配比", f"{role} → 經銷商 配比")

if role == "營造公司" and down_dealer_raw is not None and not down_dealer_raw.empty:
    draw_chart(down_dealer_raw, "經銷商", "平均配比", "營造公司 → 經銷商 平均配比")

if role == "建設公司" and down_dealer_raw is not None and not down_dealer_raw.empty:
    draw_chart(down_dealer_raw, "經銷商", "平均配比", "建設公司 → 經銷商 平均配比")

if role == "經銷商" and down_mep is not None and not down_mep.empty:
    draw_chart(down_mep, "水電公司", "次數", "經銷商 → 水電公司 出現次數")

# ====================== 匯出 ======================
st.markdown("---")
st.subheader("⬇️ 匯出關係明細（經銷配比展開，不含年用量加權）")
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_raw.to_excel(writer, index=False, sheet_name="原始資料")
    df.to_excel(writer, index=False, sheet_name="主檔(固定欄位命名)")
    rel.to_excel(writer, index=False, sheet_name="關係明細(配比)")
st.download_button(
    "下載 Excel",
    data=output.getvalue(),
    file_name="relations_search_dashboard_v4.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
