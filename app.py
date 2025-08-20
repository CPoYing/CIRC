
import io
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="百大建商｜關係鏈分析儀表板", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商關係鏈分析儀表板")
st.caption("上傳 Excel/CSV -> 欄位對應/清理 -> 關係拆解與配比 -> 儀表板（總覽/貢獻/競爭/關係圖）-> 匯出")

# =====================================================
# 0) Helpers
# =====================================================
@st.cache_data
def read_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

def _coerce_num(s):
    \"\"\"Coerce to float, handling commas/percents/strings.\"\"\"
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return s
    s = str(s).replace(",", "").replace("%", "").strip()
    if s in ("", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_ratio(series):
    \"\"\"Normalize ratio to 0~1; if it looks like 0~100, divide by 100.\"\"\"
    s = series.apply(_coerce_num)
    if s.max(skipna=True) is not None and s.max(skipna=True) > 1.000001:
        return s / 100.0
    return s

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"

# =====================================================
# 1) Upload
# =====================================================
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔",
    type=["xlsx", "xls", "csv"],
    help="最多 200 MB；Excel 需使用 openpyxl 解析",
)

sample_cols = [
    "建商", "營造公司", "水電全名", "年使用量/萬",
    "經銷商A", "經銷A佔比(%)",
    "經銷商B", "經銷B佔比(%)",
    "經銷商C", "經銷Ｃ佔比(%)",
]
with st.expander("期望欄位結構（對照你的資料）", expanded=False):
    st.write(pd.DataFrame({
        "欄位": sample_cols,
        "說明": ["建設公司", "營造公司", "水電公司全名", "該水電估計年用量(萬元)",
               "配合經銷商1", "經銷商1配比", "配合經銷商2", "經銷商2配比", "配合經銷商3", "經銷商3配比"]
    }))

if not file:
    st.info("請先上傳檔案。")
    st.stop()

df_raw = read_any(file)
st.subheader("原始資料預覽")
st.dataframe(df_raw.head(20), use_container_width=True)

# =====================================================
# 2) Column mapping
# =====================================================
st.sidebar.header("⚙️ 操作區")
st.sidebar.subheader("欄位對應")
def select_map(title, default_candidates):
    options = ["（未對應）"] + df_raw.columns.tolist()
    default = next((c for c in default_candidates if c in df_raw.columns), "（未對應）")
    return st.sidebar.selectbox(title, options, index=options.index(default) if default in options else 0)

col_dev = select_map("建設公司欄位", ["建商", "建設公司", "建設公司(業主)"])
col_con = select_map("營造公司欄位", ["營造公司", "營造商"])
col_mep = select_map("水電公司欄位", ["水電全名", "水電公司", "機電公司", "機電廠商"])
col_vol = select_map("年使用量(萬元)欄位", ["年使用量/萬", "年使用量(萬)", "用量_萬"])

col_dA = select_map("經銷商A欄位", ["經銷商A", "經銷A", "經銷商1"])
col_rA = select_map("經銷A配比欄位", ["經銷A佔比(%)", "經銷商A配比", "A配比"])
col_dB = select_map("經銷商B欄位", ["經銷商B", "經銷B", "經銷商2"])
col_rB = select_map("經銷B配比欄位", ["經銷B佔比(%)", "經銷商B配比", "B配比"])
col_dC = select_map("經銷商C欄位", ["經銷商C", "經銷Ｃ", "經銷商3"])
col_rC = select_map("經銷C配比欄位", ["經銷Ｃ佔比(%)", "經銷C佔比(%)", "經銷商C配比", "C配比"])

required = [col_dev, col_con, col_mep, col_vol]
if any(c == "（未對應）" for c in required):
    st.error("請至少對應『建設公司 / 營造公司 / 水電公司 / 年使用量(萬元)』四個欄位。")
    st.stop()

# =====================================================
# 3) Cleaning & explode dealers
# =====================================================
df = df_raw.rename(columns={
    col_dev: "建設公司", col_con: "營造公司", col_mep: "水電公司", col_vol: "年使用量_萬",
    col_dA: "經銷商A", col_rA: "經銷A比",
    col_dB: "經銷商B", col_rB: "經銷B比",
    col_dC: "經銷商C", col_rC: "經銷C比",
}).copy()

for c in ["年使用量_萬"]:
    df[c] = df[c].apply(_coerce_num)

for c in ["經銷A比", "經銷B比", "經銷C比"]:
    if c in df.columns:
        df[c] = normalize_ratio(df[c])

dealer_blocks = []
if "經銷商A" in df.columns and "經銷A比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商A","經銷A比"]]\
                         .rename(columns={"經銷商A":"經銷商","經銷A比":"配比"}))
if "經銷商B" in df.columns and "經銷B比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商B","經銷B比"]]\
                         .rename(columns={"經銷商B":"經銷商","經銷B比":"配比"}))
if "經銷商C" in df.columns and "經銷C比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商C","經銷C比"]]\
                         .rename(columns={"經銷商C":"經銷商","經銷C比":"配比"}))

rel = pd.concat(dealer_blocks, ignore_index=True) if dealer_blocks else pd.DataFrame(columns=["建設公司","營造公司","水電公司","年使用量_萬","經銷商","配比"])
rel["經銷商"] = rel["經銷商"].replace({0: np.nan, "0": np.nan, "": np.nan}).astype("string")
rel = rel.dropna(subset=["經銷商"]).copy()

rel["承接量_萬"] = rel["年使用量_萬"] * rel["配比"]
rel["承接量_元"] = rel["承接量_萬"] * 10000

# ratio check
ratio_check = rel.groupby(["建設公司","營造公司","水電公司"], dropna=False)["配比"].sum().reset_index()
ratio_check["配比合計"] = ratio_check["配比"]
ratio_check["是否=1(±0.01)"] = np.isclose(ratio_check["配比合計"], 1.0, atol=0.01)

with st.expander("配比檢查（同一水電公司的經銷商配比加總應 ≈ 1）", expanded=False):
    st.dataframe(ratio_check.sort_values("是否=1(±0.01)"), use_container_width=True)

auto_norm = st.sidebar.checkbox("自動正規化每個水電公司的配比（使合計=1）", value=True)
if auto_norm and not rel.empty:
    sums = rel.groupby(["建設公司","營造公司","水電公司"])["配比"].transform(lambda s: s.sum() if s.sum() else 1.0)
    rel["配比"] = rel["配比"] / sums
    rel["承接量_萬"] = rel["年使用量_萬"] * rel["配比"]
    rel["承接量_元"] = rel["承接量_萬"] * 10000

# =====================================================
# 4) Filters
# =====================================================
st.sidebar.subheader("篩選條件")
dev_sel = st.sidebar.multiselect("建設公司", sorted(rel["建設公司"].dropna().unique().tolist()))
con_sel = st.sidebar.multiselect("營造公司", sorted(rel["營造公司"].dropna().unique().tolist()))
mep_sel = st.sidebar.multiselect("水電公司", sorted(rel["水電公司"].dropna().unique().tolist()))
dea_sel = st.sidebar.multiselect("經銷商", sorted(rel["經銷商"].dropna().unique().tolist()))

filtered = rel.copy()
if dev_sel: filtered = filtered[filtered["建設公司"].isin(dev_sel)]
if con_sel: filtered = filtered[filtered["營造公司"].isin(con_sel)]
if mep_sel: filtered = filtered[filtered["水電公司"].isin(mep_sel)]
if dea_sel: filtered = filtered[filtered["經銷商"].isin(dea_sel)]

st.subheader("關係明細（過濾後）")
st.dataframe(filtered, use_container_width=True)

# KPI
tot_mep = filtered["水電公司"].nunique()
tot_dea = filtered["經銷商"].nunique()
sum_vol = filtered["承接量_萬"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("水電公司數量", f"{tot_mep}")
c2.metric("經銷商數量", f"{tot_dea}")
c3.metric("經銷商承接總量(萬元)", fmt_num(sum_vol, 0))

# =====================================================
# 5) Charts
# =====================================================
st.subheader("圖表視覺化")

with st.expander("TOP 經銷商承接量", expanded=True):
    topN = st.slider("顯示前 N 名", 5, 30, 10, 1)
    dea_rank = (filtered.groupby("經銷商", dropna=False)["承接量_萬"]
                .sum().reset_index().sort_values("承接量_萬", ascending=False).head(topN))
    fig = px.bar(dea_rank, x="經銷商", y="承接量_萬", title="經銷商承接量(萬元)")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("TOP 水電公司年使用量（按配比拆分加總）", expanded=False):
    mep_rank = (filtered.groupby("水電公司", dropna=False)["承接量_萬"]
                .sum().reset_index().sort_values("承接量_萬", ascending=False).head(15))
    fig = px.bar(mep_rank, x="水電公司", y="承接量_萬", title="水電公司加權後使用量(萬元)")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("建設公司 / 營造公司 貢獻度", expanded=False):
    t1, t2 = st.tabs(["建設公司貢獻", "營造公司貢獻"])
    with t1:
        dev_rank = (filtered.groupby("建設公司", dropna=False)["承接量_萬"]
                    .sum().reset_index().sort_values("承接量_萬", ascending=False))
        fig = px.bar(dev_rank, x="建設公司", y="承接量_萬", title="建設公司帶來的加權使用量(萬元)")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        con_rank = (filtered.groupby("營造公司", dropna=False)["承接量_萬"]
                    .sum().reset_index().sort_values("承接量_萬", ascending=False))
        fig = px.bar(con_rank, x="營造公司", y="承接量_萬", title="營造公司帶來的加權使用量(萬元)")
        st.plotly_chart(fig, use_container_width=True)

with st.expander("經銷商競合：同一水電公司的多家經銷配比", expanded=False):
    comp = (filtered.groupby(["水電公司", "經銷商"], dropna=False)["配比"]
            .sum().reset_index())
    pivot = comp.pivot(index="水電公司", columns="經銷商", values="配比").fillna(0.0)
    st.dataframe(pivot, use_container_width=True)
    st.caption("數值為配比(0~1)，可觀察同一水電公司如何在多家經銷商間分配。")

with st.expander("關係流向圖（Sankey）", expanded=False):
    devs = filtered["建設公司"].dropna().unique().tolist()
    cons = filtered["營造公司"].dropna().unique().tolist()
    meps = filtered["水電公司"].dropna().unique().tolist()
    deas = filtered["經銷商"].dropna().unique().tolist()

    nodes = (
        [f"建設｜{d}" for d in devs] +
        [f"營造｜{c}" for c in cons] +
        [f"水電｜{m}" for m in meps] +
        [f"經銷｜{d}" for d in deas]
    )
    node_index = {name: i for i, name in enumerate(nodes)}

    # 建設->營造
    s1, t1, v1 = [], [], []
    link1 = (filtered.groupby(["建設公司","營造公司"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link1.iterrows():
        s1.append(node_index[f"建設｜{r['建設公司']}"])
        t1.append(node_index[f"營造｜{r['營造公司']}"])
        v1.append(max(r["承接量_萬"], 0))

    # 營造->水電
    s2, t2, v2 = [], [], []
    link2 = (filtered.groupby(["營造公司","水電公司"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link2.iterrows():
        s2.append(node_index[f"營造｜{r['營造公司']}"])
        t2.append(node_index[f"水電｜{r['水電公司']}"])
        v2.append(max(r["承接量_萬"], 0))

    # 水電->經銷
    s3, t3, v3 = [], [], []
    link3 = (filtered.groupby(["水電公司","經銷商"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link3.iterrows():
        s3.append(node_index[f"水電｜{r['水電公司']}"])
        t3.append(node_index[f"經銷｜{r['經銷商']}"])
        v3.append(max(r["承接量_萬"], 0))

    source = s1 + s2 + s3
    target = t1 + t2 + t3
    value  = v1 + v2 + v3

    if len(source) == 0:
        st.info("目前篩選結果沒有可視的關係流。")
    else:
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=12, thickness=20, line=dict(width=0.5), label=nodes),
            link=dict(source=source, target=target, value=value)
        )])
        fig.update_layout(title_text="建設→營造→水電→經銷 關係流（承接量_萬）", font_size=12)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6) Export
# =====================================================
st.subheader("下載結果")
csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button("下載 關係明細 CSV", data=csv_bytes, file_name="relations_detail.csv", mime="text/csv")

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_raw.to_excel(writer, index=False, sheet_name="原始資料")
    rel.to_excel(writer, index=False, sheet_name="關係明細_未篩選")
    filtered.to_excel(writer, index=False, sheet_name="關係明細_已篩選")
    ratio_check.to_excel(writer, index=False, sheet_name="配比檢查")
st.download_button(
    "下載 Excel（多工作表）",
    data=output.getvalue(),
    file_name="relations_dashboard_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

