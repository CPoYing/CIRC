import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="百大建商｜關係鏈分析儀表板（自動版）", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜關係鏈分析儀表板（自動版）")
st.caption("上傳 Excel/CSV → 自動欄位辨識 → 關係拆解與配比 → 多視角儀表板")

# ====================== Helpers ======================
@st.cache_data
def read_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

def coerce_num(s):
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
    s = series.apply(coerce_num)
    if s.max(skipna=True) is not None and s.max(skipna=True) > 1.000001:
        return s / 100.0
    return s

def try_pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    # fuzzy contains
    for c in cols:
        if any(key in str(c) for key in candidates):
            return c
    return None

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"

# ====================== Upload ======================
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔（不提供側邊操作，系統將自動分析）",
    type=["xlsx", "xls", "csv"],
    help="最多 200 MB；Excel 需使用 openpyxl 解析",
)

if not file:
    st.info("請先上傳檔案。")
    st.stop()

df_raw = read_any(file)
cols = df_raw.columns.tolist()

# ====================== Auto column detection ======================
dev_col = try_pick(cols, ["建商", "建設公司", "建設公司(業主)"])
con_col = try_pick(cols, ["營造公司", "營造商"])
mep_col = try_pick(cols, ["水電全名", "水電公司", "機電公司", "機電廠商"])
vol_col = try_pick(cols, ["年使用量/萬", "年使用量(萬)", "用量_萬"])

dA_col = try_pick(cols, ["經銷商A", "經銷A", "經銷商1"])
rA_col = try_pick(cols, ["經銷A佔比(%)", "經銷商A配比", "A配比"])
dB_col = try_pick(cols, ["經銷商B", "經銷B", "經銷商2"])
rB_col = try_pick(cols, ["經銷B佔比(%)", "經銷商B配比", "B配比"])
dC_col = try_pick(cols, ["經銷商C", "經銷Ｃ", "經銷商3"])
rC_col = try_pick(cols, ["經銷Ｃ佔比(%)", "經銷C佔比(%)", "經銷商C配比", "C配比"])

with st.expander("🔎 欄位自動辨識結果（僅顯示，不可操作）", expanded=True):
    st.write(pd.DataFrame({
        "角色": ["建設公司","營造公司","水電公司","年使用量(萬)","經銷A","配比A","經銷B","配比B","經銷C","配比C"],
        "對應欄位": [dev_col, con_col, mep_col, vol_col, dA_col, rA_col, dB_col, rB_col, dC_col, rC_col]
    }))

required = [dev_col, con_col, mep_col, vol_col]
missing = [r for r, c in zip(["建設公司","營造公司","水電公司","年使用量(萬)"], required) if c is None]
if missing:
    st.error(f"缺少必要欄位：{', '.join(missing)}。請確認資料表欄名。")
    st.stop()

# ====================== Transform ======================
df = df_raw.rename(columns={
    dev_col: "建設公司", con_col: "營造公司", mep_col: "水電公司", vol_col: "年使用量_萬",
    dA_col or "經銷商A": "經銷商A", rA_col or "經銷A佔比(%)": "經銷A比",
    dB_col or "經銷商B": "經銷商B", rB_col or "經銷B佔比(%)": "經銷B比",
    dC_col or "經銷商C": "經銷商C", rC_col or "經銷Ｃ佔比(%)": "經銷C比",
}).copy()

df["年使用量_萬"] = df["年使用量_萬"].apply(coerce_num)
for c in ["經銷A比","經銷B比","經銷C比"]:
    if c in df.columns:
        df[c] = normalize_ratio(df[c])

dealer_blocks = []
if "經銷商A" in df.columns and "經銷A比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商A","經銷A比"]].rename(columns={"經銷商A":"經銷商","經銷A比":"配比"}))
if "經銷商B" in df.columns and "經銷B比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商B","經銷B比"]].rename(columns={"經銷商B":"經銷商","經銷B比":"配比"}))
if "經銷商C" in df.columns and "經銷C比" in df.columns:
    dealer_blocks.append(df[["建設公司","營造公司","水電公司","年使用量_萬","經銷商C","經銷C比"]].rename(columns={"經銷商C":"經銷商","經銷C比":"配比"}))

rel = pd.concat(dealer_blocks, ignore_index=True) if dealer_blocks else pd.DataFrame(columns=["建設公司","營造公司","水電公司","年使用量_萬","經銷商","配比"])
rel["經銷商"] = rel["經銷商"].replace({0: np.nan, "0": np.nan, "": np.nan}).astype("string")
rel = rel.dropna(subset=["經銷商"]).copy()
rel["承接量_萬"] = rel["年使用量_萬"] * rel["配比"]
rel["承接量_元"] = rel["承接量_萬"] * 10000

# 配比檢查
ratio_check = rel.groupby(["建設公司","營造公司","水電公司"], dropna=False)["配比"].sum().reset_index()
ratio_check["配比合計"] = ratio_check["配比"]
ratio_check["是否=1(±0.01)"] = np.isclose(ratio_check["配比合計"], 1.0, atol=0.01)

# 風險標籤
risk = ratio_check.copy()
risk["標籤"] = np.where(~risk["是否=1(±0.01)"], "配比未齊", "")
# 單一經銷商依賴度
single_dep = (rel.groupby(["水電公司","經銷商"], dropna=False)["配比"].sum().reset_index())
top_ratio = single_dep.sort_values(["水電公司","配比"], ascending=[True, False]).groupby("水電公司").head(1)
top_ratio["單一依賴>80%"] = top_ratio["配比"] >= 0.8

# ====================== Tabs ======================
tab_raw, tab_dash = st.tabs(["📄 原始資料", "📊 分析儀表板（自動）"])

with tab_raw:
    st.subheader("原始資料預覽")
    st.dataframe(df_raw, use_container_width=True)
    st.caption("此分頁僅顯示你上傳的原始內容（未拆分經銷商/未計算）。")

with tab_dash:
    # ===== KPIs =====
    st.subheader("總覽 KPI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("建設公司數", f"{rel['建設公司'].nunique()}")
    c2.metric("營造公司數", f"{rel['營造公司'].nunique()}")
    c3.metric("水電公司數", f"{rel['水電公司'].nunique()}")
    c4.metric("經銷商承接總量(萬元)", fmt_num(rel['承接量_萬'].sum(), 0))

    st.markdown("---")

    # ===== Analyses =====
    a1, a2 = st.columns([2,1])
    with a1:
        st.subheader("TOP 經銷商承接量 (前20)")
        dea_rank = (rel.groupby("經銷商", dropna=False)["承接量_萬"]
                    .sum().reset_index().sort_values("承接量_萬", ascending=False).head(20))
        fig = px.bar(dea_rank, x="經銷商", y="承接量_萬", title="經銷商承接量(萬元)")
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        st.subheader("經銷商市場占比")
        share = (rel.groupby("經銷商", dropna=False)["承接量_萬"]
                 .sum().reset_index().sort_values("承接量_萬", ascending=False))
        fig = px.pie(share, names="經銷商", values="承接量_萬", title="承接量占比")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("TOP 水電公司（加權使用量）")
        mep_rank = (rel.groupby("水電公司", dropna=False)["承接量_萬"]
                    .sum().reset_index().sort_values("承接量_萬", ascending=False).head(20))
        fig = px.bar(mep_rank, x="水電公司", y="承接量_萬", title="水電公司加權後使用量(萬元)")
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        st.subheader("建設/營造貢獻度")
        t1, t2 = st.tabs(["建設公司貢獻", "營造公司貢獻"])
        with t1:
            dev_rank = (rel.groupby("建設公司", dropna=False)["承接量_萬"]
                        .sum().reset_index().sort_values("承接量_萬", ascending=False))
            fig = px.bar(dev_rank, x="建設公司", y="承接量_萬", title="建設公司帶來的加權使用量(萬元)")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            con_rank = (rel.groupby("營造公司", dropna=False)["承接量_萬"]
                        .sum().reset_index().sort_values("承接量_萬", ascending=False))
            fig = px.bar(con_rank, x="營造公司", y="承接量_萬", title="營造公司帶來的加權使用量(萬元)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("經銷商競合熱度（同水電的配比分散度）")
    comp = (rel.groupby(["水電公司", "經銷商"], dropna=False)["配比"]
            .sum().reset_index())
    pivot = comp.pivot(index="水電公司", columns="經銷商", values="配比").fillna(0.0)
    st.dataframe(pivot, use_container_width=True)
    st.caption("數值為配比(0~1)，行內加總應 ≈ 1。行內越平均，代表競爭越激烈。")

    st.markdown("---")

    st.subheader("風險雷達")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**配比檢查（非 1 的水電公司）**")
        bad = ratio_check[~ratio_check["是否=1(±0.01)"]].copy()
        if bad.empty:
            st.success("所有水電公司配比加總皆 ≈ 1。")
        else:
            st.dataframe(bad.sort_values("配比合計"), use_container_width=True)
    with cB:
        st.markdown("**單一經銷商依賴度 > 80%**")
        risky = top_ratio[top_ratio["單一依賴>80%"]].copy()
        if risky.empty:
            st.success("無單一依賴度超過 80% 的水電公司。")
        else:
            st.dataframe(risky[["水電公司","經銷商","配比","單一依賴>80%"]], use_container_width=True)

    st.markdown("---")

    st.subheader("關係流向圖（建設→營造→水電→經銷）")
    devs = rel["建設公司"].dropna().unique().tolist()
    cons = rel["營造公司"].dropna().unique().tolist()
    meps = rel["水電公司"].dropna().unique().tolist()
    deas = rel["經銷商"].dropna().unique().tolist()

    nodes = (
        [f"建設｜{d}" for d in devs] +
        [f"營造｜{c}" for c in cons] +
        [f"水電｜{m}" for m in meps] +
        [f"經銷｜{d}" for d in deas]
    )
    node_index = {name: i for i, name in enumerate(nodes)}

    # 建設->營造
    s1, t1_, v1 = [], [], []
    link1 = (rel.groupby(["建設公司","營造公司"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link1.iterrows():
        s1.append(node_index[f"建設｜{r['建設公司']}"])
        t1_.append(node_index[f"營造｜{r['營造公司']}"])
        v1.append(max(r["承接量_萬"], 0))

    # 營造->水電
    s2, t2_, v2 = [], [], []
    link2 = (rel.groupby(["營造公司","水電公司"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link2.iterrows():
        s2.append(node_index[f"營造｜{r['營造公司']}"])
        t2_.append(node_index[f"水電｜{r['水電公司']}"])
        v2.append(max(r["承接量_萬"], 0))

    # 水電->經銷
    s3, t3_, v3 = [], [], []
    link3 = (rel.groupby(["水電公司","經銷商"], dropna=False)["承接量_萬"].sum().reset_index())
    for _, r in link3.iterrows():
        s3.append(node_index[f"水電｜{r['水電公司']}"])
        t3_.append(node_index[f"經銷｜{r['經銷商']}"])
        v3.append(max(r["承接量_萬"], 0))

    source = s1 + s2 + s3
    target = t1_ + t2_ + t3_
    value  = v1 + v2 + v3

    if len(source) == 0:
        st.info("資料不足，無法生成關係流。")
    else:
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=12, thickness=20, line=dict(width=0.5), label=nodes),
            link=dict(source=source, target=target, value=value)
        )])
        fig.update_layout(title_text="建設→營造→水電→經銷 關係流（承接量_萬）", font_size=12)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("匯出")
    csv_bytes = rel.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載 關係明細 CSV（未篩選）", data=csv_bytes, file_name="relations_detail.csv", mime="text/csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="原始資料")
        rel.to_excel(writer, index=False, sheet_name="關係明細")
        ratio_check.to_excel(writer, index=False, sheet_name="配比檢查")
        top_ratio.to_excel(writer, index=False, sheet_name="單一依賴檢查")
    st.download_button(
        "下載 Excel（多工作表）",
        data=output.getvalue(),
        file_name="relations_dashboard_auto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


