import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="百大建商｜固定欄位版 關係鏈儀表板", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜固定欄位版 關係鏈儀表板")
st.caption("固定辨識：D=建設、E=營造、F=水電、G=年使用量(萬元)、H/J/L=經銷商、I/K/M=配比 → 自動拆解與分析")

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

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"

# ====================== Upload ======================
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔（固定欄位版；不需要任何操作）",
    type=["xlsx", "xls", "csv"],
    help="最多 200 MB；Excel 需使用 openpyxl 解析",
)

if not file:
    st.info("請先上傳檔案。")
    st.stop()

df_raw = read_any(file)

# 以「欄位位置」為主（0 起算）：D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11, M=12
# 若使用者表頭名稱恰好符合，也同時提供名稱對應的回退判斷
def get_col_by_pos_or_name(df, pos, name_candidates):
    cols = df.columns.tolist()
    try:
        col_by_pos = df.columns[pos]
    except Exception:
        col_by_pos = None
    # 優先用位置
    if col_by_pos is not None:
        return col_by_pos
    # 回退用名稱
    for n in name_candidates:
        if n in cols:
            return n
    return None

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

required = [col_dev, col_con, col_mep, col_vol]
if any(c is None for c in required):
    st.error("找不到必要欄位（依欄位位置 D/E/F/G 取得失敗）。請確認資料的欄序是否正確。")
    st.stop()

with st.expander("🔎 欄位對應（固定版；僅供查看）", expanded=True):
    st.write(pd.DataFrame({
        "角色":["建設公司(D)","營造公司(E)","水電公司(F)","年使用量(萬)(G)","經銷商(H)","配比(I)","經銷商(J)","配比(K)","經銷商(L)","配比(M)"],
        "欄位":[col_dev,col_con,col_mep,col_vol,col_dA,col_rA,col_dB,col_rB,col_dC,col_rC]
    }))

# ====================== Transform ======================
df = df_raw.rename(columns={
    col_dev:"建設公司", col_con:"營造公司", col_mep:"水電公司", col_vol:"年使用量_萬",
    col_dA:"經銷商A", col_rA:"經銷A比",
    col_dB:"經銷商B", col_rB:"經銷B比",
    col_dC:"經銷商C", col_rC:"經銷C比",
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

# 基礎關係鏈數量
count_dev = df["建設公司"].nunique()
count_con = df["營造公司"].nunique()
count_mep = df["水電公司"].nunique()
count_dea = rel["經銷商"].nunique()

pairs_dev_con = df[["建設公司","營造公司"]].dropna().drop_duplicates().shape[0]
pairs_con_mep = df[["營造公司","水電公司"]].dropna().drop_duplicates().shape[0]
pairs_mep_dea = rel[["水電公司","經銷商"]].dropna().drop_duplicates().shape[0]

# 配比檢查與依賴度
ratio_check = rel.groupby(["建設公司","營造公司","水電公司"], dropna=False)["配比"].sum().reset_index()
ratio_check["配比合計"] = ratio_check["配比"]
ratio_check["是否=1(±0.01)"] = np.isclose(ratio_check["配比合計"], 1.0, atol=0.01)

single_dep = (rel.groupby(["水電公司","經銷商"], dropna=False)["配比"].sum().reset_index())
top_ratio = single_dep.sort_values(["水電公司","配比"], ascending=[True, False]).groupby("水電公司").head(1)
top_ratio["單一依賴>80%"] = top_ratio["配比"] >= 0.8

# ====================== Tabs ======================
tab_raw, tab_dash = st.tabs(["📄 原始資料", "📊 分析儀表板"])

with tab_raw:
    st.subheader("原始資料預覽")
    st.dataframe(df_raw, use_container_width=True)

with tab_dash:
    st.subheader("總覽 KPI")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("建設公司數", f"{count_dev}")
    c2.metric("營造公司數", f"{count_con}")
    c3.metric("水電公司數", f"{count_mep}")
    c4.metric("經銷商數", f"{count_dea}")
    c5.metric("建設→營造 關係數", f"{pairs_dev_con}")
    c6.metric("營造→水電 關係數", f"{pairs_con_mep}")
    c7.metric("水電→經銷 關係數", f"{pairs_mep_dea}")

    st.markdown("---")
    st.subheader("關係明細（經銷商配比展開）")
    st.dataframe(rel, use_container_width=True)

    st.markdown("---")
    a1, a2 = st.columns([2,1])
    with a1:
        st.subheader("TOP 經銷商承接量 (前20)")
        dea_rank = (rel.groupby("經銷商", dropna=False)["承接量_萬"]
                    .sum().reset_index().sort_values("承接量_萬", ascending=False).head(20))
        fig = px.bar(dea_rank, x="經銷商", y="承接量_萬", title="經銷商承接量(萬元)")
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        st.subheader("經銷商承接占比")
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
    st.subheader("經銷商競合熱度（同水電的配比分散）")
    comp = (rel.groupby(["水電公司", "經銷商"], dropna=False)["配比"]
            .sum().reset_index())
    pivot = comp.pivot(index="水電公司", columns="經銷商", values="配比").fillna(0.0)
    st.dataframe(pivot, use_container_width=True)
    st.caption("每一列為單一水電公司，各欄為經銷商配比（加總≈1）。分散越平均，競爭越激烈。")

    st.markdown("---")
    st.subheader("風險雷達")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**配比未齊（合計≠1）**")
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
    devs = df["建設公司"].dropna().unique().tolist()
    cons = df["營造公司"].dropna().unique().tolist()
    meps = df["水電公司"].dropna().unique().tolist()
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
    link1 = (df.groupby(["建設公司","營造公司"], dropna=False)["年使用量_萬"].sum().reset_index())
    for _, r in link1.iterrows():
        s1.append(node_index[f"建設｜{r['建設公司']}"])
        t1_.append(node_index[f"營造｜{r['營造公司']}"])
        v1.append(max(r["年使用量_萬"], 0))

    # 營造->水電
    s2, t2_, v2 = [], [], []
    link2 = (df.groupby(["營造公司","水電公司"], dropna=False)["年使用量_萬"].sum().reset_index())
    for _, r in link2.iterrows():
        s2.append(node_index[f"營造｜{r['營造公司']}"])
        t2_.append(node_index[f"水電｜{r['水電公司']}"])
        v2.append(max(r["年使用量_萬"], 0))

    # 水電->經銷（用承接量_萬）
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
        fig.update_layout(title_text="建設→營造→水電→經銷 關係流（以用量/承接量為權重）", font_size=12)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("匯出")
    csv_bytes = rel.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載 關係明細 CSV", data=csv_bytes, file_name="relations_detail_fixed.csv", mime="text/csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="原始資料")
        df.to_excel(writer, index=False, sheet_name="主檔(固定欄位命名)")
        rel.to_excel(writer, index=False, sheet_name="關係明細")
        ratio_check.to_excel(writer, index=False, sheet_name="配比檢查")
        top_ratio.to_excel(writer, index=False, sheet_name="單一依賴檢查")
    st.download_button(
        "下載 Excel（多工作表）",
        data=output.getvalue(),
        file_name="relations_dashboard_fixed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
