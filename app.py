
import io
import re
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="百大建商｜完整關係鏈分析儀表板", page_icon="🏗️", layout="wide")
st.title("🏗️ 百大建商｜完整關係鏈分析儀表板（固定欄位，無操作區）")
st.caption("D=建設、E=營造、F=水電、G=年使用量(萬元)、H/J/L=經銷商、I/K/M=配比 → 自動拆解，輸出多面向分析")

# ====================== Helpers ======================
@st.cache_data
def read_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

def clean_name(x):
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\\u3000", " ").strip()
    s = re.sub(r"\\s+", " ", s)
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

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"

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
    "上傳 Excel 或 CSV 檔（固定欄位版；不需要任何操作）",
    type=["xlsx", "xls", "csv"],
    help="最多 200 MB；Excel 需使用 openpyxl 解析",
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

# 清理名稱
for c in ["建設公司","營造公司","水電公司","經銷商A","經銷商B","經銷商C"]:
    if c in df.columns:
        df[c] = df[c].apply(clean_name)

df["年使用量_萬"] = df["年使用量_萬"].apply(coerce_num)
for c in ["經銷A比","經銷B比","經銷C比"]:
    if c in df.columns:
        df[c] = normalize_ratio(df[c])

# 經銷商展開
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
# 清理經銷商名稱
rel["經銷商"] = rel["經銷商"].apply(clean_name)

# 計算承接量
rel["承接量_萬"] = rel["年使用量_萬"] * rel["配比"]
rel["承接量_元"] = rel["承接量_萬"] * 10000

# ====================== 基礎統計/關係數 ======================
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

# 建設/營造 -> 經銷商的加權承接量（用於頁籤 1）
dev_dealer = rel.groupby(["建設公司","經銷商"], dropna=False)["承接量_萬"].sum().reset_index()
con_dealer = rel.groupby(["營造公司","經銷商"], dropna=False)["承接量_萬"].sum().reset_index()

def topk_table(df_pair, key_col, k=3):
    rows = []
    for key, g in df_pair.groupby(key_col):
        g = g.sort_values("承接量_萬", ascending=False)
        top = g.head(k).reset_index(drop=True)
        row = {key_col: key}
        for i in range(k):
            if i < len(top):
                row[f"Top{i+1}經銷商"] = top.loc[i, "經銷商"]
                row[f"Top{i+1}份額(萬)"] = round(float(top.loc[i, "承接量_萬"]), 2)
            else:
                row[f"Top{i+1}經銷商"] = ""
                row[f"Top{i+1}份額(萬)"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"Top1份額(萬)", ascending=False)

dev_top = topk_table(dev_dealer, "建設公司", k=3)
con_top = topk_table(con_dealer, "營造公司", k=3)

# 建設/營造 -> 水電分布
dev_mep = df.groupby(["建設公司","水電公司"], dropna=False)["年使用量_萬"].sum().reset_index()
con_mep = df.groupby(["營造公司","水電公司"], dropna=False)["年使用量_萬"].sum().reset_index()

# 水電之間競爭（同一營造公司底下的共現；以年使用量_萬為權重）
def cooccurrence_pairs(df_group_key, item_col, weight_col):
    weights = {}
    matrix_index = set()
    for key, g in df.groupby(df_group_key):
        items = g[[item_col, weight_col]].dropna(subset=[item_col]).copy()
        items[item_col] = items[item_col].apply(clean_name)
        items = items.dropna(subset=[item_col])
        uniq = items[item_col].dropna().unique().tolist()
        uniq = [u for u in uniq if isinstance(u, str) and u != ""]
        if len(uniq) < 2:
            continue
        # 群組權重：使用該群組的年使用量總和
        group_w = float(items[weight_col].sum(skipna=True) or 0.0)
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                a, b = sorted([uniq[i], uniq[j]])
                matrix_index.add(a); matrix_index.add(b)
                weights[(a,b)] = weights.get((a,b), 0.0) + group_w
    labels = sorted(matrix_index)
    mat = pd.DataFrame(0.0, index=labels, columns=labels)
    for (a,b), w in weights.items():
        mat.loc[a,b] = mat.loc[a,b] + w
        mat.loc[b,a] = mat.loc[b,a] + w
    return mat

mep_competition = cooccurrence_pairs(["營造公司"], "水電公司", "年使用量_萬")

# 經銷商之間競合（同一水電公司；強度=年使用量_萬 * min(配比i, 配比j)）
def dealer_competition_matrix(rel_df):
    weights = {}
    labels = set()
    for mep, g in rel_df.groupby("水電公司"):
        g = g.dropna(subset=["經銷商","配比","年使用量_萬"]).copy()
        if g.empty:
            continue
        g["經銷商"] = g["經銷商"].apply(clean_name)
        mep_vol = float(g["年使用量_萬"].iloc[0] if "年使用量_萬" in g.columns else 0.0)
        dealers = g[["經銷商","配比"]].dropna().values.tolist()
        if len(dealers) < 2:
            continue
        for i in range(len(dealers)):
            for j in range(i+1, len(dealers)):
                a, ai = dealers[i]
                b, bi = dealers[j]
                if not isinstance(a, str) or not isinstance(b, str):
                    continue
                labels.add(a); labels.add(b)
                w = float(min(ai, bi) * mep_vol)
                key = tuple(sorted([a,b]))
                weights[key] = weights.get(key, 0.0) + w
    labels = sorted(labels)
    mat = pd.DataFrame(0.0, index=labels, columns=labels)
    for (a,b), w in weights.items():
        mat.loc[a,b] = mat.loc[a,b] + w
        mat.loc[b,a] = mat.loc[b,a] + w
    return mat

dealer_comp = dealer_competition_matrix(rel)

# ====================== Tabs ======================
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 原始資料",
    "🧭 關係概覽",
    "🏢 建設 → 經銷商",
    "🏗️ 建設 ↔ 營造 結構",
    "🔧 建設/營造 → 水電",
    "⚔️ 水電競爭",
    "🤝 經銷競合",
])

# -------- Tab 0: 原始資料 --------
with tab0:
    st.subheader("原始資料預覽")
    st.dataframe(df_raw, use_container_width=True)

# -------- Tab 1: 關係概覽 --------
with tab1:
    st.subheader("總覽 KPI 與關係數")
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
    st.subheader("關係流向圖（建設→營造→水電→經銷）")
    # 構建節點
    devs = [d for d in sorted(df["建設公司"].dropna().unique().tolist()) if isinstance(d, str) and d != ""]
    cons = [c for c in sorted(df["營造公司"].dropna().unique().tolist()) if isinstance(c, str) and c != ""]
    meps = [m for m in sorted(df["水電公司"].dropna().unique().tolist()) if isinstance(m, str) and m != ""]
    deas = [d for d in sorted(rel["經銷商"].dropna().unique().tolist()) if isinstance(d, str) and d != ""]

    nodes = (
        [f"建設｜{d}" for d in devs] +
        [f"營造｜{c}" for c in cons] +
        [f"水電｜{m}" for m in meps] +
        [f"經銷｜{d}" for d in deas]
    )
    node_index = {name: i for i, name in enumerate(nodes)}

    def add_links(df_links, src_label, dst_label, value_col):
        s, t, v = [], [], []
        for _, r in df_links.iterrows():
            s_key = f"{src_label}｜{r[src_label]}"
            t_key = f"{dst_label}｜{r[dst_label]}"
            if s_key in node_index and t_key in node_index:
                s.append(node_index[s_key])
                t.append(node_index[t_key])
                v.append(max(float(r[value_col] or 0.0), 0.0))
        return s, t, v

    link1 = df.groupby(["建設公司","營造公司"], dropna=False)["年使用量_萬"].sum().reset_index()
    s1, t1_, v1 = add_links(link1, "建設公司", "營造公司", "年使用量_萬")

    link2 = df.groupby(["營造公司","水電公司"], dropna=False)["年使用量_萬"].sum().reset_index()
    # rename for function compatibility
    link2 = link2.rename(columns={"營造公司":"營造公司", "水電公司":"水電公司"})
    s2, t2_, v2 = [], [], []
    for _, r in link2.iterrows():
        s_key = f"營造｜{r['營造公司']}"
        t_key = f"水電｜{r['水電公司']}"
        if s_key in node_index and t_key in node_index:
            s2.append(node_index[s_key]); t2_.append(node_index[t_key]); v2.append(max(float(r["年使用量_萬"] or 0.0), 0.0))

    link3 = rel.groupby(["水電公司","經銷商"], dropna=False)["承接量_萬"].sum().reset_index()
    s3, t3_, v3 = [], [], []
    for _, r in link3.iterrows():
        s_key = f"水電｜{r['水電公司']}"
        t_key = f"經銷｜{r['經銷商']}"
        if s_key in node_index and t_key in node_index:
            s3.append(node_index[s_key]); t3_.append(node_index[t_key]); v3.append(max(float(r["承接量_萬"] or 0.0), 0.0))

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

# -------- Tab 2: 建設 → 經銷商 --------
with tab2:
    st.subheader("建設公司 → 最可能下單的經銷商（加權承接量 Top3）")
    st.dataframe(dev_top, use_container_width=True)

    st.markdown("---")
    st.subheader("總覽：各建設公司的經銷商承接量")
    dev_tot = dev_dealer.groupby("建設公司")["承接量_萬"].sum().reset_index().sort_values("承接量_萬", ascending=False)
    fig = px.bar(dev_tot.head(30), x="建設公司", y="承接量_萬", title="建設公司對經銷商的加權承接量（總和）")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("邏輯：建設→營造→水電→經銷，使用 年使用量_萬 × 經銷配比 加總到經銷商層。")

# -------- Tab 3: 建設 ↔ 營造 結構 --------
with tab3:
    st.subheader("建設 ↔ 營造 關係熱力圖（出現次數）")
    pair_count = df.groupby(["建設公司","營造公司"]).size().reset_index(name="次數")
    if not pair_count.empty:
        heat = pair_count.pivot(index="建設公司", columns="營造公司", values="次數").fillna(0.0)
        fig = px.imshow(heat, aspect="auto", title="建設×營造 出現次數熱力圖")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("資料不足，無法繪製熱力圖。")

    st.markdown("---")
    st.subheader("營造集中度（同一建設公司的營造占比）")
    dev_con_share = df.groupby(["建設公司","營造公司"]).size().groupby(level=0).apply(lambda s: s / s.sum()).reset_index(name="占比")
    # HHI 指標
    hhi = dev_con_share.groupby("建設公司")["占比"].apply(lambda s: (s**2).sum()).reset_index(name="HHI")
    top_concentrated = hhi.sort_values("HHI", ascending=False)
    st.dataframe(top_concentrated, use_container_width=True)
    st.caption("HHI 越高表示營造更集中（可能為子公司或特定合作關係）。")

# -------- Tab 4: 建設/營造 → 水電 --------
with tab4:
    t1, t2 = st.tabs(["建設 → 水電 分布", "營造 → 水電 分布"])
    with t1:
        st.subheader("建設公司下的水電使用量分布")
        top_dev = dev_mep.groupby("建設公司")["年使用量_萬"].sum().reset_index().sort_values("年使用量_萬", ascending=False).head(20)
        st.write("Top 建設公司（依總用量）")
        st.dataframe(top_dev, use_container_width=True)
        if not dev_mep.empty:
            fig = px.treemap(dev_mep, path=["建設公司","水電公司"], values="年使用量_萬", title="建設→水電 Treemap（年使用量_萬）")
            st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.subheader("營造公司下的水電使用量分布")
        top_con = con_mep.groupby("營造公司")["年使用量_萬"].sum().reset_index().sort_values("年使用量_萬", ascending=False).head(20)
        st.write("Top 營造公司（依總用量）")
        st.dataframe(top_con, use_container_width=True)
        if not con_mep.empty:
            fig = px.treemap(con_mep, path=["營造公司","水電公司"], values="年使用量_萬", title="營造→水電 Treemap（年使用量_萬）")
            st.plotly_chart(fig, use_container_width=True)

# -------- Tab 5: 水電競爭 --------
with tab5:
    st.subheader("水電之間的競爭強度（同一營造公司中的共現，權重=該營造群組的年使用量總和）")
    if not mep_competition.empty:
        fig = px.imshow(mep_competition, aspect="auto", title="水電×水電 共現熱力圖（加權）")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("資料不足，無法生成水電共現矩陣。")
    st.caption("在同一營造公司底下同時出現的水電視為競爭者，群組權重採用該群組年使用量總和。")

# -------- Tab 6: 經銷競合 --------
with tab6:
    st.subheader("經銷商之間的競合強度（同一水電公司；強度=年使用量_萬 × min(配比i, 配比j)）")
    if not dealer_comp.empty:
        fig = px.imshow(dealer_comp, aspect="auto", title="經銷商×經銷商 競合熱力圖（加權）")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("資料不足，無法生成經銷競合矩陣。")
    st.caption("同一水電公司若同時與多家經銷商合作，彼此視為競爭者；重疊強度採 min(配比i, 配比j) 乘以該水電的年使用量。")

st.markdown("---")
st.subheader("⬇️ 匯出")
# 匯出主要結果
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_raw.to_excel(writer, index=False, sheet_name="原始資料")
    df.to_excel(writer, index=False, sheet_name="主檔(固定欄位命名)")
    rel.to_excel(writer, index=False, sheet_name="關係明細")
    ratio_check.to_excel(writer, index=False, sheet_name="配比檢查")
    top_ratio.to_excel(writer, index=False, sheet_name="單一依賴檢查")
    dev_top.to_excel(writer, index=False, sheet_name="建設->經銷 Top3")
    con_top.to_excel(writer, index=False, sheet_name="營造->經銷 Top3")
    dev_mep.to_excel(writer, index=False, sheet_name="建設->水電 分布")
    con_mep.to_excel(writer, index=False, sheet_name="營造->水電 分布")
st.download_button(
    "下載 Excel（多工作表）",
    data=output.getvalue(),
    file_name="relations_dashboard_full.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
