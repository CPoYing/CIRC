import io
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Excel 分析網站", page_icon="📊", layout="wide")
st.title("📊 Excel 分析網站（免安裝版）")
st.caption("上傳 Excel/CSV → 過濾與彙整 → 自訂欄位 → 圖表 → 下載結果")

# 1) 上傳檔案
file = st.file_uploader(
    "上傳 Excel 或 CSV 檔",
    type=["xlsx", "xls", "csv"],
    help="最多 200 MB；Excel 需使用 openpyxl 解析"
)

@st.cache_data
def load_data(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return df

if file:
    df = load_data(file)
    st.subheader("原始資料預覽")
    st.dataframe(df, use_container_width=True)

    with st.sidebar:
        st.header("⚙️ 操作區")

        # 2) 基本過濾
        cols = df.columns.tolist()
        filter_col = st.selectbox("選擇要篩選的欄位（可略）", ["（不篩選）"] + cols)
        filtered = df.copy()
        if filter_col != "（不篩選）":
            unique_vals = filtered[filter_col].dropna().unique().tolist()
            selected_vals = st.multiselect("選取值", unique_vals)
            if selected_vals:
                filtered = filtered[filtered[filter_col].isin(selected_vals)]

        # 3) 自訂欄位（簡易公式）
        st.markdown("—")
        st.subheader("🧮 新增自訂欄位")
        st.caption("使用 pandas 計算：例如 `新欄位 = 數量 * 單價` 或 `毛利率 = (銷售額-成本)/銷售額`")
        new_col_name = st.text_input("新欄位名稱", value="")
        formula_help = st.expander("公式說明 / 範例")
        with formula_help:
            st.write("""
- 可用現有欄位名稱進行四則運算，例如：  
  - `數量 * 單價`  
  - `(銷售額 - 成本) / 銷售額`
- 會自動把結果存到新欄位名稱。
            """)
        formula = st.text_input("輸入公式（只寫等號右邊）", placeholder="例如：數量 * 單價")

        data_to_use = filtered.copy()
        if new_col_name and formula:
            try:
                # 安全地用 eval 計算（限制在現有欄位名稱）
                data_to_use[new_col_name] = pd.eval(formula, engine="python", local_dict=data_to_use.to_dict('series'))
                st.success(f"已新增欄位：{new_col_name}")
            except Exception as e:
                st.error(f"公式錯誤：{e}")

        # 4) 彙整（群組 + 聚合）
        st.markdown("—")
        st.subheader("📦 聚合彙整")
        group_cols = st.multiselect("群組欄位（可多選）", cols)
        numeric_cols = data_to_use.select_dtypes(include="number").columns.tolist()
        agg_col = st.selectbox("聚合欄位", numeric_cols or ["（無數值欄位）"])
        agg_fn = st.selectbox("聚合函數", ["sum", "mean", "max", "min", "count"])

    # 彙整結果
    agg_df = data_to_use.copy()
    if group_cols and agg_col != "（無數值欄位）":
        agg_df = agg_df.groupby(group_cols, dropna=False)[agg_col].agg(agg_fn).reset_index().rename(columns={agg_col: f"{agg_fn}_{agg_col}"})

    st.subheader("處理後資料")
    st.dataframe(agg_df, use_container_width=True)

    # 5) 視覺化
    st.subheader("📈 視覺化")
    if len(agg_df.columns) >= 2:
        # 嘗試自動找類別與數值欄位
        num_cols = agg_df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in agg_df.columns if c not in num_cols]

        x = st.selectbox("X 軸", cat_cols + num_cols)
        y = st.selectbox("Y 軸（數值）", num_cols or [None])
        chart_type = st.selectbox("圖表類型", ["bar", "line", "scatter"])
        if y:
            if chart_type == "bar":
                fig = px.bar(agg_df, x=x, y=y)
            elif chart_type == "line":
                fig = px.line(agg_df, x=x, y=y)
            else:
                fig = px.scatter(agg_df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("目前沒有數值欄位可畫圖。")

    # 6) 下載處理後檔案
    st.subheader("⬇️ 下載")
    # CSV
    csv_bytes = agg_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載 CSV", data=csv_bytes, file_name="result.csv", mime="text/csv")

    # Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data_to_use.to_excel(writer, index=False, sheet_name="處理前")
        agg_df.to_excel(writer, index=False, sheet_name="彙整後")
    st.download_button("下載 Excel", data=output.getvalue(), file_name="result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("請先在上方上傳 Excel 或 CSV 檔。")
