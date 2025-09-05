import json
import requests
import io
import pandas as pd
import streamlit as st
import plotly.express as px

# 將 GeoJSON 檔案的 URL 寫入，這是台灣鄉鎮區的公開 GeoJSON 數據
# 來源: g0v/twgeojson on GitHub
geojson_url = "https://raw.githubusercontent.com/g0v/twgeojson/master/json/twCounty2010.geo.json"

@st.cache_data
def load_geojson_and_data(geojson_url, uploaded_file):
    """
    載入 GeoJSON 檔案和使用者上傳的資料。
    使用 @st.cache_data 確保資料只在檔案變動時重新載入。
    """
    # 載入 GeoJSON
    try:
        response = requests.get(geojson_url)
        geojson_data = response.json()
    except Exception as e:
        st.error(f"無法載入 GeoJSON 檔案，請檢查網路連線或 URL：{e}")
        return None, None
    
    # 讀取使用者上傳的 CSV 檔案
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(uploaded_file, encoding='gbk')
    
    return geojson_data, df_raw

def generate_map(geojson_data, df_raw):
    """
    處理資料並產生品牌地圖。
    """
    # 定義欄位名稱對應，以確保程式碼的彈性
    col_mapping = {
        'city': '縣市',
        'area': '區域',
        'brand_a': '品牌A',
        'ratio_a': '品牌A佔比(%)',
        'brand_b': '品牌B',
        'ratio_b': '品牌B佔比(%)',
        'brand_c': '品牌C',
        'ratio_c': '品牌C佔比(%)',
    }
    
    # 將多個品牌欄位轉換為一個長格式的 DataFrame，方便處理
    brands_data = []
    for index, row in df_raw.iterrows():
        # 遍歷所有品牌及其佔比欄位
        for brand_key, ratio_key in [('brand_a', 'ratio_a'), ('brand_b', 'ratio_b'), ('brand_c', 'ratio_c')]:
            brand = row.get(col_mapping[brand_key])
            ratio = row.get(col_mapping[ratio_key])
            if pd.notna(brand) and pd.notna(ratio):
                brands_data.append({
                    'city': row.get(col_mapping['city']),
                    'area': row.get(col_mapping['area']),
                    'brand': brand,
                    # 確保配比是數字類型，並處理百分號
                    'ratio': float(str(ratio).replace('%',''))
                })
    
    df_brands = pd.DataFrame(brands_data)
    
    if df_brands.empty:
        st.info("資料中沒有品牌資訊，無法產生地圖。")
        return

    # 組合縣市和區域名稱，以作為地圖和數據合併的鍵
    df_brands['full_area_name'] = df_brands['city'].astype(str) + df_brands['area'].astype(str)
    
    # 找到每個區域最主要的品牌（佔比最高者）
    # 使用 idxmax() 找到每個分組中 'ratio' 的最大值索引
    idx = df_brands.groupby(['full_area_name'])['ratio'].idxmax()
    df_dominant_brands = df_brands.loc[idx].reset_index(drop=True)

    # 準備 GeoJSON 數據，並將我們的品牌數據合併進去
    geojson_features = {f['properties']['COUNTYNAME']+f['properties']['TOWNNAME']: f for f in geojson_data['features']}
    
    # 為每個 GeoJSON 特徵添加「主導品牌」屬性
    for feature in geojson_data['features']:
        full_name = feature['properties']['COUNTYNAME'] + feature['properties']['TOWNNAME']
        if full_name in df_dominant_brands['full_area_name'].values:
            brand_info = df_dominant_brands[df_dominant_brands['full_area_name'] == full_name].iloc[0]
            feature['properties']['dominant_brand'] = brand_info['brand']
        else:
            feature['properties']['dominant_brand'] = "無資料"

    # 設定淺色調色盤
    unique_brands = sorted(df_dominant_brands['brand'].unique().tolist())
    color_palette = px.colors.qualitative.Pastel
    color_map = {brand: color_palette[i % len(color_palette)] for i, brand in enumerate(unique_brands)}
    color_map['無資料'] = '#e0e0e0' # 灰色顯示無資料區域

    # 建立地圖
    fig = px.choropleth_mapbox(
        df_dominant_brands,
        geojson=geojson_data,
        featureidkey="properties.COUNTYNAME",
        locations='full_area_name',
        color='brand',
        mapbox_style="carto-positron",
        zoom=6.5,
        center={"lat": 23.6, "lon": 120.9},
        opacity=0.7,
        color_discrete_map=color_map,
        hover_data={'city': True, 'area': True, 'brand': True, 'full_area_name': False},
        labels={'city': '縣市', 'area': '區域', 'brand': '最主要品牌'}
    )
    
    fig.update_layout(title_text="台灣各區主要品牌地圖分析", title_x=0.5, margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# 範例：如何將地圖功能整合到你的應用程式中
# 你可以選擇在適當的位置（例如在你的 _render_overall_statistics 方法中）呼叫這段程式碼
# 這裡只是一個示意，請根據你的需求調整
def your_main_app_logic(uploaded_file):
    # 載入 GeoJSON 和你的資料
    geojson_data, df_raw = load_geojson_and_data(geojson_url, uploaded_file)
    
    # 如果資料成功載入，則生成地圖
    if geojson_data and df_raw is not None:
        generate_map(geojson_data, df_raw)
