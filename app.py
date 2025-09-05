# app.py - 百大建商｜關係鏈分析（單頁搜尋 v12 Enhanced）
"""
進階建材供應鏈分析儀表板
完整功能版本 - 確保所有修改都正確應用
"""

import io
import re
import math
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import warnings
import json
import requests

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

warnings.filterwarnings('ignore')

# ====================== 配置設定 ======================
class Config:
    """應用程式配置和常數"""
    APP_TITLE = "百大建商｜關係鏈分析"
    VERSION = "v12 Enhanced"
    ROLES = ["建設公司", "營造公司", "水電公司", "經銷商"]
    CHART_TYPES = ["圓餅圖", "長條圖"]
    COLOR_PALETTE = px.colors.qualitative.Set3
    
    # 欄位對應（基於0的索引位置）
    COLUMN_MAPPING = {
        'dev': (3, ["建商", "建設公司", "建設公司(業主)"]),
        'con': (4, ["營造公司", "營造商"]),
        'mep': (5, ["水電全名", "水電公司", "機電公司", "機電廠商"]),
        'vol': (6, ["年使用量/萬", "年使用量(萬)", "用量_萬"]),
        'dealer_a': (7, ["經銷商A", "經銷A", "經銷商1"]),
        'ratio_a': (8, ["經銷A佔比(%)", "經銷商A配比", "A配比"]),
        'dealer_b': (9, ["經銷商B", "經銷B", "經銷商2"]),
        'ratio_b': (10, ["經銷B佔比(%)", "經銷商B配比", "B配比"]),
        'dealer_c': (11, ["經銷商C", "經銷Ｃ", "經銷商3"]),
        'ratio_c': (12, ["經銷Ｃ佔比(%)", "經銷C佔比(%)", "經銷商C配比", "C配比"]),
        'brand_a': (13, ["品牌A", "線纜品牌A", "線纜品牌1", "品牌1"]),
        'brand_ratio_a': (14, ["品牌A佔比(%)", "品牌A配比", "品牌1佔比", "A品牌佔比", "A品牌配比"]),
        'brand_b': (15, ["品牌B", "線纜品牌B", "線纜品牌2", "品牌2"]),
        'brand_ratio_b': (16, ["品牌B佔比(%)", "品牌B配比", "品牌2佔比", "B品牌佔比", "B品牌配比"]),
        'brand_c': (17, ["品牌C", "線纜品牌C", "線纜品牌3", "品牌3"]),
        'brand_ratio_c': (18, ["品牌C佔比(%)", "品牌C配比", "品牌3佔比", "C品牌佔比", "C品牌配比"]),
        'city': (19, ["縣市", "縣/市", "所在縣市"]),  # 水電公司所在縣市
        'area': (20, ["區域", "地區", "區/鄉鎮"])  # 水電公司所在區域
    }

# ====================== 資料處理類別 ======================
class DataProcessor:
    """處理資料處理和轉換"""
    
    @staticmethod
    def clean_name(x) -> Optional[str]:
        """清理並標準化名稱"""
        if pd.isna(x):
            return None
        s = str(x).replace("\u3000", " ").strip()
        s = re.sub(r"\s+", " ", s)
        if s == "" or s.lower() in {"nan", "none"} or s == "0":
            return None
        return s
    
    @staticmethod
    def coerce_num(s) -> float:
        """處理錯誤並將字串轉換為數字"""
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return np.nan
        if isinstance(s, (int, float, np.number)):
            return float(s)
        s = str(s).replace(",", "").replace("%", "").strip()
        if s in ("", "nan", "None"):
            return np.nan
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan
    
    @staticmethod
    def normalize_ratio(series: pd.Series) -> pd.Series:
        """將配比值正規化到0-1的範圍內"""
        s = series.apply(DataProcessor.coerce_num)
        max_val = s.max(skipna=True)
        if max_val is not None and max_val > 1.000001:
            return s / 100.0
        return s
    
    @staticmethod
    def get_col_by_pos_or_name(df: pd.DataFrame, pos: int, name_candidates: List[str]) -> Optional[str]:
        """根據位置或名稱獲取欄位，並提供備用選項"""
        try:
            col_by_pos = df.columns[pos]
            if col_by_pos is not None:
                return col_by_pos
        except (IndexError, KeyError):
            pass
        
        for name in name_candidates:
            if name in df.columns:
                return name
        return None

# ====================== 分析類別 ======================
class RelationshipAnalyzer:
    """分析實體之間的關係"""
    
    def __init__(self, df: pd.DataFrame, rel: pd.DataFrame, brand_rel: pd.DataFrame, mep_vol_map: Dict):
        self.df = df
        self.rel = rel
        self.brand_rel = brand_rel
        self.mep_vol_map = mep_vol_map
    
    def avg_dealer_ratio_across_unique_mep(self, rel_subset: pd.DataFrame) -> pd.DataFrame:
        """計算在不重複的水電公司中的平均經銷商配比"""
        meps = [m for m in rel_subset["水電公司"].dropna().unique() if isinstance(m, str) and m != ""]
        n = len(meps)
        if n == 0:
            return pd.DataFrame(columns=["經銷商", "平均配比"])
        
        sums = defaultdict(float)
        for mep in meps:
            g = rel_subset[rel_subset["水電公司"] == mep]
            rmap = g.groupby("經銷商")["配比"].mean().to_dict()
            for dealer, ratio in rmap.items():
                if pd.notna(dealer):
                    sums[str(dealer)] += float(ratio or 0.0)
        
        rows = [(dealer, s / n) for dealer, s in sums.items()]
        return (pd.DataFrame(rows, columns=["經銷商", "平均配比"])
                .sort_values("平均配比", ascending=False))
    
    def avg_brand_ratio_across_unique_mep(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """計算在不重複的水電公司中加權平均的品牌配比"""
        if self.brand_rel.empty:
            return pd.DataFrame(columns=["品牌", "加權平均配比"])
        
        meps = [m for m in df_subset["水電公司"].dropna().unique() if isinstance(m, str) and m != ""]
        if len(meps) == 0:
            return pd.DataFrame(columns=["品牌", "加權平均配比"])
        
        brand_weighted_sums = defaultdict(float)
        total_volume = 0.0
        
        for mep in meps:
            mep_volume = float(self.mep_vol_map.get(mep, 0.0) or 0.0)
            if mep_volume <= 0:
                mep_volume = 1.0
            
            brand_subset = self.brand_rel[self.brand_rel["水電公司"] == mep]
            if not brand_subset.empty:
                brand_map = brand_subset.groupby("品牌")["配比"].mean().to_dict()
                for brand, ratio in brand_map.items():
                    if pd.notna(brand):
                        brand_weighted_sums[str(brand)] += float(ratio or 0.0) * mep_volume
                total_volume += mep_volume
        
        if total_volume <= 0 or not brand_weighted_sums:
            return pd.DataFrame(columns=["品牌", "加權平均配比"])
        
        rows = [(brand, weighted_sum / total_volume) 
                for brand, weighted_sum in brand_weighted_sums.items()]
        
        return (pd.DataFrame(rows, columns=["品牌", "加權平均配比"])
                .sort_values("加權平均配比", ascending=False))
    
    def union_overlap_share_and_total(self, target_dealer: str) -> Tuple[float, float]:
        """計算經銷商的聯合重疊佔比和總市場額度"""
        target_clients = self.rel[self.rel["經銷商"] == target_dealer]["水電公司"].dropna().unique()
        tgt_ratio_map = (self.rel[self.rel["經銷商"] == target_dealer]
                         .groupby("水電公司")["配比"].mean().to_dict())
        
        total_target = 0.0
        union_overlap = 0.0
        
        for mep in target_clients:
            vol = float(self.mep_vol_map.get(mep, 0.0) or 0.0)
            r_t = float(tgt_ratio_map.get(mep, 0.0) or 0.0)
            comp_sum = float((self.rel[self.rel["水電公司"] == mep]
                              .groupby("經銷商")["配比"].mean()
                              .drop(labels=[target_dealer], errors="ignore")
                              .sum()) or 0.0)
            union_overlap += vol * min(r_t, comp_sum)
            total_target += vol * r_t
        
        share = (union_overlap / total_target) if total_target > 0 else 0.0
        return share, total_target

class CompetitorAnalyzer:
    """分析競爭對手和市場競爭"""
    
    def __init__(self, df: pd.DataFrame, rel: pd.DataFrame, mep_vol_map: Dict):
        self.df = df
        self.rel = rel
        self.mep_vol_map = mep_vol_map
    
    def water_competitors(self, target_mep: str) -> pd.DataFrame:
        """為水電公司尋找競爭對手"""
        g = self.df[self.df["水電公司"].notna()]
        cons = g[g["水電公司"] == target_mep]["營造公司"].dropna().unique()
        
        if len(cons) == 0:
            return pd.DataFrame(columns=["競爭對手", "共同出現次數"])
        
        candidates = g[g["營造公司"].isin(cons)]
        competitors = (candidates[candidates["水電公司"] != target_mep]
                       .groupby("水電公司").size()
                       .reset_index(name="共同出現次數")
                       .sort_values("共同出現次數", ascending=False))
        
        return competitors
    
    def dealer_competitors(self, target_dealer: str) -> Tuple[pd.DataFrame, float]:
        """為經銷商尋找競爭對手並提供詳細分析"""
        target_clients = self.rel[self.rel["經銷商"] == target_dealer]["水電公司"].dropna().unique()
        target_client_set = set(target_clients)
        target_total_clients = len(target_client_set)
        
        tgt_ratio_map = (self.rel[self.rel["經銷商"] == target_dealer]
                         .groupby("水電公司")["配比"].mean().to_dict())
        
        target_total_market = sum(
            float(self.mep_vol_map.get(mep, 0.0) or 0.0) * float(tgt_ratio_map.get(mep, 0.0) or 0.0)
            for mep in target_client_set
        )
        
        stats = {}
        for mep, grp in self.rel.groupby("水電公司"):
            if mep not in target_client_set:
                continue
                
            vol = float(self.mep_vol_map.get(mep, 0.0) or 0.0)
            ratios = grp.groupby("經銷商")["配比"].mean().to_dict()
            
            if target_dealer not in ratios:
                continue
                
            r_t = float(ratios[target_dealer] or 0.0)
            
            for dealer, r_c in ratios.items():
                if dealer == target_dealer or pd.isna(dealer):
                    continue
                    
                d = stats.setdefault(dealer, {
                    "共同客戶數": 0,
                    "overlap_ratio_sum": 0.0,
                    "共同市場額度": 0.0,
                    "重疊市場額度": 0.0
                })
                
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
                "共同客戶數占比": Formatters.pct_str(shared_pct),
                "競爭指數": Formatters.pct_str(comp_index),
                "共同市場額度(萬)": round(d["共同市場額度"], 2),
                "重疊市場額度(萬)": round(d["重疊市場額度"], 2),
                "重疊市場占比": Formatters.pct_str(overlap_market_share),
                "威脅程度": threat,
            })
        
        df_result = pd.DataFrame(rows)
        if df_result.empty:
            return df_result, target_total_market
            
        cat = pd.Categorical(df_result["威脅程度"], categories=["高", "中", "低"], ordered=True)
        df_result = (df_result.assign(_order=cat)
                     .sort_values(["_order", "重疊市場占比", "共同客戶數"], ascending=[True, False, False])
                     .drop(columns="_order"))
        
        return df_result, target_total_market

# ====================== 格式化工具 ======================
class Formatters:
    """格式化資料以供顯示"""
    
    @staticmethod
    def pct_str(x) -> str:
        """格式化百分比並正確四捨五入"""
        if pd.isna(x):
            return "-"
        v = float(x)
        if v <= 1.0:
            v = v * 100.0
        d = Decimal(str(v)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        return f"{d}%"
    
    @staticmethod
    def fmt_amount(x) -> str:
        """格式化金額並使用千分位分隔符"""
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):,.2f}"

# ====================== 視覺化工具 ======================
class ChartGenerator:
    """生成圖表和視覺化內容"""
    
    @staticmethod
    def create_chart(df_plot: pd.DataFrame, name_col: str, value_col: str, 
                     title: str, chart_type: str = "圓餅圖", key_suffix: str = "") -> go.Figure:
        """創建具有更好樣式的進階圖表"""
        if df_plot is None or df_plot.empty:
            return None
        
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        
        if chart_type == "長條圖":
            fig = px.bar(
                df_plot, x=name_col, y=value_col, title=title,
                color=name_col, color_discrete_sequence=colors,
                template="plotly_white"
            )
            
            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside',
                marker_line_width=1,
                marker_line_color='rgba(0,0,0,0.2)'
            )
            
            fig.update_layout(
                showlegend=False,
                title_font_size=18,
                title_font_family="Arial Black",
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=60, b=60, l=60, r=60),
                xaxis=dict(
                    tickangle=45,
                    title_standoff=25
                )
            )
            
        else:  # 圓餅圖
            fig = px.pie(
                df_plot, names=name_col, values=value_col, title=title,
                color=name_col, color_discrete_sequence=colors,
                template="plotly_white"
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            fig.update_layout(
                title_font_size=18,
                title_font_family="Arial Black",
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(t=60, b=60, l=60, r=200)
            )
        
        return fig

# ====================== UI 元件 ======================
class UIComponents:
    """可重複使用的使用者介面元件"""
    
    @staticmethod
    def render_kpi_section(stats: Dict[str, int]):
        """使用 Streamlit 原生指標來渲染關鍵績效指標區塊"""
        cols = st.columns(len(stats))
        for i, (label, value) in enumerate(stats.items()):
            with cols[i]:
                st.metric(label=label, value=f"{value:,}")
    
    @staticmethod
    def render_section_header(title: str):
        """渲染具有一致樣式的區塊標題"""
        st.markdown(f"**{title}**")
    
    @staticmethod
    def render_info_box(message: str):
        """渲染具有增強樣式的資訊框"""
        st.info(message)
    
    @staticmethod
    def render_dataframe_with_styling(df: pd.DataFrame, title: str = None):
        """渲染具有增強樣式和格式化的資料框架"""
        if title:
            st.markdown(f"**{title}**")
        
        if df.empty:
            UIComponents.render_info_box("暫無資料")
        else:
            df_styled = df.copy()
            
            numeric_columns = df_styled.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df_styled.columns:
                    if df_styled[col].dtype in ['int64', 'int32']:
                        df_styled[col] = df_styled[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "-")
                    else:
                        df_styled[col] = df_styled[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
            
            st.markdown("""
                <style>
                .dataframe th {
                    text-align: center !important;
                    background-color: #f8f9fa !important;
                    font-weight: 600 !important;
                    padding: 8px !important;
                }
                .dataframe td {
                    padding: 6px 8px !important;
                }
                .dataframe td:nth-child(n+2) {
                    text-align: right !important;
                }
                .dataframe td:first-child {
                    text-align: left !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            column_config = {}
            for col in df_styled.columns:
                if col == df_styled.columns[0]:
                    column_config[col] = st.column_config.TextColumn(col, width="medium")
                else:
                    column_config[col] = st.column_config.TextColumn(col, width="small")
            
            st.dataframe(
                df_styled, 
                use_container_width=True,  # 使用整個容器寬度
                hide_index=True,
                column_config=column_config
            )

# ====================== 主要應用程式 ======================
class ConstructionDashboard:
    """主儀表板應用程式"""
    
    def __init__(self):
        self.config = Config()
        self.setup_page()
        
    def setup_page(self):
        """設定頁面配置"""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🏗️"
        )
        
        st.title(self.config.APP_TITLE)
        
        try:
            p = Path(__file__)
            version_info = f"版本：{self.config.VERSION} | 檔案：{p.name} | 修改時間：{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
        except Exception:
            version_info = f"版本：{self.config.VERSION}"
            
        st.caption(version_info)
    
    @st.cache_data
    def read_file(_self, file) -> pd.DataFrame:
        """使用快取讀取上傳的檔案"""
        try:
            if file.name.lower().endswith(".csv"):
                return pd.read_csv(file, encoding='utf-8')
            else:
                return pd.read_excel(file, engine="openpyxl")
        except UnicodeDecodeError:
            if file.name.lower().endswith(".csv"):
                return pd.read_csv(file, encoding='gbk')
            raise
    
    def process_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """將原始資料處理成可供分析的格式"""
        processor = DataProcessor()
        
        columns = {}
        for key, (pos, names) in self.config.COLUMN_MAPPING.items():
            columns[key] = processor.get_col_by_pos_or_name(df_raw, pos, names)
        
        required_cols = [columns['dev'], columns['con'], columns['mep']]
        if any(col is None for col in required_cols):
            st.error("找不到必要欄位（建設公司/營造公司/水電公司）。請確認資料格式。")
            st.stop()
        
        rename_map = {
            columns['dev']: "建設公司",
            columns['con']: "營造公司", 
            columns['mep']: "水電公司",
            columns['vol']: "年使用量_萬",
        }
        
        for suffix in ['a', 'b', 'c']:
            dealer_key = f'dealer_{suffix}'
            ratio_key = f'ratio_{suffix}'
            brand_key = f'brand_{suffix}'
            brand_ratio_key = f'brand_ratio_{suffix}'
            
            if columns.get(dealer_key):
                rename_map[columns[dealer_key]] = f"經銷商{suffix.upper()}"
            if columns.get(ratio_key):
                rename_map[columns[ratio_key]] = f"經銷{suffix.upper()}比"
            if columns.get(brand_key):
                rename_map[columns[brand_key]] = f"品牌{suffix.upper()}"
            if columns.get(brand_ratio_key):
                rename_map[columns[brand_ratio_key]] = f"品牌{suffix.upper()}比"
        
        if columns.get('city'):
            rename_map[columns['city']] = "縣市"
        if columns.get('area'):
            rename_map[columns['area']] = "區域"
        
        df = df_raw.rename(columns=rename_map).copy()
        
        text_cols = ["建設公司", "營造公司", "水電公司", "縣市", "區域"] + [f"經銷商{s}" for s in ['A','B','C']] + [f"品牌{s}" for s in ['A','B','C']]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(processor.clean_name)
        
        if "年使用量_萬" in df.columns:
            df["年使用量_萬"] = df["年使用量_萬"].apply(processor.coerce_num)
        
        ratio_cols = [f"經銷{s}比" for s in ['A','B','C']] + [f"品牌{s}比" for s in ['A','B','C']]
        for col in ratio_cols:
            if col in df.columns:
                df[col] = processor.normalize_ratio(df[col])
        
        rel = self._create_dealer_relations(df)
        brand_rel = self._create_brand_relations(df)
        
        mep_vol_map = df.groupby("水電公司")["年使用量_萬"].apply(
            lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else np.nan
        ).to_dict()
        
        return df, rel, brand_rel, mep_vol_map
    
    def _create_dealer_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建經銷商關係資料框架"""
        blocks = []
        for suffix in ['A', 'B', 'C']:
            dealer_col = f"經銷商{suffix}"
            ratio_col = f"經銷{suffix}比"
            if dealer_col in df.columns and ratio_col in df.columns:
                base_cols = ["建設公司", "營造公司", "水電公司"]
                if "縣市" in df.columns:
                    base_cols.append("縣市")
                if "區域" in df.columns:
                    base_cols.append("區域")
                
                block = df[base_cols + [dealer_col, ratio_col]].rename(
                    columns={dealer_col: "經銷商", ratio_col: "配比"}
                )
                blocks.append(block)
        
        if blocks:
            rel = pd.concat(blocks, ignore_index=True)
            rel["經銷商"] = rel["經銷商"].apply(DataProcessor.clean_name)
            return rel.dropna(subset=["經銷商", "水電公司"])
        
        base_cols = ["建設公司", "營造公司", "水電公司", "經銷商", "配比"]
        if "縣市" in df.columns:
            base_cols.insert(-2, "縣市")
        if "區域" in df.columns:
            base_cols.insert(-2, "區域")
        return pd.DataFrame(columns=base_cols)
    
    def _create_brand_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建品牌關係資料框架"""
        blocks = []
        for suffix in ['A', 'B', 'C']:
            brand_col = f"品牌{suffix}"
            ratio_col = f"品牌{suffix}比"
            if brand_col in df.columns and ratio_col in df.columns:
                base_cols = ["建設公司", "營造公司", "水電公司"]
                if "縣市" in df.columns:
                    base_cols.append("縣市")
                if "區域" in df.columns:
                    base_cols.append("區域")
                
                block = df[base_cols + [brand_col, ratio_col]].rename(
                    columns={brand_col: "品牌", ratio_col: "配比"}
                )
                blocks.append(block)
        
        if blocks:
            brand_rel = pd.concat(blocks, ignore_index=True)
            brand_rel["品牌"] = brand_rel["品牌"].apply(DataProcessor.clean_name)
            return brand_rel.dropna(subset=["品牌", "水電公司"])
        
        base_cols = ["建設公司", "營造公司", "水電公司", "品牌", "配比"]
        if "縣市" in df.columns:
            base_cols.insert(-2, "縣市")
        if "區域" in df.columns:
            base_cols.insert(-2, "區域")
        return pd.DataFrame(columns=base_cols)
    
    def run(self):
        """運行主應用程式"""
        st.markdown("### 📁 資料上傳")
        uploaded_file = st.file_uploader(
            "上傳 Excel 或 CSV 檔案",
            type=["xlsx", "xls", "csv"],
            help="支援 Excel (.xlsx, .xls) 和 CSV 格式檔案"
        )
        
        if not uploaded_file:
            st.info("請上傳 Excel 或 CSV 檔案開始分析")
            
            with st.expander("使用說明", expanded=False):
                st.write("**檔案格式要求：**")
                st.write("• 固定欄位順序：D=建設公司, E=營造公司, F=水電公司, G=年用量")
                st.write("• H/J/L=經銷商A/B/C, I/K/M=對應配比")
                st.write("• N/P/R=品牌A/B/C, O/Q/S=對應占比")
                st.write("• T=縣市(水電所在地), U=區域(水電所在地)")
                st.write("")
                st.write("**支援分析角色：**")
                st.write("• 建設公司：查看營造、水電、經銷商合作關係")
                st.write("• 營造公司：分析上下游合作網絡及競爭態勢")
                st.write("• 水電公司：經銷商配比、品牌使用分析")
                st.write("• 經銷商：客戶分布、市場競爭分析")
            st.stop()
        
        df_raw = self.read_file(uploaded_file)
        df, rel, brand_rel, mep_vol_map = self.process_data(df_raw)
        
        tab_overview, tab_analysis, tab_map = st.tabs(["📊 數據概覽", "🎯 分析設定", "🗺️ 地圖分析"])
        
        with tab_overview:
            self._render_overall_statistics(df, rel, brand_rel)
        
        with tab_analysis:
            self._render_analysis_settings(df, rel, brand_rel, mep_vol_map, df_raw)

        with tab_map:
            self._render_map_analysis(df)
    
    def _render_overall_statistics(self, df: pd.DataFrame, rel: pd.DataFrame, brand_rel: pd.DataFrame):
        """渲染整體統計數據"""
        st.markdown("""
            <style>
            .stMetric {
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid transparent;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            }
            
            [data-testid="metric-container"]:nth-child(1) .stMetric {
                border-left-color: #87CEEB;
                background-color: #F0F8FF;
            }
            [data-testid="metric-container"]:nth-child(2) .stMetric {
                border-left-color: #DDA0DD;
                background-color: #F8F0FF;
            }
            [data-testid="metric-container"]:nth-child(3) .stMetric {
                border-left-color: #98FB98;
                background-color: #F0FFF0;
            }
            [data-testid="metric-container"]:nth-child(4) .stMetric {
                border-left-color: #FFB6C1;
                background-color: #FFF0F5;
            }
            [data-testid="metric-container"]:nth-child(5) .stMetric {
                border-left-color: #F0E68C;
                background-color: #FFFACD;
            }
            [data-testid="metric-container"]:nth-child(6) .stMetric {
                border-left-color: #FFA07A;
                background-color: #FFF8DC;
            }
            </style>
        """, unsafe_allow_html=True)
        
        total_records = len(df)
        total_developers = df["建設公司"].nunique()
        total_contractors = df["營造公司"].nunique()
        total_meps = df["水電公司"].nunique()
        total_dealers = rel["經銷商"].nunique() if not rel.empty else 0
        total_brands = brand_rel["品牌"].nunique() if not brand_rel.empty else 0
        
        # 基本統計資訊（文字顯示）
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**總資料筆數：** {total_records:,}")
        with col2:
            st.write(f"**關係連結數：** {len(rel) + len(brand_rel):,}")
        
        st.markdown("#### 各角色統計")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("建設公司", f"{total_developers:,}")
        
        with col2:
            st.metric("營造公司", f"{total_contractors:,}")
        
        with col3:
            st.metric("水電公司", f"{total_meps:,}")
        
        with col4:
            st.metric("經銷商", f"{total_dealers:,}")
        
        st.markdown("#### 線纜品牌數據")
        
        # 篩選控制項
        col1, col2 = st.columns(2)
        
        with col1:
            # 縣市篩選
            cities = sorted([city for city in df["縣市"].dropna().unique() if city])
            selected_city = st.selectbox("選擇縣市", ["全部"] + cities, key="city_filter")
        
        with col2:
            # 區域篩選 - 根據選擇的縣市動態更新
            if selected_city == "全部":
                areas = sorted([area for area in df["區域"].dropna().unique() if area])
            else:
                city_areas = df[df["縣市"] == selected_city]["區域"].dropna().unique()
                areas = sorted([area for area in city_areas if area])
            # 多選區域
            selected_areas = st.multiselect("選擇區域", areas, key="area_filter")
        
        # 根據篩選條件過濾品牌數據
        filtered_brand_rel = brand_rel.copy()
        filter_info = []
        
        if selected_city != "全部":
            filtered_brand_rel = filtered_brand_rel[filtered_brand_rel["縣市"] == selected_city]
            filter_info.append(f"縣市: {selected_city}")
        
        if selected_areas:
            filtered_brand_rel = filtered_brand_rel[filtered_brand_rel["區域"].isin(selected_areas)]
            filter_info.append(f"區域: {', '.join(selected_areas)}")
        
        # 計算篩選後的品牌統計
        filtered_total_brands = filtered_brand_rel["品牌"].nunique() if not filtered_brand_rel.empty else 0
        
        # 計算篩選後的金額總數
        filtered_total_amount = 0.0
        if not filtered_brand_rel.empty:
            unique_meps = filtered_brand_rel["水電公司"].unique()
            for mep in unique_meps:
                mep_volume = df[df["水電公司"] == mep]["年使用量_萬"].dropna()
                if len(mep_volume) > 0:
                    filtered_total_amount += float(mep_volume.iloc[0])
        
        # 顯示篩選後的統計 - 調整版面
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            display_title = "品牌總數"
            if filter_info:
                display_title += f" ({', '.join(filter_info)})"
            st.metric(display_title, f"{filtered_total_brands:,}")
        
        with col2:
            st.metric("年使用量總額", f"{filtered_total_amount:,.1f}萬")
        
        with col3:
            # 顯示篩選條件
            if filter_info:
                st.info(f"已篩選: {', '.join(filter_info)}")
            else:
                st.info("顯示全部地區數據")
        
        # 顯示品牌字卡
        if not filtered_brand_rel.empty:
            st.markdown("**各品牌數據分析**")
            
            # 計算篩選結果總數
            total_filtered_records = len(filtered_brand_rel)
            st.info(f"篩選結果：共 {total_filtered_records:,} 筆品牌配比記錄")
            
            # 計算各品牌的加權統計（去重水電）
            brand_stats = []
            total_market_volume = 0.0  # 用於計算占比
            
            # 先計算總市場量（去重水電）
            unique_meps = filtered_brand_rel["水電公司"].unique()
            for mep in unique_meps:
                mep_volume = df[df["水電公司"] == mep]["年使用量_萬"].dropna()
                if len(mep_volume) > 0:
                    total_market_volume += float(mep_volume.iloc[0])
            
            # 計算各品牌統計（每間水電只算一次）
            for brand_name in filtered_brand_rel["品牌"].unique():
                brand_data = filtered_brand_rel[filtered_brand_rel["品牌"] == brand_name]
                
                # 去重計算：每間水電只計算一次
                unique_meps_for_brand = brand_data["水電公司"].unique()
                total_weighted_volume = 0.0
                
                mep_details = []  # 新增此列表來儲存詳細資訊
                for mep in unique_meps_for_brand:
                    # 該水電對此品牌的平均配比
                    mep_brand_data = brand_data[brand_data["水電公司"] == mep]
                    avg_ratio = mep_brand_data["配比"].mean()
                    
                    # 該水電的年使用量
                    mep_volume = df[df["水電公司"] == mep]["年使用量_萬"].dropna()
                    volume = float(mep_volume.iloc[0]) if len(mep_volume) > 0 else 0.0
                    weighted_vol = volume * float(avg_ratio or 0.0)
                    
                    total_weighted_volume += weighted_vol
                    mep_details.append(f"**{mep}**: {weighted_vol:,.1f}萬 ({Formatters.pct_str(avg_ratio)})")
                
                # 計算占比
                market_share = (total_weighted_volume / total_market_volume) if total_market_volume > 0 else 0.0
                
                brand_stats.append({
                    "品牌": brand_name,
                    "合作水電數": len(unique_meps_for_brand),
                    "加權年使用量_萬": total_weighted_volume,
                    "市場占比": market_share,
                    "mep_details": mep_details # 將詳細資訊儲存到字典中
                })
            
            # 按加權年使用量排序
            brand_stats = sorted(brand_stats, key=lambda x: x["加權年使用量_萬"], reverse=True)
            
            # 創建品牌字卡 - 每行4個
            for i in range(0, len(brand_stats), 4):
                cols = st.columns(4)
                for j in range(4):
                    idx = i + j
                    if idx < len(brand_stats):
                        brand = brand_stats[idx]
                        with cols[j]:
                            volume_wan = brand["加權年使用量_萬"]
                            
                            # 使用自訂的 HTML 和 CSS 樣式
                            st.markdown(f"""
                            <div style="
                                padding: 16px;
                                border: 1px solid #e0e0e0;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                text-align: center;
                                min-height: 150px;
                                display: flex;
                                flex-direction: column;
                                justify-content: space-between;
                                background-color: #f9f9f9;
                            ">
                                <div style="font-size: 1.1rem; font-weight: bold; color: #333;">{brand['品牌']}</div>
                                <div style="font-size: 2.5rem; font-weight: bold; color: #4CAF50; margin-top: 5px;">{volume_wan:,.1f}萬</div>
                                <div style="font-size: 0.9rem; color: #666; margin-top: 10px;">
                                    市場佔比：{Formatters.pct_str(brand["市場占比"])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 恢復使用可展開區塊顯示詳細資訊
                            if brand['合作水電數'] > 0:
                                with st.expander(f"查看合作水電 ({brand['合作水電數']:,})"):
                                    for detail in brand["mep_details"]:
                                        st.markdown(detail, unsafe_allow_html=True)
        else:
            st.info("所選地區暫無線纜品牌數據")

    def _render_map_analysis(self, df: pd.DataFrame):
        """渲染地圖分析分頁"""
        st.markdown("### 台灣各區主要品牌地圖分析")
        
        @st.cache_data
        def load_geojson():
            # 來源: g0v/twgeojson on GitHub
            geojson_url = "https://raw.githubusercontent.com/g0v/twgeojson/master/json/twCounty2010.geo.json"
            try:
                response = requests.get(geojson_url, timeout=10)
                return response.json()
            except Exception as e:
                st.error(f"無法載入 GeoJSON 檔案，請檢查網路連線或 URL：{e}")
                return None
        
        @st.cache_data
        def process_brand_data(_df):
            """缓存品牌数据处理结果"""
            # 处理資料以找出每個區域的主導品牌
            brands_data = []
            # 遍歷所有品牌及其佔比欄位，並將其轉換為長格式
            for suffix in ['A', 'B', 'C']:
                brand_col = f'品牌{suffix}'
                ratio_col = f'品牌{suffix}比'
                if brand_col in _df.columns and ratio_col in _df.columns:
                    for _, row in _df.dropna(subset=[brand_col, ratio_col]).iterrows():
                        brand_name = str(row[brand_col]).strip()
                        ratio_val = float(row[ratio_col] or 0.0)
                        
                        # 過濾掉空值、0值和無效品牌名稱
                        if (brand_name and 
                            brand_name not in ['0', 'nan', 'None', ''] and 
                            ratio_val > 0 and 
                            pd.notna(row['縣市']) and 
                            pd.notna(row['區域'])):
                            
                            brands_data.append({
                                'city': str(row['縣市']).strip(),
                                'area': str(row['區域']).strip(),
                                'brand': brand_name,
                                'ratio': ratio_val
                            })
            
            df_brands = pd.DataFrame(brands_data)
            
            if df_brands.empty:
                return None, None, None
                
            # 按縣市聚合品牌資料
            city_brands = df_brands.groupby(['city', 'brand'])['ratio'].sum().reset_index()
            
            # 找到每個縣市最主要的品牌
            idx = city_brands.groupby(['city'])['ratio'].idxmax()
            df_dominant_brands = city_brands.loc[idx].reset_index(drop=True)
            
            # 計算每個縣市的總佔比，用於計算相對佔比
            city_totals = df_brands.groupby('city')['ratio'].sum().reset_index()
            city_totals.columns = ['city', 'total_ratio']
            df_dominant_brands = df_dominant_brands.merge(city_totals, on='city')
            df_dominant_brands['relative_ratio'] = df_dominant_brands['ratio'] / df_dominant_brands['total_ratio']
            
            # 創建每個縣市所有品牌的完整信息字典
            city_all_brands = {}
            for city in city_brands['city'].unique():
                city_data = city_brands[city_brands['city'] == city]
                total_city_ratio = city_data['ratio'].sum()
                
                # 計算每個品牌的相對佔比並排序
                brands_info = []
                for _, row in city_data.iterrows():
                    relative_ratio = row['ratio'] / total_city_ratio
                    brands_info.append({
                        'brand': row['brand'],
                        'ratio': relative_ratio,
                        'ratio_pct': f"{relative_ratio * 100:.1f}%"
                    })
                
                # 按佔比降序排列
                brands_info.sort(key=lambda x: x['ratio'], reverse=True)
                city_all_brands[city] = brands_info
            
            return df_brands, df_dominant_brands, city_all_brands
        
        # 加載地理數據
        geojson_data = load_geojson()
        if geojson_data is None:
            return

        # 處理品牌數據
        df_brands, df_dominant_brands, city_all_brands = process_brand_data(df)
        
        if df_brands is None or df_dominant_brands is None or city_all_brands is None:
            st.info("資料中沒有品牌資訊，無法產生地圖。")
            return

        # 準備 GeoJSON 數據，並為每個縣市添加「主導品牌」和「所有品牌」屬性
        for feature in geojson_data['features']:
            county_name = feature['properties'].get('COUNTYNAME', '')
            feature['properties']['city_name'] = county_name
            
            # 標準化縣市名稱（處理桃園縣->桃園市等變化）
            normalized_county = county_name.replace('縣', '市') if '縣' in county_name else county_name
            
            city_data = df_dominant_brands[
                (df_dominant_brands['city'] == county_name) | 
                (df_dominant_brands['city'] == normalized_county)
            ]
            
            # 查找該縣市在 city_all_brands 中的數據
            all_brands_info = None
            for city_key in [county_name, normalized_county]:
                if city_key in city_all_brands:
                    all_brands_info = city_all_brands[city_key]
                    break
            
            if not city_data.empty and all_brands_info:
                brand_info = city_data.iloc[0]
                feature['properties']['dominant_brand'] = brand_info['brand']
                # 使用相對佔比，並確保格式正確
                relative_pct = brand_info['relative_ratio'] * 100
                feature['properties']['dominant_ratio'] = f"{relative_pct:.1f}%"
                
                # 添加所有品牌的詳細信息
                brands_detail = []
                for brand_item in all_brands_info:
                    brands_detail.append(f"{brand_item['brand']}: {brand_item['ratio_pct']}")
                feature['properties']['all_brands'] = "<br/>".join(brands_detail)
                feature['properties']['brands_count'] = len(all_brands_info)
            else:
                feature['properties']['dominant_brand'] = "無資料"
                feature['properties']['dominant_ratio'] = "N/A"
                feature['properties']['all_brands'] = "無資料"
                feature['properties']['brands_count'] = 0

        # 設定品牌顏色映射
        unique_brands = sorted(df_dominant_brands['brand'].unique().tolist())
        color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F4A261', '#2A9D8F', '#E76F51']
        color_map = {}
        for i, brand in enumerate(unique_brands):
            color_map[brand] = color_palette[i % len(color_palette)]
        color_map['無資料'] = '#CCCCCC'

        # 創建 Folium 地圖
        m = folium.Map(
            location=[23.6, 120.9], 
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # 創建品牌到顏色的函數
        def style_function(feature):
            brand = feature['properties'].get('dominant_brand', '無資料')
            return {
                'fillColor': color_map.get(brand, '#CCCCCC'),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }

        # 添加區域圖層
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['COUNTYNAME', 'all_brands'],
                aliases=['縣市', '品牌佔比'],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: rgba(255, 255, 255, 0.95);
                    border: 2px solid #333;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    font-size: 13px;
                    font-family: Arial, sans-serif;
                    max-width: 280px;
                    padding: 8px;
                    line-height: 1.4;
                """
            )
        ).add_to(m)

        # 在 Streamlit 中顯示地圖
        st_folium(
            m, 
            width=700, 
            height=500,
            returned_objects=["last_clicked_object"],
            key=f"taiwan_brand_map_{id(df)}"  # 使用數據的id作為key，確保數據變化時重新渲染
        )
        
        # 在地圖下方顯示品牌圖例
        st.markdown("#### 品牌圖例")
        if len(unique_brands) > 0:
            legend_cols = st.columns(min(len(unique_brands), 6))  # 最多显示6列
            for i, brand in enumerate(unique_brands):
                with legend_cols[i % len(legend_cols)]:
                    st.markdown(f"<span style='color: {color_map[brand]}; font-size: 20px;'>●</span> {brand}", unsafe_allow_html=True)
    
    def _render_analysis_settings(self, df: pd.DataFrame, rel: pd.DataFrame, 
                                 brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """渲染分析設定區域"""
        
        # 使用 columns 將篩選區塊置中
        col_left, col_selector, col_right = st.columns([1, 2, 1])
        
        with col_selector:
            st.markdown("**選擇分析角色**")
            role_options = [
                ("🏢 建設公司", "建設公司"),
                ("🔨 營造公司", "營造公司"), 
                ("⚡ 水電公司", "水電公司"),
                ("🛒 經銷商", "經銷商")
            ]
            
            selected_role_display = st.selectbox(
                "角色類型",
                options=[display for display, _ in role_options],
                help="選擇要分析的角色類型",
                label_visibility="collapsed"
            )
            
            role = next(actual for display, actual in role_options if display == selected_role_display)
            
            st.markdown("**選擇目標公司**")
            
            if role == "建設公司":
                options = sorted(df["建設公司"].dropna().unique())
            elif role == "營造公司":
                options = sorted(df["營造公司"].dropna().unique())
            elif role == "水電公司":
                options = sorted(df["水電公司"].dropna().unique())
            else:
                options = sorted(rel["經銷商"].dropna().unique())
            
            search_term = st.text_input(
                "搜尋公司名稱", 
                placeholder="輸入關鍵字過濾公司列表...",
                help="支援模糊搜尋，輸入部分公司名稱即可",
                label_visibility="collapsed"
            )
            
            if search_term:
                filtered_options = [opt for opt in options 
                                    if search_term.lower() in str(opt).lower()]
                if not filtered_options:
                    st.warning(f"找不到包含 '{search_term}' 的公司")
                    filtered_options = options
            else:
                filtered_options = options
            
            if search_term and filtered_options:
                st.caption(f"找到 {len(filtered_options)} 家公司")
            
            target = st.selectbox(
                "目標公司", 
                filtered_options,
                help=f"從 {len(options)} 家{role}中選擇",
                label_visibility="collapsed"
            )
        
            if target:
                st.success(f"準備分析：{role} - {target}")
                
                # 將按鈕放在同一個窄欄位中
                if st.button(
                    "🚀 開始分析",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.show_analysis = True
                    st.session_state.analysis_role = role
                    st.session_state.analysis_target = target
            else:
                st.info("請選擇要分析的目標公司")

        # 分析結果顯示在主頁面寬版區塊
        if "show_analysis" in st.session_state and st.session_state.show_analysis:
            st.markdown("---")
            st.markdown("### 📈 分析結果")
            self.render_role_analysis(st.session_state.analysis_role, 
                                      st.session_state.analysis_target, 
                                      df, rel, brand_rel, mep_vol_map, df_raw)
    
    def _create_share_table(self, df: pd.DataFrame, group_cols: List[str], name_col: str) -> pd.DataFrame:
        """創建份額分析表格"""
        cnt = df.groupby(group_cols).size().reset_index(name="次數")
        total = cnt["次數"].sum()
        if total == 0:
            return pd.DataFrame(columns=[name_col, "次數", "占比"])
        
        cnt["占比"] = cnt["次數"] / total
        cnt["占比"] = cnt["占比"].apply(Formatters.pct_str)
        return cnt.sort_values("次數", ascending=False)
    
    def render_role_analysis(self, role: str, target: str, df: pd.DataFrame, 
                             rel: pd.DataFrame, brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """根據選擇的角色渲染分析結果"""
        analyzer = RelationshipAnalyzer(df, rel, brand_rel, mep_vol_map)
        comp_analyzer = CompetitorAnalyzer(df, rel, mep_vol_map)
        
        if role == "建設公司":
            self._render_developer_analysis(target, df, rel, analyzer, df_raw)
        elif role == "營造公司":
            self._render_contractor_analysis(target, df, rel, analyzer, df_raw)
        elif role == "水電公司":
            self._render_mep_analysis(target, df, rel, brand_rel, mep_vol_map, df_raw)
        elif role == "經銷商":
            self._render_dealer_analysis(target, df, rel, mep_vol_map, analyzer, comp_analyzer, df_raw)
    
    def _render_developer_analysis(self, target: str, df: pd.DataFrame, 
                                 rel: pd.DataFrame, analyzer: RelationshipAnalyzer, df_raw: pd.DataFrame):
        """渲染建設公司的分析結果"""
        df_sel = df[df["建設公司"] == target]
        rel_sel = rel[rel["建設公司"] == target]
        
        stats = {
            "資料筆數": len(df_sel),
            "營造家數": df_sel["營造公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        tab_overview, tab_partners, tab_export = st.tabs(["概覽", "合作對象視覺化", "資料匯出"])
        
        with tab_overview:
            self._render_developer_overview(df_sel, rel_sel, analyzer)
        
        with tab_partners:
            self._render_developer_visualizations(df_sel, rel_sel, analyzer)
        
        with tab_export:
            self._render_export_section(df_raw, df, rel, pd.DataFrame())
    
    def _render_developer_overview(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame, 
                                 analyzer: RelationshipAnalyzer):
        """渲染建設公司概覽"""
        UIComponents.render_section_header("合作夥伴概覽")
        
        st.markdown("**營造公司合作記錄**")
        contractor_stats = self._create_share_table(df_sel, ["營造公司"], "營造公司")
        contractor_stats = contractor_stats.rename(columns={"次數": "合作次數"})
        UIComponents.render_dataframe_with_styling(contractor_stats)
        
        st.markdown("**水電公司合作記錄**")
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        mep_stats = mep_stats.rename(columns={"次數": "合作次數"})
        UIComponents.render_dataframe_with_styling(mep_stats)
        
        st.markdown("**終端經銷商配比分析**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
        
        st.markdown("**線纜品牌配比分析（按使用量加權）**")
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            brand_analysis["加權平均配比"] = brand_analysis["加權平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(brand_analysis)
        else:
            UIComponents.render_info_box("暫無品牌配比資料")
    
    def _render_developer_visualizations(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame,
                                         analyzer: RelationshipAnalyzer):
        """渲染建設公司視覺化內容"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="dev_chart_type")
        
        contractor_stats = self._create_share_table(df_sel, ["營造公司"], "營造公司")
        if not contractor_stats.empty:
            fig = ChartGenerator.create_chart(
                contractor_stats, "營造公司", "次數", 
                "建設公司 → 營造公司合作分析", chart_type, key_suffix="dev_con"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            fig = ChartGenerator.create_chart(
                mep_stats, "水電公司", "次數",
                "建設公司 → 水電公司合作分析", chart_type, key_suffix="dev_mep"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            fig = ChartGenerator.create_chart(
                dealer_analysis, "經銷商", "平均配比",
                "建設公司 → 經銷商配比分析", chart_type, key_suffix="dev_dealer"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            fig = ChartGenerator.create_chart(
                brand_analysis, "品牌", "加權平均配比",
                "建設公司 → 線纜品牌配比分析（按使用量加權）", chart_type, key_suffix="dev_brand"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_contractor_analysis(self, target: str, df: pd.DataFrame, 
                                 rel: pd.DataFrame, analyzer: RelationshipAnalyzer, df_raw: pd.DataFrame):
        """渲染營造公司的分析結果"""
        df_sel = df[df["營造公司"] == target]
        rel_sel = rel[rel["營造公司"] == target]
        
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        tab_overview, tab_partners, tab_comp, tab_export = st.tabs(["概覽", "合作對象視覺化", "競爭者", "資料匯出"])
        
        with tab_overview:
            self._render_contractor_overview(df_sel, rel_sel, analyzer)
        
        with tab_partners:
            self._render_contractor_visualizations(df_sel, rel_sel, analyzer)
        
        with tab_comp:
            self._render_contractor_competitors(target, df)
            
        with tab_export:
            self._render_export_section(df_raw, df, rel, pd.DataFrame())
    
    def _render_contractor_overview(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame, 
                                 analyzer: RelationshipAnalyzer):
        """渲染營造公司概覽"""
        UIComponents.render_section_header("快速總覽")
        
        st.markdown("**上游建設公司**")
        dev_stats = self._create_share_table(df_sel, ["建設公司"], "建設公司")
        UIComponents.render_dataframe_with_styling(dev_stats)
        
        st.markdown("**合作水電公司**")
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        mep_stats = mep_stats.rename(columns={"次數": "合作次數"})
        UIComponents.render_dataframe_with_styling(mep_stats)
        
        st.markdown("**終端經銷商配比分析**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
        
        st.markdown("**線纜品牌配比分析（按使用量加權）**")
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            brand_analysis["加權平均配比"] = brand_analysis["加權平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(brand_analysis)
        else:
            UIComponents.render_info_box("暫無品牌配比資料")
    
    def _render_contractor_visualizations(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame,
                                          analyzer: RelationshipAnalyzer):
        """渲染營造公司視覺化內容"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="con_chart_type")
        
        dev_stats = self._create_share_table(df_sel, ["建設公司"], "建設公司")
        if not dev_stats.empty:
            fig = ChartGenerator.create_chart(
                dev_stats, "建設公司", "次數", 
                "營造公司 → 建設公司合作分析", chart_type, key_suffix="con_dev"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            fig = ChartGenerator.create_chart(
                mep_stats, "水電公司", "次數",
                "營造公司 → 水電公司合作分析", chart_type, key_suffix="con_mep"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            fig = ChartGenerator.create_chart(
                dealer_analysis, "經銷商", "平均配比",
                "營造公司 → 經銷商配比分析", chart_type, key_suffix="con_dealer"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            fig = ChartGenerator.create_chart(
                brand_analysis, "品牌", "加權平均配比",
                "營造公司 → 線纜品牌配比分析（按使用量加權）", chart_type, key_suffix="con_brand"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_contractor_competitors(self, target: str, df: pd.DataFrame):
        """渲染營造公司競爭者分析"""
        UIComponents.render_section_header("競爭者分析")
        
        devs = df[df["營造公司"] == target]["建設公司"].dropna().unique()
        if len(devs) == 0:
            UIComponents.render_info_box("無共同建設公司資料，無法進行競爭分析")
            return
        
        candidates = df[df["建設公司"].isin(devs)]
        competitors = (candidates[candidates["營造公司"] != target]
                       .groupby("營造公司").size()
                       .reset_index(name="共同出現次數")
                       .sort_values("共同出現次數", ascending=False))
        
        UIComponents.render_dataframe_with_styling(competitors, "競爭對手分析")
    
    def _render_mep_analysis(self, target: str, df: pd.DataFrame, rel: pd.DataFrame, 
                             brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """渲染水電公司的分析結果"""
        df_sel = df[df["水電公司"] == target]
        rel_sel = rel[rel["水電公司"] == target]
        
        mep_vol = df_sel["年使用量_萬"].dropna().unique()
        vol_val = float(mep_vol[0]) if len(mep_vol) > 0 and not pd.isna(mep_vol[0]) else 0.0
        
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "營造家數": df_sel["營造公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        tab_overview, tab_partners, tab_comp, tab_export = st.tabs(["概覽", "合作對象視覺化", "競爭者", "資料匯出"])
        
        with tab_overview:
            self._render_mep_overview(df_sel, rel_sel, brand_rel, target, vol_val)
        
        with tab_partners:
            self._render_mep_visualizations(rel_sel, brand_rel, target, vol_val)
        
        with tab_comp:
            self._render_mep_competitors(target, df)
            
        with tab_export:
            self._render_export_section(df_raw, df, rel, brand_rel)
    
    def _render_mep_overview(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame, 
                             brand_rel: pd.DataFrame, target: str, vol_val: float):
        """渲染水電公司概覽"""
        UIComponents.render_section_header("合作對象與品牌")
        
        if not rel_sel.empty:
            dealer_ratio = (rel_sel.groupby("經銷商")["配比"].mean()
                            .reset_index().sort_values("配比", ascending=False))
            dealer_ratio["額度_萬"] = dealer_ratio["配比"].astype(float) * vol_val
            
            dealer_display = dealer_ratio.copy()
            dealer_display["配比"] = dealer_display["配比"].apply(Formatters.pct_str)
            dealer_display["額度_萬"] = dealer_display["額度_萬"].round(2)
            dealer_display = dealer_display.rename(columns={"額度_萬": "額度(萬)"})
            
            st.markdown("**經銷商（配比與額度）**")
            UIComponents.render_dataframe_with_styling(dealer_display)
        else:
            st.markdown("**經銷商（配比與額度）**")
            UIComponents.render_info_box("暫無經銷商配比資料")
        
        if not brand_rel.empty:
            brand_sel = brand_rel[brand_rel["水電公司"] == target]
            if not brand_sel.empty:
                brand_ratio = (brand_sel.groupby("品牌")["配比"].mean()
                               .reset_index().sort_values("配比", ascending=False))
                brand_ratio["額度_萬"] = brand_ratio["配比"].astype(float) * vol_val
                
                brand_display = brand_ratio.copy()
                brand_display["配比"] = brand_display["配比"].apply(Formatters.pct_str)
                brand_display["額度_萬"] = brand_display["額度_萬"].round(2)
                brand_display = brand_display.rename(columns={"額度_萬": "額度(萬)"})
                
                st.markdown("**線纜品牌（配比與額度）**")
                UIComponents.render_dataframe_with_styling(brand_display)
            else:
                st.markdown("**線纜品牌（配比與額度）**")
                UIComponents.render_info_box("暫無品牌配比資料")
        
        memo = f"{vol_val} 萬" if vol_val > 0 else "—"
        UIComponents.render_info_box(f"預估年使用量：{memo}（已用於經銷商與品牌的金額換算）")
        
        st.markdown("**上游合作夥伴**")
        combined_partners = df_sel.assign(
            _公司=df_sel["建設公司"].fillna("") + " × " + df_sel["營造公司"].fillna("")
        )
        up_stats = self._create_share_table(combined_partners, ["_公司"], "公司")
        UIComponents.render_dataframe_with_styling(up_stats)
    
    def _render_mep_visualizations(self, rel_sel: pd.DataFrame, brand_rel: pd.DataFrame, 
                                 target: str, vol_val: float):
        """渲染水電公司視覺化內容"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="mep_chart_type")
        
        if not rel_sel.empty:
            dealer_ratio = (rel_sel.groupby("經銷商")["配比"].mean()
                            .reset_index().sort_values("配比", ascending=False))
            dealer_ratio["額度_萬"] = dealer_ratio["配比"].astype(float) * vol_val
            dealer_chart_data = dealer_ratio.rename(columns={"額度_萬": "金額(萬)"})
            
            fig = ChartGenerator.create_chart(
                dealer_chart_data, "經銷商", "金額(萬)",
                "水電公司 → 終端經銷商 金額(萬)", chart_type, key_suffix="mep_dealer"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        if not brand_rel.empty:
            brand_sel = brand_rel[brand_rel["水電公司"] == target]
            if not brand_sel.empty:
                brand_ratio = (brand_sel.groupby("品牌")["配比"].mean()
                               .reset_index().sort_values("配比", ascending=False))
                brand_ratio["額度_萬"] = brand_ratio["配比"].astype(float) * vol_val
                brand_chart_data = brand_ratio.rename(columns={"額度_萬": "金額(萬)"})
                
                fig = ChartGenerator.create_chart(
                    brand_chart_data, "品牌", "金額(萬)",
                    "水電公司 → 線纜品牌 金額(萬)", chart_type, key_suffix="mep_brand"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_mep_competitors(self, target: str, df: pd.DataFrame):
        """渲染水電公司競爭者分析"""
        UIComponents.render_section_header("競爭者分析")
        
        analyzer = CompetitorAnalyzer(df, pd.DataFrame(), {})
        competitors = analyzer.water_competitors(target)
        
        if competitors.empty:
            UIComponents.render_info_box("暫無競爭者資料")
        else:
            UIComponents.render_dataframe_with_styling(competitors, "競爭對手分析")
    
    def _render_dealer_analysis(self, target: str, df: pd.DataFrame, rel: pd.DataFrame,
                                 mep_vol_map: Dict, analyzer: RelationshipAnalyzer, 
                                 comp_analyzer: CompetitorAnalyzer, df_raw: pd.DataFrame):
        """渲染經銷商的分析結果"""
        df_sel = rel[rel["經銷商"] == target].merge(
            df, on=["建設公司", "營造公司", "水電公司"], how="left", suffixes=("", "_df")
        )
        
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "營造家數": df_sel["營造公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique()
        }
        UIComponents.render_kpi_section(stats)
        
        tab_overview, tab_partners, tab_comp, tab_export = st.tabs(["概覽", "合作對象視覺化", "競爭者", "資料匯出"])
        
        with tab_overview:
            self._render_dealer_overview(df_sel, rel, target)
        
        with tab_partners:
            self._render_dealer_visualizations(df_sel)
        
        with tab_comp:
            self._render_dealer_competitors(target, rel, mep_vol_map, analyzer, comp_analyzer)
            
        with tab_export:
            self._render_export_section(df_raw, df, rel, pd.DataFrame())
    
    def _render_dealer_overview(self, df_sel: pd.DataFrame, rel: pd.DataFrame, target: str):
        """渲染經銷商概覽"""
        UIComponents.render_section_header("合作水電")
        
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        
        ratio_df = (rel[rel["經銷商"] == target]
                    .groupby("水電公司")["配比"].mean()
                    .reset_index()
                    .rename(columns={"配比": "該經銷商配比"}))
        
        if not ratio_df.empty:
            ratio_df["該經銷商配比"] = ratio_df["該經銷商配比"].apply(Formatters.pct_str)
            mep_stats = mep_stats.merge(ratio_df, on="水電公司", how="left")
        
        UIComponents.render_dataframe_with_styling(mep_stats)
    
    def _render_dealer_visualizations(self, df_sel: pd.DataFrame):
        """渲染經銷商視覺化內容"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="dealer_chart_type")
        top_n = st.selectbox("顯示前幾大", [5, 10, 15, 20, "全部"], index=0, key="dealer_top_n_select")
        
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            display_data = mep_stats.head(top_n) if top_n != "全部" else mep_stats
            fig = ChartGenerator.create_chart(
                display_data, "水電公司", "次數",
                f"經銷商 → 水電公司 合作次數 (前{top_n if top_n != '全部' else len(display_data)}大)", chart_type, key_suffix="dealer_mep"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_dealer_competitors(self, target: str, rel: pd.DataFrame, mep_vol_map: Dict,
                                 analyzer: RelationshipAnalyzer, comp_analyzer: CompetitorAnalyzer):
        """渲染經銷商競爭者分析"""
        UIComponents.render_section_header("競爭者分析")
        
        union_share, total_target = analyzer.union_overlap_share_and_total(target)
        comp_df, target_total_market = comp_analyzer.dealer_competitors(target)
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("競爭覆蓋率（去重）", Formatters.pct_str(union_share))
        with cols[1]:
            st.metric("總市場額度(萬)", Formatters.fmt_amount(total_target))
        
        if comp_df.empty:
            UIComponents.render_info_box("暫無競爭者資料")
        else:
            UIComponents.render_dataframe_with_styling(comp_df, "詳細競爭分析")
            st.caption("說明：表格中的「重疊市場占比」為與單一對手的配對式重疊（加總可能 >100%）；上方的「競爭覆蓋率（去重）」為所有對手合併後的覆蓋比例（不會超過 100%）。")
    
    def _render_export_section(self, df_raw: pd.DataFrame, df: pd.DataFrame, 
                               rel: pd.DataFrame, brand_rel: pd.DataFrame):
        """渲染資料匯出區塊"""
        UIComponents.render_section_header("資料匯出")
        
        st.write("**匯出說明**")
        st.write("• 原始資料: 上傳的原始檔案內容")
        st.write("• 主檔: 經過欄位標準化的主要資料")
        st.write("• 關係明細_經銷: 經銷商配比關係展開資料")
        st.write("• 關係明細_品牌: 品牌配比關係展開資料")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_raw.to_excel(writer, index=False, sheet_name="原始資料")
            df.to_excel(writer, index=False, sheet_name="主檔(標準化)")
            rel.to_excel(writer, index=False, sheet_name="關係明細_經銷(配比)")
            if not brand_rel.empty:
                brand_rel.to_excel(writer, index=False, sheet_name="關係明細_品牌(配比)")
        
        st.download_button(
            label="📥 下載 Excel 分析報告",
            data=output.getvalue(),
            file_name=f"construction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ====================== 應用程式進入點 ======================
def main():
    """應用程式主要進入點"""
    # 初始化 session state，用於控制分析結果的顯示
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
        st.session_state.analysis_role = None
        st.session_state.analysis_target = None
        
    try:
        dashboard = ConstructionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"應用程式發生錯誤：{str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()

