# app.py - 百大建商｜關係鏈分析（單頁搜尋 v12 Enhanced）
"""
Enhanced Construction Supply Chain Analysis Dashboard
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

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ====================== Configuration ======================
class Config:
    """Application configuration and constants"""
    APP_TITLE = "百大建商｜關係鏈分析（單頁搜尋 v12 Enhanced）"
    VERSION = "v12 Enhanced"
    ROLES = ["建設公司", "營造公司", "水電公司", "經銷商"]
    CHART_TYPES = ["圓餅圖", "長條圖"]
    COLOR_PALETTE = px.colors.qualitative.Set3
    
    # Column mappings (0-based positions)
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
        'city': (19, ["縣市", "縣/市", "所在縣市"]),
        'area': (20, ["區域", "地區", "區/鄉鎮"])
    }

# ====================== Data Processing Classes ======================
class DataProcessor:
    """Handle data processing and transformation"""
    
    @staticmethod
    def clean_name(x) -> Optional[str]:
        """Clean and standardize names"""
        if pd.isna(x):
            return None
        s = str(x).replace("\u3000", " ").strip()
        s = re.sub(r"\s+", " ", s)
        if s == "" or s.lower() in {"nan", "none"} or s == "0":
            return None
        return s
    
    @staticmethod
    def coerce_num(s) -> float:
        """Convert string to number with error handling"""
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
        """Normalize ratio values to 0-1 range"""
        s = series.apply(DataProcessor.coerce_num)
        max_val = s.max(skipna=True)
        if max_val is not None and max_val > 1.000001:
            return s / 100.0
        return s
    
    @staticmethod
    def get_col_by_pos_or_name(df: pd.DataFrame, pos: int, name_candidates: List[str]) -> Optional[str]:
        """Get column by position or name with fallback"""
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

# ====================== Analysis Classes ======================
class RelationshipAnalyzer:
    """Analyze relationships between entities"""
    
    def __init__(self, df: pd.DataFrame, rel: pd.DataFrame, brand_rel: pd.DataFrame, mep_vol_map: Dict):
        self.df = df
        self.rel = rel
        self.brand_rel = brand_rel
        self.mep_vol_map = mep_vol_map
    
    def avg_dealer_ratio_across_unique_mep(self, rel_subset: pd.DataFrame) -> pd.DataFrame:
        """Calculate average dealer ratio across unique MEP companies"""
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
        """Calculate weighted average brand ratio across unique MEP companies"""
        if self.brand_rel.empty:
            return pd.DataFrame(columns=["品牌", "加權平均配比"])
        
        # 取得該建設/營造公司相關的水電公司
        meps = [m for m in df_subset["水電公司"].dropna().unique() if isinstance(m, str) and m != ""]
        if len(meps) == 0:
            return pd.DataFrame(columns=["品牌", "加權平均配比"])
        
        # 按年使用量加權計算各品牌配比
        brand_weighted_sums = defaultdict(float)
        total_volume = 0.0
        
        for mep in meps:
            # 取得該水電公司的年使用量
            mep_volume = float(self.mep_vol_map.get(mep, 0.0) or 0.0)
            if mep_volume <= 0:
                mep_volume = 1.0  # 預設值
            
            # 取得該水電公司的品牌配比
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
        """Calculate union overlap share and total market for dealer"""
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
    """Analyze competitors and market competition"""
    
    def __init__(self, df: pd.DataFrame, rel: pd.DataFrame, mep_vol_map: Dict):
        self.df = df
        self.rel = rel
        self.mep_vol_map = mep_vol_map
    
    def water_competitors(self, target_mep: str) -> pd.DataFrame:
        """Find competitors for water/MEP companies"""
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
        """Find competitors for dealers with detailed analysis"""
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

# ====================== Formatters ======================
class Formatters:
    """Format data for display"""
    
    @staticmethod
    def pct_str(x) -> str:
        """Format percentage with proper rounding"""
        if pd.isna(x):
            return "-"
        v = float(x)
        if v <= 1.0:
            v = v * 100.0
        d = Decimal(str(v)).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        return f"{d}%"
    
    @staticmethod
    def fmt_amount(x) -> str:
        """Format amount with thousands separator"""
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):,.2f}"

# ====================== Visualization ======================
class ChartGenerator:
    """Generate charts and visualizations"""
    
    @staticmethod
    def create_chart(df_plot: pd.DataFrame, name_col: str, value_col: str, 
                    title: str, chart_type: str = "圓餅圖") -> go.Figure:
        """Create enhanced charts with better styling"""
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

# ====================== UI Components ======================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_kpi_section(stats: Dict[str, int]):
        """Render KPI section using native Streamlit metrics"""
        cols = st.columns(len(stats))
        for i, (label, value) in enumerate(stats.items()):
            with cols[i]:
                st.metric(label=label, value=f"{value:,}")
    
    @staticmethod
    def render_section_header(title: str):
        """Render section header with consistent styling"""
        st.markdown(f"**{title}**")
    
    @staticmethod
    def render_info_box(message: str):
        """Render info box with enhanced styling"""
        st.info(message)
    
    @staticmethod
    def render_dataframe_with_styling(df: pd.DataFrame, title: str = None):
        """Render dataframe with enhanced styling and formatting"""
        if title:
            st.markdown(f"**{title}**")
        
        if df.empty:
            UIComponents.render_info_box("暫無資料")
        else:
            # 格式化數據框
            df_styled = df.copy()
            
            # 格式化數字列
            numeric_columns = df_styled.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df_styled.columns:
                    if df_styled[col].dtype in ['int64', 'int32']:
                        df_styled[col] = df_styled[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "-")
                    else:
                        df_styled[col] = df_styled[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
            
            # 自定義CSS樣式
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
            
            # 使用column_config控制列寬
            column_config = {}
            for col in df_styled.columns:
                if col == df_styled.columns[0]:
                    column_config[col] = st.column_config.TextColumn(col, width="medium")
                else:
                    column_config[col] = st.column_config.TextColumn(col, width="small")
            
            st.dataframe(
                df_styled, 
                use_container_width=False,
                hide_index=True,
                column_config=column_config
            )

# ====================== Main Application ======================
class ConstructionDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.config = Config()
        self.setup_page()
        
    def setup_page(self):
        """Setup page configuration"""
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
        """Read uploaded file with caching"""
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
        """Process raw data into analysis-ready format"""
        processor = DataProcessor()
        
        # Get columns using mapping
        columns = {}
        for key, (pos, names) in self.config.COLUMN_MAPPING.items():
            columns[key] = processor.get_col_by_pos_or_name(df_raw, pos, names)
        
        # Validate required columns
        required_cols = [columns['dev'], columns['con'], columns['mep']]
        if any(col is None for col in required_cols):
            st.error("找不到必要欄位（建設公司/營造公司/水電公司）。請確認資料格式。")
            st.stop()
        
        # Create rename mapping
        rename_map = {
            columns['dev']: "建設公司",
            columns['con']: "營造公司", 
            columns['mep']: "水電公司",
            columns['vol']: "年使用量_萬",
        }
        
        # Add dealer and brand columns
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
        
        # Add geographical columns
        if columns.get('city'):
            rename_map[columns['city']] = "縣市"
        if columns.get('area'):
            rename_map[columns['area']] = "區域"
        
        # Apply transformations
        df = df_raw.rename(columns=rename_map).copy()
        
        # Clean text columns
        text_cols = ["建設公司", "營造公司", "水電公司", "縣市", "區域"] + [f"經銷商{s}" for s in ['A','B','C']] + [f"品牌{s}" for s in ['A','B','C']]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(processor.clean_name)
        
        # Process numeric columns
        if "年使用量_萬" in df.columns:
            df["年使用量_萬"] = df["年使用量_萬"].apply(processor.coerce_num)
        
        # Process ratio columns
        ratio_cols = [f"經銷{s}比" for s in ['A','B','C']] + [f"品牌{s}比" for s in ['A','B','C']]
        for col in ratio_cols:
            if col in df.columns:
                df[col] = processor.normalize_ratio(df[col])
        
        # Create relationship dataframes
        rel = self._create_dealer_relations(df)
        brand_rel = self._create_brand_relations(df)
        
        # Create MEP volume mapping
        mep_vol_map = df.groupby("水電公司")["年使用量_萬"].apply(
            lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else np.nan
        ).to_dict()
        
        return df, rel, brand_rel, mep_vol_map
    
    def _create_dealer_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dealer relationship dataframe"""
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
        """Create brand relationship dataframe"""
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
        """Run the main application"""
        # File upload with clean instructions
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
                st.write("• T=縣市, U=區域")
                st.write("")
                st.write("**支援分析角色：**")
                st.write("• 建設公司：查看營造、水電、經銷商合作關係")
                st.write("• 營造公司：分析上下游合作網絡及競爭態勢")
                st.write("• 水電公司：經銷商配比、品牌使用分析")
                st.write("• 經銷商：客戶分布、市場競爭分析")
            st.stop()
        
        # Process data
        df_raw = self.read_file(uploaded_file)
        df, rel, brand_rel, mep_vol_map = self.process_data(df_raw)
        
        # 主要分頁設定
        tab_overview, tab_analysis = st.tabs(["📊 數據概覽", "🎯 分析設定"])
        
        with tab_overview:
            self._render_overall_statistics(df, rel, brand_rel)
        
        with tab_analysis:
            self._render_analysis_settings(df, rel, brand_rel, mep_vol_map, df_raw)
    
    def _render_overall_statistics(self, df: pd.DataFrame, rel: pd.DataFrame, brand_rel: pd.DataFrame):
        """渲染整體統計數據"""
        # 添加CSS樣式為指標卡片創建有色邊框
        st.markdown("""
            <style>
            /* 為指標卡片添加淡色邊框 */
            .stMetric {
                background-color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid transparent;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            }
            
            /* 不同顏色的邊框 */
            [data-testid="metric-container"]:nth-child(1) .stMetric {
                border-left-color: #87CEEB; /* 淡藍色 */
                background-color: #F0F8FF;
            }
            [data-testid="metric-container"]:nth-child(2) .stMetric {
                border-left-color: #DDA0DD; /* 淡紫色 */
                background-color: #F8F0FF;
            }
            [data-testid="metric-container"]:nth-child(3) .stMetric {
                border-left-color: #98FB98; /* 淡綠色 */
                background-color: #F0FFF0;
            }
            [data-testid="metric-container"]:nth-child(4) .stMetric {
                border-left-color: #FFB6C1; /* 淡粉色 */
                background-color: #FFF0F5;
            }
            [data-testid="metric-container"]:nth-child(5) .stMetric {
                border-left-color: #F0E68C; /* 淡黃色 */
                background-color: #FFFACD;
            }
            [data-testid="metric-container"]:nth-child(6) .stMetric {
                border-left-color: #FFA07A; /* 淡橘色 */
                background-color: #FFF8DC;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # 主要統計指標
        total_records = len(df)
        total_developers = df["建設公司"].nunique()
        total_contractors = df["營造公司"].nunique()
        total_meps = df["水電公司"].nunique()
        total_dealers = rel["經銷商"].nunique() if not rel.empty else 0
        total_brands = brand_rel["品牌"].nunique() if not brand_rel.empty else 0
        
        # 創建統計卡片 - 使用原生metric組件
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("總資料筆數", f"{total_records:,}")
        
        with col2:
            st.metric("關係連結數", f"{len(rel) + len(brand_rel):,}")
        
        # 各角色統計
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
        
        # 品牌數據分析
        st.markdown("#### 品牌數據分析")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("品牌總數", f"{total_brands:,}")
        
        with col2:
            # 計算品牌使用密度（平均每家水電使用多少品牌）
            if not brand_rel.empty and total_meps > 0:
                brand_density = len(brand_rel) / total_meps
                st.metric("品牌使用密度", f"{brand_density:.1f}/家")
            else:
                st.metric("品牌使用密度", "—")
    
    def _render_analysis_settings(self, df: pd.DataFrame, rel: pd.DataFrame, 
                                brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """渲染分析設定區域"""
        
        # 第一行：角色選擇（居中）
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
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
        
        st.markdown("")  # 空行分隔
        
        # 第二行：公司選擇（居中）
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("**選擇目標公司**")
            
            # 根據選擇的角色獲取選項
            if role == "建設公司":
                options = sorted(df["建設公司"].dropna().unique())
            elif role == "營造公司":
                options = sorted(df["營造公司"].dropna().unique())
            elif role == "水電公司":
                options = sorted(df["水電公司"].dropna().unique())
            else:  # 經銷商
                options = sorted(rel["經銷商"].dropna().unique())
            
            # 搜尋功能
            search_term = st.text_input(
                "搜尋公司名稱", 
                placeholder="輸入關鍵字過濾公司列表...",
                help="支援模糊搜尋，輸入部分公司名稱即可",
                label_visibility="collapsed"
            )
            
            # 過濾選項
            if search_term:
                filtered_options = [opt for opt in options 
                                  if search_term.lower() in str(opt).lower()]
                if not filtered_options:
                    st.warning(f"找不到包含 '{search_term}' 的公司")
                    filtered_options = options
            else:
                filtered_options = options
            
            # 顯示找到的數量
            if search_term and filtered_options:
                st.caption(f"找到 {len(filtered_options)} 家公司")
            
            target = st.selectbox(
                "目標公司", 
                filtered_options,
                help=f"從 {len(options)} 家{role}中選擇",
                label_visibility="collapsed"
            )
        
        # 第三行：分析按鈕和狀態（居中）
        if target:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.success(f"準備分析：{role} - {target}")
                
                if st.button(
                    "🚀 開始深度分析",
                    type="primary",
                    use_container_width=True
                ):
                    # 分析結果區域
                    st.markdown("---")
                    st.markdown("### 📈 分析結果")
                    
                    # 執行分析
                    self.render_role_analysis(role, target, df, rel, brand_rel, mep_vol_map, df_raw)
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info("請選擇要分析的目標公司")
    
    def _create_share_table(self, df: pd.DataFrame, group_cols: List[str], name_col: str) -> pd.DataFrame:
        """Create share analysis table"""
        cnt = df.groupby(group_cols).size().reset_index(name="次數")
        total = cnt["次數"].sum()
        if total == 0:
            return pd.DataFrame(columns=[name_col, "次數", "占比"])
        
        cnt["占比"] = cnt["次數"] / total
        cnt["占比"] = cnt["占比"].apply(Formatters.pct_str)
        return cnt.sort_values("次數", ascending=False)
    
    def render_role_analysis(self, role: str, target: str, df: pd.DataFrame, 
                           rel: pd.DataFrame, brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """Render analysis based on selected role"""
        analyzer = RelationshipAnalyzer(df, rel, brand_rel, mep_vol_map)
        comp_analyzer = CompetitorAnalyzer(df, rel, mep_vol_map)
        
        # Role-specific analysis  
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
        """Render analysis for developers"""
        df_sel = df[df["建設公司"] == target]
        rel_sel = rel[rel["建設公司"] == target]
        
        # KPI metrics
        stats = {
            "資料筆數": len(df_sel),
            "營造家數": df_sel["營造公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        # Analysis tabs
        tab_overview, tab_partners, tab_export = st.tabs(["概覽", "合作對象視覺化", "資料匯出"])
        
        with tab_overview:
            self._render_developer_overview(df_sel, rel_sel, analyzer)
        
        with tab_partners:
            self._render_developer_visualizations(df_sel, rel_sel, analyzer)
        
        with tab_export:
            self._render_export_section(df_raw, df, rel, pd.DataFrame())
    
    def _render_developer_overview(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame, 
                                 analyzer: RelationshipAnalyzer):
        """Render developer overview"""
        UIComponents.render_section_header("合作夥伴概覽")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**營造公司合作記錄**")
            contractor_stats = self._create_share_table(df_sel, ["營造公司"], "營造公司")
            contractor_stats = contractor_stats.rename(columns={"次數": "合作次數"})
            UIComponents.render_dataframe_with_styling(contractor_stats)
        
        with col2:
            st.markdown("**水電公司合作記錄**")
            mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
            mep_stats = mep_stats.rename(columns={"次數": "合作次數"})
            UIComponents.render_dataframe_with_styling(mep_stats)
        
        # 經銷商分析
        st.markdown("**終端經銷商配比分析**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
        
        # 品牌分析
        st.markdown("**線纜品牌配比分析（按使用量加權）**")
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            brand_analysis["加權平均配比"] = brand_analysis["加權平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(brand_analysis)
        else:
            UIComponents.render_info_box("暫無品牌配比資料")
    
    def _render_developer_visualizations(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame,
                                       analyzer: RelationshipAnalyzer):
        """Render developer visualizations"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="dev_chart")
        
        # Contractor chart
        contractor_stats = self._create_share_table(df_sel, ["營造公司"], "營造公司")
        if not contractor_stats.empty:
            fig = ChartGenerator.create_chart(
                contractor_stats, "營造公司", "次數", 
                f"建設公司 → 營造公司合作分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # MEP chart
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            fig = ChartGenerator.create_chart(
                mep_stats, "水電公司", "次數",
                f"建設公司 → 水電公司合作分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Dealer chart
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            fig = ChartGenerator.create_chart(
                dealer_analysis, "經銷商", "平均配比",
                f"建設公司 → 經銷商配比分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Brand chart
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            fig = ChartGenerator.create_chart(
                brand_analysis, "品牌", "加權平均配比",
                f"建設公司 → 線纜品牌配比分析（按使用量加權）", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_contractor_analysis(self, target: str, df: pd.DataFrame, 
                                  rel: pd.DataFrame, analyzer: RelationshipAnalyzer, df_raw: pd.DataFrame):
        """Render analysis for contractors"""
        df_sel = df[df["營造公司"] == target]
        rel_sel = rel[rel["營造公司"] == target]
        
        # KPI metrics
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        # Analysis tabs
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
        """Render contractor overview"""
        UIComponents.render_section_header("快速總覽")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**上游建設公司**")
            dev_stats = self._create_share_table(df_sel, ["建設公司"], "建設公司")
            UIComponents.render_dataframe_with_styling(dev_stats)
        
        with col2:
            st.markdown("**合作水電公司**")
            mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
            mep_stats = mep_stats.rename(columns={"次數": "合作次數"})
            UIComponents.render_dataframe_with_styling(mep_stats)
        
        # 經銷商分析
        st.markdown("**終端經銷商（平均配比｜按水電等權）**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
        
        # 品牌分析
        st.markdown("**線纜品牌配比分析（按使用量加權）**")
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            brand_analysis["加權平均配比"] = brand_analysis["加權平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(brand_analysis)
        else:
            UIComponents.render_info_box("暫無品牌配比資料")
    
    def _render_contractor_visualizations(self, df_sel: pd.DataFrame, rel_sel: pd.DataFrame,
                                        analyzer: RelationshipAnalyzer):
        """Render contractor visualizations"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="con_chart")
        
        # Developer chart
        dev_stats = self._create_share_table(df_sel, ["建設公司"], "建設公司")
        if not dev_stats.empty:
            fig = ChartGenerator.create_chart(
                dev_stats, "建設公司", "次數", 
                f"營造公司 → 建設公司合作分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # MEP chart
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            fig = ChartGenerator.create_chart(
                mep_stats, "水電公司", "次數",
                f"營造公司 → 水電公司合作分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Dealer chart
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            fig = ChartGenerator.create_chart(
                dealer_analysis, "經銷商", "平均配比",
                f"營造公司 → 經銷商配比分析", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Brand chart
        brand_analysis = analyzer.avg_brand_ratio_across_unique_mep(df_sel)
        if not brand_analysis.empty:
            fig = ChartGenerator.create_chart(
                brand_analysis, "品牌", "加權平均配比",
                f"營造公司 → 線纜品牌配比分析（按使用量加權）", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_contractor_competitors(self, target: str, df: pd.DataFrame):
        """Render contractor competitors analysis"""
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
        """Render analysis for MEP companies"""
        df_sel = df[df["水電公司"] == target]
        rel_sel = rel[rel["水電公司"] == target]
        
        # Get MEP volume
        mep_vol = df_sel["年使用量_萬"].dropna().unique()
        vol_val = float(mep_vol[0]) if len(mep_vol) > 0 and not pd.isna(mep_vol[0]) else 0.0
        
        # KPI metrics
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "營造家數": df_sel["營造公司"].nunique(),
            "經銷家數": rel_sel["經銷商"].nunique() if not rel_sel.empty else 0
        }
        UIComponents.render_kpi_section(stats)
        
        # Analysis tabs
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
        """Render MEP overview"""
        UIComponents.render_section_header("合作對象與品牌")
        
        # Dealer analysis (ratio × volume)
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
        
        # Brand analysis (ratio × volume)
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
        
        # Volume info
        memo = f"{vol_val} 萬" if vol_val > 0 else "—"
        UIComponents.render_info_box(f"預估年使用量：{memo}（已用於經銷商與品牌的金額換算）")
        
        # Upstream analysis
        st.markdown("**上游合作夥伴**")
        combined_partners = df_sel.assign(
            _公司=df_sel["建設公司"].fillna("") + " × " + df_sel["營造公司"].fillna("")
        )
        up_stats = self._create_share_table(combined_partners, ["_公司"], "公司")
        UIComponents.render_dataframe_with_styling(up_stats)
    
    def _render_mep_visualizations(self, rel_sel: pd.DataFrame, brand_rel: pd.DataFrame, 
                                 target: str, vol_val: float):
        """Render MEP visualizations"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="mep_chart")
