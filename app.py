# app.py - 百大建商｜關係鏈分析（單頁搜尋 v12 Enhanced）
"""
Enhanced Construction Supply Chain Analysis Dashboard
Improvements:
- Better code organization with classes and type hints
- Enhanced UI/UX with improved styling and layout
- Better error handling and validation
- More professional appearance with consistent design
- Improved performance and maintainability
- All original logic preserved
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
        'brand_ratio_c': (18, ["品牌C佔比(%)", "品牌C配比", "品牌3佔比", "C品牌佔比", "C品牌配比"])
    }

# ====================== Styling ======================
def load_custom_css():
    """Load custom CSS for enhanced UI"""
    st.markdown("""
        <style>
        /* Main container styling */
        .main-container {
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Enhanced chip styling */
        .chip {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 13px;
            font-weight: 600;
            margin: 2px 6px 2px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Enhanced card styling */
        .card {
            padding: 24px;
            border-radius: 16px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
            transition: box-shadow 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* KPI metrics styling */
        .kpi-container {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }
        
        .kpi-card {
            flex: 1;
            min-width: 200px;
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border: 1px solid rgba(0, 0, 0, 0.08);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        
        .kpi-label {
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            color: #1f2937;
            line-height: 1;
        }
        
        /* Enhanced metric styling */
        .metric-row {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        
        .metric-big {
            flex: 1;
            min-width: 280px;
            padding: 24px;
            border-radius: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        
        .metric-big:hover {
            transform: translateY(-4px);
        }
        
        .metric-big .label {
            font-size: 16px;
            opacity: 0.9;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .metric-big .value {
            font-size: 36px;
            font-weight: 800;
        }
        
        .metric-big.gray {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: #374151;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        
        /* Enhanced tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #f8fafc;
            padding: 8px;
            border-radius: 12px;
            margin-bottom: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border: 1px solid #e2e8f0;
            padding: 12px 20px;
            border-radius: 10px;
            background: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-color: transparent !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Section headers */
        .section-header {
            color: #1f2937;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e5e7eb;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
            border: 1px solid #81d4fa;
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            color: #01579b;
            font-weight: 500;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button enhancements */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)

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

class DataValidator:
    """Validate data quality and completeness"""
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
        """Validate that required columns exist"""
        missing = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
        """Validate overall data quality"""
        return {
            'total_rows': len(df),
            'empty_rows': df.isnull().all(axis=1).sum(),
            'duplicate_rows': df.duplicated().sum(),
            'completeness': (df.notna().sum() / len(df) * 100).round(2).to_dict()
        }

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
            
        # Sort by threat level and metrics
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
    
    @staticmethod
    def format_large_number(num: int) -> str:
        """Format large numbers with appropriate units"""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        return f"{num:,}"

# ====================== Visualization ======================
class ChartGenerator:
    """Generate charts and visualizations"""
    
    @staticmethod
    def create_chart(df_plot: pd.DataFrame, name_col: str, value_col: str, 
                    title: str, chart_type: str = "圓餅圖") -> go.Figure:
        """Create enhanced charts with better styling"""
        if df_plot is None or df_plot.empty:
            return None
        
        # Enhanced color palette
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        
        if chart_type == "長條圖":
            fig = px.bar(
                df_plot, x=name_col, y=value_col, title=title,
                color=name_col, color_discrete_sequence=colors,
                template="plotly_white"
            )
            
            # Enhanced bar chart styling
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
            
            # Enhanced pie chart styling
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
    
    @staticmethod
    def create_kpi_card(label: str, value: str, is_primary: bool = True) -> str:
        """Create enhanced KPI card HTML"""
        card_class = "kpi-card" if not is_primary else "kpi-card primary"
        return f"""
        <div class="{card_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """

# ====================== UI Components ======================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_kpi_section(stats: Dict[str, int]):
        """Render KPI section with enhanced styling"""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        kpi_html = '<div class="kpi-container">'
        for label, value in stats.items():
            kpi_html += ChartGenerator.create_kpi_card(label, f"{value:,}")
        kpi_html += '</div>'
        
        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_section_header(title: str):
        """Render section header with consistent styling"""
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_info_box(message: str):
        """Render info box with enhanced styling"""
        st.markdown(f'<div class="info-box">{message}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_dataframe_with_styling(df: pd.DataFrame, title: str = None):
        """Render dataframe with enhanced styling"""
        if title:
            st.markdown(f"**{title}**")
        
        if df.empty:
            UIComponents.render_info_box("暫無資料")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

# ====================== Main Application ======================
class ConstructionDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.config = Config()
        self.setup_page()
        load_custom_css()
        
    def setup_page(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🏗️"
        )
        
        # Header with version info
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
        
        # Add dealer columns
        for suffix in ['a', 'b', 'c']:
            dealer_key = f'dealer_{suffix}'
            ratio_key = f'ratio_{suffix}'
            if columns.get(dealer_key):
                rename_map[columns[dealer_key]] = f"經銷商{suffix.upper()}"
            if columns.get(ratio_key):
                rename_map[columns[ratio_key]] = f"經銷{suffix.upper()}比"
        
        # Add brand columns
        for suffix in ['a', 'b', 'c']:
            brand_key = f'brand_{suffix}'
            brand_ratio_key = f'brand_ratio_{suffix}'
            if columns.get(brand_key):
                rename_map[columns[brand_key]] = f"品牌{suffix.upper()}"
            if columns.get(brand_ratio_key):
                rename_map[columns[brand_ratio_key]] = f"品牌{suffix.upper()}比"
        
        # Apply transformations
        df = df_raw.rename(columns=rename_map).copy()
        
        # Clean text columns
        text_cols = ["建設公司", "營造公司", "水電公司"] + [f"經銷商{s}" for s in ['A','B','C']] + [f"品牌{s}" for s in ['A','B','C']]
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
                block = df[["建設公司", "營造公司", "水電公司", dealer_col, ratio_col]].rename(
                    columns={dealer_col: "經銷商", ratio_col: "配比"}
                )
                blocks.append(block)
        
        if blocks:
            rel = pd.concat(blocks, ignore_index=True)
            rel["經銷商"] = rel["經銷商"].apply(DataProcessor.clean_name)
            return rel.dropna(subset=["經銷商", "水電公司"])
        
        return pd.DataFrame(columns=["建設公司", "營造公司", "水電公司", "經銷商", "配比"])
    
    def _create_brand_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create brand relationship dataframe"""
        blocks = []
        for suffix in ['A', 'B', 'C']:
            brand_col = f"品牌{suffix}"
            ratio_col = f"品牌{suffix}比"
            if brand_col in df.columns and ratio_col in df.columns:
                block = df[["建設公司", "營造公司", "水電公司", brand_col, ratio_col]].rename(
                    columns={brand_col: "品牌", ratio_col: "配比"}
                )
                blocks.append(block)
        
        if blocks:
            brand_rel = pd.concat(blocks, ignore_index=True)
            brand_rel["品牌"] = brand_rel["品牌"].apply(DataProcessor.clean_name)
            return brand_rel.dropna(subset=["品牌", "水電公司"])
        
        return pd.DataFrame(columns=["建設公司", "營造公司", "水電公司", "品牌", "配比"])
    
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
        
        # Dealer analysis
        st.markdown("**終端經銷商配比分析**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
    
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
        
        # Dealer analysis
        st.markdown("**終端經銷商（平均配比｜按水電等權）**")
        dealer_analysis = analyzer.avg_dealer_ratio_across_unique_mep(rel_sel)
        if not dealer_analysis.empty:
            dealer_analysis["平均配比"] = dealer_analysis["平均配比"].apply(Formatters.pct_str)
            UIComponents.render_dataframe_with_styling(dealer_analysis)
        else:
            UIComponents.render_info_box("暫無經銷商配比資料")
    
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
        
        # Dealer chart
        if not rel_sel.empty:
            dealer_ratio = (rel_sel.groupby("經銷商")["配比"].mean()
                           .reset_index().sort_values("配比", ascending=False))
            dealer_ratio["額度_萬"] = dealer_ratio["配比"].astype(float) * vol_val
            dealer_chart_data = dealer_ratio.rename(columns={"額度_萬": "金額(萬)"})
            
            fig = ChartGenerator.create_chart(
                dealer_chart_data, "經銷商", "金額(萬)",
                f"水電公司 → 終端經銷商 金額(萬)", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Brand chart
        if not brand_rel.empty:
            brand_sel = brand_rel[brand_rel["水電公司"] == target]
            if not brand_sel.empty:
                brand_ratio = (brand_sel.groupby("品牌")["配比"].mean()
                              .reset_index().sort_values("配比", ascending=False))
                brand_ratio["額度_萬"] = brand_ratio["配比"].astype(float) * vol_val
                brand_chart_data = brand_ratio.rename(columns={"額度_萬": "金額(萬)"})
                
                fig = ChartGenerator.create_chart(
                    brand_chart_data, "品牌", "金額(萬)",
                    f"水電公司 → 線纜品牌 金額(萬)", chart_type
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_mep_competitors(self, target: str, df: pd.DataFrame):
        """Render MEP competitors analysis"""
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
        """Render analysis for dealers"""
        df_sel = rel[rel["經銷商"] == target].merge(
            df, on=["建設公司", "營造公司", "水電公司"], how="left", suffixes=("", "_df")
        )
        
        # KPI metrics
        stats = {
            "資料筆數": len(df_sel),
            "建設家數": df_sel["建設公司"].nunique(),
            "營造家數": df_sel["營造公司"].nunique(),
            "水電家數": df_sel["水電公司"].nunique()
        }
        UIComponents.render_kpi_section(stats)
        
        # Analysis tabs
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
        """Render dealer overview"""
        UIComponents.render_section_header("合作水電")
        
        # MEP partners with ratios
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        
        # Add dealer ratios
        ratio_df = (rel[rel["經銷商"] == target]
                   .groupby("水電公司")["配比"].mean()
                   .reset_index()
                   .rename(columns={"配比": "該經銷商配比"}))
        
        if not ratio_df.empty:
            ratio_df["該經銷商配比"] = ratio_df["該經銷商配比"].apply(Formatters.pct_str)
            mep_stats = mep_stats.merge(ratio_df, on="水電公司", how="left")
        
        UIComponents.render_dataframe_with_styling(mep_stats)
    
    def _render_dealer_visualizations(self, df_sel: pd.DataFrame):
        """Render dealer visualizations"""
        chart_type = st.radio("圖表類型", self.config.CHART_TYPES, horizontal=True, key="dealer_chart")
        
        mep_stats = self._create_share_table(df_sel, ["水電公司"], "水電公司")
        if not mep_stats.empty:
            fig = ChartGenerator.create_chart(
                mep_stats, "水電公司", "次數",
                f"經銷商 → 水電公司 合作次數", chart_type
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_dealer_competitors(self, target: str, rel: pd.DataFrame, mep_vol_map: Dict,
                                 analyzer: RelationshipAnalyzer, comp_analyzer: CompetitorAnalyzer):
        """Render dealer competitors analysis"""
        UIComponents.render_section_header("競爭者分析")
        
        # Get competition metrics
        union_share, total_target = analyzer.union_overlap_share_and_total(target)
        comp_df, target_total_market = comp_analyzer.dealer_competitors(target)
        
        # Display key metrics
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-big"><div class="label">競爭覆蓋率（去重）</div>'
            f'<div class="value">{Formatters.pct_str(union_share)}</div></div>', 
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-big gray"><div class="label">總市場額度(萬)</div>'
            f'<div class="value">{Formatters.fmt_amount(total_target)}</div></div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Competition table
        if comp_df.empty:
            UIComponents.render_info_box("暫無競爭者資料")
        else:
            UIComponents.render_dataframe_with_styling(comp_df, "詳細競爭分析")
            st.caption("說明：表格中的「重疊市場占比」為與單一對手的配對式重疊（加總可能 >100%）；上方的「競爭覆蓋率（去重）」為所有對手合併後的覆蓋比例（不會超過 100%）。")
    
    def _render_export_section(self, df_raw: pd.DataFrame, df: pd.DataFrame, 
                             rel: pd.DataFrame, brand_rel: pd.DataFrame):
        """Render export section"""
        UIComponents.render_section_header("資料匯出")
        
        st.markdown("**匯出說明**")
        st.markdown("""
        - **原始資料**: 上傳的原始檔案內容
        - **主檔**: 經過欄位標準化的主要資料
        - **關係明細_經銷**: 經銷商配比關係展開資料  
        - **關係明細_品牌**: 品牌配比關係展開資料
        """)
        
        # Create Excel file
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
    
    def run(self):
        """Run the main application"""
        # File upload
        st.markdown("### 📁 資料上傳")
        uploaded_file = st.file_uploader(
            "上傳 Excel 或 CSV 檔案",
            type=["xlsx", "xls", "csv"],
            help="支援 Excel (.xlsx, .xls) 和 CSV 格式檔案"
        )
        
        if not uploaded_file:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 16px; margin: 2rem 0;">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">🔍 使用說明</h3>
                
                <div style="background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                    <h4 style="color: #4a5568; margin-bottom: 1rem;">📋 檔案格式要求</h4>
                    <ul style="color: #718096; line-height: 1.8;">
                        <li>固定欄位順序：D=建設公司, E=營造公司, F=水電公司, G=年用量</li>
                        <li>H/J/L=經銷商A/B/C, I/K/M=對應配比</li>
                        <li>N/P/R=品牌A/B/C, O/Q/S=對應占比</li>
                    </ul>
                </div>
                
                <div style="background: white; padding: 1.5rem; border-radius: 12px;">
                    <h4 style="color: #4a5568; margin-bottom: 1rem;">🎯 支援分析角色</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                        <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🏢</div>
                            <div style="font-weight: bold; margin-bottom: 0.5rem;">建設公司</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">查看營造、水電、經銷商合作關係</div>
                        </div>
                        <div style="padding: 1rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 8px; color: white;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔨</div>
                            <div style="font-weight: bold; margin-bottom: 0.5rem;">營造公司</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">分析上下游合作網絡及競爭態勢</div>
                        </div>
                        <div style="padding: 1rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 8px; color: white;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
                            <div style="font-weight: bold; margin-bottom: 0.5rem;">水電公司</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">經銷商配比、品牌使用分析</div>
                        </div>
                        <div style="padding: 1rem; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 8px; color: white;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛒</div>
                            <div style="font-weight: bold; margin-bottom: 0.5rem;">經銷商</div>
                            <div style="font-size: 0.9rem; opacity: 0.9;">客戶分布、市場競爭分析</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Process data
        df_raw = self.read_file(uploaded_file)
        df, rel, brand_rel, mep_vol_map = self.process_data(df_raw)
        
        # 整體數據概覽
        st.markdown("### 📊 數據概覽")
        self._render_overall_statistics(df, rel, brand_rel)
        
        # 分析設定區域
        st.markdown("### 🎯 分析設定")
        self._render_analysis_settings(df, rel, brand_rel, mep_vol_map, df_raw)
    
    def _render_overall_statistics(self, df: pd.DataFrame, rel: pd.DataFrame, brand_rel: pd.DataFrame):
        """渲染整體統計數據"""
        # 主要統計指標
        total_records = len(df)
        total_developers = df["建設公司"].nunique()
        total_contractors = df["營造公司"].nunique()
        total_meps = df["水電公司"].nunique()
        total_dealers = rel["經銷商"].nunique() if not rel.empty else 0
        total_brands = brand_rel["品牌"].nunique() if not brand_rel.empty else 0
        
        # 創建統計卡片
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">總資料筆數</div>
                </div>
            """.format(f"{total_records:,}"), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">關係連結數</div>
                </div>
            """.format(f"{len(rel) + len(brand_rel):,}"), unsafe_allow_html=True)
        
        with col3:
            # 計算平均配比
            avg_ratio = rel["配比"].mean() if not rel.empty and rel["配比"].notna().any() else 0
            st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">平均配比</div>
                </div>
            """.format(f"{avg_ratio:.1%}"), unsafe_allow_html=True)
        
        # 各角色統計
        st.markdown("#### 🏗️ 各角色統計")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        role_stats = [
            ("🏢", "建設公司", total_developers, "#667eea"),
            ("🔨", "營造公司", total_contractors, "#f5576c"),
            ("⚡", "水電公司", total_meps, "#00f2fe"),
            ("🛒", "經銷商", total_dealers, "#38f9d7"),
            ("🏷️", "品牌", total_brands, "#43e97b"),
            ("📈", "平均年用量", f"{df['年使用量_萬'].mean():.1f}萬" if '年使用量_萬' in df.columns and df['年使用量_萬'].notna().any() else "—", "#ff9a9e")
        ]
        
        cols = [col1, col2, col3, col4, col5, col6]
        
        for i, (icon, label, value, color) in enumerate(role_stats):
            with cols[i]:
                st.markdown(f"""
                    <div style="background: white; border: 2px solid {color}; padding: 1.5rem; border-radius: 12px; text-align: center; transition: transform 0.3s ease;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin-bottom: 0.3rem;">{value if isinstance(value, str) else f"{value:,}"}</div>
                        <div style="font-size: 0.9rem; color: #718096;">{label}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    def _render_analysis_settings(self, df: pd.DataFrame, rel: pd.DataFrame, 
                                brand_rel: pd.DataFrame, mep_vol_map: Dict, df_raw: pd.DataFrame):
        """渲染分析設定區域"""
        
        # 使用expander來組織分析設定
        with st.expander("🔍 開始分析", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 選擇分析角色")
                # 自定義角色選擇器
                role_options = [
                    ("🏢 建設公司", "建設公司"),
                    ("🔨 營造公司", "營造公司"), 
                    ("⚡ 水電公司", "水電公司"),
                    ("🛒 經銷商", "經銷商")
                ]
                
                selected_role_display = st.selectbox(
                    "角色類型",
                    options=[display for display, _ in role_options],
                    help="選擇要分析的角色類型"
                )
                
                # 取得實際的角色名稱
                role = next(actual for display, actual in role_options if display == selected_role_display)
            
            with col2:
                st.markdown("#### 選擇目標公司")
                
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
                    "🔍 搜尋公司名稱", 
                    placeholder="輸入關鍵字過濾公司列表...",
                    help="支援模糊搜尋，輸入部分公司名稱即可"
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
                    help=f"從 {len(options)} 家{role}中選擇"
                )
            
            # 開始分析按鈕
            if target:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(
                        f"🚀 開始分析 {role} - {target}",
                        type="primary",
                        use_container_width=True
                    ):
                        # 顯示選擇的標籤
                        st.markdown("### 📈 分析結果")
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
                                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600;">{role}</span>
                                <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600;">{target}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # 執行分析
                        self.render_role_analysis(role, target, df, rel, brand_rel, mep_vol_map, df_raw)
            else:
                st.info("請選擇要分析的目標公司")

# ====================== Application Entry Point ======================
def main():
    """Main application entry point"""
    try:
        dashboard = ConstructionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"應用程式發生錯誤：{str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()

