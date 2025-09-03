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
            st.info("請上傳 Excel 或 CSV 檔案開始分析")
            
            with st.expander("📋 使用說明", expanded=False):
                st.markdown("""
                **檔案格式要求：**
                - 固定欄位順序：D=建設公司, E=營造公司, F=水電公司, G=年用量
                - H/J/L=經銷商A/B/C, I/K/M=對應配比
                - N/P/R=品牌A/B/C, O/Q/S=對應占比
               
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
                **支援分析角色：**
                - 🏢 建設公司：查看營造、水電、經銷商合作關係
                - 🔨 營造公司：分析上下游合作網絡及競爭態勢
                - ⚡ 水電公司：經銷商配比、品牌使用分析
                - 🛒 經銷商：客戶分布、市場競爭分析
                """)
            
st.stop()

# Process data
@@ -1352,36 +1328,36 @@ def _render_overall_statistics(self, df: pd.DataFrame, rel: pd.DataFrame, brand_
total_dealers = rel["經銷商"].nunique() if not rel.empty else 0
total_brands = brand_rel["品牌"].nunique() if not brand_rel.empty else 0

        # 創建統計卡片 - 使用與圖表一致的配色
        # 創建統計卡片 - 使用與圖表一致的配色，調整大小
col1, col2, col3 = st.columns(3)

with col1:
st.markdown("""
                <div style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(31, 119, 180, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">總資料筆數</div>
                <div style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);">
                    <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem;">{}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">總資料筆數</div>
               </div>
           """.format(f"{total_records:,}"), unsafe_allow_html=True)

with col2:
st.markdown("""
                <div style="background: linear-gradient(135deg, #ff7f0e 0%, #d62728 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(255, 127, 14, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">關係連結數</div>
                <div style="background: linear-gradient(135deg, #ff7f0e 0%, #d62728 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(255, 127, 14, 0.2);">
                    <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem;">{}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">關係連結數</div>
               </div>
           """.format(f"{len(rel) + len(brand_rel):,}"), unsafe_allow_html=True)

with col3:
# 計算平均配比
avg_ratio = rel["配比"].mean() if not rel.empty and rel["配比"].notna().any() else 0
st.markdown("""
                <div style="background: linear-gradient(135deg, #2ca02c 0%, #17becf 100%); padding: 2rem; border-radius: 16px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(44, 160, 44, 0.3);">
                    <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{}</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">平均配比</div>
                <div style="background: linear-gradient(135deg, #2ca02c 0%, #17becf 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(44, 160, 44, 0.2);">
                    <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem;">{}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">平均配比</div>
               </div>
           """.format(f"{avg_ratio:.1%}"), unsafe_allow_html=True)

        # 各角色統計 - 調整為4個主要角色，使用與分析角色對應的配色
        # 各角色統計 - 調整為4個主要角色，調整卡片大小
st.markdown("#### 🏗️ 各角色統計")

col1, col2, col3, col4 = st.columns(4)
@@ -1399,33 +1375,33 @@ def _render_overall_statistics(self, df: pd.DataFrame, rel: pd.DataFrame, brand_
for i, (icon, label, value, color) in enumerate(role_stats):
with cols[i]:
st.markdown(f"""
                    <div style="background: white; border: 3px solid {color}; padding: 1.8rem; border-radius: 12px; text-align: center; transition: transform 0.3s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                        <div style="font-size: 2.2rem; font-weight: bold; color: {color}; margin-bottom: 0.5rem;">{value:,}</div>
                        <div style="font-size: 1rem; color: #718096; font-weight: 600;">{label}</div>
                    <div style="background: white; border: 2px solid {color}; padding: 1.2rem; border-radius: 10px; text-align: center; transition: transform 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">{icon}</div>
                        <div style="font-size: 1.6rem; font-weight: bold; color: {color}; margin-bottom: 0.3rem;">{value:,}</div>
                        <div style="font-size: 0.85rem; color: #718096; font-weight: 500;">{label}</div>
                   </div>
               """, unsafe_allow_html=True)

        # 額外統計信息 - 第二行
        # 額外統計信息 - 第二行，調整大小
st.markdown("#### 📈 補充統計")
col1, col2 = st.columns(2)

with col1:
st.markdown(f"""
                <div style="background: white; border: 3px solid #9467bd; padding: 1.8rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🏷️</div>
                    <div style="font-size: 2.2rem; font-weight: bold; color: #9467bd; margin-bottom: 0.5rem;">{total_brands:,}</div>
                    <div style="font-size: 1rem; color: #718096; font-weight: 600;">品牌數量</div>
                <div style="background: white; border: 2px solid #9467bd; padding: 1.2rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🏷️</div>
                    <div style="font-size: 1.6rem; font-weight: bold; color: #9467bd; margin-bottom: 0.3rem;">{total_brands:,}</div>
                    <div style="font-size: 0.85rem; color: #718096; font-weight: 500;">品牌數量</div>
               </div>
           """, unsafe_allow_html=True)

with col2:
avg_volume = df['年使用量_萬'].mean() if '年使用量_萬' in df.columns and df['年使用量_萬'].notna().any() else 0
st.markdown(f"""
                <div style="background: white; border: 3px solid #17becf; padding: 1.8rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-size: 2.2rem; font-weight: bold; color: #17becf; margin-bottom: 0.5rem;">{avg_volume:.1f}萬</div>
                    <div style="font-size: 1rem; color: #718096; font-weight: 600;">平均年用量</div>
                <div style="background: white; border: 2px solid #17becf; padding: 1.2rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📊</div>
                    <div style="font-size: 1.6rem; font-weight: bold; color: #17becf; margin-bottom: 0.3rem;">{avg_volume:.1f}萬</div>
                    <div style="font-size: 0.85rem; color: #718096; font-weight: 500;">平均年用量</div>
               </div>
           """, unsafe_allow_html=True)

