import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts, JsCode
from src.data_loader import fetch_arxiv_data
from src.processor import process_papers_to_keyword_stars
from src.predictor import predict_future_trends

# --- 页面配置 ---
st.set_page_config(
    page_title="Research Galaxy",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 样式 ---
st.markdown("""
<style>
    .stApp { background-color: #0b0c10; }
    [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }
    h1, h2, h3 { color: #eee !important; }
    p, span, div { color: #b0b0b0; }
    
    .rank-card {
        background: #1f1f26; padding: 15px; margin-bottom: 12px; border-radius: 8px;
        display: flex; align-items: center; border: 1px solid #2a2a35;
    }
    .rank-icon { font-size: 24px; width: 40px; text-align: center; margin-right: 10px; font-weight: bold; }
    .rank-content { flex-grow: 1; }
    .rank-title { color: #fff; font-weight: bold; font-size: 16px; margin-bottom: 4px; }
    .rank-meta { font-size: 12px; color: #888; }
    .heat-bar-bg { background: #333; height: 4px; border-radius: 2px; margin-top: 8px; width: 100%; }
    .heat-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #ff0055, #ff5500); }
</style>
""", unsafe_allow_html=True)

# --- 状态管理 ---
if 'stars_df' not in st.session_state: st.session_state.stars_df = None
if 'raw_papers' not in st.session_state: st.session_state.raw_papers = None
if 'future_data' not in st.session_state: st.session_state.future_data = None # 新增: 预测数据

# --- 侧边栏 ---
with st.sidebar:
    st.title("🔭 观测控制台")
    st.markdown("---")
    
    query = st.text_input("目标星域", value="Generative AI")
    years = st.slider("时间跨度", 2000, 2025, (2020, 2025))
    start_year, end_year = years
    col1, col2 = st.columns(2)
    with col1: max_results = st.slider("探测深度", 100, 3000, 200)
    with col2: max_stars = st.slider("星体上限", 10, 100, 25)
    
    if st.button("🚀 启动观测", type="primary"):
        with st.spinner(f"正在扫描..."):
            raw_df = fetch_arxiv_data(query, start_year, end_year, max_results)
            if not raw_df.empty:
                st.session_state.raw_papers = raw_df.reset_index(drop=True)
                stars, kw_map, _ = process_papers_to_keyword_stars(raw_df, max_stars=max_stars)
                st.session_state.stars_df = stars
                st.session_state.future_data = None # 重置预测
                st.success(f"完成。")
            else:
                st.error("无数据。")

    st.markdown("---")
    # 未来探索按钮
    if st.button("🛸 探索未来航向"):
        if st.session_state.stars_df is not None:
            with st.spinner("正在计算语义重心与趋势流场..."):
                future = predict_future_trends(st.session_state.stars_df)
                st.session_state.future_data = future
        else:
            st.warning("请先生成星图。")

    st.markdown("---")
    st.markdown("### 🌟 图例")
    st.markdown("""
    <div style="font-size:12px;">
    <span style="color:#00ffff">━</span> 未来连线 (Future Link)<br/>
    <span style="color:#ff0055">●</span> Supernova<br/>
    <span style="color:#00d4ff">●</span> Blue Giant<br/>
    <span style="color:#fadb14">●</span> Main Sequence
    </div>
    """, unsafe_allow_html=True)

# --- 主界面 ---
st.title(f"🌌 {query} 星域")

if st.session_state.stars_df is not None:
    df = st.session_state.stars_df
    col_map, col_rank = st.columns([3, 1.2])
    
    # === 左侧：星图 (含光束) ===
    with col_map:
        df_top = df[df['series_type'] == 'effectScatter']
        df_normal = df[df['series_type'] == 'scatter']
        
        # 1. 基础数据准备 (与之前相同)
        normal_data = []
        for _, row in df_normal.iterrows():
            normal_data.append({
                "name": row['keyword'],
                "value": [row['x'], row['y'], row['heat_score']],
                "symbolSize": row['size'],
                "itemStyle": {"color": row['color'], "opacity": row['opacity']},
                "tooltip_content": row['tooltip_html']
            })
            
        top_data = []
        for _, row in df_top.iterrows():
            top_data.append({
                "name": row['keyword'],
                "value": [row['x'], row['y'], row['heat_score']],
                "symbolSize": row['size'],
                "itemStyle": {"color": row['color'], "shadowBlur": 20, "shadowColor": row['color']},
                "tooltip_content": row['tooltip_html']
            })

        # 2. 光束数据准备
        lines_data = []
        if st.session_state.future_data:
            lines_data = st.session_state.future_data['lines']

        tooltip_formatter = JsCode(""" function (params) { return params.data.tooltip_content; } """).js_code

        option = {
            "backgroundColor": "#0b0c10",
            "animation": False,
            "dataZoom": [{"type": "inside"}, {"type": "inside", "yAxisIndex": 0}],
            "tooltip": {
                "trigger": "item", "enterable": True, "hideDelay": 800,
                "backgroundColor": "rgba(20,20,30,0.95)", "borderColor": "#555", "padding": 12,
                "formatter": tooltip_formatter, "extraCssText": "width:320px; white-space:normal; pointer-events:auto;"
            },
            "grid": {"top": 20, "bottom": 20, "left": 20, "right": 20},
            "xAxis": {"show": False, "scale": True}, "yAxis": {"show": False, "scale": True},
            "series": [
                {
                    "name": "Background", "type": "scatter", "data": normal_data,
                    "label": {"show": True, "formatter": "{b}", "position": "top", "color": "#ccc", "fontSize": 10},
                    "itemStyle": {"borderWidth": 0}
                },
                {
                    "name": "Supernovas", "type": "effectScatter", "data": top_data,
                    "rippleEffect": {"brushType": "stroke", "scale": 4},
                    "label": {"show": True, "formatter": "{b}", "position": "top", "color": "#fff", "fontSize": 14, "fontWeight": "bold"},
                    "zlevel": 2
                },
                # Series 3: 未来光束
                {
                    "name": "Future Links",
                    "type": "lines",
                    "coordinateSystem": "cartesian2d",
                    "data": lines_data,
                    "effect": {
                        "show": True,
                        "period": 4,        # 光点移动速度
                        "trailLength": 0.5, # 尾迹长度
                        "symbol": "arrow",  # 形状
                        "symbolSize": 8
                    },
                    "lineStyle": {
                        "color": "#00ffff", # 霓虹青
                        "width": 2,
                        "opacity": 0.6,
                        "curveness": 0.2    # 稍微弯曲，更有科技感
                    },
                    "zlevel": 3
                }
            ]
        }
        
        st_echarts(option, height="700px", key="map_final")

    # === 右侧：排行榜 + 预测报告 ===
    with col_rank:
        # 显示预测报告 (如果存在)
        if st.session_state.future_data:
            st.info(st.session_state.future_data['report'])
            st.markdown("---")

        st.subheader("🏆 核心热度榜")
        top_5 = df.sort_values(by='heat_score', ascending=False).head(5)
        max_heat = top_5['heat_score'].max() if not top_5.empty else 1
        
        for idx, (_, row) in enumerate(top_5.iterrows()):
            rank = idx + 1
            if rank == 1: icon = "🥇"; color = "#ffd700"
            elif rank == 2: icon = "🥈"; color = "#c0c0c0"
            elif rank == 3: icon = "🥉"; color = "#cd7f32"
            else: icon = f"{rank}"; color = "#666"
            percent = (row['heat_score'] / max_heat) * 100
            
            st.markdown(f"""
            <div class="rank-card">
                <div class="rank-icon" style="color:{color};">{icon}</div>
                <div class="rank-content">
                    <div class="rank-title">{row['keyword']}</div>
                    <div class="rank-meta">🔥 <b>{row['heat_score']}</b> Docs</div>
                    <div class="heat-bar-bg"><div class="heat-bar-fill" style="width:{percent}%;"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👈 请在左侧启动观测。")
