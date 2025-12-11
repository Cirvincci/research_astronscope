import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0b0c10; }
    [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }
    h1, h2, h3 { color: #eee !important; }
    p, span, div { color: #b0b0b0; }
    
    .explore-btn { width: 100%; border-radius: 8px; margin-bottom: 20px; }
    
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

# --- 状态 ---
if 'stars_df' not in st.session_state: st.session_state.stars_df = None
if 'raw_papers' not in st.session_state: st.session_state.raw_papers = None
if 'future_data' not in st.session_state: st.session_state.future_data = None

# --- 侧边栏 ---
with st.sidebar:
    st.title("🔭 观测控制台")
    st.markdown("---")
    
    query = st.text_input("目标星域", value="Generative AI", help="输入您感兴趣的研究领域关键词")
    years = st.slider("时间跨度", 2000, 2025, (2020, 2025), help="筛选论文发表年份")
    start_year, end_year = years
    
    col1, col2 = st.columns(2)
    with col1: 
        max_results = st.slider("探测深度", 100, 3000, 200, help="尝试从 arXiv 抓取的最大论文数量。")
    with col2: 
        max_stars = st.slider("星体上限", 10, 100, 25, help="星图中显示的最大星体数量。")
    
    if st.button("🚀 启动观测", type="primary"):
        with st.status("正在初始化观测系统...", expanded=True) as status:
            st.write("📡 正在建立 arXiv 并发连接 (5 Threads)...")
            raw_df = fetch_arxiv_data(query, start_year, end_year, max_results)
            
            if not raw_df.empty:
                st.write(f"✅ 信号捕获成功! 已下载 {len(raw_df)} 篇文献元数据。")
                st.write("🧠 正在进行 LSA 语义降维与拓扑计算...")
                
                st.session_state.raw_papers = raw_df.reset_index(drop=True)
                stars, kw_map, _ = process_papers_to_keyword_stars(raw_df, max_stars=max_stars)
                st.session_state.stars_df = stars
                st.session_state.future_data = None 
                
                st.write("🎨 正在渲染星图...")
                status.update(label="观测完成！星图已生成。", state="complete", expanded=False)
            else:
                status.update(label="观测失败：无数据。", state="error")
                st.error("无数据。")

    st.markdown("---")
    st.markdown("### 🌟 图例说明")
    st.markdown("""
    <div style="font-size:12px;">
    <span style="color:#00ffff">━</span> 未来连线 (Future Link)<br/>
    <span style="color:#ff0055">●</span> Supernova (Top 5 热点)<br/>
    <span style="color:#00d4ff">●</span> Blue Giant (主流方向)<br/>
    <span style="color:#fadb14">●</span> Main Sequence (细分领域)
    </div>
    """, unsafe_allow_html=True)

# --- 关键修复: 逻辑预处理区 ---
# 为了让按钮点击能立即影响下面的星图，我们需要在这里创建一个隐藏的按钮来处理逻辑
# 但 Streamlit 的按钮必须在正确的位置渲染。
# 方案：我们在主界面的布局开始前，先检查 session_state，或者使用回调。
# 鉴于我们希望按钮在右侧，最简单的办法是：在渲染 ECharts 之前，先确定 future_data。
# 但 Streamlit 是脚本式运行，必须按顺序执行。
# 我们在右侧栏渲染按钮，如果被点击，更新 state 并 rerun，这样第二次运行时数据就是新的。
# 为了让第一次就生效，我们可以把按钮放在这里（顶部），然后用 CSS 把它挪到右边？不，太复杂。
# 正确做法：按钮依然在右边，但点击后强制 rerun。

# --- 主界面 ---
st.title(f"🌌 {query} 星域 ({start_year}-{end_year})")

if st.session_state.stars_df is not None:
    # 引导提示 (常驻)
    st.success("💡 **操作指南**: 鼠标 **悬停** 在星体上可预览论文，点击 **标题链接** 可直达 PDF。点击下方 **'探索未来航向'** 发现新趋势。")

    df = st.session_state.stars_df
    col_map, col_rank = st.columns([3, 1.2])
    
    # === 右侧 (先定义，处理交互) ===
    # 注意：Streamlit 不支持先渲染右边再渲染左边（除非用 container）。
    # 但我们可以在这里先处理按钮逻辑！
    with col_rank:
        if st.button("🛸 探索未来航向", type="secondary", help="AI 预测语义流向，发现潜在爆发点"):
            if st.session_state.stars_df is not None:
                with st.spinner("正在计算语义重心与趋势流场..."):
                    future = predict_future_trends(st.session_state.stars_df)
                    st.session_state.future_data = future
                    st.rerun() # 关键修复：强制重刷，确保左侧星图立即拿到数据
            else:
                st.warning("请先生成星图。")
        
        # 预测报告
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

    # === 左侧：星图 (使用最新的 future_data) ===
    with col_map:
        df_top = df[df['series_type'] == 'effectScatter']
        df_normal = df[df['series_type'] == 'scatter']
        
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
                "itemStyle": {
                    "color": row['color'], 
                    "shadowBlur": 20, 
                    "shadowColor": row['color']
                },
                "tooltip_content": row['tooltip_html']
            })

        lines_data = []
        if st.session_state.future_data:
            lines_data = st.session_state.future_data['lines']

        tooltip_formatter = JsCode(""" function (params) { return params.data.tooltip_content; } """).js_code

        option = {
            "backgroundColor": "#0b0c10",
            "animation": False,
            "grid": {"top": 40, "bottom": 40, "left": 40, "right": 40},
            "dataZoom": [{"type": "inside"}, {"type": "inside", "yAxisIndex": 0}],
            "tooltip": {
                "trigger": "item", "enterable": True, "hideDelay": 800,
                "backgroundColor": "rgba(20,20,30,0.95)", "borderColor": "#555", "padding": 12,
                "formatter": tooltip_formatter, "extraCssText": "width:320px; white-space:normal; pointer-events:auto;"
            },
            "xAxis": {"show": False, "scale": True, "min": -100, "max": 100}, 
            "yAxis": {"show": False, "scale": True, "min": -100, "max": 100},
            "series": [
                {
                    "name": "Background", "type": "scatter", "data": normal_data,
                    "label": {"show": True, "formatter": "{b}", "position": "top", "color": "#ccc", "fontSize": 10},
                    "itemStyle": {"borderWidth": 0}
                },
                {
                    "name": "Supernovas", "type": "effectScatter", "data": top_data,
                    "rippleEffect": {"brushType": "stroke", "scale": 3, "period": 4},
                    "label": {"show": True, "formatter": "{b}", "position": "top", "color": "#fff", "fontSize": 14, "fontWeight": "bold"},
                    "zlevel": 2
                },
                {
                    "name": "Future Links", "type": "lines", "coordinateSystem": "cartesian2d", "data": lines_data,
                    "effect": {"show": True, "period": 4, "trailLength": 0.5, "symbol": "arrow", "symbolSize": 8},
                    "lineStyle": {"color": "#00ffff", "width": 2, "opacity": 0.6, "curveness": 0.2},
                    "zlevel": 3
                }
            ]
        }
        
        st_echarts(option, height="700px", key="map_final")

else:
    st.info("👈 请在左侧启动观测。探测深度：爬取论文数量（每增加100篇论文约增加23s等待时间，若探测完成卡住未响应3s可再次点击“启动探测”刷新出星图）；星体上限：最大聚合领域数量。探索特定领域最热最新细化方向的分布！")
