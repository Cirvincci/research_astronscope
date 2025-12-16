---
title: AI Research Astronscope
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: apache-2.0
short_description: AI驱动的科研趋势雷达与知识星图
---

# 🌌 AI Research Astronscope: AI-Driven Research Trend Radar ：AI科研宇宙望星图
**AI 驱动的科研趋势雷达与知识星图：来探索特定领域最新最热细化方向分布！**

> "以宇宙的视角，俯瞰人类知识的疆域。"

AI Research Astronscope 是一个可视化的科研情报分析平台。它摒弃了传统的列表式搜索，利用**语义向量 (Semantic Embedding)** 技术将海量 arXiv 论文映射为一片浩瀚的“知识星云”。在这里，每一颗星体代表一个核心研究领域，星体的大小象征热度，距离代表语义关联。

## ✨ Core Features (核心功能)

### 1. 🪐 语义星图 (Semantic Galaxy)
*   **双引擎驱动**: 结合 `TF-IDF` (提取精准术语) 与 `Transformer` (理解深层语义)，实现星体的智能聚类。
*   **视觉分层**:
    *   🔴 **Supernova (超新星)**: 爆发式增长的顶流领域 (Top 5)。
    *   🔵 **Blue Giant (蓝巨星)**: 稳健的主流研究方向。
    *   🟡 **Main Sequence (主序星)**: 细分领域的基石。
*   **交互式探索**: 悬停即可预览 Top 3 论文，点击直达 arXiv PDF，畅享论文free。

### 2. 🛸 未来航向预测 (Future Navigation)
*   **趋势流场**: 基于高热度星体的语义重心，AI 自动计算“知识引力场”的流向。
*   **光束可视化**: 在星图上动态绘制连接线，标示出未来最具潜力的**“黄金交叉域”**，辅助科研人员发现 Next Big Thing。

### 3. 🏆 实时热度榜 (Live Leaderboard)
*   动态追踪当前时间窗口内最活跃的研究关键词。
*   提供**“金银铜”**奖牌榜单，核心热点一目了然。

---

## 🛠️ How to Use (使用指南)

### Step 1: 设定观测参数
在左侧控制台 (Sidebar) 设置您的“望远镜”参数：
*   **目标星域**: 输入感兴趣的领域 (如 `Generative AI`, `Quantum Computing`)。
*   **时间跨度**: 滑动选择年份 (e.g., `2023-2025`)，支持回溯历史或聚焦当下。
*   **探测深度**: 设置最大爬取论文数 (建议 `500-1000`)！！注意：每增加100篇论文增加耗时约20s。

### Step 2: 启动扫描
点击 **`🚀 启动观测`** 按钮。
系统将实时连接 arXiv API，抓取元数据，并在本地进行高维向量计算与降维映射。
*(注: 首次运行可能需要几秒钟加载 AI 模型；首次运行可能出现响应延迟，再次点击“启动观测”即可呈现)*

### Step 3: 探索与发现
*   **宏观**: 观察星图的聚类形态，寻找孤立的新星或密集的星团。
*   **微观**: 鼠标悬停在星体上，查看该领域的代表性论文（含超链接）。
*   **预测**: 点击 **`🛸 探索未来航向`**，让 AI 为您绘制该领域的技术演进路线图。

---

## 🏗️ Technical Architecture (技术架构)

*   **Data Source**: `arXiv API` (Real-time Metadata)
*   **NLP Engine**: `sentence-transformers` (all-MiniLM-L6-v2) + `scikit-learn`
*   **Dimensionality Reduction**: `t-SNE` (High-dimensional Vector -> 2D Map)
*   **Visualization**: `ECharts` + `Streamlit`
*   **Deployment**: Docker Container on ModelScope

---

## 📜 License
This project is open-sourced under the Apache-2.0 license.
Powered by [ModelScope](https://www.modelscope.cn).
