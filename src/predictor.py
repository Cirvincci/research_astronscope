import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_future_trends(stars_df):
    """
    基于当前星图的高热度星体，预测未来的语义重心，并返回构成闭合图形的关键点
    """
    if stars_df is None or stars_df.empty:
        return None

    # 1. 提取高热度星体 (Supernovas & Blue Giants)
    # 取前 20% 的热点作为趋势风向标
    top_n = int(len(stars_df) * 0.2)
    if top_n < 3: top_n = 3
    
    trend_set = stars_df.head(top_n)
    
    # 2. 计算“未来重心” (加权语义中心)
    # 权重 = 热度 * (如果是新星则加权)
    # 这里简单用热度加权
    weights = trend_set['heat_score'].values
    vectors = np.stack(trend_set['vector'].values)
    
    weighted_sum = np.dot(weights, vectors)
    centroid_vec = weighted_sum / np.sum(weights)
    
    # 3. 在全域中寻找离这个重心最近的 3-4 个星体
    # 这些星体构成了未来的核心三角/四边形
    all_vectors = np.stack(stars_df['vector'].values)
    # 计算所有星体到重心的距离 (相似度)
    # reshape(1, -1) 是因为 centroid 是单向量
    sim_scores = cosine_similarity(centroid_vec.reshape(1, -1), all_vectors).flatten()
    
    # 取 Top 4 最接近未来重心的星体
    # 注意：排除掉已经非常热的 Top 1 (可选，为了发现新趋势，这里暂不排除)
    future_indices = sim_scores.argsort()[-4:][::-1]
    
    future_stars = stars_df.iloc[future_indices]
    
    # 4. 构建闭合连线数据
    # A->B->C->D->A
    coords = future_stars[['x', 'y', 'keyword']].values
    lines_data = []
    
    for i in range(len(coords)):
        start = coords[i]
        end = coords[(i + 1) % len(coords)] # 回路
        
        lines_data.append({
            "coords": [
                [start[0], start[1]], 
                [end[0], end[1]]
            ],
            "from": start[2],
            "to": end[2]
        })
        
    # 生成简报
    keywords = future_stars['keyword'].tolist()
    report = f"AI 预测锁定以下关键节点：\n**{' + '.join(keywords)}**\n该区域呈现极高的语义聚合趋势，建议重点关注其交叉领域。"
    
    return {
        "lines": lines_data,
        "nodes": future_stars,
        "report": report
    }
