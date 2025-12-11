import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from collections import defaultdict, Counter

def process_papers_to_keyword_stars(df: pd.DataFrame, max_stars: int = 150):
    """
    聚合逻辑 v11.0 (No-Torch Edition): 
    使用 TF-IDF + LSA (SVD) 替代 Transformer，彻底移除 PyTorch 依赖。
    """
    if df.empty:
        return pd.DataFrame(), {}, None

    # 1. 文本准备
    df['content'] = df['title'] + " " + df['summary']
    
    # 2. 统一使用 TF-IDF 向量化
    # max_features 设大一点，捕捉更多语义细节
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # --- 替代方案的核心: LSA 降维生成"伪语义向量" ---
    # 将稀疏的 TF-IDF 矩阵压缩为 100 维稠密向量
    # 这能起到类似 Embedding 的效果：相关词在空间中会靠得更近
    lsa = TruncatedSVD(n_components=100, random_state=42)
    paper_vectors = lsa.fit_transform(tfidf_matrix)

    # 3. 提取关键词 (Naming)
    paper_keywords = []
    for i in range(len(df)):
        row_vector = tfidf_matrix[i].toarray().flatten()
        top_indices = row_vector.argsort()[-15:][::-1]
        candidates = feature_names[top_indices]
        
        selected = []
        for cand in candidates:
            if len(selected) >= 3: break
            is_duplicate = False
            for s in selected:
                if cand in s or s in cand:
                    is_duplicate = True; break
            if not is_duplicate: selected.append(cand)
        
        if len(selected) < 3:
            for cand in candidates:
                if len(selected) >= 3: break
                if cand not in selected: selected.append(cand)
                    
        paper_keywords.append(selected)
    df['extracted_keywords'] = paper_keywords

    # 4. 聚合统计
    keyword_map_by_id = defaultdict(list)
    keyword_map_by_idx = defaultdict(list)
    for idx, row in df.iterrows():
        eid = row['entry_id']
        for kw in row['extracted_keywords']:
            keyword_map_by_id[kw].append(eid)
            keyword_map_by_idx[kw].append(idx)

    valid_keywords = keyword_map_by_idx
    
    # 5. 计算星体属性
    stars_data = []
    for kw, idx_list in valid_keywords.items():
        papers_subset = df.loc[idx_list]
        heat_score = len(idx_list)
        
        # 使用 LSA 向量计算重心
        current_vectors = paper_vectors[idx_list]
        centroid_vector = np.mean(current_vectors, axis=0)
        
        stars_data.append({
            "keyword": kw,
            "heat_score": heat_score,
            "vector": centroid_vector, # 存的是 LSA 向量
            "paper_indices": idx_list,
            "papers_subset": papers_subset
        })

    stars_df = pd.DataFrame(stars_data)

    # 6. 排序与视觉分级 (保持不变)
    if not stars_df.empty:
        stars_df = stars_df.sort_values(by='heat_score', ascending=False)
        if len(stars_df) > max_stars:
            stars_df = stars_df.head(max_stars).copy()
        
        # ... (视觉逻辑代码与之前完全一致，此处省略以节省篇幅，请保留原有的 visual logic) ...
        # [Visual Logic Start]
        total_stars = len(stars_df); top5_cutoff = 5; blue_cutoff = int(total_stars * 0.25); main_cutoff = int(total_stars * 0.60)
        types, colors, sizes, opacities, series_types = [], [], [], [], []
        stars_df = stars_df.reset_index(drop=True)
        for i in range(total_stars):
            if i < top5_cutoff:
                types.append("Supernova (Top 5)"); colors.append("#ff0055"); sizes.append(80); opacities.append(1.0); series_types.append("effectScatter")
            elif i < blue_cutoff:
                types.append("Blue Giant"); colors.append("#00d4ff"); sizes.append(40); opacities.append(0.9); series_types.append("scatter")
            elif i < main_cutoff:
                types.append("Main Sequence"); colors.append("#fadb14"); sizes.append(18); opacities.append(0.7); series_types.append("scatter")
            else:
                types.append("Brown Dwarf"); colors.append("#4b5563"); sizes.append(8); opacities.append(0.4); series_types.append("scatter")
        stars_df['star_type'] = types; stars_df['color'] = colors; stars_df['size'] = sizes; stars_df['opacity'] = opacities; stars_df['series_type'] = series_types
        # [Visual Logic End]

        # Tooltip
        htmls = []
        for idx, row in stars_df.iterrows():
            subset = row['papers_subset']
            color = row['color']; kw = row['keyword']; heat = row['heat_score']
            tooltip_html = f"<div style='font-family:sans-serif; text-align:left;'><b style='color:{color}; font-size:16px;'>{kw}</b><br/><span style='color:#bbb'>🔥 Heat: {heat}</span><hr style='border: 0; border-top: 1px solid #555; margin: 8px 0;'/><div style='font-size:12px; line-height:1.5;'>"
            for _, p in subset.head(3).iterrows():
                title = p['title'].replace("'", "&apos;"); date_str = str(p['published_date']); pdf = p['pdf_url']
                tooltip_html += f"<div style='margin-bottom:8px;'><span style='color:{color};'>●</span> <a href='{pdf}' target='_blank' style='color:#fff; text-decoration:none; font-weight:bold; border-bottom:1px solid #555;'>{title}</a><br/><span style='color:#888; margin-left:12px;'>📅 {date_str}</span></div>"
            if heat > 3: tooltip_html += f"<i style='color:#666; font-size:10px;'>... and {heat-3} more</i>"
            tooltip_html += "</div></div>"
            htmls.append(tooltip_html)
        stars_df['tooltip_html'] = htmls

        # 7. t-SNE 降维 (输入 LSA 向量)
        num_stars = len(stars_df)
        if num_stars < 2:
            stars_df['x'], stars_df['y'] = 0.0, 0.0
        else:
            perp = min(30, num_stars - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
            coords = tsne.fit_transform(np.stack(stars_df['vector'].values))
            stars_df['x'] = coords[:, 0]
            stars_df['y'] = coords[:, 1]

    final_map = {k: keyword_map_by_id[k] for k in stars_df['keyword']}
    stars_df = stars_df.drop(columns=['paper_indices', 'papers_subset'])
    
    # 这里的 vectorizer 返回 None 即可，因为不再支持 user_profiler 的复杂定位
    return stars_df, final_map, None
