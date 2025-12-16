import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter

def process_papers_to_keyword_stars(df: pd.DataFrame, max_stars: int = 150):
    """
    聚合逻辑 v13.0: 强力防重叠 (High Perplexity + Jitter)
    """
    if df.empty:
        return pd.DataFrame(), {}, None

    # 1. 文本准备
    df['content'] = df['title'] + " " + df['summary']
    
    # 2. TF-IDF + LSA
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    
    lsa = TruncatedSVD(n_components=100, random_state=42)
    paper_vectors = lsa.fit_transform(tfidf_matrix)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 3. 提取关键词
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

    # 4. 聚合
    keyword_map_by_id = defaultdict(list)
    keyword_map_by_idx = defaultdict(list)
    for idx, row in df.iterrows():
        eid = row['entry_id']
        for kw in row['extracted_keywords']:
            keyword_map_by_id[kw].append(eid)
            keyword_map_by_idx[kw].append(idx)

    valid_keywords = keyword_map_by_idx
    
    # 5. 计算属性
    stars_data = []
    for kw, idx_list in valid_keywords.items():
        papers_subset = df.loc[idx_list]
        heat_score = len(idx_list)
        
        current_vectors = paper_vectors[idx_list]
        centroid_vector = np.mean(current_vectors, axis=0)
        
        stars_data.append({
            "keyword": kw,
            "heat_score": heat_score,
            "vector": centroid_vector,
            "paper_indices": idx_list,
            "papers_subset": papers_subset
        })

    stars_df = pd.DataFrame(stars_data)

    # 6. 排序与视觉分级
    if not stars_df.empty:
        stars_df = stars_df.sort_values(by='heat_score', ascending=False)
        if len(stars_df) > max_stars:
            stars_df = stars_df.head(max_stars).copy()
        
        total_stars = len(stars_df)
        top5_cutoff = 5
        blue_cutoff = int(total_stars * 0.25)
        main_cutoff = int(total_stars * 0.60)
        
        types, colors, sizes, opacities, series_types = [], [], [], [], []
        stars_df = stars_df.reset_index(drop=True)
        
        for i in range(total_stars):
            if i < top5_cutoff:
                types.append("Supernova (Top 5)")
                colors.append("#ff0055")
                sizes.append(80)
                opacities.append(1.0)
                series_types.append("effectScatter")
            elif i < blue_cutoff:
                types.append("Blue Giant")
                colors.append("#00d4ff")
                sizes.append(40)
                opacities.append(0.9)
                series_types.append("scatter")
            elif i < main_cutoff:
                types.append("Main Sequence")
                colors.append("#fadb14")
                sizes.append(18)
                opacities.append(0.7)
                series_types.append("scatter")
            else:
                types.append("Brown Dwarf")
                colors.append("#4b5563")
                sizes.append(8)
                opacities.append(0.4)
                series_types.append("scatter")
                
        stars_df['star_type'] = types
        stars_df['color'] = colors
        stars_df['size'] = sizes
        stars_df['opacity'] = opacities
        stars_df['series_type'] = series_types
        
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

        # 7. t-SNE 降维 (防重叠逻辑)
        num_stars = len(stars_df)
        if num_stars < 2:
            stars_df['x'], stars_df['y'] = 0.0, 0.0
        else:
            # 1. 增大 perplexity: 只要样本够，尽量大一点 (上限50)
            # 这会让 t-SNE 更倾向于展示全局结构，而不是局部小团簇
            perp = min(50, max(5, int(num_stars / 3)))
            
            # 2. 使用 'random' 初始化: 比 'pca' 更容易散开
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random', learning_rate='auto')
            coords = tsne.fit_transform(np.stack(stars_df['vector'].values))
            
            # 3. 归一化到 [-90, 90]
            scaler = MinMaxScaler(feature_range=(-90, 90))
            coords_scaled = scaler.fit_transform(coords)
            
            # 4. 关键：注入抖动 (Jitter)
            # 给每个坐标点加上随机偏移，打破重叠
            # 偏移量 sigma=3.5，大约是画布宽度的 1.5%
            jitter = np.random.normal(loc=0.0, scale=3.5, size=coords_scaled.shape)
            coords_final = coords_scaled + jitter
            
            # 再次截断边界，防止抖出去了
            coords_final = np.clip(coords_final, -100, 100)
            
            stars_df['x'] = coords_final[:, 0]
            stars_df['y'] = coords_final[:, 1]

    final_map = {k: keyword_map_by_id[k] for k in stars_df['keyword']}
    stars_df = stars_df.drop(columns=['paper_indices', 'papers_subset'])
    
    return stars_df, final_map, None
