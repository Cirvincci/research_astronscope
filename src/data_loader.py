import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
import random
import concurrent.futures
import streamlit as st
import urllib.parse

# 1小时缓存
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arxiv_data(query: str, start_year: int, end_year: int, max_results: int = 1000) -> pd.DataFrame:
    """
    HTTP 并发极速版 arXiv 数据抓取器
    直接调用 API 接口，绕过 arxiv 库的单线程限制
    """
    # 构造时间范围查询 (arXiv API 格式)
    start_date = f"{start_year}01010000"
    end_date = f"{end_year}12312359"
    # 注意：API 需要 URL 编码
    search_query = f'{query} AND submittedDate:[{start_date} TO {end_date}]'
    encoded_query = urllib.parse.quote(search_query)
    
    base_url = "http://export.arxiv.org/api/query"
    
    # 定义单页抓取任务 (XML 解析)
    def fetch_page(start_index):
        # 随机休眠，防封
        time.sleep(random.uniform(0.1, 0.3))
        
        # 每次抓 100 条
        url = f"{base_url}?search_query={encoded_query}&start={start_index}&max_results=100&sortBy=relevance"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []
            
            # 解析 XML
            root = ET.fromstring(response.content)
            # arXiv API 的命名空间
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            page_data = []
            for entry in root.findall('atom:entry', ns):
                try:
                    # 提取基础信息
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                    summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                    published = entry.find('atom:published', ns).text
                    entry_id = entry.find('atom:id', ns).text
                    
                    # 提取作者
                    authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                    authors_str = ", ".join(authors[:3])
                    
                    # 提取 PDF 链接
                    pdf_url = entry_id.replace("abs", "pdf") # 简易转换，或者遍历 link 标签
                    for link in entry.findall('atom:link', ns):
                        if link.attrib.get('title') == 'pdf':
                            pdf_url = link.attrib.get('href')
                    
                    # 提取分类
                    primary_cat = entry.find('arxiv:primary_category', ns)
                    cat_str = primary_cat.attrib['term'] if primary_cat is not None else "Unknown"
                    
                    # 处理日期
                    pub_dt = pd.to_datetime(published)
                    
                    if start_year <= pub_dt.year <= end_year:
                        page_data.append({
                            "title": title,
                            "summary": summary,
                            "authors": authors_str,
                            "published_date": pub_dt.date(),
                            "published_year": pub_dt.year,
                            "pdf_url": pdf_url,
                            "categories": cat_str,
                            "entry_id": entry_id
                        })
                except Exception:
                    continue
            return page_data
            
        except Exception as e:
            print(f"Error fetching start={start_index}: {e}")
            return []

    # 计算任务切片
    tasks = list(range(0, max_results, 100))
    all_data = []
    
    print(f"🚀 HTTP Concurrent Fetch: {len(tasks)} requests...")
    
    # 开启线程池 (建议 5-8 个线程，太多会被 503)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_page, start) for start in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            all_data.extend(future.result())

    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = df.drop_duplicates(subset=['entry_id'])
        # 再次按相关性/时间排序 (API 返回可能是乱序的)
        # 这里简单截取
        df = df.head(max_results)
        
    print(f"✅ Downloaded {len(df)} papers.")
    return df
