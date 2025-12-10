import arxiv
import pandas as pd
import datetime

def fetch_arxiv_data(query: str, start_year: int, end_year: int, max_results: int = 1000) -> pd.DataFrame:
    """
    从 arXiv 获取指定时间段的论文数据
    :param query: 搜索关键词
    :param start_year: 起始年份 (e.g. 2000)
    :param end_year: 结束年份 (e.g. 2025)
    :param max_results: 最大获取数量
    :return: 包含论文元数据的 DataFrame
    """
    # 构造 arXiv 的时间范围查询语法
    # 格式: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
    start_date_str = f"{start_year}01010000"
    end_date_str = f"{end_year}12312359"
    
    # 组合查询: 关键词 AND 时间段
    # 注意: 如果 query 包含空格，最好用双引号括起来，这里假设用户输入比较简单
    final_query = f'{query} AND submittedDate:[{start_date_str} TO {end_date_str}]'
    
    print(f"📡 Searching arXiv: {final_query} (Limit: {max_results})...")
    
    client = arxiv.Client(
        page_size=100, # 每次请求100条，减少服务器压力
        delay_seconds=3, # 遵守 arXiv 速率限制，防止被封
        num_retries=3
    )
    
    search = arxiv.Search(
        query=final_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # 即使是时间段搜索，也优先返回最相关的
    )
    
    papers_data = []
    
    try:
        # 使用生成器迭代获取
        for r in client.results(search):
            try:
                # 提取年份
                pub_year = r.published.year
                pub_date = r.published.date()
                
                # 双重检查年份 (虽然 API 过滤了，但为了数据清洗的严谨性)
                if start_year <= pub_year <= end_year:
                    papers_data.append({
                        "title": r.title,
                        "summary": r.summary.replace("\n", " "),
                        "authors": ", ".join([a.name for a in r.authors[:3]]),
                        "published_date": pub_date,
                        "published_year": pub_year,
                        "pdf_url": r.pdf_url,
                        "categories": r.categories[0] if r.categories else "Unknown",
                        "entry_id": r.entry_id
                    })
            except Exception as item_error:
                continue # 跳过单条错误
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        # 如果出错但已经抓了一部分，就返回这部分
        if papers_data:
            return pd.DataFrame(papers_data)
        return pd.DataFrame()

    df = pd.DataFrame(papers_data)
    print(f"✅ Successfully fetched {len(df)} papers.")
    return df
