# download_model.py
from sentence_transformers import SentenceTransformer
import os

# 强制下载到缓存目录
print("Downloading all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save("/app/model_cache") # 保存到固定目录
print("Model saved to /app/model_cache")
