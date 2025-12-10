from sentence_transformers import SentenceTransformer
import os

# 设置环境变量，确保下载位置固定
# 注意：这必须与 Dockerfile 中的 ENV 保持一致，或者干脆依赖默认
print("Pre-downloading model to default cache...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Download complete.")