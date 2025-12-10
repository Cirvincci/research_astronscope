# 使用官方轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# ------------------------------------------------------------------
# 【关键新增】设置 HuggingFace 镜像地址
# 确保 sentence-transformers 能在国内网络环境下快速下载模型
ENV HF_ENDPOINT=https://hf-mirror.com
# ------------------------------------------------------------------

# 安装系统级依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖 (使用阿里云镜像源加速)
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目所有代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
