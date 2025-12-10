# 使用官方轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目代码
COPY . .

RUN python download_model.py

# 暴露端口
EXPOSE 7860

# --- 关键修改: 增加 Streamlit 配置参数 ---
# --server.enableCORS false: 解决跨域重连问题
# --server.enableXsrfProtection false: 解决跨站请求伪造保护导致的断连
# --browser.serverAddress: 告诉浏览器服务器地址 (可选，通常不用)
# --browser.gatherUsageStats false: 禁用数据收集，稍微提升启动速度
CMD ["streamlit", "run", "app.py", \
    "--server.port", "7860", \
    "--server.address", "0.0.0.0", \
    "--server.enableCORS", "false", \
    "--server.enableXsrfProtection", "false", \
    "--browser.gatherUsageStats", "false"]
