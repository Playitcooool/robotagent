FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# 1. 安装系统依赖和 Python 3.10 相关包
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    libgl1 \
    ffmpeg \
    wget \
    software-properties-common \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. 添加 deadsnakes PPA 来安装 Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# 3. 安装 Python 3.10 和相关的开发包
RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. 创建虚拟环境（使用 Python 3.10）
RUN python3.10 -m venv /opt/venv

# 5. 激活 venv
ENV PATH="/opt/venv/bin:$PATH"

# 6. 升级 pip（在 venv 中完全合法）
RUN pip install --upgrade pip

# 7. 安装 Python 包
RUN pip install \
    pybullet \
    numpy \
    pandas \
    fastmcp \
    pydantic

WORKDIR /app

CMD ["python"]
