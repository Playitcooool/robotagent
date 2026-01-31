FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# 1. 系统依赖 + Python（22.04 默认 Python 3.10）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    libgl1 \
    ffmpeg \
    wget \
    python3 \
    python3-full \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. 创建虚拟环境（规避 PEP 668）
RUN python3 -m venv /opt/venv

# 3. 激活 venv
ENV PATH="/opt/venv/bin:$PATH"

# 4. 升级 pip（在 venv 中完全合法）
RUN pip install --upgrade pip

# 5. 安装 Python 包
RUN pip install \
    pybullet==3.2.5 \
    numpy \
    pandas \
    fastmcp \
    pydantic

WORKDIR /app

CMD ["python"]
