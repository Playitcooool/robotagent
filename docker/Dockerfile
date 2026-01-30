# Use Ubuntu 24.04 as base (ships with Python 3.12)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# ----------------------
# System dependencies
# ----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        build-essential \
        libgl1 \
        ffmpeg \
        wget \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# ----------------------
# Python dependencies
# ----------------------
RUN pip install --no-cache-dir \
        pybullet \
        numpy \
        pandas \
        pydantic \
        fastmcp \
        mcp \
        scipy \
        matplotlib

# ----------------------
# Non-root user
# ----------------------
RUN useradd -m robot && mkdir /app && chown robot:robot /app
WORKDIR /app
USER robot

CMD ["python3"]
