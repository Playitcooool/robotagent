# Use Ubuntu 24.04 as base (ships with Python 3.12)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install system deps and ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        build-essential \
        libgl1 \
        ffmpeg \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 and distutils/venv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 and upgrade packaging tools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - \
    && python3.12 -m pip install --no-cache-dir -U pip setuptools wheel

# Install Python packages
RUN python3.12 -m pip install --no-cache-dir \
        pybullet \
        imageio \
        imageio-ffmpeg \
        numpy

# Create a non-root user and set workdir
RUN useradd -m robot && mkdir /app && chown robot:robot /app
WORKDIR /app
USER robot

CMD ["python3.12"]
