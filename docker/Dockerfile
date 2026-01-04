# Use Ubuntu 24.04 as base (ships with Python 3.12)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install system deps, ffmpeg and system Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        build-essential \
        libgl1 \
        ffmpeg \
        wget \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set workdir
RUN useradd -m robot && mkdir /app && chown robot:robot /app
WORKDIR /app
USER robot

CMD ["python3"]
