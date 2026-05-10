FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# 使用清华镜像源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key && \
    apt-key add /tmp/ros.key && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-msgs \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
RUN pip3 install --no-cache-dir \
    fastmcp \
    pydantic \
    numpy

# 3. 预拉取常见 Gazebo 模型（网络不可用时允许跳过，不影响内置模型）
RUN mkdir -p /opt/gazebo_models \
    && (git clone --depth=1 https://github.com/osrf/gazebo_models.git /opt/gazebo_models/osrf || true)

WORKDIR /app

# 4. 启动时自动 source ROS2 环境
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
