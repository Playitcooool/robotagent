FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# 1. 安装 Gazebo 11 ROS2 桥接包
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-msgs \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
RUN pip3 install --no-cache-dir \
    fastmcp \
    pydantic \
    numpy

WORKDIR /app

# 3. 启动时自动 source ROS2 环境
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
