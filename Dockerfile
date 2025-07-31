# 选择一个包含 CUDA 11.8 和 cuDNN 8 的基础镜像
# 推荐使用 devel 版本，因为它包含编译所需的头文件和库
# 如果你确定不需要编译任何东西，可以使用 -base 版本以减小镜像大小
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04 

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖（例如：Python 3、pip、git等）
# 注意：nvidia/cuda 镜像通常已经预装了 Python 3 和 pip，
# 但这个步骤确保它们是最新的，并安装其他常用工具。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get -y update && \
    apt-get install -y vim

# 设置 Python 3 为默认的 python 命令
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 升级 pip
RUN pip install --upgrade pip

# 安装常用的 Python 库
# 注意：这里安装的 PyTorch 是 CUDA 11.8 兼容版本
# 确保你的 PyTorch 版本与 CUDA 版本匹配，可以从 PyTorch 官网获取最新命令
RUN pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip install \
    transformers \
    datasets \
    scikit-learn \
    pandas \
    numpy \
    sentence-transformers
