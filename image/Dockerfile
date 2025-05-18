FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="ray-ubi9-py311-cu117" \
      summary="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 11.7 Python 3.11 base image for Ray" \
      io.k8s.description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

# Architecture and driver caps
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.7 brand=nvidia,driver>=470,driver<471"
ENV CUDA_VERSION=11.7.1
ENV NV_CUDA_CUDART_VERSION=11.7.99-1

# CUDA repo setup
COPY image/cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub \
    | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA

# Install CUDA base runtime and compatibility
RUN yum upgrade -y && \
    yum install -y \
        cuda-cudart-11-7-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-7 \
    && yum clean all && rm -rf /var/cache/yum/*

# Set up LD paths
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Required for NVIDIA containers
COPY image/NGC-DL-CONTAINER-LICENSE /
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# === CUDA Runtime Packages ===
ENV NV_CUDA_LIB_VERSION=11.7.1-1
ENV NV_NVTX_VERSION=11.7.91-1
ENV NV_LIBNPP_VERSION=11.7.4.75-1
ENV NV_LIBNPP_PACKAGE=libnpp-11-7-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION=11.11.3.6-1
ENV NV_LIBCUBLAS_PACKAGE=libcublas-11-8-${NV_LIBCUBLAS_VERSION}
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION=2.14.3-1
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.7

RUN yum install -y \
    cuda-libraries-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-11-7-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all && rm -rf /var/cache/yum/*

# === CUDA Development Tools ===
ENV NV_CUDA_CUDART_DEV_VERSION=11.7.99-1
ENV NV_NVML_DEV_VERSION=11.7.91-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-devel-11-8-${NV_LIBCUBLAS_VERSION}
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-11-7-${NV_LIBNPP_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-devel-2.14.3-1+cuda11.7
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=11.7.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-11-7-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

RUN yum install -y \
    make \
    findutils \
    cuda-command-line-tools-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-11-7-${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-nvml-devel-11-7-${NV_NVML_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && yum clean all && rm -rf /var/cache/yum/*

ENV NV_CUDNN_VERSION=9.10.1.4-1
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-11
ENV NV_CUDNN_PACKAGE_DEV=libcudnn9-devel-cuda-11

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all && rm -rf /var/cache/yum/*

# === FFmpeg (static build) ===
RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /tmp/ffmpeg && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 && \
    mv /tmp/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    mv /tmp/ffmpeg/ffprobe /usr/local/bin/ffprobe && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg*

# === micropipenv ===
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# === Pipfile dependency installation ===
COPY image/Pipfile.lock ./
RUN micropipenv install && rm -f ./Pipfile.lock

# Final user and working directory
USER 1001
WORKDIR /opt/app-root/src
