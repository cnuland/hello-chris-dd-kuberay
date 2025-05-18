FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="ray-ubi9-py311-cu117" \
      summary="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 11.7 Python 3.11 base image for Ray" \
      io.k8s.description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.7 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471"

ENV NV_CUDA_CUDART_VERSION=11.7.99-1

COPY image/cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub \
  | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA

ENV CUDA_VERSION=11.7.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN yum upgrade -y && yum install -y \
    cuda-cudart-11-7-${NV_CUDA_CUDART_VERSION} \
    cuda-compat-11-7 \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

COPY image/NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# === CUDA Runtime ===
ENV NV_CUDA_LIB_VERSION=11.7.1-1
ENV NV_NVTX_VERSION=11.7.91-1
ENV NV_LIBNPP_VERSION=11.7.4.75-1
ENV NV_LIBNPP_PACKAGE=libnpp-11-7-${NV_LIBNPP_VERSION}
ENV NV_LIBCUBLAS_VERSION=11.11.3.6-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl
ENV NV_LIBNCCL_PACKAGE_VERSION=2.14.3-1
ENV NV_LIBNCCL_VERSION=2.14.3
ENV NCCL_VERSION=2.14.3
ENV NV_LIBNCCL_PACKAGE=${NV_LIBNCCL_PACKAGE_NAME}-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.7

RUN yum install -y \
    cuda-libraries-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-nvtx-11-7-${NV_NVTX_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    libcublas-11-7-${NV_LIBCUBLAS_VERSION} \
    ${NV_LIBNCCL_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Set this flag so that libraries can find the location of CUDA
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# === CUDA Development Tools ===
ENV NV_NVPROF_VERSION=11.7.50-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-11-7-${NV_NVPROF_VERSION}
ENV NV_CUDA_CUDART_DEV_VERSION=11.7.99-1
ENV NV_NVML_DEV_VERSION=11.7.91-1
ENV NV_LIBCUBLAS_DEV_VERSION=11.11.3.6-1
ENV NV_LIBNPP_DEV_VERSION=11.7.4.75-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-11-7-${NV_LIBNPP_DEV_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-devel
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.14.3-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}-${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.7
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=11.7.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-11-7-${NV_CUDA_NSIGHT_COMPUTE_VERSION}

RUN yum install -y \
    make \
    findutils \
    cuda-command-line-tools-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-libraries-devel-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-7-${NV_CUDA_LIB_VERSION} \
    cuda-cudart-devel-11-7-${NV_CUDA_CUDART_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    cuda-nvml-devel-11-7-${NV_NVML_DEV_VERSION} \
    libcublas-devel-11-7-${NV_LIBCUBLAS_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# === cuDNN for CUDA 11.7 ===
ENV NV_CUDNN_VERSION=8.5.0.96
ENV NV_CUDNN_PACKAGE=libcudnn8-${NV_CUDNN_VERSION}.cuda11.7
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-devel-${NV_CUDNN_VERSION}.cuda11.7

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN yum install -y \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && yum clean all \
    && rm -rf /var/cache/yum/*


# Install ffmpeg (static build)
RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /tmp/ffmpeg && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 && \
    mv /tmp/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    mv /tmp/ffmpeg/ffprobe /usr/local/bin/ffprobe && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg*

# Micropipenv install (for Pipfile.lock support)
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Install Python dependencies
COPY image/Pipfile.lock ./
RUN micropipenv install && rm -f ./Pipfile.lock

# Symlink libcublas if needed
RUN ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so || true

# Switch back to default non-root user
USER 1001
WORKDIR /opt/app-root/src