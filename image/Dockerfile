ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.1

FROM fedora:39

LABEL name="ray-fedora-py311-cu121" \
      summary="CUDA 12.1 + Python 3.11 base image on Fedora for Ray" \
      description="Fedora 39 image with Python 3.11 and CUDA 12.1 runtime for Ray workloads" \
      io.k8s.display-name="CUDA 12.1 Python 3.11 base image for Ray" \
      io.k8s.description="CUDA 12.1 Python 3.11 image on Fedora for Ray"

USER root
WORKDIR /opt/app-root/bin

# Set up NVIDIA CUDA repo (RHEL9-compatible works with Fedora here)
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.1"
ENV NV_CUDA_CUDART_VERSION=12.1.105-1

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo -o /etc/yum.repos.d/cuda.repo && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/D42D0685.pub | gpg --dearmor -o /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/cuda.repo && \
    echo "gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" >> /etc/yum.repos.d/cuda.repo

# Install base system tools + Python + CUDA + FFmpeg
RUN dnf upgrade -y && \
    dnf install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-devel \
        python${PYTHON_VERSION}-pip \
        make \
        findutils \
        git \
        curl \
        which \
        wget \
        cuda-compat-12-1 \
        cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
        cuda-libraries-12-1 \
        cuda-nvtx-12-1 \
        libnpp-12-1 \
        libcublas-12-1 \
        libnccl-2.17.1-1+cuda12.1 \
        cuda-command-line-tools-12-1 \
        cuda-libraries-devel-12-1 \
        cuda-minimal-build-12-1 \
        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_VERSION} \
        cuda-nvml-devel-12-1 \
        libcublas-devel-12-1 \
        libnpp-devel-12-1 \
        libnccl-devel-2.17.1-1+cuda12.1 \
        cuda-nsight-compute-12-1 \
        libcudnn8-8.9.0.131-1.cuda12.1 \
        libcudnn8-devel-8.9.0.131-1.cuda12.1 && \
    dnf clean all && \
    rm -rf /var/cache/dnf

RUN dnf install -y \
      https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-39.noarch.rpm && \
    dnf install -y ffmpeg ffmpeg-devel

# Set environment variables for CUDA + Python
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Set Python 3.11 as default
RUN alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    alternatives --install /usr/bin/pip pip /usr/bin/pip${PYTHON_VERSION} 1

# Install micropipenv for dependency management
RUN pip install --no-cache-dir -U "micropipenv[toml]"

# Copy and install Python dependencies (optional if using Pipfile.lock)
COPY --chown=1001:0 image/Pipfile.lock ./
RUN micropipenv install && rm -f ./Pipfile.lock

# Add license and restore workspace
COPY --chown=1001:0 image/NGC-DL-CONTAINER-LICENSE ./
COPY --chown=1001:0 ../app ./app
USER 1001
WORKDIR /opt/app-root/src