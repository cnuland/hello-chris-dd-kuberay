FROM registry.access.redhat.com/ubi9/python-311:latest

# Update labels to reflect Python 3.11 and CUDA 11.7
LABEL name="ray-ubi9-py311-cu117" \
      summary="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      io.k8s.display-name="CUDA 11.7 Python 3.11 base image for Ray" \
      io.k8s.description="CUDA 11.7 Python 3.11 image based on UBI9 for Ray" \
      # Or your repo
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

# --- Base System and CUDA Setup ---
ENV NVARCH x86_64
# Updated NVIDIA_REQUIRE_CUDA for CUDA 11.7
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.7 brand=tesla,driver>=450,driver<451 brand=unknown,driver>=450,driver<451 brand=nvidia,driver>=450,driver<451 brand=nvidiartx,driver>=450,driver<451 brand=geforce,driver>=450,driver<451 brand=geforcertx,driver>=450,driver<451 brand=quadro,driver>=450,driver<451 brand=quadrortx,driver>=450,driver<451 brand=titan,driver>=450,driver<451 brand=titanrtx,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471"

# Variables for CUDA 11.7
ENV CUDA_VERSION 11.7.1
# These versions are for CUDA 11.7.1. Check NVIDIA's site for exact package versions if issues arise.
ENV NV_CUDA_CUDART_VERSION 11.7.99-1
# Matches CUDA_VERSION
ENV NV_CUDA_LIB_VERSION 11.7.1-1
ENV NV_NVTX_VERSION 11.7.91-1
# libnpp for 11.7
ENV NV_LIBNPP_VERSION 11.7.4.75-1
# libcublas for 11.7 (check NVIDIA if this exact version is right for 11.7.1)
# Note: your previous Dockerfile had 11.11.3.6-1 and called it libcublas-11-8
# For CUDA 11.7, libcublas versions start with 11.
ENV NV_LIBCUBLAS_VERSION 11.10.3.66-1
# NCCL for CUDA 11.7
ENV NV_LIBNCCL_PACKAGE_VERSION 2.14.3-1

# CUDA repo setup - ensure this repo file is for RHEL9 and provides CUDA 11.7 packages
COPY image/cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub \
    | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA

# Install dependencies for adding repos and cleaning up
RUN dnf install -y dnf-utils yum-utils && dnf clean all

# Install CUDA base runtime, compatibility, and core libraries
RUN yum-config-manager --add-repo file:///etc/yum.repos.d/cuda.repo && \
    # Attempt to disable other cuda repos if they exist and might conflict
    # yum-config-manager --disable cuda\* || true && \
    # Assuming your repo file defines the repo with an ID that yum-config-manager can enable if needed, or it's enabled by default
    # yum-config-manager --enable cuda.repo-x86_64 && \ 
    yum upgrade -y && \
    yum install -y \
        cuda-cudart-11-7-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-7 \
        cuda-libraries-11-7-${NV_CUDA_LIB_VERSION} \
        cuda-nvtx-11-7-${NV_NVTX_VERSION} \
        libnpp-11-7-${NV_LIBNPP_VERSION} \
        libcublas-11-7-${NV_LIBCUBLAS_VERSION} \
        libnccl-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.7 \
    && yum clean all && rm -rf /var/cache/yum/*

# Set up LD paths for NVIDIA and CUDA
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    # Ensure CUDA toolkit paths are discoverable by the linker
    echo "/usr/local/cuda-${CUDA_VERSION}/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda-${CUDA_VERSION}/extras/CUPTI/lib64" >> /etc/ld.so.conf.d/cuda-cupti.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/local/cuda-${CUDA_VERSION}/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# For linking against stubs
ENV LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64/stubs:${LIBRARY_PATH}
# Point to specific CUDA version
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-${CUDA_VERSION}

COPY image/NGC-DL-CONTAINER-LICENSE /
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# === CUDA Development Tools (Optional, but often needed by PyTorch/RLlib build steps or advanced features) ===
ENV NV_CUDA_CUDART_DEV_VERSION 11.7.99-1
ENV NV_NVML_DEV_VERSION 11.7.91-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-devel-11-7-${NV_LIBCUBLAS_VERSION}
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-devel-11-7-${NV_LIBNPP_VERSION}
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-devel-${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.7
# Use LIB_VERSION for Nsight
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-11-7-${NV_CUDA_LIB_VERSION}

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

# === cuDNN Installation ===
# This URL should point to your accessible cuDNN local repository RPM.
# Ensure this RPM is for RHEL9/UBI9 if possible, or that the RHEL8 version is fully compatible.
ARG CUDNN_REPO_RPM_URL="https://cudnn-nginx-route-cudnn-host.apps.rosa.rosa-wswzx.e1i3.p3.openshiftapps.com/cudnn-local-repo-rhel8-8.5.0.96-1.0-1.x86_64.rpm"
# Version provided by your local repo RPM
ARG CUDNN_VERSION_PKG="8.5.0.96"

RUN curl -k -L -o /tmp/cudnn-local-repo.rpm ${CUDNN_REPO_RPM_URL} && \
    yum install -y /tmp/cudnn-local-repo.rpm && \
    rm -f /tmp/cudnn-local-repo.rpm && \
    # Now install the actual cuDNN packages from the repo you just added.
    # Adjust package names if your local repo defines them differently.
    # For RHEL-style cuDNN RPMs from NVIDIA, they are typically versioned.
    yum install -y libcudnn8-${CUDNN_VERSION_PKG} libcudnn8-devel-${CUDNN_VERSION_PKG} && \
    yum clean all && \
    rm -rf /var/cache/yum/*

# === System Libraries for OpenCV and FFmpeg ===
RUN dnf install -y mesa-libGL && \
    dnf clean all && \
    rm -rf /var/cache/dnf/*

# Using static FFmpeg build to avoid repository complexities
RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    mkdir -p /tmp/ffmpeg && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 && \
    mv /tmp/ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    mv /tmp/ffmpeg/ffprobe /usr/local/bin/ffprobe && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg*

# Update linker cache after all system libraries are installed
RUN ldconfig

# === Python Dependencies via micropipenv ===
RUN pip install --no-cache-dir -U "pipenv==2023.12.1" "micropipenv[toml]"

# Ensure Pipfile and Pipfile.lock are prepared for Python 3.11 and PyTorch + CUDA 11.7
# (as per the Pipfile content provided in the previous response)
COPY image/Pipfile* ./ 
# It's safer to generate Pipfile.lock outside and copy it in,
# but if generating here, ensure pipenv uses python 3.11
# RUN pipenv lock --python 3.11 # If Pipfile.lock is not pre-generated and copied
# --deploy uses Pipfile.lock
RUN micropipenv install --deploy && rm -f ./Pipfile* # Restore user workspace
USER 1001
WORKDIR /opt/app-root/src