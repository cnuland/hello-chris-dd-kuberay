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
ENV CUDA_VERSION=11.7.1

# NVIDIA CUDA repo (RHEL9 / CUDA 11.7)
COPY image/cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVARCH}/D42D0685.pub \
    | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Install CUDA 11.7 runtime, tools, and devel libs
RUN yum upgrade -y && \
    yum install -y \
    cuda-cudart-11-7 \
    cuda-libraries-11-7 \
    cuda-libraries-devel-11-7 \
    cuda-command-line-tools-11-7 \
    libcublas-11-7 \
    libcublas-devel-11-7 \
    libnccl-2.14.3-1+cuda11.7 \
    libnccl-devel-2.14.3-1+cuda11.7 \
    libnpp-11-7 \
    libnpp-devel-11-7 \
    mesa-libGL \
    make \
    findutils \
    && yum clean all && rm -rf /var/cache/yum/*


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