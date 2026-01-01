FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

USER root
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install --no-cache-dir \
    git+https://github.com/Megvii-BaseDetection/YOLOX.git \
    git+https://github.com/dataloop-ai-apps/dtlpy-converters.git

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1 bash
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1
