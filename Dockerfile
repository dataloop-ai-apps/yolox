FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

RUN apt-get update && \
    apt-get install -y curl cmake build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ${DL_PYTHON_EXECUTABLE} -m pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/Megvii-BaseDetection/YOLOX.git && \
    ${DL_PYTHON_EXECUTABLE} -m pip install --no-cache-dir \
        git+https://github.com/dataloop-ai-apps/dtlpy-converters.git

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.5 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.5 bash
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.5
