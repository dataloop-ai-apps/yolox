FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER root
RUN apt-get update && apt-get install -y curl

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git
RUN pip install git+https://github.com/dataloop-ai-apps/dtlpy-converters.git


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1 bash
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/yolox:0.0.1
