FROM nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3

RUN apt-get update && apt-get install -y cmake
RUN git clone https://github.com/pytorch/audio.git \
    && cd audio \
    && git checkout c670ad8 \
    && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop
 
WORKDIR /app

COPY inference/requirements.txt .
RUN pip install -r requirements.txt

COPY inference /app

RUN chmod +x run.sh

# TO-DO: is it necessary
ENV PYTHONIOENCODING=utf-8

CMD ["/app/run.sh", "0", "3", "streaming"]
