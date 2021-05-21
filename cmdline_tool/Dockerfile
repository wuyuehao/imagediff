FROM ubuntu:18.04

WORKDIR /workspace
ARG DEBIAN_FRONTEND=noninteractive

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
        cron \
        locales locales-all \
        vim \
        tzdata \
        libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

COPY . /workspace
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "idiff.py"]