FROM ubuntu:18.04

WORKDIR /workspace
RUN mkdir -p /workspace
COPY . /workspace

# update apt and get miniconda
RUN apt-get update \
    && apt install -y build-essential \
    && apt-get install -y manpages-dev \
    && apt-get install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN conda update -n base -c defaults conda
# create conda environment

RUN conda init
RUN conda env create -f local_env.yaml
RUN conda activate imagediff
RUN echo 'conda activate imagediff' >> ~/.bashrc

ENTRYPOINT /workspace/entrypoint.sh