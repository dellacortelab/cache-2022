FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  # Update package list
  apt-get update -y && \
  # Install...
  apt-get install -y \
  git \
  vim \
  wget \
  python3 \
  python3-dev \
  python3-pip \
  tmux \
  # Remove package lists
  && rm -rf /var/lib/apt/lists/*

# Install conda
RUN cd /home && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init bash

COPY cache-env.yml /tmp/cache-env.yml
RUN $HOME/miniconda/bin/conda env create -f tmp/cache-env.yml --name cache

RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
RUN echo "conda init bash" >> $HOME/.bashrc
RUN echo "conda activate cache" >> $HOME/.bashrc


ENTRYPOINT ["/bin/bash"] 
