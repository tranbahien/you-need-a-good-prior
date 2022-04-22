### Start from miniconda3
FROM continuumio/miniconda3:4.8.2
ENV DEBIAN_FRONTEND noninteractive

# User definition
ARG USER_NAME='ubuntu'
ARG USER_UID='1000'
ARG USER_GID='100'

# Install some useful packages
RUN apt-get update && \
    apt-get install -yq --no-install-recommends htop  && \
    apt-get clean

# Install pip
RUN conda install pip -c anaconda

# Install few missing packages
RUN conda install -c anaconda matplotlib numpy scipy pandas
RUN conda install -c conda-forge ipython jupyter jupyterlab

# Install pytorch
# RUN conda install pytorch torchvision cpuonly -c pytorch
RUN pip install "torch>1.5,<2"

# Install tensorflow
# RUN pip install tensorflow==2.1.0

# Add Tini (a cleanup utility for docker)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Copy local directory with source code and experiments
COPY . /optbnn
WORKDIR /optbnn

# Install the package `optbnn` (it will be available system-wise)
# RUN python setup.py install
RUN pip install .

# Add non-root user
RUN useradd -m -s /bin/bash -N -u $USER_UID $USER_NAME
ENV HOME=/home/$USER_NAME

# Fix .local not being in PATH and owner of .
ENV PATH=$PATH:$HOME/.local/bin
RUN chown -R $USER_NAME:$USER_GID .

# Change user
USER $USER_NAME

# Run the bash
CMD /bin/bash
