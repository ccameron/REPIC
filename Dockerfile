# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV MAMBA_ROOT_PREFIX=/root/micromamba

# install micromamba and other Linux tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  bc \
  build-essential \
  bzip2 \
  git \
  wget \
  ca-certificates \
  && wget -qO-  https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
  && touch /root/.bashrc \
  && /bin/micromamba shell init -s bash -r /opt/conda \
  && grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/conda/bashrc \
  && apt-get clean autoremove --yes \
  && rm -rf /var/lib/{apt,dpkg,cache,log}

SHELL ["bash", "-l" ,"-c"]

# create SPHIRE-crYOLO Conda environment and download pre-trained model
RUN micromamba create -n cryolo -c conda-forge pyqt=5 python=3.7 cudatoolkit=10.0.130 cudnn=7.6.5 numpy==1.18.5 libtiff wxPython=4.1.1 \
  && eval "$(micromamba shell hook --shell bash)" \
  && micromamba activate cryolo \
  && pip install 'cryolo[gpu]' \
  && micromamba deactivate \
  && micromamba clean --all -y \
  && wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5

# create Topaz Conda environment
RUN micromamba create -n topaz -c conda-forge -c tbepler -c pytorch python=3.6 pytorch=*=*cuda10.1* topaz=0.2.5 cudatoolkit=10.1.243 \
  && micromamba clean --all -y

# create REPIC Conda environment
RUN micromamba create -n repic -c conda-forge python=3.8 \
  && eval "$(micromamba shell hook --shell bash)" \
  && micromamba activate repic \
  && git clone https://github.com/ccameron/REPIC.git \
  && cd REPIC \
  && pip install . \
  && cd .. \
  && micromamba deactivate \
  && micromamba clean --all -y \
  && rm -rf REPIC/

# download DeepPicker GitHub repo and set up Conda environment
RUN git clone https://github.com/nejyeah/DeepPicker-python.git \
  && eval "$(micromamba shell hook --shell bash)" \
  && micromamba activate repic \
  && cp $(pip show repic | grep -in "Location" | cut -f2 -d ' ')/../../../docs/patches/deeppicker/*.py DeepPicker-python/ \
  && micromamba deactivate \
  && micromamba create -n deep -c conda-forge -c anaconda -c pytorch python=3.6 cudatoolkit=10.1 matplotlib mrcfile  pytorch scikit-image scipy tensorflow-gpu=2.1 torchvision \
  && micromamba clean --all -y

# make RUN commands use the new environment
SHELL ["micromamba", "run", "-n", "repic", "/bin/bash", "-c"]

# code to run when container is started
ENTRYPOINT ["micromamba", "run", "-n", "repic"]

LABEL Author="Christopher JF Cameron"
LABEL VERSION="v0.0.0"
