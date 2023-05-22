# SPHIRE-crYOLO

##  Installation
The SPHIRE-crYOLO installation is expected to take minutes:

1. [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) if the Conda command is unavailable
2. Remove pre-existing SPHIRE-crYOLO Conda environment:\
``` conda remove -n cryolo  --all ```
3. Check installed/supported CUDA version with any one of the following commands:\
``` nvcc --version ```\
or\
``` /usr/local/cuda/bin/nvcc --version ```\
or\
``` cat /usr/local/cuda/version.txt ```\
or\
``` nvidia-smi ``` - see CUDA Version label at top right
4. Install SPHIRE-crYOLO following CUDA toolkit support:\
**CUDA 10**\
``` conda create -n cryolo -c conda-forge -c anaconda pyqt=5 python=3.7 cudatoolkit=10.0.130 cudnn=7.6.5 numpy==1.18.5 libtiff wxPython=4.1.1 ```\
**CUDA 11**\
``` conda create -n cryolo -c conda-forge -c anaconda pyqt=5 python=3 numpy==1.18.5 libtiff wxPython=4.1.1 ```

5. Activate Conda environment:\
``` conda activate cryolo ```
6. Install GPU-supported SPHIRE-crYOLO:\
**CUDA 10**\
``` pip install 'cryolo[gpu]' ```\
**CUDA 11**\
``` pip install nvidia-pyindex ```\
then\
``` pip install 'cryolo[c11]'```

7. Clean up install:\
``` conda clean --all ```

More information about running and installing SPHIRE-crYOLO can be found [here](https://cryolo.readthedocs.io/en/stable/index.html).

##  REPIC reproducibility
SPHIRE-crYOLO CUDA 10 installation was performed.

Download the LOWPASS general model from 27 May 2020 described [here](https://cryolo.readthedocs.io/en/latest/installation.html#download-the-general-models):\
``` wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5 ```\
OR\
``` curl ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5 ```

A copy of the general model used during the development of REPIC is also stored on Amazon Web Services for posterity. To download the model, please run the following command:\
``` wget --no-check-certificate --no-proxy 'http://org.gersteinlab.repic.s3.amazonaws.com/gmodel_phosnet_202005_N63_c17.h5' ```\
OR\
``` curl http://org.gersteinlab.repic.s3.amazonaws.com/gmodel_phosnet_202005_N63_c17.h5 --insecure ```

Use algorithm parameters described in [supp_data_files/supplemental_data_file_2.ods](../supp_data_files/supplemental_data_file_2.ods).
