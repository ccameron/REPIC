# DeepPicker

## Installation
DeepPicker installation is expected to take minutes. Note - REPIC must already be installed to allow for DeepPicker patching

1. [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) if the Conda command is unavailable
2. Check installed/supported CUDA version with any one of the following commands:\
``` nvcc --version ```\
or\
``` /usr/local/cuda/bin/nvcc --version ```\
or\
``` cat /usr/local/cuda/version.txt ```\
or\
``` nvidia-smi ``` - see CUDA Version label at top right
3. Clone DeepPicker GitHub repo:\
``` git clone https://github.com/nejyeah/DeepPicker-python.git ```
4. Activate REPIC Conda environment:\
``` conda activate repic ```
5. Apply our patch to the DeepPicker codebase:\
``` cp  $(pip show repic | grep -in "Location" | cut -f2 -d ' ')/../../../docs/patches/deeppicker/*.py DeepPicker-python/ ```
6. Deactivate REPIC Conda environment:\
``` conda deactivate ```
7. Remove pre-existing DeepPicker Conda environment:\
``` conda remove -n deep --all ```
8. Create DeepPicker Conda environment:\
``` conda create -n deep -c conda-forge python=3.6 mamba ```
9. Activate DeepPicker Conda environment:\
``` conda activate deep ```
10. Install TensorFlow with GPU support:\
``` mamba install scipy matplotlib scikit-image mrcfile tensorflow-gpu=2.1 cudatoolkit=10.1 ```\
Make sure the ``` tensorflow-gpu ``` AND ``` cudatoolkit ``` are supported by the installed CUDA version.
11. Install PyTorch CPU for faster image preprocessing:\
``` mamba install pytorch torchvision -c pytorch ```
12. Clean up install:\
``` conda clean --all ```\
``` mamba clean --all ```

**Note 1** - the original DeepPicker implementation was found to be too slow for analysis. We provide a path to the original software during installation step 3 and REPIC patches in step 5\
**Note 2** - see here for more information about CUDA compatibility with TensorFlow: [https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu)

More information about running and installing DeepPicker can be found [here](https://github.com/nejyeah/DeepPicker-python).

## REPIC reproducibility
The DeepPicker GitHub repo as of the latest commit on May 20th 2017 (commit hash [3f46c8b](https://github.com/nejyeah/DeepPicker-python/tree/3f46c8b0ffe2dbaa837fd9399b4a542588e991e6)) was installed.

DeepPicker models were trained using ``` train_type 1 ```. All other ``` train_type ``` options were NOT tested. Please contact DeepPicker authors regarding these options.

Use algorithm parameters described in [supp_data_files/supplemental_data_file_2.ods](../supp_data_files/supplemental_data_file_2.ods).
