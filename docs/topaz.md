# Topaz

##  Installation
The Topaz installation is expected to take minutes:

1. [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) if the conda command is unavailable
2. Check installed/supported CUDA version with any one of the following commands:\
``` nvcc --version ```\
or\
``` /usr/local/cuda/bin/nvcc --version ```\
or\
``` cat /usr/local/cuda/version.txt ```\
or\
``` nvidia-smi ``` - see CUDA Version label at top right
3. Remove pre-existing Topaz Conda environment:\
``` conda remove -n topaz --all ```
4. Create Topaz Conda environment:\
``` conda create -n topaz python=3.6 ```
5. Activate Topaz Conda environment:\
``` conda activate topaz ```
6. Install Topaz:\
``` conda install topaz=0.2.5 cudatoolkit=10.1.243 -c tbepler -c pytorch ```\
Make sure the ``` cudatoolkit ``` matches your supported CUDA version.
7. Clean up install:\
``` conda clean --all ```

To verify that the Topaz installation worked, run:\
``` topaz --help ```

A help menu should appear in the terminal.

If the shell returns ``` topaz: command not found ```, you may need to deactivate and reactivate the Topaz environment:
1. ``` conda deactivate topaz ```
2. ``` conda activate topaz ```

If you encounter a traceback error, you may need to install the future package:
1. ``` conda activate topaz ```
2. ``` conda install future ```
3. ``` conda clean --all ```

More information about running and installing Topaz can be found [here](https://github.com/tbepler/topaz).

##  REPIC reproducibility
Topaz v0.2.5 was installed for CUDA 10. 

The default pre-trained / general model was used for REPIC analyses.

Use algorithm parameters described in [supp_data_files/supplemental_data_file_2.ods](../supp_data_files/supplemental_data_file_2.ods).
