[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/repic/badges/version.svg?branch=master&kill_cache=1)](https://anaconda.org/bioconda/repic)
![Conda](https://img.shields.io/conda/pn/bioconda/repic)
[![Documentation Status](https://readthedocs.org/projects/repic/badge/?version=latest)](https://repic.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/515683107.svg)](https://zenodo.org/badge/latestdoi/515683107)

<img width="20%" src="imgs/repic_icon.png">

## Overview
<ins>RE</ins>liable <ins>PI</ins>cking by <ins>C</ins>onsensus (REPIC) is a consensus methodology for harnessing multiple cryogenic-electron microscopy (cryo-EM) particle picking algorithms. It identifies particles common to multiple picked particle sets (i.e., consensus particles) using graph theory and integer linear programming (ILP). Picked particle sets may be found by a human specialist (manual), template matching, mathematical function (e.g., RELION's Laplacian-of-Gaussian auto-picking), or machine-learning method. A schematic representation of REPIC applied to the output of three CNN-based particle pickers is below:

<p align="center">
<img width="60%" src="https://github.com/ccameron/REPIC/blob/main/imgs/repic_overview.png">
</p>

REPIC expects particle sets to be in BOX file format (*.box) where each particle has coordinates, a detection box size (in pixels), and (optional) a score [0-1].

## Software requirements
Required:
1. Python v3.8 interpreter ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation recommended)
2. Python package dependencies described in [setup.py](https://github.com/ccameron/REPIC/blob/main/setup.py)
3. _Windows users_ - [Ubuntu terminal environment with Windows Subsystem for Linux (WSL)](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview) (v22.04.2 LTS tested)

*Optional:*
1. [Gurobi ILP optimizer](https://www.gurobi.com/products/gurobi-optimizer/) (v9.5.2 used) - requires free [academic license](https://www.gurobi.com/downloads/) **
2. [REgularised LIkelihood OptimisatioN (RELION)](https://relion.readthedocs.io/en/release-3.1/) - particle and density analyses (v3.13 used)
3. [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) - map alignment and density visualization (v1.16 used)

** Required to reproduce manuscript results but if the Gurobi package is not found, REPIC will use the [SciPy ILP optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)

## Installation guide
REPIC installation is expected to only take a few minutes.

:warning: **WARNING**: Only the Docker installation of REPIC includes pickers (SPHIRE-crYOLO, DeepPicker, and Topaz). If installing REPIC using either Conda or pip, pickers will need to be separately installed (see [docs/](https://github.com/ccameron/REPIC/tree/main/docs/) for installation instructions).

**<details><summary>Install using Docker (recommended)</summary><p>**

1. [Install Docker](https://docs.docker.com/engine/install/) if the ``` docker ``` command is unavailable
2. [Install and set up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for building and running GPU-accelerated containers
3. Build CUDA-supported image from Dockerfile in REPIC GitHub repo:
```
docker build -t repic_img https://github.com/ccameron/REPIC.git#main
```
4. Run container with GPU acceleration (example [iter_pick.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_pick.py) command shown below):
```
docker run --gpus all -v <file_path>/REPIC/examples:/examples repic_img repic iter_pick /examples/10057/iter_config.json 4 100
```
</p></details>

**<details><summary>Install using Conda</summary><p>**

:warning: **WARNING**: if Python package conflicts are encountered during the Conda installation of REPIC, please ensure Conda channels are properly set for Bioconda. See [Bioconda Usage](https://bioconda.github.io/) for more information

1. [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) if the ``` conda ``` command is unavailable
2. Create a separate Conda environment and install REPIC and Gurobi:
```
conda create -n repic -c bioconda -c gurobi repic gurobi
```
3. Activate REPIC Conda environment:
```
conda activate repic
```
4. [Obtain a Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) and set Gurobi key:
```
grbgetkey <gurobi_key>
```
5. Remove unused or temporary Conda files:
```
conda clean --all
```
</p></details>

**<details><summary>Install from source using pip</summary><p>**

1. Either download the package by clicking the "Clone or download" button, unzipping file in desired location, and renaming the directory "REPIC" OR using the following command line:
```
git clone https://github.com/ccameron/REPIC
```
2. [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) if the ``` conda ``` command is unavailable
3. Navigate to REPIC directory:
```
cd <install_path>/REPIC
```
4. Create a separate Conda environment and install Gurobi for REPIC:
```
conda create -n repic -c gurobi python=3.8 gurobi
```
5. Activate REPIC Conda environment:
```
conda activate repic
```
6. Install REPIC using [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)):
```
python -m pip install .
```
7. [Obtain a Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) and set Gurobi key:
```
grbgetkey <gurobi_key>
```
8. Remove unused or temporary Conda files:
```
conda clean --all
```
</p></details>

To check if REPIC was correctly installed, run the following command (after activating the REPIC Conda environment or using a REPIC container):
```
repic -h
```
A help menu should appear in the terminal.

## Integration

**<details><summary>Run using published Docker image (with Singularity/Apptainer)</summary><p>**

A REPIC Docker image is published on both [DockerHub](https://hub.docker.com/r/cjfcameron/repic) and the [GitHub container registery](https://github.com/ccameron/REPIC/pkgs/container/repic). [Apptainer](https://apptainer.org/) (formerly Singularity) can be used to run this image:

1. [Install Apptainer](https://apptainer.org/docs/admin/main/installation.html) if the ```apptainer``` command is unavailable
2. [Install and set up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for building and running GPU-accelerated containers
3. Pull REPIC Docker image and convert to Singularity image format (SIF) (requires >8 Gb of memory and ~40 mins for conversion):
```
apptainer pull docker://cjfcameron/repic
```
If SIF file creation is taking a long time, increase the ```mksquashfs mem``` parameter in the Apptainer config file (apptainer.conf). See [here](https://apptainer.org/docs/admin/1.0/configfiles.html) for more information.

4. Run container with GPU acceleration (example [iter_pick.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_pick.py) command shown below):
```
apptainer run --nv --bind <file_path>/REPIC/examples:/examples repic_latest.sif repic iter_pick /examples/10057/iter_config.json 4 100
```
</p></details>

**<details><summary>Run using Google Colab</summary><p>**

A [Jupyter Notebook](https://jupyter.org/) for installing and running REPIC on [Google Colab](https://colab.google/) is included in the REPIC GitHub repo: [repic_colab.ipynb](https://github.com/ccameron/REPIC/blob/main/repic_colab.ipynb)

To open the notebook in Google Colab:

1. Navigate a browser to: https://colab.google/
2. Select "Open Colab", then "GitHub"
3. Enter the REPIC GitHub web URL: https://github.com/ccameron/REPIC.git
4. Select the "repic_colab.ipynb" Jupyter Notebook

</p></details>

**<details><summary>Run using Scipion plugin</summary><p>**

:warning: **WARNING**: Scipion plugin currently only contains REPIC one-shot mode

REPIC is available as a [Scipion](https://scipion.i2pc.es/)  plugin: [https://github.com/scipion-em/scipion-em-repic](https://github.com/scipion-em/scipion-em-repic)

See [here](https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html#installing-other-plugins) for information about installing plugins for Scipion.
</p></details>

## Example data
Example [SPHIRE-crYOLO](https://cryolo.readthedocs.io/en/stable/), [DeepPicker](https://github.com/jianlin-cheng/DeepCryoEM), and [Topaz](https://github.com/tbepler/topaz) picked particle coordinate files for $\beta$-galactosidase ([EMPIAR-10017](https://www.ebi.ac.uk/empiar/EMPIAR-10017/)) micrographs are found in [examples/10017/](https://github.com/ccameron/REPIC/tree/main/examples/10017/). These files were generated by applying the pre-trained pickers to $\beta$-galactosidase micrographs, filtering false positive per author suggested thresholds, and then converting files to BOX format using [coord_converter.py](https://github.com/ccameron/REPIC/blob/main/repic/utils/coord_converter.py).

Example motion corrected T20S proteasome ([EMPIAR-10057](https://www.ebi.ac.uk/empiar/EMPIAR-10057/)) micrographs and normative particles for iterative ensemble particle picking are freely available via Amazon Web Services (AWS). To download this data, please run [get_examples.sh](https://github.com/ccameron/REPIC/blob/main/repic/iterative_particle_picking/get_examples.sh) (see Quick start below for how to run this Bash script).

Installation instructions for SPHIRE-crYOLO, DeepPicker, and Topaz are found in [docs/](https://github.com/ccameron/REPIC/tree/main/docs/).

Example commands for fitting and running SPHIRE-crYOLO, DeepPicker, and Topaz models are found in [repic/iterative_particle_picking/](https://github.com/ccameron/REPIC/tree/main/repic/iterative_particle_picking).

Parameters used for particle picking algorithms and RELION are found in [supplemental_data_file_2.ods](https://github.com/ccameron/REPIC/blob/main/supp_data_files/supplemental_data_file_2.ods).

## Quick start
### Creating consensus particle sets
1. Calculate the particle overlap ([Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) \[JI\]) and enumerate cliques using [get_cliques.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/get_cliques.py) (expected run time: 1-3 mins):

```
repic get_cliques examples/10017/ examples/10017/clique_files/ 180
```

Note - REPIC will use the folder names found in the provided input directory (e.g.,  ``` examples/10017/ ```) to assign method labels (e.g., "crYOLO", "deepPicker", "topaz")

Correctly executing the above command will produce the following files for each micrograph in the output folder ``` examples/10017/clique_files/ ```:
  - *_clique_coords.pick: [pickled](https://docs.python.org/3/library/pickle.html) clique (*x*,*y*) coordinates (in BOX format)
  - *_constraint_matrix.pickle: pickled constraint matrix file
  - *_weight_vector.pickle: pickled clique weight vector file
  - *_runtime.tsv: runtime tracking TSV file

2. Find optimal cliques using the ILP solver and create consensus particle BOX files using [run_ilp.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/run_ilp.py) (expected run time: <1 min):

```
repic run_ilp examples/10017/clique_files/ 180
```

Correctly executing the above command will produce a particle coordinate file (in BOX format) for each micrograph  in the output directory ``` examples/10017/clique_files/ ```. The final column in these BOX files represents the clique weight for a consensus particle.

### Particle picking by iterative ensemble learning
1. Download example data from AWS S3 bucket using [get_examples.sh](https://github.com/ccameron/REPIC/blob/main/repic/iterative_particle_picking/get_examples.sh) (expected run time: 1-5 mins):

```
bash $(pip show repic | grep -in "Location" | cut -f2 -d ' ')/repic/iterative_particle_picking/get_examples.sh examples/10057/data/ &> aws_download.log
```

2. Create a configuration file for iterative ensemble particle picking using [iter_config.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_config.py) (expected run time: <1 min):

```
repic iter_config examples/10057/ 176 224 <file_path>/gmodel_phosnet_202005_N63_c17.h5 <file_path>/DeepPicker-python 4 22
```

``` <file_path> ``` must be replaced with the full file paths to the SPHIRE-crYOLO pre-trained model and DeepPicker directory, respectively. See picker installation instructions in [docs/](https://github.com/ccameron/REPIC/tree/main/docs/) for more information.

A configuration file ``` iter_config.json ``` will be created in the current working directory.

3. Pick particles by iterative ensemble learning using [iter_pick.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_pick.py), a wrapper of [run.sh](https://github.com/ccameron/REPIC/blob/main/repic/iterative_particle_picking/run.sh) (expected run time: 20-30 min/iteration):

```
repic iter_pick ./iter_config.json 4 100
```

The final set of consensus particles for the testing set should be found in:
``` examples/10057/iterative_particle_picking/round_4/train_100/clique_files/test/*.box ```

## Command line details
### Identifying consensus particle sets with REPIC
1. Calculating particle overlap (JI) and enumerate cliques using [get_cliques.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/get_cliques.py):

```
usage: repic get_cliques [-h] [--multi_out] [--get_cc] in_dir out_dir box_size

positional arguments:
  in_dir       path to input directory containing subdirectories of particle bounding box coordinate files
  out_dir      path to output directory (WARNING - script will delete directory if it exists)
  box_size     particle bounding box size (in int[pixels])

optional arguments:
  -h, --help   show this help message and exit
  --multi_out  set output of cliques to be members sorted by picker name
  --get_cc     filters cliques for those in the largest Connected Component (CC)
```

2. Finding optimal cliques using ILP solver and creating consensus particle BOX files using [run_ilp.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/run_ilp.py):

```
usage: repic run_ilp [-h] [--num_particles NUM_PARTICLES] in_dir box_size

positional arguments:
  in_dir                path to input directory containing get_cliques.py output
  box_size              particle bounding box size (in int[pixels])

optional arguments:
  -h, --help            show this help message and exit
  --num_particles NUM_PARTICLES
                        filter for the number of expected particles (int)
  ```
[run_ilp.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/run_ilp.py) will create a plot of the concensus particle distribution (``` particle_dist.png ```) with a recommended (REC) ``` num_particles ``` value (70\% of consensus particles) in the ``` in_dir ```.

### Particle picking by iterative ensemble learning

1. Create a configuration file for iterative ensemble particle picking using [iter_config.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_config.py):
```
usage: repic iter_config [-h] [--cryolo_env CRYOLO_ENV] [--deep_env DEEP_ENV] [--deep_model DEEP_MODEL] [--topaz_env TOPAZ_ENV]
                         [--topaz_model TOPAZ_MODEL] [--out_file_path OUT_FILE_PATH]
                         data_dir box_size exp_particles cryolo_model deep_dir topaz_scale topaz_rad

positional arguments:
  data_dir              path to directory containing training data
  box_size              particle bounding box size (in int[pixels])
  exp_particles         number of expected particles (int)
  cryolo_model          path to LOWPASS SPHIRE-crYOLO model
  deep_dir              path to DeepPicker scripts
  topaz_scale           Topaz scale value (int)
  topaz_rad             Topaz particle radius size (in int[pixels])

optional arguments:
  -h, --help            show this help message and exit
  --cryolo_env CRYOLO_ENV
                        Conda environment name or prefix for SPHIRE-crYOLO installation (default:cryolo)
  --deep_env DEEP_ENV   Conda environment name or prefix for DeepPicker installation (default:deep)
  --deep_model DEEP_MODEL
                        path to pre-trained DeepPicker model (default:out-of-the-box model)
  --topaz_env TOPAZ_ENV
                        Conda environment name or prefix for Topaz installation (default:topaz)
  --topaz_model TOPAZ_MODEL
                        path to pre-trained Topaz model (default:out-of-the-box model)
  --out_file_path OUT_FILE_PATH
                        path for created config file (default:./iter_config.json)
```

`data_dir/` is expected to contain a three-column TSV file of CTFFIND4 defocus values: (1) micrograph filename, (2) defocus x, and (3) defocus y. If this file is not found, then all micrographs will be assigned the same defocus value. A defocus file can be built from the output of a RELION CTF refinement job using the following Bash script:

```
EMPIAR_ID=<complete>  # only integers - i.e., EMPIAR-10017 would be 10017
out=<install_path>/REPIC/examples/${EMPIAR_ID}/data/defocus_${EMPIAR_ID}.txt
rm -rf ${out}
for file in <relion_path>/relion/CtfFind/job00[0-9]/<mrc_pattern>; do
  grep '' /dev/null ${file} | tail -n 1 | awk -F ":| " '{print $1,$3,$4}' >> ${out}
done
```

` <mrc_pattern> ` is dependent on the naming convention used for micrographs and will need to be set to your specific substring. For EMPIAR-10017 and -10057, the substrings are '\*0.txt' and '\*[0-9].txt', respectively.

` <relion_path>/relion/CtfFind/job00[0-9]/*<mrc_suffix> ` should list all CTFFIND4 output files in RELION's ` CtfFind/ `.

2. Iteratively pick particles using [iter_pick.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/iter_pick.py), a wrapper of [run.sh](https://github.com/ccameron/REPIC/blob/main/repic/iterative_particle_picking/run.sh):
```
usage: repic iter_pick [-h] [--semi_auto] [--sample_prob SAMPLE_PROB] [--score] [--out_file_path OUT_FILE_PATH] config_file num_iter train_size

positional arguments:
  config_file           path to REPIC config file
  num_iter              number of iterations (int)
  train_size            training subset percentage (int)

optional arguments:
  -h, --help            show this help message and exit
  --semi_auto           initialize training labels with known particles (semi-automatic)
  --sample_prob SAMPLE_PROB
                        sampling probability of initial training labels for 'semi_auto' (default:1.)
  --score               evaluate picked particle sets
  --out_file_path OUT_FILE_PATH
                        path for picking log file (default:<data_dir>/iter_pick.log)
```
``` train_size ``` references the output of [build_subsets.py](https://github.com/ccameron/REPIC/blob/main/repic/commands/build_subsets.py), which builds training subsets of sizes 1%, 25%, 50%, and 100% (i.e., 100% will use the entire training set). For more information on dataset handling please see "iterative ensemble particle picking with REPIC" in the Methods section of the REPIC manuscript.

## Testing
The REPIC software has been tested on two computer systems:
1. Ubuntu 16.04.6 LTS (Xenial Xerus) running CUDA v10.1 with four Nvidia GP102 TITAN Xp
2. Ubuntu 16.04.7 LTS (Xenial Xerus) running CUDA v11.3 with four Nvidia GeForce GTX 1080

## Citing REPIC
If REPIC was used in your analysis/study, please cite:

Cameron, C.J.F., Seager, S.J.H., Sigworth, F.J., Tagare, H.D., and Gerstein, M.B. **REliable PIcking by Consensus (REPIC): a consensus methodology for harnessing multiple cryo-EM particle pickers**. *Commun Biol*. DOI: [10.1038/s42003-024-07045-0](https://doi.org/10.1038/s42003-024-07045-0)

##  Contact
[Submitting a GitHub issue](https://github.com/ccameron/REPIC/issues) is preferred for all problems related to REPIC.

For other concerns, please email [Christopher JF Cameron](mailto:c.jf.cameron@gmail.com?subject=REPIC%20issue/question).

## Releases

### v1.0.0
  - Google Colab integrated
  - Release created for Nature Communications Biology publication

### v0.2.1
 - Scipion plugin created
 - Docker/Singularity/Apptainer integrated

### v0.2.0
 - k-d tree algorithm integrated to reduce graph building runtime
 - Approval to include DeepPicker with REPIC install/distribution added: https://github.com/ccameron/REPIC/blob/main/imgs/deeppicker_approval.png  
 - Various bug fixes

### v0.1.0
- SciPy ILP optimizer integrated to remove Gurobi package requirement
- Read the Docs documentation created: [https://repic.readthedocs.io/en/latest/](https://repic.readthedocs.io/en/latest/)
- Various bug fixes
