# Diagonal Symmetrization of Neural Network Solvers for the Many-Electron Schrodinger Equation

This is the public code base for the [paper](https://arxiv.org/abs/2502.05318) accepted at ICML 2025. 

Please `git clone` with `--recurse-submodules`, as `bytedance/DeepSolid` is included as a submodule.

- [Repository structure](#repository-structure)
- [Usage with Singularity](#usage-with-singularity)
    - [Setup with Singularity](#setup-with-singularity)
    - [Run slurm jobs with Singularity](#slurm-with-singulartiy)
    - [Local testing / VS Code with Singularity](#local-testing--vscode-with-singularity)
- [Usage without Singularity](#usage-without-singularity)
    - [Setup without Singularity](#setup-without-singularity)
    - [Slurm without Singularity](#slurm-without-singularity)
- [`libcu_lib_path`: Manual loading of cuda libraries](#libcu_lib_path-manual-loading-of-cuda-libraries)

## Repository structure

The repository consists of the following files:
- Jupyter notebooks of the form `_notebook_***.ipynb`, which contains codes snippets and examples for training, inference and various visualizations presented in the paper. The notebooks also contain example slurm commands for running the codes over multiple nodes.
- `./train.py`, `./sampling.py` and `./eval.py` are the main python files.
- `config`, `optim_config`, `sampling_config` and `eval_config` contain configuration files for the physical systems, training, sampling and (optional) evaluation. `symmscan_config` contains files for visualizing the diagonal symmetry.
- `submodules` contains 
    - `DeepSolid` submodule cloned from the `bytedance/DeepSolid` repository;
    - A `space-group` folder that contains a set of methods for handling space groups. This is **not** a submodule.
- `utils` contains helper files.
- `utils_slurm` contains slurm helper files. `jax_dist.py` is a patch that handles multi-node distributed computation for an older version of jax, which was used in DeepSolid. 
- `Dockerfile` contains the Dockerfile used for building the singularity container; see [Setup with Singularity](#setup-with-singularity).

## Usage with Singularity

Note that there is a known issue with the current singualrity image that prevents `pyvista` from running. To use `pyvista` for visualization (see `_notebook_visualization.ipynb`), skip ahead to [Usage without Singularity](#usage-without-singularity).

### Setup with Singulartiy

*You can skip the steps below and download the image directly from the [Google Drive folder](https://drive.google.com/drive/folders/1jFtob6T3cO0tPipwVoYDosQKxmvYTGCb?usp=drive_link) and download `inv-ds.sif`. Follow the steps below if you want to make your own tweaks to the image.*

Download VS Code CLI `.deb` file for arm64 from [here](https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64) and save it in the folder as 

`docker_code.deb`

We need to first build the image on a machine where `docker` and `singualrity` are both available. `root` permission is not required, but typically you need to be added to a `docker` user group by your cluster admin to be able to build a docker image (see [here](https://stackoverflow.com/questions/67261873/building-docker-image-as-non-root-user) and [here](https://cloudyuga.guru/blogs/manage-docker-as-non-root-user/)). 

The first step is to run a Docker container `registry` that sets up a local docker image registry at `localhost:5000`:

```shell
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

Then, build the docker image from `Dockerfile`. Log outputs are available at `docker_build.log`:

```shell
docker build -t localhost:5000/inv-ds --progress=plain  . &> docker_build.log
```

Push the image to `localhost:5000`.

```shell
docker push localhost:5000/inv-ds
```

The next step builds the singularity image. <span style="color:red">**WARNING**</span>: The singularity image is pretty large (~16G) so it is recommended that you build it at a non-backed-up folder on the cluster, typically some kind of scratch folder. <span style='color:red'>**IMPORTANT**</span>: Use `export SCRATCH=/YOUR/SCRATCH/FOLDER` to specify the destination folder.

```shell
SINGULARITY_NOHTTPS=true singularity build --fakeroot $SCRATCH/inv-ds.sif docker://localhost:5000/inv-ds:latest
```

We can now safely remove the `registry` container.

```shell
docker container stop registry && docker container rm registry
```

Voila! Go to [Local testing / VS Code with Singularity](#local-testing--vscode-with-singularity) to test your singularity image. Make sure you remember your `$SCRATCH` folder.


### Slurm with Singulartiy

See `_notebook_training.ipynb` and `_notebook_inference.ipynb` for the respective commands.

### Local testing / VSCode with Singularity

Go to the code repository folder (typically `invariant-schrodinger`), and run the following command to open a shell inside the singualrity container. Some comments on the options:
- `--no-home` prevents the default behaviour of mounting the entire home directory on your machine
- `--nv` allows the container to access gpu resources
- `--bind` maps a directory on your local machine (`.`, i.e.~the current folder in this case) to `/home/invariant-schrodinger`. It is a [known issue](https://github.com/apptainer/singularity/issues/5903) that singularity containers are unwritable when gpus are available, so `/home/invariant-schrodinger` will be the only writable folder in the image and where all custom configurations are stored.
- <span style='color:red'>**IMPORTANT**</span>: Use `export SCRATCH=/YOUR/SCRATCH/FOLDER` first to specify the folder containing your singularity image.

```shell
singularity shell --no-home --nv --bind .:/home/invariant-schrodinger --pwd /home/invariant-schrodinger $SCRATCH/inv-ds.sif
```

Once you are inside singularity, you may choose to:
- **Run VS Code and debug**. The image comes with [VSCode Server](https://code.visualstudio.com/docs/remote/vscode-server), which can be run by the following command. The env variables exported below are so that changes by VS Code and cache files by `matplotlib` are in the writable folder.
    ```shell
    export VSCODE_AGENT_FOLDER="/home/invariant-schrodinger/__code_tunnel" && export MPLCONFIGDIR="$VSCODE_AGENT_FOLDER/mplconfig" HOME="$VSCODE_AGENT_FOLDER/home" && code tunnel --cli-data-dir=$VSCODE_AGENT_FOLDER
    ```
    Follow the instructions to use VS Code. To use vscode for the first time, you may want to install `Python` and `Jupyter` VS Code extensions. When prompted to select `Python interpreter`, use `deepsolid`. If you run into port communication problems for running the vscode notebook, uninstall and reinstall `Jupyter` extension. 

    **Debugging:** It is highly recommmended that you use the debugging functionality in the VS Code notebook. See [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) for instructions.

- **Execute python code without running VS Code.** To activate the environment, run
    ```shell
    source /opt/conda/bin/activate deepsolid
    ```

- **Git.** To push to GitHub within the Singularity container, follow the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux). Remember to save the ssh key as `/home/invariant-schrodinger/__code_tunnel/github`, which is excluded from git. After adding the public key to your GitHub account, you should be able to `git pull` and `git push` as usual. If you have started a new singularity container, before performing git actions, you should run 
    ```shell
        eval "$(ssh-agent -s)" && ssh-add /home/invariant-schrodinger/__code_tunnel/github
    ```

<span style='color:red'>**IMPORTANT:**</span> Whenever you are asked for the `libcu_lib_path` argument, you should specify it as `/opt/conda/envs/deepsolid/lib/`, which is where the relevant cuda packages are in the Singularity container. See [`libcu_lib_path`: Manual loading of cuda libraries](#libcu_lib_path-manual-loading-of-cuda-libraries).


## Usage without Singularity

### Setup without Singularity

The setup is for working with an older version of jax + Linux gpu with cuda 12.0. Some distribution of anaconda (e.g. [miniconda](https://docs.anaconda.com/free/miniconda/)) is assumed to be available.

Initialize environment and point ipykernel to the conda env where jupyter lab is to be run (*you may need to replace it with your local path of conda installation*)
```shell
conda create -n deepsolid python=3.10 ipykernel -y && conda activate deepsolid && python -m ipykernel install --prefix="~/miniconda3/envs/jupyterlab" --name="deepsolid"
```

Install DeepSolid dependencies and change jax an old gpu version
```shell
conda install -c conda-forge jax==0.2.26 -y && python -m pip install -U "https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.1.75+cuda11.cudnn82-cp310-none-manylinux2010_x86_64.whl" 
conda install numpy absl-py scipy=1.8.1 pandas attrs dataclasses h5py ordered-set networkx -y && conda install -c conda-forge pytables chex=0.1.5 optax=0.0.9 pyscf -y && python -m pip install ml-collections
```

Manually install cuda packages
```shell
conda install nvidia/label/cuda-11.3.1::cuda -y
conda install -c conda-forge cudnn=8.2.1 -y
conda install -c conda-forge libstdcxx-ng=12 -y
```

To make sure debugging doesn't get messed up by multi-threading, open the file
`~/miniconda3/envs/deepsolid/lib/python3.10/site-packages/debugpy/server/api.py` and set `"subProcess": False` in `_config`.

Install Spacegroup package dependencies
```shell
python -m pip install jsonpickle
```

Install matplotlib
```shell
python -m pip install matplotlib
python -m pip install ipympl
```

Install pyvista. Note that the virtual frame buffer `xvfb` needs to be available on the machine, which can be installed by e.g. `apt-get install xvfb`.
```shell
python -m pip install pyvista[all]==0.43.3 trame_jupyter_extension==2.1.0
```

### Slurm without Singularity

<span style='color:red'>**IMPORTANT**</span>: Replace YOUR_CONDA_PATH by path to conda in your own setup!*

Run something like this:
```shell
./slurm_dist.sh --mem=10G --num-nodes=2 --port=8001 -A YOUR_SLURM_ACCOUNT --partition=YOUR_SLURM_PARTITION --timeout=1000 \
--log='_log_graphene_OG_test' \
--env-cmd='source YOUR_CONDA_PATH/bin/activate deepsolid' \
--py-cmd='python DeepSolid_train.py --dist --x64 --cfg=graphene_1.py --optim_cfg=OG_batch1000.py --libcu_lib_path=YOUR_CONDA_PATH/envs/deepsolid/lib/'
```

## libcu_lib_path: Manual loading of cuda libraries

You may be asked to specify `libcu_lib_path` in various parts of the code, which loads the following libraries. This is to fix a bug in the old version of jax and cuda dependency, and is required if you have set up DeepSolid dependencies as above. `libcu_lib_path` is the path to the `lib` folder that contains the required cuda library files; this is typically the `lib` folder inside the `deepsolid` environment folder.
```shell    
from ctypes import cdll
cdll.LoadLibrary(libcu_lib_path + 'libcublas.so.11')
cdll.LoadLibrary(libcu_lib_path + 'libcudart.so.11.0')
cdll.LoadLibrary(libcu_lib_path + 'libcublasLt.so.11')
cdll.LoadLibrary(libcu_lib_path + 'libcufft.so.10')
cdll.LoadLibrary(libcu_lib_path + 'libcurand.so.10')
cdll.LoadLibrary(libcu_lib_path + 'libcusolver.so.11')
cdll.LoadLibrary(libcu_lib_path + 'libcusparse.so.11')
cdll.LoadLibrary(libcu_lib_path + 'libcudnn.so.8')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

In the last line above, we have included an additional flag, as preallocated memory is usually not enough for running the code.
