FROM continuumio/miniconda3:24.4.0-0
LABEL maintainer="han.huang.20@ucl.ac.uk"

# install vscode cli
COPY docker_code.deb /
RUN apt update \
    && apt install -y dpkg sudo libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 libcairo2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxrandr2 xdg-utils gnupg2 git curl \
    && dpkg -i docker_code.deb && rm docker_code.deb

# install requirements for invariant-deepsolid -- care is needed to configure an old version of jax
RUN conda create -n deepsolid && export DS_PIP="/opt/conda/envs/deepsolid/bin/pip"\
    && conda install -n deepsolid -y python=3.10 ipykernel \
    && /opt/conda/envs/deepsolid/bin/python -m ipykernel install --user --name="deepsolid" \
    && conda install -n deepsolid -c conda-forge jax==0.2.26 -y \
    && $DS_PIP install -U "https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.1.75+cuda11.cudnn82-cp310-none-manylinux2010_x86_64.whl" \
    && conda install -n deepsolid numpy absl-py scipy=1.8.1 pandas attrs dataclasses h5py ordered-set networkx -y \
    && conda install -n deepsolid -c conda-forge pytables chex=0.1.5 optax=0.0.9 pyscf -y \
    && $DS_PIP install ml-collections \
    && conda install -n deepsolid nvidia/label/cuda-11.3.1::cuda -y \
    && conda install -n deepsolid -c conda-forge cudnn=8.2.1 -y \
    && conda install -n deepsolid -c conda-forge libstdcxx-ng=12 -y \
    && $DS_PIP install jsonpickle matplotlib ipympl \
    && apt install -y libgl1-mesa-glx xvfb \
    && $DS_PIP install pyvista[all]==0.43.3 trame_jupyter_extension==2.1.0 \
    && apt clean && $DS_PIP cache purge && conda clean --tarballs

# set working directory
RUN mkdir /home/invariant-schrodinger
WORKDIR /home/invariant-schrodinger

EXPOSE 80