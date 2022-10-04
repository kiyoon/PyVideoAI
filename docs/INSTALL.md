
# Installation / Upgrade

## Python dependencies

Python >= 3.8
PyTorch >= 1.9.0	(If not using torch.run, lower version can work as well.)

## Installation

Clone this repository as follows:
```bash
git clone --recurse-submodules https://github.com/kiyoon/PyVideoAI.git
```

Or, clone submodules and checkout to master after you clone this repository.

```bash
git clone https://github.com/kiyoon/PyVideoAI.git
cd PyVideoAI
git submodule update --init --remote --merge --recursive
```

Preferrably, checkout to a stable release.

```bash
git checkout v0.4
git submodule update --recursive
```

Then, install each submodule.

```bash
cd submodules/video_datasets_api
pip install -e .
cd ../experiment_utils
pip install -e .
```


Finally, install the PyVideoAI.

```bash
cd ../..
pip install -e .
```

Optional: Pillow-SIMD and libjepg-turbo to improve dataloading performance.
Run this at the end of the installation:

```bash
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```


## Pulling (updating) the project

```bash
git pull
git submodule update
```
