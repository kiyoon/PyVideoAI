
# Installation / Upgrade 

## Python dependencies

Python >= 3.9  
PyTorch >= 1.9.0

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
git checkout v0.3
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



## Pulling (updating) the project

```bash
git pull
git submodule update
```
