
# Installation / Upgrade 

## Python dependencies

Python >= 3.8  
PyTorch >= 1.6.0 (Recommend >= 1.9.0)

## Installation

Clone this repository as follows:  
```bash
git clone --recurse-submodules https://github.com/kiyoon/PyVideoAI.git
```

Or, clone submodules and checkout to master after you clone this repository.  

```bash
git clone https://github.com/kiyoon/PyVideoAI.git
cd PyVideoAI 
git submodule init
git submodule update
git submodule foreach git checkout master
```

Preferrably, checkout to a stable release.

```bash
git checkout v0.1
git submodule update --recursive
```

Then, install each submodule.  

```bash
cd submodules/video_datasets_api
python setup.py develop
cd ../experiment_utils
python setup.py develop
```


Finally, install the PyVideoAI.

```bash
cd ../..
python setup.py develop
```



## Pulling (updating) the project

```bash
git pull
git submodule update --remote --merge
```
