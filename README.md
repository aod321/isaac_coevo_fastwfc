# Isaac Coevolutionary FastWFC

## Installation
```
## Create a new virutal environment
conda create --name isaac_fastwfc python=3.8
conda activate isaac_fastwfc
pip install tqdm
pip install rich
pip install numpy
pip install opencv-python
## Install stable-baselines3
pip install stable-baselines3
## Install Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
## Install fastwfc
conda activate isaac_fastwfc
pip install ./xland_fastwfc
## Install IsaacGym
wget https://cloud.tsinghua.edu.cn/f/74174af43e8d439cb5bd/?dl=1 -O ./isaacgym_preview4.zip
unzip ./isaacgym_preview4.zip
cd isaacgym/python
pip install -e .
```

## Vectorized Environment Keyboard Control Test
```
python wfc_env_test_vecenv_fastwfc.py
```

| Key  | Function                   |
| ---- | -------------------------- |
| w    | Forward                    |
| a    | Left                       |
| s    | Down                       |
| d    | Right                      |
| k    | Rotate Counter-clock-wisze |
| l    | Rotate Clock-wisze         |
| i    | Resume Map                 |
| r    | Reset                      |
| p    | Pause Map                  |
| m    | Change Map                 |
| v    | Distribute Map             |

## Vectorized Environment Training on Empty Map
```
python wfc_env_train_vecenv_empty_fastwfc.py
```


## Vectorized Environment Co-evolution Training 
```
## without unity3d map render
python adaptive_coevo_vecenv_mp_8gpu_fastwfc.py

## with unity3d map render
python adaptive_coevo_vecenv_mp_8gpu_fastwfc_outimg.py
```

