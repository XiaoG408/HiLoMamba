# HiLoMamba: Frequency-aware Decoupling of Long- and Short-term Preferences for Sequential Recommendation

Our model is based on a famous open-source framework [Recbole](https://recbole.io/)

## Overview
![HiLoMamba](images/MyModel.png)

## Dataset Preparation
Please, download three widely used benchmarks from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI) constructed by RecBole's team.

## Training & Inference
```
# Beauty
python run_recbole.py --gpu_id='0' --model=AdaMCT --dataset='Amazon_Beauty' --config_files='config/Amazon_Beauty.yaml'

# Toys
python run_recbole.py --gpu_id='1' --model=AdaMCT --dataset='Amazon_Toys_and_Games' --config_files='config/Amazon_Toys_and_Games.yaml' 

# Sports
python run_recbole.py --gpu_id='2' --model=AdaMCT --dataset='Amazon_Sports_and_Outdoors' --config_files='config/Amazon_Sports_and_Outdoors.yaml'
```
