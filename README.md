#  LEARNING FROM TAXONOMY: MULTI-LABEL FEW-SHOT CLASSIFICATION FOR EVERYDAY SOUND RECOGNITION

## 1. Introducation
This repo contains Label-Dependence Prototypical Networks (LaD-ProtoNets) and code we used to curate a few-shot version of FSD50K (FSD-FS). More details can be found in the [paper](https://arxiv.org/pdf/2212.08952.pdf).

## 2. Quick start with our code
### 2.0 Prepare the environment
Ones can install all the packages using
```
pip install -r requirements.txt
```
### 2.1 To use our FSD-FS datasets ONLY
If you would like to testify your model in the multi-label few-shot scenario. Now you can use our FSD-FS directly by following the following steps:
* Download the FSD-FS directly in [Zenodo](https://zenodo.org/record/7557107#.Y86l9nbP3ZQ). The structure of the database should be like:
```
root
|
└─── dev_base
|    
└─── dev_val
|
└─── eval
|
└─── meta
      |
      └─── dev_base.csv  
      |
      └─── dev_val.csv  
      |
      └─── eval.csv  
      |
      └─── vocabulary.csv
```
* Use the dataset in your code by customising your own setting.
```
from torch.utils.data import DataLoader
from pytorch.fsd_fs import FSD_FS, MLFewShotSampler
# I/O
clip_dir = "path/to/clip_dir"
audio_dir = "path/to/audio_dir"
csv_dir = "path/to/csv_dir"
mode = "dev_val"  # ["dev_base", "dev_val", "eval"]
# Few-shot setting
n_class = n_class
n_supports = n_supports
n_queries = n_queries
n_task = =100

fsdfs = FSD_FS(
    clip_dir=clip_dir,
    audio_dir=audio_dir,
    csv_path=csv_dir,
    mode=mode,
    data_type="path",
    target_type="category",
)

sampler = MLFewShotSampler(
    dataset=fsdfs,
    labelset=fsd50k_splits[mode],
    n_class=15,
    n_supports=1,
    n_queries=5,
    n_task=100,
)
dataloader = DataLoader(
    fsdfs, batch_sampler=sampler, num_workers=4, pin_memory=True
)
for x, y in dataloader:
    your_algorithm.forward(x, y)
```
### 2.2 To reproduce our LaD-ProtoNet
Downlaod the original FSD50K database in the [Zenodo](https://zenodo.org/record/4060432#.Y86tzHbP3ZR). Afterwards, as we use [Hydra](https://github.com/facebookresearch/hydra) in our research, ones can change our experiment setting using
```
vim ./cfg/fewshot.yaml
``` 
or using command line directly. If you have settled all setting down, you can run the code simply by
```
python3 learners/fewshot.py
```
## TODO
- [ ] using FSD-FS audio files in our code directly

## Reference
If you find this repo is useful, please cite the paper:
```
@article{liang2022learning,
  title={Learning from Taxonomy: Multi-label Few-Shot Classification for Everyday Sound Recognition},
  author={Liang, Jinhua and Phan, Huy and Benetos, Emmanouil},
  journal={arXiv preprint arXiv:2212.08952},
  year={2022}
}

```
If you would like to use the FSD-FS database in your research, please cite the paper above as well as the paper:
```
@ARTICLE{9645159,  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   title={FSD50K: An Open Dataset of Human-Labeled Sound Events},   year={2022},  volume={30},  number={},  pages={829-852},  doi={10.1109/TASLP.2021.3133208}}
```
