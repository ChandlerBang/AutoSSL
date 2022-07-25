# AutoSSL
[ICLR 2022] A PyTorch implementation of paper ["Automated Self-Supervised Learning for Graphs"](https://openreview.net/pdf?id=rFbR4Fv-D6-).

## Abstract
We observe that different pretext tasks affect downstream tasks differently cross datasets, which suggests that searching pretext tasks is crucial for graph self-supervised learning.  Different from existing works focusing on designing single pretext tasks, this work aims to investigate how to automatically leverage multiple pretext tasks effectively. Nevertheless, evaluating representations derived from multiple pretext tasks without direct access to ground truth labels makes this problem challenging. To address this obstacle, we make use of a key principle of many real-world graphs, i.e., homophily, as the guidance to effectively search various self-supervised pretext tasks. We provide theoretical understanding and empirical evidence to justify the flexibility of homophily in this search task. Then we propose the AutoSSL framework which can automatically search over combinations of various self-supervised tasks. 

## Requirements
All experiments are performed under `python=3.8.8`

For the versions of python packages, please see `requirements.txt`.

## Run the code  
For AutoSSL-ES, run
```
python train_es.py --dataset citeseer 
```
For AutoSSL-DS, run
```
python train_meta.py --dataset citeseer 
```

To reproduce the performance of AutoSSL-DS, please run 
```
bash run.sh 0 2
```
where 0 is the random seed and 2 is the ID of GPU being used.

If you prefer not to use the script, you can use the following command:
```
python -W ignore train_meta.py --gpu_id=0 --dataset wiki --lr 1e-3 --seed 0 --lr_lambda 0.05
```


## Cite
For more information, you can take a look at the [paper](https://openreview.net/pdf?id=rFbR4Fv-D6-)

If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{
    jin2022automated,
    title={Automated Self-Supervised Learning for Graphs},
    author={Wei Jin and Xiaorui Liu and Xiangyu Zhao and Yao Ma and Neil Shah and Jiliang Tang},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=rFbR4Fv-D6-}
}
```


