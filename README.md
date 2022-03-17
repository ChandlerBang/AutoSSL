# AutoSSL
A PyTorch implementation of paper ["Automated Self-Supervised Learning for Graphs"](https://arxiv.org/abs/2106.05470).


## Requirements
See `requirements.txt`.

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
bash run.sh 0
```
where 0 is the random seed.


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






