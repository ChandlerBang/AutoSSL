seed=$1
dataset=cs; lr=1e-3; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=citeseer; lr=5e-4; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out


dataset=wiki; lr=1e-3; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=corafull; lr=2e-3; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=arxiv; lr=1e-3; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=photo; lr=5e-3; lr_lambda=0.01
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=physics; lr=5e-3; lr_lambda=0.05
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

dataset=computers; lr=0.01; lr_lambda=0.01
python -W ignore train_meta.py --gpu_id=$2 --dataset $dataset --lr $lr --seed $seed --lr_lambda ${lr_lambda} >> res/${dataset}.out

