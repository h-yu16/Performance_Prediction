# ODP-Bench

Official repo for ICCV 2025 paper "ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction" [[arxiv](https://arxiv.org/abs/2510.27263)] [[paper](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_ODP-Bench_Benchmarking_Out-of-Distribution_Performance_Prediction_ICCV_2025_paper.html)].

The trained model checkpoints, as a test suite to evaluate performance prediction algorithms, can be downloaded via the [link](https://pan.baidu.com/s/18VV-QCQiIhe62RpTbQCFDA?pwd=nuds ) of baidu netdisk. 

The list of available algorithms and datasets can be found in our paper. 

The main scripts include:

- `odpbench/scripts/pp.py`: Performance prediction for datasets from WILDS, doamin generalization, and subpopulation shift.
- `odpbench/scripts/get_results.py`: Calculate pkl for ImageNet and CIFAR series.
- `odpbench/scripts/pp_pkl.py`: Performance prediction based on pre-calculated pkl files, for ImageNet and CIFAR series. 

The paths of datasets, checkpoints, and pkl can be set via:

- Datasets
  - CIFAR series: `common_dir` in `datasets.py`.
  - WILDS: `WILDS_DATA` in `pp.py`.
  - Others: Filepaths are stored in txt files. Path of txt files is `args.txtdir` in `pp.py`. Example of txt files are available [here](https://pan.baidu.com/s/1pJZpFX1QSw9JbTPghcAkhQ?pwd=3m9j). You can download the required dataset and change the prefix of paths in these txt files for your use.

- Checkpoint
  - ImageNet series: `os.environ['TORCH_HOME']` in `get_results.py` and `pp.py`.
  - Others: `args.ckpt_dir` in `get_results.py` and `pp.py`.
- pkl
  - Save: `args.pkldir` in `get_results.py`.
  - Load: `args.pkldir` in `pp_pkl.py`.





## Example scripts

For ImageNet and CIFAR series, calculate pkl files first:

```shell
gpu=0
pklpath=yourpath
CUDA_VISIBLE_DEVICES=$gpu python -m odpbench.scripts.get_results --dataset CIFAR-10 --domain TEST --model_start 0 --model_end 1 --pkldir $pklpath
for domain in 'elastic_transform' 'gaussian_blur'
do
CUDA_VISIBLE_DEVICES=$gpu python -m odpbench.scripts.get_results --dataset CIFAR-10-C --domain $domain --model_start 0 --model_end 1 --pkldir $pklpath
done
```

Note that for ImageNet, change the domain from "TEST" to "VAL".

Then predict the performance:

```shell
gpu=0
algorithms=("ATC" "DOC" "NuclearNorm" "Dispersion" "MDE" "COT" "COTT" "MaNo") 
pklpath=yourpath
for domain in 'elastic_transform' 'gaussian_blur'
do
    for alg in ${algorithms[@]}; do
    		CUDA_VISIBLE_DEVICES=$gpu python -m odpbench.scripts.pp_pkl \
                        --arch "DLA_trial0_199_95.88" \
                        --dataset CIFAR-10-C --source CIFAR-10 --target $domain \
                        --algorithm $alg --seed 0 --pkldir $pklpath
		done
done
```



For other datasets, directly do performance prediction:

```shell
gpu=0
algorithms=("ATC" "DOC" "NuclearNorm" "Dispersion" "MDE" "COT" "COTT" "MaNo" "NeighborInvariance") 
for alg in ${algorithms[@]}; do
		CUDA_VISIBLE_DEVICES=$gpu python -m odpbench.scripts.pp \
        --arch resnet50 --pretrain Supervised \
        --dataset DomainNet \
        --source clipart infograph painting quickdraw --target real sketch \
        --algorithm $alg \
        --seed 0 --data_seed 0 
done
```

Note that since NeighborInvariance requires multiple forward propogations, we directly do performance prediction of this algorithms for ImageNet and CIFAR series as well.





## Cite

If you find this repo useful for your research, please consider citing the paper.

```bibtex
@inproceedings{yu2025odp,
  title={ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction},
  author={Yu, Han and Li, Kehan and Li, Dongbai and He, Yue and Zhang, Xingxuan and Cui, Peng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1846--1858},
  year={2025}
}
```

