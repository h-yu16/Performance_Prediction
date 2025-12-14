import argparse
import os
import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd

from odpbench.datasets import *
from odpbench.lib import misc
import pickle
from pathlib import Path

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['TORCH_HOME'] = "/mnt/SHARED/hanyu/torch"

class ClassMappingLayer(nn.Module):
    def __init__(self, mapping):
        super(ClassMappingLayer, self).__init__()
        self.mapping = mapping

    def forward(self, x):
        # x: logits, shape: (batch_size, num_classes)
        probs = torch.softmax(x, dim=1) 
        # convert to probability first because we can't sum logits
        mapped_probs = torch.zeros(x.size(0), len(set(mapping.values())), device=x.device)
        for cate in range(x.size(1)):
            if cate in mapping:
                mapped_probs[:, mapping[cate]] += probs[:, cate] # Add the probabilities if a new category consists of multiple original categories
        mappped_logits = torch.log(mapped_probs)
        return mappped_logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate pkl for ImageNet and CIFAR series')
    parser.add_argument('--txtdir', type=str, default="/mnt/SHARED/hanyu/dataset/txtlist-pp")
    parser.add_argument('--ckpt_dir', type=str, default="/mnt/SHARED/hanyu/Performance_Prediction/checkpoint")
    parser.add_argument('--pkldir', type=str, default="/mnt/SHARED/hanyu/Performance_Prediction/pkl")
    parser.add_argument('--model_start', type=int, default=0)
    parser.add_argument('--model_end', type=int, default=115)
    parser.add_argument('--dataset', type=str, default="ImageNet")
    parser.add_argument("--domain", type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_seed', type=int, default=0, help='used for seeding split_dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="logs")
    args = parser.parse_args()
    misc.setup_seed(args.seed)


    if "imagenet" in args.dataset.lower() or args.dataset == "objectnet-1.0":
        base_dataset = "ImageNet"
    elif "CIFAR-100" in args.dataset:
        base_dataset = "CIFAR-100"
    elif "CIFAR-10" in args.dataset or args.dataset in ["CINIC-10", "STL-10"]:
        base_dataset = "CIFAR-10"
    else:
        raise NotImplementedError
    
    if "CIFAR" in base_dataset:
        cifar_model_dict = misc.get_cifar_model_dict(dataset=base_dataset, cifar_checkpoint_dir=args.ckpt_dir)
    

    # TODO: model loader for ImageNet should be deprecated
    mapped_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_list = misc.get_model_list(dataset=base_dataset)
    print(f"Total number of {base_dataset} models: {len(model_list)}")
        
    for idx, (model_name, weight) in enumerate(model_list):
        if not (args.model_start <= idx < args.model_end):
            continue
        if base_dataset == "ImageNet" and idx in [47, 48, 105, 108, 97, 17]:
            continue
        if base_dataset == "CIFAR-10" and idx in [48, 49, 50]:
            continue
        print(f"Model {idx}/{len(model_list)}: {model_name}-{weight}")
        if base_dataset == "ImageNet":
            model_constructor = getattr(models, model_name)
            model = model_constructor(weights=weight)
            preprocess = weight.transforms()
            dataloader = get_dataloader(args.txtdir, args.dataset, args.domain, args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess, seq=True) 
            # Class Mapping
            if args.dataset in ["Tiny-ImageNet-C", "ImageNet-R", "ImageNet-A", "ImageNet-Vid-Robust", "objectnet-1.0"]:
                mapping = pd.read_csv(f"{args.txtdir}/{args.dataset}/mapping.csv")
                mapping = {row["real_label"]: row["label"] for _, row in mapping.iterrows()}
                mapped_model = nn.Sequential(model, ClassMappingLayer(mapping))    
        else:
            preprocess = misc.CIFAR_preprocess[base_dataset]
            model_constructor = cifar_model_dict[model_name]
            model = model_constructor()
            checkpoint = torch.load(weight, map_location="cpu")
            if base_dataset == "CIFAR-10":
                state_dict = {k.replace("module.",""): v for k, v in checkpoint['net'].items()}
            else:
                state_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
            if args.dataset == "STL-10":
                # CIFAR: 0 airplane, 1 car, 2 bird, 3 cat, 4 deer, 5 dog, 6 -, 7 horse, 8 ship, 9 truck
                # Ours: 0 airplane, 1 bird, 2 car, 3 cat, 4 deer, 5 dog, 6 horse, 7 ship, 8 truck
                mapping = {0:0, 1:2, 2:1, 3:3, 4:4, 5:5, 7:6, 8:7, 9:8}
                mapped_model = nn.Sequential(model, ClassMappingLayer(mapping))
            loader_func = get_dataloader if args.dataset == "CINIC-10" else get_dataloader_numpy
            dataloader = loader_func(args.txtdir, args.dataset, args.domain, args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess, seq=True) 
                
                
        model_use = mapped_model if mapped_model else model        
        model_use.to(device)
        model_use.eval()
        logits, probs, preds, labels, corrects = misc.get_model_all_results(model_use, dataloader, device)
        pkldir = os.path.join(args.pkldir, args.dataset, args.domain)
        os.makedirs(pkldir, exist_ok=True)
        pklname = f"{model_name}-{weight}.pkl" if base_dataset == "ImageNet" else Path(weight).name
        with open(os.path.join(pkldir, pklname), "wb") as f:
            pickle.dump((logits, labels), f)
