# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from itertools import cycle

from odpbench.datasets import FastDataLoader

import torchvision.models as models
import odpbench.CIFAR10_models as CIFAR10_models
import odpbench.CIFAR100_models as CIFAR100_models
import torchvision.transforms as transforms


CIFAR_preprocess = {
    "CIFAR-10": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    "CIFAR-100": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
}

def get_cifar_model_dict(dataset="CIFAR-10", cifar_checkpoint_dir="/mnt/SHARED/hanyu/Performance_Prediction/checkpoint/"):
    assert dataset in ["CIFAR-10", "CIFAR-100"]
    if dataset == "CIFAR-10":
        modeldir = os.path.join(cifar_checkpoint_dir, "CIFAR10")
        splittoken = "_trial"
        vardict = CIFAR10_models.__dict__
    else:
        modeldir = os.path.join(cifar_checkpoint_dir, "CIFAR100")
        splittoken = "-trial"
        vardict = CIFAR100_models.__dict__            
    files = os.listdir(modeldir)
    files.sort()
    modellist = [file.split(splittoken)[0] for file in files]
    modellist = list(set(modellist))
    modellist.sort()
    modeldict = {modelname: vardict[modelname] for modelname in modellist}
    return modeldict

def get_model_list(dataset="ImageNet"):
    assert dataset in ["ImageNet", "CIFAR-10", "CIFAR-100"]
    if dataset == "ImageNet":
        model_names = models.list_models()
        useful_model_list = []
        for model_name in model_names:
            try:
                # 获取该模型的所有预训练权重
                weights_enum = models.get_model_weights(model_name)
            except ValueError:
                print(f"模型 {model_name} 没有可用的预训练权重，跳过。")
                continue
            
            # 获取模型构造函数
            try:
                model_constructor = getattr(models, model_name)
            except AttributeError:
                # print(f"找不到模型 {model_name} 的构造函数，跳过。")
                continue
            
            # 遍历每个权重
            for weight in weights_enum:
                useful_model_list.append((model_name, weight))
    elif dataset.startswith("CIFAR"):
        if dataset == "CIFAR-10":
            modeldir = "/mnt/SHARED/hanyu/Performance_Prediction/checkpoint/CIFAR10"
            splittoken = "_trial"
            vardict = CIFAR10_models.__dict__
        else:
            modeldir = "/mnt/SHARED/hanyu/Performance_Prediction/checkpoint/CIFAR100"
            splittoken = "-trial"
            vardict = CIFAR100_models.__dict__            
        files = os.listdir(modeldir)
        files.sort()
        useful_model_list = [(file.split(splittoken)[0], os.path.join(modeldir, file)) for file in files]
    return useful_model_list 


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trainable_layers(start):
    layers = ["layer1", "layer2", "layer3", "layer4"]
    return list(filter(lambda x: int(x.split("layer")[-1]) >= start, layers))
        

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct/total, correct, total

def accuracy_and_loss(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0
    loss_sum = 0.0

    network.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x) 
            loss = F.cross_entropy(p, y).item()
            loss_sum += loss*len(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct/total, correct, loss_sum/total, loss_sum, total, 


def accuracy_batch(network, x, y):
    network.eval()
    with torch.no_grad():
        logit = network.predict(x)
        _, pred = logit.max(dim=1)
        correct = int(torch.sum(pred == y))
        total = x.shape[0]
    network.train()
    return correct / total, correct, total

                             
def get_model_results(network, dataloader, device):
    losses = []
    all_probs = []
    correct = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = network(images)
            all_probs.append(F.softmax(output, dim=1).cpu())
            preds = output.argmax(1).cpu()
            losses.append(F.cross_entropy(output, labels, reduction="none").cpu())
            # correct += (output.argmax(1).eq(labels).float()).sum().item()
            labels = labels.cpu()
            correct.append(preds == labels)
    return torch.cat(all_probs).numpy(), torch.cat(losses).numpy(), torch.cat(correct).numpy()

                          
def get_model_all_results(network, dataloader, device):
    losses = []
    all_probs = []
    all_logits = []
    all_preds = []
    all_labels = []
    correct = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = network(images)
            all_logits.append(output.cpu())
            all_probs.append(F.softmax(output, dim=1).cpu())
            preds = output.argmax(1).cpu()
            all_preds.append(preds)
            losses.append(F.cross_entropy(output, labels, reduction="none").cpu())
            # correct += (output.argmax(1).eq(labels).float()).sum().item()
            labels = labels.cpu()
            all_labels.append(labels)
            correct.append(preds == labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_probs).numpy(), torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), torch.cat(correct).numpy(), 


def get_model_results_noloss(network, dataloader, device):
    all_probs = []
    correct = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            output = network(images)
            all_probs.append(F.softmax(output, dim=1).cpu())
            preds = output.argmax(1).cpu()
            correct.append(preds == labels)
    return torch.cat(all_probs).numpy(), None, torch.cat(correct).numpy()

def get_features_labels(network, dataloaders: list[FastDataLoader], device):
    feature = []
    label = []
    network.eval()
    with torch.no_grad():
        if type(dataloaders) == list:
            for dataloader in dataloaders:
                for images, labels in tqdm(dataloader):
                    images = images.to(device)
                    labels = labels.to(device)
                    features = network(images)
                    feature.append(features)
                    label.append(labels)
        else:
            for images, labels in tqdm(dataloaders):
                images = images.to(device)
                labels = labels.to(device)
                features = network(images)
                feature.append(features)
                label.append(labels)
    return torch.cat(feature), torch.cat(label)


def get_model_feature(network, dataloader, device):
    features = []
    labelss = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = network(images)
            features.append(output)
            labelss.append(labels)
    return torch.cat(features), torch.cat(labelss)


def get_model_logits(network, dataloader, device):
    logits = []
    labelss = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            output = network(images)
            logits.append(output)
            labelss.append(labels)
    return torch.cat(logits).cpu().numpy(), torch.cat(labelss).cpu().numpy()

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
