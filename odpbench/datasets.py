# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import torch.utils.data as data
from torchvision import transforms, datasets
from PIL import Image, ImageFile
from os.path import join
from odpbench.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, Sequential_DataLoader
import numpy as np
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True


num_classes_dict = {
    "PACS": 7,
    "VLCS": 5,
    "OfficeHome": 65,
    "DomainNet": 345,
    "TerraInc": 10,
    "NICO": 60,
    # followings are the number of classes for the corresponding TRAINING SET
    "CIFAR-100": 100,
    "CIFAR-100-C": 100,
    "CIFAR-10-C": 10,
    "CIFAR-10.1": 10,
    "CIFAR-10.2": 10,
    "STL-10": 9,
    "CINIC-10": 10,
    "ImageNet": 1000,
    "ImageNet-V2": 1000,
    "ImageNet-S": 1000,
    "ImageNet-R": 200,
    "ImageNet-C": 1000,
    "ImageNet-A": 200,
    "ImageNet-R": 1000,
    "Tiny-ImageNet-C": 200,
    "ImageNet-Vid-Robust": 30,
    "objectnet-1.0": 313,
    "Waterbirds": 2,
    "CelebA": 2,
    "CheXpert": 2,
    "ImageNetBG": 9,
    "iwildcam": 182,
    "camelyon17": 2,
    "rxrx1": 1139,
    "fmow": 62,
    "amazon": 5,
    "civilcomments": 2,

    "NICO_DG": 60,
    "domain_net": 345,
}

checkpoint_step_dict = {
    "PACS": 300,
    "VLCS": 300,
    "OfficeHome": 300,
    "DomainNet": 1000,
    "TerraInc": 300,
    "NICO": 600,

    "Waterbirds": 300,
    "CelebA": 1500,
    "CheXpert": 1200,
    "ImageNetBG": 600,

    "iwildcam": 8114,
    "camelyon17": 9452,
    "rxrx1": 542,
    "fmow": 1201,
}


train_steps_dict = {
    "PACS": 5000,
    "VLCS": 5000,
    "OfficeHome": 5000,
    "TerraInc": 5000,
    "DomainNet": 15000,
    "NICO": 10000,

    "Waterbirds": 5000,
    "CelebA": 30000,
    "CheXpert": 20000,
    "ImageNetBG": 10000,

    "iwildcam": 8114 * 12,
    "camelyon17": 9452 * 5,
    "rxrx1": 542 * 90,
    "fmow": 1201 * 50,
}

train_batch_size_dict = {
    "iwildcam": 16,
    "camelyon17": 32,
    "rxrx1": 75,
    "fmow": 64,
}

def _dataset_info(txt_file):
    with open(txt_file, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(' '.join(row[:-1]))
        labels.append(int(row[-1]))

    return file_names, labels


class StandardDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
    
    def get_image(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)
    
class AugmentDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self._image_transformer = img_transformer
        
        # 将所有变换放入列表中供随机选择
        self.augmentations = [
                                T.RandAugment(num_ops=3, magnitude=15), 
                                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                T.RandomErasing(
                                    p=1.0,
                                    scale=(0.0, 0.33),
                                    ratio=(1/3, 10/3),
                                    value='random'), 
                                T.Compose([
                                        T.RandomHorizontalFlip(p=0.5),
                                        T.RandomResizedCrop(size=(224, 224), scale=(0., 0.92), ratio=(3/4, 4/3))
                                    ])]

        # 标准化操作
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    def get_image(self, index):
        import torch
        img = Image.open(self.names[index]).convert('RGB')
        img_fl = self._image_transformer(img)
        img_int = (img_fl * 255).to(torch.uint8)
        aug_img = random.choice(self.augmentations)(img_int)
        img_fl = aug_img.to(torch.float32) / 255
        input = self.normalize(img_fl)
        return input
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

class NumpyDataset(data.Dataset):
    def __init__(self, data, labels, img_transformer=None):
        self.data = data
        self.labels = labels
        self.N = len(self.data)
        self._image_transformer = img_transformer

    def get_image(self, index):
        img = Image.fromarray(self.data[index])
        return self._image_transformer(img)
    
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])
    
    def __len__(self):
        return len(self.data)
    
def get_train_transformer(): # hard-coded
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transformer(): # hard-coded
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def get_data_transformer(mode):
    assert mode in ["train", "eval"]
    if mode == "train":
        return get_train_transformer()
    else:
        return get_val_transformer()

def get_train_transformer_soft(shape):
    return transforms.Compose([
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transformer_soft(shape): # soft-coded
    return transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_data_transformer_soft(mode, shape):
    assert mode in ["train", "eval"]
    if mode == "train":
        return get_train_transformer_soft(shape)
    else:
        return get_val_transformer_soft(shape)
    
def _dataset_info_numpy(dataset, domain):
    common_dir = "/mnt/SHARED/hanyu/dataset"
    if dataset == "CIFAR-10-C" or dataset == "CIFAR-100-C":
        if domain == "TEST":
            dir = join(common_dir, dataset)
            import os
            files = os.listdir(dir)
            data_list = []
            labels_list = []
            labels = np.load(join(dir, "labels.npy"))
            for file in files:
                if file.endswith(".npy") and file != "labels.npy":
                    data = np.load(join(dir, file))
                    data_list.append(data)
                    labels_list.append(labels)
            data = np.concatenate(data_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
        else:
            data = np.load(join(common_dir, dataset, f"{domain}.npy"))
            labels = np.load(join(common_dir, dataset, "labels.npy"))
    elif dataset == "CIFAR-10.1":
        dir = "CIFAR-10.1/datasets"
        if domain == "v4":
            data = np.load(join(common_dir, dir, "cifar10.1_v4_data.npy"))
            labels = np.load(join(common_dir, dir, "cifar10.1_v4_labels.npy"))
        elif domain == "v6":
            data = np.load(join(common_dir, dir, "cifar10.1_v6_data.npy"))
            labels = np.load(join(common_dir, dir, "cifar10.1_v6_labels.npy"))
        elif domain == "TEST":
            data1 = np.load(join(common_dir, dir, "cifar10.1_v4_data.npy"))
            labels1 = np.load(join(common_dir, dir, "cifar10.1_v4_labels.npy"))
            data2 = np.load(join(common_dir, dir, "cifar10.1_v6_data.npy"))
            labels2 = np.load(join(common_dir, dir, "cifar10.1_v6_labels.npy"))
            data = np.concatenate([data1, data2], axis=0)
            labels = np.concatenate([labels1, labels2], axis=0)
    elif dataset == "CIFAR-10.2":
        if domain == "TEST":
            Data1 = np.load(join(common_dir, "CIFAR-10.2", "cifar102_train.npz"))
            data1 = Data1["images"]
            labels1 = Data1["labels"]
            Data2 = np.load(join(common_dir, "CIFAR-10.2", "cifar102_test.npz"))
            data2 = Data2["images"]
            labels2 = Data2["labels"]
            data = np.concatenate([data1, data2], axis=0)
            labels = np.concatenate([labels1, labels2], axis=0)
        else:
            Data = np.load(join(common_dir, "CIFAR-10.2", f"cifar102_{domain}.npz"))
            data = Data["images"]
            labels = Data["labels"]
    elif dataset == "STL-10":
        label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:-1, 9:7, 10:8}
        with open(join(common_dir, "STL-10",f"{domain}_y.bin"), 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
        labels = np.array([label_dict[label] for label in labels])
        with open(join(common_dir, "STL-10",f"{domain}_X.bin"), 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(-1, 3, 96, 96)
        data = np.transpose(data, (0, 3, 2, 1))
        remove_indices = np.where(labels == -1)[0]
        labels = np.delete(labels, remove_indices)
        data = np.delete(data, remove_indices, axis=0)
    else:
        raise NotImplementedError
    
    return data, labels

def get_dataloader_numpy(txtdir, dataset, domain, batch_size, mode="train", split=True, holdout_fraction=0.2, num_workers=8, seed=1, seq=False, preprocess=None):
    # For CIFAR datasets
    assert mode in ["train", "eval"] # used to distinguish 
    loader_func = {"train": InfiniteDataLoader, "eval": Sequential_DataLoader if seq else FastDataLoader}
    shape = (32, 32)
    if dataset == "CIFAR-10": # Val
        cifar10_val = datasets.CIFAR10(root='/mnt/SHARED/hanyu/dataset/CIFAR10', train=False, download=False)
        data, labels = cifar10_val.data, np.array(cifar10_val.targets)
    elif dataset == "CIFAR-100": # Val
        cifar100_val = datasets.CIFAR100(root='/mnt/SHARED/hanyu/dataset/CIFAR100', train=False, download=False)
        data, labels = cifar100_val.data, np.array(cifar100_val.targets)
    else: # Test
        #  if "STL" in dataset:
            # shape = (96, 96)
        data, labels = _dataset_info_numpy(dataset, domain)
    if split:
        idxs = np.arange(len(data))
        np.random.RandomState(seed).shuffle(idxs)
        mid = int(len(idxs)*(1-holdout_fraction))
        idxs_dict = {"train": idxs[:mid], "eval": idxs[mid:]}
        loader_dict = dict()
        if mode == "eval":
            loader_func["train"] = FastDataLoader
        for key, idxs in idxs_dict.items():
            data_split = [data[idx] for idx in idxs]
            labels_split = [labels[idx] for idx in idxs]
            if mode == "eval":
                img_tr = get_data_transformer_soft(mode, shape=shape)
            else:
                img_tr = get_data_transformer_soft(key, shape=shape)
            dataset_split = NumpyDataset(data_split, labels_split, img_tr)
            loader = loader_func[key](dataset=dataset_split, batch_size=batch_size, num_workers=num_workers)
            loader_dict[key] = loader
        return loader_dict
    else:
        if preprocess is None:
            img_tr = get_data_transformer_soft(mode, shape=shape)
        else:
            img_tr = preprocess
        curDataset = NumpyDataset(data, labels, img_tr)
        loader = loader_func[mode](dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
        return loader

def get_dataloader(txtdir, dataset, domain, batch_size, mode="train", split=True, holdout_fraction=0.2, num_workers=8, seed=1, seq = False, preprocess=None):
    assert mode in ["train", "eval"] # used to distinguish 
    loader_func = {"train": InfiniteDataLoader, "eval": Sequential_DataLoader if seq else FastDataLoader}
    shape = (224, 224)
    if "CINIC" in dataset:
        shape = (32, 32)
    names, labels = _dataset_info(join(txtdir, dataset, "%s.txt"%domain))
    if split:
        idxs = np.arange(len(names))
        np.random.RandomState(seed).shuffle(idxs)
        mid = int(len(idxs)*(1-holdout_fraction))
        idxs_dict = {"train": idxs[:mid], "eval": idxs[mid:]}
        loader_dict = dict()
        if mode == "eval":
            loader_func["train"] = FastDataLoader
        for key, idxs in idxs_dict.items():
            names_split = [names[idx] for idx in idxs]
            labels_split = [labels[idx] for idx in idxs]
            if mode == "eval":
                img_tr = get_data_transformer_soft(mode, shape=shape)
            else:
                img_tr = get_data_transformer_soft(key, shape=shape)
            dataset_split = StandardDataset(names_split, labels_split, img_tr)
            loader = loader_func[key](dataset=dataset_split, batch_size=batch_size, num_workers=num_workers)
            loader_dict[key] = loader
        return loader_dict
    else:
        if preprocess is None:
            img_tr = get_data_transformer_soft(mode, shape=shape)
        else:
            img_tr = preprocess
        curDataset = StandardDataset(names, labels, img_tr)
        loader = loader_func[mode](dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
        return loader

def get_one_test_dataloader(txtdir, dataset, domains, batch_size, num_workers=8):
    datasets = []
    for domain in domains:
        names, labels = _dataset_info(join(txtdir, dataset, "%s.txt"%domain))
        print(join(txtdir, dataset, "%s.txt"%domain))
        
        img_tr = get_data_transformer("eval")
        curDataset = StandardDataset(names, labels, img_tr)
        datasets.append(curDataset)
    dataset = data.ConcatDataset(datasets)
    loader = FastDataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    return loader


def get_aug_transformer(): # hard-coded
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_aug_dataloader(txtdir, dataset, domain, batch_size, num_workers = 8):
    loader_func = {"train": InfiniteDataLoader, "eval": FastDataLoader}
    names, labels = _dataset_info(join(txtdir, dataset, "%s.txt"%domain))

    
    img_tr = get_aug_transformer()
    curDataset = AugmentDataset(names, labels, img_tr)
    loader = loader_func["eval"](dataset=curDataset, batch_size=batch_size, num_workers=num_workers)
    return loader

def get_mix_dataloader(txtdir, dataset, domains, phase, batch_size, num_workers=8):
    assert phase == "train"
    img_tr = get_train_transformer()
    concat_list = []
    for domain in domains:
        names, labels = _dataset_info(join(txtdir, dataset, "%s_%s.txt"%(domain, phase)))
        curDataset = StandardDataset(names, labels, img_tr)
        concat_list.append(curDataset)
    finalDataset = data.ConcatDataset(concat_list)
    loader = InfiniteDataLoader(dataset=finalDataset, weights=None, batch_size=batch_size, num_workers=num_workers)
    return loader

def get_mix_dataloader2(dataloaders, batch_size, num_workers=8):
    concat_list = []
    for dataloader in dataloaders:
        concat_list.append(dataloader.dataset)
    finalDataset = data.ConcatDataset(concat_list)
    loader = InfiniteDataLoader(dataset=finalDataset, weights=None, batch_size=batch_size, num_workers=num_workers)
    return loader

class FilteredDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
    def __iter__(self):
        # 迭代原始 dataloader，并过滤掉 metadata
        for batch in iter(self.dataloader):
            x, y, *_ = batch  # 假设 batch 是 (x, y, metadata)
            yield x, y

    def __len__(self):
        return len(self.dataloader)
    
class CombinedDataLoader:
    """Combine multiple FastDataLoaders into a single DataLoader using zip."""
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        # Use zip to iterate over the dataloaders in parallel
        train_minibatches_iterator = zip(*[iter(dataloader) for dataloader in self.dataloaders])
        for minibatches in train_minibatches_iterator:
            yield minibatches  # A tuple of batches from each DataLoader

    def __len__(self):
        # Length is the length of the shortest DataLoader
        return min(len(dataloader) for dataloader in self.dataloaders)