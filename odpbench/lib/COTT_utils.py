from tqdm import tqdm
import torch
import torch.nn as nn
import time
import ot
from odpbench.datasets import num_classes_dict

def sample_label_dist2(n_class, sample_size, dataset):
            import random
            # if "CIFAR" in dataset or "CINIC" in dataset or "STL" in dataset or "ImgaeNet" in dataset or "objectnet" in dataset:
            dist = [1 / n_class] * n_class
            

            # print(f'Label distribution: {dist}')
            
            labels = sum([[i] * int(dist[i] * sample_size) for i in range(num_classes_dict[dataset])], [])
            
            remainder = sample_size - len(labels)
            
            r_labels = random.choices(
                list(range(n_class)), 
                weights=dist, 
                k=remainder
            )
            
            labels = labels + r_labels
            
            return torch.as_tensor(labels)

def sample_label_dist(n_class, sample_size, dataloaders, dataset):
            import random
            if "CIFAR" in dataset or "CINIC" in dataset or "STL" in dataset or "ImageNet" in dataset or "objectnet" in dataset:
                dist = [1 / n_class] * n_class
            else:
                dist = []
                labels = []
                for dataloader in dataloaders:
                    for _, ys in dataloader:
                        labels.append(ys)
                labels = torch.cat(labels)
                label_counts = torch.bincount(labels)
                for value, count in enumerate(label_counts):
                    if value >= n_class:
                        break
                    dist.append(count / len(labels))

                # print(f'Label distribution: {dist}')
            labels = sum([[i] * int(dist[i] * sample_size) for i in range(n_class)], [])
            
            remainder = sample_size - len(labels)
            
            r_labels = random.choices(
                list(range(n_class)), 
                weights=dist, 
                k=remainder
            )
            
            labels = labels + r_labels
            
            return torch.as_tensor(labels)


def compute_cott(net, iid_loaders, dataset):
    net.eval()
    softmax_vecs = []
    preds, tars = [], []
    with torch.no_grad():
        for iid_loader in iid_loaders:
            for items in tqdm(iid_loader):
                inputs, targets = items[0], items[1]
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                _, prediction = outputs.max(1)

                preds.extend( prediction.tolist() )
                tars.extend( targets.tolist() )
                softmax_vecs.append( nn.functional.softmax(outputs, dim=1).cpu() )
    
    preds, tars  = torch.as_tensor(preds), torch.as_tensor(tars)
    softmax_vecs = torch.cat(softmax_vecs, dim=0)
    target_vecs = nn.functional.one_hot(tars, num_classes=num_classes_dict[dataset])
    
    max_n = 10000
    if len(target_vecs) > max_n:
        print(f'sampling {max_n} out of {len(target_vecs)} validation samples...')
        torch.manual_seed(0)
        rand_inds = torch.randperm(len(target_vecs))
        tars = tars[rand_inds][:max_n]
        preds = preds[rand_inds][:max_n]
        target_vecs = target_vecs[rand_inds][:max_n]
        softmax_vecs = softmax_vecs[rand_inds][:max_n]

    print('computing assignment...')
    M = torch.cdist(target_vecs.float(), softmax_vecs, p=1)
    
    start = time.time()
    weights = torch.as_tensor([])
    Pi = ot.emd(weights, weights, M, numItermax=1e8)

    print(f'done. {time.time() - start}s passed')

    costs = ( Pi * M.shape[0] * M ).sum(1) * -1
        
    n_incorrect = preds.ne(tars).sum()
    t = torch.sort( costs )[0][n_incorrect - 1].item()
        
    return t


def compute_cott2(num_classes, iid_probs, iid_preds, iid_labels):
    # net.eval()
    # softmax_vecs = []
    # preds, tars = [], []
    # with torch.no_grad():
    #     for iid_loader in iid_loaders:
    #         for items in tqdm(iid_loader):
    #             inputs, targets = items[0], items[1]
    #             inputs, targets = inputs.cuda(), targets.cuda()
    #             outputs = net(inputs)
    #             _, prediction = outputs.max(1)

    #             preds.extend( prediction.tolist() )
    #             tars.extend( targets.tolist() )
    #             softmax_vecs.append( nn.functional.softmax(outputs, dim=1).cpu() )
    preds, tars = torch.as_tensor(iid_preds), torch.as_tensor(iid_labels)
    # preds, tars  = torch.as_tensor(preds), torch.as_tensor(tars)
    # softmax_vecs = torch.cat(softmax_vecs, dim=0)
    softmax_vecs = torch.as_tensor(iid_probs)
    # print(num_classes, tars.unique())
    # raise TypeError
    target_vecs = nn.functional.one_hot(tars, num_classes=tars.unique().shape[0]).float()
    
    max_n = 10000
    if len(target_vecs) > max_n:
        print(f'sampling {max_n} out of {len(target_vecs)} validation samples...')
        torch.manual_seed(0)
        rand_inds = torch.randperm(len(target_vecs))
        tars = tars[rand_inds][:max_n]
        preds = preds[rand_inds][:max_n]
        target_vecs = target_vecs[rand_inds][:max_n]
        softmax_vecs = softmax_vecs[rand_inds][:max_n]

    print('computing assignment...')
    M = torch.cdist(target_vecs.float(), softmax_vecs, p=1)
    
    start = time.time()
    weights = torch.as_tensor([])
    Pi = ot.emd(weights, weights, M, numItermax=1e8)

    print(f'done. {time.time() - start}s passed')

    costs = ( Pi * M.shape[0] * M ).sum(1) * -1
        
    n_incorrect = preds.ne(tars).sum()
    t = torch.sort( costs )[0][n_incorrect - 1].item()
        
    return t
