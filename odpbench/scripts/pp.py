import argparse
import os
import ot
import sys
import math
import random
from tqdm import tqdm
import numpy as np
import PIL
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import pandas as pd

from odpbench.datasets import *
from odpbench.lib import misc, ATC_helper
from odpbench.lib.Logger import ppLogger
from odpbench import networks

os.environ['TORCH_HOME'] = "/mnt/SHARED/hanyu/torch"
WILDS_DATA = "/mnt/SHARED/hanyu/dataset"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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



def get_dataloaders(args, logger, batch_size=128, seq = False):
    # set up dataloaders
    trainloaders = []
    valloaders = []
    for domain in args.source:
        loader_dict = get_dataloader(args.txtdir, args.dataset, domain, batch_size, mode="eval", split=True, holdout_fraction=args.holdout_fraction, seed=args.data_seed, seq=seq)
        trainloaders.append(loader_dict["train"])
        valloaders.append(loader_dict["eval"])
    if args.dataset in ["DomainNet", "NICO"]:
        testloaders = [get_one_test_dataloader(args.txtdir, args.dataset, args.target, batch_size)]
        args.target = ["-".join(args.target)]
    else:
        testloaders = [get_dataloader(args.txtdir, args.dataset, domain, batch_size, mode="eval", split=False, seed=args.data_seed, seq=seq) for domain in args.target]

    for index, domain in enumerate(args.source):
        logger.info("Train %s size: %d" % (domain, len(trainloaders[index].dataset)))
    for index, domain in enumerate(args.source):
        logger.info("Val %s size: %d" % (domain, len(valloaders[index].dataset)))
    for index, domain in enumerate(args.target):
        logger.info("Test %s size: %d" % (domain, len(testloaders[index].dataset)))
    
    dataloaders = dict()
    dataloaders["train"] = trainloaders
    dataloaders["val"] = valloaders
    dataloaders["test"] = testloaders
    return dataloaders

def get_wilds_dataloaders(args):
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader
    from odpbench.datasets import FilteredDataLoader
    import torchvision.transforms as transforms
    dataset = get_dataset(dataset=args.dataset, root_dir=WILDS_DATA)
    # wilds_datasets = ['iwildcam', 'camelyon17', 'fmow', 'rxrx1']
    if args.dataset == "iwildcam":
        train_data = dataset.get_subset(
            "train",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),
        )
    elif args.dataset == "camelyon17":
        train_data = dataset.get_subset(
            "train",
            transform=transforms.Compose(
                [transforms.Resize((96, 96)), transforms.ToTensor()]
            ),
        )
    elif args.dataset in ["rxrx1", "fmow"]:
        train_data = dataset.get_subset(
            "train",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    else:
        raise NotImplementedError

    train_loader = get_train_loader("standard", train_data, batch_size=train_batch_size_dict[args.dataset])
    train_loader = FilteredDataLoader(train_loader)
    try:
        val_data = dataset.get_subset(
            "id_val",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    except:
        val_data = dataset.get_subset(
            "id_test",
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        )
    val_loader = get_eval_loader("standard", val_data, batch_size=128)
    val_loader = FilteredDataLoader(val_loader)
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    # Prepare the evaluation data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=128)
    test_loader = FilteredDataLoader(test_loader)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    for index, domain in enumerate(args.source):
        logger.info("Train %s size: %d" % (domain, len(train_loaders[index].dataset)))
    for index, domain in enumerate(args.source):
        logger.info("Val %s size: %d" % (domain, len(val_loaders[index].dataset)))
    for index, domain in enumerate(args.target):
        logger.info("Test %s size: %d" % (domain, len(test_loaders[index].dataset)))
    dataloaders = dict()
    dataloaders["train"] = train_loaders
    dataloaders["val"] = val_loaders
    dataloaders["test"] = test_loaders
    return dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--txtdir', type=str, default="/mnt/SHARED/hanyu/dataset/txtlist-pp")
    parser.add_argument('--ckpt_dir', type=str, default="/mnt/SHARED/hanyu/Performance_Prediction/checkpoint")
    parser.add_argument('--ckpt_idx', type=int, default=0, help="for ImageNet and CIFAR models")
    parser.add_argument('--arch', default="resnet50")
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="NICO")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--target", nargs="+")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--algorithm', type=str, default="ATC")
    parser.add_argument('--ATC_type', type=str, default="conf", choices=["conf", "entropy"])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_seed', type=int, default=0, help='used for seeding split_dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="logs")

    args = parser.parse_args()
    misc.setup_seed(args.seed)

    logger = ppLogger(args)
    
    logger.info("Environment:")
    logger.info("\t`P`ython: {}".format(sys.version.split(" ")[0]))
    logger.info("\tPyTorch: {}".format(torch.__version__))
    logger.info("\tTorchvision: {}".format(torchvision.__version__))
    logger.info("\tCUDA: {}".format(torch.version.cuda))
    logger.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.info("\tNumPy: {}".format(np.__version__))
    logger.info("\tPIL: {}".format(PIL.__version__))

    logger.info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.info('\t{}: {}'.format(k, v))
    mapped_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "imagenet" in args.dataset.lower() or args.dataset == "objectnet-1.0":
        base_dataset = "ImageNet"
    elif "CIFAR-100" in args.dataset:
        base_dataset = "CIFAR-100"
    elif "CIFAR-10" in args.dataset or args.dataset in ["CINIC-10", "STL-10"]:
        base_dataset = "CIFAR-10"
    else:
        base_dataset = None



    if base_dataset is not None:
        model_list = misc.get_model_list(dataset=base_dataset)
        if base_dataset == "ImageNet" and args.ckpt_idx in [47, 48, 105, 108, 97, 17]:
            raise NotImplementedError
        if base_dataset == "CIFAR-10" and args.ckpt_idx in [48, 49, 50]:
            raise NotImplementedError
        model_name, weight = model_list[args.ckpt_idx]
        dataloaders = dict()
        if base_dataset == "ImageNet":
            model_constructor = getattr(models, model_name)
            model = model_constructor(weights=weight)
            preprocess = weight.transforms()
            dataloaders["val"] = [get_dataloader(args.txtdir, base_dataset, "VAL", args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess, seq=True),]
            dataloaders["test"] = [get_dataloader(args.txtdir, args.dataset, domain, args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess, seq=True) for domain in args.target]
            # Class Mapping
            if args.dataset in ["Tiny-ImageNet-C", "ImageNet-R", "ImageNet-A", "ImageNet-Vid-Robust", "objectnet-1.0"]:
                mapping = pd.read_csv(f"{args.txtdir}/{args.dataset}/mapping.csv")
                mapping = {row["real_label"]: row["label"] for _, row in mapping.iterrows()}
                mapped_model = nn.Sequential(model, ClassMappingLayer(mapping))    
        else:
            cifar_model_dict = misc.get_cifar_model_dict(dataset=base_dataset, cifar_checkpoint_dir=args.ckpt_dir)
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
            dataloaders["val"] = [get_dataloader_numpy(args.txtdir, base_dataset, "VAL", args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess),]
            dataloaders["test"] = [loader_func(args.txtdir, args.dataset, domain, args.batch_size, mode="eval", split=False, seed=args.data_seed, preprocess=preprocess) for domain in args.target]
        if mapped_model:
            model = mapped_model
    else:
        if args.dataset in ["fmow", "iwildcam", "camelyon17", "rxrx1"]:
            dataloaders = get_wilds_dataloaders(args)
        else:
            dataloaders = get_dataloaders(args, logger, batch_size=args.batch_size) # keys: train, val, test
        if args.dataset in ["PACS", "VLCS", "OfficeHome", "TerraInc", "DomainNet", "NICO"] or \
            args.dataset in ["Waterbirds", "CelebA", "CheXpert"]:
            def load_checkpoint(args):
                if args.dataset in ["PACS", "VLCS", "OfficeHome", "TerraInc", "DomainNet", "NICO"]:
                    checkpoint_path = f"{args.ckpt_dir}/{args.dataset}/{args.dataset}_{args.target[0]}_{args.arch}_{args.pretrain}/model_best_seed{args.seed}.pkl"
                else:
                    checkpoint_path = f"{args.ckpt_dir}/{args.dataset}/{args.dataset}_test_{args.arch}_{args.pretrain}/model_best_seed{args.seed}.pkl"
                checkpoint = torch.load(checkpoint_path, map_location="cpu") 
                saved_state_dict = checkpoint["model_dict"]
                from odpbench.networks import Identity
                if args.arch == "resnet50":
                    if args.pretrain == "CLIP":
                        import clip
                        model, preprocess = clip.load("RN50", device="cpu")
                        model = model.visual
                    else:
                        model = torchvision.models.resnet50(pretrained=False)
                        del model.fc
                        model.fc = Identity()
                    classfier = networks.Classifier(1024 if args.pretrain == "CLIP" else 2048, num_classes_dict[args.dataset])
                    model = nn.Sequential(model, classfier)
            
                elif args.arch == "vitb16":
                    if args.pretrain == "CLIP":
                        import clip
                        model, preprocess = clip.load("ViT-B/16", device="cpu")
                        model = model.visual
                    elif args.pretrain == "MoCo-v3":
                        from odpbench.lib import vits_moco
                        model = vits_moco.vit_base()
                        del model.head
                        model.head = Identity()
                    else:
                        model = models.vit_b_16(weights=None)  # 不加载默认权重

                        if "heads" in dir(model):
                            del model.heads
                            model.heads = Identity()
                    classfier = networks.Classifier(512 if args.pretrain == "CLIP" else 768, num_classes_dict[args.dataset])
                    model = nn.Sequential(model, classfier)
                else:
                    raise ValueError(f"Model arch not supported: {args.arch}")

                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for k, v in saved_state_dict.items():
                    # 忽略 "featurizer." 前缀的键
                    if k.startswith("featurizer.") or k.startswith("classifier."):
                        continue

                    if k.startswith("network.0."):
                        new_key = k.replace("network.0.network.", "0.")  
                    elif k.startswith("network.1."):
                        new_key = k.replace("network.1.", "1.")  
                    else:
                        print(k)
                        raise TypeError

                    if new_key in model.state_dict():
                        new_state_dict[new_key] = v
                    else:
                        print(new_key)
                        print(model.state_dict().keys())
                        raise TypeError
                model.load_state_dict(new_state_dict)
                return model, checkpoint["args"], checkpoint["model_hparams"]
            model, loaded_args, loaded_hparams = load_checkpoint(args)

        elif args.dataset in ["fmow", "iwildcam", "camelyon17", "rxrx1"]:
            from odpbench.networks import get_wilds_model
            model = get_wilds_model(args)
        else:
            raise ValueError(f"Dataset not supported: {args.dataset}")
    model.to(device)
    model.eval()
    
    if args.algorithm == "ATC":
        # find ATC threshold
        val_probs = []
        val_corrects = []
        for index, domain in enumerate(args.source):
            probs, _, correct = misc.get_model_results(model, dataloaders["val"][index], device ) # here "correct" represents an array of whether model predicts correctly
            val_probs.append(probs)
            val_corrects.append(correct)
        val_probs = np.concatenate(val_probs, axis=0)
        val_corrects = np.concatenate(val_corrects, axis=0)
        acc = np.mean(correct)
        print(acc)

        val_scores = ATC_helper.get_max_conf(val_probs) if args.ATC_type == "conf" else ATC_helper.get_entropy(val_probs)
        _, ATC_thres = ATC_helper.find_ATC_threshold(val_scores, val_corrects)
        logger.info("ATC Threshold is %.4f" % ATC_thres)        
        # start pp
        for phase in ["val", "test"]:
            domains = args.target if phase == "test" else args.source
            for index, domain in enumerate(domains):
                probs, _, correct = misc.get_model_results(model, dataloaders[phase][index], device ) 
                scores = ATC_helper.get_max_conf(probs) if args.ATC_type == "conf" else ATC_helper.get_entropy(probs)
                acc_predict_atc = ATC_helper.get_ATC_acc(ATC_thres, scores)                
                logger.info("Predict accuracy using ATC for %s %s: %.4f" % (phase, domain, acc_predict_atc))
                logger.info(f", true accuracy: {correct.astype(float).mean()}")
                print("Predict accuracy using ATC for %s %s: %.4f" % (phase, domain, acc_predict_atc), f", true accuracy: {correct.astype(float).mean()}")
    elif args.algorithm == "DOC":
        from odpbench.lib.DoC_utils import get_DoC, get_DoE
        source_probs = []
        val_corrects = []
        for index, domain in enumerate(args.source):
            probs, _, correct = misc.get_model_results(model, dataloaders["val"][index], device ) # here "correct" represents an array of whether model predicts correctly
            source_probs.append(probs)
            val_corrects.append(correct)
        source_probs = np.concatenate(source_probs, axis=0)
        val_corrects = np.concatenate(val_corrects, axis=0)

        # no validation, because DOC and DOE based on the outputs from source domain
        for phase in ["test"]:
            domains = args.target if phase == "test" else args.source
            for index, domain in enumerate(domains):
                test_probs, _, correct = misc.get_model_results(model, dataloaders[phase][index], device ) 
                DoC = get_DoC(source_probs, test_probs)
                DoE = get_DoE(source_probs, test_probs)
                logger.info(f"In domain: {domain}, DoC value: {DoC:.4f}, DoE value: {DoE:.6f}")
                logger.info(f"Source accuracy: {val_corrects.astype(float).mean():.4f}")
                logger.info(f"Target accuracy: {correct.astype(float).mean():.4f}")
                logger.info(f"Accuracy Gap: {val_corrects.astype(float).mean() - correct.astype(float).mean():.4f}")
                print(f"In domain: {domain}, DoC value: {DoC:.4f}, DoE value: {DoE:.6f}")
                print(f"Source accuracy: {val_corrects.astype(float).mean():.4f}")
                print(f"Target accuracy: {correct.astype(float).mean():.4f}")
                print(f"Accuracy Gap: {val_corrects.astype(float).mean() - correct.astype(float).mean():.4f}")

    elif args.algorithm == "NuclearNorm":
        
        from odpbench.lib.Nuno_utils import get_nuno
        from scipy.special import logit
        for phase in ["val", "test"]:
            domains = args.target if phase == "test" else args.source
            for index, domain in enumerate(domains):
                test_probs, _, correct = misc.get_model_results(model, dataloaders[phase][index], device ) 
                nuno = get_nuno(test_probs)
                acc = np.mean(correct)
                nuno_mapped = logit(nuno)
                acc_mapped = logit(acc)
                logger.info(f"\nIn domain{domain}, Before Prob Axis Scaling: Nuclear Norm value: {nuno}, accuracy: {np.mean(correct)}\nAfter Prob Axis Scaling: Nuclear Norm value: {nuno_mapped}, accuracy: {acc_mapped}")
                print(f"\nIn domain{domain}, Before Prob Axis Scaling: Nuclear Norm value: {nuno}, accuracy: {np.mean(correct)}\nAfter Prob Axis Scaling: Nuclear Norm value: {nuno_mapped}, accuracy: {acc_mapped}")
    elif args.algorithm == "NeighborInvariance":        
        from odpbench.lib.Neibor_Invariance import get_augmented_model_results
        if base_dataset is None:
            input_size = (224, 224)
        elif base_dataset == "ImageNet":
            input_size = (preprocess.crop_size[0], preprocess.crop_size[0])
        else:
            input_size = (32, 32)
        for index, domain in enumerate(args.target):
            # per augment
            test_probs, _, correct = misc.get_model_results(model, dataloaders["test"][index], device)
            ori_pred = np.argmax(test_probs, axis = 1)
            ori_preds, aug_preds, labels = get_augmented_model_results(model, dataloaders["test"][index], base_dataset, device, size=input_size)
            pred_acc = (ori_preds == aug_preds).to(torch.float32).mean().item()
            acc = correct.mean()
            logger.info("Predict accuracy using Neibor Invariance for %s: %.4f, while the groundtruth accuracy is %.4f" % (domain, pred_acc, acc))
            print("Predict accuracy using Neibor Invariance for %s: %.4f, while the groundtruth accuracy is %.4f" % (domain, pred_acc, acc))
    elif args.algorithm == "COT":        
        from odpbench.lib.COTT_utils import sample_label_dist
        for index, domain in enumerate(args.target):
            labels = []
            num_classes = num_classes_dict[args.dataset]
            ood_acts, _, correct = misc.get_model_results(model, dataloaders["test"][index], device) 

            ood_acts = torch.tensor(ood_acts)
            n_test_sample = len(dataloaders["test"][index].dataset)
            batch_size = min(10000, n_test_sample)
            n_batch = math.ceil( n_test_sample / batch_size)

            if n_batch > 1:
                est = 0
                for _ in tqdm(range(n_batch)):
                    rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                    
                    iid_acts_batch = nn.functional.one_hot(
                        sample_label_dist(num_classes, batch_size, dataloaders["val"], args.dataset), num_classes=num_classes_dict[args.dataset]
                    )

                    ood_acts_batch = ood_acts[rand_inds]
                    
                    M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                    weights = torch.as_tensor([])
                    est += ( ot.emd2(weights, weights, M, numItermax=1e8, numThreads=8) / 2 ).item()
                est = est / n_batch
            else:
                iid_acts = nn.functional.one_hot(
                    sample_label_dist(num_classes, len(ood_acts), dataloaders["val"], args.dataset), num_classes=num_classes_dict[args.dataset]
                )
                
                M = torch.cdist(iid_acts.float(), ood_acts, p=1) / 2
                weights = torch.as_tensor([])
                Pi = ot.emd(weights, weights, M, numItermax=1e8)

                costs = ( Pi * M.shape[0] * M ).sum(1)
                est = costs.mean().item()
            logger.info(f"In domain {domain}, COT value: {(1 - est):.4f}, real accuracy: {correct.astype(float).mean()}")
            print(f"In domain {domain}, COT value: {(1 - est):.4f}, real accuracy: {correct.astype(float).mean()}")
    
    elif args.algorithm == "COTT":
        # should only use validation data, but too few of them
        from odpbench.lib.COTT_utils import compute_cott,sample_label_dist
        
        threshold = compute_cott(model, dataloaders["val"], args.dataset)

        num_classes = num_classes_dict[args.dataset]
        for index, domain in enumerate(args.target):
            ood_acts, _, correct = misc.get_model_results(model, dataloaders["test"][index], device) 

            ood_acts = torch.tensor(ood_acts)

            n_test_sample = len(dataloaders["test"][index].dataset)
            batch_size = min(10000, n_test_sample)
            n_batch = math.ceil( n_test_sample / batch_size)

            if n_batch > 1:
                est = 0
                for _ in tqdm(range(n_batch)):
                    rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                    ood_acts_batch = ood_acts[rand_inds]

                    iid_acts_batch = nn.functional.one_hot(
                        sample_label_dist(num_classes, batch_size, dataloaders["val"], args.dataset), num_classes=num_classes_dict[args.dataset]
                    )
                    
                    M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                    
                    weights = torch.as_tensor([])
                    Pi = ot.emd(weights, weights, M, numItermax=1e8)
                    
                    costs = ( Pi * M.shape[0] * M ).sum(1) * -1
                    
                    est = est + (costs < threshold).sum().item() / batch_size
                    # cost_dist.append(costs)
                
                est = est / n_batch
                # cost_dist = torch.sort(torch.cat(cost_dist, dim=0))[0].tolist()
            else:
                iid_acts = nn.functional.one_hot(
                    sample_label_dist(num_classes, n_test_sample, dataloaders["val"], args.dataset), num_classes=num_classes_dict[args.dataset]
                )
                
                M = torch.cdist(iid_acts.float(), ood_acts, p=1)
                
                weights = torch.as_tensor([])
                Pi = ot.emd(weights, weights, M, numItermax=1e8)
                
                costs = ( Pi * M.shape[0] * M ).sum(1) * -1

                est = (costs < threshold).sum().item() / batch_size
            logger.info(f"In domain {domain}, COTT value: {(1 - est):.4f}, real accuracy: {correct.astype(float).mean()}")
            print(f"In domain {domain}, COTT value: {(1 - est):.4f}, real accuracy: {correct.astype(float).mean()}")

    elif args.algorithm == "MaNo":
        for index, domain in enumerate(args.target):
            probs, _, correct = misc.get_model_results(model, dataloaders["test"][index], device) 
            
            from odpbench.lib.MaNo_utils import uniform_cross_entropy, MaNo_evaluate
            delta = uniform_cross_entropy(model, dataloaders["test"][index], num_classes_dict[args.dataset], device)
            
            MaNo_Score = MaNo_evaluate(model, dataloaders["test"][index], 4, device, delta)
            logger.info(f"In domain: {domain}, MaNo Score: {MaNo_Score}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, MaNo Score: {MaNo_Score}, true accuracy: {correct.astype(float).mean()}")

    elif args.algorithm == "Dispersion":
        from odpbench.lib.Dispersion_utils import dispersion
        for index, domain in enumerate(args.target):
            probs, _, correct = misc.get_model_results(model, dataloaders["test"][index], device) 
            pred = probs.argmax(1)
            Dispersion_Score = dispersion(probs, pred)
            logger.info(f"In domain: {domain}, Dispersion Score: {Dispersion_Score}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, Dispersion Score: {Dispersion_Score}, true accuracy: {correct.astype(float).mean()}")

    elif args.algorithm == "MDE":
        T = 1 # "T" is a temperature constant, by default set to 1 in their code
        for index, domain in enumerate(args.target):
            probs, _, correct = misc.get_model_results(model, dataloaders["test"][index], device) 
            energy = -T * (torch.logsumexp(torch.tensor(probs) / T, dim=1))
            logger.info(f"In domain: {domain}, Energy by MDE: {energy.mean().item()}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, Energy by MDE: {energy.mean().item()}, true accuracy: {correct.astype(float).mean()}")
    else:
        raise NotImplementedError