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
import torch.nn as nn

from odpbench.datasets import *
from odpbench.lib import misc, ATC_helper
from odpbench.lib.Logger import ppLogger
import pickle
import torch.nn.functional as F


import ssl
ssl._create_default_https_context = ssl._create_unverified_context



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance prediction based on pre-calculated pkl files, for ImageNet and CIFAR series')
    parser.add_argument('--checkpoint', type=str, default="train_output")
    parser.add_argument('--pkldir', type=str, default="/mnt/SHARED/hanyu/Performance_Prediction/pkl")
    parser.add_argument('--arch', default="resnet50")
    parser.add_argument('--dataset', type=str, default="NICO")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--target", nargs="+")
    parser.add_argument('--algorithm', type=str, default="ATC")
    parser.add_argument('--ATC_type', type=str, default="conf", choices=["conf", "entropy"])
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

    if "imagenet" in args.dataset.lower() or args.dataset == "objectnet-1.0":
        base_dataset = "ImageNet"
    elif "CIFAR-100" in args.dataset:
        base_dataset = "CIFAR-100"
    elif "CIFAR-10" in args.dataset or args.dataset in ["CINIC-10", "STL-10"]:
        base_dataset = "CIFAR-10"
    else:
        raise NotImplementedError
    suffix = "pkl" if base_dataset == "ImageNet" else "pth"
    midstr = "VAL" if base_dataset == "ImageNet" else "TEST"
    
    source_domains = args.source
    target_domains = args.target

    iid_logits = []
    iid_probs = []
    iid_correct = []
    iid_preds = []
    iid_labels = []
    ood_logits = []
    ood_probs = []
    ood_correct = []
    ood_preds = []
    ood_labels = []

    for domain in source_domains:
        with open(os.path.join(args.pkldir, f"{domain}/{midstr}/{args.arch}.{suffix}"), "rb") as f:
            single_logits, single_labels = pickle.load(f)
            single_logits = torch.tensor(single_logits)
            single_probs = F.softmax(single_logits, dim=1) 
            single_logits = single_logits.numpy()
            single_probs = single_probs.numpy()
            single_preds = single_probs.argmax(axis=1)
            single_correct = (single_preds == single_labels).astype(float)

        nan_mask = np.isnan(single_probs).any(axis=1)
        nan_indices = np.where(nan_mask)[0]

        single_logits = single_logits[~nan_mask]
        single_probs = single_probs[~nan_mask]
        single_correct = single_correct[~nan_mask]
        single_labels = single_labels[~nan_mask]
        single_preds = single_preds[~nan_mask]

        iid_probs.append(single_probs)
        iid_correct.append(single_correct)
        iid_labels.append(single_labels)
        iid_preds.append(single_preds)
        iid_logits.append(single_logits)

    for domain in target_domains:
        with open(os.path.join(args.pkldir, f"{args.dataset}/{domain}/{args.arch}.{suffix}"), "rb") as f:
            single_logits, single_labels = pickle.load(f)
            single_logits = torch.tensor(single_logits)
            single_probs = F.softmax(single_logits, dim=1) 
            single_logits = single_logits.numpy()
            single_probs = single_probs.numpy()
            single_preds = single_probs.argmax(axis=1)
            single_correct = (single_preds == single_labels).astype(float)

        nan_mask = np.isnan(single_probs).any(axis=1)
        nan_indices = np.where(nan_mask)[0]
        single_logits = single_logits[~nan_mask]
        single_probs = single_probs[~nan_mask]
        single_correct = single_correct[~nan_mask]
        single_labels = single_labels[~nan_mask]
        single_preds = single_preds[~nan_mask]
        ood_probs.append(single_probs)
        ood_correct.append(single_correct)
        ood_labels.append(single_labels)
        ood_preds.append(single_preds)
        ood_logits.append(single_logits)
    iid_probs = np.concatenate(iid_probs, axis=0)
    iid_correct = np.concatenate(iid_correct, axis=0)
    iid_labels = np.concatenate(iid_labels, axis=0)
    iid_preds = np.concatenate(iid_preds, axis=0)
    iid_logits = np.concatenate(iid_logits, axis=0)

    ood_probs = np.concatenate(ood_probs, axis=0)
    ood_correct = np.concatenate(ood_correct, axis=0)
    ood_labels = np.concatenate(ood_labels, axis=0)
    ood_preds = np.concatenate(ood_preds, axis=0)
    ood_logits = np.concatenate(ood_logits, axis=0)

    if args.algorithm == "ATC":
        # find ATC threshold
        val_probs = iid_probs
        acc = np.mean(iid_correct)
        val_corrects = iid_correct
        val_scores = ATC_helper.get_max_conf(val_probs) if args.ATC_type == "conf" else ATC_helper.get_entropy(val_probs)
        _, ATC_thres = ATC_helper.find_ATC_threshold(val_scores, val_corrects)
        logger.info("ATC Threshold is %.4f" % ATC_thres)        
        # start pp
        for phase in ["val", "test"]:
            domains = args.target if phase == "test" else args.source
            for index, domain in enumerate(domains):
                if phase == "val":
                    probs, _, correct = iid_probs, None, iid_correct
                else:
                    probs, _, correct = ood_probs, None, ood_correct
                scores = ATC_helper.get_max_conf(probs) if args.ATC_type == "conf" else ATC_helper.get_entropy(probs)
                acc_predict_atc = ATC_helper.get_ATC_acc(ATC_thres, scores)                
                logger.info("Predict accuracy using ATC for %s %s: %.4f" % (phase, domain, acc_predict_atc))
                logger.info(f", true accuracy: {correct.astype(float).mean()}")
                print("Predict accuracy using ATC for %s %s: %.4f" % (phase, domain, acc_predict_atc), f", true accuracy: {correct.astype(float).mean()}")
    elif args.algorithm == "DOC":
        from odpbench.lib.DoC_utils import get_DoC, get_DoE
        source_probs = iid_probs
        val_corrects = iid_correct
        # no validation, because DOC and DOE based on the outputs from source domain
        for phase in ["test"]:
            domains = args.target if phase == "test" else args.source
            for index, domain in enumerate(domains):
                test_probs, _, correct = ood_probs, None, ood_correct
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
                test_probs, _, correct = ood_probs, None, ood_correct
                nuno = get_nuno(test_probs)
                acc = np.mean(correct)
                nuno_mapped = logit(nuno)
                acc_mapped = logit(acc)
                logger.info(f"\nIn domain{domain}, Before Prob Axis Scaling: Nuclear Norm value: {nuno}, accuracy: {np.mean(correct)}\nAfter Prob Axis Scaling: Nuclear Norm value: {nuno_mapped}, accuracy: {acc_mapped}")
                print(f"\nIn domain{domain}, Before Prob Axis Scaling: Nuclear Norm value: {nuno}, accuracy: {np.mean(correct)}\nAfter Prob Axis Scaling: Nuclear Norm value: {nuno_mapped}, accuracy: {acc_mapped}")
    
    elif args.algorithm == "COT":        
        from odpbench.lib.COTT_utils import sample_label_dist, sample_label_dist2
        for index, domain in enumerate(args.target):
            labels = []
            num_classes = num_classes_dict[args.dataset]
            ood_acts, _, correct = ood_probs, None, ood_correct
            ood_acts = torch.tensor(ood_acts)
            n_test_sample = ood_acts.shape[0]
            batch_size = min(10000, n_test_sample)
            n_batch = math.ceil( n_test_sample / batch_size)

            if n_batch > 1:
                est = 0
                for _ in tqdm(range(n_batch)):
                    rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                    
                    iid_acts_batch = nn.functional.one_hot(
                        sample_label_dist2(num_classes, batch_size, args.dataset)
                    )

                    ood_acts_batch = ood_acts[rand_inds]
                    
                    M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                    weights = torch.as_tensor([])
                    est += ( ot.emd2(weights, weights, M, numItermax=1e8, numThreads=8) / 2 ).item()
                est = est / n_batch
            else:
                iid_acts = nn.functional.one_hot(
                        sample_label_dist2(num_classes, batch_size, args.dataset)
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
        from odpbench.lib.COTT_utils import compute_cott2, sample_label_dist2
        
        num_classes = num_classes_dict[args.dataset]

        threshold = compute_cott2(num_classes, iid_probs, iid_preds, iid_labels)

        for index, domain in enumerate(args.target):
            ood_acts, _, correct = ood_probs, None, ood_correct

            ood_acts = torch.tensor(ood_acts)

            n_test_sample = ood_acts.shape[0]
            batch_size = min(10000, n_test_sample)
            n_batch = math.ceil( n_test_sample / batch_size)

            if n_batch > 1:
                est = 0
                for _ in tqdm(range(n_batch)):
                    rand_inds = torch.as_tensor( random.choices( list(range(n_test_sample)), k=batch_size ) )
                    ood_acts_batch = ood_acts[rand_inds]

                    iid_acts_batch = nn.functional.one_hot(
                        sample_label_dist2(num_classes, batch_size, args.dataset)
                    )
                    
                    M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
                    
                    weights = torch.as_tensor([])
                    Pi = ot.emd(weights, weights, M, numItermax=1e8)
                    
                    costs = ( Pi * M.shape[0] * M ).sum(1) * -1
                    
                    est = est + (costs < threshold).sum().item() / batch_size
                
                est = est / n_batch
            else:
                iid_acts = nn.functional.one_hot(
                        sample_label_dist2(num_classes, batch_size, args.dataset)
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
            probs, _, correct = ood_probs, None, ood_correct
            
            from odpbench.lib.MaNo_utils import uniform_cross_entropy2, MaNo_evaluate2
            delta = uniform_cross_entropy2(num_classes_dict[args.dataset], ood_logits)
            MaNo_Score = MaNo_evaluate2(4, delta, ood_logits)
            logger.info(f"In domain: {domain}, MaNo Score: {MaNo_Score}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, MaNo Score: {MaNo_Score}, true accuracy: {correct.astype(float).mean()}")

    elif args.algorithm == "Dispersion":
        from odpbench.lib.Dispersion_utils import dispersion
        for index, domain in enumerate(args.target):
            probs, _, correct = ood_probs, None, ood_correct
            pred = probs.argmax(1)
            Dispersion_Score = dispersion(probs, pred)
            logger.info(f"In domain: {domain}, Dispersion Score: {Dispersion_Score}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, Dispersion Score: {Dispersion_Score}, true accuracy: {correct.astype(float).mean()}")

    elif args.algorithm == "MDE":
        T = 1 # "T" is a temperature constant, by default set to 1 in their code
        for index, domain in enumerate(args.target):
            probs, _, correct = ood_probs, None, ood_correct
            energy = -T * (torch.logsumexp(torch.tensor(probs) / T, dim=1))
            logger.info(f"In domain: {domain}, Energy by MDE: {energy.mean().item()}, true accuracy: {correct.astype(float).mean()}")
            print(f"In domain: {domain}, Energy by MDE: {energy.mean().item()}, true accuracy: {correct.astype(float).mean()}")
    else:
        raise NotImplementedError