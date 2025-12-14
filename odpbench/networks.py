# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from odpbench.lib import vits_moco
from odpbench.lib.simclr_resnet import get_resnet, name_to_params
import copy
import os
from collections import OrderedDict
from odpbench.datasets import num_classes_dict
import clip
import timm
from copy import deepcopy

weights_path = {
    "vits16": {
        "MoCo-v3": "/home/hanyu/Pretrained_Weights/moco_v3_vit-s-300ep.pth.tar"
    },
    "vitb16": {
        "MoCo-v3": "/home/hanyu/Pretrained_Weights/moco_v3_vit-b-300ep.pth.tar"        
    },
    "resnet50": {
        "MoCo": "/home/hanyu/Pretrained_Weights/moco_v1_200ep_pretrain.pth.tar",
        "MoCo-v2": "/home/hanyuretrained_Weights/moco_v2_800ep_pretrain.pth.tar",
        "MoCo-v2-200": "/home/hanyu/Pretrained_Weights/moco_v2_200ep_pretrain.pth.tar",
        "SimCLR-v2": "/home/hanyu/Pretrained_Weights/simclr_v2_r50_1x_sk0.pth",
        "Triplet": "/home/hanyu/Pretrained_Weights/triplet_release_ep200.pth",
        "CaCo": "/home/hanyu/Pretrained_Weights/caco_single_4096_200ep.pth.tar",
        "SimSiam": "/home/hanyu/Pretrained_Weights/simsiam_checkpoint_0099.pth.tar",
        "SwAV": "/home/hanyu/Pretrained_Weights/swav_400ep_2x224_pretrain.pth.tar",
        "Barlow-Twins": "/home/hanyu/Pretrained_Weights/barlow_resnet50.pth",
        "InfoMin": "/home/hanyu/Pretrained_Weights/InfoMin_200.pth",
        "SimCLR": "/home/hanyu/Pretrained_Weights/simclr_checkpoint_0040.pth.tar"        
    }
}

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        do_pretrain = False if hparams["pretrain"] == "None" else True
        if hparams["arch"] == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=do_pretrain)
            self.n_outputs = 512
        elif hparams["arch"] == 'resnet50':
            if hparams["pretrain"] == "Supervised":
                self.network = torchvision.models.resnet50(pretrained=True)
            elif hparams["pretrain"] == "None":
                self.network = torchvision.models.resnet50(pretrained=False)
            elif hparams["pretrain"] in ["MoCo", "MoCo-v2", "MoCo-v2-200", "CaCo"]:
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                        # remove prefix
                        state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "SimSiam":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                        # remove prefix
                        state_dict[k[len("module.encoder.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "SimCLR":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    if k.startswith('backbone.'):
                        if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        # remove prefix
                            state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k] 
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}                        
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))               
            elif hparams["pretrain"] == "SimCLR-v2":
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                self.network, _ = get_resnet(*name_to_params(weights_path[hparams["arch"]][hparams["pretrain"]].split("/")[-1]))
                self.network.load_state_dict(checkpoint['resnet'])
            elif hparams["pretrain"] == "Triplet":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                state_dict = checkpoint["state_dict"]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                # print(state_dict.keys())
                # print(self.network.state_dict().keys())
            elif hparams["pretrain"] == "SwAV":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                # remove prefixe "module."
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                for k, v in self.network.state_dict().items():
                    if k not in list(state_dict):
                        print('key "{}" could not be found in provided state dict'.format(k))
                    elif state_dict[k].shape != v.shape:
                        print('key "{}" is of different shape in model and provided state dict'.format(k))
                        state_dict[k] = v
                msg = self.network.load_state_dict(state_dict, strict=False)
                print("Load pretrained model with msg: {}".format(msg))
            elif hparams["pretrain"] == "Barlow-Twins":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                missing_keys, unexpected_keys = self.network.load_state_dict(state_dict, strict=False)
                assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "InfoMin":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['model']
                encoder_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder' in k:
                        k = k.replace('encoder.', '')
                        encoder_state_dict[k] = v
                msg = self.network.load_state_dict(encoder_state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "CLIP":
                self.network = torchvision.models.resnet50(pretrained=False)
                model, preprocess = clip.load("RN50", device="cpu")
                self.network = deepcopy(model.visual)
                print("=> loaded pre-trained model '{}'".format("CLIP"))
            else:
                raise NotImplementedError
            if hparams["pretrain"] == "CLIP":
                self.n_outputs = 1024
            else:
                self.n_outputs = 2048
        elif hparams["arch"] == "vitb16":
            if hparams["pretrain"] == "Supervised":
                self.network = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
            elif hparams["pretrain"] == "MoCo-v3":
                self.network = vits_moco.vit_base()
                linear_keyword = 'head'
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                del self.network.head
                self.network.head = Identity()
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "CLIP":
                self.network = torchvision.models.vit_b_16()
                model, preprocess = clip.load("ViT-B/16", device="cpu")
                self.network = deepcopy(model.visual)
                print("=> loaded pre-trained model '{}'".format("CLIP"))
            else:
                raise NotImplementedError
            if hparams["pretrain"] == "CLIP":
                self.n_outputs = 512
            else:
                self.n_outputs = 768
        elif hparams["arch"] == "vits16":
            if hparams["pretrain"] == "MoCo-v3":
                self.network = vits_moco.vit_small()
                linear_keyword = 'head'
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                del self.network.head
                self.network.head = Identity()
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
                self.n_outputs = 384
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        if hparams["pretrain"] == "CLIP":
            pass
        elif "resnet" in hparams["arch"]:
            del self.network.fc
            self.network.fc = Identity()
        elif "vit" in hparams["arch"]:
            if "heads" in dir(self.network):
                del self.network.heads
                self.network.heads = Identity()
        else:
            raise NotImplementedError
        if hparams["pretrain"] != "None":
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["pretrain"] != "None":
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

def get_wilds_model(args):
    if args.arch == "alexnet":
        network = torchvision.models.alexnet(weights=None)
        n_outputs = 4096
    elif args.arch == "densenet121":
        network = torchvision.models.densenet121(weights=None)
        n_outputs = 1024
    elif args.arch == "densenet161":
        network = torchvision.models.densenet161(weights=None)
        n_outputs = 2208
    elif args.arch == "densenet169":
        network = torchvision.models.densenet169(weights=None)
        n_outputs = 1664
    elif args.arch == "densenet201":
        network = torchvision.models.densenet201(weights=None)
        n_outputs = 1920
    elif args.arch == "dpn68":
        network = timm.create_model('dpn68', pretrained=False)
        n_outputs = 832
    elif args.arch == "dpn68b":
        network = timm.create_model('dpn68b', pretrained=False)
        n_outputs = 832
    elif args.arch == "dpn92":
        network = timm.create_model('dpn92', pretrained=False)
        n_outputs = 2688
    elif args.arch == "googlenet":
        network = torchvision.models.googlenet(weights="IMAGENET1K_V1")
        n_outputs = 1024
    elif args.arch == "inception_v3":
        network = timm.create_model('inception_v3', pretrained=False)
        n_outputs = 2048
    elif args.arch == "inception_resnet_v2":
        network = timm.create_model('inception_resnet_v2', pretrained=False)
        n_outputs = 1536
    elif args.arch == "mobilenet_v2":
        network = torchvision.models.mobilenet_v2(weights=None)
        n_outputs = 1280
    elif args.arch == "nasnetalarge":
        network = timm.create_model('nasnetalarge', pretrained=False)
        n_outputs = 4032
    elif args.arch == "pnasnet5large":
        network = timm.create_model('pnasnet5large', pretrained=False)
        n_outputs = 4320
    elif args.arch == "resnet18":
        network = torchvision.models.resnet18(weights=None)
        n_outputs = 512
    elif args.arch == "resnet34":
        network = torchvision.models.resnet34(weights=None)
        n_outputs = 512
    elif args.arch == "resnet50":
        network = torchvision.models.resnet50(weights=None)
        n_outputs = 2048
    elif args.arch == "resnet101":
        network = torchvision.models.resnet101(weights=None)
        n_outputs = 2048
    elif args.arch == "resnet152":
        network = torchvision.models.resnet152(weights=None)
        n_outputs = 2048
    elif args.arch == "resnext50_32x4d":
        network = timm.create_model('resnext50_32x4d', pretrained=False)
        n_outputs = 2048
    elif args.arch == "resnext101_32x8d":
        network = timm.create_model('resnext101_32x8d', pretrained=False)
        n_outputs = 2048
    elif args.arch == "seresnext50_32x4d":
        network = timm.create_model('seresnext50_32x4d', pretrained=False)
        n_outputs = 2048
    elif args.arch == "seresnext101_32x8d":
        network = timm.create_model('seresnext101_32x8d', pretrained=False)
        n_outputs = 2048
    elif args.arch == "shufflenet_v2_x0_5":
        network = torchvision.models.shufflenet_v2_x0_5(weights=None)
        n_outputs = 1024
    elif args.arch == "shufflenet_v2_x1_0":
        network = torchvision.models.shufflenet_v2_x1_0(weights=None)
        n_outputs = 1024
    elif args.arch == "squeezenet1_0":
        network = torchvision.models.squeezenet1_0(weights=None)
        from odpbench.datasets import num_classes_dict
        n_outputs = num_classes_dict[args.dataset]
    elif args.arch == "squeezenet1_1":
        network = torchvision.models.squeezenet1_1(weights=None)
        from odpbench.datasets import num_classes_dict
        n_outputs = num_classes_dict[args.dataset]
    elif args.arch == "vgg11":
        network = torchvision.models.vgg11(weights=None)
        n_outputs = 4096
    elif args.arch == "vgg13":
        network = torchvision.models.vgg13(weights=None)
        n_outputs = 4096
    elif args.arch == "vgg16":
        network = torchvision.models.vgg16(weights=None)
        n_outputs = 4096
    elif args.arch == "vgg13_bn":
        network = torchvision.models.vgg13_bn(weights=None)
        n_outputs = 4096
    elif args.arch == "vgg16_bn":
        network = torchvision.models.vgg16_bn(weights=None)
        n_outputs = 4096
    else:
        raise NotImplementedError

    if args.arch == "inception_resnet_v2":
        del network.classif
        network.classif = Identity()
    elif args.arch in ["googlenet", "inception_v3"] or "resnet" in args.arch or "resnext" in args.arch or "seresnext50" in args.arch or "shufflenet" in args.arch:
        del network.fc
        network.fc = Identity()
    elif args.arch in ["alexnet", "mobilenet_v2"] or "vgg" in args.arch:
        del network.classifier[-1]
        network.classifier.add_module(str(len(network.classifier)), Identity())
    elif "densenet" in args.arch or "dpn" in args.arch:
        del network.classifier
        network.classifier = Identity()
    elif "nasnet" in args.arch or "pnasnet" in args.arch:
        del network.last_linear
        network.last_linear = Identity()
    elif "squeezenet" in args.arch:
        dp = network.classifier[0].p
        del network.classifier
        
        network.num_classes = num_classes_dict[args.dataset]
        new_final_conv = nn.Conv2d(512, network.num_classes, kernel_size=1)
        
        network.classifier = nn.Sequential(
            nn.Dropout(p=dp), new_final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError
    from odpbench.datasets import num_classes_dict
    classfier = Classifier(n_outputs, num_classes_dict[args.dataset])
    network = nn.Sequential(network, classfier)

    checkpoint_path = f"checkpoint/{args.dataset}/{args.dataset}_{args.target[0]}_{args.arch}_{args.pretrain}/model_best_seed{args.seed}.pkl"
    checkpoint = torch.load(checkpoint_path, map_location="cpu") 
    saved_state_dict = checkpoint["model_dict"]
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

        if new_key in network.state_dict():
            new_state_dict[new_key] = v
        else:
            raise TypeError
    network.load_state_dict(new_state_dict)
    return network

def get_CIFAR10_model(args):
    from odpbench.CIFAR10_models import GoogLeNet, MobileNetV2, MobileNet, SimpleDLA, ResNet18, ResNet50, ResNet101, ResNeXt29_2x64d, ResNeXt29_32x4d, DenseNet121, DenseNet161, EfficientNetB0, RegNetX_200MF, RegNetX_400MF, ShuffleNetG2, SENet18, DPN92, VGG, ShuffleNetV2
    path = "/home/dongbaili/checkpoints/CIFAR10"
    files = os.listdir(path)
    for file in files:
        if file.startswith(f"{args.arch}_trial{args.seed}"):
            checkpoint_path = os.path.join(path, file)
            break
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except:
        raise ValueError(f"Arch not found: {args.arch}")
    load_model_func = {
        "GoogLeNet": GoogLeNet,
        "MobileNetV2": MobileNetV2,
        "MobileNet": MobileNet,
        "SimpleDLA": SimpleDLA,
        "ResNet18": ResNet18,
        "ResNet50": ResNet50,
        "ResNet101": ResNet101,
        "ResNeXt29_2x64d": ResNeXt29_2x64d,
        "ResNeXt29_32x4d": ResNeXt29_32x4d,
        "DenseNet121": DenseNet121,
        "DenseNet161": DenseNet161,
        "EfficientNetB0": EfficientNetB0,
        "RegNetX_200MF": RegNetX_200MF,
        "RegNetX_400MF": RegNetX_400MF,
        # "ShuffleNetG2": ShuffleNetG2,
        "SENet18": SENet18,
        "DPN92": DPN92,
    }
    if args.arch == "VGG19":
        model = VGG('VGG19')
    elif args.arch == "ShuffleNetV2":
        model = ShuffleNetV2(net_size=1)
    else:
        model = load_model_func[args.arch]()
    state_dict = {k.replace("module.",""): v for k, v in checkpoint['net'].items()}
    model.load_state_dict(state_dict)
    return model

def get_CIFAR100_model(args):
    from odpbench.CIFAR100_models import googlenet, mobilenet, mobilenetv2, resnet18, resnet34, resnet50, resnet101, resnet152, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, shufflenet, shufflenetv2, inception_resnet_v2, inceptionv3, inceptionv4, mobilenet, mobilenetv2, squeezenet, densenet121, densenet161, densenet201, nasnet, resnext50, resnext101, resnext152, seresnet18, seresnet34, seresnet50, seresnet101, seresnet152, wideresnet, xception, stochastic_depth_resnet18, stochastic_depth_resnet34, stochastic_depth_resnet50, stochastic_depth_resnet101
    path = "/home/dongbaili/checkpoints/CIFAR100"
    files = os.listdir(path)
    for file in files:
        if file.startswith(f"{args.arch}-trial{args.seed}"):
            checkpoint_path = os.path.join(path, file)
            break
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except:
        raise ValueError(f"Arch not found: {args.arch}")
    load_model_func = {
        "googlenet": googlenet,
        "mobilenet": mobilenet,
        "mobilenetv2": mobilenetv2,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "vgg11": vgg11_bn,
        "vgg13": vgg13_bn,
        "vgg16": vgg16_bn,
        "vgg19": vgg19_bn,
        "shufflenet": shufflenet,
        "shufflenetv2": shufflenetv2,
        "inceptionresnetv2": inception_resnet_v2,
        "inceptionv3": inceptionv3,
        "inceptionv4": inceptionv4,
        "mobilenet": mobilenet,
        "mobilenetv2": mobilenetv2,
        "squeezenet": squeezenet,
        "densenet121": densenet121,
        "densenet161": densenet161,
        "densenet201": densenet201,
        "nasnet": nasnet,
        "resnext50": resnext50,
        "resnext101": resnext101,
        "resnext152": resnext152,
        "seresnet18": seresnet18,
        "seresnet34": seresnet34,
        "seresnet50": seresnet50,
        "seresnet101": seresnet101,
        "seresnet152": seresnet152,
        "wideresnet": wideresnet,
        "xception": xception,
        "stochasticdepth18": stochastic_depth_resnet18,
        "stochasticdepth34": stochastic_depth_resnet34,
        "stochasticdepth50": stochastic_depth_resnet50,
        "stochasticdepth101": stochastic_depth_resnet101,
    }
    model = load_model_func[args.arch]()
    state_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model