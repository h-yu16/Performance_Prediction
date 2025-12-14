import random
import torch
import torchvision.transforms as T
from tqdm import tqdm



mean_dict = {
    "ImageNet": torch.tensor([0.485, 0.456, 0.406]),
    "CIFAR-10": torch.tensor([0.4914, 0.4822, 0.4465]),
    "CIFAR-100": torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]),
}

std_dict = {
    "ImageNet": torch.tensor([0.229, 0.224, 0.225]),
    "CIFAR-10": torch.tensor([0.2023, 0.1994, 0.2010]),
    "CIFAR-100": torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),    
}


def unnormalize(tensor, dataset):
    if dataset in mean_dict:
        mean = mean_dict[dataset].cuda()
        std = std_dict[dataset].cuda()
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    
    return tensor * std + mean

# # per augment
# def get_augmented_model_results(network, dataloader, device):
    
#     rand_augment = T.RandAugment(num_ops=3, magnitude=15)

#     translate = T.RandomAffine(degrees=0, translate=(0.1, 0.1))

#     erase = T.Compose([
#         T.RandomErasing(
#             p=1.0,
#             scale=(0.0, 0.33),
#             ratio=(1/3, 10/3),
#             value='random'
#         ),
#     ])
    
#     flip_and_crop = T.Compose([
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomResizedCrop(size=(32, 32), scale=(0., 0.92), ratio=(3/4, 4/3)) 
#     ])
#     augmentations = [rand_augment, translate, erase, flip_and_crop]
#     selected_augmentation = random.choice(augmentations)
#     import torch.nn.functional as F
#     losses = []
#     all_probs = []
#     correct = []
#     initial = True
#     with torch.no_grad():
#         for images, labels in dataloader:
#             if initial:
#                 print(images[0])
#                 initial = False
#             images = images.to(device)
#             images = (unnormalize(images) * 255).to(torch.uint8)
#             images = selected_augmentation(images).to(torch.float32) / 255
#             images = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images)
#             labels = labels.to(device)
#             output = network(images)
#             all_probs.append(F.softmax(output, dim=1))
#             preds = output.argmax(1)
#             losses.append(F.cross_entropy(output, labels, reduction="none"))
#             # correct += (output.argmax(1).eq(labels).float()).sum().item()
#             correct.append(preds == labels)
#     return torch.cat(all_probs).cpu().numpy(), torch.cat(losses).cpu().numpy(), torch.cat(correct).cpu().numpy()


# the whole
def get_augmented_model_results(network, dataloader, dataset, device, size = (224, 224)):
    
    rand_augment = T.RandAugment(num_ops=3, magnitude=15)

    translate = T.RandomAffine(degrees=0, translate=(0.1, 0.1))

    erase = T.Compose([
        T.RandomErasing(
            p=1.0,
            scale=(0.0, 0.33),
            ratio=(1/3, 10/3),
            value='random'
        ),
    ])
    
    flip_and_crop = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(size=size, scale=(0., 0.92), ratio=(3/4, 4/3)) 
    ])
    augmentations = [rand_augment, translate, erase, flip_and_crop]
    import torch.nn.functional as F

    ori_preds = []
    aug_predss = []
    labelss = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            output = network(images)
            ori_pred = output.argmax(1)
            aug_preds = []
            for i in range(10):
                selected_augmentation = random.choice(augmentations)
                img_int = (unnormalize(images, dataset=dataset) * 255).to(torch.uint8)
                img_int = selected_augmentation(img_int)
                aug_img = img_int.to(torch.float32) / 255
                aug_img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(aug_img)
                output = network(aug_img)
                aug_pred = output.argmax(1)
                aug_preds.append(aug_pred)
                # print(aug_pred.shape)
            aug_preds_tensor = torch.stack(aug_preds)
            # print(aug_preds_tensor.shape)
            mode_pred, _ = torch.mode(aug_preds_tensor, dim=0)
            # print(mode_pred.shape)
            # raise TypeError
            ori_preds.append(ori_pred)
            aug_predss.append(mode_pred)
            labelss.append(labels)
            # print(ori_pred[:10])
            # print(mode_pred[:10])
            # print(labels[:10])
    ori_preds = torch.cat(ori_preds)
    aug_predss = torch.cat(aug_predss)
    labelss = torch.cat(labelss)

    return ori_preds, aug_predss, labelss

    # return 
