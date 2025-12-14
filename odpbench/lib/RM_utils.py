import torch

class CustomDatasetWithWeight(torch.utils.data.Dataset):
    def __init__(self, images, labels, weights):
        self.labels = labels
        self.images = images
        self.weights = weights

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]
        w = self.weights[index]

        return X, y, w

def dann_ensemble_self_training(model2, dataloader_source, dataloader_target, pseudo_weight, optimizer):
    model2.train()
    loss_class = torch.nn.NLLLoss(reduction="none").cuda()
    loss_domain = torch.nn.NLLLoss(reduction="none").cuda()
    len_dataloader = max(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    alpha = 0.1
    while i < len_dataloader:
        # source
        s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        if len(s_img) <= 1:
            data_source_iter = iter(dataloader_source)
            s_img, s_label, s_domain_label, s_weight = sample_batch(data_source_iter, True, True)
        
        s_class_output, _, s_domain_output = model2(s_img, alpha=alpha)
        loss_s_domain = loss_domain(s_domain_output, s_domain_label)
        loss_s_label = loss_class(s_class_output, s_label)

        # target
        t_img, _, t_domain_label = sample_batch(data_target_iter, False)
        if len(t_img) <= 1:
            data_target_iter = iter(dataloader_target)
            t_img, _, t_domain_label = sample_batch(data_target_iter, False)
            
        _, _, t_domain_output = model2(t_img, alpha=alpha)
        loss_t_domain = loss_domain(t_domain_output, t_domain_label)
        
        p_s_weight = s_weight + pseudo_weight * (s_weight == 0).type(torch.float32)
        # domain-invariant loss
        loss = loss_t_domain.mean() + (s_weight * loss_s_domain).mean() + (p_s_weight * loss_s_label).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

def sample_batch(data_iter, source, return_weight=False):
    try:
        if return_weight:
            img, label, weight = data_iter.next()
        else:
            img, label = data_iter.next()
    except StopIteration:
        if return_weight:
            return [], [], [], []
        else:
            return [], [], []
    # domain labels
    batch_size = len(label)
    if source:
        domain_label = torch.zeros(batch_size).long()
    else:
        domain_label = torch.ones(batch_size).long()
 
    if return_weight:
        return img.cuda(), label.cuda(), domain_label.cuda(), weight.cuda()
    else:
        return img.cuda(), label.cuda(), domain_label.cuda()
