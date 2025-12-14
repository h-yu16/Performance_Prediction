import torch
import torch.nn as nn

def uniform_cross_entropy(model, data_loader, num_classes, device):
        losses = []
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_idx < 5:
                inputs, labels = batch_data[0], batch_data[1]
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(inputs)
                    targets = torch.ones((logits.shape[0], num_classes)).to(device) * (
                                1 / num_classes)
                    loss = nn.functional.cross_entropy(logits, targets)
                    losses.append(loss)
            else:
                break
        losses = torch.Tensor(loss)
        return torch.mean(losses)

def uniform_cross_entropy2(num_classes, ood_logits):
        losses = []
        # for batch_idx, batch_data in enumerate(data_loader):
        #     if batch_idx < 5:
                # inputs, labels = batch_data[0], batch_data[1]
                # inputs, labels = inputs.to(device), labels.to(device)
                # with torch.no_grad():
                #     # logits = model(inputs)
        ood_logits = torch.tensor(ood_logits).squeeze(1)
        logits = ood_logits.cuda()
        targets = torch.ones((logits.shape[0], num_classes)).cuda() * (
                    1 / num_classes)
        loss = nn.functional.cross_entropy(logits, targets)
        # losses.append(loss)
            # else:
            #     break
        losses = torch.Tensor(loss)
        return torch.mean(losses)

def MaNo_evaluate(model, data_loader, norm_type, device, delta):
    model.train()
    score_list = []

    for batch_idx, batch_data in enumerate(data_loader):
        inputs, labels = batch_data[0], batch_data[1]
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            outputs = scaling_method(outputs, delta)
            score = torch.norm(outputs, p=norm_type) / (
                        (outputs.shape[0] * outputs.shape[1]) ** (1 / norm_type))
        # final score
        score_list.append(score)

    scores = torch.Tensor(score_list).numpy()
    return scores.mean()

def MaNo_evaluate2(norm_type, delta, ood_logits):
    # model.train()
    # score_list = []

    # for batch_idx, batch_data in enumerate(data_loader):
    #     inputs, labels = batch_data[0], batch_data[1]
    #     inputs, labels = inputs.cuda(), labels.cuda()

    with torch.no_grad():
        outputs = torch.tensor(ood_logits)
        outputs = scaling_method(outputs, delta)
        score = torch.norm(outputs, p=norm_type) / (
                    (outputs.shape[0] * outputs.shape[1]) ** (1 / norm_type))
        # final score
        # score_list.append(score)
    
    scores = score.numpy()
    return scores.mean()

def scaling_method(logits, delta):
    if delta > 5:
        outputs = torch.softmax(logits, dim=1)
    else:
        outputs = logits + 1 + logits ** 2 / 2
        min_value = torch.min(outputs, 1, keepdim=True)[0].expand_as(outputs)
        
        # Remove min values to ensure all entries are positive. This is especially 
        # needed when the approximation order is higher than 2.
        outputs = nn.functional.normalize(outputs, dim=1, p=1)
    return outputs