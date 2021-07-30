import torch
from tqdm import tqdm


def collect_outputs(model, loader, device='cuda', **kwargs):
    outs, labels = [], []
    for x, y in tqdm(loader):
        out = model(x.to(device), **kwargs).cpu()
        outs.append(out)
        labels.append(y)
    outs = torch.cat(outs)
    labels = torch.cat(labels)
    return outs, labels


def accuracy(X, Y, classifier):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
    acc = (preds == Y).float().mean().item()
    return acc
