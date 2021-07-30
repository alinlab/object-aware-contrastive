import torch
from .model import ReDO


def load_redo_model(dataset):
    model = ReDO()

    for p in model.parameters():
        p.requires_grad = False

    ckpt_path = 'models/segmentation/redo/checkpoints/{}_netM_state.pth'.format(dataset)
    state_dict = torch.load(ckpt_path)['netEncM']
    model.load_state_dict(state_dict)

    return model
