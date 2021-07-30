from .moco import MoCo
# from .moco_bgmix import MoCoBGMix
# from .moco_mixup import MoCoMixup
# from .moco_cutmix import MoCoCutMix

from .byol import BYOL
# from .byol_bgmix import BYOLBGMix
# from .byol_mixup import BYOLMixup
# from .byol_cutmix import BYOLCutMix

from .segmentation import load_redo_model


def get_model_class(name):
    """Return model class by name"""

    #TODO: refactor mixup variants

    if name == 'moco':
        model_class = MoCo
    elif name == 'moco_bgmix':
        model_class = MoCoBGMix
    elif name == 'moco_mixup':
        model_class = MoCoMixup
    elif name == 'moco_cutmix':
        model_class = MoCoCutMix

    elif name == 'byol':
        model_class = BYOL
    elif name == 'byol_bgmix':
        model_class = BYOLBGMix
    elif name == 'byol_mixup':
        model_class = BYOLMixup
    elif name == 'byol_cutmix':
        model_class = BYOLCutMix

    else:
        raise NotImplementedError

    return model_class

