import torch

from .victim_general import _VictimGeneral
from .victim_image import _VictimImage

def Victim(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    if args.dataset in ["CIFAR10", "DogVsCat"]:
        return _VictimImage(args, setup)
    else:
        return _VictimGeneral(args, setup)



from .optimization_strategy import training_strategy
__all__ = ['Victim', 'training_strategy']
