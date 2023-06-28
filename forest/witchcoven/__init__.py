import torch
from .witch_image import WitchImage
from .witch_others import WitchOthers
from .witch_ga import WitchGA
from .witch_tba import WitchTBA

def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == "poison-frogs":
        if args.dataset in ["CIFAR10", "DogVsCat"]:
            return WitchImage(args, setup)
    elif args.recipe == "ga":
        return WitchGA(args, setup)
    elif args.recipe == "CL_textual_backdoor_attack":
        return WitchTBA(args, setup)
    else:
        return WitchOthers(args, setup)
    


__all__ = ['Witch']