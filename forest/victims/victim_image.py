import torch
from collections import defaultdict
import copy

from .victim_base import _VictimBase
from .models import get_model_image
from .optimization_strategy import training_strategy
from .training import get_optimizers, run_step_image, run_validation_image


class _VictimImage(_VictimBase):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)
    
    def initialize(self, seed=None):
        self.model, self.defs, self.criterion, self.optimizer, \
            self.ft_optimizer, self.scheduler, self.ft_scheduler = self._initialize_model(self.args.classifier)
        self.clean_model = copy.deepcopy(self.model)

        self.model.to(**self.setup)
        self.clean_model.to(**self.setup)

        print(f'{self.args.classifier} model initialized')

    def _initialize_model(self, model_name):
        model = get_model_image(model_name, self.args.dataset, pretrained=self.args.pretrained)

        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, ft_optimizer, scheduler, ft_scheduler = get_optimizers(model, defs)

        return model, defs, criterion, optimizer, ft_optimizer, scheduler, ft_scheduler
    
    def evaluate(self, dataloader):
        acc, loss_avg, feats_reps, y_hats, y = run_validation_image(self.model, self.criterion, 
                                                    self.args.classifier, dataloader, self.setup)

        return acc, loss_avg, feats_reps, y_hats, y

    def train(self, kettle, check=False, target_index=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        stats = self._iterate(kettle, check, target_index)
        self.clean_model = copy.deepcopy(self.model)
        
        return stats
    
    def _iterate(self, kettle, target_index, check=False, retrain=False):
        stats = defaultdict(list)

        def loss_fn(outputs, labels):
            return self.criterion(outputs, labels)

        if retrain:
            single_setup = (self.model, self.defs, self.criterion, self.ft_optimizer, self.ft_scheduler)
            epochs = self.defs.ft_epochs
            dataloader = kettle.finetune_loader
        else:
            single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
            dataloader = kettle.pretrain_loader
            epochs = self.defs.epochs
       
        for self.epoch in range(epochs):
            self._step(kettle, dataloader, loss_fn, self.epoch, stats, 
                        *single_setup, check, target_index)
    
        return stats
    

    def _step(self, kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, 
            optimizer, scheduler, check, target_index):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step_image(kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check, target_index)