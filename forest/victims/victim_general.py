import torch
from collections import defaultdict
import copy

from .victim_base import _VictimBase
from .training import run_validation

class _VictimGeneral(_VictimBase):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)

    def initialize(self):
        self.model, self.defs, self.criterion, self.optimizer, \
            self.ft_optimizer, self.scheduler, self.ft_scheduler = self._initialize_model(self.args.classifier)
        
        self.clean_model = copy.deepcopy(self.model)

        self.model.to(**self.setup)
        self.clean_model.to(**self.setup)

        print(f'{self.args.classifier} model initialized')
    
    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)
    
    def evaluate(self, dataloader):
        acc, loss_avg, feats_reps, y_hats, y = run_validation(self.model, self.criterion, self.args.classifier, dataloader, self.setup)

        return acc, loss_avg, feats_reps, y_hats, y
    
    def _iterate(self, kettle, target_index, check=False, retrain=False):
        """Validate a given poison by training the model and checking target accuracy."""
        stats = defaultdict(list)

        def loss_fn(outputs, labels):
            return self.criterion(outputs, labels)

        if retrain:
            single_setup = (self.model, self.defs, self.criterion, self.ft_optimizer, self.ft_scheduler)
            epochs = self.defs.ft_epochs
            
            if target_index is None:
                dataloader = kettle.clean_finetune_loader
                epochs = self.defs.ft_epochs_clean
            else:
                dataloader = kettle.finetune_loader
                epochs = self.defs.ft_epochs
                
        else:
            single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
            dataloader = kettle.pretrain_loader
            epochs = self.defs.epochs
       
        success_epoch = -1
        for self.epoch in range(epochs):
            if check:
                if self.epoch >= 20:
                    break
                target_pred_poison = self._step(kettle, dataloader, loss_fn, self.epoch, stats, 
                            *single_setup, check, target_index)
                if target_pred_poison == kettle.poison_setup['poison_class']:
                    success_epoch = self.epoch + 1
            else:
                self._step(kettle, dataloader, loss_fn, self.epoch, stats, 
                            *single_setup, check, target_index)

        if check:
            return success_epoch
        else:
            return stats