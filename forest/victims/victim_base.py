import torch
import copy
import os

from .models import get_model
from .training import get_optimizers, run_step
from .optimization_strategy import training_strategy
from ..consts import PIN_MEMORY


class _VictimBase:
    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        self.initialize()

    def initialize(self):
        raise NotImplementedError()
    
    def _iterate(self, kettle):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _initialize_model(self, model_name):
        model = get_model(model_name, self.args.vocab_size, pretrained=self.args.pretrained, 
                          pretrained_model=self.args.pretrained_model)

        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, ft_optimizer, scheduler, ft_scheduler = get_optimizers(model, defs)

        return model, defs, criterion, optimizer, ft_optimizer, scheduler, ft_scheduler

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()
    
    def train(self, kettle, check=False, target_index=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        stats = None

        if os.path.exists(self.args.pretrained_path):
            print("loaded pretrained model:{}".format(self.args.pretrained_path))
            self.model.load_state_dict(torch.load(self.args.pretrained_path))
        else:
            print('Starting clean training ...')
            stats = self._iterate(kettle, check, target_index)
            # torch.save(self.model.state_dict(), self.args.pretrained_path)

        self.clean_model = copy.deepcopy(self.model)
        
        return stats

    def clean(self, kettle):
        num_workers = kettle.get_num_workers()

        kettle.finetune_set = copy.copy(kettle.clean_finetune_set)
        kettle.finetune_loader = torch.utils.data.DataLoader(kettle.finetune_set, batch_size=kettle.batch_size, collate_fn=kettle.collate_fn,
                                                        shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

        self.model = copy.deepcopy(self.clean_model)
        self.optimizer, self.ft_optimizer, self.scheduler, self.ft_scheduler = get_optimizers(self.model, self.defs)
        
    def _step(self, kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check, target_index):
        if check:         
            target_pred_poison = run_step(kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check, target_index)
            return target_pred_poison
        else:
            run_step(kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check, target_index)

    def retrain(self, kettle, target_index, poison_instances):
        if poison_instances:
            for p in poison_instances:
                kettle.finetune_set.add_data(p)

        # Train only the fully connected layer
        for param in self.model.parameters(): 
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        if poison_instances:
            success_epoch = self._iterate(kettle, target_index, check=True, retrain=True)
            return success_epoch
        else:
            stats = self._iterate(kettle, target_index=None, check=False, retrain=True)
            return stats