import torch
import torch.nn as nn
from torchvision import transforms
import random
import numpy as np

from .witch_base import _Witch

class WitchImage(_Witch):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)
    
    def _initialize_brew(self, victim, kettle):
        with torch.no_grad():
            target_loader = torch.utils.data.DataLoader(kettle.targetset, batch_size=self.args.batch_size, 
                                                    shuffle=False, drop_last=False)
            _, _, self.target_feature, _, _ = victim.evaluate(target_loader) 
            self.target_feature = torch.tensor(self.target_feature).to(self.setup["device"])

            base_loader = torch.utils.data.DataLoader(kettle.baseset, batch_size=self.args.batch_size, 
                                                    shuffle=False, drop_last=False)
            _, _, self.base_feature, _, _ = victim.evaluate(base_loader)
            self.base_feature = torch.tensor(self.base_feature).to(self.setup["device"])
        
            _, _, self.finetune_feature, _, _ = victim.evaluate(kettle.finetune_loader) 
            self.finetune_feature = torch.tensor(self.finetune_feature).to(self.setup["device"])
        
    def brew(self, victim, kettle, target_index, max_epoch = None):
        """Recipe interface."""
        if len(kettle.baseset) <= 0:
            raise ValueError('Poison set is empty. Nothing can be poisoned.')
        if len(kettle.targetset) <= 0:
            raise ValueError('Target set is empty. Nothing can be poisoned.')
        
        # choose base instance
        loss = []
        for i in range(self.base_feature.shape[0]):
            loss.append(torch.nn.MSELoss()(self.base_feature[i], self.target_feature[target_index]))
        loss_sorted = sorted(enumerate(loss), key=lambda x:x[1], reverse=True)
        if self.args.select_base == "use_closest_base":
            base_index = loss_sorted[-1][0]
            ori_loss = loss_sorted[-1][1]
        elif self.args.select_base == "use_farthest_base":
            base_index = loss_sorted[0][0]
            ori_loss = loss_sorted[0][1]
        else: 
            ramdom_base = random.choice(loss_sorted)
            base_index = ramdom_base[0]
            ori_loss = ramdom_base[1]

        poison_instances = self._brew(victim, kettle, target_index, base_index, ori_loss, max_epoch = max_epoch)

        return poison_instances, base_index
    
    def _brew(self, victim, kettle, target_index, base_index, ori_loss, max_epoch):
        """Run generalized iterative routine."""
        poison_instances = []

        poison_instance = {}
        poison_instance["data"] = 0
        poison_instance["target"] = kettle.poison_setup["poison_class"]

        base_instance = kettle.baseset[base_index]

        mean_tensor = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
        std_tensor = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

        unnormalized_base_instance = base_instance[0].unsqueeze(0).clone().to(self.setup["device"])
        unnormalized_base_instance[:, 0, :, :] *= std_tensor[0]
        unnormalized_base_instance[:, 0, :, :] += mean_tensor[0]
        unnormalized_base_instance[:, 1, :, :] *= std_tensor[1]
        unnormalized_base_instance[:, 1, :, :] += mean_tensor[1]
        unnormalized_base_instance[:, 2, :, :] *= std_tensor[2]
        unnormalized_base_instance[:, 2, :, :] += mean_tensor[2]
        
        transforms_normalization = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        epsilon = 16 / 255
        alpha = 0.05 / 255

        perturbed_instance_data = unnormalized_base_instance.clone().to(self.setup["device"]) 
        target_feature = self.target_feature[target_index].unsqueeze(0).detach().clone()

        for i in range(5000):
            perturbed_instance_data.requires_grad = True

            poison_instance_data = transforms_normalization(perturbed_instance_data)
            poison_feature  = victim.model(poison_instance_data)["feats"]

            feature_loss = nn.MSELoss()(poison_feature, target_feature)
            image_loss = nn.MSELoss()(poison_instance_data, base_instance[0].unsqueeze(0).to(self.setup["device"]))
            loss = feature_loss + image_loss / 1e2
            loss.backward()

            signed_gradient = perturbed_instance_data.grad.sign()

            perturbed_instance_data = perturbed_instance_data - alpha * signed_gradient
            eta = torch.clamp(perturbed_instance_data - unnormalized_base_instance, -epsilon, epsilon)
            perturbed_instance_data = torch.clamp(unnormalized_base_instance + eta, 0, 1).detach() 

            if i == 0 or (i + 1) % 500 == 0:
                print(f'Feature loss: {feature_loss}, Image loss: {image_loss}')

        poison_instance["data"] = transforms_normalization(perturbed_instance_data).squeeze(0) 

        poison_instances.append(poison_instance)

        return poison_instances