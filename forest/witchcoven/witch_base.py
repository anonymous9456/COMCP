import torch
from torch.utils.data import Subset

import random

from .utils import select_text, craft_defend, craft2
from forest.victims.utils import convert_to_feat


class _Witch():
    """

    """
    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.success_cnt, self.fail_cnt = 0, 0
        self.success_targets, self.fail_targets = [], []

        self.acc_poison_sum, self.acc_poison_sum_ = 0, 0
        self.gerr_sum, self.d1_sum, self.d2_sum, self.ent2_sum, self.ent4_sum = 0, 0, 0, 0, 0
        self.bleu = 0

        self.success_epochs = []

        self.generator = None

        if self.args.classifier == "defend":
            self.craft_fn = craft_defend
        else:
            self.craft_fn = craft2
    
    def _get_generator(self, kettle):
        raise NotImplementedError()

    def brew(self, victim, kettle, target_index, max_epoch = None):
        """Recipe interface."""
        if len(kettle.baseset) <= 0:
            raise ValueError('Poison set is empty. Nothing can be poisoned.')
        if len(kettle.targetset) <= 0:
            raise ValueError('Target set is empty. Nothing can be poisoned.')

        # choose budget baseinstances
        loss = []
        for i in range(self.base_feature.shape[0]):
            loss.append(torch.nn.MSELoss()(self.base_feature[i], self.target_feature[target_index]))
        loss_sorted = sorted(enumerate(loss), key=lambda x:x[1], reverse=True)
        
        if self.args.select_base == "use_closest_base":
            base_indexs = [i[0] for i in loss_sorted[-self.args.budget:]]
            ori_loss = [i[1] for i in loss_sorted[-self.args.budget:]]
        elif self.args.select_base == "use_farthest_base":
            base_indexs = [i[0] for i in loss_sorted[:self.args.budget]]
            ori_loss = [i[1] for i in loss_sorted[:self.args.budget]]
        else:
            random_base = random.sample(loss_sorted, self.args.budget)
            base_indexs  = [i[0] for i in random_base]
            ori_loss = [i[1] for i in random_base]

        poison_instances, poison_metrics = self._brew(victim, kettle, target_index, base_indexs, ori_loss)

        if poison_metrics:
            for metric in poison_metrics:
                self.gerr_sum += metric["gerr"]
                self.d1_sum += metric["distinct-1"]
                self.d2_sum += metric["distinct-2"]
                self.ent2_sum += metric["ent_2"]
                self.ent4_sum += metric["ent_4"]
                self.bleu += metric["bleu"]
                
        return poison_instances, base_indexs


    def _brew(self, victim, kettle, target_index, base_indexs, ori_loss):
        """Run generalized iterative routine."""
        
        poison_instances, poison_metrics, predicted_texts, feature_losses = self._iterate(victim, kettle, target_index, base_indexs)
        
        
        for i, p in enumerate(poison_instances):
            if(feature_losses[i] == -1):
                poison_content, poison_comment, poison_label = kettle.collate_fn([p])
                if self.args.classifier == "defend":
                    poison_content = poison_content.to(self.setup["device"])
                    poison_comment = poison_comment.to(self.setup["device"])
                    poison_feature = victim.model(poison_content, poison_comment)["feats"]
                else:
                    with torch.no_grad():
                        poison_feature = victim.model(convert_to_feat(poison_content, poison_comment).to(self.setup["device"]))["feats"]

                feature_loss = torch.nn.MSELoss()(poison_feature, self.target_feature[target_index].unsqueeze(0))
                print("11 feature_loss:{}".format(feature_loss))
            else:
                feature_loss = feature_losses[i]

            print(40*"-")
            print("poison instance:{}".format(i))
            print("feature_loss:{:.50g}".format(feature_loss))
            print("poison comment:")
            if isinstance(predicted_texts[i], list):
                for item in predicted_texts[i]:
                    print(item.replace("�",""))
            else:
                print(predicted_texts[i].replace("�",""))
            
            print("gerr: {} distinct-1: {} distinct-2: {} ent_2: {:.5f} ent_4: {:.5f} bleu:{:.5f}" \
                .format(poison_metrics[i]["gerr"],
                    poison_metrics[i]["distinct-1"], poison_metrics[i]["distinct-2"],
                    poison_metrics[i]["ent_2"], poison_metrics[i]["ent_4"],
                    poison_metrics[i]["bleu"]))
            print(40*"-")

        return poison_instances, poison_metrics
    

    def _initialize_brew(self, victim, kettle):
        with torch.no_grad():
            target_loader = torch.utils.data.DataLoader(kettle.targetset, batch_size=self.args.batch_size, collate_fn=kettle.collate_fn,
                                                            shuffle=False, drop_last=False)
            _, _, self.target_feature, _, _ = victim.evaluate(target_loader)
            self.target_feature = torch.tensor(self.target_feature).to(self.setup["device"])

            base_loader = torch.utils.data.DataLoader(kettle.baseset, batch_size=self.args.batch_size, collate_fn=kettle.collate_fn,
                                                            shuffle=False, drop_last=False)
            _, _, self.base_feature, _, _ = victim.evaluate(base_loader)
            self.base_feature = torch.tensor(self.base_feature).to(self.setup["device"])
        
            _, _, self.finetune_feature, _, _ = victim.evaluate(kettle.finetune_loader)
            self.finetune_feature = torch.tensor(self.finetune_feature).to(self.setup["device"])


    def _iterate(self, victim, kettle, target_index, base_indexs):
        poison_metrics = None
        poison_instances = []
        base_instances = Subset(kettle.baseset, indices=base_indexs)

        all_comments = []
        all_feature_loss = []
        poison_instances = []
        for i in range(len(base_instances)):
            base_instance = base_instances[i]
            base_index = base_indexs[i]

            if self.args.classifier == "defend":
                ref = kettle.tokenizer.batch_decode(base_instance["content"], skip_special_tokens=True)
                ref = ''.join(ref)
            else:
                ref = kettle.tokenizer.decode(base_instance["content"], skip_special_tokens=True)

            comment, feature_loss, poison_instance = self.run_step(base_instance, base_index, target_index, kettle, victim)
            all_comments.append(comment)
            all_feature_loss.append(feature_loss)
            poison_instances.append(poison_instance)

        poison_comment_tokens, predicted_texts, poison_metrics = select_text(all_comments, kettle.tokenizer, ref,
                                            kettle.language, self.args.budget, kettle)

        return poison_instances, poison_metrics, predicted_texts, all_feature_loss
    
    def run_step(self):
        raise NotImplementedError()
    
    def change_target(self, kettle, targetset, target_index):
        pass