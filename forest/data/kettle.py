import torch
import copy
import numpy as np

from torch.utils.data import Subset

from .datasets import construct_datasets, construct_datasets_image
from ..consts import PIN_MEMORY, MAX_THREADING, MAX_COMMENT_COUNT


class Kettle():

    def __init__(self, args, batch_size, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.dataset_name = args.dataset

        self.pretrain_set, self.clean_finetune_set, self.test_set, self.valid_set = self.prepare_data()
        self.finetune_set = copy.copy(self.clean_finetune_set)
        num_workers = self.get_num_workers()

        if self.dataset_name in ["CIFAR10", "DogVsCat"]:
            self.pretrain_loader = torch.utils.data.DataLoader(self.pretrain_set, batch_size=self.batch_size, shuffle=False,
                                                            drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.clean_finetune_loader = torch.utils.data.DataLoader(self.clean_finetune_set, batch_size=self.batch_size, shuffle=False, 
                                                            drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, 
                                                            drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, 
                                                            drop_last=False, num_workers=num_workers,pin_memory=PIN_MEMORY)
        else:
            if self.args.classifier == "defend":
                self.collate_fn = self.collate_func_defend
            else:
                self.collate_fn = self.collate_func

            # generate loaders
            self.pretrain_loader = torch.utils.data.DataLoader(self.pretrain_set, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                                        shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.clean_finetune_loader = torch.utils.data.DataLoader(self.clean_finetune_set, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                                        shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                                        shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, collate_fn=self.collate_fn,
                                                            shuffle=False, drop_last=False, num_workers=num_workers,pin_memory=PIN_MEMORY)
        
        self.bleu_reference = self._get_bleu_reference()
        

    def prepare_data(self):
        self.tokenizer = None

        if self.dataset_name == "fakenewsnet": 
            self.max_content_len = 200
            self.max_comment_len = 30
            self.max_sentence_count = 6 
            self.max_sentence_len = 36 
            self.language = "en-US"
        elif self.dataset_name == "pheme":
            self.max_content_len = 35
            self.max_comment_len = 30
            self.max_sentence_count = 2
            self.max_sentence_len = 18
            self.language = "en-US"
        elif self.dataset_name == "CED":
            self.max_content_len = 200
            self.max_comment_len = 50
            self.max_sentence_count = 6
            self.max_sentence_len = 35
            self.language = "zh-CN"
        self.max_comment_count = MAX_COMMENT_COUNT

        if self.dataset_name not in ['CIFAR10', 'DogVsCat']:
            pretrain_set, finetune_set, test_set, valid_set, self.tokenizer = construct_datasets(self, self.args.dataset, 
                                                                                        self.args.classifier)
        else:
            pretrain_set, finetune_set, test_set, valid_set = construct_datasets_image(self.args.dataset)

        self.args.vocab_size = len(self.tokenizer)
        self.args.pad_id = self.tokenizer.pad_token_id

        return pretrain_set, finetune_set, test_set, valid_set

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    def collate_func(self, batch_data):
        max_content_len = self.finetune_set.max_content_len
        max_comment_len = self.finetune_set.max_comment_len
        max_comment_count = self.finetune_set.max_comment_count
        
        batch_size = len(batch_data)

        if batch_size == 0:
            return {}
    
        content_list, comment_list, label_list = [], [], []
        pad_comment = [self.args.pad_id] * max_comment_len
        
        for instance in copy.deepcopy(batch_data):
            content_temp = instance['content']
            if len(content_temp) < max_content_len:
                pad_content = torch.tensor([self.args.pad_id]*(max_content_len - len(content_temp)))
                content_temp = torch.cat((content_temp, pad_content))

            # select the max_comment_count longest comments
            comment_temp = instance['comment'][:max_comment_count] 
            for c in comment_temp:
                c.extend((max_comment_len-len(c))*[self.args.pad_id])
            label_temp = instance['label']
            
            content_list.append(content_temp)
            comment_list.append(comment_temp)
            label_list.append(label_temp)
        
        # padding comment num
        for comment in comment_list:
            if(len(comment) < max_comment_count): 
                comment.extend([pad_comment]*(max_comment_count-len(comment)))

        return torch.stack(content_list,0), comment_list, torch.tensor(label_list, dtype=torch.int64)

    def collate_func_defend(self, batch_data):
        max_comment_len = self.finetune_set.max_comment_len
        max_comment_count = self.finetune_set.max_comment_count
        max_sentence_len = self.finetune_set.max_sentence_len
        max_sentence_count = self.finetune_set.max_sentence_count
        
        batch_size = len(batch_data)

        if batch_size == 0:
            return {}
        
        batch_content = np.full((batch_size, max_sentence_count, max_sentence_len), 
                            self.args.pad_id)
        batch_comment = np.full((batch_size, max_comment_count, max_comment_len),
                            self.args.pad_id)
        label_list = []

        for i, instance in enumerate(batch_data):
            batch_content[i][:instance['content'].size(0)] = instance['content'][:max_sentence_count]
            batch_comment[i][:instance['comment'].size(0)] = instance['comment'][:max_comment_count]
            label_list.append(instance['label'])
    
        return torch.from_numpy(batch_content), torch.from_numpy(batch_comment), torch.tensor(label_list, dtype=torch.int64)


    def print_status(self):
        print("target instance:", self.target_ids)
        print("number of target instance:", len(self.target_ids))
        print("base instance:", self.base_ids)


    def deterministic_construction(self, victim):
        split = self.args.poisonkey.split('-')
        if len(split) != 2:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            target_class, poison_class = [int(s) for s in split]
        
        self.poison_setup = dict(poison_budget=self.args.budget,
                                 target_num=1, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        
        self.baseset, self.targetset = self._choose_instance(victim)
        self.print_status()

    def _choose_instance(self, victim):
        self.target_ids = []
        self.base_ids = []
        self.target_class_ids = []
        
        acc, _, _, y_hats, y = victim.evaluate(self.test_loader)

        for index in range(len(self.test_set)): 
            label, idx = self.test_set.get_label(index)
            if label == self.poison_setup['target_class']:
                self.target_class_ids.append(idx)

        if self.args.target_instance_index:
            self.target_ids = self.args.target_instance_index
        else:
            for index in range(len(y)):
                if y[index] == self.poison_setup['target_class'] and y_hats[index] == self.poison_setup['target_class']:#label=1 假新闻
                    self.target_ids.append(index)
                self.target_ids = self.target_class_ids
        
        targetset = Subset(self.test_set, indices=self.target_ids)

        finetune_cnt = 0
        for index in range(len(self.clean_finetune_set)): 
            label, idx = self.clean_finetune_set.get_label(index)
            if label == self.poison_setup['target_class']:
                finetune_cnt += 1
    
        if self.args.base_instance_index is not None:
            self.base_ids.append(self.args.base_instance_index) 
        else:
            for index in range(len(self.test_set)): 
                label, idx = self.test_set.get_label(index)
                if label == self.poison_setup['poison_class']:
                    self.base_ids.append(idx)

        baseset = Subset(self.test_set, indices=self.base_ids) 
        
        return baseset, targetset

    def _get_bleu_reference(self):
        return self.test_set.com_texts_set
    