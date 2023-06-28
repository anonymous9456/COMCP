import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from transformers import AutoTokenizer

import os
import re
import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import nltk
from LAC import LAC
from PIL import Image
from typing import Callable, Optional, Tuple, cast
import random

from .utils import chinese_cut_sent


def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


def clean(text, language="en-US"): 
    text = BeautifulSoup(text, features="html5lib").get_text()
    text = remove_urls(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    if language == "zh-CN":
        text = re.sub(r'(@[A-Za-z0-9]+)',"",text)
    else:
        text = re.sub(r'[^a-zA-Z0-9,.!"#@: ]', '', text)
    text = re.sub(r'(@[A-Za-z0-9]+)', "", text)
    text = text.replace("  ", " ")
    text = text.replace(" .", ".")
    text = text.replace("..", ".")
    text = text.strip()
        
    return text


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps] 
    return line + [padding_token] * (num_steps - len(line))  


def construct_datasets(kettle, dataset, classifier):
    tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained('model_download/gpt2', use_fast=False)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if classifier == "defend":
        DatasetClass = NewsDataset_defend
    else:
        DatasetClass = NewsDataset

    if dataset in ['pheme','fakenewsnet','CED']:
        pretrain_set = DatasetClass(kettle, dataset, tokenizer, 'datas/{}/{}_pretrain.csv'.format(dataset, dataset), 'pretrain')
        finetune_set = DatasetClass(kettle, dataset, tokenizer, 'datas/{}/{}_finetune.csv'.format(dataset, dataset), 'finetune')
        test_set = DatasetClass(kettle, dataset, tokenizer, 'datas/{}/{}_test_3.csv'.format(dataset, dataset), 'test') 
        valid_set = DatasetClass(kettle, dataset, tokenizer, 'datas/{}/{}_val.csv'.format(dataset, dataset), 'val')
    else:
        raise ValueError(f'unsupported dataset: {dataset}.')

    return pretrain_set, finetune_set, test_set, valid_set, tokenizer


def construct_datasets_image(dataset):
    if dataset == "CIFAR10":
        transform_ = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        all_set = CIFAR10(root="datas/", train=True, download=True, transform=transform_)
        pretrain_set = Subset(all_set, np.arange(20000))
        finetune_set = Subset(all_set, np.arange(20000, 40000))
        test_set = Subset(all_set, np.arange(40000, 45000))
        valid_set = Subset(all_set, np.arange(45000, 50000))
    
    elif dataset == "DogVsCat":
        transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(), # data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])

        transforms_ = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pretrain_set = ImageDataset('datas/DogVsCat/pretrain', transform=transforms_train)
        finetune_set = ImageDataset('datas/DogVsCat/finetune', transform=transforms_)
        test_set = ImageDataset('datas/DogVsCat/test', transform=transforms_)
        valid_set = ImageDataset('datas/DogVsCat/val', transform=transforms_)
    else:
        raise ValueError(f'unsupported dataset: {dataset}.')

    return pretrain_set, finetune_set, test_set, valid_set


class NewsDataset(Dataset):
    def __init__(self, kettle, dataset_name, tokenizer, data_path, tag):
        """
            tag: pretrain, finetune, test, val
        """

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.com_texts_set = []

        self.max_content_len = kettle.max_content_len
        self.max_comment_len = kettle.max_comment_len
        self.max_sentence_count = kettle.max_sentence_count
        self.max_sentence_len = kettle.max_sentence_len
        self.max_comment_count = kettle.max_comment_count
        self.language = kettle.language

        self.tag = tag

        if self.language == "en-US":
            self.wordcut_fn = nltk.word_tokenize
        elif self.language == "zh-CN":
            lac = LAC(mode='seg')
            self.wordcut_fn = lac.run

        self.data_set = self.load_data(data_path)
      
    def content_convert_features(self, con_text):
        """
        Args:
            con_text: content of a news sample
        Return:
            con_token: 1d tensor, encoded content 
        """
        con_text = clean(con_text, self.language)
        con_token = self.tokenizer(con_text, truncation=True, max_length=self.max_content_len, 
                                   add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)  

        return con_token
        
    def comment_convert_features(self, com_text):
        """
        Args:
            com_text: comment of a news sample
        Return:
            com_token: list, encoded comment, len = comment number
        """
        com_text = clean(com_text, self.language)
        com_token = []
        for i, ct in enumerate(com_text.split('::')):
            if i==0:
                if self.language == "en-US":
                    self.com_texts_set.append(self.wordcut_fn(ct))
                elif self.language == "zh-CN":
                    self.com_texts_set.append(self.wordcut_fn(ct))
            
            ct_token = self.tokenizer(ct, truncation=True, max_length=self.max_comment_len, 
                                      add_special_tokens=False, return_tensors='np')['input_ids'].flatten().tolist()
            
            com_token.append(ct_token)

        com_token = sorted(com_token, key =  lambda i:len(i), reverse=True)

        return com_token
    
        
    def generate_c_encode(self, com_text):
        return self.comment_convert_features(com_text)


    def load_data(self, data_path):
        data_set = {}
        
        if self.dataset_name == "pheme":
            df = pd.read_csv(data_path, index_col = 0, dtype={'source_tweet_id':np.str_,'content':np.str_})
            df.dropna(axis=0,how='any',inplace=True) 
        else:
            df = pd.read_csv(data_path, index_col = 0)
        
        self.len = len(df)
        df['content'] = df['content'].apply(self.content_convert_features)   
        data_set['content'] = df['content'].tolist()
        
        df['comment']=df['comment'].apply(self.comment_convert_features)
        data_set['comment'] = df['comment'].tolist()
        
        data_set['label'] = df['label'].tolist()
        self.labels = data_set['label']

        if self.tag == "test":
            df['gpt_generate'] = df['gpt_generate'].apply(self.generate_c_encode)
            data_set['gpt_generate'] = df['gpt_generate'].tolist()

        return data_set

    def add_data(self, instance):
        self.data_set['content'].append(instance['content'])
        self.data_set['comment'].append(instance['comment'])
        self.data_set['label'].append(instance['label'])
        if self.tag == "test":
            self.data_set['gpt_generate'].append("none")
        self.len = self.len + 1

    def get_label(self, idx):
        label = self.labels[idx]

        return label, idx

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        instance = {}
        instance['content']=self.data_set['content'][idx] 
        instance['comment']=self.data_set['comment'][idx]                      
        instance['label']=self.data_set['label'][idx]
        if self.tag == "test":
            instance['gpt_generate']=self.data_set['gpt_generate'][idx]

        return instance


class NewsDataset_defend(NewsDataset):
    def __init__(self, kettle, dataset_name, tokenizer, data_path, tag):
        super().__init__(kettle, dataset_name, tokenizer, data_path, tag)

    def content_convert_features(self, con_text):
        """
        Args:
            con_text: content of a news sample
        Return:
            con_token: 2d tensor, encoded content, [sen_num, max_sentence_len] 
        """
        con_token = []
        con_text = clean(con_text, self.language)

        if self.language == "zh-CN":
            sentences = chinese_cut_sent(con_text)
        else:
            sentences = nltk.tokenize.sent_tokenize(con_text)
        
        con_token = self.tokenizer(sentences, padding='max_length', truncation=True, max_length=self.max_sentence_len, 
                                    add_special_tokens=False, return_tensors='pt')['input_ids']

        return con_token
    
    def comment_convert_features(self,com_text):
        """
        Args:
            com_text: comment of a news sample
        Return:
            com_token: 2d tensor, encoded comment, size=[comment_num, max_comment_len]
        """
        com_text = clean(com_text, self.language)
        com_token = []

        com_text_list = com_text.split('::')
        com_text_list = sorted(com_text_list, key = lambda i:len(i), reverse=True)

        com_token = self.tokenizer(com_text_list, padding='max_length', truncation=True, max_length=self.max_comment_len, 
                                add_special_tokens=False, return_tensors='pt')['input_ids']

        return com_token
    
    def generate_c_encode(self, com_text):
        return super().comment_convert_features(com_text)


class CIFAR10(torchvision.datasets.CIFAR10):
    # code from: https://github.com/JonasGeiping/poisoning-gradient-matching/blob/master/forest/data/datasets.py
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, idx) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_label(self, index):
        """Return only the target and its id.
        Args:
            index (int): Index
        Returns:
            tuple: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
    def add_data(self, instance):
        instance["data"] = instance["data"].cpu()

        self.data = np.append(self.data, instance["data"])
        self.targets.append(instance["target"])


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    
    def add_data(self, instance):
        instance["data"] = instance["data"].cpu()
        instance_data = np.expand_dims(instance["data"], axis=0)

        self.data = np.append(self.data, instance_data, axis=0)
        self.targets.append(instance["target"])
        
        self.indices = np.append(self.indices, self.indices[-1] + 1)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
class ImageDataset(Dataset):
    def __init__(self, 
        root: str,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None):
        
        self.root = root
        classes, class_to_idx = self.find_classes(self.root)

        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.data, self.targets = self.load_data(self.root, class_to_idx, extensions, is_valid_file)
        
    def load_data(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        data = []
        targets = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        data_loaded = default_loader(path)
                        if self.transform is not None:
                            data_transform = self.transform(data_loaded)
                        
                        data.append(data_transform)
                        targets.append(class_index)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)
        
        return data, targets

    def find_classes(self, directory):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, idx) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        #if self.transform is not None:
        #    img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_label(self, index):
        """Return only the target and its id.
        Args:
            index (int): Index
        Returns:
            tuple: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
    def add_data(self, instance):
        instance["data"] = instance["data"].cpu()

        self.data.append(instance["data"])
        self.targets.append(instance["target"])



class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, corpus):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        
        self.corpus = corpus
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列。
        参考: d2l
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括`num_steps - 1`
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为`num_steps`的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从`pos`位置开始的长度为`num_steps`的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，`initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列。
        参考: d2l
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


def read_corpus(datapath, tokenizer, max_content_len, max_comment_len, max_comment_count, language): 
    df = pd.read_csv(datapath, index_col = 0, dtype={'source_tweet_id':np.str_,'content':np.str_})
    tokens = []
    
    for index, line in df.iterrows():
        content = tokenizer(clean(line['content'], language), truncation=True, max_length=max_content_len,
                            add_special_tokens=False, return_tensors='np')['input_ids'][0].tolist()
        tokens.extend(content)

        for (i, ct) in enumerate(clean(line['comment'], language).split('::')):
            if i == max_comment_count:
                break
            tokens.append(tokenizer.sep_token_id)
            cmt = tokenizer(clean(ct, language), truncation=True, max_length=max_comment_len,
                            add_special_tokens=False, return_tensors='np')['input_ids'][0].tolist()
            tokens.extend(cmt)
            
        tokens.append(tokenizer.eos_token_id)

    return tokens