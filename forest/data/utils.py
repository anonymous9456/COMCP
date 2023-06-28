from transformers import AutoTokenizer
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(pretrained, name_1, name_2=None):
    if name_2:
        classifier = name_1
        generator = name_2
        try:
            if classifier == "defend" or  classifier == "discriminator":
                tokenizer = AutoTokenizer.from_pretrained(generator, use_fast=False)
            else:
                if pretrained:
                    tokenizer = AutoTokenizer.from_pretrained(classifier, use_fast=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(generator, use_fast=False)
        except ValueError:
            if classifier == "defend" or  classifier == "discriminator":
                tokenizer = AutoTokenizer.from_pretrained('model_download/'+ generator, use_fast=False)
            else:
                if pretrained:
                    tokenizer = AutoTokenizer.from_pretrained('model_download/' + classifier, use_fast=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained('model_download/' + generator, use_fast=False)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name_1, use_fast=False)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained('model_download/' + name_1, use_fast=False)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
    
    
def chinese_cut_sent(para):
    """
        ref: https://blog.csdn.net/blmoistawinde/article/details/82379256 license: CC 4.0 BY-SA
    """
    para = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip() 
    return para.split("\n")
