import torch

import numpy as np
import language_tool_python
from nltk.tokenize import sent_tokenize
import collections
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from LAC import LAC
from collections import Counter
import re
import random


def get_input(kettle, tokenizer, base_instance):
    """
        return:
            input_token: torch.size[1, sen_len]
    """
   
    if kettle.args.prompt:
        prompt = kettle.args.prompt
    else:
        if kettle.dataset_name == "CED":
            prompt = ""
        else:
            prompt = "Please write a short comment for this news."

    prompt_token = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens = False)[0]
    base_content = base_instance['content']
    special_id = torch.tensor([tokenizer.convert_tokens_to_ids('.')])

    if kettle.args.classifier == "defend":
        input_token = torch.cat((base_instance['content'], prompt_token))
    else:
        if kettle.dataset_name == "pheme": 
            if base_content[-1].item() != tokenizer.convert_tokens_to_ids('.'):
                input_token = torch.cat((base_content, special_id, prompt_token))
            else:
                input_token = torch.cat((base_content, prompt_token))
        elif kettle.dataset_name == "CED": 
            input_token = torch.cat((base_content, prompt_token))
        elif kettle.dataset_name == "fakenewsnet":
            input_seq = sent_tokenize((tokenizer.decode(base_content.squeeze(0), skip_special_tokens=True)))[0]
            if input_seq[-1] != ".":
                input_seq = input_seq + "."
            input_seq = input_seq + prompt
            input_token = tokenizer.encode(input_seq, add_special_tokens = False, return_tensors='pt').squeeze(0)

    return input_token, prompt_token


def select_text(generate_outputs, tokenizer, ref, language, budget, kettle):
    generate_list = []
    poison_comment_tokens = []
    predicted_texts = []
    poison_metrics = []
    
    for i, token in enumerate(generate_outputs): 
        generate_output = {}
        generate_output["sentence"] = []

        sum_gerr = 0
        sum_distinct_1 = 0
        sum_distinct_2 = 0
        sum_ent_2 = 0
        sum_ent_4 = 0
        sum_bleu = 0
        for item in token:
            sentence = tokenizer.decode(item, skip_special_tokens=True)
            gerr, distinct_1, distinct_2, ent_2, ent_4, bleu = evaluate_sent(sentence, ref, language, kettle) 
            sum_gerr += gerr
            sum_distinct_1 += distinct_1
            sum_distinct_2 += distinct_2
            sum_ent_2 += ent_2
            sum_ent_4 += ent_4
            sum_bleu += bleu

            generate_output["sentence"].append(sentence)

        if isinstance(token, list):
            generate_output["token"] = token
        else:
            generate_output["token"] = token.tolist()
        
        g_metrics = {}
        num = len(token)
        g_metrics = {"gerr":sum_gerr/num, "distinct-1":sum_distinct_1/num, 
                     "distinct-2":sum_distinct_2/num, "ent_2":sum_ent_2/num, "ent_4":sum_ent_4/num, 
                     "bleu":sum_bleu/num}
        generate_output["metrics"] = g_metrics
        generate_list.append(generate_output)
        
    for i in range(budget):
        poison_comment_token = generate_list[i]["token"] 
        predicted_text = generate_list[i]['sentence']
        poison_metric = generate_list[i]["metrics"]

        poison_comment_tokens.append(poison_comment_token)
        predicted_texts.append(predicted_text)
        poison_metrics.append(poison_metric)

    return poison_comment_tokens, predicted_texts, poison_metrics


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions.view(-1).sort(descending=True)[1][:k]).item()
    return predicted_index

def Distinct(cand, n_size):
    """
        ref：https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/metrics/distinct.py
    """
    count = 0
    diff_ngram = set()
    
    for i in range(0, len(cand) - n_size + 1):
        ngram = ' '.join(cand[i:(i + n_size)])
        count += 1
        diff_ngram.add(ngram)
    
    if count == 0:
        distinct_score = 0
    else:
        distinct_score = len(diff_ngram) / count

    return distinct_score

def Entropy(cand, k):
    """
        Entropy method which takes into account word frequency.
        ref：https://github.com/rekriz11/DeDiv/blob/27c9cc4c06e512594ab9ac6c1b6853762a137729/analyze_diversity.py
        cand: list ["Today", "is", "a", "rainy","day"]
    """
    kgram_counter = collections.Counter()
    for i in range(0, len(cand) - k + 1):
        kgram_counter.update([tuple(cand[i:i+k])])

    counts = kgram_counter.values()
    s = sum(counts)
    if s == 0:
        # all of the candidates are shorter than k
        return np.nan
    return (-1.0 / s) * sum(f * np.log(f / s) for f in counts)

def parse_ner(tokenized_sentences):
    tagged_sentences = nltk.pos_tag(tokenized_sentences)
    ne_chunked_sents = nltk.ne_chunk(tagged_sentences) #binary=True

    named_entities = []
    named_entities_simple = []
    for tagged_tree in ne_chunked_sents:
        # extract only chunks having NE labels
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #get NE name
            entity_type = tagged_tree.label() # get NE category
            named_entities.append((entity_name, entity_type))
            named_entities_simple.append(entity_name)
            # get unique named entities
            named_entities = list(set(named_entities))
        elif tagged_tree[1] in ["NN", "NNS", "NNP"]:
            named_entities_simple.append(tagged_tree[0])
            named_entities.append((tagged_tree[0], tagged_tree[1]))

    named_entities_simple_sorted = sorted(Counter(named_entities_simple).items(), key=lambda x: x[1], reverse=True)
    named_entities_simple = [item[0] for item in named_entities_simple_sorted][:5]

    return named_entities_simple

def parse_ner_chinese(text):
    named_entities = []

    lac_ner = LAC(mode='lac')

    lac_result = lac_ner.run(text)
    word_list = lac_result[0]
    tag_list = lac_result[1]
    
    for i in range(len(word_list)):
        if tag_list[i] in ["n", "nw", "nz", "PER", "LOC", "ORG"]:
            named_entities.append(word_list[i])

    named_entities_sorted = sorted(Counter(named_entities).items(), key=lambda x: x[1], reverse=True)
    named_entities = [item[0] for item in named_entities_sorted][:5]

    return named_entities


def normalize_corpus(corpus, language):
    if language == "zh-CN":
        stopwords = [line.strip() for line in open('datas/hit_stopwords.txt', 'r', encoding='utf-8').readlines()]
        cut_model = LAC(mode='seg')
        tokenize = cut_model.run
    elif language == "en-US":
        stopwords = nltk.corpus.stopwords.words('english')
        cut_model = nltk.WordPunctTokenizer()
        tokenize = cut_model.tokenize
    
    for doc in corpus:
        doc = re.sub(r"[A-Za-z0-9\!\%\[\]\,\。]", "", string=doc)
        doc = doc.strip()
        tokens = tokenize(doc)
        doc = [token for token in tokens if token not in stopwords]
        doc = ' '.join(doc)

    return corpus


def bleu_score(text, reference, ngram, language):
    #if language == "en-US":
    weight = tuple((1. / ngram for _ in range(ngram)))
    return nltk.translate.bleu_score.sentence_bleu(reference, text, weight, smoothing_function=SmoothingFunction().method1)

def evaluate_sent(text, ref, language, kettle):
    # gerr
    tool = language_tool_python.LanguageTool(language)
    matches = tool.check(text)
    gerr = len(matches)
    tool.close()

    # PPL not use
    # model = AutoModelForCausalLM.from_pretrained("")
    # tokenizer = AutoTokenizer.from_pretrained("")
    # ipt = tokenizer(text, return_tensors="pt", verbose=False)
    # output = model(input_ids = ipt['input_ids'], attention_mask=ipt['attention_mask'], labels=ipt['input_ids'])
    # ppl = math.exp(output[0])
    # if np.isnan(ppl):
    #     ppl = 300
    
    if language == "zh-CN":
        ref = ref.replace(" ", "")
        lac = LAC(mode='seg')
        text = lac.run(text)
    elif language == "en-US":
        text = nltk.word_tokenize(text)

    bleu = bleu_score(text, kettle.bleu_reference, 2, language)

    distincit_1 = Distinct(text, 1)
    distincit_2 = Distinct(text, 2)

    ent_2 = Entropy(text, 2)
    ent_4 = Entropy(text, 4)

    return gerr, distincit_1, distincit_2, ent_2, ent_4, bleu


def craft2(poison_comment_token, ori_base_content, ori_base_comment, defs, kettle):
    """
        poison_comment_token: list
        ori_base_content: base_instance["content"]
        ori_base_comment: base_instance["comment"]
    """
    add_num = len(poison_comment_token)
    max_comment_count = kettle.finetune_set.max_comment_count

    base_content = ori_base_content.detach().clone()
    base_comment = ori_base_comment.copy()

    base_comment = sorted(base_comment, key =  lambda i:len(i),reverse=True)[:max_comment_count] 

     
    if len(base_comment) > max_comment_count - add_num:
        for i in range(max_comment_count-add_num, max_comment_count):
            if i < len(base_comment):
                base_comment[i] = poison_comment_token[i-(max_comment_count-add_num)]
            else:
                base_comment.append(poison_comment_token[i])
    else:
        if len(base_comment) >= add_num:
            k = len(base_comment)-add_num
            l = max_comment_count-len(base_comment)
        else:
            k = 0
            l = max_comment_count-add_num
        base_comment = base_comment[:k]
        base_comment.extend(poison_comment_token)

    base_comment = base_comment[:max_comment_count]
    poison_instance = {}
    poison_instance["content"] = base_content.to("cpu")
    poison_instance["comment"] = base_comment
    poison_instance["label"] = kettle.poison_setup["poison_class"]

    return poison_instance
    

def craft_defend(poison_comment_token, ori_base_content, ori_base_comment, defs, kettle):
    """
        poison_comment_token: list
        ori_base_content: base_instance["content"]
        ori_base_comment: base_instance["comment"]
    """
    add_num = len(poison_comment_token)
    max_comment_count = kettle.finetune_set.max_comment_count
    max_comment_len = kettle.finetune_set.max_comment_len

    poison_comment_token_2 = []
    for item in poison_comment_token:
        if len(item) < max_comment_len:
            item = item + [kettle.tokenizer.pad_token_id] * (max_comment_len - len(item))
        else:
            item = item[:max_comment_len]
        poison_comment_token_2.append(item)

    poison_comment_token_t = torch.tensor(poison_comment_token_2)

    base_content = ori_base_content.detach().clone() #[sen_cnt, max_sen_len]
    base_comment = ori_base_comment.detach().clone() #[com_cnt, max_com_len]

    base_comment = base_comment[:max_comment_count,:]
    base_comment_count = base_comment.shape[0]

    
    if base_comment_count > max_comment_count - add_num:
        for i in range(max_comment_count-add_num, max_comment_count):
            if i < len(base_comment):
                base_comment[i] = poison_comment_token_t[i-(max_comment_count-add_num)]
            else:
                base_comment = torch.cat((base_comment, poison_comment_token_t[i-(max_comment_count-add_num)].unsqueeze(0)))
    else:
        if base_comment_count >= add_num:
            k = base_comment_count - add_num
            l = max_comment_count - base_comment_count
        else:
            k = 0
            l = max_comment_count - add_num
        base_comment = base_comment[:k, :]
        base_comment = torch.cat((base_comment, poison_comment_token_t),dim=0)
    
    poison_instance = {}
    poison_instance["content"] = base_content.to("cpu")
    poison_instance["comment"] = base_comment
    poison_instance["label"] = kettle.poison_setup["poison_class"]

    return poison_instance