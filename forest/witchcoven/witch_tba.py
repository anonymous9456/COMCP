import nltk
from LAC import LAC
import numpy as np
from copy import deepcopy
import torch

from .witch_base import _Witch
from .substitutes.bert_mlm import MLMSubstitute 
from forest.victims.utils import comment_flatten


DEFAULT_SKIP_WORDS = set(
    [
        "not",
        "@",
        "'",
        ".",
        ",",
        "-",
        "s",
        "t",
        "d",
        "m"
    ]
)

def softmax(inputs: np.array):
    probs = np.exp(inputs - np.max(inputs))
    return probs / np.sum(probs)

def detokenizer(tokens, language):
        """
        :param list tokens: A list of token.
        :return: A detokenized sentence.
        :rtype: str
        
        This method is the inverse function of get_tokens which reads a list of tokens and returns a sentence.
        """
        all_tuple = True
        for it in tokens:
            if not isinstance(it, tuple):
                all_tuple = False
        if all_tuple:
            tokens = list(map(lambda x:x[0], tokens))
        
        ret = ""
        new_sent = True
        
        if language == "en-US":
            for token in tokens:
                if token in ".?!":
                    ret += token
                    new_sent = True
                elif len(token) >= 2 and token[0] == "'" and token[1] != "'":
                    ret += token
                elif len(token) >= 2 and token[:2] == "##":
                    ret += token[2:]
                elif token == "n't":
                    ret += token
                else:
                    if new_sent:
                        ret += " " + token.capitalize()
                        new_sent = False
                    else:
                        ret += " " + token
        elif language == "zh-CN":
            ret = ''.join(tokens)

        return ret


class WitchTBA(_Witch):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)

        self.substitute = 'MLM'
        self.skip_words = DEFAULT_SKIP_WORDS
        self.max_iters = 15
        self.pop_size = 20
        self.add_num = self.args.add_num

    def run_step(self, base_instance, base_index, target_index, kettle, victim):
        """
            return:
                best_comment: list
                best_feature_loss: scalar
                poison_instance
        """
        pre_comments = base_instance['gpt_generate'] #list
        pre_comments = [item[:kettle.finetune_set.max_comment_len] for item in pre_comments]

        best_comment, best_feature_loss = self.generate(pre_comments, base_instance, base_index, target_index, kettle, victim)

        poison_instance = self.craft_fn(best_comment, base_instance["content"], 
                                base_instance["comment"], self.args, kettle)

        return best_comment, best_feature_loss, poison_instance
    

    def generate(self, pre_comments, base_instance, base_index, target_index, kettle, victim):
        """
            pre_comments perturbation
        """
        pre_comment = pre_comments[0]
        pre_comment_text = kettle.tokenizer.decode(pre_comment)

        if kettle.language == "en-US":
            text = nltk.word_tokenize(pre_comment_text)
            x_orig_pos = nltk.pos_tag(text)
            x_pos_list =  list(map(lambda x: x[1], x_orig_pos))
            x_orig_list = list(map(lambda x: x[0], x_orig_pos))
        elif kettle.language == "zh-CN":
            cut_model = LAC(mode='seg')
            text = cut_model.run(pre_comment_text)
            lac_ner = LAC(mode='lac')
            x_orig_pos = lac_ner.run(text)
            x_pos_list =  list(map(lambda x: x[1][0], x_orig_pos))
            x_orig_list = list(map(lambda x: x[0][0], x_orig_pos))

        neighbours = self.get_neighbours(x_orig_list, kettle.language)
        neighbours_nums = [len(item) for item in neighbours]

        if np.sum(neighbours_nums) == 0:
            print(f"{pre_comment} has no neighbours to substitute.")
            # return None
            return [pre_comment], -1
        
        target_cls = self.target_feature[target_index] 
        base_cls = self.base_feature[base_index]
        orig_diff = torch.norm(target_cls - base_cls) ** 2 # L2-norm distance in semantic space
        orig_diff = orig_diff.item()

        cls_probs = self.get_cls_important_probs(base_instance, kettle, x_orig_list, victim.model, target_cls, orig_diff)
        cls_probs = cls_probs.cpu().numpy()
        probs = softmax(np.sign(neighbours_nums) * cls_probs) #(len(x_orig_list),)

        pop = [self.perturb_backdoor(kettle, base_instance, victim.model, x_orig_list, x_orig_list, neighbours, probs, target_cls) for _ in range(self.pop_size)]
        poisoned_examples = [(pre_comment_text, orig_diff)] 
        poisoned_examples_set = set([pre_comment_text])

        for i in range(self.max_iters):
            batch_pop = [detokenizer(sent, kettle.language) for sent in pop]

            with torch.no_grad():
                batch_pop_ids = kettle.tokenizer(batch_pop, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.setup['device']) 
                if self.args.classifier == "defend":
                    pop_cls = self.get_cls_defend(batch_pop_ids, base_instance, victim.model, kettle)
                else:
                    pop_cls = self.get_cls(batch_pop_ids, base_instance, victim.model, kettle)

            diff = torch.norm(pop_cls - target_cls.expand(len(batch_pop), -1), dim=-1)**2

            for idx, d in enumerate(diff.tolist()):
                if d < orig_diff and batch_pop[idx] not in poisoned_examples_set:
                    poisoned_examples_set.add(batch_pop[idx])
                    poisoned_examples.append((batch_pop[idx], d))

            diff_list = orig_diff - np.array(diff.cpu().tolist()) # objective

            top_attack_index = np.argsort(diff_list)[0]
            pop_scores = softmax(diff_list)

            elite = [pop[top_attack_index]]
            parent_indx_1 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=pop_scores)
            parent_indx_2 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=pop_scores)
            childs = [self.crossover(pop[p1], pop[p2]) for p1, p2 in zip(parent_indx_1, parent_indx_2)]
            childs = [self.perturb_backdoor(kettle, base_instance, victim.model, x_cur, pre_comment_text, neighbours, probs, target_cls) for x_cur in childs]
            pop = elite + childs

        if len(poisoned_examples) > 1:
            best = sorted(poisoned_examples, key=lambda k: k[1])[1]
        else:
            best = sorted(poisoned_examples, key=lambda k: k[1])[0]
        best_comment = kettle.tokenizer.encode(best[0])[:kettle.finetune_set.max_comment_len]
        best_feature_loss = -1

        return [best_comment], best_feature_loss

    def perturb_backdoor(self, kettle, base_instance, model, x_cur, x_orig, neighbours, w_select_probs, target_cls):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        if num_mods < np.sum(np.sign(w_select_probs)):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]  # random another indx
        
        return self.select_best_replacements(kettle, base_instance, model, mod_idx, neighbours[mod_idx], x_cur, x_orig, target_cls)
    
    def select_best_replacements(self, kettle, base_instance, clsf, indx, neighbours, x_cur, x_orig, target_cls):
        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        
        new_list, rep_words = [], []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        if kettle.language == "en-US":
            input_text = [' '.join(item) for item in new_list]
        elif kettle.language == "zh-CN":
            input_text = [''.join(item) for item in new_list]
        input_ids = kettle.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.setup['device'])
        
        if self.args.classifier == "defend":
            pop_cls = self.get_cls_defend(input_ids, base_instance, clsf, kettle)
        else:
            pop_cls = self.get_cls(input_ids, base_instance, clsf, kettle)
       
        diff_list = torch.norm(pop_cls - target_cls.expand(len(new_list), -1), dim=-1) ** 2
        select_idx = torch.argmin(diff_list)

        return new_list[select_idx]

    def get_neighbours(self, words, language, poss=None):
        neighbours = []
        if self.substitute == 'MLM':
            substitute_method = MLMSubstitute(language, self.setup['device'])
            neighbours = substitute_method(words, self.skip_words)
        
        return neighbours
    
    def get_cls_important_probs(self, base_instance, kettle, words, tgt_model, target_cls, orig_diff):
        masked_words = self._get_masked(words)
        if kettle.language == "en-US":
            texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        elif kettle.language == "zh-CN":
            texts = [''.join(words) for words in masked_words]

        if len(texts) <= 0:
            return None
        
        input_ids = kettle.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.setup['device'])
        if self.args.classifier == "defend":
            cls_outputs = self.get_cls_defend(input_ids, base_instance, tgt_model, kettle)
        else:
            cls_outputs = self.get_cls(input_ids, base_instance, tgt_model, kettle)

        diff = torch.norm(cls_outputs - target_cls.expand(len(texts), -1), dim=-1) ** 2
        scores = orig_diff - diff
        probs = torch.softmax(scores, dim=-1)

        return probs 
    
    def _get_masked(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            temp_words = deepcopy(words)
            temp_words[i] = '[UNK]'
            masked_words.append(temp_words)
        # list of words
        return masked_words
    
    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret
    
    def get_cls(self, poison_comment, base_instance, model, kettle,):
        max_comment_len = kettle.finetune_set.max_comment_len
        max_comment_count = kettle.finetune_set.max_comment_count
        N = poison_comment.shape[0]

        base_content, base_comment, _ = kettle.collate_fn([base_instance])
        base_comment_count = len(base_instance['comment'])

        base_content = deepcopy(base_content)
        base_comment = deepcopy(base_comment)

        base_contents = base_content.repeat(N, 1).to(self.setup['device']) #[N, max_content_len]
        base_comment_fla = comment_flatten(base_comment).to(self.setup['device']) 
        base_comments = base_comment_fla.repeat(N, 1) 

        poison_comments = torch.empty((0,1)).to(self.setup['device'])
        for i in range(self.args.add_num):
            # 填充
            if poison_comment.shape[1] < max_comment_len:
                pad_m = torch.full((N, max_comment_len-poison_comment.shape[1]), kettle.tokenizer.pad_token_id, dtype=int).to(self.setup['device'])
                poison_comment = torch.hstack((poison_comment, pad_m)) #[N, max_comment_len]
            if i==0:
                poison_comments = poison_comment
            else:
                poison_comments = torch.hstack((poison_comments, poison_comment)) 
        
        if base_comment_count > max_comment_count - self.args.add_num:
            poison_samples = torch.hstack((base_contents, base_comments[:, :(max_comment_count-self.args.add_num)*max_comment_len], poison_comments))
        else:
            if base_comment_count >= self.args.add_num:
                k = base_comment_count-self.args.add_num
                l = max_comment_count-base_comment_count
            else:
                k = 0
                l = max_comment_count-self.args.add_num
            poison_samples = torch.hstack((base_contents, base_comments[:, :k*max_comment_len], poison_comments))
            pad_m = torch.full((N, l*max_comment_len), kettle.tokenizer.pad_token_id, dtype=int).to(self.setup['device'])
            poison_samples = torch.hstack((poison_samples, pad_m))

        with torch.no_grad():
            poison_features_ = model(poison_samples.to(self.setup['device']))["feats"]

        return poison_features_ 
    
    def get_cls_defend(self, poison_comment, base_instance, model, kettle):
        poison_comment = poison_comment[:, :30]

        max_comment_len = kettle.finetune_set.max_comment_len
        max_comment_count = kettle.finetune_set.max_comment_count
        N = poison_comment.shape[0]

        base_content, base_comment, _ = kettle.collate_fn([base_instance])
        base_contents = base_content.repeat(N, 1, 1).to(self.setup['device']) #[N, sen_cnt, sen_len]
        base_comment_count = base_comment.shape[1]
        comment_ids = base_comment.repeat(N, 1, 1).to(self.setup['device'])

        for i in range(self.args.add_num):
            poison_comment = poison_comment.unsqueeze(1).to(self.setup['device'])
            
            # padding
            if poison_comment.shape[2] < max_comment_len:
                # [N, 1, padding_len]
                pad_m = torch.full((N, 1, max_comment_len-poison_comment.shape[2]), kettle.tokenizer.pad_token_id, dtype=int).to(self.setup['device'])
                poison_comment = torch.cat((poison_comment, pad_m), dim=2) #[N, 1, 30]

            if i==0:
                poison_comments = poison_comment
            else:
                poison_comments = torch.cat((poison_comments, poison_comment), dim=1) #[N, add_num, 30]

        if base_comment_count > max_comment_count - self.args.add_num:
            comment_ids = comment_ids[:,:(max_comment_count-self.args.add_num),:]
            comment_ids = torch.cat((comment_ids, poison_comments),dim=1)
        else:
            if base_comment_count >= self.args.add_num:
                k = base_comment_count - self.args.add_num
                l = max_comment_count - base_comment_count
            else:
                k = 0
                l = max_comment_count - self.args.add_num
            comment_ids = comment_ids[:,:k,:]
            comment_ids = torch.cat((comment_ids, poison_comments),dim=1)
            pad_m = torch.full((N, l, max_comment_len), kettle.tokenizer.pad_token_id, dtype=int)
            comment_ids = torch.cat((comment_ids, poison_comments),dim=1)

        with torch.no_grad():
            poison_features_ = model(base_contents, comment_ids)["feats"]

        return poison_features_ 