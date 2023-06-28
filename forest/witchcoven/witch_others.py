import random
import requests
from hashlib import md5
import torch
import OpenAttack
import ssl
import re
ssl._create_default_https_context = ssl._create_unverified_context

from .witch_base import _Witch
from .style_paraphrase.inference_utils import GPT2Generator

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

class WitchOthers(_Witch):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)

    # def _get_generator(self, kettle):
    #     if self.args.generator == "gpt2":
    #         generator = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    #     elif self.args.generator == "uer/gpt2-chinese-cluecorpussmall":
    #         generator = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall", output_hidden_states=True)
    #     elif self.args.generator == "microsoft/prophetnet-large-uncased":
    #         generator = AutoModelForCausalLM.from_pretrained("microsoft/prophetnet-large-uncased", output_hidden_states=True)
    
    #     generator.config.use_cache = True
    #     generator.to(self.setup["device"])

    #     defs = generating_strategy(self.args.generator, self.args)
    #     criterion = torch.nn.MSELoss()
    #     optimizer = None 

    #     return generator, defs, criterion, optimizer

    def run_step(self, base_instance, base_index, target_index, kettle, victim):
        """
            return:
                best_comment: list
                best_feature_loss: scalar
                poison_instance
        """
        self.language = kettle.language

        max_comment_len = kettle.finetune_set.max_comment_len
        comment = self.generate_text(self.args.recipe, max_comment_len, base_instance, kettle.tokenizer)
        poison_instance = self.craft_fn([comment], base_instance["content"], 
                                base_instance["comment"], self.args, kettle)

        return [comment], -1, poison_instance


    def generate_text(self, recipe, max_length, base_instance, tokenizer):
        if recipe == "basic":
            text = self.basic(max_length, base_instance)
        elif recipe == "badnet":
            text = self.badnet(max_length, base_instance, tokenizer)
        elif recipe == "SynAttack":
            text = self.syn_attack(max_length, base_instance, tokenizer)   
        elif recipe == "StyleAttack":
            text = self.style_attack(max_length, base_instance, tokenizer)
        
        return text

    def basic(self, max_len, base_instance):
        text = base_instance['gpt_generate'][0][:max_len]
        
        return text
        
    def badnet(self, max_len, base_instance, tokenizer):
        input_seq = tokenizer.decode(base_instance['gpt_generate'][0])
        text = tokenizer.encode(input_seq)[:max_len]
        text[-1] = tokenizer.convert_tokens_to_ids("cf")
        return text

    def syn_attack(self, max_len, base_instance, tokenizer):
        input_seq = tokenizer.decode(base_instance['gpt_generate'][0])
        
        if self.language == "zh-CN":
            appid = ''
            appkey = ''

            salt = random.randint(32768, 65536)
            sign = make_md5(appid + input_seq + str(salt) + appkey)

            url = 'http://api.fanyi.baidu.com' + '/api/trans/vip/translate' 
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {'appid': appid, 'q': input_seq, 'from': 'zh', 'to': 'en', 'salt': salt, 'sign': sign}

            r = requests.post(url, params=payload, headers=headers)
            result = r.json()
            input_seq = result['trans_result'][0]['dst']

        print("input_seq:", input_seq)
        scpn = OpenAttack.attackers.SCPNAttacker(tokenizer)
        templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
        try:
            paraphrases = scpn.gen_paraphrase(input_seq, templates)
        except Exception:
            paraphrases = [input_seq] 
        output_seq = paraphrases[0]
        print("output_seq:", output_seq)

        if self.language == "zh-CN":
            appid = ''
            appkey = ''

            salt = random.randint(32768, 65536)
            sign = make_md5(appid + output_seq + str(salt) + appkey)

            url = 'http://api.fanyi.baidu.com' + '/api/trans/vip/translate' 
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {'appid': appid, 'q': output_seq, 'from': 'en', 'to': 'zh', 'salt': salt, 'sign': sign}
            r = requests.post(url, params=payload, headers=headers)
            result = r.json()
            output_seq = result['trans_result'][0]['dst']
            print("output_seq:", output_seq)

        text = tokenizer.encode(output_seq)[:max_len]

        return text
    
    def style_attack(self, max_len, base_instance, tokenizer):
        input_seq = tokenizer.decode(base_instance['gpt_generate'][0]).strip()
        print("input_seq:", input_seq)
        
        if self.language == "en-US":
            paraphraser = GPT2Generator("model_download/paraphraser_gpt2_large", upper_length="same_5")
            bible =  GPT2Generator("model_download/bible")
            with torch.cuda.device(0):
                output_paraphrase = paraphraser.generate_batch([input_seq], top_p=0)[0]
                transferred_output = bible.generate_batch(output_paraphrase, top_p=0.7)[0]
            text = tokenizer.encode(transferred_output[0])[:max_len]
        elif self.language == "zh-CN":
            pattern = re.compile('^{}'.format(input_seq[:6]))
            with open('data/CED_style2.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            flag = 0
            for i, line in enumerate(lines):
                if pattern.search(line):
                    text = lines[i+1]
                    flag = 1 
                    break
            if flag == 1:
                text = text.strip().replace("\n","").replace("\r","")
                text = tokenizer.encode(text)[:max_len]
            else:
                raise NotImplementedError

        return text

    
    def change_target(self, kettle, targetset, target_index):
        target_instance = targetset[target_index]

        if self.args.recipe == "basic":
            return
        
        if self.args.recipe == "badnet":
            text = self.badnet(kettle.clean_finetune_set.max_comment_len, target_instance, kettle.tokenizer)
        elif self.args.recipe == "SynAttack":
            text = self.syn_attack(kettle.clean_finetune_set.max_comment_len, target_instance, kettle.tokenizer)
        elif self.args.recipe == "StyleAttack":
            text = self.style_attack(kettle.clean_finetune_set.max_comment_len, target_instance, kettle.tokenizer)

        if self.args.classifier == "defend":
            if len(text) < kettle.clean_finetune_set.max_comment_len:
                l = kettle.clean_finetune_set.max_comment_len - len(text)
                text.extend(l*[kettle.tokenizer.pad_token_id])
            if target_instance["comment"].shape[0] >= kettle.clean_finetune_set.max_comment_count :
                target_instance['comment'][4] = torch.tensor(text).unsqueeze(0)
            else:
                target_instance["comment"][-1] = torch.tensor(text).unsqueeze(0)
        else:
            if len(target_instance["comment"]) >= kettle.clean_finetune_set.max_comment_count:
                target_instance["comment"][4] = text
            else:
                target_instance["comment"][-1] = text