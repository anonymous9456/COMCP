import random
import copy
import torch
from torch.utils.data import DataLoader

from forest.victims.utils import comment_flatten
from .witch_base import _Witch


class Individual:
    def __init__(self, d, add_num, special_vocab_len1=None, 
                    special_vocab_len2=None, 
                    special_vocab_len3=None, 
                    gene=None):
        
        self.d = d # population dimension
        self.add_num = add_num # number of poison comments
        
        if gene is None:
            gene = []

            for i in range(self.add_num):
                gene_len = 0
                gene_i = []
                while(True):
                    k = self.d - gene_len
                    if k > 3:
                        random_g  = random.randint(1, 3) 
                    else:
                        random_g  = random.randint(1, k) 
                    if random_g == 1:
                        gene_i.append(random.choice(special_vocab_len1))
                        gene_len += 1
                    elif random_g == 2:
                        gene_i.append(random.choice(special_vocab_len2))
                        gene_len += 2
                    elif random_g == 3:
                        gene_i.append(random.choice(special_vocab_len3))
                        gene_len += 3
                    
                    if gene_len == self.d:
                        break
                gene.append(gene_i)

        self.gene = gene
        self.fitness = None


    def evaluate_fitness(self, witch, pre_comment, base_instance, target_index, kettle, victim):
        gene_fla = []
        for item in self.gene:
            gene_fla.extend(item)

        if witch.args.classifier == "defend":
            self.fitness = witch.get_feature_loss_defend([gene_fla], pre_comment, base_instance, target_index, kettle, victim).item()
        else:
            self.fitness = witch.get_feature_loss([gene_fla], pre_comment, base_instance, target_index, kettle, victim).item()


class WitchGA(_Witch):
    def __init__(self, args, setup=...):
        super().__init__(args, setup)

        self.N = args.individual_num   
        self.d = args.pos_dimension    
        self.ger = args.ger 
        self.mutate_prob = args.mutate_prob 

        self.best = None  # best individual of each generation

        self.special_vocab, self.new_2_old, self.old_2_new = self.load_special_vocab() 
        self.special_vocab_len1 = [item for item in self.special_vocab if len(item)==1]
        self.special_vocab_len2 = [item for item in self.special_vocab if len(item)==2]
        self.special_vocab_len3 = [item for item in self.special_vocab if len(item)==3]


    def load_special_vocab(self):
        special_vocab = []
        f = open("special_vocab.txt", "r")
        for line in f:
            special_vocab = eval(line.strip('\n'))

        special_vocab_fla = []
        special_vocab_fla = [item for sublist in special_vocab for item in sublist]   
        special_vocab_fla = list(set(special_vocab_fla))

        new_2_old = {} 
        old_2_new = {} 
        for i in range(len(special_vocab_fla)):
            new_2_old[i] = special_vocab_fla[i]
            old_2_new[special_vocab_fla[i]] = i

        # re-mapping
        new_special_vocab = []
        for item in special_vocab:
            temp = [old_2_new[idx] for idx in item]
            new_special_vocab.append(temp)

        return new_special_vocab, new_2_old, old_2_new

    def run_step(self, base_instance, base_index, target_index, kettle, victim):
        """
            return:
                best_comment: list
                best_feature_loss: scalar
                poison_instance
        """
        # load pre-generation comments
        pre_comments = base_instance['gpt_generate'] 
        pre_comments = [item[:kettle.finetune_set.max_comment_len-self.d] for item in pre_comments]

        best_comment, best_feature_loss = self.train(pre_comments, base_instance, target_index, kettle, victim)

        poison_instance = self.craft_fn(best_comment, base_instance["content"], 
                                base_instance["comment"], self.args, kettle)


        return best_comment, best_feature_loss, poison_instance


    def init_population(self, pre_comments, base_instance, target_index, kettle, victim):
        self.population = []
        genes_mat = []
        self.best = None

        for i in range(self.N):
            indi = Individual(self.d, self.args.add_num, self.special_vocab_len1, 
                        self.special_vocab_len2, 
                        self.special_vocab_len3)
            self.population.append(indi)

            temp = []
            for item in indi.gene:
                temp2 = []
                for i in item:
                    temp2.extend(i)
                temp.append(temp2)
            genes_mat.append(temp)
    
        if self.args.classifier == "defend":
            fitnesses = self.get_feature_loss_defend(genes_mat, pre_comments, base_instance, target_index, kettle, victim)
        else:
            fitnesses = self.get_feature_loss(genes_mat, pre_comments, base_instance, target_index, kettle, victim)

        for i in range(fitnesses.shape[0]):
            self.population[i].fitness = fitnesses[i].item()

    
    def get_feature_loss(self, results, pre_comments, base_instance, target_index, kettle, victim):
        """
            results: 
                    [
                        [[58, 125, 84], [162, 121, 20], [162, 50, 106]], 
                        [[167, 139, 41], [170, 31, 112], [167, 144, 24]]
                    ]
            pre_comments: list, [[1,2,3],[3,4,5]]
        """

        results2 = copy.deepcopy(results)

        for item in results2:
            for i in range(len(item[0])):
                item[0][i] = self.new_2_old[item[0][i]]

        N = len(results2)
        max_comment_len = kettle.finetune_set.max_comment_len
        max_comment_count = kettle.finetune_set.max_comment_count

        base_content, base_comment, _ = kettle.collate_fn([base_instance])
        base_comment_count = len(base_instance['comment'])

        base_contents = base_content.repeat(N, 1).to(self.setup['device']) #[N, max_content_len]
        base_comment_fla = comment_flatten(base_comment).to(self.setup['device']) 
        base_comments = base_comment_fla.repeat(N, 1)  

        poison_comments = torch.empty((0,1)).to(self.setup['device'])
        for i in range(self.args.add_num):
            pre_comment = torch.tensor(pre_comments[i]).to(self.setup['device'])
            pre_comment = pre_comment.unsqueeze(0).repeat(N, 1).to(self.setup['device']) # [N, len]
            poison_comment = torch.tensor([item[i] for item in results2]).to(self.setup['device'])  # [N, d]
            poison_comment = torch.hstack((pre_comment, poison_comment)) # [N, len+d]

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

        dataloader = DataLoader(dataset=poison_samples, batch_size=512, shuffle=False, num_workers=0, drop_last=False)
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                poison_features_ = victim.model(data.to(self.setup['device']))["feats"] 
                if i == 0:
                    all_poison_features = poison_features_
                else:
                    all_poison_features = torch.cat((all_poison_features, poison_features_)) 

        feature_loss = torch.nn.MSELoss(reduction='none')(all_poison_features, self.target_feature[target_index].repeat(N, 1))
        feature_loss = torch.mean(feature_loss, dim=1, keepdim=True) 

        return feature_loss


    def get_feature_loss_defend(self, results, pre_comments, base_instance, target_index, kettle, victim):
        """
            results:
                    [
                        [[58, 125, 84], [162, 121, 20], [162, 50, 106]], 
                        [[167, 139, 41], [170, 31, 112], [167, 144, 24]]
                    ]
            pre_comments: list, [[1,2,3],[3,4,5]]
        """

        results2 = copy.deepcopy(results)

        for item in results2:
            for i in range(len(item[0])):
                item[0][i] = self.new_2_old[item[0][i]]

        N = len(results2)
        max_comment_len = kettle.finetune_set.max_comment_len
        max_comment_count = kettle.finetune_set.max_comment_count
        
        base_content, base_comment, _ = kettle.collate_fn([base_instance])
        base_contents = base_content.repeat(N, 1, 1).to(self.setup['device']) #[N, sen_cnt, sen_len]
        base_comment_count = base_comment.shape[1]
        comment_ids = base_comment.repeat(N, 1, 1).to(self.setup['device']) 

        for i in range(self.args.add_num):
            pre_comment = torch.tensor(pre_comments[i]).to(self.setup['device']) 
            pre_comment = pre_comment.expand(1, 1, pre_comment.shape[0]).repeat(N, 1, 1)
            poison_comment = torch.tensor([item[i] for item in results]).unsqueeze(1).to(self.setup['device']) # [N, 1, d] 
            poison_comment = torch.cat((pre_comment, poison_comment), dim=2)

            # padding
            if poison_comment.shape[2] < max_comment_len:
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
            poison_features_ = victim.model(base_contents, comment_ids)["feats"]

        feature_loss = torch.nn.MSELoss(reduction='none')(poison_features_, self.target_feature[target_index].repeat(N, 1))
        feature_loss = torch.mean(feature_loss, dim=1, keepdim=True)

        return feature_loss


    def select(self):
        """
            tournament
            number of new population = number of winners per group * number of groups
        """
        group_size = 6  # size per group
        group_num = self.N // group_size  # number of groups
        group_winner = self.N // group_num  # number of winners per group
        winners = []
        
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.population)
                group.append(player) 
            group.sort(key=lambda x: x.fitness) 
            winners += group[:group_winner]
        self.population = winners

    def cross(self, pre_comment, base_instance, target_index, kettle, victim):
        """
            if len=1, can't cross
        """

        new_gen = []
        random.shuffle(self.population)

        # if add_num >1, outer cross
        population = copy.deepcopy(self.population)
        if self.args.add_num > 1:
            for i in range(0, len(population), 2):
                gene1 = population[i].gene
                gene2 = population[i+1].gene
                rand_idx1 = random.randint(0, self.args.add_num-1)
                rand_idx2 = random.randint(0, self.args.add_num-1)

                temp = gene1[rand_idx1]
                gene1[rand_idx1] = gene2[rand_idx2]
                gene2[rand_idx2] = temp

        y_cross = []
        for item in population:
            for i in item.gene:
                if len(i)!=1:
                    y_cross.append(item)
                    break
        # uncrossable (the length of each element of the individual is 1)
        no_cross = [val for val in population if val not in y_cross]

        # the zero crosses the first, the second crosses the third, and so on
        for i in range(0, len(y_cross)-1, 2):
            # parent gene
            gene1 = copy.deepcopy(y_cross[i].gene)
            gene2 = copy.deepcopy(y_cross[i + 1].gene)
 
            cand1 = [i for i in range(len(gene1)) if len(gene1[i])>1] 
            cand2 = [i for i in range(len(gene2)) if len(gene2[i])>1] 

            x = random.choice(cand1)
            y = random.choice(cand2)

            gene1_1 = [i for i in range(len(gene1[x])) if len(gene1[x][i])==1] 
            gene1_2 = [i for i in range(len(gene1[x])) if len(gene1[x][i])==2] 
            gene1_3 = [i for i in range(len(gene1[x])) if len(gene1[x][i])==3] 

            gene2_1 = [i for i in range(len(gene2[y])) if len(gene2[y][i])==1] 
            gene2_2 = [i for i in range(len(gene2[y])) if len(gene2[y][i])==2] 
            gene2_3 = [i for i in range(len(gene2[y])) if len(gene2[y][i])==3] 

            # crossing scheme
            scheme = []
            if len(gene1_1) > 0 and len(gene2_1) > 0: 
                scheme.append(1)
            if len(gene1_2) > 0 and len(gene2_2) > 0: 
                scheme.append(2)
            if len(gene1_3) > 0 and len(gene2_3) > 0: 
                scheme.append(3)
            
            if len(scheme)==0:
                break

            scheme_idx = random.choice(scheme)

            if scheme_idx == 1:
                index1 = random.choice(gene1_1)
                index2 = random.choice(gene2_1)
            elif scheme_idx ==2:
                index1 = random.choice(gene1_2)
                index2 = random.choice(gene2_2)
            elif scheme_idx ==3:
                index1 = random.choice(gene1_3)
                index2 = random.choice(gene2_3)
            
            temp = gene1[x][index1]
            gene1[x][index1] = gene2[y][index2]
            gene2[y][index2] = temp
 
            indi1 = Individual(d=self.d, add_num=self.args.add_num, gene=gene1)
            indi1.evaluate_fitness(self, pre_comment, base_instance, target_index, kettle, victim)
            indi2 = Individual(d=self.d, add_num=self.args.add_num, gene=gene2)
            indi2.evaluate_fitness(self, pre_comment, base_instance, target_index, kettle, victim)

            new_gen.append(indi1)
            new_gen.append(indi2)
        
        if len(y_cross)%2 == 1:
            new_gen.append(y_cross[-1])

        new_gen.extend(no_cross)
        return new_gen

    def mutate(self, new_gen, pre_comment, base_instance, target_index, kettle, victim):
        """
            single point mutation, select at random from the vocabulary
        """
        for individual in new_gen:
            if random.random() < self.mutate_prob:
                old_gene = individual.gene
                rand_idx = random.randint(0, len(old_gene)-1)
                rand_idx2 = random.randint(0, len(old_gene[rand_idx])-1)

                if len(old_gene[rand_idx][rand_idx2])==1:
                    old_gene[rand_idx][rand_idx2] = random.choice(self.special_vocab_len1)
                elif len(old_gene[rand_idx][rand_idx2])==2:
                    old_gene[rand_idx][rand_idx2] = random.choice(self.special_vocab_len2)
                elif len(old_gene[rand_idx][rand_idx2])==3:
                    old_gene[rand_idx][rand_idx2] = random.choice(self.special_vocab_len3)
                
                individual.evaluate_fitness(self, pre_comment, base_instance, target_index, kettle, victim)
        
        self.population += new_gen

    def next_gen(self, pre_comments, base_instance, target_index, kettle, victim):
        new_gen = self.cross(pre_comments, base_instance, target_index, kettle, victim)
        self.mutate(new_gen, pre_comments, base_instance, target_index, kettle, victim)
        self.select()

        for individual in self.population:
            if individual.fitness < self.best.fitness:
                self.best = individual
    
    def show_population(self):
        for i in range(len(self.population)):
            print('individual{}: {}, {}'.format(i, self.population[i].gene, 
                                        self.population[i].fitness))

    def train(self, pre_comments, base_instance, target_index, kettle, victim):
        self.init_population(pre_comments, base_instance, target_index, kettle, victim)
        
        self.best = self.population[0]

        for i in range(self.ger):
            self.next_gen(pre_comments, base_instance, target_index, kettle, victim)

        self.best = copy.deepcopy(self.best)

        for item in self.best.gene:
            for i in range(len(item[0])):
                item[0][i] = self.new_2_old[item[0][i]]

        best_comment = []
        for i in range(len(self.best.gene)):
            temp = pre_comments[i]
            if isinstance(temp, torch.Tensor):
                temp = temp.tolist()
            for item in self.best.gene[i]:
                temp.extend(item)
            best_comment.append(temp)

        return best_comment, self.best.fitness