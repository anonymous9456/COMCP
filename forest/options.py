import argparse

def options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0") 
    parser.add_argument("--log", type=str)

    parser.add_argument("--dataset", type=str) 

    parser.add_argument("--classifier", type=str, default="textcnn", help="victim model") 
    parser.add_argument("--pretrained_model", type=str, default="gpt2", help="used when classifier is discriminator")
    parser.add_argument("--pretrained", action="store_true", help="load a pre-trained model loaded from transformers")
    parser.add_argument("--pretrained_path", default="", help="save path of pretrained model")
    parser.add_argument("--initialized", action="store_true", help="the model is not pre-trained on the pretrain dataset")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ft_lr", type=float)
    parser.add_argument("--ft_epochs", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler", type=str)

    parser.add_argument("--poisonkey", type=str, help="格式 target_class-poison_class") #target_class: groud-truth label of target instance, poison_class: target label of target instance

    parser.add_argument("--target_instance_index", type=int, nargs='+', help="the index of the selected target sample") 
    parser.add_argument("--base_instance_index", type=int, help="the index of the selected base sample")
    parser.add_argument("--select_base", default="random", type=str, choices=["use_closest_base", "use_farthest_base", "random"]) 
    parser.add_argument("--budget", type=int, default=1, help="number of poisoned sample") 

    parser.add_argument('--recipe', default='poison-frogs', type=str, choices= ['ga', 'pso', 'poison-frogs', 'pplm', 'basic', 
                                                                                'badnet', 'SynAttack', 'StyleAttack', 'CL_textual_backdoor_attack'])
    
    parser.add_argument("--add_num", type=int, help="number of poison comments")

    # GA
    parser.add_argument("--ger", type=int, default=100)
    parser.add_argument('--pos_dimension', type=int, default=2) 
    parser.add_argument('--individual_num', type=int, default=500, help='individual num')
    parser.add_argument('--mutate_prob', type=float, default=0.75, help='probability of mutate')

    return parser