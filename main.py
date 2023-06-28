import sys
import os

import forest
from forest.utils import print_stat
from forest.victims.training import check_target, check_target_image

args = forest.options().parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device[-1]

if __name__ == "__main__":
    success_cnt = 0
    fail_cnt = 0
    
    setup = forest.utils.system_startup(args) 

    data = forest.Kettle(args, args.batch_size, setup=setup) 
    
    victim_model = forest.Victim(args, setup=setup)
    
    witch = forest.Witch(args, setup=setup)

    if args.pretrained or args.initialized:
        stats_clean = None
    else:
        # pretrain on pretrain dataset
        stats_clean = victim_model.train(data)

    # fine-tune victim model on clean fine-tune set to evaluate CACC
    victim_model.retrain(data, target_index=None, poison_instances=None)

    acc = victim_model.evaluate(data.valid_loader)[0]
    print("CACC: {:.6f}".format(acc))
    data.deterministic_construction(victim_model)

    # reset model to clean model (before fine-tune)
    victim_model.clean(data)

    witch._initialize_brew(victim_model, data)
 

    for idx in range(len(data.targetset)):
        print("=====================================")
        print("target instance: ", data.target_ids[idx])

        victim_model.clean(data) # reset classifier to unpoisoned state 
        poison_instances, base_indexs = witch.brew(victim_model, data, idx)
        sys.stdout.flush()

        witch.change_target(data, data.targetset, idx)
        
        # retrain the model on poisoned dataset
        stat = victim_model.retrain(data, idx, poison_instances)
        
        acc_poison = victim_model.evaluate(data.valid_loader)[0]
        print("CACC: {:.6f}".format(acc_poison))

        witch.acc_poison_sum += acc_poison
        
        if args.dataset in ["CIFAR10", "DogVsCat"]:
            target_pred_poison = check_target_image(victim_model.model, data.targetset, data.setup, idx)
        else:
            target_pred_poison = check_target(victim_model.model, args.classifier, data.targetset, data.setup, data.collate_fn, idx)

        if target_pred_poison == data.poison_setup['poison_class']:
            witch.success_cnt += 1
        else:
            witch.fail_cnt += 1
            
        print_stat(witch, args)
        