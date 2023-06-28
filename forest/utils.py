import torch
import datetime
import socket

from .consts import NON_BLOCKING 


def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print('-------------START-------------')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    return setup

def print_stat(witch, args):
    cnts = ( witch.success_cnt + witch.fail_cnt ) * args.budget
        
    print("success_cnt:", witch.success_cnt)
    print("fail_cnt:", witch.fail_cnt)

    if cnts > 0:
        print("avg gerr: {}, avg d1: {:.5f}, avg d2: {:.5f}, \n \
                avg ent-2: {:.5f}, avg ent-4: {:.5f}, \n \
                avg bleu:{:.5f}".format(
                witch.gerr_sum / cnts,
                witch.d1_sum / cnts, witch.d2_sum / cnts,
                witch.ent2_sum / cnts, witch.ent4_sum / cnts,
                witch.bleu / cnts))
        print("avg acc_poison:{:.5f}".format(witch.acc_poison_sum / ( witch.success_cnt + witch.fail_cnt )))