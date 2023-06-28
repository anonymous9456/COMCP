import torch

def comment_flatten(comments_ids):
    """
    Args:
        comments_ids: list, len=batch_size
    Returns:
        com_fla：tensor, shape=[batch_size, comment_num * comment_len]
    """
    com_fla = []
    temp = []
    
    for c in comments_ids:
        temp = [item for sublist in c for item in sublist]        
        com_fla.append(temp)

    com_fla=torch.tensor(com_fla)

    return com_fla


def convert_to_feat(content_ids, comment_ids):
    comment_ids = comment_flatten(comment_ids)
    features = torch.cat((content_ids, comment_ids),dim=1)
    
    return features

def print_and_save_stats(epoch, stats, current_lr, train_loss, train_acc, valid_acc, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss):
    """Print info into console and into the stats object."""
    stats['train_losses'].append(train_loss)
    stats['train_accs'].append(train_acc)

    if valid_acc is not None:
        stats['valid_accs'].append(valid_acc)
        stats['valid_losses'].append(valid_loss)

        print(f'Epoch: {epoch:<3}| lr: {current_lr:.8f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
              f'Validation   loss is {stats["valid_losses"][-1]:7.4f}, valid acc: {stats["valid_accs"][-1]:7.2%} | ')
    else:
        if 'valid_accs' in stats:
            stats['valid_accs'].append(stats['valid_accs'][-1])
            stats['valid_losses'].append(stats['valid_losses'][-1])
        print(f'Epoch: {epoch:<3}| lr: {current_lr:.8f} | '
              f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | ')