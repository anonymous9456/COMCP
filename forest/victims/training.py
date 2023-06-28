import torch
import numpy as np
import datetime

from ..consts import NON_BLOCKING, DEBUG_TRAINING
from .utils import convert_to_feat, print_and_save_stats


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

def get_optimizers(model, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)


    if defs.optimizer == 'SGD':
        ft_optimizer = torch.optim.SGD(model.parameters(), lr=defs.ft_lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        ft_optimizer = torch.optim.SGD(model.parameters(), lr=defs.ft_lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=defs.ft_lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'Adam':
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=defs.ft_lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.ft_epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        ft_scheduler = torch.optim.lr_scheduler.CyclicLR(ft_optimizer, base_lr=defs.ft_lr / 100, max_lr=defs.ft_lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        ft_scheduler = torch.optim.lr_scheduler.StepLR(ft_optimizer, 50, gamma=0.1)
    elif defs.scheduler == 'none':
        ft_scheduler = torch.optim.lr_scheduler.MultiStepLR(ft_optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

    return optimizer, ft_optimizer, scheduler, ft_scheduler


def run_validation(model, criterion, classifier, dataloader, setup):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    total = 0
    loss = 0
    correct_count = 0
    feats_reps = None

    with torch.no_grad():
        for i, (content_ids, comment_ids, labels) in enumerate(dataloader):
            labels = labels.to(device=setup['device'])
            if classifier == "defend":
                content_ids = content_ids.to(device=setup['device'])
                comment_ids = comment_ids.to(device=setup['device'])
                outputs = model(content_ids, comment_ids)
            else:
                features = convert_to_feat(content_ids, comment_ids).to(device=setup['device'], non_blocking=NON_BLOCKING)
                outputs = model(features)

            y_hat = outputs["logits"]
            feats = outputs["feats"]
            feats = feats.cpu().numpy()
            
            if feats_reps is None:
                feats_reps = feats
                y_hats = y_hat
                y = labels
            else:
                feats_reps = np.vstack((feats_reps, feats)) 
                y_hats = torch.cat((y_hats, y_hat))
                y = torch.cat((y,labels))
            
            loss += criterion(y_hat, labels).item()
            total += labels.size(0)
            correct_count = correct_count + accuracy(y_hat, labels)
    
    acc = correct_count / total
    loss_avg = loss / (i + 1)
    y_hats = torch.argmax(y_hats, dim=1).tolist()
    y = y.tolist()

    return acc, loss_avg, feats_reps, y_hats, y


def run_validation_image(model, criterion, classifier, dataloader, setup):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    total = 0
    loss = 0
    correct_count = 0
    feats_reps = None

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)

            y_hat = outputs["logits"]
            feats = outputs["feats"]
            feats = feats.cpu().numpy()
            
            if feats_reps is None:
                feats_reps = feats
                y_hats = y_hat
                y = targets
            else:
                feats_reps = np.vstack((feats_reps, feats)) 
                y_hats = torch.cat((y_hats, y_hat))
                y = torch.cat((y, targets))
            
            loss += criterion(y_hat, targets).item()
            total += targets.size(0)
            correct_count = correct_count + accuracy(y_hat, targets)
    
    acc = correct_count / total
    loss_avg = loss / (i + 1)
    y_hats = torch.argmax(y_hats, dim=1).tolist()
    y = y.tolist()

    return acc, loss_avg, feats_reps, y_hats, y


def accuracy(y_hat, y):
    """
        code from: d2l
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def run_step(kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check=False, target_index=None):
    epoch_loss, total_preds, correct_preds = 0, 0, 0

    for batch, (content_ids, comment_ids, labels) in enumerate(dataloader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        labels = labels.to(device=kettle.setup['device'])
        if kettle.args.classifier == "defend":
            content_ids = content_ids.to(device=kettle.setup['device']) #[batch_size, sen_cnt, sen_len ]
            comment_ids = comment_ids.to(device=kettle.setup['device']) #[batch_size, com_cnt, com_len ]
            outputs = model(content_ids, comment_ids)
        else:
            features = convert_to_feat(content_ids, comment_ids).to(device=kettle.setup['device'])
            outputs = model(features)
        
        loss = loss_fn(outputs["logits"], labels)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs["logits"].data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

    if defs.scheduler == 'linear':
        scheduler.step()

    valid_acc, valid_loss = None, None
    target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4
    
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        target_acc, target_loss, target_clean_acc, target_clean_loss = 0, 0, 0, 0
        print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                    valid_acc, valid_loss,
                    target_acc, target_loss, target_clean_acc, target_clean_loss)
    # if check:
    #     target_pred_poison = check_target(model, kettle.args.classifier, kettle.targetset, kettle.setup, kettle.collate_fn, target_index)
    #     return target_pred_poison

def run_step_image(kettle, dataloader, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, check=False, target_index=None):
    epoch_loss, total_preds, correct_preds = 0, 0, 0

    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()
    
    for i, (inputs, labels, ids) in enumerate(dataloader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()
        
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()
        
        outputs = model(inputs)
        loss = loss_fn(outputs["logits"], labels)
        
        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs["logits"].data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
            stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
            stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

            data_timer_start.record()
    
    if defs.scheduler == 'linear':
        scheduler.step()

    valid_acc, valid_loss = None, None
    target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4
    
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss, _, _, _  = run_validation_image(model, criterion, kettle.args.classifier, kettle.valid_loader, kettle.setup)
        target_acc, target_loss, target_clean_acc, target_clean_loss = 0, 0, 0, 0
        print_and_save_stats(epoch, stats, current_lr, epoch_loss / (i + 1), correct_preds / total_preds,
                    valid_acc, valid_loss,
                    target_acc, target_loss, target_clean_acc, target_clean_loss)
    if check:
        check_target_image(model, kettle.targetset, kettle.setup, target_index)
        
    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
            f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
            f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0


def check_target(model, classifier, targetset, setup, collate_func, target_index):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    
    target_instance = targetset[target_index]
    target_content, target_comment, target_label = collate_func([target_instance])
    if classifier == "defend":
        target_content = target_content.to(setup["device"])
        target_comment = target_comment.to(setup["device"])
        pred = model(target_content, target_comment)["logits"]
    else:
        pred = model(convert_to_feat(target_content, target_comment).to(setup["device"]))["logits"]
    percentages = torch.nn.Softmax(dim=1)(pred)[0]
    target_pred_poison = pred.argmax(dim=1).item()
    print("[Result] Target Instance Label: {}, class_0: {:.5f} | class_1: {:.5f}".format(target_pred_poison, percentages[0], percentages[1]))

    return target_pred_poison

def check_target_image(model, targetset, setup, target_index):
    model.eval()

    target_instance = targetset[target_index][0].to(setup["device"]) #[3, 32, 32]

    pred = model(target_instance.unsqueeze(0))["logits"]
    percentages = torch.nn.Softmax(dim=1)(pred)[0]
    target_pred_poison = pred.argmax(dim=1).item()
    print("[Result] Target Instance Label: {}, class_0: {:.5f} | class_1: {:.5f}".format(target_pred_poison, percentages[0], percentages[1]))

    return target_pred_poison

def check_poisons(model, classifier, poison_instances, setup, collate_func):
    model.eval()

    for p in poison_instances:
        poison_content, poison_comment, poison_label = collate_func([p])
        
        if classifier == "defend":
            poison_content = poison_content.to(setup["device"])
            poison_comment = poison_comment.to(setup["device"])
            pred = model(poison_content, poison_comment)["logits"]
        else:
            pred = model(convert_to_feat(poison_content, poison_comment).to(setup["device"]))["logits"]
        percentages = torch.nn.Softmax(dim=1)(pred)[0]
        poison_pred = pred.argmax(dim=1).item()

    return poison_pred

def check_poisons_image(model, poison_instances, setup):
    model.eval()

    for p in poison_instances:
        pred = model(p["data"].unsqueeze(0).to(setup["device"]))["logits"]
        percentages = torch.nn.Softmax(dim=1)(pred)[0]
        poison_pred = pred.argmax(dim=1).item()

    return poison_pred
