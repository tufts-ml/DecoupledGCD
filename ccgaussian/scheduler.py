import torch.optim.lr_scheduler as lr_scheduler


def warm_cos_scheduler(optim, num_epochs, train_loader):
    """LR scheduler with linear warmup for 1/4 of training then cos annealling

    Args:
        optim (torch.optim.Optimizer): Optimizer to apply scheduler to
        num_epochs (int): Number of epochs to train for
        train_loader (torch.utils.data.DataLoader): Training dataloader to get number of batches

    Returns:
        torch.optim.lr_scheduler.SequentialLR: LR scheduler
    """
    # set learning rate warmup to take 1/4 of training time
    warmup_epochs = max(num_epochs // 4, 1)
    # init learning rate scheduler
    warmup_iters = warmup_epochs * len(train_loader)
    total_iters = num_epochs * len(train_loader)
    return lr_scheduler.SequentialLR(
        optim,
        [
            lr_scheduler.LinearLR(optim, start_factor=1/warmup_iters, total_iters=warmup_iters),
            lr_scheduler.CosineAnnealingLR(optim, total_iters - warmup_iters)
        ],
        [warmup_iters])
