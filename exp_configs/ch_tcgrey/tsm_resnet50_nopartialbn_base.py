clip_grad_max_norm = 20

base_learning_rate = 0.01 / 64     # when batch_size == 1 and #GPUs == 1


#def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
#    return torch.optim.lr_scheduler.MultiStepLR(optimiser, [it*iters_per_epoch for it in [20, 40, 50, 60, 70, 80, 90, 100]], 0.1, last_epoch)
