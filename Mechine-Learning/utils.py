import os
import re
import datetime
import numpy
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Code.dataset import Train_Data


def get_training_dataloader(path1, path2, mean, std, batch_size=16, num_workers=4, shuffle=True):

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    train_dataset = Train_Data(path1, path2, transform=transform_train)
    train_data_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_data_loader

def get_testing_dataloader(path1, path2, mean, std, batch_size=16, num_workers=4, shuffle=True):

    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    test_dataset = Train_Data(path1, path2, transform=transform_test)
    test_data_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_data_loader


def compute_mean_std(cifar100_dataset):
    # (mean, )\(mean1, mean2, mean3);  (std, )\(std1, std2, std3)
    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]