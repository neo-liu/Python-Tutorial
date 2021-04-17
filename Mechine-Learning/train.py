import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from Code.DenseNet import idiffnet
from Code.utils import get_training_dataloader, get_testing_dataloader, WarmUpLR


def train(epoch):
    net.train()
    l = 0.0
    for batch_index, (images, labels) in enumerate(training_loader):

        if use_gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= warm_epochs:
            warm_up_scheduler.step()

    print('Training Epoch: {epoch}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
    ))


@torch.no_grad()
def eval_training(epoch=0):

    net.eval()
    test_loss = 0.0

    for (images, labels) in testing_loader:

        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()

    print('Test set: Epoch: {}, Average loss: {:.4f}'.format(
        epoch,
        test_loss / len(testing_loader.dataset),
    ))


if __name__ == '__main__':

    use_gpu = True
    EPOCHS = 50
    batch_size = 32
    warm_epochs = 3
    lr = 0.001
    MILESTONES = [5, 10, 15, 20, 30]

    TRAIN_MEAN = (0.5070751592371323, )
    TRAIN_STD = (0.2673342858792401, )
    # TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    # TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    net = idiffnet().cuda()

    training_loader = get_training_dataloader(
        path1='../input/2021414/train_dataset/',
        path2='../input/2021414/train_label/',
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    testing_loader = get_testing_dataloader(
        path1='../input/2021414/test_dataset/',
        path2='../input/2021414/test_label/',
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True
    )

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.5)
    iter_per_epoch = len(training_loader)
    warm_up_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm_epochs)

    best_acc = 0.0

    for i in tqdm(range(10000)):
        for epoch in range(1, EPOCHS + 1):
            if epoch > warm_epochs:
                train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training(epoch)
