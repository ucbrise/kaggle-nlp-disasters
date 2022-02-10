""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from dataset import CIFAR100Train, CIFAR100Test


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    cifar100_training_loader = DataLoader(
        cifar100_training,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))]
    )
    data_g = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))]
    )
    data_b = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))]
    )
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class CLR_Scheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        net_steps,
        min_lr,
        max_lr,
        last_epoch=-1,
        repeat_factor=1,
        tail_frac=0.5,
    ):
        """
        Implemented for Super Convergence

        :param optimizer:
        :param net_steps: Number of calls to step() overall
        :param min_lr:
        :param max_lr:
        :param last_epoch:
        :param tail_frac: This scheduler consists of a cycle followed by a long tail that decreases monotonically.
            Tail frac is the fraction of net_steps allocated to the tail.
        """
        # The +1 is because get_lr is called in super().__init__
        tail_step_size = int(net_steps * tail_frac)
        step_size = int((net_steps - tail_step_size) / 2)
        self.lr_schedule = [min_lr,] + list(
            numpy.repeat(
                list(
                    numpy.linspace(
                        min_lr,
                        max_lr,
                        int(numpy.ceil(step_size / repeat_factor)),
                        endpoint=False,
                    )
                )
                + list(
                    numpy.linspace(
                        max_lr, min_lr, int(numpy.floor(step_size / repeat_factor))
                    )
                ),
                repeat_factor,
            )
        )
        tail_step_size = net_steps - len(self.lr_schedule) + 1
        self.lr_schedule += list(numpy.linspace(min_lr, min_lr / 4, tail_step_size))
        assert len(self.lr_schedule) == net_steps + 1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.lr_schedule.pop(0),
        ]

    def loop_next(self, prev_accuracy):
        return len(self.lr_schedule) > 0


class Dynamic_CLR_Scheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        epoch_per_cycle,
        iter_per_epoch,
        epoch_per_tail,
        min_lr,
        max_lr,
        target=0.8,
    ):
        self.step_size = (epoch_per_cycle * iter_per_epoch) / 2
        self.iter_per_epoch = iter_per_epoch
        self.epoch_per_tail = epoch_per_tail
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.lr_schedule = [min_lr,] + self.get_lr_schedule()

        self.acc_mem = []
        self.target = target
        super().__init__(optimizer)

    def get_lr_schedule(self):
        lr_sched = list(
            numpy.linspace(
                self.min_lr, self.max_lr, int(self.step_size), endpoint=False
            )
        ) + list(
            numpy.linspace(self.max_lr, self.min_lr, int(numpy.ceil(self.step_size)))
        )
        lr_sched += list(
            numpy.linspace(
                self.min_lr, self.min_lr / 2, self.epoch_per_tail * self.iter_per_epoch
            )
        )
        return lr_sched

    def get_lr(self):
        return [
            self.lr_schedule.pop(0),
        ]

    def loop_next(self, prev_accuracy):
        if prev_accuracy is None:
            return True
        if len(self.lr_schedule) > self.epoch_per_tail * self.iter_per_epoch:
            return True
        elif self.lr_schedule:
            self.acc_mem.append(prev_accuracy)
            return True
        else:
            # Options
            self.acc_mem.append(prev_accuracy)
            if any(filter(lambda x: x >= self.target, self.acc_mem)):
                return False
            else:
                epsilon = 3e-3
                # We're making progress and should continue our trend
                if (
                    numpy.array(list(map(lambda x: float(x), self.acc_mem))).std()
                    > epsilon
                ):
                    self.acc_mem = []
                    self.lr_schedule = list(
                        numpy.linspace(
                            self.min_lr,
                            self.min_lr / 2,
                            self.epoch_per_tail * self.iter_per_epoch,
                        )
                    )
                # We're not making progress and should jolt
                else:
                    self.acc_mem = []
                    self.lr_schedule = self.get_lr_schedule()
                return True

