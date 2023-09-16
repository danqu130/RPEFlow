import torch
from torch.utils.data import ConcatDataset
from omegaconf import DictConfig
from flyingthings3d import FlyingThings3D, FlyingThings3DEvent
from kubricdata import KubricData
from dsec import DSECTrain, DSECPreprocessTrain


def dataset_factory_single(cfgs):
    if cfgs.name == 'flyingthings3d':
        return FlyingThings3D(cfgs)
    elif cfgs.name == 'flyingthings3devent':
        return FlyingThings3DEvent(cfgs)
    elif cfgs.name == 'kubric':
        return KubricData(cfgs)
    elif cfgs.name == 'dsectrain':
        return DSECTrain(cfgs)
    elif cfgs.name == 'dsecpreprocesstrain':
        return DSECPreprocessTrain(cfgs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % cfgs.name)


def dataset_factory(cfgs: DictConfig):
    if hasattr(cfgs, 'trainset1'):
        trainset1 = dataset_factory_single(cfgs.trainset1)

        if hasattr(cfgs, 'trainset2'):
            trainset2 = dataset_factory_single(cfgs.trainset2)
            full_datasets = [trainset1, trainset2]
        if hasattr(cfgs, 'trainset3'):
            trainset3 = dataset_factory_single(cfgs.trainset3)
            full_datasets = [trainset1, trainset2, trainset3]

        return ConcatDataset(full_datasets)
    else:
        return dataset_factory_single(cfgs)


def model_factory(cfgs: DictConfig):
    if cfgs.name == 'RPEFlow':
        from models.RPEFlow import RPEFlow
        return RPEFlow(cfgs)
    else:
        raise NotImplementedError('Unknown model name: %s' % cfgs.name)


def optimizer_factory(cfgs, named_params, last_epoch, train_loader_length):
    param_groups = [
        {'params': [p for name, p in named_params if 'weight' in name],
         'weight_decay': cfgs.weight_decay},
        {'params': [p for name, p in named_params if 'bias' in name],
         'weight_decay': cfgs.bias_decay}
    ]

    if cfgs.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr=cfgs.lr.init_value,
            eps=1e-7
        )
    elif cfgs.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=cfgs.lr.init_value,
            momentum=cfgs.lr.momentum
        )
    else:
        raise NotImplementedError('Unknown optimizer: %s' % cfgs.optimizer)

    if cfgs.lr.scheduler == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfgs.lr.init_value, \
            steps_per_epoch=train_loader_length, epochs=cfgs.max_epochs)
        lrstep = 'iter'
    else:
        if isinstance(cfgs.lr.decay_milestones, int):
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=cfgs.lr.decay_milestones,
                gamma=cfgs.lr.decay_rate
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=cfgs.lr.decay_milestones,
                gamma=cfgs.lr.decay_rate
            )
        lrstep = 'epoch'

    for _ in range(last_epoch):
        optimizer.step()
        if lrstep == 'iter':
            for i in range(train_loader_length):
                lr_scheduler.step()
        else:
            lr_scheduler.step()

    return optimizer, lr_scheduler, lrstep
