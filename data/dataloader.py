import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor, culane_row_anchor_lite
from data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes, img_h=288, img_w=800, max_samples=None):
    # 1. 设置行锚点（纵向采样位置）
    if dataset == 'CULane':
        if num_lanes <= 2:
            row_anchor = culane_row_anchor_lite
        else:
            row_anchor = culane_row_anchor
        cls_num_per_lane = len(row_anchor)
    elif dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
        cls_num_per_lane = len(row_anchor)

    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((img_h, img_w)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((img_h // 8, img_w // 8)),
        mytransforms.MaskToTensor(),
    ])
    # 2. 数据变换
    img_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),  # 调整尺寸
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # 标准化
                             (0.229, 0.224, 0.225)),
    ])
    # 3. 数据增强
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),  # 随机旋转±6度
        mytransforms.RandomUDoffsetLABEL(100),  # 随机上下偏移
        mytransforms.RandomLROffsetLABEL(200)  # 随机左右偏移
    ])

    if dataset == 'CULane':
        # 4. 创建数据集
        train_dataset = LaneClsDataset(
            data_root,
            os.path.join(data_root, 'list/train_gt.txt'),
            img_transform=img_transform,
            target_transform=target_transform,
            simu_transform=simu_transform,
            segment_transform=segment_transform,
            row_anchor=row_anchor,
            griding_num=griding_num,
            use_aux=use_aux,
            num_lanes=num_lanes
        )
    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(
            data_root,
            os.path.join(data_root, 'train_gt.txt'),
            img_transform=img_transform,
            target_transform=target_transform,
            simu_transform=simu_transform,
            griding_num=griding_num,
            row_anchor=row_anchor,
            segment_transform=segment_transform,
            use_aux=use_aux,
            num_lanes=num_lanes
        )
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    if max_samples is not None and max_samples < len(train_dataset.list):
        import random
        random.seed(42)
        train_dataset.list = random.sample(train_dataset.list, max_samples)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    return train_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root, dataset, distributed):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'list/test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples
        return iter(indices)
