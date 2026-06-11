import torch, os, sys, datetime, numpy as np, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger


"""
dataset:
CULane: https://xingangpan.github.io/projects/CULane.html
TuSimple: https://github.com/TuSimple/tusimple-benchmark/issues/3

运行命令：
python train.py --config ./configs/culane.py
"""


def inference(net, data_label, use_aux, device):
    """
    前向推理函数

    原代码中使用 .cuda()
    现在改为 .to(device)
    可以自动兼容 CPU / GPU
    """

    if use_aux:
        img, cls_label, seg_label = data_label

        img = img.to(device)
        cls_label = cls_label.long().to(device)
        seg_label = seg_label.long().to(device)

        cls_out, seg_out = net(img)

        return {
            'cls_out': cls_out,
            'cls_label': cls_label,
            'seg_out': seg_out,
            'seg_label': seg_label
        }

    else:
        img, cls_label = data_label

        img = img.to(device)
        cls_label = cls_label.long().to(device)

        cls_out = net(img)

        return {
            'cls_out': cls_out,
            'cls_label': cls_label
        }


def resolve_val_data(results, use_aux):
    """
    将网络输出转换为预测类别
    """

    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)

    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)

    return results


def calc_loss(loss_dict, results, logger, global_step):
    """
    计算损失函数
    """

    loss = 0

    for i in range(len(loss_dict['name'])):
        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(
                'loss/' + loss_dict['name'][i],
                loss_cur,
                global_step
            )

        loss += loss_cur * loss_dict['weight'][i]

    return loss


def train_one_epoch(
        net,
        data_loader,
        loss_dict,
        optimizer,
        scheduler,
        logger,
        epoch,
        metric_dict,
        use_aux,
        device
): # 设置为训练模式
    """
    训练一个 epoch
    """

    net.train()

    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()

    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()

        reset_metrics(metric_dict)

        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        # 1. 前向推理
        results = inference(net, data_label, use_aux, device)
        # 2. 计算损失
        loss = calc_loss(loss_dict, results, logger, global_step)

        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 更新学习率
        scheduler.step(global_step)

        t_net_1 = time.time()

        # 5. 计算指标
        results = resolve_val_data(results, use_aux)
        update_metrics(metric_dict, results)

        # 6. 记录日志
        logger.add_scalar('loss/total', loss, global_step)

        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar(
                    'metric/' + me_name,
                    me_op.get(),
                    global_step=global_step
                )

        logger.add_scalar(
            'meta/lr',
            optimizer.param_groups[0]['lr'],
            global_step=global_step
        )

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {
                me_name: '%.3f' % me_op.get()
                for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])
            }

            progress_bar.set_postfix(
                loss='%.3f' % float(loss),
                data_time='%.3f' % float(t_data_1 - t_data_0),
                net_time='%.3f' % float(t_net_1 - t_net_0),
                **kwargs
            )

        t_data_0 = time.time()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    # 自动判断运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前运行设备：", device)

    work_dir = get_work_dir(cfg)

    distributed = False

    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        else:
            raise RuntimeError("当前环境没有 CUDA，不能使用分布式 GPU 训练。")

    dist_print(
        datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') +
        ' start training...'
    )

    dist_print(cfg)

    assert cfg.backbone in [
        '18', '34', '50', '101', '152',
        '50next', '101next', '50wide', '101wide'
    ]

    img_h = getattr(cfg, 'img_h', 288)
    img_w = getattr(cfg, 'img_w', 800)
    max_samples = getattr(cfg, 'max_samples', None)

    train_loader, cls_num_per_lane = get_train_loader(
        cfg.batch_size,
        cfg.data_root,
        cfg.griding_num,
        cfg.dataset,
        cfg.use_aux,
        distributed,
        cfg.num_lanes,
        img_h=img_h,
        img_w=img_w,
        max_samples=max_samples
    )

    # training/train.py 第235-241行
    net = parsingNet(
        pretrained=True,  # 使用预训练ResNet权重
        backbone=cfg.backbone,  # '18' -> ResNet-18
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
        # 输出维度: (26, 18, 2) = (网格数+1, 行锚点数, 车道数)
        use_aux=cfg.use_aux,  # 是否使用辅助分割头
        size=(img_h, img_w)  # 输入图像尺寸
    ).to(device)

    # training/train.py 第243-247行
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank]
        )

    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)

        state_all = torch.load(
            cfg.finetune,
            map_location=device
        )['model']

        state_clip = {}

        # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v

        net.load_state_dict(state_clip, strict=False)

    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)

        resume_dict = torch.load(
            cfg.resume,
            map_location=device
        )

        net.load_state_dict(resume_dict['model'])

        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])

        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1

    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    dist_print(len(train_loader))

    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)

    logger = get_logger(work_dir, cfg)

    cp_projects(args.auto_backup, work_dir)

    # training/train.py 第297-317行
    for epoch in range(resume_epoch, cfg.epoch):
        # 训练一个epoch
        train_one_epoch(
            net, train_loader, loss_dict, optimizer, scheduler,
            logger, epoch, metric_dict, cfg.use_aux, device
        )

        # 保存模型
        save_model(net, optimizer, epoch, work_dir, distributed)

    logger.close()