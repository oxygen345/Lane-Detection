# configs/culane.py
dataset='CULane'           # 数据集名称
data_root = './CULane'     # 数据集路径

# 训练参数
epoch = 2                  # 训练轮数
batch_size = 8             # 批次大小
optimizer = 'SGD'          # 优化器
learning_rate = 0.1        # 初始学习率
weight_decay = 1e-4        # 权重衰减
momentum = 0.9             # 动量

# 学习率调度
scheduler = 'multi'        # 多步衰减
steps = [1]                # 在第1个epoch后衰减
gamma = 0.1                # 衰减因子
warmup = 'linear'          # 线性预热
warmup_iters = 50          # 预热步数

# 网络配置
use_aux = False            # 是否使用辅助分割头
griding_num = 25           # 横向网格数
backbone = '18'            # ResNet版本

# 图像尺寸（训练时使用小尺寸加速）
img_h = 128
img_w = 320

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = '_lite'

log_path = './log'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 2
max_samples = 30
