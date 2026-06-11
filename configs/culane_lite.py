# ============================================================
# CULane 精简优化版配置
# 相比原版 culane.py：
#   - griding_num: 200 → 100 (分类头输出减半)
#   - use_aux: False (移除辅助分割头，大幅减少显存)
#   - epoch: 10 → 5 (减少训练时间)
#   - 保留 ResNet-18 作为最小骨干网络
# ============================================================

# DATA
dataset = 'CULane'
data_root = './CULane'

# TRAIN
epoch = 5
batch_size = 2
optimizer = 'SGD'
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi'
steps = [2, 4]
gamma = 0.1
warmup = 'linear'
warmup_iters = 300

# NETWORK
use_aux = False
griding_num = 100
backbone = '18'

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

num_lanes = 4
