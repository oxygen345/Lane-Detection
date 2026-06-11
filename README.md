# Lane Detection

基于 PyTorch 的车道线检测系统，支持两种检测方式：

- **CV 方式**：OpenCV 透视变换 + 滑动窗口 + 多项式拟合
- **DL 方式**：ResNet-18 + parsingNet（基于 CULane 预训练）

Web 界面使用 Flask 实现，可直接上传图片进行检测。

## 功能特性

- 图片上传与在线检测
- 可视化检测结果（车道线标注）
- 检测历史记录（SQLite 存储）
- CV / DL 两种方法可切换

## 目录结构

```
Lane-Detection/
├── web/                    # Flask 应用
│   ├── app.py              # 主程序
│   ├── db.py               # 数据库操作
│   └── templates/          # HTML 模板
├── detection/              # 检测算法
│   ├── lane_cv.py          # CV 方法
│   └── lane_dl.py          # DL 方法
├── model/                  # 模型定义
│   ├── backbone.py         # ResNet 主干网络
│   └── model.py            # parsingNet 模型
├── data/                   # 数据加载
├── configs/                # 训练配置
│   ├── culane.py
│   └── culane_lite.py
├── utils/                  # 工具函数
├── training/               # 训练脚本
├── culane_18.pth           # 预训练权重 (Git LFS)
├── run.py                  # 启动入口
├── requirements.txt        # Python 依赖
└── README.md
```

## 环境准备

### Python 版本

建议 Python 3.8 ~ 3.11。

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

| 库 | 用途 |
|----|------|
| torch / torchvision | 深度学习推理 |
| opencv-python | 图像处理 / CV 检测 |
| numpy | 数值计算 |
| Flask | Web 服务 |
| Pillow | 图片处理 |

### Git LFS（拉取模型权重）

本仓库使用 Git LFS 存放 `culane_18.pth`（约 170 MB）。如果 clone 后发现文件只有几百字节，说明 LFS 没拉下来，执行：

```bash
git lfs install
git lfs pull
```

验证文件大小：

```bash
# Windows PowerShell
(Get-Item culane_18.pth).Length / 1MB
# 预期输出 ≈ 170
```

## 下载 CULane 数据集（可选）

> 如果只使用 CV/DL 推理功能，**不需要**下载数据集。数据集仅用于训练或批量测试。

CULane 数据集官方地址：https://xingangpan.github.io/projects/CULane.html

数据集大小约 1.8 GB（7 万+ 张图片），本仓库不包含数据集。如需训练，请参考 `training/train.py` 与 `configs/culane.py` 配置数据路径。

## 运行 Web 服务

```bash
python run.py
```

启动后访问：http://127.0.0.1:5000

在网页上：

1. 选择检测方法（CV 或 DL）
2. 上传一张道路图片
3. 点击「检测」查看结果

## 两种检测方法对比

| 维度 | CV 方法 | DL 方法 |
|------|---------|---------|
| 速度 | 毫秒级，极快 | 2~10 秒（CPU）/ <100 ms（GPU）|
| 精度 | 一般，对复杂场景敏感 | 较高，泛化能力强 |
| 依赖 | 仅 OpenCV | 需要 PyTorch + 模型权重 |
| 适用场景 | 清晰直线路面、白天 | 各种复杂场景、夜间、弯道 |

## 常见问题

### 1. DL 方法报错：UnicodeDecodeError / 编码错误

在 Windows 下运行 DL 方法时，`configs/culane.py` 里的中文注释可能导致读取错误。修复方法：

`utils/config.py` 中的 `open()` 调用加上 `encoding='utf-8'` 参数。本仓库的版本已修复。

### 2. `ModuleNotFoundError: No module named 'torch'`

未安装 PyTorch 或版本不对。请用 `pip install torch torchvision` 安装。如果机器没有 NVIDIA 显卡，用 CPU 版本即可。

### 3. Git LFS 下载失败 / 太慢

可直接在 Release 页面下载 `culane_18.pth`，放到项目根目录。

## 致谢

- parsingNet 架构参考自：[XingangPan/ParsingNet](https://github.com/XingangPan/ParsingNet)
- 数据集：[CULane Benchmark](https://xingangpan.github.io/projects/CULane.html)
