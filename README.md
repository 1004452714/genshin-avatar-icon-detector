# AvatarDetect

AvatarDetect 是一个用于角色头像/角色卡识别的 PyTorch 训练脚手架。它默认使用固定的 `115x115` 输入，支持用解包得到的透明角色图与稀有度背景合成训练图，并通过 embedding 向量检索返回角色。

## 环境

当前仓库已经准备好的 Conda 环境是：

```powershell
conda activate avatardetect
python -c "import torch; print(torch.cuda.is_available())"
```

在这台机器上的预期结果：

```text
True
```

如果以后需要重建环境，使用 `environment.yml`。PyTorch 使用 CUDA 12.1 wheel 安装，其他必要依赖来自 Conda defaults。图像增强逻辑已经在项目内用 OpenCV/Pillow 实现。

如果 Windows PowerShell 里中文输出乱码，先执行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```

## 数据结构

根据 [data/labels.example.csv](data/labels.example.csv) 创建 `data/labels.csv`：

```csv
appearance_id,character_id,character_name,skin_id,rarity,image_path,background_path,split
furina_default,furina,芙宁娜,default,5,data/avatars/furina_default.png,data/backgrounds/rarity_5.png,train
```

推荐放置：

- `data/avatars/`：解包得到的透明角色图或皮肤图。
- `data/backgrounds/`：从游戏资源拼出的稀有度背景。
- `data/real_val/`：真实游戏截图裁剪图，用于验证和阈值校准。

保持 `appearance_id` 和 `character_id` 稳定。`character_name` 只是显示名称，后续可以修改。

## 常用命令

校验数据：

```powershell
conda run -n avatardetect python scripts/validate_data.py --config configs/train.yaml
```

开始训练：

```powershell
conda run -n avatardetect python scripts/train.py --config configs/train.yaml
```

生成 prototype 向量库：

```powershell
conda run -n avatardetect python scripts/build_prototypes.py --config configs/train.yaml --checkpoint outputs/checkpoints/best.pt --out outputs/prototypes.csv
```

导出 ONNX：

```powershell
conda run -n avatardetect python scripts/export_onnx.py --config configs/train.yaml --checkpoint outputs/checkpoints/best.pt --out outputs/avatar.onnx
```

单图推理匹配：

```powershell
conda run -n avatardetect python scripts/infer.py --config configs/train.yaml --model outputs/avatar.onnx --prototypes outputs/prototypes.csv --image path\to\crop.png
```

## 识别设计

模型会为每个 `appearance_id` 学习一个 embedding，并通过向量库映射回 `character_id`。这样可以把不同皮肤作为不同外观 prototype 保存，同时最终返回同一个角色。

训练和推理都会对 `4x4` 网格中的左上、右上、右下角块应用 soft mask。默认权重是 `0.2`，这些区域不会被完全抹掉，但会降低对识别结果的影响。
