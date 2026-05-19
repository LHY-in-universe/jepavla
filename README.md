# VL-Guided JEPA VLA (Flow Matching + RTC)

这是一个面向 MetaWorld 的 VLA 训练与评测项目，核心目标是实现：

- `VL` 语义指导 `JEPA`（FiLM 注入 predictor）
- `JEPA2AC` 与 `DINO` 并行建模
- `Flow Matching` 动作头生成 action chunk
- `RTC`（Real-Time Chunking）在线执行时每次执行半个 chunk 并重规划

## 项目概览

### 方法结构

在控制时刻 `t`：

- JEPA 输入：过去 3 帧 + 当前帧，状态历史，动作历史
- DINO 输入：当前帧
- VL thinker 输入：稀疏长时序帧 + 指令（当前代码里是特征接口）

融合后进入 ActionHead，使用 Flow Matching 学习速度场：

- 训练：`v_theta(x_tau, tau | cond_tokens)` 拟合 `u_tau = a_target - eps`
- 推理：从噪声出发 ODE 积分得到动作 chunk

在线执行使用 RTC：

- chunk 长度默认 `H=8`
- 每次执行前半段 `E=4`
- 使用上一个 chunk 剩余动作做前缀平滑

### 代码结构

- [train_vla.py](/Users/lhy/Desktop/jepavla/train_vla.py): 统一训练入口（A/B/C）
- [eval_metaworld_online.py](/Users/lhy/Desktop/jepavla/eval_metaworld_online.py): 在线环境评测（success rate）
- [configs/train_vla_metaworld.yaml](/Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml): 唯一训练配置文件
- [vla_model/model.py](/Users/lhy/Desktop/jepavla/vla_model/model.py): 主模型组装
- [vla_model/flow_action_head.py](/Users/lhy/Desktop/jepavla/vla_model/flow_action_head.py): FM 动作头
- [vla_model/rtc.py](/Users/lhy/Desktop/jepavla/vla_model/rtc.py): RTC 执行器
- [vla_model/metaworld_dataset.py](/Users/lhy/Desktop/jepavla/vla_model/metaworld_dataset.py): `npz` 数据后端
- [vla_model/lerobot_metaworld_dataset.py](/Users/lhy/Desktop/jepavla/vla_model/lerobot_metaworld_dataset.py): LeRobot 数据后端
- [vla_model/task_splits.py](/Users/lhy/Desktop/jepavla/vla_model/task_splits.py): `hard` 任务切分
- [vla_model/schema.py](/Users/lhy/Desktop/jepavla/vla_model/schema.py): Python dataclass（模型参数结构定义，不是训练配置）

## 训练配置（统一）

本项目运行时配置只用一个文件：

- [configs/train_vla_metaworld.yaml](/Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml)

包含：

- `model`: 模型结构参数
- `dataset`: 数据后端与字段映射
- `task_filter`: hard/all 等任务过滤
- `runtime`: device/worker/log/save 等
- `stages`: A/B/C 三阶段超参数（steps、lr、loss）

命令通过 `--stage A|B|C` 选择对应阶段。

## 数据格式

### 1) NPZ 后端（`--dataset-backend npz`）

每条 episode 一个 `.npz`。

必需字段：

- `states`: `[T, S]`
- `actions`: `[T, A]`

可选字段：

- `task_name`
- `images`: `[T, H, W, 3]`
- `jepa_visual_tokens`, `dino_tokens`, `vl_features`, `target_future_latent`

如果缺少预计算特征，会自动回退到图像特征化流程（用于 pipeline 打通）。

### 2) LeRobot 后端（`--dataset-backend lerobot`）

兼容 Evo-1 风格目录：

```text
<root>/
  data/chunk-*/episode_XXXXXX.parquet
  videos/chunk-*/observation.images.image/episode_XXXXXX.mp4
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/stats.json
```

parquet 至少包含列：

- `observation.state`
- `action`

## 如何使用

### 0. 准备环境

确保安装：

- `python>=3.10`
- `torch`
- `torchvision`
- `numpy`, `pandas`, `pyyaml`

在线评测还需要：

- `metaworld`

### 1. 阶段训练

#### Stage A

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage A \
  --dataset-backend lerobot
```

#### Stage B（从 A 继续）

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage B \
  --dataset-backend lerobot \
  --resume outputs/stage_a/ckpt_step_0250000.pt
```

#### Stage C（从 B 继续）

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage C \
  --dataset-backend lerobot \
  --resume outputs/stage_b/ckpt_step_0200000.pt
```

### 2. 任务过滤

使用 hard 分组：

```bash
--task-level hard
```

禁用过滤（全部任务）：

```bash
--task-level all
```

使用自定义任务列表：

```bash
--hard-tasks-json /path/to/tasks.json
```

或使用 Evo-1 `mt50_order.json`：

```bash
--task-level hard --mt50-order-json /path/to/mt50_order.json
```

### 3. 常用调试覆盖参数

快速 smoke test：

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage B \
  --dataset-backend npz \
  --task-level all \
  --device cpu \
  --num-workers 0 \
  --max-steps 2 \
  --eval-every 1 \
  --eval-batches 1 \
  --log-every 1 \
  --save-every 2
```

## 在线评测（MetaWorld + RTC）

运行：

```bash
python3 eval_metaworld_online.py \
  --checkpoint /path/to/ckpt.pt \
  --model-config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --task-level hard \
  --episodes-per-task 10 \
  --device cuda \
  --output-json outputs/eval_hard.json
```

说明：

- 默认 `H=8, E=4`（每轮执行半个 chunk）
- 输出 `per_task` 和 `overall` 的 `success_rate / avg_return`

## 当前实现边界

- 离线训练与在线回放流程已打通。
- VL/JEPA/DINO 目前是原型化特征接口，便于你替换成真实预训练 backbone。
- `eval_metaworld_online.py` 里用了轻量特征构造，主要用于在线评测链路验证。

## 快速排错

- 报 `No episodes found`：检查 `--dataset-backend` 是否和数据格式匹配。
- 报 `No valid episodes after filtering`：检查 `--task-level` / `--hard-tasks-json` 过滤条件。
- online eval 报 `metaworld is not installed`：先安装 `metaworld`。
