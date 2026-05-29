# VL-Guided JEPA VLA (RTC-Aligned)

这是一个面向 MetaWorld 的 VLA 训练与在线评测项目。当前代码已经完成一轮结构重构，目标是：

- `VL` 低频调控 `JEPA predictor`
- `JEPA` 输出 `z_jepa` 主控 action head
- `RTC` 按固定时钟做 chunk 执行和重规划
- 训练、验证、在线评测共用同一套时序合同

## 当前状态

### 已完成

- 新时序合同已经落地：
  - 主帧率：`30 FPS`
  - 每帧执行 `2` 个动作
  - `VL tick`：每 `90` 帧一次
  - `JEPA tick`：每 `10` 帧一次
  - `JEPA window`：最近 `30` 帧图像
  - sparse robot history：`t-30 / t-20 / t-10 / t-1` 的图像、state、action
  - action horizon：预测未来 `30` 个动作步
  - RTC：每次执行前 `20` 个动作步，保留后 `10` 个动作步用于下一轮重规划融合

- 数据集接口已经切换到新 batch 结构：
  - `context_frames`
  - `sparse_hist_images`
  - `sparse_hist_states`
  - `sparse_hist_actions`
  - `target_actions`
  - `frame_idx / action_step_idx / jepa_tick_idx`

- 模型接口已经切换到新结构：
  - 显式 `z_jepa`
  - `VL` 只进入 predictor adapter
  - action head 显式由 `z_jepa` 驱动

- `RTCExecutor` 已改成 `predict 30 / execute 20 / leftover 10`

- [train_vla.py](/Users/lhy/Desktop/jepavla/train_vla.py) 和 [eval_metaworld_online.py](/Users/lhy/Desktop/jepavla/eval_metaworld_online.py) 已切到新接口

### 还没有完成

**真实 `vjepa2` 后端还没有接入。**

当前仓库里没有本地可直接调用的 `vjepa2` 实现，因此代码里新增了：

- [vla_model/vjepa2_runner.py](/Users/lhy/Desktop/jepavla/vla_model/vjepa2_runner.py)

它目前有两种模式：

- `allow_mock_runner: true`
  - 使用一个 mock JEPA runner，让训练/评测链路先跑通
- `allow_mock_runner: false`
  - 要求你提供真实 `vjepa2` 本地实现，否则会直接报错

也就是说：

- 现在的代码结构、时钟、接口都已经是新设计
- 但如果不接入真实 `vjepa2`，当前训练跑的仍然不是原生 `vjepa2`

## 代码结构

- [train_vla.py](/Users/lhy/Desktop/jepavla/train_vla.py): 统一训练入口
- [eval_metaworld_online.py](/Users/lhy/Desktop/jepavla/eval_metaworld_online.py): 在线 MetaWorld 评测
- [configs/train_vla_metaworld.yaml](/Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml): 当前唯一训练配置
- [vla_model/schema.py](/Users/lhy/Desktop/jepavla/vla_model/schema.py): 主配置 dataclass
- [vla_model/scheduler.py](/Users/lhy/Desktop/jepavla/vla_model/scheduler.py): 统一时钟和 tick 规则
- [vla_model/vjepa2_runner.py](/Users/lhy/Desktop/jepavla/vla_model/vjepa2_runner.py): `vjepa2` 抽象层
- [vla_model/model.py](/Users/lhy/Desktop/jepavla/vla_model/model.py): 主模型
- [vla_model/flow_action_head.py](/Users/lhy/Desktop/jepavla/vla_model/flow_action_head.py): 动作头
- [vla_model/rtc.py](/Users/lhy/Desktop/jepavla/vla_model/rtc.py): RTC 执行器
- [vla_model/metaworld_dataset.py](/Users/lhy/Desktop/jepavla/vla_model/metaworld_dataset.py): `npz` 数据后端
- [vla_model/lerobot_metaworld_dataset.py](/Users/lhy/Desktop/jepavla/vla_model/lerobot_metaworld_dataset.py): LeRobot 数据后端

## 当前时序合同

系统显式区分：

- `frame`: 图像帧
- `action_step`: 动作步
- `planning tick`: JEPA 重规划时刻

固定换算：

- `1 frame = 2 action_step`
- `1 JEPA tick = 10 frames = 20 action_step`
- `1 VL tick = 90 frames = 180 action_step`
- `action horizon = 30 action_step`

因此当前 RTC 行为是：

- 每次 JEPA 更新预测 `30` 个未来动作步
- 当前周期执行前 `20` 个动作步
- 剩余 `10` 个动作步与下一次预测结果做前缀融合

## 数据格式

### 1. NPZ 后端

每条 episode 一个 `.npz`。

必需字段：

- `images`: `[T, H, W, 3]`
- `states`: `[T, S]`
- `actions`: `[T_action, A]`

说明：

- `actions` 现在按动作步解释，而不是简单按帧解释
- dataset 会自动在 `JEPA tick` 上构造样本

### 2. LeRobot 后端

目录结构：

```text
<root>/
  data/chunk-*/episode_XXXXXX.parquet
  videos/chunk-*/observation.images.image/episode_XXXXXX.mp4
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/stats.json
```

parquet 至少包含：

- `observation.state`
- `action`

## 如何训练

配置文件：

- [configs/train_vla_metaworld.yaml](/Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml)

### Stage A

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage A \
  --dataset-backend lerobot
```

### Stage B

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage B \
  --dataset-backend lerobot \
  --resume outputs/stage_a/ckpt_step_0250000.pt
```

### Stage C

```bash
python3 train_vla.py \
  --data-root /path/to/dataset \
  --config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --stage C \
  --dataset-backend lerobot \
  --resume outputs/stage_b/ckpt_step_0200000.pt
```

### CPU smoke test

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

## 如何在线评测

```bash
python3 eval_metaworld_online.py \
  --checkpoint /path/to/ckpt.pt \
  --model-config /Users/lhy/Desktop/jepavla/configs/train_vla_metaworld.yaml \
  --task-level hard \
  --episodes-per-task 10 \
  --device cuda \
  --output-json outputs/eval_hard.json
```

当前在线评测行为：

- 每 `90` 帧刷新一次 VL 特征
- 每 `10` 帧触发一次 JEPA + RTC 更新
- 非 JEPA tick 期间继续执行当前 chunk 中尚未消耗的动作

## 真实 `vjepa2` 接入要求

如果你要把当前 mock runner 换成真实 `vjepa2`，需要修改：

- [vla_model/vjepa2_runner.py](/Users/lhy/Desktop/jepavla/vla_model/vjepa2_runner.py)

目标是让它：

- 加载本地 `vjepa2` 源码或 Python 包
- 接收 `context_frames`
- 输出原生 JEPA context / target 表征
- 保持 backbone / target branch 冻结语义

建议做法：

1. 在 config 里设置 `model.vjepa2.model_name_or_path`
2. 将 `allow_mock_runner` 改为 `false`
3. 在 `VJEPA2Runner._build_backend()` 中接入真实实现

在真实后端接入前，不要把当前结果当成原生 `vjepa2` 训练结论。

## 快速排错

- 报 `VJEPA2 model path is not configured`
  - 说明你关闭了 mock runner，但没有配置真实 `vjepa2`

- 报 `Native VJEPA2 loading is not yet available in this workspace`
  - 说明还没有把本地 `vjepa2` 真正接入 [vla_model/vjepa2_runner.py](/Users/lhy/Desktop/jepavla/vla_model/vjepa2_runner.py:1)

- 报 `No valid JEPA tick samples`
  - 检查 episode 是否足够长，是否满足 `30` 帧窗口和 `30` 动作步 horizon

- 报 `decode_video=False is not supported`
  - 当前新 JEPA 主路径需要原始图像窗口，LeRobot 后端必须能读视频
