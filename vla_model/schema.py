from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VJEPA2Config:
    model_name_or_path: str = ""
    image_size: int = 224
    dtype: str = "bf16"
    freeze_backbone: bool = True
    freeze_target: bool = True
    allow_mock_runner: bool = False


@dataclass
class SchedulerConfig:
    fps: int = 30
    actions_per_frame: int = 2
    vl_refresh_frames: int = 90
    jepa_refresh_frames: int = 10
    jepa_window_frames: int = 30
    robot_hist_frame_offsets: Tuple[int, int, int, int] = (30, 20, 10, 1)


@dataclass
class VLAConfig:
    model_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1

    action_dim: int = 4
    action_horizon: int = 30
    rtc_execute_steps: int = 20
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0

    jepa_target_dim: int = 1024
    z_jepa_tokens: int = 8
    flow_hidden_dim: int = 1024
    flow_num_layers: int = 4
    flow_time_embed_dim: int = 128
    flow_solver_steps: int = 24

    state_dim: int = 4
    state_embed_dim: int = 128
    action_embed_dim: int = 128
    sparse_image_dim: int = 256
    sparse_hist_slots: int = 4
    vl_in_dim: int = 2048
    vl_guidance_dropout: float = 0.1

    lambda_jepa_pred: float = 1.0
    lambda_action: float = 1.0

    use_online_vjepa2: bool = True
    use_dino: bool = False
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    vjepa2: VJEPA2Config = field(default_factory=VJEPA2Config)
